from __future__ import annotations

import csv
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

from yolox.tracker import matching
from .basetrack import TrackState
from .byte_tracker import (
    BYTETracker,
    STrack,
    joint_stracks,
    remove_duplicate_stracks,
    sub_stracks,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from models.local_conflict_graph_common import (  # noqa: E402
    build_topk_bipartite_components,
    filter_local_conflict_clusters_by_size,
    solve_assignment_with_private_defer,
)
from models.local_conflict_set_predictor import (  # noqa: E402
    HostConditionedLocalConflictSetPredictor,
    encode_host_variant,
    normalize_host_vocab,
    pair_geometry_features,
    softmax_probs_1d,
    zscore_1d,
)
from models.posthost_one_edit_hierarchical import (  # noqa: E402
    PosthostOneEditHierarchical,
)


def _tlbr_array_to_cxcywh(tlbr_list: Sequence[np.ndarray]) -> torch.Tensor:
    if not tlbr_list:
        return torch.zeros((0, 4), dtype=torch.float32)
    rows: List[List[float]] = []
    for tlbr in tlbr_list:
        arr = np.asarray(tlbr, dtype=np.float32)
        x1, y1, x2, y2 = [float(v) for v in arr.tolist()]
        w = max(x2 - x1, 1e-6)
        h = max(y2 - y1, 1e-6)
        rows.append([x1 + 0.5 * w, y1 + 0.5 * h, w, h])
    return torch.tensor(rows, dtype=torch.float32)


class ByteTrackerLocalConflict(BYTETracker):
    def __init__(self, args, frame_rate: int = 30):
        super().__init__(args, frame_rate=frame_rate)
        self.use_local_conflict = bool(getattr(args, "use_local_conflict", False))
        self.use_posthost_oracle_edit = bool(getattr(args, "use_posthost_oracle_edit", False))
        self.use_posthost_hierarchical_edit = bool(getattr(args, "use_posthost_hierarchical_edit", False))
        self.local_conflict_checkpoint = str(getattr(args, "local_conflict_checkpoint", "") or "")
        self.local_conflict_topk = max(int(getattr(args, "local_conflict_topk", 8)), 1)
        self.local_conflict_min_detections = max(int(getattr(args, "local_conflict_min_detections", 2)), 2)
        self.local_conflict_min_committed_matches = max(
            int(getattr(args, "local_conflict_min_committed_matches", 2)),
            1,
        )
        self.local_conflict_max_detections = max(int(getattr(args, "local_conflict_max_detections", 8)), 0)
        self.local_conflict_max_tracks = max(int(getattr(args, "local_conflict_max_tracks", 32)), 0)
        self.local_conflict_cluster_gate_thresh = float(
            getattr(args, "local_conflict_cluster_gate_thresh", 0.5)
        )
        self.local_conflict_cluster_gate_temp = float(getattr(args, "local_conflict_cluster_gate_temp", 1.0))
        self.local_conflict_cluster_gate_bias = float(getattr(args, "local_conflict_cluster_gate_bias", 0.0))
        self.local_conflict_max_commits_per_cluster = max(
            int(getattr(args, "local_conflict_max_commits_per_cluster", 0)),
            0,
        )
        self.local_conflict_replacement_budget_ratio = max(
            float(getattr(args, "local_conflict_replacement_budget_ratio", 0.0)),
            0.0,
        )
        self.local_conflict_max_replaced_clusters = max(
            int(getattr(args, "local_conflict_max_replaced_clusters", 0)),
            0,
        )
        self.local_conflict_min_commit_margin = float(getattr(args, "local_conflict_min_commit_margin", 0.0))
        self.local_conflict_host_variant = str(
            getattr(args, "local_conflict_host_variant", "official_bytetrack")
        ).strip() or "official_bytetrack"
        self.posthost_oracle_data_root = str(getattr(args, "posthost_oracle_data_root", "") or "").strip()
        self.posthost_oracle_min_iou = float(getattr(args, "posthost_oracle_min_iou", 0.5))
        self.posthost_hierarchical_keep_thresh = float(
            getattr(args, "posthost_hierarchical_keep_thresh", 0.5)
        )
        self.posthost_hierarchical_swap_thresh = float(
            getattr(args, "posthost_hierarchical_swap_thresh", 0.5)
        )
        self.posthost_hierarchical_candidate_min_refined_score = float(
            getattr(args, "posthost_hierarchical_candidate_min_refined_score", 0.10)
        )
        self.posthost_hierarchical_host_summary_prior_alpha = float(
            getattr(args, "posthost_hierarchical_host_summary_prior_alpha", 0.0)
        )
        self.local_conflict_dump_dir = str(getattr(args, "local_conflict_dump_dir", "") or "").strip()
        self.local_conflict_dump_topk = max(int(getattr(args, "local_conflict_dump_topk", 8)), 0)
        self.local_conflict_dump_min_score = float(getattr(args, "local_conflict_dump_min_score", 0.0))
        self.sequence_name = ""
        self._local_conflict_dump_file = None
        self._local_conflict_dump_writer = None
        self._oracle_gt_cache_by_sequence: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}

        self.local_conflict_model: HostConditionedLocalConflictSetPredictor | None = None
        self.posthost_hierarchical_model: PosthostOneEditHierarchical | None = None
        self.local_conflict_model_family = ""
        self.local_conflict_feature_version = ""
        self.local_conflict_host_vocab = ["unknown"]
        self._local_conflict_stats_total = self._empty_local_conflict_stats()

        enabled_modes = int(self.use_local_conflict) + int(self.use_posthost_oracle_edit) + int(self.use_posthost_hierarchical_edit)
        if enabled_modes > 1:
            raise RuntimeError("Official ByteTrack tracker supports only one plugin mode at a time")
        if self.use_local_conflict:
            if not self.local_conflict_checkpoint:
                raise RuntimeError("--use-local-conflict requires --local-conflict-checkpoint")
            checkpoint_meta = torch.load(self.local_conflict_checkpoint, map_location="cpu")
            model_family = str(checkpoint_meta.get("model_family", "") or "")
            if model_family != "set_predictor_v2":
                raise RuntimeError(
                    f"Official ByteTrack plugin currently supports only set_predictor_v2, got: {model_family or 'unknown'}"
                )
            self.local_conflict_model = HostConditionedLocalConflictSetPredictor.from_checkpoint(
                self.local_conflict_checkpoint,
                map_location="cpu",
            )
            self.local_conflict_model_family = model_family
            self.local_conflict_feature_version = str(checkpoint_meta.get("feature_version", "v2_hostnorm_geom"))
            self.local_conflict_host_vocab = normalize_host_vocab(checkpoint_meta.get("host_vocab", ["unknown"]))
            if torch.cuda.is_available():
                self.local_conflict_model = self.local_conflict_model.cuda()
            self.local_conflict_model.eval()
        elif self.use_posthost_hierarchical_edit:
            if not self.local_conflict_checkpoint:
                raise RuntimeError("--use-posthost-hierarchical-edit requires --local-conflict-checkpoint")
            checkpoint_meta = torch.load(self.local_conflict_checkpoint, map_location="cpu")
            model_family = str(checkpoint_meta.get("model_family", "") or "")
            if model_family != "posthost_one_edit_hierarchical_v1":
                raise RuntimeError(
                    f"Official ByteTrack post-host hierarchical mode expects posthost_one_edit_hierarchical_v1, got: {model_family or 'unknown'}"
                )
            self.posthost_hierarchical_model = PosthostOneEditHierarchical.from_checkpoint(
                self.local_conflict_checkpoint,
                map_location="cpu",
            )
            self.local_conflict_model_family = model_family
            if torch.cuda.is_available():
                self.posthost_hierarchical_model = self.posthost_hierarchical_model.cuda()
            self.posthost_hierarchical_model.eval()
        elif self.use_posthost_oracle_edit:
            if not self.posthost_oracle_data_root:
                raise RuntimeError("--use-posthost-oracle-edit requires --posthost-oracle-data-root")

    def set_sequence_name(self, sequence_name: str) -> None:
        seq = str(sequence_name or "").strip()
        if seq == self.sequence_name:
            return
        self.close_local_conflict_dump()
        self.sequence_name = seq

    def close_local_conflict_dump(self) -> None:
        if self._local_conflict_dump_file is not None:
            try:
                self._local_conflict_dump_file.close()
            except Exception:
                pass
        self._local_conflict_dump_file = None
        self._local_conflict_dump_writer = None

    def _ensure_local_conflict_dump_writer(self) -> None:
        if not self.local_conflict_dump_dir or self._local_conflict_dump_writer is not None:
            return
        seq_name = str(self.sequence_name or "unknown_seq")
        dump_path = Path(self.local_conflict_dump_dir).resolve() / f"{seq_name}.csv"
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        self._local_conflict_dump_file = dump_path.open("a", encoding="utf-8", newline="")
        self._local_conflict_dump_writer = csv.writer(self._local_conflict_dump_file)
        if dump_path.stat().st_size == 0:
            self._local_conflict_dump_writer.writerow(
                [
                    "seq",
                    "frame",
                    "assoc_mode",
                    "group_id",
                    "candidate_count_total",
                    "det_index",
                    "track_rank",
                    "track_id",
                    "is_selected",
                    "det_score",
                    "base_score",
                    "refined_score",
                    "motion_score",
                    "track_gap",
                    "track_hist_len",
                    "det_cx",
                    "det_cy",
                    "det_w",
                    "det_h",
                    "track_cx",
                    "track_cy",
                    "track_w",
                    "track_h",
                ]
            )
            self._local_conflict_dump_file.flush()

    def _maybe_dump_runtime_candidates(
        self,
        *,
        base_scores: np.ndarray,
        refined_scores: np.ndarray,
        detections: Sequence[STrack],
        tracks: Sequence[STrack],
        det_scores: Sequence[float],
    ) -> None:
        if not self.local_conflict_dump_dir:
            return
        if len(detections) == 0 or len(tracks) == 0:
            return
        try:
            self._ensure_local_conflict_dump_writer()
            if self._local_conflict_dump_writer is None:
                return
            det_boxes = _tlbr_array_to_cxcywh([det.tlbr for det in detections])
            track_boxes = _tlbr_array_to_cxcywh([track.tlbr for track in tracks])
            score_mat = torch.as_tensor(refined_scores, dtype=torch.float32).transpose(0, 1).contiguous()
            base_mat = torch.as_tensor(base_scores, dtype=torch.float32).transpose(0, 1).contiguous()
            track_gaps = [
                max(int(self.frame_id - int(track.frame_id)), 0)
                for track in tracks
            ]
            track_hist_lens = [
                max(
                    int(getattr(track, "tracklet_len", 0) + 1),
                    int(int(track.frame_id) - int(track.start_frame) + 1),
                )
                for track in tracks
            ]
            assoc_mode = "official_bytetrack_first_stage"

            for det_idx in range(int(score_mat.shape[0])):
                row = score_mat[det_idx]
                candidate_count_total = int(row.numel())
                if candidate_count_total <= 0:
                    continue
                if self.local_conflict_dump_topk <= 0:
                    topk = candidate_count_total
                else:
                    topk = min(int(self.local_conflict_dump_topk), candidate_count_total)
                top_indices = torch.topk(row, k=topk, dim=0, sorted=True).indices.tolist()
                group_id = f"{self.sequence_name or ''}:{int(self.frame_id)}:{int(det_idx)}"
                det_box = det_boxes[det_idx]
                det_score = float(det_scores[det_idx]) if det_idx < len(det_scores) else 0.0

                for rank, track_idx in enumerate(top_indices, start=1):
                    refined_val = float(score_mat[det_idx, track_idx].item())
                    if refined_val < float(self.local_conflict_dump_min_score):
                        continue
                    track_box = track_boxes[track_idx]
                    self._local_conflict_dump_writer.writerow(
                        [
                            str(self.sequence_name or ""),
                            int(self.frame_id),
                            assoc_mode,
                            group_id,
                            candidate_count_total,
                            int(det_idx),
                            int(rank),
                            int(tracks[track_idx].track_id),
                            0,
                            det_score,
                            float(base_mat[det_idx, track_idx].item()),
                            refined_val,
                            0.0,
                            int(track_gaps[track_idx]),
                            int(track_hist_lens[track_idx]),
                            float(det_box[0].item()),
                            float(det_box[1].item()),
                            float(det_box[2].item()),
                            float(det_box[3].item()),
                            float(track_box[0].item()),
                            float(track_box[1].item()),
                            float(track_box[2].item()),
                            float(track_box[3].item()),
                        ]
                    )
            self._local_conflict_dump_file.flush()
        except Exception:
            self.close_local_conflict_dump()

    @staticmethod
    def _empty_local_conflict_stats() -> Dict[str, int]:
        return {
            "eligible_clusters": 0,
            "replaced_clusters": 0,
            "host_same_commit_clusters": 0,
            "delta_replaced_clusters": 0,
            "resolved_dets": 0,
            "matched_dets": 0,
            "delta_commit_pairs": 0,
            "delta_drop_pairs": 0,
            "null_dets": 0,
            "deferred_dets": 0,
            "blocked_tracks": 0,
            "gate_pass_clusters": 0,
            "gate_filtered_clusters": 0,
            "trigger_filtered_clusters": 0,
            "skipped_large_clusters": 0,
            "budget_filtered_clusters": 0,
            "margin_filtered_pairs": 0,
            "capped_commit_pairs": 0,
            "all_defer_clusters": 0,
            "empty_pair_candidate_clusters": 0,
            "post_filter_empty_clusters": 0,
            "posthost_selected_clusters": 0,
            "posthost_swap_clusters": 0,
            "posthost_add_clusters": 0,
            "posthost_defer_clusters": 0,
        }

    def get_local_conflict_diagnostics(self) -> Dict[str, Any]:
        if self.use_local_conflict:
            graph_mode = "learned_commit"
        elif self.use_posthost_oracle_edit:
            graph_mode = "posthost_one_edit_oracle"
        elif self.use_posthost_hierarchical_edit:
            graph_mode = "posthost_one_edit_hierarchical"
        else:
            graph_mode = "disabled"
        return {
            "enabled": bool(self.use_local_conflict or self.use_posthost_oracle_edit),
            "graph_mode": graph_mode,
            "graph_checkpoint": self.local_conflict_checkpoint if self.use_local_conflict else "",
            "graph_topk": int(self.local_conflict_topk),
            "graph_min_detections": int(self.local_conflict_min_detections),
            "graph_min_committed_matches": int(self.local_conflict_min_committed_matches),
            "graph_max_detections": int(self.local_conflict_max_detections),
            "graph_max_tracks": int(self.local_conflict_max_tracks),
            "graph_cluster_gate_thresh": float(self.local_conflict_cluster_gate_thresh),
            "graph_cluster_gate_temp": float(self.local_conflict_cluster_gate_temp),
            "graph_cluster_gate_bias": float(self.local_conflict_cluster_gate_bias),
            "graph_max_commits_per_cluster": int(self.local_conflict_max_commits_per_cluster),
            "graph_replacement_budget_ratio": float(self.local_conflict_replacement_budget_ratio),
            "graph_max_replaced_clusters": int(self.local_conflict_max_replaced_clusters),
            "graph_min_commit_margin": float(self.local_conflict_min_commit_margin),
            "host_variant": self.local_conflict_host_variant,
            "posthost_oracle_min_iou": float(self.posthost_oracle_min_iou),
            "posthost_hierarchical_keep_thresh": float(self.posthost_hierarchical_keep_thresh),
            "posthost_hierarchical_swap_thresh": float(self.posthost_hierarchical_swap_thresh),
            "posthost_hierarchical_candidate_min_refined_score": float(
                self.posthost_hierarchical_candidate_min_refined_score
            ),
            "posthost_hierarchical_host_summary_prior_alpha": float(
                self.posthost_hierarchical_host_summary_prior_alpha
            ),
            "model_family": self.local_conflict_model_family,
            "feature_version": self.local_conflict_feature_version,
            **{key: int(value) for key, value in self._local_conflict_stats_total.items()},
        }

    def _accumulate_local_conflict_stats(self, stats: Dict[str, Any]) -> None:
        for key in self._local_conflict_stats_total.keys():
            self._local_conflict_stats_total[key] += int(stats.get(key, 0) or 0)

    @staticmethod
    def _posthost_zero_pair_stats() -> Dict[str, float]:
        return {
            "refined_score": 0.0,
            "base_score": 0.0,
            "iou": 0.0,
            "bbox_dist": 0.0,
            "row_degree": 0.0,
            "row_margin": 0.0,
            "row_entropy": 0.0,
            "col_degree": 0.0,
            "track_gap": 0.0,
            "track_hist": 0.0,
            "delta_cx": 0.0,
            "delta_cy": 0.0,
            "delta_log_w": 0.0,
            "delta_log_h": 0.0,
        }

    @staticmethod
    def _posthost_action_one_hot(action_type: str) -> List[float]:
        action = str(action_type)
        return [
            1.0 if action == "keep" else 0.0,
            1.0 if action == "add" else 0.0,
            1.0 if action == "swap" else 0.0,
            1.0 if action == "defer" else 0.0,
        ]

    @staticmethod
    def _posthost_mean_pool(rows: Sequence[Sequence[float]], dim: int) -> List[float]:
        if not rows:
            return [0.0 for _ in range(dim)]
        tensor = torch.tensor(rows, dtype=torch.float32)
        return [float(x) for x in tensor.mean(dim=0).tolist()]

    @staticmethod
    def _posthost_max_pool(rows: Sequence[Sequence[float]], dim: int) -> List[float]:
        if not rows:
            return [0.0 for _ in range(dim)]
        tensor = torch.tensor(rows, dtype=torch.float32)
        return [float(x) for x in tensor.max(dim=0).values.tolist()]

    @classmethod
    def _build_posthost_cluster_feature(
        cls,
        candidate_features: Sequence[Sequence[float]],
        candidate_action_types: Sequence[str],
    ) -> List[float]:
        feature_dim = int(len(candidate_features[0]))
        keep_feature = list(candidate_features[0])
        defer_rows = [
            candidate_features[idx]
            for idx, action in enumerate(candidate_action_types)
            if str(action) == "defer"
        ]
        swap_rows = [
            candidate_features[idx]
            for idx, action in enumerate(candidate_action_types)
            if str(action) == "swap"
        ]
        add_rows = [
            candidate_features[idx]
            for idx, action in enumerate(candidate_action_types)
            if str(action) == "add"
        ]
        cluster_feature = (
            keep_feature
            + cls._posthost_mean_pool(defer_rows, feature_dim)
            + cls._posthost_max_pool(defer_rows, feature_dim)
            + cls._posthost_mean_pool(swap_rows, feature_dim)
            + cls._posthost_max_pool(swap_rows, feature_dim)
            + [
                float(len(defer_rows)),
                float(len(swap_rows)),
                float(len(add_rows)),
                float(len(candidate_features)),
            ]
        )
        return [float(x) for x in cluster_feature]

    @staticmethod
    def _build_posthost_pair_lookup(feature_pack: Dict[str, Any]) -> Dict[tuple[int, int], Dict[str, float]]:
        lookup: Dict[tuple[int, int], Dict[str, float]] = {}
        det_features = feature_pack["det_features"].detach().cpu()
        track_features = feature_pack["track_features"].detach().cpu()
        edge_features = feature_pack["edge_features"].detach().cpu()
        edge_det_index = feature_pack["edge_det_index"].detach().cpu().tolist()
        edge_track_index = feature_pack["edge_track_index"].detach().cpu().tolist()
        for edge_idx, (det_idx, track_idx) in enumerate(zip(edge_det_index, edge_track_index)):
            edge = edge_features[edge_idx]
            lookup[(int(det_idx), int(track_idx))] = {
                "base_score": float(edge[0].item()),
                "refined_score": float(edge[1].item()),
                "iou": float(edge[12].item()),
                "bbox_dist": float(edge[13].item()),
                "delta_cx": float(edge[14].item()),
                "delta_cy": float(edge[15].item()),
                "delta_log_w": float(edge[16].item()),
                "delta_log_h": float(edge[17].item()),
                "row_degree": float(det_features[int(det_idx), 1].item()),
                "row_margin": float(det_features[int(det_idx), 2].item()),
                "row_entropy": float(det_features[int(det_idx), 3].item()),
                "col_degree": float(track_features[int(track_idx), 2].item()),
                "track_gap": float(track_features[int(track_idx), 0].item()),
                "track_hist": float(track_features[int(track_idx), 1].item()),
            }
        return lookup

    def _build_posthost_host_summary(
        self,
        host_pairs_local: Sequence[tuple[int, int]],
    ) -> Dict[str, float]:
        # Runtime inference has no GT. Use a weak prior instead of hard-zeroing the
        # GT-only fields, which collapses the hierarchical gate fully open.
        pair_count = float(len(host_pairs_local))
        alpha = max(0.0, min(float(self.posthost_hierarchical_host_summary_prior_alpha), 1.0))
        positive_prior = float(alpha * pair_count)
        return {
            "pair_count": pair_count,
            "positive_count": positive_prior,
            "negative_count": 0.0,
            "score": positive_prior,
        }

    @classmethod
    def _build_posthost_candidate_feature(
        cls,
        *,
        num_detections: int,
        num_tracks: int,
        num_edges: int,
        is_large_component: int,
        host_summary: Dict[str, float],
        action_type: str,
        add_pair: tuple[int, int] | None,
        remove_pair: tuple[int, int] | None,
        pair_lookup: Dict[tuple[int, int], Dict[str, float]],
    ) -> List[float]:
        add_stats = dict(pair_lookup.get(add_pair, cls._posthost_zero_pair_stats()))
        remove_stats = dict(pair_lookup.get(remove_pair, cls._posthost_zero_pair_stats()))
        delta_stats = {
            "refined_score": float(add_stats["refined_score"] - remove_stats["refined_score"]),
            "base_score": float(add_stats["base_score"] - remove_stats["base_score"]),
            "iou": float(add_stats["iou"] - remove_stats["iou"]),
            "bbox_dist": float(add_stats["bbox_dist"] - remove_stats["bbox_dist"]),
            "row_margin": float(add_stats["row_margin"] - remove_stats["row_margin"]),
            "row_entropy": float(add_stats["row_entropy"] - remove_stats["row_entropy"]),
            "track_gap": float(add_stats["track_gap"] - remove_stats["track_gap"]),
            "track_hist": float(add_stats["track_hist"] - remove_stats["track_hist"]),
        }
        values = (
            cls._posthost_action_one_hot(action_type)
            + [
                float(num_detections),
                float(num_tracks),
                float(num_edges),
                float(is_large_component),
                float(host_summary["pair_count"]),
                float(host_summary["positive_count"]),
                float(host_summary["negative_count"]),
                float(host_summary["score"]),
                float(add_stats["refined_score"]),
                float(add_stats["base_score"]),
                float(add_stats["iou"]),
                float(add_stats["bbox_dist"]),
                float(add_stats["row_degree"]),
                float(add_stats["row_margin"]),
                float(add_stats["row_entropy"]),
                float(add_stats["col_degree"]),
                float(add_stats["track_gap"]),
                float(add_stats["track_hist"]),
                float(add_stats["delta_cx"]),
                float(add_stats["delta_cy"]),
                float(add_stats["delta_log_w"]),
                float(add_stats["delta_log_h"]),
                float(remove_stats["refined_score"]),
                float(remove_stats["base_score"]),
                float(remove_stats["iou"]),
                float(remove_stats["bbox_dist"]),
                float(remove_stats["row_degree"]),
                float(remove_stats["row_margin"]),
                float(remove_stats["row_entropy"]),
                float(remove_stats["col_degree"]),
                float(remove_stats["track_gap"]),
                float(remove_stats["track_hist"]),
                float(remove_stats["delta_cx"]),
                float(remove_stats["delta_cy"]),
                float(remove_stats["delta_log_w"]),
                float(remove_stats["delta_log_h"]),
                float(delta_stats["refined_score"]),
                float(delta_stats["base_score"]),
                float(delta_stats["iou"]),
                float(delta_stats["bbox_dist"]),
                float(delta_stats["row_margin"]),
                float(delta_stats["row_entropy"]),
                float(delta_stats["track_gap"]),
                float(delta_stats["track_hist"]),
                float(1 if remove_pair is not None else 0),
            ]
        )
        return [float(x) for x in values]

    def _enumerate_posthost_hierarchical_candidates(
        self,
        *,
        feature_pack: Dict[str, Any],
        host_pairs_local: Sequence[tuple[int, int]],
        is_large_component: int,
    ) -> List[Dict[str, Any]]:
        pair_lookup = self._build_posthost_pair_lookup(feature_pack)
        host_summary = self._build_posthost_host_summary(host_pairs_local)
        host_pair_set = {tuple(int(v) for v in pair) for pair in host_pairs_local}
        host_det_to_track = {int(det_idx): int(track_idx) for det_idx, track_idx in host_pair_set}
        host_track_to_det = {int(track_idx): int(det_idx) for det_idx, track_idx in host_pair_set}
        candidates: List[Dict[str, Any]] = [
            {
                "action_type": "keep",
                "add_pair": None,
                "remove_pair": None,
                "candidate_features": self._build_posthost_candidate_feature(
                    num_detections=int(feature_pack["det_features"].shape[0]),
                    num_tracks=int(feature_pack["track_features"].shape[0]),
                    num_edges=int(feature_pack["edge_features"].shape[0]),
                    is_large_component=int(is_large_component),
                    host_summary=host_summary,
                    action_type="keep",
                    add_pair=None,
                    remove_pair=None,
                    pair_lookup=pair_lookup,
                ),
            }
        ]

        for pair, edge in sorted(pair_lookup.items()):
            det_idx, track_idx = int(pair[0]), int(pair[1])
            refined_score = float(edge["refined_score"])
            if pair not in host_pair_set and refined_score >= float(self.posthost_hierarchical_candidate_min_refined_score):
                remove_pairs: set[tuple[int, int]] = set()
                existing_track = host_det_to_track.get(int(det_idx))
                if existing_track is not None and int(existing_track) != int(track_idx):
                    remove_pairs.add((int(det_idx), int(existing_track)))
                existing_det = host_track_to_det.get(int(track_idx))
                if existing_det is not None and int(existing_det) != int(det_idx):
                    remove_pairs.add((int(existing_det), int(track_idx)))
                if len(remove_pairs) > 1:
                    continue
                remove_pair = next(iter(remove_pairs)) if remove_pairs else None
                action_type = "swap" if remove_pair is not None else "add"
                candidates.append(
                    {
                        "action_type": action_type,
                        "add_pair": (int(det_idx), int(track_idx)),
                        "remove_pair": remove_pair,
                        "candidate_features": self._build_posthost_candidate_feature(
                            num_detections=int(feature_pack["det_features"].shape[0]),
                            num_tracks=int(feature_pack["track_features"].shape[0]),
                            num_edges=int(feature_pack["edge_features"].shape[0]),
                            is_large_component=int(is_large_component),
                            host_summary=host_summary,
                            action_type=action_type,
                            add_pair=(int(det_idx), int(track_idx)),
                            remove_pair=remove_pair,
                            pair_lookup=pair_lookup,
                        ),
                    }
                )

        for det_idx, track_idx in sorted(host_pair_set):
            candidates.append(
                {
                    "action_type": "defer",
                    "add_pair": None,
                    "remove_pair": (int(det_idx), int(track_idx)),
                    "candidate_features": self._build_posthost_candidate_feature(
                        num_detections=int(feature_pack["det_features"].shape[0]),
                        num_tracks=int(feature_pack["track_features"].shape[0]),
                        num_edges=int(feature_pack["edge_features"].shape[0]),
                        is_large_component=int(is_large_component),
                        host_summary=host_summary,
                        action_type="defer",
                        add_pair=None,
                        remove_pair=(int(det_idx), int(track_idx)),
                        pair_lookup=pair_lookup,
                    ),
                }
            )
        return candidates

    def _predict_posthost_hierarchical_candidate(
        self,
        *,
        feature_pack: Dict[str, Any],
        host_pairs_local: Sequence[tuple[int, int]],
        is_large_component: int,
    ) -> Dict[str, Any]:
        result = {
            "selected_candidate": None,
            "decision": "keep",
            "gate_prob": 0.0,
            "swap_prob": 0.0,
        }
        if self.posthost_hierarchical_model is None:
            return result
        candidates = self._enumerate_posthost_hierarchical_candidates(
            feature_pack=feature_pack,
            host_pairs_local=host_pairs_local,
            is_large_component=int(is_large_component),
        )
        if len(candidates) <= 1:
            result["decision"] = "no_candidate"
            return result

        candidate_action_types = [str(candidate["action_type"]) for candidate in candidates]
        candidate_features = [candidate["candidate_features"] for candidate in candidates]
        cluster_feature = self._build_posthost_cluster_feature(candidate_features, candidate_action_types)
        device = next(self.posthost_hierarchical_model.parameters()).device
        cluster_tensor = torch.tensor(cluster_feature, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.inference_mode():
            gate_prob = float(torch.sigmoid(self.posthost_hierarchical_model.keep_edit_gate(cluster_tensor)).item())
            result["gate_prob"] = gate_prob
            if gate_prob < float(self.posthost_hierarchical_keep_thresh):
                result["decision"] = "gate_keep"
                return result

            swap_prob = float(torch.sigmoid(self.posthost_hierarchical_model.defer_swap_selector(cluster_tensor)).item())
            result["swap_prob"] = swap_prob

            defer_indices = [
                idx for idx, action in enumerate(candidate_action_types)
                if str(action) == "defer"
            ]
            swap_indices = [
                idx for idx, action in enumerate(candidate_action_types)
                if str(action) == "swap"
            ]

            choose_swap = swap_prob >= float(self.posthost_hierarchical_swap_thresh)
            if choose_swap and swap_indices:
                active_indices = swap_indices
                active_action = "swap"
                ranker = self.posthost_hierarchical_model.swap_ranker
            elif defer_indices:
                active_indices = defer_indices
                active_action = "defer"
                ranker = self.posthost_hierarchical_model.defer_ranker
            elif swap_indices:
                active_indices = swap_indices
                active_action = "swap"
                ranker = self.posthost_hierarchical_model.swap_ranker
            else:
                result["decision"] = "no_supported_action"
                return result

            ranker_features = torch.tensor(
                [candidate_features[idx] for idx in active_indices],
                dtype=torch.float32,
                device=device,
            )
            local_best = int(ranker(ranker_features).argmax().item())
            best_index = int(active_indices[local_best])
            result["selected_candidate"] = dict(candidates[best_index])
            result["decision"] = str(active_action)
            return result

    @staticmethod
    def _pairwise_iou_matrix(boxes_a: Sequence[np.ndarray], boxes_b: Sequence[np.ndarray]) -> np.ndarray:
        if len(boxes_a) == 0 or len(boxes_b) == 0:
            return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
        a = np.asarray([np.asarray(box, dtype=np.float32) for box in boxes_a], dtype=np.float32)
        b = np.asarray([np.asarray(box, dtype=np.float32) for box in boxes_b], dtype=np.float32)
        ax1 = a[:, 0:1]
        ay1 = a[:, 1:2]
        ax2 = a[:, 2:3]
        ay2 = a[:, 3:4]
        bx1 = b[:, 0][None, :]
        by1 = b[:, 1][None, :]
        bx2 = b[:, 2][None, :]
        by2 = b[:, 3][None, :]
        inter_x1 = np.maximum(ax1, bx1)
        inter_y1 = np.maximum(ay1, by1)
        inter_x2 = np.minimum(ax2, bx2)
        inter_y2 = np.minimum(ay2, by2)
        inter_w = np.clip(inter_x2 - inter_x1, a_min=0.0, a_max=None)
        inter_h = np.clip(inter_y2 - inter_y1, a_min=0.0, a_max=None)
        inter_area = inter_w * inter_h
        area_a = np.clip((ax2 - ax1), a_min=0.0, a_max=None) * np.clip((ay2 - ay1), a_min=0.0, a_max=None)
        area_b = np.clip((bx2 - bx1), a_min=0.0, a_max=None) * np.clip((by2 - by1), a_min=0.0, a_max=None)
        union = np.clip(area_a + area_b - inter_area, a_min=1e-6, a_max=None)
        return (inter_area / union).astype(np.float32)

    def _load_sequence_oracle_gt(self, sequence_name: str) -> Dict[int, List[Dict[str, Any]]]:
        seq = str(sequence_name or "").strip()
        if not seq:
            return {}
        cached = self._oracle_gt_cache_by_sequence.get(seq)
        if cached is not None:
            return cached
        frames: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        if not self.posthost_oracle_data_root:
            self._oracle_gt_cache_by_sequence[seq] = {}
            return {}
        seq_root = Path(self.posthost_oracle_data_root) / "MOT17" / "train" / seq / "gt"
        gt_candidates = [seq_root / "gt_val_half.txt", seq_root / "gt.txt"]
        gt_path = next((path for path in gt_candidates if path.is_file()), None)
        if gt_path is None:
            self._oracle_gt_cache_by_sequence[seq] = {}
            return {}
        for line in gt_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 6:
                continue
            try:
                frame_id = int(float(parts[0]))
                gt_id = int(float(parts[1]))
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
                mark = int(float(parts[6])) if len(parts) > 6 else 1
                label = int(float(parts[7])) if len(parts) > 7 else 1
            except Exception:
                continue
            if frame_id <= 0 or gt_id <= 0 or mark <= 0 or label != 1:
                continue
            frames[int(frame_id)].append(
                {
                    "gt_id": int(gt_id),
                    "tlbr": np.asarray([x, y, x + w, y + h], dtype=np.float32),
                }
            )
        loaded = {int(frame_id): list(rows) for frame_id, rows in frames.items()}
        self._oracle_gt_cache_by_sequence[seq] = loaded
        return loaded

    def _annotate_detection_gt_ids(self, detections: Sequence[STrack]) -> List[int]:
        if not self.use_posthost_oracle_edit or not detections:
            return [-1 for _ in detections]
        frame_gt = self._load_sequence_oracle_gt(self.sequence_name).get(int(self.frame_id), [])
        if not frame_gt:
            for det in detections:
                setattr(det, "oracle_gt_id", -1)
            return [-1 for _ in detections]
        det_boxes = [np.asarray(det.tlbr, dtype=np.float32) for det in detections]
        gt_boxes = [np.asarray(row["tlbr"], dtype=np.float32) for row in frame_gt]
        iou_mat = self._pairwise_iou_matrix(det_boxes, gt_boxes)
        assignments: List[int] = [-1 for _ in detections]
        candidates: List[tuple[float, int, int]] = []
        for det_idx in range(iou_mat.shape[0]):
            for gt_idx in range(iou_mat.shape[1]):
                iou_val = float(iou_mat[det_idx, gt_idx])
                if iou_val >= float(self.posthost_oracle_min_iou):
                    candidates.append((iou_val, int(det_idx), int(gt_idx)))
        candidates.sort(key=lambda item: (float(item[0]), -int(item[1]), -int(item[2])), reverse=True)
        used_dets: set[int] = set()
        used_gts: set[int] = set()
        for _, det_idx, gt_idx in candidates:
            if det_idx in used_dets or gt_idx in used_gts:
                continue
            assignments[det_idx] = int(frame_gt[gt_idx]["gt_id"])
            used_dets.add(int(det_idx))
            used_gts.add(int(gt_idx))
        for det_idx, det in enumerate(detections):
            setattr(det, "oracle_gt_id", int(assignments[det_idx]))
        return assignments

    @staticmethod
    def _update_track_oracle_identity(track: STrack, detection: STrack) -> None:
        gt_id = int(getattr(detection, "oracle_gt_id", -1) or -1)
        if gt_id <= 0:
            return
        counts = dict(getattr(track, "oracle_gt_counts", {}) or {})
        counts[int(gt_id)] = int(counts.get(int(gt_id), 0)) + 1
        setattr(track, "oracle_gt_counts", counts)
        best_gt_id = max(counts.items(), key=lambda item: (int(item[1]), -int(item[0])))[0]
        setattr(track, "oracle_gt_id", int(best_gt_id))

    @staticmethod
    def _score_pair_set_against_oracle(
        pair_set: Sequence[tuple[int, int]],
        *,
        det_gt_ids: Sequence[int],
        track_gt_ids: Sequence[int],
    ) -> Dict[str, float]:
        correct = 0
        wrong = 0
        for det_idx, track_idx in pair_set:
            if not (0 <= int(det_idx) < len(det_gt_ids) and 0 <= int(track_idx) < len(track_gt_ids)):
                continue
            det_gt = int(det_gt_ids[int(det_idx)])
            track_gt = int(track_gt_ids[int(track_idx)])
            if det_gt <= 0 or track_gt <= 0:
                continue
            if det_gt == track_gt:
                correct += 1
            else:
                wrong += 1
        return {
            "score": float(correct - wrong),
            "correct": float(correct),
            "wrong": float(wrong),
        }

    def _build_local_conflict_runtime_features_v2(
        self,
        *,
        base_scores: torch.Tensor,
        score_mat: torch.Tensor,
        det_scores: torch.Tensor,
        track_gaps: torch.Tensor,
        track_hist_lens: torch.Tensor,
        det_boxes_cxcywh: torch.Tensor,
        track_boxes_cxcywh: torch.Tensor,
    ) -> Dict[str, Any]:
        device = score_mat.device
        dtype = torch.float32
        num_local_dets = int(score_mat.shape[0])
        num_local_tracks = int(score_mat.shape[1])
        topk = int(max(min(self.local_conflict_topk, num_local_tracks), 1))

        top_vals, top_idx = torch.topk(score_mat, k=topk, dim=1, sorted=True)
        top_base = torch.gather(base_scores, dim=1, index=top_idx)
        top_motion = torch.zeros_like(top_vals)

        det_features: List[List[float]] = []
        edge_records: List[Dict[str, Any]] = []
        dense_edge_mask = torch.zeros((num_local_dets, num_local_tracks), device=device, dtype=torch.bool)
        dense_refined_scores = torch.zeros((num_local_dets, num_local_tracks), device=device, dtype=dtype)
        row_entropy_values: List[float] = []
        row_margin_values: List[float] = []
        track_gap_acc: Dict[int, List[float]] = defaultdict(list)
        track_hist_acc: Dict[int, List[float]] = defaultdict(list)
        col_refined_scores: Dict[int, List[float]] = defaultdict(list)

        for det_local_idx in range(num_local_dets):
            row_refined = top_vals[det_local_idx].to(dtype=torch.float32)
            row_base = top_base[det_local_idx].to(dtype=torch.float32)
            row_motion = top_motion[det_local_idx].to(dtype=torch.float32)
            row_probs = softmax_probs_1d(row_refined)
            row_entropy = float((-(row_probs * torch.log(row_probs.clamp(min=1e-8))).sum()).item())
            row_margin = 0.0
            if row_refined.numel() > 1:
                row_margin = float((row_refined[0] - row_refined[1]).item())
            row_entropy_values.append(row_entropy)
            row_margin_values.append(row_margin)

            det_box = det_boxes_cxcywh[det_local_idx]
            det_cx = float(det_box[0].item())
            det_cy = float(det_box[1].item())
            det_w = max(float(det_box[2].item()), 1e-6)
            det_h = max(float(det_box[3].item()), 1e-6)
            det_features.append(
                [
                    float(det_scores[det_local_idx].item()) if det_scores.numel() > det_local_idx else 0.0,
                    0.0,
                    row_margin,
                    row_entropy,
                    det_cx,
                    det_cy,
                    float(np.log(det_w)),
                    float(np.log(det_h)),
                    float(det_w / max(det_h, 1e-6)),
                ]
            )

            row_base_z = zscore_1d(row_base)
            row_refined_z = zscore_1d(row_refined)
            row_motion_z = zscore_1d(row_motion)
            row_top1 = float(row_refined.max().item()) if row_refined.numel() > 0 else 0.0

            for local_rank in range(int(top_vals.shape[1])):
                track_local_idx = int(top_idx[det_local_idx, local_rank].item())
                score_val = float(top_vals[det_local_idx, local_rank].item())
                if score_val <= 0.0:
                    continue
                dense_edge_mask[det_local_idx, track_local_idx] = True
                dense_refined_scores[det_local_idx, track_local_idx] = score_mat[det_local_idx, track_local_idx]
                track_gap_acc[track_local_idx].append(float(track_gaps[track_local_idx].item()))
                track_hist_acc[track_local_idx].append(float(track_hist_lens[track_local_idx].item()))

                geom = pair_geometry_features(
                    det_box.view(1, 4),
                    track_boxes_cxcywh[track_local_idx].view(1, 4),
                )
                edge_records.append(
                    {
                        "det_local_idx": int(det_local_idx),
                        "track_local_idx": int(track_local_idx),
                        "base_score_raw": float(top_base[det_local_idx, local_rank].item()),
                        "refined_score_raw": score_val,
                        "motion_score_raw": 0.0,
                        "base_score_row_z": float(row_base_z[local_rank].item()),
                        "refined_score_row_z": float(row_refined_z[local_rank].item()),
                        "motion_score_row_z": float(row_motion_z[local_rank].item()),
                        "refined_score_row_softmax": float(row_probs[local_rank].item()),
                        "refined_gap_to_row_top1": float(row_top1 - score_val),
                        "rank_frac": float(local_rank + 1) / float(max(int(top_vals.shape[1]), 1)),
                        "iou": float(geom["iou"].view(-1)[0].item()),
                        "bbox_dist_score": float(geom["bbox_dist_score"].view(-1)[0].item()),
                        "delta_cx_norm": float(geom["delta_cx_norm"].view(-1)[0].item()),
                        "delta_cy_norm": float(geom["delta_cy_norm"].view(-1)[0].item()),
                        "delta_log_w": float(geom["delta_log_w"].view(-1)[0].item()),
                        "delta_log_h": float(geom["delta_log_h"].view(-1)[0].item()),
                    }
                )
                col_refined_scores[track_local_idx].append(score_val)

        edge_det_index = [int(record["det_local_idx"]) for record in edge_records]
        edge_track_index = [int(record["track_local_idx"]) for record in edge_records]
        row_degree = torch.zeros((num_local_dets,), device=device, dtype=dtype)
        col_degree = torch.zeros((num_local_tracks,), device=device, dtype=dtype)
        for det_local_idx, track_local_idx in zip(edge_det_index, edge_track_index):
            row_degree[det_local_idx] += 1.0
            col_degree[track_local_idx] += 1.0
        for det_local_idx in range(num_local_dets):
            det_features[det_local_idx][1] = float(row_degree[det_local_idx].item())

        track_features: List[List[float]] = []
        for local_track_idx in range(num_local_tracks):
            track_box = track_boxes_cxcywh[local_track_idx]
            track_cx = float(track_box[0].item())
            track_cy = float(track_box[1].item())
            track_w = max(float(track_box[2].item()), 1e-6)
            track_h = max(float(track_box[3].item()), 1e-6)
            gap_vals = track_gap_acc.get(local_track_idx, [])
            hist_vals = track_hist_acc.get(local_track_idx, [])
            gap_mean = float(sum(gap_vals) / len(gap_vals)) if gap_vals else float(track_gaps[local_track_idx].item())
            hist_mean = (
                float(sum(hist_vals) / len(hist_vals))
                if hist_vals
                else float(track_hist_lens[local_track_idx].item())
            )
            track_features.append(
                [
                    float(np.log1p(max(gap_mean, 0.0))),
                    float(np.log1p(max(hist_mean, 0.0))),
                    float(col_degree[local_track_idx].item()),
                    track_cx,
                    track_cy,
                    float(np.log(track_w)),
                    float(np.log(track_h)),
                    float(track_w / max(track_h, 1e-6)),
                ]
            )

        col_z_by_track: Dict[int, List[float]] = {}
        for local_track_idx, scores_for_track in col_refined_scores.items():
            score_tensor = torch.tensor(scores_for_track, device=device, dtype=torch.float32)
            col_z = zscore_1d(score_tensor)
            col_z_by_track[local_track_idx] = [float(x) for x in col_z.tolist()]

        col_offsets: Dict[int, int] = defaultdict(int)
        edge_features: List[List[float]] = []
        for record in edge_records:
            local_track_idx = int(record["track_local_idx"])
            local_offset = int(col_offsets[local_track_idx])
            col_offsets[local_track_idx] += 1
            refined_score_col_z = 0.0
            if local_track_idx in col_z_by_track and local_offset < len(col_z_by_track[local_track_idx]):
                refined_score_col_z = float(col_z_by_track[local_track_idx][local_offset])
            edge_features.append(
                [
                    float(record["base_score_raw"]),
                    float(record["refined_score_raw"]),
                    float(record["motion_score_raw"]),
                    float(record["base_score_row_z"]),
                    float(record["refined_score_row_z"]),
                    float(record["motion_score_row_z"]),
                    float(record["refined_score_row_softmax"]),
                    float(record["refined_gap_to_row_top1"]),
                    float(record["rank_frac"]),
                    refined_score_col_z,
                    float(record["refined_score_raw"] - record["base_score_raw"]),
                    float(record["motion_score_raw"] - record["refined_score_raw"]),
                    float(record["iou"]),
                    float(record["bbox_dist_score"]),
                    float(record["delta_cx_norm"]),
                    float(record["delta_cy_norm"]),
                    float(record["delta_log_w"]),
                    float(record["delta_log_h"]),
                ]
            )

        cluster_features = torch.tensor(
            [
                float(num_local_dets),
                float(num_local_tracks),
                float(len(edge_features)),
                float(row_degree.mean().item()) if row_degree.numel() > 0 else 0.0,
                float(row_degree.max().item()) if row_degree.numel() > 0 else 0.0,
                float(col_degree.mean().item()) if col_degree.numel() > 0 else 0.0,
                float(col_degree.max().item()) if col_degree.numel() > 0 else 0.0,
                float(np.mean(row_entropy_values)) if row_entropy_values else 0.0,
                float(np.max(row_entropy_values)) if row_entropy_values else 0.0,
                float(np.mean(row_margin_values)) if row_margin_values else 0.0,
                float(np.max(row_margin_values)) if row_margin_values else 0.0,
            ],
            device=device,
            dtype=dtype,
        )

        if edge_features:
            edge_features_tensor = torch.tensor(edge_features, device=device, dtype=dtype)
            edge_det_index_tensor = torch.tensor(edge_det_index, device=device, dtype=torch.long)
            edge_track_index_tensor = torch.tensor(edge_track_index, device=device, dtype=torch.long)
        else:
            edge_features_tensor = torch.zeros((0, 18), device=device, dtype=dtype)
            edge_det_index_tensor = torch.zeros((0,), device=device, dtype=torch.long)
            edge_track_index_tensor = torch.zeros((0,), device=device, dtype=torch.long)

        host_variant_id = int(encode_host_variant(self.local_conflict_host_variant, self.local_conflict_host_vocab))
        return {
            "det_features": torch.tensor(det_features, device=device, dtype=dtype),
            "track_features": torch.tensor(track_features, device=device, dtype=dtype),
            "edge_features": edge_features_tensor,
            "edge_det_index": edge_det_index_tensor,
            "edge_track_index": edge_track_index_tensor,
            "cluster_features": cluster_features,
            "dense_edge_mask": dense_edge_mask,
            "dense_refined_scores": dense_refined_scores,
            "host_variant_id": host_variant_id,
        }

    def _get_local_conflict_plan(
        self,
        *,
        iou_cost: np.ndarray,
        refined_cost: np.ndarray,
        detections: Sequence[STrack],
        tracks: Sequence[STrack],
        thresh: float,
    ) -> Dict[str, Any]:
        plan = {
            "assignments": [],
            "resolved_dets": set(),
            "blocked_tracks": set(),
            "stats": self._empty_local_conflict_stats(),
        }
        if (
            not self.use_local_conflict
            or self.local_conflict_model is None
            or len(detections) == 0
            or len(tracks) == 0
        ):
            return plan

        device = next(self.local_conflict_model.parameters()).device
        score_mat = torch.as_tensor(1.0 - refined_cost, device=device, dtype=torch.float32).transpose(0, 1).contiguous()
        base_scores = torch.as_tensor(1.0 - iou_cost, device=device, dtype=torch.float32).transpose(0, 1).contiguous()
        det_scores = torch.tensor([float(det.score) for det in detections], device=device, dtype=torch.float32)
        det_boxes_cxcywh = _tlbr_array_to_cxcywh([det.tlbr for det in detections]).to(device=device)
        track_boxes_cxcywh = _tlbr_array_to_cxcywh([track.tlbr for track in tracks]).to(device=device)
        track_gaps = torch.tensor(
            [max(float(self.frame_id - int(track.frame_id)), 0.0) for track in tracks],
            device=device,
            dtype=torch.float32,
        )
        track_hist_lens = torch.tensor(
            [
                max(
                    float(getattr(track, "tracklet_len", 0) + 1),
                    float(int(track.frame_id) - int(track.start_frame) + 1),
                )
                for track in tracks
            ],
            device=device,
            dtype=torch.float32,
        )

        components = build_topk_bipartite_components(
            score_mat=score_mat,
            topk=int(self.local_conflict_topk),
            min_edge_score=0.0,
        )
        eligible_components, skipped_large = filter_local_conflict_clusters_by_size(
            components,
            min_detections=int(self.local_conflict_min_detections),
            max_detections=int(self.local_conflict_max_detections),
            max_tracks=int(self.local_conflict_max_tracks),
        )
        plan["stats"]["skipped_large_clusters"] = int(skipped_large)
        if not eligible_components:
            return plan

        eligible_cluster_count = int(len(eligible_components))
        budget_limit = 0
        if float(self.local_conflict_replacement_budget_ratio) > 0.0 and eligible_cluster_count > 0:
            budget_limit = max(
                int(math.ceil(float(self.local_conflict_replacement_budget_ratio) * float(eligible_cluster_count))),
                1,
            )
        if int(self.local_conflict_max_replaced_clusters) > 0:
            budget_limit = (
                min(int(self.local_conflict_max_replaced_clusters), budget_limit)
                if budget_limit > 0
                else int(self.local_conflict_max_replaced_clusters)
            )

        feasible_score_thresh = 1.0 - float(thresh)
        for component in eligible_components:
            det_rows = [int(x) for x in component.get("det_rows", [])]
            track_cols = [int(x) for x in component.get("track_cols", [])]
            if not det_rows or not track_cols:
                continue
            plan["stats"]["eligible_clusters"] += 1

            host_matches_local, _, _ = matching.linear_assignment(
                refined_cost[np.ix_(track_cols, det_rows)],
                thresh=float(thresh),
            )
            host_det_to_track_local = {
                int(det_local_idx): int(track_local_idx)
                for track_local_idx, det_local_idx in host_matches_local.tolist()
            }
            host_pair_set = {
                (int(det_rows[int(det_local_idx)]), int(track_cols[int(track_local_idx)]))
                for track_local_idx, det_local_idx in host_matches_local.tolist()
            }
            host_matched_track_local = {
                int(track_local_idx)
                for track_local_idx, _ in host_matches_local.tolist()
            }
            free_track_local = [
                int(track_local_idx)
                for track_local_idx in range(len(track_cols))
                if int(track_local_idx) not in host_matched_track_local
            ]

            feature_pack = self._build_local_conflict_runtime_features_v2(
                base_scores=base_scores.index_select(0, torch.tensor(det_rows, device=device)).index_select(
                    1, torch.tensor(track_cols, device=device)
                ),
                score_mat=score_mat.index_select(0, torch.tensor(det_rows, device=device)).index_select(
                    1, torch.tensor(track_cols, device=device)
                ),
                det_scores=det_scores.index_select(0, torch.tensor(det_rows, device=device)),
                track_gaps=track_gaps.index_select(0, torch.tensor(track_cols, device=device)),
                track_hist_lens=track_hist_lens.index_select(0, torch.tensor(track_cols, device=device)),
                det_boxes_cxcywh=det_boxes_cxcywh.index_select(0, torch.tensor(det_rows, device=device)),
                track_boxes_cxcywh=track_boxes_cxcywh.index_select(0, torch.tensor(track_cols, device=device)),
            )
            if feature_pack["edge_features"].shape[0] == 0:
                plan["stats"]["trigger_filtered_clusters"] += 1
                plan["stats"]["deferred_dets"] += int(len(det_rows))
                continue

            with torch.inference_mode():
                outputs = self.local_conflict_model(
                    det_features=feature_pack["det_features"],
                    track_features=feature_pack["track_features"],
                    edge_features=feature_pack["edge_features"],
                    edge_det_index=feature_pack["edge_det_index"],
                    edge_track_index=feature_pack["edge_track_index"],
                    cluster_features=feature_pack["cluster_features"],
                    host_variant_id=int(feature_pack.get("host_variant_id", 0)),
                )
                gate_logit = outputs.get("cluster_utility_logit", outputs["cluster_commit_logit"]).view(())
                gate_logit = (
                    gate_logit / max(float(self.local_conflict_cluster_gate_temp), 1e-6)
                    + float(self.local_conflict_cluster_gate_bias)
                )
                cluster_gate_prob = float(torch.sigmoid(gate_logit).item())
                if cluster_gate_prob < float(self.local_conflict_cluster_gate_thresh):
                    plan["stats"]["gate_filtered_clusters"] += 1
                    plan["stats"]["deferred_dets"] += int(len(det_rows))
                    continue
                plan["stats"]["gate_pass_clusters"] += 1

                dense_logits = HostConditionedLocalConflictSetPredictor.build_dense_assignment_logits(
                    num_detections=len(det_rows),
                    num_tracks=len(track_cols),
                    edge_logits=outputs["edge_logits"],
                    edge_det_index=feature_pack["edge_det_index"],
                    edge_track_index=feature_pack["edge_track_index"],
                    defer_logits=outputs["defer_logits"],
                )

            feasible_mask = feature_pack["dense_edge_mask"] & (
                feature_pack["dense_refined_scores"] >= float(feasible_score_thresh)
            )
            edit_score_sub = dense_logits[:, : len(track_cols)].new_full(
                (len(det_rows), len(track_cols)),
                -1e6,
            )
            edit_feasible_mask = torch.zeros_like(feasible_mask)
            free_track_mask = torch.zeros((len(track_cols),), device=device, dtype=torch.bool)
            if free_track_local:
                free_track_mask[torch.tensor(free_track_local, device=device, dtype=torch.long)] = True
            for det_local_idx in range(len(det_rows)):
                row_feasible = feasible_mask[det_local_idx]
                candidate_mask = row_feasible & free_track_mask
                if not bool(candidate_mask.any().item()):
                    continue
                host_track_local_idx = host_det_to_track_local.get(int(det_local_idx), None)
                defer_logit = dense_logits[det_local_idx, len(track_cols)]
                baseline = defer_logit
                if host_track_local_idx is not None and 0 <= int(host_track_local_idx) < len(track_cols):
                    baseline = torch.maximum(
                        baseline,
                        dense_logits[det_local_idx, int(host_track_local_idx)],
                    )
                utility_row = dense_logits[det_local_idx, : len(track_cols)] - baseline
                edit_score_sub[det_local_idx, candidate_mask] = utility_row[candidate_mask]
                edit_feasible_mask[det_local_idx, candidate_mask] = True

            assignments = solve_assignment_with_private_defer(
                score_sub=edit_score_sub,
                feasible_mask=edit_feasible_mask,
                defer_scores=torch.zeros((len(det_rows),), device=device, dtype=dense_logits.dtype),
                use_hungarian=True,
            )

            pair_candidates: List[Dict[str, float | int]] = []
            any_assignment_commit = False
            for assignment in assignments:
                det_local_idx = int(assignment.get("det_local_idx", -1))
                track_local_idx = assignment.get("track_local_idx", None)
                if det_local_idx < 0 or track_local_idx is None:
                    continue
                track_local_idx = int(track_local_idx)
                if not (0 <= det_local_idx < len(det_rows) and 0 <= track_local_idx < len(track_cols)):
                    continue
                if not bool(edit_feasible_mask[det_local_idx, track_local_idx].item()):
                    continue
                utility_margin = float(edit_score_sub[det_local_idx, track_local_idx].item())
                if utility_margin <= 0.0:
                    continue
                any_assignment_commit = True
                det_row = int(det_rows[det_local_idx])
                track_col = int(track_cols[track_local_idx])
                pair_candidates.append(
                    {
                        "det_row": int(det_row),
                        "track_col": int(track_col),
                        "commit_margin": float(utility_margin),
                    }
                )

            if not any_assignment_commit:
                plan["stats"]["all_defer_clusters"] += 1
                plan["stats"]["trigger_filtered_clusters"] += 1
                plan["stats"]["deferred_dets"] += int(len(det_rows))
                continue
            if not pair_candidates:
                plan["stats"]["empty_pair_candidate_clusters"] += 1
                plan["stats"]["trigger_filtered_clusters"] += 1
                plan["stats"]["deferred_dets"] += int(len(det_rows))
                continue

            if float(self.local_conflict_min_commit_margin) > 0.0:
                before_margin = len(pair_candidates)
                pair_candidates = [
                    row
                    for row in pair_candidates
                    if float(row["commit_margin"]) >= float(self.local_conflict_min_commit_margin)
                ]
                plan["stats"]["margin_filtered_pairs"] += max(before_margin - len(pair_candidates), 0)

            pair_candidates.sort(
                key=lambda row: (float(row["commit_margin"]), -int(row["det_row"]), -int(row["track_col"])),
                reverse=True,
            )
            if int(self.local_conflict_max_commits_per_cluster) > 0 and len(pair_candidates) > int(
                self.local_conflict_max_commits_per_cluster
            ):
                plan["stats"]["capped_commit_pairs"] += int(
                    len(pair_candidates) - int(self.local_conflict_max_commits_per_cluster)
                )
                pair_candidates = pair_candidates[: int(self.local_conflict_max_commits_per_cluster)]

            if not pair_candidates:
                plan["stats"]["post_filter_empty_clusters"] += 1
                plan["stats"]["trigger_filtered_clusters"] += 1
                plan["stats"]["deferred_dets"] += int(len(det_rows))
                continue

            matched_pairs = [
                (int(row["det_row"]), int(row["track_col"]))
                for row in pair_candidates
            ]
            matched_track_cols = {int(track_col) for _, track_col in matched_pairs}

            if len(matched_pairs) < int(self.local_conflict_min_committed_matches):
                plan["stats"]["trigger_filtered_clusters"] += 1
                plan["stats"]["deferred_dets"] += int(len(det_rows))
                continue

            if budget_limit > 0 and int(plan["stats"]["replaced_clusters"]) >= int(budget_limit):
                plan["stats"]["budget_filtered_clusters"] += 1
                plan["stats"]["deferred_dets"] += int(len(det_rows))
                continue

            plan["stats"]["replaced_clusters"] += 1
            plugin_pair_set = set(matched_pairs)
            edited_det_rows = {int(det_row) for det_row, _ in plugin_pair_set}
            final_pair_set = {
                pair
                for pair in host_pair_set
                if int(pair[0]) not in edited_det_rows
            }
            final_pair_set.update(plugin_pair_set)
            if final_pair_set == host_pair_set:
                plan["stats"]["host_same_commit_clusters"] += 1
            else:
                plan["stats"]["delta_replaced_clusters"] += 1
            plan["stats"]["delta_commit_pairs"] += int(len(final_pair_set - host_pair_set))
            plan["stats"]["delta_drop_pairs"] += int(len(host_pair_set - final_pair_set))
            plan["stats"]["deferred_dets"] += int(len(det_rows) - len(matched_pairs))
            for det_row, track_col in matched_pairs:
                plan["resolved_dets"].add(int(det_row))
                plan["assignments"].append((int(det_row), int(track_col)))
                plan["stats"]["matched_dets"] += 1
            for track_col in matched_track_cols:
                plan["blocked_tracks"].add(int(track_col))

        plan["assignments"].sort(key=lambda x: (int(x[0]), int(x[1])))
        plan["stats"]["resolved_dets"] = int(len(plan["resolved_dets"]))
        plan["stats"]["blocked_tracks"] = int(len(plan["blocked_tracks"]))
        return plan

    def _get_posthost_one_edit_oracle_plan(
        self,
        *,
        refined_cost: np.ndarray,
        host_matches: np.ndarray,
        host_u_track: np.ndarray,
        host_u_detection: np.ndarray,
        detections: Sequence[STrack],
        tracks: Sequence[STrack],
        thresh: float,
    ) -> Dict[str, Any]:
        plan = {
            "matches": np.asarray(host_matches, dtype=int).reshape(-1, 2)
            if np.asarray(host_matches).size > 0
            else np.empty((0, 2), dtype=int),
            "u_track": np.asarray(host_u_track, dtype=int).reshape(-1),
            "u_detection": np.asarray(host_u_detection, dtype=int).reshape(-1),
            "stats": self._empty_local_conflict_stats(),
        }
        if (
            not self.use_posthost_oracle_edit
            or len(detections) == 0
            or len(tracks) == 0
        ):
            return plan

        det_gt_ids = [int(getattr(det, "oracle_gt_id", -1) or -1) for det in detections]
        track_gt_ids = [int(getattr(track, "oracle_gt_id", -1) or -1) for track in tracks]
        if not any(gt_id > 0 for gt_id in det_gt_ids) or not any(gt_id > 0 for gt_id in track_gt_ids):
            return plan

        score_mat = torch.as_tensor(1.0 - refined_cost, dtype=torch.float32).transpose(0, 1).contiguous()
        components = build_topk_bipartite_components(
            score_mat=score_mat,
            topk=int(self.local_conflict_topk),
            min_edge_score=0.0,
        )
        eligible_components, skipped_large = filter_local_conflict_clusters_by_size(
            components,
            min_detections=int(self.local_conflict_min_detections),
            max_detections=int(self.local_conflict_max_detections),
            max_tracks=int(self.local_conflict_max_tracks),
        )
        plan["stats"]["skipped_large_clusters"] = int(skipped_large)
        if not eligible_components:
            return plan

        feasible_score_thresh = float(1.0 - float(thresh))
        match_set_det_track = {
            (int(det_idx), int(track_idx))
            for track_idx, det_idx in np.asarray(host_matches, dtype=int).reshape(-1, 2).tolist()
        }
        unmatched_track_set = {int(x) for x in np.asarray(host_u_track, dtype=int).reshape(-1).tolist()}
        unmatched_det_set = {int(x) for x in np.asarray(host_u_detection, dtype=int).reshape(-1).tolist()}

        for component in eligible_components:
            det_rows = [int(x) for x in component.get("det_rows", [])]
            track_cols = [int(x) for x in component.get("track_cols", [])]
            if not det_rows or not track_cols:
                continue
            plan["stats"]["eligible_clusters"] += 1

            component_host_pairs = {
                (int(det_idx), int(track_idx))
                for det_idx, track_idx in match_set_det_track
                if int(det_idx) in det_rows and int(track_idx) in track_cols
            }
            host_det_to_track = {int(det_idx): int(track_idx) for det_idx, track_idx in component_host_pairs}
            host_track_to_det = {int(track_idx): int(det_idx) for det_idx, track_idx in component_host_pairs}
            host_score = self._score_pair_set_against_oracle(
                list(component_host_pairs),
                det_gt_ids=det_gt_ids,
                track_gt_ids=track_gt_ids,
            )

            candidates: List[Dict[str, Any]] = []
            for det_idx in det_rows:
                det_gt = int(det_gt_ids[int(det_idx)])
                if det_gt <= 0:
                    continue
                for track_idx in track_cols:
                    track_gt = int(track_gt_ids[int(track_idx)])
                    if track_gt <= 0 or det_gt != track_gt:
                        continue
                    if float(score_mat[int(det_idx), int(track_idx)].item()) < feasible_score_thresh:
                        continue
                    if (int(det_idx), int(track_idx)) in component_host_pairs:
                        continue
                    remove_pairs: set[tuple[int, int]] = set()
                    existing_track = host_det_to_track.get(int(det_idx))
                    if existing_track is not None and int(existing_track) != int(track_idx):
                        remove_pairs.add((int(det_idx), int(existing_track)))
                    existing_det = host_track_to_det.get(int(track_idx))
                    if existing_det is not None and int(existing_det) != int(det_idx):
                        remove_pairs.add((int(existing_det), int(track_idx)))
                    if len(remove_pairs) > 1:
                        continue
                    new_pairs = set(component_host_pairs)
                    new_pairs.difference_update(remove_pairs)
                    new_pairs.add((int(det_idx), int(track_idx)))
                    new_score = self._score_pair_set_against_oracle(
                        list(new_pairs),
                        det_gt_ids=det_gt_ids,
                        track_gt_ids=track_gt_ids,
                    )
                    utility = float(new_score["score"] - host_score["score"])
                    if utility <= 0.0:
                        continue
                    action_type = "add" if len(remove_pairs) == 0 else "swap"
                    candidates.append(
                        {
                            "action_type": action_type,
                            "add_pair": (int(det_idx), int(track_idx)),
                            "remove_pairs": sorted(remove_pairs),
                            "utility": utility,
                            "delta_correct": float(new_score["correct"] - host_score["correct"]),
                            "delta_wrong": float(new_score["wrong"] - host_score["wrong"]),
                        }
                    )

            for det_idx, track_idx in sorted(component_host_pairs):
                det_gt = int(det_gt_ids[int(det_idx)])
                track_gt = int(track_gt_ids[int(track_idx)])
                if det_gt <= 0 or track_gt <= 0 or det_gt == track_gt:
                    continue
                new_pairs = {pair for pair in component_host_pairs if pair != (int(det_idx), int(track_idx))}
                new_score = self._score_pair_set_against_oracle(
                    list(new_pairs),
                    det_gt_ids=det_gt_ids,
                    track_gt_ids=track_gt_ids,
                )
                utility = float(new_score["score"] - host_score["score"])
                if utility <= 0.0:
                    continue
                candidates.append(
                    {
                        "action_type": "defer",
                        "add_pair": None,
                        "remove_pairs": [(int(det_idx), int(track_idx))],
                        "utility": utility,
                        "delta_correct": float(new_score["correct"] - host_score["correct"]),
                        "delta_wrong": float(new_score["wrong"] - host_score["wrong"]),
                    }
                )

            if not candidates:
                plan["stats"]["trigger_filtered_clusters"] += 1
                continue

            action_priority = {"swap": 2, "add": 1, "defer": 0}
            candidates.sort(
                key=lambda row: (
                    float(row["utility"]),
                    float(row["delta_correct"]),
                    -float(row["delta_wrong"]),
                    int(action_priority.get(str(row["action_type"]), -1)),
                ),
                reverse=True,
            )
            best = candidates[0]
            remove_pairs = {(int(det_idx), int(track_idx)) for det_idx, track_idx in best["remove_pairs"]}
            add_pair = (
                (int(best["add_pair"][0]), int(best["add_pair"][1]))
                if best.get("add_pair", None) is not None
                else None
            )
            final_pairs = set(component_host_pairs)
            final_pairs.difference_update(remove_pairs)
            if add_pair is not None:
                final_pairs.add(add_pair)

            if final_pairs == component_host_pairs:
                plan["stats"]["host_same_commit_clusters"] += 1
                continue

            plan["stats"]["replaced_clusters"] += 1
            plan["stats"]["delta_replaced_clusters"] += 1
            plan["stats"]["posthost_selected_clusters"] += 1
            if str(best["action_type"]) == "swap":
                plan["stats"]["posthost_swap_clusters"] += 1
            elif str(best["action_type"]) == "add":
                plan["stats"]["posthost_add_clusters"] += 1
            elif str(best["action_type"]) == "defer":
                plan["stats"]["posthost_defer_clusters"] += 1

            for det_idx, track_idx in remove_pairs:
                if (int(det_idx), int(track_idx)) in match_set_det_track:
                    match_set_det_track.remove((int(det_idx), int(track_idx)))
                    unmatched_det_set.add(int(det_idx))
                    unmatched_track_set.add(int(track_idx))
                    plan["stats"]["delta_drop_pairs"] += 1
                    if add_pair is None or int(add_pair[0]) != int(det_idx):
                        plan["stats"]["deferred_dets"] += 1
            if add_pair is not None and add_pair not in match_set_det_track:
                match_set_det_track.add(add_pair)
                unmatched_det_set.discard(int(add_pair[0]))
                unmatched_track_set.discard(int(add_pair[1]))
                plan["stats"]["delta_commit_pairs"] += 1
                plan["stats"]["matched_dets"] += 1

        final_matches = sorted(
            [[int(track_idx), int(det_idx)] for det_idx, track_idx in match_set_det_track],
            key=lambda row: (int(row[0]), int(row[1])),
        )
        plan["matches"] = np.asarray(final_matches, dtype=int).reshape(-1, 2) if final_matches else np.empty((0, 2), dtype=int)
        plan["u_track"] = np.asarray(sorted(unmatched_track_set), dtype=int)
        plan["u_detection"] = np.asarray(sorted(unmatched_det_set), dtype=int)
        return plan

    def _get_posthost_one_edit_hierarchical_plan(
        self,
        *,
        iou_cost: np.ndarray,
        refined_cost: np.ndarray,
        host_matches: np.ndarray,
        host_u_track: np.ndarray,
        host_u_detection: np.ndarray,
        detections: Sequence[STrack],
        tracks: Sequence[STrack],
        thresh: float,
    ) -> Dict[str, Any]:
        plan = {
            "matches": np.asarray(host_matches, dtype=int).reshape(-1, 2)
            if np.asarray(host_matches).size > 0
            else np.empty((0, 2), dtype=int),
            "u_track": np.asarray(host_u_track, dtype=int).reshape(-1),
            "u_detection": np.asarray(host_u_detection, dtype=int).reshape(-1),
            "stats": self._empty_local_conflict_stats(),
        }
        if (
            not self.use_posthost_hierarchical_edit
            or self.posthost_hierarchical_model is None
            or len(detections) == 0
            or len(tracks) == 0
        ):
            return plan

        device = next(self.posthost_hierarchical_model.parameters()).device
        score_mat = torch.as_tensor(1.0 - refined_cost, device=device, dtype=torch.float32).transpose(0, 1).contiguous()
        base_scores = torch.as_tensor(1.0 - iou_cost, device=device, dtype=torch.float32).transpose(0, 1).contiguous()
        det_scores = torch.tensor([float(det.score) for det in detections], device=device, dtype=torch.float32)
        det_boxes_cxcywh = _tlbr_array_to_cxcywh([det.tlbr for det in detections]).to(device=device)
        track_boxes_cxcywh = _tlbr_array_to_cxcywh([track.tlbr for track in tracks]).to(device=device)
        track_gaps = torch.tensor(
            [max(float(self.frame_id - int(track.frame_id)), 0.0) for track in tracks],
            device=device,
            dtype=torch.float32,
        )
        track_hist_lens = torch.tensor(
            [
                max(
                    float(getattr(track, "tracklet_len", 0) + 1),
                    float(int(track.frame_id) - int(track.start_frame) + 1),
                )
                for track in tracks
            ],
            device=device,
            dtype=torch.float32,
        )

        components = build_topk_bipartite_components(
            score_mat=score_mat,
            topk=int(self.local_conflict_topk),
            min_edge_score=0.0,
        )
        eligible_components, skipped_large = filter_local_conflict_clusters_by_size(
            components,
            min_detections=int(self.local_conflict_min_detections),
            max_detections=int(self.local_conflict_max_detections),
            max_tracks=int(self.local_conflict_max_tracks),
        )
        plan["stats"]["skipped_large_clusters"] = int(skipped_large)
        if not eligible_components:
            return plan

        match_set_det_track = {
            (int(det_idx), int(track_idx))
            for track_idx, det_idx in np.asarray(host_matches, dtype=int).reshape(-1, 2).tolist()
        }
        unmatched_track_set = {int(x) for x in np.asarray(host_u_track, dtype=int).reshape(-1).tolist()}
        unmatched_det_set = {int(x) for x in np.asarray(host_u_detection, dtype=int).reshape(-1).tolist()}

        for component in eligible_components:
            det_rows = [int(x) for x in component.get("det_rows", [])]
            track_cols = [int(x) for x in component.get("track_cols", [])]
            if not det_rows or not track_cols:
                continue
            plan["stats"]["eligible_clusters"] += 1

            component_host_pairs = {
                (int(det_idx), int(track_idx))
                for det_idx, track_idx in match_set_det_track
                if int(det_idx) in det_rows and int(track_idx) in track_cols
            }
            det_to_local = {int(det_idx): int(local_idx) for local_idx, det_idx in enumerate(det_rows)}
            track_to_local = {int(track_idx): int(local_idx) for local_idx, track_idx in enumerate(track_cols)}
            host_pairs_local = [
                (int(det_to_local[int(det_idx)]), int(track_to_local[int(track_idx)]))
                for det_idx, track_idx in sorted(component_host_pairs)
            ]

            feature_pack = self._build_local_conflict_runtime_features_v2(
                base_scores=base_scores.index_select(0, torch.tensor(det_rows, device=device)).index_select(
                    1, torch.tensor(track_cols, device=device)
                ),
                score_mat=score_mat.index_select(0, torch.tensor(det_rows, device=device)).index_select(
                    1, torch.tensor(track_cols, device=device)
                ),
                det_scores=det_scores.index_select(0, torch.tensor(det_rows, device=device)),
                track_gaps=track_gaps.index_select(0, torch.tensor(track_cols, device=device)),
                track_hist_lens=track_hist_lens.index_select(0, torch.tensor(track_cols, device=device)),
                det_boxes_cxcywh=det_boxes_cxcywh.index_select(0, torch.tensor(det_rows, device=device)),
                track_boxes_cxcywh=track_boxes_cxcywh.index_select(0, torch.tensor(track_cols, device=device)),
            )
            if feature_pack["edge_features"].shape[0] == 0:
                plan["stats"]["trigger_filtered_clusters"] += 1
                continue

            is_large_component = int(max(len(det_rows), len(track_cols)) >= 4)
            prediction = self._predict_posthost_hierarchical_candidate(
                feature_pack=feature_pack,
                host_pairs_local=host_pairs_local,
                is_large_component=int(is_large_component),
            )
            selected = prediction.get("selected_candidate", None)
            if selected is None:
                if str(prediction.get("decision", "")) == "gate_keep":
                    plan["stats"]["gate_filtered_clusters"] += 1
                else:
                    plan["stats"]["trigger_filtered_clusters"] += 1
                continue

            add_pair_local = selected.get("add_pair", None)
            remove_pair_local = selected.get("remove_pair", None)
            add_pair_global = None
            if add_pair_local is not None:
                add_pair_global = (
                    int(det_rows[int(add_pair_local[0])]),
                    int(track_cols[int(add_pair_local[1])]),
                )
            remove_pairs_global: set[tuple[int, int]] = set()
            if remove_pair_local is not None:
                remove_pairs_global.add(
                    (
                        int(det_rows[int(remove_pair_local[0])]),
                        int(track_cols[int(remove_pair_local[1])]),
                    )
                )

            final_pairs = set(component_host_pairs)
            final_pairs.difference_update(remove_pairs_global)
            if add_pair_global is not None:
                final_pairs.add(add_pair_global)

            if final_pairs == component_host_pairs:
                plan["stats"]["host_same_commit_clusters"] += 1
                continue

            plan["stats"]["replaced_clusters"] += 1
            plan["stats"]["delta_replaced_clusters"] += 1
            plan["stats"]["posthost_selected_clusters"] += 1
            action_type = str(selected.get("action_type", ""))
            if action_type == "swap":
                plan["stats"]["posthost_swap_clusters"] += 1
            elif action_type == "add":
                plan["stats"]["posthost_add_clusters"] += 1
            elif action_type == "defer":
                plan["stats"]["posthost_defer_clusters"] += 1

            for det_idx, track_idx in remove_pairs_global:
                if (int(det_idx), int(track_idx)) in match_set_det_track:
                    match_set_det_track.remove((int(det_idx), int(track_idx)))
                    unmatched_det_set.add(int(det_idx))
                    unmatched_track_set.add(int(track_idx))
                    plan["stats"]["delta_drop_pairs"] += 1
                    if add_pair_global is None or int(add_pair_global[0]) != int(det_idx):
                        plan["stats"]["deferred_dets"] += 1

            if add_pair_global is not None and add_pair_global not in match_set_det_track:
                match_set_det_track.add(add_pair_global)
                unmatched_det_set.discard(int(add_pair_global[0]))
                unmatched_track_set.discard(int(add_pair_global[1]))
                plan["stats"]["delta_commit_pairs"] += 1
                plan["stats"]["matched_dets"] += 1

        final_matches = sorted(
            [[int(track_idx), int(det_idx)] for det_idx, track_idx in match_set_det_track],
            key=lambda row: (int(row[0]), int(row[1])),
        )
        plan["matches"] = np.asarray(final_matches, dtype=int).reshape(-1, 2) if final_matches else np.empty((0, 2), dtype=int)
        plan["u_track"] = np.asarray(sorted(unmatched_track_set), dtype=int)
        plan["u_detection"] = np.asarray(sorted(unmatched_det_set), dtype=int)
        return plan

    def update(self, output_results, img_info, img_size):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]
        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale

        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.args.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        detections = (
            [STrack(STrack.tlbr_to_tlwh(tlbr), s) for (tlbr, s) in zip(dets, scores_keep)]
            if len(dets) > 0
            else []
        )
        self._annotate_detection_gt_ids(detections)

        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        STrack.multi_predict(strack_pool)
        iou_dists = matching.iou_distance(strack_pool, detections)
        dists = iou_dists.copy()
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        if len(strack_pool) > 0 and len(detections) > 0:
            self._maybe_dump_runtime_candidates(
                base_scores=1.0 - iou_dists,
                refined_scores=1.0 - dists,
                detections=detections,
                tracks=strack_pool,
                det_scores=scores_keep.tolist() if hasattr(scores_keep, "tolist") else list(scores_keep),
            )

        if self.use_local_conflict and len(strack_pool) > 0 and len(detections) > 0:
            local_plan = self._get_local_conflict_plan(
                iou_cost=iou_dists,
                refined_cost=dists,
                detections=detections,
                tracks=strack_pool,
                thresh=self.args.match_thresh,
            )
            self._accumulate_local_conflict_stats(local_plan["stats"])
            if not local_plan["assignments"]:
                matches, u_track, u_detection = matching.linear_assignment(
                    dists,
                    thresh=self.args.match_thresh,
                )
            else:
                resolved_track_rows = sorted({int(track_col) for _, track_col in local_plan["assignments"]})
                resolved_det_cols = sorted({int(det_row) for det_row, _ in local_plan["assignments"]})
                resolved_track_rows_set = set(resolved_track_rows)
                resolved_det_cols_set = set(resolved_det_cols)
                residual_track_rows = [
                    idx for idx in range(len(strack_pool)) if idx not in resolved_track_rows_set
                ]
                residual_det_cols = [
                    idx for idx in range(len(detections)) if idx not in resolved_det_cols_set
                ]
                residual_dists = dists[np.ix_(residual_track_rows, residual_det_cols)]
                host_matches, u_track_local, u_detection_local = matching.linear_assignment(
                    residual_dists,
                    thresh=self.args.match_thresh,
                )
                matches_list: List[List[int]] = [
                    [int(track_col), int(det_row)] for det_row, track_col in local_plan["assignments"]
                ]
                for track_local_idx, det_local_idx in host_matches.tolist():
                    matches_list.append(
                        [
                            int(residual_track_rows[int(track_local_idx)]),
                            int(residual_det_cols[int(det_local_idx)]),
                        ]
                    )
                matches = (
                    np.asarray(matches_list, dtype=int).reshape(-1, 2)
                    if matches_list
                    else np.empty((0, 2), dtype=int)
                )
                u_track = np.asarray(
                    [int(residual_track_rows[int(i)]) for i in list(u_track_local)],
                    dtype=int,
                )
                u_detection = np.asarray(
                    [int(residual_det_cols[int(i)]) for i in list(u_detection_local)],
                    dtype=int,
                )
        elif (self.use_posthost_oracle_edit or self.use_posthost_hierarchical_edit) and len(strack_pool) > 0 and len(detections) > 0:
            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)
            if self.use_posthost_oracle_edit:
                oracle_plan = self._get_posthost_one_edit_oracle_plan(
                    refined_cost=dists,
                    host_matches=matches,
                    host_u_track=u_track,
                    host_u_detection=u_detection,
                    detections=detections,
                    tracks=strack_pool,
                    thresh=self.args.match_thresh,
                )
                self._accumulate_local_conflict_stats(oracle_plan["stats"])
                matches = oracle_plan["matches"]
                u_track = oracle_plan["u_track"]
                u_detection = oracle_plan["u_detection"]
            else:
                learned_plan = self._get_posthost_one_edit_hierarchical_plan(
                    iou_cost=iou_dists,
                    refined_cost=dists,
                    host_matches=matches,
                    host_u_track=u_track,
                    host_u_detection=u_detection,
                    detections=detections,
                    tracks=strack_pool,
                    thresh=self.args.match_thresh,
                )
                self._accumulate_local_conflict_stats(learned_plan["stats"])
                matches = learned_plan["matches"]
                u_track = learned_plan["u_track"]
                u_detection = learned_plan["u_detection"]
        else:
            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                self._update_track_oracle_identity(track, det)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                self._update_track_oracle_identity(track, det)
                refind_stracks.append(track)

        detections_second = (
            [STrack(STrack.tlbr_to_tlwh(tlbr), s) for (tlbr, s) in zip(dets_second, scores_second)]
            if len(dets_second) > 0
            else []
        )
        self._annotate_detection_gt_ids(detections_second)
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                self._update_track_oracle_identity(track, det)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                self._update_track_oracle_identity(track, det)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            self._update_track_oracle_identity(unconfirmed[itracked], detections[idet])
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            self._update_track_oracle_identity(track, detections[inew])
            activated_starcks.append(track)
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        return output_stracks
