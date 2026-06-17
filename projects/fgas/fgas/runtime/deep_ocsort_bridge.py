from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from projects.fgas.fgas.runtime.block_refiner import FGASBlockRefiner, FGASConfig, FGASControllerActions


REPO_ROOT = Path(__file__).resolve().parents[4]


@dataclass
class _TrackProxy:
    tlbr: np.ndarray
    tracklet_len: float
    time_since_update: int = 0
    hit_streak: int = 0
    hits: int = 0
    age: int = 0
    fcaa_low: Optional[np.ndarray] = None
    fcaa_mid: Optional[np.ndarray] = None
    fcaa_high: Optional[np.ndarray] = None


@dataclass
class _DetectionProxy:
    tlbr: np.ndarray
    score: float


def _resolve_device(device_name: str) -> str:
    name = str(device_name or "cpu").lower()
    if name == "gpu":
        return "cuda"
    return name


def _resolve_checkpoint_path(path_value: str) -> str:
    raw = str(path_value or "").strip()
    if not raw:
        return ""
    path = Path(raw)
    if path.is_file():
        return str(path.resolve())
    alt = (REPO_ROOT / raw).resolve()
    if alt.is_file():
        return str(alt)
    return raw


def _linear_assignment(cost_matrix: np.ndarray) -> np.ndarray:
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int)
    try:
        import lap

        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[row_idx, col_idx] for row_idx, col_idx in enumerate(x) if col_idx >= 0], dtype=int)
    except ImportError:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(rows, cols)), dtype=int)


def apply_deep_ocsort_fgas_controller(
    *,
    det_track_cost: np.ndarray,
    det_track_iou: np.ndarray,
    iou_threshold: float,
    controller_actions: Optional[FGASControllerActions],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    empty_matches = np.empty((0, 2), dtype=int)
    base_debug = {
        "controller_applied_forced_matches": 0,
        "controller_applied_blocked_rows": 0,
        "controller_applied_blocked_cols": 0,
    }
    if det_track_cost.size == 0:
        return (
            empty_matches,
            np.arange(det_track_cost.shape[0], dtype=int),
            np.arange(det_track_cost.shape[1], dtype=int),
            base_debug,
        )

    if controller_actions is None:
        matched_indices = _linear_assignment(det_track_cost)
        unmatched_detections = []
        unmatched_trackers = []
        for det_idx in range(det_track_cost.shape[0]):
            if det_idx not in matched_indices[:, 0]:
                unmatched_detections.append(det_idx)
        for trk_idx in range(det_track_cost.shape[1]):
            if trk_idx not in matched_indices[:, 1]:
                unmatched_trackers.append(trk_idx)
        matches = []
        for det_idx, trk_idx in matched_indices:
            if det_track_iou[det_idx, trk_idx] < float(iou_threshold):
                unmatched_detections.append(int(det_idx))
                unmatched_trackers.append(int(trk_idx))
            else:
                matches.append([int(det_idx), int(trk_idx)])
        matches_array = np.asarray(matches, dtype=int) if matches else empty_matches
        return (
            matches_array,
            np.asarray(sorted(set(unmatched_detections)), dtype=int),
            np.asarray(sorted(set(unmatched_trackers)), dtype=int),
            base_debug,
        )

    track_det_cost = np.asarray(det_track_cost.T, dtype=np.float32)
    track_det_iou = np.asarray(det_track_iou.T, dtype=np.float32)

    raw_forced = list(getattr(controller_actions, "forced_matches", []) or [])
    blocked_rows = {int(v) for v in (getattr(controller_actions, "blocked_rows", []) or [])}
    blocked_cols = {int(v) for v in (getattr(controller_actions, "blocked_cols", []) or [])}

    forced_matches_track_det = []
    used_rows = set()
    used_cols = set()
    for row_idx, col_idx in raw_forced:
        row_idx = int(row_idx)
        col_idx = int(col_idx)
        if row_idx < 0 or row_idx >= track_det_cost.shape[0] or col_idx < 0 or col_idx >= track_det_cost.shape[1]:
            continue
        if row_idx in blocked_rows or col_idx in blocked_cols:
            continue
        if row_idx in used_rows or col_idx in used_cols:
            continue
        if not np.isfinite(track_det_cost[row_idx, col_idx]):
            continue
        if float(track_det_iou[row_idx, col_idx]) < float(iou_threshold):
            continue
        forced_matches_track_det.append((row_idx, col_idx))
        used_rows.add(row_idx)
        used_cols.add(col_idx)

    blocked_rows.difference_update(used_rows)
    blocked_cols.difference_update(used_cols)

    remain_rows = [idx for idx in range(track_det_cost.shape[0]) if idx not in used_rows and idx not in blocked_rows]
    remain_cols = [idx for idx in range(track_det_cost.shape[1]) if idx not in used_cols and idx not in blocked_cols]

    hungarian_matches_track_det = []
    unmatched_rows = set(blocked_rows)
    unmatched_cols = set(blocked_cols)
    if remain_rows and remain_cols:
        sub_cost = track_det_cost[np.ix_(remain_rows, remain_cols)]
        sub_matches = _linear_assignment(sub_cost)
        for sub_row_idx, sub_col_idx in sub_matches.tolist():
            track_idx = int(remain_rows[sub_row_idx])
            det_idx = int(remain_cols[sub_col_idx])
            if float(track_det_iou[track_idx, det_idx]) < float(iou_threshold):
                unmatched_rows.add(track_idx)
                unmatched_cols.add(det_idx)
                continue
            hungarian_matches_track_det.append((track_idx, det_idx))

        matched_rows = {track_idx for track_idx, _ in hungarian_matches_track_det}
        matched_cols = {det_idx for _, det_idx in hungarian_matches_track_det}
        unmatched_rows.update(int(idx) for idx in remain_rows if idx not in matched_rows)
        unmatched_cols.update(int(idx) for idx in remain_cols if idx not in matched_cols)
    else:
        unmatched_rows.update(int(idx) for idx in remain_rows)
        unmatched_cols.update(int(idx) for idx in remain_cols)

    all_track_det_matches = forced_matches_track_det + hungarian_matches_track_det
    det_track_matches = [[int(det_idx), int(track_idx)] for track_idx, det_idx in all_track_det_matches]
    matches_array = np.asarray(det_track_matches, dtype=int) if det_track_matches else empty_matches
    debug = {
        "controller_applied_forced_matches": int(len(forced_matches_track_det)),
        "controller_applied_blocked_rows": int(len(blocked_rows)),
        "controller_applied_blocked_cols": int(len(blocked_cols)),
    }
    return (
        matches_array,
        np.asarray(sorted(unmatched_cols), dtype=int),
        np.asarray(sorted(unmatched_rows), dtype=int),
        debug,
    )


class DeepOCSORTFGASBridge:
    def __init__(self, args) -> None:
        self.refiner: Optional[FGASBlockRefiner] = None
        if not bool(getattr(args, "fgas_enable", False)):
            return
        if bool(getattr(args, "emb_off", False)):
            raise ValueError("FGAS on Deep-OC-SORT requires embeddings; do not use --emb_off.")
        self.refiner = FGASBlockRefiner(
            FGASConfig(
                enabled=True,
                resolver_checkpoint=_resolve_checkpoint_path(getattr(args, "fgas_resolver_checkpoint", "") or ""),
                block_primitive_checkpoint=_resolve_checkpoint_path(getattr(args, "fgas_block_primitive_checkpoint", "") or ""),
                block_primitive_conf_thresh=float(getattr(args, "fgas_block_primitive_conf_thresh", 0.5)),
                block_matcher_checkpoint=_resolve_checkpoint_path(getattr(args, "fgas_block_matcher_checkpoint", "") or ""),
                block_matcher_margin_thresh=float(getattr(args, "fgas_block_matcher_margin_thresh", 0.0)),
                block_matcher_base_margin_thresh=float(getattr(args, "fgas_block_matcher_base_margin_thresh", 0.0)),
                block_matcher_base_logit_scale_override=(
                    None
                    if getattr(args, "fgas_block_matcher_base_logit_scale_override", None) is None
                    else float(getattr(args, "fgas_block_matcher_base_logit_scale_override"))
                ),
                block_matcher_force_only=bool(getattr(args, "fgas_block_matcher_force_only", False)),
                block_matcher_forceonly_keep_base_on_nomatch=bool(
                    getattr(args, "fgas_block_matcher_forceonly_keep_base_on_nomatch", False)
                ),
                block_matcher_skip_single_row_unchanged_takeover=bool(getattr(args, "fgas_block_matcher_skip_single_row_unchanged_takeover", False)),
                block_matcher_stale_match_bias=float(getattr(args, "fgas_block_matcher_stale_match_bias", 0.0)),
                block_matcher_stale_match_mode=str(getattr(args, "fgas_block_matcher_stale_match_mode", "all_edges")),
                block_matcher_stale_match_min_time_since_update=int(
                    getattr(args, "fgas_block_matcher_stale_match_min_time_since_update", 0)
                ),
                block_matcher_stale_match_max_hit_streak=int(
                    getattr(args, "fgas_block_matcher_stale_match_max_hit_streak", 0)
                ),
                block_matcher_stale_match_min_hits=int(getattr(args, "fgas_block_matcher_stale_match_min_hits", 0)),
                block_matcher_stale_match_max_component_rows=int(
                    getattr(args, "fgas_block_matcher_stale_match_max_component_rows", 0)
                ),
                block_matcher_stale_match_max_component_cols=int(
                    getattr(args, "fgas_block_matcher_stale_match_max_component_cols", 0)
                ),
                pair_scorer_checkpoint=_resolve_checkpoint_path(getattr(args, "fgas_pair_scorer_checkpoint", "") or ""),
                block_gate_checkpoint=_resolve_checkpoint_path(getattr(args, "fgas_block_gate_checkpoint", "") or ""),
                block_gate_thresh=float(getattr(args, "fgas_block_gate_thresh", 0.5)),
                top_k=int(getattr(args, "fgas_topk", 5)),
                proximity_thresh=max(0.0, 1.0 - float(getattr(args, "iou_thresh", 0.3))),
                appearance_thresh=1.0,
                max_rows=int(getattr(args, "fgas_max_rows", 3)),
                max_cols=int(getattr(args, "fgas_max_cols", 3)),
                crop_height=int(getattr(args, "fgas_crop_height", 128)),
                crop_width=int(getattr(args, "fgas_crop_width", 64)),
                blend_weight=float(getattr(args, "fgas_blend_weight", 0.5)),
                assignment_mode=str(getattr(args, "fgas_assignment_mode", "blend")),
                row_nomatch_weight=float(getattr(args, "fgas_row_nomatch_weight", 0.0)),
                controller_enable=bool(getattr(args, "fgas_controller_enable", False)),
                controller_edge_thresh=float(getattr(args, "fgas_controller_edge_thresh", 0.6)),
                controller_row_defer_thresh=float(getattr(args, "fgas_controller_row_defer_thresh", 0.6)),
                controller_col_newborn_thresh=float(getattr(args, "fgas_controller_col_newborn_thresh", 0.6)),
                controller_margin_thresh=float(getattr(args, "fgas_controller_margin_thresh", 0.05)),
                controller_ambiguity_margin=float(getattr(args, "fgas_controller_ambiguity_margin", 0.05)),
                controller_mutual_top1_only=bool(getattr(args, "fgas_controller_mutual_top1_only", True)),
                controller_require_base_top1=bool(getattr(args, "fgas_controller_require_base_top1", True)),
                controller_only_changed_blocks=bool(getattr(args, "fgas_controller_only_changed_blocks", False)),
                primitive_direct_takeover=bool(getattr(args, "fgas_primitive_direct_takeover", False)),
                soft_apply_only_changed_blocks=bool(getattr(args, "fgas_soft_only_changed_blocks", False)),
                soft_apply_only_changed_rows=bool(getattr(args, "fgas_soft_only_changed_rows", False)),
                soft_apply_only_changed_frontier=bool(getattr(args, "fgas_soft_only_changed_frontier", False)),
                soft_allow_without_takeover=bool(getattr(args, "fgas_soft_allow_fallback", False)),
                soft_row_base_margin_thresh=float(getattr(args, "fgas_soft_row_base_margin_thresh", 1.0)),
                soft_changed_row_flip_gap_thresh=float(getattr(args, "fgas_soft_changed_row_flip_gap_thresh", 0.0)),
                soft_changed_row_refined_margin_thresh=float(getattr(args, "fgas_soft_changed_row_refined_margin_thresh", 0.0)),
                pair_ambiguity_margin=float(getattr(args, "fgas_pair_ambiguity_margin", 0.05)),
                acceptance_gate_checkpoint=_resolve_checkpoint_path(getattr(args, "fgas_acceptance_gate_checkpoint", "") or ""),
                acceptance_gate_thresh=float(getattr(args, "fgas_acceptance_gate_thresh", 0.5)),
                device=_resolve_device(getattr(args, "device", "cpu")),
            )
        )

    def is_active(self) -> bool:
        return bool(self.refiner is not None and self.refiner.is_active())

    def uses_frequency(self) -> bool:
        return bool(self.refiner is not None and self.refiner.uses_frequency())

    def crop_hw(self) -> Tuple[int, int]:
        if self.refiner is None:
            return (128, 64)
        return (int(self.refiner.config.crop_height), int(self.refiner.config.crop_width))

    def propose_controller_actions(
        self,
        *,
        tracks: Sequence[object],
        track_boxes: np.ndarray,
        detections: np.ndarray,
        emb_similarity: np.ndarray,
        iou_similarity: np.ndarray,
        image: np.ndarray,
        seq_name: str = "",
    ) -> Tuple[Optional[FGASControllerActions], Optional[np.ndarray], Optional[np.ndarray], Dict[str, object]]:
        if not self.is_active():
            return None, None, None, {}
        if len(tracks) == 0 or detections.shape[0] == 0:
            return None, None, None, {}

        track_proxies = [
            _TrackProxy(
                tlbr=np.asarray(track_boxes[idx], dtype=np.float32),
                tracklet_len=float(max(int(getattr(track, "hit_streak", 0)), int(getattr(track, "age", 0)))),
                time_since_update=int(getattr(track, "time_since_update", 0)),
                hit_streak=int(getattr(track, "hit_streak", 0)),
                hits=int(getattr(track, "hits", 0)),
                age=int(getattr(track, "age", 0)),
                fcaa_low=(None if getattr(track, "fcaa_low", None) is None else np.asarray(track.fcaa_low, dtype=np.float32)),
                fcaa_mid=(None if getattr(track, "fcaa_mid", None) is None else np.asarray(track.fcaa_mid, dtype=np.float32)),
                fcaa_high=(None if getattr(track, "fcaa_high", None) is None else np.asarray(track.fcaa_high, dtype=np.float32)),
            )
            for idx, track in enumerate(tracks)
        ]
        det_proxies = [
            _DetectionProxy(
                tlbr=np.asarray(detections[idx, :4], dtype=np.float32),
                score=float(detections[idx, 4]),
            )
            for idx in range(detections.shape[0])
        ]

        emb_similarity = np.clip(np.asarray(emb_similarity, dtype=np.float32), 0.0, 1.0)
        iou_similarity = np.clip(np.asarray(iou_similarity, dtype=np.float32), 0.0, 1.0)
        emb_dists = np.clip(1.0 - emb_similarity.T, 0.0, 1.0)
        raw_ious_dists = np.clip(1.0 - iou_similarity.T, 0.0, 1.0)

        refined_track_det_cost, refined_track_det_mask, debug, actions = self.refiner.refine_primary_cost(
            track_pool=track_proxies,
            detections=det_proxies,
            emb_dists=emb_dists,
            raw_ious_dists=raw_ious_dists,
            image=np.asarray(image),
            seq_name=str(seq_name or ""),
        )
        refined_det_track_similarity = np.clip(1.0 - np.asarray(refined_track_det_cost.T, dtype=np.float32), 0.0, 1.0)
        refined_det_track_mask = np.asarray(refined_track_det_mask.T, dtype=bool)
        return actions, refined_det_track_similarity, refined_det_track_mask, debug
