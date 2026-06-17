from __future__ import annotations

import math
import os.path as osp
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List

import torch

from models.graph_assoc_commit_policy import GraphAssocCommitPolicy
from models.local_conflict_commit import LocalConflictCommitRefiner
from models.local_conflict_set_predictor import HostConditionedLocalConflictSetPredictor, pair_geometry_features
from models.local_conflict_graph_common import compute_component_degree_features


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _finite_or(value: Any, fallback: float) -> float:
    try:
        out = float(value)
    except Exception:
        return float(fallback)
    if not math.isfinite(out):
        return float(fallback)
    return float(out)


def _normalize_commit_decision_mode(mode: str, model_type: str) -> str:
    requested = str(mode or "").strip().lower()
    if not requested:
        return "policy_score"
    aliases = {
        "policy_score": "policy_score",
        "selection_score": "selection_score",
        "learned_selection": "selection_score",
        "residual_selection": "policy_score",
        "selection_residual": "policy_score",
        "legacy_policy_score": "legacy_policy_score",
        "legacy_mixture": "legacy_policy_score",
        "positive_probability": "positive_probability",
        "positive_minus_weighted_neutral": "positive_minus_weighted_neutral",
        "positive_minus_neutral": "positive_minus_neutral",
        "gain_minus_neutral": "positive_minus_neutral",
        "gain-neutral": "positive_minus_neutral",
        "positive_times_one_minus_neutral": "positive_times_one_minus_neutral",
        "gain_times_one_minus_neutral": "positive_times_one_minus_neutral",
        "gain*(1-neutral)": "positive_times_one_minus_neutral",
        "min_positive_vs_one_minus_neutral": "min_positive_vs_one_minus_neutral",
        "min_gain_vs_one_minus_neutral": "min_positive_vs_one_minus_neutral",
        "min(gain,1-neutral)": "min_positive_vs_one_minus_neutral",
        "dual_threshold": "dual_threshold",
        "positive_and_not_neutral": "dual_threshold",
        "gain_plus_action": "policy_score",
        "gain+action": "policy_score",
        "gain_only": "gain_pred",
        "gain_pred": "gain_pred",
        "gain": "gain_pred",
        "action_margin": "action_margin",
        "rewrite_minus_reject": "action_margin",
        "rewrite-reject": "action_margin",
        "router_margin": "router_margin",
        "route_margin": "router_margin",
        "router_confidence": "router_confidence",
        "router_entropy": "router_entropy",
        "mixture_score": "policy_score",
    }
    normalized = aliases.get(requested, requested)
    if str(model_type) != "action_policy":
        return normalized
    if normalized not in {
        "policy_score",
        "selection_score",
        "legacy_policy_score",
        "positive_probability",
        "positive_minus_weighted_neutral",
        "positive_minus_neutral",
        "positive_times_one_minus_neutral",
        "min_positive_vs_one_minus_neutral",
        "dual_threshold",
        "gain_pred",
        "action_margin",
        "router_margin",
        "router_confidence",
        "router_entropy",
    }:
        return "policy_score"
    return normalized


def _score_entropy(values: List[float]) -> float:
    if not values:
        return 0.0
    arr = torch.tensor(values, dtype=torch.float32)
    arr = arr - torch.max(arr)
    prob = torch.softmax(arr, dim=0)
    entropy = -(prob * torch.log(torch.clamp(prob, min=1e-8))).sum()
    return float(entropy.item())


def _infer_edge_rank_frac(edge_meta_rows: List[Dict[str, Any]]) -> Dict[tuple[int, int], float]:
    by_col: Dict[int, List[Dict[str, Any]]] = {}
    for row in edge_meta_rows:
        col_idx = _safe_int(row.get("col_idx", -1), -1)
        if col_idx < 0:
            continue
        by_col.setdefault(int(col_idx), []).append(row)
    rank_frac: Dict[tuple[int, int], float] = {}
    for col_idx, rows in by_col.items():
        rows_sorted = sorted(
            rows,
            key=lambda item: (-_safe_float(item.get("utility", 0.0), 0.0), _safe_float(item.get("cost", 1.0), 1.0)),
        )
        denom = max(len(rows_sorted), 1)
        for rank, row in enumerate(rows_sorted, 1):
            row_idx = _safe_int(row.get("row_idx", -1), -1)
            if row_idx < 0:
                continue
            rank_frac[(int(row_idx), int(col_idx))] = float(rank) / float(denom)
    return rank_frac


def _merge_required_edge_rows(candidate_row: Dict[str, Any]) -> List[Dict[str, Any]]:
    merged: Dict[tuple[int, int], Dict[str, Any]] = {}
    for key in ("edge_meta", "baseline_pair_meta", "chosen_pair_meta"):
        for row in list(candidate_row.get(key, [])):
            row_idx = _safe_int(row.get("row_idx", -1), -1)
            col_idx = _safe_int(row.get("col_idx", -1), -1)
            if row_idx < 0 or col_idx < 0:
                continue
            merged[(int(row_idx), int(col_idx))] = dict(row)
    return list(merged.values())


def _pair_local_map(
    pairs: List[List[int]] | List[tuple[int, int]],
    *,
    col_to_local: Dict[int, int],
    row_to_local: Dict[int, int],
) -> Dict[int, int]:
    local_map: Dict[int, int] = {}
    for pair in pairs:
        if len(pair) < 2:
            continue
        row_idx = _safe_int(pair[0], -1)
        col_idx = _safe_int(pair[1], -1)
        if row_idx not in row_to_local or col_idx not in col_to_local:
            continue
        local_map[int(col_to_local[col_idx])] = int(row_to_local[row_idx])
    return local_map


@dataclass
class GraphAssocCommitScore:
    baseline_score: float
    chosen_score: float
    score_delta: float
    chosen_better: bool
    pred_score: float
    pred_pairs_local: Dict[int, int]
    chosen_pairs_local: Dict[int, int]
    baseline_pairs_local: Dict[int, int]
    gain_pred: float = 0.0
    selection_score: float = 0.0
    legacy_policy_score: float = 0.0
    policy_score: float = 0.0
    action_margin: float = 0.0
    router_margin: float = 0.0
    router_confidence: float = 0.0
    router_entropy: float = 0.0
    positive_probability: float = 0.0
    neutral_probability: float = 0.0
    reject_probability: float = 0.0
    policy_score_mode: str = ""
    decision_score: float = 0.0
    model_type: str = "commit_refiner"
    decision_mode: str = "assignment_delta"
    threshold: float = 0.0
    positive_threshold: float = 0.0
    neutral_threshold: float = 0.0
    neutral_risk_weight: float = 1.0
    accept: bool = False


class GraphAssocCommitScorer:
    def __init__(
        self,
        checkpoint_path: str,
        *,
        device: str = "cpu",
        decision_mode: str = "",
        threshold: float | None = None,
        neutral_risk_weight: float | None = None,
        positive_threshold: float | None = None,
        neutral_threshold: float | None = None,
    ) -> None:
        requested_path = str(checkpoint_path)
        if osp.isabs(requested_path):
            resolved_path = requested_path
        else:
            repo_root = osp.abspath(osp.join(osp.dirname(__file__), ".."))
            resolved_path = osp.join(repo_root, requested_path)
        self.checkpoint_path = str(resolved_path)
        requested_device = str(device or "cpu").strip() or "cpu"
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            requested_device = "cpu"
        self.device = torch.device(requested_device)
        payload = torch.load(self.checkpoint_path, map_location="cpu")
        self.model_type = str(payload.get("model_type", "commit_refiner") or "commit_refiner")
        self.model_family = str(payload.get("model_family", self.model_type) or self.model_type)
        if self.model_family == "set_predictor_v2":
            self.decision_mode = str(decision_mode or payload.get("decision_mode", "") or "cluster_gate_probability")
        else:
            self.decision_mode = _normalize_commit_decision_mode(
                str(decision_mode or payload.get("decision_mode", "") or ""),
                self.model_type,
            )
        payload_threshold = _finite_or(payload.get("threshold", float("nan")), 0.0)
        if self.model_family == "set_predictor_v2":
            payload_threshold = _finite_or(
                payload.get("cluster_gate_thresh_calibrated", payload.get("cluster_gate_thresh", 0.5)),
                0.5,
            )
            self.cluster_gate_temp = _finite_or(payload.get("cluster_gate_temp", 1.0), 1.0)
            self.cluster_gate_bias = _finite_or(payload.get("cluster_gate_bias", 0.0), 0.0)
            self.host_vocab = list(payload.get("host_vocab", ["unknown"]))
        else:
            self.cluster_gate_temp = 1.0
            self.cluster_gate_bias = 0.0
            self.host_vocab = ["unknown"]
        self.threshold = _finite_or(threshold, payload_threshold) if threshold is not None else float(payload_threshold)
        self.neutral_risk_weight = _finite_or(
            neutral_risk_weight,
            _finite_or(payload.get("neutral_risk_weight", 1.0), 1.0),
        ) if neutral_risk_weight is not None else _finite_or(payload.get("neutral_risk_weight", 1.0), 1.0)
        self.positive_threshold = _finite_or(
            positive_threshold,
            _finite_or(payload.get("positive_threshold", self.threshold), self.threshold),
        ) if positive_threshold is not None else _finite_or(payload.get("positive_threshold", self.threshold), self.threshold)
        self.neutral_threshold = _finite_or(
            neutral_threshold,
            _finite_or(payload.get("neutral_threshold", 0.5), 0.5),
        ) if neutral_threshold is not None else _finite_or(payload.get("neutral_threshold", 0.5), 0.5)
        if self.model_family == "set_predictor_v2":
            self.model = HostConditionedLocalConflictSetPredictor.from_checkpoint(self.checkpoint_path, map_location=self.device)
        elif self.model_type == "action_policy":
            self.model = GraphAssocCommitPolicy.from_checkpoint(self.checkpoint_path, map_location=self.device)
        else:
            self.model = LocalConflictCommitRefiner.from_checkpoint(self.checkpoint_path, map_location=self.device)
        self.model.to(self.device)
        self.model.eval()

    def _action_policy_decision_score(
        self,
        *,
        gain_pred: float,
        selection_score: float,
        legacy_policy_score: float,
        policy_score: float,
        action_margin: float,
        router_margin: float,
        router_confidence: float,
        router_entropy: float,
        positive_probability: float,
        neutral_probability: float,
    ) -> tuple[float, float]:
        if self.decision_mode == "gain_pred":
            return float(gain_pred), float(self.threshold)
        if self.decision_mode == "action_margin":
            return float(action_margin), float(self.threshold)
        if self.decision_mode == "selection_score":
            return float(selection_score), float(self.threshold)
        if self.decision_mode == "legacy_policy_score":
            return float(legacy_policy_score), float(self.threshold)
        if self.decision_mode == "router_margin":
            return float(router_margin), float(self.threshold)
        if self.decision_mode == "router_confidence":
            return float(router_confidence), float(self.threshold)
        if self.decision_mode == "router_entropy":
            return float(-router_entropy), float(self.threshold)
        if self.decision_mode == "positive_probability":
            return float(positive_probability), float(self.threshold)
        if self.decision_mode == "dual_threshold":
            positive_margin = float(positive_probability - self.positive_threshold)
            neutral_margin = float(self.neutral_threshold - neutral_probability)
            return float(min(positive_margin, neutral_margin)), 0.0
        if self.decision_mode == "positive_minus_neutral":
            return float(positive_probability - neutral_probability), float(self.threshold)
        if self.decision_mode == "positive_times_one_minus_neutral":
            return float(positive_probability * max(0.0, 1.0 - neutral_probability)), float(self.threshold)
        if self.decision_mode == "min_positive_vs_one_minus_neutral":
            return float(min(positive_probability, max(0.0, 1.0 - neutral_probability))), float(self.threshold)
        if self.decision_mode == "positive_minus_weighted_neutral":
            return (
                float(positive_probability - self.neutral_risk_weight * neutral_probability),
                float(self.threshold),
            )
        return float(policy_score), float(self.threshold)

    def _build_tensors(self, candidate_row: Dict[str, Any]) -> Dict[str, Any]:
        row_states = list(candidate_row.get("row_states", []))
        col_states = list(candidate_row.get("col_states", []))
        edge_meta_rows = _merge_required_edge_rows(candidate_row)
        if not row_states or not col_states or not edge_meta_rows:
            raise ValueError("candidate row is missing row_states/col_states/edge_meta")

        track_rows = [_safe_int(v.get("row_idx", -1), -1) for v in row_states]
        det_rows = [_safe_int(v.get("col_idx", -1), -1) for v in col_states]
        if any(v < 0 for v in track_rows) or any(v < 0 for v in det_rows):
            raise ValueError("candidate row contains invalid row_idx/col_idx")

        row_to_local = {int(row_idx): local_idx for local_idx, row_idx in enumerate(track_rows)}
        col_to_local = {int(col_idx): local_idx for local_idx, col_idx in enumerate(det_rows)}
        rank_frac = _infer_edge_rank_frac(edge_meta_rows)

        edge_features: List[List[float]] = []
        edge_det_index: List[int] = []
        edge_track_index: List[int] = []
        utility_by_det: Dict[int, List[float]] = {}

        for edge in edge_meta_rows:
            row_idx = _safe_int(edge.get("row_idx", -1), -1)
            col_idx = _safe_int(edge.get("col_idx", -1), -1)
            if row_idx not in row_to_local or col_idx not in col_to_local:
                continue
            local_track_idx = int(row_to_local[row_idx])
            local_det_idx = int(col_to_local[col_idx])
            track_state = row_states[local_track_idx]
            utility = _safe_float(edge.get("utility", 0.0), 0.0)
            feat = [
                1.0 - _safe_float(edge.get("cost", 1.0), 1.0),
                utility,
                _safe_float(edge.get("box_iou", 0.0), 0.0),
                _safe_float(track_state.get("gap", 0.0), 0.0),
                _safe_float(track_state.get("tracklet_len", 0.0), 0.0),
                float(rank_frac.get((row_idx, col_idx), 1.0)),
            ]
            edge_features.append([float(x) for x in feat])
            edge_det_index.append(local_det_idx)
            edge_track_index.append(local_track_idx)
            utility_by_det.setdefault(local_det_idx, []).append(float(utility))

        degree = compute_component_degree_features(
            num_detections=len(col_states),
            num_tracks=len(row_states),
            edge_det_index=edge_det_index,
            edge_track_index=edge_track_index,
        )
        row_degree = degree["row_degree"].tolist()
        col_degree = degree["col_degree"].tolist()

        det_features: List[List[float]] = []
        for local_det_idx, det in enumerate(col_states):
            utilities = sorted(utility_by_det.get(local_det_idx, []), reverse=True)
            top_margin = 0.0
            if len(utilities) == 1:
                top_margin = float(utilities[0])
            elif len(utilities) >= 2:
                top_margin = float(utilities[0] - utilities[1])
            det_features.append(
                [
                    _safe_float(det.get("score", 0.0), 0.0),
                    float(row_degree[local_det_idx]) if local_det_idx < len(row_degree) else 0.0,
                    float(top_margin),
                    float(_score_entropy(utilities)),
                ]
            )

        track_features: List[List[float]] = []
        for local_track_idx, track in enumerate(row_states):
            track_features.append(
                [
                    _safe_float(track.get("gap", 0.0), 0.0),
                    _safe_float(track.get("tracklet_len", 0.0), 0.0),
                    float(col_degree[local_track_idx]) if local_track_idx < len(col_degree) else 0.0,
                ]
            )

        row_degree_tensor = torch.tensor(row_degree, dtype=torch.float32)
        col_degree_tensor = torch.tensor(col_degree, dtype=torch.float32)
        introduced_rows = list(candidate_row.get("introduced_rows", []))
        suppressed_rows = list(candidate_row.get("suppressed_rows", []))
        recent_owner_rows = list(candidate_row.get("recent_owner_rows", []))
        protected_young_active_rows = list(candidate_row.get("protected_young_active_rows", []))
        cluster_features = [
            float(len(col_states)),
            float(len(row_states)),
            float(len(edge_features)),
            float(row_degree_tensor.mean().item()) if row_degree_tensor.numel() > 0 else 0.0,
            float(row_degree_tensor.max().item()) if row_degree_tensor.numel() > 0 else 0.0,
            float(col_degree_tensor.mean().item()) if col_degree_tensor.numel() > 0 else 0.0,
            float(col_degree_tensor.max().item()) if col_degree_tensor.numel() > 0 else 0.0,
            float(_safe_float(candidate_row.get("utility_gain", 0.0), 0.0)),
            float(_safe_float(candidate_row.get("cost_delta", 0.0), 0.0)),
            float(_safe_float(candidate_row.get("baseline_cost", 0.0), 0.0)),
            float(_safe_float(candidate_row.get("chosen_cost", 0.0), 0.0)),
            float(len(introduced_rows)),
            float(len(suppressed_rows)),
            float(len(recent_owner_rows)),
            float(len(protected_young_active_rows)),
        ]

        expected_cluster_dim = int(getattr(self.model, "cluster_dim", len(cluster_features)))
        if len(cluster_features) > expected_cluster_dim:
            cluster_features = cluster_features[:expected_cluster_dim]
        elif len(cluster_features) < expected_cluster_dim:
            cluster_features = cluster_features + [0.0] * (expected_cluster_dim - len(cluster_features))

        baseline_local = _pair_local_map(
            list(candidate_row.get("baseline_pairs", [])),
            col_to_local=col_to_local,
            row_to_local=row_to_local,
        )
        chosen_local = _pair_local_map(
            list(candidate_row.get("chosen_pairs", [])),
            col_to_local=col_to_local,
            row_to_local=row_to_local,
        )

        return {
            "det_features": torch.tensor(det_features, dtype=torch.float32, device=self.device),
            "track_features": torch.tensor(track_features, dtype=torch.float32, device=self.device),
            "edge_features": torch.tensor(edge_features, dtype=torch.float32, device=self.device),
            "edge_det_index": torch.tensor(edge_det_index, dtype=torch.long, device=self.device),
            "edge_track_index": torch.tensor(edge_track_index, dtype=torch.long, device=self.device),
            "cluster_features": torch.tensor(cluster_features, dtype=torch.float32, device=self.device),
            "baseline_local": baseline_local,
            "chosen_local": chosen_local,
            "num_detections": len(col_states),
            "num_tracks": len(row_states),
        }

    @staticmethod
    def _tlwh_to_cxcywh(tlwh: Any) -> torch.Tensor:
        values: List[float]
        if isinstance(tlwh, (list, tuple)):
            values = [float(v) for v in tlwh[:4]]
        else:
            try:
                values = [float(v) for v in list(tlwh)[:4]]
            except Exception:
                values = []
        x = float(values[0]) if len(values) > 0 else 0.0
        y = float(values[1]) if len(values) > 1 else 0.0
        w = max(float(values[2]) if len(values) > 2 else 1e-6, 1e-6)
        h = max(float(values[3]) if len(values) > 3 else 1e-6, 1e-6)
        return torch.tensor([x + 0.5 * w, y + 0.5 * h, w, h], dtype=torch.float32)

    def _build_set_predictor_tensors(self, candidate_row: Dict[str, Any]) -> Dict[str, Any]:
        row_states = list(candidate_row.get("row_states", []))
        col_states = list(candidate_row.get("col_states", []))
        edge_meta_rows = _merge_required_edge_rows(candidate_row)
        if not row_states or not col_states or not edge_meta_rows:
            raise ValueError("candidate row is missing row_states/col_states/edge_meta")

        device = self.device
        row_to_local = {int(row.get("row_idx", idx)): int(idx) for idx, row in enumerate(row_states)}
        col_to_local = {int(col.get("col_idx", idx)): int(idx) for idx, col in enumerate(col_states)}

        det_scores = torch.tensor(
            [_safe_float(col.get("score", 0.0), 0.0) for col in col_states],
            dtype=torch.float32,
            device=device,
        )
        det_boxes = torch.stack([self._tlwh_to_cxcywh(col.get("tlwh", None)) for col in col_states], dim=0).to(device)
        track_boxes = torch.stack([self._tlwh_to_cxcywh(row.get("tlwh", None)) for row in row_states], dim=0).to(device)
        track_gap = torch.tensor(
            [_safe_float(row.get("gap", 0.0), 0.0) for row in row_states],
            dtype=torch.float32,
            device=device,
        )
        track_hist_len = torch.tensor(
            [_safe_float(row.get("tracklet_len", 0.0), 0.0) for row in row_states],
            dtype=torch.float32,
            device=device,
        )

        edge_records: List[Dict[str, Any]] = []
        det_to_edge_indices: Dict[int, List[int]] = defaultdict(list)
        track_to_edge_indices: Dict[int, List[int]] = defaultdict(list)
        row_entropy_values: List[float] = []
        row_margin_values: List[float] = []
        for edge in edge_meta_rows:
            row_idx = _safe_int(edge.get("row_idx", -1), -1)
            col_idx = _safe_int(edge.get("col_idx", -1), -1)
            if row_idx not in row_to_local or col_idx not in col_to_local:
                continue
            local_track_idx = int(row_to_local[row_idx])
            local_det_idx = int(col_to_local[col_idx])
            cost = max(_safe_float(edge.get("cost", 0.0), 0.0), 0.0)
            det_score = _safe_float(edge.get("det_score", 0.0), 0.0)
            box_iou = _safe_float(edge.get("box_iou", 0.0), 0.0)
            base_score = float(det_score)
            refined_score = float(1.0 / (1.0 + cost))
            motion_score = float(box_iou)
            geom = pair_geometry_features(
                det_boxes[local_det_idx].view(1, 4),
                track_boxes[local_track_idx].view(1, 4),
            )
            edge_records.append(
                {
                    "det_local_idx": int(local_det_idx),
                    "track_local_idx": int(local_track_idx),
                    "base_score_raw": float(base_score),
                    "refined_score_raw": float(refined_score),
                    "motion_score_raw": float(motion_score),
                    "track_gap": float(track_gap[local_track_idx].item()),
                    "track_hist_len": float(track_hist_len[local_track_idx].item()),
                    "det_score": float(det_scores[local_det_idx].item()),
                    "iou": float(geom["iou"].view(-1)[0].item()),
                    "bbox_dist_score": float(geom["bbox_dist_score"].view(-1)[0].item()),
                    "delta_cx_norm": float(geom["delta_cx_norm"].view(-1)[0].item()),
                    "delta_cy_norm": float(geom["delta_cy_norm"].view(-1)[0].item()),
                    "delta_log_w": float(geom["delta_log_w"].view(-1)[0].item()),
                    "delta_log_h": float(geom["delta_log_h"].view(-1)[0].item()),
                }
            )
            edge_idx = int(len(edge_records) - 1)
            det_to_edge_indices[local_det_idx].append(edge_idx)
            track_to_edge_indices[local_track_idx].append(edge_idx)

        if not edge_records:
            raise ValueError("candidate row has no usable edges")

        row_degree = torch.zeros((len(col_states),), device=device, dtype=torch.float32)
        col_degree = torch.zeros((len(row_states),), device=device, dtype=torch.float32)
        for rec in edge_records:
            row_degree[int(rec["det_local_idx"])] += 1.0
            col_degree[int(rec["track_local_idx"])] += 1.0

        def _zscore_tensor(values: torch.Tensor) -> torch.Tensor:
            values = values.to(dtype=torch.float32)
            if values.numel() <= 1:
                return torch.zeros_like(values)
            mean = values.mean()
            std = values.std(unbiased=False)
            if float(std.item()) < 1e-6:
                return torch.zeros_like(values)
            return (values - mean) / std

        det_features: List[List[float]] = []
        for local_det_idx, det in enumerate(col_states):
            edge_indices = det_to_edge_indices.get(local_det_idx, [])
            if edge_indices:
                base_vals = torch.tensor([edge_records[idx]["base_score_raw"] for idx in edge_indices], device=device)
                refined_vals = torch.tensor([edge_records[idx]["refined_score_raw"] for idx in edge_indices], device=device)
                motion_vals = torch.tensor([edge_records[idx]["motion_score_raw"] for idx in edge_indices], device=device)
                row_base_z = _zscore_tensor(base_vals)
                row_refined_z = _zscore_tensor(refined_vals)
                row_motion_z = _zscore_tensor(motion_vals)
                row_refined_softmax = torch.softmax(refined_vals - refined_vals.max(), dim=0)
                row_top1 = float(refined_vals.max().item())
                if refined_vals.numel() > 1:
                    top2 = torch.topk(refined_vals, k=2, dim=0, sorted=True).values
                    row_margin = float((top2[0] - top2[1]).item())
                else:
                    row_margin = float(row_top1)
                row_entropy = float(
                    (-(row_refined_softmax * torch.log(row_refined_softmax.clamp(min=1e-8)))).sum().item()
                )
                row_entropy_values.append(row_entropy)
                row_margin_values.append(row_margin)
                order = sorted(
                    range(len(edge_indices)),
                    key=lambda pos: float(edge_records[edge_indices[pos]]["refined_score_raw"]),
                    reverse=True,
                )
                for rank_pos, pos in enumerate(order):
                    edge_idx = edge_indices[pos]
                    edge_records[edge_idx]["base_score_row_z"] = float(row_base_z[pos].item())
                    edge_records[edge_idx]["refined_score_row_z"] = float(row_refined_z[pos].item())
                    edge_records[edge_idx]["motion_score_row_z"] = float(row_motion_z[pos].item())
                    edge_records[edge_idx]["refined_score_row_softmax"] = float(row_refined_softmax[pos].item())
                    edge_records[edge_idx]["refined_gap_to_row_top1"] = float(row_top1 - float(edge_records[edge_idx]["refined_score_raw"]))
                    edge_records[edge_idx]["rank_frac"] = float(rank_pos + 1) / float(max(len(edge_indices), 1))
            else:
                row_top1 = 0.0
                row_margin = 0.0
                row_entropy = 0.0
                row_entropy_values.append(row_entropy)
                row_margin_values.append(row_margin)
                for edge_idx in edge_indices:
                    edge_records[edge_idx]["base_score_row_z"] = 0.0
                    edge_records[edge_idx]["refined_score_row_z"] = 0.0
                    edge_records[edge_idx]["motion_score_row_z"] = 0.0
                    edge_records[edge_idx]["refined_score_row_softmax"] = 0.0
                    edge_records[edge_idx]["refined_gap_to_row_top1"] = 0.0
                    edge_records[edge_idx]["rank_frac"] = 0.0

            det_box = det_boxes[local_det_idx]
            w = max(float(det_box[2].item()), 1e-6)
            h = max(float(det_box[3].item()), 1e-6)
            det_features.append(
                [
                    float(det_scores[local_det_idx].item()),
                    float(row_degree[local_det_idx].item()),
                    float(row_margin),
                    float(row_entropy),
                    float(det_box[0].item()),
                    float(det_box[1].item()),
                    float(math.log(w)),
                    float(math.log(h)),
                    float(w / max(h, 1e-6)),
                ]
            )

        track_features: List[List[float]] = []
        for local_track_idx, track in enumerate(row_states):
            track_box = track_boxes[local_track_idx]
            tw = max(float(track_box[2].item()), 1e-6)
            th = max(float(track_box[3].item()), 1e-6)
            track_features.append(
                [
                    float(math.log1p(max(float(track_gap[local_track_idx].item()), 0.0))),
                    float(math.log1p(max(float(track_hist_len[local_track_idx].item()), 0.0))),
                    float(col_degree[local_track_idx].item()),
                    float(track_box[0].item()),
                    float(track_box[1].item()),
                    float(math.log(tw)),
                    float(math.log(th)),
                    float(tw / max(th, 1e-6)),
                ]
            )

        track_col_z: Dict[int, List[float]] = {}
        for local_track_idx, edge_indices in track_to_edge_indices.items():
            refined_vals = torch.tensor([edge_records[idx]["refined_score_raw"] for idx in edge_indices], device=device)
            col_z = _zscore_tensor(refined_vals)
            track_col_z[local_track_idx] = [float(v) for v in col_z.tolist()]
            for offset, edge_idx in enumerate(edge_indices):
                edge_records[edge_idx]["refined_score_col_z"] = float(col_z[offset].item())
        for local_track_idx, edge_indices in enumerate([track_to_edge_indices.get(i, []) for i in range(len(row_states))]):
            if local_track_idx not in track_col_z:
                continue
            for offset, edge_idx in enumerate(edge_indices):
                edge_records[edge_idx]["refined_score_col_z"] = float(track_col_z[local_track_idx][offset])

        cluster_features = [
            float(len(col_states)),
            float(len(row_states)),
            float(len(edge_records)),
            float(row_degree.mean().item()) if row_degree.numel() > 0 else 0.0,
            float(row_degree.max().item()) if row_degree.numel() > 0 else 0.0,
            float(col_degree.mean().item()) if col_degree.numel() > 0 else 0.0,
            float(col_degree.max().item()) if col_degree.numel() > 0 else 0.0,
            float(sum(row_entropy_values) / len(row_entropy_values)) if row_entropy_values else 0.0,
            float(max(row_entropy_values)) if row_entropy_values else 0.0,
            float(sum(row_margin_values) / len(row_margin_values)) if row_margin_values else 0.0,
            float(max(row_margin_values)) if row_margin_values else 0.0,
        ]

        host_variant_id = _safe_int(candidate_row.get("host_variant_id", 0), 0)
        if host_variant_id < 0:
            host_variant_id = 0

        baseline_local = _pair_local_map(
            list(candidate_row.get("baseline_pairs", [])),
            col_to_local=col_to_local,
            row_to_local=row_to_local,
        )
        chosen_local = _pair_local_map(
            list(candidate_row.get("chosen_pairs", [])),
            col_to_local=col_to_local,
            row_to_local=row_to_local,
        )

        edge_features = [
            [
                float(rec.get("base_score_raw", 0.0)),
                float(rec.get("refined_score_raw", 0.0)),
                float(rec.get("motion_score_raw", 0.0)),
                float(rec.get("base_score_row_z", 0.0)),
                float(rec.get("refined_score_row_z", 0.0)),
                float(rec.get("motion_score_row_z", 0.0)),
                float(rec.get("refined_score_row_softmax", 0.0)),
                float(rec.get("refined_gap_to_row_top1", 0.0)),
                float(rec.get("rank_frac", 0.0)),
                float(rec.get("refined_score_col_z", 0.0)),
                float(rec.get("refined_score_raw", 0.0) - rec.get("base_score_raw", 0.0)),
                float(rec.get("motion_score_raw", 0.0) - rec.get("refined_score_raw", 0.0)),
                float(rec.get("iou", 0.0)),
                float(rec.get("bbox_dist_score", 0.0)),
                float(rec.get("delta_cx_norm", 0.0)),
                float(rec.get("delta_cy_norm", 0.0)),
                float(rec.get("delta_log_w", 0.0)),
                float(rec.get("delta_log_h", 0.0)),
            ]
            for rec in edge_records
        ]

        dense_edge_mask = torch.zeros((len(col_states), len(row_states)), device=device, dtype=torch.bool)
        dense_refined_scores = torch.zeros((len(col_states), len(row_states)), device=device, dtype=torch.float32)
        for rec in edge_records:
            dense_edge_mask[int(rec["det_local_idx"]), int(rec["track_local_idx"])] = True
            dense_refined_scores[int(rec["det_local_idx"]), int(rec["track_local_idx"])] = float(rec["refined_score_raw"])

        return {
            "det_features": torch.tensor(det_features, device=device, dtype=torch.float32),
            "track_features": torch.tensor(track_features, device=device, dtype=torch.float32),
            "edge_features": torch.tensor(edge_features, device=device, dtype=torch.float32),
            "edge_det_index": torch.tensor([int(rec["det_local_idx"]) for rec in edge_records], device=device, dtype=torch.long),
            "edge_track_index": torch.tensor([int(rec["track_local_idx"]) for rec in edge_records], device=device, dtype=torch.long),
            "cluster_features": torch.tensor(cluster_features, device=device, dtype=torch.float32),
            "baseline_local": baseline_local,
            "chosen_local": chosen_local,
            "num_detections": len(col_states),
            "num_tracks": len(row_states),
            "dense_edge_mask": dense_edge_mask,
            "dense_refined_scores": dense_refined_scores,
            "host_variant_id": int(host_variant_id),
        }

    def _build_tensors_for_assignment(
        self,
        candidate_row: Dict[str, Any],
        assignment_local: Dict[int, int],
        *,
        assignment_key: str = "assignment_local",
    ) -> Dict[str, Any]:
        payload = dict(self._build_tensors(candidate_row))
        payload[assignment_key] = dict(assignment_local)
        return payload

    @staticmethod
    def _assignment_score(dense_logits: torch.Tensor, assignment_local: Dict[int, int]) -> float:
        if dense_logits.ndim != 2:
            return 0.0
        num_tracks = int(dense_logits.shape[1] - 1)
        total = 0.0
        for det_local_idx in range(int(dense_logits.shape[0])):
            track_local_idx = assignment_local.get(int(det_local_idx), None)
            chosen_col = int(track_local_idx) if track_local_idx is not None else num_tracks
            total += float(dense_logits[det_local_idx, chosen_col].item())
        return total

    def score_candidate_row(self, candidate_row: Dict[str, Any]) -> GraphAssocCommitScore:
        if self.model_family == "set_predictor_v2":
            payload = self._build_set_predictor_tensors(candidate_row)
        else:
            payload = self._build_tensors(candidate_row)
        with torch.inference_mode():
            if self.model_family == "set_predictor_v2":
                outputs = self.model(
                    det_features=payload["det_features"],
                    track_features=payload["track_features"],
                    edge_features=payload["edge_features"],
                    edge_det_index=payload["edge_det_index"],
                    edge_track_index=payload["edge_track_index"],
                    cluster_features=payload["cluster_features"],
                    host_variant_id=torch.tensor([int(payload["host_variant_id"])], device=self.device, dtype=torch.long),
                )
                dense_logits = HostConditionedLocalConflictSetPredictor.build_dense_assignment_logits(
                    num_detections=int(payload["det_features"].shape[0]),
                    num_tracks=int(payload["track_features"].shape[0]),
                    edge_logits=outputs["edge_logits"],
                    edge_det_index=payload["edge_det_index"],
                    edge_track_index=payload["edge_track_index"],
                    defer_logits=outputs["defer_logits"],
                )
            else:
                outputs = self.model(
                    det_features=payload["det_features"],
                    track_features=payload["track_features"],
                    edge_features=payload["edge_features"],
                    edge_det_index=payload["edge_det_index"],
                    edge_track_index=payload["edge_track_index"],
                    cluster_features=payload["cluster_features"],
                )
                dense_logits = LocalConflictCommitRefiner.build_dense_assignment_logits(
                    num_detections=int(payload["num_detections"]),
                    num_tracks=int(payload["num_tracks"]),
                    edge_logits=outputs["edge_logits"],
                    edge_det_index=payload["edge_det_index"],
                    edge_track_index=payload["edge_track_index"],
                    defer_logits=outputs["defer_logits"],
                )

        baseline_score = self._assignment_score(dense_logits, payload["baseline_local"])
        chosen_score = self._assignment_score(dense_logits, payload["chosen_local"])

        if self.model_family == "set_predictor_v2":
            gate_logit = outputs["cluster_commit_logit"].view(())
            gate_score = torch.sigmoid(
                gate_logit / max(float(self.cluster_gate_temp), 1e-6) + float(self.cluster_gate_bias)
            )
            decision_score = float(gate_score.item())
            accept_threshold = float(self.threshold)
            score_delta = float(decision_score - accept_threshold)
            pred_track_by_det = dense_logits.argmax(dim=-1).detach().cpu().tolist()
            pred_pairs_local: Dict[int, int] = {}
            for det_local_idx, track_or_defer in enumerate(pred_track_by_det):
                if int(track_or_defer) >= int(payload["num_tracks"]):
                    continue
                pred_pairs_local[int(det_local_idx)] = int(track_or_defer)
            pred_score = float(decision_score)
            chosen_better = bool(score_delta >= 0.0)
            gain_pred = 0.0
            selection_score = 0.0
            legacy_policy_score = 0.0
            policy_score = float(decision_score)
            action_margin = float(score_delta)
            router_margin = 0.0
            router_confidence = float(decision_score)
            router_entropy = 0.0
            positive_probability = float(decision_score)
            neutral_probability = 0.0
            reject_probability = float(max(0.0, 1.0 - decision_score))
            decision_mode = "cluster_gate_probability"
        else:
            gain_pred = float(outputs.get("gain_pred", torch.tensor(0.0, device=self.device)).item()) if self.model_type == "action_policy" else 0.0
            selection_score = float(outputs.get("selection_score", torch.tensor(0.0, device=self.device)).item()) if self.model_type == "action_policy" else 0.0
            legacy_policy_score = float(outputs.get("legacy_policy_score", torch.tensor(0.0, device=self.device)).item()) if self.model_type == "action_policy" else 0.0
            policy_score = float(outputs.get("policy_score", torch.tensor(0.0, device=self.device)).item()) if self.model_type == "action_policy" else 0.0
            action_margin = float(outputs.get("action_margin", torch.tensor(0.0, device=self.device)).item()) if self.model_type == "action_policy" else 0.0
            router_margin = float(outputs.get("router_margin", torch.tensor(0.0, device=self.device)).item()) if self.model_type == "action_policy" else 0.0
            router_entropy = float(outputs.get("router_entropy", torch.tensor(0.0, device=self.device)).item()) if self.model_type == "action_policy" else 0.0
            positive_probability = 0.0
            neutral_probability = 0.0
            reject_probability = 0.0
            action_probs = outputs.get("action_probs", None) if self.model_type == "action_policy" else None
            if isinstance(action_probs, torch.Tensor) and action_probs.numel() >= 3:
                flat_probs = action_probs.view(-1)
                positive_probability = float(flat_probs[0].item())
                neutral_probability = float(flat_probs[1].item())
                reject_probability = float(flat_probs[2].item())
            router_probs = outputs.get("router_probs", None) if self.model_type == "action_policy" else None
            if isinstance(router_probs, torch.Tensor) and router_probs.numel() > 0:
                router_confidence = float(router_probs.max().item())
                if int(router_probs.numel()) > 1:
                    top2 = torch.topk(router_probs.view(-1), k=2, sorted=True).values
                    router_margin = float(top2[0].item() - top2[1].item())
                else:
                    router_margin = float(router_confidence)
            else:
                router_confidence = 0.0
            pred_track_by_det = dense_logits.argmax(dim=-1).detach().cpu().tolist()
            pred_pairs_local: Dict[int, int] = {}
            for det_local_idx, track_or_defer in enumerate(pred_track_by_det):
                if int(track_or_defer) >= int(payload["num_tracks"]):
                    continue
                pred_pairs_local[int(det_local_idx)] = int(track_or_defer)
            pred_score = self._assignment_score(dense_logits, pred_pairs_local)
            if self.model_type == "action_policy":
                decision_score, accept_threshold = self._action_policy_decision_score(
                    gain_pred=float(gain_pred),
                    selection_score=float(selection_score),
                    legacy_policy_score=float(legacy_policy_score),
                    policy_score=float(policy_score),
                    action_margin=float(action_margin),
                    router_margin=float(router_margin),
                    router_confidence=float(router_confidence),
                    router_entropy=float(router_entropy),
                    positive_probability=float(positive_probability),
                    neutral_probability=float(neutral_probability),
                )
            else:
                decision_score = float(outputs.get("decision_score", outputs.get("policy_score", outputs.get("gain_pred", torch.tensor(0.0, device=self.device)))).item())
                accept_threshold = 0.0
            if self.model_type == "action_policy":
                score_delta = float(decision_score - accept_threshold)
                pred_score = float(decision_score)
                chosen_better = bool(score_delta >= 0.0)
                decision_mode = str(self.decision_mode or "policy_score")
            else:
                score_delta = float(chosen_score - baseline_score)
                chosen_better = bool(score_delta >= 0.0)
                decision_mode = str(self.decision_mode or "assignment_delta")
        return GraphAssocCommitScore(
            baseline_score=float(baseline_score),
            chosen_score=float(chosen_score),
            score_delta=float(score_delta),
            chosen_better=bool(chosen_better),
            pred_score=float(pred_score),
            gain_pred=float(gain_pred),
            selection_score=float(selection_score),
            legacy_policy_score=float(legacy_policy_score),
            policy_score=float(policy_score),
            action_margin=float(action_margin),
            router_margin=float(router_margin),
            router_confidence=float(router_confidence),
            router_entropy=float(router_entropy),
            positive_probability=float(positive_probability),
            neutral_probability=float(neutral_probability),
            reject_probability=float(reject_probability),
            policy_score_mode=str(getattr(self.model, "policy_score_mode", "")),
            pred_pairs_local=pred_pairs_local,
            chosen_pairs_local=dict(payload["chosen_local"]),
            baseline_pairs_local=dict(payload["baseline_local"]),
            decision_score=float(decision_score),
            model_type=str(self.model_family if self.model_family == "set_predictor_v2" else self.model_type),
            decision_mode=str(decision_mode),
            threshold=float(accept_threshold),
            positive_threshold=float(self.positive_threshold),
            neutral_threshold=float(self.neutral_threshold),
            neutral_risk_weight=float(self.neutral_risk_weight),
            accept=bool(chosen_better),
        )

    def score_candidate_row_variant(
        self,
        candidate_row: Dict[str, Any],
        assignment_local: Dict[int, int],
        *,
        assignment_key: str = "assignment_local",
    ) -> Dict[str, Any]:
        """
        Score an arbitrary assignment variant inside the same block.

        This reuses the current learned scorer, but lets the runtime compare multiple
        candidate assignments before deciding which one to commit.
        """
        payload = self._build_tensors_for_assignment(candidate_row, assignment_local, assignment_key=assignment_key)
        with torch.inference_mode():
            if self.model_family == "set_predictor_v2":
                outputs = self.model(
                    det_features=payload["det_features"],
                    track_features=payload["track_features"],
                    edge_features=payload["edge_features"],
                    edge_det_index=payload["edge_det_index"],
                    edge_track_index=payload["edge_track_index"],
                    cluster_features=payload["cluster_features"],
                    host_variant_id=torch.tensor([int(payload["host_variant_id"])], device=self.device, dtype=torch.long),
                )
                dense_logits = HostConditionedLocalConflictSetPredictor.build_dense_assignment_logits(
                    num_detections=int(payload["det_features"].shape[0]),
                    num_tracks=int(payload["track_features"].shape[0]),
                    edge_logits=outputs["edge_logits"],
                    edge_det_index=payload["edge_det_index"],
                    edge_track_index=payload["edge_track_index"],
                    defer_logits=outputs["defer_logits"],
                )
            else:
                outputs = self.model(
                    det_features=payload["det_features"],
                    track_features=payload["track_features"],
                    edge_features=payload["edge_features"],
                    edge_det_index=payload["edge_det_index"],
                    edge_track_index=payload["edge_track_index"],
                    cluster_features=payload["cluster_features"],
                )
                dense_logits = LocalConflictCommitRefiner.build_dense_assignment_logits(
                    num_detections=int(payload["num_detections"]),
                    num_tracks=int(payload["num_tracks"]),
                    edge_logits=outputs["edge_logits"],
                    edge_det_index=payload["edge_det_index"],
                    edge_track_index=payload["edge_track_index"],
                    defer_logits=outputs["defer_logits"],
                )

        if self.model_family == "set_predictor_v2":
            gate_logit = outputs["cluster_commit_logit"].view(())
            decision_score = float(
                torch.sigmoid(gate_logit / max(float(self.cluster_gate_temp), 1e-6) + float(self.cluster_gate_bias)).item()
            )
            accept_threshold = float(self.threshold)
            baseline_score = accept_threshold
            variant_score = decision_score
            assignment_delta = float(self._assignment_score(dense_logits, dict(assignment_local)) - self._assignment_score(dense_logits, payload["baseline_local"]))
            decision_margin = float(decision_score - accept_threshold)
            chosen_better = bool(decision_margin >= 0.0)
            pred_score = float(decision_score)
        elif self.model_type == "action_policy":
            decision_score, accept_threshold = self._action_policy_decision_score(
                gain_pred=float(outputs.get("gain_pred", torch.tensor(0.0, device=self.device)).item()),
                selection_score=float(outputs.get("selection_score", torch.tensor(0.0, device=self.device)).item()),
                legacy_policy_score=float(outputs.get("legacy_policy_score", torch.tensor(0.0, device=self.device)).item()),
                policy_score=float(outputs.get("policy_score", torch.tensor(0.0, device=self.device)).item()),
                action_margin=float(outputs.get("action_margin", torch.tensor(0.0, device=self.device)).item()),
                router_margin=float(outputs.get("router_margin", torch.tensor(0.0, device=self.device)).item()),
                router_confidence=float(outputs.get("router_probs", torch.tensor([], device=self.device)).max().item())
                if isinstance(outputs.get("router_probs", None), torch.Tensor) and outputs["router_probs"].numel() > 0
                else 0.0,
                router_entropy=float(outputs.get("router_entropy", torch.tensor(0.0, device=self.device)).item()),
                positive_probability=float(outputs.get("action_probs", torch.tensor([], device=self.device)).reshape(-1)[0].item())
                if isinstance(outputs.get("action_probs", None), torch.Tensor) and outputs["action_probs"].numel() >= 3
                else 0.0,
                neutral_probability=float(outputs.get("action_probs", torch.tensor([], device=self.device)).reshape(-1)[1].item())
                if isinstance(outputs.get("action_probs", None), torch.Tensor) and outputs["action_probs"].numel() >= 3
                else 0.0,
            )
            baseline_score = self._assignment_score(dense_logits, payload["baseline_local"])
            variant_score = self._assignment_score(dense_logits, dict(assignment_local))
            assignment_delta = float(variant_score - baseline_score)
            decision_margin = float(decision_score - accept_threshold)
            chosen_better = bool(assignment_delta >= 0.0)
            pred_score = float(decision_score)
        else:
            decision_score = float(outputs.get("decision_score", outputs.get("policy_score", outputs.get("gain_pred", torch.tensor(0.0, device=self.device)))).item())
            accept_threshold = 0.0
            baseline_score = self._assignment_score(dense_logits, payload["baseline_local"])
            variant_score = self._assignment_score(dense_logits, dict(assignment_local))
            assignment_delta = float(variant_score - baseline_score)
            decision_margin = float(decision_score - accept_threshold)
            chosen_better = bool(assignment_delta >= 0.0)
            pred_score = float(decision_score)
        return {
            "baseline_score": float(baseline_score),
            "variant_score": float(variant_score),
            "assignment_delta": float(assignment_delta),
            "decision_score": float(decision_score),
            "decision_margin": float(decision_margin),
            "accept_threshold": float(accept_threshold),
            "score_delta": float(decision_margin if self.model_family == "set_predictor_v2" else assignment_delta),
            "chosen_better": bool(chosen_better),
            "pred_score": float(pred_score),
            "assignment_local": dict(assignment_local),
            "dense_logits": dense_logits,
            "outputs": outputs,
        }
