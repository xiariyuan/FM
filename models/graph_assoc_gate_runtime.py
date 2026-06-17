from __future__ import annotations

import os.path as osp
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch

from models.graph_assoc_gate import GraphAssocDualHeadGate, GraphAssocGate


GRAPH_ASSOC_GATE_FEATURE_NAMES = [
    "num_rows",
    "num_cols",
    "num_edges",
    "num_ambiguous_rows",
    "num_ambiguous_cols",
    "num_reclaimable_rows",
    "num_recent_owner_rows",
    "num_introduced_rows",
    "num_suppressed_rows",
    "num_protected_young_active_rows",
    "num_suppressed_protected_rows",
    "num_suppressed_stale_lost_rows",
    "num_active_introduced_rows",
    "row_involved_block",
    "col_only_block",
    "has_gap_advantaged_reclaim",
    "baseline_pair_count",
    "chosen_pair_count",
    "pair_count_delta",
    "baseline_reclaim_matches",
    "chosen_reclaim_matches",
    "reclaim_match_delta",
    "baseline_recent_matches",
    "chosen_recent_matches",
    "recent_match_delta",
    "baseline_tracked_matches",
    "chosen_tracked_matches",
    "tracked_match_delta",
    "baseline_lost_matches",
    "chosen_lost_matches",
    "lost_match_delta",
    "suppressed_tracked_rows",
    "suppressed_lost_rows",
    "suppressed_recent_owner_rows",
    "suppressed_mean_gap",
    "suppressed_max_gap",
    "baseline_cost",
    "chosen_cost",
    "cost_delta",
    "baseline_utility",
    "chosen_utility",
    "utility_gain",
    "required_min_gain",
    "required_max_cost_delta",
    "gain_margin",
    "cost_slack",
    "enumerated_assignments",
    "max_introduced_utility",
    "mean_row_gap",
    "max_row_gap",
    "mean_tracklet_len",
    "max_tracklet_len",
    "mean_row_score",
    "min_row_score",
    "max_row_score",
    "mean_reclaim_strength",
    "max_reclaim_strength",
    "mean_recent_penalty",
    "max_recent_penalty",
    "tracked_row_frac",
    "lost_row_frac",
    "mean_edge_cost",
    "min_edge_cost",
    "max_edge_cost",
    "mean_edge_utility",
    "min_edge_utility",
    "max_edge_utility",
    "mean_edge_iou",
    "max_edge_iou",
    "mean_edge_det_score",
    "max_edge_det_score",
    "baseline_pair_mean_cost",
    "chosen_pair_mean_cost",
    "baseline_pair_mean_utility",
    "chosen_pair_mean_utility",
    "learned_commit_available",
    "learned_commit_score_delta",
    "learned_commit_baseline_score",
    "learned_commit_chosen_score",
    "learned_commit_pred_score",
    "learned_commit_chosen_better",
    "learned_commit_score_margin_pass",
]

DUAL_HEAD_DECISION_MODES = {
    "positive_minus_weighted_neutral",
    "positive_minus_neutral",
    "positive_times_one_minus_neutral",
    "min_positive_vs_one_minus_neutral",
    "dual_threshold",
}


def _resolve_checkpoint_path(checkpoint_path: str) -> str:
    requested_path = str(checkpoint_path)
    if osp.isabs(requested_path):
        return requested_path
    repo_root = osp.abspath(osp.join(osp.dirname(__file__), ".."))
    return osp.join(repo_root, requested_path)


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


def _normalize_decision_mode(mode: str, model_type: str) -> str:
    requested = str(mode or "").strip().lower()
    if not requested:
        return "positive_minus_weighted_neutral" if str(model_type) == "dual_head" else "positive_probability"
    aliases = {
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
    }
    normalized = aliases.get(requested, requested)
    if str(model_type) != "dual_head":
        return "positive_probability"
    if normalized not in DUAL_HEAD_DECISION_MODES:
        return "positive_minus_weighted_neutral"
    return normalized


def _bool01(value: Any) -> float:
    return 1.0 if bool(value) else 0.0


def _list_count(values: Any) -> int:
    if isinstance(values, list):
        return int(len(values))
    return 0


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    if not vals:
        return 0.0
    return float(sum(vals) / float(len(vals)))


def _min(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    if not vals:
        return 0.0
    return float(min(vals))


def _max(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    if not vals:
        return 0.0
    return float(max(vals))


def _fraction(count: int, total: int) -> float:
    if int(total) <= 0:
        return 0.0
    return float(count) / float(total)


def _pair_mean(pair_rows: Sequence[Dict[str, Any]], key: str) -> float:
    return _mean(_safe_float(row.get(key, 0.0), 0.0) for row in pair_rows)


def infer_rules_passed_before_gate(candidate_row: Dict[str, Any]) -> bool:
    explicit_value = candidate_row.get("rules_passed_before_learned_gate", None)
    if explicit_value not in (None, ""):
        return bool(explicit_value)

    accepted = bool(candidate_row.get("accepted", False))
    decision_source = str(candidate_row.get("decision_source", ""))
    if accepted and decision_source in {"rules", "rules+learned_gate"}:
        return True

    baseline_pairs = list(candidate_row.get("baseline_pairs", []) or [])
    chosen_pairs = list(candidate_row.get("chosen_pairs", []) or [])
    if len(chosen_pairs) != len(baseline_pairs):
        return False

    utility_gain = _safe_float(candidate_row.get("utility_gain", 0.0), 0.0)
    required_min_gain = _safe_float(candidate_row.get("required_min_gain", 0.0), 0.0)
    if utility_gain < required_min_gain:
        return False

    cost_delta = _safe_float(candidate_row.get("cost_delta", 0.0), 0.0)
    required_max_cost_delta = _safe_float(candidate_row.get("required_max_cost_delta", 0.0), 0.0)
    if cost_delta > required_max_cost_delta:
        return False

    chosen_reclaim = _safe_int(candidate_row.get("chosen_reclaim_matches", 0), 0)
    baseline_reclaim = _safe_int(candidate_row.get("baseline_reclaim_matches", 0), 0)
    chosen_recent = _safe_int(candidate_row.get("chosen_recent_matches", 0), 0)
    baseline_recent = _safe_int(candidate_row.get("baseline_recent_matches", 0), 0)
    if chosen_reclaim < baseline_reclaim:
        return False
    if chosen_reclaim == baseline_reclaim and chosen_recent >= baseline_recent:
        return False

    protected_rows = list(candidate_row.get("protected_young_active_rows", []) or [])
    if protected_rows and not bool(candidate_row.get("has_gap_advantaged_reclaim", False)):
        return False

    stale_rows = list(candidate_row.get("suppressed_stale_lost_rows", []) or [])
    active_introduced_rows = list(candidate_row.get("active_introduced_rows", []) or [])
    if stale_rows and active_introduced_rows:
        if _safe_float(candidate_row.get("max_introduced_utility", 0.0), 0.0) < 0.0:
            return False

    return True


def build_graph_assoc_gate_feature_vector(candidate_row: Dict[str, Any]) -> List[float]:
    row_states = list(candidate_row.get("row_states", []) or [])
    edge_meta = list(candidate_row.get("edge_meta", []) or [])
    baseline_pair_meta = list(candidate_row.get("baseline_pair_meta", []) or [])
    chosen_pair_meta = list(candidate_row.get("chosen_pair_meta", []) or [])

    row_gaps = [_safe_float(row.get("gap", 0.0), 0.0) for row in row_states]
    row_tracklet_lens = [_safe_float(row.get("tracklet_len", 0.0), 0.0) for row in row_states]
    row_scores = [_safe_float(row.get("score", 0.0), 0.0) for row in row_states]
    row_reclaim = [_safe_float(row.get("reclaim_strength", 0.0), 0.0) for row in row_states]
    row_recent_penalty = [_safe_float(row.get("recent_penalty", 0.0), 0.0) for row in row_states]

    tracked_rows = sum(1 for row in row_states if str(row.get("state", "")) == "Tracked")
    lost_rows = sum(1 for row in row_states if str(row.get("state", "")) in {"Lost", "LongLost"})

    edge_costs = [_safe_float(edge.get("cost", 0.0), 0.0) for edge in edge_meta]
    edge_utilities = [_safe_float(edge.get("utility", 0.0), 0.0) for edge in edge_meta]
    edge_ious = [_safe_float(edge.get("box_iou", 0.0), 0.0) for edge in edge_meta]
    edge_det_scores = [_safe_float(edge.get("det_score", 0.0), 0.0) for edge in edge_meta]

    baseline_pairs = list(candidate_row.get("baseline_pairs", []) or [])
    chosen_pairs = list(candidate_row.get("chosen_pairs", []) or [])
    baseline_pair_count = int(len(baseline_pairs))
    chosen_pair_count = int(len(chosen_pairs))

    row_by_idx = {_safe_int(row.get("row_idx", -1), -1): row for row in row_states}

    def _pair_state_counts(pairs: Sequence[Sequence[int]]) -> tuple[int, int]:
        tracked_count = 0
        lost_count = 0
        for pair in pairs:
            if len(pair) < 2:
                continue
            row_idx = _safe_int(pair[0], -1)
            row = row_by_idx.get(row_idx)
            if row is None:
                continue
            state = str(row.get("state", ""))
            if state == "Tracked":
                tracked_count += 1
            elif state in {"Lost", "LongLost"}:
                lost_count += 1
        return int(tracked_count), int(lost_count)

    baseline_tracked_matches, baseline_lost_matches = _pair_state_counts(baseline_pairs)
    chosen_tracked_matches, chosen_lost_matches = _pair_state_counts(chosen_pairs)

    suppressed_rows = list(candidate_row.get("suppressed_rows", []) or [])
    suppressed_row_states = [row_by_idx.get(_safe_int(row_idx, -1), {}) for row_idx in suppressed_rows]
    suppressed_tracked_rows = sum(1 for row in suppressed_row_states if str(row.get("state", "")) == "Tracked")
    suppressed_lost_rows = sum(1 for row in suppressed_row_states if str(row.get("state", "")) in {"Lost", "LongLost"})
    suppressed_recent_owner_rows = sum(1 for row in suppressed_row_states if bool(row.get("is_recent_owner", False)))
    suppressed_gaps = [_safe_float(row.get("gap", 0.0), 0.0) for row in suppressed_row_states if row]

    features = [
        float(_list_count(candidate_row.get("rows", []))),
        float(_list_count(candidate_row.get("cols", []))),
        float(len(edge_meta)),
        float(_list_count(candidate_row.get("ambiguous_rows", []))),
        float(_list_count(candidate_row.get("ambiguous_cols", []))),
        float(_list_count(candidate_row.get("reclaimable_rows", []))),
        float(_list_count(candidate_row.get("recent_owner_rows", []))),
        float(_list_count(candidate_row.get("introduced_rows", []))),
        float(_list_count(candidate_row.get("suppressed_rows", []))),
        float(_list_count(candidate_row.get("protected_young_active_rows", []))),
        float(_list_count(candidate_row.get("suppressed_protected_rows", []))),
        float(_list_count(candidate_row.get("suppressed_stale_lost_rows", []))),
        float(_list_count(candidate_row.get("active_introduced_rows", []))),
        _bool01(candidate_row.get("row_involved_block", False)),
        _bool01(candidate_row.get("col_only_block", False)),
        _bool01(candidate_row.get("has_gap_advantaged_reclaim", False)),
        float(baseline_pair_count),
        float(chosen_pair_count),
        float(chosen_pair_count - baseline_pair_count),
        float(_safe_int(candidate_row.get("baseline_reclaim_matches", 0), 0)),
        float(_safe_int(candidate_row.get("chosen_reclaim_matches", 0), 0)),
        float(_safe_int(candidate_row.get("chosen_reclaim_matches", 0), 0) - _safe_int(candidate_row.get("baseline_reclaim_matches", 0), 0)),
        float(_safe_int(candidate_row.get("baseline_recent_matches", 0), 0)),
        float(_safe_int(candidate_row.get("chosen_recent_matches", 0), 0)),
        float(_safe_int(candidate_row.get("chosen_recent_matches", 0), 0) - _safe_int(candidate_row.get("baseline_recent_matches", 0), 0)),
        float(baseline_tracked_matches),
        float(chosen_tracked_matches),
        float(chosen_tracked_matches - baseline_tracked_matches),
        float(baseline_lost_matches),
        float(chosen_lost_matches),
        float(chosen_lost_matches - baseline_lost_matches),
        float(suppressed_tracked_rows),
        float(suppressed_lost_rows),
        float(suppressed_recent_owner_rows),
        _mean(suppressed_gaps),
        _max(suppressed_gaps),
        _safe_float(candidate_row.get("baseline_cost", 0.0), 0.0),
        _safe_float(candidate_row.get("chosen_cost", 0.0), 0.0),
        _safe_float(candidate_row.get("cost_delta", 0.0), 0.0),
        _safe_float(candidate_row.get("baseline_utility", 0.0), 0.0),
        _safe_float(candidate_row.get("chosen_utility", 0.0), 0.0),
        _safe_float(candidate_row.get("utility_gain", 0.0), 0.0),
        _safe_float(candidate_row.get("required_min_gain", 0.0), 0.0),
        _safe_float(candidate_row.get("required_max_cost_delta", 0.0), 0.0),
        _safe_float(candidate_row.get("utility_gain", 0.0), 0.0) - _safe_float(candidate_row.get("required_min_gain", 0.0), 0.0),
        _safe_float(candidate_row.get("required_max_cost_delta", 0.0), 0.0) - _safe_float(candidate_row.get("cost_delta", 0.0), 0.0),
        float(_safe_int(candidate_row.get("enumerated_assignments", 0), 0)),
        _safe_float(candidate_row.get("max_introduced_utility", 0.0), 0.0),
        _mean(row_gaps),
        _max(row_gaps),
        _mean(row_tracklet_lens),
        _max(row_tracklet_lens),
        _mean(row_scores),
        _min(row_scores),
        _max(row_scores),
        _mean(row_reclaim),
        _max(row_reclaim),
        _mean(row_recent_penalty),
        _max(row_recent_penalty),
        _fraction(tracked_rows, len(row_states)),
        _fraction(lost_rows, len(row_states)),
        _mean(edge_costs),
        _min(edge_costs),
        _max(edge_costs),
        _mean(edge_utilities),
        _min(edge_utilities),
        _max(edge_utilities),
        _mean(edge_ious),
        _max(edge_ious),
        _mean(edge_det_scores),
        _max(edge_det_scores),
        _pair_mean(baseline_pair_meta, "cost"),
        _pair_mean(chosen_pair_meta, "cost"),
        _pair_mean(baseline_pair_meta, "utility"),
        _pair_mean(chosen_pair_meta, "utility"),
        _bool01(candidate_row.get("learned_commit_available", False)),
        _safe_float(candidate_row.get("learned_commit_score_delta", 0.0), 0.0),
        _safe_float(candidate_row.get("learned_commit_baseline_score", 0.0), 0.0),
        _safe_float(candidate_row.get("learned_commit_chosen_score", 0.0), 0.0),
        _safe_float(candidate_row.get("learned_commit_pred_score", 0.0), 0.0),
        _bool01(candidate_row.get("learned_commit_chosen_better", False)),
        _bool01(candidate_row.get("learned_commit_score_margin_pass", False)),
    ]
    return [float(value) for value in features]


@dataclass
class GraphAssocGateScore:
    baseline_score: float
    chosen_score: float
    score_delta: float
    chosen_better: bool
    pred_score: float
    decision_score: float
    positive_probability: float
    neutral_probability: float
    model_type: str
    decision_mode: str
    positive_threshold: float
    neutral_threshold: float
    neutral_risk_weight: float
    pred_pairs_local: Dict[int, int]
    chosen_pairs_local: Dict[int, int]
    baseline_pairs_local: Dict[int, int]
    probability: float
    accept: bool
    threshold: float
    rules_passed: bool


def checkpoint_looks_like_graph_assoc_gate(checkpoint_path: str) -> bool:
    try:
        payload = torch.load(_resolve_checkpoint_path(checkpoint_path), map_location="cpu")
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False
    return "gate_threshold" in payload and "feature_names" in payload and "model_state" in payload


class GraphAssocGateScorer:
    def __init__(
        self,
        checkpoint_path: str,
        *,
        device: str = "cpu",
        decision_mode: str = "",
        threshold: Optional[float] = None,
        neutral_risk_weight: Optional[float] = None,
        positive_threshold: Optional[float] = None,
        neutral_threshold: Optional[float] = None,
    ) -> None:
        self.checkpoint_path = _resolve_checkpoint_path(checkpoint_path)
        requested_device = str(device or "cpu").strip() or "cpu"
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            requested_device = "cpu"
        self.device = torch.device(requested_device)

        payload = torch.load(self.checkpoint_path, map_location="cpu")
        self.feature_names = list(payload.get("feature_names", GRAPH_ASSOC_GATE_FEATURE_NAMES))
        self.model_type = str(payload.get("model_type", "single_head") or "single_head")
        self.decision_mode = _normalize_decision_mode(str(decision_mode or payload.get("decision_mode", "")), self.model_type)
        payload_threshold = float(payload.get("gate_threshold", payload.get("acceptance_gate_thresh", 0.5)))
        self.threshold = float(payload_threshold if threshold is None else threshold)
        payload_neutral_weight = float(payload.get("neutral_risk_weight", 1.0))
        self.neutral_risk_weight = float(payload_neutral_weight if neutral_risk_weight is None else neutral_risk_weight)
        self.positive_threshold = float(
            payload.get("positive_threshold", self.threshold) if positive_threshold is None else positive_threshold
        )
        self.neutral_threshold = float(payload.get("neutral_threshold", 0.5) if neutral_threshold is None else neutral_threshold)
        if self.model_type == "dual_head":
            self.model = GraphAssocDualHeadGate(
                input_dim=int(payload.get("input_dim", len(self.feature_names))),
                hidden_dim=int(payload.get("hidden_dim", 32)),
                dropout=float(payload.get("dropout", 0.0)),
                num_hidden_layers=int(payload.get("num_hidden_layers", 1)),
            )
        else:
            self.model = GraphAssocGate(
                input_dim=int(payload.get("input_dim", len(self.feature_names))),
                hidden_dim=int(payload.get("hidden_dim", 32)),
                dropout=float(payload.get("dropout", 0.0)),
                num_hidden_layers=int(payload.get("num_hidden_layers", 1)),
            )
        self.model.load_state_dict(payload["model_state"])
        self.model.to(self.device)
        self.model.eval()

    def score_candidate_row(self, candidate_row: Dict[str, Any]) -> GraphAssocGateScore:
        feature_map = dict(zip(GRAPH_ASSOC_GATE_FEATURE_NAMES, build_graph_assoc_gate_feature_vector(candidate_row)))
        features = [float(feature_map.get(name, 0.0)) for name in self.feature_names]
        with torch.inference_mode():
            feature_tensor = torch.tensor(features, dtype=torch.float32, device=self.device).view(1, -1)
            if self.model_type == "dual_head":
                gain_logit, neutral_logit = self.model(feature_tensor)
                positive_probability = float(torch.sigmoid(gain_logit.view(())).item())
                neutral_probability = float(torch.sigmoid(neutral_logit.view(())).item())
                if self.decision_mode == "dual_threshold":
                    positive_margin = float(positive_probability - self.positive_threshold)
                    neutral_margin = float(self.neutral_threshold - neutral_probability)
                    decision_score = float(min(positive_margin, neutral_margin))
                    accept_threshold = 0.0
                    baseline_score = 0.0
                elif self.decision_mode == "positive_minus_neutral":
                    decision_score = float(positive_probability - neutral_probability)
                    accept_threshold = float(self.threshold)
                    baseline_score = float(self.threshold)
                elif self.decision_mode == "positive_times_one_minus_neutral":
                    decision_score = float(positive_probability * max(0.0, 1.0 - neutral_probability))
                    accept_threshold = float(self.threshold)
                    baseline_score = float(self.threshold)
                elif self.decision_mode == "min_positive_vs_one_minus_neutral":
                    decision_score = float(min(positive_probability, max(0.0, 1.0 - neutral_probability)))
                    accept_threshold = float(self.threshold)
                    baseline_score = float(self.threshold)
                else:
                    decision_score = float(positive_probability - self.neutral_risk_weight * neutral_probability)
                    accept_threshold = float(self.threshold)
                    baseline_score = float(self.threshold)
            else:
                logit = self.model(feature_tensor).view(())
                positive_probability = float(torch.sigmoid(logit).item())
                neutral_probability = 0.0
                decision_score = float(positive_probability)
                accept_threshold = float(self.threshold)
                baseline_score = float(self.threshold)
        rules_passed = bool(infer_rules_passed_before_gate(candidate_row))
        score_delta = float(decision_score - float(accept_threshold))
        return GraphAssocGateScore(
            baseline_score=float(baseline_score),
            chosen_score=float(decision_score),
            score_delta=float(score_delta),
            chosen_better=bool(score_delta >= 0.0),
            pred_score=float(decision_score),
            decision_score=float(decision_score),
            positive_probability=float(positive_probability),
            neutral_probability=float(neutral_probability),
            model_type=str(self.model_type),
            decision_mode=str(self.decision_mode),
            positive_threshold=float(self.positive_threshold),
            neutral_threshold=float(self.neutral_threshold),
            neutral_risk_weight=float(self.neutral_risk_weight),
            pred_pairs_local={},
            chosen_pairs_local={},
            baseline_pairs_local={},
            probability=float(positive_probability),
            accept=bool(score_delta >= 0.0),
            threshold=float(accept_threshold),
            rules_passed=rules_passed,
        )
