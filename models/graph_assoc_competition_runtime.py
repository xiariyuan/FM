from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch

from models.competition_assoc import CANDIDATE_FEATURES, OBSERVED_GROUP_FEATURES, CompetitionAssociationController
from models.graph_assoc_gate_runtime import infer_rules_passed_before_gate


COMPETITION_ASSOC_FEATURE_VERSION = "competition_assoc_v1"


def _resolve_checkpoint_path(checkpoint_path: str) -> str:
    requested_path = str(checkpoint_path)
    if Path(requested_path).is_absolute():
        return requested_path
    repo_root = Path(__file__).resolve().parents[1]
    return str((repo_root / requested_path).resolve())


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


def _mean(values: Iterable[float]) -> float:
    seq = [float(v) for v in values]
    if not seq:
        return 0.0
    return float(sum(seq) / float(len(seq)))


def _min(values: Iterable[float]) -> float:
    seq = [float(v) for v in values]
    if not seq:
        return 0.0
    return float(min(seq))


def _max(values: Iterable[float]) -> float:
    seq = [float(v) for v in values]
    if not seq:
        return 0.0
    return float(max(seq))


def _entropy(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    tensor = torch.tensor([float(v) for v in values], dtype=torch.float32)
    tensor = tensor - tensor.max()
    probs = torch.softmax(tensor, dim=0)
    return float((-(probs * torch.log(probs.clamp(min=1e-8))).sum()).item())


def _pair_local_map(
    pairs: Sequence[Sequence[int]],
    *,
    row_to_local: Dict[int, int],
    col_to_local: Dict[int, int],
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


def _infer_rank_frac(edge_meta_rows: Sequence[Dict[str, Any]]) -> Dict[tuple[int, int], float]:
    by_col: Dict[int, List[Dict[str, Any]]] = {}
    for row in edge_meta_rows:
        col_idx = _safe_int(row.get("col_idx", -1), -1)
        if col_idx < 0:
            continue
        by_col.setdefault(int(col_idx), []).append(dict(row))

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


@dataclass
class GraphAssocCompetitionScore:
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
    gain_pred: float = 0.0
    policy_score: float = 0.0
    action_margin: float = 0.0
    keep_probability: float = 0.0
    rerank_probability: float = 0.0
    null_probability: float = 0.0
    continuity_probability: float = 0.0


def _normalize_decision_mode(mode: str) -> str:
    requested = str(mode or "").strip().lower()
    if not requested:
        return "positive_minus_weighted_neutral"
    aliases = {
        "positive_probability": "positive_probability",
        "positive_minus_weighted_neutral": "positive_minus_weighted_neutral",
        "positive_minus_neutral": "positive_minus_neutral",
        "gain_minus_neutral": "positive_minus_neutral",
        "positive_times_one_minus_neutral": "positive_times_one_minus_neutral",
        "gain_times_one_minus_neutral": "positive_times_one_minus_neutral",
        "min_positive_vs_one_minus_neutral": "min_positive_vs_one_minus_neutral",
        "dual_threshold": "dual_threshold",
        "positive_and_not_neutral": "dual_threshold",
    }
    return aliases.get(requested, requested)


def _build_group_feature_tensor(candidate_row: Dict[str, Any]) -> torch.Tensor:
    row_states = list(candidate_row.get("row_states", []) or [])
    col_states = list(candidate_row.get("col_states", []) or [])
    edge_meta = list(candidate_row.get("edge_meta", []) or [])

    row_gaps = [_safe_float(row.get("gap", 0.0), 0.0) for row in row_states]
    row_tracklet_lens = [_safe_float(row.get("tracklet_len", 0.0), 0.0) for row in row_states]
    edge_utilities = [_safe_float(edge.get("utility", 0.0), 0.0) for edge in edge_meta]
    utilities_sorted = sorted(edge_utilities, reverse=True)
    col_scores = [_safe_float(col.get("score", 0.0), 0.0) for col in col_states]
    rank_margin = 0.0
    if len(utilities_sorted) >= 2:
        rank_margin = float(utilities_sorted[0] - utilities_sorted[1])
    elif len(utilities_sorted) == 1:
        rank_margin = float(utilities_sorted[0])

    row_involved_block = bool(candidate_row.get("row_involved_block", False))
    col_only_block = bool(candidate_row.get("col_only_block", False))
    ambiguous_rows = list(candidate_row.get("ambiguous_rows", []) or [])
    ambiguous_cols = list(candidate_row.get("ambiguous_cols", []) or [])
    reclaimable_rows = list(candidate_row.get("reclaimable_rows", []) or [])
    introduced_rows = list(candidate_row.get("introduced_rows", []) or [])

    group_is_ambiguous = int(bool(row_involved_block or col_only_block or ambiguous_rows or ambiguous_cols))
    group_is_recoverable = int(bool(reclaimable_rows))
    group_is_background = int(not bool(reclaimable_rows) and not bool(introduced_rows) and max(edge_utilities or [0.0]) <= 0.0)
    positive_in_topk = int(max(edge_utilities or [0.0]) > 0.0)

    feature_values = [
        float(len(row_states)),
        float(len(edge_meta)),
        float(rank_margin),
        float(_entropy(edge_utilities)),
        float(positive_in_topk),
        float(group_is_ambiguous),
        float(group_is_recoverable),
        float(group_is_background),
        float(min(row_gaps) if row_gaps else 0.0),
        float(_mean(row_gaps)),
        float(_mean(row_tracklet_lens)),
        float(_mean(col_scores)),
        float(_safe_int(candidate_row.get("future_visible_count", 0), 0)),
        float(_safe_int(candidate_row.get("next_same_gt_gap", -1), -1)),
    ]
    return torch.tensor(feature_values, dtype=torch.float32).view(-1)


def _build_candidate_feature_tensor(candidate_row: Dict[str, Any]) -> torch.Tensor:
    row_states = list(candidate_row.get("row_states", []) or [])
    edge_meta_rows = list(candidate_row.get("edge_meta", []) or [])
    if not edge_meta_rows:
        return torch.zeros((0, 6), dtype=torch.float32)

    row_by_idx = {_safe_int(row.get("row_idx", -1), -1): row for row in row_states}
    rank_frac = _infer_rank_frac(edge_meta_rows)

    features: List[List[float]] = []
    for edge in edge_meta_rows:
        row_idx = _safe_int(edge.get("row_idx", -1), -1)
        col_idx = _safe_int(edge.get("col_idx", -1), -1)
        if row_idx not in row_by_idx or col_idx < 0:
            continue
        row_state = row_by_idx[row_idx]
        cost = _safe_float(edge.get("cost", 0.0), 0.0)
        utility = _safe_float(edge.get("utility", 0.0), 0.0)
        box_iou = _safe_float(edge.get("box_iou", 0.0), 0.0)
        gap = _safe_float(row_state.get("gap", 0.0), 0.0)
        tracklet_len = _safe_float(row_state.get("tracklet_len", 0.0), 0.0)
        features.append(
            [
                float(1.0 - cost),
                float(utility),
                float(box_iou),
                float(gap),
                float(tracklet_len),
                float(rank_frac.get((row_idx, col_idx), 1.0)),
            ]
        )
    if not features:
        return torch.zeros((0, 6), dtype=torch.float32)
    return torch.tensor(features, dtype=torch.float32)


class GraphAssocCompetitionScorer:
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
        self.checkpoint_path = _resolve_checkpoint_path(checkpoint_path)
        requested_device = str(device or "cpu").strip() or "cpu"
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            requested_device = "cpu"
        self.device = torch.device(requested_device)

        payload = torch.load(self.checkpoint_path, map_location="cpu")
        config = dict(payload.get("config", {}))
        self.model_family = str(payload.get("model_family", "competition_assoc") or "competition_assoc")
        self.model_type = str(payload.get("model_type", "competition_controller") or "competition_controller")
        self.feature_version = str(payload.get("feature_version", COMPETITION_ASSOC_FEATURE_VERSION) or COMPETITION_ASSOC_FEATURE_VERSION)
        self.decision_mode = _normalize_decision_mode(str(decision_mode or payload.get("decision_mode", "")))
        self.threshold = float(payload.get("threshold", 0.0) if threshold is None else threshold)
        self.neutral_risk_weight = float(payload.get("neutral_risk_weight", 1.0) if neutral_risk_weight is None else neutral_risk_weight)
        self.positive_threshold = float(payload.get("positive_threshold", self.threshold) if positive_threshold is None else positive_threshold)
        self.neutral_threshold = float(payload.get("neutral_threshold", 0.5) if neutral_threshold is None else neutral_threshold)

        group_dim = int(config.get("group_dim", len(OBSERVED_GROUP_FEATURES)))
        candidate_dim = int(config.get("candidate_dim", len(CANDIDATE_FEATURES)))
        hidden_dim = int(config.get("hidden_dim", 128))
        dropout = float(config.get("dropout", 0.1))
        num_heads = int(config.get("num_heads", 4))
        self.model = CompetitionAssociationController(
            group_dim=group_dim,
            candidate_dim=candidate_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        state_dict = payload.get("model", payload)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()

    def _score_from_actions(
        self,
        *,
        keep_prob: float,
        rerank_prob: float,
        null_prob: float,
        continuity_prob: float,
    ) -> float:
        if self.decision_mode == "dual_threshold":
            positive_margin = float(rerank_prob - self.positive_threshold)
            neutral_margin = float(self.neutral_threshold - null_prob)
            return float(min(positive_margin, neutral_margin))
        if self.decision_mode == "positive_minus_neutral":
            return float(rerank_prob - null_prob)
        if self.decision_mode == "positive_times_one_minus_neutral":
            return float(rerank_prob * max(0.0, 1.0 - null_prob))
        if self.decision_mode == "min_positive_vs_one_minus_neutral":
            return float(min(rerank_prob, max(0.0, 1.0 - null_prob)))
        return float(rerank_prob - self.neutral_risk_weight * null_prob)

    def score_candidate_row(self, candidate_row: Dict[str, Any]) -> GraphAssocCompetitionScore:
        group_features = _build_group_feature_tensor(candidate_row).to(device=self.device, dtype=torch.float32).view(1, -1)
        candidate_features = _build_candidate_feature_tensor(candidate_row).to(device=self.device, dtype=torch.float32)
        valid_mask = torch.ones((1, int(candidate_features.shape[0])), dtype=torch.bool, device=self.device)
        if candidate_features.shape[0] == 0:
            candidate_features = torch.zeros((1, 0, len(CANDIDATE_FEATURES)), dtype=torch.float32, device=self.device)
        else:
            candidate_features = candidate_features.unsqueeze(0)

        with torch.inference_mode():
            outputs = self.model(
                group_features=group_features,
                candidate_features=candidate_features,
                valid_mask=valid_mask,
            )

        action_logits = outputs["action_logits"].view(-1)
        action_prob = outputs["action_prob"].view(-1)
        candidate_logits = outputs["candidate_logits"].view(-1)
        continuity_prob = float(outputs["continuity_prob"].view(()).item())
        keep_prob = float(action_prob[0].item()) if action_prob.numel() > 0 else 0.0
        rerank_prob = float(action_prob[1].item()) if action_prob.numel() > 1 else 0.0
        null_prob = float(action_prob[2].item()) if action_prob.numel() > 2 else 0.0
        decision_score = self._score_from_actions(
            keep_prob=keep_prob,
            rerank_prob=rerank_prob,
            null_prob=null_prob,
            continuity_prob=continuity_prob,
        )
        threshold = float(self.threshold if self.decision_mode != "dual_threshold" else 0.0)
        score_delta = float(decision_score - threshold)
        baseline_pairs = list(candidate_row.get("baseline_pairs", []) or [])
        chosen_pairs = list(candidate_row.get("chosen_pairs", []) or [])
        if score_delta >= 0.0:
            pred_pairs = chosen_pairs
        else:
            pred_pairs = baseline_pairs

        row_states = list(candidate_row.get("row_states", []) or [])
        col_states = list(candidate_row.get("col_states", []) or [])
        row_to_local = {_safe_int(row.get("row_idx", -1), -1): idx for idx, row in enumerate(row_states)}
        col_to_local = {_safe_int(col.get("col_idx", -1), -1): idx for idx, col in enumerate(col_states)}
        baseline_local = _pair_local_map(baseline_pairs, row_to_local=row_to_local, col_to_local=col_to_local)
        chosen_local = _pair_local_map(chosen_pairs, row_to_local=row_to_local, col_to_local=col_to_local)
        pred_local = _pair_local_map(pred_pairs, row_to_local=row_to_local, col_to_local=col_to_local)

        rules_passed = bool(infer_rules_passed_before_gate(candidate_row))
        return GraphAssocCompetitionScore(
            baseline_score=float(keep_prob),
            chosen_score=float(rerank_prob),
            score_delta=float(score_delta),
            chosen_better=bool(rerank_prob >= keep_prob),
            pred_score=float(decision_score),
            decision_score=float(decision_score),
            positive_probability=float(rerank_prob),
            neutral_probability=float(null_prob),
            model_type=str(self.model_type),
            decision_mode=str(self.decision_mode),
            positive_threshold=float(self.positive_threshold),
            neutral_threshold=float(self.neutral_threshold),
            neutral_risk_weight=float(self.neutral_risk_weight),
            pred_pairs_local=pred_local,
            chosen_pairs_local=chosen_local,
            baseline_pairs_local=baseline_local,
            probability=float(rerank_prob),
            accept=bool(score_delta >= 0.0),
            threshold=float(threshold),
            rules_passed=bool(rules_passed),
            gain_pred=float(rerank_prob - keep_prob),
            policy_score=float(decision_score),
            action_margin=float(rerank_prob - keep_prob),
            keep_probability=float(keep_prob),
            rerank_probability=float(rerank_prob),
            null_probability=float(null_prob),
            continuity_probability=float(continuity_prob),
        )


def checkpoint_looks_like_graph_assoc_competition(checkpoint_path: str) -> bool:
    try:
        payload = torch.load(_resolve_checkpoint_path(checkpoint_path), map_location="cpu")
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False
    config = payload.get("config", {})
    return "model" in payload and isinstance(config, dict) and "group_dim" in config and "candidate_dim" in config
