from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


ACCEPTANCE_FEATURE_NAMES = [
    "base_row_margin",
    "refined_row_margin",
    "flip_gap",
    "refined_top1_prob",
    "base_top1_prob_under_refiner",
    "row_nomatch_prob",
    "row_candidate_count_norm",
    "row_best_base_similarity",
    "row_second_base_similarity",
    "row_margin_base_similarity",
    "base_candidate_base_similarity",
    "refined_candidate_base_similarity",
    "base_candidate_reid",
    "refined_candidate_reid",
    "base_candidate_fused_iou_similarity",
    "refined_candidate_fused_iou_similarity",
    "base_candidate_det_score",
    "refined_candidate_det_score",
    "base_candidate_rank_norm",
    "refined_candidate_rank_norm",
    "base_col_margin_base_similarity",
    "refined_col_margin_base_similarity",
    "base_col_multi_track_flag",
    "refined_col_multi_track_flag",
]


def _feature_map(names: Sequence[str], values: Sequence[float] | np.ndarray) -> Dict[str, float]:
    return {str(name): float(values[idx]) for idx, name in enumerate(names)}


def build_acceptance_feature_vector(
    *,
    edge_feature_names: Sequence[str],
    row_feature_names: Sequence[str],
    col_feature_names: Sequence[str],
    edge_features: np.ndarray,
    row_features: np.ndarray,
    col_features: np.ndarray,
    valid_mask: np.ndarray,
    probs: np.ndarray,
    row_nomatch_probs: np.ndarray,
    base_best: np.ndarray,
    refined_best: np.ndarray,
    base_row_margin: np.ndarray,
    refined_row_margin: np.ndarray,
    row_idx: int,
) -> List[float]:
    row_idx = int(row_idx)
    base_col = int(base_best[row_idx])
    refined_col = int(refined_best[row_idx])
    if row_idx < 0 or row_idx >= edge_features.shape[0] or base_col < 0 or refined_col < 0:
        return [0.0 for _ in ACCEPTANCE_FEATURE_NAMES]

    row_map = _feature_map(row_feature_names, row_features[row_idx]) if row_features.size else {}
    base_edge_map = _feature_map(edge_feature_names, edge_features[row_idx, base_col]) if edge_features.size else {}
    refined_edge_map = _feature_map(edge_feature_names, edge_features[row_idx, refined_col]) if edge_features.size else {}
    base_col_map = _feature_map(col_feature_names, col_features[base_col]) if col_features.size else {}
    refined_col_map = _feature_map(col_feature_names, col_features[refined_col]) if col_features.size else {}

    candidate_count = float(np.count_nonzero(valid_mask[row_idx])) / max(float(valid_mask.shape[1]), 1.0)
    row_candidate_count_norm = float(row_map.get("row_candidate_count_norm", min(candidate_count, 1.0)))

    refined_prob = float(probs[row_idx, refined_col])
    base_prob = float(probs[row_idx, base_col])
    flip_gap = float(refined_prob - base_prob)

    return [
        float(base_row_margin[row_idx]),
        float(refined_row_margin[row_idx]),
        flip_gap,
        refined_prob,
        base_prob,
        float(row_nomatch_probs[row_idx]),
        row_candidate_count_norm,
        float(row_map.get("row_best_base_similarity", 0.0)),
        float(row_map.get("row_second_base_similarity", 0.0)),
        float(row_map.get("row_margin_base_similarity", 0.0)),
        float(base_edge_map.get("base_similarity", 0.0)),
        float(refined_edge_map.get("base_similarity", 0.0)),
        float(base_edge_map.get("s_reid", 0.0)),
        float(refined_edge_map.get("s_reid", 0.0)),
        float(base_edge_map.get("fused_iou_similarity", 0.0)),
        float(refined_edge_map.get("fused_iou_similarity", 0.0)),
        float(base_edge_map.get("det_score", 0.0)),
        float(refined_edge_map.get("det_score", 0.0)),
        float(base_edge_map.get("candidate_rank_norm", 0.0)),
        float(refined_edge_map.get("candidate_rank_norm", 0.0)),
        float(base_col_map.get("col_margin_base_similarity", 0.0)),
        float(refined_col_map.get("col_margin_base_similarity", 0.0)),
        float(base_col_map.get("col_multi_track_flag", 0.0)),
        float(refined_col_map.get("col_multi_track_flag", 0.0)),
    ]
