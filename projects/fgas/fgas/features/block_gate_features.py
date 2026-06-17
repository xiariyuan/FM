from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from projects.fgas.fgas.features.edge_features import build_domain_feature_vector


BLOCK_GATE_FEATURE_NAMES = [
    "block_rows_norm",
    "block_cols_norm",
    "block_edges_norm",
    "block_ambiguous_flag",
    "changed_row_rate",
    "changed_row_count_norm",
    "mean_base_row_margin",
    "min_base_row_margin",
    "mean_refined_row_margin",
    "min_refined_row_margin",
    "mean_flip_gap_changed",
    "max_flip_gap_changed",
    "mean_refined_win_prob",
    "mean_base_prob_under_refiner",
    "mean_row_candidate_count_norm",
    "mean_row_best_base_similarity",
    "mean_row_margin_base_similarity",
    "det_is_dpm",
    "det_is_frcnn",
    "det_is_sdp",
    "seq_id_norm",
]


def _row_feature_map(row_feature_names: Sequence[str], row_features: np.ndarray, row_idx: int) -> Dict[str, float]:
    if row_features.size == 0 or row_idx < 0 or row_idx >= row_features.shape[0]:
        return {}
    return {
        str(name): float(row_features[row_idx, idx])
        for idx, name in enumerate(row_feature_names)
    }


def build_block_gate_feature_vector(
    *,
    row_feature_names: Sequence[str],
    row_features: np.ndarray,
    valid_mask: np.ndarray,
    base_similarity: np.ndarray,
    probs: np.ndarray,
    base_best: np.ndarray,
    refined_best: np.ndarray,
    base_row_margin: np.ndarray,
    refined_row_margin: np.ndarray,
    seq_name: str,
    block_ambiguous_flag: float,
) -> List[float]:
    row_count = int(valid_mask.shape[0]) if valid_mask.ndim == 2 else 0
    col_count = int(valid_mask.shape[1]) if valid_mask.ndim == 2 else 0
    edge_count = int(np.count_nonzero(valid_mask))
    changed_mask = (base_best >= 0) & (refined_best >= 0) & (base_best != refined_best)
    changed_local_rows = np.where(changed_mask)[0].tolist()
    flip_gaps: List[float] = []
    refined_win_probs: List[float] = []
    base_probs_under_refiner: List[float] = []
    row_candidate_counts: List[float] = []
    row_best_base_sims: List[float] = []
    row_margin_base_sims: List[float] = []
    for row_idx in range(row_count):
        row_map = _row_feature_map(row_feature_names, row_features, row_idx)
        row_candidate_counts.append(float(row_map.get("row_candidate_count_norm", 0.0)))
        row_best_base_sims.append(float(row_map.get("row_best_base_similarity", 0.0)))
        row_margin_base_sims.append(float(row_map.get("row_margin_base_similarity", 0.0)))
    for local_r in changed_local_rows:
        base_local_c = int(base_best[local_r])
        refined_local_c = int(refined_best[local_r])
        if base_local_c < 0 or refined_local_c < 0:
            continue
        refined_prob = float(probs[local_r, refined_local_c])
        base_prob = float(probs[local_r, base_local_c])
        flip_gaps.append(refined_prob - base_prob)
        refined_win_probs.append(refined_prob)
        base_probs_under_refiner.append(base_prob)
    if not refined_win_probs:
        for local_r in range(row_count):
            refined_local_c = int(refined_best[local_r])
            base_local_c = int(base_best[local_r])
            if refined_local_c >= 0:
                refined_win_probs.append(float(probs[local_r, refined_local_c]))
            if base_local_c >= 0:
                base_probs_under_refiner.append(float(probs[local_r, base_local_c]))
    domain_features = build_domain_feature_vector(seq_name)
    return [
        min(float(row_count), 6.0) / 6.0,
        min(float(col_count), 6.0) / 6.0,
        min(float(edge_count), 16.0) / 16.0,
        float(block_ambiguous_flag),
        float(np.mean(changed_mask.astype(np.float32))) if row_count > 0 else 0.0,
        min(float(np.count_nonzero(changed_mask)), 6.0) / 6.0,
        float(np.mean(base_row_margin)) if row_count > 0 else 0.0,
        float(np.min(base_row_margin)) if row_count > 0 else 0.0,
        float(np.mean(refined_row_margin)) if row_count > 0 else 0.0,
        float(np.min(refined_row_margin)) if row_count > 0 else 0.0,
        float(np.mean(flip_gaps)) if flip_gaps else 0.0,
        float(np.max(flip_gaps)) if flip_gaps else 0.0,
        float(np.mean(refined_win_probs)) if refined_win_probs else 0.0,
        float(np.mean(base_probs_under_refiner)) if base_probs_under_refiner else 0.0,
        float(np.mean(row_candidate_counts)) if row_candidate_counts else 0.0,
        float(np.mean(row_best_base_sims)) if row_best_base_sims else 0.0,
        float(np.mean(row_margin_base_sims)) if row_margin_base_sims else 0.0,
        float(domain_features[0]),
        float(domain_features[1]),
        float(domain_features[2]),
        float(domain_features[3]),
    ]
