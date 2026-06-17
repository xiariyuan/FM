from __future__ import annotations

from typing import Dict, List, Mapping, Sequence


MATCHER_ACCEPTANCE_FEATURE_NAMES = [
    "base_row_margin",
    "refined_row_margin",
    "flip_gap",
    "refined_top1_prob",
    "base_top1_prob_under_refiner",
    "row_nomatch_prob",
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
    "base_col_multi_track_flag",
    "refined_col_multi_track_flag",
    "matcher_assignment_margin_norm",
    "component_row_count_norm",
    "component_col_count_norm",
    "changed_row_count_norm",
    "target_track_time_since_update_norm",
    "base_best_det_raw_owner_track_time_since_update_norm",
    "target_track_hit_streak_norm",
    "base_best_det_raw_owner_track_hit_streak_norm",
    "target_track_fresh_flag",
    "base_best_det_raw_owner_track_fresh_flag",
    "target_track_stale_flag",
    "base_best_det_raw_owner_track_stale_flag",
    "owner_minus_target_time_since_update_norm",
    "target_minus_owner_hit_streak_norm",
    "base_best_det_raw_owned_by_other_track",
    "base_best_det_raw_owner_local_top1_is_base_best_det",
    "base_best_det_raw_owner_local_top1_margin_norm",
    "base_best_det_raw_owner_base_best_vs_target_raw_score_margin_norm",
]


TSU_SHORT_CAP = 3.0
HIT_STREAK_CAP = 8.0
COUNT_CAP = 3.0
MARGIN_CAP = 1.0
ASSIGNMENT_MARGIN_CAP = 1.0


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _feature_map(names: Sequence[object], values: Sequence[object]) -> Dict[str, float]:
    return {
        str(name): _safe_float(values[idx], 0.0)
        for idx, name in enumerate(names)
        if idx < len(values)
    }


def _clipped_norm(value: object, cap: float) -> float:
    cap = max(float(cap), 1e-6)
    return min(max(_safe_float(value, 0.0), 0.0), cap) / cap


def _symmetric_norm(value: object, cap: float) -> float:
    cap = max(float(cap), 1e-6)
    clipped = min(max(_safe_float(value, 0.0), -cap), cap)
    return clipped / cap


def _fresh_flag(time_since_update: object) -> float:
    return 1.0 if _safe_float(time_since_update, 999.0) <= 1.0 else 0.0


def _stale_flag(time_since_update: object) -> float:
    return 1.0 if _safe_float(time_since_update, 0.0) >= 2.0 else 0.0


def build_matcher_acceptance_feature_map(row: Mapping[str, object]) -> Dict[str, float]:
    base_feature_names = list(row.get("feature_names", []) or [])
    base_features = list(row.get("features", []) or [])
    feature_map = _feature_map(base_feature_names, base_features)

    matcher_assignment_margin = _safe_float(row.get("matcher_assignment_margin", 0.0))
    component_row_count = _safe_float(row.get("component_row_count", 0.0))
    component_col_count = _safe_float(row.get("component_col_count", 0.0))
    changed_row_count = _safe_float(row.get("changed_row_count", 0.0))

    target_tsu = _safe_float(row.get("target_track_time_since_update", 0.0))
    owner_tsu = _safe_float(row.get("base_best_det_raw_owner_track_time_since_update", 0.0))
    target_hit_streak = _safe_float(row.get("target_track_hit_streak", 0.0))
    owner_hit_streak = _safe_float(row.get("base_best_det_raw_owner_track_hit_streak", 0.0))
    owner_local_top1_margin = _safe_float(row.get("base_best_det_raw_owner_local_top1_margin", 0.0))
    owner_base_best_vs_target_raw_margin = _safe_float(
        row.get("base_best_det_raw_owner_base_best_vs_target_raw_score_margin", 0.0)
    )

    feature_map.update(
        {
            "base_row_margin": _safe_float(row.get("base_row_margin", feature_map.get("base_row_margin", 0.0))),
            "refined_row_margin": _safe_float(row.get("refined_row_margin", feature_map.get("refined_row_margin", 0.0))),
            "flip_gap": _safe_float(row.get("flip_gap", feature_map.get("flip_gap", 0.0))),
            "refined_top1_prob": _safe_float(row.get("refined_choice_prob", feature_map.get("refined_top1_prob", 0.0))),
            "base_top1_prob_under_refiner": _safe_float(
                row.get("base_prob_under_matcher", feature_map.get("base_top1_prob_under_refiner", 0.0))
            ),
            "row_nomatch_prob": _safe_float(row.get("row_nomatch_prob", feature_map.get("row_nomatch_prob", 0.0))),
            "matcher_assignment_margin_norm": _clipped_norm(matcher_assignment_margin, ASSIGNMENT_MARGIN_CAP),
            "component_row_count_norm": _clipped_norm(component_row_count, COUNT_CAP),
            "component_col_count_norm": _clipped_norm(component_col_count, COUNT_CAP),
            "changed_row_count_norm": _clipped_norm(changed_row_count, COUNT_CAP),
            "target_track_time_since_update_norm": _clipped_norm(target_tsu, TSU_SHORT_CAP),
            "base_best_det_raw_owner_track_time_since_update_norm": _clipped_norm(owner_tsu, TSU_SHORT_CAP),
            "target_track_hit_streak_norm": _clipped_norm(target_hit_streak, HIT_STREAK_CAP),
            "base_best_det_raw_owner_track_hit_streak_norm": _clipped_norm(owner_hit_streak, HIT_STREAK_CAP),
            "target_track_fresh_flag": _fresh_flag(target_tsu),
            "base_best_det_raw_owner_track_fresh_flag": _fresh_flag(owner_tsu),
            "target_track_stale_flag": _stale_flag(target_tsu),
            "base_best_det_raw_owner_track_stale_flag": _stale_flag(owner_tsu),
            "owner_minus_target_time_since_update_norm": _symmetric_norm(owner_tsu - target_tsu, TSU_SHORT_CAP),
            "target_minus_owner_hit_streak_norm": _symmetric_norm(target_hit_streak - owner_hit_streak, HIT_STREAK_CAP),
            "base_best_det_raw_owned_by_other_track": _safe_float(
                row.get("base_best_det_raw_owned_by_other_track", 0.0)
            ),
            "base_best_det_raw_owner_local_top1_is_base_best_det": _safe_float(
                row.get("base_best_det_raw_owner_local_top1_is_base_best_det", 0.0)
            ),
            "base_best_det_raw_owner_local_top1_margin_norm": _clipped_norm(owner_local_top1_margin, MARGIN_CAP),
            "base_best_det_raw_owner_base_best_vs_target_raw_score_margin_norm": _clipped_norm(
                owner_base_best_vs_target_raw_margin, MARGIN_CAP
            ),
        }
    )
    return feature_map


def build_matcher_acceptance_feature_vector(row: Mapping[str, object]) -> List[float]:
    feature_map = build_matcher_acceptance_feature_map(row)
    return [float(feature_map.get(name, 0.0)) for name in MATCHER_ACCEPTANCE_FEATURE_NAMES]
