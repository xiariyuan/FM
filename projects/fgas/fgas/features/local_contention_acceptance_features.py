from __future__ import annotations

from typing import Dict, List, Mapping


LOCAL_CONTENTION_ACCEPTANCE_FEATURE_NAMES = [
    "local_contention_ranker_score",
    "local_contention_ranker_margin_to_second",
    "local_contention_ranker_group_size",
    "owner_edge_score",
    "challenger_edge_score",
    "challenger_edge_advantage_vs_owner",
    "owner_best_box_iou",
    "challenger_best_box_iou",
    "owner_score_margin_to_best_other_track",
    "owner_best_alt_det_score",
    "owner_best_alt_det_box_iou",
    "challenger_best_alt_det_score",
    "challenger_best_alt_det_box_iou",
    "owner_hits",
    "owner_age",
    "owner_hit_streak",
    "owner_time_since_update",
    "owner_latest_observation_valid",
    "owner_is_weak_hits",
    "challenger_rank_in_unit",
    "challenger_hits",
    "challenger_age",
    "challenger_hit_streak",
    "challenger_time_since_update",
    "challenger_latest_observation_valid",
    "challenger_det_rank",
    "challenger_hit_gap_vs_owner",
    "challenger_age_gap_vs_owner",
    "owner_minus_target_time_since_update",
    "target_minus_owner_hit_streak",
]


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def build_local_contention_acceptance_feature_map(row: Mapping[str, object]) -> Dict[str, float]:
    owner_tsu = _safe_float(row.get("owner_time_since_update", 0.0))
    challenger_tsu = _safe_float(
        row.get("challenger_time_since_update", row.get("track_time_since_update", 0.0))
    )
    owner_hit_streak = _safe_float(row.get("owner_hit_streak", 0.0))
    challenger_hit_streak = _safe_float(
        row.get("challenger_hit_streak", row.get("track_hit_streak", 0.0))
    )
    return {
        "local_contention_ranker_score": _safe_float(row.get("local_contention_ranker_score", -1.0)),
        "local_contention_ranker_margin_to_second": _safe_float(
            row.get("local_contention_ranker_margin_to_second", -1.0)
        ),
        "local_contention_ranker_group_size": _safe_float(row.get("local_contention_ranker_group_size", 0.0)),
        "owner_edge_score": _safe_float(row.get("owner_edge_score", 0.0)),
        "challenger_edge_score": _safe_float(row.get("challenger_edge_score", row.get("edge_score", 0.0))),
        "challenger_edge_advantage_vs_owner": _safe_float(
            row.get("challenger_edge_advantage_vs_owner", row.get("edge_advantage_vs_owner", 0.0))
        ),
        "owner_best_box_iou": _safe_float(row.get("owner_best_box_iou", 0.0)),
        "challenger_best_box_iou": _safe_float(row.get("challenger_best_box_iou", row.get("best_box_iou", 0.0))),
        "owner_score_margin_to_best_other_track": _safe_float(
            row.get("owner_score_margin_to_best_other_track", 0.0)
        ),
        "owner_best_alt_det_score": _safe_float(row.get("owner_best_alt_det_score", -1.0)),
        "owner_best_alt_det_box_iou": _safe_float(row.get("owner_best_alt_det_box_iou", 0.0)),
        "challenger_best_alt_det_score": _safe_float(row.get("challenger_best_alt_det_score", -1.0)),
        "challenger_best_alt_det_box_iou": _safe_float(row.get("challenger_best_alt_det_box_iou", 0.0)),
        "owner_hits": _safe_float(row.get("owner_hits", 0.0)),
        "owner_age": _safe_float(row.get("owner_age", 0.0)),
        "owner_hit_streak": owner_hit_streak,
        "owner_time_since_update": owner_tsu,
        "owner_latest_observation_valid": _safe_float(row.get("owner_latest_observation_valid", 0.0)),
        "owner_is_weak_hits": _safe_float(row.get("owner_is_weak_hits", 0.0)),
        "challenger_rank_in_unit": _safe_float(row.get("challenger_rank_in_unit", 0.0)),
        "challenger_hits": _safe_float(row.get("challenger_hits", row.get("track_hits", 0.0))),
        "challenger_age": _safe_float(row.get("challenger_age", row.get("track_age", 0.0))),
        "challenger_hit_streak": challenger_hit_streak,
        "challenger_time_since_update": challenger_tsu,
        "challenger_latest_observation_valid": _safe_float(
            row.get("challenger_latest_observation_valid", row.get("track_latest_observation_valid", 0.0))
        ),
        "challenger_det_rank": _safe_float(row.get("challenger_det_rank", row.get("det_rank", 0.0))),
        "challenger_hit_gap_vs_owner": _safe_float(
            row.get("challenger_hit_gap_vs_owner", row.get("hit_gap_vs_owner", 0.0))
        ),
        "challenger_age_gap_vs_owner": _safe_float(
            row.get("challenger_age_gap_vs_owner", row.get("age_gap", 0.0))
        ),
        "owner_minus_target_time_since_update": float(owner_tsu - challenger_tsu),
        "target_minus_owner_hit_streak": float(challenger_hit_streak - owner_hit_streak),
    }


def build_local_contention_acceptance_feature_vector(row: Mapping[str, object]) -> List[float]:
    feature_map = build_local_contention_acceptance_feature_map(row)
    return [float(feature_map.get(name, 0.0)) for name in LOCAL_CONTENTION_ACCEPTANCE_FEATURE_NAMES]
