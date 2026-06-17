from __future__ import annotations

import re
from typing import Dict, List, Sequence


EDGE_FEATURE_NAMES = [
    "s_reid",
    "s_low",
    "s_mid",
    "s_high",
    "base_similarity",
    "det_score",
    "track_age",
    "track_age_short_norm",
    "track_fresh_flag",
    "track_stale_flag",
    "raw_iou_similarity",
    "fused_iou_similarity",
    "track_width",
    "track_height",
    "det_width",
    "det_height",
    "track_aspect",
    "det_aspect",
    "center_dx",
    "center_dy",
    "area_ratio",
    "candidate_rank_norm",
]

DOMAIN_FEATURE_NAMES = [
    "det_is_dpm",
    "det_is_frcnn",
    "det_is_sdp",
    "seq_id_norm",
]

ROW_CONTEXT_FEATURE_NAMES = [
    "row_track_age_norm",
    "row_track_age_short_norm",
    "row_track_fresh_flag",
    "row_track_stale_flag",
    "row_candidate_count_norm",
    "row_best_base_similarity",
    "row_second_base_similarity",
    "row_margin_base_similarity",
    "row_mean_base_similarity",
    "row_best_reid",
    "row_best_fused_iou_similarity",
    "row_ambiguous_flag",
]

COL_CONTEXT_FEATURE_NAMES = [
    "col_det_score",
    "col_candidate_count_norm",
    "col_best_base_similarity",
    "col_second_base_similarity",
    "col_margin_base_similarity",
    "col_mean_base_similarity",
    "col_best_reid",
    "col_best_fused_iou_similarity",
    "col_best_track_age_short_norm",
    "col_best_track_fresh_flag",
    "col_best_track_stale_flag",
    "col_second_track_age_short_norm",
    "col_second_track_fresh_flag",
    "col_second_track_stale_flag",
    "col_multi_track_flag",
]


TRACK_AGE_LONG_CAP = 30.0
TRACK_AGE_SHORT_CAP = 3.0
TRACK_FRESH_MAX = 1.0
TRACK_STALE_MIN = 2.0


def parse_mot_sequence_name(seq_name: str) -> tuple[str, int, str]:
    raw_name = str(seq_name or "").strip()
    detector = ""
    seq_id = 0
    tokens = raw_name.split("-")
    if len(tokens) >= 3:
        detector = str(tokens[-1]).upper()
    if len(tokens) >= 2:
        match = re.search(r"(\d+)", str(tokens[1]))
        if match:
            seq_id = int(match.group(1))
    return raw_name, int(seq_id), detector


def build_domain_feature_vector(seq_name: str) -> List[float]:
    _, seq_id, detector = parse_mot_sequence_name(seq_name)
    return [
        1.0 if detector == "DPM" else 0.0,
        1.0 if detector == "FRCNN" else 0.0,
        1.0 if detector == "SDP" else 0.0,
        min(max(float(seq_id), 0.0), 20.0) / 20.0,
    ]


def domain_feature_map(seq_name: str) -> Dict[str, float]:
    values = build_domain_feature_vector(seq_name)
    return {name: float(value) for name, value in zip(DOMAIN_FEATURE_NAMES, values)}


def _track_age_long_norm(track_age: float) -> float:
    return min(max(float(track_age), 0.0), TRACK_AGE_LONG_CAP) / TRACK_AGE_LONG_CAP


def _track_age_short_norm(track_age: float) -> float:
    return min(max(float(track_age), 0.0), TRACK_AGE_SHORT_CAP) / TRACK_AGE_SHORT_CAP


def _track_fresh_flag(track_age: float) -> float:
    return 1.0 if float(track_age) <= TRACK_FRESH_MAX else 0.0


def _track_stale_flag(track_age: float) -> float:
    return 1.0 if float(track_age) >= TRACK_STALE_MIN else 0.0


def _track_age_feature_triplet(track_age: float) -> tuple[float, float, float]:
    return (
        _track_age_short_norm(track_age),
        _track_fresh_flag(track_age),
        _track_stale_flag(track_age),
    )


def _row_track_age(row: Dict[str, object] | None) -> float:
    if row is None:
        return 0.0
    return float(row.get("track_age", 0.0))


def build_edge_feature_vector(
    *,
    s_reid: float,
    s_low: float,
    s_mid: float,
    s_high: float,
    base_similarity: float,
    det_score: float,
    track_age: float,
    raw_iou_cost: float,
    fused_iou_cost: float,
    track_box: Sequence[float],
    det_box: Sequence[float],
    candidate_rank: float,
    top_k: float = 5.0,
) -> List[float]:
    tx1, ty1, tx2, ty2 = [float(v) for v in track_box]
    dx1, dy1, dx2, dy2 = [float(v) for v in det_box]
    track_w = max(tx2 - tx1, 1.0)
    track_h = max(ty2 - ty1, 1.0)
    det_w = max(dx2 - dx1, 1.0)
    det_h = max(dy2 - dy1, 1.0)
    track_cx = 0.5 * (tx1 + tx2)
    track_cy = 0.5 * (ty1 + ty2)
    det_cx = 0.5 * (dx1 + dx2)
    det_cy = 0.5 * (dy1 + dy2)
    norm_scale = max((track_w + det_w) * 0.5, (track_h + det_h) * 0.5, 1.0)
    center_dx = (det_cx - track_cx) / norm_scale
    center_dy = (det_cy - track_cy) / norm_scale
    track_aspect = track_w / track_h
    det_aspect = det_w / det_h
    area_ratio = (det_w * det_h) / max(track_w * track_h, 1.0)
    candidate_rank_norm = float(candidate_rank) / max(float(top_k), 1.0)
    track_age_long_norm = _track_age_long_norm(track_age)
    track_age_short_norm, track_fresh_flag, track_stale_flag = _track_age_feature_triplet(track_age)
    return [
        float(s_reid),
        float(s_low),
        float(s_mid),
        float(s_high),
        float(base_similarity),
        float(det_score),
        track_age_long_norm,
        track_age_short_norm,
        track_fresh_flag,
        track_stale_flag,
        1.0 - min(max(float(raw_iou_cost), 0.0), 1.0),
        1.0 - min(max(float(fused_iou_cost), 0.0), 1.0),
        min(track_w / 200.0, 2.0),
        min(track_h / 400.0, 2.0),
        min(det_w / 200.0, 2.0),
        min(det_h / 400.0, 2.0),
        min(track_aspect, 4.0) / 4.0,
        min(det_aspect, 4.0) / 4.0,
        max(min(center_dx, 2.0), -2.0) / 2.0,
        max(min(center_dy, 2.0), -2.0) / 2.0,
        min(area_ratio, 4.0) / 4.0,
        min(candidate_rank_norm, 1.0),
    ]


def feature_vector_from_row(row: Dict[str, object], *, top_k: float = 5.0) -> List[float]:
    return build_edge_feature_vector(
        s_reid=float(row.get("s_reid", 0.0)),
        s_low=float(row.get("s_low", 0.0)),
        s_mid=float(row.get("s_mid", 0.0)),
        s_high=float(row.get("s_high", 0.0)),
        base_similarity=float(row.get("base_similarity", 0.0)),
        det_score=float(row.get("det_score", 0.0)),
        track_age=float(row.get("track_age", 0.0)),
        raw_iou_cost=float(row.get("raw_iou_cost", 1.0)),
        fused_iou_cost=float(row.get("fused_iou_cost", 1.0)),
        track_box=row.get("track_box", [0.0, 0.0, 1.0, 1.0]),
        det_box=row.get("det_box", [0.0, 0.0, 1.0, 1.0]),
        candidate_rank=float(row.get("candidate_rank", 0.0)),
        top_k=float(top_k),
    )


def _top2(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    ordered = sorted((float(v) for v in values), reverse=True)
    if len(ordered) == 1:
        return ordered[0], 0.0
    return ordered[0], ordered[1]


def _top2_rows_by_base_similarity(rows: Sequence[Dict[str, object]]) -> tuple[Dict[str, object] | None, Dict[str, object] | None]:
    if not rows:
        return None, None
    ordered = sorted(rows, key=lambda row: float(row.get("base_similarity", 0.0)), reverse=True)
    if len(ordered) == 1:
        return ordered[0], None
    return ordered[0], ordered[1]


def build_row_context_from_rows(
    rows: Sequence[Dict[str, object]],
    *,
    candidate_limit: float,
) -> List[float]:
    if not rows:
        return [0.0 for _ in ROW_CONTEXT_FEATURE_NAMES]
    base_scores = [float(row.get("base_similarity", 0.0)) for row in rows]
    reid_scores = [float(row.get("s_reid", 0.0)) for row in rows]
    fused_iou_scores = [1.0 - min(max(float(row.get("fused_iou_cost", 1.0)), 0.0), 1.0) for row in rows]
    best_base, second_base = _top2(base_scores)
    candidate_count = float(len(rows)) / max(float(candidate_limit), 1.0)
    track_age = max(float(row.get("track_age", 0.0)) for row in rows)
    track_age_short_norm, track_fresh_flag, track_stale_flag = _track_age_feature_triplet(track_age)
    ambiguous_flag = 1.0 if any(bool(int(row.get("ambiguous_flag", 0))) for row in rows) else 0.0
    return [
        _track_age_long_norm(track_age),
        track_age_short_norm,
        track_fresh_flag,
        track_stale_flag,
        min(candidate_count, 1.0),
        best_base,
        second_base,
        max(best_base - second_base, 0.0),
        sum(base_scores) / max(len(base_scores), 1),
        max(reid_scores) if reid_scores else 0.0,
        max(fused_iou_scores) if fused_iou_scores else 0.0,
        ambiguous_flag,
    ]


def build_col_context_from_rows(
    rows: Sequence[Dict[str, object]],
    *,
    candidate_limit: float,
) -> List[float]:
    if not rows:
        return [0.0 for _ in COL_CONTEXT_FEATURE_NAMES]
    base_scores = [float(row.get("base_similarity", 0.0)) for row in rows]
    reid_scores = [float(row.get("s_reid", 0.0)) for row in rows]
    fused_iou_scores = [1.0 - min(max(float(row.get("fused_iou_cost", 1.0)), 0.0), 1.0) for row in rows]
    best_base, second_base = _top2(base_scores)
    det_score = max(float(row.get("det_score", 0.0)) for row in rows)
    candidate_count = float(len(rows)) / max(float(candidate_limit), 1.0)
    best_row, second_row = _top2_rows_by_base_similarity(rows)
    best_age_short_norm, best_fresh_flag, best_stale_flag = _track_age_feature_triplet(_row_track_age(best_row))
    second_age_short_norm, second_fresh_flag, second_stale_flag = _track_age_feature_triplet(_row_track_age(second_row))
    multi_track_flag = 1.0 if len(rows) > 1 else 0.0
    return [
        det_score,
        min(candidate_count, 1.0),
        best_base,
        second_base,
        max(best_base - second_base, 0.0),
        sum(base_scores) / max(len(base_scores), 1),
        max(reid_scores) if reid_scores else 0.0,
        max(fused_iou_scores) if fused_iou_scores else 0.0,
        best_age_short_norm,
        best_fresh_flag,
        best_stale_flag,
        second_age_short_norm,
        second_fresh_flag,
        second_stale_flag,
        multi_track_flag,
    ]
