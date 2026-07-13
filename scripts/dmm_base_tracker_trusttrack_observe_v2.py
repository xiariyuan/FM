#!/usr/bin/env python3
"""Association-aligned observe-only wrapper for the recovered DMM tracker.

The wrapper does not change tracker decisions. It replaces only the association
debug callback and streams actual Hungarian matches plus optional top-K
candidates to CSV. Every chosen pair receives chosen-specific signed row/column
margins computed from the real final cost matrix after GMC/Kalman prediction and
all baseline masks/fusion.

No GT is read here. Ground truth must be joined offline after tracking.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

import dmm_base_tracker_shadow_pda as spda
import dmm_base_tracker_shadow_pda_v3 as v3

_ORIGINAL_RECORD_ASSOC_DEBUG = spda.base.DMMBaseTracker._record_assoc_debug
_WRITER: csv.DictWriter | None = None
_HANDLE = None
_OBSERVE_CONFIG = None
_STATS: dict = {}
_CHOSEN_KEYS: set[Tuple[int, str, int, int]] = set()

_FIELDS = [
    "frame",
    "stage",
    "track_row",
    "det_col",
    "chosen",
    "chosen_rank",
    "logged_reason",
    "assignment_threshold",
    "final_valid",
    "track_id",
    "det_global_idx",
    "track_state",
    "track_is_activated",
    "track_start_frame",
    "track_last_frame",
    "track_age",
    "tracklet_len",
    "lost_age",
    "track_alpha",
    "feature_history_len",
    "det_score",
    "soft_map_hit",
    "soft_planned_alpha",
    "soft_planned_reason",
    "soft_approx_trust",
    "soft_approx_collapse",
    "soft_approx_app_pair_margin",
    "soft_approx_motion_pair_margin",
    "soft_approx_iou_pair_margin",
    "soft_approx_shape_pair_margin",
    "final_cost",
    "raw_iou_cost",
    "iou_similarity",
    "embedding_cost",
    "embedding_available",
    "smooth_current_cosine",
    "row_alt_cost_all",
    "col_alt_cost_all",
    "row_signed_margin_all",
    "col_signed_margin_all",
    "pair_signed_margin_all",
    "row_alt_cost_valid",
    "col_alt_cost_valid",
    "row_signed_margin_valid",
    "col_signed_margin_valid",
    "pair_signed_margin_valid",
    "row_valid_competitors",
    "col_valid_competitors",
    "row_candidate_count",
    "col_candidate_count",
    "track_x1",
    "track_y1",
    "track_x2",
    "track_y2",
    "det_x1",
    "det_y1",
    "det_x2",
    "det_y2",
]


def _reset_stats() -> None:
    global _STATS, _CHOSEN_KEYS
    _STATS = {
        "assoc_calls": 0,
        "matrix_pairs": 0,
        "chosen_pairs_total": 0,
        "chosen_pairs_logged": 0,
        "logged_rows": 0,
        "chosen_rank_gt_top_k": 0,
        "invalid_match_indices": 0,
        "duplicate_chosen_keys": 0,
        "stage_calls": Counter(),
        "stage_chosen_pairs": Counter(),
        "stage_logged_rows": Counter(),
        "chosen_rank_histogram": Counter(),
    }
    _CHOSEN_KEYS = set()


def _float_or_nan(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _unit_cosine(a, b) -> float:
    if a is None or b is None:
        return math.nan
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    if aa.size == 0 or bb.size == 0 or aa.shape != bb.shape:
        return math.nan
    na = float(np.linalg.norm(aa))
    nb = float(np.linalg.norm(bb))
    if na < 1e-12 or nb < 1e-12:
        return math.nan
    return float(np.dot(aa / na, bb / nb))


def _state_name(state) -> str:
    mapping = {}
    for name in ("New", "Tracked", "Lost", "Removed"):
        if hasattr(spda.base.TrackState, name):
            mapping[getattr(spda.base.TrackState, name)] = name.lower()
    return mapping.get(state, str(state))


def _stage_threshold(tracker, stage: str) -> float:
    stage = str(stage)
    if stage.startswith("primary"):
        return float(tracker.cfg.match_thresh)
    if stage.startswith("secondary"):
        return float(tracker.cfg.second_match_thresh)
    if stage.startswith("unconfirmed"):
        return float(tracker.cfg.unconfirmed_match_thresh)
    return math.nan


def _alternative_cost(
    values: np.ndarray,
    chosen_index: int,
    *,
    threshold: float | None = None,
) -> Tuple[float, int]:
    if values.size <= 1:
        return math.nan, 0
    mask = np.ones(values.shape[0], dtype=bool)
    mask[int(chosen_index)] = False
    mask &= np.isfinite(values)
    if threshold is not None and math.isfinite(threshold):
        mask &= values <= float(threshold) + np.finfo(float).eps
    alternatives = values[mask]
    if alternatives.size == 0:
        return math.nan, 0
    return float(np.min(alternatives)), int(alternatives.size)


def _margin(alt_cost: float, chosen_cost: float) -> float:
    if not math.isfinite(alt_cost) or not math.isfinite(chosen_cost):
        return math.nan
    return float(alt_cost - chosen_cost)


def _pair_min(a: float, b: float) -> float:
    vals = [value for value in (a, b) if math.isfinite(value)]
    return float(min(vals)) if vals else math.nan


def _rank_by_cost(row: np.ndarray, chosen_col: int) -> int:
    # Stable sorting makes tied ranks deterministic and preserves column order.
    order = np.argsort(row, kind="mergesort")
    where = np.flatnonzero(order == int(chosen_col))
    return int(where[0]) + 1 if where.size else -1


def _box(track) -> Tuple[float, float, float, float]:
    values = np.asarray(track.tlbr, dtype=np.float32).reshape(-1)
    if values.size < 4:
        return (math.nan, math.nan, math.nan, math.nan)
    return tuple(float(value) for value in values[:4])


def _patched_record_assoc_debug(
    self,
    stage: str,
    tracks: Sequence,
    detections: Sequence,
    cost: np.ndarray,
    matches: np.ndarray,
    row_m: np.ndarray,
    col_m: np.ndarray,
    debug: Dict[str, np.ndarray],
) -> None:
    del row_m, col_m  # Existing margins are not chosen-specific.
    if _WRITER is None or _OBSERVE_CONFIG is None:
        raise RuntimeError("observe writer is not initialized")

    matrix = np.asarray(cost, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"association cost must be 2-D, got {matrix.shape}")
    n_tracks, n_detections = matrix.shape
    threshold = _stage_threshold(self, str(stage))
    raw_iou = np.asarray(debug.get("raw_iou", np.full_like(matrix, np.nan)), dtype=np.float32)
    embedding = debug.get("emb")
    if embedding is None:
        embedding_matrix = np.full_like(matrix, np.nan, dtype=np.float32)
    else:
        embedding_matrix = np.asarray(embedding, dtype=np.float32)
        if embedding_matrix.shape != matrix.shape:
            embedding_matrix = np.full_like(matrix, np.nan, dtype=np.float32)
    if raw_iou.shape != matrix.shape:
        raw_iou = np.full_like(matrix, np.nan, dtype=np.float32)

    match_array = np.asarray(matches, dtype=np.int64)
    if match_array.size == 0:
        match_array = np.zeros((0, 2), dtype=np.int64)
    else:
        match_array = match_array.reshape(-1, 2)

    _STATS["assoc_calls"] += 1
    _STATS["matrix_pairs"] += int(matrix.size)
    _STATS["stage_calls"][str(stage)] += 1
    _STATS["chosen_pairs_total"] += int(len(match_array))
    _STATS["stage_chosen_pairs"][str(stage)] += int(len(match_array))

    chosen_by_track: Dict[int, int] = {}
    chosen_pairs: set[Tuple[int, int]] = set()
    for track_row, det_col in match_array.tolist():
        if not (0 <= track_row < n_tracks and 0 <= det_col < n_detections):
            _STATS["invalid_match_indices"] += 1
            continue
        chosen_by_track[int(track_row)] = int(det_col)
        chosen_pairs.add((int(track_row), int(det_col)))

    top_k = max(0, int(_OBSERVE_CONFIG.observe_top_k))
    for track_row, track in enumerate(tracks):
        columns: set[int] = set()
        if top_k > 0 and n_detections > 0:
            order = np.argsort(matrix[track_row], kind="mergesort")
            columns.update(int(value) for value in order[: min(top_k, n_detections)].tolist())
        if track_row in chosen_by_track:
            columns.add(int(chosen_by_track[track_row]))

        for det_col in sorted(columns, key=lambda value: (float(matrix[track_row, value]), value)):
            det = detections[det_col]
            chosen = (track_row, det_col) in chosen_pairs
            chosen_rank = _rank_by_cost(matrix[track_row], det_col)
            chosen_cost = float(matrix[track_row, det_col])

            row_alt_all, _ = _alternative_cost(matrix[track_row], det_col)
            col_alt_all, _ = _alternative_cost(matrix[:, det_col], track_row)
            row_alt_valid, row_valid_count = _alternative_cost(
                matrix[track_row], det_col, threshold=threshold
            )
            col_alt_valid, col_valid_count = _alternative_cost(
                matrix[:, det_col], track_row, threshold=threshold
            )
            row_margin_all = _margin(row_alt_all, chosen_cost)
            col_margin_all = _margin(col_alt_all, chosen_cost)
            row_margin_valid = _margin(row_alt_valid, chosen_cost)
            col_margin_valid = _margin(col_alt_valid, chosen_cost)

            track_box = _box(track)
            det_box = _box(det)
            smooth_feat = getattr(track, "smooth_feat", None)
            current_feat = getattr(det, "curr_feat", None)
            raw_iou_cost = float(raw_iou[track_row, det_col])
            emb_cost = float(embedding_matrix[track_row, det_col])
            history = getattr(track, "features", ())
            try:
                history_len = len(history)
            except TypeError:
                history_len = -1

            soft_module = sys.modules.get("dmm_base_tracker_trusttrack_soft")
            soft_key = (
                int(getattr(track, "track_id", -1)),
                int(getattr(det, "det_global_idx", -1)),
            )
            soft_alpha_map = (
                getattr(soft_module, "_TRUST_SOFT_ALPHA_MAP", {})
                if soft_module is not None
                else {}
            )
            soft_reason_map = (
                getattr(soft_module, "_TRUST_SOFT_REASON_MAP", {})
                if soft_module is not None
                else {}
            )
            soft_meta_map = (
                getattr(soft_module, "_TRUST_PAIR_META", {})
                if soft_module is not None
                else {}
            )
            soft_meta = dict(soft_meta_map.get(soft_key, {}))
            soft_planned_alpha = soft_alpha_map.get(soft_key)

            reason_parts = []
            if chosen:
                reason_parts.append("chosen")
            if top_k > 0 and chosen_rank <= top_k:
                reason_parts.append("top_k")
            row = {
                "frame": int(self.frame_id),
                "stage": str(stage),
                "track_row": int(track_row),
                "det_col": int(det_col),
                "chosen": int(chosen),
                "chosen_rank": int(chosen_rank),
                "logged_reason": "|".join(reason_parts),
                "assignment_threshold": float(threshold),
                "final_valid": int(chosen_cost <= threshold + np.finfo(float).eps),
                "track_id": int(getattr(track, "track_id", -1)),
                "det_global_idx": int(getattr(det, "det_global_idx", -1)),
                "track_state": _state_name(getattr(track, "state", "unknown")),
                "track_is_activated": int(bool(getattr(track, "is_activated", False))),
                "track_start_frame": int(getattr(track, "start_frame", -1)),
                "track_last_frame": int(getattr(track, "frame_id", -1)),
                "track_age": int(self.frame_id - int(getattr(track, "start_frame", self.frame_id))),
                "tracklet_len": int(getattr(track, "tracklet_len", -1)),
                "lost_age": int(max(0, self.frame_id - int(getattr(track, "frame_id", self.frame_id)))),
                "track_alpha": float(getattr(track, "alpha", math.nan)),
                "feature_history_len": int(history_len),
                "det_score": float(getattr(det, "score", math.nan)),
                "soft_map_hit": int(soft_planned_alpha is not None),
                "soft_planned_alpha": (
                    float(soft_planned_alpha)
                    if soft_planned_alpha is not None
                    else math.nan
                ),
                "soft_planned_reason": str(soft_reason_map.get(soft_key, "")),
                "soft_approx_trust": _float_or_nan(soft_meta.get("trust")),
                "soft_approx_collapse": _float_or_nan(soft_meta.get("collapse")),
                "soft_approx_app_pair_margin": _float_or_nan(
                    soft_meta.get("app_pair_margin")
                ),
                "soft_approx_motion_pair_margin": _float_or_nan(
                    soft_meta.get("motion_pair_margin")
                ),
                "soft_approx_iou_pair_margin": _float_or_nan(
                    soft_meta.get("iou_pair_margin")
                ),
                "soft_approx_shape_pair_margin": _float_or_nan(
                    soft_meta.get("shape_pair_margin")
                ),
                "final_cost": chosen_cost,
                "raw_iou_cost": raw_iou_cost,
                "iou_similarity": 1.0 - raw_iou_cost if math.isfinite(raw_iou_cost) else math.nan,
                "embedding_cost": emb_cost,
                "embedding_available": int(math.isfinite(emb_cost)),
                "smooth_current_cosine": _unit_cosine(smooth_feat, current_feat),
                "row_alt_cost_all": row_alt_all,
                "col_alt_cost_all": col_alt_all,
                "row_signed_margin_all": row_margin_all,
                "col_signed_margin_all": col_margin_all,
                "pair_signed_margin_all": _pair_min(row_margin_all, col_margin_all),
                "row_alt_cost_valid": row_alt_valid,
                "col_alt_cost_valid": col_alt_valid,
                "row_signed_margin_valid": row_margin_valid,
                "col_signed_margin_valid": col_margin_valid,
                "pair_signed_margin_valid": _pair_min(row_margin_valid, col_margin_valid),
                "row_valid_competitors": int(row_valid_count),
                "col_valid_competitors": int(col_valid_count),
                "row_candidate_count": int(n_detections),
                "col_candidate_count": int(n_tracks),
                "track_x1": track_box[0],
                "track_y1": track_box[1],
                "track_x2": track_box[2],
                "track_y2": track_box[3],
                "det_x1": det_box[0],
                "det_y1": det_box[1],
                "det_x2": det_box[2],
                "det_y2": det_box[3],
            }
            _WRITER.writerow(row)
            _STATS["logged_rows"] += 1
            _STATS["stage_logged_rows"][str(stage)] += 1

            if chosen:
                _STATS["chosen_pairs_logged"] += 1
                _STATS["chosen_rank_histogram"][str(chosen_rank)] += 1
                if chosen_rank > top_k and top_k > 0:
                    _STATS["chosen_rank_gt_top_k"] += 1
                key = (
                    int(self.frame_id),
                    str(stage),
                    int(getattr(track, "track_id", -1)),
                    int(getattr(det, "det_global_idx", -1)),
                )
                if key in _CHOSEN_KEYS:
                    _STATS["duplicate_chosen_keys"] += 1
                _CHOSEN_KEYS.add(key)


def _jsonable_stats() -> dict:
    result = dict(_STATS)
    for key in (
        "stage_calls",
        "stage_chosen_pairs",
        "stage_logged_rows",
        "chosen_rank_histogram",
    ):
        result[key] = dict(result[key])
    result["chosen_log_completeness"] = (
        int(result["chosen_pairs_logged"]) / max(1, int(result["chosen_pairs_total"]))
    )
    return result


def parse_observe_args() -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--observe-csv", required=True)
    parser.add_argument("--observe-summary-json", required=True)
    parser.add_argument("--observe-top-k", type=int, default=0)
    args, remaining = parser.parse_known_args()
    return args, remaining


def main() -> None:
    global _WRITER, _HANDLE, _OBSERVE_CONFIG
    observe, remaining = parse_observe_args()
    if observe.observe_top_k < 0:
        raise ValueError("--observe-top-k must be >= 0")
    if "--debug-csv" in remaining:
        raise ValueError("Do not pass --debug-csv; use --observe-csv")
    if "--debug-assoc" not in remaining:
        remaining.append("--debug-assoc")

    csv_path = Path(observe.observe_csv)
    summary_path = Path(observe.observe_summary_json)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    _reset_stats()
    _OBSERVE_CONFIG = observe

    old_argv = sys.argv
    spda.base.DMMBaseTracker._record_assoc_debug = _patched_record_assoc_debug
    failure: str | None = None
    try:
        _HANDLE = csv_path.open("w", newline="", encoding="utf-8")
        _WRITER = csv.DictWriter(_HANDLE, fieldnames=_FIELDS)
        _WRITER.writeheader()
        sys.argv = [sys.argv[0]] + remaining
        v3.main()
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        sys.argv = old_argv
        spda.base.DMMBaseTracker._record_assoc_debug = _ORIGINAL_RECORD_ASSOC_DEBUG
        if _HANDLE is not None:
            _HANDLE.flush()
            _HANDLE.close()
        _WRITER = None
        payload = {
            "observe_csv": str(csv_path),
            "top_k": int(observe.observe_top_k),
            "failure": failure,
            "stats": _jsonable_stats(),
        }
        summary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    stats = _jsonable_stats()
    if int(stats["invalid_match_indices"]) != 0:
        raise RuntimeError(f"invalid match indices observed: {stats}")
    if int(stats["chosen_pairs_logged"]) != int(stats["chosen_pairs_total"]):
        raise RuntimeError(f"chosen-pair logging incomplete: {stats}")
    if int(stats["duplicate_chosen_keys"]) != 0:
        raise RuntimeError(f"duplicate chosen keys observed: {stats}")


if __name__ == "__main__":
    main()
