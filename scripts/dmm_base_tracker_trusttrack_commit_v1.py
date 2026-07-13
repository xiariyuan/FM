#!/usr/bin/env python3
"""TrustTrack commit-aware v1.

This wrapper keeps TrustTrack-soft v1 unchanged except for one online,
association-aligned temporal override:

If the same track actually received a soft feature update on the immediately
previous frame, and the current *actual primary Hungarian match* is a strong
geometric commit, cancel soft attenuation for the current update and use the
baseline track alpha.

The frozen v1 rule is fully online and GT-free:

- previous actual soft update gap == 1 frame;
- current pair is a planned TrustTrack-soft update;
- actual primary chosen rank == 1;
- chosen-specific signed pair margin >= 0.02;
- raw IoU cost <= 0.10;
- detection score >= 0.60.

The recovered/stable tracker core is not modified. The association callback is
invoked after the real cost matrix and Hungarian assignment are available and
before DMMTrack.update consumes the soft alpha map.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

import dmm_base_tracker_shadow_pda as spda
import dmm_base_tracker_trusttrack_soft as soft

_ORIGINAL_RECORD_ASSOC_DEBUG = spda.base.DMMBaseTracker._record_assoc_debug
_ORIGINAL_TRACK_UPDATE = spda.base.DMMTrack.update
_SOFT_UPDATE_IMPL = None
_CONFIG = None

_LAST_ACTUAL_SOFT_FRAME: Dict[int, int] = {}
_EVENT_BY_UPDATE_KEY: Dict[Tuple[int, int, int], dict] = {}
_EVENT_ROWS: List[dict] = []
_SNAPSHOT_KEYS: set[Tuple[int, int, int]] = set()
_SNAPSHOT_ROWS: List[dict] = []
_STATS: dict = {}


_SNAPSHOT_FIELDS = [
    "frame",
    "track_id",
    "det_global_idx",
    "update_mode",
    "soft_alpha_at_update",
    "soft_reason_at_update",
    "override_planned",
    "override_actual_update",
    "track_alpha",
    "feature_available",
    "smooth_before_hash",
    "smooth_after_hash",
    "current_feature_hash",
    "smooth_before_current_cosine",
    "smooth_after_current_cosine",
    "smooth_before_after_cosine",
    "history_len_before",
    "history_len_after",
    "history_last_before_current_cosine",
    "history_last_after_current_cosine",
]


def _feature_copy(value):
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32).copy()
    return array if array.size else None


def _feature_hash(value) -> str:
    array = _feature_copy(value)
    if array is None:
        return ""
    return hashlib.sha256(array.tobytes()).hexdigest()


def _cosine(a, b) -> float:
    aa = _feature_copy(a)
    bb = _feature_copy(b)
    if aa is None or bb is None or aa.shape != bb.shape:
        return math.nan
    na = float(np.linalg.norm(aa))
    nb = float(np.linalg.norm(bb))
    if na < 1e-12 or nb < 1e-12:
        return math.nan
    return float(np.dot(aa / na, bb / nb))


def _history_last(track):
    history = getattr(track, "features", None)
    if history is None:
        return None
    try:
        if len(history) == 0:
            return None
        return _feature_copy(history[-1])
    except (TypeError, IndexError):
        return None


def _load_snapshot_keys(path: str) -> set[Tuple[int, int, int]]:
    if not path:
        return set()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("keys", payload.get("events", []))
    if not isinstance(payload, list):
        raise ValueError("snapshot key JSON must contain a list or keys/events")
    keys = set()
    for item in payload:
        if isinstance(item, dict):
            keys.add((int(item["frame"]), int(item.get("track_id", item.get("track"))), int(item.get("det_global_idx", item.get("det")))))
        elif isinstance(item, (list, tuple)) and len(item) == 3:
            keys.add((int(item[0]), int(item[1]), int(item[2])))
        else:
            raise ValueError(f"unsupported snapshot key: {item!r}")
    return keys

_LOG_FIELDS = [
    "frame",
    "stage",
    "track_id",
    "det_global_idx",
    "soft_planned_before",
    "soft_alpha_before",
    "soft_reason_before",
    "previous_actual_soft_frame",
    "previous_frame_was_soft",
    "chosen_rank",
    "final_cost",
    "raw_iou_cost",
    "embedding_cost",
    "row_signed_margin",
    "col_signed_margin",
    "pair_signed_margin",
    "det_score",
    "rule_rank_ok",
    "rule_margin_ok",
    "rule_iou_ok",
    "rule_score_ok",
    "override_planned",
    "override_actual_update",
    "soft_applied_after_override",
    "feature_available",
]


def _reset_soft_state() -> None:
    soft._TRUST_SOFT_ALPHA_MAP = {}
    soft._TRUST_SOFT_REASON_MAP = {}
    soft._TRUST_PAIR_META = {}
    soft._TRUST_MATCH_ROWS.clear()
    soft._TRUST_STATS.clear()
    soft._TRUST_STATS.update(
        {
            "frames": 0,
            "candidate_pairs": 0,
            "soft_pairs_predicted": 0,
            "feature_updates_soft": 0,
            "feature_updates_normal": 0,
            "soft_alpha_sum": 0.0,
        }
    )
    soft._PATCHED = False


def _reset_commit_state() -> None:
    global _LAST_ACTUAL_SOFT_FRAME, _EVENT_BY_UPDATE_KEY, _EVENT_ROWS, _SNAPSHOT_ROWS, _STATS
    _LAST_ACTUAL_SOFT_FRAME = {}
    _EVENT_BY_UPDATE_KEY = {}
    _EVENT_ROWS = []
    _SNAPSHOT_ROWS = []
    _STATS = {
        "association_callbacks": 0,
        "primary_callbacks": 0,
        "primary_chosen_pairs": 0,
        "primary_chosen_soft_pairs": 0,
        "previous_frame_soft_candidates": 0,
        "override_planned": 0,
        "override_actual_update": 0,
        "override_missing_update": 0,
        "override_feature_missing": 0,
        "override_soft_still_present": 0,
        "actual_soft_updates_seen": 0,
        "normal_updates_seen": 0,
        "duplicate_event_keys": 0,
        "stage_calls": Counter(),
    }


def _rank_by_cost(row: np.ndarray, chosen_col: int) -> int:
    order = np.argsort(row, kind="mergesort")
    where = np.flatnonzero(order == int(chosen_col))
    return int(where[0]) + 1 if where.size else -1


def _alternative_min(values: np.ndarray, chosen_index: int) -> float:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size <= 1:
        return math.nan
    mask = np.ones(values.size, dtype=bool)
    mask[int(chosen_index)] = False
    mask &= np.isfinite(values)
    alternatives = values[mask]
    if alternatives.size == 0:
        return math.nan
    return float(np.min(alternatives))


def _signed_margin(alt_cost: float, chosen_cost: float) -> float:
    if not math.isfinite(alt_cost) or not math.isfinite(chosen_cost):
        return math.nan
    return float(alt_cost - chosen_cost)


def _pair_margin(row_margin: float, col_margin: float) -> float:
    values = [value for value in (row_margin, col_margin) if math.isfinite(value)]
    return float(min(values)) if values else math.nan


def _matrix_or_nan(debug: dict, key: str, shape: Tuple[int, int]) -> np.ndarray:
    value = debug.get(key)
    if value is None:
        return np.full(shape, np.nan, dtype=np.float32)
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.shape != shape:
        return np.full(shape, np.nan, dtype=np.float32)
    return matrix


def _commit_record_assoc_debug(
    self,
    stage: str,
    tracks: Sequence,
    detections: Sequence,
    cost: np.ndarray,
    matches: np.ndarray,
    row_m: np.ndarray,
    col_m: np.ndarray,
    debug: dict,
) -> None:
    del row_m, col_m
    if _CONFIG is None:
        raise RuntimeError("commit configuration is not initialized")

    stage = str(stage)
    _STATS["association_callbacks"] += 1
    _STATS["stage_calls"][stage] += 1
    if stage != "primary":
        return
    _STATS["primary_callbacks"] += 1

    matrix = np.asarray(cost, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"association cost must be 2-D, got {matrix.shape}")
    raw_iou = _matrix_or_nan(debug, "raw_iou", matrix.shape)
    embedding = _matrix_or_nan(debug, "emb", matrix.shape)

    match_array = np.asarray(matches, dtype=np.int64)
    if match_array.size == 0:
        return
    match_array = match_array.reshape(-1, 2)
    _STATS["primary_chosen_pairs"] += int(len(match_array))

    for track_row, det_col in match_array.tolist():
        if not (0 <= track_row < len(tracks) and 0 <= det_col < len(detections)):
            raise IndexError(
                f"invalid primary match index {(track_row, det_col)} for {matrix.shape}"
            )
        track = tracks[int(track_row)]
        detection = detections[int(det_col)]
        track_id = int(getattr(track, "track_id", -1))
        det_global_idx = int(getattr(detection, "det_global_idx", -1))
        soft_key = (track_id, det_global_idx)
        soft_alpha = soft._TRUST_SOFT_ALPHA_MAP.get(soft_key)
        if soft_alpha is None:
            continue

        _STATS["primary_chosen_soft_pairs"] += 1
        frame = int(self.frame_id)
        update_key = (frame, track_id, det_global_idx)
        if update_key in _EVENT_BY_UPDATE_KEY:
            _STATS["duplicate_event_keys"] += 1
            raise RuntimeError(f"duplicate commit event key: {update_key}")

        chosen_cost = float(matrix[track_row, det_col])
        row_alt = _alternative_min(matrix[track_row, :], det_col)
        col_alt = _alternative_min(matrix[:, det_col], track_row)
        row_margin = _signed_margin(row_alt, chosen_cost)
        col_margin = _signed_margin(col_alt, chosen_cost)
        pair_margin = _pair_margin(row_margin, col_margin)
        chosen_rank = _rank_by_cost(matrix[track_row, :], det_col)
        raw_iou_cost = float(raw_iou[track_row, det_col])
        embedding_cost = float(embedding[track_row, det_col])
        det_score = float(getattr(detection, "score", math.nan))
        previous_soft_frame = _LAST_ACTUAL_SOFT_FRAME.get(track_id)
        previous_frame_was_soft = int(previous_soft_frame == frame - 1)
        if previous_frame_was_soft:
            _STATS["previous_frame_soft_candidates"] += 1

        rank_ok = chosen_rank == int(_CONFIG.rank)
        margin_ok = math.isfinite(pair_margin) and pair_margin >= float(
            _CONFIG.min_pair_margin
        )
        iou_ok = math.isfinite(raw_iou_cost) and raw_iou_cost <= float(
            _CONFIG.max_iou_cost
        )
        score_ok = math.isfinite(det_score) and det_score >= float(
            _CONFIG.min_det_score
        )
        override = bool(
            _CONFIG.enable
            and previous_frame_was_soft
            and rank_ok
            and margin_ok
            and iou_ok
            and score_ok
        )

        event = {
            "frame": frame,
            "stage": stage,
            "track_id": track_id,
            "det_global_idx": det_global_idx,
            "soft_planned_before": 1,
            "soft_alpha_before": float(soft_alpha),
            "soft_reason_before": str(soft._TRUST_SOFT_REASON_MAP.get(soft_key, "")),
            "previous_actual_soft_frame": (
                int(previous_soft_frame) if previous_soft_frame is not None else ""
            ),
            "previous_frame_was_soft": previous_frame_was_soft,
            "chosen_rank": chosen_rank,
            "final_cost": chosen_cost,
            "raw_iou_cost": raw_iou_cost,
            "embedding_cost": embedding_cost,
            "row_signed_margin": row_margin,
            "col_signed_margin": col_margin,
            "pair_signed_margin": pair_margin,
            "det_score": det_score,
            "rule_rank_ok": int(rank_ok),
            "rule_margin_ok": int(margin_ok),
            "rule_iou_ok": int(iou_ok),
            "rule_score_ok": int(score_ok),
            "override_planned": int(override),
            "override_actual_update": 0,
            "soft_applied_after_override": -1,
            "feature_available": -1,
        }
        _EVENT_BY_UPDATE_KEY[update_key] = event
        _EVENT_ROWS.append(event)

        if override:
            _STATS["override_planned"] += 1
            # This callback is executed before the track.update loop. Removing
            # the key here changes only the current feature-update alpha; it
            # does not alter the already-computed Hungarian assignment.
            soft._TRUST_SOFT_ALPHA_MAP.pop(soft_key, None)
            soft._TRUST_SOFT_REASON_MAP.pop(soft_key, None)
            soft._TRUST_PAIR_META.pop(soft_key, None)


def _commit_track_update(self, new_track, frame_id):
    if _SOFT_UPDATE_IMPL is None:
        raise RuntimeError("soft update implementation is not initialized")
    frame = int(frame_id)
    track_id = int(getattr(self, "track_id", -1))
    det_global_idx = int(getattr(new_track, "det_global_idx", -1))
    soft_key = (track_id, det_global_idx)
    update_key = (frame, track_id, det_global_idx)
    feature_available = getattr(new_track, "curr_feat", None) is not None
    soft_alpha_at_update = soft._TRUST_SOFT_ALPHA_MAP.get(soft_key)
    soft_reason_at_update = str(soft._TRUST_SOFT_REASON_MAP.get(soft_key, ""))
    soft_will_apply = bool(soft_alpha_at_update is not None and feature_available)
    event = _EVENT_BY_UPDATE_KEY.get(update_key)

    if event is not None:
        event["feature_available"] = int(feature_available)
        event["soft_applied_after_override"] = int(soft_will_apply)
        if int(event["override_planned"]):
            if soft_key in soft._TRUST_SOFT_ALPHA_MAP:
                _STATS["override_soft_still_present"] += 1
            if not feature_available:
                _STATS["override_feature_missing"] += 1
            else:
                event["override_actual_update"] = 1
                _STATS["override_actual_update"] += 1

    snapshot = update_key in _SNAPSHOT_KEYS
    if snapshot:
        smooth_before = _feature_copy(getattr(self, "smooth_feat", None))
        current_feature = _feature_copy(getattr(new_track, "curr_feat", None))
        history_before = getattr(self, "features", ())
        try:
            history_len_before = len(history_before)
        except TypeError:
            history_len_before = -1
        history_last_before = _history_last(self)

    _SOFT_UPDATE_IMPL(self, new_track, frame_id)

    if snapshot:
        smooth_after = _feature_copy(getattr(self, "smooth_feat", None))
        history_after = getattr(self, "features", ())
        try:
            history_len_after = len(history_after)
        except TypeError:
            history_len_after = -1
        history_last_after = _history_last(self)
        _SNAPSHOT_ROWS.append(
            {
                "frame": frame,
                "track_id": track_id,
                "det_global_idx": det_global_idx,
                "update_mode": "soft" if soft_will_apply else "normal",
                "soft_alpha_at_update": float(soft_alpha_at_update) if soft_alpha_at_update is not None else "",
                "soft_reason_at_update": soft_reason_at_update,
                "override_planned": int(event["override_planned"]) if event is not None else 0,
                "override_actual_update": int(event["override_actual_update"]) if event is not None else 0,
                "track_alpha": float(getattr(self, "alpha", math.nan)),
                "feature_available": int(feature_available),
                "smooth_before_hash": _feature_hash(smooth_before),
                "smooth_after_hash": _feature_hash(smooth_after),
                "current_feature_hash": _feature_hash(current_feature),
                "smooth_before_current_cosine": _cosine(smooth_before, current_feature),
                "smooth_after_current_cosine": _cosine(smooth_after, current_feature),
                "smooth_before_after_cosine": _cosine(smooth_before, smooth_after),
                "history_len_before": history_len_before,
                "history_len_after": history_len_after,
                "history_last_before_current_cosine": _cosine(history_last_before, current_feature),
                "history_last_after_current_cosine": _cosine(history_last_after, current_feature),
            }
        )

    if soft_will_apply:
        _LAST_ACTUAL_SOFT_FRAME[track_id] = frame
        _STATS["actual_soft_updates_seen"] += 1
    else:
        _STATS["normal_updates_seen"] += 1


def _jsonable_stats() -> dict:
    result = dict(_STATS)
    result["stage_calls"] = dict(result["stage_calls"])
    result["last_actual_soft_tracks"] = len(_LAST_ACTUAL_SOFT_FRAME)
    return result


def _write_outputs(log_csv: Path, summary_json: Path, failure: str | None) -> None:
    planned_keys = {
        (int(row["frame"]), int(row["track_id"]), int(row["det_global_idx"]))
        for row in _EVENT_ROWS
        if int(row["override_planned"])
    }
    actual_keys = {
        (int(row["frame"]), int(row["track_id"]), int(row["det_global_idx"]))
        for row in _EVENT_ROWS
        if int(row["override_actual_update"])
    }
    _STATS["override_missing_update"] = len(planned_keys - actual_keys)

    log_csv.parent.mkdir(parents=True, exist_ok=True)
    with log_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_LOG_FIELDS)
        writer.writeheader()
        writer.writerows(_EVENT_ROWS)

    snapshot_actual_keys = {
        (int(row["frame"]), int(row["track_id"]), int(row["det_global_idx"]))
        for row in _SNAPSHOT_ROWS
    }
    if getattr(_CONFIG, "snapshot_csv", ""):
        snapshot_path = Path(_CONFIG.snapshot_csv)
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        with snapshot_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=_SNAPSHOT_FIELDS)
            writer.writeheader()
            writer.writerows(_SNAPSHOT_ROWS)

    payload = {
        "failure": failure,
        "config": vars(_CONFIG) if _CONFIG is not None else {},
        "stats": _jsonable_stats(),
        "override_planned_keys": [list(key) for key in sorted(planned_keys)],
        "override_actual_keys": [list(key) for key in sorted(actual_keys)],
        "planned_actual_equal": planned_keys == actual_keys,
        "snapshot_requested_keys": [list(key) for key in sorted(_SNAPSHOT_KEYS)],
        "snapshot_actual_keys": [list(key) for key in sorted(snapshot_actual_keys)],
        "snapshot_missing_keys": [list(key) for key in sorted(_SNAPSHOT_KEYS - snapshot_actual_keys)],
        "diagnostic_only": False,
        "uses_ground_truth": False,
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_commit_args() -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--trust-commit-enable", action="store_true")
    parser.add_argument("--trust-commit-rank", type=int, default=1)
    parser.add_argument("--trust-commit-min-pair-margin", type=float, default=0.02)
    parser.add_argument("--trust-commit-max-iou-cost", type=float, default=0.10)
    parser.add_argument("--trust-commit-min-det-score", type=float, default=0.60)
    parser.add_argument("--trust-commit-log-csv", required=True)
    parser.add_argument("--trust-commit-summary-json", required=True)
    parser.add_argument("--trust-memory-snapshot-keys-json", default="")
    parser.add_argument("--trust-memory-snapshot-csv", default="")
    args, remaining = parser.parse_known_args()
    if args.trust_commit_rank < 1:
        raise ValueError("--trust-commit-rank must be >= 1")
    if not 0.0 <= args.trust_commit_max_iou_cost <= 1.0:
        raise ValueError("--trust-commit-max-iou-cost must be in [0, 1]")
    if not 0.0 <= args.trust_commit_min_det_score <= 1.0:
        raise ValueError("--trust-commit-min-det-score must be in [0, 1]")
    if bool(args.trust_memory_snapshot_keys_json) != bool(args.trust_memory_snapshot_csv):
        raise ValueError("snapshot keys JSON and snapshot CSV must be supplied together")
    config = argparse.Namespace(
        enable=bool(args.trust_commit_enable),
        rank=int(args.trust_commit_rank),
        min_pair_margin=float(args.trust_commit_min_pair_margin),
        max_iou_cost=float(args.trust_commit_max_iou_cost),
        min_det_score=float(args.trust_commit_min_det_score),
        log_csv=str(args.trust_commit_log_csv),
        summary_json=str(args.trust_commit_summary_json),
        snapshot_keys_json=str(args.trust_memory_snapshot_keys_json),
        snapshot_csv=str(args.trust_memory_snapshot_csv),
    )
    return config, remaining


def main() -> None:
    global _CONFIG, _SOFT_UPDATE_IMPL, _SNAPSHOT_KEYS
    config, remaining = parse_commit_args()
    if "--debug-csv" in remaining:
        raise ValueError(
            "--debug-csv is not supported by commit v1; use --trust-commit-log-csv"
        )
    if "--debug-assoc" not in remaining:
        remaining.append("--debug-assoc")

    _CONFIG = config
    _SNAPSHOT_KEYS = _load_snapshot_keys(config.snapshot_keys_json)
    _reset_commit_state()
    _reset_soft_state()
    soft._ensure_patch()
    _SOFT_UPDATE_IMPL = spda.base.DMMTrack.update

    old_argv = sys.argv
    failure: str | None = None
    spda.base.DMMBaseTracker._record_assoc_debug = _commit_record_assoc_debug
    spda.base.DMMTrack.update = _commit_track_update
    try:
        sys.argv = [sys.argv[0]] + remaining
        soft.main()
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        sys.argv = old_argv
        spda.base.DMMBaseTracker._record_assoc_debug = _ORIGINAL_RECORD_ASSOC_DEBUG
        spda.base.DMMTrack.update = _ORIGINAL_TRACK_UPDATE
        soft._PATCHED = False
        _write_outputs(
            Path(config.log_csv),
            Path(config.summary_json),
            failure,
        )

    stats = _jsonable_stats()
    if int(stats["duplicate_event_keys"]) != 0:
        raise RuntimeError(f"duplicate commit event keys: {stats}")
    if int(stats["override_soft_still_present"]) != 0:
        raise RuntimeError(f"override keys remained in soft map: {stats}")
    if int(stats["override_feature_missing"]) != 0:
        raise RuntimeError(f"override chosen update lacked a feature: {stats}")
    if int(stats["override_missing_update"]) != 0:
        raise RuntimeError(f"planned override did not reach track.update: {stats}")
    if int(stats["override_planned"]) != int(stats["override_actual_update"]):
        raise RuntimeError(f"planned/applied override mismatch: {stats}")


if __name__ == "__main__":
    main()
