#!/usr/bin/env python3
"""Summarize actual TrustTrack-soft interventions against a baseline observe log.

Both inputs must be GT-annotated observe-v2 chosen-pair CSVs. Soft events are
identified by soft_map_hit=1. The baseline comparison uses the same
(frame, det_global_idx) chosen detection when present. GT labels remain offline
and are never used by the tracker.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def as_int(row: dict, key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default)))
    except (TypeError, ValueError):
        return int(default)


def as_float(row: dict, key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return math.nan


def load_rows(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def unique_detection_index(rows: List[dict]) -> Tuple[Dict[Tuple[int, int], dict], int]:
    index: Dict[Tuple[int, int], dict] = {}
    duplicates = 0
    for row in rows:
        key = (as_int(row, "frame"), as_int(row, "det_global_idx", -1))
        if key in index:
            duplicates += 1
            # Prefer primary, then output-aligned rows for deterministic joins.
            old = index[key]
            old_score = (old.get("stage") == "primary", as_int(old, "chosen_track_is_current_output"))
            new_score = (row.get("stage") == "primary", as_int(row, "chosen_track_is_current_output"))
            if new_score > old_score:
                index[key] = row
        else:
            index[key] = row
    return index, duplicates


def compact_event(soft: dict, baseline: dict | None) -> dict:
    event = {
        "frame": as_int(soft, "frame"),
        "track_id": as_int(soft, "track_id", -1),
        "det_global_idx": as_int(soft, "det_global_idx", -1),
        "det_gt": as_int(soft, "det_gt", -1) if soft.get("det_gt") not in (None, "") else None,
        "det_score": as_float(soft, "det_score"),
        "alpha": as_float(soft, "soft_planned_alpha"),
        "approx_collapse": as_float(soft, "soft_approx_collapse"),
        "actual_final_cost": as_float(soft, "final_cost"),
        "actual_iou_cost": as_float(soft, "raw_iou_cost"),
        "actual_emb_cost": as_float(soft, "embedding_cost"),
        "actual_row_margin": as_float(soft, "row_signed_margin_all"),
        "actual_col_margin": as_float(soft, "col_signed_margin_all"),
        "actual_pair_margin": as_float(soft, "pair_signed_margin_all"),
        "chosen_rank": as_int(soft, "chosen_rank", -1),
        "smooth_cos": as_float(soft, "smooth_current_cosine"),
        "track_history_label": soft.get("track_history_label", "unknown"),
        "gt_transition_label": soft.get("gt_transition_label", "unknown"),
        "future3": as_int(soft, "gt_future_transition_3"),
        "future10": as_int(soft, "gt_future_transition_10"),
        "chosen_output_aligned": as_int(soft, "chosen_track_is_current_output"),
        "baseline_present": int(baseline is not None),
    }
    if baseline is None:
        event.update(
            {
                "baseline_track": None,
                "same_baseline_track": 0,
                "baseline_transition": "absent",
                "baseline_future3": 0,
                "baseline_future10": 0,
                "baseline_pair_margin": math.nan,
                "baseline_final_cost": math.nan,
            }
        )
    else:
        baseline_track = as_int(baseline, "track_id", -1)
        event.update(
            {
                "baseline_track": baseline_track,
                "same_baseline_track": int(baseline_track == event["track_id"]),
                "baseline_transition": baseline.get("gt_transition_label", "unknown"),
                "baseline_future3": as_int(baseline, "gt_future_transition_3"),
                "baseline_future10": as_int(baseline, "gt_future_transition_10"),
                "baseline_pair_margin": as_float(baseline, "pair_signed_margin_all"),
                "baseline_final_cost": as_float(baseline, "final_cost"),
            }
        )
    return event


def summarize(events: List[dict], bin_width: int) -> dict:
    stage = Counter()
    track_history = Counter()
    transitions = Counter()
    baseline_transitions = Counter()
    bins: Dict[str, dict] = {}
    for event in events:
        stage["primary"] += 1
        track_history[event["track_history_label"]] += 1
        transitions[event["gt_transition_label"]] += 1
        baseline_transitions[event["baseline_transition"]] += 1
        start = ((int(event["frame"]) - 1) // bin_width) * bin_width + 1
        end = start + bin_width - 1
        key = f"{start:04d}-{end:04d}"
        bucket = bins.setdefault(
            key,
            {
                "events": 0,
                "different_track": 0,
                "soft_changed": 0,
                "baseline_changed": 0,
                "soft_future10": 0,
                "baseline_future10": 0,
                "negative_real_margin": 0,
                "rank_gt1": 0,
                "tracks": set(),
            },
        )
        bucket["events"] += 1
        bucket["different_track"] += int(not event["same_baseline_track"])
        bucket["soft_changed"] += int(event["gt_transition_label"] == "changed")
        bucket["baseline_changed"] += int(event["baseline_transition"] == "changed")
        bucket["soft_future10"] += int(event["future10"])
        bucket["baseline_future10"] += int(event["baseline_future10"])
        bucket["negative_real_margin"] += int(
            math.isfinite(event["actual_pair_margin"]) and event["actual_pair_margin"] < 0
        )
        bucket["rank_gt1"] += int(event["chosen_rank"] > 1)
        bucket["tracks"].add(int(event["track_id"]))
    for bucket in bins.values():
        bucket["tracks"] = sorted(bucket["tracks"])
    return {
        "soft_events": len(events),
        "soft_stage_counts": dict(stage),
        "soft_track_history_labels": dict(track_history),
        "soft_gt_transition_labels": dict(transitions),
        "baseline_gt_transition_labels_same_detection": dict(baseline_transitions),
        "baseline_absent": sum(not event["baseline_present"] for event in events),
        "different_track": sum(not event["same_baseline_track"] for event in events),
        "soft_changed": sum(event["gt_transition_label"] == "changed" for event in events),
        "baseline_changed_same_det": sum(event["baseline_transition"] == "changed" for event in events),
        "soft_future3": sum(int(event["future3"]) for event in events),
        "baseline_future3_same_det": sum(int(event["baseline_future3"]) for event in events),
        "soft_future10": sum(int(event["future10"]) for event in events),
        "baseline_future10_same_det": sum(int(event["baseline_future10"]) for event in events),
        "soft_actual_negative_pair_margin": sum(
            math.isfinite(event["actual_pair_margin"]) and event["actual_pair_margin"] < 0
            for event in events
        ),
        "soft_actual_negative_row_margin": sum(
            math.isfinite(event["actual_row_margin"]) and event["actual_row_margin"] < 0
            for event in events
        ),
        "soft_actual_negative_col_margin": sum(
            math.isfinite(event["actual_col_margin"]) and event["actual_col_margin"] < 0
            for event in events
        ),
        "soft_actual_rank_gt1": sum(event["chosen_rank"] > 1 for event in events),
        "soft_chosen_output_aligned": sum(int(event["chosen_output_aligned"]) for event in events),
        "time_bins": bins,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--soft-annotated-csv", required=True)
    parser.add_argument("--baseline-annotated-csv", required=True)
    parser.add_argument("--out-events-csv", required=True)
    parser.add_argument("--out-summary-json", required=True)
    parser.add_argument("--bin-width", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bin_width <= 0:
        raise ValueError("--bin-width must be positive")
    soft_rows = load_rows(Path(args.soft_annotated_csv))
    baseline_rows = load_rows(Path(args.baseline_annotated_csv))
    baseline_index, baseline_duplicate_keys = unique_detection_index(baseline_rows)
    events = []
    for row in soft_rows:
        if as_int(row, "soft_map_hit") != 1:
            continue
        key = (as_int(row, "frame"), as_int(row, "det_global_idx", -1))
        events.append(compact_event(row, baseline_index.get(key)))
    events.sort(key=lambda row: (row["frame"], row["track_id"], row["det_global_idx"]))

    event_path = Path(args.out_events_csv)
    event_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(events[0]) if events else []
    with event_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(events)

    summary = summarize(events, args.bin_width)
    summary.update(
        {
            "soft_annotated_csv": args.soft_annotated_csv,
            "baseline_annotated_csv": args.baseline_annotated_csv,
            "out_events_csv": args.out_events_csv,
            "bin_width": int(args.bin_width),
            "baseline_rows": len(baseline_rows),
            "baseline_duplicate_detection_keys": baseline_duplicate_keys,
        }
    )
    summary_path = Path(args.out_summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
