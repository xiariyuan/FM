#!/usr/bin/env python3
"""Offline ceiling analysis from track result files.

For each wrong_recovery_recoverable event, check whether the oracle_track_id
is still active (has a bbox) at the reentry_frame.  If yes, the event is
theoretically improvable by better association.

This gives a lower bound on the ceiling: it doesn't guarantee the oracle bbox
overlaps the GT well enough for a correct association, but it tells us how
many wrong events have the correct branch still "alive" at the right moment.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

REPO_ROOT = Path("/gemini/code/FMtrack-main/FM-Track")
REGISTRY_CSV = REPO_ROOT / "outputs" / "experiment_registry.csv"

SUMMARY_FIELDS = [
    "status",
    "error",
    "total_wrong_recoverable",
    "oracle_active_at_reentry",
    "oracle_not_active_at_reentry",
    "oracle_active_rate",
    "ceiling_correct_lower_bound",
    "ceiling_wrong_lower_bound",
    "ceiling_f1_lower_bound",
]

EVENT_FIELDS = [
    "seq_name",
    "gt_id",
    "exit_frame",
    "reentry_frame",
    "gap",
    "oracle_track_id",
    "tracker_track_id",
    "oracle_active_at_reentry",
    "oracle_bbox_count_at_reentry",
]


def write_single_row_csv(path: Path, fieldnames, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})


def write_rows(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def append_registry(summary_csv, status, notes):
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "append_experiment_record.py"),
        "--csv", str(REGISTRY_CSV),
        "--kind", "analysis",
        "--status", status,
        "--script", "scripts/analyze_reentry_ceiling_from_tracks.py",
        "--dataset", "DanceTrack",
        "--split", "val",
        "--tracker-family", "reentry_ceiling",
        "--variant", summary_csv.parent.name,
        "--tag", summary_csv.parent.name,
        "--run-root", str(summary_csv.parent.resolve()),
        "--summary-csv", str(summary_csv.resolve()),
        "--notes", notes,
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=False)


def load_track_results(track_dir: Path) -> Dict[str, Dict[int, Set[int]]]:
    """Load per-sequence track results: seq_name -> frame -> set of active track_ids."""
    results: Dict[str, Dict[int, Set[int]]] = {}
    if not track_dir.is_dir():
        return results
    for p in sorted(track_dir.glob("*.txt")):
        seq_name = p.stem
        frames: Dict[int, Set[int]] = defaultdict(set)
        with p.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 6:
                    continue
                frame = int(float(row[0]))
                tid = int(float(row[1]))
                # Only count rows with positive area (non-empty bbox)
                w = float(row[4])
                h = float(row[5])
                if w > 0 and h > 0:
                    frames[frame].add(tid)
        results[seq_name] = frames
    return results


def load_gt_events(event_details_csv: Path, tracker_name: str) -> List[dict]:
    with event_details_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [
            r for r in reader
            if r["tracker_name"] == tracker_name
            and r["status"] == "wrong_recovery_recoverable"
        ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Ceiling analysis from track result files.")
    parser.add_argument("--event-details-csv", required=True)
    parser.add_argument("--track-results-dir", required=True,
                        help="Directory containing per-sequence track result .txt files")
    parser.add_argument("--tracker-name", required=True)
    parser.add_argument("--baseline-correct", type=int, default=123,
                        help="Current correct count for F1 computation")
    parser.add_argument("--baseline-total", type=int, default=344,
                        help="Total events for F1 computation")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "summary.csv"
    detail_csv = out_dir / "oracle_active_events.csv"

    summary_row = {
        "status": "running",
        "error": "",
        "total_wrong_recoverable": 0,
        "oracle_active_at_reentry": 0,
        "oracle_not_active_at_reentry": 0,
        "oracle_active_rate": 0.0,
        "ceiling_correct_lower_bound": 0,
        "ceiling_wrong_lower_bound": 0,
        "ceiling_f1_lower_bound": 0.0,
    }
    write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)
    write_rows(detail_csv, EVENT_FIELDS, [])
    append_registry(summary_csv, "running", "ceiling analysis from track results")

    try:
        track_data = load_track_results(Path(args.track_results_dir))
        wrong_events = load_gt_events(Path(args.event_details_csv), args.tracker_name)

        total = len(wrong_events)
        active_count = 0
        not_active_count = 0
        detail_rows: List[dict] = []

        for event in wrong_events:
            seq = event["seq_name"]
            reentry_frame = int(event["reentry_frame"])
            oracle_tid = int(event["oracle_track_id"]) if event["oracle_track_id"] else -1
            tracker_tid = int(event["tracker_track_id"]) if event["tracker_track_id"] else -1

            seq_frames = track_data.get(seq, {})
            frame_tids = seq_frames.get(reentry_frame, set())
            oracle_active = oracle_tid in frame_tids

            if oracle_active:
                active_count += 1
            else:
                not_active_count += 1

            detail_rows.append({
                "seq_name": seq,
                "gt_id": event["gt_id"],
                "exit_frame": event["exit_frame"],
                "reentry_frame": reentry_frame,
                "gap": event["gap"],
                "oracle_track_id": oracle_tid,
                "tracker_track_id": tracker_tid,
                "oracle_active_at_reentry": int(oracle_active),
                "oracle_bbox_count_at_reentry": len(frame_tids),
            })

        # Ceiling: if all oracle-active events could be fixed
        ceiling_correct = args.baseline_correct + active_count
        ceiling_wrong = total - active_count
        total_events = args.baseline_total
        precision = ceiling_correct / (ceiling_correct + ceiling_wrong) if (ceiling_correct + ceiling_wrong) > 0 else 0
        recall = ceiling_correct / total_events if total_events > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        summary_row = {
            "status": "success",
            "error": "",
            "total_wrong_recoverable": total,
            "oracle_active_at_reentry": active_count,
            "oracle_not_active_at_reentry": not_active_count,
            "oracle_active_rate": round(active_count / total, 6) if total else 0.0,
            "ceiling_correct_lower_bound": ceiling_correct,
            "ceiling_wrong_lower_bound": ceiling_wrong,
            "ceiling_f1_lower_bound": round(f1, 6),
        }
        write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)
        write_rows(detail_csv, EVENT_FIELDS, detail_rows)
        append_registry(
            summary_csv, "success",
            f"ceiling: {active_count}/{total} oracle active at reentry frame, "
            f"F1 ceiling={f1:.4f} (from baseline {args.baseline_correct}/{args.baseline_total})"
        )
        return 0

    except Exception as exc:
        summary_row["status"] = "failed"
        summary_row["error"] = str(exc)
        write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)
        append_registry(summary_csv, "failed", f"ceiling analysis failed: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
