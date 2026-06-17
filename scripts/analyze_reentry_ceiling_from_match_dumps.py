#!/usr/bin/env python3
"""Offline ceiling analysis: how many wrong_recovery_recoverable events have the
correct oracle_track_id inside the engine's top-k candidates at the reentry frame?

Usage:
    python scripts/analyze_reentry_ceiling_from_match_dumps.py \
        --event-details-csv outputs/.../event_details.csv \
        --match-dump-dir .../engine_match_dumps \
        --tracker-name engine_recent_rerank_fullval \
        --top-k 8 \
        --out-dir outputs/...
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set

REPO_ROOT = Path("/gemini/code/FMtrack-main/FM-Track")
REGISTRY_CSV = REPO_ROOT / "outputs" / "experiment_registry.csv"

SUMMARY_FIELDS = [
    "status",
    "error",
    "total_wrong_recoverable",
    "wrong_with_oracle_in_topk",
    "wrong_without_oracle_in_topk",
    "ceiling_correct",
    "ceiling_wrong",
    "ceiling_improvable_rate",
    "top_k",
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
        "--script", "scripts/analyze_reentry_ceiling_from_match_dumps.py",
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


def load_match_dumps(dump_dir: Path) -> Dict[str, Dict[int, List[dict]]]:
    """Load match dumps keyed by seq_name -> frame -> list of match rows."""
    dumps: Dict[str, Dict[int, List[dict]]] = defaultdict(lambda: defaultdict(list))
    if not dump_dir.is_dir():
        return dumps
    for p in sorted(dump_dir.glob("*_matches.jsonl")):
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                frame = int(row["frame"])
                # The seq_name is encoded in the filename
                seq_name = p.name.replace("_matches.jsonl", "")
                dumps[seq_name][frame].append(row)
    return dumps


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-details-csv", required=True)
    parser.add_argument("--match-dump-dir", required=True,
                        help="Directory containing per-sequence *_matches.jsonl files")
    parser.add_argument("--tracker-name", required=True)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "summary.csv"
    detail_csv = out_dir / "improvable_events.csv"

    IMPROVABLE_FIELDS = [
        "seq_name", "gt_id", "exit_frame", "reentry_frame", "gap",
        "oracle_track_id", "tracker_track_id",
        "oracle_in_topk", "oracle_rank", "oracle_score", "selected_score",
        "score_delta",
    ]

    summary_row = {
        "status": "running",
        "error": "",
        "total_wrong_recoverable": 0,
        "wrong_with_oracle_in_topk": 0,
        "wrong_without_oracle_in_topk": 0,
        "ceiling_correct": 0,
        "ceiling_wrong": 0,
        "ceiling_improvable_rate": 0.0,
        "top_k": args.top_k,
    }
    write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)
    append_registry(summary_csv, "running", "analyzing ceiling from match dumps")

    try:
        # Load event details
        with open(args.event_details_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            wrong_events = [
                r for r in reader
                if r["tracker_name"] == args.tracker_name
                and r["status"] == "wrong_recovery_recoverable"
            ]

        # Load match dumps
        dumps = load_match_dumps(Path(args.match_dump_dir))

        total_wrong = len(wrong_events)
        with_oracle = 0
        without_oracle = 0
        improvable_rows: List[dict] = []

        for event in wrong_events:
            seq = event["seq_name"]
            reentry_frame = int(event["reentry_frame"])
            oracle_tid = int(event["oracle_track_id"]) if event["oracle_track_id"] else None
            tracker_tid = int(event["tracker_track_id"]) if event["tracker_track_id"] else None

            if oracle_tid is None:
                without_oracle += 1
                continue

            frame_matches = dumps.get(seq, {}).get(reentry_frame, [])
            if not frame_matches:
                without_oracle += 1
                continue

            # Take the first match row for this frame (one per detection)
            # Find the row that selected the wrong tracker_tid
            relevant = [m for m in frame_matches if int(m["selected_track_id"]) == tracker_tid]
            if not relevant:
                # Try any row at this frame
                relevant = frame_matches

            if not relevant:
                without_oracle += 1
                continue

            match_row = relevant[0]
            top_k_tracks = match_row["top_k_tracks"][:args.top_k]
            top_k_scores = match_row["top_k_scores"][:args.top_k]
            top_k_exit_frames = match_row.get("top_k_exit_frames", [])[:args.top_k]

            oracle_found = False
            oracle_rank = -1
            oracle_score = 0.0
            for i, tid in enumerate(top_k_tracks):
                if int(tid) == oracle_tid:
                    oracle_found = True
                    oracle_rank = i + 1
                    oracle_score = float(top_k_scores[i]) if i < len(top_k_scores) else 0.0
                    break

            if oracle_found:
                with_oracle += 1
                improvable_rows.append({
                    "seq_name": seq,
                    "gt_id": event["gt_id"],
                    "exit_frame": event["exit_frame"],
                    "reentry_frame": reentry_frame,
                    "gap": event["gap"],
                    "oracle_track_id": oracle_tid,
                    "tracker_track_id": tracker_tid,
                    "oracle_in_topk": 1,
                    "oracle_rank": oracle_rank,
                    "oracle_score": round(oracle_score, 6),
                    "selected_score": round(float(match_row["selected_composite_score"]), 6),
                    "score_delta": round(float(match_row["selected_composite_score"]) - oracle_score, 6),
                })
            else:
                without_oracle += 1

        ceiling_correct = 123 + with_oracle  # baseline correct + improvables
        ceiling_wrong = total_wrong - with_oracle

        summary_row = {
            "status": "success",
            "error": "",
            "total_wrong_recoverable": total_wrong,
            "wrong_with_oracle_in_topk": with_oracle,
            "wrong_without_oracle_in_topk": without_oracle,
            "ceiling_correct": ceiling_correct,
            "ceiling_wrong": ceiling_wrong,
            "ceiling_improvable_rate": round(with_oracle / total_wrong, 6) if total_wrong else 0.0,
            "top_k": args.top_k,
        }
        write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)
        write_rows(detail_csv, IMPROVABLE_FIELDS, improvable_rows)
        append_registry(summary_csv, "success",
                        f"ceiling: {with_oracle}/{total_wrong} wrong events improvable at top-{args.top_k}")
        return 0

    except Exception as exc:
        summary_row["status"] = "failed"
        summary_row["error"] = str(exc)
        write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)
        append_registry(summary_csv, "failed", f"ceiling analysis failed: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
