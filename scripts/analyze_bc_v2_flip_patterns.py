#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path


REPO_ROOT = Path("/gemini/code/FMtrack-main/FM-Track")
REGISTRY_CSV = REPO_ROOT / "outputs" / "experiment_registry.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze BC-v2 status-flip patterns from paired-event rows.")
    parser.add_argument("--paired-events-csv", required=True)
    parser.add_argument("--engine-summary-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--registry-csv", default=str(REGISTRY_CSV))
    return parser.parse_args()


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def append_registry(summary_csv: Path, args: argparse.Namespace, status: str, notes: str) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "append_experiment_record.py"),
        "--csv",
        str(args.registry_csv),
        "--kind",
        "analysis",
        "--status",
        status,
        "--script",
        "scripts/analyze_bc_v2_flip_patterns.py",
        "--dataset",
        "DanceTrack",
        "--split",
        "val",
        "--tracker-family",
        "bc_v2_flip_patterns",
        "--variant",
        Path(args.out_dir).name,
        "--tag",
        Path(args.out_dir).name,
        "--run-root",
        str(Path(args.out_dir).resolve()),
        "--summary-csv",
        str(summary_csv.resolve()),
        "--notes",
        notes,
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=False)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = out_dir / "summary.csv"
    seq_csv = out_dir / "sequence_flip_summary.csv"
    bucket_csv = out_dir / "bucket_flip_summary.csv"
    event_csv = out_dir / "status_flip_events.csv"
    engine_csv = out_dir / "engine_runtime_summary.csv"

    write_rows(
        summary_csv,
        [
            "status",
            "status_changed_events",
            "positive_flips",
            "negative_flips",
            "positive_short_gap",
            "negative_short_gap",
            "net_correct_delta",
        ],
        [{"status": "running"}],
    )
    append_registry(summary_csv, args, "running", "analyzing bc-v2 flip patterns")

    paired_rows = list(csv.DictReader(Path(args.paired_events_csv).open("r", encoding="utf-8")))
    status_rows = [row for row in paired_rows if str(row.get("status_changed", "0")) == "1"]

    seq_counter: dict[str, Counter] = defaultdict(Counter)
    bucket_counter: dict[tuple[str, str], int] = defaultdict(int)
    event_rows: list[dict[str, object]] = []

    positive = 0
    negative = 0
    positive_short = 0
    negative_short = 0

    for row in status_rows:
        direction = "positive" if str(row.get("compare_correct_baseline_not", "0")) == "1" else "negative"
        gap = int(row["gap"])
        bucket = str(row["bucket"])
        seq = str(row["seq_name"])
        short_gap = bucket == "10-30"

        if direction == "positive":
            positive += 1
            if short_gap:
                positive_short += 1
        else:
            negative += 1
            if short_gap:
                negative_short += 1

        seq_counter[seq]["status_changed"] += 1
        seq_counter[seq][direction] += 1
        if short_gap:
            seq_counter[seq][f"{direction}_short_gap"] += 1
        bucket_counter[(bucket, direction)] += 1

        event_rows.append(
            {
                "seq_name": seq,
                "gt_id": row["gt_id"],
                "exit_frame": row["exit_frame"],
                "reentry_frame": row["reentry_frame"],
                "gap": gap,
                "bucket": bucket,
                "direction": direction,
                "baseline_status": row["baseline_status"],
                "compare_status": row["compare_status"],
                "baseline_tracker_tid": row["baseline_tracker_tid"],
                "compare_tracker_tid": row["compare_tracker_tid"],
                "baseline_oracle_tid": row["baseline_oracle_tid"],
                "compare_oracle_tid": row["compare_oracle_tid"],
            }
        )

    seq_rows = []
    for seq_name in sorted(seq_counter):
        ctr = seq_counter[seq_name]
        seq_rows.append(
            {
                "seq_name": seq_name,
                "status_changed": int(ctr.get("status_changed", 0)),
                "positive": int(ctr.get("positive", 0)),
                "negative": int(ctr.get("negative", 0)),
                "positive_short_gap": int(ctr.get("positive_short_gap", 0)),
                "negative_short_gap": int(ctr.get("negative_short_gap", 0)),
                "net_delta": int(ctr.get("positive", 0) - ctr.get("negative", 0)),
            }
        )

    bucket_rows = []
    for bucket in sorted({bucket for bucket, _ in bucket_counter}):
        bucket_rows.append(
            {
                "bucket": bucket,
                "positive": int(bucket_counter.get((bucket, "positive"), 0)),
                "negative": int(bucket_counter.get((bucket, "negative"), 0)),
                "net_delta": int(bucket_counter.get((bucket, "positive"), 0) - bucket_counter.get((bucket, "negative"), 0)),
            }
        )

    engine_rows = []
    for summary_path in sorted(Path(args.engine_summary_dir).glob("*_summary.csv")):
        row = next(csv.DictReader(summary_path.open("r", encoding="utf-8")))
        engine = ast.literal_eval(row.get("engine", "{}") or "{}")
        recent = engine.get("recent_rerank", {}) or {}
        commit = engine.get("commit", {}) or {}
        engine_rows.append(
            {
                "seq_name": row.get("seq_name", summary_path.stem.replace("_summary", "")),
                "proposals": int(commit.get("proposals", 0)),
                "commits": int(commit.get("commits", 0)),
                "recent_considered": int(recent.get("considered", 0)),
                "recent_swaps": int(recent.get("swaps", 0)),
            }
        )

    write_rows(
        summary_csv,
        [
            "status",
            "status_changed_events",
            "positive_flips",
            "negative_flips",
            "positive_short_gap",
            "negative_short_gap",
            "net_correct_delta",
        ],
        [
            {
                "status": "success",
                "status_changed_events": int(len(status_rows)),
                "positive_flips": int(positive),
                "negative_flips": int(negative),
                "positive_short_gap": int(positive_short),
                "negative_short_gap": int(negative_short),
                "net_correct_delta": int(positive - negative),
            }
        ],
    )
    write_rows(
        seq_csv,
        ["seq_name", "status_changed", "positive", "negative", "positive_short_gap", "negative_short_gap", "net_delta"],
        seq_rows,
    )
    write_rows(
        bucket_csv,
        ["bucket", "positive", "negative", "net_delta"],
        bucket_rows,
    )
    write_rows(
        event_csv,
        [
            "seq_name",
            "gt_id",
            "exit_frame",
            "reentry_frame",
            "gap",
            "bucket",
            "direction",
            "baseline_status",
            "compare_status",
            "baseline_tracker_tid",
            "compare_tracker_tid",
            "baseline_oracle_tid",
            "compare_oracle_tid",
        ],
        event_rows,
    )
    write_rows(
        engine_csv,
        ["seq_name", "proposals", "commits", "recent_considered", "recent_swaps"],
        engine_rows,
    )

    append_registry(summary_csv, args, "success", "bc-v2 flip pattern analysis complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
