#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


REPO_ROOT = Path("/gemini/code/FMtrack-main/FM-Track")
REGISTRY_CSV = REPO_ROOT / "outputs" / "experiment_registry.csv"

SUMMARY_FIELDS = [
    "status",
    "error",
    "event_count",
    "changed_event_count",
    "status_changed_event_count",
    "same_status_diff_tid_event_count",
    "baseline_correct_to_compare_not",
    "compare_correct_to_baseline_not",
]

PAIR_FIELDS = [
    "seq_name",
    "gt_id",
    "exit_frame",
    "reentry_frame",
    "gap",
    "bucket",
    "baseline_status",
    "baseline_status_group",
    "baseline_recoverable",
    "baseline_tracker_tid",
    "baseline_oracle_tid",
    "compare_status",
    "compare_status_group",
    "compare_recoverable",
    "compare_tracker_tid",
    "compare_oracle_tid",
    "status_changed",
    "tracker_tid_changed",
    "oracle_tid_changed",
    "baseline_correct_compare_not",
    "compare_correct_baseline_not",
]

TRANSITION_FIELDS = [
    "from_status",
    "to_status",
    "count",
]

BUCKET_TRANSITION_FIELDS = [
    "bucket",
    "from_status_group",
    "to_status_group",
    "count",
]

SEQUENCE_FIELDS = [
    "seq_name",
    "event_count",
    "changed_event_count",
    "status_changed_event_count",
    "baseline_correct",
    "compare_correct",
    "delta_correct",
    "baseline_wrong",
    "compare_wrong",
    "delta_wrong",
    "baseline_missed",
    "compare_missed",
    "delta_missed",
]

ENGINE_SEQ_FIELDS = [
    "seq_name",
    "engine_frames",
    "engine_queries",
    "engine_matches_above_threshold",
    "engine_proposals",
    "engine_updates",
    "engine_resets",
    "engine_confirmations",
    "engine_commits",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze paired event-level flips between two re-entry evaluations.")
    parser.add_argument("--event-details-csv", required=True)
    parser.add_argument("--baseline-name", required=True)
    parser.add_argument("--compare-name", required=True)
    parser.add_argument("--engine-reentry-analysis-dir", default="", help="Optional per-sequence engine reentry_analysis directory.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--registry-csv", default=str(REGISTRY_CSV))
    return parser.parse_args()


def write_rows(path: Path, fieldnames: Iterable[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_single_row_csv(path: Path, fieldnames: Iterable[str], row: Dict[str, object]) -> None:
    write_rows(path, fieldnames, [row])


def append_registry(args: argparse.Namespace, summary_csv: Path, status: str, notes: str) -> None:
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
        "scripts/analyze_reentry_paired_flips.py",
        "--dataset",
        "DanceTrack",
        "--split",
        "val",
        "--tracker-family",
        "reentry_paired_flips",
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


def load_engine_seq_rows(engine_dir: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if not engine_dir.is_dir():
        return rows
    for summary_csv in sorted(engine_dir.glob("*_summary.csv")):
        with summary_csv.open("r", encoding="utf-8") as handle:
            row = next(csv.DictReader(handle))
        engine_blob = row.get("engine", "")
        engine = ast.literal_eval(engine_blob) if engine_blob else {}
        commit = engine.get("commit", {})
        metrics = engine.get("metrics", {})
        rows.append(
            {
                "seq_name": row.get("seq_name", summary_csv.name.replace("_summary.csv", "")),
                "engine_frames": int(commit.get("frames", 0)),
                "engine_queries": int(metrics.get("total_queries", 0)),
                "engine_matches_above_threshold": int(metrics.get("total_matches_above_threshold", 0)),
                "engine_proposals": int(commit.get("proposals", 0)),
                "engine_updates": int(commit.get("updates", 0)),
                "engine_resets": int(commit.get("resets", 0)),
                "engine_confirmations": int(commit.get("confirmations", 0)),
                "engine_commits": int(commit.get("commits", 0)),
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "summary.csv"

    write_single_row_csv(
        summary_csv,
        SUMMARY_FIELDS,
        {
            "status": "running",
            "error": "",
            "event_count": 0,
            "changed_event_count": 0,
            "status_changed_event_count": 0,
            "same_status_diff_tid_event_count": 0,
            "baseline_correct_to_compare_not": 0,
            "compare_correct_to_baseline_not": 0,
        },
    )
    append_registry(summary_csv=summary_csv, args=args, status="running", notes=f"{args.baseline_name} vs {args.compare_name}")

    try:
        rows = list(csv.DictReader(Path(args.event_details_csv).expanduser().resolve().open("r", encoding="utf-8")))
        by_key: Dict[Tuple[str, str, str, str], Dict[str, dict]] = defaultdict(dict)
        for row in rows:
            key = (row["seq_name"], row["gt_id"], row["exit_frame"], row["reentry_frame"])
            by_key[key][row["tracker_name"]] = row

        pair_rows: List[Dict[str, object]] = []
        transition_counter: Counter = Counter()
        bucket_transition_counter: Counter = Counter()
        seq_stats: Dict[str, Counter] = defaultdict(Counter)

        changed_event_count = 0
        status_changed_event_count = 0
        same_status_diff_tid_event_count = 0
        baseline_correct_to_compare_not = 0
        compare_correct_to_baseline_not = 0

        for key in sorted(by_key):
            pair = by_key[key]
            base = pair.get(args.baseline_name)
            comp = pair.get(args.compare_name)
            if not base or not comp:
                continue

            base_status = base["status"]
            comp_status = comp["status"]
            base_group = base["status_group"]
            comp_group = comp["status_group"]
            base_tid = base["tracker_track_id"]
            comp_tid = comp["tracker_track_id"]
            base_oracle = base["oracle_track_id"]
            comp_oracle = comp["oracle_track_id"]

            status_changed = int(base_status != comp_status)
            tracker_tid_changed = int(base_tid != comp_tid)
            oracle_tid_changed = int(base_oracle != comp_oracle)
            changed = int(status_changed or tracker_tid_changed or oracle_tid_changed)

            if changed:
                changed_event_count += 1
            if status_changed:
                status_changed_event_count += 1
            elif tracker_tid_changed:
                same_status_diff_tid_event_count += 1

            base_correct = int(base_status == "correct_recovery")
            comp_correct = int(comp_status == "correct_recovery")
            if base_correct and not comp_correct:
                baseline_correct_to_compare_not += 1
            if comp_correct and not base_correct:
                compare_correct_to_baseline_not += 1

            transition_counter[(base_status, comp_status)] += 1
            bucket_transition_counter[(base["bucket"], base_group, comp_group)] += 1

            seq_name = base["seq_name"]
            seq_stats[seq_name]["event_count"] += 1
            seq_stats[seq_name]["changed_event_count"] += changed
            seq_stats[seq_name]["status_changed_event_count"] += status_changed
            seq_stats[seq_name]["baseline_correct"] += base_correct
            seq_stats[seq_name]["compare_correct"] += comp_correct
            seq_stats[seq_name]["baseline_wrong"] += int(base_group == "wrong_recovery")
            seq_stats[seq_name]["compare_wrong"] += int(comp_group == "wrong_recovery")
            seq_stats[seq_name]["baseline_missed"] += int(base_group == "missed_recovery")
            seq_stats[seq_name]["compare_missed"] += int(comp_group == "missed_recovery")

            pair_rows.append(
                {
                    "seq_name": base["seq_name"],
                    "gt_id": base["gt_id"],
                    "exit_frame": base["exit_frame"],
                    "reentry_frame": base["reentry_frame"],
                    "gap": base["gap"],
                    "bucket": base["bucket"],
                    "baseline_status": base_status,
                    "baseline_status_group": base_group,
                    "baseline_recoverable": base["recoverable"],
                    "baseline_tracker_tid": base_tid,
                    "baseline_oracle_tid": base_oracle,
                    "compare_status": comp_status,
                    "compare_status_group": comp_group,
                    "compare_recoverable": comp["recoverable"],
                    "compare_tracker_tid": comp_tid,
                    "compare_oracle_tid": comp_oracle,
                    "status_changed": status_changed,
                    "tracker_tid_changed": tracker_tid_changed,
                    "oracle_tid_changed": oracle_tid_changed,
                    "baseline_correct_compare_not": int(base_correct and not comp_correct),
                    "compare_correct_baseline_not": int(comp_correct and not base_correct),
                }
            )

        transition_rows = [
            {"from_status": src, "to_status": dst, "count": count}
            for (src, dst), count in sorted(transition_counter.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))
        ]
        bucket_transition_rows = [
            {"bucket": bucket, "from_status_group": src, "to_status_group": dst, "count": count}
            for (bucket, src, dst), count in sorted(
                bucket_transition_counter.items(),
                key=lambda item: (item[0][0], -item[1], item[0][1], item[0][2]),
            )
        ]
        sequence_rows = []
        for seq_name, stats in sorted(seq_stats.items()):
            sequence_rows.append(
                {
                    "seq_name": seq_name,
                    "event_count": int(stats["event_count"]),
                    "changed_event_count": int(stats["changed_event_count"]),
                    "status_changed_event_count": int(stats["status_changed_event_count"]),
                    "baseline_correct": int(stats["baseline_correct"]),
                    "compare_correct": int(stats["compare_correct"]),
                    "delta_correct": int(stats["compare_correct"] - stats["baseline_correct"]),
                    "baseline_wrong": int(stats["baseline_wrong"]),
                    "compare_wrong": int(stats["compare_wrong"]),
                    "delta_wrong": int(stats["compare_wrong"] - stats["baseline_wrong"]),
                    "baseline_missed": int(stats["baseline_missed"]),
                    "compare_missed": int(stats["compare_missed"]),
                    "delta_missed": int(stats["compare_missed"] - stats["baseline_missed"]),
                }
            )

        engine_seq_rows = load_engine_seq_rows(Path(args.engine_reentry_analysis_dir).expanduser().resolve()) if args.engine_reentry_analysis_dir else []

        write_rows(out_dir / "paired_event_rows.csv", PAIR_FIELDS, pair_rows)
        write_rows(out_dir / "status_transition_summary.csv", TRANSITION_FIELDS, transition_rows)
        write_rows(out_dir / "bucket_transition_summary.csv", BUCKET_TRANSITION_FIELDS, bucket_transition_rows)
        write_rows(out_dir / "sequence_gain_loss_summary.csv", SEQUENCE_FIELDS, sequence_rows)
        if engine_seq_rows:
            write_rows(out_dir / "engine_sequence_summary.csv", ENGINE_SEQ_FIELDS, engine_seq_rows)
        write_single_row_csv(
            summary_csv,
            SUMMARY_FIELDS,
            {
                "status": "success",
                "error": "",
                "event_count": len(pair_rows),
                "changed_event_count": changed_event_count,
                "status_changed_event_count": status_changed_event_count,
                "same_status_diff_tid_event_count": same_status_diff_tid_event_count,
                "baseline_correct_to_compare_not": baseline_correct_to_compare_not,
                "compare_correct_to_baseline_not": compare_correct_to_baseline_not,
            },
        )
        append_registry(summary_csv=summary_csv, args=args, status="success", notes=f"{args.baseline_name} vs {args.compare_name} complete")
        return 0
    except Exception as exc:
        write_single_row_csv(
            summary_csv,
            SUMMARY_FIELDS,
            {
                "status": "failed",
                "error": str(exc),
                "event_count": 0,
                "changed_event_count": 0,
                "status_changed_event_count": 0,
                "same_status_diff_tid_event_count": 0,
                "baseline_correct_to_compare_not": 0,
                "compare_correct_to_baseline_not": 0,
            },
        )
        append_registry(summary_csv=summary_csv, args=args, status="failed", notes=f"{args.baseline_name} vs {args.compare_name} failed: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
