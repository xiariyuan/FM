#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path("/gemini/code/FMtrack-main/FM-Track")
REGISTRY_CSV = REPO_ROOT / "outputs" / "experiment_registry.csv"
DEFAULT_DATA_ROOT = Path("/gemini/code/datasets/DanceTrack/extracted")
RUN_SCRIPT = REPO_ROOT / "scripts" / "run_botsort_dancetrack_val.sh"
GT_JSON = DEFAULT_DATA_ROOT / "annotations" / "val.json"

SUMMARY_FIELDS = [
    "seq_id",
    "seq_name",
    "status",
    "started_at",
    "finished_at",
    "result_file",
    "reentry_summary_csv",
    "notes",
]


def iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def append_registry(summary_csv: Path, status: str, tag: str, notes: str) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "append_experiment_record.py"),
        "--csv",
        str(REGISTRY_CSV),
        "--kind",
        "eval",
        "--status",
        status,
        "--script",
        "scripts/run_botsort_dancetrack_reentry_engine_queue.py",
        "--dataset",
        "DanceTrack",
        "--split",
        "val",
        "--tracker-family",
        "BoT-SORT",
        "--variant",
        "reentry_engine_queue",
        "--tag",
        tag,
        "--run-root",
        str(summary_csv.parent.resolve()),
        "--summary-csv",
        str(summary_csv.resolve()),
        "--notes",
        notes,
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Queue per-sequence DanceTrack re-entry engine runs.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--seq-ids", nargs="+", type=int, required=True)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument(
        "--track-arg",
        action="append",
        default=[],
        help="Extra argument token appended to tools/track.py after the default re-entry args. Repeat once per token.",
    )
    parser.add_argument(
        "--disable-engine",
        action="store_true",
        help="Do not pass --reentry-engine-enable. Useful for memory-only control runs.",
    )
    parser.add_argument(
        "--disable-memory",
        action="store_true",
        help="Do not pass --reentry-memory-enable or its override args. Useful for current-code baseline runs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    summary_csv = out_dir / "summary.csv"
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    extra_args_note = " ".join(str(token) for token in args.track_arg).strip()

    rows: list[dict[str, str]] = []
    for seq_id in args.seq_ids:
        seq_name = f"dancetrack{seq_id:04d}"
        rows.append(
            {
                "seq_id": str(seq_id),
                "seq_name": seq_name,
                "status": "pending",
                "started_at": "",
                "finished_at": "",
                "result_file": str(
                    REPO_ROOT
                    / "external"
                    / "BoT-SORT-main"
                    / "YOLOX_outputs"
                    / args.exp_name
                    / "track_results"
                    / f"{seq_name}.txt"
                ),
                "reentry_summary_csv": str(
                    REPO_ROOT
                    / "external"
                    / "BoT-SORT-main"
                    / "YOLOX_outputs"
                    / args.exp_name
                    / "reentry_analysis"
                    / f"{seq_name}_summary.csv"
                ),
                "notes": "",
            }
        )
    write_rows(summary_csv, SUMMARY_FIELDS, rows)
    start_note = "starting per-sequence DanceTrack reentry queue"
    if args.disable_engine:
        start_note += " engine=off"
    if args.disable_memory:
        start_note += " memory=off"
    if extra_args_note:
        start_note += f" extra_args={extra_args_note}"
    append_registry(summary_csv, "running", Path(args.out_dir).name, start_note)

    for row in rows:
        row["status"] = "running"
        row["started_at"] = iso_now()
        write_rows(summary_csv, SUMMARY_FIELDS, rows)

        seq_id = row["seq_id"]
        seq_name = row["seq_name"]
        log_path = log_dir / f"{seq_name}.log"
        cmd = [
            "env",
            f"DATA_ROOT={Path(args.data_root).resolve()}",
            "PYTHONUNBUFFERED=1",
            "bash",
            str(RUN_SCRIPT),
            "base",
            "--experiment-name",
            args.exp_name,
            "--seq-ids",
            str(seq_id),
        ]
        if not args.disable_memory:
            cmd.extend(
                [
                    "--reentry-memory-enable",
                    "--reentry-memory-max-gap",
                    "60",
                    "--reentry-memory-max-size",
                    "256",
                    "--reentry-memory-min-similarity",
                    "0.60",
                    "--reentry-memory-confirm-streak",
                    "2",
                    "--reentry-memory-confirm-gap",
                    "2",
                    "--reentry-memory-confirm-min-similarity",
                    "0.65",
                    "--reentry-memory-min-det-score",
                    "0.10",
                    "--reentry-memory-appearance-weight",
                    "0.55",
                    "--reentry-memory-iou-weight",
                    "0.25",
                    "--reentry-memory-score-weight",
                    "0.10",
                    "--reentry-memory-gap-weight",
                    "0.10",
                ]
            )
        if not args.disable_engine:
            cmd.extend(
                [
                    "--reentry-engine-hilbert-order",
                    "8",
                    "--reentry-engine-bf-threshold",
                    "50",
                    "--reentry-engine-spatial-radius",
                    "2",
                    "--reentry-engine-max-spatial-radius",
                    "4",
                ]
            )
        if not args.disable_engine:
            cmd.append("--reentry-engine-enable")
        cmd.extend(str(token) for token in args.track_arg)
        with log_path.open("w", encoding="utf-8") as handle:
            proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=handle, stderr=subprocess.STDOUT)

        row["finished_at"] = iso_now()
        if proc.returncode == 0:
            row["status"] = "success"
            row["notes"] = str(log_path)
        else:
            row["status"] = "failed"
            row["notes"] = f"{log_path} rc={proc.returncode}"
            write_rows(summary_csv, SUMMARY_FIELDS, rows)
            fail_note = f"queue failed on {seq_name}"
            if args.disable_engine:
                fail_note += " engine=off"
            if args.disable_memory:
                fail_note += " memory=off"
            if extra_args_note:
                fail_note += f" extra_args={extra_args_note}"
            append_registry(summary_csv, "failed", Path(args.out_dir).name, fail_note)
            return proc.returncode
        write_rows(summary_csv, SUMMARY_FIELDS, rows)

    success_note = "per-sequence DanceTrack reentry queue complete"
    if args.disable_engine:
        success_note += " engine=off"
    if args.disable_memory:
        success_note += " memory=off"
    if extra_args_note:
        success_note += f" extra_args={extra_args_note}"
    append_registry(summary_csv, "success", Path(args.out_dir).name, success_note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
