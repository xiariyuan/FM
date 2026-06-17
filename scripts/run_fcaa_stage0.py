#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List


REPO_ROOT = Path("/gemini/code/FMtrack-main/FM-Track")


SUMMARY_FIELDS = [
    "step",
    "name",
    "status",
    "out_dir",
    "summary_csv",
    "log_path",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FCAA stage0: pair-bank + control/freq scorers.")
    parser.add_argument("--benchmark", default="MOT17")
    parser.add_argument("--split", default="train")
    parser.add_argument("--seq-names", nargs="*", default=["MOT17-02-FRCNN", "MOT17-13-FRCNN"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-root", default="")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--max-frames", type=int, default=0)
    return parser.parse_args()


def write_rows(path: Path, fieldnames: Iterable[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def run_step(cmd: List[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(cmd, cwd=REPO_ROOT, stdout=handle, stderr=subprocess.STDOUT)
    return int(process.returncode)


def append_registry(summary_csv: Path, run_root: Path, status: str, notes: str) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/append_experiment_record.py"),
        "--csv",
        str(REPO_ROOT / "outputs" / "experiment_registry.csv"),
        "--kind",
        "other",
        "--status",
        status,
        "--script",
        "scripts/run_fcaa_stage0.py",
        "--dataset",
        "MOT17",
        "--split",
        "stage0",
        "--tracker-family",
        "botsort_fcaa",
        "--variant",
        run_root.name,
        "--tag",
        "fcaa_stage0",
        "--run-root",
        str(run_root),
        "--summary-csv",
        str(summary_csv),
        "--notes",
        notes,
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=False)


def update_row(rows: List[Dict[str, object]], step: str, **kwargs: object) -> None:
    for row in rows:
        if str(row["step"]) == step:
            row.update(kwargs)
            return


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_root) if args.out_root else REPO_ROOT / "outputs" / f"fcaa_stage0_{timestamp}"
    out_root.mkdir(parents=True, exist_ok=True)
    queue_summary = out_root / "summary.csv"

    pairbank_dir = out_root / "pairbank"
    control_dir = out_root / "scorer_control"
    freq_dir = out_root / "scorer_freq"
    rows: List[Dict[str, object]] = [
        {
            "step": "pairbank",
            "name": "fcaa_pairbank",
            "status": "running",
            "out_dir": str(pairbank_dir),
            "summary_csv": str(pairbank_dir / "summary.csv"),
            "log_path": str(out_root / "logs" / "pairbank.log"),
            "notes": "",
        },
        {
            "step": "scorer_control",
            "name": "fcaa_control",
            "status": "pending",
            "out_dir": str(control_dir),
            "summary_csv": str(control_dir / "summary.csv"),
            "log_path": str(out_root / "logs" / "scorer_control.log"),
            "notes": "",
        },
        {
            "step": "scorer_freq",
            "name": "fcaa_freq",
            "status": "pending",
            "out_dir": str(freq_dir),
            "summary_csv": str(freq_dir / "summary.csv"),
            "log_path": str(out_root / "logs" / "scorer_freq.log"),
            "notes": "",
        },
    ]
    write_rows(queue_summary, SUMMARY_FIELDS, rows)
    append_registry(queue_summary, out_root, "running", "fcaa stage0 queue started")

    pairbank_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/build_fcaa_pairbank.py"),
        "--benchmark",
        args.benchmark,
        "--split",
        args.split,
        "--out-dir",
        str(pairbank_dir),
        "--dataset-name",
        f"fcaa_{args.benchmark.lower()}_{args.split}_pairbank",
        "--device",
        args.device,
        "--top-k",
        str(args.top_k),
    ]
    if args.seq_names:
        pairbank_cmd.extend(["--seq-names", *list(args.seq_names)])
    if int(args.max_frames) > 0:
        pairbank_cmd.extend(["--max-frames", str(int(args.max_frames))])
    pairbank_rc = run_step(pairbank_cmd, Path(rows[0]["log_path"]))
    if pairbank_rc != 0:
        update_row(rows, "pairbank", status="failed", notes=f"exit_code={pairbank_rc}")
        write_rows(queue_summary, SUMMARY_FIELDS, rows)
        append_registry(queue_summary, out_root, "failed", "fcaa stage0 failed in pairbank step")
        return
    update_row(rows, "pairbank", status="success")
    update_row(rows, "scorer_control", status="running")
    write_rows(queue_summary, SUMMARY_FIELDS, rows)

    pairbank_jsonl = pairbank_dir / "pairbank.jsonl"
    control_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/train_fcaa_pair_scorer.py"),
        "--pairbank-jsonl",
        str(pairbank_jsonl),
        "--out-dir",
        str(control_dir),
        "--run-name",
        "fcaa_control",
        "--mode",
        "control",
        "--train-ratio",
        str(args.train_ratio),
    ]
    control_rc = run_step(control_cmd, Path(rows[1]["log_path"]))
    if control_rc != 0:
        update_row(rows, "scorer_control", status="failed", notes=f"exit_code={control_rc}")
        write_rows(queue_summary, SUMMARY_FIELDS, rows)
        append_registry(queue_summary, out_root, "failed", "fcaa stage0 failed in control scorer step")
        return
    update_row(rows, "scorer_control", status="success")
    update_row(rows, "scorer_freq", status="running")
    write_rows(queue_summary, SUMMARY_FIELDS, rows)

    freq_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/train_fcaa_pair_scorer.py"),
        "--pairbank-jsonl",
        str(pairbank_jsonl),
        "--out-dir",
        str(freq_dir),
        "--run-name",
        "fcaa_freq",
        "--mode",
        "freq",
        "--train-ratio",
        str(args.train_ratio),
    ]
    freq_rc = run_step(freq_cmd, Path(rows[2]["log_path"]))
    if freq_rc != 0:
        update_row(rows, "scorer_freq", status="failed", notes=f"exit_code={freq_rc}")
        write_rows(queue_summary, SUMMARY_FIELDS, rows)
        append_registry(queue_summary, out_root, "failed", "fcaa stage0 failed in freq scorer step")
        return
    update_row(rows, "scorer_freq", status="success")
    write_rows(queue_summary, SUMMARY_FIELDS, rows)
    append_registry(queue_summary, out_root, "success", "fcaa stage0 finished")


if __name__ == "__main__":
    main()
