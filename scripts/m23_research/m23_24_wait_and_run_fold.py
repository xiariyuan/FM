#!/usr/bin/env python3
from __future__ import annotations

"""Wait for one M23-24 ReID fold and run its frozen downstream pipeline."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


SEQUENCES = ("MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05")


def run(command: list[str]) -> None:
    print("[M23-24 orchestrator] run:", " ".join(command), flush=True)
    completed = subprocess.run(command)
    if completed.returncode:
        raise RuntimeError(f"command failed with exit code {completed.returncode}: {command}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--held-seq", required=True, choices=SEQUENCES)
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--phase-root", required=True)
    parser.add_argument("--graph-root", required=True)
    parser.add_argument("--transaction-root", required=True)
    parser.add_argument("--timeout-hours", type=float, default=8.0)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()

    train_dir = Path(args.train_dir)
    summary_path = train_dir / "summary.json"
    checkpoint = train_dir / "model_best.pth"
    deadline = time.time() + 3600.0 * args.timeout_hours
    while not summary_path.is_file():
        if time.time() >= deadline:
            raise TimeoutError(f"training summary did not appear: {summary_path}")
        print(f"[M23-24 orchestrator] waiting for {summary_path}", flush=True)
        time.sleep(max(5.0, min(args.poll_seconds, 60.0)))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("held_sequence") != args.held_seq or not checkpoint.is_file():
        raise RuntimeError(f"invalid training completion under {train_dir}")

    run([
        sys.executable,
        "-u",
        "scripts/m23_research/m23_24_reembed_phase0.py",
        "--checkpoint",
        str(checkpoint),
        "--held-seq",
        args.held_seq,
        "--output-root",
        args.phase_root,
        "--batch-size",
        "96",
    ])
    run([
        sys.executable,
        "-u",
        "scripts/m23_research/m23_24_build_finetuned_micrograph.py",
        "--held-seq",
        args.held_seq,
        "--phase-root",
        args.phase_root,
        "--out-root",
        args.graph_root,
    ])
    run([
        sys.executable,
        "-u",
        "scripts/m23_research/m23_24_train_transaction_fold.py",
        "--held-seq",
        args.held_seq,
        "--graph-root",
        args.graph_root,
        "--output-root",
        args.transaction_root,
    ])
    result = {
        "held_sequence": args.held_seq,
        "training_summary": str(summary_path),
        "checkpoint": str(checkpoint),
        "phase_root": args.phase_root,
        "graph_root": args.graph_root,
        "transaction_root": args.transaction_root,
        "status": "completed",
    }
    output = Path(args.transaction_root) / "orchestrator_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
