#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

from queue_deep_ocsort_preassoc_force_rewrite_next2h import QUEUE_FIELDS, REPO_ROOT, now_iso, timestamp_tag, write_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Watch the main two-hour queue and launch a tail-fill queue automatically if it ends early without 3-sequence confirmation."
    )
    parser.add_argument("--primary-summary-csv", required=True)
    parser.add_argument("--primary-run-root", required=True)
    parser.add_argument("--out-root", default="")
    parser.add_argument(
        "--tail-script",
        default=str(REPO_ROOT / "scripts" / "queue_deep_ocsort_preassoc_force_rewrite_tail_fill.py"),
    )
    parser.add_argument("--poll-sec", type=int, default=30)
    return parser.parse_args()


def read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def summary_terminal(rows: List[Dict[str, str]]) -> bool:
    if not rows:
        return False
    return all(str(row.get("status", "")).strip() not in {"", "pending", "running"} for row in rows)


def main() -> int:
    args = parse_args()
    primary_summary_csv = Path(args.primary_summary_csv).expanduser().resolve()
    primary_run_root = Path(args.primary_run_root).expanduser().resolve()
    run_root = (
        Path(args.out_root).expanduser().resolve()
        if args.out_root
        else (REPO_ROOT / "outputs" / f"deep_ocsort_preassoc_force_rewrite_watch_{timestamp_tag()}").resolve()
    )
    tail_script = Path(args.tail_script).expanduser().resolve()

    summary_csv = run_root / "summary.csv"
    logs_dir = run_root / "logs"
    queue_rows: List[Dict[str, object]] = [
        {
            "step": "wait_primary",
            "name": f"{run_root.name}_wait_primary",
            "status": "running",
            "out_dir": str(primary_run_root),
            "summary_csv": str(primary_summary_csv),
            "log_path": str((logs_dir / "wait_primary.log").resolve()),
            "started_at": now_iso(),
            "finished_at": "",
            "notes": "waiting for primary two-hour queue to reach a terminal state",
        },
        {
            "step": "launch_tail_fill",
            "name": f"{run_root.name}_launch_tail_fill",
            "status": "pending",
            "out_dir": "",
            "summary_csv": "",
            "log_path": str((logs_dir / "launch_tail_fill.log").resolve()),
            "started_at": "",
            "finished_at": "",
            "notes": "launch a tail-fill queue only if primary queue ends without successful 3-sequence confirmation",
        },
    ]
    write_rows(summary_csv, QUEUE_FIELDS, queue_rows)

    logs_dir.mkdir(parents=True, exist_ok=True)
    confirm_status = "missing"
    while True:
        rows = read_rows(primary_summary_csv)
        if summary_terminal(rows):
            confirm_row = next((row for row in rows if str(row.get("step", "")) == "dance3_best_confirm"), None)
            if confirm_row is not None:
                confirm_status = str(confirm_row.get("status", "") or "missing")
            queue_rows[0]["status"] = "success"
            queue_rows[0]["finished_at"] = now_iso()
            queue_rows[0]["notes"] = f"primary queue finished with dance3_best_confirm status={confirm_status}"
            write_rows(summary_csv, QUEUE_FIELDS, queue_rows)
            break
        time.sleep(float(args.poll_sec))

    if confirm_status == "success":
        queue_rows[1]["status"] = "cancelled"
        queue_rows[1]["finished_at"] = now_iso()
        queue_rows[1]["notes"] = "primary queue already launched and finished 3-sequence confirmation; no tail fill needed"
        write_rows(summary_csv, QUEUE_FIELDS, queue_rows)
        return 0

    tail_out_root = (REPO_ROOT / "outputs" / f"deep_ocsort_preassoc_force_rewrite_tailfill_{timestamp_tag()}").resolve()
    launcher_log = (tail_out_root / "launcher.log").resolve()
    tail_summary_csv = (tail_out_root / "summary.csv").resolve()
    tail_out_root.mkdir(parents=True, exist_ok=True)

    queue_rows[1]["status"] = "running"
    queue_rows[1]["out_dir"] = str(tail_out_root)
    queue_rows[1]["summary_csv"] = str(tail_summary_csv)
    queue_rows[1]["started_at"] = now_iso()
    write_rows(summary_csv, QUEUE_FIELDS, queue_rows)

    with launcher_log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            [sys.executable, str(tail_script), "--out-root", str(tail_out_root)],
            cwd=REPO_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    time.sleep(4.0)
    if process.poll() is not None or not tail_summary_csv.is_file():
        queue_rows[1]["status"] = "failed"
        queue_rows[1]["finished_at"] = now_iso()
        queue_rows[1]["notes"] = f"tail-fill launch failed to stay alive; launcher_log={launcher_log}"
        write_rows(summary_csv, QUEUE_FIELDS, queue_rows)
        return 1

    queue_rows[1]["status"] = "success"
    queue_rows[1]["finished_at"] = now_iso()
    queue_rows[1]["notes"] = f"tail-fill queue launched successfully with pid={process.pid}"
    write_rows(summary_csv, QUEUE_FIELDS, queue_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
