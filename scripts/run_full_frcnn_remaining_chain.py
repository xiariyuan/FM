#!/usr/bin/env python3
from __future__ import annotations

import csv
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path("/gemini/code/FMtrack-main/FM-Track")
PYTHON_BIN = Path("/root/miniconda3/bin/python")
GENERIC_RUNNER = REPO_ROOT / "scripts" / "run_local_conflict_graph_commitmatches_hardtrigger_oracle_generic.sh"
PLAN_SCRIPT = REPO_ROOT / "scripts" / "upsert_experiment_plan.py"
ORACLE_JSONL = REPO_ROOT / "outputs" / "competition_assoc_base_reid_da_proxy0213_hybriddumpfix" / "labeled_replay_top8.groups.jsonl"
TARGET_JOB_KEYS = [
    "06_full_frcnn_topk8_md2_mm3",
    "07_full_frcnn_topk8_md3_mm2",
    "08_full_frcnn_topk8_md4_mm2",
]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = [dict(row) for row in reader]
        return rows, list(reader.fieldnames or [])


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def get_queue_row(summary_csv: Path, job_key: str) -> dict[str, str]:
    rows, _ = read_csv(summary_csv)
    for row in rows:
        if row.get("job_key") == job_key:
            return row
    raise KeyError(f"job_key not found: {job_key}")


def read_run_terminal_status(run_root: Path) -> str:
    run_summary = run_root / "summary.csv"
    if not run_summary.is_file():
        return ""
    rows, _ = read_csv(run_summary)
    if not rows:
        return ""
    return rows[-1].get("status", "")


def read_run_metrics(run_root: Path) -> dict[str, str]:
    run_summary = run_root / "summary.csv"
    if not run_summary.is_file():
        return {}
    rows, _ = read_csv(run_summary)
    if not rows:
        return {}
    row = rows[-1]
    return {key: row.get(key, "") for key in ("HOTA", "AssA", "IDF1", "MOTA", "IDSW")}


def has_live_process(run_root: Path) -> bool:
    cmd = ["ps", "-eo", "pid,args"]
    output = subprocess.check_output(cmd, text=True)
    run_root_s = str(run_root)
    needles = (
        "run_local_conflict_graph_commitmatches_hardtrigger_oracle_generic.sh",
        "run_bytetrack_profile.py",
        "submit_bytetrack.py",
    )
    for line in output.splitlines():
        if run_root_s not in line:
            continue
        if any(needle in line for needle in needles):
            return True
    return False


def update_queue_summary(summary_csv: Path, job_key: str, status: str, started_at: str = "", finished_at: str = "") -> None:
    rows, fieldnames = read_csv(summary_csv)
    for row in rows:
        if row.get("job_key") != job_key:
            continue
        row["status"] = status
        if started_at:
            row["started_at"] = started_at
        if finished_at:
            row["finished_at"] = finished_at
        metrics = read_run_metrics(Path(row["out_dir"]))
        for key, value in metrics.items():
            if value != "":
                row[key] = value
    write_csv(summary_csv, rows, fieldnames)


def upsert_plan(queue_name: str, queue_root: Path, row: dict[str, str], status: str) -> None:
    plan_key = row["plan_key"]
    cmd = [
        str(PYTHON_BIN),
        str(PLAN_SCRIPT),
        "--csv",
        str(REPO_ROOT / "outputs" / "experiment_plan.csv"),
        "--key",
        plan_key,
        "--status",
        status,
        "--kind",
        "eval",
        "--script",
        "scripts/run_local_conflict_graph_commitmatches_hardtrigger_oracle_generic.sh",
        "--dataset",
        "MOT17",
        "--split",
        row["phase"],
        "--tracker-family",
        "ByteTrack",
        "--variant",
        row["run_name"],
        "--tag",
        "local_conflict_graph_mainline",
        "--run-root",
        row["out_dir"],
        "--summary-csv",
        str(Path(row["out_dir"]) / "summary.csv"),
        "--log-path",
        str(Path(row["out_dir"]) / "run.log"),
        "--notes",
        row["notes"],
        "--extra",
        f"queue_name={queue_name}",
        f"queue_root={queue_root}",
        f"oracle_group_jsonl={ORACLE_JSONL}",
        f"graph_topk={row['graph_topk']}",
        f"graph_min_detections={row['graph_min_detections']}",
        f"graph_min_committed_matches={row['graph_min_committed_matches']}",
        f"detector_filter={row['detector_filter']}",
        f"val_sequences={row['val_sequences']}",
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def launch_job(queue_log: Path, row: dict[str, str]) -> int:
    cmd = [
        "bash",
        str(GENERIC_RUNNER),
        row["out_dir"],
        row["run_name"],
        str(ORACLE_JSONL),
        row["graph_topk"],
        row["graph_min_detections"],
        row["graph_min_committed_matches"],
        row["detector_filter"],
        row["val_sequences"],
        row["notes"],
    ]
    with queue_log.open("a", encoding="utf-8") as log_fp:
        proc = subprocess.run(cmd, stdout=log_fp, stderr=subprocess.STDOUT)
        return proc.returncode


def poll_until_finished(run_root: Path, poll_sec: int = 20) -> str:
    while True:
        terminal_status = read_run_terminal_status(run_root)
        if terminal_status in {"ok", "failed"}:
            return terminal_status
        if not has_live_process(run_root):
            return terminal_status
        time.sleep(poll_sec)


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("usage: run_full_frcnn_remaining_chain.py <queue_root> [queue_name]")
    queue_root = Path(sys.argv[1]).resolve()
    queue_name = sys.argv[2] if len(sys.argv) >= 3 else queue_root.name
    summary_csv = queue_root / "summary.csv"
    current_job_txt = queue_root / "current_job.txt"
    queue_log = queue_root / "queue.log"

    if not summary_csv.is_file():
        raise SystemExit(f"missing queue summary: {summary_csv}")

    queue_log.parent.mkdir(parents=True, exist_ok=True)
    queue_log.touch(exist_ok=True)

    for job_key in TARGET_JOB_KEYS:
        row = get_queue_row(summary_csv, job_key)
        run_root = Path(row["out_dir"])
        current_job_txt.write_text(job_key + "\n", encoding="utf-8")

        terminal_status = read_run_terminal_status(run_root)
        if terminal_status == "ok":
            update_queue_summary(summary_csv, job_key, "completed")
            upsert_plan(queue_name, queue_root, row, "completed")
            continue

        if has_live_process(run_root):
            update_queue_summary(summary_csv, job_key, "running")
            upsert_plan(queue_name, queue_root, row, "running")
            terminal_status = poll_until_finished(run_root)
            row = get_queue_row(summary_csv, job_key)
            if terminal_status == "ok":
                update_queue_summary(summary_csv, job_key, "completed", finished_at=now_iso())
                upsert_plan(queue_name, queue_root, row, "completed")
                continue

        update_queue_summary(summary_csv, job_key, "running", started_at=now_iso())
        row = get_queue_row(summary_csv, job_key)
        upsert_plan(queue_name, queue_root, row, "running")
        rc = launch_job(queue_log, row)
        terminal_status = read_run_terminal_status(run_root)
        row = get_queue_row(summary_csv, job_key)
        if rc == 0 and terminal_status == "ok":
            update_queue_summary(summary_csv, job_key, "completed", finished_at=now_iso())
            upsert_plan(queue_name, queue_root, row, "completed")
        else:
            update_queue_summary(summary_csv, job_key, "failed", finished_at=now_iso())
            upsert_plan(queue_name, queue_root, row, "failed")
            current_job_txt.write_text(job_key + "\n", encoding="utf-8")
            return 1

    current_job_txt.write_text("", encoding="utf-8")
    with queue_log.open("a", encoding="utf-8") as f:
        f.write(f"[full-frcnn-chain] completed at {now_iso()}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
