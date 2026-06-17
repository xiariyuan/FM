#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

from queue_deep_ocsort_preassoc_force_rewrite_next2h import (
    QUEUE_FIELDS,
    REPO_ROOT,
    REGISTRY_CSV,
    ensure_child_success,
    now_iso,
    read_rows,
    run_step,
    timestamp_tag,
    update_row,
    write_rows,
)


DEFAULT_WAIT_RUN_ROOT = REPO_ROOT / "outputs" / "deep_ocsort_preassoc_force_recovery_anchor_collect_dance8_20260409_1"
DEFAULT_WAIT_SUMMARY_CSV = DEFAULT_WAIT_RUN_ROOT / "summary.csv"
DEFAULT_BASE_RUN_ROOTS = [
    REPO_ROOT / "outputs" / "deep_ocsort_preassoc_force_recovery_anchor_seq0090_debug" / "runs" / "anchor_gap3_safe015",
    REPO_ROOT / "outputs" / "deep_ocsort_preassoc_force_recovery_anchor_seq0090_debug" / "runs" / "anchor_gap2_safe015",
    REPO_ROOT / "outputs" / "deep_ocsort_preassoc_force_recovery_anchor_seq0090_debug" / "runs" / "anchor_gap3_safe020",
    REPO_ROOT / "outputs" / "deep_ocsort_preassoc_force_recovery_anchor_confirm_dance3_debug" / "runs" / "anchor_gap2_safe015_dance3",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wait for the recovery-anchor collection queue, then build a larger dataset, sequence split, and first anchor-gate training run."
    )
    parser.add_argument("--out-root", default="")
    parser.add_argument("--wait-summary-csv", default=str(DEFAULT_WAIT_SUMMARY_CSV))
    parser.add_argument("--wait-run-root", default=str(DEFAULT_WAIT_RUN_ROOT))
    parser.add_argument(
        "--wait-process-pattern",
        default=str(DEFAULT_WAIT_RUN_ROOT.name),
        help="substring expected in the live process table while the waited queue is still running",
    )
    parser.add_argument(
        "--base-run-roots",
        nargs="*",
        default=[str(path) for path in DEFAULT_BASE_RUN_ROOTS],
    )
    parser.add_argument("--dataset-out-dir", default="")
    parser.add_argument("--split-out-dir", default="")
    parser.add_argument("--train-out-dir", default="")
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--train-epochs", type=int, default=80)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--train-lr", type=float, default=1e-3)
    parser.add_argument("--train-weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-hidden-dim", type=int, default=16)
    parser.add_argument("--train-dropout", type=float, default=0.1)
    parser.add_argument("--train-device", default="cuda")
    parser.add_argument("--registry-csv", default=str(REGISTRY_CSV))
    return parser.parse_args()


def append_registry(summary_csv: Path, run_root: Path, status: str, notes: str, registry_csv: str) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "append_experiment_record.py"),
        "--csv",
        str(registry_csv),
        "--kind",
        "other",
        "--status",
        status,
        "--script",
        "scripts/queue_recovery_anchor_learning_followup.py",
        "--dataset",
        "DanceTrack",
        "--split",
        "recovery_anchor_followup",
        "--tracker-family",
        "deep_ocsort_preassoc_force_recovery_anchor",
        "--variant",
        run_root.name,
        "--tag",
        "recovery_anchor_followup",
        "--run-root",
        str(run_root.resolve()),
        "--summary-csv",
        str(summary_csv.resolve()),
        "--notes",
        notes,
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=False)


def read_single_row(summary_csv: Path) -> Dict[str, str]:
    rows = read_rows(summary_csv)
    if not rows:
        raise FileNotFoundError(f"Missing summary rows: {summary_csv}")
    return rows[0]


def queue_status(summary_csv: Path) -> List[Dict[str, str]]:
    rows = read_rows(summary_csv)
    if not rows:
        raise FileNotFoundError(f"Missing queue summary rows: {summary_csv}")
    return rows


def process_alive(pattern: str) -> bool:
    result = subprocess.run(["ps", "-eo", "cmd"], capture_output=True, text=True, check=False)
    for line in result.stdout.splitlines():
        if pattern in line and "ps -eo cmd" not in line:
            return True
    return False


def wait_for_queue(wait_summary_csv: Path, wait_process_pattern: str, poll_seconds: int) -> str:
    while True:
        rows = queue_status(wait_summary_csv)
        statuses = {str(row.get("status", "")).strip() for row in rows}
        if statuses == {"success"}:
            finished = [str(row.get("finished_at", "") or "") for row in rows if str(row.get("finished_at", ""))]
            return max(finished) if finished else now_iso()
        if "failed" in statuses:
            failed_steps = [str(row.get("step", "")) for row in rows if str(row.get("status", "")) == "failed"]
            raise RuntimeError(f"Waited queue failed at steps: {failed_steps}")
        if not process_alive(wait_process_pattern):
            raise RuntimeError(
                "Waited queue summary still not successful, but no live process matches pattern "
                f"{shlex.quote(wait_process_pattern)}"
            )
        time.sleep(max(int(poll_seconds), 10))


def main() -> int:
    args = parse_args()
    tag = timestamp_tag()
    run_root = (
        Path(args.out_root).expanduser().resolve()
        if args.out_root
        else (REPO_ROOT / "outputs" / f"recovery_anchor_learning_followup_{tag}").resolve()
    )
    dataset_out_dir = (
        Path(args.dataset_out_dir).expanduser().resolve()
        if args.dataset_out_dir
        else (REPO_ROOT / "outputs" / f"recovery_anchor_extension_dataset_{tag}").resolve()
    )
    split_out_dir = (
        Path(args.split_out_dir).expanduser().resolve()
        if args.split_out_dir
        else (REPO_ROOT / "outputs" / f"recovery_anchor_sequence_split_{tag}").resolve()
    )
    train_out_dir = (
        Path(args.train_out_dir).expanduser().resolve()
        if args.train_out_dir
        else (REPO_ROOT / "outputs" / f"train_recovery_anchor_gate_{tag}").resolve()
    )

    wait_summary_csv = Path(args.wait_summary_csv).expanduser().resolve()
    wait_run_root = Path(args.wait_run_root).expanduser().resolve()
    base_run_roots = [str(Path(item).expanduser().resolve()) for item in list(args.base_run_roots or [])]
    all_run_roots = list(base_run_roots) + [str(wait_run_root)]

    summary_csv = run_root / "summary.csv"
    logs_dir = run_root / "logs"
    queue_rows: List[Dict[str, object]] = [
        {
            "step": "wait_collect",
            "name": f"{run_root.name}_wait_collect",
            "status": "pending",
            "out_dir": str(wait_run_root),
            "summary_csv": str(wait_summary_csv),
            "log_path": str((logs_dir / "wait_collect.log").resolve()),
            "started_at": "",
            "finished_at": "",
            "notes": "wait for the 8-sequence recovery-anchor collection queue to finish successfully",
        },
        {
            "step": "build_dataset",
            "name": f"{run_root.name}_build_dataset",
            "status": "pending",
            "out_dir": str(dataset_out_dir),
            "summary_csv": str((dataset_out_dir / "summary.csv").resolve()),
            "log_path": str((logs_dir / "build_dataset.log").resolve()),
            "started_at": "",
            "finished_at": "",
            "notes": "build the enlarged recovery-anchor anchor/extension JSONL dataset",
        },
        {
            "step": "split_anchor",
            "name": f"{run_root.name}_split_anchor",
            "status": "pending",
            "out_dir": str(split_out_dir),
            "summary_csv": str((split_out_dir / "summary.csv").resolve()),
            "log_path": str((logs_dir / "split_anchor.log").resolve()),
            "started_at": "",
            "finished_at": "",
            "notes": "build a sequence-level train/val split for anchor candidates",
        },
        {
            "step": "train_anchor",
            "name": f"{run_root.name}_train_anchor",
            "status": "pending",
            "out_dir": str(train_out_dir),
            "summary_csv": str((train_out_dir / "summary.csv").resolve()),
            "log_path": str((logs_dir / "train_anchor.log").resolve()),
            "started_at": "",
            "finished_at": "",
            "notes": "train the first learned anchor gate prototype",
        },
    ]
    write_rows(summary_csv, QUEUE_FIELDS, queue_rows)
    append_registry(summary_csv, run_root, "running", "started recovery-anchor follow-up queue", args.registry_csv)

    overall_status = "success"
    overall_notes = "completed recovery-anchor follow-up queue"

    try:
        update_row(queue_rows, "wait_collect", status="running", started_at=now_iso())
        write_rows(summary_csv, QUEUE_FIELDS, queue_rows)
        finished_at = wait_for_queue(
            wait_summary_csv=wait_summary_csv,
            wait_process_pattern=str(args.wait_process_pattern),
            poll_seconds=int(args.poll_seconds),
        )
        update_row(
            queue_rows,
            "wait_collect",
            status="success",
            finished_at=finished_at,
            notes=f"waited queue finished successfully: {wait_run_root.name}",
        )
        write_rows(summary_csv, QUEUE_FIELDS, queue_rows)

        update_row(queue_rows, "build_dataset", status="running", started_at=now_iso())
        write_rows(summary_csv, QUEUE_FIELDS, queue_rows)
        build_cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "build_recovery_anchor_extension_dataset.py"),
            "--out-dir",
            str(dataset_out_dir),
            "--run-roots",
            *all_run_roots,
        ]
        rc = run_step(build_cmd, logs_dir / "build_dataset.log", cwd=REPO_ROOT)
        if rc != 0:
            raise RuntimeError(f"build dataset step returned code {rc}")
        ensure_child_success(dataset_out_dir / "summary.csv")
        dataset_summary = read_single_row(dataset_out_dir / "summary.csv")
        update_row(
            queue_rows,
            "build_dataset",
            status="success",
            finished_at=now_iso(),
            notes=(
                f"anchor_rows={dataset_summary.get('anchor_rows', '0')} "
                f"anchor_pos={dataset_summary.get('anchor_positive_rows', '0')} "
                f"extension_pos={dataset_summary.get('extension_positive_rows', '0')}"
            ),
        )
        write_rows(summary_csv, QUEUE_FIELDS, queue_rows)

        update_row(queue_rows, "split_anchor", status="running", started_at=now_iso())
        write_rows(summary_csv, QUEUE_FIELDS, queue_rows)
        split_cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "build_recovery_anchor_sequence_split.py"),
            "--anchor-jsonl",
            str(dataset_out_dir / "anchor_dataset.jsonl"),
            "--out-dir",
            str(split_out_dir),
        ]
        rc = run_step(split_cmd, logs_dir / "split_anchor.log", cwd=REPO_ROOT)
        if rc != 0:
            raise RuntimeError(f"split anchor step returned code {rc}")
        ensure_child_success(split_out_dir / "summary.csv")
        split_summary = read_single_row(split_out_dir / "summary.csv")
        update_row(
            queue_rows,
            "split_anchor",
            status="success",
            finished_at=now_iso(),
            notes=(
                f"val_seq={split_summary.get('val_seq_names', '')} "
                f"train_pos={split_summary.get('train_pos', '0')} "
                f"val_pos={split_summary.get('val_pos', '0')}"
            ),
        )
        write_rows(summary_csv, QUEUE_FIELDS, queue_rows)

        update_row(queue_rows, "train_anchor", status="running", started_at=now_iso())
        write_rows(summary_csv, QUEUE_FIELDS, queue_rows)
        train_cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "train_fgas_acceptance_gate.py"),
            "--train-jsonl",
            str(split_out_dir / "train.jsonl"),
            "--val-jsonl",
            str(split_out_dir / "val.jsonl"),
            "--out-dir",
            str(train_out_dir),
            "--device",
            str(args.train_device),
            "--epochs",
            str(int(args.train_epochs)),
            "--batch-size",
            str(int(args.train_batch_size)),
            "--lr",
            str(float(args.train_lr)),
            "--weight-decay",
            str(float(args.train_weight_decay)),
            "--hidden-dim",
            str(int(args.train_hidden_dim)),
            "--dropout",
            str(float(args.train_dropout)),
            "--registry-dataset",
            "DanceTrack",
            "--registry-split",
            "recovery_anchor_jsonl",
            "--registry-tracker-family",
            "deep_ocsort_preassoc_force_recovery_anchor",
            "--registry-variant",
            train_out_dir.name,
            "--registry-tag",
            "recovery_anchor_gate_v1",
            "--registry-notes",
            "recovery-anchor gate training",
        ]
        rc = run_step(train_cmd, logs_dir / "train_anchor.log", cwd=REPO_ROOT)
        if rc != 0:
            raise RuntimeError(f"train anchor step returned code {rc}")
        ensure_child_success(train_out_dir / "summary.csv")
        train_summary = read_single_row(train_out_dir / "summary.csv")
        update_row(
            queue_rows,
            "train_anchor",
            status="success",
            finished_at=now_iso(),
            notes=(
                f"best_metric={train_summary.get('best_metric', '0')} "
                f"val_bal_acc={train_summary.get('val_balanced_accuracy', '0')} "
                f"val_f1={train_summary.get('val_f1', '0')}"
            ),
        )
        write_rows(summary_csv, QUEUE_FIELDS, queue_rows)

        overall_notes = (
            f"dataset={dataset_out_dir.name} split={split_out_dir.name} "
            f"train={train_out_dir.name}"
        )
    except Exception as exc:
        overall_status = "failed"
        overall_notes = str(exc)
        for row in queue_rows:
            if str(row.get("status", "")) == "running":
                row["status"] = "failed"
                row["finished_at"] = now_iso()
                row["notes"] = str(exc)
        write_rows(summary_csv, QUEUE_FIELDS, queue_rows)

    append_registry(summary_csv, run_root, overall_status, overall_notes, args.registry_csv)
    return 0 if overall_status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
