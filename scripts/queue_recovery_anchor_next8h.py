#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List

from queue_deep_ocsort_preassoc_force_rewrite_next2h import (
    QUEUE_FIELDS,
    REPO_ROOT,
    REGISTRY_CSV,
    child_finished_at,
    ensure_child_success,
    now_iso,
    read_rows,
    run_step,
    timestamp_tag,
    update_row,
    write_rows,
)


LOCAL_CONTENTION_RANKER = (
    REPO_ROOT / "outputs" / "local_contention_ranker_mot17_mot20_dance_seqholdout_20260406_1" / "model.pt"
)
LOCAL_CONTENTION_ACCEPTANCE = (
    REPO_ROOT / "outputs" / "local_contention_acceptance_gate_mot17_mot20_dance_seqholdout_20260408_1" / "best.pt"
)

DEFAULT_WAIT_QUEUE_ROOT = REPO_ROOT / "outputs" / "queue_recovery_anchor_learning_followup_20260409_1"
DEFAULT_WAIT_SUMMARY_CSV = DEFAULT_WAIT_QUEUE_ROOT / "summary.csv"
DEFAULT_BATCH1_COLLECT_ROOT = REPO_ROOT / "outputs" / "deep_ocsort_preassoc_force_recovery_anchor_collect_dance8_20260409_1"
DEFAULT_STAGE1_DATASET_ROOT = REPO_ROOT / "outputs" / "recovery_anchor_extension_dataset_20260409_4"
DEFAULT_STAGE1_SPLIT_ROOT = REPO_ROOT / "outputs" / "recovery_anchor_sequence_split_20260409_1"

DEFAULT_BASE_RUN_ROOTS = [
    REPO_ROOT / "outputs" / "deep_ocsort_preassoc_force_recovery_anchor_seq0090_debug" / "runs" / "anchor_gap3_safe015",
    REPO_ROOT / "outputs" / "deep_ocsort_preassoc_force_recovery_anchor_seq0090_debug" / "runs" / "anchor_gap2_safe015",
    REPO_ROOT / "outputs" / "deep_ocsort_preassoc_force_recovery_anchor_seq0090_debug" / "runs" / "anchor_gap3_safe020",
    REPO_ROOT / "outputs" / "deep_ocsort_preassoc_force_recovery_anchor_confirm_dance3_debug" / "runs" / "anchor_gap2_safe015_dance3",
]

BATCH2_SEQS = [
    "dancetrack0005",
    "dancetrack0007",
    "dancetrack0014",
    "dancetrack0018",
    "dancetrack0025",
    "dancetrack0026",
    "dancetrack0030",
    "dancetrack0035",
]

BATCH3_SEQS = [
    "dancetrack0041",
    "dancetrack0043",
    "dancetrack0063",
    "dancetrack0073",
    "dancetrack0077",
    "dancetrack0079",
]

TRAIN_VARIANTS = [
    {
        "name": "h8do00_seed42",
        "hidden_dim": 8,
        "dropout": 0.0,
        "seed": 42,
    },
    {
        "name": "h16do01_seed43",
        "hidden_dim": 16,
        "dropout": 0.1,
        "seed": 43,
    },
    {
        "name": "h32do01_seed44",
        "hidden_dim": 32,
        "dropout": 0.1,
        "seed": 44,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Schedule the next 6-8 hours of recovery-anchor data collection and anchor-gate training sweeps."
    )
    parser.add_argument("--out-root", default="")
    parser.add_argument("--wait-summary-csv", default=str(DEFAULT_WAIT_SUMMARY_CSV))
    parser.add_argument("--wait-run-root", default=str(DEFAULT_WAIT_QUEUE_ROOT))
    parser.add_argument(
        "--wait-process-pattern",
        default=str(DEFAULT_WAIT_QUEUE_ROOT.name),
        help="substring expected in the live process table while the waited queue is still running",
    )
    parser.add_argument(
        "--base-run-roots",
        nargs="*",
        default=[str(path) for path in DEFAULT_BASE_RUN_ROOTS],
    )
    parser.add_argument("--stage1-split-root", default=str(DEFAULT_STAGE1_SPLIT_ROOT))
    parser.add_argument("--batch1-collect-root", default=str(DEFAULT_BATCH1_COLLECT_ROOT))
    parser.add_argument("--batch2-collect-root", default=str(REPO_ROOT / "outputs" / "deep_ocsort_preassoc_force_recovery_anchor_collect_dance8b_20260409_1"))
    parser.add_argument("--batch3-collect-root", default=str(REPO_ROOT / "outputs" / "deep_ocsort_preassoc_force_recovery_anchor_collect_dance6c_20260409_1"))
    parser.add_argument("--stage2-dataset-root", default=str(REPO_ROOT / "outputs" / "recovery_anchor_extension_dataset_20260409_5"))
    parser.add_argument("--stage2-split-root", default=str(REPO_ROOT / "outputs" / "recovery_anchor_sequence_split_20260409_2"))
    parser.add_argument("--stage3-dataset-root", default=str(REPO_ROOT / "outputs" / "recovery_anchor_extension_dataset_20260409_6"))
    parser.add_argument("--stage3-split-root", default=str(REPO_ROOT / "outputs" / "recovery_anchor_sequence_split_20260409_3"))
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--train-epochs", type=int, default=80)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--train-lr", type=float, default=1e-3)
    parser.add_argument("--train-weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-device", default="cpu")
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
        "scripts/queue_recovery_anchor_next8h.py",
        "--dataset",
        "DanceTrack",
        "--split",
        "recovery_anchor_next8h",
        "--tracker-family",
        "deep_ocsort_preassoc_force_recovery_anchor",
        "--variant",
        run_root.name,
        "--tag",
        "recovery_anchor_next8h",
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


def read_queue_rows(summary_csv: Path) -> List[Dict[str, str]]:
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
        rows = read_queue_rows(wait_summary_csv)
        statuses = {str(row.get("status", "")).strip() for row in rows}
        if statuses == {"success"}:
            return child_finished_at(wait_summary_csv)
        if "failed" in statuses:
            failed_steps = [str(row.get("step", "")) for row in rows if str(row.get("status", "")) == "failed"]
            raise RuntimeError(f"Waited queue failed at steps: {failed_steps}")
        if not process_alive(wait_process_pattern):
            raise RuntimeError(
                "Waited queue summary still not successful, but no live process matches pattern "
                f"{shlex.quote(wait_process_pattern)}"
            )
        time.sleep(max(int(poll_seconds), 10))


def build_collection_cmd(*, out_root: Path, seq_names: Iterable[str]) -> List[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_deep_ocsort_preassoc_competition_dataset_eval.py"),
        "--benchmark",
        "DanceTrack",
        "--seq-names",
        *list(seq_names),
        "--out-root",
        str(out_root),
        "--preassoc-stale-competition-min-time-since-update",
        "2",
        "--preassoc-stale-competition-max-time-since-update",
        "8",
        "--preassoc-stale-competition-min-hits",
        "8",
        "--preassoc-stale-competition-min-box-iou",
        "0.5",
        "--preassoc-stale-competition-min-edge-score",
        "0.0",
        "--preassoc-stale-competition-bias",
        "0.1",
        "--preassoc-stale-competition-iou-scale",
        "0.0",
        "--preassoc-stale-competition-require-raw-owner",
        "--preassoc-stale-competition-min-age-gap-vs-owner",
        "10",
        "--preassoc-stale-competition-owner-edge-penalty",
        "0.05",
        "--preassoc-stale-competition-takeover-soft-margin-floor",
        "0.1",
        "--preassoc-stale-competition-takeover-soft-edge-advantage-floor",
        "0.05",
        "--preassoc-stale-competition-owner-alt-det-bias",
        "0.3",
        "--preassoc-stale-competition-owner-alt-det-min-score",
        "0.35",
        "--preassoc-stale-competition-owner-alt-det-min-box-iou",
        "0.30",
        "--preassoc-stale-competition-max-owner-edge-deficit",
        "-1.0",
        "--preassoc-stale-competition-force-owner-edge-deficit-arg",
        "--preassoc-stale-competition-block-owner-on-reclaim",
        "--preassoc-stale-competition-force-rewrite-enable",
        "--preassoc-stale-competition-force-rewrite-min-score",
        "0.65",
        "--preassoc-stale-competition-force-rewrite-gate-weight",
        "0.45",
        "--preassoc-stale-competition-force-rewrite-iou-weight",
        "0.30",
        "--preassoc-stale-competition-force-rewrite-ranker-weight",
        "0.15",
        "--preassoc-stale-competition-force-rewrite-age-weight",
        "0.0",
        "--preassoc-stale-competition-force-rewrite-age-cap",
        "20",
        "--preassoc-stale-competition-force-rewrite-owner-alt-bonus",
        "0.0",
        "--preassoc-stale-competition-force-rewrite-neighborhood-enable",
        "--preassoc-stale-competition-force-rewrite-min-neighborhood-gain",
        "0.0",
        "--preassoc-stale-competition-force-rewrite-trapped-owner-min-neighborhood-gain",
        "0.0",
        "--preassoc-stale-competition-force-rewrite-reroute-ready-min-neighborhood-gain",
        "0.0",
        "--preassoc-stale-competition-force-rewrite-trapped-owner-negative-gain-min-challenger-alt-box-iou",
        "-1.0",
        "--preassoc-stale-competition-force-rewrite-neighborhood-keep-challenger-alt-weight",
        "0.25",
        "--preassoc-stale-competition-force-rewrite-neighborhood-rewrite-owner-alt-weight",
        "1.0",
        "--preassoc-stale-competition-force-rewrite-neighborhood-shared-alt-penalty",
        "0.25",
        "--preassoc-stale-competition-force-rewrite-neighborhood-trapped-owner-bonus",
        "0.75",
        "--preassoc-stale-competition-force-rewrite-neighborhood-reroute-ready-penalty",
        "0.6",
        "--preassoc-stale-competition-force-rewrite-min-box-iou",
        "0.55",
        "--preassoc-stale-competition-force-rewrite-max-age-gap",
        "200",
        "--preassoc-stale-competition-force-rewrite-max-owner-alt-det-box-iou",
        "0.5",
        "--preassoc-stale-competition-force-rewrite-reroute-ready-min-box-iou",
        "0.8",
        "--preassoc-stale-competition-force-rewrite-recovery-memory-enable",
        "--preassoc-stale-competition-force-rewrite-recovery-memory-max-frame-gap",
        "2",
        "--preassoc-stale-competition-force-rewrite-recovery-memory-min-score",
        "0.85",
        "--preassoc-stale-competition-force-rewrite-recovery-memory-min-box-iou",
        "0.8",
        "--preassoc-stale-competition-force-rewrite-recovery-memory-min-challenger-alt-box-iou",
        "0.12",
        "--preassoc-stale-competition-force-rewrite-recovery-memory-warmup-min-neighborhood-gain",
        "-0.08",
        "--preassoc-stale-competition-force-rewrite-recovery-memory-bonus",
        "0.05",
        "--preassoc-stale-competition-force-rewrite-recovery-memory-bonus-max-streak",
        "2",
        "--preassoc-stale-competition-force-rewrite-recovery-memory-gate-bonus",
        "0.0001",
        "--preassoc-stale-competition-force-rewrite-recovery-memory-anchor-enable",
        "--preassoc-stale-competition-force-rewrite-recovery-memory-anchor-min-raw-neighborhood-gain",
        "0.05",
        "--preassoc-stale-competition-force-rewrite-recovery-memory-anchor-max-edge-deficit",
        "0.35",
        "--preassoc-stale-competition-force-rewrite-recovery-memory-extension-max-edge-deficit-delta",
        "0.15",
        "--preassoc-stale-competition-export-jsonl",
        str(out_root / "preassoc_candidates.jsonl"),
        "--local-contention-export-jsonl",
        str(out_root / "local_contention_units.jsonl"),
        "--local-contention-topk",
        "3",
        "--local-contention-min-box-iou",
        "0.5",
        "--local-contention-max-time-since-update",
        "8",
        "--local-contention-min-challenger-hits",
        "3",
        "--local-contention-owner-weak-hits",
        "8",
        "--local-contention-ranker-checkpoint",
        str(LOCAL_CONTENTION_RANKER),
        "--local-contention-ranker-thresh",
        "0.99",
        "--local-contention-ranker-bias",
        "0.0",
        "--local-contention-ranker-min-margin-to-second",
        "0.05",
        "--local-contention-ranker-margin-bias",
        "0.5",
        "--preassoc-stale-competition-acceptance-gate-checkpoint",
        str(LOCAL_CONTENTION_ACCEPTANCE),
        "--preassoc-stale-competition-acceptance-gate-thresh",
        "0.9995",
    ]


def build_dataset_cmd(*, out_dir: Path, run_roots: Iterable[Path]) -> List[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "scripts" / "build_recovery_anchor_extension_dataset.py"),
        "--out-dir",
        str(out_dir),
        "--run-roots",
        *[str(path) for path in run_roots],
    ]


def build_split_cmd(*, anchor_jsonl: Path, out_dir: Path) -> List[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "scripts" / "build_recovery_anchor_sequence_split.py"),
        "--anchor-jsonl",
        str(anchor_jsonl),
        "--out-dir",
        str(out_dir),
    ]


def build_train_cmd(
    *,
    split_root: Path,
    out_dir: Path,
    hidden_dim: int,
    dropout: float,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: str,
) -> List[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "scripts" / "train_fgas_acceptance_gate.py"),
        "--train-jsonl",
        str(split_root / "train.jsonl"),
        "--val-jsonl",
        str(split_root / "val.jsonl"),
        "--out-dir",
        str(out_dir),
        "--device",
        str(device),
        "--epochs",
        str(int(epochs)),
        "--batch-size",
        str(int(batch_size)),
        "--lr",
        str(float(lr)),
        "--weight-decay",
        str(float(weight_decay)),
        "--hidden-dim",
        str(int(hidden_dim)),
        "--dropout",
        str(float(dropout)),
        "--seed",
        str(int(seed)),
        "--registry-dataset",
        "DanceTrack",
        "--registry-split",
        "recovery_anchor_jsonl",
        "--registry-tracker-family",
        "deep_ocsort_preassoc_force_recovery_anchor",
        "--registry-variant",
        out_dir.name,
        "--registry-tag",
        out_dir.name,
        "--registry-notes",
        "recovery-anchor gate sweep training",
    ]


def summary_note_from_child(summary_csv: Path) -> str:
    rows = read_rows(summary_csv)
    if not rows:
        return ""
    if len(rows) == 1:
        row = rows[0]
        note_parts = []
        if "anchor_positive_rows" in row:
            note_parts.append(
                f"anchor_pos={row.get('anchor_positive_rows', '0')} extension_pos={row.get('extension_positive_rows', '0')}"
            )
        elif "train_pos" in row:
            note_parts.append(f"train_pos={row.get('train_pos', '0')} val_pos={row.get('val_pos', '0')}")
        elif "best_metric" in row:
            note_parts.append(
                f"best_metric={row.get('best_metric', '0')} val_bal_acc={row.get('val_balanced_accuracy', '0')} val_f1={row.get('val_f1', '0')}"
            )
        return " ".join(note_parts)
    for row in rows:
        if str(row.get("step", "")) == "compare":
            return str(row.get("notes", ""))
    return ""


def make_row(step: str, out_dir: Path, log_path: Path, notes: str, summary_csv: Path | None = None) -> Dict[str, object]:
    return {
        "step": step,
        "name": step,
        "status": "pending",
        "out_dir": str(out_dir.resolve()),
        "summary_csv": str((summary_csv or (out_dir / "summary.csv")).resolve()),
        "log_path": str(log_path.resolve()),
        "started_at": "",
        "finished_at": "",
        "notes": notes,
    }


def run_child_step(
    *,
    queue_rows: List[Dict[str, object]],
    summary_csv: Path,
    step: str,
    cmd: List[str],
    log_path: Path,
    child_summary_csv: Path,
) -> str:
    update_row(queue_rows, step, status="running", started_at=now_iso())
    write_rows(summary_csv, QUEUE_FIELDS, queue_rows)
    rc = run_step(cmd, log_path, cwd=REPO_ROOT)
    if rc != 0:
        raise RuntimeError(f"{step} returned code {rc}")
    ensure_child_success(child_summary_csv)
    finished_at = child_finished_at(child_summary_csv)
    notes = summary_note_from_child(child_summary_csv)
    update_row(queue_rows, step, status="success", finished_at=finished_at, notes=notes or step)
    write_rows(summary_csv, QUEUE_FIELDS, queue_rows)
    return notes


def main() -> int:
    args = parse_args()
    tag = timestamp_tag()
    run_root = (
        Path(args.out_root).expanduser().resolve()
        if args.out_root
        else (REPO_ROOT / "outputs" / f"queue_recovery_anchor_next8h_{tag}").resolve()
    )
    run_root.mkdir(parents=True, exist_ok=True)

    wait_summary_csv = Path(args.wait_summary_csv).expanduser().resolve()
    wait_run_root = Path(args.wait_run_root).expanduser().resolve()
    stage1_split_root = Path(args.stage1_split_root).expanduser().resolve()
    batch1_collect_root = Path(args.batch1_collect_root).expanduser().resolve()
    batch2_collect_root = Path(args.batch2_collect_root).expanduser().resolve()
    batch3_collect_root = Path(args.batch3_collect_root).expanduser().resolve()
    stage2_dataset_root = Path(args.stage2_dataset_root).expanduser().resolve()
    stage2_split_root = Path(args.stage2_split_root).expanduser().resolve()
    stage3_dataset_root = Path(args.stage3_dataset_root).expanduser().resolve()
    stage3_split_root = Path(args.stage3_split_root).expanduser().resolve()
    base_run_roots = [Path(item).expanduser().resolve() for item in list(args.base_run_roots or [])]

    logs_dir = run_root / "logs"
    summary_csv = run_root / "summary.csv"

    queue_rows: List[Dict[str, object]] = [
        make_row(
            step="wait_stage1_followup",
            out_dir=wait_run_root,
            log_path=logs_dir / "wait_stage1_followup.log",
            notes="wait for the current recovery-anchor follow-up queue to finish",
            summary_csv=wait_summary_csv,
        ),
    ]
    for stage_name, split_root in [
        ("stage1", stage1_split_root),
        ("stage2", stage2_split_root),
        ("stage3", stage3_split_root),
    ]:
        for variant in TRAIN_VARIANTS:
            out_dir = REPO_ROOT / "outputs" / f"train_recovery_anchor_gate_{stage_name}_{variant['name']}_20260409_1"
            queue_rows.append(
                make_row(
                    step=f"{stage_name}_train_{variant['name']}",
                    out_dir=out_dir,
                    log_path=logs_dir / f"{stage_name}_train_{variant['name']}.log",
                    notes=f"{stage_name} anchor-gate sweep {variant['name']}",
                )
            )
        if stage_name == "stage1":
            queue_rows.append(
                make_row(
                    step="collect_batch2",
                    out_dir=batch2_collect_root,
                    log_path=logs_dir / "collect_batch2.log",
                    notes="collect anchor exports on the next 8 uncovered DanceTrack validation sequences",
                )
            )
            queue_rows.append(
                make_row(
                    step="build_dataset_stage2",
                    out_dir=stage2_dataset_root,
                    log_path=logs_dir / "build_dataset_stage2.log",
                    notes="rebuild anchor dataset with batch1 and batch2 collections merged",
                )
            )
            queue_rows.append(
                make_row(
                    step="split_anchor_stage2",
                    out_dir=stage2_split_root,
                    log_path=logs_dir / "split_anchor_stage2.log",
                    notes="build a sequence split for the stage2 anchor dataset",
                )
            )
        if stage_name == "stage2":
            queue_rows.append(
                make_row(
                    step="collect_batch3",
                    out_dir=batch3_collect_root,
                    log_path=logs_dir / "collect_batch3.log",
                    notes="collect anchor exports on the final 6 uncovered DanceTrack validation sequences",
                )
            )
            queue_rows.append(
                make_row(
                    step="build_dataset_stage3",
                    out_dir=stage3_dataset_root,
                    log_path=logs_dir / "build_dataset_stage3.log",
                    notes="rebuild anchor dataset with all 25 DanceTrack validation sequences merged",
                )
            )
            queue_rows.append(
                make_row(
                    step="split_anchor_stage3",
                    out_dir=stage3_split_root,
                    log_path=logs_dir / "split_anchor_stage3.log",
                    notes="build a sequence split for the final anchor dataset",
                )
            )

    write_rows(summary_csv, QUEUE_FIELDS, queue_rows)
    append_registry(summary_csv, run_root, "running", "started recovery-anchor next8h queue", args.registry_csv)

    overall_status = "success"
    overall_notes = "completed recovery-anchor next8h queue"

    try:
        update_row(queue_rows, "wait_stage1_followup", status="running", started_at=now_iso())
        write_rows(summary_csv, QUEUE_FIELDS, queue_rows)
        finished_at = wait_for_queue(
            wait_summary_csv=wait_summary_csv,
            wait_process_pattern=str(args.wait_process_pattern),
            poll_seconds=int(args.poll_seconds),
        )
        update_row(
            queue_rows,
            "wait_stage1_followup",
            status="success",
            finished_at=finished_at,
            notes=f"stage1 follow-up queue finished successfully: {wait_run_root.name}",
        )
        write_rows(summary_csv, QUEUE_FIELDS, queue_rows)

        for variant in TRAIN_VARIANTS:
            step = f"stage1_train_{variant['name']}"
            out_dir = REPO_ROOT / "outputs" / f"train_recovery_anchor_gate_stage1_{variant['name']}_20260409_1"
            run_child_step(
                queue_rows=queue_rows,
                summary_csv=summary_csv,
                step=step,
                cmd=build_train_cmd(
                    split_root=stage1_split_root,
                    out_dir=out_dir,
                    hidden_dim=int(variant["hidden_dim"]),
                    dropout=float(variant["dropout"]),
                    seed=int(variant["seed"]),
                    epochs=int(args.train_epochs),
                    batch_size=int(args.train_batch_size),
                    lr=float(args.train_lr),
                    weight_decay=float(args.train_weight_decay),
                    device=str(args.train_device),
                ),
                log_path=logs_dir / f"{step}.log",
                child_summary_csv=out_dir / "summary.csv",
            )

        run_child_step(
            queue_rows=queue_rows,
            summary_csv=summary_csv,
            step="collect_batch2",
            cmd=build_collection_cmd(out_root=batch2_collect_root, seq_names=BATCH2_SEQS),
            log_path=logs_dir / "collect_batch2.log",
            child_summary_csv=batch2_collect_root / "summary.csv",
        )
        run_child_step(
            queue_rows=queue_rows,
            summary_csv=summary_csv,
            step="build_dataset_stage2",
            cmd=build_dataset_cmd(
                out_dir=stage2_dataset_root,
                run_roots=[*base_run_roots, batch1_collect_root, batch2_collect_root],
            ),
            log_path=logs_dir / "build_dataset_stage2.log",
            child_summary_csv=stage2_dataset_root / "summary.csv",
        )
        run_child_step(
            queue_rows=queue_rows,
            summary_csv=summary_csv,
            step="split_anchor_stage2",
            cmd=build_split_cmd(
                anchor_jsonl=stage2_dataset_root / "anchor_dataset.jsonl",
                out_dir=stage2_split_root,
            ),
            log_path=logs_dir / "split_anchor_stage2.log",
            child_summary_csv=stage2_split_root / "summary.csv",
        )

        for variant in TRAIN_VARIANTS:
            step = f"stage2_train_{variant['name']}"
            out_dir = REPO_ROOT / "outputs" / f"train_recovery_anchor_gate_stage2_{variant['name']}_20260409_1"
            run_child_step(
                queue_rows=queue_rows,
                summary_csv=summary_csv,
                step=step,
                cmd=build_train_cmd(
                    split_root=stage2_split_root,
                    out_dir=out_dir,
                    hidden_dim=int(variant["hidden_dim"]),
                    dropout=float(variant["dropout"]),
                    seed=int(variant["seed"]),
                    epochs=int(args.train_epochs),
                    batch_size=int(args.train_batch_size),
                    lr=float(args.train_lr),
                    weight_decay=float(args.train_weight_decay),
                    device=str(args.train_device),
                ),
                log_path=logs_dir / f"{step}.log",
                child_summary_csv=out_dir / "summary.csv",
            )

        run_child_step(
            queue_rows=queue_rows,
            summary_csv=summary_csv,
            step="collect_batch3",
            cmd=build_collection_cmd(out_root=batch3_collect_root, seq_names=BATCH3_SEQS),
            log_path=logs_dir / "collect_batch3.log",
            child_summary_csv=batch3_collect_root / "summary.csv",
        )
        run_child_step(
            queue_rows=queue_rows,
            summary_csv=summary_csv,
            step="build_dataset_stage3",
            cmd=build_dataset_cmd(
                out_dir=stage3_dataset_root,
                run_roots=[*base_run_roots, batch1_collect_root, batch2_collect_root, batch3_collect_root],
            ),
            log_path=logs_dir / "build_dataset_stage3.log",
            child_summary_csv=stage3_dataset_root / "summary.csv",
        )
        run_child_step(
            queue_rows=queue_rows,
            summary_csv=summary_csv,
            step="split_anchor_stage3",
            cmd=build_split_cmd(
                anchor_jsonl=stage3_dataset_root / "anchor_dataset.jsonl",
                out_dir=stage3_split_root,
            ),
            log_path=logs_dir / "split_anchor_stage3.log",
            child_summary_csv=stage3_split_root / "summary.csv",
        )

        for variant in TRAIN_VARIANTS:
            step = f"stage3_train_{variant['name']}"
            out_dir = REPO_ROOT / "outputs" / f"train_recovery_anchor_gate_stage3_{variant['name']}_20260409_1"
            run_child_step(
                queue_rows=queue_rows,
                summary_csv=summary_csv,
                step=step,
                cmd=build_train_cmd(
                    split_root=stage3_split_root,
                    out_dir=out_dir,
                    hidden_dim=int(variant["hidden_dim"]),
                    dropout=float(variant["dropout"]),
                    seed=int(variant["seed"]),
                    epochs=int(args.train_epochs),
                    batch_size=int(args.train_batch_size),
                    lr=float(args.train_lr),
                    weight_decay=float(args.train_weight_decay),
                    device=str(args.train_device),
                ),
                log_path=logs_dir / f"{step}.log",
                child_summary_csv=out_dir / "summary.csv",
            )

        overall_notes = (
            f"stage2_dataset={stage2_dataset_root.name} "
            f"stage3_dataset={stage3_dataset_root.name} "
            f"final_split={stage3_split_root.name}"
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
