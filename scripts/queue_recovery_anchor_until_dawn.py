#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

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


DEFAULT_WAIT_QUEUE_ROOT = REPO_ROOT / "outputs" / "queue_recovery_anchor_next8h_20260409_1"
DEFAULT_WAIT_SUMMARY_CSV = DEFAULT_WAIT_QUEUE_ROOT / "summary.csv"
DEFAULT_FINAL_DATASET_ROOT = REPO_ROOT / "outputs" / "recovery_anchor_extension_dataset_20260409_6"
DEFAULT_FINAL_SPLIT_ROOT = REPO_ROOT / "outputs" / "recovery_anchor_sequence_split_20260409_3"

DEFAULT_STAGE1_TRAIN_ROOTS = [
    REPO_ROOT / "outputs" / "train_recovery_anchor_gate_stage1_h8do00_seed42_20260409_1",
    REPO_ROOT / "outputs" / "train_recovery_anchor_gate_stage1_h16do01_seed43_20260409_1",
    REPO_ROOT / "outputs" / "train_recovery_anchor_gate_stage1_h32do01_seed44_20260409_1",
]
DEFAULT_STAGE2_TRAIN_ROOTS = [
    REPO_ROOT / "outputs" / "train_recovery_anchor_gate_stage2_h8do00_seed42_20260409_1",
    REPO_ROOT / "outputs" / "train_recovery_anchor_gate_stage2_h16do01_seed43_20260409_1",
    REPO_ROOT / "outputs" / "train_recovery_anchor_gate_stage2_h32do01_seed44_20260409_1",
]
DEFAULT_STAGE3_TRAIN_ROOTS = [
    REPO_ROOT / "outputs" / "train_recovery_anchor_gate_stage3_h8do00_seed42_20260409_1",
    REPO_ROOT / "outputs" / "train_recovery_anchor_gate_stage3_h16do01_seed43_20260409_1",
    REPO_ROOT / "outputs" / "train_recovery_anchor_gate_stage3_h32do01_seed44_20260409_1",
]

CONFIRM_SEEDS = [101, 202, 303, 404]
RESCUE_VARIANTS = [
    {"name": "h8do00_seed101", "hidden_dim": 8, "dropout": 0.0, "seed": 101},
    {"name": "h8do00_seed202", "hidden_dim": 8, "dropout": 0.0, "seed": 202},
    {"name": "h16do00_seed303", "hidden_dim": 16, "dropout": 0.0, "seed": 303},
    {"name": "h16do01_seed404", "hidden_dim": 16, "dropout": 0.1, "seed": 404},
    {"name": "h32do00_seed505", "hidden_dim": 32, "dropout": 0.0, "seed": 505},
    {"name": "h32do01_seed606", "hidden_dim": 32, "dropout": 0.1, "seed": 606},
]

SELECTION_FIELDS = [
    "stage",
    "variant_name",
    "summary_csv",
    "best_metric",
    "val_balanced_accuracy",
    "val_f1",
    "hidden_dim",
    "dropout",
    "seed",
    "train_rows",
    "val_rows",
    "selected",
]

OVERNIGHT_RESULT_FIELDS = [
    "step",
    "branch",
    "split_name",
    "summary_csv",
    "hidden_dim",
    "dropout",
    "seed",
    "best_metric",
    "val_balanced_accuracy",
    "val_f1",
    "train_rows",
    "val_rows",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wait for the current recovery-anchor queue, then automatically continue confirmation or rescue experiments overnight."
    )
    parser.add_argument("--out-root", default="")
    parser.add_argument("--wait-summary-csv", default=str(DEFAULT_WAIT_SUMMARY_CSV))
    parser.add_argument("--wait-run-root", default=str(DEFAULT_WAIT_QUEUE_ROOT))
    parser.add_argument(
        "--wait-process-pattern",
        default=str(DEFAULT_WAIT_QUEUE_ROOT.name),
        help="Substring expected in the live process table while the waited queue is still running.",
    )
    parser.add_argument(
        "--stage1-train-roots",
        nargs="*",
        default=[str(path) for path in DEFAULT_STAGE1_TRAIN_ROOTS],
    )
    parser.add_argument(
        "--stage2-train-roots",
        nargs="*",
        default=[str(path) for path in DEFAULT_STAGE2_TRAIN_ROOTS],
    )
    parser.add_argument(
        "--stage3-train-roots",
        nargs="*",
        default=[str(path) for path in DEFAULT_STAGE3_TRAIN_ROOTS],
    )
    parser.add_argument("--final-dataset-root", default=str(DEFAULT_FINAL_DATASET_ROOT))
    parser.add_argument("--final-split-root", default=str(DEFAULT_FINAL_SPLIT_ROOT))
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--confirm-epochs", type=int, default=160)
    parser.add_argument("--rescue-epochs", type=int, default=220)
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
        "scripts/queue_recovery_anchor_until_dawn.py",
        "--dataset",
        "DanceTrack",
        "--split",
        "recovery_anchor_overnight",
        "--tracker-family",
        "deep_ocsort_preassoc_force_recovery_anchor",
        "--variant",
        run_root.name,
        "--tag",
        "recovery_anchor_overnight",
        "--run-root",
        str(run_root.resolve()),
        "--summary-csv",
        str(summary_csv.resolve()),
        "--notes",
        notes,
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=False)


def process_alive(pattern: str) -> bool:
    result = subprocess.run(["ps", "-eo", "cmd"], capture_output=True, text=True, check=False)
    for line in result.stdout.splitlines():
        if pattern in line and "ps -eo cmd" not in line:
            return True
    return False


def wait_for_queue(wait_summary_csv: Path, wait_process_pattern: str, poll_seconds: int) -> str:
    while True:
        rows = read_rows(wait_summary_csv)
        if not rows:
            raise FileNotFoundError(f"Missing queue summary rows: {wait_summary_csv}")
        statuses = {str(row.get("status", "")).strip() for row in rows}
        if statuses == {"success"}:
            return child_finished_at(wait_summary_csv)
        if "failed" in statuses:
            failed_steps = [str(row.get("step", "")) for row in rows if str(row.get("status", "")) == "failed"]
            raise RuntimeError(f"Waited queue failed at steps: {failed_steps}")
        if not process_alive(wait_process_pattern):
            raise RuntimeError(
                "Waited queue summary still not successful, but no live process matches pattern "
                f"{wait_process_pattern!r}"
            )
        time.sleep(max(int(poll_seconds), 10))


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


def mark_step(
    queue_rows: List[Dict[str, object]],
    summary_csv: Path,
    *,
    step: str,
    status: str,
    notes: str,
) -> None:
    update_row(queue_rows, step, status=status, finished_at=now_iso(), notes=notes)
    write_rows(summary_csv, QUEUE_FIELDS, queue_rows)


def mark_steps_skipped(
    queue_rows: List[Dict[str, object]],
    summary_csv: Path,
    steps: Iterable[str],
    reason: str,
) -> None:
    for step in steps:
        update_row(queue_rows, step, status="success", finished_at=now_iso(), notes=f"skipped: {reason}")
    write_rows(summary_csv, QUEUE_FIELDS, queue_rows)


def write_log(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(str(line))
            handle.write("\n")


def read_single_summary(summary_csv: Path) -> Dict[str, str]:
    rows = read_rows(summary_csv)
    if not rows:
        raise FileNotFoundError(f"Missing summary rows: {summary_csv}")
    return rows[0]


def read_train_candidate(summary_csv: Path, stage: str) -> Dict[str, object]:
    row = read_single_summary(summary_csv)
    return {
        "stage": stage,
        "variant_name": summary_csv.parent.name,
        "summary_csv": str(summary_csv.resolve()),
        "best_metric": float(row.get("best_metric", 0.0) or 0.0),
        "val_balanced_accuracy": float(row.get("val_balanced_accuracy", 0.0) or 0.0),
        "val_f1": float(row.get("val_f1", 0.0) or 0.0),
        "hidden_dim": int(float(row.get("hidden_dim", 0) or 0)),
        "dropout": float(row.get("dropout", 0.0) or 0.0),
        "seed": int(float(row.get("seed", 0) or 0)) if row.get("seed", "") not in {"", None} else 0,
        "train_rows": int(float(row.get("train_rows", 0) or 0)),
        "val_rows": int(float(row.get("val_rows", 0) or 0)),
    }


def stage_rank(stage: str) -> int:
    if stage == "stage3":
        return 3
    if stage == "stage2":
        return 2
    return 1


def select_best_candidate(candidates: Sequence[Dict[str, object]]) -> Dict[str, object]:
    if not candidates:
        raise ValueError("No train candidates were found.")
    return max(
        candidates,
        key=lambda item: (
            float(item["best_metric"]),
            float(item["val_f1"]),
            int(stage_rank(str(item["stage"]))),
            -float(item["dropout"]),
            -int(item["hidden_dim"]),
        ),
    )


def write_selection_csv(path: Path, rows: Sequence[Dict[str, object]], selected_summary_csv: str) -> None:
    output_rows: List[Dict[str, object]] = []
    for row in rows:
        item = dict(row)
        item["selected"] = int(str(row.get("summary_csv", "")) == str(selected_summary_csv))
        output_rows.append(item)
    write_rows(path, SELECTION_FIELDS, output_rows)


def build_split_cmd(
    *,
    anchor_jsonl: Path,
    out_dir: Path,
    target_val_fraction: float,
    min_val_positive: int,
    min_train_positive: int,
    max_auto_val_seqs: int,
) -> List[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "scripts" / "build_recovery_anchor_sequence_split.py"),
        "--anchor-jsonl",
        str(anchor_jsonl),
        "--out-dir",
        str(out_dir),
        "--target-val-fraction",
        str(float(target_val_fraction)),
        "--min-val-positive",
        str(int(min_val_positive)),
        "--min-train-positive",
        str(int(min_train_positive)),
        "--max-auto-val-seqs",
        str(int(max_auto_val_seqs)),
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
        "overnight recovery-anchor follow-up training",
    ]


def child_note(summary_csv: Path) -> str:
    row = read_single_summary(summary_csv)
    if "best_metric" in row:
        return (
            f"best_metric={row.get('best_metric', '0')} "
            f"val_bal_acc={row.get('val_balanced_accuracy', '0')} "
            f"val_f1={row.get('val_f1', '0')}"
        )
    if "train_pos" in row:
        return f"train_pos={row.get('train_pos', '0')} val_pos={row.get('val_pos', '0')}"
    if "anchor_positive_rows" in row:
        return (
            f"anchor_pos={row.get('anchor_positive_rows', '0')} "
            f"extension_pos={row.get('extension_positive_rows', '0')}"
        )
    return str(row)


def run_child_step(
    *,
    queue_rows: List[Dict[str, object]],
    summary_csv: Path,
    step: str,
    cmd: List[str],
    log_path: Path,
    child_summary_csv: Path,
) -> Dict[str, str]:
    update_row(queue_rows, step, status="running", started_at=now_iso())
    write_rows(summary_csv, QUEUE_FIELDS, queue_rows)
    rc = run_step(cmd, log_path, cwd=REPO_ROOT)
    if rc != 0:
        raise RuntimeError(f"{step} returned code {rc}")
    ensure_child_success(child_summary_csv)
    note = child_note(child_summary_csv)
    update_row(queue_rows, step, status="success", finished_at=child_finished_at(child_summary_csv), notes=note)
    write_rows(summary_csv, QUEUE_FIELDS, queue_rows)
    return read_single_summary(child_summary_csv)


def try_build_split_step(
    *,
    queue_rows: List[Dict[str, object]],
    summary_csv: Path,
    step: str,
    log_path: Path,
    anchor_jsonl: Path,
    out_dir: Path,
    candidates: Sequence[Dict[str, object]],
) -> Path | None:
    update_row(queue_rows, step, status="running", started_at=now_iso())
    write_rows(summary_csv, QUEUE_FIELDS, queue_rows)
    last_error = "unknown"
    for params in candidates:
        cmd = build_split_cmd(
            anchor_jsonl=anchor_jsonl,
            out_dir=out_dir,
            target_val_fraction=float(params["target_val_fraction"]),
            min_val_positive=int(params["min_val_positive"]),
            min_train_positive=int(params["min_train_positive"]),
            max_auto_val_seqs=int(params["max_auto_val_seqs"]),
        )
        rc = run_step(cmd, log_path, cwd=REPO_ROOT)
        if rc != 0:
            last_error = f"rc={rc} params={json.dumps(params, ensure_ascii=True)}"
            continue
        child_summary_csv = out_dir / "summary.csv"
        try:
            ensure_child_success(child_summary_csv)
        except Exception as exc:
            last_error = f"{exc} params={json.dumps(params, ensure_ascii=True)}"
            continue
        note = child_note(child_summary_csv)
        note = f"{note} split_params={json.dumps(params, ensure_ascii=True)}"
        update_row(queue_rows, step, status="success", finished_at=child_finished_at(child_summary_csv), notes=note)
        write_rows(summary_csv, QUEUE_FIELDS, queue_rows)
        return out_dir
    update_row(queue_rows, step, status="success", finished_at=now_iso(), notes=f"skipped: no feasible split ({last_error})")
    write_rows(summary_csv, QUEUE_FIELDS, queue_rows)
    return None


def append_overnight_result(results_csv: Path, row: Dict[str, object]) -> None:
    existing = read_rows(results_csv)
    rows: List[Dict[str, object]] = [dict(item) for item in existing]
    rows.append({key: row.get(key, "") for key in OVERNIGHT_RESULT_FIELDS})
    write_rows(results_csv, OVERNIGHT_RESULT_FIELDS, rows)


def main() -> int:
    args = parse_args()
    tag = timestamp_tag()
    run_root = (
        Path(args.out_root).expanduser().resolve()
        if args.out_root
        else (REPO_ROOT / "outputs" / f"queue_recovery_anchor_until_dawn_{tag}").resolve()
    )
    run_root.mkdir(parents=True, exist_ok=True)

    wait_summary_csv = Path(args.wait_summary_csv).expanduser().resolve()
    wait_run_root = Path(args.wait_run_root).expanduser().resolve()
    final_dataset_root = Path(args.final_dataset_root).expanduser().resolve()
    final_split_root = Path(args.final_split_root).expanduser().resolve()

    stage1_train_roots = [Path(item).expanduser().resolve() for item in list(args.stage1_train_roots or [])]
    stage2_train_roots = [Path(item).expanduser().resolve() for item in list(args.stage2_train_roots or [])]
    stage3_train_roots = [Path(item).expanduser().resolve() for item in list(args.stage3_train_roots or [])]

    logs_dir = run_root / "logs"
    summary_csv = run_root / "summary.csv"
    selection_csv = run_root / "selection_summary.csv"
    overnight_results_csv = run_root / "overnight_results.csv"

    robust_split_root = run_root / "stage3_robust_split"

    queue_rows: List[Dict[str, object]] = [
        make_row(
            step="wait_current_queue",
            out_dir=wait_run_root,
            log_path=logs_dir / "wait_current_queue.log",
            notes="wait for the current recovery-anchor queue to finish",
            summary_csv=wait_summary_csv,
        ),
        make_row(
            step="analyze_current_results",
            out_dir=run_root,
            log_path=logs_dir / "analyze_current_results.log",
            notes="analyze stage1/stage2/stage3 metrics and choose the overnight branch",
            summary_csv=summary_csv,
        ),
        make_row(
            step="build_stage3_robust_split",
            out_dir=robust_split_root,
            log_path=logs_dir / "build_stage3_robust_split.log",
            notes="build a harder validation split on the final merged stage3 anchor dataset",
        ),
    ]

    confirm_steps: List[str] = []
    for split_name in ["default", "robust"]:
        for seed in CONFIRM_SEEDS:
            step = f"confirm_{split_name}_seed{seed}"
            confirm_steps.append(step)
            queue_rows.append(
                make_row(
                    step=step,
                    out_dir=run_root / step,
                    log_path=logs_dir / f"{step}.log",
                    notes=f"confirmation training on {split_name} split seed={seed}",
                )
            )

    rescue_steps: List[str] = []
    for variant in RESCUE_VARIANTS:
        step = f"rescue_{variant['name']}"
        rescue_steps.append(step)
        queue_rows.append(
            make_row(
                step=step,
                out_dir=run_root / step,
                log_path=logs_dir / f"{step}.log",
                notes=f"rescue sweep {variant['name']} on the final stage3 split",
            )
        )

    write_rows(summary_csv, QUEUE_FIELDS, queue_rows)
    write_rows(selection_csv, SELECTION_FIELDS, [])
    write_rows(overnight_results_csv, OVERNIGHT_RESULT_FIELDS, [])
    append_registry(summary_csv, run_root, "running", "started overnight recovery-anchor autopilot queue", args.registry_csv)

    overall_status = "success"
    overall_notes = "completed overnight recovery-anchor autopilot queue"

    try:
        update_row(queue_rows, "wait_current_queue", status="running", started_at=now_iso())
        write_rows(summary_csv, QUEUE_FIELDS, queue_rows)
        waited_finished_at = wait_for_queue(
            wait_summary_csv=wait_summary_csv,
            wait_process_pattern=str(args.wait_process_pattern),
            poll_seconds=int(args.poll_seconds),
        )
        update_row(
            queue_rows,
            "wait_current_queue",
            status="success",
            finished_at=waited_finished_at,
            notes=f"waited queue finished successfully: {wait_run_root.name}",
        )
        write_rows(summary_csv, QUEUE_FIELDS, queue_rows)

        update_row(queue_rows, "analyze_current_results", status="running", started_at=now_iso())
        write_rows(summary_csv, QUEUE_FIELDS, queue_rows)

        candidates: List[Dict[str, object]] = []
        for root in stage1_train_roots:
            candidates.append(read_train_candidate(root / "summary.csv", "stage1"))
        for root in stage2_train_roots:
            candidates.append(read_train_candidate(root / "summary.csv", "stage2"))
        for root in stage3_train_roots:
            candidates.append(read_train_candidate(root / "summary.csv", "stage3"))

        stage2_best = select_best_candidate([item for item in candidates if str(item["stage"]) == "stage2"])
        stage3_best = select_best_candidate([item for item in candidates if str(item["stage"]) == "stage3"])
        selected = select_best_candidate(candidates)
        write_selection_csv(selection_csv, candidates, str(selected["summary_csv"]))

        confirmation_mode = float(stage3_best["best_metric"]) >= float(stage2_best["best_metric"])
        branch_name = "confirmation" if confirmation_mode else "rescue"

        write_log(
            logs_dir / "analyze_current_results.log",
            [
                f"[started_at] {now_iso()}",
                f"[branch] {branch_name}",
                f"[stage2_best] {json.dumps(stage2_best, ensure_ascii=True)}",
                f"[stage3_best] {json.dumps(stage3_best, ensure_ascii=True)}",
                f"[selected] {json.dumps(selected, ensure_ascii=True)}",
                f"[selection_csv] {selection_csv}",
                f"[finished_at] {now_iso()}",
            ],
        )
        analyze_note = (
            f"branch={branch_name} selected={Path(str(selected['summary_csv'])).parent.name} "
            f"stage2_best={stage2_best['best_metric']:.4f} stage3_best={stage3_best['best_metric']:.4f}"
        )
        update_row(
            queue_rows,
            "analyze_current_results",
            status="success",
            finished_at=now_iso(),
            notes=analyze_note,
        )
        write_rows(summary_csv, QUEUE_FIELDS, queue_rows)

        if confirmation_mode:
            mark_steps_skipped(queue_rows, summary_csv, rescue_steps, "confirmation branch selected")
        else:
            mark_steps_skipped(queue_rows, summary_csv, confirm_steps, "rescue branch selected")

        anchor_jsonl = final_dataset_root / "anchor_dataset.jsonl"
        robust_split_candidates = [
            {"target_val_fraction": 0.33, "min_val_positive": 4, "min_train_positive": 8, "max_auto_val_seqs": 4},
            {"target_val_fraction": 0.30, "min_val_positive": 4, "min_train_positive": 8, "max_auto_val_seqs": 5},
            {"target_val_fraction": 0.25, "min_val_positive": 4, "min_train_positive": 8, "max_auto_val_seqs": 5},
        ]
        robust_split_ready = try_build_split_step(
            queue_rows=queue_rows,
            summary_csv=summary_csv,
            step="build_stage3_robust_split",
            log_path=logs_dir / "build_stage3_robust_split.log",
            anchor_jsonl=anchor_jsonl,
            out_dir=robust_split_root,
            candidates=robust_split_candidates,
        )

        if confirmation_mode:
            confirm_hidden_dim = int(selected["hidden_dim"])
            confirm_dropout = float(selected["dropout"])
            for seed in CONFIRM_SEEDS:
                step = f"confirm_default_seed{seed}"
                train_summary = run_child_step(
                    queue_rows=queue_rows,
                    summary_csv=summary_csv,
                    step=step,
                    cmd=build_train_cmd(
                        split_root=final_split_root,
                        out_dir=run_root / step,
                        hidden_dim=confirm_hidden_dim,
                        dropout=confirm_dropout,
                        seed=int(seed),
                        epochs=int(args.confirm_epochs),
                        batch_size=int(args.train_batch_size),
                        lr=float(args.train_lr),
                        weight_decay=float(args.train_weight_decay),
                        device=str(args.train_device),
                    ),
                    log_path=logs_dir / f"{step}.log",
                    child_summary_csv=(run_root / step / "summary.csv"),
                )
                append_overnight_result(
                    overnight_results_csv,
                    {
                        "step": step,
                        "branch": branch_name,
                        "split_name": "stage3_default",
                        "summary_csv": str((run_root / step / "summary.csv").resolve()),
                        "hidden_dim": confirm_hidden_dim,
                        "dropout": confirm_dropout,
                        "seed": int(seed),
                        "best_metric": train_summary.get("best_metric", 0),
                        "val_balanced_accuracy": train_summary.get("val_balanced_accuracy", 0),
                        "val_f1": train_summary.get("val_f1", 0),
                        "train_rows": train_summary.get("train_rows", 0),
                        "val_rows": train_summary.get("val_rows", 0),
                        "notes": "default split confirmation",
                    },
                )

            if robust_split_ready is None:
                mark_steps_skipped(queue_rows, summary_csv, [f"confirm_robust_seed{seed}" for seed in CONFIRM_SEEDS], "robust split unavailable")
            else:
                for seed in CONFIRM_SEEDS:
                    step = f"confirm_robust_seed{seed}"
                    train_summary = run_child_step(
                        queue_rows=queue_rows,
                        summary_csv=summary_csv,
                        step=step,
                        cmd=build_train_cmd(
                            split_root=robust_split_root,
                            out_dir=run_root / step,
                            hidden_dim=confirm_hidden_dim,
                            dropout=confirm_dropout,
                            seed=int(seed),
                            epochs=int(args.confirm_epochs),
                            batch_size=int(args.train_batch_size),
                            lr=float(args.train_lr),
                            weight_decay=float(args.train_weight_decay),
                            device=str(args.train_device),
                        ),
                        log_path=logs_dir / f"{step}.log",
                        child_summary_csv=(run_root / step / "summary.csv"),
                    )
                    append_overnight_result(
                        overnight_results_csv,
                        {
                            "step": step,
                            "branch": branch_name,
                            "split_name": "stage3_robust",
                            "summary_csv": str((run_root / step / "summary.csv").resolve()),
                            "hidden_dim": confirm_hidden_dim,
                            "dropout": confirm_dropout,
                            "seed": int(seed),
                            "best_metric": train_summary.get("best_metric", 0),
                            "val_balanced_accuracy": train_summary.get("val_balanced_accuracy", 0),
                            "val_f1": train_summary.get("val_f1", 0),
                            "train_rows": train_summary.get("train_rows", 0),
                            "val_rows": train_summary.get("val_rows", 0),
                            "notes": "robust split confirmation",
                        },
                    )
        else:
            if robust_split_ready is not None:
                mark_step(
                    queue_rows,
                    summary_csv,
                    step="build_stage3_robust_split",
                    status="success",
                    notes=f"{read_single_summary(robust_split_root / 'summary.csv').get('val_seq_names', '')} rescue split ready",
                )
            for variant in RESCUE_VARIANTS:
                step = f"rescue_{variant['name']}"
                train_summary = run_child_step(
                    queue_rows=queue_rows,
                    summary_csv=summary_csv,
                    step=step,
                    cmd=build_train_cmd(
                        split_root=final_split_root,
                        out_dir=run_root / step,
                        hidden_dim=int(variant["hidden_dim"]),
                        dropout=float(variant["dropout"]),
                        seed=int(variant["seed"]),
                        epochs=int(args.rescue_epochs),
                        batch_size=int(args.train_batch_size),
                        lr=float(args.train_lr),
                        weight_decay=float(args.train_weight_decay),
                        device=str(args.train_device),
                    ),
                    log_path=logs_dir / f"{step}.log",
                    child_summary_csv=(run_root / step / "summary.csv"),
                )
                append_overnight_result(
                    overnight_results_csv,
                    {
                        "step": step,
                        "branch": branch_name,
                        "split_name": "stage3_default",
                        "summary_csv": str((run_root / step / "summary.csv").resolve()),
                        "hidden_dim": variant["hidden_dim"],
                        "dropout": variant["dropout"],
                        "seed": variant["seed"],
                        "best_metric": train_summary.get("best_metric", 0),
                        "val_balanced_accuracy": train_summary.get("val_balanced_accuracy", 0),
                        "val_f1": train_summary.get("val_f1", 0),
                        "train_rows": train_summary.get("train_rows", 0),
                        "val_rows": train_summary.get("val_rows", 0),
                        "notes": "default split rescue sweep",
                    },
                )

        overall_notes = (
            f"completed overnight recovery-anchor autopilot queue branch={branch_name} "
            f"selected={Path(str(selected['summary_csv'])).parent.name}"
        )
        append_registry(summary_csv, run_root, "success", overall_notes, args.registry_csv)
        return 0
    except Exception as exc:
        overall_status = "failed"
        overall_notes = f"overnight recovery-anchor autopilot failed: {exc}"
        for row in queue_rows:
            status = str(row.get("status", ""))
            if status == "running":
                row["status"] = "failed"
                row["finished_at"] = now_iso()
                row["notes"] = str(exc)
        write_rows(summary_csv, QUEUE_FIELDS, queue_rows)
        append_registry(summary_csv, run_root, overall_status, overall_notes, args.registry_csv)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
