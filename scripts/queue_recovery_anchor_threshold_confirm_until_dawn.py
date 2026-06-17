#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from datetime import datetime, timedelta
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


DEFAULT_WAIT_QUEUE_ROOT = REPO_ROOT / "outputs" / "queue_recovery_anchor_until_dawn_20260410_1"
DEFAULT_WAIT_SUMMARY_CSV = DEFAULT_WAIT_QUEUE_ROOT / "summary.csv"
DEFAULT_WAIT_RESULTS_CSV = DEFAULT_WAIT_QUEUE_ROOT / "overnight_results.csv"
DEFAULT_DEFAULT_SPLIT_ROOT = REPO_ROOT / "outputs" / "recovery_anchor_sequence_split_20260409_3"
DEFAULT_ROBUST_SPLIT_ROOT = DEFAULT_WAIT_QUEUE_ROOT / "stage3_robust_split"

SELECTION_FIELDS = [
    "rank",
    "source_step",
    "split_name",
    "summary_csv",
    "hidden_dim",
    "dropout",
    "seed",
    "best_metric",
    "val_balanced_accuracy",
    "val_f1",
    "selected",
]

RESULT_FIELDS = [
    "step",
    "split_name",
    "summary_csv",
    "hidden_dim",
    "dropout",
    "seed",
    "best_metric",
    "best_threshold",
    "val_balanced_accuracy",
    "val_f1",
    "train_rows",
    "val_rows",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run threshold-calibrated confirmation sweeps on the best recovery-anchor configs until the local dawn cutoff."
    )
    parser.add_argument("--out-root", default="")
    parser.add_argument("--wait-summary-csv", default=str(DEFAULT_WAIT_SUMMARY_CSV))
    parser.add_argument("--wait-results-csv", default=str(DEFAULT_WAIT_RESULTS_CSV))
    parser.add_argument("--default-split-root", default=str(DEFAULT_DEFAULT_SPLIT_ROOT))
    parser.add_argument("--robust-split-root", default=str(DEFAULT_ROBUST_SPLIT_ROOT))
    parser.add_argument("--stop-at", default="")
    parser.add_argument("--stop-hour-local", type=int, default=8)
    parser.add_argument("--max-configs", type=int, default=4)
    parser.add_argument("--seed-start", type=int, default=701)
    parser.add_argument("--seed-end", type=int, default=999)
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--train-lr", type=float, default=1e-3)
    parser.add_argument("--train-weight-decay", type=float, default=1e-4)
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
        "scripts/queue_recovery_anchor_threshold_confirm_until_dawn.py",
        "--dataset",
        "DanceTrack",
        "--split",
        "recovery_anchor_threshold_confirm",
        "--tracker-family",
        "deep_ocsort_preassoc_force_recovery_anchor",
        "--variant",
        run_root.name,
        "--tag",
        "recovery_anchor_threshold_confirm",
        "--run-root",
        str(run_root.resolve()),
        "--summary-csv",
        str(summary_csv.resolve()),
        "--notes",
        notes,
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=False)


def resolve_stop_at(stop_at_text: str, stop_hour_local: int) -> datetime:
    now = datetime.now().astimezone()
    if str(stop_at_text).strip():
        return datetime.fromisoformat(str(stop_at_text))
    cutoff = now.replace(hour=int(stop_hour_local), minute=0, second=0, microsecond=0)
    if cutoff <= now:
        cutoff += timedelta(days=1)
    return cutoff


def all_success(summary_csv: Path) -> bool:
    rows = read_rows(summary_csv)
    if not rows:
        raise FileNotFoundError(f"Missing queue summary rows: {summary_csv}")
    statuses = {str(row.get("status", "")).strip() for row in rows}
    return statuses == {"success"}


def wait_for_success(summary_csv: Path) -> str:
    if not all_success(summary_csv):
        raise RuntimeError(f"Waited queue is not fully successful: {summary_csv}")
    return child_finished_at(summary_csv)


def read_single_summary(summary_csv: Path) -> Dict[str, str]:
    rows = read_rows(summary_csv)
    if not rows:
        raise FileNotFoundError(f"Missing summary rows: {summary_csv}")
    return rows[0]


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


def ensure_queue_row(
    queue_rows: List[Dict[str, object]],
    summary_csv: Path,
    *,
    step: str,
    out_dir: Path,
    log_path: Path,
    notes: str,
) -> None:
    if any(str(row.get("step", "")) == step for row in queue_rows):
        return
    queue_rows.append(make_row(step=step, out_dir=out_dir, log_path=log_path, notes=notes))
    write_rows(summary_csv, QUEUE_FIELDS, queue_rows)


def load_candidate_rows(results_csv: Path) -> List[Dict[str, object]]:
    rows = read_rows(results_csv)
    if not rows:
        raise FileNotFoundError(f"Missing overnight results rows: {results_csv}")
    candidates: List[Dict[str, object]] = []
    for row in rows:
        candidates.append(
            {
                "source_step": str(row.get("step", "")),
                "split_name": str(row.get("split_name", "")),
                "summary_csv": str(row.get("summary_csv", "")),
                "hidden_dim": int(float(row.get("hidden_dim", 0) or 0)),
                "dropout": float(row.get("dropout", 0.0) or 0.0),
                "seed": int(float(row.get("seed", 0) or 0)),
                "best_metric": float(row.get("best_metric", 0.0) or 0.0),
                "val_balanced_accuracy": float(row.get("val_balanced_accuracy", 0.0) or 0.0),
                "val_f1": float(row.get("val_f1", 0.0) or 0.0),
            }
        )
    return candidates


def select_configs(candidates: Sequence[Dict[str, object]], max_configs: int) -> List[Dict[str, object]]:
    unique: Dict[tuple[int, float], Dict[str, object]] = {}
    for candidate in candidates:
        key = (int(candidate["hidden_dim"]), float(candidate["dropout"]))
        current = unique.get(key)
        if current is None:
            unique[key] = dict(candidate)
            continue
        old_key = (
            float(current["best_metric"]),
            float(current["val_f1"]),
            -int(current["hidden_dim"]),
            -float(current["dropout"]),
        )
        new_key = (
            float(candidate["best_metric"]),
            float(candidate["val_f1"]),
            -int(candidate["hidden_dim"]),
            -float(candidate["dropout"]),
        )
        if new_key > old_key:
            unique[key] = dict(candidate)
    selected = sorted(
        unique.values(),
        key=lambda item: (
            float(item["best_metric"]),
            float(item["val_f1"]),
            -int(item["hidden_dim"]),
            -float(item["dropout"]),
        ),
        reverse=True,
    )
    return [dict(item) for item in selected[: max(int(max_configs), 1)]]


def write_selection_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    output_rows: List[Dict[str, object]] = []
    for index, row in enumerate(rows, start=1):
        item = dict(row)
        item["rank"] = int(index)
        item["selected"] = 1
        output_rows.append(item)
    write_rows(path, SELECTION_FIELDS, output_rows)


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
        "threshold-calibrated recovery-anchor confirmation",
    ]


def child_note(summary_csv: Path) -> str:
    row = read_single_summary(summary_csv)
    return (
        f"best_metric={row.get('best_metric', '0')} "
        f"best_threshold={row.get('best_threshold', '0.5')} "
        f"val_bal_acc={row.get('val_balanced_accuracy', '0')} "
        f"val_f1={row.get('val_f1', '0')}"
    )


def upsert_result(results_csv: Path, result_row: Dict[str, object]) -> None:
    rows = read_rows(results_csv)
    output_rows: List[Dict[str, object]] = []
    replaced = False
    for row in rows:
        if str(row.get("step", "")) == str(result_row.get("step", "")):
            merged = dict(row)
            merged.update(result_row)
            output_rows.append({key: merged.get(key, "") for key in RESULT_FIELDS})
            replaced = True
        else:
            output_rows.append({key: row.get(key, "") for key in RESULT_FIELDS})
    if not replaced:
        output_rows.append({key: result_row.get(key, "") for key in RESULT_FIELDS})
    write_rows(results_csv, RESULT_FIELDS, output_rows)


def record_result(results_csv: Path, step: str, split_name: str, cfg: Dict[str, object], summary_csv: Path, notes: str) -> None:
    row = read_single_summary(summary_csv)
    upsert_result(
        results_csv,
        {
            "step": step,
            "split_name": split_name,
            "summary_csv": str(summary_csv.resolve()),
            "hidden_dim": int(cfg["hidden_dim"]),
            "dropout": float(cfg["dropout"]),
            "seed": int(cfg["seed"]),
            "best_metric": row.get("best_metric", 0),
            "best_threshold": row.get("best_threshold", 0.5),
            "val_balanced_accuracy": row.get("val_balanced_accuracy", 0),
            "val_f1": row.get("val_f1", 0),
            "train_rows": row.get("train_rows", 0),
            "val_rows": row.get("val_rows", 0),
            "notes": notes,
        },
    )


def run_child_step(
    *,
    queue_rows: List[Dict[str, object]],
    queue_summary_csv: Path,
    results_csv: Path,
    step: str,
    split_name: str,
    cfg: Dict[str, object],
    cmd: List[str],
    log_path: Path,
    child_summary_csv: Path,
) -> None:
    if child_summary_csv.is_file():
        try:
            ensure_child_success(child_summary_csv)
            update_row(
                queue_rows,
                step,
                status="success",
                finished_at=child_finished_at(child_summary_csv),
                notes=child_note(child_summary_csv),
            )
            write_rows(queue_summary_csv, QUEUE_FIELDS, queue_rows)
            record_result(results_csv, step, split_name, cfg, child_summary_csv, "resumed existing success")
            return
        except Exception:
            pass

    update_row(queue_rows, step, status="running", started_at=now_iso())
    write_rows(queue_summary_csv, QUEUE_FIELDS, queue_rows)
    rc = run_step(cmd, log_path, cwd=REPO_ROOT)
    if rc != 0:
        raise RuntimeError(f"{step} returned code {rc}")
    ensure_child_success(child_summary_csv)
    note = child_note(child_summary_csv)
    update_row(queue_rows, step, status="success", finished_at=child_finished_at(child_summary_csv), notes=note)
    write_rows(queue_summary_csv, QUEUE_FIELDS, queue_rows)
    record_result(results_csv, step, split_name, cfg, child_summary_csv, "threshold-calibrated confirmation")


def config_name(hidden_dim: int, dropout: float) -> str:
    return f"h{int(hidden_dim)}do{int(round(float(dropout) * 10)):02d}"


def main() -> int:
    args = parse_args()
    tag = timestamp_tag()
    run_root = (
        Path(args.out_root).expanduser().resolve()
        if args.out_root
        else (REPO_ROOT / "outputs" / f"queue_recovery_anchor_threshold_confirm_until_dawn_{tag}").resolve()
    )
    run_root.mkdir(parents=True, exist_ok=True)

    wait_summary_csv = Path(args.wait_summary_csv).expanduser().resolve()
    wait_results_csv = Path(args.wait_results_csv).expanduser().resolve()
    default_split_root = Path(args.default_split_root).expanduser().resolve()
    robust_split_root = Path(args.robust_split_root).expanduser().resolve()
    stop_at = resolve_stop_at(args.stop_at, int(args.stop_hour_local))

    logs_dir = run_root / "logs"
    queue_summary_csv = run_root / "summary.csv"
    selection_csv = run_root / "selection_summary.csv"
    results_csv = run_root / "results.csv"

    queue_rows: List[Dict[str, object]] = [
        make_row(
            step="wait_previous_queue",
            out_dir=wait_summary_csv.parent,
            log_path=logs_dir / "wait_previous_queue.log",
            notes="wait for the previous recovery-anchor rescue queue",
            summary_csv=wait_summary_csv,
        ),
        make_row(
            step="select_configs",
            out_dir=run_root,
            log_path=logs_dir / "select_configs.log",
            notes="select top recovery-anchor configs for threshold-calibrated confirmation",
            summary_csv=queue_summary_csv,
        ),
    ]

    write_rows(queue_summary_csv, QUEUE_FIELDS, queue_rows)
    write_rows(selection_csv, SELECTION_FIELDS, [])
    write_rows(results_csv, RESULT_FIELDS, [])
    append_registry(queue_summary_csv, run_root, "running", "started threshold-calibrated confirmation queue", args.registry_csv)

    completed_runs = 0
    overall_notes = "threshold-calibrated confirmation queue finished"

    try:
        update_row(queue_rows, "wait_previous_queue", status="running", started_at=now_iso())
        write_rows(queue_summary_csv, QUEUE_FIELDS, queue_rows)
        waited_finished_at = wait_for_success(wait_summary_csv)
        update_row(
            queue_rows,
            "wait_previous_queue",
            status="success",
            finished_at=waited_finished_at,
            notes=f"waited queue finished successfully: {wait_summary_csv.parent.name}",
        )
        write_rows(queue_summary_csv, QUEUE_FIELDS, queue_rows)

        update_row(queue_rows, "select_configs", status="running", started_at=now_iso())
        write_rows(queue_summary_csv, QUEUE_FIELDS, queue_rows)
        selected_configs = select_configs(load_candidate_rows(wait_results_csv), int(args.max_configs))
        if not selected_configs:
            raise RuntimeError(f"No candidate configs found in {wait_results_csv}")
        write_selection_csv(selection_csv, selected_configs)
        selection_notes = "|".join(
            f"{config_name(int(cfg['hidden_dim']), float(cfg['dropout']))}:{float(cfg['best_metric']):.4f}"
            for cfg in selected_configs
        )
        update_row(
            queue_rows,
            "select_configs",
            status="success",
            finished_at=now_iso(),
            notes=f"selected {len(selected_configs)} configs {selection_notes}",
        )
        write_rows(queue_summary_csv, QUEUE_FIELDS, queue_rows)

        for seed in range(int(args.seed_start), int(args.seed_end) + 1):
            if datetime.now().astimezone() >= stop_at:
                overall_notes = f"reached cutoff {stop_at.isoformat()} after {completed_runs} runs"
                break
            for cfg in selected_configs:
                if datetime.now().astimezone() >= stop_at:
                    overall_notes = f"reached cutoff {stop_at.isoformat()} after {completed_runs} runs"
                    break
                for split_name, split_root, split_alias in [
                    ("stage3_default", default_split_root, "dflt"),
                    ("stage3_robust", robust_split_root, "rbst"),
                ]:
                    if datetime.now().astimezone() >= stop_at:
                        overall_notes = f"reached cutoff {stop_at.isoformat()} after {completed_runs} runs"
                        break
                    cfg_run = dict(cfg)
                    cfg_run["seed"] = int(seed)
                    step = f"{split_alias}_{config_name(int(cfg['hidden_dim']), float(cfg['dropout']))}_seed{seed}"
                    out_dir = run_root / step
                    log_path = logs_dir / f"{step}.log"
                    ensure_queue_row(
                        queue_rows,
                        queue_summary_csv,
                        step=step,
                        out_dir=out_dir,
                        log_path=log_path,
                        notes=f"threshold-calibrated confirmation on {split_name} seed={seed}",
                    )
                    run_child_step(
                        queue_rows=queue_rows,
                        queue_summary_csv=queue_summary_csv,
                        results_csv=results_csv,
                        step=step,
                        split_name=split_name,
                        cfg=cfg_run,
                        cmd=build_train_cmd(
                            split_root=split_root,
                            out_dir=out_dir,
                            hidden_dim=int(cfg["hidden_dim"]),
                            dropout=float(cfg["dropout"]),
                            seed=int(seed),
                            epochs=int(args.epochs),
                            batch_size=int(args.train_batch_size),
                            lr=float(args.train_lr),
                            weight_decay=float(args.train_weight_decay),
                            device=str(args.train_device),
                        ),
                        log_path=log_path,
                        child_summary_csv=out_dir / "summary.csv",
                    )
                    completed_runs += 1
                else:
                    continue
                break
            else:
                continue
            break

        append_registry(queue_summary_csv, run_root, "success", overall_notes, args.registry_csv)
        return 0
    except Exception as exc:
        for row in queue_rows:
            if str(row.get("status", "")) == "running":
                row["status"] = "failed"
                row["finished_at"] = now_iso()
                row["notes"] = str(exc)
        write_rows(queue_summary_csv, QUEUE_FIELDS, queue_rows)
        append_registry(
            queue_summary_csv,
            run_root,
            "failed",
            f"threshold-calibrated confirmation queue failed: {exc}",
            args.registry_csv,
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
