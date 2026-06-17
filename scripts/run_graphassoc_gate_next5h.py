#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional


REPO_ROOT = Path("/gemini/code/FMtrack-main/FM-Track")
REGISTRY_CSV = REPO_ROOT / "outputs" / "experiment_registry.csv"
PLAN_CSV = REPO_ROOT / "outputs" / "experiment_plan.csv"

BUILD_DATASET_SCRIPT = REPO_ROOT / "scripts" / "build_graph_assoc_gate_dataset.py"
TRAIN_GATE_SCRIPT = REPO_ROOT / "scripts" / "train_graph_assoc_gate.py"
EVAL_SCRIPT = REPO_ROOT / "scripts" / "run_botsort_graphassoc_mot20_eval.py"

DEFAULT_SOURCE_MANIFEST = (
    REPO_ROOT
    / "outputs"
    / "graph_assoc_commit_mot17_expand_20260424_150607"
    / "source_manifest.csv"
)
DEFAULT_INIT_CHECKPOINT = (
    REPO_ROOT
    / "outputs"
    / "graphassoc_gate_next5h_20260424_230311"
    / "train_gate"
    / "best.pt"
)

QUEUE_FIELDS = [
    "step",
    "name",
    "status",
    "run_root",
    "summary_csv",
    "log_path",
    "started_at",
    "finished_at",
    "artifact_path",
    "artifact_path_2",
    "source_manifest",
    "train_jsonl",
    "val_jsonl",
    "checkpoint",
    "best_epoch",
    "best_metric",
    "best_threshold",
    "train_rows",
    "val_rows",
    "sources",
    "rows_total",
    "rules_pass_rows",
    "hard_negative_checkpoint",
    "hard_negative_score_threshold",
    "hard_negative_weight_multiplier",
    "train_hard_negative_boosted",
    "val_hard_negative_boosted",
    "val_accuracy",
    "val_balanced_accuracy",
    "val_precision",
    "val_recall",
    "val_f1",
    "delta_hota",
    "delta_assa",
    "delta_idf1",
    "delta_mota",
    "delta_ids",
    "delta_frag",
    "notes",
    "params_json",
]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Run the next five hours of graph-association gate experiments.")
    parser.add_argument("--run-root", default=str(REPO_ROOT / "outputs" / f"graphassoc_gate_next5h_{ts}"))
    parser.add_argument("--queue-name", default=f"graphassoc_gate_next5h_{ts}")
    parser.add_argument("--variant-name", default="graphassoc_gate_next5h")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--max-hours", type=float, default=5.0)
    parser.add_argument("--dataset", default="MOT20", choices=["MOT17", "MOT20"])
    parser.add_argument("--data-root", default="/gemini/code/datasets")
    parser.add_argument("--split", default="train")
    parser.add_argument("--split-part", default="val_half", choices=["full", "train_half", "val_half"])
    parser.add_argument(
        "--val-patterns",
        default="",
        help="Comma-separated substrings used to route rows_jsonl inputs to the validation split.",
    )
    parser.add_argument("--source-manifest", default=str(DEFAULT_SOURCE_MANIFEST))
    parser.add_argument(
        "--rows-jsonl",
        nargs="*",
        default=[],
        help="Optional direct candidate-row jsonl inputs. When set, the gate dataset is built from these files instead of a source manifest.",
    )
    parser.add_argument("--reuse-dataset-dir", default="", help="Reuse an existing gate dataset directory instead of rebuilding from source rows.")
    parser.add_argument("--init-checkpoint", default=str(DEFAULT_INIT_CHECKPOINT))
    parser.add_argument("--train-device", default="cuda")
    parser.add_argument("--eval-device", default="cuda")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--num-hidden-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--model-type", default="dual_head", choices=["single_head", "dual_head"])
    parser.add_argument("--pos-weight", type=float, default=1.0)
    parser.add_argument("--neutral-pos-weight", type=float, default=1.0)
    parser.add_argument("--neutral-loss-weight", type=float, default=0.5)
    parser.add_argument("--neutral-risk-weight", type=float, default=1.25)
    parser.add_argument("--rank-loss-weight", type=float, default=0.0)
    parser.add_argument("--rank-margin", type=float, default=0.1)
    parser.add_argument("--ordinal-rank-loss-weight", type=float, default=2.0)
    parser.add_argument("--ordinal-rank-margin", type=float, default=0.12)
    parser.add_argument("--positive-target", type=float, default=1.0)
    parser.add_argument("--neutral-target", type=float, default=0.25)
    parser.add_argument("--harmful-target", type=float, default=0.0)
    parser.add_argument("--neutral-weight", type=float, default=0.5)
    parser.add_argument("--harmful-weight", type=float, default=1.25)
    parser.add_argument("--positive-weight-scale", type=float, default=0.25)
    parser.add_argument("--hard-negative-checkpoint", default="")
    parser.add_argument("--hard-negative-score-threshold", type=float, default=-1.0)
    parser.add_argument("--hard-negative-weight-multiplier", type=float, default=1.0)
    parser.add_argument("--commit-score-margin", type=float, default=0.0)
    parser.add_argument("--reference-results-dirs", nargs="+", default=[])
    parser.add_argument("--seq-ids", nargs="+", type=int, default=[2, 5])
    parser.add_argument("--skip-eval", action="store_true", help="Only build dataset and train gate; skip MOT20 evaluation.")
    return parser.parse_args()


def write_rows(path: Path, fieldnames: Iterable[str], rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def update_row(rows: List[Dict[str, object]], step: str, **updates: object) -> None:
    for row in rows:
        if str(row.get("step", "")) == str(step):
            row.update(updates)
            return
    raise KeyError(f"Missing queue step: {step}")


def append_row(
    rows: List[Dict[str, object]],
    *,
    step: str,
    name: str,
    status: str,
    run_root: Path,
    summary_csv: Path,
    log_path: Path,
    artifact_path: str = "",
    artifact_path_2: str = "",
    source_manifest: str = "",
    train_jsonl: str = "",
    val_jsonl: str = "",
    checkpoint: str = "",
    best_epoch: str = "",
    best_metric: str = "",
    best_threshold: str = "",
    train_rows: str = "",
    val_rows: str = "",
    sources: str = "",
    rows_total: str = "",
    rules_pass_rows: str = "",
    hard_negative_checkpoint: str = "",
    hard_negative_score_threshold: str = "",
    hard_negative_weight_multiplier: str = "",
    train_hard_negative_boosted: str = "",
    val_hard_negative_boosted: str = "",
    val_accuracy: str = "",
    val_balanced_accuracy: str = "",
    val_precision: str = "",
    val_recall: str = "",
    val_f1: str = "",
    delta_hota: str = "",
    delta_assa: str = "",
    delta_idf1: str = "",
    delta_mota: str = "",
    delta_ids: str = "",
    delta_frag: str = "",
    notes: str = "",
    params_json: str = "",
) -> None:
    rows.append(
        {
            "step": step,
            "name": name,
            "status": status,
            "run_root": str(run_root),
            "summary_csv": str(summary_csv),
            "log_path": str(log_path),
            "started_at": "",
            "finished_at": "",
            "artifact_path": artifact_path,
            "artifact_path_2": artifact_path_2,
            "source_manifest": source_manifest,
            "train_jsonl": train_jsonl,
            "val_jsonl": val_jsonl,
            "checkpoint": checkpoint,
            "best_epoch": best_epoch,
            "best_metric": best_metric,
            "best_threshold": best_threshold,
            "train_rows": train_rows,
            "val_rows": val_rows,
            "sources": sources,
            "rows_total": rows_total,
            "rules_pass_rows": rules_pass_rows,
            "hard_negative_checkpoint": hard_negative_checkpoint,
            "hard_negative_score_threshold": hard_negative_score_threshold,
            "hard_negative_weight_multiplier": hard_negative_weight_multiplier,
            "train_hard_negative_boosted": train_hard_negative_boosted,
            "val_hard_negative_boosted": val_hard_negative_boosted,
            "val_accuracy": val_accuracy,
            "val_balanced_accuracy": val_balanced_accuracy,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_f1": val_f1,
            "delta_hota": delta_hota,
            "delta_assa": delta_assa,
            "delta_idf1": delta_idf1,
            "delta_mota": delta_mota,
            "delta_ids": delta_ids,
            "delta_frag": delta_frag,
            "notes": notes,
            "params_json": params_json,
        }
    )


def parse_single_row_csv(path: Path) -> Dict[str, str]:
    rows = read_csv_rows(path)
    return rows[0] if rows else {}


def parse_dataset_summary(summary_csv: Path) -> Dict[str, object]:
    row = parse_single_row_csv(summary_csv)
    return {
        "source_manifest": str(row.get("source_manifest", "")),
        "train_jsonl": str(row.get("train_jsonl", "")),
        "val_jsonl": str(row.get("val_jsonl", "")),
        "sources": int(float(row.get("sources", 0) or 0)),
        "rows_total": int(float(row.get("rows_total", 0) or 0)),
        "rules_pass_rows": int(float(row.get("rules_pass_rows", 0) or 0)),
        "hard_negative_checkpoint": str(row.get("hard_negative_checkpoint", "")),
        "hard_negative_score_threshold": float(row.get("hard_negative_score_threshold", 0.0) or 0.0),
        "hard_negative_weight_multiplier": float(row.get("hard_negative_weight_multiplier", 0.0) or 0.0),
        "train_hard_negative_boosted": int(float(row.get("train_hard_negative_boosted", 0) or 0)),
        "val_hard_negative_boosted": int(float(row.get("val_hard_negative_boosted", 0) or 0)),
        "train_rows": int(float(row.get("train_rows", 0) or 0)),
        "val_rows": int(float(row.get("val_rows", 0) or 0)),
        "status": str(row.get("status", "")).strip(),
    }


def parse_train_summary(summary_csv: Path) -> Dict[str, object]:
    row = parse_single_row_csv(summary_csv)
    return {
        "best_epoch": int(float(row.get("best_epoch", 0) or 0)) if str(row.get("best_epoch", "")).strip() else "",
        "best_metric": float(row.get("best_metric", 0.0) or 0.0),
        "best_threshold": float(row.get("best_threshold", 0.0) or 0.0),
        "train_rows": int(float(row.get("train_rows", 0) or 0)),
        "val_rows": int(float(row.get("val_rows", 0) or 0)),
        "val_accuracy": float(row.get("val_accuracy", 0.0) or 0.0),
        "val_balanced_accuracy": float(row.get("val_balanced_accuracy", 0.0) or 0.0),
        "val_precision": float(row.get("val_precision", 0.0) or 0.0),
        "val_recall": float(row.get("val_recall", 0.0) or 0.0),
        "val_f1": float(row.get("val_f1", 0.0) or 0.0),
        "status": str(row.get("status", "")).strip(),
    }


def parse_eval_metrics(run_root: Path) -> Dict[str, object]:
    compare_rows = read_csv_rows(run_root / "metrics_compare.csv")
    delta_rows = read_csv_rows(run_root / "metrics_delta.csv")
    graph_assoc_row = next((row for row in compare_rows if str(row.get("name", "")) == "graph_assoc"), {})
    delta_row = delta_rows[0] if delta_rows else {}
    return {
        "eval_hota": float(graph_assoc_row.get("HOTA", 0.0) or 0.0),
        "eval_assa": float(graph_assoc_row.get("AssA", 0.0) or 0.0),
        "eval_idf1": float(graph_assoc_row.get("IDF1", 0.0) or 0.0),
        "eval_mota": float(graph_assoc_row.get("MOTA", 0.0) or 0.0),
        "eval_ids": int(round(float(graph_assoc_row.get("IDs", 0.0) or 0.0))),
        "eval_frag": int(round(float(graph_assoc_row.get("Frag", 0.0) or 0.0))),
        "delta_hota": float(delta_row.get("delta_HOTA", 0.0) or 0.0),
        "delta_assa": float(delta_row.get("delta_AssA", 0.0) or 0.0),
        "delta_idf1": float(delta_row.get("delta_IDF1", 0.0) or 0.0),
        "delta_mota": float(delta_row.get("delta_MOTA", 0.0) or 0.0),
        "delta_ids": int(round(float(delta_row.get("delta_IDs", 0.0) or 0.0))),
        "delta_frag": int(round(float(delta_row.get("delta_Frag", 0.0) or 0.0))),
    }


def queue_plan_status(args: argparse.Namespace, status: str, summary_csv: Path, log_path: Path, notes: str = "") -> None:
    cmd = [
        args.python_bin,
        str(REPO_ROOT / "scripts" / "upsert_experiment_plan.py"),
        "--csv",
        str(PLAN_CSV),
        "--key",
        f"run_root:{Path(args.run_root).expanduser().resolve()}",
        "--status",
        status,
        "--kind",
        "analysis",
        "--script",
        "scripts/run_graphassoc_gate_next5h.py",
        "--dataset",
        str(args.dataset),
        "--split",
        "val_half",
        "--tracker-family",
        "BoT-SORT",
        "--variant",
        str(args.variant_name),
        "--tag",
        str(args.queue_name),
        "--run-root",
        str(Path(args.run_root).expanduser().resolve()),
        "--summary-csv",
        str(summary_csv),
        "--log-path",
        str(log_path),
        "--notes",
        notes,
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=False)


def queue_registry(args: argparse.Namespace, status: str, summary_csv: Path, log_path: Path, notes: str = "") -> None:
    cmd = [
        args.python_bin,
        str(REPO_ROOT / "scripts" / "append_experiment_record.py"),
        "--csv",
        str(REGISTRY_CSV),
        "--kind",
        "analysis",
        "--status",
        status,
        "--script",
        "scripts/run_graphassoc_gate_next5h.py",
        "--dataset",
        str(args.dataset),
        "--split",
        "val_half",
        "--tracker-family",
        "BoT-SORT",
        "--variant",
        str(args.variant_name),
        "--tag",
        str(args.queue_name),
        "--run-root",
        str(Path(args.run_root).expanduser().resolve()),
        "--summary-csv",
        str(summary_csv),
        "--log-path",
        str(log_path),
        "--notes",
        notes,
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=False)


def run_step(cmd: List[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"[started_at] {now_iso()}\n")
        handle.write("[cmd] " + " ".join(cmd) + "\n\n")
        handle.flush()
        proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=handle, stderr=subprocess.STDOUT)
        handle.write(f"\n[finished_at] {now_iso()}\n")
        handle.write(f"[return_code] {proc.returncode}\n")
    return int(proc.returncode)


def sync_records(args: argparse.Namespace, status: str, summary_csv: Path, log_path: Path, notes: str = "") -> None:
    queue_plan_status(args, status, summary_csv, log_path, notes=notes)
    queue_registry(args, status, summary_csv, log_path, notes=notes)


def build_dataset_cmd(args: argparse.Namespace, dataset_dir: Path) -> List[str]:
    cmd = [
        args.python_bin,
        str(BUILD_DATASET_SCRIPT),
        "--out-dir",
        str(dataset_dir),
        "--dataset",
        str(args.dataset),
        "--data-root",
        str(args.data_root),
        "--split",
        str(args.split),
        "--split-part",
        str(args.split_part),
        "--positive-target",
        str(args.positive_target),
        "--neutral-target",
        str(args.neutral_target),
        "--harmful-target",
        str(args.harmful_target),
        "--neutral-weight",
        str(args.neutral_weight),
        "--harmful-weight",
        str(args.harmful_weight),
        "--positive-weight-scale",
        str(args.positive_weight_scale),
        "--hard-negative-checkpoint",
        str(args.hard_negative_checkpoint),
        "--hard-negative-score-threshold",
        str(args.hard_negative_score_threshold),
        "--hard-negative-weight-multiplier",
        str(args.hard_negative_weight_multiplier),
    ]
    reuse_dataset_dir = str(args.reuse_dataset_dir or "").strip()
    if reuse_dataset_dir:
        cmd.extend(["--reuse-dataset-dir", str(Path(reuse_dataset_dir).expanduser().resolve())])
    elif list(args.rows_jsonl or []):
        cmd.extend(["--rows-jsonl", *[str(Path(v).expanduser().resolve()) for v in list(args.rows_jsonl or [])]])
        if str(args.val_patterns or "").strip():
            cmd.extend(["--val-patterns", str(args.val_patterns).strip()])
    else:
        cmd.extend(["--source-manifest", str(Path(args.source_manifest).expanduser().resolve())])
    return cmd


def train_gate_cmd(args: argparse.Namespace, dataset_dir: Path, train_dir: Path, train_jsonl: Path, val_jsonl: Path) -> List[str]:
    cmd = [
        args.python_bin,
        str(TRAIN_GATE_SCRIPT),
        "--train-jsonl",
        str(train_jsonl),
        "--val-jsonl",
        str(val_jsonl),
        "--out-dir",
        str(train_dir),
        "--device",
        str(args.train_device),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--hidden-dim",
        str(args.hidden_dim),
        "--num-hidden-layers",
        str(args.num_hidden_layers),
        "--dropout",
        str(args.dropout),
        "--model-type",
        str(args.model_type),
        "--pos-weight",
        str(args.pos_weight),
        "--neutral-pos-weight",
        str(args.neutral_pos_weight),
        "--neutral-loss-weight",
        str(args.neutral_loss_weight),
        "--neutral-risk-weight",
        str(args.neutral_risk_weight),
        "--rank-loss-weight",
        str(args.rank_loss_weight),
        "--rank-margin",
        str(args.rank_margin),
        "--ordinal-rank-loss-weight",
        str(args.ordinal_rank_loss_weight),
        "--ordinal-rank-margin",
        str(args.ordinal_rank_margin),
        "--seed",
        "42",
        "--registry-dataset",
        str(args.dataset),
        "--registry-split",
        "graph_assoc_gate_jsonl",
        "--registry-notes",
        f"gate training from {dataset_dir.name} with warm start",
    ]
    if str(args.init_checkpoint or "").strip():
        cmd.extend(["--init-checkpoint", str(Path(args.init_checkpoint).expanduser().resolve())])
    return cmd


def eval_cmd(args: argparse.Namespace, eval_dir: Path, checkpoint: Path) -> List[str]:
    cmd = [
        args.python_bin,
        str(EVAL_SCRIPT),
        "--run-root",
        str(eval_dir),
        "--experiment-name",
        f"{args.queue_name}_mot20_replace_rules",
        "--variant-name",
        "botsort_graphassoc_gate_replace_rules",
        "--seq-ids",
        *[str(v) for v in args.seq_ids],
        "--graph-assoc-commit-checkpoint",
        str(checkpoint),
        "--graph-assoc-commit-device",
        str(args.eval_device),
        "--graph-assoc-commit-score-margin",
        str(args.commit_score_margin),
        "--graph-assoc-commit-replace-rules",
    ]
    if list(args.reference_results_dirs or []):
        cmd.extend(["--reference-results-dirs", *[str(Path(v).expanduser().resolve()) for v in args.reference_results_dirs]])
    return cmd


def main() -> None:
    args = parse_args()
    queue_root = Path(args.run_root).expanduser().resolve()
    queue_root.mkdir(parents=True, exist_ok=True)
    logs_dir = queue_root / "logs"
    queue_log = logs_dir / "queue.log"
    summary_csv = queue_root / "summary.csv"
    dataset_dir = queue_root / "gate_dataset"
    train_dir = queue_root / "train_gate"
    eval_dir = queue_root / "mot20_eval"
    deadline = time.time() + max(0.5, float(args.max_hours)) * 3600.0

    rows: List[Dict[str, object]] = []
    queue_params = json.dumps(
        {
            "source_manifest": str(Path(args.source_manifest).expanduser().resolve()),
            "rows_jsonl": [str(Path(v).expanduser().resolve()) for v in list(args.rows_jsonl or [])],
            "reuse_dataset_dir": str(Path(args.reuse_dataset_dir).expanduser().resolve()) if str(args.reuse_dataset_dir or "").strip() else "",
            "init_checkpoint": str(Path(args.init_checkpoint).expanduser().resolve()) if str(args.init_checkpoint or "").strip() else "",
            "dataset": str(args.dataset),
            "split": str(args.split),
            "split_part": str(args.split_part),
            "val_patterns": str(args.val_patterns),
            "train_device": str(args.train_device),
            "eval_device": str(args.eval_device),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "hidden_dim": int(args.hidden_dim),
            "num_hidden_layers": int(args.num_hidden_layers),
            "dropout": float(args.dropout),
            "model_type": str(args.model_type),
            "positive_target": float(args.positive_target),
            "neutral_target": float(args.neutral_target),
            "harmful_target": float(args.harmful_target),
            "neutral_weight": float(args.neutral_weight),
            "harmful_weight": float(args.harmful_weight),
            "positive_weight_scale": float(args.positive_weight_scale),
            "hard_negative_checkpoint": str(Path(args.hard_negative_checkpoint).expanduser().resolve())
            if str(args.hard_negative_checkpoint or "").strip()
            else "",
            "hard_negative_score_threshold": float(args.hard_negative_score_threshold),
            "hard_negative_weight_multiplier": float(args.hard_negative_weight_multiplier),
            "commit_score_margin": float(args.commit_score_margin),
            "seq_ids": [int(v) for v in args.seq_ids],
            "skip_eval": bool(args.skip_eval),
        },
        ensure_ascii=False,
        sort_keys=True,
    )

    append_row(
        rows,
        step="queue",
        name=str(args.queue_name),
        status="running",
        run_root=queue_root,
        summary_csv=summary_csv,
        log_path=queue_log,
        artifact_path=str(queue_root),
        artifact_path_2=str(Path(args.source_manifest).expanduser().resolve()),
        source_manifest=str(Path(args.source_manifest).expanduser().resolve()),
        notes="gate queue started",
        params_json=queue_params,
    )
    append_row(
        rows,
        step="build_dataset",
        name="build_graph_assoc_gate_dataset",
        status="pending",
        run_root=queue_root,
        summary_csv=dataset_dir / "summary.csv",
        log_path=logs_dir / "build_dataset.log",
        artifact_path=str(dataset_dir / "summary.csv"),
        artifact_path_2=str(dataset_dir / "train.jsonl"),
        source_manifest=str(Path(args.source_manifest).expanduser().resolve()),
        notes="build gate dataset from current expanded manifest",
        params_json=queue_params,
    )
    append_row(
        rows,
        step="train_gate",
        name="train_graph_assoc_gate_dual_head",
        status="pending",
        run_root=queue_root,
        summary_csv=train_dir / "summary.csv",
        log_path=logs_dir / "train_gate.log",
        artifact_path=str(train_dir / "best.pt"),
        artifact_path_2=str(train_dir / "summary.csv"),
        source_manifest=str(Path(args.source_manifest).expanduser().resolve()),
        train_jsonl=str(dataset_dir / "train.jsonl"),
        val_jsonl=str(dataset_dir / "val.jsonl"),
        checkpoint=str(train_dir / "best.pt"),
        notes="warm-start dual-head gate training",
        params_json=queue_params,
    )
    append_row(
        rows,
        step="eval_replace_rules",
        name="eval_graphassoc_gate_replace_rules",
        status="skipped" if bool(args.skip_eval) else "pending",
        run_root=queue_root,
        summary_csv=eval_dir / "summary.csv",
        log_path=logs_dir / "eval_replace_rules.log",
        artifact_path=str(eval_dir / "metrics_compare.csv"),
        artifact_path_2=str(eval_dir / "metrics_delta.csv"),
        source_manifest=str(Path(args.source_manifest).expanduser().resolve()),
        checkpoint=str(train_dir / "best.pt"),
        notes=(
            "MOT20 evaluation skipped by flag"
            if bool(args.skip_eval)
            else "MOT20 evaluation with learned gate replacing hand-coded rules"
        ),
        params_json=queue_params,
    )
    write_rows(summary_csv, QUEUE_FIELDS, rows)
    sync_records(args, "running", summary_csv, queue_log, notes="gate queue started")

    try:
        # Step 1: build dataset.
        if time.time() >= deadline:
            raise TimeoutError("Queue deadline reached before dataset build.")
        update_row(rows, "build_dataset", status="running", started_at=now_iso())
        write_rows(summary_csv, QUEUE_FIELDS, rows)
        rc = run_step(build_dataset_cmd(args, dataset_dir), logs_dir / "build_dataset.log")
        if rc != 0:
            raise RuntimeError(f"build_graph_assoc_gate_dataset failed with return code {rc}")
        dataset_summary = parse_dataset_summary(dataset_dir / "summary.csv")
        update_row(
            rows,
            "build_dataset",
            status="success",
            finished_at=now_iso(),
            artifact_path=str(dataset_dir / "summary.csv"),
            artifact_path_2=str(dataset_dir / "train.jsonl"),
            source_manifest=dataset_summary["source_manifest"] or str(Path(args.source_manifest).expanduser().resolve()),
            train_jsonl=dataset_summary["train_jsonl"],
            val_jsonl=dataset_summary["val_jsonl"],
            sources=str(dataset_summary["sources"]),
            rows_total=str(dataset_summary["rows_total"]),
            rules_pass_rows=str(dataset_summary["rules_pass_rows"]),
            hard_negative_checkpoint=dataset_summary["hard_negative_checkpoint"],
            hard_negative_score_threshold=str(dataset_summary["hard_negative_score_threshold"]),
            hard_negative_weight_multiplier=str(dataset_summary["hard_negative_weight_multiplier"]),
            train_hard_negative_boosted=str(dataset_summary["train_hard_negative_boosted"]),
            val_hard_negative_boosted=str(dataset_summary["val_hard_negative_boosted"]),
            train_rows=str(dataset_summary["train_rows"]),
            val_rows=str(dataset_summary["val_rows"]),
            notes=(
                f"gate dataset ready, sources={dataset_summary['sources']} "
                f"train_rows={dataset_summary['train_rows']} val_rows={dataset_summary['val_rows']}"
            ),
        )
        write_rows(summary_csv, QUEUE_FIELDS, rows)
        sync_records(
            args,
            "running",
            summary_csv,
            queue_log,
            notes=f"dataset ready, train_rows={dataset_summary['train_rows']} val_rows={dataset_summary['val_rows']}",
        )

        # Step 2: train gate.
        if time.time() >= deadline:
            raise TimeoutError("Queue deadline reached before gate training.")
        train_jsonl = dataset_dir / "train.jsonl"
        val_jsonl = dataset_dir / "val.jsonl"
        update_row(rows, "train_gate", status="running", started_at=now_iso())
        write_rows(summary_csv, QUEUE_FIELDS, rows)
        train_cmd = train_gate_cmd(args, dataset_dir, train_dir, train_jsonl, val_jsonl)
        rc = run_step(train_cmd, logs_dir / "train_gate.log")
        if rc != 0:
            raise RuntimeError(f"train_graph_assoc_gate failed with return code {rc}")
        train_summary = parse_train_summary(train_dir / "summary.csv")
        best_ckpt = train_dir / "best.pt"
        update_row(
            rows,
            "train_gate",
            status="success",
            finished_at=now_iso(),
            artifact_path=str(best_ckpt),
            artifact_path_2=str(train_dir / "summary.csv"),
            train_jsonl=str(train_jsonl),
            val_jsonl=str(val_jsonl),
            checkpoint=str(best_ckpt),
            best_epoch=str(train_summary["best_epoch"]),
            best_metric=str(train_summary["best_metric"]),
            best_threshold=str(train_summary["best_threshold"]),
            train_rows=str(train_summary["train_rows"]),
            val_rows=str(train_summary["val_rows"]),
            val_accuracy=str(train_summary["val_accuracy"]),
            val_balanced_accuracy=str(train_summary["val_balanced_accuracy"]),
            val_precision=str(train_summary["val_precision"]),
            val_recall=str(train_summary["val_recall"]),
            val_f1=str(train_summary["val_f1"]),
            notes=(
                f"gate training complete, best_epoch={train_summary['best_epoch']} "
                f"best_metric={train_summary['best_metric']:.4f} best_threshold={train_summary['best_threshold']:.3f}"
            ),
        )
        write_rows(summary_csv, QUEUE_FIELDS, rows)
        sync_records(
            args,
            "running",
            summary_csv,
            queue_log,
            notes=(
                f"gate trained, best_epoch={train_summary['best_epoch']} "
                f"best_metric={train_summary['best_metric']:.4f}"
            ),
        )

        if bool(args.skip_eval):
            final_note = (
                f"gate train-only queue completed, best_epoch={train_summary['best_epoch']}, "
                f"best_metric={train_summary['best_metric']:.4f}; eval skipped"
            )
            update_row(rows, "queue", status="completed", finished_at=now_iso(), notes=final_note)
            write_rows(summary_csv, QUEUE_FIELDS, rows)
            sync_records(args, "completed", summary_csv, queue_log, notes=final_note)
            return 0

        # Step 3: run MOT20 evaluation with learned gate replacing rules.
        if time.time() >= deadline:
            raise TimeoutError("Queue deadline reached before MOT20 evaluation.")
        eval_checkpoint = train_dir / "best.pt"
        update_row(rows, "eval_replace_rules", status="running", started_at=now_iso())
        write_rows(summary_csv, QUEUE_FIELDS, rows)
        eval_run_root = eval_dir
        eval_command = eval_cmd(args, eval_run_root, eval_checkpoint)
        rc = run_step(eval_command, logs_dir / "eval_replace_rules.log")
        if rc != 0:
            raise RuntimeError(f"run_botsort_graphassoc_mot20_eval failed with return code {rc}")
        eval_metrics = parse_eval_metrics(eval_run_root)
        update_row(
            rows,
            "eval_replace_rules",
            status="success",
            finished_at=now_iso(),
            artifact_path=str(eval_run_root / "metrics_compare.csv"),
            artifact_path_2=str(eval_run_root / "metrics_delta.csv"),
            checkpoint=str(eval_checkpoint),
            delta_hota=str(eval_metrics["delta_hota"]),
            delta_assa=str(eval_metrics["delta_assa"]),
            delta_idf1=str(eval_metrics["delta_idf1"]),
            delta_mota=str(eval_metrics["delta_mota"]),
            delta_ids=str(eval_metrics["delta_ids"]),
            delta_frag=str(eval_metrics["delta_frag"]),
            notes=(
                f"MOT20 eval complete, delta_HOTA={eval_metrics['delta_hota']:.3f} "
                f"delta_IDF1={eval_metrics['delta_idf1']:.3f}"
            ),
        )
        write_rows(summary_csv, QUEUE_FIELDS, rows)

        final_note = (
            f"gate queue completed, delta_HOTA={eval_metrics['delta_hota']:.3f}, "
            f"best_epoch={train_summary['best_epoch']}"
        )
        update_row(rows, "queue", status="completed", finished_at=now_iso(), notes=final_note)
        write_rows(summary_csv, QUEUE_FIELDS, rows)
        sync_records(args, "completed", summary_csv, queue_log, notes=final_note)
    except Exception as exc:
        finished_at = now_iso()
        for row in rows:
            status = str(row.get("status", "")).strip().lower()
            if status == "running":
                row["status"] = "failed"
                row["finished_at"] = finished_at
                row["notes"] = f"{row.get('notes', '')} | failed: {exc}".strip()
            elif status == "pending":
                row["status"] = "cancelled"
                row["finished_at"] = finished_at
                row["notes"] = f"{row.get('notes', '')} | cancelled_after_failure".strip()
        update_row(rows, "queue", status="failed", finished_at=finished_at, notes=str(exc))
        write_rows(summary_csv, QUEUE_FIELDS, rows)
        sync_records(args, "failed", summary_csv, queue_log, notes=str(exc))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
