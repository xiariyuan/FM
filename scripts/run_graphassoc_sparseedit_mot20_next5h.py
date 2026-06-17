#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List


REPO_ROOT = Path("/gemini/code/FMtrack-main/FM-Track")
REGISTRY_CSV = REPO_ROOT / "outputs" / "experiment_registry.csv"
PLAN_CSV = REPO_ROOT / "outputs" / "experiment_plan.csv"

BUILD_DATASET_SCRIPT = REPO_ROOT / "scripts" / "build_local_conflict_set_predictor_dataset.py"
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_local_conflict_set_predictor.py"
EVAL_SCRIPT = REPO_ROOT / "scripts" / "run_botsort_graphassoc_mot20_eval.py"

DEFAULT_SOURCE_MANIFEST = (
    REPO_ROOT
    / "outputs"
    / "20260509_100047_graphassoc_bridge_mot20_next5h"
    / "set_predictor_inputs"
    / "source_manifest.csv"
)

SUMMARY_FIELDS = [
    "step",
    "name",
    "status",
    "run_root",
    "summary_csv",
    "log_path",
    "started_at",
    "finished_at",
    "teacher_mode",
    "split_target_key",
    "cluster_target_key",
    "assignment_target_key",
    "assignment_row_mask_key",
    "edge_loss_mask_key",
    "edge_target_key",
    "train_sequences",
    "val_sequences",
    "source_manifest",
    "dataset_dir",
    "train_dir",
    "eval_dir",
    "checkpoint",
    "best_epoch",
    "best_metric",
    "train_examples",
    "val_examples",
    "train_target_positives",
    "val_target_positives",
    "trigger_pass_clusters",
    "cluster_utility_gain",
    "cluster_edit_gain",
    "cluster_edit_utility_gain",
    "cluster_sparse_utility_gain",
    "val_cluster_gate_thresh_calibrated",
    "val_cluster_gate_temp",
    "val_cluster_gate_bias",
    "val_cluster_gate_precision_cal",
    "val_cluster_gate_recall_cal",
    "val_cluster_gate_f0_5",
    "val_cluster_gate_utility_cal",
    "val_cluster_gate_coverage_cal",
    "val_cluster_gate_bounded_utility",
    "eval_HOTA",
    "eval_AssA",
    "eval_IDF1",
    "eval_MOTA",
    "eval_IDSW",
    "eval_gate_filtered_clusters",
    "eval_replaced_clusters",
    "delta_HOTA",
    "delta_AssA",
    "delta_IDF1",
    "delta_MOTA",
    "delta_IDSW",
    "delta_Frag",
    "notes",
    "params_json",
]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_args() -> argparse.Namespace:
    ts = timestamp()
    parser = argparse.ArgumentParser(description="Run a sparse-edit MOT20 set-predictor batch.")
    parser.add_argument("--run-root", default=str(REPO_ROOT / "outputs" / "mot20" / f"{ts}_graphassoc_sparseedit"))
    parser.add_argument("--queue-name", default=f"graphassoc_sparseedit_{ts}")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--source-manifest", default=str(DEFAULT_SOURCE_MANIFEST))
    parser.add_argument("--teacher-mode", choices=["edit_utility", "sparse_edit"], default="sparse_edit")
    parser.add_argument("--train-sequences", default="MOT20-01,MOT20-02,MOT20-03")
    parser.add_argument("--val-sequences", default="MOT20-05")
    parser.add_argument("--feature-version", default="graphassoc_commit_v1")
    parser.add_argument("--dataset-tag", default="graphassoc_commit_set_predictor_mot20")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-conflict-blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--min-detections", type=int, default=2)
    parser.add_argument("--min-committed-matches", type=int, default=2)
    parser.add_argument("--max-detections", type=int, default=8)
    parser.add_argument("--max-tracks", type=int, default=32)
    parser.add_argument("--cluster-gate-thresh", type=float, default=0.5)
    parser.add_argument("--cluster-gate-calibration", choices=["none", "temp_bias"], default="temp_bias")
    parser.add_argument(
        "--cluster-gate-select-metric",
        choices=["f0.5", "utility", "bounded_utility"],
        default="bounded_utility",
    )
    parser.add_argument("--cluster-gate-beta", type=float, default=0.5)
    parser.add_argument("--cluster-gate-fp-weight", type=float, default=2.0)
    parser.add_argument("--cluster-gate-search-min", type=float, default=0.01)
    parser.add_argument("--cluster-gate-search-max", type=float, default=0.80)
    parser.add_argument("--cluster-gate-search-steps", type=int, default=19)
    parser.add_argument("--cluster-gate-loss-mode", choices=["bce", "weighted_bce"], default="weighted_bce")
    parser.add_argument("--cluster-gate-positive-weight", type=float, default=4.0)
    parser.add_argument("--cluster-gate-negative-weight", type=float, default=1.0)
    parser.add_argument(
        "--train-positive-cluster-oversample",
        type=float,
        default=1.5,
        help="Oversample positive clusters less aggressively than the bridge run.",
    )
    parser.add_argument(
        "--model-selection-metric",
        choices=[
            "val_loss",
            "val_cluster_gate_f0_5",
            "val_cluster_gate_utility",
            "val_selective_utility_targetcov",
            "selective_utility_cov",
            "commit_viability_utility_cov",
            "hybrid_gate_f0_5_loss",
            "hybrid_gate_utility_loss",
        ],
        default="selective_utility_cov",
    )
    parser.add_argument("--target-gate-coverage-min", type=float, default=0.01)
    parser.add_argument("--target-gate-coverage-max", type=float, default=0.05)
    parser.add_argument("--coverage-penalty-weight", type=float, default=1.0)
    parser.add_argument("--keep-row-loss-weight", type=float, default=0.0)
    parser.add_argument("--edit-row-loss-weight", type=float, default=2.0)
    parser.add_argument("--edit-edge-positive-weight", type=float, default=32.0)
    parser.add_argument("--loss-assign-weight", type=float, default=1.0)
    parser.add_argument("--loss-edge-weight", type=float, default=0.5)
    parser.add_argument("--loss-cluster-weight", type=float, default=0.25)
    parser.add_argument("--loss-margin-weight", type=float, default=0.25)
    parser.add_argument("--margin-commit", type=float, default=0.2)
    parser.add_argument("--margin-row", type=float, default=0.2)
    parser.add_argument("--margin-defer", type=float, default=0.2)
    parser.add_argument("--margin-host-edit", type=float, default=0.1)
    parser.add_argument("--edge-focal-alpha", type=float, default=0.25)
    parser.add_argument("--edge-focal-gamma", type=float, default=2.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--skip-step-grad-norm", type=float, default=1000000.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-device", default="cuda")
    parser.add_argument("--eval-device", default="cuda")
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


def read_single_row(path: Path) -> Dict[str, str]:
    rows = read_csv_rows(path)
    return rows[0] if rows else {}


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


def write_record(path: Path, row: Dict[str, object]) -> None:
    write_rows(path, SUMMARY_FIELDS, [row])


def upsert_plan(args: argparse.Namespace, status: str, summary_csv: Path, log_path: Path, notes: str = "") -> None:
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
        "scripts/run_graphassoc_sparseedit_mot20_next5h.py",
        "--dataset",
        "MOT20",
        "--split",
        "val_half",
        "--tracker-family",
        "BoT-SORT",
        "--variant",
        "graphassoc_sparseedit_mot20",
        "--tag",
        args.queue_name,
        "--run-root",
        str(Path(args.run_root).expanduser().resolve()),
        "--summary-csv",
        str(summary_csv),
        "--log-path",
        str(log_path),
        "--notes",
        notes,
        "--extra",
        f"teacher_mode={args.teacher_mode}",
        f"source_manifest={args.source_manifest}",
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=False)


def append_registry(args: argparse.Namespace, status: str, summary_csv: Path, log_path: Path, notes: str, checkpoint: Path) -> None:
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
        "scripts/run_graphassoc_sparseedit_mot20_next5h.py",
        "--dataset",
        "MOT20",
        "--split",
        "val_half",
        "--tracker-family",
        "BoT-SORT",
        "--variant",
        "graphassoc_sparseedit_mot20",
        "--tag",
        args.queue_name,
        "--run-root",
        str(Path(args.run_root).expanduser().resolve()),
        "--summary-csv",
        str(summary_csv),
        "--checkpoint",
        str(checkpoint) if checkpoint.is_file() else "",
        "--log-path",
        str(log_path),
        "--notes",
        notes,
        "--extra",
        f"teacher_mode={args.teacher_mode}",
        f"source_manifest={args.source_manifest}",
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=False)


def target_keys_for_teacher_mode(teacher_mode: str) -> dict[str, str]:
    if teacher_mode == "edit_utility":
        return {
            "cluster_target_key": "cluster_should_intervene_edit",
            "assignment_target_key": "target_by_det_edit",
            "assignment_row_mask_key": "row_edit_mask",
            "edge_loss_mask_key": "row_edit_mask",
            "edge_target_key": "edge_is_edit_commit",
        }
    return {
        "cluster_target_key": "cluster_should_intervene_sparse",
        "assignment_target_key": "target_by_det_sparse_edit",
        "assignment_row_mask_key": "row_sparse_edit_mask",
        "edge_loss_mask_key": "row_sparse_edit_mask",
        "edge_target_key": "edge_is_sparse_edit",
    }


def main() -> int:
    args = parse_args()
    run_root = Path(args.run_root).expanduser().resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    logs_dir = run_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = run_root / "summary.csv"
    result_csv = run_root / "result.csv"
    queue_log = logs_dir / "queue.log"
    dataset_dir = run_root / "01_dataset"
    train_dir = run_root / "02_train"
    eval_dir = run_root / "03_eval_valhalf"

    key_map = target_keys_for_teacher_mode(args.teacher_mode)
    row: Dict[str, Any] = {
        "step": "01_init",
        "name": "graphassoc_sparseedit_mot20",
        "status": "running",
        "run_root": str(run_root),
        "summary_csv": str(summary_csv),
        "log_path": str(queue_log),
        "started_at": now_iso(),
        "finished_at": "",
        "teacher_mode": args.teacher_mode,
        "split_target_key": key_map["cluster_target_key"],
        "cluster_target_key": key_map["cluster_target_key"],
        "assignment_target_key": key_map["assignment_target_key"],
        "assignment_row_mask_key": key_map["assignment_row_mask_key"],
        "edge_loss_mask_key": key_map["edge_loss_mask_key"],
        "edge_target_key": key_map["edge_target_key"],
        "train_sequences": args.train_sequences,
        "val_sequences": args.val_sequences,
        "source_manifest": str(Path(args.source_manifest).expanduser().resolve()),
        "dataset_dir": str(dataset_dir),
        "train_dir": str(train_dir),
        "eval_dir": str(eval_dir),
        "checkpoint": "",
        "best_epoch": "",
        "best_metric": "",
        "train_examples": "",
        "val_examples": "",
        "train_target_positives": "",
        "val_target_positives": "",
        "trigger_pass_clusters": "",
        "cluster_utility_gain": "",
        "cluster_edit_gain": "",
        "cluster_edit_utility_gain": "",
        "cluster_sparse_utility_gain": "",
        "val_cluster_gate_thresh_calibrated": "",
        "val_cluster_gate_temp": "",
        "val_cluster_gate_bias": "",
        "val_cluster_gate_precision_cal": "",
        "val_cluster_gate_recall_cal": "",
        "val_cluster_gate_f0_5": "",
        "val_cluster_gate_utility_cal": "",
        "val_cluster_gate_coverage_cal": "",
        "val_cluster_gate_bounded_utility": "",
        "eval_HOTA": "",
        "eval_AssA": "",
        "eval_IDF1": "",
        "eval_MOTA": "",
        "eval_IDSW": "",
        "eval_gate_filtered_clusters": "",
        "eval_replaced_clusters": "",
        "delta_HOTA": "",
        "delta_AssA": "",
        "delta_IDF1": "",
        "delta_MOTA": "",
        "delta_IDSW": "",
        "delta_Frag": "",
        "notes": "",
        "params_json": json.dumps(
            {
                "source_manifest": str(Path(args.source_manifest).expanduser().resolve()),
                "teacher_mode": args.teacher_mode,
                "train_sequences": args.train_sequences,
                "val_sequences": args.val_sequences,
                "target_keys": key_map,
                "topk": args.topk,
                "min_detections": args.min_detections,
                "min_committed_matches": args.min_committed_matches,
                "max_detections": args.max_detections,
                "max_tracks": args.max_tracks,
            },
            ensure_ascii=False,
        ),
    }

    write_rows(summary_csv, SUMMARY_FIELDS, [row])
    upsert_plan(args, "running", summary_csv, queue_log, notes="graphassoc sparse-edit batch started")
    append_registry(args, "running", summary_csv, queue_log, "graphassoc sparse-edit batch started", checkpoint=Path(""))

    dataset_summary: Dict[str, Any] = {}
    train_summary: Dict[str, str] = {}
    eval_summary: Dict[str, str] = {}
    eval_delta: Dict[str, float | int] = {}

    try:
        # Step 01: build dataset
        row.update(
            {
                "step": "01_build_dataset",
                "status": "running",
                "started_at": now_iso(),
                "notes": f"building {args.teacher_mode} dataset",
            }
        )
        write_rows(summary_csv, SUMMARY_FIELDS, [row])
        dataset_log = logs_dir / "build_dataset.log"
        dataset_cmd = [
            args.python_bin,
            str(BUILD_DATASET_SCRIPT),
            "--source-manifest",
            str(Path(args.source_manifest).expanduser().resolve()),
            "--out-dir",
            str(dataset_dir),
            "--topk",
            str(args.topk),
            "--min-detections",
            str(args.min_detections),
            "--min-committed-matches",
            str(args.min_committed_matches),
            "--max-detections",
            str(args.max_detections),
            "--max-tracks",
            str(args.max_tracks),
            "--train-sequences",
            args.train_sequences,
            "--val-sequences",
            args.val_sequences,
            "--strict-sequence-split",
            "--feature-version",
            args.feature_version,
            "--dataset-tag",
            args.dataset_tag,
            "--teacher-mode",
            args.teacher_mode,
            "--edit-utility-min-gain",
            "0.5",
            "--edit-utility-commit-cost",
            "0.20",
            "--edit-utility-force-defer-gain",
            "0.50",
            "--runtime-host-match-thresh",
            "0.90",
            "--soft-rescue-weight",
            "0.75",
            "--rescue-force-defer-gain",
            "0.50",
            "--rescue-min-gain",
            "1.25" if args.teacher_mode == "sparse_edit" else "0.50",
            "--bridge-crowded-row-degree-thresh",
            "4",
            "--bridge-crowded-bonus",
            "0.25",
            "--bridge-large-component-bonus",
            "0.50",
            "--bridge-commit-cost",
            "0.20",
            "--bridge-min-gain",
            "0.75",
        ]
        rc = run_step(dataset_cmd, dataset_log)
        if rc != 0:
            raise RuntimeError(f"dataset build failed with exit code {rc}")
        dataset_summary_path = dataset_dir / "summary.json"
        if dataset_summary_path.is_file():
            dataset_summary = json.loads(dataset_summary_path.read_text(encoding="utf-8"))
        row.update(
            {
                "status": "success",
                "finished_at": now_iso(),
                "trigger_pass_clusters": dataset_summary.get("trigger_pass_clusters", ""),
                "cluster_utility_gain": dataset_summary.get("cluster_utility_gain", ""),
                "cluster_edit_gain": dataset_summary.get("cluster_edit_gain", ""),
                "cluster_edit_utility_gain": dataset_summary.get("cluster_edit_utility_gain", ""),
                "cluster_sparse_utility_gain": dataset_summary.get("cluster_sparse_utility_gain", ""),
                "train_target_positives": dataset_summary.get("split_breakdown", {}).get("train", {}).get(
                    "cluster_should_intervene_edit_clusters"
                    if args.teacher_mode == "edit_utility"
                    else "cluster_should_intervene_sparse_clusters",
                    "",
                ),
                "val_target_positives": dataset_summary.get("split_breakdown", {}).get("val", {}).get(
                    "cluster_should_intervene_edit_clusters"
                    if args.teacher_mode == "edit_utility"
                    else "cluster_should_intervene_sparse_clusters",
                    "",
                ),
                "notes": f"dataset built: {dataset_summary.get('eligible_clusters', '')} eligible clusters",
            }
        )
        write_rows(summary_csv, SUMMARY_FIELDS, [row])

        # Step 02: train
        row.update(
            {
                "step": "02_train_stage1",
                "status": "running",
                "started_at": now_iso(),
                "finished_at": "",
                "notes": f"training {args.teacher_mode} set predictor",
            }
        )
        write_rows(summary_csv, SUMMARY_FIELDS, [row])
        train_log = logs_dir / "train.log"
        train_cmd = [
            args.python_bin,
            str(TRAIN_SCRIPT),
            "--data-jsonl",
            str(dataset_dir / "cluster_examples.jsonl"),
            "--out-dir",
            str(train_dir),
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
            "--num-heads",
            str(args.num_heads),
            "--num-conflict-blocks",
            str(args.num_conflict_blocks),
            "--dropout",
            str(args.dropout),
            "--train-sequences",
            args.train_sequences,
            "--val-sequences",
            args.val_sequences,
            "--strict-sequence-split",
            "--split-strategy",
            "auto",
            "--split-target-key",
            key_map["cluster_target_key"],
            "--val-fraction",
            "0.2",
            "--min-val-examples",
            "64",
            "--dataset-tag",
            args.dataset_tag,
            "--source-manifest",
            str(Path(args.source_manifest).expanduser().resolve()),
            "--feature-version",
            args.feature_version,
            "--cluster-gate-thresh",
            str(args.cluster_gate_thresh),
            "--cluster-gate-calibration",
            args.cluster_gate_calibration,
            "--cluster-gate-select-metric",
            args.cluster_gate_select_metric,
            "--cluster-gate-beta",
            str(args.cluster_gate_beta),
            "--cluster-gate-fp-weight",
            str(args.cluster_gate_fp_weight),
            "--cluster-gate-search-min",
            str(args.cluster_gate_search_min),
            "--cluster-gate-search-max",
            str(args.cluster_gate_search_max),
            "--cluster-gate-search-steps",
            str(args.cluster_gate_search_steps),
            "--cluster-gate-loss-mode",
            args.cluster_gate_loss_mode,
            "--cluster-gate-positive-weight",
            str(args.cluster_gate_positive_weight),
            "--cluster-gate-negative-weight",
            str(args.cluster_gate_negative_weight),
            "--train-positive-cluster-oversample",
            str(args.train_positive_cluster_oversample),
            "--model-selection-metric",
            args.model_selection_metric,
            "--assignment-target-key",
            key_map["assignment_target_key"],
            "--assignment-row-mask-key",
            key_map["assignment_row_mask_key"],
            "--edge-loss-mask-key",
            key_map["edge_loss_mask_key"],
            "--edge-target-key",
            key_map["edge_target_key"],
            "--cluster-target-key",
            key_map["cluster_target_key"],
            "--target-gate-coverage-min",
            str(args.target_gate_coverage_min),
            "--target-gate-coverage-max",
            str(args.target_gate_coverage_max),
            "--coverage-penalty-weight",
            str(args.coverage_penalty_weight),
            "--keep-row-loss-weight",
            str(args.keep_row_loss_weight),
            "--edit-row-loss-weight",
            str(args.edit_row_loss_weight),
            "--edit-edge-positive-weight",
            str(args.edit_edge_positive_weight),
            "--loss-assign-weight",
            str(args.loss_assign_weight),
            "--loss-edge-weight",
            str(args.loss_edge_weight),
            "--loss-cluster-weight",
            str(args.loss_cluster_weight),
            "--loss-margin-weight",
            str(args.loss_margin_weight),
            "--margin-commit",
            str(args.margin_commit),
            "--margin-row",
            str(args.margin_row),
            "--margin-defer",
            str(args.margin_defer),
            "--margin-host-edit",
            str(args.margin_host_edit),
            "--edge-focal-alpha",
            str(args.edge_focal_alpha),
            "--edge-focal-gamma",
            str(args.edge_focal_gamma),
            "--grad-clip-norm",
            str(args.grad_clip_norm),
            "--skip-step-grad-norm",
            str(args.skip_step_grad_norm),
            "--seed",
            str(args.seed),
        ]
        rc = run_step(train_cmd, train_log)
        if rc != 0:
            raise RuntimeError(f"training failed with exit code {rc}")
        train_summary = read_single_row(train_dir / "summary.csv")
        train_ckpt = train_dir / "best.pt"
        row.update(
            {
                "status": "success",
                "finished_at": now_iso(),
                "checkpoint": str(train_ckpt),
                "best_epoch": train_summary.get("best_epoch", ""),
                "best_metric": train_summary.get("best_metric", ""),
                "train_examples": train_summary.get("train_examples", ""),
                "val_examples": train_summary.get("val_examples", ""),
                "val_cluster_gate_thresh_calibrated": train_summary.get("val_cluster_gate_thresh_calibrated", ""),
                "val_cluster_gate_temp": train_summary.get("val_cluster_gate_temp", ""),
                "val_cluster_gate_bias": train_summary.get("val_cluster_gate_bias", ""),
                "val_cluster_gate_precision_cal": train_summary.get("val_cluster_gate_precision_cal", ""),
                "val_cluster_gate_recall_cal": train_summary.get("val_cluster_gate_recall_cal", ""),
                "val_cluster_gate_f0_5": train_summary.get("val_cluster_gate_f0_5", ""),
                "val_cluster_gate_utility_cal": train_summary.get("val_cluster_gate_utility_cal", ""),
                "val_cluster_gate_coverage_cal": train_summary.get("val_cluster_gate_coverage_cal", ""),
                "val_cluster_gate_bounded_utility": train_summary.get("val_cluster_gate_bounded_utility", ""),
                "notes": "training complete; proceeding to MOT20 val_half eval",
            }
        )
        write_rows(summary_csv, SUMMARY_FIELDS, [row])

        # Step 03: eval
        row.update(
            {
                "step": "03_eval_valhalf",
                "status": "running",
                "started_at": now_iso(),
                "finished_at": "",
                "notes": "running MOT20 val_half eval",
            }
        )
        write_rows(summary_csv, SUMMARY_FIELDS, [row])
        eval_log = logs_dir / "eval.log"
        eval_cmd = [
            args.python_bin,
            str(EVAL_SCRIPT),
            "--run-root",
            str(eval_dir),
            "--experiment-name",
            args.queue_name,
            "--variant-name",
            "graphassoc_sparseedit_mot20",
            "--seq-ids",
            "2",
            "5",
            "--graph-assoc-commit-checkpoint",
            str(train_ckpt),
            "--graph-assoc-commit-device",
            str(args.eval_device),
            "--graph-assoc-commit-score-margin",
            "0.0",
            "--graph-assoc-commit-replace-rules",
            "--graph-assoc-dump-candidate-rows",
        ]
        rc = run_step(eval_cmd, eval_log)
        if rc != 0:
            raise RuntimeError(f"eval failed with exit code {rc}")
        eval_summary = read_single_row(eval_dir / "summary.csv")
        eval_delta_rows = read_csv_rows(eval_dir / "metrics_delta.csv")
        if eval_delta_rows:
            eval_delta = {
                "delta_HOTA": float(eval_delta_rows[0].get("delta_HOTA", 0.0) or 0.0),
                "delta_AssA": float(eval_delta_rows[0].get("delta_AssA", 0.0) or 0.0),
                "delta_IDF1": float(eval_delta_rows[0].get("delta_IDF1", 0.0) or 0.0),
                "delta_MOTA": float(eval_delta_rows[0].get("delta_MOTA", 0.0) or 0.0),
                "delta_IDSW": int(round(float(eval_delta_rows[0].get("delta_IDs", 0.0) or 0.0))),
                "delta_Frag": int(round(float(eval_delta_rows[0].get("delta_Frag", 0.0) or 0.0))),
            }
        row.update(
            {
                "status": "success",
                "finished_at": now_iso(),
                "eval_HOTA": eval_summary.get("HOTA", ""),
                "eval_AssA": eval_summary.get("AssA", ""),
                "eval_IDF1": eval_summary.get("IDF1", ""),
                "eval_MOTA": eval_summary.get("MOTA", ""),
                "eval_IDSW": eval_summary.get("IDSW", ""),
                "eval_gate_filtered_clusters": eval_summary.get("gate_filtered_clusters", ""),
                "eval_replaced_clusters": eval_summary.get("replaced_clusters", ""),
                "delta_HOTA": eval_delta.get("delta_HOTA", ""),
                "delta_AssA": eval_delta.get("delta_AssA", ""),
                "delta_IDF1": eval_delta.get("delta_IDF1", ""),
                "delta_MOTA": eval_delta.get("delta_MOTA", ""),
                "delta_IDSW": eval_delta.get("delta_IDSW", ""),
                "delta_Frag": eval_delta.get("delta_Frag", ""),
                "notes": "evaluation complete",
            }
        )
        write_rows(summary_csv, SUMMARY_FIELDS, [row])
        write_record(result_csv, row)
        upsert_plan(args, "completed", result_csv, queue_log, notes="graphassoc sparse-edit batch completed")
        append_registry(args, "success", summary_csv, queue_log, "graphassoc sparse-edit batch completed", checkpoint=train_ckpt)
        return 0
    except Exception as exc:
        row.update(
            {
                "status": "failed",
                "finished_at": now_iso(),
                "notes": f"failed: {exc}",
            }
        )
        write_rows(summary_csv, SUMMARY_FIELDS, [row])
        write_record(result_csv, row)
        upsert_plan(args, "failed", result_csv, queue_log, notes=str(exc))
        append_registry(args, "failed", summary_csv, queue_log, str(exc), checkpoint=Path(row.get("checkpoint", "")))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
