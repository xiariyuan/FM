#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import fcntl
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path("/gemini/code/FMtrack-main/FM-Track")
REGISTRY_CSV = REPO_ROOT / "outputs" / "experiment_registry.csv"
TERMINAL_STATUSES = {"ok", "success", "failed", "error", "cancelled", "aborted", "skipped"}

SUMMARY_FIELDS = [
    "step",
    "status",
    "depends_on",
    "decision",
    "script",
    "run_root",
    "checkpoint",
    "current_stage",
    "delta_HOTA",
    "delta_AssA",
    "delta_IDF1",
    "delta_MOTA",
    "delta_IDSW",
    "plugin_replaced_clusters",
    "plugin_gate_pass_clusters",
    "plugin_all_defer_clusters",
    "error",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Queue a single official ByteTrack bridge-commit follow-up after the current mainline run finishes."
    )
    parser.add_argument(
        "--main-run-root",
        default=str(
            REPO_ROOT / "outputs" / "official_bytetrack_stage1_largecomp4_sparseedit_posboost_lc4_20260327_015600"
        ),
    )
    parser.add_argument("--queue-root", default="")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--poll-sec", type=int, default=180)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size-stage1", type=int, default=4)
    parser.add_argument("--cluster-gate-positive-weight", type=float, default=8.0)
    parser.add_argument("--cluster-gate-negative-weight", type=float, default=1.0)
    parser.add_argument("--cluster-gate-loss-mode", default="weighted_bce")
    parser.add_argument("--cluster-gate-calibration", default="temp_bias")
    parser.add_argument("--cluster-gate-select-metric", default="f0.5")
    parser.add_argument("--cluster-gate-beta", type=float, default=0.5)
    parser.add_argument("--cluster-gate-fp-weight", type=float, default=2.0)
    parser.add_argument("--cluster-gate-search-min", type=float, default=0.05)
    parser.add_argument("--cluster-gate-search-max", type=float, default=0.80)
    parser.add_argument("--cluster-gate-search-steps", type=int, default=19)
    parser.add_argument("--model-selection-metric", default="commit_viability_utility_cov")
    parser.add_argument("--bridge-crowded-row-degree-thresh", type=int, default=4)
    parser.add_argument("--bridge-crowded-bonus", type=float, default=0.25)
    parser.add_argument("--bridge-large-component-bonus", type=float, default=0.50)
    parser.add_argument("--bridge-commit-cost", type=float, default=0.20)
    parser.add_argument("--bridge-min-gain", type=float, default=0.75)
    parser.add_argument("--loss-assign-weight", type=float, default=1.0)
    parser.add_argument("--loss-edge-weight", type=float, default=0.5)
    parser.add_argument("--loss-cluster-weight", type=float, default=0.25)
    parser.add_argument("--loss-margin-weight", type=float, default=1.0)
    parser.add_argument("--registry-csv", default=str(REGISTRY_CSV))
    return parser.parse_args()


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in SUMMARY_FIELDS})


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def read_single_row(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        return next(csv.DictReader(f), {})


def update_step(summary_csv: Path, step: str, **updates: Any) -> None:
    rows = read_rows(summary_csv)
    for row in rows:
        if row.get("step") == step:
            for key, value in updates.items():
                row[key] = value
            break
    write_rows(summary_csv, rows)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def process_alive(pattern: str) -> bool:
    proc = subprocess.run(
        ["bash", "-lc", f"ps -ef | grep {pattern!r} | grep -v grep >/dev/null 2>&1"],
        cwd=REPO_ROOT,
    )
    return proc.returncode == 0


def pipeline_outcome(run_root: Path) -> dict[str, Any]:
    summary = read_single_row(run_root / "summary.csv")
    return {
        "status": str(summary.get("status", "")).strip(),
        "current_stage": str(summary.get("current_stage", "")).strip(),
        "run_root": str(run_root.resolve()),
        "checkpoint": str(summary.get("checkpoint", "")).strip() or str((run_root / "01_stage1" / "best.pt").resolve()),
        "delta_HOTA": _safe_float(summary.get("delta_HOTA", 0.0), 0.0),
        "delta_AssA": _safe_float(summary.get("delta_AssA", 0.0), 0.0),
        "delta_IDF1": _safe_float(summary.get("delta_IDF1", 0.0), 0.0),
        "delta_MOTA": _safe_float(summary.get("delta_MOTA", 0.0), 0.0),
        "delta_IDSW": _safe_int(summary.get("delta_IDSW", 0), 0),
        "plugin_replaced_clusters": _safe_int(summary.get("plugin_replaced_clusters", 0), 0),
        "plugin_gate_pass_clusters": _safe_int(summary.get("plugin_gate_pass_clusters", 0), 0),
        "plugin_all_defer_clusters": _safe_int(summary.get("plugin_all_defer_clusters", 0), 0),
        "error": str(summary.get("error", "")).strip(),
        "teacher_mode": str(summary.get("teacher_mode", "sparse_edit")).strip() or "sparse_edit",
        "large_component_max_subclusters": _safe_int(summary.get("large_component_max_subclusters", 4), 4),
        "split_strategy": str(summary.get("split_strategy", "auto")).strip() or "auto",
        "split_target_key": str(summary.get("split_target_key", "cluster_should_intervene_sparse")).strip() or "cluster_should_intervene_sparse",
        "assignment_target_key": str(summary.get("assignment_target_key", "target_by_det_sparse_edit")).strip() or "target_by_det_sparse_edit",
        "assignment_row_mask_key": str(summary.get("assignment_row_mask_key", "row_sparse_edit_mask")).strip() or "row_sparse_edit_mask",
        "edge_loss_mask_key": str(summary.get("edge_loss_mask_key", "assignment_row_mask")).strip() or "assignment_row_mask",
        "edge_target_key": str(summary.get("edge_target_key", "edge_is_sparse_edit")).strip() or "edge_is_sparse_edit",
        "cluster_target_key": str(summary.get("cluster_target_key", "cluster_should_intervene_sparse")).strip() or "cluster_should_intervene_sparse",
        "target_gate_coverage_min": _safe_float(summary.get("target_gate_coverage_min", 0.015), 0.015),
        "target_gate_coverage_max": _safe_float(summary.get("target_gate_coverage_max", 0.05), 0.05),
        "coverage_penalty_weight": _safe_float(summary.get("coverage_penalty_weight", 1.0), 1.0),
        "keep_row_loss_weight": _safe_float(summary.get("keep_row_loss_weight", 0.0), 0.0),
        "edit_row_loss_weight": _safe_float(summary.get("edit_row_loss_weight", 2.0), 2.0),
        "edit_edge_positive_weight": _safe_float(summary.get("edit_edge_positive_weight", 32.0), 32.0),
        "margin_host_edit": _safe_float(summary.get("margin_host_edit", 0.5), 0.5),
        "train_sequences": str(summary.get("train_sequences", "")).strip(),
        "val_sequences": str(summary.get("val_sequences", "")).strip(),
        "graph_topk": _safe_int(summary.get("graph_topk", 8), 8),
        "graph_min_detections": _safe_int(summary.get("graph_min_detections", 2), 2),
        "graph_min_committed_matches": _safe_int(summary.get("graph_min_committed_matches", 1), 1),
        "graph_max_detections": _safe_int(summary.get("graph_max_detections", 8), 8),
        "graph_max_tracks": _safe_int(summary.get("graph_max_tracks", 32), 32),
        "seed": _safe_int(summary.get("seed", 42), 42),
    }


def fill_step_from_outcome(summary_csv: Path, step: str, outcome: dict[str, Any], decision: str) -> None:
    update_step(
        summary_csv,
        step,
        status=str(outcome.get("status", "")),
        decision=decision,
        run_root=str(outcome.get("run_root", "")),
        checkpoint=str(outcome.get("checkpoint", "")),
        current_stage=str(outcome.get("current_stage", "")),
        delta_HOTA=f"{_safe_float(outcome.get('delta_HOTA', 0.0), 0.0):.3f}",
        delta_AssA=f"{_safe_float(outcome.get('delta_AssA', 0.0), 0.0):.3f}",
        delta_IDF1=f"{_safe_float(outcome.get('delta_IDF1', 0.0), 0.0):.3f}",
        delta_MOTA=f"{_safe_float(outcome.get('delta_MOTA', 0.0), 0.0):.3f}",
        delta_IDSW=str(_safe_int(outcome.get("delta_IDSW", 0), 0)),
        plugin_replaced_clusters=str(_safe_int(outcome.get("plugin_replaced_clusters", 0), 0)),
        plugin_gate_pass_clusters=str(_safe_int(outcome.get("plugin_gate_pass_clusters", 0), 0)),
        plugin_all_defer_clusters=str(_safe_int(outcome.get("plugin_all_defer_clusters", 0), 0)),
        error=str(outcome.get("error", "")),
    )


def _registry_base_row(queue_root: Path, summary_csv: Path) -> dict[str, str]:
    return {
        "timestamp": _now(),
        "kind": "other",
        "status": "running",
        "script": "scripts/queue_official_bytetrack_possampler_followup.py",
        "dataset": "MOT17",
        "split": "followup_queue",
        "tracker_family": "official_bytetrack",
        "variant": "official_bytetrack_bridgecommit_followup",
        "tag": queue_root.name,
        "run_root": str(queue_root.resolve()),
        "summary_csv": str(summary_csv.resolve()),
        "checkpoint": "",
        "calibrator_npz": "",
        "log_path": str((queue_root / "queue.log").resolve()),
        "notes": "queue current official ByteTrack run, then launch one bridge-commit follow-up if needed",
    }


def _registry_fields(existing_rows: list[dict[str, str]], new_rows: list[dict[str, str]]) -> list[str]:
    fields: list[str] = []
    seen: set[str] = set()
    for rows in (existing_rows, new_rows):
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    fields.append(key)
                    seen.add(key)
    return fields


def sync_registry_rows(args: argparse.Namespace, queue_root: Path, summary_csv: Path) -> None:
    registry_csv = Path(args.registry_csv).resolve()
    base = _registry_base_row(queue_root, summary_csv)
    summary_rows = read_rows(summary_csv)
    new_rows: list[dict[str, str]] = []
    for row in summary_rows:
        merged = dict(base)
        merged.update({key: str(value) for key, value in row.items()})
        merged["timestamp"] = _now()
        merged["status"] = str(row.get("status", "")).strip() or "pending"
        merged["checkpoint"] = str(row.get("checkpoint", "")).strip()
        merged["run_root"] = str(row.get("run_root", "")).strip() or base["run_root"]
        new_rows.append(merged)

    lock_path = registry_csv.with_suffix(registry_csv.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock_fp:
        fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
        if registry_csv.is_file():
            with registry_csv.open("r", encoding="utf-8", newline="") as f:
                existing_rows = [dict(row) for row in csv.DictReader(f)]
        else:
            existing_rows = []

        step_index: dict[tuple[str, str, str, str, str], int] = {}
        for idx, row in enumerate(existing_rows):
            key = (
                str(row.get("kind", "")),
                str(row.get("script", "")),
                str(row.get("variant", "")),
                str(row.get("summary_csv", "")),
                str(row.get("step", "")),
            )
            step_index[key] = idx

        for row in new_rows:
            key = (
                str(row.get("kind", "")),
                str(row.get("script", "")),
                str(row.get("variant", "")),
                str(row.get("summary_csv", "")),
                str(row.get("step", "")),
            )
            if key in step_index:
                existing_rows[step_index[key]] = row
            else:
                step_index[key] = len(existing_rows)
                existing_rows.append(row)

        fieldnames = _registry_fields(existing_rows, new_rows)
        tmp_path = registry_csv.with_suffix(registry_csv.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in existing_rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})
        os.replace(tmp_path, registry_csv)
        fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)


def wait_for_pipeline(run_root: Path, poll_sec: int, log_fp) -> dict[str, Any]:
    summary_path = run_root / "summary.csv"
    while True:
        outcome = pipeline_outcome(run_root)
        alive = process_alive(str(run_root))
        status = str(outcome.get("status", "")).strip()
        log_fp.write(
            f"[{_now()}] monitor status={status or 'missing'} stage={outcome.get('current_stage', '') or 'unknown'} alive={int(alive)}\n"
        )
        log_fp.flush()
        if summary_path.is_file() and status in TERMINAL_STATUSES:
            return outcome
        if not alive and summary_path.is_file() and status == "running":
            outcome["status"] = "failed"
            outcome["error"] = "process_exited_while_summary_still_running"
            return outcome
        if not alive and not summary_path.is_file():
            outcome["status"] = "failed"
            outcome["error"] = "process_exited_before_summary_created"
            return outcome
        time.sleep(max(int(poll_sec), 30))


def init_summary(summary_csv: Path, main_run_root: Path) -> None:
    rows = [
        {
            "step": "01_main_sparseedit_posboost_lc4",
            "status": "running",
            "depends_on": "",
            "decision": "monitor_current_run",
            "script": "scripts/run_official_bytetrack_local_conflict_stage1_trainhalf.py",
            "run_root": str(main_run_root.resolve()),
            "checkpoint": "",
            "current_stage": "",
            "delta_HOTA": "",
            "delta_AssA": "",
            "delta_IDF1": "",
            "delta_MOTA": "",
            "delta_IDSW": "",
            "plugin_replaced_clusters": "",
            "plugin_gate_pass_clusters": "",
            "plugin_all_defer_clusters": "",
            "error": "",
            "notes": "monitor current mainline run",
        },
        {
            "step": "02_bridgecommit_v1",
            "status": "pending",
            "depends_on": "01_main_sparseedit_posboost_lc4",
            "decision": "",
            "script": "scripts/run_official_bytetrack_local_conflict_stage1_trainhalf.py",
            "run_root": "",
            "checkpoint": "",
            "current_stage": "",
            "delta_HOTA": "",
            "delta_AssA": "",
            "delta_IDF1": "",
            "delta_MOTA": "",
            "delta_IDSW": "",
            "plugin_replaced_clusters": "",
            "plugin_gate_pass_clusters": "",
            "plugin_all_defer_clusters": "",
            "error": "",
            "notes": "launch only if the current run ends as no-op or non-positive",
        },
    ]
    write_rows(summary_csv, rows)


def should_launch_followup(outcome: dict[str, Any]) -> bool:
    if str(outcome.get("status", "")).strip() not in {"ok", "success"}:
        return False
    if _safe_int(outcome.get("plugin_replaced_clusters", 0), 0) == 0:
        return True
    if _safe_float(outcome.get("delta_HOTA", 0.0), 0.0) <= 0.0:
        return True
    if _safe_float(outcome.get("delta_IDF1", 0.0), 0.0) <= 0.0:
        return True
    return False


def build_followup_cmd(args: argparse.Namespace, main_outcome: dict[str, Any], out_dir: Path) -> list[str]:
    experiment_name = f"{Path(main_outcome['run_root']).name}_bridgecommit_v1"
    protocol_tag = "official_bytetrack_bridgecommit_v1_followup"
    return [
        args.python_bin,
        str(REPO_ROOT / "scripts" / "run_official_bytetrack_local_conflict_stage1_trainhalf.py"),
        "--out-dir",
        str(out_dir.resolve()),
        "--experiment-name",
        experiment_name,
        "--protocol-tag",
        protocol_tag,
        "--dataset-tag",
        "official_bytetrack_bridgecommit_v1_followup",
        "--teacher-mode",
        "bridge_commit_contract_utility",
        "--epochs",
        str(int(args.epochs)),
        "--batch-size-stage1",
        str(int(args.batch_size_stage1)),
        "--large-component-max-subclusters",
        str(_safe_int(main_outcome.get("large_component_max_subclusters", 4), 4)),
        "--split-strategy",
        str(main_outcome.get("split_strategy", "auto")),
        "--split-target-key",
        "cluster_should_intervene_bridge",
        "--assignment-target-key",
        "target_by_det_bridge",
        "--assignment-row-mask-key",
        "row_bridge_mask",
        "--edge-loss-mask-key",
        "assignment_row_mask",
        "--edge-target-key",
        "edge_is_bridge_commit",
        "--cluster-target-key",
        "cluster_should_intervene_bridge",
        "--target-gate-coverage-min",
        "0.01",
        "--target-gate-coverage-max",
        "0.05",
        "--coverage-penalty-weight",
        str(_safe_float(main_outcome.get("coverage_penalty_weight", 1.0), 1.0)),
        "--cluster-gate-calibration",
        str(args.cluster_gate_calibration),
        "--cluster-gate-select-metric",
        str(args.cluster_gate_select_metric),
        "--cluster-gate-beta",
        str(float(args.cluster_gate_beta)),
        "--cluster-gate-fp-weight",
        str(float(args.cluster_gate_fp_weight)),
        "--cluster-gate-search-min",
        str(float(args.cluster_gate_search_min)),
        "--cluster-gate-search-max",
        str(float(args.cluster_gate_search_max)),
        "--cluster-gate-search-steps",
        str(int(args.cluster_gate_search_steps)),
        "--cluster-gate-loss-mode",
        str(args.cluster_gate_loss_mode),
        "--cluster-gate-positive-weight",
        str(float(args.cluster_gate_positive_weight)),
        "--cluster-gate-negative-weight",
        str(float(args.cluster_gate_negative_weight)),
        "--model-selection-metric",
        str(args.model_selection_metric),
        "--keep-row-loss-weight",
        str(_safe_float(main_outcome.get("keep_row_loss_weight", 0.0), 0.0)),
        "--edit-row-loss-weight",
        str(_safe_float(main_outcome.get("edit_row_loss_weight", 2.0), 2.0)),
        "--edit-edge-positive-weight",
        str(_safe_float(main_outcome.get("edit_edge_positive_weight", 32.0), 32.0)),
        "--margin-host-edit",
        str(_safe_float(main_outcome.get("margin_host_edit", 0.5), 0.5)),
        "--bridge-crowded-row-degree-thresh",
        str(int(args.bridge_crowded_row_degree_thresh)),
        "--bridge-crowded-bonus",
        str(float(args.bridge_crowded_bonus)),
        "--bridge-large-component-bonus",
        str(float(args.bridge_large_component_bonus)),
        "--bridge-commit-cost",
        str(float(args.bridge_commit_cost)),
        "--bridge-min-gain",
        str(float(args.bridge_min_gain)),
        "--loss-assign-weight",
        str(float(args.loss_assign_weight)),
        "--loss-edge-weight",
        str(float(args.loss_edge_weight)),
        "--loss-cluster-weight",
        str(float(args.loss_cluster_weight)),
        "--loss-margin-weight",
        str(float(args.loss_margin_weight)),
        "--train-sequences",
        str(main_outcome.get("train_sequences", "")),
        "--val-sequences",
        str(main_outcome.get("val_sequences", "")),
        "--topk",
        str(_safe_int(main_outcome.get("graph_topk", 8), 8)),
        "--min-detections",
        str(_safe_int(main_outcome.get("graph_min_detections", 2), 2)),
        "--min-committed-matches",
        str(_safe_int(main_outcome.get("graph_min_committed_matches", 1), 1)),
        "--max-detections",
        str(_safe_int(main_outcome.get("graph_max_detections", 8), 8)),
        "--max-tracks",
        str(_safe_int(main_outcome.get("graph_max_tracks", 32), 32)),
        "--seed",
        str(_safe_int(main_outcome.get("seed", 42), 42)),
    ]


def run_command(cmd: list[str], log_fp) -> None:
    log_fp.write("$ " + " ".join(cmd) + "\n")
    log_fp.flush()
    subprocess.run(cmd, check=True, cwd=REPO_ROOT, stdout=log_fp, stderr=subprocess.STDOUT)
    log_fp.write("\n")
    log_fp.flush()


def main() -> int:
    args = parse_args()
    main_run_root = Path(args.main_run_root).resolve()
    if not main_run_root.is_dir():
        raise FileNotFoundError(f"Missing main run root: {main_run_root}")
    queue_root = (
        Path(args.queue_root).resolve()
        if str(args.queue_root).strip()
        else REPO_ROOT / "outputs" / f"official_bytetrack_bridgecommit_followup_queue_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    queue_root.mkdir(parents=True, exist_ok=True)
    summary_csv = queue_root / "summary.csv"
    queue_log = queue_root / "queue.log"
    if not summary_csv.is_file():
        init_summary(summary_csv, main_run_root)
    sync_registry_rows(args, queue_root, summary_csv)

    with queue_log.open("a", encoding="utf-8") as log_fp:
        log_fp.write(f"[{_now()}] start queue_root={queue_root}\n")
        log_fp.flush()

        main_outcome = wait_for_pipeline(main_run_root, int(args.poll_sec), log_fp)
        fill_step_from_outcome(summary_csv, "01_main_sparseedit_posboost_lc4", main_outcome, decision="mainline_complete")
        sync_registry_rows(args, queue_root, summary_csv)

        if not should_launch_followup(main_outcome):
            update_step(
                summary_csv,
                "02_bridgecommit_v1",
                status="skipped",
                decision="mainline_positive_enough",
                error="",
            )
            sync_registry_rows(args, queue_root, summary_csv)
            log_fp.write(f"[{_now()}] mainline already positive enough; skip follow-up\n")
            return 0

        followup_out = queue_root / "02_bridgecommit_v1"
        update_step(
            summary_csv,
            "02_bridgecommit_v1",
            status="running",
            decision="launch_followup",
            run_root=str(followup_out.resolve()),
        )
        sync_registry_rows(args, queue_root, summary_csv)

        cmd = build_followup_cmd(args, main_outcome, followup_out)
        run_command(cmd, log_fp)
        followup_outcome = pipeline_outcome(followup_out)
        fill_step_from_outcome(summary_csv, "02_bridgecommit_v1", followup_outcome, decision="followup_complete")
        sync_registry_rows(args, queue_root, summary_csv)
        log_fp.write(f"[{_now()}] queue complete\n")
        return 0 if str(followup_outcome.get("status", "")).strip() in {"ok", "success"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
