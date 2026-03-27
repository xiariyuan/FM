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
    "plugin_delta_commit_pairs",
    "plugin_delta_drop_pairs",
    "plugin_gate_pass_clusters",
    "plugin_gate_filtered_clusters",
    "plugin_all_defer_clusters",
    "error",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overnight queue for official ByteTrack sparse-edit follow-up experiments."
    )
    parser.add_argument(
        "--main-run-root",
        default=str(
            REPO_ROOT / "outputs" / "official_bytetrack_stage1_largecomp4_sparseedit_20260326_235955"
        ),
    )
    parser.add_argument("--queue-root", default="")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--poll-sec", type=int, default=180)
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


def update_step(summary_csv: Path, step: str, **updates: Any) -> None:
    rows = read_rows(summary_csv)
    for row in rows:
        if row.get("step") == step:
            for key, value in updates.items():
                row[key] = value
            break
    write_rows(summary_csv, rows)


def mark_remaining(summary_csv: Path, from_step: str, status: str, decision: str) -> None:
    rows = read_rows(summary_csv)
    seen = False
    for row in rows:
        if row.get("step") == from_step:
            seen = True
            continue
        if seen and row.get("status") in {"pending", "running"}:
            row["status"] = status
            row["decision"] = decision
    write_rows(summary_csv, rows)


def read_single_row(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        return next(csv.DictReader(f), {})


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
    stage1_summary = read_single_row(run_root / "01_stage1" / "summary.csv")
    outcome = {
        "kind": "pipeline",
        "status": str(summary.get("status", "")).strip(),
        "current_stage": str(summary.get("current_stage", "")).strip(),
        "run_root": str(run_root.resolve()),
        "checkpoint": str(summary.get("checkpoint", "")).strip() or str((run_root / "01_stage1" / "best.pt").resolve()),
        "official_exp_file": str(summary.get("official_eval_exp_file", "")).strip(),
        "official_checkpoint": str(summary.get("official_checkpoint", "")).strip(),
        "host_variant": str(summary.get("host_variant", "official_bytetrack")).strip() or "official_bytetrack",
        "graph_topk": _safe_int(summary.get("graph_topk", 8), 8),
        "graph_min_detections": _safe_int(summary.get("graph_min_detections", 2), 2),
        "graph_min_committed_matches": _safe_int(summary.get("graph_min_committed_matches", 1), 1),
        "graph_max_detections": _safe_int(summary.get("graph_max_detections", 8), 8),
        "graph_max_tracks": _safe_int(summary.get("graph_max_tracks", 32), 32),
        "graph_max_commits_per_cluster": _safe_int(summary.get("graph_max_commits_per_cluster", 1), 1),
        "graph_max_replaced_clusters": _safe_int(summary.get("graph_max_replaced_clusters", 0), 0),
        "graph_cluster_gate_thresh": _safe_float(
            summary.get("graph_cluster_gate_thresh", stage1_summary.get("val_cluster_gate_thresh_calibrated", 0.5)),
            0.5,
        ),
        "graph_cluster_gate_temp": _safe_float(
            summary.get("graph_cluster_gate_temp", stage1_summary.get("val_cluster_gate_temp", 1.0)),
            1.0,
        ),
        "graph_cluster_gate_bias": _safe_float(
            summary.get("graph_cluster_gate_bias", stage1_summary.get("val_cluster_gate_bias", 0.0)),
            0.0,
        ),
        "delta_HOTA": _safe_float(summary.get("delta_HOTA", 0.0), 0.0),
        "delta_AssA": _safe_float(summary.get("delta_AssA", 0.0), 0.0),
        "delta_IDF1": _safe_float(summary.get("delta_IDF1", 0.0), 0.0),
        "delta_MOTA": _safe_float(summary.get("delta_MOTA", 0.0), 0.0),
        "delta_IDSW": _safe_int(summary.get("delta_IDSW", 0), 0),
        "plugin_replaced_clusters": _safe_int(summary.get("plugin_replaced_clusters", 0), 0),
        "plugin_delta_commit_pairs": _safe_int(summary.get("plugin_delta_commit_pairs", 0), 0),
        "plugin_delta_drop_pairs": _safe_int(summary.get("plugin_delta_drop_pairs", 0), 0),
        "plugin_gate_pass_clusters": _safe_int(summary.get("plugin_gate_pass_clusters", 0), 0),
        "plugin_gate_filtered_clusters": _safe_int(summary.get("plugin_gate_filtered_clusters", 0), 0),
        "plugin_all_defer_clusters": _safe_int(summary.get("plugin_all_defer_clusters", 0), 0),
        "error": str(summary.get("error", "")).strip(),
    }
    return outcome


def pair_outcome(run_root: Path) -> dict[str, Any]:
    result = read_single_row(run_root / "result.csv")
    plugin = read_single_row(run_root / "01_host_plus_plugin" / "summary.csv")
    outcome = {
        "kind": "pair_eval",
        "status": str(result.get("status", "")).strip(),
        "current_stage": "done" if str(result.get("status", "")).strip() else "",
        "run_root": str(run_root.resolve()),
        "checkpoint": str(result.get("graph_checkpoint", "")).strip(),
        "delta_HOTA": _safe_float(result.get("delta_HOTA", 0.0), 0.0),
        "delta_AssA": _safe_float(result.get("delta_AssA", 0.0), 0.0),
        "delta_IDF1": _safe_float(result.get("delta_IDF1", 0.0), 0.0),
        "delta_MOTA": _safe_float(result.get("delta_MOTA", 0.0), 0.0),
        "delta_IDSW": _safe_int(result.get("delta_IDSW", 0), 0),
        "plugin_replaced_clusters": _safe_int(plugin.get("replaced_clusters", 0), 0),
        "plugin_delta_commit_pairs": _safe_int(plugin.get("delta_commit_pairs", 0), 0),
        "plugin_delta_drop_pairs": _safe_int(plugin.get("delta_drop_pairs", 0), 0),
        "plugin_gate_pass_clusters": _safe_int(plugin.get("gate_pass_clusters", 0), 0),
        "plugin_gate_filtered_clusters": _safe_int(plugin.get("gate_filtered_clusters", 0), 0),
        "plugin_all_defer_clusters": _safe_int(plugin.get("all_defer_clusters", 0), 0),
        "error": str(result.get("error", "")).strip() or str(plugin.get("error", "")).strip(),
    }
    return outcome


def fill_step_from_outcome(summary_csv: Path, step: str, outcome: dict[str, Any], decision: str = "") -> None:
    update_step(
        summary_csv,
        step,
        status=str(outcome.get("status", "")),
        decision=str(decision),
        run_root=str(outcome.get("run_root", "")),
        checkpoint=str(outcome.get("checkpoint", "")),
        current_stage=str(outcome.get("current_stage", "")),
        delta_HOTA=f"{_safe_float(outcome.get('delta_HOTA', 0.0), 0.0):.3f}",
        delta_AssA=f"{_safe_float(outcome.get('delta_AssA', 0.0), 0.0):.3f}",
        delta_IDF1=f"{_safe_float(outcome.get('delta_IDF1', 0.0), 0.0):.3f}",
        delta_MOTA=f"{_safe_float(outcome.get('delta_MOTA', 0.0), 0.0):.3f}",
        delta_IDSW=str(_safe_int(outcome.get("delta_IDSW", 0), 0)),
        plugin_replaced_clusters=str(_safe_int(outcome.get("plugin_replaced_clusters", 0), 0)),
        plugin_delta_commit_pairs=str(_safe_int(outcome.get("plugin_delta_commit_pairs", 0), 0)),
        plugin_delta_drop_pairs=str(_safe_int(outcome.get("plugin_delta_drop_pairs", 0), 0)),
        plugin_gate_pass_clusters=str(_safe_int(outcome.get("plugin_gate_pass_clusters", 0), 0)),
        plugin_gate_filtered_clusters=str(_safe_int(outcome.get("plugin_gate_filtered_clusters", 0), 0)),
        plugin_all_defer_clusters=str(_safe_int(outcome.get("plugin_all_defer_clusters", 0), 0)),
        error=str(outcome.get("error", "")),
    )


def _registry_base_row(queue_root: Path, summary_csv: Path) -> dict[str, str]:
    return {
        "timestamp": _now(),
        "kind": "other",
        "status": "running",
        "script": "scripts/queue_official_bytetrack_sparseedit_overnight.py",
        "dataset": "MOT17",
        "split": "overnight_queue",
        "tracker_family": "official_bytetrack",
        "variant": "official_bytetrack_sparseedit_overnight",
        "tag": queue_root.name,
        "run_root": str(queue_root.resolve()),
        "summary_csv": str(summary_csv.resolve()),
        "checkpoint": "",
        "calibrator_npz": "",
        "log_path": str((queue_root / "queue.log").resolve()),
        "notes": "overnight queue for official ByteTrack sparse-edit follow-up experiments",
    }


def _registry_fields(existing_rows: list[dict[str, str]], new_rows: list[dict[str, str]]) -> list[str]:
    fields = []
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
        step_status = str(row.get("status", "")).strip() or "pending"
        merged["status"] = step_status
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


def should_forcegate(outcome: dict[str, Any]) -> bool:
    return _safe_int(outcome.get("plugin_replaced_clusters", 0), 0) == 0


def is_drop_dominant_negative(outcome: dict[str, Any]) -> bool:
    return (
        _safe_int(outcome.get("plugin_replaced_clusters", 0), 0) > 0
        and _safe_int(outcome.get("plugin_delta_drop_pairs", 0), 0)
        > _safe_int(outcome.get("plugin_delta_commit_pairs", 0), 0)
        and _safe_float(outcome.get("delta_HOTA", 0.0), 0.0) <= 0.0
        and _safe_float(outcome.get("delta_IDF1", 0.0), 0.0) <= 0.0
    )


def wait_for_pipeline(run_root: Path, poll_sec: int, log_fp) -> dict[str, Any]:
    summary_path = run_root / "summary.csv"
    while True:
        outcome = pipeline_outcome(run_root)
        alive = process_alive(str(run_root))
        status = str(outcome.get("status", "")).strip()
        log_fp.write(
            f"[{_now()}] monitor main status={outcome['status'] or 'missing'} stage={outcome['current_stage'] or 'unknown'} alive={int(alive)}\n"
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


def run_command(cmd: list[str], log_fp) -> None:
    log_fp.write("$ " + " ".join(cmd) + "\n")
    log_fp.flush()
    subprocess.run(cmd, check=True, cwd=REPO_ROOT, stdout=log_fp, stderr=subprocess.STDOUT)
    log_fp.write("\n")
    log_fp.flush()


def build_pair_eval_cmd(
    *,
    args: argparse.Namespace,
    queue_root: Path,
    out_dir_name: str,
    experiment_name: str,
    protocol_tag: str,
    exp_file: str,
    ckpt: str,
    graph_ckpt: str,
    host_variant: str,
    topk: int,
    min_detections: int,
    min_committed_matches: int,
    max_detections: int,
    max_tracks: int,
    gate_thresh: float,
    gate_temp: float,
    gate_bias: float,
    max_commits_per_cluster: int,
    replacement_budget_ratio: float,
    max_replaced_clusters: int,
    min_commit_margin: float,
    seed: int,
) -> tuple[list[str], Path]:
    out_dir = queue_root / out_dir_name
    cmd = [
        args.python_bin,
        str(REPO_ROOT / "scripts" / "run_official_bytetrack_local_conflict_halfval_pair.py"),
        "--out-dir",
        str(out_dir.resolve()),
        "--exp-file",
        str(Path(exp_file).resolve()),
        "--ckpt",
        str(Path(ckpt).resolve()),
        "--graph-ckpt",
        str(Path(graph_ckpt).resolve()),
        "--experiment-name",
        experiment_name,
        "--protocol-tag",
        protocol_tag,
        "--host-variant",
        host_variant,
        "--graph-topk",
        str(int(topk)),
        "--graph-min-detections",
        str(int(min_detections)),
        "--graph-min-committed-matches",
        str(int(min_committed_matches)),
        "--graph-max-detections",
        str(int(max_detections)),
        "--graph-max-tracks",
        str(int(max_tracks)),
        "--graph-cluster-gate-thresh",
        str(float(gate_thresh)),
        "--graph-cluster-gate-temp",
        str(float(gate_temp)),
        "--graph-cluster-gate-bias",
        str(float(gate_bias)),
        "--graph-max-commits-per-cluster",
        str(int(max_commits_per_cluster)),
        "--graph-replacement-budget-ratio",
        str(float(replacement_budget_ratio)),
        "--graph-max-replaced-clusters",
        str(int(max_replaced_clusters)),
        "--graph-min-commit-margin",
        str(float(min_commit_margin)),
        "--seed",
        str(int(seed)),
    ]
    return cmd, out_dir


def init_summary(summary_csv: Path, main_run_root: Path) -> None:
    rows = [
        {
            "step": "01_main_sparseedit",
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
            "plugin_delta_commit_pairs": "",
            "plugin_delta_drop_pairs": "",
            "plugin_gate_pass_clusters": "",
            "plugin_gate_filtered_clusters": "",
            "plugin_all_defer_clusters": "",
            "error": "",
            "notes": "monitor current sparse-edit mainline until completion",
        },
        {
            "step": "02_forcegate0_probe",
            "status": "pending",
            "depends_on": "01_main_sparseedit",
            "decision": "",
            "script": "scripts/run_official_bytetrack_local_conflict_halfval_pair.py",
            "run_root": "",
            "checkpoint": "",
            "current_stage": "",
            "delta_HOTA": "",
            "delta_AssA": "",
            "delta_IDF1": "",
            "delta_MOTA": "",
            "delta_IDSW": "",
            "plugin_replaced_clusters": "",
            "plugin_delta_commit_pairs": "",
            "plugin_delta_drop_pairs": "",
            "plugin_gate_pass_clusters": "",
            "plugin_gate_filtered_clusters": "",
            "plugin_all_defer_clusters": "",
            "error": "",
            "notes": "run only if mainline ends as near no-op",
        },
        {
            "step": "03_tight_runtime_rerun",
            "status": "pending",
            "depends_on": "01_main_sparseedit or 02_forcegate0_probe",
            "decision": "",
            "script": "scripts/run_official_bytetrack_local_conflict_halfval_pair.py",
            "run_root": "",
            "checkpoint": "",
            "current_stage": "",
            "delta_HOTA": "",
            "delta_AssA": "",
            "delta_IDF1": "",
            "delta_MOTA": "",
            "delta_IDSW": "",
            "plugin_replaced_clusters": "",
            "plugin_delta_commit_pairs": "",
            "plugin_delta_drop_pairs": "",
            "plugin_gate_pass_clusters": "",
            "plugin_gate_filtered_clusters": "",
            "plugin_all_defer_clusters": "",
            "error": "",
            "notes": "run if outcome is still drop-dominant and non-positive",
        },
        {
            "step": "04_ultratight_runtime_rerun",
            "status": "pending",
            "depends_on": "03_tight_runtime_rerun",
            "decision": "",
            "script": "scripts/run_official_bytetrack_local_conflict_halfval_pair.py",
            "run_root": "",
            "checkpoint": "",
            "current_stage": "",
            "delta_HOTA": "",
            "delta_AssA": "",
            "delta_IDF1": "",
            "delta_MOTA": "",
            "delta_IDSW": "",
            "plugin_replaced_clusters": "",
            "plugin_delta_commit_pairs": "",
            "plugin_delta_drop_pairs": "",
            "plugin_gate_pass_clusters": "",
            "plugin_gate_filtered_clusters": "",
            "plugin_all_defer_clusters": "",
            "error": "",
            "notes": "run if tight-runtime rerun is still drop-dominant and non-positive",
        },
    ]
    write_rows(summary_csv, rows)


def main() -> int:
    args = parse_args()
    main_run_root = Path(args.main_run_root).resolve()
    if not main_run_root.is_dir():
        raise FileNotFoundError(f"Missing main run root: {main_run_root}")
    queue_root = (
        Path(args.queue_root).resolve()
        if str(args.queue_root).strip()
        else REPO_ROOT / "outputs" / f"official_bytetrack_sparseedit_overnight_queue_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
        fill_step_from_outcome(summary_csv, "01_main_sparseedit", main_outcome, decision="mainline_complete")
        sync_registry_rows(args, queue_root, summary_csv)
        if str(main_outcome.get("status", "")).strip() not in {"ok", "success"}:
            mark_remaining(summary_csv, "01_main_sparseedit", "skipped", "mainline_failed")
            sync_registry_rows(args, queue_root, summary_csv)
            log_fp.write(f"[{_now()}] mainline failed; stop queue\n")
            return 1

        ckpt = str(main_outcome.get("checkpoint", "")).strip()
        exp_file = str(main_outcome.get("official_exp_file", "")).strip()
        official_ckpt = str(main_outcome.get("official_checkpoint", "")).strip()
        host_variant = str(main_outcome.get("host_variant", "official_bytetrack")).strip() or "official_bytetrack"

        gate_thresh = float(main_outcome.get("graph_cluster_gate_thresh", 0.5))
        gate_temp = float(main_outcome.get("graph_cluster_gate_temp", 1.0))
        gate_bias = float(main_outcome.get("graph_cluster_gate_bias", 0.0))
        topk = int(main_outcome.get("graph_topk", 8))
        min_detections = int(main_outcome.get("graph_min_detections", 2))
        min_committed_matches = int(main_outcome.get("graph_min_committed_matches", 1))
        max_detections = int(main_outcome.get("graph_max_detections", 8))
        max_tracks = int(main_outcome.get("graph_max_tracks", 32))
        max_commits_per_cluster = int(main_outcome.get("graph_max_commits_per_cluster", 1))
        max_replaced_clusters = int(main_outcome.get("graph_max_replaced_clusters", 0))

        forcegate_outcome: dict[str, Any] | None = None
        if should_forcegate(main_outcome):
            update_step(summary_csv, "02_forcegate0_probe", status="running", decision="trigger_forcegate0", checkpoint=ckpt)
            sync_registry_rows(args, queue_root, summary_csv)
            cmd, run_root = build_pair_eval_cmd(
                args=args,
                queue_root=queue_root,
                out_dir_name="02_forcegate0_probe",
                experiment_name=f"{main_run_root.name}_forcegate0",
                protocol_tag="official_bytetrack_sparseedit_forcegate0_probe",
                exp_file=exp_file,
                ckpt=official_ckpt,
                graph_ckpt=ckpt,
                host_variant=host_variant,
                topk=topk,
                min_detections=min_detections,
                min_committed_matches=min_committed_matches,
                max_detections=max_detections,
                max_tracks=max_tracks,
                gate_thresh=0.0,
                gate_temp=gate_temp,
                gate_bias=gate_bias,
                max_commits_per_cluster=max_commits_per_cluster,
                replacement_budget_ratio=0.05,
                max_replaced_clusters=max_replaced_clusters,
                min_commit_margin=0.05,
                seed=42,
            )
            run_command(cmd, log_fp)
            forcegate_outcome = pair_outcome(run_root)
            fill_step_from_outcome(summary_csv, "02_forcegate0_probe", forcegate_outcome, decision="forcegate0_complete")
            sync_registry_rows(args, queue_root, summary_csv)
        else:
            update_step(summary_csv, "02_forcegate0_probe", status="skipped", decision="mainline_not_noop")
            sync_registry_rows(args, queue_root, summary_csv)

        tight_seed_outcome = forcegate_outcome if forcegate_outcome is not None else main_outcome
        tight_should_run = (
            str(tight_seed_outcome.get("status", "")).strip() == "success"
            and is_drop_dominant_negative(tight_seed_outcome)
        )
        tight_outcome: dict[str, Any] | None = None
        if tight_should_run:
            update_step(summary_csv, "03_tight_runtime_rerun", status="running", decision="trigger_tight_runtime", checkpoint=ckpt)
            sync_registry_rows(args, queue_root, summary_csv)
            cmd, run_root = build_pair_eval_cmd(
                args=args,
                queue_root=queue_root,
                out_dir_name="03_tight_runtime_rerun",
                experiment_name=f"{main_run_root.name}_tight_runtime",
                protocol_tag="official_bytetrack_sparseedit_tight_runtime",
                exp_file=exp_file,
                ckpt=official_ckpt,
                graph_ckpt=ckpt,
                host_variant=host_variant,
                topk=topk,
                min_detections=min_detections,
                min_committed_matches=min_committed_matches,
                max_detections=max_detections,
                max_tracks=max_tracks,
                gate_thresh=gate_thresh,
                gate_temp=gate_temp,
                gate_bias=gate_bias,
                max_commits_per_cluster=max_commits_per_cluster,
                replacement_budget_ratio=0.02,
                max_replaced_clusters=max_replaced_clusters,
                min_commit_margin=0.10,
                seed=42,
            )
            run_command(cmd, log_fp)
            tight_outcome = pair_outcome(run_root)
            fill_step_from_outcome(summary_csv, "03_tight_runtime_rerun", tight_outcome, decision="tight_runtime_complete")
            sync_registry_rows(args, queue_root, summary_csv)
        else:
            update_step(summary_csv, "03_tight_runtime_rerun", status="skipped", decision="drop_dominant_not_detected")
            sync_registry_rows(args, queue_root, summary_csv)

        if tight_outcome is not None and str(tight_outcome.get("status", "")).strip() == "success" and is_drop_dominant_negative(tight_outcome):
            update_step(summary_csv, "04_ultratight_runtime_rerun", status="running", decision="trigger_ultratight_runtime", checkpoint=ckpt)
            sync_registry_rows(args, queue_root, summary_csv)
            cmd, run_root = build_pair_eval_cmd(
                args=args,
                queue_root=queue_root,
                out_dir_name="04_ultratight_runtime_rerun",
                experiment_name=f"{main_run_root.name}_ultratight_runtime",
                protocol_tag="official_bytetrack_sparseedit_ultratight_runtime",
                exp_file=exp_file,
                ckpt=official_ckpt,
                graph_ckpt=ckpt,
                host_variant=host_variant,
                topk=topk,
                min_detections=min_detections,
                min_committed_matches=min_committed_matches,
                max_detections=max_detections,
                max_tracks=max_tracks,
                gate_thresh=max(gate_thresh, 0.05),
                gate_temp=gate_temp,
                gate_bias=gate_bias,
                max_commits_per_cluster=max_commits_per_cluster,
                replacement_budget_ratio=0.01,
                max_replaced_clusters=max_replaced_clusters,
                min_commit_margin=0.15,
                seed=42,
            )
            run_command(cmd, log_fp)
            ultra_outcome = pair_outcome(run_root)
            fill_step_from_outcome(summary_csv, "04_ultratight_runtime_rerun", ultra_outcome, decision="ultratight_runtime_complete")
            sync_registry_rows(args, queue_root, summary_csv)
        else:
            update_step(summary_csv, "04_ultratight_runtime_rerun", status="skipped", decision="tight_runtime_not_needed_or_already_stable")
            sync_registry_rows(args, queue_root, summary_csv)

        log_fp.write(f"[{_now()}] queue complete summary_csv={summary_csv}\n")
        log_fp.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
