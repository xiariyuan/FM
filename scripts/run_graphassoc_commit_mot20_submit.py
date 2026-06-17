#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List


REPO_ROOT = Path("/gemini/code/FMtrack-main/FM-Track")
BOT_ROOT = REPO_ROOT / "external" / "BoT-SORT-main"
REGISTRY_CSV = REPO_ROOT / "outputs" / "experiment_registry.csv"
PLAN_CSV = REPO_ROOT / "outputs" / "experiment_plan.csv"

DATASET_CONFIGS = {
    "MOT20": {
        "exp_file": "./yolox/exps/example/mot/yolox_x_mix_mot20_ch.py",
        "ckpt": "./pretrained/bytetrack_x_mot20.pth.tar",
        "fast_reid_config": "fast_reid/configs/MOT20/sbs_S50.yml",
        "fast_reid_weights": "pretrained/mot20_sbs_S50.pth",
        "check_script": "scripts/check_mot20_submission.py",
        "check_profile": "mot20_test_4",
        "expected_files": ["MOT20-04.txt", "MOT20-06.txt", "MOT20-07.txt", "MOT20-08.txt"],
    },
    "MOT17": {
        "exp_file": "./yolox/exps/example/mot/yolox_x_mix_det.py",
        "ckpt": "./pretrained/bytetrack_x_mot17.pth.tar",
        "fast_reid_config": "fast_reid/configs/MOT17/sbs_S50.yml",
        "fast_reid_weights": "pretrained/mot17_sbs_S50.pth",
        "check_script": "scripts/check_mot17_submission.py",
        "check_profile": "mot17_test_public_21",
        "expected_files": [
            f"MOT17-{seq:02d}-{det}.txt"
            for seq in (1, 3, 6, 7, 8, 12, 14)
            for det in ("DPM", "FRCNN", "SDP")
        ],
    },
}

SUMMARY_FIELDS = [
    "step",
    "name",
    "status",
    "out_dir",
    "summary_csv",
    "log_path",
    "started_at",
    "finished_at",
    "zip_path",
    "notes",
]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Package the graph-assoc commit mainline as a MOT test submission.")
    parser.add_argument("--dataset", default="MOT20", choices=sorted(DATASET_CONFIGS.keys()))
    parser.add_argument("--run-root", default="")
    parser.add_argument("--experiment-name", default="")
    parser.add_argument("--variant-name", default="")
    parser.add_argument("--registry-csv", default=str(REGISTRY_CSV))
    parser.add_argument("--plan-csv", default=str(PLAN_CSV))
    parser.add_argument("--python-bin", default="/root/miniconda3/bin/python")
    parser.add_argument("--data-root", default="/gemini/code/datasets")
    parser.add_argument("--cmc-method", default="file")
    parser.add_argument("--track-high-thresh", type=float, default=0.6)
    parser.add_argument("--track-low-thresh", type=float, default=0.1)
    parser.add_argument("--new-track-thresh", type=float, default=0.7)
    parser.add_argument("--track-buffer", type=int, default=30)
    parser.add_argument("--match-thresh", type=float, default=0.7)
    parser.add_argument("--proximity-thresh", type=float, default=0.5)
    parser.add_argument("--appearance-thresh", type=float, default=0.25)
    parser.add_argument("--interp-n-min", type=int, default=5)
    parser.add_argument("--interp-n-dti", type=int, default=20)
    parser.add_argument("--sanitize-precision", type=int, default=2)
    parser.add_argument("--graph-assoc-top-k", type=int, default=3)
    parser.add_argument("--graph-assoc-max-rows", type=int, default=4)
    parser.add_argument("--graph-assoc-max-cols", type=int, default=4)
    parser.add_argument("--graph-assoc-row-margin", type=float, default=0.03)
    parser.add_argument("--graph-assoc-col-margin", type=float, default=0.03)
    parser.add_argument("--graph-assoc-min-reclaim-time-since-update", type=int, default=1)
    parser.add_argument("--graph-assoc-max-reclaim-time-since-update", type=int, default=8)
    parser.add_argument("--graph-assoc-min-reclaim-tracklet-len", type=int, default=20)
    parser.add_argument("--graph-assoc-recent-owner-max-time-since-update", type=int, default=1)
    parser.add_argument("--graph-assoc-recent-owner-max-tracklet-len", type=int, default=8)
    parser.add_argument("--graph-assoc-young-active-max-time-since-update", type=int, default=1)
    parser.add_argument("--graph-assoc-young-active-max-tracklet-len", type=int, default=20)
    parser.add_argument("--graph-assoc-young-active-min-reclaim-gap", type=int, default=2)
    parser.add_argument("--graph-assoc-young-active-max-cost-delta", type=float, default=-1.0)
    parser.add_argument("--graph-assoc-stale-lost-owner-min-time-since-update", type=int, default=9)
    parser.add_argument("--graph-assoc-stale-lost-owner-min-tracklet-len", type=int, default=100)
    parser.add_argument("--graph-assoc-stale-lost-owner-active-max-time-since-update", type=int, default=1)
    parser.add_argument("--graph-assoc-stale-lost-owner-min-introduced-edge-utility", type=float, default=0.0)
    parser.add_argument("--graph-assoc-min-box-iou", type=float, default=0.6)
    parser.add_argument("--graph-assoc-reclaim-bonus", type=float, default=0.08)
    parser.add_argument("--graph-assoc-recent-owner-penalty", type=float, default=0.05)
    parser.add_argument("--graph-assoc-iou-bonus", type=float, default=0.04)
    parser.add_argument("--graph-assoc-score-bonus", type=float, default=0.02)
    parser.add_argument("--graph-assoc-min-assignment-gain", type=float, default=0.01)
    parser.add_argument("--graph-assoc-max-cost-delta", type=float, default=0.05)
    parser.add_argument("--graph-assoc-row-involved-min-assignment-gain", type=float, default=0.01)
    parser.add_argument("--graph-assoc-col-only-min-assignment-gain", type=float, default=0.01)
    parser.add_argument("--graph-assoc-col-only-max-cost-delta", type=float, default=0.05)
    parser.add_argument("--graph-assoc-force-match-cost", type=float, default=0.0)
    parser.add_argument(
        "--graph-assoc-commit-checkpoint",
        default="outputs/graph_assoc_commit_policy_balanced_policyloss_20260420_3/best.pt",
    )
    parser.add_argument("--graph-assoc-commit-device", default="cpu")
    parser.add_argument("--graph-assoc-commit-score-margin", type=float, default=0.11)
    parser.add_argument("--graph-assoc-commit-decision-mode", default="")
    parser.add_argument("--graph-assoc-commit-threshold", type=float, default=float("nan"))
    parser.add_argument("--graph-assoc-commit-neutral-risk-weight", type=float, default=float("nan"))
    parser.add_argument("--graph-assoc-commit-positive-threshold", type=float, default=float("nan"))
    parser.add_argument("--graph-assoc-commit-neutral-threshold", type=float, default=float("nan"))
    parser.add_argument("--graph-assoc-commit-safety-min-gain", type=float, default=float("nan"))
    parser.add_argument("--graph-assoc-commit-safety-max-cost-delta", type=float, default=float("nan"))
    parser.add_argument("--graph-assoc-commit-safety-require-reclaim-improve", action="store_true")
    parser.add_argument("--graph-assoc-commit-safety-require-same-match-count", action="store_true")
    parser.add_argument("--graph-assoc-learned-commit-rerank-candidates", action="store_true")
    parser.add_argument("--no-run-interpolation", action="store_true")
    parser.add_argument("--no-sanitize-results", action="store_true")
    parser.add_argument("--graph-assoc-no-col-only-blocks", action="store_true")
    parser.add_argument("--graph-assoc-require-row-involved-strict-reclaim", action="store_true")
    parser.add_argument("--graph-assoc-allow-match-count-drop", action="store_true")
    parser.add_argument("--graph-assoc-dump-candidate-rows", action="store_true")
    parser.add_argument("--graph-assoc-commit-gate-only", action="store_true")
    parser.add_argument("--no-graph-assoc-commit-replace-rules", action="store_true")
    parser.add_argument("--no-graph-assoc-protect-young-active-rows", action="store_true")
    parser.add_argument("--no-graph-assoc-protect-stale-lost-owner-rows", action="store_true")
    parser.add_argument("--zip-name", default="")
    return parser.parse_args()


def dataset_config(dataset: str) -> Dict[str, object]:
    try:
        return DATASET_CONFIGS[dataset]
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset: {dataset}") from exc


def default_run_root(dataset: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "outputs" / dataset.lower() / f"{timestamp}_graphassoc"


def write_rows(path: Path, fieldnames: Iterable[str], rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def update_row(rows: List[Dict[str, object]], step: str, **updates: object) -> None:
    for row in rows:
        if str(row.get("step", "")) == step:
            row.update(updates)
            return
    raise KeyError(f"Missing summary row: {step}")


def run_step(cmd: List[str], log_path: Path, cwd: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"[started_at] {now_iso()}\n")
        handle.write(f"[cwd] {cwd}\n")
        handle.write("[cmd] " + " ".join(cmd) + "\n\n")
        handle.flush()
        proc = subprocess.run(cmd, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT)
        handle.write(f"\n[finished_at] {now_iso()}\n")
        handle.write(f"[return_code] {proc.returncode}\n")
    return int(proc.returncode)


def update_plan_status(
    args: argparse.Namespace,
    status: str,
    run_root: Path,
    summary_csv: Path,
    log_path: Path,
    notes: str = "",
    zip_path: str = "",
) -> None:
    cmd = [
        args.python_bin,
        str(REPO_ROOT / "scripts" / "upsert_experiment_plan.py"),
        "--csv",
        str(args.plan_csv),
        "--key",
        f"run_root:{run_root}",
        "--status",
        status,
        "--kind",
        "eval",
        "--script",
        "scripts/run_graphassoc_commit_mot20_submit.py",
        "--dataset",
        args.dataset,
        "--split",
        "test",
        "--tracker-family",
        "BoT-SORT",
        "--variant",
        args.variant_name,
        "--tag",
        args.experiment_name,
        "--run-root",
        str(run_root),
        "--summary-csv",
        str(summary_csv),
        "--log-path",
        str(log_path),
        "--notes",
        notes,
        "--extra",
        f"experiment_name={args.experiment_name}",
        f"package_mode=test",
        f"zip_path={zip_path}",
        f"graph_assoc_top_k={args.graph_assoc_top_k}",
        f"graph_assoc_max_rows={args.graph_assoc_max_rows}",
        f"graph_assoc_max_cols={args.graph_assoc_max_cols}",
        f"graph_assoc_row_margin={args.graph_assoc_row_margin}",
        f"graph_assoc_col_margin={args.graph_assoc_col_margin}",
        f"graph_assoc_commit_checkpoint={args.graph_assoc_commit_checkpoint}",
        f"graph_assoc_commit_device={args.graph_assoc_commit_device}",
        f"graph_assoc_commit_score_margin={args.graph_assoc_commit_score_margin}",
        f"graph_assoc_commit_decision_mode={args.graph_assoc_commit_decision_mode}",
        f"graph_assoc_commit_threshold={args.graph_assoc_commit_threshold}",
        f"graph_assoc_commit_neutral_risk_weight={args.graph_assoc_commit_neutral_risk_weight}",
        f"graph_assoc_commit_positive_threshold={args.graph_assoc_commit_positive_threshold}",
        f"graph_assoc_commit_neutral_threshold={args.graph_assoc_commit_neutral_threshold}",
        f"graph_assoc_commit_safety_min_gain={args.graph_assoc_commit_safety_min_gain}",
        f"graph_assoc_commit_safety_max_cost_delta={args.graph_assoc_commit_safety_max_cost_delta}",
        f"graph_assoc_commit_safety_require_reclaim_improve={int(args.graph_assoc_commit_safety_require_reclaim_improve)}",
        f"graph_assoc_commit_safety_require_same_match_count={int(args.graph_assoc_commit_safety_require_same_match_count)}",
        f"graph_assoc_learned_commit_rerank_candidates={int(args.graph_assoc_learned_commit_rerank_candidates)}",
        f"graph_assoc_commit_replace_rules={int(not args.no_graph_assoc_commit_replace_rules)}",
        f"graph_assoc_commit_gate_only={int(args.graph_assoc_commit_gate_only)}",
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=False)


def append_registry(
    args: argparse.Namespace,
    summary_csv: Path,
    run_root: Path,
    status: str,
    notes: str,
    log_path: Path,
    zip_path: str = "",
) -> None:
    cmd = [
        args.python_bin,
        str(REPO_ROOT / "scripts" / "append_experiment_record.py"),
        "--csv",
        str(args.registry_csv),
        "--kind",
        "eval",
        "--status",
        status,
        "--script",
        "scripts/run_graphassoc_commit_mot20_submit.py",
        "--dataset",
        args.dataset,
        "--split",
        "test",
        "--tracker-family",
        "BoT-SORT",
        "--variant",
        args.variant_name,
        "--tag",
        args.experiment_name,
        "--run-root",
        str(run_root),
        "--summary-csv",
        str(summary_csv),
        "--log-path",
        str(log_path),
        "--notes",
        notes,
        "--extra",
        f"package_mode=test",
        f"zip_path={zip_path}",
        f"graph_assoc_commit_checkpoint={args.graph_assoc_commit_checkpoint}",
        f"graph_assoc_commit_device={args.graph_assoc_commit_device}",
        f"graph_assoc_commit_score_margin={args.graph_assoc_commit_score_margin}",
        f"graph_assoc_commit_decision_mode={args.graph_assoc_commit_decision_mode}",
        f"graph_assoc_commit_safety_min_gain={args.graph_assoc_commit_safety_min_gain}",
        f"graph_assoc_commit_safety_max_cost_delta={args.graph_assoc_commit_safety_max_cost_delta}",
        f"graph_assoc_commit_safety_require_reclaim_improve={int(args.graph_assoc_commit_safety_require_reclaim_improve)}",
        f"graph_assoc_commit_safety_require_same_match_count={int(args.graph_assoc_commit_safety_require_same_match_count)}",
        f"graph_assoc_learned_commit_rerank_candidates={int(args.graph_assoc_learned_commit_rerank_candidates)}",
        f"graph_assoc_commit_replace_rules={int(not args.no_graph_assoc_commit_replace_rules)}",
        f"graph_assoc_commit_gate_only={int(args.graph_assoc_commit_gate_only)}",
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=False)


def mark_running_rows_failed(rows: List[Dict[str, object]], summary_csv: Path, reason: str) -> None:
    finished_at = now_iso()
    changed = False
    for row in rows:
        status = str(row.get("status", ""))
        if status == "running":
            row["status"] = "failed"
            row["finished_at"] = finished_at
            row["notes"] = f"{row.get('notes', '')} | failed: {reason}".strip()
            changed = True
        elif status == "pending":
            row["status"] = "cancelled"
            row["finished_at"] = finished_at
            row["notes"] = f"{row.get('notes', '')} | cancelled_after_failure: {reason}".strip()
            changed = True
    if changed:
        write_rows(summary_csv, SUMMARY_FIELDS, rows)


def expected_files_present(directory: Path, expected_files: List[str]) -> bool:
    return all((directory / name).is_file() for name in expected_files)


def zip_submission(package_dir: Path, zip_path: Path, expected_files: List[str]) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for name in expected_files:
            src = package_dir / name
            if not src.is_file():
                raise FileNotFoundError(f"Missing expected result file: {src}")
            handle.write(src, arcname=name)


def write_meta(args: argparse.Namespace, run_root: Path, zip_name: str) -> None:
    protect_young = int(not args.no_graph_assoc_protect_young_active_rows)
    protect_stale = int(not args.no_graph_assoc_protect_stale_lost_owner_rows)
    replace_rules = int(not args.no_graph_assoc_commit_replace_rules)
    lines = [
        f"dataset={args.dataset}",
        "package_mode=test",
        f"experiment_name={args.experiment_name}",
        f"variant_name={args.variant_name}",
        f"out_dir={run_root}",
        f"zip_name={zip_name}",
        f"data_root={args.data_root}",
        f"run_interpolation={int(not args.no_run_interpolation)}",
        f"sanitize_results={int(not args.no_sanitize_results)}",
        f"sanitize_precision={args.sanitize_precision}",
        f"graph_assoc_top_k={args.graph_assoc_top_k}",
        f"graph_assoc_max_rows={args.graph_assoc_max_rows}",
        f"graph_assoc_max_cols={args.graph_assoc_max_cols}",
        f"graph_assoc_row_margin={args.graph_assoc_row_margin}",
        f"graph_assoc_col_margin={args.graph_assoc_col_margin}",
        f"graph_assoc_min_reclaim_time_since_update={args.graph_assoc_min_reclaim_time_since_update}",
        f"graph_assoc_max_reclaim_time_since_update={args.graph_assoc_max_reclaim_time_since_update}",
        f"graph_assoc_min_reclaim_tracklet_len={args.graph_assoc_min_reclaim_tracklet_len}",
        f"graph_assoc_recent_owner_max_time_since_update={args.graph_assoc_recent_owner_max_time_since_update}",
        f"graph_assoc_recent_owner_max_tracklet_len={args.graph_assoc_recent_owner_max_tracklet_len}",
        f"graph_assoc_protect_young_active_rows={protect_young}",
        f"graph_assoc_young_active_max_time_since_update={args.graph_assoc_young_active_max_time_since_update}",
        f"graph_assoc_young_active_max_tracklet_len={args.graph_assoc_young_active_max_tracklet_len}",
        f"graph_assoc_young_active_min_reclaim_gap={args.graph_assoc_young_active_min_reclaim_gap}",
        f"graph_assoc_young_active_max_cost_delta={args.graph_assoc_young_active_max_cost_delta}",
        f"graph_assoc_protect_stale_lost_owner_rows={protect_stale}",
        f"graph_assoc_stale_lost_owner_min_time_since_update={args.graph_assoc_stale_lost_owner_min_time_since_update}",
        f"graph_assoc_stale_lost_owner_min_tracklet_len={args.graph_assoc_stale_lost_owner_min_tracklet_len}",
        f"graph_assoc_stale_lost_owner_active_max_time_since_update={args.graph_assoc_stale_lost_owner_active_max_time_since_update}",
        f"graph_assoc_stale_lost_owner_min_introduced_edge_utility={args.graph_assoc_stale_lost_owner_min_introduced_edge_utility}",
        f"graph_assoc_min_box_iou={args.graph_assoc_min_box_iou}",
        f"graph_assoc_reclaim_bonus={args.graph_assoc_reclaim_bonus}",
        f"graph_assoc_recent_owner_penalty={args.graph_assoc_recent_owner_penalty}",
        f"graph_assoc_iou_bonus={args.graph_assoc_iou_bonus}",
        f"graph_assoc_score_bonus={args.graph_assoc_score_bonus}",
        f"graph_assoc_min_assignment_gain={args.graph_assoc_min_assignment_gain}",
        f"graph_assoc_max_cost_delta={args.graph_assoc_max_cost_delta}",
        f"graph_assoc_row_involved_min_assignment_gain={args.graph_assoc_row_involved_min_assignment_gain}",
        f"graph_assoc_col_only_min_assignment_gain={args.graph_assoc_col_only_min_assignment_gain}",
        f"graph_assoc_col_only_max_cost_delta={args.graph_assoc_col_only_max_cost_delta}",
        f"graph_assoc_force_match_cost={args.graph_assoc_force_match_cost}",
        f"graph_assoc_allow_match_count_drop={int(args.graph_assoc_allow_match_count_drop)}",
        f"graph_assoc_dump_candidate_rows={int(args.graph_assoc_dump_candidate_rows)}",
        f"graph_assoc_commit_checkpoint={args.graph_assoc_commit_checkpoint}",
        f"graph_assoc_commit_device={args.graph_assoc_commit_device}",
        f"graph_assoc_commit_score_margin={args.graph_assoc_commit_score_margin}",
        f"graph_assoc_commit_decision_mode={args.graph_assoc_commit_decision_mode}",
        f"graph_assoc_commit_threshold={args.graph_assoc_commit_threshold}",
        f"graph_assoc_commit_neutral_risk_weight={args.graph_assoc_commit_neutral_risk_weight}",
        f"graph_assoc_commit_positive_threshold={args.graph_assoc_commit_positive_threshold}",
        f"graph_assoc_commit_neutral_threshold={args.graph_assoc_commit_neutral_threshold}",
        f"graph_assoc_commit_safety_min_gain={args.graph_assoc_commit_safety_min_gain}",
        f"graph_assoc_commit_safety_max_cost_delta={args.graph_assoc_commit_safety_max_cost_delta}",
        f"graph_assoc_commit_safety_require_reclaim_improve={int(args.graph_assoc_commit_safety_require_reclaim_improve)}",
        f"graph_assoc_commit_safety_require_same_match_count={int(args.graph_assoc_commit_safety_require_same_match_count)}",
        f"graph_assoc_learned_commit_rerank_candidates={int(args.graph_assoc_learned_commit_rerank_candidates)}",
        f"graph_assoc_commit_gate_only={int(args.graph_assoc_commit_gate_only)}",
        f"graph_assoc_commit_replace_rules={replace_rules}",
        f"start_time={datetime.now().astimezone().strftime('%F %T %z')}",
    ]
    (run_root / "meta.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = dataset_config(args.dataset)
    if not args.run_root:
        args.run_root = str(default_run_root(args.dataset))
    run_root = Path(args.run_root).expanduser().resolve()
    if not args.experiment_name:
        args.experiment_name = run_root.name
    margin_tag = str(args.graph_assoc_commit_score_margin).replace(".", "")
    if not args.variant_name:
        args.variant_name = f"graphassoc_commit_margin{margin_tag}"
    if not args.zip_name:
        args.zip_name = f"{run_root.name}.zip"
    expected_files = list(config["expected_files"])  # type: ignore[arg-type]
    logs_dir = run_root / "logs"
    summary_csv = run_root / "summary.csv"
    track_results_dir = BOT_ROOT / "YOLOX_outputs" / args.experiment_name / "track_results"
    graph_analysis_dir = run_root / "graph_assoc_analysis"
    sanitized_dir = run_root / "sanitized_results"
    zip_path = run_root / args.zip_name

    rows: List[Dict[str, object]] = [
        {
            "step": "track_test",
            "name": args.experiment_name,
            "status": "running",
            "out_dir": str(track_results_dir),
            "summary_csv": str(summary_csv),
            "log_path": str(logs_dir / "track_test.log"),
            "started_at": now_iso(),
            "finished_at": "",
            "zip_path": "",
            "notes": f"{args.dataset} test tracking running",
        },
        {
            "step": "interpolation",
            "name": f"{args.experiment_name}_interpolation",
            "status": "pending",
            "out_dir": str(track_results_dir),
            "summary_csv": str(summary_csv),
            "log_path": str(logs_dir / "interpolation.log"),
            "started_at": "",
            "finished_at": "",
            "zip_path": "",
            "notes": "",
        },
        {
            "step": "sanitize",
            "name": f"{args.experiment_name}_sanitize",
            "status": "pending",
            "out_dir": str(sanitized_dir),
            "summary_csv": str(summary_csv),
            "log_path": str(logs_dir / "sanitize.log"),
            "started_at": "",
            "finished_at": "",
            "zip_path": "",
            "notes": "",
        },
        {
            "step": "package",
            "name": f"{args.experiment_name}_package",
            "status": "pending",
            "out_dir": str(run_root),
            "summary_csv": str(summary_csv),
            "log_path": str(logs_dir / "package.log"),
            "started_at": "",
            "finished_at": "",
            "zip_path": str(zip_path),
            "notes": "",
        },
        {
            "step": "precheck",
            "name": f"{args.experiment_name}_precheck",
            "status": "pending",
            "out_dir": str(run_root),
            "summary_csv": str(summary_csv),
            "log_path": str(run_root / "precheck.log"),
            "started_at": "",
            "finished_at": "",
            "zip_path": str(zip_path),
            "notes": "",
        },
    ]

    run_root.mkdir(parents=True, exist_ok=True)
    write_rows(summary_csv, SUMMARY_FIELDS, rows)
    write_meta(args, run_root, args.zip_name)
    update_plan_status(
        args,
        "running",
        run_root,
        summary_csv,
        logs_dir / "track_test.log",
        notes=f"graphassoc commit {args.dataset} test submission started",
    )
    append_registry(
        args,
        summary_csv,
        run_root,
        "running",
        f"graphassoc commit {args.dataset} test submission started",
        logs_dir / "track_test.log",
    )

    try:
        track_cmd = [
            args.python_bin,
            "-u",
            "tools/track.py",
            f"{args.data_root}/{args.dataset}",
            "--benchmark",
            args.dataset,
            "--eval",
            "test",
            "-f",
            str(config["exp_file"]),
            "-c",
            str(config["ckpt"]),
            "--with-reid",
            "--fast-reid-config",
            str(config["fast_reid_config"]),
            "--fast-reid-weights",
            str(config["fast_reid_weights"]),
            "--cmc-method",
            args.cmc_method,
            "--fuse",
            "--experiment-name",
            args.experiment_name,
            "--run-manifest-path",
            str(run_root / "run_manifest.json"),
            "--track_high_thresh",
            str(args.track_high_thresh),
            "--track_low_thresh",
            str(args.track_low_thresh),
            "--new_track_thresh",
            str(args.new_track_thresh),
            "--track_buffer",
            str(args.track_buffer),
            "--match_thresh",
            str(args.match_thresh),
            "--proximity_thresh",
            str(args.proximity_thresh),
            "--appearance_thresh",
            str(args.appearance_thresh),
            "--graph-assoc-enable",
            "--graph-assoc-analysis-dir",
            str(graph_analysis_dir),
            "--graph-assoc-top-k",
            str(args.graph_assoc_top_k),
            "--graph-assoc-max-rows",
            str(args.graph_assoc_max_rows),
            "--graph-assoc-max-cols",
            str(args.graph_assoc_max_cols),
            "--graph-assoc-row-margin",
            str(args.graph_assoc_row_margin),
            "--graph-assoc-col-margin",
            str(args.graph_assoc_col_margin),
            "--graph-assoc-min-reclaim-time-since-update",
            str(args.graph_assoc_min_reclaim_time_since_update),
            "--graph-assoc-max-reclaim-time-since-update",
            str(args.graph_assoc_max_reclaim_time_since_update),
            "--graph-assoc-min-reclaim-tracklet-len",
            str(args.graph_assoc_min_reclaim_tracklet_len),
            "--graph-assoc-recent-owner-max-time-since-update",
            str(args.graph_assoc_recent_owner_max_time_since_update),
            "--graph-assoc-recent-owner-max-tracklet-len",
            str(args.graph_assoc_recent_owner_max_tracklet_len),
            "--graph-assoc-young-active-max-time-since-update",
            str(args.graph_assoc_young_active_max_time_since_update),
            "--graph-assoc-young-active-max-tracklet-len",
            str(args.graph_assoc_young_active_max_tracklet_len),
            "--graph-assoc-young-active-min-reclaim-gap",
            str(args.graph_assoc_young_active_min_reclaim_gap),
            "--graph-assoc-young-active-max-cost-delta",
            str(args.graph_assoc_young_active_max_cost_delta),
            "--graph-assoc-stale-lost-owner-min-time-since-update",
            str(args.graph_assoc_stale_lost_owner_min_time_since_update),
            "--graph-assoc-stale-lost-owner-min-tracklet-len",
            str(args.graph_assoc_stale_lost_owner_min_tracklet_len),
            "--graph-assoc-stale-lost-owner-active-max-time-since-update",
            str(args.graph_assoc_stale_lost_owner_active_max_time_since_update),
            "--graph-assoc-stale-lost-owner-min-introduced-edge-utility",
            str(args.graph_assoc_stale_lost_owner_min_introduced_edge_utility),
            "--graph-assoc-min-box-iou",
            str(args.graph_assoc_min_box_iou),
            "--graph-assoc-reclaim-bonus",
            str(args.graph_assoc_reclaim_bonus),
            "--graph-assoc-recent-owner-penalty",
            str(args.graph_assoc_recent_owner_penalty),
            "--graph-assoc-iou-bonus",
            str(args.graph_assoc_iou_bonus),
            "--graph-assoc-score-bonus",
            str(args.graph_assoc_score_bonus),
            "--graph-assoc-min-assignment-gain",
            str(args.graph_assoc_min_assignment_gain),
            "--graph-assoc-max-cost-delta",
            str(args.graph_assoc_max_cost_delta),
            "--graph-assoc-row-involved-min-assignment-gain",
            str(args.graph_assoc_row_involved_min_assignment_gain),
            "--graph-assoc-col-only-min-assignment-gain",
            str(args.graph_assoc_col_only_min_assignment_gain),
            "--graph-assoc-col-only-max-cost-delta",
            str(args.graph_assoc_col_only_max_cost_delta),
            "--graph-assoc-force-match-cost",
            str(args.graph_assoc_force_match_cost),
            "--graph-assoc-commit-checkpoint",
            args.graph_assoc_commit_checkpoint,
            "--graph-assoc-commit-device",
            args.graph_assoc_commit_device,
            "--graph-assoc-commit-score-margin",
            str(args.graph_assoc_commit_score_margin),
        ]
        if not math.isnan(float(args.graph_assoc_commit_threshold)):
            track_cmd.extend(["--graph-assoc-commit-threshold", str(args.graph_assoc_commit_threshold)])
        if not math.isnan(float(args.graph_assoc_commit_neutral_risk_weight)):
            track_cmd.extend(["--graph-assoc-commit-neutral-risk-weight", str(args.graph_assoc_commit_neutral_risk_weight)])
        if not math.isnan(float(args.graph_assoc_commit_positive_threshold)):
            track_cmd.extend(["--graph-assoc-commit-positive-threshold", str(args.graph_assoc_commit_positive_threshold)])
        if not math.isnan(float(args.graph_assoc_commit_neutral_threshold)):
            track_cmd.extend(["--graph-assoc-commit-neutral-threshold", str(args.graph_assoc_commit_neutral_threshold)])
        if not math.isnan(float(args.graph_assoc_commit_safety_min_gain)):
            track_cmd.extend(["--graph-assoc-commit-safety-min-gain", str(args.graph_assoc_commit_safety_min_gain)])
        if not math.isnan(float(args.graph_assoc_commit_safety_max_cost_delta)):
            track_cmd.extend(["--graph-assoc-commit-safety-max-cost-delta", str(args.graph_assoc_commit_safety_max_cost_delta)])
        if args.graph_assoc_commit_safety_require_reclaim_improve:
            track_cmd.append("--graph-assoc-commit-safety-require-reclaim-improve")
        if args.graph_assoc_commit_safety_require_same_match_count:
            track_cmd.append("--graph-assoc-commit-safety-require-same-match-count")
        if args.graph_assoc_learned_commit_rerank_candidates:
            track_cmd.append("--graph-assoc-learned-commit-rerank-candidates")
        if args.graph_assoc_commit_decision_mode:
            track_cmd.extend(["--graph-assoc-commit-decision-mode", args.graph_assoc_commit_decision_mode])
        if args.graph_assoc_no_col_only_blocks:
            track_cmd.append("--graph-assoc-no-col-only-blocks")
        if args.graph_assoc_require_row_involved_strict_reclaim:
            track_cmd.append("--graph-assoc-require-row-involved-strict-reclaim")
        if not args.no_graph_assoc_protect_young_active_rows:
            track_cmd.append("--graph-assoc-protect-young-active-rows")
        if not args.no_graph_assoc_protect_stale_lost_owner_rows:
            track_cmd.append("--graph-assoc-protect-stale-lost-owner-rows")
        if args.graph_assoc_allow_match_count_drop:
            track_cmd.append("--graph-assoc-allow-match-count-drop")
        if args.graph_assoc_dump_candidate_rows:
            track_cmd.append("--graph-assoc-dump-candidate-rows")
        if args.graph_assoc_commit_gate_only:
            track_cmd.append("--graph-assoc-commit-gate-only")
        if not args.no_graph_assoc_commit_replace_rules:
            track_cmd.append("--graph-assoc-commit-replace-rules")

        if expected_files_present(track_results_dir, expected_files):
            update_row(rows, "track_test", status="success", finished_at=now_iso(), notes=f"reused existing {args.dataset} test tracking results")
        else:
            rc = run_step(track_cmd, logs_dir / "track_test.log", cwd=BOT_ROOT)
            if rc != 0:
                raise RuntimeError(f"track_test failed with exit code {rc}")
            if not expected_files_present(track_results_dir, expected_files):
                raise FileNotFoundError(f"Missing expected tracking outputs under {track_results_dir}")
            update_row(rows, "track_test", status="success", finished_at=now_iso(), notes=f"{args.dataset} test tracking complete")
        write_rows(summary_csv, SUMMARY_FIELDS, rows)

        if args.no_run_interpolation:
            update_row(rows, "interpolation", status="success", started_at=now_iso(), finished_at=now_iso(), notes="interpolation disabled")
        else:
            update_row(rows, "interpolation", status="running", started_at=now_iso(), notes="interpolation running")
            write_rows(summary_csv, SUMMARY_FIELDS, rows)
            rc = run_step(
                [
                    args.python_bin,
                    "tools/interpolation.py",
                    "--txt_path",
                    str(track_results_dir),
                    "--n_min",
                    str(args.interp_n_min),
                    "--n_dti",
                    str(args.interp_n_dti),
                ],
                logs_dir / "interpolation.log",
                cwd=BOT_ROOT,
            )
            if rc != 0:
                raise RuntimeError(f"interpolation failed with exit code {rc}")
            update_row(rows, "interpolation", status="success", finished_at=now_iso(), notes="interpolation complete")
        write_rows(summary_csv, SUMMARY_FIELDS, rows)

        package_dir = track_results_dir
        if args.no_sanitize_results:
            update_row(rows, "sanitize", status="success", started_at=now_iso(), finished_at=now_iso(), notes="sanitize disabled")
        else:
            update_row(rows, "sanitize", status="running", started_at=now_iso(), notes="sanitize running")
            write_rows(summary_csv, SUMMARY_FIELDS, rows)
            if not expected_files_present(sanitized_dir, expected_files):
                rc = run_step(
                    [
                        args.python_bin,
                        str(REPO_ROOT / "scripts" / "sanitize_mot_submission.py"),
                        "--input-dir",
                        str(track_results_dir),
                        "--output-dir",
                        str(sanitized_dir),
                        "--data-root",
                        args.data_root,
                        "--benchmark",
                        args.dataset,
                        "--precision",
                        str(args.sanitize_precision),
                    ],
                    logs_dir / "sanitize.log",
                    cwd=REPO_ROOT,
                )
                if rc != 0:
                    raise RuntimeError(f"sanitize failed with exit code {rc}")
            if not expected_files_present(sanitized_dir, expected_files):
                raise FileNotFoundError(f"Missing expected sanitized outputs under {sanitized_dir}")
            package_dir = sanitized_dir
            update_row(rows, "sanitize", status="success", finished_at=now_iso(), notes="sanitize complete")
        write_rows(summary_csv, SUMMARY_FIELDS, rows)

        update_row(rows, "package", status="running", started_at=now_iso(), notes="packaging submission zip")
        write_rows(summary_csv, SUMMARY_FIELDS, rows)
        zip_submission(package_dir, zip_path, expected_files)
        (run_root / "latest_zip.txt").write_text(str(zip_path) + "\n", encoding="utf-8")
        update_row(rows, "package", status="success", finished_at=now_iso(), zip_path=str(zip_path), notes=f"zip ready from {package_dir}")
        write_rows(summary_csv, SUMMARY_FIELDS, rows)

        update_row(rows, "precheck", status="running", started_at=now_iso(), notes="submission precheck running")
        write_rows(summary_csv, SUMMARY_FIELDS, rows)
        rc = run_step(
            [
                args.python_bin,
                str(REPO_ROOT / str(config["check_script"])),
                "--zip-path",
                str(zip_path),
                "--profile",
                str(config["check_profile"]),
            ],
            run_root / "precheck.log",
            cwd=REPO_ROOT,
        )
        if rc != 0:
            raise RuntimeError(f"precheck failed with exit code {rc}")
        update_row(rows, "precheck", status="success", finished_at=now_iso(), zip_path=str(zip_path), notes="submission precheck passed")
        write_rows(summary_csv, SUMMARY_FIELDS, rows)

        update_plan_status(
            args,
            "completed",
            run_root,
            summary_csv,
            run_root / "precheck.log",
            notes=f"graphassoc commit {args.dataset} test submission complete",
            zip_path=str(zip_path),
        )
        append_registry(
            args,
            summary_csv,
            run_root,
            "success",
            f"graphassoc commit {args.dataset} test submission complete",
            run_root / "precheck.log",
            zip_path=str(zip_path),
        )
    except Exception as exc:
        mark_running_rows_failed(rows, summary_csv, str(exc))
        update_plan_status(
            args,
            "failed",
            run_root,
            summary_csv,
            logs_dir / "track_test.log",
            notes=str(exc),
            zip_path=str(zip_path),
        )
        append_registry(
            args,
            summary_csv,
            run_root,
            "failed",
            f"graphassoc commit {args.dataset} test submission failed: {exc}",
            logs_dir / "track_test.log",
            zip_path=str(zip_path),
        )
        raise


if __name__ == "__main__":
    main()
