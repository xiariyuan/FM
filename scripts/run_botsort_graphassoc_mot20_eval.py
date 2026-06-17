#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List


REPO_ROOT = Path("/gemini/code/FMtrack-main/FM-Track")
BOT_ROOT = REPO_ROOT / "external" / "BoT-SORT-main"
REGISTRY_CSV = REPO_ROOT / "outputs" / "experiment_registry.csv"
PLAN_CSV = REPO_ROOT / "outputs" / "experiment_plan.csv"

QUEUE_FIELDS = [
    "step",
    "name",
    "status",
    "out_dir",
    "summary_csv",
    "log_path",
    "started_at",
    "finished_at",
    "notes",
]

METRIC_FIELDS = [
    "name",
    "seq",
    "HOTA",
    "AssA",
    "IDF1",
    "MOTA",
    "IDs",
    "Frag",
    "summary_txt",
    "detailed_csv",
    "tracker_dir",
]

DELTA_FIELDS = [
    "name",
    "seq",
    "delta_HOTA",
    "delta_AssA",
    "delta_IDF1",
    "delta_MOTA",
    "delta_IDs",
    "delta_Frag",
]

PER_SEQUENCE_FIELDS = [
    "name",
    "seq",
    "HOTA",
    "AssA",
    "IDF1",
    "MOTA",
    "IDs",
    "Frag",
]

RUNTIME_FIELDS = [
    "name",
    "seq",
    "frames",
    "trigger_blocks",
    "changed_blocks",
    "trigger_rows",
    "trigger_cols",
    "forced_matches",
    "forced_rows",
    "suppressed_rows",
    "event_count",
    "learned_commit_scored_candidates",
    "learned_commit_margin_accept_count",
    "learned_commit_margin_reject_count",
    "learned_commit_gate_applied_count",
    "learned_commit_error_count",
    "skip_learned_commit_margin",
    "skip_learned_commit_gate",
    "summary_source",
]

DIFF_FIELDS = [
    "seq",
    "reference_file",
    "graphassoc_file",
    "identical",
    "reference_lines",
    "graphassoc_lines",
    "reference_md5",
    "graphassoc_md5",
]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Run standalone graph-association BoT-SORT evaluation on MOT20 val_half.")
    parser.add_argument("--run-root", default=str(REPO_ROOT / "outputs" / f"botsort_graphassoc_mot20_eval_{ts}"))
    parser.add_argument("--experiment-name", default=f"mot20_graphassoc_{ts}")
    parser.add_argument("--variant-name", default="botsort_graphassoc_mot20")
    parser.add_argument("--registry-csv", default=str(REGISTRY_CSV))
    parser.add_argument("--plan-csv", default=str(PLAN_CSV))
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--data-root", default="/gemini/code/datasets")
    parser.add_argument(
        "--reference-results-dirs",
        nargs="+",
        default=[
            str(BOT_ROOT / "YOLOX_outputs" / "botsort_current_base_mot20_seq2_rebuild_20260417_175642" / "track_results"),
            str(BOT_ROOT / "YOLOX_outputs" / "botsort_current_base_mot20_seq5_rebuild_20260417_162603" / "track_results"),
        ],
    )
    parser.add_argument("--seq-ids", nargs="+", type=int, default=[2, 5])
    parser.add_argument("--exp-file", default="./yolox/exps/example/mot/yolox_x_mix_mot20_ch.py")
    parser.add_argument("--ckpt", default="./pretrained/bytetrack_x_mot20.pth.tar")
    parser.add_argument("--fast-reid-config", default="fast_reid/configs/MOT20/sbs_S50.yml")
    parser.add_argument("--fast-reid-weights", default="pretrained/mot20_sbs_S50.pth")
    parser.add_argument("--cmc-method", default="file")
    parser.add_argument("--track-high-thresh", type=float, default=0.6)
    parser.add_argument("--track-low-thresh", type=float, default=0.1)
    parser.add_argument("--new-track-thresh", type=float, default=0.7)
    parser.add_argument("--track-buffer", type=int, default=30)
    parser.add_argument("--match-thresh", type=float, default=0.7)
    parser.add_argument("--proximity-thresh", type=float, default=0.5)
    parser.add_argument("--appearance-thresh", type=float, default=0.25)
    parser.add_argument("--graph-assoc-top-k", type=int, default=3)
    parser.add_argument("--graph-assoc-no-col-only-blocks", action="store_true")
    parser.add_argument("--graph-assoc-require-row-involved-strict-reclaim", action="store_true")
    parser.add_argument("--graph-assoc-max-rows", type=int, default=4)
    parser.add_argument("--graph-assoc-max-cols", type=int, default=4)
    parser.add_argument("--graph-assoc-row-margin", type=float, default=0.03)
    parser.add_argument("--graph-assoc-col-margin", type=float, default=0.03)
    parser.add_argument("--graph-assoc-min-reclaim-time-since-update", type=int, default=1)
    parser.add_argument("--graph-assoc-max-reclaim-time-since-update", type=int, default=8)
    parser.add_argument("--graph-assoc-min-reclaim-tracklet-len", type=int, default=20)
    parser.add_argument("--graph-assoc-recent-owner-max-time-since-update", type=int, default=1)
    parser.add_argument("--graph-assoc-recent-owner-max-tracklet-len", type=int, default=8)
    parser.add_argument("--graph-assoc-protect-young-active-rows", action="store_true")
    parser.add_argument("--graph-assoc-young-active-max-time-since-update", type=int, default=1)
    parser.add_argument("--graph-assoc-young-active-max-tracklet-len", type=int, default=20)
    parser.add_argument("--graph-assoc-young-active-min-reclaim-gap", type=int, default=2)
    parser.add_argument("--graph-assoc-young-active-max-cost-delta", type=float, default=-1.0)
    parser.add_argument("--graph-assoc-protect-stale-lost-owner-rows", action="store_true")
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
    parser.add_argument("--graph-assoc-allow-match-count-drop", action="store_true")
    parser.add_argument("--graph-assoc-dump-candidate-rows", action="store_true")
    parser.add_argument("--graph-assoc-candidate-rerank-top-k", type=int, default=6)
    parser.add_argument("--graph-assoc-learned-commit-rerank-candidates", action="store_true")
    parser.add_argument("--graph-assoc-commit-checkpoint", default="")
    parser.add_argument("--graph-assoc-commit-device", default="")
    parser.add_argument("--graph-assoc-commit-score-margin", type=float, default=0.0)
    parser.add_argument("--graph-assoc-commit-gate-only", action="store_true")
    parser.add_argument("--graph-assoc-commit-replace-rules", action="store_true")
    parser.add_argument("--graph-assoc-commit-decision-mode", default="")
    parser.add_argument("--graph-assoc-commit-threshold", type=float, default=float("nan"))
    parser.add_argument("--graph-assoc-commit-neutral-risk-weight", type=float, default=float("nan"))
    parser.add_argument("--graph-assoc-commit-positive-threshold", type=float, default=float("nan"))
    parser.add_argument("--graph-assoc-commit-neutral-threshold", type=float, default=float("nan"))
    parser.add_argument("--graph-assoc-commit-safety-min-gain", type=float, default=float("nan"))
    parser.add_argument("--graph-assoc-commit-safety-max-cost-delta", type=float, default=float("nan"))
    parser.add_argument("--graph-assoc-commit-safety-require-reclaim-improve", action="store_true")
    parser.add_argument("--graph-assoc-commit-safety-require-same-match-count", action="store_true")
    parser.add_argument(
        "--skip-existing-results",
        action="store_true",
        help="skip sequences whose final track result file already exists",
    )
    return parser.parse_args()


def write_rows(path: Path, fieldnames: Iterable[str], rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def update_row(rows: List[Dict[str, object]], step: str, **updates: object) -> None:
    for row in rows:
        if str(row.get("step")) == str(step):
            row.update(updates)
            return
    raise KeyError(f"Missing step row: {step}")


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


def update_plan_status(args: argparse.Namespace, status: str, run_root: Path, summary_csv: Path, log_path: Path, notes: str = "") -> None:
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
        "scripts/run_botsort_graphassoc_mot20_eval.py",
        "--dataset",
        "MOT20",
        "--split",
        "val_half",
        "--tracker-family",
        "BoT-SORT",
        "--variant",
        args.variant_name,
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
        f"seq_ids={'|'.join(str(v) for v in args.seq_ids)}",
        f"reference_results_dirs={'|'.join(str(Path(v).expanduser().resolve()) for v in args.reference_results_dirs)}",
        f"graph_assoc_top_k={args.graph_assoc_top_k}",
        f"graph_assoc_allow_col_only_blocks={int(not bool(args.graph_assoc_no_col_only_blocks))}",
        f"graph_assoc_require_row_involved_strict_reclaim={int(bool(args.graph_assoc_require_row_involved_strict_reclaim))}",
        f"graph_assoc_max_rows={args.graph_assoc_max_rows}",
        f"graph_assoc_max_cols={args.graph_assoc_max_cols}",
        f"graph_assoc_row_margin={args.graph_assoc_row_margin}",
        f"graph_assoc_col_margin={args.graph_assoc_col_margin}",
        f"graph_assoc_min_reclaim_time_since_update={args.graph_assoc_min_reclaim_time_since_update}",
        f"graph_assoc_max_reclaim_time_since_update={args.graph_assoc_max_reclaim_time_since_update}",
        f"graph_assoc_min_reclaim_tracklet_len={args.graph_assoc_min_reclaim_tracklet_len}",
        f"graph_assoc_recent_owner_max_time_since_update={args.graph_assoc_recent_owner_max_time_since_update}",
        f"graph_assoc_recent_owner_max_tracklet_len={args.graph_assoc_recent_owner_max_tracklet_len}",
        f"graph_assoc_protect_young_active_rows={int(bool(args.graph_assoc_protect_young_active_rows))}",
        f"graph_assoc_young_active_max_time_since_update={args.graph_assoc_young_active_max_time_since_update}",
        f"graph_assoc_young_active_max_tracklet_len={args.graph_assoc_young_active_max_tracklet_len}",
        f"graph_assoc_young_active_min_reclaim_gap={args.graph_assoc_young_active_min_reclaim_gap}",
        f"graph_assoc_young_active_max_cost_delta={args.graph_assoc_young_active_max_cost_delta}",
        f"graph_assoc_protect_stale_lost_owner_rows={int(bool(args.graph_assoc_protect_stale_lost_owner_rows))}",
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
        f"graph_assoc_allow_match_count_drop={int(bool(args.graph_assoc_allow_match_count_drop))}",
        f"graph_assoc_dump_candidate_rows={int(bool(args.graph_assoc_dump_candidate_rows))}",
        f"graph_assoc_candidate_rerank_top_k={args.graph_assoc_candidate_rerank_top_k}",
        f"graph_assoc_learned_commit_rerank_candidates={int(bool(args.graph_assoc_learned_commit_rerank_candidates))}",
        f"graph_assoc_commit_checkpoint={args.graph_assoc_commit_checkpoint}",
        f"graph_assoc_commit_device={args.graph_assoc_commit_device}",
        f"graph_assoc_commit_score_margin={args.graph_assoc_commit_score_margin}",
        f"graph_assoc_commit_gate_only={int(bool(args.graph_assoc_commit_gate_only))}",
        f"graph_assoc_commit_replace_rules={int(bool(args.graph_assoc_commit_replace_rules))}",
        f"graph_assoc_commit_decision_mode={args.graph_assoc_commit_decision_mode}",
        f"graph_assoc_commit_threshold={args.graph_assoc_commit_threshold}",
        f"graph_assoc_commit_neutral_risk_weight={args.graph_assoc_commit_neutral_risk_weight}",
        f"graph_assoc_commit_positive_threshold={args.graph_assoc_commit_positive_threshold}",
        f"graph_assoc_commit_neutral_threshold={args.graph_assoc_commit_neutral_threshold}",
        f"graph_assoc_commit_safety_min_gain={args.graph_assoc_commit_safety_min_gain}",
        f"graph_assoc_commit_safety_max_cost_delta={args.graph_assoc_commit_safety_max_cost_delta}",
        f"graph_assoc_commit_safety_require_reclaim_improve={int(bool(args.graph_assoc_commit_safety_require_reclaim_improve))}",
        f"graph_assoc_commit_safety_require_same_match_count={int(bool(args.graph_assoc_commit_safety_require_same_match_count))}",
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=False)


def append_registry(args: argparse.Namespace, summary_csv: Path, run_root: Path, status: str, notes: str, log_path: Path) -> None:
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
        "scripts/run_botsort_graphassoc_mot20_eval.py",
        "--dataset",
        "MOT20",
        "--split",
        "val_half",
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
        f"seq_ids={'|'.join(str(v) for v in args.seq_ids)}",
        f"reference_results_dirs={'|'.join(str(Path(v).expanduser().resolve()) for v in args.reference_results_dirs)}",
        f"graph_assoc_top_k={args.graph_assoc_top_k}",
        f"graph_assoc_allow_col_only_blocks={int(not bool(args.graph_assoc_no_col_only_blocks))}",
        f"graph_assoc_require_row_involved_strict_reclaim={int(bool(args.graph_assoc_require_row_involved_strict_reclaim))}",
        f"graph_assoc_max_rows={args.graph_assoc_max_rows}",
        f"graph_assoc_max_cols={args.graph_assoc_max_cols}",
        f"graph_assoc_row_margin={args.graph_assoc_row_margin}",
        f"graph_assoc_col_margin={args.graph_assoc_col_margin}",
        f"graph_assoc_min_reclaim_time_since_update={args.graph_assoc_min_reclaim_time_since_update}",
        f"graph_assoc_max_reclaim_time_since_update={args.graph_assoc_max_reclaim_time_since_update}",
        f"graph_assoc_min_reclaim_tracklet_len={args.graph_assoc_min_reclaim_tracklet_len}",
        f"graph_assoc_recent_owner_max_time_since_update={args.graph_assoc_recent_owner_max_time_since_update}",
        f"graph_assoc_recent_owner_max_tracklet_len={args.graph_assoc_recent_owner_max_tracklet_len}",
        f"graph_assoc_protect_young_active_rows={int(bool(args.graph_assoc_protect_young_active_rows))}",
        f"graph_assoc_young_active_max_time_since_update={args.graph_assoc_young_active_max_time_since_update}",
        f"graph_assoc_young_active_max_tracklet_len={args.graph_assoc_young_active_max_tracklet_len}",
        f"graph_assoc_young_active_min_reclaim_gap={args.graph_assoc_young_active_min_reclaim_gap}",
        f"graph_assoc_young_active_max_cost_delta={args.graph_assoc_young_active_max_cost_delta}",
        f"graph_assoc_protect_stale_lost_owner_rows={int(bool(args.graph_assoc_protect_stale_lost_owner_rows))}",
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
        f"graph_assoc_allow_match_count_drop={int(bool(args.graph_assoc_allow_match_count_drop))}",
        f"graph_assoc_dump_candidate_rows={int(bool(args.graph_assoc_dump_candidate_rows))}",
        f"graph_assoc_candidate_rerank_top_k={args.graph_assoc_candidate_rerank_top_k}",
        f"graph_assoc_learned_commit_rerank_candidates={int(bool(args.graph_assoc_learned_commit_rerank_candidates))}",
        f"graph_assoc_commit_checkpoint={args.graph_assoc_commit_checkpoint}",
        f"graph_assoc_commit_device={args.graph_assoc_commit_device}",
        f"graph_assoc_commit_score_margin={args.graph_assoc_commit_score_margin}",
        f"graph_assoc_commit_gate_only={int(bool(args.graph_assoc_commit_gate_only))}",
        f"graph_assoc_commit_replace_rules={int(bool(args.graph_assoc_commit_replace_rules))}",
        f"graph_assoc_commit_decision_mode={args.graph_assoc_commit_decision_mode}",
        f"graph_assoc_commit_threshold={args.graph_assoc_commit_threshold}",
        f"graph_assoc_commit_neutral_risk_weight={args.graph_assoc_commit_neutral_risk_weight}",
        f"graph_assoc_commit_positive_threshold={args.graph_assoc_commit_positive_threshold}",
        f"graph_assoc_commit_neutral_threshold={args.graph_assoc_commit_neutral_threshold}",
        f"graph_assoc_commit_safety_min_gain={args.graph_assoc_commit_safety_min_gain}",
        f"graph_assoc_commit_safety_max_cost_delta={args.graph_assoc_commit_safety_max_cost_delta}",
        f"graph_assoc_commit_safety_require_reclaim_improve={int(bool(args.graph_assoc_commit_safety_require_reclaim_improve))}",
        f"graph_assoc_commit_safety_require_same_match_count={int(bool(args.graph_assoc_commit_safety_require_same_match_count))}",
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
        write_rows(summary_csv, QUEUE_FIELDS, rows)


def parse_summary_txt(path: Path) -> Dict[str, float]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter=" ")
        parsed: List[List[str]] = []
        for row in reader:
            filtered = [token for token in row if token]
            if filtered:
                parsed.append(filtered)
    if len(parsed) < 2:
        raise RuntimeError(f"Unexpected TrackEval summary format: {path}")
    metrics: Dict[str, float] = {}
    for key, value in zip(parsed[0], parsed[1]):
        try:
            metrics[key] = float(value)
        except ValueError:
            continue
    return metrics


def load_per_sequence_metrics(detailed_csv: Path, label: str) -> List[Dict[str, float | str]]:
    rows: List[Dict[str, float | str]] = []
    with detailed_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            seq = str(row.get("seq", ""))
            if not seq or seq == "COMBINED":
                continue
            rows.append(
                {
                    "name": label,
                    "seq": seq,
                    "HOTA": float(row["HOTA___AUC"]) * 100.0,
                    "AssA": float(row["AssA___AUC"]) * 100.0,
                    "IDF1": float(row["IDF1"]) * 100.0,
                    "MOTA": float(row["MOTA"]) * 100.0,
                    "IDs": int(round(float(row["IDs"]))),
                    "Frag": int(round(float(row["Frag"]))),
                }
            )
    return rows


def load_graph_assoc_runtime_rows(analysis_dir: Path, label: str) -> List[Dict[str, int | str]]:
    rows: List[Dict[str, int | str]] = []
    if not analysis_dir.is_dir():
        return rows
    for csv_path in sorted(analysis_dir.glob("*_summary.csv")):
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows.append(
                    {
                        "name": label,
                        "seq": str(row.get("seq_name", "")),
                        "frames": int(float(row.get("frames", 0) or 0)),
                        "trigger_blocks": int(float(row.get("trigger_blocks", 0) or 0)),
                        "changed_blocks": int(float(row.get("changed_blocks", 0) or 0)),
                        "trigger_rows": int(float(row.get("trigger_rows", 0) or 0)),
                        "trigger_cols": int(float(row.get("trigger_cols", 0) or 0)),
                        "forced_matches": int(float(row.get("forced_matches", 0) or 0)),
                        "forced_rows": int(float(row.get("forced_rows", 0) or 0)),
                        "suppressed_rows": int(float(row.get("suppressed_rows", 0) or 0)),
                        "event_count": int(float(row.get("event_count", 0) or 0)),
                        "learned_commit_scored_candidates": int(float(row.get("learned_commit_scored_candidates", 0) or 0)),
                        "learned_commit_margin_accept_count": int(float(row.get("learned_commit_margin_accept_count", 0) or 0)),
                        "learned_commit_margin_reject_count": int(float(row.get("learned_commit_margin_reject_count", 0) or 0)),
                        "learned_commit_gate_applied_count": int(float(row.get("learned_commit_gate_applied_count", 0) or 0)),
                        "learned_commit_error_count": int(float(row.get("learned_commit_error_count", 0) or 0)),
                        "skip_learned_commit_margin": int(float(row.get("skip_learned_commit_margin", 0) or 0)),
                        "skip_learned_commit_gate": int(float(row.get("skip_learned_commit_gate", 0) or 0)),
                        "summary_source": str(csv_path),
                    }
                )
    return rows


def summarize_runtime_rows(runtime_rows: List[Dict[str, int | str]], label: str, seq_label: str, summary_source: str) -> Dict[str, int | str]:
    total: Dict[str, int | str] = {
        "name": label,
        "seq": seq_label,
        "frames": 0,
        "trigger_blocks": 0,
        "changed_blocks": 0,
        "trigger_rows": 0,
        "trigger_cols": 0,
        "forced_matches": 0,
        "forced_rows": 0,
        "suppressed_rows": 0,
        "event_count": 0,
        "learned_commit_scored_candidates": 0,
        "learned_commit_margin_accept_count": 0,
        "learned_commit_margin_reject_count": 0,
        "learned_commit_gate_applied_count": 0,
        "learned_commit_error_count": 0,
        "skip_learned_commit_margin": 0,
        "skip_learned_commit_gate": 0,
        "summary_source": summary_source,
    }
    for row in runtime_rows:
        for key in [
            "frames",
            "trigger_blocks",
            "changed_blocks",
            "trigger_rows",
            "trigger_cols",
            "forced_matches",
            "forced_rows",
            "suppressed_rows",
            "event_count",
            "learned_commit_scored_candidates",
            "learned_commit_margin_accept_count",
            "learned_commit_margin_reject_count",
            "learned_commit_gate_applied_count",
            "learned_commit_error_count",
            "skip_learned_commit_margin",
            "skip_learned_commit_gate",
        ]:
            total[key] = int(total[key]) + int(row.get(key, 0))
    return total


def md5_path(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_reference_subset(reference_dirs: List[Path], seq_names: List[str], dst_dir: Path) -> Path:
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    for seq_name in seq_names:
        found = None
        for ref_dir in reference_dirs:
            candidate = ref_dir / f"{seq_name}.txt"
            if candidate.is_file():
                found = candidate
                break
        if found is None:
            raise FileNotFoundError(f"Missing reference tracking result for {seq_name} under {reference_dirs}")
        shutil.copy2(found, dst_dir / found.name)
    return dst_dir


def compare_track_results(reference_dir: Path, graph_dir: Path, seq_names: List[str]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for seq_name in seq_names:
        reference_file = reference_dir / f"{seq_name}.txt"
        graph_file = graph_dir / f"{seq_name}.txt"
        if not reference_file.is_file():
            raise FileNotFoundError(f"Missing reference file: {reference_file}")
        if not graph_file.is_file():
            raise FileNotFoundError(f"Missing graph-assoc file: {graph_file}")
        reference_text = reference_file.read_text(encoding="utf-8")
        graph_text = graph_file.read_text(encoding="utf-8")
        rows.append(
            {
                "seq": seq_name,
                "reference_file": str(reference_file),
                "graphassoc_file": str(graph_file),
                "identical": int(reference_text == graph_text),
                "reference_lines": int(len(reference_text.splitlines())),
                "graphassoc_lines": int(len(graph_text.splitlines())),
                "reference_md5": md5_path(reference_file),
                "graphassoc_md5": md5_path(graph_file),
            }
        )
    return rows


def ensure_tracking_results(track_results_dir: Path, step_name: str) -> None:
    result_files = sorted(track_results_dir.glob("*.txt"))
    if not result_files:
        raise RuntimeError(f"{step_name} produced no tracking txt files in: {track_results_dir}")


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root).expanduser().resolve()
    logs_dir = run_root / "logs"
    summary_csv = run_root / "summary.csv"
    metrics_compare_csv = run_root / "metrics_compare.csv"
    metrics_delta_csv = run_root / "metrics_delta.csv"
    per_sequence_csv = run_root / "per_sequence_metrics.csv"
    runtime_compare_csv = run_root / "runtime_compare.csv"
    runtime_per_sequence_csv = run_root / "runtime_per_sequence.csv"
    track_diff_csv = run_root / "track_diff_summary.csv"
    compare_log = logs_dir / "compare.log"

    reference_dirs = [Path(path).expanduser().resolve() for path in args.reference_results_dirs]
    seq_names = [f"MOT20-{seq_id:02d}" for seq_id in args.seq_ids]
    seq_label = "|".join(seq_names)
    reference_subset_dir = prepare_reference_subset(reference_dirs, seq_names, run_root / "reference_tracker_subset")
    track_results_dir = BOT_ROOT / "YOLOX_outputs" / args.experiment_name / "track_results"
    reference_eval_dir = run_root / "eval" / "reference"
    graph_eval_dir = run_root / "eval" / "graph_assoc"
    graph_analysis_dir = run_root / "graph_assoc_analysis"
    reference_tracker_name = f"{args.experiment_name}_reference"
    reference_summary_txt = reference_eval_dir / "eval" / reference_tracker_name / "pedestrian_summary.txt"
    reference_detailed_csv = reference_eval_dir / "eval" / reference_tracker_name / "pedestrian_detailed.csv"

    rows: List[Dict[str, object]] = [
        {
            "step": "reference_eval",
            "name": reference_tracker_name,
            "status": "running",
            "out_dir": str(reference_eval_dir),
            "summary_csv": str(summary_csv),
            "log_path": str(logs_dir / "reference_eval.log"),
            "started_at": now_iso(),
            "finished_at": "",
            "notes": f"reference subset eval from {reference_subset_dir}",
        },
        {
            "step": "graph_track",
            "name": args.experiment_name,
            "status": "pending",
            "out_dir": str(track_results_dir),
            "summary_csv": str(summary_csv),
            "log_path": str(logs_dir / "graph_track.log"),
            "started_at": "",
            "finished_at": "",
            "notes": "",
        },
        {
            "step": "graph_eval",
            "name": args.experiment_name,
            "status": "pending",
            "out_dir": str(graph_eval_dir),
            "summary_csv": str(summary_csv),
            "log_path": str(logs_dir / "graph_eval.log"),
            "started_at": "",
            "finished_at": "",
            "notes": "",
        },
        {
            "step": "compare",
            "name": run_root.name,
            "status": "pending",
            "out_dir": str(run_root),
            "summary_csv": str(summary_csv),
            "log_path": str(compare_log),
            "started_at": "",
            "finished_at": "",
            "notes": "",
        },
    ]
    write_rows(summary_csv, QUEUE_FIELDS, rows)
    update_plan_status(args, "running", run_root, summary_csv, logs_dir / "graph_track.log", notes="graph-assoc MOT20 eval started")
    append_registry(args, summary_csv, run_root, "running", "started BoT-SORT graph-assoc MOT20 val_half eval", logs_dir / "graph_track.log")

    try:
        reference_eval_cmd = [
            args.python_bin,
            str(REPO_ROOT / "scripts" / "eval_botsort_halfval_trackeval.py"),
            "--dataset",
            "MOT20",
            "--data-root",
            args.data_root,
            "--results-dir",
            str(reference_subset_dir),
            "--tracker-name",
            reference_tracker_name,
            "--work-dir",
            str(reference_eval_dir),
            "--remap-results-from-fullval",
        ]
        rc = run_step(reference_eval_cmd, logs_dir / "reference_eval.log", cwd=REPO_ROOT)
        if rc != 0:
            raise RuntimeError(f"reference subset evaluation failed with exit code {rc}")
        update_row(rows, "reference_eval", status="success", finished_at=now_iso(), notes=f"reference subset eval complete for {seq_label}")
        update_row(rows, "graph_track", status="running", started_at=now_iso(), notes="graph-assoc tracking running")
        write_rows(summary_csv, QUEUE_FIELDS, rows)

        track_cmd = [
            args.python_bin,
            "-u",
            "tools/track.py",
            f"{args.data_root}/MOT20",
            "--benchmark",
            "MOT20",
            "--eval",
            "val",
            "--seq-ids",
            *[str(v) for v in args.seq_ids],
            "-f",
            args.exp_file,
            "-c",
            args.ckpt,
            "--with-reid",
            "--fast-reid-config",
            args.fast_reid_config,
            "--fast-reid-weights",
            args.fast_reid_weights,
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
            *(["--graph-assoc-learned-commit-rerank-candidates"] if args.graph_assoc_learned_commit_rerank_candidates else []),
            *(["--graph-assoc-no-col-only-blocks"] if args.graph_assoc_no_col_only_blocks else []),
            *(["--graph-assoc-require-row-involved-strict-reclaim"] if args.graph_assoc_require_row_involved_strict_reclaim else []),
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
        ]
        if bool(args.skip_existing_results):
            track_cmd.append("--skip-existing-results")
        if args.graph_assoc_commit_checkpoint:
            track_cmd.extend(
                [
                    "--graph-assoc-commit-checkpoint",
                    args.graph_assoc_commit_checkpoint,
                    "--graph-assoc-commit-device",
                    args.graph_assoc_commit_device,
                    "--graph-assoc-commit-score-margin",
                    str(args.graph_assoc_commit_score_margin),
                ]
            )
            if args.graph_assoc_commit_decision_mode:
                track_cmd.extend(
                    [
                        "--graph-assoc-commit-decision-mode",
                        str(args.graph_assoc_commit_decision_mode),
                    ]
                )
            if not math.isnan(float(args.graph_assoc_commit_threshold)):
                track_cmd.extend(
                    [
                        "--graph-assoc-commit-threshold",
                        str(args.graph_assoc_commit_threshold),
                    ]
                )
            if not math.isnan(float(args.graph_assoc_commit_neutral_risk_weight)):
                track_cmd.extend(
                    [
                        "--graph-assoc-commit-neutral-risk-weight",
                        str(args.graph_assoc_commit_neutral_risk_weight),
                    ]
                )
            if not math.isnan(float(args.graph_assoc_commit_positive_threshold)):
                track_cmd.extend(
                    [
                        "--graph-assoc-commit-positive-threshold",
                        str(args.graph_assoc_commit_positive_threshold),
                    ]
                )
            if not math.isnan(float(args.graph_assoc_commit_neutral_threshold)):
                track_cmd.extend(
                    [
                        "--graph-assoc-commit-neutral-threshold",
                        str(args.graph_assoc_commit_neutral_threshold),
                    ]
                )
            if not math.isnan(float(args.graph_assoc_commit_safety_min_gain)):
                track_cmd.extend(
                    [
                        "--graph-assoc-commit-safety-min-gain",
                        str(args.graph_assoc_commit_safety_min_gain),
                    ]
                )
            if not math.isnan(float(args.graph_assoc_commit_safety_max_cost_delta)):
                track_cmd.extend(
                    [
                        "--graph-assoc-commit-safety-max-cost-delta",
                        str(args.graph_assoc_commit_safety_max_cost_delta),
                    ]
                )
        if args.graph_assoc_commit_replace_rules:
            track_cmd.append("--graph-assoc-commit-replace-rules")
        if args.graph_assoc_commit_gate_only:
            track_cmd.append("--graph-assoc-commit-gate-only")
        if args.graph_assoc_commit_safety_require_reclaim_improve:
            track_cmd.append("--graph-assoc-commit-safety-require-reclaim-improve")
        if args.graph_assoc_commit_safety_require_same_match_count:
            track_cmd.append("--graph-assoc-commit-safety-require-same-match-count")
        if args.graph_assoc_protect_young_active_rows:
            track_cmd.append("--graph-assoc-protect-young-active-rows")
        if args.graph_assoc_protect_stale_lost_owner_rows:
            track_cmd.append("--graph-assoc-protect-stale-lost-owner-rows")
        if args.graph_assoc_allow_match_count_drop:
            track_cmd.append("--graph-assoc-allow-match-count-drop")
        if args.graph_assoc_dump_candidate_rows:
            track_cmd.append("--graph-assoc-dump-candidate-rows")
        rc = run_step(track_cmd, logs_dir / "graph_track.log", cwd=BOT_ROOT)
        if rc != 0:
            raise RuntimeError(f"graph-assoc tracking failed with exit code {rc}")
        ensure_tracking_results(track_results_dir, "graph-assoc tracking")

        update_row(rows, "graph_track", status="success", finished_at=now_iso(), notes="graph-assoc tracking complete")
        update_row(rows, "graph_eval", status="running", started_at=now_iso(), notes="TrackEval running on graph-assoc results")
        write_rows(summary_csv, QUEUE_FIELDS, rows)

        graph_eval_cmd = [
            args.python_bin,
            str(REPO_ROOT / "scripts" / "eval_botsort_halfval_trackeval.py"),
            "--dataset",
            "MOT20",
            "--data-root",
            args.data_root,
            "--results-dir",
            str(track_results_dir),
            "--tracker-name",
            args.experiment_name,
            "--work-dir",
            str(graph_eval_dir),
            "--remap-results-from-fullval",
        ]
        rc = run_step(graph_eval_cmd, logs_dir / "graph_eval.log", cwd=REPO_ROOT)
        if rc != 0:
            raise RuntimeError(f"graph-assoc evaluation failed with exit code {rc}")

        update_row(rows, "graph_eval", status="success", finished_at=now_iso(), notes="graph-assoc TrackEval complete")
        update_row(rows, "compare", status="running", started_at=now_iso(), notes="building compare tables")
        write_rows(summary_csv, QUEUE_FIELDS, rows)

        graph_summary_txt = graph_eval_dir / "eval" / args.experiment_name / "pedestrian_summary.txt"
        graph_detailed_csv = graph_eval_dir / "eval" / args.experiment_name / "pedestrian_detailed.csv"
        reference_metrics = parse_summary_txt(reference_summary_txt)
        graph_metrics = parse_summary_txt(graph_summary_txt)
        compare_rows = [
            {
                "name": "reference_base",
                "seq": seq_label,
                "HOTA": reference_metrics.get("HOTA", ""),
                "AssA": reference_metrics.get("AssA", ""),
                "IDF1": reference_metrics.get("IDF1", ""),
                "MOTA": reference_metrics.get("MOTA", ""),
                "IDs": int(round(reference_metrics.get("IDs", 0.0))),
                "Frag": int(round(reference_metrics.get("Frag", 0.0))),
                "summary_txt": str(reference_summary_txt),
                "detailed_csv": str(reference_detailed_csv),
                "tracker_dir": str(reference_subset_dir),
            },
            {
                "name": "graph_assoc",
                "seq": seq_label,
                "HOTA": graph_metrics.get("HOTA", ""),
                "AssA": graph_metrics.get("AssA", ""),
                "IDF1": graph_metrics.get("IDF1", ""),
                "MOTA": graph_metrics.get("MOTA", ""),
                "IDs": int(round(graph_metrics.get("IDs", 0.0))),
                "Frag": int(round(graph_metrics.get("Frag", 0.0))),
                "summary_txt": str(graph_summary_txt),
                "detailed_csv": str(graph_detailed_csv),
                "tracker_dir": str(track_results_dir),
            },
        ]
        write_rows(metrics_compare_csv, METRIC_FIELDS, compare_rows)

        delta_rows = [
            {
                "name": "graph_assoc-minus-reference_base",
                "seq": seq_label,
                "delta_HOTA": float(graph_metrics.get("HOTA", 0.0)) - float(reference_metrics.get("HOTA", 0.0)),
                "delta_AssA": float(graph_metrics.get("AssA", 0.0)) - float(reference_metrics.get("AssA", 0.0)),
                "delta_IDF1": float(graph_metrics.get("IDF1", 0.0)) - float(reference_metrics.get("IDF1", 0.0)),
                "delta_MOTA": float(graph_metrics.get("MOTA", 0.0)) - float(reference_metrics.get("MOTA", 0.0)),
                "delta_IDs": int(round(graph_metrics.get("IDs", 0.0) - reference_metrics.get("IDs", 0.0))),
                "delta_Frag": int(round(graph_metrics.get("Frag", 0.0) - reference_metrics.get("Frag", 0.0))),
            }
        ]
        write_rows(metrics_delta_csv, DELTA_FIELDS, delta_rows)

        per_sequence_rows = load_per_sequence_metrics(reference_detailed_csv, "reference_base")
        per_sequence_rows.extend(load_per_sequence_metrics(graph_detailed_csv, "graph_assoc"))
        write_rows(per_sequence_csv, PER_SEQUENCE_FIELDS, per_sequence_rows)

        runtime_rows = load_graph_assoc_runtime_rows(graph_analysis_dir, "graph_assoc")
        runtime_summary_rows = list(runtime_rows)
        runtime_summary_rows.append(
            summarize_runtime_rows(
                runtime_rows=runtime_rows,
                label="graph_assoc",
                seq_label=seq_label,
                summary_source=str(graph_analysis_dir),
            )
        )
        write_rows(runtime_per_sequence_csv, RUNTIME_FIELDS, runtime_summary_rows)
        write_rows(runtime_compare_csv, RUNTIME_FIELDS, runtime_summary_rows[-1:])

        diff_rows = compare_track_results(reference_subset_dir, track_results_dir, seq_names)
        write_rows(track_diff_csv, DIFF_FIELDS, diff_rows)

        compare_log.parent.mkdir(parents=True, exist_ok=True)
        with compare_log.open("w", encoding="utf-8") as handle:
            handle.write(f"[reference_summary] {reference_summary_txt}\n")
            handle.write(f"[graph_summary] {graph_summary_txt}\n")
            handle.write(f"[metrics_compare_csv] {metrics_compare_csv}\n")
            handle.write(f"[metrics_delta_csv] {metrics_delta_csv}\n")
            handle.write(f"[runtime_per_sequence_csv] {runtime_per_sequence_csv}\n")
            handle.write(f"[track_diff_csv] {track_diff_csv}\n")

        update_row(rows, "compare", status="success", finished_at=now_iso(), notes="compare tables complete")
        write_rows(summary_csv, QUEUE_FIELDS, rows)
        update_plan_status(args, "completed", run_root, summary_csv, compare_log, notes="graph-assoc MOT20 eval completed")
        append_registry(args, summary_csv, run_root, "success", "graph-assoc MOT20 eval complete", compare_log)
    except Exception as exc:
        mark_running_rows_failed(rows, summary_csv, str(exc))
        update_plan_status(args, "failed", run_root, summary_csv, compare_log, notes=str(exc))
        append_registry(args, summary_csv, run_root, "failed", str(exc), compare_log)
        raise


if __name__ == "__main__":
    main()
