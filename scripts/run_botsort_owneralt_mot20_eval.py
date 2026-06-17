#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
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
    "candidate_detections",
    "candidate_pairs",
    "rewrites",
    "owner_rows_released",
    "alt_edges_reweighted",
    "blocked_owner_reclaims",
    "event_count",
    "summary_source",
]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Run standalone BoT-SORT OwnerAlt evaluation on MOT20 val_half.")
    parser.add_argument("--run-root", default=str(REPO_ROOT / "outputs" / f"botsort_owneralt_mot20_eval_{ts}"))
    parser.add_argument("--experiment-name", default=f"mot20_owneralt_{ts}")
    parser.add_argument("--variant-name", default="botsort_owneralt_mot20")
    parser.add_argument("--registry-csv", default=str(REGISTRY_CSV))
    parser.add_argument("--plan-csv", default=str(PLAN_CSV))
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--data-root", default="/gemini/code/datasets")
    parser.add_argument("--reuse-base-eval", default=str(REPO_ROOT / "outputs" / "botsort_ltra_stage2" / "MOT20" / "base_eval"))
    parser.add_argument("--seq-ids", nargs="+", type=int, default=[1, 2, 3, 5])
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
    parser.add_argument("--owneralt-competition-min-time-since-update", type=int, default=2)
    parser.add_argument("--owneralt-competition-max-time-since-update", type=int, default=8)
    parser.add_argument("--owneralt-competition-min-tracklet-len", type=int, default=20)
    parser.add_argument("--owneralt-competition-min-box-iou", type=float, default=0.75)
    parser.add_argument("--owneralt-competition-gap1-min-box-iou", type=float, default=-1.0)
    parser.add_argument("--owneralt-competition-owner-max-tracklet-len", type=int, default=8)
    parser.add_argument("--owneralt-competition-owner-alt-det-min-score", type=float, default=0.6)
    parser.add_argument("--owneralt-competition-owner-alt-det-min-box-iou", type=float, default=0.5)
    parser.add_argument("--owneralt-competition-gap1-owner-alt-det-min-box-iou", type=float, default=-1.0)
    parser.add_argument("--owneralt-competition-max-owner-edge-deficit", type=float, default=0.10)
    parser.add_argument("--owneralt-competition-gap1-max-owner-edge-deficit", type=float, default=-1.0)
    parser.add_argument("--owneralt-competition-evidence-mode", type=str, default="legacy", choices=["legacy", "joint"])
    parser.add_argument("--owneralt-competition-max-joint-penalty", type=float, default=-1.0)
    parser.add_argument("--owneralt-competition-gap1-max-joint-penalty", type=float, default=-1.0)
    parser.add_argument("--owneralt-competition-owner-alt-bonus", type=float, default=0.10)
    parser.add_argument("--owneralt-competition-block-owner-on-reclaim", action="store_true")
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


def ensure_tracking_results(track_results_dir: Path, step_name: str) -> None:
    result_files = sorted(track_results_dir.glob("*.txt"))
    if not result_files:
        raise RuntimeError(f"{step_name} produced no tracking txt files in: {track_results_dir}")


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
        "scripts/run_botsort_owneralt_mot20_eval.py",
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
        f"owneralt_min_gap={args.owneralt_competition_min_time_since_update}",
        f"owneralt_max_gap={args.owneralt_competition_max_time_since_update}",
        f"owneralt_min_tracklet_len={args.owneralt_competition_min_tracklet_len}",
        f"owneralt_min_box_iou={args.owneralt_competition_min_box_iou}",
        f"owneralt_gap1_min_box_iou={args.owneralt_competition_gap1_min_box_iou}",
        f"owneralt_owner_max_tracklet_len={args.owneralt_competition_owner_max_tracklet_len}",
        f"owneralt_owner_alt_det_min_score={args.owneralt_competition_owner_alt_det_min_score}",
        f"owneralt_owner_alt_det_min_box_iou={args.owneralt_competition_owner_alt_det_min_box_iou}",
        f"owneralt_gap1_owner_alt_det_min_box_iou={args.owneralt_competition_gap1_owner_alt_det_min_box_iou}",
        f"owneralt_max_owner_edge_deficit={args.owneralt_competition_max_owner_edge_deficit}",
        f"owneralt_gap1_max_owner_edge_deficit={args.owneralt_competition_gap1_max_owner_edge_deficit}",
        f"owneralt_evidence_mode={args.owneralt_competition_evidence_mode}",
        f"owneralt_max_joint_penalty={args.owneralt_competition_max_joint_penalty}",
        f"owneralt_gap1_max_joint_penalty={args.owneralt_competition_gap1_max_joint_penalty}",
        f"owneralt_owner_alt_bonus={args.owneralt_competition_owner_alt_bonus}",
        f"owneralt_block_owner_on_reclaim={int(bool(args.owneralt_competition_block_owner_on_reclaim))}",
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
        "scripts/run_botsort_owneralt_mot20_eval.py",
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
        f"owneralt_min_gap={args.owneralt_competition_min_time_since_update}",
        f"owneralt_max_gap={args.owneralt_competition_max_time_since_update}",
        f"owneralt_min_tracklet_len={args.owneralt_competition_min_tracklet_len}",
        f"owneralt_min_box_iou={args.owneralt_competition_min_box_iou}",
        f"owneralt_gap1_min_box_iou={args.owneralt_competition_gap1_min_box_iou}",
        f"owneralt_owner_max_tracklet_len={args.owneralt_competition_owner_max_tracklet_len}",
        f"owneralt_owner_alt_det_min_score={args.owneralt_competition_owner_alt_det_min_score}",
        f"owneralt_owner_alt_det_min_box_iou={args.owneralt_competition_owner_alt_det_min_box_iou}",
        f"owneralt_gap1_owner_alt_det_min_box_iou={args.owneralt_competition_gap1_owner_alt_det_min_box_iou}",
        f"owneralt_max_owner_edge_deficit={args.owneralt_competition_max_owner_edge_deficit}",
        f"owneralt_gap1_max_owner_edge_deficit={args.owneralt_competition_gap1_max_owner_edge_deficit}",
        f"owneralt_evidence_mode={args.owneralt_competition_evidence_mode}",
        f"owneralt_max_joint_penalty={args.owneralt_competition_max_joint_penalty}",
        f"owneralt_gap1_max_joint_penalty={args.owneralt_competition_gap1_max_joint_penalty}",
        f"owneralt_owner_alt_bonus={args.owneralt_competition_owner_alt_bonus}",
        f"owneralt_block_owner_on_reclaim={int(bool(args.owneralt_competition_block_owner_on_reclaim))}",
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


def load_owneralt_runtime_rows(analysis_dir: Path, label: str) -> List[Dict[str, int | str]]:
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
                        "candidate_detections": int(float(row.get("candidate_detections", 0) or 0)),
                        "candidate_pairs": int(float(row.get("candidate_pairs", 0) or 0)),
                        "rewrites": int(float(row.get("rewrites", 0) or 0)),
                        "owner_rows_released": int(float(row.get("owner_rows_released", 0) or 0)),
                        "alt_edges_reweighted": int(float(row.get("alt_edges_reweighted", 0) or 0)),
                        "blocked_owner_reclaims": int(float(row.get("blocked_owner_reclaims", 0) or 0)),
                        "event_count": int(float(row.get("event_count", 0) or 0)),
                        "summary_source": str(csv_path),
                    }
                )
    return rows


def summarize_runtime_rows(runtime_rows: List[Dict[str, int | str]], label: str, seq_label: str, summary_source: str) -> Dict[str, int | str]:
    total: Dict[str, int | str] = {
        "name": label,
        "seq": seq_label,
        "frames": 0,
        "candidate_detections": 0,
        "candidate_pairs": 0,
        "rewrites": 0,
        "owner_rows_released": 0,
        "alt_edges_reweighted": 0,
        "blocked_owner_reclaims": 0,
        "event_count": 0,
        "summary_source": summary_source,
    }
    for row in runtime_rows:
        for key in [
            "frames",
            "candidate_detections",
            "candidate_pairs",
            "rewrites",
            "owner_rows_released",
            "alt_edges_reweighted",
            "blocked_owner_reclaims",
            "event_count",
        ]:
            total[key] = int(total[key]) + int(row.get(key, 0))
    return total


def resolve_base_tracker_dir(base_eval_root: Path) -> Path:
    tracker_dirs = sorted((base_eval_root / "trackers").glob("*"))
    if not tracker_dirs:
        raise FileNotFoundError(f"Missing base tracker directory under {base_eval_root}")
    tracker_dir = tracker_dirs[0] / "data"
    if not tracker_dir.is_dir():
        raise FileNotFoundError(f"Missing base tracking results under {tracker_dirs[0]}")
    return tracker_dir


def prepare_base_subset_tracker_dir(base_tracker_dir: Path, seq_names: List[str], dst_dir: Path) -> Path:
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    for seq_name in seq_names:
        src_path = base_tracker_dir / f"{seq_name}.txt"
        if not src_path.is_file():
            raise FileNotFoundError(f"Missing base tracking result for {seq_name}: {src_path}")
        shutil.copy2(src_path, dst_dir / src_path.name)
    return dst_dir


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root).expanduser().resolve()
    base_eval_root = Path(args.reuse_base_eval).expanduser().resolve()
    logs_dir = run_root / "logs"
    summary_csv = run_root / "summary.csv"
    metrics_compare_csv = run_root / "metrics_compare.csv"
    metrics_delta_csv = run_root / "metrics_delta.csv"
    per_sequence_csv = run_root / "per_sequence_metrics.csv"
    runtime_compare_csv = run_root / "runtime_compare.csv"
    runtime_per_sequence_csv = run_root / "runtime_per_sequence.csv"
    compare_log = logs_dir / "compare.log"

    base_tracker_dir = resolve_base_tracker_dir(base_eval_root)
    seq_names = [f"MOT20-{seq_id:02d}" for seq_id in args.seq_ids]
    seq_label = "|".join(seq_names)
    base_subset_tracker_dir = prepare_base_subset_tracker_dir(base_tracker_dir, seq_names, run_root / "base_tracker_subset")
    track_results_dir = BOT_ROOT / "YOLOX_outputs" / args.experiment_name / "track_results"
    base_eval_dir = run_root / "eval" / "base"
    owneralt_analysis_dir = run_root / "owneralt_analysis"
    owneralt_eval_dir = run_root / "eval" / "owneralt"
    base_tracker_name = f"{args.experiment_name}_base_reuse"
    base_summary_txt = base_eval_dir / "eval" / base_tracker_name / "pedestrian_summary.txt"
    base_detailed_csv = base_eval_dir / "eval" / base_tracker_name / "pedestrian_detailed.csv"

    rows: List[Dict[str, object]] = [
        {
            "step": "base_eval_reuse",
            "name": base_tracker_name,
            "status": "running",
            "out_dir": str(base_eval_dir),
            "summary_csv": str(summary_csv),
            "log_path": str(logs_dir / "base_eval.log"),
            "started_at": now_iso(),
            "finished_at": "",
            "notes": f"reuse base tracking from {base_subset_tracker_dir}",
        },
        {
            "step": "owneralt_track",
            "name": args.experiment_name,
            "status": "pending",
            "out_dir": str(track_results_dir),
            "summary_csv": str(summary_csv),
            "log_path": str(logs_dir / "owneralt_track.log"),
            "started_at": "",
            "finished_at": "",
            "notes": "",
        },
        {
            "step": "owneralt_eval",
            "name": args.experiment_name,
            "status": "pending",
            "out_dir": str(owneralt_eval_dir),
            "summary_csv": str(summary_csv),
            "log_path": str(logs_dir / "owneralt_eval.log"),
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
    update_plan_status(args, "running", run_root, summary_csv, logs_dir / "owneralt_track.log", notes="owneralt MOT20 eval started")
    append_registry(args, summary_csv, run_root, "running", "started BoT-SORT OwnerAlt MOT20 val_half eval", logs_dir / "owneralt_track.log")

    try:
        base_eval_cmd = [
            args.python_bin,
            str(REPO_ROOT / "scripts" / "eval_botsort_halfval_trackeval.py"),
            "--dataset",
            "MOT20",
            "--data-root",
            args.data_root,
            "--results-dir",
            str(base_subset_tracker_dir),
            "--tracker-name",
            base_tracker_name,
            "--work-dir",
            str(base_eval_dir),
            "--remap-results-from-fullval",
        ]
        rc = run_step(base_eval_cmd, logs_dir / "base_eval.log", cwd=REPO_ROOT)
        if rc != 0:
            raise RuntimeError(f"base subset evaluation failed with exit code {rc}")
        update_row(rows, "base_eval_reuse", status="success", finished_at=now_iso(), notes=f"base subset eval complete for {seq_label}")
        update_row(rows, "owneralt_track", status="running", started_at=now_iso(), notes="owneralt tracking running")
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
            "--owneralt-competition-enable",
            "--owneralt-analysis-dir",
            str(owneralt_analysis_dir),
            "--owneralt-competition-min-time-since-update",
            str(args.owneralt_competition_min_time_since_update),
            "--owneralt-competition-max-time-since-update",
            str(args.owneralt_competition_max_time_since_update),
            "--owneralt-competition-min-tracklet-len",
            str(args.owneralt_competition_min_tracklet_len),
            "--owneralt-competition-min-box-iou",
            str(args.owneralt_competition_min_box_iou),
            "--owneralt-competition-gap1-min-box-iou",
            str(args.owneralt_competition_gap1_min_box_iou),
            "--owneralt-competition-owner-max-tracklet-len",
            str(args.owneralt_competition_owner_max_tracklet_len),
            "--owneralt-competition-owner-alt-det-min-score",
            str(args.owneralt_competition_owner_alt_det_min_score),
            "--owneralt-competition-owner-alt-det-min-box-iou",
            str(args.owneralt_competition_owner_alt_det_min_box_iou),
            "--owneralt-competition-gap1-owner-alt-det-min-box-iou",
            str(args.owneralt_competition_gap1_owner_alt_det_min_box_iou),
            "--owneralt-competition-max-owner-edge-deficit",
            str(args.owneralt_competition_max_owner_edge_deficit),
            "--owneralt-competition-gap1-max-owner-edge-deficit",
            str(args.owneralt_competition_gap1_max_owner_edge_deficit),
            "--owneralt-competition-evidence-mode",
            str(args.owneralt_competition_evidence_mode),
            "--owneralt-competition-max-joint-penalty",
            str(args.owneralt_competition_max_joint_penalty),
            "--owneralt-competition-gap1-max-joint-penalty",
            str(args.owneralt_competition_gap1_max_joint_penalty),
            "--owneralt-competition-owner-alt-bonus",
            str(args.owneralt_competition_owner_alt_bonus),
        ]
        if args.owneralt_competition_block_owner_on_reclaim:
            track_cmd.append("--owneralt-competition-block-owner-on-reclaim")
        rc = run_step(track_cmd, logs_dir / "owneralt_track.log", cwd=BOT_ROOT)
        if rc != 0:
            raise RuntimeError(f"owneralt tracking failed with exit code {rc}")
        ensure_tracking_results(track_results_dir, "owneralt tracking")

        update_row(rows, "owneralt_track", status="success", finished_at=now_iso(), notes="owneralt tracking complete")
        update_row(rows, "owneralt_eval", status="running", started_at=now_iso(), notes="TrackEval running on owneralt results")
        write_rows(summary_csv, QUEUE_FIELDS, rows)

        eval_cmd = [
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
            str(owneralt_eval_dir),
            "--remap-results-from-fullval",
        ]
        rc = run_step(eval_cmd, logs_dir / "owneralt_eval.log", cwd=REPO_ROOT)
        if rc != 0:
            raise RuntimeError(f"owneralt evaluation failed with exit code {rc}")

        update_row(rows, "owneralt_eval", status="success", finished_at=now_iso(), notes="owneralt TrackEval complete")
        update_row(rows, "compare", status="running", started_at=now_iso(), notes="building compare tables")
        write_rows(summary_csv, QUEUE_FIELDS, rows)

        owneralt_summary_txt = owneralt_eval_dir / "eval" / args.experiment_name / "pedestrian_summary.txt"
        owneralt_detailed_csv = owneralt_eval_dir / "eval" / args.experiment_name / "pedestrian_detailed.csv"
        owneralt_metrics = parse_summary_txt(owneralt_summary_txt)
        base_metrics = parse_summary_txt(base_summary_txt)

        compare_rows = [
            {
                "name": "base",
                "seq": seq_label,
                "HOTA": base_metrics.get("HOTA", ""),
                "AssA": base_metrics.get("AssA", ""),
                "IDF1": base_metrics.get("IDF1", ""),
                "MOTA": base_metrics.get("MOTA", ""),
                "IDs": base_metrics.get("IDs", ""),
                "Frag": base_metrics.get("Frag", ""),
                "summary_txt": str(base_summary_txt),
                "detailed_csv": str(base_detailed_csv),
                "tracker_dir": str(base_tracker_dir),
            },
            {
                "name": "owneralt",
                "seq": seq_label,
                "HOTA": owneralt_metrics.get("HOTA", ""),
                "AssA": owneralt_metrics.get("AssA", ""),
                "IDF1": owneralt_metrics.get("IDF1", ""),
                "MOTA": owneralt_metrics.get("MOTA", ""),
                "IDs": owneralt_metrics.get("IDs", ""),
                "Frag": owneralt_metrics.get("Frag", ""),
                "summary_txt": str(owneralt_summary_txt),
                "detailed_csv": str(owneralt_detailed_csv),
                "tracker_dir": str(track_results_dir),
            },
        ]
        delta_rows = [
            {
                "name": "owneralt_minus_base",
                "seq": seq_label,
                "delta_HOTA": float(owneralt_metrics.get("HOTA", 0.0)) - float(base_metrics.get("HOTA", 0.0)),
                "delta_AssA": float(owneralt_metrics.get("AssA", 0.0)) - float(base_metrics.get("AssA", 0.0)),
                "delta_IDF1": float(owneralt_metrics.get("IDF1", 0.0)) - float(base_metrics.get("IDF1", 0.0)),
                "delta_MOTA": float(owneralt_metrics.get("MOTA", 0.0)) - float(base_metrics.get("MOTA", 0.0)),
                "delta_IDs": float(owneralt_metrics.get("IDs", 0.0)) - float(base_metrics.get("IDs", 0.0)),
                "delta_Frag": float(owneralt_metrics.get("Frag", 0.0)) - float(base_metrics.get("Frag", 0.0)),
            }
        ]
        per_sequence_rows = load_per_sequence_metrics(base_detailed_csv, "base") + load_per_sequence_metrics(owneralt_detailed_csv, "owneralt")
        write_rows(metrics_compare_csv, METRIC_FIELDS, compare_rows)
        write_rows(metrics_delta_csv, DELTA_FIELDS, delta_rows)
        write_rows(per_sequence_csv, PER_SEQUENCE_FIELDS, per_sequence_rows)

        owneralt_runtime_rows = load_owneralt_runtime_rows(owneralt_analysis_dir, "owneralt")
        runtime_compare_rows = [
            summarize_runtime_rows([], "base", seq_label, ""),
            summarize_runtime_rows(owneralt_runtime_rows, "owneralt", seq_label, str(owneralt_analysis_dir)),
        ]
        write_rows(runtime_compare_csv, RUNTIME_FIELDS, runtime_compare_rows)
        write_rows(runtime_per_sequence_csv, RUNTIME_FIELDS, owneralt_runtime_rows)

        compare_log.write_text(
            "\n".join(
                [
                    f"base_summary={base_summary_txt}",
                    f"owneralt_summary={owneralt_summary_txt}",
                    f"metrics_compare={metrics_compare_csv}",
                    f"metrics_delta={metrics_delta_csv}",
                    f"per_sequence_metrics={per_sequence_csv}",
                    f"runtime_compare={runtime_compare_csv}",
                    f"runtime_per_sequence={runtime_per_sequence_csv}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        update_row(rows, "compare", status="success", finished_at=now_iso(), notes="compare tables complete")
        write_rows(summary_csv, QUEUE_FIELDS, rows)
        update_plan_status(args, "completed", run_root, summary_csv, compare_log, notes="owneralt MOT20 eval complete")
        append_registry(args, summary_csv, run_root, "success", "completed BoT-SORT OwnerAlt MOT20 val_half eval", compare_log)
    except Exception as exc:
        mark_running_rows_failed(rows, summary_csv, str(exc))
        update_plan_status(args, "failed", run_root, summary_csv, logs_dir / "owneralt_track.log", notes=str(exc))
        append_registry(args, summary_csv, run_root, "failed", f"BoT-SORT OwnerAlt MOT20 eval failed: {exc}", logs_dir / "owneralt_track.log")
        raise


if __name__ == "__main__":
    main()
