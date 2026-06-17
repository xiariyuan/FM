#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Run current-code BoT-SORT base evaluation on MOT20 val_half.")
    parser.add_argument("--run-root", default=str(REPO_ROOT / "outputs" / f"botsort_current_base_mot20_eval_{ts}"))
    parser.add_argument("--experiment-name", default=f"mot20_current_base_{ts}")
    parser.add_argument("--variant-name", default="botsort_current_base_mot20")
    parser.add_argument("--registry-csv", default=str(REGISTRY_CSV))
    parser.add_argument("--plan-csv", default=str(PLAN_CSV))
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--data-root", default="/gemini/code/datasets")
    parser.add_argument(
        "--reference-results-dir",
        default=str(REPO_ROOT / "outputs" / "botsort_ltra_stage2" / "MOT20" / "base_eval" / "trackers" / "botsort_mot20_val_base" / "data"),
    )
    parser.add_argument("--reference-label", default="reference_base")
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
    parser.add_argument("--reentry-memory-enable", action="store_true")
    parser.add_argument("--reentry-memory-use-low-score", action="store_true")
    parser.add_argument("--reentry-memory-max-gap", type=int, default=60)
    parser.add_argument("--reentry-memory-max-size", type=int, default=256)
    parser.add_argument("--reentry-memory-min-similarity", type=float, default=0.60)
    parser.add_argument("--reentry-memory-confirm-streak", type=int, default=2)
    parser.add_argument("--reentry-memory-confirm-gap", type=int, default=2)
    parser.add_argument("--reentry-memory-confirm-min-similarity", type=float, default=0.65)
    parser.add_argument("--reentry-memory-min-det-score", type=float, default=0.10)
    parser.add_argument("--reentry-memory-appearance-weight", type=float, default=0.55)
    parser.add_argument("--reentry-memory-iou-weight", type=float, default=0.25)
    parser.add_argument("--reentry-memory-score-weight", type=float, default=0.10)
    parser.add_argument("--reentry-memory-gap-weight", type=float, default=0.10)
    parser.add_argument("--reentry-memory-compete-primary", action="store_true")
    parser.add_argument("--reentry-engine-enable", action="store_true")
    parser.add_argument("--reentry-engine-hilbert-order", type=int, default=8)
    parser.add_argument("--reentry-engine-bf-threshold", type=int, default=50)
    parser.add_argument("--reentry-engine-spatial-radius", type=int, default=2)
    parser.add_argument("--reentry-engine-max-spatial-radius", type=int, default=4)
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
        "scripts/run_botsort_current_base_mot20_eval.py",
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
        f"reference_results_dir={args.reference_results_dir}",
        f"track_high_thresh={args.track_high_thresh}",
        f"track_low_thresh={args.track_low_thresh}",
        f"new_track_thresh={args.new_track_thresh}",
        f"track_buffer={args.track_buffer}",
        f"match_thresh={args.match_thresh}",
        f"proximity_thresh={args.proximity_thresh}",
        f"appearance_thresh={args.appearance_thresh}",
        f"reentry_memory_enable={int(bool(args.reentry_memory_enable))}",
        f"reentry_memory_use_low_score={int(bool(args.reentry_memory_use_low_score))}",
        f"reentry_memory_max_gap={args.reentry_memory_max_gap}",
        f"reentry_memory_max_size={args.reentry_memory_max_size}",
        f"reentry_memory_min_similarity={args.reentry_memory_min_similarity}",
        f"reentry_memory_confirm_streak={args.reentry_memory_confirm_streak}",
        f"reentry_memory_confirm_gap={args.reentry_memory_confirm_gap}",
        f"reentry_memory_confirm_min_similarity={args.reentry_memory_confirm_min_similarity}",
        f"reentry_memory_min_det_score={args.reentry_memory_min_det_score}",
        f"reentry_memory_app_weight={args.reentry_memory_appearance_weight}",
        f"reentry_memory_iou_weight={args.reentry_memory_iou_weight}",
        f"reentry_memory_score_weight={args.reentry_memory_score_weight}",
        f"reentry_memory_gap_weight={args.reentry_memory_gap_weight}",
        f"reentry_memory_compete_primary={int(bool(getattr(args, 'reentry_memory_compete_primary', False)))}",
        f"reentry_engine_enable={int(bool(getattr(args, 'reentry_engine_enable', False)))}",
        f"reentry_engine_hilbert_order={int(getattr(args, 'reentry_engine_hilbert_order', 8))}",
        f"reentry_engine_bf_threshold={int(getattr(args, 'reentry_engine_bf_threshold', 50))}",
        f"reentry_engine_spatial_radius={int(getattr(args, 'reentry_engine_spatial_radius', 2))}",
        f"reentry_engine_max_spatial_radius={int(getattr(args, 'reentry_engine_max_spatial_radius', 4))}",
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
        "scripts/run_botsort_current_base_mot20_eval.py",
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
        f"reference_results_dir={args.reference_results_dir}",
        f"track_high_thresh={args.track_high_thresh}",
        f"track_low_thresh={args.track_low_thresh}",
        f"new_track_thresh={args.new_track_thresh}",
        f"track_buffer={args.track_buffer}",
        f"match_thresh={args.match_thresh}",
        f"proximity_thresh={args.proximity_thresh}",
        f"appearance_thresh={args.appearance_thresh}",
        f"reentry_memory_enable={int(bool(args.reentry_memory_enable))}",
        f"reentry_memory_use_low_score={int(bool(args.reentry_memory_use_low_score))}",
        f"reentry_memory_max_gap={args.reentry_memory_max_gap}",
        f"reentry_memory_max_size={args.reentry_memory_max_size}",
        f"reentry_memory_min_similarity={args.reentry_memory_min_similarity}",
        f"reentry_memory_confirm_streak={args.reentry_memory_confirm_streak}",
        f"reentry_memory_confirm_gap={args.reentry_memory_confirm_gap}",
        f"reentry_memory_confirm_min_similarity={args.reentry_memory_confirm_min_similarity}",
        f"reentry_memory_min_det_score={args.reentry_memory_min_det_score}",
        f"reentry_memory_app_weight={args.reentry_memory_appearance_weight}",
        f"reentry_memory_iou_weight={args.reentry_memory_iou_weight}",
        f"reentry_memory_score_weight={args.reentry_memory_score_weight}",
        f"reentry_memory_gap_weight={args.reentry_memory_gap_weight}",
        f"reentry_engine_enable={int(bool(getattr(args, 'reentry_engine_enable', False)))}",
        f"reentry_engine_hilbert_order={int(getattr(args, 'reentry_engine_hilbert_order', 8))}",
        f"reentry_engine_bf_threshold={int(getattr(args, 'reentry_engine_bf_threshold', 50))}",
        f"reentry_engine_spatial_radius={int(getattr(args, 'reentry_engine_spatial_radius', 2))}",
        f"reentry_engine_max_spatial_radius={int(getattr(args, 'reentry_engine_max_spatial_radius', 4))}",
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


def prepare_subset_tracker_dir(src_dir: Path, seq_names: List[str], dst_dir: Path) -> Path:
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    for seq_name in seq_names:
        src_path = src_dir / f"{seq_name}.txt"
        if not src_path.is_file():
            raise FileNotFoundError(f"Missing tracker result for {seq_name}: {src_path}")
        shutil.copy2(src_path, dst_dir / src_path.name)
    return dst_dir


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root).expanduser().resolve()
    logs_dir = run_root / "logs"
    summary_csv = run_root / "summary.csv"
    metrics_compare_csv = run_root / "metrics_compare.csv"
    metrics_delta_csv = run_root / "metrics_delta.csv"
    per_sequence_csv = run_root / "per_sequence_metrics.csv"
    compare_log = logs_dir / "compare.log"
    seq_names = [f"MOT20-{seq_id:02d}" for seq_id in args.seq_ids]
    seq_label = "|".join(seq_names)

    current_results_dir = BOT_ROOT / "YOLOX_outputs" / args.experiment_name / "track_results"
    current_eval_dir = run_root / "eval" / "current_base"
    current_tracker_name = args.experiment_name
    current_summary_txt = current_eval_dir / "eval" / current_tracker_name / "pedestrian_summary.txt"
    current_detailed_csv = current_eval_dir / "eval" / current_tracker_name / "pedestrian_detailed.csv"

    reference_dir = Path(args.reference_results_dir).expanduser().resolve() if args.reference_results_dir else None
    reference_eval_dir = run_root / "eval" / "reference"
    reference_tracker_name = f"{args.experiment_name}_{args.reference_label}"
    reference_summary_txt = reference_eval_dir / "eval" / reference_tracker_name / "pedestrian_summary.txt"
    reference_detailed_csv = reference_eval_dir / "eval" / reference_tracker_name / "pedestrian_detailed.csv"
    reference_subset_dir = run_root / "reference_subset"

    rows: List[Dict[str, object]] = []
    if reference_dir is not None:
        rows.append(
            {
                "step": "reference_eval_reuse",
                "name": reference_tracker_name,
                "status": "running",
                "out_dir": str(reference_eval_dir),
                "summary_csv": str(summary_csv),
                "log_path": str(logs_dir / "reference_eval.log"),
                "started_at": now_iso(),
                "finished_at": "",
                "notes": f"reuse reference tracking from {reference_dir}",
            }
        )
    rows.extend(
        [
            {
                "step": "current_base_track",
                "name": current_tracker_name,
                "status": "pending" if reference_dir is not None else "running",
                "out_dir": str(current_results_dir),
                "summary_csv": str(summary_csv),
                "log_path": str(logs_dir / "current_base_track.log"),
                "started_at": "" if reference_dir is not None else now_iso(),
                "finished_at": "",
                "notes": "" if reference_dir is not None else "current-code base tracking running",
            },
            {
                "step": "current_base_eval",
                "name": current_tracker_name,
                "status": "pending",
                "out_dir": str(current_eval_dir),
                "summary_csv": str(summary_csv),
                "log_path": str(logs_dir / "current_base_eval.log"),
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
    )

    write_rows(summary_csv, QUEUE_FIELDS, rows)
    update_plan_status(args, "running", run_root, summary_csv, logs_dir / "current_base_track.log", notes="current-base MOT20 eval started")
    append_registry(args, summary_csv, run_root, "running", "started current-code BoT-SORT base MOT20 val_half eval", logs_dir / "current_base_track.log")

    try:
        if reference_dir is not None:
            prepare_subset_tracker_dir(reference_dir, seq_names, reference_subset_dir)
            ref_eval_cmd = [
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
            rc = run_step(ref_eval_cmd, logs_dir / "reference_eval.log", cwd=REPO_ROOT)
            if rc != 0:
                raise RuntimeError(f"reference evaluation failed with exit code {rc}")
            update_row(rows, "reference_eval_reuse", status="success", finished_at=now_iso(), notes=f"reference subset eval complete for {seq_label}")
            update_row(rows, "current_base_track", status="running", started_at=now_iso(), notes="current-code base tracking running")
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
            *(["--reentry-memory-enable"] if args.reentry_memory_enable else []),
            *(["--reentry-memory-use-low-score"] if args.reentry_memory_use_low_score else []),
            "--reentry-memory-max-gap",
            str(args.reentry_memory_max_gap),
            "--reentry-memory-max-size",
            str(args.reentry_memory_max_size),
            "--reentry-memory-min-similarity",
            str(args.reentry_memory_min_similarity),
            "--reentry-memory-confirm-streak",
            str(args.reentry_memory_confirm_streak),
            "--reentry-memory-confirm-gap",
            str(args.reentry_memory_confirm_gap),
            "--reentry-memory-confirm-min-similarity",
            str(args.reentry_memory_confirm_min_similarity),
            "--reentry-memory-min-det-score",
            str(args.reentry_memory_min_det_score),
            "--reentry-memory-appearance-weight",
            str(args.reentry_memory_appearance_weight),
            "--reentry-memory-iou-weight",
            str(args.reentry_memory_iou_weight),
            "--reentry-memory-score-weight",
            str(args.reentry_memory_score_weight),
            "--reentry-memory-gap-weight",
            str(args.reentry_memory_gap_weight),
            *(["--reentry-memory-compete-primary"] if getattr(args, "reentry_memory_compete_primary", False) else []),
            *(["--reentry-engine-enable"] if getattr(args, "reentry_engine_enable", False) else []),
            "--reentry-engine-hilbert-order",
            str(getattr(args, "reentry_engine_hilbert_order", 8)),
            "--reentry-engine-bf-threshold",
            str(getattr(args, "reentry_engine_bf_threshold", 50)),
            "--reentry-engine-spatial-radius",
            str(getattr(args, "reentry_engine_spatial_radius", 2)),
            "--reentry-engine-max-spatial-radius",
            str(getattr(args, "reentry_engine_max_spatial_radius", 4)),
        ]
        rc = run_step(track_cmd, logs_dir / "current_base_track.log", cwd=BOT_ROOT)
        if rc != 0:
            raise RuntimeError(f"current base tracking failed with exit code {rc}")

        update_row(rows, "current_base_track", status="success", finished_at=now_iso(), notes="current-code base tracking complete")
        update_row(rows, "current_base_eval", status="running", started_at=now_iso(), notes="TrackEval running on current base results")
        write_rows(summary_csv, QUEUE_FIELDS, rows)

        eval_cmd = [
            args.python_bin,
            str(REPO_ROOT / "scripts" / "eval_botsort_halfval_trackeval.py"),
            "--dataset",
            "MOT20",
            "--data-root",
            args.data_root,
            "--results-dir",
            str(current_results_dir),
            "--tracker-name",
            current_tracker_name,
            "--work-dir",
            str(current_eval_dir),
            "--remap-results-from-fullval",
        ]
        rc = run_step(eval_cmd, logs_dir / "current_base_eval.log", cwd=REPO_ROOT)
        if rc != 0:
            raise RuntimeError(f"current base evaluation failed with exit code {rc}")

        update_row(rows, "current_base_eval", status="success", finished_at=now_iso(), notes="current-code base TrackEval complete")
        update_row(rows, "compare", status="running", started_at=now_iso(), notes="building compare tables")
        write_rows(summary_csv, QUEUE_FIELDS, rows)

        compare_rows: List[Dict[str, object]] = []
        per_sequence_rows: List[Dict[str, object]] = []
        delta_rows: List[Dict[str, object]] = []

        current_metrics = parse_summary_txt(current_summary_txt)
        compare_rows.append(
            {
                "name": "current_base",
                "seq": seq_label,
                "HOTA": current_metrics.get("HOTA", ""),
                "AssA": current_metrics.get("AssA", ""),
                "IDF1": current_metrics.get("IDF1", ""),
                "MOTA": current_metrics.get("MOTA", ""),
                "IDs": current_metrics.get("IDs", ""),
                "Frag": current_metrics.get("Frag", ""),
                "summary_txt": str(current_summary_txt),
                "detailed_csv": str(current_detailed_csv),
                "tracker_dir": str(current_results_dir),
            }
        )
        per_sequence_rows.extend(load_per_sequence_metrics(current_detailed_csv, "current_base"))

        if reference_dir is not None:
            reference_metrics = parse_summary_txt(reference_summary_txt)
            compare_rows.insert(
                0,
                {
                    "name": args.reference_label,
                    "seq": seq_label,
                    "HOTA": reference_metrics.get("HOTA", ""),
                    "AssA": reference_metrics.get("AssA", ""),
                    "IDF1": reference_metrics.get("IDF1", ""),
                    "MOTA": reference_metrics.get("MOTA", ""),
                    "IDs": reference_metrics.get("IDs", ""),
                    "Frag": reference_metrics.get("Frag", ""),
                    "summary_txt": str(reference_summary_txt),
                    "detailed_csv": str(reference_detailed_csv),
                    "tracker_dir": str(reference_dir),
                },
            )
            per_sequence_rows = load_per_sequence_metrics(reference_detailed_csv, args.reference_label) + per_sequence_rows
            delta_rows.append(
                {
                    "name": f"current_base_minus_{args.reference_label}",
                    "seq": seq_label,
                    "delta_HOTA": float(current_metrics.get("HOTA", 0.0)) - float(reference_metrics.get("HOTA", 0.0)),
                    "delta_AssA": float(current_metrics.get("AssA", 0.0)) - float(reference_metrics.get("AssA", 0.0)),
                    "delta_IDF1": float(current_metrics.get("IDF1", 0.0)) - float(reference_metrics.get("IDF1", 0.0)),
                    "delta_MOTA": float(current_metrics.get("MOTA", 0.0)) - float(reference_metrics.get("MOTA", 0.0)),
                    "delta_IDs": float(current_metrics.get("IDs", 0.0)) - float(reference_metrics.get("IDs", 0.0)),
                    "delta_Frag": float(current_metrics.get("Frag", 0.0)) - float(reference_metrics.get("Frag", 0.0)),
                }
            )

        write_rows(metrics_compare_csv, METRIC_FIELDS, compare_rows)
        write_rows(metrics_delta_csv, DELTA_FIELDS, delta_rows)
        write_rows(per_sequence_csv, PER_SEQUENCE_FIELDS, per_sequence_rows)

        compare_log.write_text(
            "\n".join(
                [
                    f"current_summary={current_summary_txt}",
                    f"current_results_dir={current_results_dir}",
                    f"metrics_compare={metrics_compare_csv}",
                    f"metrics_delta={metrics_delta_csv}",
                    f"per_sequence_metrics={per_sequence_csv}",
                    f"reference_results_dir={reference_dir or ''}",
                    f"reference_summary={reference_summary_txt if reference_dir is not None else ''}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        update_row(rows, "compare", status="success", finished_at=now_iso(), notes="compare tables complete")
        write_rows(summary_csv, QUEUE_FIELDS, rows)
        update_plan_status(args, "completed", run_root, summary_csv, compare_log, notes="current-base MOT20 eval complete")
        append_registry(args, summary_csv, run_root, "success", "completed current-code BoT-SORT base MOT20 val_half eval", compare_log)
    except Exception as exc:
        mark_running_rows_failed(rows, summary_csv, str(exc))
        update_plan_status(args, "failed", run_root, summary_csv, logs_dir / "current_base_track.log", notes=str(exc))
        append_registry(args, summary_csv, run_root, "failed", f"current-code BoT-SORT base MOT20 eval failed: {exc}", logs_dir / "current_base_track.log")
        raise


if __name__ == "__main__":
    main()
