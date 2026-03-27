#!/usr/bin/env python3
from __future__ import annotations

import argparse
import configparser
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path("/gemini/code/FMtrack-main/FM-Track")
BYTE_ROOT = REPO_ROOT / "third_party" / "ByteTrack"
TRACK_SCRIPT = BYTE_ROOT / "tools" / "track.py"
TRACK_EVAL_SCRIPT = REPO_ROOT / "TrackEval" / "scripts" / "run_mot_challenge.py"
SHARED_PAIR_CORE_SCRIPT = REPO_ROOT / "scripts" / "run_official_bytetrack_shared_detection_pair_core.py"
REGISTRY_CSV = REPO_ROOT / "outputs" / "experiment_registry.csv"

DEFAULT_EXP_FILE = BYTE_ROOT / "exps" / "example" / "mot" / "yolox_x_mix_det_valhalf.py"
DEFAULT_CKPT = REPO_ROOT / "weight" / "bytetrack_x_mot17.pth.tar"
DEFAULT_GRAPH_CKPT = (
    REPO_ROOT
    / "outputs"
    / "local_conflict_set_predictor_large_base_stable_20260325_023500"
    / "01_stage1"
    / "best.pt"
)
DEFAULT_DATA_ROOT = Path("/gemini/code/datasets")

SUMMARY_FIELDS = [
    "arm",
    "run_dir",
    "exp_name",
    "protocol_tag",
    "official_exp_file",
    "official_checkpoint",
    "official_experiment_name",
    "host_variant",
    "plugin_mode",
    "tracker_mode",
    "graph_checkpoint",
    "graph_topk",
    "graph_min_detections",
    "graph_min_committed_matches",
    "graph_max_detections",
    "graph_max_tracks",
    "graph_cluster_gate_thresh",
    "graph_cluster_gate_temp",
    "graph_cluster_gate_bias",
    "graph_max_commits_per_cluster",
    "graph_replacement_budget_ratio",
    "graph_max_replaced_clusters",
    "graph_min_commit_margin",
    "posthost_oracle_min_iou",
    "seed",
    "track_thresh",
    "track_buffer",
    "match_thresh",
    "min_box_area",
    "results_dir",
    "eval_dir",
    "log_path",
    "eligible_clusters",
    "replaced_clusters",
    "host_same_commit_clusters",
    "delta_replaced_clusters",
    "matched_dets",
    "delta_commit_pairs",
    "delta_drop_pairs",
    "deferred_dets",
    "blocked_tracks",
    "gate_pass_clusters",
    "gate_filtered_clusters",
    "trigger_filtered_clusters",
    "skipped_large_clusters",
    "budget_filtered_clusters",
    "margin_filtered_pairs",
    "capped_commit_pairs",
    "all_defer_clusters",
    "empty_pair_candidate_clusters",
    "post_filter_empty_clusters",
    "posthost_selected_clusters",
    "posthost_swap_clusters",
    "posthost_add_clusters",
    "posthost_defer_clusters",
    "HOTA",
    "AssA",
    "IDF1",
    "MOTA",
    "IDSW",
    "status",
    "error",
]

DELTA_FIELDS = [
    "exp_name",
    "protocol_tag",
    "official_exp_file",
    "official_checkpoint",
    "host_variant",
    "plugin_mode",
    "graph_checkpoint",
    "host_only_dir",
    "host_plus_plugin_dir",
    "delta_HOTA",
    "delta_AssA",
    "delta_IDF1",
    "delta_MOTA",
    "delta_IDSW",
    "status",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run official ByteTrack half-val paired evaluation with and without local-conflict plugin."
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--exp-file", default=str(DEFAULT_EXP_FILE))
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    parser.add_argument("--graph-ckpt", default=str(DEFAULT_GRAPH_CKPT))
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--experiment-name", default="official_bytetrack_local_conflict_halfval_pair")
    parser.add_argument("--protocol-tag", default="official_bytetrack_x_mot17_valhalf_pair")
    parser.add_argument("--host-variant", default="official_bytetrack_x_mot17_valhalf")
    parser.add_argument("--plugin-mode", choices=["learned_commit", "posthost_one_edit_oracle"], default="learned_commit")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--conf", type=float, default=0.01)
    parser.add_argument("--nms", type=float, default=0.7)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")
    parser.add_argument("--fuse", action="store_true", default=True)
    parser.add_argument("--no-fuse", dest="fuse", action="store_false")
    parser.add_argument("--track-thresh", type=float, default=0.6)
    parser.add_argument("--track-buffer", type=int, default=30)
    parser.add_argument("--match-thresh", type=float, default=0.9)
    parser.add_argument("--min-box-area", type=float, default=100.0)
    parser.add_argument("--graph-topk", type=int, default=8)
    parser.add_argument("--graph-min-detections", type=int, default=2)
    parser.add_argument("--graph-min-committed-matches", type=int, default=1)
    parser.add_argument("--graph-max-detections", type=int, default=8)
    parser.add_argument("--graph-max-tracks", type=int, default=32)
    parser.add_argument("--graph-cluster-gate-thresh", type=float, default=0.5)
    parser.add_argument("--graph-cluster-gate-temp", type=float, default=1.0)
    parser.add_argument("--graph-cluster-gate-bias", type=float, default=0.0)
    parser.add_argument("--graph-max-commits-per-cluster", type=int, default=1)
    parser.add_argument("--graph-replacement-budget-ratio", type=float, default=0.05)
    parser.add_argument("--graph-max-replaced-clusters", type=int, default=0)
    parser.add_argument("--graph-min-commit-margin", type=float, default=0.05)
    parser.add_argument("--posthost-oracle-min-iou", type=float, default=0.5)
    parser.add_argument("--registry-csv", default=str(REGISTRY_CSV))
    return parser.parse_args()


def write_rows(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_single_row(path: Path, fieldnames: List[str], row: Dict[str, Any]) -> None:
    write_rows(path, fieldnames, [row])


def initial_summary_row(args: argparse.Namespace, *, arm: str, tracker_mode: str, arm_dir: Path) -> Dict[str, Any]:
    plugin_enabled = str(tracker_mode) != "host_only"
    graph_ckpt = (
        str(Path(args.graph_ckpt).resolve())
        if plugin_enabled and str(args.plugin_mode) == "learned_commit" and str(args.graph_ckpt).strip()
        else ""
    )
    return {
        "arm": arm,
        "run_dir": str(arm_dir.resolve()),
        "exp_name": args.experiment_name,
        "protocol_tag": args.protocol_tag,
        "official_exp_file": str(Path(args.exp_file).resolve()),
        "official_checkpoint": str(Path(args.ckpt).resolve()),
        "official_experiment_name": "",
        "host_variant": args.host_variant,
        "plugin_mode": args.plugin_mode if plugin_enabled else "",
        "tracker_mode": tracker_mode,
        "graph_checkpoint": graph_ckpt,
        "graph_topk": args.graph_topk if plugin_enabled else "",
        "graph_min_detections": args.graph_min_detections if plugin_enabled else "",
        "graph_min_committed_matches": (
            args.graph_min_committed_matches if plugin_enabled else ""
        ),
        "graph_max_detections": args.graph_max_detections if plugin_enabled else "",
        "graph_max_tracks": args.graph_max_tracks if plugin_enabled else "",
        "graph_cluster_gate_thresh": args.graph_cluster_gate_thresh if plugin_enabled else "",
        "graph_cluster_gate_temp": args.graph_cluster_gate_temp if plugin_enabled else "",
        "graph_cluster_gate_bias": args.graph_cluster_gate_bias if plugin_enabled else "",
        "graph_max_commits_per_cluster": (
            args.graph_max_commits_per_cluster if plugin_enabled else ""
        ),
        "graph_replacement_budget_ratio": (
            args.graph_replacement_budget_ratio if plugin_enabled else ""
        ),
        "graph_max_replaced_clusters": (
            args.graph_max_replaced_clusters if plugin_enabled else ""
        ),
        "graph_min_commit_margin": args.graph_min_commit_margin if plugin_enabled else "",
        "posthost_oracle_min_iou": args.posthost_oracle_min_iou if plugin_enabled else "",
        "seed": args.seed,
        "track_thresh": args.track_thresh,
        "track_buffer": args.track_buffer,
        "match_thresh": args.match_thresh,
        "min_box_area": args.min_box_area,
        "results_dir": str((arm_dir / "track_results").resolve()),
        "eval_dir": str((arm_dir / "trackeval").resolve()),
        "log_path": str((arm_dir / "run.log").resolve()),
        "eligible_clusters": "",
        "replaced_clusters": "",
        "host_same_commit_clusters": "",
        "delta_replaced_clusters": "",
        "matched_dets": "",
        "delta_commit_pairs": "",
        "delta_drop_pairs": "",
        "deferred_dets": "",
        "blocked_tracks": "",
        "gate_pass_clusters": "",
        "gate_filtered_clusters": "",
        "trigger_filtered_clusters": "",
        "skipped_large_clusters": "",
        "budget_filtered_clusters": "",
        "margin_filtered_pairs": "",
        "capped_commit_pairs": "",
        "all_defer_clusters": "",
        "empty_pair_candidate_clusters": "",
        "post_filter_empty_clusters": "",
        "posthost_selected_clusters": "",
        "posthost_swap_clusters": "",
        "posthost_add_clusters": "",
        "posthost_defer_clusters": "",
        "HOTA": "",
        "AssA": "",
        "IDF1": "",
        "MOTA": "",
        "IDSW": "",
        "status": "running",
        "error": "",
    }


def initial_delta_row(args: argparse.Namespace, out_dir: Path) -> Dict[str, Any]:
    return {
        "exp_name": args.experiment_name,
        "protocol_tag": args.protocol_tag,
        "official_exp_file": str(Path(args.exp_file).resolve()),
        "official_checkpoint": str(Path(args.ckpt).resolve()),
        "host_variant": args.host_variant,
        "plugin_mode": args.plugin_mode,
        "graph_checkpoint": (
            str(Path(args.graph_ckpt).resolve())
            if str(args.plugin_mode) == "learned_commit" and str(args.graph_ckpt).strip()
            else ""
        ),
        "host_only_dir": str((out_dir / "00_host_only").resolve()),
        "host_plus_plugin_dir": str((out_dir / "01_host_plus_plugin").resolve()),
        "delta_HOTA": "",
        "delta_AssA": "",
        "delta_IDF1": "",
        "delta_MOTA": "",
        "delta_IDSW": "",
        "status": "running",
        "error": "",
    }


def prepare_halfval_gt_root(*, data_root: Path, gt_root: Path, sequences: List[str]) -> None:
    if gt_root.exists():
        shutil.rmtree(gt_root)
    for seq in sequences:
        src_seq_dir = data_root / "MOT17" / "train" / seq
        src_ini = src_seq_dir / "seqinfo.ini"
        src_gt = src_seq_dir / "gt" / "gt_val_half.txt"
        if not src_ini.is_file():
            raise FileNotFoundError(f"Missing seqinfo.ini for {seq}: {src_ini}")
        if not src_gt.is_file():
            raise FileNotFoundError(f"Missing gt_val_half for {seq}: {src_gt}")
        dst_seq_dir = gt_root / seq
        (dst_seq_dir / "gt").mkdir(parents=True, exist_ok=True)
        rows = [line.strip().split(",") for line in src_gt.read_text(encoding="utf-8").splitlines() if line.strip()]
        with (dst_seq_dir / "gt" / "gt.txt").open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

        parser = configparser.ConfigParser()
        parser.read(src_ini)
        if "Sequence" not in parser:
            raise ValueError(f"Invalid seqinfo.ini: {src_ini}")
        max_frame = max(int(float(row[0])) for row in rows) if rows else 0
        parser["Sequence"]["seqLength"] = str(max_frame)
        with (dst_seq_dir / "seqinfo.ini").open("w", encoding="utf-8") as f:
            parser.write(f)


def build_seqmap(sequences: List[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name"])
        for seq in sequences:
            writer.writerow([seq])


def run_trackeval(
    *,
    python_bin: str,
    gt_root: Path,
    results_dir: Path,
    tracker_name: str,
    work_dir: Path,
    sequences: List[str],
) -> Path:
    tracker_root = work_dir / "trackers"
    tracker_data = tracker_root / tracker_name / "data"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    tracker_data.mkdir(parents=True, exist_ok=True)
    for result_file in sorted(results_dir.glob("*.txt")):
        shutil.copy2(result_file, tracker_data / result_file.name)
    seqmap_path = work_dir / "seqmaps" / "MOT17_val_half.txt"
    build_seqmap(sequences, seqmap_path)
    output_root = work_dir / "eval"
    cmd = [
        python_bin,
        str(TRACK_EVAL_SCRIPT),
        "--GT_FOLDER",
        str(gt_root),
        "--TRACKERS_FOLDER",
        str(tracker_root),
        "--OUTPUT_FOLDER",
        str(output_root),
        "--TRACKERS_TO_EVAL",
        tracker_name,
        "--BENCHMARK",
        "MOT17",
        "--SPLIT_TO_EVAL",
        "train",
        "--SEQMAP_FILE",
        str(seqmap_path),
        "--SKIP_SPLIT_FOL",
        "True",
        "--DO_PREPROC",
        "True",
        "--TRACKER_SUB_FOLDER",
        "data",
        "--OUTPUT_SUB_FOLDER",
        "",
        "--PRINT_ONLY_COMBINED",
        "True",
        "--METRICS",
        "HOTA",
        "CLEAR",
        "Identity",
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    return output_root / tracker_name


def parse_trackeval_summary(eval_dir: Path) -> Dict[str, float]:
    summary_path = eval_dir / "pedestrian_summary.txt"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing TrackEval summary: {summary_path}")
    lines = [line.strip() for line in summary_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError(f"Malformed TrackEval summary: {summary_path}")
    keys = lines[0].split()
    values = lines[1].split()
    if len(keys) != len(values):
        raise ValueError(f"Header/value mismatch in {summary_path}")
    row = {key: float(value) for key, value in zip(keys, values)}
    return {
        "HOTA": float(row["HOTA"]),
        "AssA": float(row["AssA"]),
        "IDF1": float(row["IDF1"]),
        "MOTA": float(row["MOTA"]),
        "IDSW": float(row["IDSW"]),
    }


def aggregate_diagnostics(diag_dir: Path) -> Dict[str, int]:
    totals = {
        "eligible_clusters": 0,
        "replaced_clusters": 0,
        "host_same_commit_clusters": 0,
        "delta_replaced_clusters": 0,
        "matched_dets": 0,
        "delta_commit_pairs": 0,
        "delta_drop_pairs": 0,
        "deferred_dets": 0,
        "blocked_tracks": 0,
        "gate_pass_clusters": 0,
        "gate_filtered_clusters": 0,
        "trigger_filtered_clusters": 0,
        "skipped_large_clusters": 0,
        "budget_filtered_clusters": 0,
        "margin_filtered_pairs": 0,
        "capped_commit_pairs": 0,
        "all_defer_clusters": 0,
        "empty_pair_candidate_clusters": 0,
        "post_filter_empty_clusters": 0,
        "posthost_selected_clusters": 0,
        "posthost_swap_clusters": 0,
        "posthost_add_clusters": 0,
        "posthost_defer_clusters": 0,
    }
    if not diag_dir.is_dir():
        return totals
    for path in sorted(diag_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for key in totals:
            totals[key] += int(payload.get(key, 0) or 0)
    return totals


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def run_shared_pair_core(args: argparse.Namespace, *, out_dir: Path) -> Path:
    log_path = out_dir / "shared_pair_core.log"
    cmd = [
        args.python_bin,
        str(SHARED_PAIR_CORE_SCRIPT),
        "--out-dir",
        str(out_dir.resolve()),
        "--exp-file",
        str(Path(args.exp_file).resolve()),
        "--ckpt",
        str(Path(args.ckpt).resolve()),
        "--data-root",
        str(Path(args.data_root).resolve()),
        "--batch-size",
        str(args.batch_size),
        "--devices",
        str(args.devices),
        "--seed",
        str(args.seed),
        "--conf",
        str(args.conf),
        "--nms",
        str(args.nms),
        "--track-thresh",
        str(args.track_thresh),
        "--track-buffer",
        str(args.track_buffer),
        "--match-thresh",
        str(args.match_thresh),
        "--min-box-area",
        str(args.min_box_area),
        "--host-variant",
        str(args.host_variant),
        "--plugin-mode",
        str(args.plugin_mode),
        "--graph-topk",
        str(args.graph_topk),
        "--graph-min-detections",
        str(args.graph_min_detections),
        "--graph-min-committed-matches",
        str(args.graph_min_committed_matches),
        "--graph-max-detections",
        str(args.graph_max_detections),
        "--graph-max-tracks",
        str(args.graph_max_tracks),
        "--graph-cluster-gate-thresh",
        str(args.graph_cluster_gate_thresh),
        "--graph-cluster-gate-temp",
        str(args.graph_cluster_gate_temp),
        "--graph-cluster-gate-bias",
        str(args.graph_cluster_gate_bias),
        "--graph-max-commits-per-cluster",
        str(args.graph_max_commits_per_cluster),
        "--graph-replacement-budget-ratio",
        str(args.graph_replacement_budget_ratio),
        "--graph-max-replaced-clusters",
        str(args.graph_max_replaced_clusters),
        "--graph-min-commit-margin",
        str(args.graph_min_commit_margin),
        "--posthost-oracle-data-root",
        str(Path(args.data_root).resolve()),
        "--posthost-oracle-min-iou",
        str(args.posthost_oracle_min_iou),
    ]
    if str(args.plugin_mode) == "learned_commit":
        cmd.extend(
            [
                "--graph-ckpt",
                str(Path(args.graph_ckpt).resolve()),
            ]
        )
    if args.fp16:
        cmd.append("--fp16")
    if args.fuse:
        cmd.append("--fuse")

    env = os.environ.copy()
    pythonpath_parts = [str(BYTE_ROOT.resolve()), str(REPO_ROOT.resolve())]
    existing_pythonpath = env.get("PYTHONPATH", "").strip()
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    with log_path.open("w", encoding="utf-8") as log_fp:
        subprocess.run(
            cmd,
            check=True,
            cwd=REPO_ROOT,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            env=env,
        )
    summary_json = out_dir / "shared_pair_core_summary.json"
    if not summary_json.is_file():
        raise FileNotFoundError(f"Missing shared pair core summary: {summary_json}")
    return log_path


def evaluate_arm_outputs(
    args: argparse.Namespace,
    *,
    arm_summary: Dict[str, Any],
    tracker_mode: str,
    out_dir: Path,
    shared_log_path: Path,
) -> Dict[str, Any]:
    arm = str(arm_summary["arm"])
    arm_dir = out_dir / arm
    diagnostics_dir = arm_dir / "diagnostics"
    local_results_dir = arm_dir / "track_results"
    official_experiment_name = f"{args.experiment_name}_shared_detection_pair"
    try:
        if not local_results_dir.is_dir():
            raise FileNotFoundError(f"Missing tracking outputs dir: {local_results_dir}")
        result_files = sorted(local_results_dir.glob("*.txt"))
        if not result_files:
            raise FileNotFoundError(f"No tracking outputs produced in {local_results_dir}")
        sequences = [path.stem for path in result_files]
        gt_root = arm_dir / "trackeval_gt"
        prepare_halfval_gt_root(
            data_root=Path(args.data_root),
            gt_root=gt_root,
            sequences=sequences,
        )
        eval_dir = run_trackeval(
            python_bin=args.python_bin,
            gt_root=gt_root,
            results_dir=local_results_dir,
            tracker_name=arm,
            work_dir=arm_dir / "trackeval",
            sequences=sequences,
        )
        metrics = parse_trackeval_summary(eval_dir)
        diagnostics = aggregate_diagnostics(diagnostics_dir) if tracker_mode != "host_only" else {}
        arm_summary.update(
            {
                "official_experiment_name": official_experiment_name,
                "results_dir": str(local_results_dir.resolve()),
                "eval_dir": str(eval_dir.resolve()),
                "log_path": str(shared_log_path.resolve()),
                "HOTA": f"{metrics['HOTA']:.3f}",
                "AssA": f"{metrics['AssA']:.3f}",
                "IDF1": f"{metrics['IDF1']:.3f}",
                "MOTA": f"{metrics['MOTA']:.3f}",
                "IDSW": str(int(round(metrics["IDSW"]))),
                "status": "success",
                "error": "",
            }
        )
        for key in [
            "eligible_clusters",
            "replaced_clusters",
            "host_same_commit_clusters",
            "delta_replaced_clusters",
            "matched_dets",
            "delta_commit_pairs",
            "delta_drop_pairs",
            "deferred_dets",
            "blocked_tracks",
            "gate_pass_clusters",
            "gate_filtered_clusters",
            "trigger_filtered_clusters",
            "skipped_large_clusters",
            "budget_filtered_clusters",
            "margin_filtered_pairs",
            "capped_commit_pairs",
            "all_defer_clusters",
            "empty_pair_candidate_clusters",
            "post_filter_empty_clusters",
            "posthost_selected_clusters",
            "posthost_swap_clusters",
            "posthost_add_clusters",
            "posthost_defer_clusters",
        ]:
            arm_summary[key] = diagnostics.get(key, 0) if tracker_mode != "host_only" else 0
    except Exception as exc:
        arm_summary.update(
            {
                "official_experiment_name": official_experiment_name,
                "log_path": str(shared_log_path.resolve()),
                "status": "failed",
                "error": str(exc),
            }
        )

    write_single_row(arm_dir / "summary.csv", SUMMARY_FIELDS, arm_summary)
    write_single_row(arm_dir / "result.csv", SUMMARY_FIELDS, arm_summary)
    return arm_summary


def append_registry(args: argparse.Namespace, *, out_dir: Path, status: str) -> None:
    cmd = [
        args.python_bin,
        str(REPO_ROOT / "scripts" / "append_experiment_record.py"),
        "--csv",
        str(Path(args.registry_csv).resolve()),
        "--kind",
        "eval",
        "--status",
        status,
        "--script",
        "scripts/run_official_bytetrack_local_conflict_halfval_pair.py",
        "--dataset",
        "MOT17",
        "--split",
        "val_half",
        "--tracker-family",
        "official_bytetrack",
        "--variant",
        args.protocol_tag,
        "--tag",
        args.experiment_name,
        "--run-root",
        str(out_dir.resolve()),
        "--summary-csv",
        str((out_dir / "summary.csv").resolve()),
        "--checkpoint",
        str(Path(args.ckpt).resolve()),
        "--notes",
        f"official ByteTrack half-val paired host-only vs {args.plugin_mode}",
        "--extra",
        f"protocol_tag={args.protocol_tag}",
        f"host_variant={args.host_variant}",
        f"plugin_mode={args.plugin_mode}",
        f"graph_checkpoint={(str(Path(args.graph_ckpt).resolve()) if str(args.plugin_mode) == 'learned_commit' and str(args.graph_ckpt).strip() else '')}",
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not Path(args.exp_file).is_file():
        raise FileNotFoundError(f"Missing exp file: {args.exp_file}")
    if not Path(args.ckpt).is_file():
        raise FileNotFoundError(f"Missing checkpoint: {args.ckpt}")
    if str(args.plugin_mode) == "learned_commit" and not Path(args.graph_ckpt).is_file():
        raise FileNotFoundError(f"Missing graph checkpoint: {args.graph_ckpt}")

    host_only_row = initial_summary_row(
        args,
        arm="00_host_only",
        tracker_mode="host_only",
        arm_dir=out_dir / "00_host_only",
    )
    plugin_row = initial_summary_row(
        args,
        arm="01_host_plus_plugin",
        tracker_mode="host_plus_plugin",
        arm_dir=out_dir / "01_host_plus_plugin",
    )
    delta_row = initial_delta_row(args, out_dir)
    write_single_row(out_dir / "00_host_only" / "summary.csv", SUMMARY_FIELDS, host_only_row)
    write_single_row(out_dir / "00_host_only" / "result.csv", SUMMARY_FIELDS, host_only_row)
    write_single_row(out_dir / "01_host_plus_plugin" / "summary.csv", SUMMARY_FIELDS, plugin_row)
    write_single_row(out_dir / "01_host_plus_plugin" / "result.csv", SUMMARY_FIELDS, plugin_row)
    write_rows(out_dir / "summary.csv", SUMMARY_FIELDS, [host_only_row, plugin_row])
    write_single_row(out_dir / "result.csv", DELTA_FIELDS, delta_row)
    manifest = {
        "exp_name": args.experiment_name,
        "protocol_tag": args.protocol_tag,
        "official_exp_file": str(Path(args.exp_file).resolve()),
        "official_checkpoint": str(Path(args.ckpt).resolve()),
        "graph_checkpoint": (
            str(Path(args.graph_ckpt).resolve())
            if str(args.plugin_mode) == "learned_commit" and str(args.graph_ckpt).strip()
            else ""
        ),
        "host_variant": args.host_variant,
        "plugin_mode": args.plugin_mode,
        "posthost_oracle_min_iou": float(args.posthost_oracle_min_iou),
        "tracker_modes": ["host_only", "host_plus_plugin"],
        "eval_mode": "shared_detection_pair",
        "status": "running",
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    shared_log_path = out_dir / "shared_pair_core.log"
    core_error = ""
    try:
        shared_log_path = run_shared_pair_core(args, out_dir=out_dir)
    except Exception as exc:
        core_error = str(exc)
        host_only_row.update({"log_path": str(shared_log_path.resolve()), "status": "failed", "error": core_error})
        plugin_row.update({"log_path": str(shared_log_path.resolve()), "status": "failed", "error": core_error})
        write_single_row(out_dir / "00_host_only" / "summary.csv", SUMMARY_FIELDS, host_only_row)
        write_single_row(out_dir / "00_host_only" / "result.csv", SUMMARY_FIELDS, host_only_row)
        write_single_row(out_dir / "01_host_plus_plugin" / "summary.csv", SUMMARY_FIELDS, plugin_row)
        write_single_row(out_dir / "01_host_plus_plugin" / "result.csv", SUMMARY_FIELDS, plugin_row)
        write_rows(out_dir / "summary.csv", SUMMARY_FIELDS, [host_only_row, plugin_row])

    if not core_error:
        host_only_row = evaluate_arm_outputs(
            args,
            arm_summary=host_only_row,
            tracker_mode="host_only",
            out_dir=out_dir,
            shared_log_path=shared_log_path,
        )
        write_rows(out_dir / "summary.csv", SUMMARY_FIELDS, [host_only_row, plugin_row])

        plugin_row = evaluate_arm_outputs(
            args,
            arm_summary=plugin_row,
            tracker_mode="host_plus_plugin",
            out_dir=out_dir,
            shared_log_path=shared_log_path,
        )
        write_rows(out_dir / "summary.csv", SUMMARY_FIELDS, [host_only_row, plugin_row])

    if not core_error and host_only_row["status"] == "success" and plugin_row["status"] == "success":
        delta_row.update(
            {
                "delta_HOTA": f"{float(plugin_row['HOTA']) - float(host_only_row['HOTA']):.3f}",
                "delta_AssA": f"{float(plugin_row['AssA']) - float(host_only_row['AssA']):.3f}",
                "delta_IDF1": f"{float(plugin_row['IDF1']) - float(host_only_row['IDF1']):.3f}",
                "delta_MOTA": f"{float(plugin_row['MOTA']) - float(host_only_row['MOTA']):.3f}",
                "delta_IDSW": str(int(round(float(plugin_row["IDSW"]) - float(host_only_row["IDSW"])))),
                "status": "success",
                "error": "",
            }
        )
        manifest["status"] = "success"
    else:
        errors = [row.get("error", "") for row in (host_only_row, plugin_row) if row.get("error")]
        delta_row.update(
            {
                "status": "failed",
                "error": " | ".join(errors),
            }
        )
        manifest["status"] = "failed"

    append_registry(args, out_dir=out_dir, status=manifest["status"])
    write_single_row(out_dir / "result.csv", DELTA_FIELDS, delta_row)
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
