#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict


REPO_ROOT = Path("/gemini/code/FMtrack-main/FM-Track")
BYTE_ROOT = REPO_ROOT / "third_party" / "ByteTrack"
TRACK_SCRIPT = BYTE_ROOT / "tools" / "track.py"
REGISTRY_CSV = REPO_ROOT / "outputs" / "experiment_registry.csv"

DEFAULT_DUMP_EXP_FILE = BYTE_ROOT / "exps" / "example" / "mot" / "yolox_x_mix_det_trainhalf_dump.py"
DEFAULT_EVAL_EXP_FILE = BYTE_ROOT / "exps" / "example" / "mot" / "yolox_x_mix_det_valhalf.py"
DEFAULT_CKPT = REPO_ROOT / "weight" / "bytetrack_x_mot17.pth.tar"
DEFAULT_DATA_ROOT = Path("/gemini/code/datasets")

SUMMARY_FIELDS = [
    "exp_name",
    "protocol_tag",
    "host_variant",
    "official_dump_exp_file",
    "official_eval_exp_file",
    "official_checkpoint",
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
    "seed",
    "teacher_mode",
    "bridge_crowded_row_degree_thresh",
    "bridge_crowded_bonus",
    "bridge_large_component_bonus",
    "bridge_commit_cost",
    "bridge_min_gain",
    "large_component_max_subclusters",
    "split_strategy",
    "split_target_key",
    "val_fraction",
    "assignment_target_key",
    "assignment_row_mask_key",
    "edge_loss_mask_key",
    "edge_target_key",
    "cluster_target_key",
    "target_gate_coverage_min",
    "target_gate_coverage_max",
    "coverage_penalty_weight",
    "train_positive_cluster_oversample",
    "keep_row_loss_weight",
    "edit_row_loss_weight",
    "edit_edge_positive_weight",
    "margin_host_edit",
    "train_sequences",
    "val_sequences",
    "dump_dir",
    "labeled_rows_csv",
    "group_jsonl",
    "dataset_dir",
    "stage1_dir",
    "pair_eval_dir",
    "pipeline_log",
    "checkpoint",
    "cluster_examples",
    "cluster_trigger_pass_clusters",
    "cluster_skipped_large_clusters",
    "train_examples",
    "val_examples",
    "train_host_variants",
    "val_host_variants",
    "split_mode",
    "host_only_HOTA",
    "host_only_AssA",
    "host_only_IDF1",
    "host_only_MOTA",
    "host_only_IDSW",
    "plugin_HOTA",
    "plugin_AssA",
    "plugin_IDF1",
    "plugin_MOTA",
    "plugin_IDSW",
    "plugin_eligible_clusters",
    "plugin_replaced_clusters",
    "plugin_matched_dets",
    "plugin_deferred_dets",
    "plugin_gate_pass_clusters",
    "plugin_gate_filtered_clusters",
    "plugin_trigger_filtered_clusters",
    "plugin_budget_filtered_clusters",
    "plugin_margin_filtered_pairs",
    "plugin_capped_commit_pairs",
    "plugin_all_defer_clusters",
    "plugin_empty_pair_candidate_clusters",
    "plugin_post_filter_empty_clusters",
    "delta_HOTA",
    "delta_AssA",
    "delta_IDF1",
    "delta_MOTA",
    "delta_IDSW",
    "current_stage",
    "status",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build official ByteTrack host runtime data on train_half, retrain the local-conflict module, then run strict paired half-val evaluation."
    )
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--dump-exp-file", default=str(DEFAULT_DUMP_EXP_FILE))
    parser.add_argument("--eval-exp-file", default=str(DEFAULT_EVAL_EXP_FILE))
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    parser.add_argument("--experiment-name", default="official_bytetrack_bridgecommit_v1")
    parser.add_argument("--protocol-tag", default="official_bytetrack_mot17_trainhalf_stage1_then_valhalf_pair")
    parser.add_argument("--host-variant", default="official_bytetrack")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--conf", type=float, default=0.01)
    parser.add_argument("--nms", type=float, default=0.7)
    parser.add_argument("--track-thresh", type=float, default=0.6)
    parser.add_argument("--track-buffer", type=int, default=30)
    parser.add_argument("--match-thresh", type=float, default=0.9)
    parser.add_argument("--min-box-area", type=float, default=100.0)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")
    parser.add_argument("--fuse", action="store_true", default=True)
    parser.add_argument("--no-fuse", dest="fuse", action="store_false")
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--min-detections", type=int, default=2)
    parser.add_argument("--min-committed-matches", type=int, default=1)
    parser.add_argument("--max-detections", type=int, default=8)
    parser.add_argument("--max-tracks", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size-stage1", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-conflict-blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--min-val-examples", type=int, default=64)
    parser.add_argument("--dataset-tag", default="official_bytetrack_bridgecommit_v1_trainhalf")
    parser.add_argument("--feature-version", default="v2_hostnorm_geom")
    parser.add_argument(
        "--teacher-mode",
        choices=["oracle_commit", "delta_utility", "edit_utility", "rescue_utility", "sparse_edit", "bridge_commit_contract_utility"],
        default="bridge_commit_contract_utility",
    )
    parser.add_argument("--edit-utility-commit-cost", type=float, default=0.20)
    parser.add_argument("--edit-utility-min-gain", type=float, default=0.75)
    parser.add_argument("--edit-utility-force-defer-gain", type=float, default=0.50)
    parser.add_argument("--runtime-host-match-thresh", type=float, default=0.90)
    parser.add_argument("--soft-rescue-weight", type=float, default=0.75)
    parser.add_argument("--rescue-force-defer-gain", type=float, default=0.50)
    parser.add_argument("--rescue-min-gain", type=float, default=0.50)
    parser.add_argument("--bridge-crowded-row-degree-thresh", type=int, default=4)
    parser.add_argument("--bridge-crowded-bonus", type=float, default=0.25)
    parser.add_argument("--bridge-large-component-bonus", type=float, default=0.50)
    parser.add_argument("--bridge-commit-cost", type=float, default=0.20)
    parser.add_argument("--bridge-min-gain", type=float, default=0.75)
    parser.add_argument("--large-component-max-subclusters", type=int, default=0)
    parser.add_argument("--train-sequences", default="MOT17-04-FRCNN,MOT17-05-FRCNN,MOT17-09-FRCNN,MOT17-10-FRCNN,MOT17-11-FRCNN")
    parser.add_argument("--val-sequences", default="MOT17-02-FRCNN,MOT17-13-FRCNN")
    parser.add_argument("--split-strategy", choices=["auto", "sequence", "random", "stratified_random"], default="auto")
    parser.add_argument("--split-target-key", default="cluster_should_intervene_bridge")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--cluster-gate-thresh", type=float, default=0.5)
    parser.add_argument("--cluster-gate-calibration", choices=["none", "temp_bias"], default="temp_bias")
    parser.add_argument("--cluster-gate-select-metric", choices=["f0.5", "utility", "bounded_utility"], default="f0.5")
    parser.add_argument("--cluster-gate-beta", type=float, default=0.5)
    parser.add_argument("--cluster-gate-fp-weight", type=float, default=2.0)
    parser.add_argument("--cluster-gate-search-min", type=float, default=0.05)
    parser.add_argument("--cluster-gate-search-max", type=float, default=0.80)
    parser.add_argument("--cluster-gate-search-steps", type=int, default=19)
    parser.add_argument("--cluster-gate-loss-mode", choices=["bce", "weighted_bce"], default="weighted_bce")
    parser.add_argument("--cluster-gate-positive-weight", type=float, default=16.0)
    parser.add_argument("--cluster-gate-negative-weight", type=float, default=1.0)
    parser.add_argument("--model-selection-metric", default="commit_viability_utility_cov")
    parser.add_argument("--assignment-target-key", default="target_by_det_bridge")
    parser.add_argument("--assignment-row-mask-key", default="row_bridge_mask")
    parser.add_argument("--edge-loss-mask-key", default="assignment_row_mask")
    parser.add_argument("--edge-target-key", default="edge_is_bridge_commit")
    parser.add_argument("--cluster-target-key", default="cluster_should_intervene_bridge")
    parser.add_argument("--target-gate-coverage-min", type=float, default=0.01)
    parser.add_argument("--target-gate-coverage-max", type=float, default=0.05)
    parser.add_argument("--coverage-penalty-weight", type=float, default=1.0)
    parser.add_argument("--train-positive-cluster-oversample", type=float, default=4.0)
    parser.add_argument("--keep-row-loss-weight", type=float, default=0.0)
    parser.add_argument("--edit-row-loss-weight", type=float, default=2.0)
    parser.add_argument("--edit-edge-positive-weight", type=float, default=32.0)
    parser.add_argument("--margin-host-edit", type=float, default=0.1)
    parser.add_argument("--loss-assign-weight", type=float, default=1.0)
    parser.add_argument("--loss-edge-weight", type=float, default=0.5)
    parser.add_argument("--loss-cluster-weight", type=float, default=0.25)
    parser.add_argument("--loss-margin-weight", type=float, default=0.25)
    parser.add_argument("--margin-commit", type=float, default=0.2)
    parser.add_argument("--margin-row", type=float, default=0.2)
    parser.add_argument("--margin-defer", type=float, default=0.2)
    parser.add_argument("--graph-max-commits-per-cluster", type=int, default=1)
    parser.add_argument("--graph-replacement-budget-ratio", type=float, default=0.05)
    parser.add_argument("--graph-max-replaced-clusters", type=int, default=0)
    parser.add_argument("--graph-min-commit-margin", type=float, default=0.05)
    parser.add_argument("--edge-focal-alpha", type=float, default=0.25)
    parser.add_argument("--edge-focal-gamma", type=float, default=2.0)
    parser.add_argument("--score-jitter-std", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--skip-step-grad-norm", type=float, default=1000000.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--registry-csv", default=str(REGISTRY_CSV))
    return parser.parse_args()


def write_single_row(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in SUMMARY_FIELDS})


def read_single_row(path: Path) -> Dict[str, str]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        return next(csv.DictReader(f), {})


def update_root_row(path_summary: Path, path_result: Path, row: Dict[str, Any], **updates: Any) -> Dict[str, Any]:
    row.update(updates)
    write_single_row(path_summary, row)
    write_single_row(path_result, row)
    return row


def run_logged(cmd: list[str], *, cwd: Path, env: dict[str, str], log_fp) -> None:
    log_fp.write("$ " + " ".join(cmd) + "\n")
    log_fp.flush()
    subprocess.run(cmd, check=True, cwd=cwd, env=env, stdout=log_fp, stderr=subprocess.STDOUT)
    log_fp.write("\n")
    log_fp.flush()


def load_dataset_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def count_cluster_examples_by_sequence(path: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if not path.is_file():
        return counts
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            seq = str(row.get("seq", "")).strip()
            if not seq:
                continue
            counts[seq] = int(counts.get(seq, 0)) + 1
    return counts


def choose_fallback_sequence_split(
    *,
    cluster_examples_jsonl: Path,
    min_val_examples: int,
) -> tuple[str, str]:
    counts = count_cluster_examples_by_sequence(cluster_examples_jsonl)
    if len(counts) < 2:
        raise RuntimeError(
            f"Cannot build a strict train/val split from {cluster_examples_jsonl}: sequence_counts={counts}"
        )
    ordered = sorted(counts.items(), key=lambda item: (int(item[1]), str(item[0])))
    candidate_val = next((seq for seq, count in ordered if int(count) >= int(min_val_examples)), "")
    if not candidate_val:
        candidate_val = ordered[-1][0]
    train_sequences = [seq for seq, _ in sorted(counts.items()) if seq != candidate_val]
    if not train_sequences:
        raise RuntimeError(f"Fallback split left no train sequences: sequence_counts={counts}")
    return ",".join(train_sequences), str(candidate_val)


def append_registry(args: argparse.Namespace, *, out_dir: Path, summary_csv: Path, checkpoint: Path, log_path: Path, status: str) -> None:
    cmd = [
        args.python_bin,
        str(REPO_ROOT / "scripts" / "append_experiment_record.py"),
        "--csv",
        str(Path(args.registry_csv).resolve()),
        "--kind",
        "train",
        "--status",
        status,
        "--script",
        "scripts/run_official_bytetrack_local_conflict_stage1_trainhalf.py",
        "--dataset",
        "MOT17",
        "--split",
        "official_trainhalf_to_valhalf",
        "--tracker-family",
        "ByteTrack",
        "--variant",
        "official_bytetrack_local_conflict_stage1_trainhalf",
        "--tag",
        "official_bytetrack_local_conflict_mainline",
        "--run-root",
        str(out_dir.resolve()),
        "--summary-csv",
        str(summary_csv.resolve()),
        "--checkpoint",
        str(checkpoint.resolve()) if checkpoint.is_file() else "",
        "--log-path",
        str(log_path.resolve()),
        "--notes",
        "official ByteTrack frozen-host module retrain on train_half and strict paired half-val evaluation",
        "--extra",
        f"host_variant={args.host_variant}",
        f"graph_topk={args.topk}",
        f"graph_min_detections={args.min_detections}",
        f"graph_min_committed_matches={args.min_committed_matches}",
        f"graph_max_detections={args.max_detections}",
        f"graph_max_tracks={args.max_tracks}",
        f"seed={args.seed}",
        f"train_sequences={args.train_sequences}",
        f"val_sequences={args.val_sequences}",
        f"split_strategy={args.split_strategy}",
        f"split_target_key={args.split_target_key}",
        f"val_fraction={args.val_fraction}",
        f"assignment_target_key={args.assignment_target_key}",
        f"train_positive_cluster_oversample={args.train_positive_cluster_oversample}",
        f"margin_host_edit={args.margin_host_edit}",
        f"dataset_tag={args.dataset_tag}",
        f"feature_version={args.feature_version}",
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def main() -> int:
    args = parse_args()
    if str(args.out_dir).strip():
        out_dir = Path(args.out_dir).resolve()
    else:
        from datetime import datetime
        out_dir = REPO_ROOT / "outputs" / f"official_bytetrack_local_conflict_stage1_trainhalf_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = out_dir / "summary.csv"
    result_csv = out_dir / "result.csv"
    pipeline_log = out_dir / "pipeline.log"
    dump_dir = out_dir / "00_official_dump_trainhalf"
    runtime_dump_dir = dump_dir / "runtime_dump"
    labeled_rows_csv = out_dir / f"labeled_replay_top{int(args.topk)}.csv"
    group_jsonl = out_dir / f"labeled_replay_top{int(args.topk)}.groups.jsonl"
    recoverability_json = out_dir / f"labeled_replay_top{int(args.topk)}.recoverability.json"
    source_manifest = out_dir / "source_manifest.csv"
    dataset_dir = out_dir / "cluster_set_predictor_data"
    stage1_dir = out_dir / "01_stage1"
    pair_eval_dir = out_dir / "02_official_halfval_pair"
    current_train_sequences = str(args.train_sequences)
    current_val_sequences = str(args.val_sequences)

    row: Dict[str, Any] = {
        "exp_name": args.experiment_name,
        "protocol_tag": args.protocol_tag,
        "host_variant": args.host_variant,
        "official_dump_exp_file": str(Path(args.dump_exp_file).resolve()),
        "official_eval_exp_file": str(Path(args.eval_exp_file).resolve()),
        "official_checkpoint": str(Path(args.ckpt).resolve()),
        "graph_topk": int(args.topk),
        "graph_min_detections": int(args.min_detections),
        "graph_min_committed_matches": int(args.min_committed_matches),
        "graph_max_detections": int(args.max_detections),
        "graph_max_tracks": int(args.max_tracks),
        "graph_cluster_gate_thresh": float(args.cluster_gate_thresh),
        "graph_cluster_gate_temp": 1.0,
        "graph_cluster_gate_bias": 0.0,
        "graph_max_commits_per_cluster": int(args.graph_max_commits_per_cluster),
        "graph_replacement_budget_ratio": float(args.graph_replacement_budget_ratio),
        "graph_max_replaced_clusters": int(args.graph_max_replaced_clusters),
        "graph_min_commit_margin": float(args.graph_min_commit_margin),
        "seed": int(args.seed),
        "teacher_mode": str(args.teacher_mode),
        "bridge_crowded_row_degree_thresh": int(args.bridge_crowded_row_degree_thresh),
        "bridge_crowded_bonus": float(args.bridge_crowded_bonus),
        "bridge_large_component_bonus": float(args.bridge_large_component_bonus),
        "bridge_commit_cost": float(args.bridge_commit_cost),
        "bridge_min_gain": float(args.bridge_min_gain),
        "large_component_max_subclusters": int(args.large_component_max_subclusters),
        "split_strategy": str(args.split_strategy),
        "split_target_key": str(args.split_target_key),
        "val_fraction": float(args.val_fraction),
        "assignment_target_key": str(args.assignment_target_key),
        "assignment_row_mask_key": str(args.assignment_row_mask_key),
        "edge_loss_mask_key": str(args.edge_loss_mask_key),
        "edge_target_key": str(args.edge_target_key),
        "cluster_target_key": str(args.cluster_target_key),
        "target_gate_coverage_min": float(args.target_gate_coverage_min),
        "target_gate_coverage_max": float(args.target_gate_coverage_max),
        "coverage_penalty_weight": float(args.coverage_penalty_weight),
        "train_positive_cluster_oversample": float(args.train_positive_cluster_oversample),
        "keep_row_loss_weight": float(args.keep_row_loss_weight),
        "edit_row_loss_weight": float(args.edit_row_loss_weight),
        "edit_edge_positive_weight": float(args.edit_edge_positive_weight),
        "margin_host_edit": float(args.margin_host_edit),
        "train_sequences": current_train_sequences,
        "val_sequences": current_val_sequences,
        "dump_dir": str(runtime_dump_dir.resolve()),
        "labeled_rows_csv": str(labeled_rows_csv.resolve()),
        "group_jsonl": str(group_jsonl.resolve()),
        "dataset_dir": str(dataset_dir.resolve()),
        "stage1_dir": str(stage1_dir.resolve()),
        "pair_eval_dir": str(pair_eval_dir.resolve()),
        "pipeline_log": str(pipeline_log.resolve()),
        "checkpoint": "",
        "cluster_examples": "",
        "cluster_trigger_pass_clusters": "",
        "cluster_skipped_large_clusters": "",
        "train_examples": "",
        "val_examples": "",
        "train_host_variants": "",
        "val_host_variants": "",
        "split_mode": "",
        "host_only_HOTA": "",
        "host_only_AssA": "",
        "host_only_IDF1": "",
        "host_only_MOTA": "",
        "host_only_IDSW": "",
        "plugin_HOTA": "",
        "plugin_AssA": "",
        "plugin_IDF1": "",
        "plugin_MOTA": "",
        "plugin_IDSW": "",
        "plugin_eligible_clusters": "",
        "plugin_replaced_clusters": "",
        "plugin_matched_dets": "",
        "plugin_deferred_dets": "",
        "plugin_gate_pass_clusters": "",
        "plugin_gate_filtered_clusters": "",
        "plugin_trigger_filtered_clusters": "",
        "plugin_budget_filtered_clusters": "",
        "plugin_margin_filtered_pairs": "",
        "plugin_capped_commit_pairs": "",
        "plugin_all_defer_clusters": "",
        "plugin_empty_pair_candidate_clusters": "",
        "plugin_post_filter_empty_clusters": "",
        "delta_HOTA": "",
        "delta_AssA": "",
        "delta_IDF1": "",
        "delta_MOTA": "",
        "delta_IDSW": "",
        "current_stage": "init",
        "status": "running",
        "error": "",
    }
    write_single_row(summary_csv, row)
    write_single_row(result_csv, row)

    env = os.environ.copy()
    pythonpath_parts = [str(BYTE_ROOT.resolve()), str(REPO_ROOT.resolve())]
    existing_pythonpath = env.get("PYTHONPATH", "").strip()
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    stage1_checkpoint = stage1_dir / "best.pt"
    status = "failed"
    try:
        with pipeline_log.open("w", encoding="utf-8") as log_fp:
            dump_experiment_name = f"{args.experiment_name}_dump_trainhalf"
            official_dump_output_dir = BYTE_ROOT / "YOLOX_outputs" / dump_experiment_name
            if official_dump_output_dir.exists():
                shutil.rmtree(official_dump_output_dir)

            update_root_row(summary_csv, result_csv, row, current_stage="00_official_dump_trainhalf", status="running", error="")
            dump_cmd = [
                args.python_bin,
                str(TRACK_SCRIPT),
                "-f",
                str(Path(args.dump_exp_file).resolve()),
                "-c",
                str(Path(args.ckpt).resolve()),
                "-b",
                str(args.batch_size),
                "-d",
                str(args.devices),
                "--experiment-name",
                dump_experiment_name,
                "--conf",
                str(args.conf),
                "--nms",
                str(args.nms),
                "--track_thresh",
                str(args.track_thresh),
                "--track_buffer",
                str(args.track_buffer),
                "--match_thresh",
                str(args.match_thresh),
                "--min-box-area",
                str(args.min_box_area),
                "--local-conflict-dump-dir",
                str(runtime_dump_dir.resolve()),
                "--local-conflict-dump-topk",
                str(args.topk),
                "--local-conflict-dump-min-score",
                "0.0",
            ]
            if args.fp16:
                dump_cmd.append("--fp16")
            if args.fuse:
                dump_cmd.append("--fuse")
            run_logged(dump_cmd, cwd=BYTE_ROOT, env=env, log_fp=log_fp)
            dump_csvs = sorted(runtime_dump_dir.glob("*.csv"))
            if not dump_csvs:
                raise FileNotFoundError(f"No runtime dump CSV files found under {runtime_dump_dir}")

            update_root_row(summary_csv, result_csv, row, current_stage="01_build_replay_labels", status="running", error="")
            label_cmd = [
                args.python_bin,
                str(REPO_ROOT / "scripts" / "build_runtime_assoc_replay_labels.py"),
                "--dump-root",
                str(runtime_dump_dir.resolve()),
                "--dataset",
                "MOT17",
                "--data-root",
                str(Path(args.data_root).resolve()),
                "--split",
                "train",
                "--split-part",
                "train_half",
                "--out-csv",
                str(labeled_rows_csv.resolve()),
                "--summary-json",
                str((out_dir / "labeled_replay.summary.json").resolve()),
                "--out-group-jsonl",
                str(group_jsonl.resolve()),
                "--out-recoverability-json",
                str(recoverability_json.resolve()),
                "--topk",
                str(args.topk),
                "--rank-score-col",
                "refined_score",
                "--ambiguity-margin",
                "0.10",
            ]
            run_logged(label_cmd, cwd=REPO_ROOT, env=env, log_fp=log_fp)

            update_root_row(summary_csv, result_csv, row, current_stage="02_build_dataset", status="running", error="")
            manifest_cmd = [
                args.python_bin,
                str(REPO_ROOT / "scripts" / "build_local_conflict_set_predictor_dataset_manifest.py"),
                "--out-csv",
                str(source_manifest.resolve()),
                "--rows-csv",
                str(labeled_rows_csv.resolve()),
                "--group-jsonl",
                str(group_jsonl.resolve()),
                "--host-variant",
                str(args.host_variant),
                "--source-tag",
                "official_bytetrack_trainhalf",
                "--split-tag",
                "auto",
                "--dataset-tag",
                str(args.dataset_tag),
                "--feature-version",
                str(args.feature_version),
            ]
            run_logged(manifest_cmd, cwd=REPO_ROOT, env=env, log_fp=log_fp)

            dataset_cmd = [
                args.python_bin,
                str(REPO_ROOT / "scripts" / "build_local_conflict_set_predictor_dataset.py"),
                "--source-manifest",
                str(source_manifest.resolve()),
                "--out-dir",
                str(dataset_dir.resolve()),
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
                str(current_train_sequences),
                "--val-sequences",
                str(current_val_sequences),
                "--strict-sequence-split",
                "--feature-version",
                str(args.feature_version),
                "--dataset-tag",
                str(args.dataset_tag),
                "--teacher-mode",
                str(args.teacher_mode),
                "--edit-utility-commit-cost",
                str(args.edit_utility_commit_cost),
                "--edit-utility-min-gain",
                str(args.edit_utility_min_gain),
                "--edit-utility-force-defer-gain",
                str(args.edit_utility_force_defer_gain),
                "--runtime-host-match-thresh",
                str(args.runtime_host_match_thresh),
                "--soft-rescue-weight",
                str(args.soft_rescue_weight),
                "--rescue-force-defer-gain",
                str(args.rescue_force_defer_gain),
                "--rescue-min-gain",
                str(args.rescue_min_gain),
                "--bridge-crowded-row-degree-thresh",
                str(args.bridge_crowded_row_degree_thresh),
                "--bridge-crowded-bonus",
                str(args.bridge_crowded_bonus),
                "--bridge-large-component-bonus",
                str(args.bridge_large_component_bonus),
                "--bridge-commit-cost",
                str(args.bridge_commit_cost),
                "--bridge-min-gain",
                str(args.bridge_min_gain),
                "--large-component-max-subclusters",
                str(args.large_component_max_subclusters),
            ]
            run_logged(dataset_cmd, cwd=REPO_ROOT, env=env, log_fp=log_fp)

            dataset_summary = load_dataset_summary(dataset_dir / "summary.json")
            val_eligible = int(
                dataset_summary.get("split_breakdown", {}).get("val", {}).get("eligible_clusters", 0) or 0
            )
            if val_eligible < int(args.min_val_examples):
                fallback_train, fallback_val = choose_fallback_sequence_split(
                    cluster_examples_jsonl=dataset_dir / "cluster_examples.jsonl",
                    min_val_examples=int(args.min_val_examples),
                )
                current_train_sequences = fallback_train
                current_val_sequences = fallback_val
                log_fp.write(
                    "[fallback_split] requested val sequences produced too few eligible clusters; "
                    f"retrying with train={current_train_sequences} val={current_val_sequences}\n"
                )
                log_fp.flush()
                update_root_row(
                    summary_csv,
                    result_csv,
                    row,
                    train_sequences=current_train_sequences,
                    val_sequences=current_val_sequences,
                )
                dataset_cmd = [
                    args.python_bin,
                    str(REPO_ROOT / "scripts" / "build_local_conflict_set_predictor_dataset.py"),
                    "--source-manifest",
                    str(source_manifest.resolve()),
                    "--out-dir",
                    str(dataset_dir.resolve()),
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
                    str(current_train_sequences),
                    "--val-sequences",
                    str(current_val_sequences),
                    "--strict-sequence-split",
                    "--feature-version",
                    str(args.feature_version),
                    "--dataset-tag",
                    str(args.dataset_tag),
                    "--teacher-mode",
                    str(args.teacher_mode),
                    "--edit-utility-commit-cost",
                    str(args.edit_utility_commit_cost),
                    "--edit-utility-min-gain",
                    str(args.edit_utility_min_gain),
                    "--edit-utility-force-defer-gain",
                    str(args.edit_utility_force_defer_gain),
                    "--runtime-host-match-thresh",
                    str(args.runtime_host_match_thresh),
                    "--soft-rescue-weight",
                    str(args.soft_rescue_weight),
                    "--rescue-force-defer-gain",
                    str(args.rescue_force_defer_gain),
                    "--rescue-min-gain",
                    str(args.rescue_min_gain),
                    "--bridge-crowded-row-degree-thresh",
                    str(args.bridge_crowded_row_degree_thresh),
                    "--bridge-crowded-bonus",
                    str(args.bridge_crowded_bonus),
                    "--bridge-large-component-bonus",
                    str(args.bridge_large_component_bonus),
                    "--bridge-commit-cost",
                    str(args.bridge_commit_cost),
                    "--bridge-min-gain",
                    str(args.bridge_min_gain),
                    "--large-component-max-subclusters",
                    str(args.large_component_max_subclusters),
                ]
                run_logged(dataset_cmd, cwd=REPO_ROOT, env=env, log_fp=log_fp)

            update_root_row(summary_csv, result_csv, row, current_stage="03_train_stage1", status="running", error="")
            train_cmd = [
                args.python_bin,
                str(REPO_ROOT / "scripts" / "train_local_conflict_set_predictor.py"),
                "--data-jsonl",
                str((dataset_dir / "cluster_examples.jsonl").resolve()),
                "--out-dir",
                str(stage1_dir.resolve()),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size_stage1),
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
                str(current_train_sequences),
                "--val-sequences",
                str(current_val_sequences),
                "--strict-sequence-split",
                "--split-strategy",
                str(args.split_strategy),
                "--split-target-key",
                str(args.split_target_key),
                "--val-fraction",
                str(args.val_fraction),
                "--min-val-examples",
                str(args.min_val_examples),
                "--dataset-tag",
                str(args.dataset_tag),
                "--source-manifest",
                str(source_manifest.resolve()),
                "--feature-version",
                str(args.feature_version),
                "--cluster-gate-thresh",
                str(args.cluster_gate_thresh),
                "--cluster-gate-calibration",
                str(args.cluster_gate_calibration),
                "--cluster-gate-select-metric",
                str(args.cluster_gate_select_metric),
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
                str(args.cluster_gate_loss_mode),
                "--cluster-gate-positive-weight",
                str(args.cluster_gate_positive_weight),
                "--cluster-gate-negative-weight",
                str(args.cluster_gate_negative_weight),
                "--model-selection-metric",
                str(args.model_selection_metric),
                "--assignment-target-key",
                str(args.assignment_target_key),
                "--assignment-row-mask-key",
                str(args.assignment_row_mask_key),
                "--edge-loss-mask-key",
                str(args.edge_loss_mask_key),
                "--edge-target-key",
                str(args.edge_target_key),
                "--cluster-target-key",
                str(args.cluster_target_key),
                "--target-gate-coverage-min",
                str(args.target_gate_coverage_min),
                "--target-gate-coverage-max",
                str(args.target_gate_coverage_max),
                "--coverage-penalty-weight",
                str(args.coverage_penalty_weight),
                "--train-positive-cluster-oversample",
                str(args.train_positive_cluster_oversample),
                "--keep-row-loss-weight",
                str(args.keep_row_loss_weight),
                "--edit-row-loss-weight",
                str(args.edit_row_loss_weight),
                "--edit-edge-positive-weight",
                str(args.edit_edge_positive_weight),
                "--margin-host-edit",
                str(args.margin_host_edit),
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
                "--edge-focal-alpha",
                str(args.edge_focal_alpha),
                "--edge-focal-gamma",
                str(args.edge_focal_gamma),
                "--score-jitter-std",
                str(args.score_jitter_std),
                "--grad-clip-norm",
                str(args.grad_clip_norm),
                "--skip-step-grad-norm",
                str(args.skip_step_grad_norm),
                "--seed",
                str(args.seed),
            ]
            run_logged(train_cmd, cwd=REPO_ROOT, env=env, log_fp=log_fp)
            if not stage1_checkpoint.is_file():
                raise FileNotFoundError(f"Missing stage1 checkpoint: {stage1_checkpoint}")

            stage1_summary = read_single_row(stage1_dir / "summary.csv")
            dataset_summary = load_dataset_summary(dataset_dir / "summary.json")
            gate_thresh = stage1_summary.get("val_cluster_gate_thresh_calibrated", "") or str(args.cluster_gate_thresh)
            gate_temp = stage1_summary.get("val_cluster_gate_temp", "") or "1.0"
            gate_bias = stage1_summary.get("val_cluster_gate_bias", "") or "0.0"
            update_root_row(
                summary_csv,
                result_csv,
                row,
                checkpoint=str(stage1_checkpoint.resolve()),
                graph_cluster_gate_thresh=gate_thresh,
                graph_cluster_gate_temp=gate_temp,
                graph_cluster_gate_bias=gate_bias,
                graph_max_commits_per_cluster=int(args.graph_max_commits_per_cluster),
                graph_replacement_budget_ratio=float(args.graph_replacement_budget_ratio),
                graph_max_replaced_clusters=int(args.graph_max_replaced_clusters),
                graph_min_commit_margin=float(args.graph_min_commit_margin),
                teacher_mode=str(args.teacher_mode),
                cluster_examples=dataset_summary.get("eligible_clusters", ""),
                cluster_trigger_pass_clusters=dataset_summary.get("trigger_pass_clusters", ""),
                cluster_skipped_large_clusters=dataset_summary.get("skipped_large_clusters", ""),
                train_examples=stage1_summary.get("train_examples", ""),
                val_examples=stage1_summary.get("val_examples", ""),
                train_host_variants=stage1_summary.get("train_host_variants", ""),
                val_host_variants=stage1_summary.get("val_host_variants", ""),
                split_mode=stage1_summary.get("split_mode", ""),
            )

            update_root_row(summary_csv, result_csv, row, current_stage="04_official_halfval_pair_eval", status="running", error="")
            pair_cmd = [
                args.python_bin,
                str(REPO_ROOT / "scripts" / "run_official_bytetrack_local_conflict_halfval_pair.py"),
                "--out-dir",
                str(pair_eval_dir.resolve()),
                "--exp-file",
                str(Path(args.eval_exp_file).resolve()),
                "--ckpt",
                str(Path(args.ckpt).resolve()),
                "--graph-ckpt",
                str(stage1_checkpoint.resolve()),
                "--python-bin",
                str(args.python_bin),
                "--data-root",
                str(Path(args.data_root).resolve()),
                "--experiment-name",
                f"{args.experiment_name}_paired_halfval",
                "--protocol-tag",
                f"{args.protocol_tag}_paired_halfval",
                "--host-variant",
                str(args.host_variant),
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
                "--graph-topk",
                str(args.topk),
                "--graph-min-detections",
                str(args.min_detections),
                "--graph-min-committed-matches",
                str(args.min_committed_matches),
                "--graph-max-detections",
                str(args.max_detections),
                "--graph-max-tracks",
                str(args.max_tracks),
                "--graph-cluster-gate-thresh",
                str(gate_thresh),
                "--graph-cluster-gate-temp",
                str(gate_temp),
                "--graph-cluster-gate-bias",
                str(gate_bias),
                "--graph-max-commits-per-cluster",
                str(args.graph_max_commits_per_cluster),
                "--graph-replacement-budget-ratio",
                str(args.graph_replacement_budget_ratio),
                "--graph-max-replaced-clusters",
                str(args.graph_max_replaced_clusters),
                "--graph-min-commit-margin",
                str(args.graph_min_commit_margin),
                "--registry-csv",
                str(Path(args.registry_csv).resolve()),
            ]
            if args.fp16:
                pair_cmd.append("--fp16")
            if args.fuse:
                pair_cmd.append("--fuse")
            run_logged(pair_cmd, cwd=REPO_ROOT, env=env, log_fp=log_fp)

        pair_summary_rows = []
        pair_summary_csv = pair_eval_dir / "summary.csv"
        if pair_summary_csv.is_file():
            with pair_summary_csv.open("r", encoding="utf-8", newline="") as f:
                pair_summary_rows = [dict(r) for r in csv.DictReader(f)]
        host_only = next((r for r in pair_summary_rows if str(r.get("arm", "")) == "00_host_only"), {})
        plugin = next((r for r in pair_summary_rows if str(r.get("arm", "")) == "01_host_plus_plugin"), {})
        delta = read_single_row(pair_eval_dir / "result.csv")
        if str(delta.get("status", "")) != "success":
            raise RuntimeError(f"Paired official half-val eval failed: {delta.get('error', 'unknown_error')}")

        update_root_row(
            summary_csv,
            result_csv,
            row,
            host_only_HOTA=host_only.get("HOTA", ""),
            host_only_AssA=host_only.get("AssA", ""),
            host_only_IDF1=host_only.get("IDF1", ""),
            host_only_MOTA=host_only.get("MOTA", ""),
            host_only_IDSW=host_only.get("IDSW", ""),
            plugin_HOTA=plugin.get("HOTA", ""),
            plugin_AssA=plugin.get("AssA", ""),
            plugin_IDF1=plugin.get("IDF1", ""),
            plugin_MOTA=plugin.get("MOTA", ""),
            plugin_IDSW=plugin.get("IDSW", ""),
            plugin_eligible_clusters=plugin.get("eligible_clusters", ""),
            plugin_replaced_clusters=plugin.get("replaced_clusters", ""),
            plugin_matched_dets=plugin.get("matched_dets", ""),
            plugin_deferred_dets=plugin.get("deferred_dets", ""),
            plugin_gate_pass_clusters=plugin.get("gate_pass_clusters", ""),
            plugin_gate_filtered_clusters=plugin.get("gate_filtered_clusters", ""),
            plugin_trigger_filtered_clusters=plugin.get("trigger_filtered_clusters", ""),
            plugin_budget_filtered_clusters=plugin.get("budget_filtered_clusters", ""),
            plugin_margin_filtered_pairs=plugin.get("margin_filtered_pairs", ""),
            plugin_capped_commit_pairs=plugin.get("capped_commit_pairs", ""),
            plugin_all_defer_clusters=plugin.get("all_defer_clusters", ""),
            plugin_empty_pair_candidate_clusters=plugin.get("empty_pair_candidate_clusters", ""),
            plugin_post_filter_empty_clusters=plugin.get("post_filter_empty_clusters", ""),
            delta_HOTA=delta.get("delta_HOTA", ""),
            delta_AssA=delta.get("delta_AssA", ""),
            delta_IDF1=delta.get("delta_IDF1", ""),
            delta_MOTA=delta.get("delta_MOTA", ""),
            delta_IDSW=delta.get("delta_IDSW", ""),
            current_stage="done",
            status="ok",
            error="",
        )
        status = "success"
    except Exception as exc:
        update_root_row(summary_csv, result_csv, row, current_stage=row.get("current_stage", "failed"), status="failed", error=str(exc))
        status = "failed"

    try:
        append_registry(
            args,
            out_dir=out_dir,
            summary_csv=summary_csv,
            checkpoint=stage1_checkpoint,
            log_path=pipeline_log,
            status=status,
        )
    except Exception:
        pass

    return 0 if status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
