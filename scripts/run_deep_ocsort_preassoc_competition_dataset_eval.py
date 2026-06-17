#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List


REPO_ROOT = Path("/gemini/code/FMtrack-main/FM-Track")
DEEP_ROOT = REPO_ROOT / "external" / "Deep-OC-SORT-main"
REGISTRY_CSV = REPO_ROOT / "outputs" / "experiment_registry.csv"

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
    "frames_seen",
    "preassoc_stale_competition_rows",
    "preassoc_stale_competition_candidate_rows",
    "preassoc_stale_competition_biased_edges",
    "preassoc_stale_competition_takeover_risk_rejected_rows",
    "preassoc_stale_competition_owner_alt_biased_edges",
    "preassoc_stale_competition_owner_alt_risk_rejected_rows",
    "preassoc_stale_competition_owner_alt_released_owners",
    "preassoc_stale_competition_selected_matches",
    "preassoc_stale_competition_forced_gate_rejected_rows",
    "preassoc_stale_competition_acceptance_gate_scored_rows",
    "preassoc_stale_competition_acceptance_gate_rejected_rows",
    "preassoc_stale_competition_acceptance_gate_accepted_rows",
    "preassoc_stale_competition_acceptance_gate_high_conf_reclaim_override_rows",
    "preassoc_stale_competition_recovery_anchor_gate_scored_rows",
    "preassoc_stale_competition_recovery_anchor_gate_rejected_rows",
    "preassoc_stale_competition_recovery_anchor_gate_accepted_rows",
    "preassoc_stale_competition_force_rewrite_scored_rows",
    "preassoc_stale_competition_force_rewrite_rejected_rows",
    "preassoc_stale_competition_force_rewrite_accepted_rows",
    "preassoc_stale_competition_force_rewrite_recovery_anchor_override_rows",
    "preassoc_stale_competition_force_rewrite_acceptance_nearmiss_owner_alt_override_rows",
    "preassoc_stale_competition_force_rewrite_recovery_anchor_nearmiss_owner_alt_override_rows",
    "local_contention_units",
    "local_contention_candidate_pairs",
    "local_contention_labeled_pairs",
    "local_contention_positive_pairs",
    "local_contention_negative_pairs",
    "summary_csv",
]

RUNTIME_PER_SEQUENCE_FIELDS = [
    "name",
    "seq",
    "frames_seen",
    "preassoc_stale_competition_rows",
    "preassoc_stale_competition_candidate_rows",
    "preassoc_stale_competition_biased_edges",
    "preassoc_stale_competition_takeover_risk_rejected_rows",
    "preassoc_stale_competition_owner_alt_biased_edges",
    "preassoc_stale_competition_owner_alt_risk_rejected_rows",
    "preassoc_stale_competition_owner_alt_released_owners",
    "preassoc_stale_competition_selected_matches",
    "preassoc_stale_competition_forced_gate_rejected_rows",
    "preassoc_stale_competition_acceptance_gate_scored_rows",
    "preassoc_stale_competition_acceptance_gate_rejected_rows",
    "preassoc_stale_competition_acceptance_gate_accepted_rows",
    "preassoc_stale_competition_acceptance_gate_high_conf_reclaim_override_rows",
    "preassoc_stale_competition_recovery_anchor_gate_scored_rows",
    "preassoc_stale_competition_recovery_anchor_gate_rejected_rows",
    "preassoc_stale_competition_recovery_anchor_gate_accepted_rows",
    "preassoc_stale_competition_force_rewrite_scored_rows",
    "preassoc_stale_competition_force_rewrite_rejected_rows",
    "preassoc_stale_competition_force_rewrite_accepted_rows",
    "preassoc_stale_competition_force_rewrite_recovery_anchor_override_rows",
    "preassoc_stale_competition_force_rewrite_acceptance_nearmiss_owner_alt_override_rows",
    "preassoc_stale_competition_force_rewrite_recovery_anchor_nearmiss_owner_alt_override_rows",
    "local_contention_units",
    "local_contention_candidate_pairs",
    "local_contention_labeled_pairs",
    "local_contention_positive_pairs",
    "local_contention_negative_pairs",
]

LOCAL_CONTENTION_SUMMARY_FIELDS = [
    "name",
    "rows",
    "units",
    "labeled_rows",
    "positive_rows",
    "negative_rows",
    "avg_candidates_per_unit",
    "jsonl_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a recorded Deep-OC-SORT raw vs pre-association stale-competition paired eval on MOT17, MOT20, or DanceTrack sequences."
    )
    parser.add_argument("--benchmark", choices=["MOT17", "MOT20", "DanceTrack"], required=True)
    parser.add_argument("--seq-name", default="")
    parser.add_argument("--seq-names", nargs="+", default=None)
    parser.add_argument("--out-root", default="")
    parser.add_argument("--reuse-raw-from", default="", help="optional existing run root whose raw arm should be reused")
    parser.add_argument("--disable-preassoc-stale-competition", action="store_true")
    parser.add_argument("--preassoc-stale-competition-min-time-since-update", type=int, default=2)
    parser.add_argument("--preassoc-stale-competition-max-time-since-update", type=int, default=8)
    parser.add_argument("--preassoc-stale-competition-min-hits", type=int, default=20)
    parser.add_argument("--preassoc-stale-competition-min-box-iou", type=float, default=0.75)
    parser.add_argument("--preassoc-stale-competition-min-edge-score", type=float, default=0.0)
    parser.add_argument("--preassoc-stale-competition-bias", type=float, default=0.1)
    parser.add_argument("--preassoc-stale-competition-iou-scale", type=float, default=0.0)
    parser.add_argument("--preassoc-stale-competition-require-raw-owner", action="store_true")
    parser.add_argument("--preassoc-stale-competition-min-hit-gap-vs-owner", type=int, default=0)
    parser.add_argument("--preassoc-stale-competition-min-age-gap-vs-owner", type=int, default=50)
    parser.add_argument("--preassoc-stale-competition-owner-max-hits", type=int, default=8)
    parser.add_argument("--preassoc-stale-competition-owner-max-age", type=int, default=0)
    parser.add_argument("--preassoc-stale-competition-owner-edge-penalty", type=float, default=0.05)
    parser.add_argument("--preassoc-stale-competition-owner-alt-det-bias", type=float, default=0.0)
    parser.add_argument("--preassoc-stale-competition-owner-alt-det-min-score", type=float, default=0.0)
    parser.add_argument("--preassoc-stale-competition-owner-alt-det-min-box-iou", type=float, default=0.0)
    parser.add_argument("--preassoc-stale-competition-owner-alt-det-min-ranker-score", type=float, default=-1.0)
    parser.add_argument("--preassoc-stale-competition-owner-alt-det-min-ranker-margin", type=float, default=-1.0)
    parser.add_argument("--preassoc-stale-competition-owner-alt-det-min-edge-advantage", type=float, default=-1.0)
    parser.add_argument("--preassoc-stale-competition-takeover-soft-margin-floor", type=float, default=-1.0)
    parser.add_argument("--preassoc-stale-competition-takeover-soft-edge-advantage-floor", type=float, default=-1.0)
    parser.add_argument("--preassoc-stale-competition-takeover-min-force-risk-scale", type=float, default=-1.0)
    parser.add_argument("--preassoc-stale-competition-min-ranker-margin-to-second", type=float, default=-1.0)
    parser.add_argument("--preassoc-stale-competition-min-edge-advantage-vs-owner", type=float, default=-1.0)
    parser.add_argument("--preassoc-stale-competition-max-owner-edge-deficit", type=float, default=0.20)
    parser.add_argument(
        "--preassoc-stale-competition-force-owner-edge-deficit-arg",
        action="store_true",
        help="always forward --preassoc-stale-competition-max-owner-edge-deficit to main.py, even when the value is negative to explicitly disable the gate",
    )
    parser.add_argument("--preassoc-stale-competition-block-owner-on-reclaim", action="store_true")
    parser.add_argument("--preassoc-stale-competition-require-det-top1", action="store_true")
    parser.add_argument("--preassoc-stale-competition-max-det-rank", type=int, default=0)
    parser.add_argument("--preassoc-stale-competition-force-rewrite-enable", action="store_true")
    parser.add_argument("--preassoc-stale-competition-force-rewrite-min-score", type=float, default=-1.0)
    parser.add_argument("--preassoc-stale-competition-force-rewrite-gate-weight", type=float, default=0.45)
    parser.add_argument("--preassoc-stale-competition-force-rewrite-iou-weight", type=float, default=0.30)
    parser.add_argument("--preassoc-stale-competition-force-rewrite-ranker-weight", type=float, default=0.15)
    parser.add_argument("--preassoc-stale-competition-force-rewrite-age-weight", type=float, default=0.10)
    parser.add_argument("--preassoc-stale-competition-force-rewrite-age-cap", type=int, default=30)
    parser.add_argument("--preassoc-stale-competition-force-rewrite-owner-alt-bonus", type=float, default=0.15)
    parser.add_argument("--preassoc-stale-competition-force-rewrite-neighborhood-enable", action="store_true")
    parser.add_argument("--preassoc-stale-competition-force-rewrite-min-neighborhood-gain", type=float, default=-1.0)
    parser.add_argument(
        "--preassoc-stale-competition-force-rewrite-trapped-owner-min-neighborhood-gain",
        type=float,
        default=999.0,
    )
    parser.add_argument(
        "--preassoc-stale-competition-force-rewrite-reroute-ready-min-neighborhood-gain",
        type=float,
        default=999.0,
    )
    parser.add_argument(
        "--preassoc-stale-competition-force-rewrite-trapped-owner-negative-gain-min-challenger-alt-box-iou",
        type=float,
        default=-1.0,
    )
    parser.add_argument(
        "--preassoc-stale-competition-force-rewrite-neighborhood-keep-challenger-alt-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--preassoc-stale-competition-force-rewrite-neighborhood-rewrite-owner-alt-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--preassoc-stale-competition-force-rewrite-neighborhood-shared-alt-penalty",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--preassoc-stale-competition-force-rewrite-neighborhood-trapped-owner-bonus",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--preassoc-stale-competition-force-rewrite-neighborhood-reroute-ready-penalty",
        type=float,
        default=0.0,
    )
    parser.add_argument("--preassoc-stale-competition-force-rewrite-min-box-iou", type=float, default=-1.0)
    parser.add_argument("--preassoc-stale-competition-force-rewrite-max-age-gap", type=int, default=-1)
    parser.add_argument(
        "--preassoc-stale-competition-force-rewrite-max-owner-alt-det-box-iou",
        type=float,
        default=-1.0,
    )
    parser.add_argument(
        "--preassoc-stale-competition-force-rewrite-owner-alt-soft-override-enable",
        action="store_true",
    )
    parser.add_argument(
        "--preassoc-stale-competition-force-rewrite-owner-alt-soft-override-max-owner-alt-det-box-iou",
        type=float,
        default=-1.0,
    )
    parser.add_argument(
        "--preassoc-stale-competition-force-rewrite-owner-alt-soft-override-max-owner-hits",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--preassoc-stale-competition-force-rewrite-owner-alt-soft-override-min-box-iou",
        type=float,
        default=-1.0,
    )
    parser.add_argument(
        "--preassoc-stale-competition-force-rewrite-owner-alt-soft-override-min-score",
        type=float,
        default=-1.0,
    )
    parser.add_argument(
        "--preassoc-stale-competition-force-rewrite-owner-alt-soft-override-min-raw-neighborhood-gain",
        type=float,
        default=-1.0,
    )
    parser.add_argument(
        "--preassoc-stale-competition-force-rewrite-acceptance-nearmiss-owner-alt-override-enable",
        action="store_true",
    )
    parser.add_argument(
        "--preassoc-stale-competition-force-rewrite-acceptance-nearmiss-owner-alt-override-min-acceptance-score",
        type=float,
        default=-1.0,
    )
    parser.add_argument(
        "--preassoc-stale-competition-force-rewrite-acceptance-nearmiss-owner-alt-override-max-owner-alt-det-box-iou",
        type=float,
        default=-1.0,
    )
    parser.add_argument(
        "--preassoc-stale-competition-force-rewrite-acceptance-nearmiss-owner-alt-override-max-owner-hits",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--preassoc-stale-competition-force-rewrite-acceptance-nearmiss-owner-alt-override-min-box-iou",
        type=float,
        default=-1.0,
    )
    parser.add_argument(
        "--preassoc-stale-competition-force-rewrite-acceptance-nearmiss-owner-alt-override-min-score",
        type=float,
        default=-1.0,
    )
    parser.add_argument(
        "--preassoc-stale-competition-force-rewrite-acceptance-nearmiss-owner-alt-override-min-raw-neighborhood-gain",
        type=float,
        default=-1.0,
    )
    parser.add_argument(
        "--preassoc-stale-competition-force-rewrite-stable-owner-min-hits",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--preassoc-stale-competition-force-rewrite-stable-owner-min-raw-neighborhood-gain",
        type=float,
        default=-1.0,
    )
    parser.add_argument(
        "--preassoc-stale-competition-force-rewrite-reroute-ready-min-box-iou",
        type=float,
        default=-1.0,
    )
    parser.add_argument(
        "--preassoc-stale-competition-force-rewrite-recovery-anchor-override-enable",
        action="store_true",
    )
    parser.add_argument(
        "--preassoc-stale-competition-force-rewrite-recovery-anchor-min-raw-neighborhood-gain",
        type=float,
        default=-1.0,
    )
    parser.add_argument(
        "--preassoc-stale-competition-force-rewrite-recovery-anchor-nearmiss-owner-alt-override-enable",
        action="store_true",
    )
    parser.add_argument(
        "--preassoc-stale-competition-force-rewrite-recovery-anchor-nearmiss-owner-alt-override-min-recovery-anchor-score",
        type=float,
        default=-1.0,
    )
    parser.add_argument(
        "--preassoc-stale-competition-force-rewrite-recovery-anchor-nearmiss-owner-alt-override-max-owner-alt-det-box-iou",
        type=float,
        default=-1.0,
    )
    parser.add_argument(
        "--preassoc-stale-competition-force-rewrite-recovery-anchor-nearmiss-owner-alt-override-max-owner-hits",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--preassoc-stale-competition-force-rewrite-recovery-anchor-nearmiss-owner-alt-override-min-box-iou",
        type=float,
        default=-1.0,
    )
    parser.add_argument(
        "--preassoc-stale-competition-force-rewrite-recovery-anchor-nearmiss-owner-alt-override-min-score",
        type=float,
        default=-1.0,
    )
    parser.add_argument(
        "--preassoc-stale-competition-force-rewrite-recovery-anchor-nearmiss-owner-alt-override-min-raw-neighborhood-gain",
        type=float,
        default=-1.0,
    )
    parser.add_argument("--preassoc-stale-competition-force-rewrite-recovery-memory-enable", action="store_true")
    parser.add_argument(
        "--preassoc-stale-competition-force-rewrite-recovery-memory-max-frame-gap",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--preassoc-stale-competition-force-rewrite-recovery-memory-min-score",
        type=float,
        default=0.85,
    )
    parser.add_argument(
        "--preassoc-stale-competition-force-rewrite-recovery-memory-min-box-iou",
        type=float,
        default=0.80,
    )
    parser.add_argument(
        "--preassoc-stale-competition-force-rewrite-recovery-memory-min-challenger-alt-box-iou",
        type=float,
        default=0.12,
    )
    parser.add_argument(
        "--preassoc-stale-competition-force-rewrite-recovery-memory-warmup-min-neighborhood-gain",
        type=float,
        default=-0.08,
    )
    parser.add_argument(
        "--preassoc-stale-competition-force-rewrite-recovery-memory-bonus",
        type=float,
        default=0.06,
    )
    parser.add_argument(
        "--preassoc-stale-competition-force-rewrite-recovery-memory-bonus-max-streak",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--preassoc-stale-competition-force-rewrite-recovery-memory-gate-bonus",
        type=float,
        default=1e-4,
    )
    parser.add_argument("--preassoc-stale-competition-force-rewrite-recovery-memory-anchor-enable", action="store_true")
    parser.add_argument(
        "--preassoc-stale-competition-force-rewrite-recovery-memory-anchor-min-raw-neighborhood-gain",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--preassoc-stale-competition-force-rewrite-recovery-memory-anchor-max-edge-deficit",
        type=float,
        default=0.35,
    )
    parser.add_argument(
        "--preassoc-stale-competition-force-rewrite-recovery-memory-extension-max-edge-deficit-delta",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--preassoc-stale-competition-force-rewrite-high-conf-reclaim-min-score",
        type=float,
        default=0.665,
    )
    parser.add_argument(
        "--preassoc-stale-competition-force-rewrite-high-conf-reclaim-min-box-iou",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--preassoc-stale-competition-force-rewrite-high-conf-reclaim-min-ranker-score",
        type=float,
        default=0.998,
    )
    parser.add_argument(
        "--preassoc-stale-competition-force-rewrite-high-conf-reclaim-min-acceptance-score",
        type=float,
        default=0.999,
    )
    parser.add_argument(
        "--preassoc-stale-competition-force-rewrite-high-conf-reclaim-min-track-hits",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--preassoc-stale-competition-force-rewrite-high-conf-reclaim-max-owner-hits",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--preassoc-stale-competition-force-rewrite-high-conf-reclaim-max-owner-alt-box-iou",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--preassoc-stale-competition-force-rewrite-high-conf-reclaim-min-challenger-alt-box-iou",
        type=float,
        default=0.08,
    )
    parser.add_argument(
        "--preassoc-stale-competition-force-rewrite-high-conf-reclaim-min-edge-advantage",
        type=float,
        default=-0.25,
    )
    parser.add_argument(
        "--preassoc-stale-competition-force-rewrite-high-conf-reclaim-min-neighborhood-gain",
        type=float,
        default=-0.10,
    )
    parser.add_argument(
        "--preassoc-stale-competition-acceptance-gate-high-conf-reclaim-min-score",
        type=float,
        default=0.99935,
    )
    parser.add_argument(
        "--preassoc-stale-competition-acceptance-gate-high-conf-reclaim-min-recovery-anchor-score",
        type=float,
        default=0.59,
    )
    parser.add_argument("--preassoc-stale-competition-export-jsonl", default="")
    parser.add_argument("--local-contention-export-jsonl", default="")
    parser.add_argument("--local-contention-topk", type=int, default=3)
    parser.add_argument("--local-contention-min-box-iou", type=float, default=0.5)
    parser.add_argument("--local-contention-max-time-since-update", type=int, default=8)
    parser.add_argument("--local-contention-min-challenger-hits", type=int, default=3)
    parser.add_argument("--local-contention-owner-weak-hits", type=int, default=8)
    parser.add_argument("--local-contention-ranker-checkpoint", default="")
    parser.add_argument("--local-contention-ranker-thresh", type=float, default=0.5)
    parser.add_argument("--local-contention-ranker-bias", type=float, default=0.0)
    parser.add_argument("--local-contention-ranker-min-margin-to-second", type=float, default=0.0)
    parser.add_argument("--local-contention-ranker-margin-bias", type=float, default=0.0)
    parser.add_argument("--preassoc-stale-competition-acceptance-gate-checkpoint", default="")
    parser.add_argument("--preassoc-stale-competition-acceptance-gate-thresh", type=float, default=0.5)
    parser.add_argument("--preassoc-stale-competition-recovery-anchor-gate-checkpoint", default="")
    parser.add_argument("--preassoc-stale-competition-recovery-anchor-gate-thresh", type=float, default=0.5)
    parser.add_argument(
        "--competition-track-max-frames-per-batch",
        type=int,
        default=0,
        help="optional maximum total frame budget per competition tracking batch; >0 splits long sequence lists into restartable batches and merges outputs",
    )
    parser.add_argument("--registry-csv", default=str(REGISTRY_CSV))
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def timestamp_tag() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


def ensure_dataset_link(benchmark: str) -> Path:
    dataset_dir_name = {
        "MOT17": "mot",
        "MOT20": "MOT20",
        "DanceTrack": "dancetrack",
    }.get(benchmark, "")
    if not dataset_dir_name:
        raise ValueError(f"Unsupported benchmark for dataset link check: {benchmark}")

    link_root = DEEP_ROOT / "data" / dataset_dir_name
    required_annotation_name = "val.json" if benchmark == "DanceTrack" else "val_half.json"
    required_annotation = link_root / "annotations" / required_annotation_name
    if required_annotation.is_file():
        return link_root

    if benchmark == "DanceTrack":
        source_root = (REPO_ROOT.parents[1] / "datasets" / "DanceTrack" / "extracted").resolve()
        source_annotation = source_root / "annotations" / "val.json"
    else:
        source_root = (REPO_ROOT.parents[1] / "datasets" / benchmark).resolve()
        source_annotation = source_root / "annotations" / "val_half.json"
    if not source_annotation.is_file():
        raise FileNotFoundError(f"Missing {benchmark} validation annotation: {source_annotation}")

    if link_root.exists():
        raise FileExistsError(f"Dataset link root exists but required annotation is missing: {link_root}")
    if link_root.is_symlink():
        link_root.unlink()

    link_root.symlink_to(source_root, target_is_directory=True)
    if not required_annotation.is_file():
        raise FileNotFoundError(f"Dataset link created but annotation is still missing: {required_annotation}")
    return link_root


def count_sequence_frames(dataset_root: Path, seq_name: str) -> int:
    candidate_dirs = [
        dataset_root / "val" / seq_name / "img1",
        dataset_root / "train" / seq_name / "img1",
        dataset_root / seq_name / "img1",
    ]
    for img_dir in candidate_dirs:
        if img_dir.is_dir():
            return len(list(img_dir.glob("*.jpg"))) + len(list(img_dir.glob("*.png")))
    raise FileNotFoundError(f"Unable to locate image directory for sequence {seq_name} under {dataset_root}")


def split_seq_names_by_frame_budget(dataset_root: Path, seq_names: List[str], frame_budget: int) -> List[List[str]]:
    if frame_budget <= 0 or len(seq_names) <= 1:
        return [list(seq_names)]
    batches: List[List[str]] = []
    current_batch: List[str] = []
    current_frames = 0
    for seq_name in seq_names:
        seq_frames = count_sequence_frames(dataset_root, seq_name)
        if current_batch and current_frames + seq_frames > frame_budget:
            batches.append(current_batch)
            current_batch = [seq_name]
            current_frames = seq_frames
        else:
            current_batch.append(seq_name)
            current_frames += seq_frames
    if current_batch:
        batches.append(current_batch)
    return batches


def resolve_weight_source(filename: str) -> Path:
    candidates = [
        REPO_ROOT / "external" / "BoT-SORT-main" / "pretrained" / filename,
        REPO_ROOT.parents[1] / "lapmot_assoc_proto" / filename,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Unable to find required Deep-OC-SORT weight {filename} in known local sources")


def ensure_reid_weights(benchmark: str) -> Path:
    weights_root = DEEP_ROOT / "external" / "weights"
    weights_root.mkdir(parents=True, exist_ok=True)

    required_filenames = ["osnet_ain_ms_d_c.pth.tar"]
    if benchmark == "MOT17":
        required_filenames.append("mot17_sbs_S50.pth")
    elif benchmark == "MOT20":
        required_filenames.append("mot20_sbs_S50.pth")
    elif benchmark == "DanceTrack":
        required_filenames.append("dance_sbs_S50.pth")
    else:
        raise ValueError(f"Unsupported benchmark for weight check: {benchmark}")

    for filename in required_filenames:
        target = weights_root / filename
        if target.is_file():
            continue
        if target.exists():
            raise FileExistsError(f"Weight target exists but is not a regular file: {target}")
        if target.is_symlink():
            target.unlink()
        target.symlink_to(resolve_weight_source(filename))
    return weights_root


def ensure_detector_weights(benchmark: str) -> Path:
    weights_root = DEEP_ROOT / "external" / "weights"
    weights_root.mkdir(parents=True, exist_ok=True)

    detector_filename = {
        "MOT17": "bytetrack_x_mot17.pth.tar",
        "MOT20": "bytetrack_x_mot20.tar",
        "DanceTrack": "bytetrack_dance_model.pth.tar",
    }.get(benchmark, "")
    if not detector_filename:
        raise ValueError(f"Unsupported benchmark for detector weight check: {benchmark}")

    target = weights_root / detector_filename
    if target.is_file():
        return weights_root
    if target.exists():
        raise FileExistsError(f"Detector weight target exists but is not a regular file: {target}")
    if target.is_symlink():
        target.unlink()
    target.symlink_to(resolve_weight_source(detector_filename))
    return weights_root


def write_rows(path: Path, fieldnames: Iterable[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def update_row(rows: List[Dict[str, object]], step: str, **updates: object) -> None:
    for row in rows:
        if str(row["step"]) == step:
            row.update(updates)
            return
    raise KeyError(f"Missing queue step: {step}")


def start_orion_keepalive(*, step_log_path: Path, env: Dict[str, str], interval_sec: int) -> tuple[threading.Thread, threading.Event, Path] | tuple[None, None, None]:
    nvidia_smi = shutil.which("nvidia-smi", path=env.get("PATH"))
    if not nvidia_smi:
        return None, None, None

    keepalive_log = step_log_path.with_name(f"{step_log_path.stem}_orion_keepalive.log")
    stop_event = threading.Event()

    def worker() -> None:
        keepalive_log.parent.mkdir(parents=True, exist_ok=True)
        with keepalive_log.open("w", encoding="utf-8") as handle:
            handle.write(f"[started_at] {now_iso()}\n")
            handle.write(f"[nvidia_smi] {nvidia_smi}\n")
            handle.write(f"[interval_sec] {interval_sec}\n\n")
            handle.flush()
            while not stop_event.is_set():
                heartbeat_at = now_iso()
                try:
                    result = subprocess.run(
                        [nvidia_smi, "--query-gpu=name,memory.total", "--format=csv,noheader"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        env=env,
                        timeout=20,
                        check=False,
                    )
                    preview = " | ".join(line.strip() for line in (result.stdout or "").splitlines()[:2] if line.strip())
                    handle.write(f"[heartbeat_at] {heartbeat_at} rc={result.returncode} output={preview}\n")
                except subprocess.TimeoutExpired:
                    handle.write(f"[heartbeat_at] {heartbeat_at} timeout=20s\n")
                except Exception as exc:
                    handle.write(f"[heartbeat_at] {heartbeat_at} error={exc}\n")
                handle.flush()
                if stop_event.wait(interval_sec):
                    break
            handle.write(f"\n[finished_at] {now_iso()}\n")
            handle.flush()

    thread = threading.Thread(target=worker, name=f"orion-keepalive-{step_log_path.stem}", daemon=True)
    thread.start()
    return thread, stop_event, keepalive_log


def run_step(cmd: List[str], log_path: Path, *, cwd: Path, enable_orion_keepalive: bool = False, orion_keepalive_interval_sec: int = 240) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    sanitized_env_names: List[str] = []
    for env_name in ("ORION_TASK_IDLE_TIME",):
        if env.pop(env_name, None) is not None:
            sanitized_env_names.append(env_name)
    keepalive_thread: threading.Thread | None = None
    keepalive_stop_event: threading.Event | None = None
    keepalive_log_path: Path | None = None
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"[started_at] {now_iso()}\n")
        handle.write(f"[cwd] {cwd}\n")
        handle.write("[cmd] " + " ".join(cmd) + "\n\n")
        if sanitized_env_names:
            handle.write("[sanitized_env] unset " + ",".join(sanitized_env_names) + "\n\n")
        if enable_orion_keepalive:
            keepalive_thread, keepalive_stop_event, keepalive_log_path = start_orion_keepalive(
                step_log_path=log_path,
                env=env.copy(),
                interval_sec=orion_keepalive_interval_sec,
            )
            if keepalive_log_path is not None:
                handle.write(
                    "[orion_keepalive] "
                    f"enabled interval_sec={orion_keepalive_interval_sec} log={keepalive_log_path}\n\n"
                )
            else:
                handle.write("[orion_keepalive] skipped because nvidia-smi is unavailable\n\n")
        handle.flush()
        try:
            process = subprocess.run(cmd, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT, env=env)
        finally:
            if keepalive_stop_event is not None:
                keepalive_stop_event.set()
            if keepalive_thread is not None:
                keepalive_thread.join(timeout=30)
        handle.write(f"\n[finished_at] {now_iso()}\n")
        handle.write(f"[return_code] {process.returncode}\n")
    return int(process.returncode)


def append_registry(summary_csv: Path, run_root: Path, status: str, notes: str, registry_csv: str, benchmark: str, split_label: str) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "append_experiment_record.py"),
        "--csv",
        str(registry_csv),
        "--kind",
        "eval",
        "--status",
        status,
        "--script",
        "scripts/run_deep_ocsort_preassoc_competition_dataset_eval.py",
        "--dataset",
        benchmark,
        "--split",
        split_label,
        "--tracker-family",
        "deep_ocsort_preassoc_competition",
        "--variant",
        run_root.name,
        "--tag",
        "deep_ocsort_preassoc_competition_dataset_eval",
        "--run-root",
        str(run_root),
        "--summary-csv",
        str(summary_csv),
        "--notes",
        notes,
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
        rows = []
        for row in reader:
            filtered = [token for token in row if token != ""]
            if filtered:
                rows.append(filtered)
    if len(rows) < 2:
        raise RuntimeError(f"Unexpected TrackEval summary format: {path}")
    fields = rows[0]
    values = rows[1]
    data: Dict[str, float] = {}
    for key, value in zip(fields, values):
        try:
            data[key] = float(value)
        except ValueError:
            continue
    return data


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


def resolve_seq_names(args: argparse.Namespace) -> List[str]:
    if args.seq_names:
        return [str(seq) for seq in args.seq_names]
    if args.seq_name:
        return [str(args.seq_name)]
    if args.benchmark == "MOT17":
        return [
            "MOT17-02-FRCNN",
            "MOT17-04-FRCNN",
            "MOT17-05-FRCNN",
            "MOT17-09-FRCNN",
            "MOT17-10-FRCNN",
            "MOT17-11-FRCNN",
            "MOT17-13-FRCNN",
        ]
    if args.benchmark == "DanceTrack":
        dance_val_root = (REPO_ROOT.parents[1] / "datasets" / "DanceTrack" / "extracted" / "val").resolve()
        if not dance_val_root.is_dir():
            raise FileNotFoundError(f"Missing DanceTrack val directory: {dance_val_root}")
        return sorted(path.name for path in dance_val_root.iterdir() if path.is_dir())
    return ["MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05"]


def seq_note(seq_names: List[str]) -> str:
    return "|".join(seq_names)


def resolve_runtime_summary(track_dir: Path) -> Path | None:
    runtime_csv = track_dir / "fgas_analysis" / f"{track_dir.name}_summary.csv"
    if runtime_csv.is_file():
        return runtime_csv.resolve()
    return None


def load_runtime_rows(summary_csv: Path, label: str) -> List[Dict[str, int | str]]:
    rows: List[Dict[str, int | str]] = []
    if not summary_csv.is_file():
        return rows
    with summary_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "name": label,
                    "seq": str(row.get("seq_name", "")),
                    "frames_seen": int(row.get("frames_seen", 0) or 0),
                    "preassoc_stale_competition_rows": int(row.get("preassoc_stale_competition_rows", 0) or 0),
                    "preassoc_stale_competition_candidate_rows": int(row.get("preassoc_stale_competition_candidate_rows", 0) or 0),
                    "preassoc_stale_competition_biased_edges": int(row.get("preassoc_stale_competition_biased_edges", 0) or 0),
                    "preassoc_stale_competition_takeover_risk_rejected_rows": int(
                        row.get("preassoc_stale_competition_takeover_risk_rejected_rows", 0) or 0
                    ),
                    "preassoc_stale_competition_owner_alt_biased_edges": int(
                        row.get("preassoc_stale_competition_owner_alt_biased_edges", 0) or 0
                    ),
                    "preassoc_stale_competition_owner_alt_risk_rejected_rows": int(
                        row.get("preassoc_stale_competition_owner_alt_risk_rejected_rows", 0) or 0
                    ),
                    "preassoc_stale_competition_owner_alt_released_owners": int(
                        row.get("preassoc_stale_competition_owner_alt_released_owners", 0) or 0
                    ),
                    "preassoc_stale_competition_selected_matches": int(row.get("preassoc_stale_competition_selected_matches", 0) or 0),
                    "preassoc_stale_competition_forced_gate_rejected_rows": int(
                        row.get("preassoc_stale_competition_forced_gate_rejected_rows", 0) or 0
                    ),
                    "preassoc_stale_competition_acceptance_gate_scored_rows": int(
                        row.get("preassoc_stale_competition_acceptance_gate_scored_rows", 0) or 0
                    ),
                    "preassoc_stale_competition_acceptance_gate_rejected_rows": int(
                        row.get("preassoc_stale_competition_acceptance_gate_rejected_rows", 0) or 0
                    ),
                    "preassoc_stale_competition_acceptance_gate_accepted_rows": int(
                        row.get("preassoc_stale_competition_acceptance_gate_accepted_rows", 0) or 0
                    ),
                    "preassoc_stale_competition_acceptance_gate_high_conf_reclaim_override_rows": int(
                        row.get("preassoc_stale_competition_acceptance_gate_high_conf_reclaim_override_rows", 0) or 0
                    ),
                    "preassoc_stale_competition_recovery_anchor_gate_scored_rows": int(
                        row.get("preassoc_stale_competition_recovery_anchor_gate_scored_rows", 0) or 0
                    ),
                    "preassoc_stale_competition_recovery_anchor_gate_rejected_rows": int(
                        row.get("preassoc_stale_competition_recovery_anchor_gate_rejected_rows", 0) or 0
                    ),
                    "preassoc_stale_competition_recovery_anchor_gate_accepted_rows": int(
                        row.get("preassoc_stale_competition_recovery_anchor_gate_accepted_rows", 0) or 0
                    ),
                    "preassoc_stale_competition_force_rewrite_scored_rows": int(
                        row.get("preassoc_stale_competition_force_rewrite_scored_rows", 0) or 0
                    ),
                    "preassoc_stale_competition_force_rewrite_rejected_rows": int(
                        row.get("preassoc_stale_competition_force_rewrite_rejected_rows", 0) or 0
                    ),
                    "preassoc_stale_competition_force_rewrite_accepted_rows": int(
                        row.get("preassoc_stale_competition_force_rewrite_accepted_rows", 0) or 0
                    ),
                    "preassoc_stale_competition_force_rewrite_recovery_anchor_override_rows": int(
                        row.get("preassoc_stale_competition_force_rewrite_recovery_anchor_override_rows", 0) or 0
                    ),
                    "preassoc_stale_competition_force_rewrite_acceptance_nearmiss_owner_alt_override_rows": int(
                        row.get(
                            "preassoc_stale_competition_force_rewrite_acceptance_nearmiss_owner_alt_override_rows",
                            0,
                        )
                        or 0
                    ),
                    "preassoc_stale_competition_force_rewrite_recovery_anchor_nearmiss_owner_alt_override_rows": int(
                        row.get(
                            "preassoc_stale_competition_force_rewrite_recovery_anchor_nearmiss_owner_alt_override_rows",
                            0,
                        )
                        or 0
                    ),
                    "local_contention_units": int(row.get("local_contention_units", 0) or 0),
                    "local_contention_candidate_pairs": int(row.get("local_contention_candidate_pairs", 0) or 0),
                    "local_contention_labeled_pairs": int(row.get("local_contention_labeled_pairs", 0) or 0),
                    "local_contention_positive_pairs": int(row.get("local_contention_positive_pairs", 0) or 0),
                    "local_contention_negative_pairs": int(row.get("local_contention_negative_pairs", 0) or 0),
                }
            )
    return rows


def summarize_runtime_rows(*, runtime_rows: List[Dict[str, int | str]], label: str, seq_label: str, summary_csv: Path | None) -> Dict[str, int | str]:
    total = {
        "name": label,
        "seq": seq_label,
        "frames_seen": 0,
        "preassoc_stale_competition_rows": 0,
        "preassoc_stale_competition_candidate_rows": 0,
        "preassoc_stale_competition_biased_edges": 0,
        "preassoc_stale_competition_takeover_risk_rejected_rows": 0,
        "preassoc_stale_competition_owner_alt_biased_edges": 0,
        "preassoc_stale_competition_owner_alt_risk_rejected_rows": 0,
        "preassoc_stale_competition_owner_alt_released_owners": 0,
        "preassoc_stale_competition_selected_matches": 0,
        "preassoc_stale_competition_forced_gate_rejected_rows": 0,
        "preassoc_stale_competition_acceptance_gate_scored_rows": 0,
        "preassoc_stale_competition_acceptance_gate_rejected_rows": 0,
        "preassoc_stale_competition_acceptance_gate_accepted_rows": 0,
        "preassoc_stale_competition_acceptance_gate_high_conf_reclaim_override_rows": 0,
        "preassoc_stale_competition_recovery_anchor_gate_scored_rows": 0,
        "preassoc_stale_competition_recovery_anchor_gate_rejected_rows": 0,
        "preassoc_stale_competition_recovery_anchor_gate_accepted_rows": 0,
        "preassoc_stale_competition_force_rewrite_scored_rows": 0,
        "preassoc_stale_competition_force_rewrite_rejected_rows": 0,
        "preassoc_stale_competition_force_rewrite_accepted_rows": 0,
        "preassoc_stale_competition_force_rewrite_recovery_anchor_override_rows": 0,
        "preassoc_stale_competition_force_rewrite_acceptance_nearmiss_owner_alt_override_rows": 0,
        "preassoc_stale_competition_force_rewrite_recovery_anchor_nearmiss_owner_alt_override_rows": 0,
        "local_contention_units": 0,
        "local_contention_candidate_pairs": 0,
        "local_contention_labeled_pairs": 0,
        "local_contention_positive_pairs": 0,
        "local_contention_negative_pairs": 0,
        "summary_csv": str(summary_csv) if summary_csv is not None else "",
    }
    for row in runtime_rows:
        total["frames_seen"] = int(total["frames_seen"]) + int(row.get("frames_seen", 0))
        total["preassoc_stale_competition_rows"] = int(total["preassoc_stale_competition_rows"]) + int(row.get("preassoc_stale_competition_rows", 0))
        total["preassoc_stale_competition_candidate_rows"] = int(total["preassoc_stale_competition_candidate_rows"]) + int(
            row.get("preassoc_stale_competition_candidate_rows", 0)
        )
        total["preassoc_stale_competition_biased_edges"] = int(total["preassoc_stale_competition_biased_edges"]) + int(
            row.get("preassoc_stale_competition_biased_edges", 0)
        )
        total["preassoc_stale_competition_takeover_risk_rejected_rows"] = int(
            total["preassoc_stale_competition_takeover_risk_rejected_rows"]
        ) + int(row.get("preassoc_stale_competition_takeover_risk_rejected_rows", 0))
        total["preassoc_stale_competition_owner_alt_biased_edges"] = int(
            total["preassoc_stale_competition_owner_alt_biased_edges"]
        ) + int(row.get("preassoc_stale_competition_owner_alt_biased_edges", 0))
        total["preassoc_stale_competition_owner_alt_risk_rejected_rows"] = int(
            total["preassoc_stale_competition_owner_alt_risk_rejected_rows"]
        ) + int(row.get("preassoc_stale_competition_owner_alt_risk_rejected_rows", 0))
        total["preassoc_stale_competition_owner_alt_released_owners"] = int(
            total["preassoc_stale_competition_owner_alt_released_owners"]
        ) + int(row.get("preassoc_stale_competition_owner_alt_released_owners", 0))
        total["preassoc_stale_competition_selected_matches"] = int(total["preassoc_stale_competition_selected_matches"]) + int(
            row.get("preassoc_stale_competition_selected_matches", 0)
        )
        total["preassoc_stale_competition_forced_gate_rejected_rows"] = int(
            total["preassoc_stale_competition_forced_gate_rejected_rows"]
        ) + int(row.get("preassoc_stale_competition_forced_gate_rejected_rows", 0))
        total["preassoc_stale_competition_acceptance_gate_scored_rows"] = int(
            total["preassoc_stale_competition_acceptance_gate_scored_rows"]
        ) + int(row.get("preassoc_stale_competition_acceptance_gate_scored_rows", 0))
        total["preassoc_stale_competition_acceptance_gate_rejected_rows"] = int(
            total["preassoc_stale_competition_acceptance_gate_rejected_rows"]
        ) + int(row.get("preassoc_stale_competition_acceptance_gate_rejected_rows", 0))
        total["preassoc_stale_competition_acceptance_gate_accepted_rows"] = int(
            total["preassoc_stale_competition_acceptance_gate_accepted_rows"]
        ) + int(row.get("preassoc_stale_competition_acceptance_gate_accepted_rows", 0))
        total["preassoc_stale_competition_acceptance_gate_high_conf_reclaim_override_rows"] = int(
            total["preassoc_stale_competition_acceptance_gate_high_conf_reclaim_override_rows"]
        ) + int(row.get("preassoc_stale_competition_acceptance_gate_high_conf_reclaim_override_rows", 0))
        total["preassoc_stale_competition_recovery_anchor_gate_scored_rows"] = int(
            total["preassoc_stale_competition_recovery_anchor_gate_scored_rows"]
        ) + int(row.get("preassoc_stale_competition_recovery_anchor_gate_scored_rows", 0))
        total["preassoc_stale_competition_recovery_anchor_gate_rejected_rows"] = int(
            total["preassoc_stale_competition_recovery_anchor_gate_rejected_rows"]
        ) + int(row.get("preassoc_stale_competition_recovery_anchor_gate_rejected_rows", 0))
        total["preassoc_stale_competition_recovery_anchor_gate_accepted_rows"] = int(
            total["preassoc_stale_competition_recovery_anchor_gate_accepted_rows"]
        ) + int(row.get("preassoc_stale_competition_recovery_anchor_gate_accepted_rows", 0))
        total["preassoc_stale_competition_force_rewrite_scored_rows"] = int(
            total["preassoc_stale_competition_force_rewrite_scored_rows"]
        ) + int(row.get("preassoc_stale_competition_force_rewrite_scored_rows", 0))
        total["preassoc_stale_competition_force_rewrite_rejected_rows"] = int(
            total["preassoc_stale_competition_force_rewrite_rejected_rows"]
        ) + int(row.get("preassoc_stale_competition_force_rewrite_rejected_rows", 0))
        total["preassoc_stale_competition_force_rewrite_accepted_rows"] = int(
            total["preassoc_stale_competition_force_rewrite_accepted_rows"]
        ) + int(row.get("preassoc_stale_competition_force_rewrite_accepted_rows", 0))
        total["preassoc_stale_competition_force_rewrite_recovery_anchor_override_rows"] = int(
            total["preassoc_stale_competition_force_rewrite_recovery_anchor_override_rows"]
        ) + int(row.get("preassoc_stale_competition_force_rewrite_recovery_anchor_override_rows", 0))
        total["preassoc_stale_competition_force_rewrite_acceptance_nearmiss_owner_alt_override_rows"] = int(
            total["preassoc_stale_competition_force_rewrite_acceptance_nearmiss_owner_alt_override_rows"]
        ) + int(row.get("preassoc_stale_competition_force_rewrite_acceptance_nearmiss_owner_alt_override_rows", 0))
        total["preassoc_stale_competition_force_rewrite_recovery_anchor_nearmiss_owner_alt_override_rows"] = int(
            total["preassoc_stale_competition_force_rewrite_recovery_anchor_nearmiss_owner_alt_override_rows"]
        ) + int(
            row.get("preassoc_stale_competition_force_rewrite_recovery_anchor_nearmiss_owner_alt_override_rows", 0)
        )
        total["local_contention_units"] = int(total["local_contention_units"]) + int(row.get("local_contention_units", 0))
        total["local_contention_candidate_pairs"] = int(total["local_contention_candidate_pairs"]) + int(
            row.get("local_contention_candidate_pairs", 0)
        )
        total["local_contention_labeled_pairs"] = int(total["local_contention_labeled_pairs"]) + int(
            row.get("local_contention_labeled_pairs", 0)
        )
        total["local_contention_positive_pairs"] = int(total["local_contention_positive_pairs"]) + int(
            row.get("local_contention_positive_pairs", 0)
        )
        total["local_contention_negative_pairs"] = int(total["local_contention_negative_pairs"]) + int(
            row.get("local_contention_negative_pairs", 0)
        )
    return total


def summarize_local_contention_export(export_jsonl: Path, label: str) -> Dict[str, object]:
    rows = 0
    units = set()
    labeled_rows = 0
    positive_rows = 0
    negative_rows = 0
    if export_jsonl.is_file():
        with export_jsonl.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                rows += 1
                units.add(str(row.get("unit_id", "")))
                raw_label = row.get("label_prefer_challenger", -1)
                label_value = int(raw_label) if raw_label is not None else -1
                if label_value >= 0:
                    labeled_rows += 1
                    if label_value == 1:
                        positive_rows += 1
                    else:
                        negative_rows += 1
    unit_count = len({unit for unit in units if unit})
    avg_candidates_per_unit = (float(rows) / float(unit_count)) if unit_count > 0 else 0.0
    return {
        "name": label,
        "rows": int(rows),
        "units": int(unit_count),
        "labeled_rows": int(labeled_rows),
        "positive_rows": int(positive_rows),
        "negative_rows": int(negative_rows),
        "avg_candidates_per_unit": float(avg_candidates_per_unit),
        "jsonl_path": str(export_jsonl),
    }


def copy_sequence_outputs(src_out: Path, dst_out: Path, seq_names: List[str]) -> None:
    src_data_dir = src_out / "data"
    dst_data_dir = dst_out / "data"
    dst_data_dir.mkdir(parents=True, exist_ok=True)
    for seq_name in seq_names:
        src_txt = src_data_dir / f"{seq_name}.txt"
        if not src_txt.is_file():
            raise FileNotFoundError(f"Missing batch sequence output: {src_txt}")
        shutil.copy2(src_txt, dst_data_dir / src_txt.name)


def merge_runtime_summaries(batch_summaries: List[Path], merged_summary: Path) -> None:
    rows: List[Dict[str, str]] = []
    fieldnames: List[str] | None = None
    for summary_path in batch_summaries:
        if not summary_path.is_file():
            continue
        with summary_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if fieldnames is None:
                fieldnames = list(reader.fieldnames or [])
            rows.extend({key: row.get(key, "") for key in fieldnames or []} for row in reader)
    if fieldnames is None or not rows:
        return
    merged_summary.parent.mkdir(parents=True, exist_ok=True)
    with merged_summary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def concatenate_jsonl(batch_jsonls: List[Path], merged_jsonl: Path) -> None:
    merged_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with merged_jsonl.open("w", encoding="utf-8") as merged_handle:
        for batch_jsonl in batch_jsonls:
            if not batch_jsonl.is_file():
                continue
            with batch_jsonl.open("r", encoding="utf-8") as batch_handle:
                for line in batch_handle:
                    if line.strip():
                        merged_handle.write(line if line.endswith("\n") else line + "\n")


def append_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)


def _missing_sequence_outputs(out_dir: Path, seq_names: List[str]) -> List[str]:
    data_dir = out_dir / "data"
    if not data_dir.is_dir():
        return [str(data_dir)]
    missing = []
    for seq_name in seq_names:
        txt_path = data_dir / f"{seq_name}.txt"
        if not txt_path.is_file():
            missing.append(str(txt_path))
    return missing


def ensure_tracking_outputs(track_out: Path, post_out: Path, seq_names: List[str]) -> None:
    missing = _missing_sequence_outputs(track_out, seq_names) + _missing_sequence_outputs(post_out, seq_names)
    if missing:
        preview = ", ".join(missing[:10])
        suffix = "" if len(missing) <= 10 else f" ... (+{len(missing) - 10} more)"
        raise FileNotFoundError(f"Missing tracking outputs after successful return code: {preview}{suffix}")


def ensure_eval_outputs(eval_out: Path) -> None:
    required_files = [
        eval_out / "pedestrian_summary.txt",
        eval_out / "pedestrian_detailed.csv",
    ]
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing evaluation outputs after successful return code: " + ", ".join(missing)
        )


def prepare_reused_eval_dir(source_post_out: Path, target_eval_out: Path) -> None:
    source_data_dir = source_post_out / "data"
    if not source_data_dir.is_dir():
        raise FileNotFoundError(f"Missing reused tracker data directory: {source_data_dir}")
    if target_eval_out.is_symlink():
        raise FileExistsError(f"Target eval dir must not be a symlink: {target_eval_out}")

    target_eval_out.mkdir(parents=True, exist_ok=True)
    target_data_dir = target_eval_out / "data"
    if target_data_dir.exists() or target_data_dir.is_symlink():
        if target_data_dir.is_symlink():
            if target_data_dir.resolve() != source_data_dir.resolve():
                target_data_dir.unlink()
                target_data_dir.symlink_to(source_data_dir, target_is_directory=True)
        elif target_data_dir.is_dir():
            shutil.rmtree(target_data_dir)
            target_data_dir.symlink_to(source_data_dir, target_is_directory=True)
        else:
            target_data_dir.unlink()
            target_data_dir.symlink_to(source_data_dir, target_is_directory=True)
    else:
        target_data_dir.symlink_to(source_data_dir, target_is_directory=True)

    for child in target_eval_out.iterdir():
        if child.name == "data":
            continue
        if child.is_symlink() or child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)


def ensure_success(step: str, return_code: int, rows: List[Dict[str, object]], summary_csv: Path, out_dir: Path, log_path: Path, notes: str) -> None:
    finished_at = now_iso()
    status = "success" if return_code == 0 else "failed"
    step_notes = notes if return_code == 0 else f"failed: {step} return_code={return_code}"
    update_row(
        rows,
        step,
        status=status,
        finished_at=finished_at,
        out_dir=str(out_dir),
        summary_csv=str(summary_csv),
        log_path=str(log_path),
        notes=step_notes,
    )
    write_rows(summary_csv, QUEUE_FIELDS, rows)
    if return_code != 0:
        raise RuntimeError(f"Step failed: {step}")


def extend_dataset_tracking_profile(cmd: List[str], dataset_name: str) -> None:
    if dataset_name == "dance":
        cmd.extend(
            [
                "--aspect_ratio_thresh",
                "1000",
                "--w_assoc_emb",
                "1.25",
                "--aw_param",
                "1",
            ]
        )
        return

    if dataset_name == "mot20":
        cmd.extend(["--track_thresh", "0.4"])

    cmd.extend(
        [
            "--w_assoc_emb",
            "0.75",
            "--aw_param",
            "0.5",
        ]
    )


def build_intervention_track_cmd(
    *,
    args: argparse.Namespace,
    dataset_name: str,
    trackers_root: Path,
    exp_name: str,
    seq_names: List[str],
) -> List[str]:
    intervention_track_cmd = [
        sys.executable,
        "main.py",
        "--dataset",
        dataset_name,
        "--result_folder",
        str(trackers_root),
        "--exp_name",
        exp_name,
        "--seq-filter",
        *seq_names,
        "--post",
        "--grid_off",
        "--new_kf_off",
    ]
    extend_dataset_tracking_profile(intervention_track_cmd, dataset_name)
    if not args.disable_preassoc_stale_competition:
        intervention_track_cmd.extend(
            [
                "--preassoc-stale-competition-enable",
                "--preassoc-stale-competition-min-time-since-update",
                str(args.preassoc_stale_competition_min_time_since_update),
                "--preassoc-stale-competition-max-time-since-update",
                str(args.preassoc_stale_competition_max_time_since_update),
                "--preassoc-stale-competition-min-hits",
                str(args.preassoc_stale_competition_min_hits),
                "--preassoc-stale-competition-min-box-iou",
                str(args.preassoc_stale_competition_min_box_iou),
                "--preassoc-stale-competition-min-edge-score",
                str(args.preassoc_stale_competition_min_edge_score),
                "--preassoc-stale-competition-bias",
                str(args.preassoc_stale_competition_bias),
                "--preassoc-stale-competition-iou-scale",
                str(args.preassoc_stale_competition_iou_scale),
            ]
        )
        if args.preassoc_stale_competition_require_raw_owner:
            intervention_track_cmd.append("--preassoc-stale-competition-require-raw-owner")
        if args.preassoc_stale_competition_min_hit_gap_vs_owner > 0:
            intervention_track_cmd.extend(
                [
                    "--preassoc-stale-competition-min-hit-gap-vs-owner",
                    str(args.preassoc_stale_competition_min_hit_gap_vs_owner),
                ]
            )
        if args.preassoc_stale_competition_min_age_gap_vs_owner > 0:
            intervention_track_cmd.extend(
                [
                    "--preassoc-stale-competition-min-age-gap-vs-owner",
                    str(args.preassoc_stale_competition_min_age_gap_vs_owner),
                ]
            )
        if args.preassoc_stale_competition_owner_max_hits > 0:
            intervention_track_cmd.extend(
                [
                    "--preassoc-stale-competition-owner-max-hits",
                    str(args.preassoc_stale_competition_owner_max_hits),
                ]
            )
        if args.preassoc_stale_competition_owner_max_age > 0:
            intervention_track_cmd.extend(
                [
                    "--preassoc-stale-competition-owner-max-age",
                    str(args.preassoc_stale_competition_owner_max_age),
                ]
            )
        if args.preassoc_stale_competition_owner_edge_penalty > 0.0:
            intervention_track_cmd.extend(
                [
                    "--preassoc-stale-competition-owner-edge-penalty",
                    str(args.preassoc_stale_competition_owner_edge_penalty),
                ]
            )
        if args.preassoc_stale_competition_takeover_soft_margin_floor >= 0.0:
            intervention_track_cmd.extend(
                [
                    "--preassoc-stale-competition-takeover-soft-margin-floor",
                    str(args.preassoc_stale_competition_takeover_soft_margin_floor),
                ]
            )
        if args.preassoc_stale_competition_takeover_soft_edge_advantage_floor >= 0.0:
            intervention_track_cmd.extend(
                [
                    "--preassoc-stale-competition-takeover-soft-edge-advantage-floor",
                    str(args.preassoc_stale_competition_takeover_soft_edge_advantage_floor),
                ]
            )
        if args.preassoc_stale_competition_takeover_min_force_risk_scale >= 0.0:
            intervention_track_cmd.extend(
                [
                    "--preassoc-stale-competition-takeover-min-force-risk-scale",
                    str(args.preassoc_stale_competition_takeover_min_force_risk_scale),
                ]
            )
        if args.preassoc_stale_competition_min_ranker_margin_to_second >= 0.0:
            intervention_track_cmd.extend(
                [
                    "--preassoc-stale-competition-min-ranker-margin-to-second",
                    str(args.preassoc_stale_competition_min_ranker_margin_to_second),
                ]
            )
        if args.preassoc_stale_competition_min_edge_advantage_vs_owner >= 0.0:
            intervention_track_cmd.extend(
                [
                    "--preassoc-stale-competition-min-edge-advantage-vs-owner",
                    str(args.preassoc_stale_competition_min_edge_advantage_vs_owner),
                ]
            )
        if args.preassoc_stale_competition_owner_alt_det_bias > 0.0:
            intervention_track_cmd.extend(
                [
                    "--preassoc-stale-competition-owner-alt-det-bias",
                    str(args.preassoc_stale_competition_owner_alt_det_bias),
                    "--preassoc-stale-competition-owner-alt-det-min-score",
                    str(args.preassoc_stale_competition_owner_alt_det_min_score),
                    "--preassoc-stale-competition-owner-alt-det-min-box-iou",
                    str(args.preassoc_stale_competition_owner_alt_det_min_box_iou),
                ]
            )
            if args.preassoc_stale_competition_owner_alt_det_min_ranker_score >= 0.0:
                intervention_track_cmd.extend(
                    [
                        "--preassoc-stale-competition-owner-alt-det-min-ranker-score",
                        str(args.preassoc_stale_competition_owner_alt_det_min_ranker_score),
                    ]
                )
            if args.preassoc_stale_competition_owner_alt_det_min_ranker_margin >= 0.0:
                intervention_track_cmd.extend(
                    [
                        "--preassoc-stale-competition-owner-alt-det-min-ranker-margin",
                        str(args.preassoc_stale_competition_owner_alt_det_min_ranker_margin),
                    ]
                )
            if args.preassoc_stale_competition_owner_alt_det_min_edge_advantage >= 0.0:
                intervention_track_cmd.extend(
                    [
                        "--preassoc-stale-competition-owner-alt-det-min-edge-advantage",
                        str(args.preassoc_stale_competition_owner_alt_det_min_edge_advantage),
                    ]
                )
        if (
            args.preassoc_stale_competition_force_owner_edge_deficit_arg
            or args.preassoc_stale_competition_max_owner_edge_deficit >= 0.0
        ):
            intervention_track_cmd.extend(
                [
                    "--preassoc-stale-competition-max-owner-edge-deficit",
                    str(args.preassoc_stale_competition_max_owner_edge_deficit),
                ]
            )
        if args.preassoc_stale_competition_block_owner_on_reclaim:
            intervention_track_cmd.append("--preassoc-stale-competition-block-owner-on-reclaim")
        if args.preassoc_stale_competition_require_det_top1:
            intervention_track_cmd.append("--preassoc-stale-competition-require-det-top1")
        if args.preassoc_stale_competition_max_det_rank > 0:
            intervention_track_cmd.extend(
                [
                    "--preassoc-stale-competition-max-det-rank",
                    str(args.preassoc_stale_competition_max_det_rank),
                ]
            )
        if args.preassoc_stale_competition_force_rewrite_enable:
            intervention_track_cmd.extend(
                [
                    "--preassoc-stale-competition-force-rewrite-enable",
                    "--preassoc-stale-competition-force-rewrite-min-score",
                    str(args.preassoc_stale_competition_force_rewrite_min_score),
                    "--preassoc-stale-competition-force-rewrite-gate-weight",
                    str(args.preassoc_stale_competition_force_rewrite_gate_weight),
                    "--preassoc-stale-competition-force-rewrite-iou-weight",
                    str(args.preassoc_stale_competition_force_rewrite_iou_weight),
                    "--preassoc-stale-competition-force-rewrite-ranker-weight",
                    str(args.preassoc_stale_competition_force_rewrite_ranker_weight),
                    "--preassoc-stale-competition-force-rewrite-age-weight",
                    str(args.preassoc_stale_competition_force_rewrite_age_weight),
                    "--preassoc-stale-competition-force-rewrite-age-cap",
                    str(args.preassoc_stale_competition_force_rewrite_age_cap),
                    "--preassoc-stale-competition-force-rewrite-owner-alt-bonus",
                    str(args.preassoc_stale_competition_force_rewrite_owner_alt_bonus),
                    "--preassoc-stale-competition-force-rewrite-min-neighborhood-gain",
                    str(args.preassoc_stale_competition_force_rewrite_min_neighborhood_gain),
                    "--preassoc-stale-competition-force-rewrite-trapped-owner-min-neighborhood-gain",
                    str(args.preassoc_stale_competition_force_rewrite_trapped_owner_min_neighborhood_gain),
                    "--preassoc-stale-competition-force-rewrite-reroute-ready-min-neighborhood-gain",
                    str(args.preassoc_stale_competition_force_rewrite_reroute_ready_min_neighborhood_gain),
                    "--preassoc-stale-competition-force-rewrite-trapped-owner-negative-gain-min-challenger-alt-box-iou",
                    str(
                        args.preassoc_stale_competition_force_rewrite_trapped_owner_negative_gain_min_challenger_alt_box_iou
                    ),
                    "--preassoc-stale-competition-force-rewrite-neighborhood-keep-challenger-alt-weight",
                    str(args.preassoc_stale_competition_force_rewrite_neighborhood_keep_challenger_alt_weight),
                    "--preassoc-stale-competition-force-rewrite-neighborhood-rewrite-owner-alt-weight",
                    str(args.preassoc_stale_competition_force_rewrite_neighborhood_rewrite_owner_alt_weight),
                    "--preassoc-stale-competition-force-rewrite-neighborhood-shared-alt-penalty",
                    str(args.preassoc_stale_competition_force_rewrite_neighborhood_shared_alt_penalty),
                    "--preassoc-stale-competition-force-rewrite-neighborhood-trapped-owner-bonus",
                    str(args.preassoc_stale_competition_force_rewrite_neighborhood_trapped_owner_bonus),
                    "--preassoc-stale-competition-force-rewrite-neighborhood-reroute-ready-penalty",
                    str(args.preassoc_stale_competition_force_rewrite_neighborhood_reroute_ready_penalty),
                    "--preassoc-stale-competition-force-rewrite-min-box-iou",
                    str(args.preassoc_stale_competition_force_rewrite_min_box_iou),
                    "--preassoc-stale-competition-force-rewrite-max-age-gap",
                    str(args.preassoc_stale_competition_force_rewrite_max_age_gap),
                    "--preassoc-stale-competition-force-rewrite-max-owner-alt-det-box-iou",
                    str(args.preassoc_stale_competition_force_rewrite_max_owner_alt_det_box_iou),
                    "--preassoc-stale-competition-force-rewrite-stable-owner-min-hits",
                    str(args.preassoc_stale_competition_force_rewrite_stable_owner_min_hits),
                    "--preassoc-stale-competition-force-rewrite-stable-owner-min-raw-neighborhood-gain",
                    str(
                        args.preassoc_stale_competition_force_rewrite_stable_owner_min_raw_neighborhood_gain
                    ),
                    "--preassoc-stale-competition-force-rewrite-reroute-ready-min-box-iou",
                    str(args.preassoc_stale_competition_force_rewrite_reroute_ready_min_box_iou),
                ]
            )
            if args.preassoc_stale_competition_force_rewrite_owner_alt_soft_override_enable:
                intervention_track_cmd.extend(
                    [
                        "--preassoc-stale-competition-force-rewrite-owner-alt-soft-override-enable",
                        "--preassoc-stale-competition-force-rewrite-owner-alt-soft-override-max-owner-alt-det-box-iou",
                        str(
                            args.preassoc_stale_competition_force_rewrite_owner_alt_soft_override_max_owner_alt_det_box_iou
                        ),
                        "--preassoc-stale-competition-force-rewrite-owner-alt-soft-override-max-owner-hits",
                        str(args.preassoc_stale_competition_force_rewrite_owner_alt_soft_override_max_owner_hits),
                        "--preassoc-stale-competition-force-rewrite-owner-alt-soft-override-min-box-iou",
                        str(args.preassoc_stale_competition_force_rewrite_owner_alt_soft_override_min_box_iou),
                        "--preassoc-stale-competition-force-rewrite-owner-alt-soft-override-min-score",
                        str(args.preassoc_stale_competition_force_rewrite_owner_alt_soft_override_min_score),
                        "--preassoc-stale-competition-force-rewrite-owner-alt-soft-override-min-raw-neighborhood-gain",
                        str(
                            args.preassoc_stale_competition_force_rewrite_owner_alt_soft_override_min_raw_neighborhood_gain
                        ),
                    ]
                )
            if args.preassoc_stale_competition_force_rewrite_acceptance_nearmiss_owner_alt_override_enable:
                intervention_track_cmd.extend(
                    [
                        "--preassoc-stale-competition-force-rewrite-acceptance-nearmiss-owner-alt-override-enable",
                        "--preassoc-stale-competition-force-rewrite-acceptance-nearmiss-owner-alt-override-min-acceptance-score",
                        str(
                            args.preassoc_stale_competition_force_rewrite_acceptance_nearmiss_owner_alt_override_min_acceptance_score
                        ),
                        "--preassoc-stale-competition-force-rewrite-acceptance-nearmiss-owner-alt-override-max-owner-alt-det-box-iou",
                        str(
                            args.preassoc_stale_competition_force_rewrite_acceptance_nearmiss_owner_alt_override_max_owner_alt_det_box_iou
                        ),
                        "--preassoc-stale-competition-force-rewrite-acceptance-nearmiss-owner-alt-override-max-owner-hits",
                        str(
                            args.preassoc_stale_competition_force_rewrite_acceptance_nearmiss_owner_alt_override_max_owner_hits
                        ),
                        "--preassoc-stale-competition-force-rewrite-acceptance-nearmiss-owner-alt-override-min-box-iou",
                        str(
                            args.preassoc_stale_competition_force_rewrite_acceptance_nearmiss_owner_alt_override_min_box_iou
                        ),
                        "--preassoc-stale-competition-force-rewrite-acceptance-nearmiss-owner-alt-override-min-score",
                        str(
                            args.preassoc_stale_competition_force_rewrite_acceptance_nearmiss_owner_alt_override_min_score
                        ),
                        "--preassoc-stale-competition-force-rewrite-acceptance-nearmiss-owner-alt-override-min-raw-neighborhood-gain",
                        str(
                            args.preassoc_stale_competition_force_rewrite_acceptance_nearmiss_owner_alt_override_min_raw_neighborhood_gain
                        ),
                    ]
                )
            if args.preassoc_stale_competition_force_rewrite_recovery_anchor_override_enable:
                intervention_track_cmd.extend(
                    [
                        "--preassoc-stale-competition-force-rewrite-recovery-anchor-override-enable",
                        "--preassoc-stale-competition-force-rewrite-recovery-anchor-min-raw-neighborhood-gain",
                        str(args.preassoc_stale_competition_force_rewrite_recovery_anchor_min_raw_neighborhood_gain),
                    ]
                )
            if args.preassoc_stale_competition_force_rewrite_recovery_anchor_nearmiss_owner_alt_override_enable:
                intervention_track_cmd.extend(
                    [
                        "--preassoc-stale-competition-force-rewrite-recovery-anchor-nearmiss-owner-alt-override-enable",
                        "--preassoc-stale-competition-force-rewrite-recovery-anchor-nearmiss-owner-alt-override-min-recovery-anchor-score",
                        str(
                            args.preassoc_stale_competition_force_rewrite_recovery_anchor_nearmiss_owner_alt_override_min_recovery_anchor_score
                        ),
                        "--preassoc-stale-competition-force-rewrite-recovery-anchor-nearmiss-owner-alt-override-max-owner-alt-det-box-iou",
                        str(
                            args.preassoc_stale_competition_force_rewrite_recovery_anchor_nearmiss_owner_alt_override_max_owner_alt_det_box_iou
                        ),
                        "--preassoc-stale-competition-force-rewrite-recovery-anchor-nearmiss-owner-alt-override-max-owner-hits",
                        str(
                            args.preassoc_stale_competition_force_rewrite_recovery_anchor_nearmiss_owner_alt_override_max_owner_hits
                        ),
                        "--preassoc-stale-competition-force-rewrite-recovery-anchor-nearmiss-owner-alt-override-min-box-iou",
                        str(
                            args.preassoc_stale_competition_force_rewrite_recovery_anchor_nearmiss_owner_alt_override_min_box_iou
                        ),
                        "--preassoc-stale-competition-force-rewrite-recovery-anchor-nearmiss-owner-alt-override-min-score",
                        str(
                            args.preassoc_stale_competition_force_rewrite_recovery_anchor_nearmiss_owner_alt_override_min_score
                        ),
                        "--preassoc-stale-competition-force-rewrite-recovery-anchor-nearmiss-owner-alt-override-min-raw-neighborhood-gain",
                        str(
                            args.preassoc_stale_competition_force_rewrite_recovery_anchor_nearmiss_owner_alt_override_min_raw_neighborhood_gain
                        ),
                    ]
                )
            if args.preassoc_stale_competition_force_rewrite_neighborhood_enable:
                intervention_track_cmd.append("--preassoc-stale-competition-force-rewrite-neighborhood-enable")
            if args.preassoc_stale_competition_force_rewrite_recovery_memory_enable:
                intervention_track_cmd.extend(
                    [
                        "--preassoc-stale-competition-force-rewrite-recovery-memory-enable",
                        "--preassoc-stale-competition-force-rewrite-recovery-memory-max-frame-gap",
                        str(args.preassoc_stale_competition_force_rewrite_recovery_memory_max_frame_gap),
                        "--preassoc-stale-competition-force-rewrite-recovery-memory-min-score",
                        str(args.preassoc_stale_competition_force_rewrite_recovery_memory_min_score),
                        "--preassoc-stale-competition-force-rewrite-recovery-memory-min-box-iou",
                        str(args.preassoc_stale_competition_force_rewrite_recovery_memory_min_box_iou),
                        "--preassoc-stale-competition-force-rewrite-recovery-memory-min-challenger-alt-box-iou",
                        str(args.preassoc_stale_competition_force_rewrite_recovery_memory_min_challenger_alt_box_iou),
                        "--preassoc-stale-competition-force-rewrite-recovery-memory-warmup-min-neighborhood-gain",
                        str(
                            args.preassoc_stale_competition_force_rewrite_recovery_memory_warmup_min_neighborhood_gain
                        ),
                        "--preassoc-stale-competition-force-rewrite-recovery-memory-bonus",
                        str(args.preassoc_stale_competition_force_rewrite_recovery_memory_bonus),
                        "--preassoc-stale-competition-force-rewrite-recovery-memory-bonus-max-streak",
                        str(args.preassoc_stale_competition_force_rewrite_recovery_memory_bonus_max_streak),
                        "--preassoc-stale-competition-force-rewrite-recovery-memory-gate-bonus",
                        str(args.preassoc_stale_competition_force_rewrite_recovery_memory_gate_bonus),
                        "--preassoc-stale-competition-force-rewrite-recovery-memory-anchor-min-raw-neighborhood-gain",
                        str(
                            args.preassoc_stale_competition_force_rewrite_recovery_memory_anchor_min_raw_neighborhood_gain
                        ),
                        "--preassoc-stale-competition-force-rewrite-recovery-memory-anchor-max-edge-deficit",
                        str(args.preassoc_stale_competition_force_rewrite_recovery_memory_anchor_max_edge_deficit),
                        "--preassoc-stale-competition-force-rewrite-recovery-memory-extension-max-edge-deficit-delta",
                        str(
                            args.preassoc_stale_competition_force_rewrite_recovery_memory_extension_max_edge_deficit_delta
                        ),
                    ]
                )
                if args.preassoc_stale_competition_force_rewrite_recovery_memory_anchor_enable:
                    intervention_track_cmd.append(
                        "--preassoc-stale-competition-force-rewrite-recovery-memory-anchor-enable"
                    )
            intervention_track_cmd.extend(
                [
                    "--preassoc-stale-competition-force-rewrite-high-conf-reclaim-min-score",
                    str(args.preassoc_stale_competition_force_rewrite_high_conf_reclaim_min_score),
                    "--preassoc-stale-competition-force-rewrite-high-conf-reclaim-min-box-iou",
                    str(args.preassoc_stale_competition_force_rewrite_high_conf_reclaim_min_box_iou),
                    "--preassoc-stale-competition-force-rewrite-high-conf-reclaim-min-ranker-score",
                    str(args.preassoc_stale_competition_force_rewrite_high_conf_reclaim_min_ranker_score),
                    "--preassoc-stale-competition-force-rewrite-high-conf-reclaim-min-acceptance-score",
                    str(args.preassoc_stale_competition_force_rewrite_high_conf_reclaim_min_acceptance_score),
                    "--preassoc-stale-competition-force-rewrite-high-conf-reclaim-min-track-hits",
                    str(args.preassoc_stale_competition_force_rewrite_high_conf_reclaim_min_track_hits),
                    "--preassoc-stale-competition-force-rewrite-high-conf-reclaim-max-owner-hits",
                    str(args.preassoc_stale_competition_force_rewrite_high_conf_reclaim_max_owner_hits),
                    "--preassoc-stale-competition-force-rewrite-high-conf-reclaim-max-owner-alt-box-iou",
                    str(args.preassoc_stale_competition_force_rewrite_high_conf_reclaim_max_owner_alt_box_iou),
                    "--preassoc-stale-competition-force-rewrite-high-conf-reclaim-min-challenger-alt-box-iou",
                    str(args.preassoc_stale_competition_force_rewrite_high_conf_reclaim_min_challenger_alt_box_iou),
                    "--preassoc-stale-competition-force-rewrite-high-conf-reclaim-min-edge-advantage",
                    str(args.preassoc_stale_competition_force_rewrite_high_conf_reclaim_min_edge_advantage),
                    "--preassoc-stale-competition-force-rewrite-high-conf-reclaim-min-neighborhood-gain",
                    str(args.preassoc_stale_competition_force_rewrite_high_conf_reclaim_min_neighborhood_gain),
                    "--preassoc-stale-competition-acceptance-gate-high-conf-reclaim-min-score",
                    str(args.preassoc_stale_competition_acceptance_gate_high_conf_reclaim_min_score),
                    "--preassoc-stale-competition-acceptance-gate-high-conf-reclaim-min-recovery-anchor-score",
                    str(args.preassoc_stale_competition_acceptance_gate_high_conf_reclaim_min_recovery_anchor_score),
                ]
            )
        if args.preassoc_stale_competition_export_jsonl:
            export_jsonl_path = Path(str(args.preassoc_stale_competition_export_jsonl)).expanduser()
            if not export_jsonl_path.is_absolute():
                export_jsonl_path = (REPO_ROOT / export_jsonl_path).resolve()
            intervention_track_cmd.extend(["--preassoc-stale-competition-export-jsonl", str(export_jsonl_path)])
    if args.local_contention_export_jsonl:
        local_export_jsonl_path = Path(str(args.local_contention_export_jsonl)).expanduser()
        if not local_export_jsonl_path.is_absolute():
            local_export_jsonl_path = (REPO_ROOT / local_export_jsonl_path).resolve()
        intervention_track_cmd.extend(
            [
                "--local-contention-export-jsonl",
                str(local_export_jsonl_path),
                "--local-contention-topk",
                str(args.local_contention_topk),
                "--local-contention-min-box-iou",
                str(args.local_contention_min_box_iou),
                "--local-contention-max-time-since-update",
                str(args.local_contention_max_time_since_update),
                "--local-contention-min-challenger-hits",
                str(args.local_contention_min_challenger_hits),
                "--local-contention-owner-weak-hits",
                str(args.local_contention_owner_weak_hits),
            ]
        )
    if args.local_contention_ranker_checkpoint:
        ranker_checkpoint_path = Path(str(args.local_contention_ranker_checkpoint)).expanduser()
        if not ranker_checkpoint_path.is_absolute():
            ranker_checkpoint_path = (REPO_ROOT / ranker_checkpoint_path).resolve()
        intervention_track_cmd.extend(
            [
                "--local-contention-ranker-checkpoint",
                str(ranker_checkpoint_path),
                "--local-contention-ranker-thresh",
                str(args.local_contention_ranker_thresh),
                "--local-contention-ranker-bias",
                str(args.local_contention_ranker_bias),
                "--local-contention-ranker-min-margin-to-second",
                str(args.local_contention_ranker_min_margin_to_second),
                "--local-contention-ranker-margin-bias",
                str(args.local_contention_ranker_margin_bias),
            ]
        )
    if args.preassoc_stale_competition_acceptance_gate_checkpoint:
        gate_checkpoint_path = Path(str(args.preassoc_stale_competition_acceptance_gate_checkpoint)).expanduser()
        if not gate_checkpoint_path.is_absolute():
            gate_checkpoint_path = (REPO_ROOT / gate_checkpoint_path).resolve()
        intervention_track_cmd.extend(
            [
                "--preassoc-stale-competition-acceptance-gate-checkpoint",
                str(gate_checkpoint_path),
                "--preassoc-stale-competition-acceptance-gate-thresh",
                str(args.preassoc_stale_competition_acceptance_gate_thresh),
            ]
        )
    if args.preassoc_stale_competition_recovery_anchor_gate_checkpoint:
        recovery_anchor_gate_checkpoint_path = Path(
            str(args.preassoc_stale_competition_recovery_anchor_gate_checkpoint)
        ).expanduser()
        if not recovery_anchor_gate_checkpoint_path.is_absolute():
            recovery_anchor_gate_checkpoint_path = (REPO_ROOT / recovery_anchor_gate_checkpoint_path).resolve()
        intervention_track_cmd.extend(
            [
                "--preassoc-stale-competition-recovery-anchor-gate-checkpoint",
                str(recovery_anchor_gate_checkpoint_path),
                "--preassoc-stale-competition-recovery-anchor-gate-thresh",
                str(args.preassoc_stale_competition_recovery_anchor_gate_thresh),
            ]
        )
    return intervention_track_cmd


def run_intervention_tracking(
    *,
    args: argparse.Namespace,
    benchmark: str,
    dataset_root: Path,
    dataset_name: str,
    tracker_split: str,
    seq_names: List[str],
    run_root: Path,
    logs_dir: Path,
    trackers_root: Path,
    intervention_exp: str,
    intervention_track_out: Path,
    intervention_eval_out: Path,
) -> Path | None:
    intervention_track_log = logs_dir / "competition_track.log"
    batch_frame_budget = int(args.competition_track_max_frames_per_batch or 0)
    if batch_frame_budget <= 0:
        intervention_track_cmd = build_intervention_track_cmd(
            args=args,
            dataset_name=dataset_name,
            trackers_root=trackers_root,
            exp_name=intervention_exp,
            seq_names=seq_names,
        )
        return_code = run_step(intervention_track_cmd, intervention_track_log, cwd=DEEP_ROOT, enable_orion_keepalive=True)
        if return_code == 0:
            ensure_tracking_outputs(intervention_track_out, intervention_eval_out, seq_names)
        if return_code != 0:
            raise RuntimeError(f"competition tracking return_code={return_code}")
        return resolve_runtime_summary(intervention_track_out)

    batch_seq_groups = split_seq_names_by_frame_budget(dataset_root, seq_names, batch_frame_budget)
    intervention_track_log.parent.mkdir(parents=True, exist_ok=True)
    if intervention_track_log.is_file() and intervention_track_log.stat().st_size > 0:
        append_text(intervention_track_log, f"[resumed_at] {now_iso()}\n")
    else:
        intervention_track_log.write_text(
            "\n".join(
                [
                    f"[started_at] {now_iso()}",
                    f"[mode] chunked competition tracking",
                    f"[frame_budget_per_batch] {batch_frame_budget}",
                    *[
                        f"[batch_{idx:02d}] {'|'.join(batch_seqs)}"
                        for idx, batch_seqs in enumerate(batch_seq_groups, start=1)
                    ],
                    "",
                ]
            ),
            encoding="utf-8",
        )
    intervention_track_out.mkdir(parents=True, exist_ok=True)
    intervention_eval_out.mkdir(parents=True, exist_ok=True)
    merged_runtime_summaries: List[Path] = []
    merged_local_jsonls: List[Path] = []
    for batch_idx, batch_seqs in enumerate(batch_seq_groups, start=1):
        batch_root = run_root / "competition_batch_runs" / f"batch_{batch_idx:02d}"
        batch_trackers_root = batch_root / "results" / "trackers"
        batch_exp = f"{intervention_exp}_batch{batch_idx:02d}"
        batch_log = logs_dir / f"competition_track_batch_{batch_idx:02d}.log"
        batch_track_out = batch_trackers_root / tracker_split / batch_exp
        batch_eval_out = batch_trackers_root / tracker_split / (batch_exp + "_post")
        batch_local_jsonl = batch_root / "local_contention_units.jsonl"
        batch_ready = False
        if batch_root.exists():
            try:
                ensure_tracking_outputs(batch_track_out, batch_eval_out, batch_seqs)
                batch_ready = True
            except FileNotFoundError:
                shutil.rmtree(batch_root)
        if not batch_ready:
            batch_args = argparse.Namespace(**vars(args))
            if args.local_contention_export_jsonl:
                batch_args.local_contention_export_jsonl = str(batch_local_jsonl)
            batch_cmd = build_intervention_track_cmd(
                args=batch_args,
                dataset_name=dataset_name,
                trackers_root=batch_trackers_root,
                exp_name=batch_exp,
                seq_names=batch_seqs,
            )
            append_text(
                intervention_track_log,
                f"[batch_{batch_idx:02d}_started_at] {now_iso()} seqs={'|'.join(batch_seqs)} log={batch_log}\n",
            )
            return_code = run_step(batch_cmd, batch_log, cwd=DEEP_ROOT, enable_orion_keepalive=True)
            if return_code != 0:
                raise RuntimeError(f"competition tracking batch {batch_idx} return_code={return_code}")
            ensure_tracking_outputs(batch_track_out, batch_eval_out, batch_seqs)
            append_text(
                intervention_track_log,
                f"[batch_{batch_idx:02d}_finished_at] {now_iso()} status=success seqs={'|'.join(batch_seqs)}\n",
            )
        else:
            append_text(
                intervention_track_log,
                f"[batch_{batch_idx:02d}_reused_at] {now_iso()} status=success seqs={'|'.join(batch_seqs)} log={batch_log}\n",
            )
        copy_sequence_outputs(batch_track_out, intervention_track_out, batch_seqs)
        copy_sequence_outputs(batch_eval_out, intervention_eval_out, batch_seqs)
        batch_runtime_summary = resolve_runtime_summary(batch_track_out)
        if batch_runtime_summary is not None:
            merged_runtime_summaries.append(batch_runtime_summary)
        if args.local_contention_export_jsonl:
            if batch_local_jsonl.is_file():
                merged_local_jsonls.append(batch_local_jsonl)
    if merged_runtime_summaries:
        merged_runtime_summary = intervention_track_out / "fgas_analysis" / f"{intervention_track_out.name}_summary.csv"
        merge_runtime_summaries(merged_runtime_summaries, merged_runtime_summary)
    if args.local_contention_export_jsonl:
        merged_local_export = Path(str(args.local_contention_export_jsonl)).expanduser()
        if not merged_local_export.is_absolute():
            merged_local_export = (REPO_ROOT / merged_local_export).resolve()
        if merged_local_export.exists():
            merged_local_export.unlink()
        concatenate_jsonl(merged_local_jsonls, merged_local_export)
    ensure_tracking_outputs(intervention_track_out, intervention_eval_out, seq_names)
    append_text(intervention_track_log, f"[finished_at] {now_iso()}\n[status] success\n")
    return resolve_runtime_summary(intervention_track_out)


def main() -> None:
    args = parse_args()
    benchmark = str(args.benchmark)
    dataset_name = {
        "MOT17": "mot17",
        "MOT20": "mot20",
        "DanceTrack": "dance",
    }[benchmark]
    eval_benchmark = {
        "MOT17": "MOT17",
        "MOT20": "MOT20",
        "DanceTrack": "DANCE",
    }[benchmark]
    split_label = "val" if benchmark == "DanceTrack" else "val_half"
    tracker_split = f"{eval_benchmark}-val"
    dataset_root = ensure_dataset_link(benchmark)
    ensure_reid_weights(benchmark)
    ensure_detector_weights(benchmark)
    seq_names = resolve_seq_names(args)
    seq_label = seq_note(seq_names)
    run_root = (
        Path(args.out_root)
        if args.out_root
        else REPO_ROOT / "outputs" / f"deep_ocsort_preassoc_competition_{dataset_name}_val_{timestamp_tag()}"
    ).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    logs_dir = run_root / "logs"
    trackers_root = (run_root / "results" / "trackers").resolve()
    metrics_compare_csv = run_root / "metrics_compare.csv"
    metrics_delta_csv = run_root / "metrics_delta.csv"
    per_sequence_csv = run_root / "per_sequence_metrics.csv"
    runtime_compare_csv = run_root / "runtime_compare.csv"
    runtime_per_sequence_csv = run_root / "runtime_per_sequence.csv"
    local_contention_summary_csv = run_root / "local_contention_summary.csv"
    summary_csv = run_root / "summary.csv"
    local_export_jsonl_path: Path | None = None
    if args.local_contention_export_jsonl:
        local_export_jsonl_path = Path(str(args.local_contention_export_jsonl)).expanduser()
        if not local_export_jsonl_path.is_absolute():
            local_export_jsonl_path = (REPO_ROOT / local_export_jsonl_path).resolve()

    reuse_raw_root = Path(str(args.reuse_raw_from)).resolve() if args.reuse_raw_from else None
    raw_exp = f"{run_root.name}_raw"
    intervention_exp = f"{run_root.name}_competition"

    raw_reuse_post_out: Path | None = None
    if reuse_raw_root is not None:
        raw_exp = f"{reuse_raw_root.name}_raw"
        raw_track_out = reuse_raw_root / "results" / "trackers" / tracker_split / raw_exp
        raw_reuse_post_out = reuse_raw_root / "results" / "trackers" / tracker_split / (raw_exp + "_post")
        raw_eval_out = trackers_root / tracker_split / (raw_exp + "_post")
    else:
        raw_track_out = trackers_root / tracker_split / raw_exp
        raw_eval_out = trackers_root / tracker_split / (raw_exp + "_post")
    intervention_track_out = trackers_root / tracker_split / intervention_exp
    intervention_eval_out = trackers_root / tracker_split / (intervention_exp + "_post")

    rows: List[Dict[str, object]] = [
        {
            "step": "raw_track",
            "name": raw_exp,
            "status": "success" if reuse_raw_root is not None else "running",
            "out_dir": str(raw_track_out) if reuse_raw_root is not None else "",
            "summary_csv": str(summary_csv),
            "log_path": str(logs_dir / "raw_track.log"),
            "started_at": now_iso() if reuse_raw_root is None else "",
            "finished_at": now_iso() if reuse_raw_root is not None else "",
            "notes": f"reuse raw tracking from {reuse_raw_root}" if reuse_raw_root is not None else f"Deep-OC-SORT raw tracking on {benchmark} {seq_label}",
        },
        {
            "step": "raw_eval",
            "name": raw_exp,
            "status": "pending",
            "out_dir": "",
            "summary_csv": str(summary_csv),
            "log_path": str(logs_dir / "raw_eval.log"),
            "started_at": "",
            "finished_at": "",
            "notes": (
                f"subset raw eval using reused tracking from {reuse_raw_root}"
                if reuse_raw_root is not None
                else f"TrackEval for {raw_exp}"
            ),
        },
        {
            "step": "competition_track",
            "name": intervention_exp,
            "status": "pending",
            "out_dir": "",
            "summary_csv": str(summary_csv),
            "log_path": str(logs_dir / "competition_track.log"),
            "started_at": "",
            "finished_at": "",
            "notes": f"Deep-OC-SORT preassoc competition tracking on {benchmark} {seq_label}",
        },
        {
            "step": "competition_eval",
            "name": intervention_exp,
            "status": "pending",
            "out_dir": "",
            "summary_csv": str(summary_csv),
            "log_path": str(logs_dir / "competition_eval.log"),
            "started_at": "",
            "finished_at": "",
            "notes": f"TrackEval for {intervention_exp}",
        },
        {
            "step": "compare",
            "name": run_root.name,
            "status": "pending",
            "out_dir": str(run_root),
            "summary_csv": str(summary_csv),
            "log_path": str(logs_dir / "compare.log"),
            "started_at": "",
            "finished_at": "",
            "notes": f"Compare raw vs preassoc competition on {benchmark} {seq_label}",
        },
    ]
    write_rows(summary_csv, QUEUE_FIELDS, rows)
    append_registry(summary_csv, run_root, "running", f"started paired preassoc competition eval on {benchmark} {seq_label}", args.registry_csv, benchmark, split_label)
    try:
        if reuse_raw_root is None:
            raw_track_cmd = [
                sys.executable,
                "main.py",
                "--dataset",
                dataset_name,
                "--result_folder",
                str(trackers_root),
                "--exp_name",
                raw_exp,
                "--seq-filter",
                *seq_names,
                "--post",
                "--grid_off",
                "--new_kf_off",
            ]
            extend_dataset_tracking_profile(raw_track_cmd, dataset_name)
            raw_track_log = logs_dir / "raw_track.log"
            return_code = run_step(raw_track_cmd, raw_track_log, cwd=DEEP_ROOT, enable_orion_keepalive=True)
            if return_code == 0:
                ensure_tracking_outputs(raw_track_out, raw_eval_out, seq_names)
            ensure_success("raw_track", return_code, rows, summary_csv, raw_track_out, raw_track_log, f"raw tracking complete for {seq_label}")

            update_row(rows, "raw_eval", status="running", started_at=now_iso())
            write_rows(summary_csv, QUEUE_FIELDS, rows)
            raw_eval_cmd = [
                sys.executable,
                "external/TrackEval/scripts/run_mot_challenge.py",
                "--BENCHMARK",
                eval_benchmark,
                "--SPLIT_TO_EVAL",
                "val",
                "--GT_FOLDER",
                str(DEEP_ROOT / "results" / "gt"),
                "--TRACKERS_FOLDER",
                str(trackers_root),
                "--TRACKERS_TO_EVAL",
                raw_exp + "_post",
                "--SEQ_INFO",
                *seq_names,
                "--METRICS",
                "HOTA",
                "CLEAR",
                "Identity",
                "--USE_PARALLEL",
                "False",
                "--PRINT_ONLY_COMBINED",
                "True",
            ]
            raw_eval_log = logs_dir / "raw_eval.log"
            return_code = run_step(raw_eval_cmd, raw_eval_log, cwd=DEEP_ROOT)
            if return_code == 0:
                ensure_eval_outputs(raw_eval_out)
            ensure_success("raw_eval", return_code, rows, summary_csv, raw_eval_out, raw_eval_log, f"raw eval complete for {seq_label}")
        else:
            if not raw_track_out.is_dir():
                raise FileNotFoundError(f"Missing raw tracker dir: {raw_track_out}")
            if raw_reuse_post_out is None or not raw_reuse_post_out.is_dir():
                raise FileNotFoundError(f"Missing reused raw post dir: {raw_reuse_post_out}")
            ensure_tracking_outputs(raw_track_out, raw_reuse_post_out, seq_names)
            update_row(rows, "raw_eval", status="running", started_at=now_iso())
            write_rows(summary_csv, QUEUE_FIELDS, rows)
            prepare_reused_eval_dir(raw_reuse_post_out, raw_eval_out)
            raw_eval_cmd = [
                sys.executable,
                "external/TrackEval/scripts/run_mot_challenge.py",
                "--BENCHMARK",
                eval_benchmark,
                "--SPLIT_TO_EVAL",
                "val",
                "--GT_FOLDER",
                str(DEEP_ROOT / "results" / "gt"),
                "--TRACKERS_FOLDER",
                str(trackers_root),
                "--TRACKERS_TO_EVAL",
                raw_exp + "_post",
                "--SEQ_INFO",
                *seq_names,
                "--METRICS",
                "HOTA",
                "CLEAR",
                "Identity",
                "--USE_PARALLEL",
                "False",
                "--PRINT_ONLY_COMBINED",
                "True",
            ]
            raw_eval_log = logs_dir / "raw_eval.log"
            return_code = run_step(raw_eval_cmd, raw_eval_log, cwd=DEEP_ROOT)
            if return_code == 0:
                ensure_tracking_outputs(raw_track_out, raw_eval_out, seq_names)
                ensure_eval_outputs(raw_eval_out)
            ensure_success(
                "raw_eval",
                return_code,
                rows,
                summary_csv,
                raw_eval_out,
                raw_eval_log,
                f"subset raw eval complete for {seq_label} using reused tracking",
            )
            write_rows(summary_csv, QUEUE_FIELDS, rows)

        update_row(rows, "competition_track", status="running", started_at=now_iso())
        write_rows(summary_csv, QUEUE_FIELDS, rows)
        intervention_track_log = logs_dir / "competition_track.log"
        intervention_runtime_summary = run_intervention_tracking(
            args=args,
            benchmark=benchmark,
            dataset_root=dataset_root,
            dataset_name=dataset_name,
            tracker_split=tracker_split,
            seq_names=seq_names,
            run_root=run_root,
            logs_dir=logs_dir,
            trackers_root=trackers_root,
            intervention_exp=intervention_exp,
            intervention_track_out=intervention_track_out,
            intervention_eval_out=intervention_eval_out,
        )
        ensure_success(
            "competition_track",
            0,
            rows,
            summary_csv,
            intervention_track_out,
            intervention_track_log,
            f"preassoc competition tracking complete for {seq_label}",
        )

        update_row(rows, "competition_eval", status="running", started_at=now_iso())
        write_rows(summary_csv, QUEUE_FIELDS, rows)
        intervention_eval_cmd = [
            sys.executable,
            "external/TrackEval/scripts/run_mot_challenge.py",
            "--BENCHMARK",
            eval_benchmark,
            "--SPLIT_TO_EVAL",
            "val",
            "--GT_FOLDER",
            str(DEEP_ROOT / "results" / "gt"),
            "--TRACKERS_FOLDER",
            str(trackers_root),
            "--TRACKERS_TO_EVAL",
            intervention_exp + "_post",
            "--SEQ_INFO",
            *seq_names,
            "--METRICS",
            "HOTA",
            "CLEAR",
            "Identity",
            "--USE_PARALLEL",
            "False",
            "--PRINT_ONLY_COMBINED",
            "True",
        ]
        intervention_eval_log = logs_dir / "competition_eval.log"
        return_code = run_step(intervention_eval_cmd, intervention_eval_log, cwd=DEEP_ROOT)
        if return_code == 0:
            ensure_eval_outputs(intervention_eval_out)
        ensure_success("competition_eval", return_code, rows, summary_csv, intervention_eval_out, intervention_eval_log, f"preassoc competition eval complete for {seq_label}")

        update_row(rows, "compare", status="running", started_at=now_iso())
        write_rows(summary_csv, QUEUE_FIELDS, rows)

        raw_summary_txt = raw_eval_out / "pedestrian_summary.txt"
        raw_detailed_csv = raw_eval_out / "pedestrian_detailed.csv"
        intervention_summary_txt = intervention_eval_out / "pedestrian_summary.txt"
        intervention_detailed_csv = intervention_eval_out / "pedestrian_detailed.csv"
        raw_metrics = parse_summary_txt(raw_summary_txt)
        intervention_metrics = parse_summary_txt(intervention_summary_txt)
        per_sequence_rows = load_per_sequence_metrics(raw_detailed_csv, "raw") + load_per_sequence_metrics(intervention_detailed_csv, "competition")
        compare_rows = [
            {
                "name": "raw",
                "seq": seq_label,
                "HOTA": raw_metrics.get("HOTA", ""),
                "AssA": raw_metrics.get("AssA", ""),
                "IDF1": raw_metrics.get("IDF1", ""),
                "MOTA": raw_metrics.get("MOTA", ""),
                "IDs": raw_metrics.get("IDs", ""),
                "Frag": raw_metrics.get("Frag", ""),
                "summary_txt": str(raw_summary_txt),
                "detailed_csv": str(raw_detailed_csv),
                "tracker_dir": str(raw_track_out),
            },
            {
                "name": "competition",
                "seq": seq_label,
                "HOTA": intervention_metrics.get("HOTA", ""),
                "AssA": intervention_metrics.get("AssA", ""),
                "IDF1": intervention_metrics.get("IDF1", ""),
                "MOTA": intervention_metrics.get("MOTA", ""),
                "IDs": intervention_metrics.get("IDs", ""),
                "Frag": intervention_metrics.get("Frag", ""),
                "summary_txt": str(intervention_summary_txt),
                "detailed_csv": str(intervention_detailed_csv),
                "tracker_dir": str(intervention_track_out),
            },
        ]
        delta_rows = [
            {
                "name": "competition_minus_raw",
                "seq": seq_label,
                "delta_HOTA": float(intervention_metrics.get("HOTA", 0.0)) - float(raw_metrics.get("HOTA", 0.0)),
                "delta_AssA": float(intervention_metrics.get("AssA", 0.0)) - float(raw_metrics.get("AssA", 0.0)),
                "delta_IDF1": float(intervention_metrics.get("IDF1", 0.0)) - float(raw_metrics.get("IDF1", 0.0)),
                "delta_MOTA": float(intervention_metrics.get("MOTA", 0.0)) - float(raw_metrics.get("MOTA", 0.0)),
                "delta_IDs": float(intervention_metrics.get("IDs", 0.0)) - float(raw_metrics.get("IDs", 0.0)),
                "delta_Frag": float(intervention_metrics.get("Frag", 0.0)) - float(raw_metrics.get("Frag", 0.0)),
            }
        ]
        write_rows(metrics_compare_csv, METRIC_FIELDS, compare_rows)
        write_rows(metrics_delta_csv, DELTA_FIELDS, delta_rows)
        write_rows(per_sequence_csv, PER_SEQUENCE_FIELDS, per_sequence_rows)

        raw_runtime_summary = resolve_runtime_summary(raw_track_out)
        if intervention_runtime_summary is None:
            intervention_runtime_summary = resolve_runtime_summary(intervention_track_out)
        raw_runtime_rows = load_runtime_rows(raw_runtime_summary, "raw") if raw_runtime_summary is not None else []
        intervention_runtime_rows = load_runtime_rows(intervention_runtime_summary, "competition") if intervention_runtime_summary is not None else []
        runtime_compare_rows = [
            summarize_runtime_rows(runtime_rows=raw_runtime_rows, label="raw", seq_label=seq_label, summary_csv=raw_runtime_summary),
            summarize_runtime_rows(runtime_rows=intervention_runtime_rows, label="competition", seq_label=seq_label, summary_csv=intervention_runtime_summary),
        ]
        write_rows(runtime_compare_csv, RUNTIME_FIELDS, runtime_compare_rows)
        write_rows(runtime_per_sequence_csv, RUNTIME_PER_SEQUENCE_FIELDS, raw_runtime_rows + intervention_runtime_rows)
        if args.local_contention_export_jsonl:
            write_rows(
                local_contention_summary_csv,
                LOCAL_CONTENTION_SUMMARY_FIELDS,
                [summarize_local_contention_export(local_export_jsonl_path, "competition")],
            )

        compare_log = logs_dir / "compare.log"
        compare_log.write_text(
            "\n".join(
                [
                    f"raw_summary={raw_summary_txt}",
                    f"competition_summary={intervention_summary_txt}",
                    f"metrics_compare={metrics_compare_csv}",
                    f"metrics_delta={metrics_delta_csv}",
                    f"per_sequence_metrics={per_sequence_csv}",
                    f"runtime_compare={runtime_compare_csv}",
                    f"runtime_per_sequence={runtime_per_sequence_csv}",
                    (
                        f"local_contention_summary={local_contention_summary_csv}"
                        if args.local_contention_export_jsonl
                        else ""
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        update_row(rows, "compare", status="success", finished_at=now_iso(), out_dir=str(run_root), summary_csv=str(summary_csv), log_path=str(compare_log), notes=f"compare complete for {seq_label}")
        write_rows(summary_csv, QUEUE_FIELDS, rows)
        append_registry(summary_csv, run_root, "success", f"completed paired preassoc competition eval on {benchmark} {seq_label}", args.registry_csv, benchmark, split_label)
    except Exception as exc:
        mark_running_rows_failed(rows, summary_csv, str(exc))
        append_registry(summary_csv, run_root, "failed", f"paired preassoc competition eval failed on {benchmark} {seq_label}: {exc}", args.registry_csv, benchmark, split_label)
        raise


if __name__ == "__main__":
    main()
