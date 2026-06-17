#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

REPO_ROOT = Path("/gemini/code/FMtrack-main/FM-Track")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.posthost_one_edit_scorer import CANDIDATE_FEATURE_NAMES, feature_dim


REGISTRY_CSV = REPO_ROOT / "outputs" / "experiment_registry.csv"


SUMMARY_FIELDS = [
    "dataset_name",
    "source_cluster_jsonl",
    "candidate_min_refined_score",
    "host_summary_mode",
    "host_summary_prior_alpha",
    "swap_utility_bonus",
    "add_utility_bonus",
    "defer_utility_penalty",
    "clusters",
    "train_clusters",
    "val_clusters",
    "target_keep_clusters",
    "target_add_clusters",
    "target_swap_clusters",
    "target_defer_clusters",
    "target_nonkeep_clusters",
    "positive_utility_clusters",
    "mean_candidates_per_cluster",
    "max_candidates_per_cluster",
    "mean_nonkeep_candidates_per_cluster",
    "max_nonkeep_candidates_per_cluster",
    "mean_best_positive_nonkeep_adjusted_utility",
    "mean_positive_nonkeep_utility_margin_to_second",
    "status",
    "error",
]

SPLIT_FIELDS = [
    "split_tag",
    "clusters",
    "target_keep_clusters",
    "target_add_clusters",
    "target_swap_clusters",
    "target_defer_clusters",
    "target_nonkeep_clusters",
    "positive_utility_clusters",
    "mean_candidates_per_cluster",
    "max_candidates_per_cluster",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build post-host one-edit action-scoring dataset from existing official cluster_examples.jsonl."
    )
    parser.add_argument("--cluster-jsonl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dataset-name", default="official_bytetrack_posthost_one_edit_dataset")
    parser.add_argument("--candidate-min-refined-score", type=float, default=0.10)
    parser.add_argument(
        "--host-summary-mode",
        choices=["gt_oracle", "runtime_safe_zero", "runtime_prior_alpha"],
        default="runtime_safe_zero",
    )
    parser.add_argument("--host-summary-prior-alpha", type=float, default=0.0)
    parser.add_argument("--swap-utility-bonus", type=float, default=0.0)
    parser.add_argument("--add-utility-bonus", type=float, default=0.0)
    parser.add_argument("--defer-utility-penalty", type=float, default=0.0)
    parser.add_argument("--registry-csv", default=str(REGISTRY_CSV))
    return parser.parse_args()


def write_rows(path: Path, fieldnames: List[str], rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def score_pair_set(pair_set: set[tuple[int, int]], gt_by_pair: Dict[tuple[int, int], int]) -> Dict[str, float]:
    correct = 0
    wrong = 0
    for pair in pair_set:
        if int(gt_by_pair.get((int(pair[0]), int(pair[1])), 0)) > 0:
            correct += 1
        else:
            wrong += 1
    return {
        "correct": float(correct),
        "wrong": float(wrong),
        "score": float(correct - wrong),
    }


def zero_pair_stats() -> Dict[str, float]:
    return {
        "refined_score": 0.0,
        "base_score": 0.0,
        "iou": 0.0,
        "bbox_dist": 0.0,
        "row_degree": 0.0,
        "row_margin": 0.0,
        "row_entropy": 0.0,
        "col_degree": 0.0,
        "track_gap": 0.0,
        "track_hist": 0.0,
        "delta_cx": 0.0,
        "delta_cy": 0.0,
        "delta_log_w": 0.0,
        "delta_log_h": 0.0,
    }


def pair_stats(
    sample: Dict[str, Any],
    pair_lookup: Dict[tuple[int, int], Dict[str, float]],
    pair: tuple[int, int] | None,
) -> Dict[str, float]:
    if pair is None:
        return zero_pair_stats()
    det_idx = int(pair[0])
    track_idx = int(pair[1])
    det_features = sample["det_features"][det_idx]
    track_features = sample["track_features"][track_idx]
    edge = pair_lookup[(det_idx, track_idx)]
    return {
        "refined_score": float(edge["refined_score"]),
        "base_score": float(edge["base_score"]),
        "iou": float(edge["iou"]),
        "bbox_dist": float(edge["bbox_dist"]),
        "row_degree": float(det_features[1]),
        "row_margin": float(det_features[2]),
        "row_entropy": float(det_features[3]),
        "col_degree": float(track_features[2]),
        "track_gap": float(track_features[0]),
        "track_hist": float(track_features[1]),
        "delta_cx": float(edge["delta_cx"]),
        "delta_cy": float(edge["delta_cy"]),
        "delta_log_w": float(edge["delta_log_w"]),
        "delta_log_h": float(edge["delta_log_h"]),
    }


def action_one_hot(action_type: str) -> list[float]:
    action = str(action_type)
    return [
        1.0 if action == "keep" else 0.0,
        1.0 if action == "add" else 0.0,
        1.0 if action == "swap" else 0.0,
        1.0 if action == "defer" else 0.0,
    ]


def adapt_host_summary(
    host_summary: Dict[str, float],
    *,
    mode: str,
    prior_alpha: float,
) -> Dict[str, float]:
    adapted = dict(host_summary)
    pair_count = float(adapted.get("pair_count", 0.0))
    if mode == "gt_oracle":
        return adapted
    if mode == "runtime_safe_zero":
        adapted["positive_count"] = 0.0
        adapted["negative_count"] = 0.0
        adapted["score"] = 0.0
        return adapted
    if mode == "runtime_prior_alpha":
        alpha = max(0.0, min(float(prior_alpha), 1.0))
        adapted["positive_count"] = float(alpha * pair_count)
        adapted["negative_count"] = 0.0
        adapted["score"] = float(alpha * pair_count)
        return adapted
    raise ValueError(f"Unsupported host_summary_mode: {mode}")


def adjusted_candidate_utility(
    *,
    raw_utility: float,
    action_type: str,
    swap_utility_bonus: float,
    add_utility_bonus: float,
    defer_utility_penalty: float,
) -> float:
    adjusted = float(raw_utility)
    action = str(action_type)
    if action == "swap":
        adjusted += float(swap_utility_bonus)
    elif action == "add":
        adjusted += float(add_utility_bonus)
    elif action == "defer":
        adjusted -= float(defer_utility_penalty)
    return float(adjusted)


def build_candidate_feature(
    *,
    sample: Dict[str, Any],
    host_summary: Dict[str, float],
    action_type: str,
    add_pair: tuple[int, int] | None,
    remove_pair: tuple[int, int] | None,
    pair_lookup: Dict[tuple[int, int], Dict[str, float]],
) -> list[float]:
    add_stats = pair_stats(sample, pair_lookup, add_pair)
    remove_stats = pair_stats(sample, pair_lookup, remove_pair)
    delta_stats = {
        "refined_score": float(add_stats["refined_score"] - remove_stats["refined_score"]),
        "base_score": float(add_stats["base_score"] - remove_stats["base_score"]),
        "iou": float(add_stats["iou"] - remove_stats["iou"]),
        "bbox_dist": float(add_stats["bbox_dist"] - remove_stats["bbox_dist"]),
        "row_margin": float(add_stats["row_margin"] - remove_stats["row_margin"]),
        "row_entropy": float(add_stats["row_entropy"] - remove_stats["row_entropy"]),
        "track_gap": float(add_stats["track_gap"] - remove_stats["track_gap"]),
        "track_hist": float(add_stats["track_hist"] - remove_stats["track_hist"]),
    }
    cluster = [
        float(sample["num_detections"]),
        float(sample["num_tracks"]),
        float(sample["num_edges"]),
        float(sample.get("is_large_component", 0)),
        float(host_summary["pair_count"]),
        float(host_summary["positive_count"]),
        float(host_summary["negative_count"]),
        float(host_summary["score"]),
    ]
    values = (
        action_one_hot(action_type)
        + cluster
        + [
            float(add_stats["refined_score"]),
            float(add_stats["base_score"]),
            float(add_stats["iou"]),
            float(add_stats["bbox_dist"]),
            float(add_stats["row_degree"]),
            float(add_stats["row_margin"]),
            float(add_stats["row_entropy"]),
            float(add_stats["col_degree"]),
            float(add_stats["track_gap"]),
            float(add_stats["track_hist"]),
            float(add_stats["delta_cx"]),
            float(add_stats["delta_cy"]),
            float(add_stats["delta_log_w"]),
            float(add_stats["delta_log_h"]),
            float(remove_stats["refined_score"]),
            float(remove_stats["base_score"]),
            float(remove_stats["iou"]),
            float(remove_stats["bbox_dist"]),
            float(remove_stats["row_degree"]),
            float(remove_stats["row_margin"]),
            float(remove_stats["row_entropy"]),
            float(remove_stats["col_degree"]),
            float(remove_stats["track_gap"]),
            float(remove_stats["track_hist"]),
            float(remove_stats["delta_cx"]),
            float(remove_stats["delta_cy"]),
            float(remove_stats["delta_log_w"]),
            float(remove_stats["delta_log_h"]),
            float(delta_stats["refined_score"]),
            float(delta_stats["base_score"]),
            float(delta_stats["iou"]),
            float(delta_stats["bbox_dist"]),
            float(delta_stats["row_margin"]),
            float(delta_stats["row_entropy"]),
            float(delta_stats["track_gap"]),
            float(delta_stats["track_hist"]),
            float(1 if remove_pair is not None else 0),
        ]
    )
    if len(values) != feature_dim():
        raise ValueError(f"Expected {feature_dim()} candidate features, got {len(values)}")
    return [float(x) for x in values]


def build_pair_lookup(sample: Dict[str, Any]) -> Dict[tuple[int, int], Dict[str, float]]:
    lookup: Dict[tuple[int, int], Dict[str, float]] = {}
    compact_edges = {
        (int(edge["det_local_idx"]), int(edge["track_local_idx"])): edge
        for edge in sample.get("compact_edges", [])
    }
    for det_idx, track_idx, edge_features in zip(
        sample["edge_det_index"],
        sample["edge_track_index"],
        sample["edge_features"],
    ):
        pair = (int(det_idx), int(track_idx))
        compact = compact_edges.get(pair, {})
        lookup[pair] = {
            "base_score": float(edge_features[0]),
            "refined_score": float(edge_features[1]),
            "iou": float(edge_features[12]),
            "bbox_dist": float(edge_features[13]),
            "delta_cx": float(edge_features[14]),
            "delta_cy": float(edge_features[15]),
            "delta_log_w": float(edge_features[16]),
            "delta_log_h": float(edge_features[17]),
            "edge_is_gt_positive": float(compact.get("edge_is_gt_positive", 0.0)),
        }
    return lookup


def enumerate_candidates(
    sample: Dict[str, Any],
    *,
    candidate_min_refined_score: float,
    host_summary_mode: str,
    host_summary_prior_alpha: float,
    swap_utility_bonus: float,
    add_utility_bonus: float,
    defer_utility_penalty: float,
) -> Tuple[List[Dict[str, Any]], int]:
    host_pairs = {tuple(int(v) for v in pair) for pair in sample.get("host_pairs_local_runtime", [])}
    gt_by_pair = {
        (int(edge["det_local_idx"]), int(edge["track_local_idx"])): int(edge["edge_is_gt_positive"])
        for edge in sample.get("compact_edges", [])
    }
    pair_lookup = build_pair_lookup(sample)
    host_summary = score_pair_set(host_pairs, gt_by_pair)
    host_summary["pair_count"] = float(len(host_pairs))
    host_summary["positive_count"] = float(host_summary["correct"])
    host_summary["negative_count"] = float(host_summary["wrong"])
    feature_host_summary = adapt_host_summary(
        host_summary,
        mode=str(host_summary_mode),
        prior_alpha=float(host_summary_prior_alpha),
    )

    host_det_to_track = {int(det_idx): int(track_idx) for det_idx, track_idx in host_pairs}
    host_track_to_det = {int(track_idx): int(det_idx) for det_idx, track_idx in host_pairs}

    candidates: List[Dict[str, Any]] = [
        {
            "action_type": "keep",
            "add_pair": None,
            "remove_pair": None,
            "utility_delta": 0.0,
            "adjusted_utility_delta": 0.0,
            "new_correct": float(host_summary["correct"]),
            "new_wrong": float(host_summary["wrong"]),
            "candidate_features": build_candidate_feature(
                sample=sample,
                host_summary=feature_host_summary,
                action_type="keep",
                add_pair=None,
                remove_pair=None,
                pair_lookup=pair_lookup,
            ),
        }
    ]

    for pair, edge in sorted(pair_lookup.items()):
        det_idx, track_idx = pair
        refined_score = float(edge["refined_score"])
        if pair not in host_pairs and refined_score >= float(candidate_min_refined_score):
            remove_pairs: set[tuple[int, int]] = set()
            existing_track = host_det_to_track.get(int(det_idx))
            if existing_track is not None and int(existing_track) != int(track_idx):
                remove_pairs.add((int(det_idx), int(existing_track)))
            existing_det = host_track_to_det.get(int(track_idx))
            if existing_det is not None and int(existing_det) != int(det_idx):
                remove_pairs.add((int(existing_det), int(track_idx)))
            if len(remove_pairs) > 1:
                continue
            remove_pair = next(iter(remove_pairs)) if remove_pairs else None
            action_type = "swap" if remove_pair is not None else "add"
            final_pairs = set(host_pairs)
            if remove_pair is not None:
                final_pairs.discard(remove_pair)
            final_pairs.add((int(det_idx), int(track_idx)))
            new_summary = score_pair_set(final_pairs, gt_by_pair)
            utility_delta = float(new_summary["score"] - host_summary["score"])
            adjusted_utility = adjusted_candidate_utility(
                raw_utility=utility_delta,
                action_type=action_type,
                swap_utility_bonus=float(swap_utility_bonus),
                add_utility_bonus=float(add_utility_bonus),
                defer_utility_penalty=float(defer_utility_penalty),
            )
            candidates.append(
                {
                    "action_type": action_type,
                    "add_pair": [int(det_idx), int(track_idx)],
                    "remove_pair": [int(remove_pair[0]), int(remove_pair[1])] if remove_pair is not None else None,
                    "utility_delta": utility_delta,
                    "adjusted_utility_delta": adjusted_utility,
                    "new_correct": float(new_summary["correct"]),
                    "new_wrong": float(new_summary["wrong"]),
                    "candidate_features": build_candidate_feature(
                        sample=sample,
                        host_summary=feature_host_summary,
                        action_type=action_type,
                        add_pair=(int(det_idx), int(track_idx)),
                        remove_pair=remove_pair,
                        pair_lookup=pair_lookup,
                    ),
                }
            )

    for pair in sorted(host_pairs):
        det_idx, track_idx = int(pair[0]), int(pair[1])
        final_pairs = set(host_pairs)
        final_pairs.discard((det_idx, track_idx))
        new_summary = score_pair_set(final_pairs, gt_by_pair)
        utility_delta = float(new_summary["score"] - host_summary["score"])
        adjusted_utility = adjusted_candidate_utility(
            raw_utility=utility_delta,
            action_type="defer",
            swap_utility_bonus=float(swap_utility_bonus),
            add_utility_bonus=float(add_utility_bonus),
            defer_utility_penalty=float(defer_utility_penalty),
        )
        candidates.append(
            {
                "action_type": "defer",
                "add_pair": None,
                "remove_pair": [det_idx, track_idx],
                "utility_delta": utility_delta,
                "adjusted_utility_delta": adjusted_utility,
                "new_correct": float(new_summary["correct"]),
                "new_wrong": float(new_summary["wrong"]),
                "candidate_features": build_candidate_feature(
                    sample=sample,
                    host_summary=feature_host_summary,
                    action_type="defer",
                    add_pair=None,
                    remove_pair=(det_idx, track_idx),
                    pair_lookup=pair_lookup,
                ),
            }
        )

    best_index = 0
    best_key = None
    action_priority = {"keep": 0, "defer": 1, "add": 2, "swap": 3}
    for idx, candidate in enumerate(candidates):
        raw_utility = float(candidate["utility_delta"])
        adjusted_utility = float(candidate.get("adjusted_utility_delta", raw_utility))
        if raw_utility <= 0.0 or adjusted_utility <= 0.0:
            continue
        key = (
            adjusted_utility,
            raw_utility,
            float(candidate["new_correct"]),
            -float(candidate["new_wrong"]),
            int(action_priority.get(str(candidate["action_type"]), -1)),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_index = int(idx)
    return candidates, int(best_index)


def build_dataset_row(
    sample: Dict[str, Any],
    *,
    candidate_min_refined_score: float,
    host_summary_mode: str,
    host_summary_prior_alpha: float,
    swap_utility_bonus: float,
    add_utility_bonus: float,
    defer_utility_penalty: float,
) -> Dict[str, Any]:
    candidates, target_index = enumerate_candidates(
        sample,
        candidate_min_refined_score=float(candidate_min_refined_score),
        host_summary_mode=str(host_summary_mode),
        host_summary_prior_alpha=float(host_summary_prior_alpha),
        swap_utility_bonus=float(swap_utility_bonus),
        add_utility_bonus=float(add_utility_bonus),
        defer_utility_penalty=float(defer_utility_penalty),
    )
    target = candidates[int(target_index)]
    candidate_types = [str(candidate["action_type"]) for candidate in candidates]
    candidate_raw_utilities = [float(candidate["utility_delta"]) for candidate in candidates]
    candidate_adjusted_utilities = [
        float(candidate.get("adjusted_utility_delta", candidate["utility_delta"])) for candidate in candidates
    ]
    candidate_positive_utility = [int(value > 0.0) for value in candidate_adjusted_utilities]
    nonkeep_adjusted_utilities = candidate_adjusted_utilities[1:]
    best_positive_nonkeep_adjusted_utility = max(
        0.0,
        max(nonkeep_adjusted_utilities, default=0.0),
    )
    sorted_nonkeep_positive = sorted(
        [float(value) for value in nonkeep_adjusted_utilities if float(value) > 0.0],
        reverse=True,
    )
    positive_nonkeep_utility_margin_to_second = (
        float(sorted_nonkeep_positive[0] - sorted_nonkeep_positive[1])
        if len(sorted_nonkeep_positive) >= 2
        else float(sorted_nonkeep_positive[0] if sorted_nonkeep_positive else 0.0)
    )
    sorted_adjusted_utilities = sorted(candidate_adjusted_utilities, reverse=True)
    second_adjusted_utility = (
        float(sorted_adjusted_utilities[1])
        if len(sorted_adjusted_utilities) >= 2
        else 0.0
    )
    return {
        "cluster_id": sample["cluster_id"],
        "seq": sample["seq"],
        "frame": int(sample["frame"]),
        "split_tag": sample.get("split_tag", "train"),
        "host_variant": sample.get("host_variant", "official_bytetrack"),
        "num_detections": int(sample["num_detections"]),
        "num_tracks": int(sample["num_tracks"]),
        "num_edges": int(sample["num_edges"]),
        "is_large_component": int(sample.get("is_large_component", 0)),
        "candidate_count": int(len(candidates)),
        "nonkeep_candidate_count": int(sum(1 for action in candidate_types if action != "keep")),
        "candidate_action_types": candidate_types,
        "candidate_features": [candidate["candidate_features"] for candidate in candidates],
        "candidate_add_pairs": [candidate["add_pair"] for candidate in candidates],
        "candidate_remove_pairs": [candidate["remove_pair"] for candidate in candidates],
        "candidate_utility_deltas": candidate_raw_utilities,
        "candidate_adjusted_utility_deltas": candidate_adjusted_utilities,
        "candidate_is_positive_utility": candidate_positive_utility,
        "positive_utility_candidate_count": int(sum(candidate_positive_utility)),
        "positive_nonkeep_candidate_count": int(sum(candidate_positive_utility[1:])),
        "feature_host_summary_mode": str(host_summary_mode),
        "feature_host_summary_prior_alpha": float(host_summary_prior_alpha),
        "target_index": int(target_index),
        "target_action_type": str(target["action_type"]),
        "target_utility_delta": float(target["utility_delta"]),
        "target_adjusted_utility_delta": float(target.get("adjusted_utility_delta", target["utility_delta"])),
        "target_is_positive_utility": int(
            int(target_index) > 0 and float(target.get("adjusted_utility_delta", target["utility_delta"])) > 0.0
        ),
        "target_add_pair": target["add_pair"],
        "target_remove_pair": target["remove_pair"],
        "target_is_nonkeep": int(int(target_index) != 0),
        "cluster_has_positive_utility": int(best_positive_nonkeep_adjusted_utility > 0.0),
        "best_adjusted_utility": float(max(candidate_adjusted_utilities, default=0.0)),
        "second_adjusted_utility": float(second_adjusted_utility),
        "utility_margin_to_second": float(max(candidate_adjusted_utilities, default=0.0) - second_adjusted_utility),
        "best_positive_nonkeep_adjusted_utility": float(best_positive_nonkeep_adjusted_utility),
        "positive_nonkeep_utility_margin_to_second": float(positive_nonkeep_utility_margin_to_second),
        "source_cluster_should_intervene_edit": int(sample.get("cluster_should_intervene_edit", 0)),
        "source_cluster_should_intervene_soft": int(sample.get("cluster_should_intervene_soft", 0)),
        "source_cluster_should_intervene_bridge": int(sample.get("cluster_should_intervene_bridge", 0)),
    }


def append_registry(args: argparse.Namespace, *, out_dir: Path, status: str) -> None:
    cmd = [
        "python",
        str(REPO_ROOT / "scripts" / "append_experiment_record.py"),
        "--csv",
        str(Path(args.registry_csv).resolve()),
        "--kind",
        "analysis",
        "--status",
        status,
        "--script",
        "scripts/build_posthost_one_edit_dataset.py",
        "--dataset",
        "MOT17",
        "--split",
        "official_trainhalf_manifest_split",
        "--tracker-family",
        "official_bytetrack",
        "--variant",
        "posthost_one_edit_dataset",
        "--tag",
        args.dataset_name,
        "--run-root",
        str(out_dir.resolve()),
        "--summary-csv",
        str((out_dir / "summary.csv").resolve()),
        "--notes",
        "build post-host one-edit action dataset from official cluster_examples",
        "--extra",
        f"source_cluster_jsonl={str(Path(args.cluster_jsonl).resolve())}",
        f"candidate_min_refined_score={float(args.candidate_min_refined_score)}",
        f"host_summary_mode={str(args.host_summary_mode)}",
        f"host_summary_prior_alpha={float(args.host_summary_prior_alpha)}",
        f"swap_utility_bonus={float(args.swap_utility_bonus)}",
        f"add_utility_bonus={float(args.add_utility_bonus)}",
        f"defer_utility_penalty={float(args.defer_utility_penalty)}",
    ]
    import subprocess

    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_row: Dict[str, Any] = {
        "dataset_name": args.dataset_name,
        "source_cluster_jsonl": str(Path(args.cluster_jsonl).resolve()),
        "candidate_min_refined_score": float(args.candidate_min_refined_score),
        "host_summary_mode": str(args.host_summary_mode),
        "host_summary_prior_alpha": float(args.host_summary_prior_alpha),
        "swap_utility_bonus": float(args.swap_utility_bonus),
        "add_utility_bonus": float(args.add_utility_bonus),
        "defer_utility_penalty": float(args.defer_utility_penalty),
        "status": "running",
        "error": "",
    }
    write_rows(out_dir / "summary.csv", SUMMARY_FIELDS, [summary_row])

    try:
        dataset_rows: List[Dict[str, Any]] = []
        split_counter: Dict[str, Counter] = defaultdict(Counter)
        target_counter: Counter = Counter()
        candidate_counter: Counter = Counter()

        cluster_path = Path(args.cluster_jsonl).resolve()
        with cluster_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                sample = json.loads(line)
                row = build_dataset_row(
                    sample,
                    candidate_min_refined_score=float(args.candidate_min_refined_score),
                    host_summary_mode=str(args.host_summary_mode),
                    host_summary_prior_alpha=float(args.host_summary_prior_alpha),
                    swap_utility_bonus=float(args.swap_utility_bonus),
                    add_utility_bonus=float(args.add_utility_bonus),
                    defer_utility_penalty=float(args.defer_utility_penalty),
                )
                dataset_rows.append(row)
                split_tag = str(row.get("split_tag", "train"))
                action_type = str(row["target_action_type"])
                split_counter[split_tag]["clusters"] += 1
                split_counter[split_tag][f"target_{action_type}_clusters"] += 1
                split_counter[split_tag]["positive_utility_clusters"] += int(row["cluster_has_positive_utility"])
                split_counter[split_tag]["candidate_count_total"] += int(row["candidate_count"])
                split_counter[split_tag]["candidate_count_max"] = max(
                    int(split_counter[split_tag].get("candidate_count_max", 0)),
                    int(row["candidate_count"]),
                )
                target_counter[f"target_{action_type}_clusters"] += 1
                target_counter["target_nonkeep_clusters"] += int(row["target_is_nonkeep"])
                target_counter["positive_utility_clusters"] += int(row["cluster_has_positive_utility"])
                candidate_counter["candidate_count_total"] += int(row["candidate_count"])
                candidate_counter["candidate_count_max"] = max(
                    int(candidate_counter.get("candidate_count_max", 0)),
                    int(row["candidate_count"]),
                )
                candidate_counter["nonkeep_candidate_count_total"] += int(row["nonkeep_candidate_count"])
                candidate_counter["nonkeep_candidate_count_max"] = max(
                    int(candidate_counter.get("nonkeep_candidate_count_max", 0)),
                    int(row["nonkeep_candidate_count"]),
                )
                candidate_counter["best_positive_nonkeep_adjusted_utility_total"] += float(
                    row["best_positive_nonkeep_adjusted_utility"]
                )
                candidate_counter["positive_nonkeep_utility_margin_to_second_total"] += float(
                    row["positive_nonkeep_utility_margin_to_second"]
                )

        dataset_jsonl = out_dir / "posthost_action_examples.jsonl"
        sample_jsonl = out_dir / "posthost_action_examples.sample.jsonl"
        with dataset_jsonl.open("w", encoding="utf-8") as f:
            for row in dataset_rows:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")
        with sample_jsonl.open("w", encoding="utf-8") as f:
            for row in dataset_rows[: min(16, len(dataset_rows))]:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")

        split_rows = []
        for split_tag in sorted(split_counter.keys()):
            counter = split_counter[split_tag]
            clusters = int(counter.get("clusters", 0))
            split_rows.append(
                {
                    "split_tag": split_tag,
                    "clusters": clusters,
                    "target_keep_clusters": int(counter.get("target_keep_clusters", 0)),
                    "target_add_clusters": int(counter.get("target_add_clusters", 0)),
                    "target_swap_clusters": int(counter.get("target_swap_clusters", 0)),
                    "target_defer_clusters": int(counter.get("target_defer_clusters", 0)),
                    "target_nonkeep_clusters": int(
                        counter.get("target_add_clusters", 0)
                        + counter.get("target_swap_clusters", 0)
                        + counter.get("target_defer_clusters", 0)
                    ),
                    "positive_utility_clusters": int(counter.get("positive_utility_clusters", 0)),
                    "mean_candidates_per_cluster": (
                        float(counter.get("candidate_count_total", 0)) / float(max(clusters, 1))
                    ),
                    "max_candidates_per_cluster": int(counter.get("candidate_count_max", 0)),
                }
            )
        write_rows(out_dir / "split_summary.csv", SPLIT_FIELDS, split_rows)

        summary_row.update(
            {
                "clusters": int(len(dataset_rows)),
                "train_clusters": int(split_counter.get("train", Counter()).get("clusters", 0)),
                "val_clusters": int(split_counter.get("val", Counter()).get("clusters", 0)),
                "target_keep_clusters": int(target_counter.get("target_keep_clusters", 0)),
                "target_add_clusters": int(target_counter.get("target_add_clusters", 0)),
                "target_swap_clusters": int(target_counter.get("target_swap_clusters", 0)),
                "target_defer_clusters": int(target_counter.get("target_defer_clusters", 0)),
                "target_nonkeep_clusters": int(target_counter.get("target_nonkeep_clusters", 0)),
                "positive_utility_clusters": int(target_counter.get("positive_utility_clusters", 0)),
                "mean_candidates_per_cluster": (
                    float(candidate_counter.get("candidate_count_total", 0)) / float(max(len(dataset_rows), 1))
                ),
                "max_candidates_per_cluster": int(candidate_counter.get("candidate_count_max", 0)),
                "mean_nonkeep_candidates_per_cluster": (
                    float(candidate_counter.get("nonkeep_candidate_count_total", 0)) / float(max(len(dataset_rows), 1))
                ),
                "max_nonkeep_candidates_per_cluster": int(candidate_counter.get("nonkeep_candidate_count_max", 0)),
                "mean_best_positive_nonkeep_adjusted_utility": (
                    float(candidate_counter.get("best_positive_nonkeep_adjusted_utility_total", 0.0))
                    / float(max(len(dataset_rows), 1))
                ),
                "mean_positive_nonkeep_utility_margin_to_second": (
                    float(candidate_counter.get("positive_nonkeep_utility_margin_to_second_total", 0.0))
                    / float(max(len(dataset_rows), 1))
                ),
                "status": "success",
                "error": "",
            }
        )
        write_rows(out_dir / "summary.csv", SUMMARY_FIELDS, [summary_row])
        append_registry(args, out_dir=out_dir, status="success")
        return 0
    except Exception as exc:
        summary_row.update({"status": "failed", "error": str(exc)})
        write_rows(out_dir / "summary.csv", SUMMARY_FIELDS, [summary_row])
        append_registry(args, out_dir=out_dir, status="failed")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
