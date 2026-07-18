#!/usr/bin/env python3
from __future__ import annotations

"""Strict sequence-LOSO sequence-calibrated transaction conflict graph.

The outer-held sequence is used only to construct GT-free structural and
appearance features. Its GT is opened only after the tracker is frozen, by
TrackEval. Model fitting and policy calibration use the other three sequences.
"""

import argparse
import csv
import importlib.util
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor


REPO = Path(__file__).resolve().parents[2]
SEQUENCES = ("MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05")
TARGET = "chain_transaction_delta_proxy"
POSITIVE = "chain_transaction_positive"
DEFAULT_PARENT = (
    "outputs/assocriskbench_p15_20260714/fused_a42_adaptive_risk_eval/"
    "l1_r2_q75_adaptive_priority/track_results"
)
PARENT_METRICS = {
    "MOT20-01": {"HOTA": 78.8187, "DetA": 81.84765, "AssA": 76.059633, "IDSW": 40},
    "MOT20-02": {"HOTA": 71.57491, "DetA": 80.63572, "AssA": 63.638735, "IDSW": 286},
    "MOT20-03": {"HOTA": 80.665165, "DetA": 81.184465, "AssA": 80.182254, "IDSW": 147},
    "MOT20-05": {"HOTA": 79.49144, "DetA": 81.95191, "AssA": 77.14945, "IDSW": 435},
}

GRAPH_FEATURES = [
    "src_conflict_degree",
    "dst_conflict_degree",
    "conflict_degree",
    "log_src_conflict_degree",
    "log_dst_conflict_degree",
    "log_conflict_degree",
    "pair_candidate_count",
    "src_candidate_fraction",
    "dst_candidate_fraction",
    "sequence_appearance_percentile",
    "sequence_segment_appearance_percentile",
    "sequence_motion_percentile",
    "sequence_gap_percentile",
    "src_appearance_percentile",
    "dst_appearance_percentile",
    "src_segment_appearance_percentile",
    "dst_segment_appearance_percentile",
    "src_motion_percentile",
    "dst_motion_percentile",
    "rank_consensus",
    "appearance_rank_consensus",
    "motion_rank_consensus",
    "appearance_upgrade",
    "segment_endpoint_gain",
    "coherence_floor",
    "coherence_imbalance",
    "motion_appearance_joint",
    "action_complexity",
    "conflict_per_merged_row",
]


def load_module(name: str, relative_path: str):
    path = REPO / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parse_float_grid(raw: str) -> List[float]:
    values = sorted({float(item) for item in raw.split(",") if item.strip()})
    if not values:
        raise ValueError("empty calibration grid")
    return values


def robust_percentile(values: pd.Series, higher_is_better: bool = True) -> pd.Series:
    ranked = values.rank(method="average", pct=True)
    return ranked if higher_is_better else 1.0 - ranked + (1.0 / max(len(values), 1))


def grouped_percentile(
    frame: pd.DataFrame, group: str, value: str, higher_is_better: bool = True
) -> pd.Series:
    ranked = frame.groupby(group, sort=False)[value].rank(method="average", pct=True)
    if higher_is_better:
        return ranked
    sizes = frame.groupby(group, sort=False)[value].transform("size").clip(lower=1)
    return 1.0 - ranked + 1.0 / sizes


def add_conflict_graph_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    required = {
        "transaction_src_track_id",
        "transaction_dst_track_id",
        "appearance_cos",
        "segment_appearance_cos",
        "motion_error_min",
    }
    missing = sorted(required - set(output.columns))
    if missing:
        raise RuntimeError(f"missing transaction graph columns: {missing}")

    src = output.transaction_src_track_id.astype(np.int64)
    dst = output.transaction_dst_track_id.astype(np.int64)
    all_tracks = pd.concat([src, dst], ignore_index=True)
    all_counts = all_tracks.value_counts()
    src_degree = src.map(all_counts).to_numpy(float) - 1.0
    dst_degree = dst.map(all_counts).to_numpy(float) - 1.0
    output["src_conflict_degree"] = np.maximum(src_degree, 0.0)
    output["dst_conflict_degree"] = np.maximum(dst_degree, 0.0)
    output["conflict_degree"] = np.maximum(src_degree + dst_degree, 0.0)
    output["log_src_conflict_degree"] = np.log1p(output.src_conflict_degree)
    output["log_dst_conflict_degree"] = np.log1p(output.dst_conflict_degree)
    output["log_conflict_degree"] = np.log1p(output.conflict_degree)
    output["pair_candidate_count"] = output.groupby(
        ["transaction_src_track_id", "transaction_dst_track_id"], sort=False
    ).src_chunk.transform("size")
    candidate_denominator = max(len(output), 1)
    output["src_candidate_fraction"] = (output.src_conflict_degree + 1.0) / candidate_denominator
    output["dst_candidate_fraction"] = (output.dst_conflict_degree + 1.0) / candidate_denominator

    output["sequence_appearance_percentile"] = robust_percentile(output.appearance_cos)
    output["sequence_segment_appearance_percentile"] = robust_percentile(
        output.segment_appearance_cos
    )
    output["sequence_motion_percentile"] = robust_percentile(
        output.motion_error_min, higher_is_better=False
    )
    output["sequence_gap_percentile"] = robust_percentile(output.gap, higher_is_better=False)
    output["src_appearance_percentile"] = grouped_percentile(
        output, "transaction_src_track_id", "appearance_cos"
    )
    output["dst_appearance_percentile"] = grouped_percentile(
        output, "transaction_dst_track_id", "appearance_cos"
    )
    output["src_segment_appearance_percentile"] = grouped_percentile(
        output, "transaction_src_track_id", "segment_appearance_cos"
    )
    output["dst_segment_appearance_percentile"] = grouped_percentile(
        output, "transaction_dst_track_id", "segment_appearance_cos"
    )
    output["src_motion_percentile"] = grouped_percentile(
        output, "transaction_src_track_id", "motion_error_min", higher_is_better=False
    )
    output["dst_motion_percentile"] = grouped_percentile(
        output, "transaction_dst_track_id", "motion_error_min", higher_is_better=False
    )
    output["appearance_rank_consensus"] = np.minimum(
        output.src_segment_appearance_percentile,
        output.dst_segment_appearance_percentile,
    )
    output["motion_rank_consensus"] = np.minimum(
        output.src_motion_percentile, output.dst_motion_percentile
    )
    output["rank_consensus"] = np.minimum(
        output.appearance_rank_consensus, output.motion_rank_consensus
    )
    output["appearance_upgrade"] = (
        output.segment_appearance_cos.to_numpy(float)
        - output.track_appearance_cos.to_numpy(float)
    )
    endpoint_floor = np.minimum(
        output.src_prefix_endpoint_cos.to_numpy(float),
        output.dst_suffix_endpoint_cos.to_numpy(float),
    )
    output["segment_endpoint_gain"] = (
        output.segment_appearance_cos.to_numpy(float) - endpoint_floor
    )
    output["coherence_floor"] = np.minimum(
        output.src_prefix_coherence.to_numpy(float),
        output.dst_suffix_coherence.to_numpy(float),
    )
    output["coherence_imbalance"] = np.abs(
        output.src_prefix_coherence.to_numpy(float)
        - output.dst_suffix_coherence.to_numpy(float)
    )
    output["motion_appearance_joint"] = (
        output.segment_appearance_cos.to_numpy(float)
        - np.log1p(np.maximum(output.motion_error_min.to_numpy(float), 0.0))
    )
    output["action_complexity"] = (
        output.transaction_removes_source_out.to_numpy(float)
        + output.transaction_removes_source_in.to_numpy(float)
    )
    output["conflict_per_merged_row"] = output.conflict_degree.to_numpy(float) / np.maximum(
        output.merged_segment_rows.to_numpy(float), 1.0
    )
    return output


def structural_transactions(meta: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    cross = edges[edges.same_source.to_numpy(int) == 0].copy()
    meta = meta.sort_values("chunk_id")
    if meta.chunk_id.tolist() != list(range(len(meta))):
        raise RuntimeError("chunk ids are not dense")
    track_for_chunk = meta.source_track_id.to_numpy(int)
    ordinal_for_chunk = meta.source_ordinal.to_numpy(int)
    chunks_per_track = meta.groupby("source_track_id").source_ordinal.max().add(1).to_dict()
    src = cross.src_chunk.to_numpy(int)
    dst = cross.dst_chunk.to_numpy(int)
    src_track = track_for_chunk[src]
    dst_track = track_for_chunk[dst]
    src_ordinal = ordinal_for_chunk[src]
    dst_ordinal = ordinal_for_chunk[dst]
    cross["transaction_src_track_id"] = src_track
    cross["transaction_dst_track_id"] = dst_track
    cross["transaction_removes_source_out"] = [
        int(ordinal + 1 < chunks_per_track[int(track)])
        for ordinal, track in zip(src_ordinal, src_track)
    ]
    cross["transaction_removes_source_in"] = (dst_ordinal > 0).astype(np.int8)
    return cross


def sequence_class_weights(frame: pd.DataFrame) -> np.ndarray:
    groups = frame.groupby(["seq", POSITIVE], sort=False).size().to_dict()
    present = max(len(groups), 1)
    weights = np.asarray(
        [len(frame) / (present * groups[(seq, value)]) for seq, value in zip(frame.seq, frame[POSITIVE])],
        dtype=np.float64,
    )
    return weights


def sequence_sign_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.seq.value_counts().to_dict()
    return np.asarray(
        [len(frame) / (len(counts) * counts[seq]) for seq in frame.seq], dtype=np.float64
    )


def sequence_target_scale(frame: pd.DataFrame) -> np.ndarray:
    scale_by_sequence: Dict[str, float] = {}
    for seq, part in frame.groupby("seq", sort=False):
        positive = part.loc[part[TARGET] > 0, TARGET].to_numpy(float)
        if len(positive):
            scale = float(np.median(positive))
        else:
            scale = 1.0
        scale_by_sequence[str(seq)] = max(scale, 1e-3)
    return frame.seq.map(scale_by_sequence).to_numpy(float)


def fit_distributional_models(
    frame: pd.DataFrame, features: Sequence[str], seed: int, max_iter: int
):
    classifier = HistGradientBoostingClassifier(
        max_iter=max_iter,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=60,
        l2_regularization=14.0,
        max_bins=255,
        random_state=seed,
        early_stopping=False,
    )
    classifier.fit(
        frame[list(features)],
        frame[POSITIVE].astype(int),
        sample_weight=sequence_class_weights(frame),
    )
    scales = sequence_target_scale(frame)
    positive = frame[TARGET].to_numpy(float) > 0
    negative = frame[TARGET].to_numpy(float) < 0
    if positive.sum() < 10 or negative.sum() < 10:
        raise RuntimeError("insufficient signed transaction labels")

    def fit_magnitude(mask: np.ndarray, target: np.ndarray, offset: int):
        model = HistGradientBoostingRegressor(
            loss="absolute_error",
            max_iter=max_iter,
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            l2_regularization=14.0,
            max_bins=255,
            random_state=seed + offset,
            early_stopping=False,
        )
        part = frame.loc[mask].copy()
        model.fit(
            part[list(features)],
            target,
            sample_weight=sequence_sign_weights(part),
        )
        return model

    gain_target = np.log1p(frame.loc[positive, TARGET].to_numpy(float) / scales[positive])
    loss_target = np.log1p(-frame.loc[negative, TARGET].to_numpy(float) / scales[negative])
    gain_model = fit_magnitude(positive, gain_target, 1000)
    loss_model = fit_magnitude(negative, loss_target, 2000)
    return classifier, gain_model, loss_model


def predict_distributional(
    frame: pd.DataFrame, models, features: Sequence[str]
) -> pd.DataFrame:
    classifier, gain_model, loss_model = models
    output = frame.copy()
    probability = classifier.predict_proba(output[list(features)])[:, 1]
    gain = np.expm1(gain_model.predict(output[list(features)]))
    loss = np.expm1(loss_model.predict(output[list(features)]))
    output["pred_positive_probability"] = probability
    output["pred_normalized_gain"] = np.maximum(gain, 0.0)
    output["pred_normalized_loss"] = np.maximum(loss, 0.0)
    output["pred_entropy"] = -(
        probability * np.log(np.maximum(probability, 1e-8))
        + (1.0 - probability) * np.log(np.maximum(1.0 - probability, 1e-8))
    )
    return output


def add_policy_score(frame: pd.DataFrame, loss_multiplier: float) -> pd.DataFrame:
    output = frame.copy()
    probability = output.pred_positive_probability.to_numpy(float)
    gain = output.pred_normalized_gain.to_numpy(float)
    loss = output.pred_normalized_loss.to_numpy(float)
    score = probability * gain - loss_multiplier * (1.0 - probability) * loss
    output["policy_score"] = score
    output["policy_score_percentile"] = robust_percentile(output.policy_score)
    return output


def maximum_weight_transaction_matching(
    frame: pd.DataFrame,
    loss_multiplier: float,
    min_probability: float,
    score_quantile: float,
) -> pd.DataFrame:
    scored = add_policy_score(frame, loss_multiplier)
    eligible = scored[
        (scored.policy_score.to_numpy(float) > 0.0)
        & (scored.pred_positive_probability.to_numpy(float) >= min_probability)
        & (scored.policy_score_percentile.to_numpy(float) >= score_quantile)
    ].copy()
    if eligible.empty:
        return eligible
    eligible.sort_values(
        [
            "transaction_src_track_id",
            "transaction_dst_track_id",
            "policy_score",
            "src_chunk",
            "dst_chunk",
        ],
        ascending=[True, True, False, True, True],
        inplace=True,
    )
    best_by_pair: Dict[Tuple[int, int], Tuple[float, int]] = {}
    for index, row in eligible.iterrows():
        left = int(row.transaction_src_track_id)
        right = int(row.transaction_dst_track_id)
        key = (min(left, right), max(left, right))
        weight = float(row.policy_score)
        current = best_by_pair.get(key)
        if current is None or weight > current[0]:
            best_by_pair[key] = (weight, int(index))

    graph = nx.Graph()
    for (left, right), (weight, index) in sorted(best_by_pair.items()):
        if left == right:
            continue
        graph.add_edge(left, right, weight=weight, row_index=index)
    if graph.number_of_edges() == 0:
        return eligible.iloc[:0].copy()
    matching = nx.algorithms.matching.max_weight_matching(
        graph, maxcardinality=False, weight="weight"
    )
    indices = []
    for left, right in matching:
        indices.append(int(graph[left][right]["row_index"]))
    selected = scored.loc[sorted(indices)].copy()
    selected.sort_values(
        ["policy_score", "src_chunk", "dst_chunk"],
        ascending=[False, True, True],
        inplace=True,
    )
    return selected


def policy_diagnostics(
    predictions: Dict[str, pd.DataFrame],
    loss_multiplier: float,
    min_probability: float,
    score_quantile: float,
) -> Dict[str, object]:
    by_sequence = []
    normalized_utilities = []
    for seq, frame in predictions.items():
        selected = maximum_weight_transaction_matching(
            frame, loss_multiplier, min_probability, score_quantile
        )
        actual = selected[TARGET].to_numpy(float)
        positive_total = float(frame.loc[frame[TARGET] > 0, TARGET].sum())
        utility = float(actual.sum())
        normalized = utility / max(positive_total, 1e-9)
        normalized_utilities.append(normalized)
        by_sequence.append(
            {
                "seq": seq,
                "actions": int(len(selected)),
                "actual_proxy_sum": utility,
                "normalized_utility": normalized,
                "positive": int((actual > 0).sum()),
                "negative": int((actual < 0).sum()),
                "precision": float((actual > 0).mean()) if len(actual) else None,
            }
        )
    values = np.asarray(normalized_utilities, dtype=float)
    return {
        "loss_multiplier": loss_multiplier,
        "min_probability": min_probability,
        "score_quantile": score_quantile,
        "worst_normalized_utility": float(values.min()) if len(values) else 0.0,
        "median_normalized_utility": float(np.median(values)) if len(values) else 0.0,
        "sum_normalized_utility": float(values.sum()),
        "positive_sequences": int((values > 0).sum()),
        "negative_sequences": int((values < 0).sum()),
        "actions": int(sum(item["actions"] for item in by_sequence)),
        "by_sequence": by_sequence,
    }


def policy_key(report: Dict[str, object]) -> Tuple[float, int, float, float, int]:
    return (
        float(report["worst_normalized_utility"]),
        int(report["positive_sequences"]),
        float(report["median_normalized_utility"]),
        float(report["sum_normalized_utility"]),
        -int(report["actions"]),
    )


def choose_policy(
    predictions: Dict[str, pd.DataFrame],
    loss_grid: Iterable[float],
    probability_grid: Iterable[float],
    quantile_grid: Iterable[float],
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    reports: List[Dict[str, object]] = [
        {
            "loss_multiplier": None,
            "min_probability": None,
            "score_quantile": None,
            "worst_normalized_utility": 0.0,
            "median_normalized_utility": 0.0,
            "sum_normalized_utility": 0.0,
            "positive_sequences": 0,
            "negative_sequences": 0,
            "actions": 0,
            "by_sequence": [
                {
                    "seq": seq,
                    "actions": 0,
                    "actual_proxy_sum": 0.0,
                    "normalized_utility": 0.0,
                    "positive": 0,
                    "negative": 0,
                    "precision": None,
                }
                for seq in predictions
            ],
        }
    ]
    for loss_multiplier in loss_grid:
        for min_probability in probability_grid:
            for score_quantile in quantile_grid:
                reports.append(
                    policy_diagnostics(
                        predictions,
                        loss_multiplier,
                        min_probability,
                        score_quantile,
                    )
                )
    chosen = max(reports, key=policy_key)
    return chosen, reports


def evaluate_held(
    track_results: Path, held: str, tracker_name: str, work_dir: Path
) -> Dict[str, float]:
    command = [
        sys.executable,
        "scripts/eval_motstyle_trackeval.py",
        "--benchmark-name",
        "MOT20",
        "--split-to-eval",
        "train",
        "--gt-root",
        "datasets/MOT20/train",
        "--results-dir",
        str(track_results),
        "--tracker-name",
        tracker_name,
        "--work-dir",
        str(work_dir),
        "--seqs",
        held,
    ]
    completed = subprocess.run(
        command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    (work_dir.parent / "eval.log").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(completed.stdout[-5000:])
    detailed = work_dir / "eval" / tracker_name / "pedestrian_detailed.csv"
    rows = list(csv.DictReader(detailed.open(encoding="utf-8")))
    row = next(item for item in rows if item["seq"] in {held, "COMBINED"})
    return {
        "HOTA": 100.0 * float(row["HOTA___AUC"]),
        "DetA": 100.0 * float(row["DetA___AUC"]),
        "AssA": 100.0 * float(row["AssA___AUC"]),
        "IDSW": int(float(row["IDSW"])),
    }


def write_metrics(path: Path, row: Dict[str, object]) -> None:
    fieldnames = [
        "status",
        "held_sequence",
        "HOTA",
        "DetA",
        "AssA",
        "IDSW",
        "delta_HOTA",
        "delta_AssA",
        "delta_IDSW",
        "selected_actions",
        "loss_multiplier",
        "min_probability",
        "score_quantile",
        "message",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def append_registry(output_root: Path, metrics: Dict[str, object], held: str) -> None:
    registry = REPO / "outputs" / "experiment_registry.csv"
    with registry.open(newline="", encoding="utf-8") as handle:
        fieldnames = next(csv.reader(handle))
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "kind": "eval",
        "status": "success",
        "script": "scripts/m23_research/m23_25_sequence_calibrated_transaction_graph.py",
        "dataset": "MOT20",
        "split": "strict_sequence_loso",
        "tracker_family": "FM-Track",
        "variant": "m23_25_sequence_calibrated_transaction_graph",
        "tag": "GT-free_outer_LOSO",
        "run_root": str(output_root),
        "summary_csv": str(output_root / "metrics.csv"),
        "notes": "outer held GT used only by final TrackEval; no fixed action budget",
        "name": f"m23_25_{held}",
        "HOTA": metrics["HOTA"],
        "DetA": metrics["DetA"],
        "AssA": metrics["AssA"],
        "IDSW": metrics["IDSW"],
        "delta_HOTA": metrics["delta_HOTA"],
        "delta_AssA": metrics["delta_AssA"],
        "delta_IDSW": metrics["delta_IDSW"],
        "protocol_tag": "strict_outer_sequence_loso",
        "seq": held,
    }
    with registry.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--held-seq", required=True, choices=SEQUENCES)
    parser.add_argument("--graph-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--parent", default=DEFAULT_PARENT)
    parser.add_argument("--loss-grid", default="1,2,4,8,16,32")
    parser.add_argument("--probability-grid", default="0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--quantile-grid", default="0.99,0.995,0.9975,0.999,0.9995")
    parser.add_argument("--max-iter", type=int, default=220)
    parser.add_argument("--max-training-rows-per-seq", type=int, default=0)
    parser.add_argument(
        "--reuse-training-label-root",
        help=(
            "Optional strict outer-training transaction label root. Each sequence "
            "must contain cross_chain_transaction_utility.parquet."
        ),
    )
    parser.add_argument("--skip-trackeval", action="store_true")
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()

    held = args.held_seq
    training_sequences = [seq for seq in SEQUENCES if seq != held]
    graph_root = Path(args.graph_root)
    output_root = Path(args.output_root)
    parent = Path(args.parent)
    output_root.mkdir(parents=True, exist_ok=True)
    metrics_path = output_root / "metrics.csv"
    write_metrics(
        metrics_path,
        {"status": "running", "held_sequence": held, "message": "model fitting"},
    )
    protocol = {
        "experiment": "M23-25 sequence-calibrated transaction conflict graph",
        "held_sequence": held,
        "training_sequences": training_sequences,
        "held_gt_use": "final TrackEval only",
        "candidate_feature_gt_use": "none",
        "training_gt_use": "training-sequence transaction utility labels only",
        "reuse_training_label_root": args.reuse_training_label_root,
        "selection": (
            "sequence-relative score quantile plus exact maximum-weight transaction "
            "matching; no fixed action budget"
        ),
        "calibration": (
            "inner leave-one-training-sequence-out; maximize worst sequence-normalized "
            "transaction utility"
        ),
        "status": "running",
    }
    (output_root / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    try:
        utility = load_module(
            "m23_25_utility", "scripts/m23_research/m23_11_add_micrograph_utility.py"
        )
        chain = load_module(
            "m23_25_chain", "scripts/m23_research/m23_12_chain_transaction_oracle.py"
        )
        base = load_module(
            "m23_25_base", "scripts/m23_research/m23_12_train_chain_transaction_loso.py"
        )
        evaluator = load_module(
            "m23_25_evaluator", "scripts/m23_research/m23_11_eval_utility_graph.py"
        )
        utility_root = output_root / "training_edge_utility"
        label_root = output_root / "training_transaction_labels"
        prediction_root = output_root / "predictions"
        track_root = output_root / "track_results"
        for path in [utility_root, label_root, prediction_root, track_root]:
            path.mkdir(parents=True, exist_ok=True)
        utility.ROOT = graph_root
        utility.OUT = utility_root
        utility.PARENT = parent
        chain.DATA = graph_root
        chain.UTILITY = utility_root
        chain.OUT = output_root / "unused_oracle_output"
        chain.PARENT = parent
        base.META = graph_root
        evaluator.DATA = graph_root
        evaluator.PARENT = parent
        features = list(base.FEATURES) + GRAPH_FEATURES

        m23 = None if args.reuse_training_label_root else utility.load_m23()
        training_frames: Dict[str, pd.DataFrame] = {}
        label_reports = []
        for seq in training_sequences:
            sequence_dir = label_root / seq
            sequence_dir.mkdir(parents=True, exist_ok=True)
            if args.reuse_training_label_root:
                reused_path = (
                    Path(args.reuse_training_label_root)
                    / seq
                    / "cross_chain_transaction_utility.parquet"
                )
                transactions = pd.read_parquet(reused_path)
                label_reports.append(
                    {
                        "seq": seq,
                        "reused": str(reused_path),
                        "rows": int(len(transactions)),
                        "held_sequence_excluded": True,
                    }
                )
            else:
                label_reports.append(utility.label_sequence(seq, m23))
                _, _, transactions = chain.label_transactions(seq, utility)
            transactions.to_parquet(
                sequence_dir / "cross_chain_transaction_utility.parquet", index=False
            )
            frame = add_conflict_graph_features(base.add_chain_features(seq, transactions))
            if args.max_training_rows_per_seq > 0 and len(frame) > args.max_training_rows_per_seq:
                frame = frame.sample(
                    args.max_training_rows_per_seq,
                    random_state=2500 + SEQUENCES.index(seq),
                ).sort_index()
            training_frames[seq] = frame

        inner_predictions: Dict[str, pd.DataFrame] = {}
        for fold_index, pseudo_held in enumerate(training_sequences):
            inner_train = pd.concat(
                [
                    training_frames[seq]
                    for seq in training_sequences
                    if seq != pseudo_held
                ],
                ignore_index=True,
                sort=False,
            )
            models = fit_distributional_models(
                inner_train, features, 25000 + fold_index, args.max_iter
            )
            predicted = predict_distributional(
                training_frames[pseudo_held], models, features
            )
            predicted.to_parquet(
                prediction_root / f"inner_{pseudo_held}.parquet", index=False
            )
            inner_predictions[pseudo_held] = predicted

        chosen, calibration_reports = choose_policy(
            inner_predictions,
            parse_float_grid(args.loss_grid),
            parse_float_grid(args.probability_grid),
            parse_float_grid(args.quantile_grid),
        )
        if chosen["loss_multiplier"] is None:
            chosen_policy = None
        else:
            chosen_policy = {
                "loss_multiplier": float(chosen["loss_multiplier"]),
                "min_probability": float(chosen["min_probability"]),
                "score_quantile": float(chosen["score_quantile"]),
            }

        outer_train = pd.concat(
            list(training_frames.values()), ignore_index=True, sort=False
        )
        outer_models = fit_distributional_models(
            outer_train, features, 25999, args.max_iter
        )
        held_meta = pd.read_parquet(graph_root / held / "microtracklets.parquet")
        held_edges = pd.read_parquet(graph_root / held / "candidate_edges.parquet")
        held_structural = structural_transactions(held_meta, held_edges)
        held_features = add_conflict_graph_features(
            base.add_chain_features(held, held_structural)
        )
        held_predictions = predict_distributional(held_features, outer_models, features)
        held_predictions.to_parquet(
            prediction_root / f"{held}_predictions.parquet", index=False
        )
        if chosen_policy is None:
            selected = held_predictions.iloc[:0].copy()
            selected["policy_score"] = np.asarray([], dtype=float)
        else:
            selected = maximum_weight_transaction_matching(
                held_predictions, **chosen_policy
            )
        selected.to_parquet(
            output_root / f"{held}_selected_transactions.parquet", index=False
        )

        applied = chain.apply_transactions(held_edges, selected.assign(
            **{TARGET: selected.policy_score.to_numpy(float)}
        ))
        for column, default in [
            ("assa_edge_delta_proxy", 0.0),
            ("assa_edge_positive", 0),
            ("assa_edge_negative", 0),
        ]:
            if column not in applied:
                applied[column] = default
        applied.to_parquet(output_root / f"{held}_applied_edges.parquet", index=False)
        tracker_report = evaluator.write_tracker(
            held, held_meta, applied, track_root / f"{held}.txt"
        )

        tracker_name = "m23_25_sequence_calibrated_transaction_graph"
        evaluation = None
        if not args.skip_trackeval:
            evaluation = evaluate_held(
                track_root, held, tracker_name, output_root / "eval_work"
            )
        baseline = PARENT_METRICS[held]
        report = {
            **protocol,
            "status": "completed" if evaluation is not None else "tracker_frozen",
            "features": features,
            "graph_features": GRAPH_FEATURES,
            "label_reports": label_reports,
            "calibration_grid_size": len(calibration_reports),
            "calibration_candidates": calibration_reports,
            "chosen_policy": chosen,
            "held_candidates": int(len(held_predictions)),
            "held_selected_actions": int(len(selected)),
            "held_predicted_score_sum": float(selected.policy_score.sum()),
            "tracker_report": tracker_report,
            "baseline": baseline,
            "eval": evaluation,
        }
        (output_root / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        if evaluation is None:
            metrics = {
                "status": "tracker_frozen",
                "held_sequence": held,
                "selected_actions": len(selected),
                "message": "TrackEval skipped",
            }
        else:
            metrics = {
                "status": "success",
                "held_sequence": held,
                **evaluation,
                "delta_HOTA": evaluation["HOTA"] - baseline["HOTA"],
                "delta_AssA": evaluation["AssA"] - baseline["AssA"],
                "delta_IDSW": evaluation["IDSW"] - baseline["IDSW"],
                "selected_actions": len(selected),
                "loss_multiplier": (
                    chosen_policy["loss_multiplier"] if chosen_policy else ""
                ),
                "min_probability": (
                    chosen_policy["min_probability"] if chosen_policy else ""
                ),
                "score_quantile": (
                    chosen_policy["score_quantile"] if chosen_policy else ""
                ),
                "message": "strict outer-held TrackEval",
            }
        write_metrics(metrics_path, metrics)
        protocol["status"] = metrics["status"]
        (output_root / "protocol.json").write_text(
            json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if evaluation is not None and not args.no_register:
            append_registry(output_root, metrics, held)
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    except Exception as exc:
        protocol["status"] = "failed"
        protocol["error"] = f"{type(exc).__name__}: {exc}"
        (output_root / "protocol.json").write_text(
            json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        write_metrics(
            metrics_path,
            {
                "status": "failed",
                "held_sequence": held,
                "message": protocol["error"],
            },
        )
        raise


if __name__ == "__main__":
    main()
