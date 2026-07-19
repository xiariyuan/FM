#!/usr/bin/env python3
from __future__ import annotations

"""M23-36: robust-fold strict nested-LOSO component matcher.

The decision target is not independent transaction positivity.  Training labels
mark transactions selected by an exact maximum-weight matching on each
outer-training sequence.  A separate catastrophic-risk head estimates negative
utility probability and severity.  At inference, GT-free transactions are
scored with a lower-confidence value and solved jointly by conflict component,
with no-op represented by excluding non-positive edges.

The policy is frozen by the worst per-sequence inner HOTA improvement over
no-op, rather than pooled COMBINED HOTA alone.  This prevents a large training
sequence from hiding a regression on another inner-held sequence.

The outer-held sequence GT is not opened until the model, policy, selected
transactions, applied graph, and tracker text are frozen.  Inner TrackEval uses
only outer-training sequences and is therefore permitted by strict sequence
LOSO.
"""

import argparse
import csv
import importlib.util
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor


REPO = Path(__file__).resolve().parents[2]
SEQUENCES = ("MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05")
TARGET = "chain_transaction_delta_proxy"
ORACLE_SELECTED = "component_oracle_selected"
CATASTROPHIC = "catastrophic_negative"
DEFAULT_PARENT = (
    "outputs/assocriskbench_p15_20260714/fused_a42_adaptive_risk_eval/"
    "l1_r2_q75_adaptive_priority/track_results"
)
STRICT_M23_25 = {
    "MOT20-01": {"HOTA": 79.181480, "DetA": 81.897146, "AssA": 76.711800, "IDSW": 43},
    "MOT20-02": {"HOTA": 72.868943, "DetA": 80.713767, "AssA": 65.885675, "IDSW": 317},
    "MOT20-03": {"HOTA": 80.571616, "DetA": 81.183845, "AssA": 79.997680, "IDSW": 139},
    "MOT20-05": {"HOTA": 79.653203, "DetA": 81.955170, "AssA": 77.457994, "IDSW": 475},
}

COMPONENT_RAW_FEATURES = (
    "segment_appearance_cos",
    "track_appearance_cos",
    "motion_error_min",
    "detached_fraction",
    "dst_cut_fraction",
    "src_cut_fraction",
    "conflict_per_merged_row",
    "rank_consensus",
    "merged_coherence_gain",
    "coherence_floor",
)

COMPONENT_FEATURES = [
    "plausible_component_edge",
    "component_edge_count",
    "component_track_count",
    "component_density",
    "log_component_edge_count",
    "log_component_track_count",
    "src_transaction_count",
    "dst_transaction_count",
]
for _column in COMPONENT_RAW_FEATURES:
    COMPONENT_FEATURES.extend(
        [
            f"component_{_column}_mean",
            f"component_{_column}_max",
            f"component_{_column}_min",
            f"component_{_column}_rank",
            f"src_{_column}_max",
            f"dst_{_column}_max",
            f"src_{_column}_mean",
            f"dst_{_column}_mean",
            f"src_{_column}_margin",
            f"dst_{_column}_margin",
        ]
    )


@dataclass(frozen=True)
class Policy:
    policy_id: str
    risk_lambda: float
    uncertainty_lambda: float
    blocking_lambda: float
    score_quantile: float
    min_selection_probability: float
    no_op_margin: float = 0.0


def default_policies() -> List[Policy]:
    return [
        Policy("noop", 0.0, 0.0, 0.0, 1.0, 1.0, math.inf),
        Policy("r1_u0_b0_q0975_p001", 1.0, 0.0, 0.0, 0.975, 0.01),
        Policy("r1_u0_b0_q0980_p001", 1.0, 0.0, 0.0, 0.980, 0.01),
        Policy("r1_u0_b0_q0990_p001", 1.0, 0.0, 0.0, 0.990, 0.01),
        Policy("r1_u0_b0_q0995_p001", 1.0, 0.0, 0.0, 0.995, 0.01),
        Policy("r2_u025_b025_q0975_p002", 2.0, 0.25, 0.25, 0.975, 0.02),
        Policy("r2_u025_b025_q0980_p002", 2.0, 0.25, 0.25, 0.980, 0.02),
        Policy("r2_u025_b025_q0990_p002", 2.0, 0.25, 0.25, 0.990, 0.02),
        Policy("r2_u025_b025_q0995_p002", 2.0, 0.25, 0.25, 0.995, 0.02),
        Policy("r4_u050_b050_q0975_p003", 4.0, 0.50, 0.50, 0.975, 0.03),
        Policy("r4_u050_b050_q0980_p003", 4.0, 0.50, 0.50, 0.980, 0.03),
        Policy("r4_u050_b050_q0990_p003", 4.0, 0.50, 0.50, 0.990, 0.03),
        Policy("r4_u050_b050_q0995_p003", 4.0, 0.50, 0.50, 0.995, 0.03),
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


class UnionFind:
    def __init__(self) -> None:
        self.parent: Dict[int, int] = {}
        self.rank: Dict[int, int] = {}

    def find(self, value: int) -> int:
        if value not in self.parent:
            self.parent[value] = value
            self.rank[value] = 0
            return value
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def safe_entropy(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(probability, 1e-8, 1.0 - 1e-8)
    return -(probability * np.log(probability) + (1.0 - probability) * np.log(1.0 - probability))


def sanitize(frame: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    output = frame.copy()
    for column in features:
        if column not in output:
            output[column] = 0.0
    output[list(features)] = (
        output[list(features)]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(np.float32)
    )
    return output.copy()


def add_component_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    required = {"transaction_src_track_id", "transaction_dst_track_id"}
    missing = sorted(required - set(output.columns))
    if missing:
        raise RuntimeError(f"missing component keys: {missing}")

    src = output.transaction_src_track_id.to_numpy(np.int64)
    dst = output.transaction_dst_track_id.to_numpy(np.int64)
    plausible = (
        (output.rank_consensus.to_numpy(float) >= 0.80)
        & (output.sequence_segment_appearance_percentile.to_numpy(float) >= 0.85)
        & (output.sequence_motion_percentile.to_numpy(float) >= 0.70)
    )
    union_find = UnionFind()
    for left, right in zip(src[plausible], dst[plausible]):
        union_find.union(int(left), int(right))
    component_keys = []
    for left, right in zip(src, dst):
        left_root = union_find.find(int(left))
        right_root = union_find.find(int(right))
        if left_root == right_root:
            component_keys.append((0, left_root, left_root))
        else:
            component_keys.append((1, min(left_root, right_root), max(left_root, right_root)))
    component_id, _ = pd.factorize(pd.Series(component_keys, dtype=object), sort=True)
    output["component_id"] = component_id.astype(np.int32)

    component_group = output.groupby("component_id", sort=False)
    feature_data: Dict[str, pd.Series | np.ndarray] = {
        "plausible_component_edge": plausible.astype(np.float32),
        "component_edge_count": component_group.src_chunk.transform("size").astype(float),
    }
    component_tracks = {
        int(component): len(
            set(part.transaction_src_track_id.astype(int)).union(
                set(part.transaction_dst_track_id.astype(int))
            )
        )
        for component, part in component_group
    }
    component_track_count = output.component_id.map(component_tracks).astype(float)
    component_edge_count = feature_data["component_edge_count"]
    possible_edges = component_track_count * np.maximum(component_track_count - 1.0, 1.0) / 2.0
    feature_data["component_track_count"] = component_track_count
    feature_data["component_density"] = component_edge_count / np.maximum(possible_edges, 1.0)
    feature_data["log_component_edge_count"] = np.log1p(component_edge_count)
    feature_data["log_component_track_count"] = np.log1p(component_track_count)
    feature_data["src_transaction_count"] = output.groupby(
        "transaction_src_track_id", sort=False
    ).src_chunk.transform("size").astype(float)
    feature_data["dst_transaction_count"] = output.groupby(
        "transaction_dst_track_id", sort=False
    ).src_chunk.transform("size").astype(float)

    component_group = output.groupby("component_id", sort=False)
    src_group = output.groupby("transaction_src_track_id", sort=False)
    dst_group = output.groupby("transaction_dst_track_id", sort=False)
    for column in COMPONENT_RAW_FEATURES:
        values = output[column].astype(float) if column in output else pd.Series(0.0, index=output.index)
        output[column] = values
        component_mean = component_group[column].transform("mean")
        component_max = component_group[column].transform("max")
        component_min = component_group[column].transform("min")
        component_rank = component_group[column].rank(
            method="average", pct=True
        )
        src_max = src_group[column].transform("max")
        dst_max = dst_group[column].transform("max")
        src_mean = src_group[column].transform("mean")
        dst_mean = dst_group[column].transform("mean")
        feature_data.update(
            {
                f"component_{column}_mean": component_mean,
                f"component_{column}_max": component_max,
                f"component_{column}_min": component_min,
                f"component_{column}_rank": component_rank,
                f"src_{column}_max": src_max,
                f"dst_{column}_max": dst_max,
                f"src_{column}_mean": src_mean,
                f"dst_{column}_mean": dst_mean,
                f"src_{column}_margin": values - src_max,
                f"dst_{column}_margin": values - dst_max,
            }
        )
    return pd.concat(
        [output, pd.DataFrame(feature_data, index=output.index)], axis=1
    ).copy()


def target_maximum_weight_matching(frame: pd.DataFrame) -> np.ndarray:
    positive = frame[frame[TARGET].to_numpy(float) > 0.0]
    best_by_pair: Dict[Tuple[int, int], Tuple[float, int, int, int]] = {}
    for index, row in positive.iterrows():
        left = int(row.transaction_src_track_id)
        right = int(row.transaction_dst_track_id)
        if left == right:
            continue
        key = (min(left, right), max(left, right))
        value = float(row[TARGET])
        current = best_by_pair.get(key)
        candidate = (value, int(index))
        if current is None or candidate > current:
            best_by_pair[key] = candidate
    graph = nx.Graph()
    for (left, right), (value, index) in sorted(best_by_pair.items()):
        graph.add_edge(left, right, weight=value, row_index=index)
    selected = np.zeros(len(frame), dtype=np.int8)
    if graph.number_of_edges():
        matching = nx.algorithms.matching.max_weight_matching(
            graph, maxcardinality=False, weight="weight"
        )
        index_to_position = {int(index): position for position, index in enumerate(frame.index)}
        for left, right in matching:
            row_index = int(graph[left][right]["row_index"])
            selected[index_to_position[row_index]] = 1
    return selected


def add_training_targets(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output[ORACLE_SELECTED] = target_maximum_weight_matching(output)
    output[CATASTROPHIC] = (output[TARGET].to_numpy(float) < 0.0).astype(np.int8)
    return output.copy()


def balanced_sequence_binary_weights(frame: pd.DataFrame, target: str) -> np.ndarray:
    keys = list(zip(frame.seq.astype(str), frame[target].astype(int)))
    counts: Dict[Tuple[str, int], int] = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
    groups = max(len(counts), 1)
    return np.asarray(
        [len(frame) / (groups * counts[key]) for key in keys], dtype=np.float64
    )


def sequence_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.seq.astype(str).value_counts().to_dict()
    return np.asarray(
        [len(frame) / (len(counts) * counts[str(seq)]) for seq in frame.seq],
        dtype=np.float64,
    )


def target_scale_by_sequence(frame: pd.DataFrame) -> Mapping[str, float]:
    result: Dict[str, float] = {}
    for seq, part in frame.groupby("seq", sort=False):
        positive = part.loc[part[TARGET] > 0.0, TARGET].to_numpy(float)
        result[str(seq)] = max(float(np.median(positive)) if len(positive) else 1.0, 1e-3)
    return result


def sample_training_frame(frame: pd.DataFrame, maximum: int, seed: int) -> pd.DataFrame:
    if maximum <= 0 or len(frame) <= maximum:
        return frame
    oracle = frame[frame[ORACLE_SELECTED] > 0].copy()
    remaining_budget = max(maximum - len(oracle), 0)
    negative = frame[(frame[TARGET] < 0) & (frame[ORACLE_SELECTED] == 0)].copy()
    positive = frame[(frame[TARGET] > 0) & (frame[ORACLE_SELECTED] == 0)].copy()
    neutral = frame[frame[TARGET] == 0].copy()

    negative_budget = min(len(negative), int(0.45 * remaining_budget))
    positive_budget = min(len(positive), int(0.45 * remaining_budget))
    neutral_budget = min(len(neutral), remaining_budget - negative_budget - positive_budget)
    unused = remaining_budget - negative_budget - positive_budget - neutral_budget
    if unused > 0:
        extra_positive = min(len(positive) - positive_budget, unused)
        positive_budget += extra_positive
        unused -= extra_positive
    if unused > 0:
        negative_budget += min(len(negative) - negative_budget, unused)

    if len(negative) > negative_budget:
        # Preserve both catastrophic tails and ordinary hard negatives.
        tail_budget = negative_budget // 2
        worst = negative.nsmallest(tail_budget, TARGET)
        rest = negative.drop(index=worst.index)
        random_budget = negative_budget - len(worst)
        random_part = rest.sample(random_budget, random_state=seed) if random_budget else rest.iloc[:0]
        negative = pd.concat([worst, random_part])
    if len(positive) > positive_budget:
        positive = positive.sample(positive_budget, random_state=seed + 1)
    if len(neutral) > neutral_budget:
        neutral = neutral.sample(neutral_budget, random_state=seed + 2)
    return pd.concat([oracle, negative, positive, neutral]).sort_index()


def fit_models(frame: pd.DataFrame, features: Sequence[str], seed: int, max_iter: int):
    training = sanitize(frame, features)
    selection = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=max_iter,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=18.0,
        max_bins=255,
        random_state=seed,
        early_stopping=False,
    )
    selection.fit(
        training[list(features)],
        training[ORACLE_SELECTED].astype(int),
        sample_weight=balanced_sequence_binary_weights(training, ORACLE_SELECTED),
    )
    risk = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=max_iter,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=18.0,
        max_bins=255,
        random_state=seed + 1000,
        early_stopping=False,
    )
    risk.fit(
        training[list(features)],
        training[CATASTROPHIC].astype(int),
        sample_weight=balanced_sequence_binary_weights(training, CATASTROPHIC),
    )

    scales = target_scale_by_sequence(training)
    scale_values = training.seq.astype(str).map(scales).to_numpy(float)
    positive_mask = training[TARGET].to_numpy(float) > 0.0
    negative_mask = training[TARGET].to_numpy(float) < 0.0
    if positive_mask.sum() < 20 or negative_mask.sum() < 20:
        raise RuntimeError("insufficient signed labels for component-risk models")

    gain = HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.05,
        max_iter=max_iter,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=18.0,
        max_bins=255,
        random_state=seed + 2000,
        early_stopping=False,
    )
    positive = training.loc[positive_mask]
    gain.fit(
        positive[list(features)],
        np.log1p(training.loc[positive_mask, TARGET].to_numpy(float) / scale_values[positive_mask]),
        sample_weight=sequence_weights(positive),
    )
    severity = HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.05,
        max_iter=max_iter,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=18.0,
        max_bins=255,
        random_state=seed + 3000,
        early_stopping=False,
    )
    negative = training.loc[negative_mask]
    severity.fit(
        negative[list(features)],
        np.log1p(-training.loc[negative_mask, TARGET].to_numpy(float) / scale_values[negative_mask]),
        sample_weight=sequence_weights(negative),
    )
    return selection, risk, gain, severity


def predict_models(frame: pd.DataFrame, models, features: Sequence[str]) -> pd.DataFrame:
    selection, risk, gain, severity = models
    output = sanitize(frame, features)
    matrix = output[list(features)]
    output["pred_component_selection_probability"] = selection.predict_proba(matrix)[:, 1]
    output["pred_catastrophic_probability"] = risk.predict_proba(matrix)[:, 1]
    output["pred_normalized_gain"] = np.maximum(np.expm1(gain.predict(matrix)), 0.0)
    output["pred_normalized_severity"] = np.maximum(np.expm1(severity.predict(matrix)), 0.0)
    output["pred_joint_entropy"] = 0.5 * (
        safe_entropy(output.pred_component_selection_probability.to_numpy(float))
        + safe_entropy(output.pred_catastrophic_probability.to_numpy(float))
    )
    return output


def add_policy_scores(frame: pd.DataFrame, policy: Policy) -> pd.DataFrame:
    output = frame.copy()
    selection_probability = output.pred_component_selection_probability.to_numpy(float)
    risk_probability = output.pred_catastrophic_probability.to_numpy(float)
    gain = output.pred_normalized_gain.to_numpy(float)
    severity = output.pred_normalized_severity.to_numpy(float)
    entropy = output.pred_joint_entropy.to_numpy(float)
    raw = (
        selection_probability * gain
        - policy.risk_lambda * risk_probability * severity
        - policy.uncertainty_lambda * entropy
    )
    output["raw_component_value"] = raw
    positive_raw = np.maximum(raw, 0.0)
    output["src_best_raw_value"] = pd.Series(positive_raw, index=output.index).groupby(
        output.transaction_src_track_id, sort=False
    ).transform("max")
    output["dst_best_raw_value"] = pd.Series(positive_raw, index=output.index).groupby(
        output.transaction_dst_track_id, sort=False
    ).transform("max")
    alternative_pressure = np.maximum(
        np.maximum(output.src_best_raw_value.to_numpy(float), output.dst_best_raw_value.to_numpy(float))
        - positive_raw,
        0.0,
    )
    output["alternative_pressure"] = alternative_pressure
    output["component_lcb_score"] = raw - policy.blocking_lambda * alternative_pressure
    output["component_lcb_percentile"] = output.component_lcb_score.rank(
        method="average", pct=True
    )
    return output


def select_transactions(frame: pd.DataFrame, policy: Policy) -> pd.DataFrame:
    if policy.policy_id == "noop":
        output = frame.iloc[:0].copy()
        output["component_lcb_score"] = np.asarray([], dtype=float)
        return output
    scored = add_policy_scores(frame, policy)
    eligible = scored[
        (scored.component_lcb_score.to_numpy(float) > policy.no_op_margin)
        & (
            scored.pred_component_selection_probability.to_numpy(float)
            >= policy.min_selection_probability
        )
        & (scored.component_lcb_percentile.to_numpy(float) >= policy.score_quantile)
    ].copy()
    if eligible.empty:
        return eligible

    best_by_pair: Dict[Tuple[int, int], Tuple[float, int]] = {}
    for index, row in eligible.iterrows():
        left = int(row.transaction_src_track_id)
        right = int(row.transaction_dst_track_id)
        if left == right:
            continue
        key = (min(left, right), max(left, right))
        candidate = (float(row.component_lcb_score), -int(row.src_chunk), -int(row.dst_chunk), int(index))
        current = best_by_pair.get(key)
        if current is None or candidate[:3] > current[:3]:
            best_by_pair[key] = candidate

    graph = nx.Graph()
    for (left, right), (weight, _neg_src, _neg_dst, index) in sorted(best_by_pair.items()):
        if weight > policy.no_op_margin:
            graph.add_edge(left, right, weight=weight, row_index=index)
    if graph.number_of_edges() == 0:
        return eligible.iloc[:0].copy()
    matching = nx.algorithms.matching.max_weight_matching(
        graph, maxcardinality=False, weight="weight"
    )
    indices = [int(graph[left][right]["row_index"]) for left, right in matching]
    selected = scored.loc[sorted(indices)].copy()
    selected.sort_values(
        ["component_lcb_score", "src_chunk", "dst_chunk"],
        ascending=[False, True, True],
        inplace=True,
    )
    return selected


def apply_and_write_tracker(
    seq: str,
    predictions: pd.DataFrame,
    policy: Policy,
    graph_root: Path,
    parent: Path,
    output_path: Path,
    chain,
    evaluator,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    meta = pd.read_parquet(graph_root / seq / "microtracklets.parquet")
    edges = pd.read_parquet(graph_root / seq / "candidate_edges.parquet")
    selected = select_transactions(predictions, policy)
    applied = chain.apply_transactions(
        edges,
        selected.assign(**{TARGET: selected.component_lcb_score.to_numpy(float)}),
    )
    for column, default in [
        ("assa_edge_delta_proxy", 0.0),
        ("assa_edge_positive", 0),
        ("assa_edge_negative", 0),
    ]:
        if column not in applied:
            applied[column] = default
    evaluator.DATA = graph_root
    evaluator.PARENT = parent
    report = evaluator.write_tracker(seq, meta, applied, output_path)
    report["selected_actions"] = int(len(selected))
    report["selected_score_sum"] = float(
        selected.component_lcb_score.sum() if len(selected) else 0.0
    )
    return selected, report


def evaluate_combined(
    track_results: Path,
    candidate_root: Path,
    tracker_name: str,
    sequences: Sequence[str],
) -> Dict[str, object]:
    work_dir = candidate_root / "eval_work"
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
        *sequences,
    ]
    completed = subprocess.run(
        command,
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    candidate_root.mkdir(parents=True, exist_ok=True)
    (candidate_root / "eval.log").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(completed.stdout[-5000:])
    detailed = work_dir / "eval" / tracker_name / "pedestrian_detailed.csv"
    rows = list(csv.DictReader(detailed.open(encoding="utf-8")))
    row = next(item for item in rows if item["seq"] == "COMBINED")
    by_sequence: Dict[str, Dict[str, float]] = {}
    for sequence in sequences:
        sequence_row = next(item for item in rows if item["seq"] == sequence)
        by_sequence[str(sequence)] = {
            "HOTA": 100.0 * float(sequence_row["HOTA___AUC"]),
            "DetA": 100.0 * float(sequence_row["DetA___AUC"]),
            "AssA": 100.0 * float(sequence_row["AssA___AUC"]),
            "IDSW": int(float(sequence_row["IDSW"])),
        }
    return {
        "HOTA": 100.0 * float(row["HOTA___AUC"]),
        "DetA": 100.0 * float(row["DetA___AUC"]),
        "AssA": 100.0 * float(row["AssA___AUC"]),
        "IDSW": int(float(row["IDSW"])),
        "by_sequence": by_sequence,
    }


def evaluate_single(
    track_results: Path,
    output_root: Path,
    tracker_name: str,
    sequence: str,
) -> Dict[str, float]:
    metrics = evaluate_combined(track_results, output_root, tracker_name, (sequence,))
    metrics.pop("by_sequence", None)
    return metrics


def policy_sort_key(
    row: Mapping[str, object],
) -> Tuple[float, int, int, float, float, float, float, int, int]:
    return (
        float(row["worst_fold_delta_HOTA"]),
        int(row["nonnegative_folds"]),
        int(row["positive_folds"]),
        float(row["median_fold_delta_HOTA"]),
        float(row["mean_fold_delta_HOTA"]),
        float(row["HOTA"]),
        float(row["AssA"]),
        -int(row["IDSW"]),
        -int(row["selected_actions"]),
    )


def audit_held_gt_zero(frame: pd.DataFrame) -> Dict[str, float]:
    fields = ("same_gt", "src_modal_gt", "dst_modal_gt", "label_confidence")
    result: Dict[str, float] = {}
    for field in fields:
        value = float(np.nan_to_num(frame[field].to_numpy(float), nan=0.0).sum()) if field in frame else 0.0
        result[f"held_{field}_sum"] = value
        if abs(value) > 1e-9:
            raise RuntimeError(f"held GT-derived field is non-zero before freeze: {field}={value}")
    return result


def prepare_sequence_frame(
    seq: str,
    graph_root: Path,
    label_root: Path | None,
    m25,
    base,
) -> pd.DataFrame:
    if label_root is None:
        meta = pd.read_parquet(graph_root / seq / "microtracklets.parquet")
        edges = pd.read_parquet(graph_root / seq / "candidate_edges.parquet")
        transactions = m25.structural_transactions(meta, edges)
    else:
        transactions = pd.read_parquet(
            label_root / seq / "cross_chain_transaction_utility.parquet"
        )
    base.META = graph_root
    frame = base.add_chain_features(seq, transactions)
    frame = m25.add_conflict_graph_features(frame)
    frame = add_component_features(frame)
    frame["seq"] = seq
    if label_root is not None:
        frame = add_training_targets(frame)
    return frame


def write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--held-seq", required=True, choices=SEQUENCES)
    parser.add_argument("--graph-root", required=True)
    parser.add_argument("--training-label-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--parent", default=DEFAULT_PARENT)
    parser.add_argument("--max-iter", type=int, default=160)
    parser.add_argument("--max-training-rows-per-seq", type=int, default=0)
    parser.add_argument("--policy-limit", type=int, default=0)
    parser.add_argument("--skip-outer-trackeval", action="store_true")
    args = parser.parse_args()

    held = args.held_seq
    training_sequences = [seq for seq in SEQUENCES if seq != held]
    graph_root = Path(args.graph_root)
    label_root = Path(args.training_label_root)
    output_root = Path(args.output_root)
    parent = Path(args.parent)
    output_root.mkdir(parents=True, exist_ok=True)
    policies = default_policies()
    if args.policy_limit > 0:
        policies = policies[: args.policy_limit]
    protocol = {
        "experiment": "M23-36 robust-fold catastrophic-risk-aware component matcher",
        "protocol": "strict nested sequence LOSO",
        "outer_held_sequence": held,
        "training_sequences": training_sequences,
        "training_gt_use": "outer-training transaction utilities and inner TrackEval only",
        "candidate_inference_gt_use": "none",
        "outer_held_gt_use": "final TrackEval only after tracker freeze",
        "outer_held_gt_read_before_frozen_tracker": False,
        "selection_target": "exact positive-utility maximum-weight matching membership",
        "risk_target": "negative transaction probability and magnitude",
        "component_decision": "risk-adjusted maximum-weight matching with implicit no-op",
        "policy_selection": (
            "maximize worst per-inner-sequence HOTA delta versus no-op; then "
            "nonnegative/positive fold counts, median and mean delta, COMBINED HOTA"
        ),
        "status": "running",
    }
    (output_root / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    m25 = load_module(
        "m23_36_m25", "scripts/m23_research/m23_25_sequence_calibrated_transaction_graph.py"
    )
    base = load_module(
        "m23_36_base", "scripts/m23_research/m23_12_train_chain_transaction_loso.py"
    )
    chain = load_module(
        "m23_36_chain", "scripts/m23_research/m23_12_chain_transaction_oracle.py"
    )
    evaluator = load_module(
        "m23_36_evaluator", "scripts/m23_research/m23_11_eval_utility_graph.py"
    )
    chain.DATA = graph_root
    chain.PARENT = parent
    evaluator.DATA = graph_root
    evaluator.PARENT = parent

    features = list(base.FEATURES) + list(m25.GRAPH_FEATURES) + COMPONENT_FEATURES
    training_frames: Dict[str, pd.DataFrame] = {}
    frame_reports = []
    for index, seq in enumerate(training_sequences):
        print(json.dumps({"stage": "prepare_training_frame", "seq": seq}), flush=True)
        frame = prepare_sequence_frame(seq, graph_root, label_root, m25, base)
        full_rows = len(frame)
        frame = sample_training_frame(
            frame, args.max_training_rows_per_seq, 35000 + index
        )
        training_frames[seq] = frame
        report = {
            "seq": seq,
            "full_rows": int(full_rows),
            "training_rows": int(len(frame)),
            "oracle_selected": int(frame[ORACLE_SELECTED].sum()),
            "catastrophic_negative": int(frame[CATASTROPHIC].sum()),
            "components": int(frame.component_id.nunique()),
        }
        frame_reports.append(report)
        print(json.dumps({"stage": "training_frame_ready", **report}), flush=True)

    inner_predictions: Dict[str, pd.DataFrame] = {}
    prediction_root = output_root / "inner_predictions"
    prediction_root.mkdir(parents=True, exist_ok=True)
    for fold_index, pseudo_held in enumerate(training_sequences):
        inner_train = pd.concat(
            [training_frames[seq] for seq in training_sequences if seq != pseudo_held],
            ignore_index=True,
            sort=False,
        )
        print(
            json.dumps(
                {
                    "stage": "inner_model_fit",
                    "pseudo_held": pseudo_held,
                    "training_rows": int(len(inner_train)),
                }
            ),
            flush=True,
        )
        models = fit_models(inner_train, features, 35100 + fold_index, args.max_iter)
        predicted = predict_models(training_frames[pseudo_held], models, features)
        predicted.to_parquet(prediction_root / f"{pseudo_held}.parquet", index=False)
        inner_predictions[pseudo_held] = predicted
        print(
            json.dumps(
                {
                    "stage": "inner_prediction_ready",
                    "pseudo_held": pseudo_held,
                    "rows": int(len(predicted)),
                }
            ),
            flush=True,
        )

    inner_rows: List[Dict[str, object]] = []
    noop_by_sequence: Dict[str, Dict[str, float]] | None = None
    candidates_root = output_root / "inner_candidates"
    for candidate_index, policy in enumerate(policies):
        candidate_root = candidates_root / policy.policy_id
        track_root = candidate_root / "track_results"
        track_root.mkdir(parents=True, exist_ok=True)
        selected_actions = 0
        tracker_reports = []
        for seq in training_sequences:
            selected, tracker_report = apply_and_write_tracker(
                seq,
                inner_predictions[seq],
                policy,
                graph_root,
                parent,
                track_root / f"{seq}.txt",
                chain,
                evaluator,
            )
            selected_actions += len(selected)
            tracker_reports.append(tracker_report)
        metrics = evaluate_combined(
            track_root,
            candidate_root,
            f"m23_36_inner_{candidate_index}",
            training_sequences,
        )
        by_sequence = metrics.pop("by_sequence")
        if policy.policy_id == "noop":
            noop_by_sequence = by_sequence
        if noop_by_sequence is None:
            raise RuntimeError("no-op candidate must be evaluated before non-noop policies")
        fold_deltas = {
            seq: float(by_sequence[seq]["HOTA"] - noop_by_sequence[seq]["HOTA"])
            for seq in training_sequences
        }
        delta_values = np.asarray(list(fold_deltas.values()), dtype=float)
        row = {
            "candidate_id": policy.policy_id,
            **metrics,
            "by_sequence": by_sequence,
            "fold_delta_HOTA": fold_deltas,
            "worst_fold_delta_HOTA": float(delta_values.min()),
            "median_fold_delta_HOTA": float(np.median(delta_values)),
            "mean_fold_delta_HOTA": float(delta_values.mean()),
            "positive_folds": int((delta_values > 1e-12).sum()),
            "nonnegative_folds": int((delta_values >= -1e-12).sum()),
            "selected_actions": int(selected_actions),
            "risk_lambda": policy.risk_lambda,
            "uncertainty_lambda": policy.uncertainty_lambda,
            "blocking_lambda": policy.blocking_lambda,
            "score_quantile": policy.score_quantile,
            "min_selection_probability": policy.min_selection_probability,
            "no_op_margin": policy.no_op_margin,
        }
        for seq in training_sequences:
            safe_seq = seq.replace("-", "_")
            row[f"{safe_seq}_HOTA"] = float(by_sequence[seq]["HOTA"])
            row[f"{safe_seq}_delta_HOTA"] = float(fold_deltas[seq])
        inner_rows.append(row)
        (candidate_root / "tracker_reports.json").write_text(
            json.dumps(tracker_reports, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_csv(output_root / "inner_metrics.csv", inner_rows)
        print(json.dumps({"stage": "inner_trackeval", **row}), flush=True)

    chosen_row = max(inner_rows, key=policy_sort_key)
    chosen_policy = next(
        policy for policy in policies if policy.policy_id == chosen_row["candidate_id"]
    )
    frozen = {
        "selected_policy": chosen_row,
        "policy": chosen_policy.__dict__,
        "selection_rule": (
            "maximum worst per-sequence inner HOTA delta versus no-op; then "
            "nonnegative/positive fold counts, median/mean delta, COMBINED HOTA, "
            "AssA, lower IDSW/actions"
        ),
        "outer_held_gt_read": False,
    }
    (output_root / "frozen_inner_selection.json").write_text(
        json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"stage": "policy_frozen", **chosen_row}), flush=True)

    outer_train = pd.concat(list(training_frames.values()), ignore_index=True, sort=False)
    print(
        json.dumps({"stage": "outer_model_fit", "training_rows": int(len(outer_train))}),
        flush=True,
    )
    outer_models = fit_models(outer_train, features, 35999, args.max_iter)
    held_frame = prepare_sequence_frame(held, graph_root, None, m25, base)
    held_audit = audit_held_gt_zero(held_frame)
    held_predictions = predict_models(held_frame, outer_models, features)
    held_prediction_path = output_root / "outer_predictions.parquet"
    held_predictions.to_parquet(held_prediction_path, index=False)

    outer_track_root = output_root / "outer_final" / "track_results"
    outer_track_root.mkdir(parents=True, exist_ok=True)
    selected, tracker_report = apply_and_write_tracker(
        held,
        held_predictions,
        chosen_policy,
        graph_root,
        parent,
        outer_track_root / f"{held}.txt",
        chain,
        evaluator,
    )
    selected.to_parquet(output_root / "outer_selected_transactions.parquet", index=False)
    frozen_tracker = {
        "held_sequence": held,
        "selected_policy": chosen_policy.__dict__,
        "selected_actions": int(len(selected)),
        "tracker_report": tracker_report,
        "held_audit": held_audit,
        "held_gt_read": False,
        "tracker_path": str(outer_track_root / f"{held}.txt"),
    }
    (output_root / "outer_tracker_frozen.json").write_text(
        json.dumps(frozen_tracker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"stage": "outer_tracker_frozen", **frozen_tracker}), flush=True)

    evaluation = None
    if not args.skip_outer_trackeval:
        evaluation = evaluate_single(
            outer_track_root,
            output_root / "outer_eval",
            "m23_36_robust_fold_component_matcher",
            held,
        )
        print(json.dumps({"stage": "outer_trackeval", **evaluation}), flush=True)

    baseline = STRICT_M23_25[held]
    report = {
        **protocol,
        "status": "completed" if evaluation is not None else "tracker_frozen",
        "features": features,
        "component_features": COMPONENT_FEATURES,
        "training_frame_reports": frame_reports,
        "inner_candidates": inner_rows,
        "frozen_inner_selection": frozen,
        "held_candidates": int(len(held_predictions)),
        "held_selected_actions": int(len(selected)),
        "held_audit": held_audit,
        "tracker_report": tracker_report,
        "strict_m23_25_baseline": baseline,
        "eval": evaluation,
        "delta_vs_strict_m23_25": (
            {
                "HOTA": evaluation["HOTA"] - baseline["HOTA"],
                "DetA": evaluation["DetA"] - baseline["DetA"],
                "AssA": evaluation["AssA"] - baseline["AssA"],
                "IDSW": evaluation["IDSW"] - baseline["IDSW"],
            }
            if evaluation is not None
            else None
        ),
    }
    (output_root / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    protocol["status"] = report["status"]
    (output_root / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
