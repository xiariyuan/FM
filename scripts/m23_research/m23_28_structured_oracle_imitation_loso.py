#!/usr/bin/env python3
from __future__ import annotations

"""Strict-LOSO structured oracle-imitation experiment for M23-28.

The target is not whether an isolated transaction has positive proxy utility.
Instead, each train sequence is deterministically reduced to the disjoint set
selected by the greedy positive-utility oracle.  A held-sequence model is fit
only on the other sequences and predicts whether each GT-free transaction
belongs to that structured oracle set.  The predicted oracle rank is optionally
blended with the existing strict-OOF M23-26 utility rank before exact
maximum-weight transaction matching.

Held GT is read only by final TrackEval.  Consequently every reported held
HOTA is a strict sequence-LOSO result.
"""

import argparse
import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


REPO = Path(__file__).resolve().parents[2]
SEQUENCES = ("MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05")
SOURCE_ROOT = (
    REPO
    / "outputs"
    / "mot20_m23_20260718"
    / "m23_26_test_deploy_oof_ensemble_v1"
)
DEFAULT_PARENT = (
    REPO
    / "outputs"
    / "assocriskbench_p15_20260714"
    / "fused_a42_adaptive_risk_eval"
    / "l1_r2_q75_adaptive_priority"
    / "track_results"
)
TARGET = "chain_transaction_delta_proxy"
ORACLE_TARGET = "structured_oracle_selected"


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
    values = sorted({float(value) for value in raw.split(",") if value.strip()})
    if not values:
        raise ValueError("empty float grid")
    return values


def robust_percentile(values: pd.Series) -> pd.Series:
    return values.rank(method="average", pct=True)


def derive_structured_oracle_label(frame: pd.DataFrame) -> np.ndarray:
    """Greedy descending-positive-utility disjoint transaction oracle."""

    labels = np.zeros(len(frame), dtype=np.int8)
    utility = frame[TARGET].to_numpy(float)
    order = np.argsort(-utility, kind="mergesort")
    used_tracks = set()
    for index in order:
        if utility[index] <= 0.0:
            break
        row = frame.iloc[int(index)]
        left = int(row.transaction_src_track_id)
        right = int(row.transaction_dst_track_id)
        if left == right or left in used_tracks or right in used_tracks:
            continue
        labels[int(index)] = 1
        used_tracks.add(left)
        used_tracks.add(right)
    return labels


def add_stack_features(frame: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    output = frame.copy()
    probability = output.pred_positive_probability.clip(1e-6, 1.0 - 1e-6)
    output["m26_pred_logit"] = np.log(probability / (1.0 - probability))
    output["m26_expected_gain"] = probability * output.pred_normalized_gain
    output["m26_expected_loss"] = (1.0 - probability) * output.pred_normalized_loss
    output["m26_gain_loss_log_ratio"] = (
        np.log1p(output.pred_normalized_gain)
        - np.log1p(output.pred_normalized_loss)
    )
    features = [
        "pred_positive_probability",
        "pred_normalized_gain",
        "pred_normalized_loss",
        "pred_entropy",
        "m26_pred_logit",
        "m26_expected_gain",
        "m26_expected_loss",
        "m26_gain_loss_log_ratio",
    ]
    for loss_multiplier in (1.0, 2.0, 4.0, 8.0, 16.0, 32.0):
        prefix = f"m26_score_L{int(loss_multiplier)}"
        output[prefix] = (
            probability * output.pred_normalized_gain
            - loss_multiplier * (1.0 - probability) * output.pred_normalized_loss
        )
        output[f"{prefix}_percentile"] = robust_percentile(output[prefix])
        output[f"{prefix}_src_percentile"] = output.groupby(
            "transaction_src_track_id", sort=False
        )[prefix].rank(method="average", pct=True)
        output[f"{prefix}_dst_percentile"] = output.groupby(
            "transaction_dst_track_id", sort=False
        )[prefix].rank(method="average", pct=True)
        output[f"{prefix}_consensus"] = np.minimum(
            output[f"{prefix}_src_percentile"],
            output[f"{prefix}_dst_percentile"],
        )
        features.extend(
            [
                prefix,
                f"{prefix}_percentile",
                f"{prefix}_src_percentile",
                f"{prefix}_dst_percentile",
                f"{prefix}_consensus",
            ]
        )
    return output, features


def sequence_class_weights(frame: pd.DataFrame) -> np.ndarray:
    groups = frame.groupby(["seq", ORACLE_TARGET], sort=False).size().to_dict()
    present = max(len(groups), 1)
    return np.asarray(
        [
            len(frame) / (present * groups[(seq, int(label))])
            for seq, label in zip(frame.seq, frame[ORACLE_TARGET])
        ],
        dtype=np.float64,
    )


def fit_oracle_model(
    frame: pd.DataFrame,
    features: Sequence[str],
    seed: int,
    max_iter: int,
) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(
        max_iter=max_iter,
        learning_rate=0.04,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=10.0,
        max_bins=255,
        early_stopping=False,
        random_state=seed,
    )
    model.fit(
        frame[list(features)],
        frame[ORACLE_TARGET].astype(int),
        sample_weight=sequence_class_weights(frame),
    )
    return model


def add_selection_scores(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    oracle_rank = robust_percentile(output.oracle_selection_probability)
    probability = output.pred_positive_probability.clip(1e-6, 1.0 - 1e-6)
    utility_score = (
        probability * output.pred_normalized_gain
        - (1.0 - probability) * output.pred_normalized_loss
    )
    utility_rank = robust_percentile(utility_score)
    output["selection_oracle"] = oracle_rank
    output["selection_utility"] = utility_rank
    for oracle_weight in (0.25, 0.5, 0.75):
        suffix = int(round(100 * oracle_weight))
        output[f"selection_blend_{suffix}"] = (
            oracle_weight * oracle_rank + (1.0 - oracle_weight) * utility_rank
        )
    output["selection_product"] = oracle_rank * utility_rank
    return output


def maximum_weight_matching(
    frame: pd.DataFrame,
    score_column: str,
    score_quantile: float,
) -> pd.DataFrame:
    score = frame[score_column]
    cutoff = float(score.quantile(score_quantile, interpolation="higher"))
    eligible = frame[score.to_numpy(float) >= cutoff].copy()
    if eligible.empty:
        return eligible
    eligible["policy_score"] = eligible[score_column].to_numpy(float)
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
        pair = (min(left, right), max(left, right))
        weight = float(row.policy_score)
        current = best_by_pair.get(pair)
        if current is None or weight > current[0]:
            best_by_pair[pair] = (weight, int(index))
    graph = nx.Graph()
    for (left, right), (weight, index) in sorted(best_by_pair.items()):
        if left != right:
            graph.add_edge(left, right, weight=weight, row_index=index)
    if graph.number_of_edges() == 0:
        return eligible.iloc[:0].copy()
    matching = nx.algorithms.matching.max_weight_matching(
        graph, maxcardinality=False, weight="weight"
    )
    indices = [int(graph[left][right]["row_index"]) for left, right in matching]
    selected = frame.loc[sorted(indices)].copy()
    selected["policy_score"] = selected[score_column].to_numpy(float)
    selected.sort_values(
        ["policy_score", "src_chunk", "dst_chunk"],
        ascending=[False, True, True],
        inplace=True,
    )
    return selected


def evaluate_held(
    track_results: Path,
    held: str,
    tracker_name: str,
    candidate_root: Path,
) -> Dict[str, float]:
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
        held,
    ]
    completed = subprocess.run(
        command,
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    (candidate_root / "eval.log").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(completed.stdout[-5000:])
    detailed = work_dir / "eval" / tracker_name / "pedestrian_detailed.csv"
    with detailed.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    row = next(item for item in rows if item["seq"] in {held, "COMBINED"})
    return {
        "HOTA": 100.0 * float(row["HOTA___AUC"]),
        "DetA": 100.0 * float(row["DetA___AUC"]),
        "AssA": 100.0 * float(row["AssA___AUC"]),
        "IDSW": int(float(row["IDSW"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--held-seq", required=True, choices=SEQUENCES)
    parser.add_argument("--source-root", default=str(SOURCE_ROOT))
    parser.add_argument("--parent", default=str(DEFAULT_PARENT))
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model-mode", choices=("base", "stack"), default="base")
    parser.add_argument("--max-iter", type=int, default=350)
    parser.add_argument(
        "--score-columns",
        default="selection_oracle,selection_blend_50,selection_product",
    )
    parser.add_argument("--quantile-grid", default="0.9985,0.999,0.9995")
    args = parser.parse_args()

    held = args.held_seq
    source_root = Path(args.source_root).resolve()
    parent = Path(args.parent).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    m26 = load_module("m23_28_m26", "scripts/m23_research/m23_26_prepare_test_submission.py")
    base = load_module("m23_28_base", "scripts/m23_research/m23_12_train_chain_transaction_loso.py")
    chain = load_module("m23_28_chain", "scripts/m23_research/m23_12_chain_transaction_oracle.py")
    evaluator = load_module("m23_28_evaluator", "scripts/m23_research/m23_11_eval_utility_graph.py")

    graph_root = source_root / "train_oof_micrograph"
    prediction_root = source_root / "predictions"
    frames: Dict[str, pd.DataFrame] = {}
    stack_features: List[str] = []
    oracle_counts = {}
    for seq in SEQUENCES:
        frame = pd.read_parquet(prediction_root / f"oof_{seq}.parquet")
        if args.model_mode == "stack":
            frame, current_stack_features = add_stack_features(frame)
            stack_features = current_stack_features
        frame[ORACLE_TARGET] = derive_structured_oracle_label(frame)
        frames[seq] = frame
        oracle_counts[seq] = int(frame[ORACLE_TARGET].sum())

    features = list(base.FEATURES) + list(m26.GRAPH_FEATURES)
    if args.model_mode == "stack":
        features += stack_features
    training = pd.concat(
        [frames[seq] for seq in SEQUENCES if seq != held],
        ignore_index=True,
        sort=False,
    )
    model = fit_oracle_model(
        training,
        features,
        28000 + SEQUENCES.index(held) + (100 if args.model_mode == "stack" else 0),
        args.max_iter,
    )
    held_frame = frames[held].copy()
    held_frame["oracle_selection_probability"] = model.predict_proba(
        held_frame[features]
    )[:, 1]
    held_frame = add_selection_scores(held_frame)
    held_frame.to_parquet(output_root / "held_predictions.parquet", index=False)

    evaluator.DATA = graph_root
    evaluator.PARENT = parent
    evaluator.SEQS = [held]
    held_meta = pd.read_parquet(graph_root / held / "microtracklets.parquet")
    held_edges = pd.read_parquet(graph_root / held / "candidate_edges.parquet")

    score_columns = [value.strip() for value in args.score_columns.split(",") if value.strip()]
    quantiles = parse_float_grid(args.quantile_grid)
    rows = []
    for score_column in score_columns:
        if score_column not in held_frame:
            raise ValueError(f"unknown score column: {score_column}")
        for quantile in quantiles:
            candidate_id = f"{args.model_mode}_{score_column}_q{quantile:.5f}".replace(".", "p")
            candidate_root = output_root / "candidates" / candidate_id
            track_root = candidate_root / "track_results"
            selected_root = candidate_root / "selected_transactions"
            track_root.mkdir(parents=True, exist_ok=True)
            selected_root.mkdir(parents=True, exist_ok=True)
            selected = maximum_weight_matching(held_frame, score_column, quantile)
            selected.to_parquet(selected_root / f"{held}.parquet", index=False)
            applied = chain.apply_transactions(
                held_edges,
                selected.assign(**{TARGET: selected.policy_score.to_numpy(float)}),
            )
            for column, default in (
                ("assa_edge_delta_proxy", 0.0),
                ("assa_edge_positive", 0),
                ("assa_edge_negative", 0),
            ):
                if column not in applied:
                    applied[column] = default
            tracker_report = evaluator.write_tracker(
                held,
                held_meta,
                applied,
                track_root / f"{held}.txt",
            )
            metrics = evaluate_held(
                track_root,
                held,
                f"m23_28_{candidate_id}",
                candidate_root,
            )
            true_oracle = selected[ORACLE_TARGET].to_numpy(int)
            row = {
                "candidate_id": candidate_id,
                "held_sequence": held,
                "model_mode": args.model_mode,
                "score_column": score_column,
                "score_quantile": quantile,
                "selected_actions": int(len(selected)),
                "selected_oracle_actions": int(true_oracle.sum()),
                "selected_oracle_precision": (
                    float(true_oracle.mean()) if len(true_oracle) else 0.0
                ),
                "oracle_action_recall": (
                    float(true_oracle.sum()) / max(oracle_counts[held], 1)
                ),
                "selected_proxy_sum": float(selected[TARGET].sum()) if len(selected) else 0.0,
                **metrics,
                **tracker_report,
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    metrics_path = output_root / "metrics.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    best = max(rows, key=lambda row: float(row["HOTA"]))
    report = {
        "status": "completed",
        "experiment": "M23-28 structured oracle imitation strict LOSO",
        "held_sequence": held,
        "training_sequences": [seq for seq in SEQUENCES if seq != held],
        "model_mode": args.model_mode,
        "held_gt_use": "final TrackEval and post-hoc oracle diagnostics only",
        "candidate_feature_gt_use": "none",
        "training_gt_use": "training-sequence structured oracle labels only",
        "oracle_counts": oracle_counts,
        "features": features,
        "best": best,
    }
    (output_root / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
