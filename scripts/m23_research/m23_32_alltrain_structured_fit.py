#!/usr/bin/env python3
from __future__ import annotations

"""Fit the final structured transaction model on all MOT20 train sequences.

This script intentionally reports a *training-fit* HOTA. It is not sequence-LOSO
validation. The model uses only GT-free candidate features at inference, while
train GT supplies the structured oracle labels and the final training TrackEval.
The purpose is to verify that the deployable model class can represent the
80+ action-space solution before applying the frozen model to MOT20 test.
"""

import argparse
import csv
import importlib.util
import json
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


REPO = Path(__file__).resolve().parents[2]
SEQUENCES = ("MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05")
DEFAULT_SOURCE = (
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
DEFAULT_OUTPUT = (
    REPO
    / "outputs"
    / "mot20_m23_20260718"
    / "m23_32_alltrain_structured_fit_v1"
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


def calibrate_probability_threshold(
    probabilities: np.ndarray, labels: np.ndarray
) -> Tuple[float, Dict[str, float]]:
    positive = probabilities[labels == 1]
    negative = probabilities[labels == 0]
    if not len(positive) or not len(negative):
        raise RuntimeError("both classes are required for threshold calibration")
    min_positive = float(positive.min())
    max_negative = float(negative.max())
    if min_positive > max_negative:
        threshold = 0.5 * (min_positive + max_negative)
    else:
        candidates = np.unique(
            np.quantile(probabilities, np.linspace(0.98, 1.0, 1001))
        )
        best = None
        for threshold_value in candidates:
            predicted = probabilities >= threshold_value
            tp = int(((predicted == 1) & (labels == 1)).sum())
            fp = int(((predicted == 1) & (labels == 0)).sum())
            fn = int(((predicted == 0) & (labels == 1)).sum())
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
            key = (f1, recall, precision, float(threshold_value))
            if best is None or key > best[0]:
                best = (key, float(threshold_value))
        if best is None:
            raise RuntimeError("threshold calibration failed")
        threshold = best[1]
    predicted = probabilities >= threshold
    tp = int(((predicted == 1) & (labels == 1)).sum())
    fp = int(((predicted == 1) & (labels == 0)).sum())
    fn = int(((predicted == 0) & (labels == 1)).sum())
    return threshold, {
        "min_positive_probability": min_positive,
        "max_negative_probability": max_negative,
        "threshold": threshold,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
    }


def threshold_matching(
    frame: pd.DataFrame, probability_threshold: float
) -> pd.DataFrame:
    eligible = frame[
        frame.oracle_selection_probability.to_numpy(float) >= probability_threshold
    ].copy()
    if eligible.empty:
        eligible["policy_score"] = np.asarray([], dtype=float)
        return eligible
    eligible["policy_score"] = eligible.oracle_selection_probability.to_numpy(float)
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
    selected["policy_score"] = selected.oracle_selection_probability.to_numpy(float)
    selected.sort_values("policy_score", ascending=False, inplace=True)
    return selected


def evaluate_combined(
    track_results: Path, output_root: Path, tracker_name: str
) -> Dict[str, float]:
    work_dir = output_root / "eval_work"
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
        *SEQUENCES,
    ]
    completed = subprocess.run(
        command,
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    (output_root / "eval.log").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(completed.stdout[-5000:])
    detailed = work_dir / "eval" / tracker_name / "pedestrian_detailed.csv"
    with detailed.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    output = {}
    for row in rows:
        if row["seq"] in set(SEQUENCES) | {"COMBINED"}:
            output[row["seq"]] = {
                "HOTA": 100.0 * float(row["HOTA___AUC"]),
                "DetA": 100.0 * float(row["DetA___AUC"]),
                "AssA": 100.0 * float(row["AssA___AUC"]),
                "IDSW": int(float(row["IDSW"])),
            }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE))
    parser.add_argument("--parent", default=str(DEFAULT_PARENT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-iter", type=int, default=500)
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    parent = Path(args.parent).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    m26 = load_module("m23_32_m26", "scripts/m23_research/m23_26_prepare_test_submission.py")
    m28 = load_module("m23_32_m28", "scripts/m23_research/m23_28_structured_oracle_imitation_loso.py")
    base = load_module("m23_32_base", "scripts/m23_research/m23_12_train_chain_transaction_loso.py")
    chain = load_module("m23_32_chain", "scripts/m23_research/m23_12_chain_transaction_oracle.py")
    evaluator = load_module("m23_32_evaluator", "scripts/m23_research/m23_11_eval_utility_graph.py")

    graph_root = source_root / "train_oof_micrograph"
    prediction_root = source_root / "predictions"
    features = list(base.FEATURES) + list(m26.GRAPH_FEATURES)
    frames: Dict[str, pd.DataFrame] = {}
    for seq in SEQUENCES:
        frame = pd.read_parquet(prediction_root / f"oof_{seq}.parquet")
        frame[ORACLE_TARGET] = m28.derive_structured_oracle_label(frame)
        frames[seq] = frame

    training = pd.concat(list(frames.values()), ignore_index=True, sort=False)
    model = HistGradientBoostingClassifier(
        max_iter=args.max_iter,
        learning_rate=0.05,
        max_leaf_nodes=63,
        min_samples_leaf=10,
        l2_regularization=4.0,
        max_bins=255,
        early_stopping=False,
        random_state=32010,
    )
    model.fit(
        training[features],
        training[ORACLE_TARGET].astype(int),
        sample_weight=sequence_class_weights(training),
    )
    all_probabilities = model.predict_proba(training[features])[:, 1]
    threshold, threshold_report = calibrate_probability_threshold(
        all_probabilities,
        training[ORACLE_TARGET].to_numpy(int),
    )

    with (output_root / "structured_model.pkl").open("wb") as handle:
        pickle.dump(
            {
                "model": model,
                "features": features,
                "probability_threshold": threshold,
                "sequences": SEQUENCES,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    evaluator.DATA = graph_root
    evaluator.PARENT = parent
    evaluator.SEQS = list(SEQUENCES)
    track_root = output_root / "track_results"
    selected_root = output_root / "selected_transactions"
    prediction_output = output_root / "predictions"
    for path in (track_root, selected_root, prediction_output):
        path.mkdir(parents=True, exist_ok=True)

    sequence_reports = []
    for seq in SEQUENCES:
        frame = frames[seq].copy()
        frame["oracle_selection_probability"] = model.predict_proba(
            frame[features]
        )[:, 1]
        frame.to_parquet(prediction_output / f"{seq}.parquet", index=False)
        selected = threshold_matching(frame, threshold)
        selected.to_parquet(selected_root / f"{seq}.parquet", index=False)
        meta = pd.read_parquet(graph_root / seq / "microtracklets.parquet")
        edges = pd.read_parquet(graph_root / seq / "candidate_edges.parquet")
        applied = chain.apply_transactions(
            edges,
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
            seq,
            meta,
            applied,
            track_root / f"{seq}.txt",
        )
        oracle = selected[ORACLE_TARGET].to_numpy(int)
        report = {
            "sequence": seq,
            "selected_actions": int(len(selected)),
            "selected_oracle_actions": int(oracle.sum()),
            "selected_oracle_precision": float(oracle.mean()) if len(oracle) else 0.0,
            "selected_proxy_sum": float(selected[TARGET].sum()) if len(selected) else 0.0,
            **tracker_report,
        }
        sequence_reports.append(report)
        print(json.dumps(report, sort_keys=True), flush=True)

    metrics = evaluate_combined(
        track_root, output_root, "m23_32_alltrain_structured_fit"
    )
    report = {
        "status": "completed",
        "experiment": "M23-32 all-train structured fit",
        "score_role": "training-fit capacity/deployment score; not strict LOSO validation",
        "candidate_feature_gt_use": "none",
        "training_gt_use": "all four train sequences supply structured oracle labels",
        "test_inference_gt_free": True,
        "features": features,
        "threshold_calibration": threshold_report,
        "sequences": sequence_reports,
        "metrics": metrics,
        "target_reached": bool(metrics["COMBINED"]["HOTA"] > 80.0),
    }
    (output_root / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
