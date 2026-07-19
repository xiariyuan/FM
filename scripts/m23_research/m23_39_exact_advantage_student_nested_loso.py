#!/usr/bin/env python3
from __future__ import annotations

"""M23-39: strict nested-LOSO student of exact HOTA action advantages.

Exact M23-38 labels are read only for outer-training sequences.  Every input is
GT-free and converted to a within-sequence percentile before fitting.  A sign
head and conditional gain/loss heads estimate risk-adjusted expected HOTA
advantage.  Drop and replacement actions are then solved jointly with a
maximum-weight matching and an implicit no-op.

The outer-held tracker is frozen before its GT is opened by TrackEval.
"""

import argparse
import csv
import importlib.util
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge


REPO = Path(__file__).resolve().parents[2]
SEQUENCES = ("MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05")
TARGET = "chain_transaction_delta_proxy"
DEFAULT_GRAPH_ROOT = Path("outputs/mot20_m23_20260718/micrograph_chunk30_v1")
DEFAULT_PARENT = Path(
    "outputs/assocriskbench_p15_20260714/fused_a42_adaptive_risk_eval/"
    "l1_r2_q75_adaptive_priority/track_results"
)
BASELINE_ROOTS = {
    "MOT20-01": Path("outputs/mot20_m23_20260718/m23_25_sequence_calibrated_graph_loso/MOT20-01"),
    "MOT20-02": Path("outputs/mot20_m23_20260718/m23_25_sequence_calibrated_graph_m02_full_v1"),
    "MOT20-03": Path("outputs/mot20_m23_20260718/m23_25_sequence_calibrated_graph_loso/MOT20-03"),
    "MOT20-05": Path("outputs/mot20_m23_20260718/m23_25_sequence_calibrated_graph_loso/MOT20-05"),
}
LABEL_PATHS = {
    "MOT20-01": Path("outputs/mot20_m23_20260718/m23_38_exact_hota_advantage_m01_v1/exact_action_labels.parquet"),
    "MOT20-02": Path("outputs/mot20_m23_20260718/m23_38_exact_hota_advantage_m02_v1/exact_action_labels.parquet"),
    "MOT20-03": Path("outputs/mot20_m23_20260718/m23_38_exact_hota_advantage_m03_v1/exact_action_labels.parquet"),
    "MOT20-05": Path("outputs/mot20_m23_20260718/m23_38_exact_hota_advantage_m05_v1/exact_action_labels.parquet"),
}
STRICT_M23_25 = {
    "MOT20-01": {"HOTA": 79.181480, "DetA": 81.897146, "AssA": 76.711800, "IDSW": 43},
    "MOT20-02": {"HOTA": 72.868943, "DetA": 80.713767, "AssA": 65.885675, "IDSW": 317},
    "MOT20-03": {"HOTA": 80.571616, "DetA": 81.183845, "AssA": 79.997680, "IDSW": 139},
    "MOT20-05": {"HOTA": 79.653203, "DetA": 81.955170, "AssA": 77.457994, "IDSW": 475},
}

# IDs, absolute positions, GT metrics, and post-TrackEval fields are excluded.
# Every continuous field below is ranked within its own sequence.
CONTINUOUS_FEATURES = (
    "gap", "log_gap", "appearance_cos", "forward_motion_error",
    "backward_motion_error", "motion_error_min", "motion_error_mean",
    "endpoint_displacement", "velocity_cos", "log_height_ratio", "src_rows",
    "dst_rows", "src_mapping_rate", "dst_mapping_rate", "mapping_rate_min",
    "src_consistency", "dst_consistency", "consistency_min", "src_match_iou",
    "dst_match_iou", "out_rank", "in_rank", "max_rank", "out_margin",
    "in_margin", "max_margin", "src_track_rows", "dst_track_rows",
    "src_track_chunks", "dst_track_chunks", "src_track_span", "dst_track_span",
    "src_prefix_rows", "src_suffix_rows", "dst_prefix_rows", "dst_suffix_rows",
    "merged_segment_rows", "detached_segment_rows", "src_cut_fraction",
    "dst_cut_fraction", "merged_balance", "detached_fraction",
    "segment_appearance_cos", "track_appearance_cos", "src_prefix_coherence",
    "dst_suffix_coherence", "src_suffix_coherence", "dst_prefix_coherence",
    "merged_segment_coherence", "merged_coherence_gain",
    "src_prefix_endpoint_cos", "dst_suffix_endpoint_cos", "conflict_degree",
    "log_conflict_degree", "pair_candidate_count", "src_candidate_fraction",
    "dst_candidate_fraction", "sequence_appearance_percentile",
    "sequence_segment_appearance_percentile", "sequence_motion_percentile",
    "sequence_gap_percentile", "src_appearance_percentile",
    "dst_appearance_percentile", "src_segment_appearance_percentile",
    "dst_segment_appearance_percentile", "src_motion_percentile",
    "dst_motion_percentile", "appearance_rank_consensus",
    "motion_rank_consensus", "rank_consensus", "appearance_upgrade",
    "segment_endpoint_gain", "coherence_floor", "coherence_imbalance",
    "motion_appearance_joint", "action_complexity", "conflict_per_merged_row",
    "pred_positive_probability", "pred_normalized_gain", "pred_normalized_loss",
    "pred_entropy", "removed_baseline_actions",
)
BINARY_FEATURES = (
    "action_is_drop", "same_source", "source_adjacent",
    "transaction_removes_source_out", "transaction_removes_source_in",
)
FEATURES = list(CONTINUOUS_FEATURES) + list(BINARY_FEATURES)


@dataclass(frozen=True)
class Policy:
    policy_id: str
    risk_lambda: float
    uncertainty_lambda: float
    score_quantile: float
    min_positive_probability: float
    include_drops: bool
    max_replacements: int = 12


def default_policies() -> List[Policy]:
    policies = [Policy("noop", 1.0, 0.0, 1.0, 1.0, False, 0)]
    settings = (
        (1.00, 0.00, 0.50, 0.40),
        (1.00, 0.00, 0.60, 0.45),
        (1.00, 0.00, 0.70, 0.50),
        (1.50, 0.05, 0.75, 0.52),
        (2.00, 0.10, 0.82, 0.55),
    )
    for risk, uncertainty, quantile, probability in settings:
        tag = f"r{risk:g}_u{uncertainty:g}_q{quantile:g}_p{probability:g}".replace(".", "p")
        policies.append(Policy(f"replace_{tag}", risk, uncertainty, quantile, probability, False))
        policies.append(Policy(f"combo_{tag}", risk, uncertainty, quantile, probability, True))
    return policies


def load_module(name: str, relative_path: str):
    path = REPO / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def action_nodes(row) -> set[int]:
    return {int(row.transaction_src_track_id), int(row.transaction_dst_track_id)}


def sanitize_numeric(values: pd.Series) -> pd.Series:
    output = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return output.fillna(0.0).astype(float)


def within_sequence_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["action_is_drop"] = (output.action_type.astype(str) == "drop").astype(float)
    for column in CONTINUOUS_FEATURES:
        values = sanitize_numeric(output[column]) if column in output else pd.Series(0.0, index=output.index)
        if values.nunique(dropna=False) <= 1:
            output[column] = 0.0
        else:
            output[column] = 2.0 * values.rank(method="average", pct=True) - 1.0
    for column in BINARY_FEATURES:
        if column not in output:
            output[column] = 0.0
        output[column] = sanitize_numeric(output[column]).clip(0.0, 1.0)
    return output


def sequence_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.seq.astype(str).value_counts()
    weight = frame.seq.astype(str).map({seq: 1.0 / count for seq, count in counts.items()}).to_numpy(float)
    return weight * len(weight) / weight.sum()


def balanced_sign_weights(frame: pd.DataFrame) -> np.ndarray:
    labels = (frame.delta_HOTA.to_numpy(float) > 0.0).astype(int)
    weight = sequence_weights(frame)
    for label in (0, 1):
        mask = labels == label
        if mask.any():
            weight[mask] *= len(labels) / (2.0 * mask.sum())
    return weight * len(weight) / weight.sum()


def target_scales(frame: pd.DataFrame) -> Dict[str, float]:
    scales: Dict[str, float] = {}
    for seq, part in frame.groupby("seq"):
        values = np.abs(part.delta_HOTA.to_numpy(float))
        nonzero = values[values > 1e-8]
        scale = float(np.median(nonzero)) if len(nonzero) else 0.01
        scales[str(seq)] = max(scale, 1e-4)
    return scales


def fit_models(frame: pd.DataFrame, seed: int):
    matrix = frame[FEATURES].to_numpy(float)
    sign = (frame.delta_HOTA.to_numpy(float) > 0.0).astype(int)
    sign_weight = balanced_sign_weights(frame)
    hgb_sign = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=140, max_leaf_nodes=7,
        min_samples_leaf=8, l2_regularization=8.0, early_stopping=False,
        random_state=seed,
    )
    linear_sign = LogisticRegression(
        C=0.20, class_weight="balanced", max_iter=2000, random_state=seed + 1
    )
    hgb_sign.fit(matrix, sign, sample_weight=sign_weight)
    linear_sign.fit(matrix, sign, sample_weight=sequence_weights(frame))

    scales = target_scales(frame)
    scale = frame.seq.astype(str).map(scales).to_numpy(float)
    normalized = np.log1p(np.abs(frame.delta_HOTA.to_numpy(float)) / scale)
    positive = sign > 0
    negative = ~positive
    if positive.sum() < 4 or negative.sum() < 4:
        raise RuntimeError("insufficient signed exact-HOTA labels")

    def fit_magnitude(mask: np.ndarray, offset: int):
        weights = sequence_weights(frame.loc[mask])
        hgb = HistGradientBoostingRegressor(
            loss="absolute_error", learning_rate=0.05, max_iter=120,
            max_leaf_nodes=7, min_samples_leaf=5, l2_regularization=8.0,
            early_stopping=False, random_state=seed + offset,
        )
        ridge = Ridge(alpha=20.0)
        hgb.fit(matrix[mask], normalized[mask], sample_weight=weights)
        ridge.fit(matrix[mask], normalized[mask], sample_weight=weights)
        return hgb, ridge

    gain = fit_magnitude(positive, 100)
    loss = fit_magnitude(negative, 200)
    return hgb_sign, linear_sign, gain, loss, scales


def predict_models(frame: pd.DataFrame, models) -> pd.DataFrame:
    hgb_sign, linear_sign, gain_models, loss_models, _scales = models
    output = frame.copy()
    matrix = output[FEATURES].to_numpy(float)
    p_hgb = hgb_sign.predict_proba(matrix)[:, 1]
    p_linear = linear_sign.predict_proba(matrix)[:, 1]
    probability = 0.60 * p_hgb + 0.40 * p_linear

    def magnitude(pair):
        hgb, ridge = pair
        prediction = 0.65 * hgb.predict(matrix) + 0.35 * ridge.predict(matrix)
        return np.clip(np.expm1(prediction), 0.0, 20.0)

    output["student_positive_probability"] = probability
    output["student_gain"] = magnitude(gain_models)
    output["student_loss"] = magnitude(loss_models)
    output["student_disagreement"] = np.abs(p_hgb - p_linear)
    output["student_p_hgb"] = p_hgb
    output["student_p_linear"] = p_linear
    return output


def load_training_frame(seq: str, path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame = frame[frame.status.astype(str) == "success"].copy()
    if frame.seq.astype(str).nunique() != 1 or str(frame.seq.iloc[0]) != seq:
        raise RuntimeError(f"label sequence mismatch for {seq}: {path}")
    if "delta_HOTA" not in frame:
        raise RuntimeError(f"missing exact-HOTA target: {path}")
    return within_sequence_features(frame)


def make_held_action_frame(seq: str, baseline_root: Path, label_module, max_replacements: int) -> pd.DataFrame:
    predictions = pd.read_parquet(baseline_root / "predictions" / f"{seq}_predictions.parquet")
    selected = pd.read_parquet(baseline_root / f"{seq}_selected_transactions.parquet")
    gt_fields = (
        "same_gt", "src_modal_gt", "dst_modal_gt", "src_purity", "dst_purity",
        "label_confidence",
    )
    for name, source in (("predictions", predictions), ("selected", selected)):
        for field in gt_fields:
            if field in source:
                total = float(np.nan_to_num(pd.to_numeric(source[field], errors="coerce").to_numpy(float), nan=0.0).sum())
                if abs(total) > 1e-9:
                    raise RuntimeError(f"held {name} contains nonzero GT-derived field {field}={total}")
        source.drop(columns=[field for field in gt_fields if field in source], inplace=True)
    rows: List[Dict[str, object]] = []
    for index, row in selected.iterrows():
        record = row.to_dict()
        record.update(
            seq=seq, action_type="drop", source_index=int(index),
            removed_baseline_actions=1, resulting_selected_actions=int(len(selected) - 1),
        )
        rows.append(record)
    replacements = label_module.residual_replacements(predictions, selected, max_replacements)
    for index, row in replacements.iterrows():
        removed = sum(not action_nodes(item).isdisjoint(action_nodes(row)) for item in selected.itertuples())
        record = row.to_dict()
        record.update(
            seq=seq, action_type="replace", source_index=int(index),
            removed_baseline_actions=int(removed),
            resulting_selected_actions=int(len(selected) - removed + 1),
        )
        rows.append(record)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"empty action frame for {seq}")
    forbidden = set(gt_fields) | {"exact_HOTA", "delta_HOTA"}
    leaked = sorted(forbidden.intersection(frame.columns))
    if leaked:
        raise RuntimeError(f"held action frame contains forbidden fields: {leaked}")
    return within_sequence_features(frame)


def add_policy_score(frame: pd.DataFrame, policy: Policy) -> pd.DataFrame:
    output = frame.copy()
    probability = output.student_positive_probability.to_numpy(float)
    raw = (
        probability * output.student_gain.to_numpy(float)
        - policy.risk_lambda * (1.0 - probability) * output.student_loss.to_numpy(float)
        - policy.uncertainty_lambda * output.student_disagreement.to_numpy(float)
    )
    output["student_value"] = raw
    output["student_value_percentile"] = output.groupby("action_type")["student_value"].rank(
        method="average", pct=True
    )
    return output


def select_policy_actions(frame: pd.DataFrame, policy: Policy) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if policy.policy_id == "noop":
        return frame.iloc[:0].copy(), frame.iloc[:0].copy()
    scored = add_policy_score(frame, policy)
    eligible = scored[
        (scored.student_value > 0.0)
        & (scored.student_positive_probability >= policy.min_positive_probability)
        & (scored.student_value_percentile >= policy.score_quantile)
    ].copy()
    drops = eligible[(eligible.action_type == "drop") & policy.include_drops].copy()
    replacements = eligible[eligible.action_type == "replace"].copy()
    if replacements.empty:
        return drops, replacements
    best_by_pair: Dict[Tuple[int, int], Tuple[float, int]] = {}
    for index, row in replacements.iterrows():
        left, right = sorted(action_nodes(row))
        if left == right:
            continue
        candidate = (float(row.student_value), int(index))
        current = best_by_pair.get((left, right))
        if current is None or candidate[0] > current[0]:
            best_by_pair[(left, right)] = candidate
    graph = nx.Graph()
    for (left, right), (weight, index) in best_by_pair.items():
        graph.add_edge(left, right, weight=weight, row_index=index)
    matching = nx.algorithms.matching.max_weight_matching(graph, maxcardinality=False, weight="weight")
    indices = [int(graph[left][right]["row_index"]) for left, right in matching]
    replacements = scored.loc[indices].nlargest(policy.max_replacements, "student_value") if indices else scored.iloc[:0].copy()
    return drops, replacements


def build_selected_transactions(
    baseline_selected: pd.DataFrame,
    original_predictions: pd.DataFrame,
    drops: pd.DataFrame,
    replacements: pd.DataFrame,
) -> Tuple[pd.DataFrame, int]:
    drop_indices = set(drops.source_index.astype(int)) if len(drops) else set()
    replacement_nodes: set[int] = set()
    for row in replacements.itertuples():
        replacement_nodes.update(action_nodes(row))
    keep = []
    for index, row in baseline_selected.iterrows():
        keep.append(int(index) not in drop_indices and action_nodes(row).isdisjoint(replacement_nodes))
    retained = baseline_selected[np.asarray(keep, dtype=bool)].copy()
    added = []
    for row in replacements.itertuples():
        candidate = original_predictions.loc[int(row.source_index)].copy()
        candidate[TARGET] = float(row.student_value)
        added.append(candidate)
    output = pd.concat([retained, pd.DataFrame(added)], ignore_index=True, sort=False) if added else retained.reset_index(drop=True)
    if TARGET not in output:
        output[TARGET] = 1.0
    output[TARGET] = output[TARGET].fillna(1.0).astype(float)
    return output, int(len(baseline_selected) - len(retained))


def apply_and_write_tracker(
    seq: str,
    action_predictions: pd.DataFrame,
    policy: Policy,
    graph_root: Path,
    parent: Path,
    output_path: Path,
    chain,
    evaluator,
) -> Dict[str, object]:
    baseline_root = BASELINE_ROOTS[seq]
    original = pd.read_parquet(baseline_root / "predictions" / f"{seq}_predictions.parquet")
    baseline_selected = pd.read_parquet(baseline_root / f"{seq}_selected_transactions.parquet")
    drops, replacements = select_policy_actions(action_predictions, policy)
    selected, removed = build_selected_transactions(baseline_selected, original, drops, replacements)
    edges = pd.read_parquet(graph_root / seq / "candidate_edges.parquet")
    meta = pd.read_parquet(graph_root / seq / "microtracklets.parquet")
    applied = chain.apply_transactions(edges, selected)
    for column, default in (("assa_edge_delta_proxy", 0.0), ("assa_edge_positive", 0), ("assa_edge_negative", 0)):
        if column not in applied:
            applied[column] = default
    evaluator.DATA = graph_root
    evaluator.PARENT = parent
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tracker_report = evaluator.write_tracker(seq, meta, applied, output_path)
    return {
        "seq": seq, "drop_actions": int(len(drops)),
        "replacement_actions": int(len(replacements)),
        "removed_baseline_actions": removed,
        "resulting_selected_actions": int(len(selected)),
        "selected_value_sum": float(drops.get("student_value", pd.Series(dtype=float)).sum() + replacements.get("student_value", pd.Series(dtype=float)).sum()),
        "tracker_report": tracker_report,
    }


def evaluate_combined(track_results: Path, output_root: Path, tracker_name: str, sequences: Sequence[str]) -> Dict[str, float]:
    work_dir = output_root / "eval_work"
    command = [
        sys.executable, "scripts/eval_motstyle_trackeval.py", "--benchmark-name", "MOT20",
        "--split-to-eval", "train", "--gt-root", "datasets/MOT20/train",
        "--results-dir", str(track_results), "--tracker-name", tracker_name,
        "--work-dir", str(work_dir), "--seqs", *sequences,
    ]
    completed = subprocess.run(command, cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "eval.log").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(completed.stdout[-5000:])
    detailed = work_dir / "eval" / tracker_name / "pedestrian_detailed.csv"
    rows = list(csv.DictReader(detailed.open(encoding="utf-8")))
    row = next(item for item in rows if item["seq"] == "COMBINED")
    return {
        "HOTA": 100.0 * float(row["HOTA___AUC"]),
        "DetA": 100.0 * float(row["DetA___AUC"]),
        "AssA": 100.0 * float(row["AssA___AUC"]),
        "IDSW": int(float(row["IDSW"])),
    }


def write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)


def policy_sort_key(row: Mapping[str, object]):
    return float(row["HOTA"]), float(row["AssA"]), -int(row["IDSW"]), -int(row["selected_actions"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--held-seq", required=True, choices=SEQUENCES)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--graph-root", default=str(DEFAULT_GRAPH_ROOT))
    parser.add_argument("--parent", default=str(DEFAULT_PARENT))
    parser.add_argument("--max-replacements", type=int, default=64)
    parser.add_argument("--policy-limit", type=int, default=0)
    parser.add_argument("--skip-outer-trackeval", action="store_true")
    args = parser.parse_args()

    held = args.held_seq
    training_sequences = [seq for seq in SEQUENCES if seq != held]
    output_root = Path(args.output_root)
    graph_root = Path(args.graph_root)
    parent = Path(args.parent)
    output_root.mkdir(parents=True, exist_ok=True)
    policies = default_policies()
    if args.policy_limit > 0:
        policies = policies[: args.policy_limit]
    protocol = {
        "experiment": "M23-39 exact-HOTA advantage student",
        "protocol": "strict nested sequence LOSO",
        "outer_held_sequence": held,
        "outer_training_sequences": training_sequences,
        "training_gt_use": "M23-38 exact action labels and inner TrackEval on outer-training sequences only",
        "held_candidate_gt_use": "none",
        "held_gt_read_before_tracker_freeze": False,
        "features": FEATURES,
        "max_replacements": args.max_replacements,
        "status": "running",
    }
    (output_root / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    label_module = load_module("m23_39_labels", "scripts/m23_research/m23_38_exact_hota_advantage_labels.py")
    chain = load_module("m23_39_chain", "scripts/m23_research/m23_12_chain_transaction_oracle.py")
    evaluator = load_module("m23_39_evaluator", "scripts/m23_research/m23_11_eval_utility_graph.py")
    training_frames = {seq: load_training_frame(seq, LABEL_PATHS[seq]) for seq in training_sequences}

    inner_predictions: Dict[str, pd.DataFrame] = {}
    for fold_index, pseudo_held in enumerate(training_sequences):
        train = pd.concat([training_frames[seq] for seq in training_sequences if seq != pseudo_held], ignore_index=True, sort=False)
        models = fit_models(train, 39000 + fold_index)
        prediction = predict_models(training_frames[pseudo_held], models)
        inner_predictions[pseudo_held] = prediction
        path = output_root / "inner_predictions" / f"{pseudo_held}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        prediction.to_parquet(path, index=False)
        print(json.dumps({"stage": "inner_prediction", "pseudo_held": pseudo_held, "rows": len(prediction)}), flush=True)

    inner_rows: List[Dict[str, object]] = []
    for policy_index, policy in enumerate(policies):
        track_root = output_root / "inner_candidates" / policy.policy_id / "track_results"
        reports = []
        for seq in training_sequences:
            reports.append(apply_and_write_tracker(
                seq, inner_predictions[seq], policy, graph_root, parent,
                track_root / f"{seq}.txt", chain, evaluator,
            ))
        metrics = evaluate_combined(
            track_root, output_root / "inner_candidates" / policy.policy_id,
            f"m23_39_inner_{held}_{policy_index}", training_sequences,
        )
        row = {
            "policy_id": policy.policy_id, **metrics,
            "selected_actions": int(sum(item["drop_actions"] + item["replacement_actions"] for item in reports)),
            "risk_lambda": policy.risk_lambda,
            "uncertainty_lambda": policy.uncertainty_lambda,
            "score_quantile": policy.score_quantile,
            "min_positive_probability": policy.min_positive_probability,
            "include_drops": policy.include_drops,
        }
        inner_rows.append(row)
        write_csv(output_root / "inner_metrics.csv", inner_rows)
        print(json.dumps({"stage": "inner_trackeval", **row}), flush=True)

    chosen_row = max(inner_rows, key=policy_sort_key)
    chosen_policy = next(policy for policy in policies if policy.policy_id == chosen_row["policy_id"])
    frozen_selection = {
        "selected_policy": chosen_policy.__dict__, "selected_inner_metrics": chosen_row,
        "selection_rule": "maximum exact inner COMBINED HOTA, then AssA, lower IDSW/actions",
        "outer_held_gt_read": False,
    }
    (output_root / "frozen_inner_selection.json").write_text(json.dumps(frozen_selection, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"stage": "policy_frozen", **chosen_row}), flush=True)

    outer_train = pd.concat(list(training_frames.values()), ignore_index=True, sort=False)
    outer_models = fit_models(outer_train, 39999)
    held_actions = make_held_action_frame(held, BASELINE_ROOTS[held], label_module, args.max_replacements)
    held_predictions = predict_models(held_actions, outer_models)
    held_predictions.to_parquet(output_root / "outer_action_predictions.parquet", index=False)
    outer_track_root = output_root / "outer_final" / "track_results"
    tracker_report = apply_and_write_tracker(
        held, held_predictions, chosen_policy, graph_root, parent,
        outer_track_root / f"{held}.txt", chain, evaluator,
    )
    frozen_tracker = {
        "held_sequence": held, "held_label_file_read": False,
        "selected_policy": chosen_policy.__dict__, "tracker_report": tracker_report,
        "tracker_path": str(outer_track_root / f"{held}.txt"),
    }
    (output_root / "outer_tracker_frozen.json").write_text(json.dumps(frozen_tracker, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"stage": "outer_tracker_frozen", **frozen_tracker}), flush=True)

    evaluation = None
    if not args.skip_outer_trackeval:
        evaluation = evaluate_combined(
            outer_track_root, output_root / "outer_eval", f"m23_39_outer_{held}", (held,)
        )
        print(json.dumps({"stage": "outer_trackeval", **evaluation}), flush=True)
    baseline = STRICT_M23_25[held]
    report = {
        **protocol, "status": "completed" if evaluation else "tracker_frozen",
        "training_rows": {seq: int(len(frame)) for seq, frame in training_frames.items()},
        "inner_candidates": inner_rows, "frozen_inner_selection": frozen_selection,
        "held_candidate_actions": int(len(held_predictions)),
        "tracker_report": tracker_report, "eval": evaluation,
        "strict_m23_25_baseline": baseline,
        "delta_vs_strict_m23_25": ({
            "HOTA": evaluation["HOTA"] - baseline["HOTA"],
            "DetA": evaluation["DetA"] - baseline["DetA"],
            "AssA": evaluation["AssA"] - baseline["AssA"],
            "IDSW": evaluation["IDSW"] - baseline["IDSW"],
        } if evaluation else None),
    }
    (output_root / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    protocol["status"] = report["status"]
    (output_root / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"stage": "completed", "held": held, "eval": evaluation, "delta": report["delta_vs_strict_m23_25"]}), flush=True)


if __name__ == "__main__":
    main()
