#!/usr/bin/env python3
from __future__ import annotations

"""M23-45 strict nested-LOSO student for parent-track source-boundary cuts.

The deployable baseline is the already frozen M23-39 outer tracker for each
sequence.  Exact M23-45 teacher labels are allowed only for outer-training
sequences.  The outer-held sequence contributes only a GT-free candidate
shortlist built from its frozen applied graph.  A policy is selected with inner
exact TrackEval by maximizing the worst per-sequence HOTA delta relative to
no-op, then the mean delta and stitched inner HOTA.  The held tracker is written
and frozen before the held GT is opened for the final TrackEval.
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

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge


REPO = Path(__file__).resolve().parents[2]
SEQUENCES = ("MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05")
DEFAULT_GRAPH_ROOT = Path("outputs/mot20_m23_20260718/micrograph_chunk30_v1")
DEFAULT_PARENT = Path(
    "outputs/assocriskbench_p15_20260714/fused_a42_adaptive_risk_eval/"
    "l1_r2_q75_adaptive_priority/track_results"
)
DEFAULT_BASELINE_CACHE = Path(
    "outputs/mot20_m23_20260718/m23_45_m23_39_deployable_baseline_cache_v1"
)
LABEL_PATHS = {
    "MOT20-01": Path("outputs/mot20_m23_20260718/m23_45_source_cut_teacher_01_v1/exact_source_cut_labels.parquet"),
    "MOT20-02": Path("outputs/mot20_m23_20260718/m23_45_source_cut_teacher_02_v1/exact_source_cut_labels.parquet"),
    "MOT20-03": Path("outputs/mot20_m23_20260718/m23_45_source_cut_teacher_03_v1/exact_source_cut_labels.parquet"),
    "MOT20-05": Path("outputs/mot20_m23_20260718/m23_45_source_cut_teacher_05_v1/exact_source_cut_labels.parquet"),
}

CONTINUOUS_FEATURES = (
    "gap", "log_gap", "appearance_cos", "forward_motion_error",
    "backward_motion_error", "motion_error_min", "motion_error_mean",
    "endpoint_displacement", "velocity_cos", "log_height_ratio", "src_rows",
    "dst_rows", "src_mapping_rate", "dst_mapping_rate", "mapping_rate_min",
    "src_consistency", "dst_consistency", "consistency_min", "src_match_iou",
    "dst_match_iou", "out_rank", "in_rank", "max_rank", "out_margin",
    "in_margin", "max_margin", "chain_rows", "chain_chunks", "prefix_rows",
    "suffix_rows", "min_side_rows", "cut_fraction", "cut_balance",
    "log_chain_rows", "log_min_side_rows", "risk_low_appearance",
    "risk_high_motion", "risk_high_rank", "risk_low_consistency",
    "risk_low_mapping", "impact_percentile", "boundary_disagreement",
    "source_cut_policy_score", "selection_rank",
)
BINARY_FEATURES = ("same_source", "source_adjacent")
CHANNELS = (
    "composite", "low_appearance", "high_motion", "high_rank",
    "low_consistency", "large_impact", "disagreement",
)
FEATURES = list(CONTINUOUS_FEATURES) + list(BINARY_FEATURES) + [
    f"channel_{name}" for name in CHANNELS
]
FORBIDDEN_HELD_FIELDS = {
    "same_gt", "src_modal_gt", "dst_modal_gt", "src_purity", "dst_purity",
    "label_confidence", "exact_HOTA", "exact_DetA", "exact_AssA",
    "delta_HOTA", "delta_DetA", "delta_AssA", "changed_raw_rows",
    "affected_frames", "changed_processed_detections", "teacher_seconds",
}


@dataclass(frozen=True)
class Policy:
    policy_id: str
    risk_lambda: float
    uncertainty_lambda: float
    score_quantile: float
    min_positive_probability: float
    max_cuts: int


def default_policies() -> List[Policy]:
    return [
        Policy("noop", 0.0, 0.0, 1.0, 1.0, 0),
        Policy("r0p5_u0_q0p75_p0p45_k16", 0.5, 0.00, 0.75, 0.45, 16),
        Policy("r0p75_u0p05_q0p80_p0p50_k12", 0.75, 0.05, 0.80, 0.50, 12),
        Policy("r1_u0p10_q0p85_p0p52_k10", 1.0, 0.10, 0.85, 0.52, 10),
        Policy("r1p5_u0p15_q0p90_p0p55_k8", 1.5, 0.15, 0.90, 0.55, 8),
        Policy("r2_u0p20_q0p93_p0p58_k6", 2.0, 0.20, 0.93, 0.58, 6),
        Policy("r3_u0p25_q0p95_p0p60_k4", 3.0, 0.25, 0.95, 0.60, 4),
        Policy("r1_u0p10_q0p90_p0p60_k6", 1.0, 0.10, 0.90, 0.60, 6),
        Policy("r2_u0p20_q0p95_p0p65_k4", 2.0, 0.20, 0.95, 0.65, 4),
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


def numeric(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)


def within_sequence_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in CONTINUOUS_FEATURES:
        values = numeric(output[column]) if column in output else pd.Series(0.0, index=output.index)
        fill = float(values.median()) if values.notna().any() else 0.0
        values = values.fillna(fill)
        if values.nunique(dropna=False) <= 1:
            output[column] = 0.0
        else:
            output[column] = 2.0 * values.rank(method="average", pct=True) - 1.0
    for column in BINARY_FEATURES:
        values = numeric(output[column]) if column in output else pd.Series(0.0, index=output.index)
        output[column] = values.fillna(0.0).clip(0.0, 1.0)
    channel = output.get("selection_channel", pd.Series("", index=output.index)).astype(str)
    for name in CHANNELS:
        output[f"channel_{name}"] = (channel == name).astype(float)
    return output


def sequence_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.seq.astype(str).value_counts().to_dict()
    weights = np.asarray([1.0 / counts[str(seq)] for seq in frame.seq], dtype=float)
    return weights * len(weights) / weights.sum()


def balanced_sign_weights(frame: pd.DataFrame) -> np.ndarray:
    labels = (frame.delta_HOTA.to_numpy(float) > 0.0).astype(int)
    weights = sequence_weights(frame)
    for label in (0, 1):
        mask = labels == label
        if mask.any():
            weights[mask] *= len(labels) / (2.0 * mask.sum())
    return weights * len(weights) / weights.sum()


def target_scales(frame: pd.DataFrame) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for seq, part in frame.groupby("seq", sort=False):
        values = np.abs(part.delta_HOTA.to_numpy(float))
        nonzero = values[values > 1e-8]
        result[str(seq)] = max(float(np.median(nonzero)) if len(nonzero) else 0.01, 1e-4)
    return result


def load_training_frame(seq: str, path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame = frame[frame.status.astype(str) == "success"].copy()
    if frame.empty or frame.seq.astype(str).nunique() != 1 or str(frame.seq.iloc[0]) != seq:
        raise RuntimeError(f"invalid training labels for {seq}: {path}")
    if "delta_HOTA" not in frame:
        raise RuntimeError(f"missing exact HOTA target: {path}")
    frame["seq"] = seq
    return within_sequence_features(frame)


def fit_models(frame: pd.DataFrame, seed: int):
    matrix = frame[FEATURES].to_numpy(float)
    sign = (frame.delta_HOTA.to_numpy(float) > 0.0).astype(int)
    if len(np.unique(sign)) < 2:
        raise RuntimeError("source-cut training fold has only one sign")
    sign_weights = balanced_sign_weights(frame)
    hgb_sign = HistGradientBoostingClassifier(
        learning_rate=0.04, max_iter=180, max_leaf_nodes=5,
        min_samples_leaf=8, l2_regularization=12.0,
        early_stopping=False, random_state=seed,
    )
    linear_sign = LogisticRegression(
        C=0.15, class_weight="balanced", max_iter=3000, random_state=seed + 1,
    )
    hgb_sign.fit(matrix, sign, sample_weight=sign_weights)
    linear_sign.fit(matrix, sign, sample_weight=sequence_weights(frame))

    scales = target_scales(frame)
    scale = frame.seq.astype(str).map(scales).to_numpy(float)
    normalized = np.log1p(np.abs(frame.delta_HOTA.to_numpy(float)) / scale)

    def magnitude(mask: np.ndarray, offset: int):
        if mask.sum() < 4:
            raise RuntimeError("insufficient signed source-cut labels")
        sub = frame.loc[mask]
        weights = sequence_weights(sub)
        hgb = HistGradientBoostingRegressor(
            loss="absolute_error", learning_rate=0.04, max_iter=160,
            max_leaf_nodes=5, min_samples_leaf=6, l2_regularization=12.0,
            early_stopping=False, random_state=seed + offset,
        )
        ridge = Ridge(alpha=30.0)
        hgb.fit(matrix[mask], normalized[mask], sample_weight=weights)
        ridge.fit(matrix[mask], normalized[mask], sample_weight=weights)
        return hgb, ridge

    gain = magnitude(sign > 0, 100)
    loss = magnitude(sign == 0, 200)
    return hgb_sign, linear_sign, gain, loss


def predict_models(frame: pd.DataFrame, models) -> pd.DataFrame:
    hgb_sign, linear_sign, gain_models, loss_models = models
    output = frame.copy()
    matrix = output[FEATURES].to_numpy(float)
    p_hgb = hgb_sign.predict_proba(matrix)[:, 1]
    p_linear = linear_sign.predict_proba(matrix)[:, 1]
    probability = 0.55 * p_hgb + 0.45 * p_linear

    def magnitude(pair):
        hgb, ridge = pair
        prediction = 0.60 * hgb.predict(matrix) + 0.40 * ridge.predict(matrix)
        return np.clip(np.expm1(prediction), 0.0, 30.0)

    output["student_positive_probability"] = probability
    output["student_gain"] = magnitude(gain_models)
    output["student_loss"] = magnitude(loss_models)
    output["student_disagreement"] = np.abs(p_hgb - p_linear)
    output["student_p_hgb"] = p_hgb
    output["student_p_linear"] = p_linear
    return output


def add_policy_score(frame: pd.DataFrame, policy: Policy) -> pd.DataFrame:
    output = frame.copy()
    probability = output.student_positive_probability.to_numpy(float)
    output["student_value"] = (
        probability * output.student_gain.to_numpy(float)
        - policy.risk_lambda * (1.0 - probability) * output.student_loss.to_numpy(float)
        - policy.uncertainty_lambda * output.student_disagreement.to_numpy(float)
    )
    output["student_value_percentile"] = output.student_value.rank(method="average", pct=True)
    return output


def select_cuts(frame: pd.DataFrame, policy: Policy) -> pd.DataFrame:
    if policy.policy_id == "noop":
        return frame.iloc[:0].copy()
    scored = add_policy_score(frame, policy)
    eligible = scored[
        (scored.student_value > 0.0)
        & (scored.student_positive_probability >= policy.min_positive_probability)
        & (scored.student_value_percentile >= policy.score_quantile)
    ].copy()
    if eligible.empty:
        return eligible
    return eligible.nlargest(policy.max_cuts, ["student_value", "student_positive_probability"])


def held_candidate_frame(seq: str, baseline_cache: Path, graph_root: Path, m43) -> pd.DataFrame:
    applied = pd.read_parquet(baseline_cache / seq / "frozen_applied_edges.parquet")
    meta = pd.read_parquet(graph_root / seq / "microtracklets.parquet")
    candidates = m43.candidate_source_cuts(applied, meta, 96)
    leaked = sorted(FORBIDDEN_HELD_FIELDS.intersection(candidates.columns))
    if leaked:
        raise RuntimeError(f"held cut candidates contain forbidden fields: {leaked}")
    candidates["seq"] = seq
    return within_sequence_features(candidates)


def apply_and_write_tracker(
    seq: str,
    predictions: pd.DataFrame,
    policy: Policy,
    baseline_cache: Path,
    graph_root: Path,
    parent: Path,
    output_path: Path,
    evaluator,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    applied = pd.read_parquet(baseline_cache / seq / "frozen_applied_edges.parquet")
    meta = pd.read_parquet(graph_root / seq / "microtracklets.parquet")
    chosen = select_cuts(predictions, policy)
    indices = chosen.source_index.astype(int).tolist() if len(chosen) else []
    if len(indices) != len(set(indices)):
        raise RuntimeError("duplicate source-cut indices")
    modified = applied.drop(index=indices).copy()
    evaluator.DATA = graph_root
    evaluator.PARENT = parent
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tracker_report = evaluator.write_tracker(seq, meta, modified, output_path)
    return chosen, {
        "seq": seq,
        "selected_cuts": int(len(chosen)),
        "selected_source_indices": indices,
        "selected_value_sum": float(chosen.student_value.sum()) if len(chosen) else 0.0,
        "remaining_applied_edges": int(len(modified)),
        "tracker_report": tracker_report,
    }


def evaluate_detailed(
    track_results: Path,
    output_root: Path,
    tracker_name: str,
    sequences: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    work_dir = output_root / "eval_work"
    command = [
        sys.executable, "scripts/eval_motstyle_trackeval.py",
        "--benchmark-name", "MOT20", "--split-to-eval", "train",
        "--gt-root", "datasets/MOT20/train", "--results-dir", str(track_results),
        "--tracker-name", tracker_name, "--work-dir", str(work_dir),
        "--seqs", *sequences,
    ]
    completed = subprocess.run(
        command, cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "eval.log").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(completed.stdout[-5000:])
    detailed = work_dir / "eval" / tracker_name / "pedestrian_detailed.csv"
    result: Dict[str, Dict[str, float]] = {}
    for row in csv.DictReader(detailed.open(encoding="utf-8")):
        if row["seq"] not in set(sequences) | {"COMBINED"}:
            continue
        result[row["seq"]] = {
            "HOTA": 100.0 * float(row["HOTA___AUC"]),
            "DetA": 100.0 * float(row["DetA___AUC"]),
            "AssA": 100.0 * float(row["AssA___AUC"]),
            "IDSW": int(float(row["IDSW"])),
        }
    missing = (set(sequences) | {"COMBINED"}) - set(result)
    if missing:
        raise RuntimeError(f"missing TrackEval rows: {sorted(missing)}")
    return result


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


def robust_sort_key(row: Mapping[str, object]) -> Tuple[float, float, float, float, int]:
    return (
        float(row["worst_fold_delta_HOTA"]),
        float(row["mean_fold_delta_HOTA"]),
        float(row["HOTA"]),
        float(row["AssA"]),
        -int(row["selected_actions"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--held-seq", required=True, choices=SEQUENCES)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--baseline-cache", default=str(DEFAULT_BASELINE_CACHE))
    parser.add_argument("--graph-root", default=str(DEFAULT_GRAPH_ROOT))
    parser.add_argument("--parent", default=str(DEFAULT_PARENT))
    parser.add_argument("--policy-limit", type=int, default=0)
    parser.add_argument("--skip-outer-trackeval", action="store_true")
    args = parser.parse_args()

    held = args.held_seq
    training_sequences = [seq for seq in SEQUENCES if seq != held]
    output_root = Path(args.output_root)
    baseline_cache = Path(args.baseline_cache)
    graph_root = Path(args.graph_root)
    parent = Path(args.parent)
    output_root.mkdir(parents=True, exist_ok=True)
    policies = default_policies()
    if args.policy_limit > 0:
        policies = policies[: args.policy_limit]

    protocol = {
        "experiment": "M23-45 domain-robust source-cut student",
        "protocol": "strict nested sequence LOSO",
        "outer_held_sequence": held,
        "outer_training_sequences": training_sequences,
        "training_gt_use": "exact source-cut labels from outer-training sequences only",
        "held_candidate_gt_use": "none",
        "policy_selection": "maximize worst per-inner-sequence HOTA delta, then mean delta and stitched HOTA",
        "held_label_file_read": False,
        "held_gt_read_before_tracker_freeze": False,
        "deployable": True,
        "status": "running",
    }
    (output_root / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")

    m43 = load_module("m23_45_candidates", "scripts/m23_research/m23_43_source_boundary_cut_teacher.py")
    evaluator = load_module("m23_45_evaluator", "scripts/m23_research/m23_11_eval_utility_graph.py")

    training_frames = {seq: load_training_frame(seq, LABEL_PATHS[seq]) for seq in training_sequences}
    frame_report = {
        seq: {
            "rows": int(len(frame)),
            "positive": int((frame.delta_HOTA > 0.0).sum()),
            "negative": int((frame.delta_HOTA < 0.0).sum()),
            "zero": int((frame.delta_HOTA == 0.0).sum()),
        }
        for seq, frame in training_frames.items()
    }
    print(json.dumps({"stage": "training_labels_ready", "frames": frame_report}), flush=True)

    inner_predictions: Dict[str, pd.DataFrame] = {}
    for fold_index, pseudo_held in enumerate(training_sequences):
        train = pd.concat(
            [training_frames[seq] for seq in training_sequences if seq != pseudo_held],
            ignore_index=True, sort=False,
        )
        models = fit_models(train, 45000 + fold_index)
        prediction = predict_models(training_frames[pseudo_held], models)
        inner_predictions[pseudo_held] = prediction
        path = output_root / "inner_predictions" / f"{pseudo_held}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        prediction.to_parquet(path, index=False)
        print(json.dumps({"stage": "inner_prediction", "pseudo_held": pseudo_held, "rows": len(prediction)}), flush=True)

    inner_rows: List[Dict[str, object]] = []
    baseline_by_seq: Dict[str, Dict[str, float]] | None = None
    for policy_index, policy in enumerate(policies):
        candidate_root = output_root / "inner_candidates" / policy.policy_id
        track_root = candidate_root / "track_results"
        reports = []
        selected_actions = 0
        for seq in training_sequences:
            chosen, tracker_report = apply_and_write_tracker(
                seq, inner_predictions[seq], policy, baseline_cache, graph_root, parent,
                track_root / f"{seq}.txt", evaluator,
            )
            selected_actions += len(chosen)
            reports.append(tracker_report)
        metrics = evaluate_detailed(
            track_root, candidate_root, f"m23_45_inner_{held}_{policy_index}", training_sequences,
        )
        if policy.policy_id == "noop":
            baseline_by_seq = {seq: metrics[seq] for seq in training_sequences}
        if baseline_by_seq is None:
            raise RuntimeError("no-op policy must be evaluated first")
        deltas = {seq: metrics[seq]["HOTA"] - baseline_by_seq[seq]["HOTA"] for seq in training_sequences}
        row: Dict[str, object] = {
            "policy_id": policy.policy_id,
            **metrics["COMBINED"],
            "selected_actions": int(selected_actions),
            "risk_lambda": policy.risk_lambda,
            "uncertainty_lambda": policy.uncertainty_lambda,
            "score_quantile": policy.score_quantile,
            "min_positive_probability": policy.min_positive_probability,
            "max_cuts": policy.max_cuts,
            "worst_fold_delta_HOTA": float(min(deltas.values())),
            "mean_fold_delta_HOTA": float(np.mean(list(deltas.values()))),
            "positive_inner_folds": int(sum(value > 0.0 for value in deltas.values())),
        }
        for seq in training_sequences:
            tag = seq.replace("MOT20-", "M")
            row[f"{tag}_HOTA"] = float(metrics[seq]["HOTA"])
            row[f"{tag}_delta_HOTA"] = float(deltas[seq])
        inner_rows.append(row)
        (candidate_root / "tracker_reports.json").write_text(json.dumps(reports, indent=2, sort_keys=True) + "\n")
        write_csv(output_root / "inner_metrics.csv", inner_rows)
        print(json.dumps({"stage": "inner_trackeval", **row}), flush=True)

    chosen_row = max(inner_rows, key=robust_sort_key)
    chosen_policy = next(policy for policy in policies if policy.policy_id == chosen_row["policy_id"])
    frozen_selection = {
        "selected_policy": chosen_policy.__dict__,
        "selected_inner_metrics": chosen_row,
        "selection_rule": "maximum worst-fold HOTA delta; then mean delta, stitched HOTA, AssA, fewer actions",
        "outer_held_gt_read": False,
        "held_label_file_read": False,
    }
    (output_root / "frozen_inner_selection.json").write_text(json.dumps(frozen_selection, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"stage": "policy_frozen", **chosen_row}), flush=True)

    outer_train = pd.concat(list(training_frames.values()), ignore_index=True, sort=False)
    outer_models = fit_models(outer_train, 45999)
    held_frame = held_candidate_frame(held, baseline_cache, graph_root, m43)
    held_predictions = predict_models(held_frame, outer_models)
    held_predictions.to_parquet(output_root / "outer_cut_predictions.parquet", index=False)

    outer_track_root = output_root / "outer_final" / "track_results"
    selected, tracker_report = apply_and_write_tracker(
        held, held_predictions, chosen_policy, baseline_cache, graph_root, parent,
        outer_track_root / f"{held}.txt", evaluator,
    )
    selected.to_parquet(output_root / "outer_selected_cuts.parquet", index=False)
    frozen_tracker = {
        "held_sequence": held,
        "held_label_file_read": False,
        "held_gt_read": False,
        "held_candidates": int(len(held_predictions)),
        "selected_policy": chosen_policy.__dict__,
        "selected_cuts": int(len(selected)),
        "selected_source_indices": selected.source_index.astype(int).tolist() if len(selected) else [],
        "tracker_report": tracker_report,
        "tracker_path": str(outer_track_root / f"{held}.txt"),
    }
    (output_root / "outer_tracker_frozen.json").write_text(json.dumps(frozen_tracker, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"stage": "outer_tracker_frozen", **frozen_tracker}), flush=True)

    evaluation = None
    if not args.skip_outer_trackeval:
        evaluation = evaluate_detailed(
            outer_track_root, output_root / "outer_eval", f"m23_45_outer_{held}", (held,),
        )["COMBINED"]
        print(json.dumps({"stage": "outer_trackeval", **evaluation}), flush=True)

    report = {
        **protocol,
        "status": "completed" if evaluation is not None else "tracker_frozen",
        "training_frame_report": frame_report,
        "inner_candidates": inner_rows,
        "frozen_inner_selection": frozen_selection,
        "held_candidates": int(len(held_predictions)),
        "held_selected_cuts": int(len(selected)),
        "tracker_report": tracker_report,
        "eval": evaluation,
    }
    (output_root / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    protocol["status"] = report["status"]
    (output_root / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
