#!/usr/bin/env python3
from __future__ import annotations

"""M23-52 strict residual transaction student.

The student starts from the byte-frozen deployable M23-39 transaction state and
uses only M23-51 exact labels from the three outer-training sequences.  Drop and
replacement actions have separate heads.  Features are sequence-normalized,
source experts are weighted by a GT-free domain-discriminator density ratio,
and a one-sided leave-one-inner-sequence conformal correction produces an exact-
HOTA lower bound.  The only policies are no-op, top-1 and top-2 abstention.

The outer-held label path is never opened.  The held tracker and an explicit
freeze manifest are written before TrackEval is allowed to read held GT.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import pairwise_distances


REPO = Path(__file__).resolve().parents[2]
SEQUENCES = ("MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05")
LABEL_PATHS = {
    seq: Path(
        f"outputs/mot20_m23_20260718/m23_51_residual_labels_m{seq[-2:]}_v1/"
        "exact_action_labels.parquet"
    )
    for seq in SEQUENCES
}
FROZEN_CACHE = Path(
    "outputs/mot20_m23_20260718/m23_45_m23_39_deployable_baseline_cache_v1"
)
STRICT_M23_39 = {
    "MOT20-01": {"HOTA": 78.805125, "DetA": 81.837410, "AssA": 76.043165, "IDSW": 46},
    "MOT20-02": {"HOTA": 73.098150, "DetA": 80.584820, "AssA": 66.407293, "IDSW": 325},
    "MOT20-03": {"HOTA": 80.603280, "DetA": 81.174386, "AssA": 80.068130, "IDSW": 146},
    "MOT20-05": {"HOTA": 79.732850, "DetA": 81.954850, "AssA": 77.612920, "IDSW": 478},
}
STRICT_M23_46 = {
    **STRICT_M23_39,
    "MOT20-05": {"HOTA": 79.770327, "DetA": 81.954810, "AssA": 77.685820, "IDSW": 479},
}
ACTION_TYPES = ("drop", "replace")
CONFORMAL_COVERAGE = 0.80
POLICY_BUDGETS = (0, 1, 2)
STUDENT_MODE = "parametric_delta"
LOCAL_NEIGHBORS = {"drop": 3, "replace": 5}
FORBIDDEN_HELD_TOKENS = (
    "same_gt", "modal_gt", "purity", "label_confidence", "exact_hota",
    "delta_hota", "delta_deta", "delta_assa", "teacher", "actual_assa",
)

# All continuous fields are converted to ranks within sequence and action type.
# No absolute track/chunk IDs or GT-derived fields are model inputs.
CONTINUOUS_FEATURES = (
    "m23_38_parent_policy_score",
    "pred_positive_probability",
    "pred_normalized_gain",
    "pred_normalized_loss",
    "pred_entropy",
    "motion_appearance_joint",
    "max_margin",
    "backward_motion_error",
    "motion_error_mean",
    "motion_error_min",
    "sequence_motion_percentile",
    "sequence_appearance_percentile",
    "sequence_segment_appearance_percentile",
    "sequence_gap_percentile",
    "motion_rank_consensus",
    "appearance_rank_consensus",
    "rank_consensus",
    "appearance_cos",
    "segment_appearance_cos",
    "track_appearance_cos",
    "segment_endpoint_gain",
    "coherence_floor",
    "coherence_imbalance",
    "merged_coherence_gain",
    "detached_fraction",
    "conflict_per_merged_row",
    "removed_baseline_actions",
    "resulting_selected_actions",
    "src_cut_fraction",
    "dst_cut_fraction",
    "merged_balance",
    "max_rank",
)
BINARY_FEATURES = (
    "same_source",
    "source_adjacent",
    "transaction_removes_source_out",
    "transaction_removes_source_in",
    "channel_parent_policy",
    "channel_motion",
    "channel_appearance",
    "channel_structure",
    "channel_exploration",
)
FEATURES = CONTINUOUS_FEATURES + BINARY_FEATURES


@dataclass(frozen=True)
class Policy:
    policy_id: str
    budget: int


def policies() -> List[Policy]:
    return [
        Policy("noop", 0),
        Policy("conformal_top1", 1),
        Policy("conformal_top2", 2),
    ]


def load_module(name: str, relative: str):
    path = REPO / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numeric(values: pd.Series) -> pd.Series:
    return (
        pd.to_numeric(values, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(float)
    )


def add_channel_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    channels = output.get(
        "m23_41_selection_channel", pd.Series("drop", index=output.index)
    ).fillna("drop").astype(str)
    groups = {
        "channel_parent_policy": {"parent_policy"},
        "channel_motion": {
            "motion_error_mean", "motion_error_min", "motion_consensus", "rank_consensus"
        },
        "channel_appearance": {"appearance_consensus", "track_appearance"},
        "channel_structure": {
            "low_detachment", "low_conflict_density", "large_merged_segment",
            "large_detached_segment",
        },
        "channel_exploration": {
            "parent_boundary", "high_entropy", "positive_probability", "predicted_gain"
        },
    }
    for feature, names in groups.items():
        output[feature] = channels.isin(names).astype(float)
    return output


def sequence_normalize(frame: pd.DataFrame) -> pd.DataFrame:
    output = add_channel_features(frame)
    if "seq" not in output or "action_type" not in output:
        raise RuntimeError("candidate frame missing seq/action_type")
    group_columns = [output.seq.astype(str), output.action_type.astype(str)]
    for column in CONTINUOUS_FEATURES:
        values = numeric(output[column]) if column in output else pd.Series(0.0, index=output.index)
        ranks = values.groupby(group_columns, sort=False).rank(method="average", pct=True)
        counts = values.groupby(group_columns, sort=False).transform("size")
        output[column] = np.where(counts > 1, 2.0 * ranks - 1.0, 0.0)
    for column in BINARY_FEATURES:
        values = numeric(output[column]) if column in output else pd.Series(0.0, index=output.index)
        output[column] = values.clip(0.0, 1.0)
    return output


def validate_raw_candidate_columns(frame: pd.DataFrame, context: str) -> None:
    forbidden = [
        column for column in frame.columns
        if any(token in column.lower() for token in FORBIDDEN_HELD_TOKENS)
    ]
    if forbidden:
        raise RuntimeError(f"{context} candidate leakage columns: {forbidden}")


def load_training_frame(seq: str, path: Path, opened: List[str]) -> pd.DataFrame:
    opened.append(str(path))
    frame = pd.read_parquet(path)
    frame = frame[frame.status.astype(str) == "success"].copy()
    if frame.empty:
        raise RuntimeError(f"empty labels: {path}")
    if frame.seq.astype(str).nunique() != 1 or str(frame.seq.iloc[0]) != seq:
        raise RuntimeError(f"label sequence mismatch: {seq} {path}")
    if "delta_HOTA" not in frame:
        raise RuntimeError(f"missing exact HOTA label: {path}")
    output = sequence_normalize(frame)
    if STUDENT_MODE == "local_rank":
        output["target_rank"] = (
            output.groupby("action_type", sort=False).delta_HOTA.rank(
                method="average", pct=True
            )
            * 2.0
            - 1.0
        )
    return output


def action_nodes(row) -> set[int]:
    return {int(row.transaction_src_track_id), int(row.transaction_dst_track_id)}


def make_held_action_frame(
    seq: str,
    max_replacements: int,
    m39,
    label_module,
) -> pd.DataFrame:
    prediction_root = m39.BASELINE_ROOTS[seq]
    predictions = pd.read_parquet(
        prediction_root / "predictions" / f"{seq}_predictions.parquet"
    )
    selected = pd.read_parquet(
        FROZEN_CACHE / seq / "frozen_selected_transactions.parquet"
    )
    gt_fields = (
        "same_gt", "src_modal_gt", "dst_modal_gt", "src_purity", "dst_purity",
        "label_confidence", "actual_assa",
    )
    for name, source in (("predictions", predictions), ("selected", selected)):
        for field in gt_fields:
            if field in source:
                total = float(np.nan_to_num(numeric(source[field]).to_numpy(float), nan=0.0).sum())
                if abs(total) > 1e-9:
                    raise RuntimeError(f"held {name} contains nonzero GT field {field}={total}")
        source.drop(columns=[field for field in gt_fields if field in source], inplace=True)

    rows: List[Dict[str, object]] = []
    for index, row in selected.iterrows():
        record = row.to_dict()
        record.update(
            seq=seq,
            action_type="drop",
            source_index=int(index),
            removed_baseline_actions=1,
            resulting_selected_actions=int(len(selected) - 1),
        )
        rows.append(record)

    replacements = label_module.residual_replacements(
        predictions, selected, max_replacements, selection_mode="diverse"
    )
    for index, row in replacements.iterrows():
        removed = sum(
            not action_nodes(item).isdisjoint(action_nodes(row))
            for item in selected.itertuples()
        )
        record = row.to_dict()
        record.update(
            seq=seq,
            action_type="replace",
            source_index=int(index),
            removed_baseline_actions=int(removed),
            resulting_selected_actions=int(len(selected) - removed + 1),
        )
        rows.append(record)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"empty held residual candidate frame: {seq}")
    validate_raw_candidate_columns(frame, "held")
    return sequence_normalize(frame)


def sequence_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.seq.astype(str).value_counts()
    weights = frame.seq.astype(str).map(
        {seq: 1.0 / count for seq, count in counts.items()}
    ).to_numpy(float)
    return weights * len(weights) / weights.sum()


def balanced_sign_weights(frame: pd.DataFrame) -> np.ndarray:
    labels = (frame.delta_HOTA.to_numpy(float) > 0.0).astype(int)
    weights = sequence_weights(frame)
    for label in (0, 1):
        mask = labels == label
        if mask.any():
            weights[mask] *= len(labels) / (2.0 * mask.sum())
    return weights * len(weights) / weights.sum()


def fit_head(frame: pd.DataFrame, seed: int) -> Dict[str, object]:
    if frame.empty:
        raise RuntimeError("cannot fit empty action head")
    matrix = frame[list(FEATURES)].to_numpy(float)
    target = frame.delta_HOTA.to_numpy(float)
    sign = (target > 0.0).astype(int)
    min_leaf = 3 if str(frame.action_type.iloc[0]) == "drop" else 8

    if np.unique(sign).size < 2:
        sign_bundle: Dict[str, object] = {
            "kind": "constant", "probability": float(sign.mean())
        }
    else:
        hgb_sign = HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_iter=160,
            max_leaf_nodes=5,
            min_samples_leaf=min_leaf,
            l2_regularization=12.0,
            early_stopping=False,
            random_state=seed,
        )
        linear_sign = LogisticRegression(
            C=0.20,
            class_weight="balanced",
            max_iter=2000,
            random_state=seed + 1,
        )
        hgb_sign.fit(matrix, sign, sample_weight=balanced_sign_weights(frame))
        linear_sign.fit(matrix, sign, sample_weight=sequence_weights(frame))
        sign_bundle = {"kind": "ensemble", "hgb": hgb_sign, "linear": linear_sign}

    hgb_delta = HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.04,
        max_iter=160,
        max_leaf_nodes=5,
        min_samples_leaf=min_leaf,
        l2_regularization=16.0,
        early_stopping=False,
        random_state=seed + 10,
    )
    ridge_delta = Ridge(alpha=24.0)
    weights = sequence_weights(frame)
    hgb_delta.fit(matrix, target, sample_weight=weights)
    ridge_delta.fit(matrix, target, sample_weight=weights)
    return {
        "sign": sign_bundle,
        "hgb_delta": hgb_delta,
        "ridge_delta": ridge_delta,
        "rows": int(len(frame)),
    }


def predict_head(frame: pd.DataFrame, bundle: Mapping[str, object]) -> Tuple[np.ndarray, np.ndarray]:
    matrix = frame[list(FEATURES)].to_numpy(float)
    sign_bundle = bundle["sign"]
    if sign_bundle["kind"] == "constant":
        probability = np.full(len(frame), float(sign_bundle["probability"]), dtype=float)
    else:
        p_hgb = sign_bundle["hgb"].predict_proba(matrix)[:, 1]
        p_linear = sign_bundle["linear"].predict_proba(matrix)[:, 1]
        probability = 0.60 * p_hgb + 0.40 * p_linear
    delta_hgb = bundle["hgb_delta"].predict(matrix)
    delta_ridge = bundle["ridge_delta"].predict(matrix)
    delta = 0.65 * delta_hgb + 0.35 * delta_ridge
    return np.asarray(delta, dtype=float), np.asarray(probability, dtype=float)


def fit_domain_support(source: pd.DataFrame, target: pd.DataFrame, seed: int) -> np.ndarray:
    if source.empty or target.empty:
        return np.zeros(len(target), dtype=float)
    combined = pd.concat([source, target], ignore_index=True, sort=False)
    labels = np.concatenate(
        [np.zeros(len(source), dtype=int), np.ones(len(target), dtype=int)]
    )
    if len(combined) < 4:
        return np.ones(len(target), dtype=float)
    model = LogisticRegression(
        C=0.20,
        class_weight="balanced",
        max_iter=2000,
        random_state=seed,
    )
    model.fit(combined[list(FEATURES)].to_numpy(float), labels)
    p_target = model.predict_proba(target[list(FEATURES)].to_numpy(float))[:, 1]
    # With equal class priors, (1-p)/p estimates source/target density ratio.
    ratio = (1.0 - p_target) / np.clip(p_target, 1e-6, 1.0)
    return np.clip(ratio, 0.0, 1.0)


def predict_centers(
    source_frames: Mapping[str, pd.DataFrame],
    target_frame: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    if STUDENT_MODE == "local_rank":
        return predict_local_rank_centers(source_frames, target_frame, seed)
    if not source_frames:
        raise RuntimeError("no source sequences")
    output = target_frame.copy()
    output["pooled_delta"] = 0.0
    output["pooled_positive_probability"] = 0.0
    output["expert_weighted_delta"] = 0.0
    output["expert_weighted_probability"] = 0.0
    output["expert_delta_std"] = 0.0
    output["domain_density_support"] = 0.0
    output["adjusted_center_delta"] = 0.0

    for type_index, action_type in enumerate(ACTION_TYPES):
        target = output[output.action_type.astype(str) == action_type].copy()
        if target.empty:
            continue
        local_sources = {
            seq: frame[frame.action_type.astype(str) == action_type].copy()
            for seq, frame in source_frames.items()
        }
        local_sources = {seq: frame for seq, frame in local_sources.items() if not frame.empty}
        if not local_sources:
            raise RuntimeError(f"no source rows for {action_type}")
        pooled = pd.concat(list(local_sources.values()), ignore_index=True, sort=False)
        pooled_model = fit_head(pooled, seed + 1000 * type_index)
        pooled_delta, pooled_probability = predict_head(target, pooled_model)

        expert_delta: List[np.ndarray] = []
        expert_probability: List[np.ndarray] = []
        expert_support: List[np.ndarray] = []
        for source_index, (seq, source) in enumerate(sorted(local_sources.items())):
            model = fit_head(source.reset_index(drop=True), seed + 1000 * type_index + 100 + source_index)
            delta, probability = predict_head(target, model)
            support = fit_domain_support(
                source, target, seed + 1000 * type_index + 500 + source_index
            )
            expert_delta.append(delta)
            expert_probability.append(probability)
            expert_support.append(support)
            output.loc[target.index, f"expert_{seq[-2:]}_delta"] = delta
            output.loc[target.index, f"expert_{seq[-2:]}_probability"] = probability
            output.loc[target.index, f"expert_{seq[-2:]}_density_ratio"] = support

        delta_matrix = np.stack(expert_delta, axis=1)
        probability_matrix = np.stack(expert_probability, axis=1)
        support_matrix = np.stack(expert_support, axis=1)
        weights = support_matrix + 1e-3
        weights /= weights.sum(axis=1, keepdims=True)
        weighted_delta = np.sum(weights * delta_matrix, axis=1)
        weighted_probability = np.sum(weights * probability_matrix, axis=1)
        disagreement = np.sqrt(np.sum(weights * (delta_matrix - weighted_delta[:, None]) ** 2, axis=1))
        coverage = np.mean(support_matrix, axis=1)
        center = 0.50 * pooled_delta + 0.50 * weighted_delta
        probability = 0.50 * pooled_probability + 0.50 * weighted_probability
        adjusted = (
            center
            - 0.50 * disagreement
            - 0.25 * (1.0 - coverage) * np.abs(center)
        )

        output.loc[target.index, "pooled_delta"] = pooled_delta
        output.loc[target.index, "pooled_positive_probability"] = pooled_probability
        output.loc[target.index, "expert_weighted_delta"] = weighted_delta
        output.loc[target.index, "expert_weighted_probability"] = weighted_probability
        output.loc[target.index, "expert_delta_std"] = disagreement
        output.loc[target.index, "domain_density_support"] = coverage
        output.loc[target.index, "adjusted_center_delta"] = adjusted
        output.loc[target.index, "student_positive_probability"] = probability
    return output


def predict_local_rank_centers(
    source_frames: Mapping[str, pd.DataFrame],
    target_frame: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    """Predict a sequence-normalized exact-delta rank with local source experts."""
    if not source_frames:
        raise RuntimeError("no source sequences")
    output = target_frame.copy()
    output["pooled_delta"] = 0.0
    output["pooled_positive_probability"] = 0.0
    output["expert_weighted_delta"] = 0.0
    output["expert_weighted_probability"] = 0.0
    output["expert_delta_std"] = 0.0
    output["domain_density_support"] = 0.0
    output["adjusted_center_delta"] = 0.0

    for type_index, action_type in enumerate(ACTION_TYPES):
        target = output[output.action_type.astype(str) == action_type].copy()
        if target.empty:
            continue
        local_sources = {
            seq: frame[frame.action_type.astype(str) == action_type].copy()
            for seq, frame in source_frames.items()
        }
        local_sources = {seq: frame for seq, frame in local_sources.items() if not frame.empty}
        if not local_sources:
            raise RuntimeError(f"no source rows for {action_type}")

        target_matrix = target[list(FEATURES)].to_numpy(float)
        expert_rank: List[np.ndarray] = []
        expert_probability: List[np.ndarray] = []
        expert_support: List[np.ndarray] = []
        for source_index, (seq, source) in enumerate(sorted(local_sources.items())):
            if "target_rank" not in source:
                raise RuntimeError(f"local-rank source missing target_rank: {seq}")
            source_matrix = source[list(FEATURES)].to_numpy(float)
            distances = pairwise_distances(target_matrix, source_matrix)
            neighbors = min(LOCAL_NEIGHBORS[action_type], len(source))
            indices = np.argpartition(distances, neighbors - 1, axis=1)[:, :neighbors]
            local_distances = np.take_along_axis(distances, indices, axis=1)
            local_weights = 1.0 / (local_distances + 0.20)
            local_weights /= local_weights.sum(axis=1, keepdims=True)
            ranks = source.target_rank.to_numpy(float)[indices]
            signs = (source.delta_HOTA.to_numpy(float)[indices] > 0.0).astype(float)
            rank_prediction = np.sum(local_weights * ranks, axis=1)
            probability = np.sum(local_weights * signs, axis=1)
            support = fit_domain_support(
                source, target, seed + 1000 * type_index + source_index
            )
            expert_rank.append(rank_prediction)
            expert_probability.append(probability)
            expert_support.append(support)
            output.loc[target.index, f"expert_{seq[-2:]}_delta"] = rank_prediction
            output.loc[target.index, f"expert_{seq[-2:]}_probability"] = probability
            output.loc[target.index, f"expert_{seq[-2:]}_density_ratio"] = support

        rank_matrix = np.stack(expert_rank, axis=1)
        probability_matrix = np.stack(expert_probability, axis=1)
        support_matrix = np.stack(expert_support, axis=1)
        weights = support_matrix + 1e-3
        weights /= weights.sum(axis=1, keepdims=True)
        weighted_rank = np.sum(weights * rank_matrix, axis=1)
        weighted_probability = np.sum(weights * probability_matrix, axis=1)
        disagreement = np.sqrt(
            np.sum(weights * (rank_matrix - weighted_rank[:, None]) ** 2, axis=1)
        )
        coverage = np.mean(support_matrix, axis=1)
        adjusted = (
            weighted_rank
            - 0.50 * disagreement
            - 0.15 * (1.0 - coverage)
        )

        output.loc[target.index, "pooled_delta"] = weighted_rank
        output.loc[target.index, "pooled_positive_probability"] = weighted_probability
        output.loc[target.index, "expert_weighted_delta"] = weighted_rank
        output.loc[target.index, "expert_weighted_probability"] = weighted_probability
        output.loc[target.index, "expert_delta_std"] = disagreement
        output.loc[target.index, "domain_density_support"] = coverage
        output.loc[target.index, "adjusted_center_delta"] = adjusted
        output.loc[target.index, "student_positive_probability"] = weighted_probability
    return output


def conformal_higher_quantile(values: np.ndarray, coverage: float) -> float:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return math.inf
    rank = int(math.ceil((clean.size + 1) * coverage))
    rank = min(max(rank, 1), clean.size)
    return float(np.sort(clean)[rank - 1])


def calibrate_quantiles(
    source_frames: Mapping[str, pd.DataFrame],
    seed: int,
) -> Tuple[Dict[str, float], Dict[str, object]]:
    scores: Dict[str, List[float]] = {action_type: [] for action_type in ACTION_TYPES}
    folds: List[Dict[str, object]] = []
    source_names = sorted(source_frames)
    if len(source_names) < 2:
        raise RuntimeError("conformal calibration requires at least two source sequences")
    for fold_index, pseudo_held in enumerate(source_names):
        train = {seq: source_frames[seq] for seq in source_names if seq != pseudo_held}
        target = source_frames[pseudo_held]
        prediction = predict_centers(train, target, seed + 10000 + fold_index)
        fold_info: Dict[str, object] = {"pseudo_held": pseudo_held, "rows": int(len(target))}
        for action_type in ACTION_TYPES:
            local = prediction[prediction.action_type.astype(str) == action_type]
            target_column = "target_rank" if STUDENT_MODE == "local_rank" else "delta_HOTA"
            residual = (
                local.adjusted_center_delta.to_numpy(float)
                - local[target_column].to_numpy(float)
            )
            scores[action_type].extend(residual.tolist())
            fold_info[f"{action_type}_rows"] = int(len(local))
        folds.append(fold_info)
    quantiles = {
        action_type: max(0.0, conformal_higher_quantile(np.asarray(values), CONFORMAL_COVERAGE))
        for action_type, values in scores.items()
    }
    audit = {
        "coverage": CONFORMAL_COVERAGE,
        "folds": folds,
        "nonconformity_rows": {key: len(value) for key, value in scores.items()},
        "quantiles": quantiles,
    }
    return quantiles, audit


def attach_conformal_lcb(
    source_frames: Mapping[str, pd.DataFrame],
    target_frame: pd.DataFrame,
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    quantiles, calibration = calibrate_quantiles(source_frames, seed)
    output = predict_centers(source_frames, target_frame, seed + 50000)
    output["conformal_quantile"] = output.action_type.astype(str).map(quantiles).astype(float)
    output["conformal_lcb"] = (
        output.adjusted_center_delta.to_numpy(float)
        - output.conformal_quantile.to_numpy(float)
    )
    output["student_value"] = output.conformal_lcb.to_numpy(float)
    output["eligible"] = (
        (output.conformal_lcb.to_numpy(float) > 0.0)
        & (output.student_positive_probability.to_numpy(float) >= 0.50)
    )
    return output, calibration


def compatible(left, right) -> bool:
    return action_nodes(left).isdisjoint(action_nodes(right))


def select_actions(frame: pd.DataFrame, budget: int) -> pd.DataFrame:
    if budget <= 0:
        return frame.iloc[:0].copy()
    eligible = frame[frame.eligible.astype(bool)].copy()
    if eligible.empty:
        return frame.iloc[:0].copy()
    eligible.sort_values(
        ["conformal_lcb", "student_positive_probability", "adjusted_center_delta"],
        ascending=[False, False, False],
        inplace=True,
    )
    pool = eligible.head(32)
    best_indices: Tuple[int, ...] = ()
    best_key: Tuple[float, float, float, int] = (0.0, 0.0, 0.0, 0)
    for index, row in pool.iterrows():
        key = (
            float(row.conformal_lcb),
            float(row.student_positive_probability),
            float(row.domain_density_support),
            -1,
        )
        if key > best_key:
            best_key = key
            best_indices = (int(index),)
    if budget >= 2:
        rows = list(pool.iterrows())
        for left_position, (left_index, left) in enumerate(rows):
            for right_index, right in rows[left_position + 1:]:
                if not compatible(left, right):
                    continue
                key = (
                    float(left.conformal_lcb + right.conformal_lcb),
                    float(left.student_positive_probability + right.student_positive_probability),
                    float(left.domain_density_support + right.domain_density_support),
                    -2,
                )
                if key > best_key:
                    best_key = key
                    best_indices = (int(left_index), int(right_index))
    if not best_indices:
        return frame.iloc[:0].copy()
    return frame.loc[list(best_indices)].sort_values("conformal_lcb", ascending=False).copy()


def write_tracker(
    seq: str,
    predictions: pd.DataFrame,
    policy: Policy,
    graph_root: Path,
    parent: Path,
    output_path: Path,
    m39,
    chain,
    evaluator,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    chosen = select_actions(predictions, policy.budget)
    cache_root = FROZEN_CACHE / seq
    baseline_tracker = cache_root / "track_results" / f"{seq}.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if chosen.empty:
        shutil.copyfile(baseline_tracker, output_path)
        return chosen, {
            "seq": seq,
            "drop_actions": 0,
            "replacement_actions": 0,
            "removed_baseline_actions": 0,
            "resulting_selected_actions": int(
                len(pd.read_parquet(cache_root / "frozen_selected_transactions.parquet"))
            ),
            "baseline_byte_exact": sha256(output_path) == sha256(baseline_tracker),
            "selected_source_indices": [],
            "tracker_path": str(output_path),
        }

    original_predictions = pd.read_parquet(
        m39.BASELINE_ROOTS[seq] / "predictions" / f"{seq}_predictions.parquet"
    )
    baseline_selected = pd.read_parquet(
        cache_root / "frozen_selected_transactions.parquet"
    )
    drops = chosen[chosen.action_type.astype(str) == "drop"].copy()
    replacements = chosen[chosen.action_type.astype(str) == "replace"].copy()
    selected, removed = m39.build_selected_transactions(
        baseline_selected, original_predictions, drops, replacements
    )
    edges = pd.read_parquet(graph_root / seq / "candidate_edges.parquet")
    meta = pd.read_parquet(graph_root / seq / "microtracklets.parquet")
    applied = chain.apply_transactions(edges, selected)
    for column, default in (
        ("assa_edge_delta_proxy", 0.0),
        ("assa_edge_positive", 0),
        ("assa_edge_negative", 0),
    ):
        if column not in applied:
            applied[column] = default
    evaluator.DATA = graph_root
    evaluator.PARENT = parent
    tracker_report = evaluator.write_tracker(seq, meta, applied, output_path)
    return chosen, {
        "seq": seq,
        "drop_actions": int(len(drops)),
        "replacement_actions": int(len(replacements)),
        "removed_baseline_actions": int(removed),
        "resulting_selected_actions": int(len(selected)),
        "baseline_byte_exact": False,
        "selected_source_indices": chosen.source_index.astype(int).tolist(),
        "selected_action_types": chosen.action_type.astype(str).tolist(),
        "selected_lcb": chosen.conformal_lcb.astype(float).tolist(),
        "selected_probability": chosen.student_positive_probability.astype(float).tolist(),
        "selected_domain_support": chosen.domain_density_support.astype(float).tolist(),
        "tracker_report": tracker_report,
        "tracker_path": str(output_path),
    }


def robust_key(row: Mapping[str, object]) -> Tuple[float, float, float, float, int]:
    return (
        float(row["worst_fold_delta_HOTA"]),
        float(row["mean_fold_delta_HOTA"]),
        float(row["HOTA"]),
        float(row["AssA"]),
        -int(row["selected_actions"]),
    )


def metric_delta(metrics: Mapping[str, float], baseline: Mapping[str, float]) -> Dict[str, object]:
    return {
        "HOTA": float(metrics["HOTA"] - baseline["HOTA"]),
        "DetA": float(metrics["DetA"] - baseline["DetA"]),
        "AssA": float(metrics["AssA"] - baseline["AssA"]),
        "IDSW": int(metrics["IDSW"] - baseline["IDSW"]),
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    global STUDENT_MODE
    parser = argparse.ArgumentParser()
    parser.add_argument("--held-seq", required=True, choices=SEQUENCES)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--graph-root", default=None)
    parser.add_argument("--parent", default=None)
    parser.add_argument("--max-replacements", type=int, default=128)
    parser.add_argument(
        "--student-mode",
        choices=("parametric_delta", "local_rank"),
        default="parametric_delta",
    )
    parser.add_argument("--skip-outer-trackeval", action="store_true")
    args = parser.parse_args()
    STUDENT_MODE = args.student_mode

    m39 = load_module(
        "m23_52_m39", "scripts/m23_research/m23_39_exact_advantage_student_nested_loso.py"
    )
    eval_helper = load_module(
        "m23_52_eval", "scripts/m23_research/m23_45_domain_robust_source_cut_student.py"
    )
    label_module = load_module(
        "m23_52_labels", "scripts/m23_research/m23_38_exact_hota_advantage_labels.py"
    )
    chain = load_module(
        "m23_52_chain", "scripts/m23_research/m23_12_chain_transaction_oracle.py"
    )
    evaluator = load_module(
        "m23_52_evaluator", "scripts/m23_research/m23_11_eval_utility_graph.py"
    )

    held = args.held_seq
    training_sequences = [seq for seq in SEQUENCES if seq != held]
    output_root = Path(args.output_root)
    graph_root = Path(args.graph_root) if args.graph_root else m39.DEFAULT_GRAPH_ROOT
    parent = Path(args.parent) if args.parent else m39.DEFAULT_PARENT
    output_root.mkdir(parents=True, exist_ok=True)

    protocol: Dict[str, object] = {
        "experiment": (
            "M23-52 local-expert rank-conformal residual transaction student"
            if STUDENT_MODE == "local_rank"
            else "M23-52 action-type conformal residual transaction student"
        ),
        "protocol": "strict nested sequence-LOSO around byte-frozen M23-39 baseline",
        "deployable": True,
        "outer_held_sequence": held,
        "outer_training_sequences": training_sequences,
        "training_gt_use": "M23-51 exact residual labels from outer-training sequences only",
        "held_candidate_gt_use": "none",
        "held_label_file_read": False,
        "held_gt_read_before_tracker_freeze": False,
        "label_regeneration": False,
        "baseline": "byte-frozen strict deployable M23-39 tracker/selected transactions",
        "candidate_space": "M23-51 GT-free diverse residual drops and replacements",
        "model": (
            "separate per-source drop/replacement local-neighbor rank experts"
            if STUDENT_MODE == "local_rank"
            else "separate drop/replacement HGB+linear heads"
        ),
        "student_mode": STUDENT_MODE,
        "local_neighbors": LOCAL_NEIGHBORS if STUDENT_MODE == "local_rank" else None,
        "feature_normalization": "rank within sequence and action type",
        "domain_adaptation": "source/target logistic density ratio, GT-free target candidates",
        "conformal": {
            "type": "one-sided leave-one-inner-sequence lower bound",
            "coverage": CONFORMAL_COVERAGE,
            "target": "sequence-normalized exact-delta rank"
            if STUDENT_MODE == "local_rank"
            else "exact HOTA delta",
        },
        "policies": [policy.__dict__ for policy in policies()],
        "policy_selection": "maximum worst inner exact-TrackEval HOTA delta; then mean, combined, AssA, fewer actions",
        "features": list(FEATURES),
        "status": "running",
    }
    (output_root / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    opened_label_paths: List[str] = []
    training_frames = {
        seq: load_training_frame(seq, LABEL_PATHS[seq], opened_label_paths)
        for seq in training_sequences
    }
    held_label_path = str(LABEL_PATHS[held])
    if held_label_path in opened_label_paths:
        raise RuntimeError(f"held label file was opened: {held_label_path}")
    label_audit = {
        "opened_label_paths": opened_label_paths,
        "held_label_path": held_label_path,
        "held_label_file_read": False,
        "training_rows": {seq: int(len(frame)) for seq, frame in training_frames.items()},
        "action_rows": {
            seq: frame.action_type.astype(str).value_counts().to_dict()
            for seq, frame in training_frames.items()
        },
    }
    (output_root / "label_access_audit.json").write_text(
        json.dumps(label_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"stage": "training_labels_ready", **label_audit}), flush=True)

    inner_predictions: Dict[str, pd.DataFrame] = {}
    inner_calibration: Dict[str, object] = {}
    for fold_index, pseudo_held in enumerate(training_sequences):
        source_frames = {
            seq: training_frames[seq]
            for seq in training_sequences if seq != pseudo_held
        }
        prediction, calibration = attach_conformal_lcb(
            source_frames, training_frames[pseudo_held], 52000 + 1000 * fold_index
        )
        inner_predictions[pseudo_held] = prediction
        inner_calibration[pseudo_held] = calibration
        path = output_root / "inner_predictions" / f"{pseudo_held}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        prediction.to_parquet(path, index=False)
        print(json.dumps({
            "stage": "inner_prediction",
            "pseudo_held": pseudo_held,
            "rows": int(len(prediction)),
            "eligible": int(prediction.eligible.sum()),
            "eligible_drop": int(
                ((prediction.action_type.astype(str) == "drop") & prediction.eligible).sum()
            ),
            "eligible_replace": int(
                ((prediction.action_type.astype(str) == "replace") & prediction.eligible).sum()
            ),
            "conformal_quantiles": calibration["quantiles"],
        }), flush=True)
    (output_root / "inner_conformal_audit.json").write_text(
        json.dumps(inner_calibration, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    inner_rows: List[Dict[str, object]] = []
    baseline_metrics: Dict[str, Mapping[str, float]] | None = None
    for policy_index, policy in enumerate(policies()):
        candidate_root = output_root / "inner_candidates" / policy.policy_id
        track_root = candidate_root / "track_results"
        reports: List[Dict[str, object]] = []
        selected_actions = 0
        for seq in training_sequences:
            selected, tracker_report = write_tracker(
                seq,
                inner_predictions[seq],
                policy,
                graph_root,
                parent,
                track_root / f"{seq}.txt",
                m39,
                chain,
                evaluator,
            )
            selected_actions += int(len(selected))
            reports.append(tracker_report)
        metrics = eval_helper.evaluate_detailed(
            track_root,
            candidate_root,
            f"m23_52_inner_{held}_{policy_index}",
            training_sequences,
        )
        if policy.budget == 0:
            baseline_metrics = {seq: metrics[seq] for seq in training_sequences}
            for seq in training_sequences:
                error = abs(float(metrics[seq]["HOTA"]) - STRICT_M23_39[seq]["HOTA"])
                if error > 0.002:
                    raise RuntimeError(
                        f"frozen M23-39 baseline mismatch {seq}: {metrics[seq]['HOTA']}"
                    )
        if baseline_metrics is None:
            raise RuntimeError("no-op must be evaluated first")
        deltas = {
            seq: float(metrics[seq]["HOTA"] - baseline_metrics[seq]["HOTA"])
            for seq in training_sequences
        }
        row: Dict[str, object] = {
            "policy_id": policy.policy_id,
            "budget": int(policy.budget),
            **metrics["COMBINED"],
            "selected_actions": int(selected_actions),
            "worst_fold_delta_HOTA": float(min(deltas.values())),
            "mean_fold_delta_HOTA": float(np.mean(list(deltas.values()))),
            "positive_inner_folds": int(sum(value > 0.0 for value in deltas.values())),
        }
        for seq in training_sequences:
            tag = seq.replace("MOT20-", "M")
            row[f"{tag}_HOTA"] = float(metrics[seq]["HOTA"])
            row[f"{tag}_delta_HOTA"] = float(deltas[seq])
        inner_rows.append(row)
        (candidate_root / "tracker_reports.json").write_text(
            json.dumps(reports, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        write_csv(output_root / "inner_metrics.csv", inner_rows)
        print(json.dumps({"stage": "inner_trackeval", **row}), flush=True)

    chosen_row = max(inner_rows, key=robust_key)
    chosen_policy = next(
        policy for policy in policies() if policy.policy_id == chosen_row["policy_id"]
    )
    frozen_selection = {
        "selected_policy": chosen_policy.__dict__,
        "selected_inner_metrics": chosen_row,
        "selection_rule": protocol["policy_selection"],
        "held_label_file_read": False,
        "outer_held_gt_read": False,
        "label_access_audit": label_audit,
    }
    (output_root / "frozen_inner_selection.json").write_text(
        json.dumps(frozen_selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"stage": "policy_frozen", **chosen_row}), flush=True)

    held_frame = make_held_action_frame(
        held, args.max_replacements, m39, label_module
    )
    validate_raw_candidate_columns(held_frame, "held normalized")
    held_predictions, outer_calibration = attach_conformal_lcb(
        training_frames, held_frame, 52999
    )
    validate_raw_candidate_columns(held_predictions, "held prediction")
    held_predictions.to_parquet(
        output_root / "outer_action_predictions.parquet", index=False
    )
    (output_root / "outer_conformal_audit.json").write_text(
        json.dumps(outer_calibration, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    outer_track_root = output_root / "outer_final" / "track_results"
    selected, tracker_report = write_tracker(
        held,
        held_predictions,
        chosen_policy,
        graph_root,
        parent,
        outer_track_root / f"{held}.txt",
        m39,
        chain,
        evaluator,
    )
    selected.to_parquet(output_root / "outer_selected_actions.parquet", index=False)
    frozen_tracker = {
        "held_sequence": held,
        "held_label_path": held_label_path,
        "held_label_file_read": False,
        "held_gt_read": False,
        "opened_label_paths": opened_label_paths,
        "held_candidates": int(len(held_predictions)),
        "held_candidate_forbidden_columns": [],
        "held_eligible_actions": int(held_predictions.eligible.sum()),
        "selected_policy": chosen_policy.__dict__,
        "selected_actions": int(len(selected)),
        "selected_action_types": selected.action_type.astype(str).tolist()
        if len(selected) else [],
        "selected_source_indices": selected.source_index.astype(int).tolist()
        if len(selected) else [],
        "tracker_report": tracker_report,
        "tracker_path": str(outer_track_root / f"{held}.txt"),
        "tracker_sha256": sha256(outer_track_root / f"{held}.txt"),
        "frozen_before_outer_trackeval": True,
    }
    if held_label_path in frozen_tracker["opened_label_paths"]:
        raise RuntimeError("held label read before freeze")
    (output_root / "outer_tracker_frozen.json").write_text(
        json.dumps(frozen_tracker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"stage": "outer_tracker_frozen", **frozen_tracker}), flush=True)

    evaluation = None
    if not args.skip_outer_trackeval:
        evaluation = eval_helper.evaluate_detailed(
            outer_track_root,
            output_root / "outer_eval",
            f"m23_52_outer_{held}",
            (held,),
        )["COMBINED"]
        print(json.dumps({"stage": "outer_trackeval", **evaluation}), flush=True)

    report = {
        **protocol,
        "status": "completed" if evaluation is not None else "tracker_frozen",
        "label_access_audit": label_audit,
        "training_rows": {seq: int(len(frame)) for seq, frame in training_frames.items()},
        "inner_candidates": inner_rows,
        "inner_conformal_audit": inner_calibration,
        "frozen_inner_selection": frozen_selection,
        "outer_conformal_audit": outer_calibration,
        "held_candidate_actions": int(len(held_predictions)),
        "held_eligible_actions": int(held_predictions.eligible.sum()),
        "held_selected_actions": int(len(selected)),
        "held_selected_drops": int(
            (selected.action_type.astype(str) == "drop").sum()
        ) if len(selected) else 0,
        "held_selected_replacements": int(
            (selected.action_type.astype(str) == "replace").sum()
        ) if len(selected) else 0,
        "tracker_report": tracker_report,
        "outer_tracker_frozen": frozen_tracker,
        "eval": evaluation,
        "strict_m23_39_baseline": STRICT_M23_39[held],
        "strict_m23_46_baseline": STRICT_M23_46[held],
        "delta_vs_strict_m23_39": metric_delta(evaluation, STRICT_M23_39[held])
        if evaluation else None,
        "delta_vs_strict_m23_46": metric_delta(evaluation, STRICT_M23_46[held])
        if evaluation else None,
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
