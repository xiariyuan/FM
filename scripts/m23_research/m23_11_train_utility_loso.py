from __future__ import annotations

# Research artifact for the MOT20 M23 sequence-LOSO audit.

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score


SEQS = ["MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05"]
ROOT = Path("outputs/mot20_m23_20260718/micrograph_chunk30_assa_utility_v1")
OUT = Path("outputs/mot20_m23_20260718/micrograph_chunk30_utility_loso_v1")
FEATURES = [
    "gap",
    "log_gap",
    "appearance_cos",
    "forward_motion_error",
    "backward_motion_error",
    "motion_error_min",
    "motion_error_mean",
    "endpoint_displacement",
    "velocity_cos",
    "log_height_ratio",
    "src_rows",
    "dst_rows",
    "src_mapping_rate",
    "dst_mapping_rate",
    "mapping_rate_min",
    "src_consistency",
    "dst_consistency",
    "consistency_min",
    "src_match_iou",
    "dst_match_iou",
    "out_rank",
    "in_rank",
    "max_rank",
    "out_margin",
    "in_margin",
    "max_margin",
]
TARGET = "assa_edge_delta_proxy"
POSITIVE = "assa_edge_positive"


def sequence_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.seq.value_counts()
    mapping = {seq: len(frame) / (len(counts) * count) for seq, count in counts.items()}
    return frame.seq.map(mapping).to_numpy(float)


def classification_weights(frame: pd.DataFrame, rare_label: int, cap: float = 25.0) -> np.ndarray:
    y = frame[POSITIVE].to_numpy(int)
    rare = int((y == rare_label).sum())
    common = int((y != rare_label).sum())
    multiplier = min(cap, common / max(rare, 1))
    return sequence_weights(frame) * np.where(y == rare_label, multiplier, 1.0)


def regression_weights(frame: pd.DataFrame) -> np.ndarray:
    magnitude = np.clip(np.abs(frame[TARGET].to_numpy(float)), 0.0, 10.0)
    return sequence_weights(frame) * (1.0 + magnitude)


def safe_classification_metrics(y: np.ndarray, score: np.ndarray) -> dict:
    if len(np.unique(y)) < 2:
        return {"auc": None, "ap": None}
    return {
        "auc": float(roc_auc_score(y, score)),
        "ap": float(average_precision_score(y, score)),
    }


def fit_role(frame: pd.DataFrame, role: str, fold_index: int):
    rare_label = 0 if role == "source" else 1
    classifier = HistGradientBoostingClassifier(
        max_iter=280,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=20 if role == "source" else 40,
        l2_regularization=8.0,
        max_bins=255,
        random_state=11100 + 10 * fold_index + (0 if role == "source" else 1),
        early_stopping=False,
    )
    classifier.fit(
        frame[FEATURES],
        frame[POSITIVE].astype(int),
        sample_weight=classification_weights(frame, rare_label=rare_label),
    )
    regressor = HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=20 if role == "source" else 40,
        l2_regularization=10.0,
        max_bins=255,
        random_state=11200 + 10 * fold_index + (0 if role == "source" else 1),
        early_stopping=False,
    )
    transformed = np.arcsinh(frame[TARGET].to_numpy(float))
    regressor.fit(
        frame[FEATURES],
        transformed,
        sample_weight=regression_weights(frame),
    )
    return classifier, regressor


def predict(frame: pd.DataFrame, classifier, regressor) -> pd.DataFrame:
    output = frame.copy()
    output["pred_utility_positive_prob"] = classifier.predict_proba(output[FEATURES])[:, 1]
    output["pred_utility_asinh"] = regressor.predict(output[FEATURES])
    output["pred_utility_delta"] = np.sinh(output.pred_utility_asinh.to_numpy(float))
    output["pred_risk_adjusted_utility"] = (
        output.pred_utility_positive_prob.to_numpy(float)
        * np.maximum(output.pred_utility_delta.to_numpy(float), 0.0)
    )
    output["pred_keep_expected_utility"] = (
        output.pred_utility_positive_prob.to_numpy(float)
        * output.pred_utility_delta.to_numpy(float)
    )
    return output


def ranking_audit(frame: pd.DataFrame, role: str) -> dict:
    actual = frame[TARGET].to_numpy(float)
    y = frame[POSITIVE].to_numpy(int)
    probability = frame.pred_utility_positive_prob.to_numpy(float)
    predicted = frame.pred_utility_delta.to_numpy(float)
    rho = spearmanr(actual, predicted).statistic if len(frame) > 1 else np.nan
    result = {
        "rows": len(frame),
        "positive": int(y.sum()),
        "positive_rate": float(y.mean()) if len(y) else 0.0,
        **safe_classification_metrics(y, probability),
        "delta_spearman": None if not np.isfinite(rho) else float(rho),
        "delta_mae": float(mean_absolute_error(actual, predicted)),
        "actual_delta_sum": float(actual.sum()),
        "predicted_delta_sum": float(predicted.sum()),
    }
    if role == "cross":
        order = np.argsort(-frame.pred_risk_adjusted_utility.to_numpy(float))
        for k in [10, 25, 50, 100, 250, 500, 1000]:
            indices = order[: min(k, len(order))]
            values = actual[indices]
            result[f"top{k}_selected"] = len(indices)
            result[f"top{k}_positive"] = int((values > 0).sum())
            result[f"top{k}_precision"] = float((values > 0).mean()) if len(values) else 0.0
            result[f"top{k}_delta_sum"] = float(values.sum())
            result[f"top{k}_positive_sum"] = float(values[values > 0].sum())
            result[f"top{k}_negative_sum"] = float(values[values < 0].sum())
    else:
        order = np.argsort(frame.pred_keep_expected_utility.to_numpy(float))
        for k in [5, 10, 25, 50, 100, 250, 500]:
            indices = order[: min(k, len(order))]
            values = actual[indices]
            result[f"break_top{k}_selected"] = len(indices)
            result[f"break_top{k}_true_negative"] = int((values < 0).sum())
            result[f"break_top{k}_precision"] = float((values < 0).mean()) if len(values) else 0.0
            result[f"break_top{k}_net_gain"] = float((-values).sum())
            result[f"break_top{k}_saved_negative"] = float((-values[values < 0]).sum())
            result[f"break_top{k}_lost_positive"] = float(values[values > 0].sum())
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frames = []
    for seq in SEQS:
        frame = pd.read_parquet(ROOT / seq / "candidate_edges_utility.parquet")
        frame.insert(0, "seq", seq)
        frames.append(frame)
    all_data = pd.concat(frames, ignore_index=True, sort=False)
    fold_reports = []
    for fold_index, held in enumerate(SEQS):
        train_pool = all_data[all_data.seq != held]
        held_pool = all_data[all_data.seq == held]
        held_report = {"held_out_seq": held}
        for role in ["source", "cross"]:
            if role == "source":
                train = train_pool[train_pool.source_adjacent == 1].copy()
                test = held_pool[held_pool.source_adjacent == 1].copy()
            else:
                train = train_pool[train_pool.same_source == 0].copy()
                test = held_pool[held_pool.same_source == 0].copy()
            classifier, regressor = fit_role(train, role, fold_index)
            predictions = predict(test, classifier, regressor)
            predictions.to_parquet(OUT / f"{held}_{role}_utility_predictions.parquet", index=False)
            held_report[role] = {
                "train_rows": len(train),
                "train_positive": int(train[POSITIVE].sum()),
                **ranking_audit(predictions, role),
            }
        fold_reports.append(held_report)
        print(json.dumps(held_report), flush=True)
    report = {
        "protocol": {
            "validation": "strict leave-one-sequence-out; held sequence excluded from classifier and regressor fitting",
            "roles": "separate source-adjacent keep/split and cross-source merge utility models",
            "target": TARGET,
            "target_transform": "asinh for regression; sign for utility-positive classifier",
            "features": FEATURES,
            "feature_gt_use": "none; explicit allowlist excludes all GT and diagnostic columns",
            "training_gt_use": "AssA-aligned utility labels and sample weights only",
            "inference": "GT-free",
            "status": "diagnostic ranking only; no TrackEval policy selected here",
        },
        "folds": fold_reports,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
