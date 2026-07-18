from __future__ import annotations

# Research artifact for the MOT20 M23 sequence-LOSO audit.

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, roc_auc_score


SEQS = ["MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05"]
ROOT = Path("outputs/mot20_m23_20260718/micrograph_chain_transaction_oracle_v1/labels")
OUT = Path("outputs/mot20_m23_20260718/micrograph_chain_expected_utility_loso_v2_segment_app")


def load_base():
    path = Path("scripts/m23_research/m23_12_train_chain_transaction_loso.py")
    spec = importlib.util.spec_from_file_location("m23_12_chain_base", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["m23_12_chain_base"] = module
    spec.loader.exec_module(module)
    return module


def fit_regressor(frame: pd.DataFrame, target: np.ndarray, seed: int):
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=360,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=12.0,
        max_bins=255,
        random_state=seed,
        early_stopping=False,
    )
    model.fit(
        frame[base.FEATURES],
        target,
        sample_weight=base.sequence_weights(frame),
    )
    return model


def fit_models(frame: pd.DataFrame, fold_index: int):
    classifier = HistGradientBoostingClassifier(
        max_iter=340,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        l2_regularization=12.0,
        max_bins=255,
        random_state=12200 + fold_index,
        early_stopping=False,
    )
    classifier.fit(
        frame[base.FEATURES],
        frame[base.POSITIVE].astype(int),
        sample_weight=base.sequence_weights(frame),
    )
    positive = frame[frame[base.TARGET] > 0].copy()
    negative = frame[frame[base.TARGET] < 0].copy()
    gain_model = fit_regressor(
        positive,
        np.log1p(positive[base.TARGET].to_numpy(float)),
        12300 + fold_index,
    )
    loss_model = fit_regressor(
        negative,
        np.log1p(-negative[base.TARGET].to_numpy(float)),
        12400 + fold_index,
    )
    return classifier, gain_model, loss_model


def predict(frame: pd.DataFrame, models) -> pd.DataFrame:
    classifier, gain_model, loss_model = models
    output = frame.copy()
    probability = classifier.predict_proba(output[base.FEATURES])[:, 1]
    gain = np.expm1(gain_model.predict(output[base.FEATURES]))
    loss = np.expm1(loss_model.predict(output[base.FEATURES]))
    gain = np.maximum(gain, 0.0)
    loss = np.maximum(loss, 0.0)
    output["pred_transaction_positive_prob_unweighted"] = probability
    output["pred_transaction_positive_gain"] = gain
    output["pred_transaction_negative_loss"] = loss
    output["pred_expected_transaction_utility"] = (
        probability * gain - (1.0 - probability) * loss
    )
    return output


def ranking_audit(frame: pd.DataFrame) -> dict:
    actual = frame[base.TARGET].to_numpy(float)
    y = frame[base.POSITIVE].to_numpy(int)
    probability = frame.pred_transaction_positive_prob_unweighted.to_numpy(float)
    score = frame.pred_expected_transaction_utility.to_numpy(float)
    rho = spearmanr(actual, score).statistic
    report = {
        "rows": len(frame),
        "positive": int(y.sum()),
        "positive_rate": float(y.mean()),
        "auc": float(roc_auc_score(y, probability)),
        "ap": float(average_precision_score(y, probability)),
        "expected_utility_spearman": None if not np.isfinite(rho) else float(rho),
        "predicted_positive_expected_utility": int((score > 0).sum()),
    }
    order = np.argsort(-score)
    for k in [10, 25, 50, 100, 250, 500, 1000, 2500]:
        indices = order[: min(k, len(order))]
        values = actual[indices]
        report[f"top{k}_selected"] = len(indices)
        report[f"top{k}_positive"] = int((values > 0).sum())
        report[f"top{k}_precision"] = float((values > 0).mean())
        report[f"top{k}_delta_sum"] = float(values.sum())
        report[f"top{k}_positive_sum"] = float(values[values > 0].sum())
        report[f"top{k}_negative_sum"] = float(values[values < 0].sum())
    positive_score_values = actual[score > 0]
    report["score_positive_selected"] = len(positive_score_values)
    report["score_positive_precision"] = (
        float((positive_score_values > 0).mean()) if len(positive_score_values) else 0.0
    )
    report["score_positive_delta_sum"] = float(positive_score_values.sum())
    return report


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frames = []
    for seq in SEQS:
        raw = pd.read_parquet(ROOT / seq / "cross_chain_transaction_utility.parquet")
        frames.append(base.add_chain_features(seq, raw))
    all_data = pd.concat(frames, ignore_index=True, sort=False)
    reports = []
    for fold_index, held in enumerate(SEQS):
        train = all_data[all_data.seq != held].copy()
        test = all_data[all_data.seq == held].copy()
        models = fit_models(train, fold_index)
        predictions = predict(test, models)
        predictions.to_parquet(
            OUT / f"{held}_chain_expected_utility_predictions.parquet", index=False
        )
        report = {
            "held_out_seq": held,
            "train_rows": len(train),
            "train_positive": int(train[base.POSITIVE].sum()),
            **ranking_audit(predictions),
        }
        reports.append(report)
        print(json.dumps(report), flush=True)
    output = {
        "protocol": {
            "validation": (
                "strict leave-one-sequence-out; held sequence excluded from "
                "all three model fits"
            ),
            "target": base.TARGET,
            "model": (
                "unweighted sign probability plus separate log positive-gain "
                "and negative-loss regressors"
            ),
            "score": "P(positive)*gain - (1-P(positive))*loss",
            "features": base.FEATURES,
            "feature_gt_use": "none; explicit allowlist",
            "training_gt_use": "chain-transaction sign and magnitude labels only",
            "inference": "GT-free",
            "status": "diagnostic ranking only; no TrackEval policy selected here",
        },
        "folds": reports,
    }
    (OUT / "report.json").write_text(json.dumps(output, indent=2) + "\n")


base = load_base()


if __name__ == "__main__":
    main()
