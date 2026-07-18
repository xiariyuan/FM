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
ROOT = Path("outputs/mot20_m23_20260718/micrograph_chain_transaction_oracle_v1/labels")
META = Path("outputs/mot20_m23_20260718/micrograph_chunk30_v1")
OUT = Path("outputs/mot20_m23_20260718/micrograph_chain_transaction_loso_v1")
TARGET = "chain_transaction_delta_proxy"
POSITIVE = "chain_transaction_positive"
EDGE_FEATURES = [
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
CHAIN_FEATURES = [
    "transaction_removes_source_out",
    "transaction_removes_source_in",
    "src_track_rows",
    "dst_track_rows",
    "src_track_chunks",
    "dst_track_chunks",
    "src_track_span",
    "dst_track_span",
    "src_prefix_rows",
    "src_suffix_rows",
    "dst_prefix_rows",
    "dst_suffix_rows",
    "merged_segment_rows",
    "detached_segment_rows",
    "src_cut_fraction",
    "dst_cut_fraction",
    "merged_balance",
    "detached_fraction",
    "log_src_track_rows",
    "log_dst_track_rows",
    "log_merged_segment_rows",
    "log_detached_segment_rows",
    "segment_appearance_cos",
    "track_appearance_cos",
    "src_prefix_coherence",
    "dst_suffix_coherence",
    "src_suffix_coherence",
    "dst_prefix_coherence",
    "merged_segment_coherence",
    "merged_coherence_gain",
    "src_prefix_endpoint_cos",
    "dst_suffix_endpoint_cos",
]
FEATURES = EDGE_FEATURES + CHAIN_FEATURES


def normalized_rows(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(values, axis=1)
    output = values / np.maximum(norms[:, None], 1e-12)
    return output, norms


def add_chain_features(seq: str, frame: pd.DataFrame) -> pd.DataFrame:
    meta = pd.read_parquet(META / seq / "microtracklets.parquet").copy()
    prototypes = np.load(META / seq / "prototypes.f16.npy").astype(np.float32)
    prototypes, _ = normalized_rows(prototypes)
    meta.sort_values(["source_track_id", "source_ordinal"], inplace=True)
    grouped = meta.groupby("source_track_id", sort=False)
    meta["track_rows"] = grouped.rows.transform("sum")
    meta["track_chunks"] = grouped.rows.transform("size")
    meta["track_first"] = grouped.first_frame.transform("min")
    meta["track_last"] = grouped.last_frame.transform("max")
    meta["track_span"] = meta.track_last - meta.track_first + 1
    meta["prefix_rows"] = grouped.rows.cumsum()
    meta["suffix_rows"] = meta.track_rows - meta.prefix_rows + meta.rows
    meta.sort_values("chunk_id", inplace=True)
    if meta.chunk_id.tolist() != list(range(len(meta))):
        raise RuntimeError(f"{seq}: chunk ids are not dense")

    prefix_vectors = np.zeros_like(prototypes)
    suffix_vectors = np.zeros_like(prototypes)
    track_vectors = np.zeros_like(prototypes)
    for _, part in meta.groupby("source_track_id", sort=False):
        ids = part.chunk_id.to_numpy(int)
        weights = part.rows.to_numpy(np.float32)
        weighted = prototypes[ids] * weights[:, None]
        prefix = np.cumsum(weighted, axis=0)
        suffix = np.cumsum(weighted[::-1], axis=0)[::-1]
        prefix_vectors[ids] = prefix
        suffix_vectors[ids] = suffix
        track_vectors[ids] = prefix[-1]
    prefix_unit, prefix_norm = normalized_rows(prefix_vectors)
    suffix_unit, suffix_norm = normalized_rows(suffix_vectors)
    track_unit, track_norm = normalized_rows(track_vectors)

    src = frame.src_chunk.to_numpy(int)
    dst = frame.dst_chunk.to_numpy(int)
    src_track_rows = meta.track_rows.to_numpy(float)[src]
    dst_track_rows = meta.track_rows.to_numpy(float)[dst]
    src_prefix = meta.prefix_rows.to_numpy(float)[src]
    src_suffix = src_track_rows - src_prefix
    dst_suffix = meta.suffix_rows.to_numpy(float)[dst]
    dst_prefix = dst_track_rows - dst_suffix
    merged = src_prefix + dst_suffix
    detached = src_suffix + dst_prefix

    output = frame.copy()
    output["src_track_rows"] = src_track_rows
    output["dst_track_rows"] = dst_track_rows
    output["src_track_chunks"] = meta.track_chunks.to_numpy(float)[src]
    output["dst_track_chunks"] = meta.track_chunks.to_numpy(float)[dst]
    output["src_track_span"] = meta.track_span.to_numpy(float)[src]
    output["dst_track_span"] = meta.track_span.to_numpy(float)[dst]
    output["src_prefix_rows"] = src_prefix
    output["src_suffix_rows"] = src_suffix
    output["dst_prefix_rows"] = dst_prefix
    output["dst_suffix_rows"] = dst_suffix
    output["merged_segment_rows"] = merged
    output["detached_segment_rows"] = detached
    output["src_cut_fraction"] = src_prefix / np.maximum(src_track_rows, 1)
    output["dst_cut_fraction"] = dst_prefix / np.maximum(dst_track_rows, 1)
    output["merged_balance"] = np.minimum(src_prefix, dst_suffix) / np.maximum(
        np.maximum(src_prefix, dst_suffix), 1
    )
    output["detached_fraction"] = detached / np.maximum(
        src_track_rows + dst_track_rows, 1
    )
    output["log_src_track_rows"] = np.log1p(src_track_rows)
    output["log_dst_track_rows"] = np.log1p(dst_track_rows)
    output["log_merged_segment_rows"] = np.log1p(merged)
    output["log_detached_segment_rows"] = np.log1p(detached)
    output["segment_appearance_cos"] = np.einsum(
        "ij,ij->i", prefix_unit[src], suffix_unit[dst]
    )
    output["track_appearance_cos"] = np.einsum(
        "ij,ij->i", track_unit[src], track_unit[dst]
    )
    output["src_prefix_coherence"] = prefix_norm[src] / np.maximum(src_prefix, 1)
    output["dst_suffix_coherence"] = suffix_norm[dst] / np.maximum(dst_suffix, 1)
    output["src_suffix_coherence"] = (
        np.linalg.norm(track_vectors[src] - prefix_vectors[src], axis=1)
        / np.maximum(src_suffix, 1)
    )
    output["dst_prefix_coherence"] = (
        np.linalg.norm(track_vectors[dst] - suffix_vectors[dst], axis=1)
        / np.maximum(dst_prefix, 1)
    )
    merged_vectors = prefix_vectors[src] + suffix_vectors[dst]
    output["merged_segment_coherence"] = (
        np.linalg.norm(merged_vectors, axis=1) / np.maximum(merged, 1)
    )
    separate_coherence = (
        prefix_norm[src] + suffix_norm[dst]
    ) / np.maximum(merged, 1)
    output["merged_coherence_gain"] = (
        output.merged_segment_coherence.to_numpy(float) - separate_coherence
    )
    output["src_prefix_endpoint_cos"] = np.einsum(
        "ij,ij->i", prefix_unit[src], prototypes[src]
    )
    output["dst_suffix_endpoint_cos"] = np.einsum(
        "ij,ij->i", suffix_unit[dst], prototypes[dst]
    )
    output.insert(0, "seq", seq)
    return output


def sequence_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.seq.value_counts()
    mapping = {seq: len(frame) / (len(counts) * count) for seq, count in counts.items()}
    return frame.seq.map(mapping).to_numpy(float)


def classification_weights(frame: pd.DataFrame) -> np.ndarray:
    y = frame[POSITIVE].to_numpy(int)
    positive = int(y.sum())
    negative = len(y) - positive
    multiplier = min(25.0, negative / max(positive, 1))
    return sequence_weights(frame) * np.where(y == 1, multiplier, 1.0)


def regression_weights(frame: pd.DataFrame) -> np.ndarray:
    magnitude = np.clip(np.abs(frame[TARGET].to_numpy(float)), 0.0, 50.0)
    return sequence_weights(frame) * (1.0 + magnitude)


def fit_models(frame: pd.DataFrame, fold_index: int):
    classifier = HistGradientBoostingClassifier(
        max_iter=320,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        l2_regularization=10.0,
        max_bins=255,
        random_state=12000 + fold_index,
        early_stopping=False,
    )
    classifier.fit(
        frame[FEATURES],
        frame[POSITIVE].astype(int),
        sample_weight=classification_weights(frame),
    )
    regressor = HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=360,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        l2_regularization=12.0,
        max_bins=255,
        random_state=12100 + fold_index,
        early_stopping=False,
    )
    regressor.fit(
        frame[FEATURES],
        np.arcsinh(frame[TARGET].to_numpy(float)),
        sample_weight=regression_weights(frame),
    )
    return classifier, regressor


def predict(frame: pd.DataFrame, classifier, regressor) -> pd.DataFrame:
    output = frame.copy()
    output["pred_transaction_positive_prob"] = classifier.predict_proba(
        output[FEATURES]
    )[:, 1]
    output["pred_transaction_asinh"] = regressor.predict(output[FEATURES])
    output["pred_transaction_delta"] = np.sinh(
        output.pred_transaction_asinh.to_numpy(float)
    )
    output["pred_risk_adjusted_transaction_utility"] = (
        output.pred_transaction_positive_prob.to_numpy(float)
        * np.maximum(output.pred_transaction_delta.to_numpy(float), 0.0)
    )
    return output


def ranking_audit(frame: pd.DataFrame) -> dict:
    actual = frame[TARGET].to_numpy(float)
    y = frame[POSITIVE].to_numpy(int)
    probability = frame.pred_transaction_positive_prob.to_numpy(float)
    predicted = frame.pred_transaction_delta.to_numpy(float)
    score = frame.pred_risk_adjusted_transaction_utility.to_numpy(float)
    rho = spearmanr(actual, predicted).statistic
    report = {
        "rows": len(frame),
        "positive": int(y.sum()),
        "positive_rate": float(y.mean()),
        "auc": float(roc_auc_score(y, probability)),
        "ap": float(average_precision_score(y, probability)),
        "delta_spearman": None if not np.isfinite(rho) else float(rho),
        "delta_mae": float(mean_absolute_error(actual, predicted)),
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
    return report


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frames = []
    for seq in SEQS:
        raw = pd.read_parquet(ROOT / seq / "cross_chain_transaction_utility.parquet")
        frames.append(add_chain_features(seq, raw))
    all_data = pd.concat(frames, ignore_index=True, sort=False)
    reports = []
    for fold_index, held in enumerate(SEQS):
        train = all_data[all_data.seq != held].copy()
        test = all_data[all_data.seq == held].copy()
        classifier, regressor = fit_models(train, fold_index)
        predictions = predict(test, classifier, regressor)
        predictions.to_parquet(
            OUT / f"{held}_chain_transaction_predictions.parquet", index=False
        )
        report = {
            "held_out_seq": held,
            "train_rows": len(train),
            "train_positive": int(train[POSITIVE].sum()),
            **ranking_audit(predictions),
        }
        reports.append(report)
        print(json.dumps(report), flush=True)
    output = {
        "protocol": {
            "validation": (
                "strict leave-one-sequence-out; held sequence excluded from "
                "classifier and regressor fitting"
            ),
            "target": TARGET,
            "target_transform": "asinh signed chain-transaction utility",
            "features": FEATURES,
            "feature_gt_use": "none; explicit allowlist",
            "training_gt_use": "chain-transaction utility labels and weights only",
            "inference": "GT-free",
            "status": "diagnostic ranking only; no TrackEval policy selected here",
        },
        "folds": reports,
    }
    (OUT / "report.json").write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
