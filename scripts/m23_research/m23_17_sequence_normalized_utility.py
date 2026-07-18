from __future__ import annotations

# Research artifact for the MOT20 M23 sequence-normalized utility audit.

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


SEQS = ["MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05"]
TOP_K_GRID = [10, 25, 50, 100, 250]
LABEL_ROOT = Path(
    "outputs/mot20_m23_20260718/micrograph_chain_transaction_oracle_v1/labels"
)
RAW_OUTER_PRED = Path(
    "outputs/mot20_m23_20260718/"
    "micrograph_chain_expected_utility_loso_v2_segment_app"
)
RAW_INNER_PRED = Path(
    "outputs/mot20_m23_20260718/"
    "m23_13_nested_chain_risk_policy_v1/inner_predictions"
)
META = Path("outputs/mot20_m23_20260718/micrograph_chunk30_v1")
UTILITY = Path("outputs/mot20_m23_20260718/micrograph_chunk30_assa_utility_v1")
OUT = Path("outputs/mot20_m23_20260718/m23_17_sequence_normalized_utility_v1")
NAME = "sequence_normalized_utility_ood_policy_v1"
TARGET = "chain_transaction_delta_proxy"
SCORE = "pred_sequence_normalized_utility"
TRAINING_GT_USE = "sequence-normalized transaction utility target on fit sequences only"
FEATURE_TRANSFORM_DESCRIPTION = (
    "continuous features are converted to within-sequence percentile ranks; "
    "low-cardinality features retain their observable values"
)
TARGET_TRANSFORM_DESCRIPTION = (
    "asinh(raw transaction utility / fit-sequence q90 absolute utility); "
    "positive samples receive a capped square-root imbalance weight"
)
STATUS_DESCRIPTION = (
    "nested sequence-normalized utility audit on reused development "
    "sequences; fixed-parent provenance remains exploratory"
)
RAW_PROBABILITY = "pred_transaction_positive_prob_unweighted"
OOD_QUANTILE = 0.90
USE_FROZEN_M14_OOD_GATE = True
EPS = 1e-9


def augment_model_features(
    seq: str,
    frame: pd.DataFrame,
    features: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    return frame, features


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def rank_normalize(frame: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, list[str]]:
    output = frame.copy()
    normalized = []
    ranked_columns = {}
    for feature in features:
        name = f"seqrank__{feature}"
        values = output[feature]
        finite = values[np.isfinite(values.to_numpy(float))]
        unique = finite.nunique(dropna=True)
        if unique <= 4:
            ranked = values.astype(float)
        else:
            ranked = values.rank(method="average", pct=True).astype(float)
        ranked_columns[name] = ranked.fillna(0.5).astype(np.float32)
        normalized.append(name)
    output = pd.concat(
        [output, pd.DataFrame(ranked_columns, index=output.index)], axis=1
    )
    return output, normalized


def target_and_weights(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict]:
    target = np.zeros(len(frame), dtype=np.float64)
    sequence_weight = np.zeros(len(frame), dtype=np.float64)
    scales = {}
    counts = frame.seq.value_counts()
    for seq, indices in frame.groupby("seq", sort=False).groups.items():
        values = frame.loc[indices, TARGET].to_numpy(float)
        nonzero = np.abs(values[np.abs(values) > EPS])
        scale = float(np.quantile(nonzero, 0.90)) if len(nonzero) else 1.0
        scale = max(scale, 1.0)
        positions = frame.index.get_indexer(indices)
        target[positions] = np.arcsinh(values / scale)
        sequence_weight[positions] = len(frame) / (len(counts) * counts[seq])
        scales[seq] = scale
    positive = int((frame[TARGET].to_numpy(float) > 0).sum())
    negative = len(frame) - positive
    positive_multiplier = min(8.0, math.sqrt(negative / max(positive, 1)))
    sign_weight = np.where(frame[TARGET].to_numpy(float) > 0, positive_multiplier, 1.0)
    magnitude_weight = 1.0 + np.minimum(np.abs(target), 3.0)
    weights = sequence_weight * sign_weight * magnitude_weight
    return target, weights, {
        "per_sequence_q90_abs_utility_scale": scales,
        "positive_multiplier": positive_multiplier,
    }


def fit_model(frame: pd.DataFrame, features: list[str], seed: int):
    target, weights, metadata = target_and_weights(frame)
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=240,
        learning_rate=0.06,
        max_leaf_nodes=31,
        min_samples_leaf=80,
        l2_regularization=16.0,
        max_bins=255,
        random_state=seed,
        early_stopping=False,
    )
    model.fit(frame[features], target, sample_weight=weights)
    return model, metadata


def truncate_then_disjoint(frame: pd.DataFrame, top_k: int) -> pd.DataFrame:
    candidates = frame.copy()
    candidates["policy_score"] = candidates[SCORE].to_numpy(float)
    candidates.sort_values(
        ["policy_score", "src_chunk", "dst_chunk"],
        ascending=[False, True, True],
        inplace=True,
    )
    candidates = candidates.head(top_k)
    used_tracks: set[int] = set()
    selected_indices = []
    for index, edge in candidates.iterrows():
        src_track = int(edge.transaction_src_track_id)
        dst_track = int(edge.transaction_dst_track_id)
        if src_track in used_tracks or dst_track in used_tracks:
            continue
        used_tracks.add(src_track)
        used_tracks.add(dst_track)
        selected_indices.append(index)
    return candidates.loc[selected_indices].copy()


def selection_audit(selected: pd.DataFrame) -> dict:
    actual = selected[TARGET].to_numpy(float)
    return {
        "selected": len(selected),
        "true_positive": int((actual > 0).sum()),
        "true_precision": float((actual > 0).mean()) if len(actual) else None,
        "true_delta_sum": float(actual.sum()),
        "predicted_score_sum": float(selected.policy_score.sum()),
    }


def calibrate_top_k(inner_predictions: dict[str, pd.DataFrame]) -> tuple[int | None, dict]:
    candidates = []
    for top_k in TOP_K_GRID:
        by_seq = {}
        deltas = []
        for seq, prediction in inner_predictions.items():
            selected = truncate_then_disjoint(prediction, top_k)
            audit = selection_audit(selected)
            by_seq[seq] = audit
            deltas.append(audit["true_delta_sum"])
        candidates.append(
            {
                "top_k": top_k,
                "median_true_delta_sum": float(np.median(deltas)),
                "total_true_delta_sum": float(np.sum(deltas)),
                "total_selected": int(sum(item["selected"] for item in by_seq.values())),
                "by_seq": by_seq,
            }
        )
    positive = [item for item in candidates if item["median_true_delta_sum"] > EPS]
    chosen = None
    if positive:
        best = max(positive, key=lambda item: (item["median_true_delta_sum"], -item["top_k"]))
        chosen = int(best["top_k"])
    return chosen, {
        "rule": (
            "choose the truncate-then-disjoint K with the largest positive "
            "median true transaction-utility sum across the three inner held "
            "sequences; otherwise no-op"
        ),
        "top_k_grid": TOP_K_GRID,
        "chosen_top_k_before_ood_gate": chosen,
        "candidates": candidates,
    }


def probability_q90(path: Path) -> float:
    frame = pd.read_parquet(path, columns=[RAW_PROBABILITY])
    return float(frame[RAW_PROBABILITY].quantile(OOD_QUANTILE))


def empty_selection(frame: pd.DataFrame) -> pd.DataFrame:
    selected = frame.iloc[0:0].copy()
    selected["policy_score"] = np.asarray([], dtype=float)
    return selected


def inference_manifest(frame: pd.DataFrame, held: str, applied_top_k: int | None) -> pd.DataFrame:
    columns = [
        "src_chunk",
        "dst_chunk",
        "transaction_src_track_id",
        "transaction_dst_track_id",
        SCORE,
        "policy_score",
    ]
    output = frame[columns].copy()
    output.insert(0, "seq", held)
    output.insert(1, "applied_top_k", applied_top_k)
    return output


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    expected = load_module(
        "m23_12_expected_seqnorm",
        Path("scripts/m23_research/m23_12_train_chain_expected_utility_loso.py"),
    )
    oracle = load_module(
        "m23_12_oracle_seqnorm",
        Path("scripts/m23_research/m23_12_chain_transaction_oracle.py"),
    )
    evaluator = load_module(
        "m23_11_evaluator_seqnorm",
        Path("scripts/m23_research/m23_11_eval_utility_graph.py"),
    )
    evaluator.NAME = NAME

    frames = []
    for seq in SEQS:
        raw = pd.read_parquet(LABEL_ROOT / seq / "cross_chain_transaction_utility.parquet")
        enriched = expected.base.add_chain_features(seq, raw)
        enriched, model_features = augment_model_features(
            seq, enriched, list(expected.base.FEATURES)
        )
        enriched, normalized_features = rank_normalize(enriched, model_features)
        frames.append(enriched)
    all_data = pd.concat(frames, ignore_index=True, sort=False)

    model_cache = {}
    model_reports = {}

    def predict(fit_sequences: list[str], target_seq: str) -> pd.DataFrame:
        key = tuple(sorted(fit_sequences))
        if key not in model_cache:
            train = all_data[all_data.seq.isin(key)].copy()
            seed = 17000 + sum((index + 1) * (SEQS.index(seq) + 1) for index, seq in enumerate(key))
            model, metadata = fit_model(train, normalized_features, seed)
            model_cache[key] = model
            model_reports["+".join(key)] = {
                "fit_sequences": list(key),
                "train_rows": len(train),
                **metadata,
            }
        target = all_data[all_data.seq == target_seq].copy()
        target[SCORE] = model_cache[key].predict(target[normalized_features])
        return target

    fold_reports = []
    frozen_manifests = []
    frozen_policy_rows = []
    selected_for_evaluation = {}
    for outer_held in SEQS:
        training_sequences = [seq for seq in SEQS if seq != outer_held]
        inner_predictions = {}
        inner_root = OUT / "inner_predictions" / f"outer_{outer_held}"
        inner_root.mkdir(parents=True, exist_ok=True)
        for inner_held in training_sequences:
            fit_sequences = [seq for seq in training_sequences if seq != inner_held]
            prediction = predict(fit_sequences, inner_held)
            inner_predictions[inner_held] = prediction
            prediction[
                [
                    "seq",
                    "src_chunk",
                    "dst_chunk",
                    "transaction_src_track_id",
                    "transaction_dst_track_id",
                    TARGET,
                    SCORE,
                ]
            ].to_parquet(inner_root / f"{inner_held}_predictions.parquet", index=False)

        calibrated_top_k, calibration = calibrate_top_k(inner_predictions)
        outer = predict(training_sequences, outer_held)
        outer_prediction_root = OUT / "outer_predictions"
        outer_prediction_root.mkdir(parents=True, exist_ok=True)
        outer[
            [
                "seq",
                "src_chunk",
                "dst_chunk",
                "transaction_src_track_id",
                "transaction_dst_track_id",
                SCORE,
            ]
        ].to_parquet(
            outer_prediction_root / f"{outer_held}_predictions_gt_free.parquet",
            index=False,
        )

        if USE_FROZEN_M14_OOD_GATE:
            inner_q90 = {
                inner_held: probability_q90(
                    RAW_INNER_PRED
                    / f"outer_{outer_held}"
                    / f"{inner_held}_predictions.parquet"
                )
                for inner_held in training_sequences
            }
            outer_q90 = probability_q90(
                RAW_OUTER_PRED
                / f"{outer_held}_chain_expected_utility_predictions.parquet"
            )
            ood_threshold = float(max(inner_q90.values()))
            upper_ood = outer_q90 > ood_threshold
            ood_source = "frozen M23-14 raw expected-utility probability q90 gate"
        else:
            inner_q90 = {}
            outer_q90 = None
            ood_threshold = None
            upper_ood = False
            ood_source = "disabled before freeze by the experiment protocol"
        applied_top_k = None if upper_ood else calibrated_top_k
        selected = (
            empty_selection(outer)
            if applied_top_k is None
            else truncate_then_disjoint(outer, applied_top_k)
        )
        selected_for_evaluation[outer_held] = selected
        frozen_manifests.append(inference_manifest(selected, outer_held, applied_top_k))
        policy_row = {
            "seq": outer_held,
            "calibrated_top_k": calibrated_top_k,
            "outer_raw_probability_q90": outer_q90,
            "inner_raw_probability_q90_max": ood_threshold,
            "upper_ood": upper_ood,
            "applied_top_k": applied_top_k,
            "outer_selected": len(selected),
            "outer_predicted_score_sum": float(selected.policy_score.sum()),
        }
        frozen_policy_rows.append(policy_row)
        fold_reports.append(
            {
                "outer_held_seq": outer_held,
                "outer_model_fit_sequences": training_sequences,
                "outer_gt_used_in_model_or_policy_selection": False,
                "calibration": calibration,
                "ood_gate": {
                    "source": ood_source,
                    "enabled": USE_FROZEN_M14_OOD_GATE,
                    "inner_q90": inner_q90,
                    "outer_q90": outer_q90,
                    "upper_threshold": ood_threshold,
                    "upper_ood": upper_ood,
                },
                "applied_top_k": applied_top_k,
                "outer_selected": len(selected),
                "outer_predicted_score_sum": float(selected.policy_score.sum()),
            }
        )
        print(json.dumps(policy_row), flush=True)

    manifest = pd.concat(frozen_manifests, ignore_index=True, sort=False)
    manifest_path = OUT / "frozen_outer_selection.csv"
    manifest.to_csv(manifest_path, index=False)
    selection_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    policy_frame = pd.DataFrame(frozen_policy_rows)
    policy_path = OUT / "frozen_outer_policy.csv"
    policy_frame.to_csv(policy_path, index=False)
    policy_sha256 = hashlib.sha256(policy_path.read_bytes()).hexdigest()
    frozen_protocol = {
        "status": "outer policy and transaction selection frozen before TrackEval",
        "selection_sha256": selection_sha256,
        "policy_sha256": policy_sha256,
        "outer_gt_used_in_model_or_policy_selection": False,
        "training_gt_use": TRAINING_GT_USE,
        "inference_gt_use": "none",
        "features": normalized_features,
        "feature_transform": FEATURE_TRANSFORM_DESCRIPTION,
        "target_transform": TARGET_TRANSFORM_DESCRIPTION,
        "models": model_reports,
        "folds": fold_reports,
    }
    (OUT / "frozen_protocol.json").write_text(json.dumps(frozen_protocol, indent=2) + "\n")

    root = OUT / NAME
    tracker_reports = []
    outer_diagnostics = []
    for held in SEQS:
        selected = selected_for_evaluation[held]
        # Outer labels are consumed only after both frozen manifests are written.
        outer_diagnostics.append({"seq": held, **selection_audit(selected)})
        edges = pd.read_parquet(UTILITY / held / "candidate_edges_utility.parquet")
        applied = oracle.apply_transactions(edges, selected)
        meta = pd.read_parquet(META / held / "microtracklets.parquet")
        tracker_reports.append(
            evaluator.write_tracker(
                held,
                meta,
                applied,
                root / "track_results" / f"{held}.txt",
            )
        )

    report = {
        "name": NAME,
        "status": STATUS_DESCRIPTION,
        "deployment_allowed": False,
        "parent": (
            "fixed GT-free exploratory fused A42 parent at 78.763497 HOTA; "
            "formal deployable anchor remains 77.699"
        ),
        "selection_sha256": selection_sha256,
        "policy_sha256": policy_sha256,
        "protocol": frozen_protocol,
        "outer_postfreeze_diagnostics": outer_diagnostics,
        "tracker_reports": tracker_reports,
        "eval": evaluator.evaluate(root),
    }
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
