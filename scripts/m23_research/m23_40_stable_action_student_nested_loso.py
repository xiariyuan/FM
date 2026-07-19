#!/usr/bin/env python3
from __future__ import annotations

"""M23-40: action-specific, stability-selected exact-HOTA student.

This wrapper reuses the fully audited M23-39 nested-LOSO execution and tracker
freezing path, but replaces its high-dimensional learner.  Drop and replacement
actions get separate models.  Features are retained only when their target
direction is stable across the available outer-training sequences; a strongly
regularized logistic head is blended with a monotone correlation ensemble.
"""

import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge


REPO = Path(__file__).resolve().parents[2]
AUDIT: List[Dict[str, object]] = []

REPLACEMENT_CANDIDATES = (
    "motion_error_mean", "motion_error_min", "forward_motion_error",
    "backward_motion_error", "sequence_motion_percentile",
    "motion_appearance_joint", "endpoint_displacement", "velocity_cos",
    "src_motion_percentile", "dst_motion_percentile", "motion_rank_consensus",
    "gap", "sequence_gap_percentile", "detached_fraction",
    "conflict_per_merged_row", "in_rank", "max_rank", "mapping_rate_min",
    "removed_baseline_actions", "pred_positive_probability",
)
DROP_CANDIDATES = (
    "detached_fraction", "merged_segment_rows", "max_rank",
    "backward_motion_error", "max_margin", "pair_candidate_count",
    "src_motion_percentile", "dst_motion_percentile", "motion_rank_consensus",
    "motion_error_mean", "dst_appearance_percentile", "src_match_iou",
    "dst_suffix_rows", "conflict_per_merged_row", "pred_positive_probability",
    "pred_normalized_gain", "pred_normalized_loss",
)


def load_m39():
    path = REPO / "scripts/m23_research/m23_39_exact_advantage_student_nested_loso.py"
    spec = importlib.util.spec_from_file_location("m23_40_m39", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


m39 = load_m39()


def default_policies():
    policies = [m39.Policy("noop", 1.0, 0.0, 1.0, 1.0, False, 0)]
    settings = (
        (1.0, 0.00, 0.65, 0.45),
        (1.0, 0.00, 0.75, 0.48),
        (1.25, 0.03, 0.82, 0.50),
        (1.50, 0.05, 0.88, 0.52),
        (2.00, 0.08, 0.93, 0.55),
    )
    for risk, uncertainty, quantile, probability in settings:
        tag = f"r{risk:g}_u{uncertainty:g}_q{quantile:g}_p{probability:g}".replace(".", "p")
        policies.append(m39.Policy(f"replace_{tag}", risk, uncertainty, quantile, probability, False))
        policies.append(m39.Policy(f"combo_{tag}", risk, uncertainty, quantile, probability, True))
    return policies


def correlation_table(frame: pd.DataFrame, candidates: Sequence[str]) -> List[Dict[str, object]]:
    sequence_count = int(frame.seq.astype(str).nunique())
    rows: List[Dict[str, object]] = []
    for feature in candidates:
        correlations = []
        for _seq, part in frame.groupby("seq"):
            if part[feature].nunique(dropna=False) <= 1 or part.delta_HOTA.nunique() <= 1:
                continue
            value = part[feature].corr(part.delta_HOTA, method="spearman")
            if pd.notna(value):
                correlations.append(float(value))
        if not correlations:
            continue
        median = float(np.median(correlations))
        direction = float(np.sign(median) or 1.0)
        agreement = float(np.mean(np.sign(correlations) == direction))
        minimum = float(min(abs(value) for value in correlations))
        score = minimum * agreement if len(correlations) > 1 else abs(median)
        stable = (
            len(correlations) >= max(1, sequence_count - 0)
            and agreement >= 0.999
            and minimum >= 0.02
        )
        rows.append({
            "feature": feature, "correlations": correlations, "direction": direction,
            "agreement": agreement, "minimum_abs_correlation": minimum,
            "stability_score": score, "stable": stable,
        })
    return rows


def select_stable_features(frame: pd.DataFrame, action_type: str):
    candidates = REPLACEMENT_CANDIDATES if action_type == "replace" else DROP_CANDIDATES
    limit = 8 if action_type == "replace" else 5
    table = correlation_table(frame, candidates)
    stable = sorted(
        [row for row in table if row["stable"]],
        key=lambda row: (float(row["stability_score"]), str(row["feature"])),
        reverse=True,
    )
    if len(stable) < 2:
        stable = sorted(
            table,
            key=lambda row: (float(row["agreement"]), float(row["stability_score"])),
            reverse=True,
        )
    selected = []
    for row in stable:
        feature = str(row["feature"])
        # Avoid spending the small feature budget on nearly duplicate ranks.
        if any(abs(frame[feature].corr(frame[other], method="spearman")) > 0.985 for other in selected):
            continue
        selected.append(feature)
        if len(selected) >= limit:
            break
    if not selected:
        selected = list(candidates[:2])
    by_feature = {str(row["feature"]): row for row in table}
    directions = np.asarray([float(by_feature[feature]["direction"]) for feature in selected])
    weights = np.asarray([max(float(by_feature[feature]["stability_score"]), 0.02) for feature in selected])
    weights /= weights.sum()
    return selected, directions, weights, table


def fit_magnitude(frame: pd.DataFrame, features: Sequence[str], mask: np.ndarray, normalized: np.ndarray):
    if mask.sum() < 2:
        return {"kind": "constant", "value": float(np.median(normalized[mask])) if mask.any() else 0.0}
    model = Ridge(alpha=12.0)
    model.fit(
        frame.loc[mask, list(features)].to_numpy(float), normalized[mask],
        sample_weight=m39.sequence_weights(frame.loc[mask]),
    )
    return {"kind": "ridge", "model": model}


def predict_magnitude(model, matrix: np.ndarray):
    if model["kind"] == "constant":
        values = np.full(len(matrix), float(model["value"]), dtype=float)
    else:
        values = model["model"].predict(matrix)
    return np.clip(np.expm1(values), 0.0, 20.0)


def fit_action_model(frame: pd.DataFrame, action_type: str, seed: int):
    part = frame[frame.action_type.astype(str) == action_type].copy()
    if part.empty:
        raise RuntimeError(f"no {action_type} actions in training data")
    sign = (part.delta_HOTA.to_numpy(float) > 0.0).astype(int)
    if np.unique(sign).size < 2:
        raise RuntimeError(f"one-class {action_type} labels")
    features, directions, weights, table = select_stable_features(part, action_type)
    matrix = part[features].to_numpy(float)
    logistic = LogisticRegression(
        C=0.08, class_weight="balanced", max_iter=3000, random_state=seed
    )
    logistic.fit(matrix, sign, sample_weight=m39.sequence_weights(part))
    monotone_score = (matrix * directions * weights).sum(axis=1).reshape(-1, 1)
    calibrator = LogisticRegression(
        C=0.25, class_weight="balanced", max_iter=1000, random_state=seed + 1
    )
    calibrator.fit(monotone_score, sign, sample_weight=m39.sequence_weights(part))

    scales = m39.target_scales(part)
    scale = part.seq.astype(str).map(scales).to_numpy(float)
    normalized = np.log1p(np.abs(part.delta_HOTA.to_numpy(float)) / scale)
    gain = fit_magnitude(part, features, sign > 0, normalized)
    loss = fit_magnitude(part, features, sign == 0, normalized)
    AUDIT.append({
        "training_sequences": sorted(part.seq.astype(str).unique()),
        "action_type": action_type, "rows": int(len(part)),
        "positive_rows": int(sign.sum()), "selected_features": features,
        "feature_directions": directions.tolist(), "feature_weights": weights.tolist(),
        "correlation_table": table,
    })
    return {
        "features": features, "directions": directions, "weights": weights,
        "logistic": logistic, "calibrator": calibrator, "gain": gain, "loss": loss,
    }


def fit_models(frame: pd.DataFrame, seed: int):
    return {
        "replace": fit_action_model(frame, "replace", seed),
        "drop": fit_action_model(frame, "drop", seed + 1000),
    }


def predict_models(frame: pd.DataFrame, models) -> pd.DataFrame:
    output = frame.copy()
    for column in (
        "student_positive_probability", "student_gain", "student_loss",
        "student_disagreement", "student_p_hgb", "student_p_linear",
        "student_selected_feature_count",
    ):
        output[column] = 0.0
    for action_type, model in models.items():
        mask = output.action_type.astype(str) == action_type
        if not mask.any():
            continue
        matrix = output.loc[mask, model["features"]].to_numpy(float)
        p_linear = model["logistic"].predict_proba(matrix)[:, 1]
        monotone_score = (matrix * model["directions"] * model["weights"]).sum(axis=1).reshape(-1, 1)
        p_monotone = model["calibrator"].predict_proba(monotone_score)[:, 1]
        probability = 0.45 * p_linear + 0.55 * p_monotone
        output.loc[mask, "student_positive_probability"] = probability
        output.loc[mask, "student_gain"] = predict_magnitude(model["gain"], matrix)
        output.loc[mask, "student_loss"] = predict_magnitude(model["loss"], matrix)
        output.loc[mask, "student_disagreement"] = np.abs(p_linear - p_monotone)
        output.loc[mask, "student_p_hgb"] = p_monotone
        output.loc[mask, "student_p_linear"] = p_linear
        output.loc[mask, "student_selected_feature_count"] = len(model["features"])
    return output


def output_root_from_argv() -> Path | None:
    if "--output-root" not in sys.argv:
        return None
    index = sys.argv.index("--output-root")
    return Path(sys.argv[index + 1]) if index + 1 < len(sys.argv) else None


def patch_reports(output_root: Path) -> None:
    for name in ("protocol.json", "report.json"):
        path = output_root / name
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["experiment"] = "M23-40 stable action-specific exact-HOTA student"
        payload["model_structure"] = (
            "separate drop/replacement heads; cross-training-sequence stability selection; "
            "regularized logistic plus monotone correlation ensemble"
        )
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_root / "stable_model_audit.json").write_text(
        json.dumps(AUDIT, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    m39.fit_models = fit_models
    m39.predict_models = predict_models
    m39.default_policies = default_policies
    output_root = output_root_from_argv()
    m39.main()
    if output_root is not None:
        patch_reports(output_root)


if __name__ == "__main__":
    main()
