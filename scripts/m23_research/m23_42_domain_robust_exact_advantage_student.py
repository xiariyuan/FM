#!/usr/bin/env python3
from __future__ import annotations

"""M23-42: domain-robust student over the expanded M23-41 action bank.

The outer-held sequence label file is never read.  Each training sequence gets
its own low-capacity expert, alongside a pooled expert.  Candidate value is a
lower-quantile consensus across source-domain experts, with explicit penalties
for cross-domain downside and disagreement.  This is intended to prevent the
large single-domain failures seen in M23-39/40 while exploiting the diverse
GT-free M23-41 shortlist.
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


def load_m39():
    path = REPO / "scripts/m23_research/m23_39_exact_advantage_student_nested_loso.py"
    spec = importlib.util.spec_from_file_location("m23_42_m39", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


m39 = load_m39()
ORIGINAL_WITHIN_SEQUENCE_FEATURES = m39.within_sequence_features
ORIGINAL_MAKE_HELD_ACTION_FRAME = m39.make_held_action_frame

m39.LABEL_PATHS = {
    seq: Path(
        f"outputs/mot20_m23_20260718/m23_41_diverse_exact_advantage_{seq[-2:]}_v1/"
        "exact_action_labels.parquet"
    )
    for seq in m39.SEQUENCES
}

CHANNEL_FEATURES = (
    "channel_parent_policy",
    "channel_motion",
    "channel_appearance",
    "channel_structure",
    "channel_exploration",
)
MODEL_FEATURES = (
    "m23_38_parent_policy_score",
    "pred_positive_probability",
    "motion_appearance_joint",
    "max_margin",
    "backward_motion_error",
    "motion_error_mean",
    "motion_error_min",
    "sequence_motion_percentile",
    "motion_rank_consensus",
    "rank_consensus",
    "appearance_cos",
    "segment_appearance_cos",
    "segment_endpoint_gain",
    "max_rank",
    "channel_parent_policy",
)


def within_sequence_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = ORIGINAL_WITHIN_SEQUENCE_FEATURES(frame)
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


def normalized_target(frame: pd.DataFrame) -> np.ndarray:
    values = frame.delta_HOTA.to_numpy(float)
    result = np.zeros(len(frame), dtype=float)
    for seq, indices in frame.groupby(frame.seq.astype(str)).groups.items():
        positions = frame.index.get_indexer(indices)
        local = np.abs(values[positions])
        nonzero = local[local > 1e-8]
        scale = float(np.median(nonzero)) if len(nonzero) else 0.01
        result[positions] = np.clip(values[positions] / max(scale, 1e-4), -8.0, 8.0)
    return result


def fit_expert(frame: pd.DataFrame, seed: int) -> Dict[str, object]:
    x = np.nan_to_num(
        frame[list(MODEL_FEATURES)].to_numpy(float),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    y = normalized_target(frame)
    sign = (frame.delta_HOTA.to_numpy(float) > 0.0).astype(int)
    weight = m39.balanced_sign_weights(frame)

    catastrophic = (frame.delta_HOTA.to_numpy(float) < -0.05).astype(int)

    def fit_classifier(labels: np.ndarray, offset: int):
        if np.unique(labels).size < 2:
            return {"kind": "constant", "value": float(labels.mean())}
        model = LogisticRegression(
            C=0.05, class_weight="balanced", max_iter=3000, random_state=seed + offset
        )
        model.fit(x, labels, sample_weight=weight)
        return {"kind": "logistic", "model": model}

    positive_classifier = fit_classifier(sign, 0)
    catastrophic_classifier = fit_classifier(catastrophic, 1)
    ridge = Ridge(alpha=80.0)
    sequence_weight = m39.sequence_weights(frame)
    ridge.fit(x, y, sample_weight=sequence_weight)
    return {
        "positive_classifier": positive_classifier,
        "catastrophic_classifier": catastrophic_classifier,
        "ridge": ridge,
        "rows": int(len(frame)),
        "positive_rows": int(sign.sum()),
        "catastrophic_rows": int(catastrophic.sum()),
    }


def fit_action_model(frame: pd.DataFrame, action_type: str, seed: int):
    part = frame[frame.action_type.astype(str) == action_type].copy()
    if part.empty:
        raise RuntimeError(f"no {action_type} actions in training data")
    experts = []
    for offset, (seq, local) in enumerate(part.groupby(part.seq.astype(str))):
        if len(local) < 8:
            continue
        experts.append((str(seq), fit_expert(local.reset_index(drop=True), seed + 100 * offset)))
    pooled = fit_expert(part.reset_index(drop=True), seed + 9000)
    AUDIT.append({
        "action_type": action_type,
        "training_sequences": sorted(part.seq.astype(str).unique()),
        "rows": int(len(part)),
        "positive_rows": int((part.delta_HOTA > 0).sum()),
        "domain_experts": [
            {"sequence": seq, "rows": model["rows"], "positive_rows": model["positive_rows"]}
            for seq, model in experts
        ],
        "features": list(MODEL_FEATURES),
    })
    return {"experts": experts, "pooled": pooled}


def fit_models(frame: pd.DataFrame, seed: int):
    return {
        "replace": fit_action_model(frame, "replace", seed),
        "drop": fit_action_model(frame, "drop", seed + 10000),
    }


def predict_expert(frame: pd.DataFrame, model: Dict[str, object]):
    x = np.nan_to_num(
        frame[list(MODEL_FEATURES)].to_numpy(float),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    def probability(bundle):
        if bundle["kind"] == "constant":
            return np.full(len(frame), float(bundle["value"]))
        return bundle["model"].predict_proba(x)[:, 1]

    positive = probability(model["positive_classifier"])
    catastrophic = probability(model["catastrophic_classifier"])
    value = model["ridge"].predict(x)
    return positive, catastrophic, value


def predict_models(frame: pd.DataFrame, models) -> pd.DataFrame:
    output = frame.copy()
    for column in (
        "student_positive_probability", "student_gain", "student_loss",
        "student_disagreement", "student_p_hgb", "student_p_linear",
        "student_robust_value", "student_domain_downside",
        "student_catastrophic_probability",
    ):
        output[column] = 0.0
    for action_type, bundle in models.items():
        mask = output.action_type.astype(str) == action_type
        if not mask.any():
            continue
        local = output.loc[mask]
        pooled_probability, pooled_catastrophic, pooled_value = predict_expert(local, bundle["pooled"])
        domain_outputs = [predict_expert(local, expert) for _seq, expert in bundle["experts"]]
        if domain_outputs:
            probabilities = np.stack([item[0] for item in domain_outputs], axis=1)
            catastrophics = np.stack([item[1] for item in domain_outputs], axis=1)
            values = np.stack([item[2] for item in domain_outputs], axis=1)
            conservative_probability = np.quantile(probabilities, 0.25, axis=1)
            conservative_catastrophic = np.quantile(catastrophics, 0.75, axis=1)
            conservative_value = np.quantile(values, 0.25, axis=1)
            disagreement = (
                np.std(values, axis=1) + np.std(probabilities, axis=1)
                + np.std(catastrophics, axis=1)
            )
        else:
            conservative_probability = pooled_probability
            conservative_catastrophic = pooled_catastrophic
            conservative_value = pooled_value
            disagreement = np.zeros(len(local), dtype=float)
        probability = 0.70 * conservative_probability + 0.30 * pooled_probability
        catastrophic = 0.70 * conservative_catastrophic + 0.30 * pooled_catastrophic
        robust_value = 0.70 * conservative_value + 0.30 * pooled_value
        output.loc[mask, "student_positive_probability"] = probability
        output.loc[mask, "student_robust_value"] = robust_value
        output.loc[mask, "student_domain_downside"] = catastrophic
        output.loc[mask, "student_catastrophic_probability"] = catastrophic
        output.loc[mask, "student_disagreement"] = disagreement
        output.loc[mask, "student_p_hgb"] = conservative_probability
        output.loc[mask, "student_p_linear"] = pooled_probability
        output.loc[mask, "student_gain"] = np.maximum(robust_value, 0.0) + 1e-3
        output.loc[mask, "student_loss"] = np.maximum(-robust_value, 0.0) + catastrophic + 1e-3
    return output


def add_policy_score(frame: pd.DataFrame, policy) -> pd.DataFrame:
    output = frame.copy()
    output["student_value"] = (
        output.student_positive_probability.to_numpy(float) - 0.50
        - policy.risk_lambda * output.student_catastrophic_probability.to_numpy(float)
        - policy.uncertainty_lambda * output.student_disagreement.to_numpy(float)
        + 0.05 * output.student_robust_value.to_numpy(float)
    )
    output["student_value_percentile"] = output.groupby("action_type")["student_value"].rank(
        method="average", pct=True
    )
    return output


def default_policies():
    policies = [m39.Policy("noop", 0.0, 0.0, 1.0, 1.0, False, 0)]
    settings = (
        (0.50, 0.00, 0.90, 0.50, 20),
        (1.00, 0.03, 0.93, 0.52, 16),
        (2.00, 0.05, 0.96, 0.55, 10),
        (4.00, 0.08, 0.98, 0.58, 6),
    )
    for risk, uncertainty, quantile, probability, budget in settings:
        tag = f"r{risk:g}_u{uncertainty:g}_q{quantile:g}_p{probability:g}_k{budget}".replace(".", "p")
        policies.append(m39.Policy(f"replace_{tag}", risk, uncertainty, quantile, probability, False, budget))
        policies.append(m39.Policy(f"combo_{tag}", risk, uncertainty, quantile, probability, True, budget))
    return policies


def make_held_action_frame(seq, baseline_root, label_module, max_replacements):
    original = label_module.residual_replacements

    def diverse(predictions, selected, maximum):
        return original(predictions, selected, maximum, selection_mode="diverse")

    label_module.residual_replacements = diverse
    try:
        return ORIGINAL_MAKE_HELD_ACTION_FRAME(
            seq, baseline_root, label_module, max_replacements
        )
    finally:
        label_module.residual_replacements = original


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
        payload["experiment"] = "M23-42 domain-robust expanded exact-HOTA student"
        payload["model_structure"] = (
            "separate action heads; pooled plus per-training-sequence experts; "
            "lower-quartile value/probability; explicit domain downside and disagreement"
        )
        payload["candidate_space"] = "M23-41 256-action diverse GT-free shortlist"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_root / "domain_robust_model_audit.json").write_text(
        json.dumps(AUDIT, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    m39.within_sequence_features = within_sequence_features
    m39.fit_models = fit_models
    m39.predict_models = predict_models
    m39.add_policy_score = add_policy_score
    m39.default_policies = default_policies
    m39.make_held_action_frame = make_held_action_frame
    output_root = output_root_from_argv()
    m39.main()
    if output_root is not None:
        patch_reports(output_root)


if __name__ == "__main__":
    main()
