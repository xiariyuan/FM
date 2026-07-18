from __future__ import annotations

# Research artifact for the MOT20 M23 sequence-normalized sign audit.

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class PositiveProbabilityModel:
    def __init__(self, classifier):
        self.classifier = classifier

    def predict(self, features):
        return self.classifier.predict_proba(features)[:, 1]


def fit_sign_model(frame: pd.DataFrame, features: list[str], seed: int):
    target = (frame.chain_transaction_delta_proxy.to_numpy(float) > 0).astype(int)
    counts = frame.seq.value_counts()
    weights = frame.seq.map(
        {seq: len(frame) / (len(counts) * count) for seq, count in counts.items()}
    ).to_numpy(float)
    classifier = HistGradientBoostingClassifier(
        max_iter=260,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=60,
        l2_regularization=14.0,
        max_bins=255,
        random_state=seed,
        early_stopping=False,
    )
    classifier.fit(frame[features], target, sample_weight=weights)
    return PositiveProbabilityModel(classifier), {
        "per_sequence_positive_rate": {
            seq: float(
                (
                    frame.loc[frame.seq == seq, "chain_transaction_delta_proxy"]
                    > 0
                ).mean()
            )
            for seq in counts.index
        },
        "class_weighting": "none beyond equal total weight per fit sequence",
    }


def main() -> None:
    base = load_module(
        "m23_17_sequence_normalized_base",
        Path("scripts/m23_research/m23_17_sequence_normalized_utility.py"),
    )
    base.OUT = Path(
        "outputs/mot20_m23_20260718/m23_19_sequence_normalized_sign_v1"
    )
    base.NAME = "sequence_normalized_sign_ood_policy_v1"
    base.SCORE = "pred_sequence_normalized_positive_probability"
    base.TRAINING_GT_USE = (
        "binary positive transaction-utility labels on fit sequences only"
    )
    base.TARGET_TRANSFORM_DESCRIPTION = (
        "binary target: raw chain transaction utility greater than zero; "
        "no class reweighting beyond equal aggregate sequence weights"
    )
    base.STATUS_DESCRIPTION = (
        "nested sequence-normalized sign audit on reused development sequences; "
        "fixed-parent provenance remains exploratory"
    )
    base.fit_model = fit_sign_model
    base.main()


if __name__ == "__main__":
    main()
