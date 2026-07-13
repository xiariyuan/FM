#!/usr/bin/env python3
"""Analyze GT-annotated TrustTrack observe-v2 chosen-pair logs.

The analysis is descriptive and cross-sequence only. It does not train or
modify an online tracker. Targets are kept separate:

- track_cross: current chosen detection GT differs from the track's prior output GT;
- gt_changed: the GT is output under a different tracker ID than at its previous
  valid observation;
- future10: the current GT changes tracker ID within the next 10 frames.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def parse_input(spec: str) -> Tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"--input must be SEQ=PATH, got {spec!r}")
    seq, path = spec.split("=", 1)
    seq = seq.strip()
    if not seq:
        raise ValueError(f"empty sequence name in {spec!r}")
    return seq, Path(path)


def as_float(row: dict, key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return math.nan


def as_int(row: dict, key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default)))
    except (TypeError, ValueError):
        return int(default)


def load_primary_rows(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [row for row in csv.DictReader(handle) if row.get("stage") == "primary"]


def margin_risk(value: float) -> float:
    # No alternative candidate means no measured ambiguity. Treat missing margin
    # as a large positive separation, hence low risk after negation.
    return -value if math.isfinite(value) else -1.0


def feature_row(row: dict) -> Dict[str, float]:
    final_cost = as_float(row, "final_cost")
    iou_cost = as_float(row, "raw_iou_cost")
    emb_cost = as_float(row, "embedding_cost")
    smooth_cos = as_float(row, "smooth_current_cosine")
    det_score = as_float(row, "det_score")
    row_margin = as_float(row, "row_signed_margin_all")
    col_margin = as_float(row, "col_signed_margin_all")
    pair_margin = as_float(row, "pair_signed_margin_all")
    row_margin_valid = as_float(row, "row_signed_margin_valid")
    col_margin_valid = as_float(row, "col_signed_margin_valid")
    pair_margin_valid = as_float(row, "pair_signed_margin_valid")
    return {
        "final_cost": final_cost,
        "raw_iou_cost": iou_cost,
        "embedding_cost": emb_cost,
        "negative_det_score": -det_score if math.isfinite(det_score) else math.nan,
        "negative_smooth_current_cosine": -smooth_cos if math.isfinite(smooth_cos) else math.nan,
        "negative_row_margin_all": margin_risk(row_margin),
        "negative_col_margin_all": margin_risk(col_margin),
        "negative_pair_margin_all": margin_risk(pair_margin),
        "negative_row_margin_valid": margin_risk(row_margin_valid),
        "negative_col_margin_valid": margin_risk(col_margin_valid),
        "negative_pair_margin_valid": margin_risk(pair_margin_valid),
        "chosen_rank": float(as_int(row, "chosen_rank", 1)),
        "lost_age": float(as_int(row, "lost_age", 0)),
        "track_age": float(as_int(row, "track_age", 0)),
        "row_valid_competitors": float(as_int(row, "row_valid_competitors", 0)),
        "col_valid_competitors": float(as_int(row, "col_valid_competitors", 0)),
        "appearance_wins_by": (
            iou_cost - emb_cost
            if math.isfinite(iou_cost) and math.isfinite(emb_cost)
            else math.nan
        ),
        "cue_cost_disagreement": (
            abs(iou_cost - emb_cost)
            if math.isfinite(iou_cost) and math.isfinite(emb_cost)
            else math.nan
        ),
    }


def target_value(row: dict, target: str) -> Tuple[bool, int]:
    if target == "track_cross":
        label = row.get("track_history_label")
        if label not in {"same_identity", "cross_identity"}:
            return False, 0
        return True, int(label == "cross_identity")
    if target == "gt_changed":
        if as_int(row, "chosen_track_is_current_output", 0) != 1:
            return False, 0
        label = row.get("gt_transition_label")
        if label not in {"stable", "changed"}:
            return False, 0
        return True, int(label == "changed")
    if target == "future10":
        if as_int(row, "chosen_track_is_current_output", 0) != 1:
            return False, 0
        label = row.get("gt_transition_label")
        if label not in {"stable", "changed"}:
            return False, 0
        return True, as_int(row, "gt_future_transition_10", 0)
    raise KeyError(target)


def binary_metrics(y: np.ndarray, score: np.ndarray) -> dict:
    finite = np.isfinite(score)
    y = y[finite]
    score = score[finite]
    result = {
        "n": int(len(y)),
        "positive": int(y.sum()),
        "rate": float(y.mean()) if len(y) else None,
        "missing": int((~finite).sum()),
    }
    if len(y) == 0 or len(np.unique(y)) < 2:
        result.update({"auc": None, "ap": None, "top10_precision": None, "top10_lift": None})
        return result
    top_n = max(1, int(math.ceil(0.10 * len(y))))
    top_indices = np.argpartition(score, -top_n)[-top_n:]
    precision = float(y[top_indices].mean())
    base_rate = float(y.mean())
    result.update(
        {
            "auc": float(roc_auc_score(y, score)),
            "ap": float(average_precision_score(y, score)),
            "top10_precision": precision,
            "top10_lift": precision / base_rate if base_rate else None,
        }
    )
    return result


def deciles(y: np.ndarray, score: np.ndarray) -> List[dict]:
    finite = np.isfinite(score)
    y = y[finite]
    score = score[finite]
    if len(y) == 0:
        return []
    order = np.argsort(score, kind="mergesort")
    bins = np.empty(len(score), dtype=np.int32)
    bins[order] = np.minimum(9, np.arange(len(score)) * 10 // len(score))
    output = []
    for index in range(10):
        mask = bins == index
        output.append(
            {
                "decile": index + 1,
                "n": int(mask.sum()),
                "score_mean": float(score[mask].mean()),
                "positive_rate": float(y[mask].mean()),
            }
        )
    return output


def prepare(rows: Sequence[dict], target: str) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    labels: List[int] = []
    feature_values: Dict[str, List[float]] = {}
    for row in rows:
        valid, label = target_value(row, target)
        if not valid:
            continue
        features = feature_row(row)
        labels.append(label)
        for key, value in features.items():
            feature_values.setdefault(key, []).append(value)
    return np.asarray(labels, dtype=np.int32), {
        key: np.asarray(values, dtype=np.float64) for key, values in feature_values.items()
    }


def matrix(features: Dict[str, np.ndarray], names: Sequence[str]) -> np.ndarray:
    return np.column_stack([features[name] for name in names]).astype(np.float64)


def cross_sequence_model(
    train_y: np.ndarray,
    train_features: Dict[str, np.ndarray],
    test_y: np.ndarray,
    test_features: Dict[str, np.ndarray],
    names: Sequence[str],
) -> dict:
    train_x = matrix(train_features, names)
    test_x = matrix(test_features, names)
    model = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        LogisticRegression(max_iter=500, class_weight="balanced", C=1.0),
    )
    model.fit(train_x, train_y)
    probability = model.predict_proba(test_x)[:, 1]
    result = binary_metrics(test_y, probability)
    result["features"] = list(names)
    return result


def analyze(inputs: Sequence[Tuple[str, Path]]) -> dict:
    rows_by_seq = {seq: load_primary_rows(path) for seq, path in inputs}
    targets = ("track_cross", "gt_changed", "future10")
    prepared = {
        seq: {target: prepare(rows, target) for target in targets}
        for seq, rows in rows_by_seq.items()
    }

    output = {"sequences": {}, "cross_sequence": {}}
    for seq, rows in rows_by_seq.items():
        seq_result = {"primary_rows": len(rows), "targets": {}}
        for target in targets:
            y, features = prepared[seq][target]
            metrics = {
                name: binary_metrics(y, values) for name, values in features.items()
            }
            seq_result["targets"][target] = {
                "n": int(len(y)),
                "positive": int(y.sum()),
                "rate": float(y.mean()) if len(y) else None,
                "features": metrics,
                "negative_pair_margin_all_deciles": deciles(
                    y, features["negative_pair_margin_all"]
                ),
                "final_cost_deciles": deciles(y, features["final_cost"]),
            }
        output["sequences"][seq] = seq_result

    model_groups = {
        "costs": [
            "final_cost",
            "raw_iou_cost",
            "embedding_cost",
            "negative_det_score",
            "negative_smooth_current_cosine",
        ],
        "chosen_margins": [
            "negative_row_margin_all",
            "negative_col_margin_all",
            "negative_pair_margin_all",
            "chosen_rank",
            "row_valid_competitors",
            "col_valid_competitors",
        ],
        "costs_plus_margins": [
            "final_cost",
            "raw_iou_cost",
            "embedding_cost",
            "negative_det_score",
            "negative_smooth_current_cosine",
            "negative_row_margin_all",
            "negative_col_margin_all",
            "negative_pair_margin_all",
            "chosen_rank",
            "row_valid_competitors",
            "col_valid_competitors",
            "cue_cost_disagreement",
            "appearance_wins_by",
        ],
    }
    if len(inputs) >= 2:
        for train_seq, _ in inputs:
            for test_seq, _ in inputs:
                if train_seq == test_seq:
                    continue
                pair_key = f"{train_seq}_to_{test_seq}"
                output["cross_sequence"].setdefault(pair_key, {})
                for target in targets:
                    train_y, train_features = prepared[train_seq][target]
                    test_y, test_features = prepared[test_seq][target]
                    output["cross_sequence"][pair_key][target] = {
                        group: cross_sequence_model(
                            train_y,
                            train_features,
                            test_y,
                            test_features,
                            names,
                        )
                        for group, names in model_groups.items()
                    }
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Sequence and annotated CSV as SEQ=PATH; pass at least twice for transfer analysis.",
    )
    parser.add_argument("--out-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = [parse_input(spec) for spec in args.input]
    result = analyze(inputs)
    output_path = Path(args.out_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
