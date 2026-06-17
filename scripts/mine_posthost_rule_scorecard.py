#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path("/gemini/code/FMtrack-main/FM-Track")
DEFAULT_DATASET = (
    REPO_ROOT
    / "outputs"
    / "official_bytetrack_posthost_one_edit_dataset_utilityaware_20260328_212500"
    / "posthost_action_examples.jsonl"
)
DEFAULT_REGISTRY = REPO_ROOT / "outputs" / "experiment_registry.csv"

POSTHOST_CANDIDATE_FEATURE_NAMES = [
    "is_keep",
    "is_add",
    "is_swap",
    "is_defer",
    "num_detections",
    "num_tracks",
    "num_edges",
    "is_large_component",
    "host_pair_count",
    "host_positive_count",
    "host_negative_count",
    "host_score",
    "add_refined",
    "add_base",
    "add_iou",
    "add_bbox_dist",
    "add_row_degree",
    "add_row_margin",
    "add_row_entropy",
    "add_col_degree",
    "add_track_gap",
    "add_track_hist",
    "add_delta_cx",
    "add_delta_cy",
    "add_delta_log_w",
    "add_delta_log_h",
    "remove_refined",
    "remove_base",
    "remove_iou",
    "remove_bbox_dist",
    "remove_row_degree",
    "remove_row_margin",
    "remove_row_entropy",
    "remove_col_degree",
    "remove_track_gap",
    "remove_track_hist",
    "remove_delta_cx",
    "remove_delta_cy",
    "remove_delta_log_w",
    "remove_delta_log_h",
    "delta_refined",
    "delta_base",
    "delta_iou",
    "delta_bbox_dist",
    "delta_row_margin",
    "delta_row_entropy",
    "delta_track_gap",
    "delta_track_hist",
    "has_remove",
]
FEATURE_INDEX = {name: idx for idx, name in enumerate(POSTHOST_CANDIDATE_FEATURE_NAMES)}

SUMMARY_FIELDS = [
    "experiment_name",
    "dataset_jsonl",
    "large_only",
    "respect_runtime_safe_prefilter",
    "feature_count",
    "train_candidates",
    "val_candidates",
    "train_positive_rate",
    "val_positive_rate",
    "best_penalty",
    "best_C",
    "best_threshold",
    "val_ap",
    "val_auc",
    "val_top1_positive_hit_rate",
    "val_top1_selected_precision",
    "val_top1_selected_clusters",
    "val_top1_positive_clusters",
    "val_utility_capture",
    "val_selected_positive_utility_sum",
    "val_selected_adjusted_utility_sum",
    "val_best_positive_utility_sum",
    "status",
    "error",
]

METRIC_FIELDS = [
    "penalty",
    "C",
    "threshold",
    "train_candidates",
    "val_candidates",
    "train_positive_rate",
    "val_positive_rate",
    "val_ap",
    "val_auc",
    "val_top1_positive_hit_rate",
    "val_top1_selected_precision",
    "val_top1_selected_clusters",
    "val_top1_positive_clusters",
    "val_utility_capture",
    "val_selected_positive_utility_sum",
    "val_selected_adjusted_utility_sum",
    "val_best_positive_utility_sum",
]


@dataclass
class CandidateRow:
    cluster_id: str
    seq: str
    split_tag: str
    candidate_index: int
    features: List[float]
    is_positive: int
    adjusted_utility: float
    utility: float
    is_large_component: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mine a linear defer-only scorecard from post-host one-edit oracle-style utility data."
    )
    parser.add_argument("--dataset-jsonl", default=str(DEFAULT_DATASET))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--experiment-name", default="official_bytetrack_posthost_rule_scorecard")
    parser.add_argument("--registry-csv", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--large-only", action="store_true", default=True)
    parser.add_argument("--allow-small", dest="large_only", action="store_false")
    parser.add_argument("--respect-runtime-safe-prefilter", dest="respect_runtime_safe_prefilter", action="store_true", default=True)
    parser.add_argument("--no-runtime-safe-prefilter", dest="respect_runtime_safe_prefilter", action="store_false")
    parser.add_argument(
        "--penalties",
        nargs="+",
        default=["l1", "l2"],
        choices=["l1", "l2"],
    )
    parser.add_argument("--c-grid", nargs="+", type=float, default=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0])
    parser.add_argument("--threshold-grid", nargs="+", type=float, default=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    parser.add_argument("--selected-precision-floor", type=float, default=0.75)
    parser.add_argument("--min-selected-clusters", type=int, default=20)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--seq-include", nargs="+", default=[])
    parser.add_argument("--seq-exclude", nargs="+", default=[])
    return parser.parse_args()


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def append_registry_row(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
    full_row = {key: "" for key in header}
    full_row.update(row)
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writerow(full_row)


def candidate_passes_runtime_safe_prefilter(features: Sequence[float]) -> bool:
    idx = FEATURE_INDEX
    if len(features) <= idx["has_remove"]:
        return False
    if int(round(float(features[idx["is_defer"]]))) <= 0:
        return False
    if int(round(float(features[idx["has_remove"]]))) <= 0:
        return False
    if float(features[idx["remove_refined"]]) > 0.35:
        return False
    if float(features[idx["remove_iou"]]) > 0.55:
        return False
    if float(features[idx["remove_row_margin"]]) > 0.15:
        return False
    if float(features[idx["remove_track_hist"]]) > 4.2:
        return False
    return True


def load_rows(args: argparse.Namespace) -> List[CandidateRow]:
    rows: List[CandidateRow] = []
    dataset_path = Path(args.dataset_jsonl)
    include_set = {str(x).strip() for x in args.seq_include if str(x).strip()}
    exclude_set = {str(x).strip() for x in args.seq_exclude if str(x).strip()}
    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            example = json.loads(line)
            seq = str(example.get("seq", ""))
            if include_set and seq not in include_set:
                continue
            if exclude_set and seq in exclude_set:
                continue
            is_large_component = int(example.get("is_large_component", 0) or 0)
            if args.large_only and is_large_component <= 0:
                continue
            candidate_types = example.get("candidate_action_types", [])
            candidate_features = example.get("candidate_features", [])
            candidate_positive = example.get("candidate_is_positive_utility", [])
            candidate_adjusted = example.get("candidate_adjusted_utility_deltas", [])
            candidate_utility = example.get("candidate_utility_deltas", [])
            for idx, action_type in enumerate(candidate_types):
                if str(action_type) != "defer":
                    continue
                feats = [float(x) for x in candidate_features[idx]]
                if args.respect_runtime_safe_prefilter and not candidate_passes_runtime_safe_prefilter(feats):
                    continue
                rows.append(
                    CandidateRow(
                        cluster_id=str(example.get("cluster_id", "")),
                        seq=seq,
                        split_tag=str(example.get("split_tag", "")),
                        candidate_index=int(idx),
                        features=feats,
                        is_positive=int(candidate_positive[idx]),
                        adjusted_utility=float(candidate_adjusted[idx]),
                        utility=float(candidate_utility[idx]),
                        is_large_component=is_large_component,
                    )
                )
    return rows


def rows_to_arrays(rows: Sequence[CandidateRow]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([row.features for row in rows], dtype=np.float32)
    y = np.asarray([row.is_positive for row in rows], dtype=np.int32)
    return x, y


def evaluate_threshold(rows: Sequence[CandidateRow], probs: np.ndarray, threshold: float) -> Dict[str, float]:
    grouped: Dict[str, List[tuple[CandidateRow, float]]] = defaultdict(list)
    for row, prob in zip(rows, probs.tolist()):
        grouped[row.cluster_id].append((row, float(prob)))

    selected_clusters = 0
    selected_positive = 0
    selected_adjusted_utility_sum = 0.0
    selected_positive_utility_sum = 0.0
    positive_clusters = 0
    positive_hit = 0
    best_positive_utility_sum = 0.0

    for cluster_rows in grouped.values():
        cluster_rows.sort(key=lambda item: item[1], reverse=True)
        best_row, best_prob = cluster_rows[0]
        cluster_best_positive = max(
            (max(row.adjusted_utility, 0.0) for row, _ in cluster_rows if row.is_positive > 0),
            default=0.0,
        )
        if cluster_best_positive > 0.0:
            positive_clusters += 1
            best_positive_utility_sum += float(cluster_best_positive)
        if best_prob < threshold:
            continue
        selected_clusters += 1
        selected_adjusted_utility_sum += float(best_row.adjusted_utility)
        if best_row.is_positive > 0:
            selected_positive += 1
            positive_hit += 1
            selected_positive_utility_sum += max(float(best_row.adjusted_utility), 0.0)
        elif cluster_best_positive > 0.0:
            positive_hit += 0

    selected_precision = (
        float(selected_positive) / float(selected_clusters) if selected_clusters > 0 else 0.0
    )
    top1_hit_rate = (
        float(positive_hit) / float(positive_clusters) if positive_clusters > 0 else 0.0
    )
    utility_capture = (
        float(selected_positive_utility_sum) / float(best_positive_utility_sum)
        if best_positive_utility_sum > 0.0
        else 0.0
    )
    return {
        "val_top1_selected_clusters": int(selected_clusters),
        "val_top1_selected_precision": float(selected_precision),
        "val_top1_positive_clusters": int(positive_clusters),
        "val_top1_positive_hit_rate": float(top1_hit_rate),
        "val_selected_positive_utility_sum": float(selected_positive_utility_sum),
        "val_selected_adjusted_utility_sum": float(selected_adjusted_utility_sum),
        "val_best_positive_utility_sum": float(best_positive_utility_sum),
        "val_utility_capture": float(utility_capture),
    }


def choose_best_metric(metric_rows: Sequence[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    viable = [
        row
        for row in metric_rows
        if float(row["val_top1_selected_precision"]) >= float(args.selected_precision_floor)
        and int(row["val_top1_selected_clusters"]) >= int(args.min_selected_clusters)
    ]
    if not viable:
        viable = list(metric_rows)
    return max(
        viable,
        key=lambda row: (
            float(row["val_utility_capture"]),
            float(row["val_top1_positive_hit_rate"]),
            float(row["val_top1_selected_precision"]),
            float(row["val_ap"]),
            -float(abs(float(row["threshold"]) - 0.8)),
        ),
    )


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "summary.csv"
    metrics_csv = out_dir / "metrics.csv"
    coefficients_json = out_dir / "coefficients.json"
    log_path = out_dir / "run.log"

    running_summary = {
        "experiment_name": args.experiment_name,
        "dataset_jsonl": str(Path(args.dataset_jsonl).resolve()),
        "large_only": bool(args.large_only),
        "respect_runtime_safe_prefilter": bool(args.respect_runtime_safe_prefilter),
        "status": "running",
        "error": "",
    }
    write_csv(summary_csv, SUMMARY_FIELDS, [running_summary])
    write_csv(metrics_csv, METRIC_FIELDS, [])
    log_path.write_text("status=running\n", encoding="utf-8")

    try:
        rows = load_rows(args)
        train_rows = [row for row in rows if row.split_tag == "train"]
        val_rows = [row for row in rows if row.split_tag == "val"]
        if not train_rows or not val_rows:
            raise RuntimeError("empty train or val candidate rows after filtering")

        x_train, y_train = rows_to_arrays(train_rows)
        x_val, y_val = rows_to_arrays(val_rows)

        metric_rows: List[Dict[str, Any]] = []
        fitted_models: Dict[tuple[str, float], Pipeline] = {}
        for penalty in args.penalties:
            for c_value in args.c_grid:
                clf = LogisticRegression(
                    penalty=penalty,
                    C=float(c_value),
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=int(args.max_iter),
                    random_state=42,
                )
                pipe = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("clf", clf),
                    ]
                )
                pipe.fit(x_train, y_train)
                fitted_models[(penalty, float(c_value))] = pipe
                val_probs = pipe.predict_proba(x_val)[:, 1]
                ap = float(average_precision_score(y_val, val_probs))
                auc = float(roc_auc_score(y_val, val_probs)) if len(np.unique(y_val)) > 1 else 0.0
                for threshold in args.threshold_grid:
                    metrics = evaluate_threshold(val_rows, val_probs, float(threshold))
                    metric_rows.append(
                        {
                            "penalty": penalty,
                            "C": float(c_value),
                            "threshold": float(threshold),
                            "train_candidates": int(len(train_rows)),
                            "val_candidates": int(len(val_rows)),
                            "train_positive_rate": float(y_train.mean()),
                            "val_positive_rate": float(y_val.mean()),
                            "val_ap": ap,
                            "val_auc": auc,
                            **metrics,
                        }
                    )

        best_metric = choose_best_metric(metric_rows, args)
        best_key = (str(best_metric["penalty"]), float(best_metric["C"]))
        best_model = fitted_models[best_key]
        scaler: StandardScaler = best_model.named_steps["scaler"]
        clf: LogisticRegression = best_model.named_steps["clf"]

        coef = clf.coef_[0].astype(float)
        intercept = float(clf.intercept_[0])
        scale = scaler.scale_.astype(float)
        mean = scaler.mean_.astype(float)
        raw_weights = coef / scale
        raw_intercept = intercept - float(np.dot(raw_weights, mean))
        coefficient_rows = [
            {
                "feature_name": name,
                "weight_standardized": float(weight_std),
                "weight_raw": float(weight_raw),
            }
            for name, weight_std, weight_raw in zip(
                POSTHOST_CANDIDATE_FEATURE_NAMES,
                coef.tolist(),
                raw_weights.tolist(),
            )
        ]
        coefficient_rows.sort(key=lambda row: abs(float(row["weight_raw"])), reverse=True)
        coefficients_payload = {
            "experiment_name": args.experiment_name,
            "dataset_jsonl": str(Path(args.dataset_jsonl).resolve()),
            "large_only": bool(args.large_only),
            "respect_runtime_safe_prefilter": bool(args.respect_runtime_safe_prefilter),
            "penalty": str(best_metric["penalty"]),
            "C": float(best_metric["C"]),
            "threshold": float(best_metric["threshold"]),
            "feature_names": list(POSTHOST_CANDIDATE_FEATURE_NAMES),
            "raw_intercept": float(raw_intercept),
            "standardized_intercept": float(intercept),
            "raw_weights": {name: float(weight) for name, weight in zip(POSTHOST_CANDIDATE_FEATURE_NAMES, raw_weights.tolist())},
            "standardized_weights": {name: float(weight) for name, weight in zip(POSTHOST_CANDIDATE_FEATURE_NAMES, coef.tolist())},
            "top_weights": coefficient_rows[:20],
            "validation_metrics": best_metric,
        }
        coefficients_json.write_text(json.dumps(coefficients_payload, indent=2), encoding="utf-8")
        write_csv(metrics_csv, METRIC_FIELDS, metric_rows)

        summary_row = {
            "experiment_name": args.experiment_name,
            "dataset_jsonl": str(Path(args.dataset_jsonl).resolve()),
            "large_only": bool(args.large_only),
            "respect_runtime_safe_prefilter": bool(args.respect_runtime_safe_prefilter),
            "feature_count": int(x_train.shape[1]),
            "train_candidates": int(len(train_rows)),
            "val_candidates": int(len(val_rows)),
            "train_positive_rate": float(y_train.mean()),
            "val_positive_rate": float(y_val.mean()),
            "best_penalty": str(best_metric["penalty"]),
            "best_C": float(best_metric["C"]),
            "best_threshold": float(best_metric["threshold"]),
            "val_ap": float(best_metric["val_ap"]),
            "val_auc": float(best_metric["val_auc"]),
            "val_top1_positive_hit_rate": float(best_metric["val_top1_positive_hit_rate"]),
            "val_top1_selected_precision": float(best_metric["val_top1_selected_precision"]),
            "val_top1_selected_clusters": int(best_metric["val_top1_selected_clusters"]),
            "val_top1_positive_clusters": int(best_metric["val_top1_positive_clusters"]),
            "val_utility_capture": float(best_metric["val_utility_capture"]),
            "val_selected_positive_utility_sum": float(best_metric["val_selected_positive_utility_sum"]),
            "val_selected_adjusted_utility_sum": float(best_metric["val_selected_adjusted_utility_sum"]),
            "val_best_positive_utility_sum": float(best_metric["val_best_positive_utility_sum"]),
            "status": "success",
            "error": "",
        }
        write_csv(summary_csv, SUMMARY_FIELDS, [summary_row])
        append_registry_row(
            Path(args.registry_csv),
            {
                "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                "kind": "analysis",
                "status": "success",
                "script": "scripts/mine_posthost_rule_scorecard.py",
                "dataset": "MOT17",
                "split": "trainhalf_valhalf",
                "tracker_family": "official_bytetrack",
                "variant": args.experiment_name,
                "tag": args.experiment_name,
                "run_root": str(out_dir),
                "summary_csv": str(summary_csv),
                "log_path": str(log_path),
                "notes": "oracle-guided posthost defer scorecard mining",
                "name": args.experiment_name,
            },
        )
        log_path.write_text("status=success\n", encoding="utf-8")
    except Exception as exc:  # pragma: no cover - operational path
        error_row = dict(running_summary)
        error_row["status"] = "failed"
        error_row["error"] = str(exc)
        write_csv(summary_csv, SUMMARY_FIELDS, [error_row])
        log_path.write_text(f"status=failed\nerror={exc}\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
