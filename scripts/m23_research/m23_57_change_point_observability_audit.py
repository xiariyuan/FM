#!/usr/bin/env python3
from __future__ import annotations

"""Strict sequence-LOSO observability audit for frozen M23-57 boundaries.

This script is hard-gated by the completed M23-57 COMBINED teacher capacity.
It cannot train unless HOTA >= 80.700000. The held sequence is not opened
until the model, scaler, threshold and SHA manifest are frozen.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

REPO = Path(__file__).resolve().parents[2]
SEQUENCES = ("MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05")
ROOT = Path("outputs/mot20_m23_20260718/m23_57_intra_node_change_point_capacity_v2")
PREREG_SHA256 = "06ef9d70f2fe319de8e787e7aedcf74895c39736b31da1146c84cec24a5778d5"
SEED = 23570057
EPOCHS = 300
BATCH_PER_SEQUENCE = 4096
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.0001
GROUP_DRO_ETA = 0.1
M23_56 = {
    "source_error_pr_auc": 0.18298375390596652,
    "precision_at_actual_edit_count": 0.21907241179210601,
    "recall_at_95_precision": 0.012017174739538172,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_event(root: Path, payload: dict[str, Any]) -> None:
    from datetime import datetime, timezone
    with (root / "protocol_events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(), **payload}, sort_keys=True) + "\n")


def peak_rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)


def verify_gate(root: Path) -> dict[str, Any]:
    prereg = root / "preregistered_protocol.json"
    if sha256_file(prereg) != PREREG_SHA256:
        raise RuntimeError("M23-57 preregistration SHA changed")
    combined = root / "capacity_combined" / "report.json"
    if not combined.exists():
        raise FileNotFoundError("combined capacity must complete before observability")
    report = json.loads(combined.read_text(encoding="utf-8"))
    if not report["capacity_gate"]["pass"]:
        raise RuntimeError(
            f"M23-57 observability is locked: capacity HOTA={report['official_trackeval']['COMBINED']['HOTA']:.6f} < 80.700000"
        )
    return report


def feature_schema(seq: str, root: Path) -> tuple[list[str], list[int]]:
    manifest = json.loads((root / "boundary_universe" / seq / "freeze_manifest.json").read_text(encoding="utf-8"))
    columns = list(manifest["feature_columns"])
    raw = [name for name in columns if not name.endswith("_pct")]
    indices = [columns.index(name) for name in raw]
    return raw, indices


def load_sequence(seq: str, root: Path, open_labels: bool) -> dict[str, Any]:
    manifest = json.loads((root / "boundary_universe" / seq / "freeze_manifest.json").read_text(encoding="utf-8"))
    raw, indices = feature_schema(seq, root)
    matrix_path = Path(manifest["artifacts"]["feature_matrix"])
    if sha256_file(matrix_path) != manifest["artifacts"]["feature_matrix_sha256"]:
        raise RuntimeError(f"{seq} feature matrix SHA changed")
    matrix = np.load(matrix_path, mmap_mode="r")
    features = np.asarray(matrix[:, indices], np.float32)
    result: dict[str, Any] = {
        "seq": seq,
        "feature_names": raw,
        "X_all": features,
        "boundaries": pd.read_parquet(manifest["artifacts"]["boundary_features"]),
    }
    if open_labels:
        labels_path = root / "teacher_labels" / seq / "boundary_labels.parquet"
        labels = pd.read_parquet(labels_path)
        if not np.array_equal(labels.boundary_id.to_numpy(np.int64), np.arange(len(labels), dtype=np.int64)):
            raise RuntimeError(f"{seq} label rows are not aligned")
        supported = labels.audit_label.to_numpy(np.int8) >= 0
        result.update(
            {
                "labels_frame": labels,
                "supported": supported,
                "X": features[supported],
                "y": labels.audit_label.to_numpy(np.int8)[supported].astype(np.float32),
                "supported_indices": np.flatnonzero(supported),
            }
        )
    return result


def fit_scaler(training: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    count = 0
    total = None
    total_square = None
    for data in training:
        values = data["X"].astype(np.float64)
        current_sum = values.sum(axis=0)
        current_square = np.square(values).sum(axis=0)
        total = current_sum if total is None else total + current_sum
        total_square = current_square if total_square is None else total_square + current_square
        count += len(values)
    mean = total / max(count, 1)
    variance = total_square / max(count, 1) - np.square(mean)
    scale = np.sqrt(np.maximum(variance, 1e-8))
    return mean.astype(np.float32), scale.astype(np.float32)


def standardize(values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.clip((values - mean) / scale, -10.0, 10.0).astype(np.float32)


def balanced_batch(rng: np.random.Generator, X: np.ndarray, y: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    positive = np.flatnonzero(y > 0.5)
    negative = np.flatnonzero(y <= 0.5)
    if not len(positive) or not len(negative):
        raise RuntimeError("each training sequence requires positive and negative supported boundaries")
    half = size // 2
    indices = np.concatenate(
        [
            rng.choice(positive, size=half, replace=True),
            rng.choice(negative, size=size - half, replace=True),
        ]
    )
    rng.shuffle(indices)
    return X[indices], y[indices]


def predict(model: torch.nn.Module, X: np.ndarray, device: torch.device, batch: int = 65536) -> np.ndarray:
    output = np.empty(len(X), np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(X), batch):
            stop = min(len(X), start + batch)
            tensor = torch.from_numpy(X[start:stop]).to(device)
            output[start:stop] = torch.sigmoid(model(tensor).squeeze(1)).cpu().numpy()
    return output


def choose_training_threshold(y: np.ndarray, score: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y, score)
    if not len(thresholds):
        return 0.5
    f1 = 2.0 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best = float(np.nanmax(f1))
    candidates = np.flatnonzero(np.isclose(f1, best, rtol=0.0, atol=1e-12))
    return float(thresholds[int(candidates[0])])


def recall_at_precision(y: np.ndarray, score: np.ndarray, target: float) -> float:
    precision, recall, _ = precision_recall_curve(y, score)
    valid = recall[precision >= target]
    return float(valid.max()) if len(valid) else 0.0


def binary_metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, Any]:
    prediction = score >= threshold
    tp = int(np.sum(prediction & (y > 0.5)))
    fp = int(np.sum(prediction & (y <= 0.5)))
    fn = int(np.sum((~prediction) & (y > 0.5)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    actual = int(np.sum(y > 0.5))
    if actual:
        order = np.argsort(-score, kind="mergesort")[:actual]
        precision_actual = float(np.mean(y[order] > 0.5))
    else:
        precision_actual = 0.0
    return {
        "base_rate": float(np.mean(y > 0.5)),
        "pr_auc": float(average_precision_score(y, score)) if len(np.unique(y)) > 1 else None,
        "roc_auc": float(roc_auc_score(y, score)) if len(np.unique(y)) > 1 else None,
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "precision_at_actual_split_count": precision_actual,
        "recall_at_90_precision": recall_at_precision(y, score, 0.90),
        "recall_at_95_precision": recall_at_precision(y, score, 0.95),
        "recall_at_99_precision": recall_at_precision(y, score, 0.99),
        "predicted_splits": int(prediction.sum()),
        "actual_splits": actual,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
    }


def stratified_metrics(y: np.ndarray, score: np.ndarray, threshold: float, values: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    result = {}
    for name in dict.fromkeys(labels.tolist()):
        mask = labels == name
        if not np.any(mask):
            continue
        result[str(name)] = {"rows": int(mask.sum()), **binary_metrics(y[mask], score[mask], threshold)}
    return result


def boundary_offsets(boundaries: pd.DataFrame, labels: pd.DataFrame, supported_indices: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    frame = boundaries.iloc[supported_indices][["fixed_chunk_id", "position"]].copy()
    frame["label"] = labels.audit_label.to_numpy(np.int8)[supported_indices]
    frame["score"] = score
    offsets: list[int] = []
    for _, group in frame.groupby("fixed_chunk_id", sort=True):
        positives = group[group.label == 1]
        if positives.empty:
            continue
        selected = group.sort_values(["score", "position"], ascending=[False, True], kind="mergesort").head(len(positives))
        candidate_positions = selected.position.to_numpy(int)
        for true_position in positives.position.to_numpy(int):
            nearest = candidate_positions[int(np.argmin(np.abs(candidate_positions - true_position)))]
            offsets.append(int(nearest - true_position))
    array = np.asarray(offsets, np.int32)
    return {
        "events": len(offsets),
        "mean_signed_offset_rows": float(array.mean()) if len(array) else None,
        "median_absolute_offset_rows": float(np.median(np.abs(array))) if len(array) else None,
        "p90_absolute_offset_rows": float(np.quantile(np.abs(array), 0.90)) if len(array) else None,
        "exact_boundary_rate": float(np.mean(array == 0)) if len(array) else None,
    }


def train_outer(held: str, root: Path) -> dict[str, Any]:
    capacity = verify_gate(root)
    output_root = root / "observability_loso" / held
    report_path = output_root / "report.json"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite observability fold: {output_root}")
    started = time.time()
    training_sequences = [seq for seq in SEQUENCES if seq != held]
    training = [load_sequence(seq, root, open_labels=True) for seq in training_sequences]
    schemas = [data["feature_names"] for data in training]
    held_schema, _ = feature_schema(held, root)
    if any(schema != schemas[0] for schema in schemas[1:]) or held_schema != schemas[0]:
        raise RuntimeError("raw feature schemas differ across sequences")
    mean, scale = fit_scaler(training)
    for data in training:
        data["X_standard"] = standardize(data["X"], mean, scale)
    torch.manual_seed(SEED + SEQUENCES.index(held))
    np.random.seed(SEED + SEQUENCES.index(held))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED + SEQUENCES.index(held))
        torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.Linear(len(schemas[0]), 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    group_weights = torch.ones(len(training), device=device) / len(training)
    rng = np.random.default_rng(SEED + SEQUENCES.index(held))
    history = []
    model.train()
    for epoch in range(EPOCHS):
        losses = []
        for data in training:
            batch_x, batch_y = balanced_batch(rng, data["X_standard"], data["y"], BATCH_PER_SEQUENCE)
            x = torch.from_numpy(batch_x).to(device)
            y = torch.from_numpy(batch_y).to(device)
            logits = model(x).squeeze(1)
            losses.append(torch.nn.functional.binary_cross_entropy_with_logits(logits, y))
        loss_tensor = torch.stack(losses)
        with torch.no_grad():
            group_weights *= torch.exp(GROUP_DRO_ETA * loss_tensor.detach())
            group_weights /= group_weights.sum()
        objective = torch.sum(group_weights * loss_tensor)
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        optimizer.step()
        if epoch % 25 == 0 or epoch == EPOCHS - 1:
            history.append(
                {
                    "epoch": epoch,
                    "objective": float(objective.detach().cpu()),
                    "group_losses": {seq: float(value.detach().cpu()) for seq, value in zip(training_sequences, loss_tensor)},
                    "group_weights": {seq: float(value.detach().cpu()) for seq, value in zip(training_sequences, group_weights)},
                }
            )
    train_y = np.concatenate([data["y"] for data in training])
    train_score = np.concatenate([predict(model, data["X_standard"], device) for data in training])
    threshold = choose_training_threshold(train_y, train_score)
    output_root.mkdir(parents=True, exist_ok=True)
    model_path = output_root / "model.pt"
    torch.save(model.state_dict(), model_path)
    scaler_path = output_root / "scaler_and_threshold.npz"
    np.savez(scaler_path, mean=mean, scale=scale, threshold=np.asarray([threshold], np.float32), feature_names=np.asarray(schemas[0], object))
    history_path = output_root / "training_history.json"
    json_write(history_path, history)
    frozen = {
        "held": held,
        "training_sequences": training_sequences,
        "held_labels_opened": False,
        "model_sha256": sha256_file(model_path),
        "scaler_sha256": sha256_file(scaler_path),
        "training_history_sha256": sha256_file(history_path),
        "feature_names": schemas[0],
        "threshold": threshold,
        "normalizer_fit_sequences": training_sequences,
        "capacity_report_sha256": sha256_file(root / "capacity_combined" / "report.json"),
    }
    frozen_path = output_root / "model_frozen_before_held_labels.json"
    json_write(frozen_path, frozen)
    append_event(root, {"event": "observability_model_frozen_before_held_labels", "held": held, "model_sha256": frozen["model_sha256"], "scaler_sha256": frozen["scaler_sha256"]})
    held_data = load_sequence(held, root, open_labels=True)
    X_held = standardize(held_data["X"], mean, scale)
    held_score = predict(model, X_held, device)
    metrics = binary_metrics(held_data["y"], held_score, threshold)
    supported_labels = held_data["labels_frame"].iloc[held_data["supported_indices"]]
    prediction = held_score >= threshold
    pure = supported_labels.pure_node.to_numpy(np.int8) > 0
    metrics["false_split_rate_on_pure_nodes"] = float(np.mean(prediction[pure])) if np.any(pure) else None
    boundaries_supported = held_data["boundaries"].iloc[held_data["supported_indices"]]
    node_length = boundaries_supported.chunk_rows.to_numpy(int)
    node_stratum = np.where(node_length <= 10, "1-10", np.where(node_length <= 20, "11-20", "21-30"))
    crowd = np.maximum(boundaries_supported.left_crowd_density.to_numpy(float), boundaries_supported.right_crowd_density.to_numpy(float))
    crowd_stratum = np.where(crowd <= 2, "low_0_2", np.where(crowd <= 6, "medium_3_6", "high_7_plus"))
    relative = boundaries_supported.boundary_relative_position.to_numpy(float)
    position_stratum = np.where(relative <= 1 / 3, "early", np.where(relative >= 2 / 3, "late", "middle"))
    strata = {
        "node_length": stratified_metrics(held_data["y"], held_score, threshold, node_length, node_stratum),
        "crowd": stratified_metrics(held_data["y"], held_score, threshold, crowd, crowd_stratum),
        "boundary_position": stratified_metrics(held_data["y"], held_score, threshold, relative, position_stratum),
    }
    offsets = boundary_offsets(held_data["boundaries"], held_data["labels_frame"], held_data["supported_indices"], held_score)
    report = {
        "experiment": "M23-57 strict sequence-LOSO local boundary observability",
        "held": held,
        "training_sequences": training_sequences,
        "teacher_only": True,
        "deployable": False,
        "capacity_gate_HOTA": capacity["official_trackeval"]["COMBINED"]["HOTA"],
        "model": "single linear logistic classifier",
        "architecture_scan": False,
        "feature_selection": False,
        "held_used_for_training": False,
        "held_used_for_normalization": False,
        "held_used_for_threshold": False,
        "feature_count": len(schemas[0]),
        "features": schemas[0],
        "metrics": metrics,
        "detection_offset": offsets,
        "strata": strata,
        "training": {
            "epochs": EPOCHS,
            "batch_per_sequence": BATCH_PER_SEQUENCE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "group_dro_eta": GROUP_DRO_ETA,
            "training_supported_rows": {data["seq"]: len(data["y"]) for data in training},
            "training_positive_rows": {data["seq"]: int(data["y"].sum()) for data in training},
            "training_threshold": threshold,
            "model_sha256": frozen["model_sha256"],
            "scaler_sha256": frozen["scaler_sha256"],
            "history_sha256": frozen["training_history_sha256"],
        },
        "held_rows": len(held_data["y"]),
        "held_positive_rows": int(held_data["y"].sum()),
        "runtime_seconds": time.time() - started,
        "peak_rss_mb": peak_rss_mb(),
        "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated() / 1024**2) if torch.cuda.is_available() else 0.0,
    }
    json_write(report_path, report)
    append_event(root, {"event": "observability_fold_completed", "held": held, "pr_auc": metrics["pr_auc"], "precision_at_actual_split_count": metrics["precision_at_actual_split_count"]})
    return report


def summarize(root: Path) -> dict[str, Any]:
    capacity = verify_gate(root)
    reports = {}
    for seq in SEQUENCES:
        path = root / "observability_loso" / seq / "report.json"
        if not path.exists():
            raise FileNotFoundError(path)
        reports[seq] = json.loads(path.read_text(encoding="utf-8"))
    keys = ("pr_auc", "precision_at_actual_split_count", "recall_at_95_precision", "base_rate", "false_split_rate_on_pure_nodes")
    aggregate = {}
    for key in keys:
        values = [reports[seq]["metrics"][key] for seq in SEQUENCES if reports[seq]["metrics"][key] is not None]
        aggregate[key] = {"mean": float(np.mean(values)), "min": float(np.min(values)), "max": float(np.max(values))}
    easy = float(np.mean([reports[seq]["metrics"]["pr_auc"] for seq in ("MOT20-01", "MOT20-03")]))
    hard = float(np.mean([reports[seq]["metrics"]["pr_auc"] for seq in ("MOT20-02", "MOT20-05")]))
    degradation = easy - hard
    rule = {
        "mean_pr_auc_at_least_0p283": aggregate["pr_auc"]["mean"] >= 0.283,
        "mean_precision_at_actual_split_count_at_least_0p35": aggregate["precision_at_actual_split_count"]["mean"] >= 0.35,
        "mean_recall_at_95_precision_at_least_0p05": aggregate["recall_at_95_precision"]["mean"] >= 0.05,
        "m02_m05_pr_auc_degradation_at_most_0p15": degradation <= 0.15,
    }
    clearly_better = all(rule.values())
    decision = "D_allow_future_preregistered_M23_58_do_not_start" if clearly_better else "C_capacity_exists_but_local_representation_not_safely_transferable"
    summary = {
        "experiment": "M23-57 boundary observability summary",
        "teacher_only": True,
        "deployable": False,
        "capacity_HOTA": capacity["official_trackeval"]["COMBINED"]["HOTA"],
        "folds": {seq: reports[seq]["metrics"] for seq in SEQUENCES},
        "aggregate": aggregate,
        "transfer_degradation": {
            "M01_M03_mean_pr_auc": easy,
            "M02_M05_mean_pr_auc": hard,
            "degradation": degradation,
        },
        "comparison_to_M23_56": {
            "M23_56": M23_56,
            "M23_57": {
                "pr_auc": aggregate["pr_auc"]["mean"],
                "precision_at_actual_split_count": aggregate["precision_at_actual_split_count"]["mean"],
                "recall_at_95_precision": aggregate["recall_at_95_precision"]["mean"],
            },
            "absolute_improvement": {
                "pr_auc": aggregate["pr_auc"]["mean"] - M23_56["source_error_pr_auc"],
                "precision_at_actual_split_count": aggregate["precision_at_actual_split_count"]["mean"] - M23_56["precision_at_actual_edit_count"],
                "recall_at_95_precision": aggregate["recall_at_95_precision"]["mean"] - M23_56["recall_at_95_precision"],
            },
        },
        "fixed_D_rule": rule,
        "clearly_better_than_M23_56": clearly_better,
        "decision": decision,
        "m23_58_started": False,
        "mot20_test_submission": False,
        "reports_sha256": {seq: sha256_file(root / "observability_loso" / seq / "report.json") for seq in SEQUENCES},
    }
    path = root / "observability_loso" / "summary.json"
    json_write(path, summary)
    append_event(root, {"event": "observability_summary_completed", "decision": decision, "clearly_better_than_M23_56": clearly_better})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run-outer")
    run.add_argument("--held", required=True, choices=SEQUENCES)
    sub.add_parser("summarize")
    args = parser.parse_args()
    if args.command == "run-outer":
        result = train_outer(args.held, args.root)
    elif args.command == "summarize":
        result = summarize(args.root)
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
