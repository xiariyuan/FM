#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


REPO_ROOT = Path("/gemini/code/FMtrack-main/FM-Track")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.fgas.fgas.features.local_contention_acceptance_features import (
    LOCAL_CONTENTION_ACCEPTANCE_FEATURE_NAMES,
    build_local_contention_acceptance_feature_vector,
)
from projects.fgas.fgas.model.acceptance_gate import FGASAcceptanceGate


REGISTRY_CSV = REPO_ROOT / "outputs" / "experiment_registry.csv"

SUMMARY_FIELDS = [
    "status",
    "out_dir",
    "source_jsonl",
    "ranker_checkpoint",
    "selected_units",
    "train_rows",
    "val_rows",
    "train_positive",
    "train_negative",
    "val_positive",
    "val_negative",
    "epochs",
    "best_epoch",
    "best_val_loss",
    "best_val_accuracy",
    "best_val_balanced_accuracy",
    "best_val_precision",
    "best_val_recall",
    "best_val_f1",
    "best_val_average_precision",
    "fixed_threshold",
    "decision_threshold",
    "best_val_swept_precision",
    "best_val_swept_recall",
    "best_val_swept_f1",
    "best_val_swept_balanced_accuracy",
    "train_sequences",
    "val_sequences",
    "selection_metric",
    "sampler_mode",
    "class_balance_ratio",
    "positive_weight",
    "positive_weight_power",
    "positive_weight_cap",
    "hidden_dim",
    "dropout",
    "model_path",
    "metrics_csv",
    "feature_stats_json",
    "notes",
]

METRICS_FIELDS = [
    "epoch",
    "train_loss",
    "val_loss",
    "val_accuracy",
    "val_balanced_accuracy",
    "val_precision",
    "val_recall",
    "val_f1",
    "val_average_precision",
    "val_fixed_threshold",
    "val_best_f1_threshold",
    "val_best_f1_precision",
    "val_best_f1_recall",
    "val_best_f1",
    "val_best_f1_balanced_accuracy",
    "val_best_balanced_accuracy_threshold",
    "val_best_balanced_accuracy_precision",
    "val_best_balanced_accuracy_recall",
    "val_best_balanced_accuracy_f1",
    "val_best_balanced_accuracy_swept",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a pre-association local-contention acceptance gate on top-ranked challenger units."
    )
    parser.add_argument("--jsonl", nargs="+", required=True)
    parser.add_argument("--ranker-checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--score-batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--train-sequences", default="")
    parser.add_argument("--val-sequences", default="")
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--sampler-mode", choices=["shuffle", "balanced"], default="shuffle")
    parser.add_argument(
        "--selection-metric",
        choices=["fixed_balanced_accuracy", "swept_f1", "swept_balanced_accuracy", "average_precision"],
        default="average_precision",
    )
    parser.add_argument("--fixed-threshold", type=float, default=0.5)
    parser.add_argument("--threshold-grid-size", type=int, default=99)
    parser.add_argument("--positive-weight-power", type=float, default=1.0)
    parser.add_argument("--positive-weight-cap", type=float, default=0.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-tag", default="")
    parser.add_argument("--split-label", default="")
    parser.add_argument("--registry-csv", default=str(REGISTRY_CSV))
    return parser.parse_args()


def write_single_row_csv(path: Path, fieldnames: Iterable[str], row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_rows(path: Path, fieldnames: Iterable[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def append_registry(*, status: str, out_dir: Path, summary_csv: Path, notes: str, registry_csv: str, dataset_tag: str, split_label: str) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "append_experiment_record.py"),
        "--csv",
        str(registry_csv),
        "--kind",
        "train",
        "--status",
        status,
        "--script",
        "scripts/train_local_contention_acceptance_gate.py",
        "--dataset",
        dataset_tag,
        "--split",
        split_label,
        "--tracker-family",
        "local_contention_acceptance_gate",
        "--variant",
        out_dir.name,
        "--tag",
        "local_contention_acceptance_gate",
        "--run-root",
        str(out_dir),
        "--summary-csv",
        str(summary_csv),
        "--notes",
        notes,
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=False)


def parse_csv_tokens(raw: str) -> list[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    return [token.strip() for token in text.split(",") if token.strip()]


def infer_dataset_tag(jsonl_paths: list[str]) -> str:
    ordered_names = ["MOT17", "MOT20", "DanceTrack"]
    detected = {name: False for name in ordered_names}
    for path_str in jsonl_paths:
        path_text = str(Path(path_str).expanduser()).lower()
        if "mot17" in path_text:
            detected["MOT17"] = True
        if "mot20" in path_text:
            detected["MOT20"] = True
        if "dance" in path_text:
            detected["DanceTrack"] = True
    names = [name for name in ordered_names if detected[name]]
    return "+".join(names) if names else "unknown"


def infer_split_label(dataset_tag: str) -> str:
    parts = [token.strip() for token in str(dataset_tag).split("+") if token.strip()]
    if not parts:
        return "unknown"
    split_labels = {"val" if part == "DanceTrack" else "val_half" for part in parts}
    if len(split_labels) == 1:
        return next(iter(split_labels))
    return "mixed"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LocalContentionRanker(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


@dataclass
class RankerBundle:
    feature_names: list[str]
    mean: np.ndarray
    std: np.ndarray
    model: nn.Module
    device: torch.device
    checkpoint_path: str


def load_ranker_bundle(checkpoint_path: str, device: torch.device) -> RankerBundle:
    path = Path(checkpoint_path).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    payload = torch.load(str(path), map_location="cpu")
    feature_names = [str(name) for name in list(payload.get("feature_names", []) or [])]
    if not feature_names:
        raise RuntimeError(f"Ranker checkpoint is missing feature_names: {path}")
    model_state = payload.get("model_state_dict")
    if model_state is None:
        model_state = payload.get("model_state")
    if model_state is None:
        raise RuntimeError(f"Ranker checkpoint is missing model weights: {path}")
    hidden_dim = int(payload.get("hidden_dim", 64))
    dropout = float(payload.get("dropout", 0.1))
    mean = np.asarray(payload.get("mean", []), dtype=np.float32).reshape(-1)
    std = np.asarray(payload.get("std", []), dtype=np.float32).reshape(-1)
    if mean.size != len(feature_names):
        mean = np.zeros((len(feature_names),), dtype=np.float32)
    if std.size != len(feature_names):
        std = np.ones((len(feature_names),), dtype=np.float32)
    std[std < 1e-6] = 1.0
    model = LocalContentionRanker(len(feature_names), hidden_dim, dropout)
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    return RankerBundle(
        feature_names=feature_names,
        mean=mean,
        std=std,
        model=model,
        device=device,
        checkpoint_path=str(path),
    )


def _candidate_sort_key(row: dict[str, object]) -> tuple[float, ...]:
    det_rank = int(row.get("challenger_det_rank", row.get("det_rank", 0)) or 0)
    return (
        float(row.get("_ranker_score", -1.0)),
        float(row.get("owner_best_alt_det_score", -1.0) or -1.0),
        float(row.get("owner_best_alt_det_box_iou", 0.0) or 0.0),
        float(row.get("challenger_edge_advantage_vs_owner", row.get("edge_advantage_vs_owner", 0.0)) or 0.0),
        -det_rank if det_rank > 0 else 0.0,
        float(row.get("challenger_edge_score", row.get("edge_score", 0.0)) or 0.0),
        float(row.get("challenger_best_box_iou", row.get("best_box_iou", 0.0)) or 0.0),
        float(row.get("challenger_age_gap_vs_owner", row.get("age_gap", 0.0)) or 0.0),
        float(row.get("challenger_hits", row.get("track_hits", 0.0)) or 0.0),
        -float(row.get("owner_age", 0.0) or 0.0),
    )


def _score_rows(rows: list[dict[str, object]], ranker: RankerBundle) -> list[float]:
    if not rows:
        return []
    matrix = np.asarray(
        [
            [float(row.get(name, 0.0) or 0.0) for name in ranker.feature_names]
            for row in rows
        ],
        dtype=np.float32,
    )
    matrix = (matrix - ranker.mean) / ranker.std
    features = torch.from_numpy(matrix).to(ranker.device)
    with torch.no_grad():
        scores = torch.sigmoid(ranker.model(features)).detach().cpu().numpy()
    return [float(value) for value in scores.reshape(-1)]


@dataclass
class SelectedRows:
    features: np.ndarray
    labels: np.ndarray
    seq_names: list[str]
    selected_units: int


def load_selected_rows(jsonl_paths: list[str], ranker: RankerBundle, score_batch_size: int) -> SelectedRows:
    feature_rows: list[list[float]] = []
    labels: list[int] = []
    seq_names: list[str] = []
    selected_units = 0

    current_unit_id: str | None = None
    current_unit_rows: list[dict[str, object]] = []
    score_buffer: list[dict[str, object]] = []

    def flush_unit(unit_rows: list[dict[str, object]]) -> None:
        nonlocal selected_units
        if not unit_rows:
            return
        ordered = sorted(unit_rows, key=_candidate_sort_key, reverse=True)
        top_row = dict(ordered[0])
        top_row["local_contention_ranker_score"] = float(top_row.get("_ranker_score", -1.0))
        top_row["local_contention_ranker_group_size"] = int(len(ordered))
        if len(ordered) > 1:
            second_score = float(ordered[1].get("_ranker_score", -1.0))
            top_row["local_contention_ranker_margin_to_second"] = float(
                float(top_row["local_contention_ranker_score"]) - second_score
            )
        else:
            top_row["local_contention_ranker_margin_to_second"] = -1.0
        label = int(top_row.get("label_prefer_challenger", -1))
        if label not in {0, 1}:
            return
        feature_rows.append(build_local_contention_acceptance_feature_vector(top_row))
        labels.append(label)
        seq_names.append(str(top_row.get("seq_name", "")).strip())
        selected_units += 1

    def flush_buffer(buffer_rows: list[dict[str, object]]) -> None:
        nonlocal current_unit_id, current_unit_rows
        if not buffer_rows:
            return
        scores = _score_rows(buffer_rows, ranker)
        for row, score in zip(buffer_rows, scores):
            row["_ranker_score"] = float(score)
            unit_id = str(row.get("unit_id", "")).strip()
            if current_unit_id is None:
                current_unit_id = unit_id
            if unit_id != current_unit_id:
                flush_unit(current_unit_rows)
                current_unit_rows = []
                current_unit_id = unit_id
            current_unit_rows.append(row)

    max_batch = max(1, int(score_batch_size))
    for path_str in jsonl_paths:
        path = Path(path_str).expanduser()
        if not path.is_absolute():
            path = (REPO_ROOT / path).resolve()
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                label = int(row.get("label_prefer_challenger", -1))
                if label not in {0, 1}:
                    continue
                score_buffer.append(row)
                if len(score_buffer) >= max_batch:
                    flush_buffer(score_buffer)
                    score_buffer = []
    if score_buffer:
        flush_buffer(score_buffer)
    if current_unit_rows:
        flush_unit(current_unit_rows)

    if not feature_rows:
        raise RuntimeError("No selected acceptance rows were built from the provided local contention JSONL files.")

    return SelectedRows(
        features=np.asarray(feature_rows, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        seq_names=seq_names,
        selected_units=int(selected_units),
    )


def default_sequence_split(seq_names: list[str], val_fraction: float) -> tuple[list[str], list[str]]:
    unique_seq_names = sorted({str(seq_name).strip() for seq_name in seq_names if str(seq_name).strip()})
    if len(unique_seq_names) <= 1:
        return unique_seq_names, unique_seq_names
    val_count = max(1, int(math.ceil(len(unique_seq_names) * max(0.05, float(val_fraction)))))
    val_sequences = unique_seq_names[-val_count:]
    train_sequences = unique_seq_names[:-val_count]
    if not train_sequences:
        train_sequences = unique_seq_names[:-1]
        val_sequences = unique_seq_names[-1:]
    return train_sequences, val_sequences


@dataclass
class SplitArrays:
    train_features: np.ndarray
    train_labels: np.ndarray
    val_features: np.ndarray
    val_labels: np.ndarray
    train_sequences: list[str]
    val_sequences: list[str]


def split_selected_rows(rows: SelectedRows, train_sequences: list[str], val_sequences: list[str], val_fraction: float) -> SplitArrays:
    if not train_sequences and not val_sequences:
        train_sequences, val_sequences = default_sequence_split(rows.seq_names, val_fraction)
    train_seq_set = set(train_sequences)
    val_seq_set = set(val_sequences)
    seq_array = np.asarray(rows.seq_names)
    train_mask = np.asarray([seq in train_seq_set for seq in seq_array], dtype=bool)
    val_mask = np.asarray([seq in val_seq_set for seq in seq_array], dtype=bool)
    if not np.any(train_mask) or not np.any(val_mask):
        train_sequences, val_sequences = default_sequence_split(rows.seq_names, val_fraction)
        train_seq_set = set(train_sequences)
        val_seq_set = set(val_sequences)
        train_mask = np.asarray([seq in train_seq_set for seq in seq_array], dtype=bool)
        val_mask = np.asarray([seq in val_seq_set for seq in seq_array], dtype=bool)
    if not np.any(train_mask) or not np.any(val_mask):
        raise RuntimeError("Unable to build a non-empty train/val split from selected acceptance rows.")
    return SplitArrays(
        train_features=rows.features[train_mask],
        train_labels=rows.labels[train_mask],
        val_features=rows.features[val_mask],
        val_labels=rows.labels[val_mask],
        train_sequences=sorted(train_seq_set),
        val_sequences=sorted(val_seq_set),
    )


class ArrayDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, mean: np.ndarray, std: np.ndarray) -> None:
        self.features = features.astype(np.float32)
        self.labels = labels.astype(np.float32)
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = (self.features[index] - self.mean) / self.std
        return torch.from_numpy(normalized), torch.tensor(float(self.labels[index]), dtype=torch.float32)


def build_feature_stats(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def count_labels(labels: np.ndarray) -> tuple[int, int]:
    positive = int(np.sum(labels == 1))
    negative = int(np.sum(labels == 0))
    return positive, negative


def compute_binary_metrics(labels_int: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, float]:
    preds = (probs >= float(threshold)).astype(np.int32)
    tp = int(np.sum((preds == 1) & (labels_int == 1)))
    tn = int(np.sum((preds == 0) & (labels_int == 0)))
    fp = int(np.sum((preds == 1) & (labels_int == 0)))
    fn = int(np.sum((preds == 0) & (labels_int == 1)))
    pos_total = max(1, int(np.sum(labels_int == 1)))
    neg_total = max(1, int(np.sum(labels_int == 0)))
    recall = float(tp) / float(max(1, tp + fn))
    precision = float(tp) / float(max(1, tp + fp))
    accuracy = float(tp + tn) / float(max(1, tp + tn + fp + fn))
    tpr = float(tp) / float(pos_total)
    tnr = float(tn) / float(neg_total)
    balanced_accuracy = 0.5 * (tpr + tnr)
    f1 = (2.0 * precision * recall / max(1e-8, precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "threshold": float(threshold),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def compute_average_precision(labels_int: np.ndarray, probs: np.ndarray) -> float:
    positive_total = int(np.sum(labels_int == 1))
    if positive_total <= 0 or labels_int.size == 0:
        return 0.0
    order = np.argsort(-probs, kind="mergesort")
    sorted_labels = labels_int[order]
    tp_cumsum = np.cumsum(sorted_labels == 1)
    fp_cumsum = np.cumsum(sorted_labels == 0)
    precision = tp_cumsum / np.maximum(1, tp_cumsum + fp_cumsum)
    recall = tp_cumsum / float(positive_total)
    average_precision = 0.0
    prev_recall = 0.0
    for idx, label in enumerate(sorted_labels):
        if int(label) != 1:
            continue
        current_recall = float(recall[idx])
        average_precision += float(precision[idx]) * max(0.0, current_recall - prev_recall)
        prev_recall = current_recall
    return float(average_precision)


def sweep_threshold_metrics(labels_int: np.ndarray, probs: np.ndarray, grid_size: int) -> tuple[dict[str, float], dict[str, float]]:
    threshold_count = max(2, int(grid_size))
    thresholds = np.linspace(0.01, 0.99, threshold_count, dtype=np.float32)
    best_f1_metrics: dict[str, float] | None = None
    best_balanced_metrics: dict[str, float] | None = None
    for threshold in thresholds:
        metrics = compute_binary_metrics(labels_int, probs, float(threshold))
        f1_score = (metrics["f1"], metrics["balanced_accuracy"], metrics["precision"], metrics["recall"])
        best_f1_score = (
            best_f1_metrics["f1"],
            best_f1_metrics["balanced_accuracy"],
            best_f1_metrics["precision"],
            best_f1_metrics["recall"],
        ) if best_f1_metrics is not None else None
        if best_f1_score is None or f1_score > best_f1_score:
            best_f1_metrics = dict(metrics)

        balanced_score = (metrics["balanced_accuracy"], metrics["f1"], metrics["precision"], metrics["recall"])
        best_balanced_score = (
            best_balanced_metrics["balanced_accuracy"],
            best_balanced_metrics["f1"],
            best_balanced_metrics["precision"],
            best_balanced_metrics["recall"],
        ) if best_balanced_metrics is not None else None
        if best_balanced_score is None or balanced_score > best_balanced_score:
            best_balanced_metrics = dict(metrics)

    if best_f1_metrics is None or best_balanced_metrics is None:
        empty = compute_binary_metrics(labels_int, probs, 0.5)
        return dict(empty), dict(empty)
    return best_f1_metrics, best_balanced_metrics


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    *,
    fixed_threshold: float,
    threshold_grid_size: int,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_rows = 0
    logits_all = []
    labels_all = []
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            loss = loss_fn(logits, labels)
            total_loss += float(loss.item()) * int(labels.shape[0])
            total_rows += int(labels.shape[0])
            logits_all.append(logits.cpu().numpy())
            labels_all.append(labels.cpu().numpy())
    logits = np.concatenate(logits_all, axis=0) if logits_all else np.zeros((0,), dtype=np.float32)
    labels = np.concatenate(labels_all, axis=0) if labels_all else np.zeros((0,), dtype=np.float32)
    probs = 1.0 / (1.0 + np.exp(-logits))
    labels_int = labels.astype(np.int32)
    fixed_metrics = compute_binary_metrics(labels_int, probs, fixed_threshold)
    best_f1_metrics, best_balanced_metrics = sweep_threshold_metrics(labels_int, probs, threshold_grid_size)
    average_precision = compute_average_precision(labels_int, probs)
    return {
        "loss": float(total_loss / float(max(1, total_rows))),
        "accuracy": float(fixed_metrics["accuracy"]),
        "balanced_accuracy": float(fixed_metrics["balanced_accuracy"]),
        "precision": float(fixed_metrics["precision"]),
        "recall": float(fixed_metrics["recall"]),
        "f1": float(fixed_metrics["f1"]),
        "average_precision": float(average_precision),
        "fixed_threshold": float(fixed_threshold),
        "swept_best_f1_threshold": float(best_f1_metrics["threshold"]),
        "swept_best_f1_precision": float(best_f1_metrics["precision"]),
        "swept_best_f1_recall": float(best_f1_metrics["recall"]),
        "swept_best_f1": float(best_f1_metrics["f1"]),
        "swept_best_f1_balanced_accuracy": float(best_f1_metrics["balanced_accuracy"]),
        "swept_best_balanced_accuracy_threshold": float(best_balanced_metrics["threshold"]),
        "swept_best_balanced_accuracy_precision": float(best_balanced_metrics["precision"]),
        "swept_best_balanced_accuracy_recall": float(best_balanced_metrics["recall"]),
        "swept_best_balanced_accuracy_f1": float(best_balanced_metrics["f1"]),
        "swept_best_balanced_accuracy": float(best_balanced_metrics["balanced_accuracy"]),
    }


def compute_selection_score(val_metrics: dict[str, float], selection_metric: str) -> tuple[float, float, float]:
    def metric_value(*keys: str) -> float:
        for key in keys:
            if key in val_metrics and str(val_metrics[key]) != "":
                return float(val_metrics[key])
        raise KeyError(keys[0])

    if selection_metric == "fixed_balanced_accuracy":
        primary = metric_value("balanced_accuracy", "best_val_balanced_accuracy")
    elif selection_metric == "swept_f1":
        primary = metric_value("swept_best_f1", "best_val_swept_f1")
    elif selection_metric == "swept_balanced_accuracy":
        primary = metric_value("swept_best_balanced_accuracy", "best_val_swept_balanced_accuracy")
    elif selection_metric == "average_precision":
        primary = metric_value("average_precision", "best_val_average_precision")
    else:
        raise ValueError(f"Unsupported selection metric: {selection_metric}")
    return (
        primary,
        metric_value("average_precision", "best_val_average_precision"),
        metric_value("swept_best_f1", "best_val_swept_f1"),
    )


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = (REPO_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = out_dir / "summary.csv"
    metrics_csv = out_dir / "metrics.csv"
    feature_stats_json = out_dir / "feature_stats.json"
    model_path = out_dir / "best.pt"
    dataset_tag = str(args.dataset_tag).strip() or infer_dataset_tag(list(args.jsonl))
    split_label = str(args.split_label).strip() or infer_split_label(dataset_tag)

    running_summary = {
        "status": "running",
        "out_dir": str(out_dir),
        "source_jsonl": "|".join(str(Path(path).expanduser()) for path in args.jsonl),
        "ranker_checkpoint": str(Path(args.ranker_checkpoint).expanduser()),
        "selected_units": 0,
        "train_rows": 0,
        "val_rows": 0,
        "train_positive": 0,
        "train_negative": 0,
        "val_positive": 0,
        "val_negative": 0,
        "epochs": int(args.epochs),
        "best_epoch": "",
        "best_val_loss": "",
        "best_val_accuracy": "",
        "best_val_balanced_accuracy": "",
        "best_val_precision": "",
        "best_val_recall": "",
        "best_val_f1": "",
        "best_val_average_precision": "",
        "fixed_threshold": float(args.fixed_threshold),
        "decision_threshold": "",
        "best_val_swept_precision": "",
        "best_val_swept_recall": "",
        "best_val_swept_f1": "",
        "best_val_swept_balanced_accuracy": "",
        "train_sequences": "",
        "val_sequences": "",
        "selection_metric": str(args.selection_metric),
        "sampler_mode": str(args.sampler_mode),
        "class_balance_ratio": "",
        "positive_weight": "",
        "positive_weight_power": float(args.positive_weight_power),
        "positive_weight_cap": float(args.positive_weight_cap),
        "hidden_dim": int(args.hidden_dim),
        "dropout": float(args.dropout),
        "model_path": str(model_path),
        "metrics_csv": str(metrics_csv),
        "feature_stats_json": str(feature_stats_json),
        "notes": "training local contention acceptance gate",
    }
    write_single_row_csv(summary_csv, SUMMARY_FIELDS, running_summary)
    append_registry(
        status="running",
        out_dir=out_dir,
        summary_csv=summary_csv,
        notes="started local contention acceptance gate training",
        registry_csv=args.registry_csv,
        dataset_tag=dataset_tag,
        split_label=split_label,
    )

    try:
        set_seed(args.seed)
        device = torch.device(args.device)
        ranker = load_ranker_bundle(args.ranker_checkpoint, device)
        selected = load_selected_rows(list(args.jsonl), ranker, int(args.score_batch_size))
        split = split_selected_rows(
            selected,
            parse_csv_tokens(args.train_sequences),
            parse_csv_tokens(args.val_sequences),
            args.val_fraction,
        )
        train_positive, train_negative = count_labels(split.train_labels)
        val_positive, val_negative = count_labels(split.val_labels)
        if train_positive == 0 or train_negative == 0:
            raise RuntimeError(
                f"Train split must contain both positive and negative labels, got positive={train_positive} negative={train_negative}"
            )
        if val_positive == 0 or val_negative == 0:
            raise RuntimeError(
                f"Val split must contain both positive and negative labels, got positive={val_positive} negative={val_negative}"
            )

        mean, std = build_feature_stats(split.train_features)
        feature_stats_json.write_text(
            json.dumps(
                {
                    "feature_names": LOCAL_CONTENTION_ACCEPTANCE_FEATURE_NAMES,
                    "mean": mean.tolist(),
                    "std": std.tolist(),
                    "ranker_checkpoint": ranker.checkpoint_path,
                    "train_sequences": split.train_sequences,
                    "val_sequences": split.val_sequences,
                    "selected_units": int(selected.selected_units),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        train_dataset = ArrayDataset(split.train_features, split.train_labels, mean, std)
        if str(args.sampler_mode) == "balanced":
            sample_weights = np.asarray(
                [
                    (1.0 / float(max(1, train_positive)))
                    if int(label) == 1
                    else (1.0 / float(max(1, train_negative)))
                    for label in split.train_labels.tolist()
                ],
                dtype=np.float64,
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=int(args.batch_size),
                shuffle=False,
                sampler=WeightedRandomSampler(
                    weights=torch.as_tensor(sample_weights, dtype=torch.double),
                    num_samples=int(split.train_labels.shape[0]),
                    replacement=True,
                ),
                drop_last=False,
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=int(args.batch_size),
                shuffle=True,
                drop_last=False,
            )
        val_loader = DataLoader(
            ArrayDataset(split.val_features, split.val_labels, mean, std),
            batch_size=int(args.batch_size),
            shuffle=False,
            drop_last=False,
        )

        model = FGASAcceptanceGate(
            input_dim=len(LOCAL_CONTENTION_ACCEPTANCE_FEATURE_NAMES),
            hidden_dim=int(args.hidden_dim),
            dropout=float(args.dropout),
        ).to(device)
        class_balance_ratio = float(max(1, train_negative) / float(max(1, train_positive)))
        if float(args.positive_weight_power) <= 0.0:
            pos_weight = 1.0
        else:
            pos_weight = float(class_balance_ratio ** float(args.positive_weight_power))
        if float(args.positive_weight_cap) > 0.0:
            pos_weight = float(min(pos_weight, float(args.positive_weight_cap)))

        running_summary.update(
            {
                "ranker_checkpoint": str(ranker.checkpoint_path),
                "selected_units": int(selected.selected_units),
                "train_rows": int(split.train_labels.shape[0]),
                "val_rows": int(split.val_labels.shape[0]),
                "train_positive": int(train_positive),
                "train_negative": int(train_negative),
                "val_positive": int(val_positive),
                "val_negative": int(val_negative),
                "train_sequences": ",".join(split.train_sequences),
                "val_sequences": ",".join(split.val_sequences),
                "class_balance_ratio": float(class_balance_ratio),
                "positive_weight": float(pos_weight),
            }
        )
        write_single_row_csv(summary_csv, SUMMARY_FIELDS, running_summary)
        append_registry(
            status="running",
            out_dir=out_dir,
            summary_csv=summary_csv,
            notes=(
                "running local contention acceptance gate training "
                f"selected_units={selected.selected_units} "
                f"train_rows={split.train_labels.shape[0]} "
                f"val_rows={split.val_labels.shape[0]} "
                f"positive_weight={pos_weight:.4f}"
            ),
            registry_csv=args.registry_csv,
            dataset_tag=dataset_tag,
            split_label=split_label,
        )

        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

        best_state = None
        best_metrics = None
        history: List[dict[str, object]] = []

        for epoch in range(1, int(args.epochs) + 1):
            model.train()
            train_loss_total = 0.0
            train_rows_seen = 0
            for features, labels in train_loader:
                features = features.to(device)
                labels = labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(features)
                loss = loss_fn(logits, labels)
                loss.backward()
                optimizer.step()
                train_loss_total += float(loss.item()) * int(labels.shape[0])
                train_rows_seen += int(labels.shape[0])

            val_metrics = evaluate(
                model,
                val_loader,
                device,
                loss_fn,
                fixed_threshold=float(args.fixed_threshold),
                threshold_grid_size=int(args.threshold_grid_size),
            )
            row = {
                "epoch": int(epoch),
                "train_loss": float(train_loss_total / float(max(1, train_rows_seen))),
                "val_loss": float(val_metrics["loss"]),
                "val_accuracy": float(val_metrics["accuracy"]),
                "val_balanced_accuracy": float(val_metrics["balanced_accuracy"]),
                "val_precision": float(val_metrics["precision"]),
                "val_recall": float(val_metrics["recall"]),
                "val_f1": float(val_metrics["f1"]),
                "val_average_precision": float(val_metrics["average_precision"]),
                "val_fixed_threshold": float(val_metrics["fixed_threshold"]),
                "val_best_f1_threshold": float(val_metrics["swept_best_f1_threshold"]),
                "val_best_f1_precision": float(val_metrics["swept_best_f1_precision"]),
                "val_best_f1_recall": float(val_metrics["swept_best_f1_recall"]),
                "val_best_f1": float(val_metrics["swept_best_f1"]),
                "val_best_f1_balanced_accuracy": float(val_metrics["swept_best_f1_balanced_accuracy"]),
                "val_best_balanced_accuracy_threshold": float(val_metrics["swept_best_balanced_accuracy_threshold"]),
                "val_best_balanced_accuracy_precision": float(val_metrics["swept_best_balanced_accuracy_precision"]),
                "val_best_balanced_accuracy_recall": float(val_metrics["swept_best_balanced_accuracy_recall"]),
                "val_best_balanced_accuracy_f1": float(val_metrics["swept_best_balanced_accuracy_f1"]),
                "val_best_balanced_accuracy_swept": float(val_metrics["swept_best_balanced_accuracy"]),
            }
            history.append(row)

            current_score = compute_selection_score(val_metrics, str(args.selection_metric))
            best_score = compute_selection_score(best_metrics, str(args.selection_metric)) if best_metrics is not None else None
            if best_score is None or current_score > best_score:
                best_metrics = {
                    "best_epoch": int(epoch),
                    "best_val_loss": float(val_metrics["loss"]),
                    "best_val_accuracy": float(val_metrics["accuracy"]),
                    "best_val_balanced_accuracy": float(val_metrics["balanced_accuracy"]),
                    "best_val_precision": float(val_metrics["precision"]),
                    "best_val_recall": float(val_metrics["recall"]),
                    "best_val_f1": float(val_metrics["f1"]),
                    "best_val_average_precision": float(val_metrics["average_precision"]),
                    "best_val_swept_threshold": float(val_metrics["swept_best_f1_threshold"]),
                    "best_val_swept_precision": float(val_metrics["swept_best_f1_precision"]),
                    "best_val_swept_recall": float(val_metrics["swept_best_f1_recall"]),
                    "best_val_swept_f1": float(val_metrics["swept_best_f1"]),
                    "best_val_swept_balanced_accuracy": float(val_metrics["swept_best_f1_balanced_accuracy"]),
                }
                best_state = {
                    "model_state": model.state_dict(),
                    "feature_names": LOCAL_CONTENTION_ACCEPTANCE_FEATURE_NAMES,
                    "mean": mean.tolist(),
                    "std": std.tolist(),
                    "hidden_dim": int(args.hidden_dim),
                    "dropout": float(args.dropout),
                    "ranker_checkpoint": ranker.checkpoint_path,
                    "train_sequences": split.train_sequences,
                    "val_sequences": split.val_sequences,
                    "selection_metric": str(args.selection_metric),
                    "fixed_threshold": float(args.fixed_threshold),
                    "decision_threshold": float(val_metrics["swept_best_f1_threshold"]),
                    "average_precision": float(val_metrics["average_precision"]),
                    "positive_weight": float(pos_weight),
                    "sampler_mode": str(args.sampler_mode),
                    "selected_units": int(selected.selected_units),
                }

        if best_state is None or best_metrics is None:
            raise RuntimeError("Training did not produce a valid best checkpoint.")

        torch.save(best_state, model_path)
        write_rows(metrics_csv, METRICS_FIELDS, history)

        summary_row = {
            "status": "success",
            "out_dir": str(out_dir),
            "source_jsonl": "|".join(str(Path(path).expanduser()) for path in args.jsonl),
            "ranker_checkpoint": str(ranker.checkpoint_path),
            "selected_units": int(selected.selected_units),
            "train_rows": int(split.train_labels.shape[0]),
            "val_rows": int(split.val_labels.shape[0]),
            "train_positive": int(train_positive),
            "train_negative": int(train_negative),
            "val_positive": int(val_positive),
            "val_negative": int(val_negative),
            "epochs": int(args.epochs),
            "best_epoch": int(best_metrics["best_epoch"]),
            "best_val_loss": float(best_metrics["best_val_loss"]),
            "best_val_accuracy": float(best_metrics["best_val_accuracy"]),
            "best_val_balanced_accuracy": float(best_metrics["best_val_balanced_accuracy"]),
            "best_val_precision": float(best_metrics["best_val_precision"]),
            "best_val_recall": float(best_metrics["best_val_recall"]),
            "best_val_f1": float(best_metrics["best_val_f1"]),
            "best_val_average_precision": float(best_metrics["best_val_average_precision"]),
            "fixed_threshold": float(args.fixed_threshold),
            "decision_threshold": float(best_metrics["best_val_swept_threshold"]),
            "best_val_swept_precision": float(best_metrics["best_val_swept_precision"]),
            "best_val_swept_recall": float(best_metrics["best_val_swept_recall"]),
            "best_val_swept_f1": float(best_metrics["best_val_swept_f1"]),
            "best_val_swept_balanced_accuracy": float(best_metrics["best_val_swept_balanced_accuracy"]),
            "train_sequences": ",".join(split.train_sequences),
            "val_sequences": ",".join(split.val_sequences),
            "selection_metric": str(args.selection_metric),
            "sampler_mode": str(args.sampler_mode),
            "class_balance_ratio": float(class_balance_ratio),
            "positive_weight": float(pos_weight),
            "positive_weight_power": float(args.positive_weight_power),
            "positive_weight_cap": float(args.positive_weight_cap),
            "hidden_dim": int(args.hidden_dim),
            "dropout": float(args.dropout),
            "model_path": str(model_path),
            "metrics_csv": str(metrics_csv),
            "feature_stats_json": str(feature_stats_json),
            "notes": "completed local contention acceptance gate training",
        }
        write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)
        append_registry(
            status="success",
            out_dir=out_dir,
            summary_csv=summary_csv,
            notes=(
                "completed local contention acceptance gate training "
                f"best_val_average_precision={best_metrics['best_val_average_precision']:.4f} "
                f"best_val_swept_f1={best_metrics['best_val_swept_f1']:.4f}"
            ),
            registry_csv=args.registry_csv,
            dataset_tag=dataset_tag,
            split_label=split_label,
        )
    except Exception as exc:
        failed_summary = dict(running_summary)
        failed_summary["status"] = "failed"
        failed_summary["notes"] = f"failed: {exc}"
        write_single_row_csv(summary_csv, SUMMARY_FIELDS, failed_summary)
        append_registry(
            status="failed",
            out_dir=out_dir,
            summary_csv=summary_csv,
            notes=f"local contention acceptance gate training failed: {exc}",
            registry_csv=args.registry_csv,
            dataset_tag=dataset_tag,
            split_label=split_label,
        )
        raise


if __name__ == "__main__":
    main()
