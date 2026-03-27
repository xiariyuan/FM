#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.local_conflict_set_predictor import (
    FEATURE_VERSION,
    HostConditionedLocalConflictSetPredictor,
    normalize_host_vocab,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HostConditionedLocalConflictSetPredictor.")
    parser.add_argument("--data-jsonl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-conflict-blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--train-sequences", type=str, default="")
    parser.add_argument("--val-sequences", type=str, default="")
    parser.add_argument("--strict-sequence-split", action="store_true")
    parser.add_argument(
        "--split-strategy",
        choices=["auto", "sequence", "random", "stratified_random"],
        default="auto",
    )
    parser.add_argument("--split-target-key", type=str, default="cluster_should_intervene_bridge")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--min-val-examples", type=int, default=64)
    parser.add_argument("--dataset-tag", type=str, default="")
    parser.add_argument("--source-manifest", type=str, default="")
    parser.add_argument("--feature-version", type=str, default=FEATURE_VERSION)
    parser.add_argument("--cluster-gate-thresh", type=float, default=0.5)
    parser.add_argument("--cluster-gate-calibration", choices=["none", "temp_bias"], default="none")
    parser.add_argument(
        "--cluster-gate-select-metric",
        choices=["f0.5", "utility", "bounded_utility"],
        default="f0.5",
    )
    parser.add_argument("--cluster-gate-beta", type=float, default=0.5)
    parser.add_argument("--cluster-gate-fp-weight", type=float, default=2.0)
    parser.add_argument("--cluster-gate-search-min", type=float, default=0.05)
    parser.add_argument("--cluster-gate-search-max", type=float, default=0.80)
    parser.add_argument("--cluster-gate-search-steps", type=int, default=19)
    parser.add_argument("--cluster-gate-loss-mode", choices=["bce", "weighted_bce"], default="weighted_bce")
    parser.add_argument("--cluster-gate-positive-weight", type=float, default=16.0)
    parser.add_argument("--cluster-gate-negative-weight", type=float, default=1.0)
    parser.add_argument(
        "--train-positive-cluster-oversample",
        type=float,
        default=4.0,
        help="If >1, oversample train clusters with positive cluster_target_key via WeightedRandomSampler.",
    )
    parser.add_argument(
        "--model-selection-metric",
        choices=[
            "val_loss",
            "val_cluster_gate_f0_5",
            "val_cluster_gate_utility",
            "val_selective_utility_targetcov",
            "selective_utility_cov",
            "commit_viability_utility_cov",
            "hybrid_gate_f0_5_loss",
            "hybrid_gate_utility_loss",
        ],
        default="commit_viability_utility_cov",
    )
    parser.add_argument("--assignment-target-key", type=str, default="target_by_det_bridge")
    parser.add_argument("--assignment-row-mask-key", type=str, default="row_bridge_mask")
    parser.add_argument("--edge-loss-mask-key", type=str, default="assignment_row_mask")
    parser.add_argument("--edge-target-key", type=str, default="edge_is_bridge_commit")
    parser.add_argument("--cluster-target-key", type=str, default="cluster_should_intervene_bridge")
    parser.add_argument("--target-gate-coverage-min", type=float, default=0.01)
    parser.add_argument("--target-gate-coverage-max", type=float, default=0.05)
    parser.add_argument("--coverage-penalty-weight", type=float, default=1.0)
    parser.add_argument("--keep-row-loss-weight", type=float, default=0.0)
    parser.add_argument("--edit-row-loss-weight", type=float, default=2.0)
    parser.add_argument("--edit-edge-positive-weight", type=float, default=32.0)
    parser.add_argument("--loss-assign-weight", type=float, default=1.0)
    parser.add_argument("--loss-edge-weight", type=float, default=0.5)
    parser.add_argument("--loss-cluster-weight", type=float, default=0.25)
    parser.add_argument("--loss-margin-weight", type=float, default=0.25)
    parser.add_argument("--margin-commit", type=float, default=0.2)
    parser.add_argument("--margin-row", type=float, default=0.2)
    parser.add_argument("--margin-defer", type=float, default=0.2)
    parser.add_argument("--margin-host-edit", type=float, default=0.1)
    parser.add_argument("--edge-focal-alpha", type=float, default=0.25)
    parser.add_argument("--edge-focal-gamma", type=float, default=2.0)
    parser.add_argument("--score-jitter-std", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--skip-step-grad-norm", type=float, default=1000000.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _write_single_row_csv(path: Path, fieldnames: list[str], row: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_csv_tokens(raw: str) -> list[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    return [token.strip() for token in text.split(",") if token.strip()]


def _sequence_aliases(seq: str) -> set[str]:
    raw = str(seq or "").strip()
    if not raw:
        return set()
    name = Path(raw).name
    aliases = {raw, name}
    if name.count("-") >= 2:
        aliases.add(name.rsplit("-", 1)[0])
    return {token for token in aliases if token}


def _sequence_matches(seq: str, tokens: list[str]) -> bool:
    if not tokens:
        return False
    aliases = _sequence_aliases(seq)
    for token in tokens:
        token = str(token or "").strip()
        if not token:
            continue
        for alias in aliases:
            if alias == token or alias.startswith(token):
                return True
    return False


class LocalConflictSetDataset(Dataset):
    def __init__(self, samples: list[dict[str, Any]], host_vocab: list[str]) -> None:
        self.samples = samples
        self.host_vocab = normalize_host_vocab(host_vocab)
        self.host_to_id = {host: idx for idx, host in enumerate(self.host_vocab)}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.samples[index]
        host_variant = str(row.get("host_variant", "")).strip()
        host_variant_id = int(self.host_to_id.get(host_variant, 0))
        target_by_det = list(row.get("target_by_det", []))
        row_edit_mask = row.get("row_edit_mask", None)
        if not isinstance(row_edit_mask, list) or len(row_edit_mask) != len(target_by_det):
            row_edit_mask = [1 for _ in target_by_det]
        return {
            "cluster_id": str(row["cluster_id"]),
            "seq": str(row["seq"]),
            "frame": int(row["frame"]),
            "source_tag": str(row.get("source_tag", "")),
            "host_variant": host_variant,
            "host_variant_id": int(host_variant_id),
            "split_tag": str(row.get("split_tag", "")),
            "feature_version": str(row.get("feature_version", "")),
            "teacher_mode": str(row.get("teacher_mode", "")),
            "det_features": torch.tensor(row["det_features"], dtype=torch.float32),
            "track_features": torch.tensor(row["track_features"], dtype=torch.float32),
            "edge_features": torch.tensor(row["edge_features"], dtype=torch.float32),
            "edge_det_index": torch.tensor(row["edge_det_index"], dtype=torch.long),
            "edge_track_index": torch.tensor(row["edge_track_index"], dtype=torch.long),
            "cluster_features": torch.tensor(row["cluster_features"], dtype=torch.float32),
            "target_by_det": torch.tensor(target_by_det, dtype=torch.long),
            "target_by_det_oracle": torch.tensor(row.get("target_by_det_oracle", target_by_det), dtype=torch.long),
            "target_by_det_delta": torch.tensor(row.get("target_by_det_delta", target_by_det), dtype=torch.long),
            "target_by_det_edit": torch.tensor(row.get("target_by_det_edit", target_by_det), dtype=torch.long),
            "target_by_det_rescue": torch.tensor(row.get("target_by_det_rescue", target_by_det), dtype=torch.long),
            "target_by_det_sparse_edit": torch.tensor(row.get("target_by_det_sparse_edit", row.get("target_by_det_pairs", target_by_det)), dtype=torch.long),
            "target_by_det_bridge": torch.tensor(row.get("target_by_det_bridge", target_by_det), dtype=torch.long),
            "trigger_pass": int(row.get("trigger_pass", 0)),
            "cluster_should_intervene": int(row.get("cluster_should_intervene", row.get("trigger_pass", 0))),
            "cluster_should_intervene_delta": int(row.get("cluster_should_intervene_delta", row.get("cluster_should_intervene", row.get("trigger_pass", 0)))),
            "cluster_should_intervene_edit": int(row.get("cluster_should_intervene_edit", row.get("cluster_should_intervene", row.get("trigger_pass", 0)))),
            "cluster_should_intervene_soft": int(row.get("cluster_should_intervene_soft", row.get("cluster_should_intervene", row.get("trigger_pass", 0)))),
            "cluster_should_intervene_sparse": int(row.get("cluster_should_intervene_sparse", row.get("cluster_should_intervene_soft", row.get("cluster_should_intervene", row.get("trigger_pass", 0))))),
            "cluster_should_intervene_bridge": int(row.get("cluster_should_intervene_bridge", row.get("trigger_pass", 0))),
            "cluster_utility_gain": float(row.get("cluster_utility_gain", 0.0)),
            "target_committed_matches": int(row.get("target_committed_matches", 0)),
            "row_edit_mask": torch.tensor(row_edit_mask, dtype=torch.float32),
            "row_rescue_mask": torch.tensor(row.get("row_rescue_mask", row_edit_mask), dtype=torch.float32),
            "row_sparse_edit_mask": torch.tensor(row.get("row_sparse_edit_mask", row.get("row_rescue_mask", row_edit_mask)), dtype=torch.float32),
            "row_bridge_mask": torch.tensor(row.get("row_bridge_mask", [0 for _ in target_by_det]), dtype=torch.float32),
            "host_action_by_det_runtime": torch.tensor(row.get("host_action_by_det_runtime", [-1 for _ in target_by_det]), dtype=torch.long),
            "is_large_component": int(row.get("is_large_component", 0)),
            "edge_is_gt_positive": torch.tensor(row.get("edge_is_gt_positive", []), dtype=torch.float32),
            "edge_is_oracle_commit": torch.tensor(row.get("edge_is_oracle_commit", []), dtype=torch.float32),
            "edge_is_delta_commit": torch.tensor(row.get("edge_is_delta_commit", []), dtype=torch.float32),
            "edge_is_edit_commit": torch.tensor(
                row.get("edge_is_edit_commit", row.get("edge_is_delta_commit", row.get("edge_is_oracle_commit", []))),
                dtype=torch.float32,
            ),
            "edge_is_soft_rescue": torch.tensor(
                row.get("edge_is_soft_rescue", row.get("edge_is_edit_commit", row.get("edge_is_delta_commit", row.get("edge_is_oracle_commit", [])))),
                dtype=torch.float32,
            ),
            "edge_is_sparse_edit": torch.tensor(
                row.get("edge_is_sparse_edit", row.get("edge_is_soft_rescue", row.get("edge_is_edit_commit", row.get("edge_is_delta_commit", row.get("edge_is_oracle_commit", []))))),
                dtype=torch.float32,
            ),
            "edge_is_bridge_commit": torch.tensor(
                row.get("edge_is_bridge_commit", row.get("edge_is_sparse_edit", row.get("edge_is_soft_rescue", row.get("edge_is_edit_commit", row.get("edge_is_delta_commit", row.get("edge_is_oracle_commit", [])))))),
                dtype=torch.float32,
            ),
        }


def _collate_samples(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return batch


def _load_examples(path: Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
    if not examples:
        raise ValueError(f"No examples found in {path}")
    return examples


def _fallback_split(examples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    split_at = max(int(len(examples) * 0.8), 1)
    train_rows = examples[:split_at]
    val_rows = examples[split_at:] or examples[-1:]
    return train_rows, val_rows


def _normalize_split_fraction(val_fraction: float) -> float:
    return float(min(max(float(val_fraction), 0.05), 0.5))


def _split_bucket_key(value: Any) -> str:
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True, ensure_ascii=True)
    return str(value)


def _random_split(
    examples: list[dict[str, Any]],
    *,
    seed: int,
    val_fraction: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(examples) <= 1:
        return _fallback_split(examples)
    rng = random.Random(int(seed))
    indices = list(range(len(examples)))
    rng.shuffle(indices)
    val_count = int(round(len(examples) * _normalize_split_fraction(val_fraction)))
    val_count = min(max(val_count, 1), len(examples) - 1)
    val_index = set(indices[:val_count])
    train_rows = [row for idx, row in enumerate(examples) if idx not in val_index]
    val_rows = [row for idx, row in enumerate(examples) if idx in val_index]
    return train_rows, val_rows


def _stratified_random_split(
    examples: list[dict[str, Any]],
    *,
    seed: int,
    val_fraction: float,
    split_target_key: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(examples) <= 1:
        return _fallback_split(examples)
    rng = random.Random(int(seed))
    groups: dict[str, list[int]] = {}
    for idx, row in enumerate(examples):
        bucket = _split_bucket_key(row.get(str(split_target_key), 0))
        groups.setdefault(bucket, []).append(idx)

    val_index: set[int] = set()
    for group_indices in groups.values():
        shuffled = list(group_indices)
        rng.shuffle(shuffled)
        if len(shuffled) <= 1:
            continue
        group_val_count = int(round(len(shuffled) * _normalize_split_fraction(val_fraction)))
        group_val_count = min(max(group_val_count, 1), len(shuffled) - 1)
        val_index.update(shuffled[:group_val_count])

    if not val_index or len(val_index) >= len(examples):
        return _random_split(examples, seed=seed, val_fraction=val_fraction)
    train_rows = [row for idx, row in enumerate(examples) if idx not in val_index]
    val_rows = [row for idx, row in enumerate(examples) if idx in val_index]
    if not train_rows or not val_rows:
        return _random_split(examples, seed=seed, val_fraction=val_fraction)
    return train_rows, val_rows


def _split_examples(
    *,
    examples: list[dict[str, Any]],
    train_sequences_arg: str,
    val_sequences_arg: str,
    strict_sequence_split: bool,
    split_strategy: str,
    split_target_key: str,
    val_fraction: float,
    min_val_examples: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], list[str], str]:
    train_tokens = _parse_csv_tokens(train_sequences_arg)
    val_tokens = _parse_csv_tokens(val_sequences_arg)
    all_sequences = sorted({str(row["seq"]) for row in examples})
    seq_counts = {seq: sum(1 for row in examples if str(row["seq"]) == seq) for seq in all_sequences}

    tagged_train = [row for row in examples if str(row.get("split_tag", "")) == "train"]
    tagged_val = [row for row in examples if str(row.get("split_tag", "")) == "val"]

    split_mode = ""
    effective_strategy = str(split_strategy or "auto")

    if effective_strategy == "random":
        train_rows, val_rows = _random_split(examples, seed=int(seed), val_fraction=float(val_fraction))
        split_mode = "random"
    elif effective_strategy == "stratified_random":
        train_rows, val_rows = _stratified_random_split(
            examples,
            seed=int(seed),
            val_fraction=float(val_fraction),
            split_target_key=str(split_target_key),
        )
        split_mode = f"stratified_random:{split_target_key}"
    elif effective_strategy == "sequence" and (train_tokens or val_tokens):
        train_rows = []
        val_rows = []
        for row in examples:
            seq = str(row["seq"])
            if _sequence_matches(seq, val_tokens):
                val_rows.append(row)
            elif train_tokens:
                if _sequence_matches(seq, train_tokens):
                    train_rows.append(row)
            else:
                train_rows.append(row)
        split_mode = "requested_sequences"
    elif tagged_train or tagged_val:
        train_rows = list(tagged_train)
        val_rows = list(tagged_val)
        split_mode = "manifest_split_tag"
    elif train_tokens or val_tokens:
        train_rows = []
        val_rows = []
        for row in examples:
            seq = str(row["seq"])
            if _sequence_matches(seq, val_tokens):
                val_rows.append(row)
            elif train_tokens:
                if _sequence_matches(seq, train_tokens):
                    train_rows.append(row)
            else:
                train_rows.append(row)
        split_mode = "requested_sequences"
    elif len(all_sequences) > 1:
        val_sequence = min(all_sequences, key=lambda seq: (seq_counts.get(seq, 0), seq))
        train_rows = [row for row in examples if str(row["seq"]) != val_sequence]
        val_rows = [row for row in examples if str(row["seq"]) == val_sequence]
        split_mode = "auto_smallest_sequence"
    else:
        train_rows, val_rows = _fallback_split(examples)
        split_mode = "fallback_80_20"

    if not train_rows or not val_rows or len(val_rows) < int(min_val_examples):
        if strict_sequence_split:
            raise ValueError(
                "Strict sequence split failed: "
                f"train_rows={len(train_rows)} val_rows={len(val_rows)} min_val_examples={int(min_val_examples)}"
            )
        train_rows, val_rows = _fallback_split(examples)
        split_mode = "fallback_80_20"

    train_sequences = sorted({str(row["seq"]) for row in train_rows})
    val_sequences = sorted({str(row["seq"]) for row in val_rows})
    return train_rows, val_rows, train_sequences, val_sequences, split_mode


def _join_unique(rows: list[dict[str, Any]], key: str) -> str:
    values = sorted({str(row.get(key, "")).strip() for row in rows if str(row.get(key, "")).strip()})
    return ",".join(values)


def _count_positive(rows: list[dict[str, Any]], key: str) -> int:
    target_key = str(key or "").strip()
    if not target_key:
        return 0
    total = 0
    for row in rows:
        value = row.get(target_key, 0)
        if isinstance(value, (int, float)):
            total += int(float(value) > 0.5)
    return int(total)


def _build_positive_cluster_sampler(
    rows: list[dict[str, Any]],
    *,
    positive_key: str,
    positive_oversample: float,
    seed: int,
) -> WeightedRandomSampler | None:
    if float(positive_oversample) <= 1.0 or not rows:
        return None
    weights: list[float] = []
    positive_count = 0
    for row in rows:
        value = row.get(str(positive_key), 0)
        is_positive = False
        if isinstance(value, (int, float)):
            is_positive = float(value) > 0.5
        weight = float(positive_oversample) if is_positive else 1.0
        weights.append(weight)
        positive_count += int(is_positive)
    if positive_count <= 0:
        return None
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )


def _compute_feature_stats(rows: list[dict[str, Any]]) -> dict[str, dict[str, list[float]]]:
    stats: dict[str, dict[str, list[float]]] = {}
    for key in ("det_features", "track_features", "edge_features", "cluster_features"):
        chunks = []
        for row in rows:
            values = row.get(key, [])
            tensor = torch.tensor(values, dtype=torch.float32)
            if tensor.numel() == 0:
                continue
            if tensor.ndim == 1:
                tensor = tensor.view(1, -1)
            chunks.append(tensor)
        if not chunks:
            stats[key] = {"mean": [], "std": []}
            continue
        merged = torch.cat(chunks, dim=0)
        stats[key] = {
            "mean": merged.mean(dim=0).tolist(),
            "std": merged.std(dim=0, unbiased=False).tolist(),
        }
    return stats


def _binary_focal_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float,
    gamma: float,
    positive_weight: float = 1.0,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if mask is not None:
        mask = mask.to(dtype=torch.bool, device=logits.device).view(-1)
        logits = logits.view(-1)[mask]
        targets = targets.view(-1)[mask]
    if logits.numel() == 0:
        return logits.new_zeros(())
    probs = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = probs * targets + (1.0 - probs) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    loss = alpha_t * (1.0 - pt).pow(gamma) * ce
    if float(positive_weight) != 1.0:
        pos_weight = torch.where(targets > 0.5, loss.new_full(loss.shape, float(positive_weight)), loss.new_ones(loss.shape))
        loss = loss * pos_weight
    return loss.mean()


def _average_precision(logits: torch.Tensor, targets: torch.Tensor) -> float:
    if logits.numel() == 0 or targets.numel() == 0:
        return 0.0
    positives = int((targets > 0.5).sum().item())
    if positives <= 0:
        return 0.0
    order = torch.argsort(logits, descending=True)
    sorted_targets = targets.index_select(0, order)
    tp = torch.cumsum(sorted_targets, dim=0)
    fp = torch.cumsum(1.0 - sorted_targets, dim=0)
    precision = tp / (tp + fp).clamp(min=1.0)
    ap = (precision * sorted_targets).sum() / float(positives)
    return float(ap.item())


def _cluster_f1(scores: torch.Tensor, targets: torch.Tensor, thresh: float) -> tuple[float, float, float]:
    if scores.numel() == 0 or targets.numel() == 0:
        return 0.0, 0.0, 0.0
    pred = scores >= float(thresh)
    target = targets >= 0.5
    tp = float((pred & target).sum().item())
    fp = float((pred & (~target)).sum().item())
    fn = float(((~pred) & target).sum().item())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    return precision, recall, f1


def _coverage_penalty_value(
    coverage: float,
    *,
    target_coverage_min: float,
    target_coverage_max: float,
) -> float:
    return max(float(target_coverage_min) - float(coverage), 0.0) + max(float(coverage) - float(target_coverage_max), 0.0)


def _cluster_gate_stats(
    scores: torch.Tensor,
    targets: torch.Tensor,
    *,
    thresh: float,
    beta: float,
    fp_weight: float,
    target_coverage_min: float,
    target_coverage_max: float,
    coverage_penalty_weight: float,
) -> dict[str, float]:
    if scores.numel() == 0 or targets.numel() == 0:
        return {
            "thresh": float(thresh),
            "tp": 0.0,
            "fp": 0.0,
            "fn": 0.0,
            "tn": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "f_beta": 0.0,
            "coverage": 0.0,
            "utility": 0.0,
            "coverage_penalty": 0.0,
            "bounded_utility": 0.0,
        }
    pred = scores >= float(thresh)
    target = targets >= 0.5
    tp = float((pred & target).sum().item())
    fp = float((pred & (~target)).sum().item())
    fn = float(((~pred) & target).sum().item())
    tn = float(((~pred) & (~target)).sum().item())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    beta_sq = float(beta) * float(beta)
    f_beta = (1.0 + beta_sq) * precision * recall / max(beta_sq * precision + recall, 1e-8)
    coverage = float(pred.to(dtype=torch.float32).mean().item()) if pred.numel() > 0 else 0.0
    utility = (tp - float(fp_weight) * fp) / max(float(scores.numel()), 1.0)
    coverage_penalty = _coverage_penalty_value(
        coverage,
        target_coverage_min=float(target_coverage_min),
        target_coverage_max=float(target_coverage_max),
    )
    bounded_utility = float(utility - float(coverage_penalty_weight) * float(coverage_penalty))
    return {
        "thresh": float(thresh),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f_beta": f_beta,
        "coverage": coverage,
        "utility": utility,
        "coverage_penalty": float(coverage_penalty),
        "bounded_utility": float(bounded_utility),
    }


def _cluster_gate_loss(
    logit: torch.Tensor,
    target: torch.Tensor,
    *,
    mode: str,
    positive_weight: float,
    negative_weight: float,
) -> torch.Tensor:
    logit = logit.view(1)
    target = target.view(1).to(dtype=torch.float32)
    if str(mode) == "bce":
        return F.binary_cross_entropy_with_logits(logit, target)
    if str(mode) == "weighted_bce":
        weight_value = float(positive_weight) if float(target.item()) >= 0.5 else float(negative_weight)
        weight = torch.full_like(target, fill_value=weight_value)
        return F.binary_cross_entropy_with_logits(logit, target, weight=weight)
    raise ValueError(f"Unsupported cluster gate loss mode: {mode}")


def fit_cluster_gate_calibration(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    mode: str = "temp_bias",
) -> dict[str, Any]:
    raw_logits = logits.detach().to(dtype=torch.float32).view(-1).cpu()
    raw_targets = targets.detach().to(dtype=torch.float32).view(-1).cpu()
    result = {
        "mode": str(mode),
        "temp": 1.0,
        "bias": 0.0,
        "scores": torch.sigmoid(raw_logits),
        "loss": 0.0,
    }
    if raw_logits.numel() == 0 or raw_targets.numel() == 0:
        return result
    if str(mode) == "none":
        loss = F.binary_cross_entropy_with_logits(raw_logits, raw_targets)
        result["loss"] = float(loss.item()) if bool(torch.isfinite(loss).item()) else 0.0
        return result
    if str(mode) != "temp_bias":
        raise ValueError(f"Unsupported cluster gate calibration mode: {mode}")
    if int(torch.unique(raw_targets).numel()) < 2:
        loss = F.binary_cross_entropy_with_logits(raw_logits, raw_targets)
        result["loss"] = float(loss.item()) if bool(torch.isfinite(loss).item()) else 0.0
        return result

    cal_logits = raw_logits.clone()
    cal_targets = raw_targets.clone()
    log_temp = torch.zeros((), dtype=torch.float32, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temp, bias],
        lr=0.1,
        max_iter=64,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        temp = torch.exp(log_temp).clamp(min=1e-3, max=1e3)
        logits_cal = (cal_logits / temp + bias).clamp(min=-40.0, max=40.0)
        loss = F.binary_cross_entropy_with_logits(logits_cal, cal_targets)
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
    except Exception:
        return result

    with torch.no_grad():
        temp = float(torch.exp(log_temp).clamp(min=1e-3, max=1e3).item())
        bias_value = float(bias.clamp(min=-20.0, max=20.0).item())
        logits_cal = (cal_logits / max(temp, 1e-6) + bias_value).clamp(min=-40.0, max=40.0)
        loss = F.binary_cross_entropy_with_logits(logits_cal, cal_targets)
        result.update(
            {
                "temp": temp,
                "bias": bias_value,
                "scores": torch.sigmoid(logits_cal),
                "loss": float(loss.item()) if bool(torch.isfinite(loss).item()) else 0.0,
            }
        )
    return result


def select_cluster_gate_threshold(
    scores: torch.Tensor,
    targets: torch.Tensor,
    *,
    metric: str,
    beta: float,
    fp_weight: float,
    search_min: float,
    search_max: float,
    search_steps: int,
    default_thresh: float,
    enable_search: bool,
    target_coverage_min: float,
    target_coverage_max: float,
    coverage_penalty_weight: float,
) -> dict[str, float]:
    score_tensor = scores.detach().to(dtype=torch.float32).view(-1).cpu()
    target_tensor = targets.detach().to(dtype=torch.float32).view(-1).cpu()
    if score_tensor.numel() == 0 or target_tensor.numel() == 0:
        return _cluster_gate_stats(
            score_tensor,
            target_tensor,
            thresh=float(default_thresh),
            beta=float(beta),
            fp_weight=float(fp_weight),
            target_coverage_min=float(target_coverage_min),
            target_coverage_max=float(target_coverage_max),
            coverage_penalty_weight=float(coverage_penalty_weight),
        )
    min_thresh = float(min(search_min, search_max))
    max_thresh = float(max(search_min, search_max))
    if not enable_search:
        return _cluster_gate_stats(
            score_tensor,
            target_tensor,
            thresh=float(default_thresh),
            beta=float(beta),
            fp_weight=float(fp_weight),
            target_coverage_min=float(target_coverage_min),
            target_coverage_max=float(target_coverage_max),
            coverage_penalty_weight=float(coverage_penalty_weight),
        )
    if int(search_steps) <= 1:
        candidate_thresholds = [float(default_thresh)]
    else:
        candidate_thresholds = torch.linspace(min_thresh, max_thresh, steps=max(int(search_steps), 2)).tolist()
        if float(default_thresh) not in candidate_thresholds:
            candidate_thresholds.append(float(default_thresh))

    best_stats: dict[str, float] | None = None
    best_key: tuple[float, float, float, float] | None = None
    for thresh in candidate_thresholds:
        stats = _cluster_gate_stats(
            score_tensor,
            target_tensor,
            thresh=float(thresh),
            beta=float(beta),
            fp_weight=float(fp_weight),
            target_coverage_min=float(target_coverage_min),
            target_coverage_max=float(target_coverage_max),
            coverage_penalty_weight=float(coverage_penalty_weight),
        )
        if str(metric) == "utility":
            primary = float(stats["utility"])
        elif str(metric) == "bounded_utility":
            primary = float(stats["bounded_utility"])
        else:
            primary = float(stats["f_beta"])
        key = (
            primary,
            float(stats["precision"]),
            -float(stats["coverage_penalty"]),
            -float(stats["coverage"]),
            -abs(float(thresh) - float(default_thresh)),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_stats = stats
    return best_stats or _cluster_gate_stats(
        score_tensor,
        target_tensor,
        thresh=float(default_thresh),
        beta=float(beta),
        fp_weight=float(fp_weight),
        target_coverage_min=float(target_coverage_min),
        target_coverage_max=float(target_coverage_max),
        coverage_penalty_weight=float(coverage_penalty_weight),
    )


def _safe_float(value: Any, *, default: float) -> float:
    try:
        numeric = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(numeric):
        return float(default)
    return numeric


def _selection_key(
    *,
    metric_name: str,
    val_metrics: dict[str, Any],
    calibrated_gate: dict[str, Any],
) -> tuple[float, ...]:
    val_loss = _safe_float(val_metrics.get("loss", float("inf")), default=1e18)
    gate_f0_5 = _safe_float(calibrated_gate.get("f_beta", 0.0), default=0.0)
    gate_utility = _safe_float(calibrated_gate.get("utility", -1e18), default=-1e18)
    bounded_utility = _safe_float(calibrated_gate.get("bounded_utility", gate_utility), default=-1e18)
    coverage_penalty = _safe_float(calibrated_gate.get("coverage_penalty", 1e18), default=1e18)
    gate_coverage = _safe_float(calibrated_gate.get("coverage", 0.0), default=0.0)
    edge_ap = _safe_float(val_metrics.get("edge_ap", 0.0), default=0.0)
    row_acc = _safe_float(val_metrics.get("row_acc", 0.0), default=0.0)
    commit_precision = _safe_float(val_metrics.get("commit_precision", 0.0), default=0.0)
    commit_recall = _safe_float(val_metrics.get("commit_recall", 0.0), default=0.0)
    pred_commits = _safe_float(val_metrics.get("pred_commits", 0.0), default=0.0)
    target_commits = _safe_float(val_metrics.get("target_commits", 0.0), default=0.0)
    tp_commits = _safe_float(val_metrics.get("tp_commits", 0.0), default=0.0)
    commit_f1 = 0.0
    if commit_precision + commit_recall > 0.0:
        commit_f1 = (2.0 * commit_precision * commit_recall) / (commit_precision + commit_recall)
    viable_commit = 1.0 if tp_commits > 0.0 and pred_commits > 0.0 and target_commits > 0.0 else 0.0
    metric_name = str(metric_name)
    if metric_name == "val_loss":
        return (-val_loss, gate_f0_5, edge_ap, row_acc)
    if metric_name == "val_cluster_gate_f0_5":
        return (gate_f0_5, -val_loss, edge_ap, row_acc)
    if metric_name == "val_cluster_gate_utility":
        return (gate_utility, gate_f0_5, -val_loss, edge_ap, row_acc)
    if metric_name in {"val_selective_utility_targetcov", "selective_utility_cov"}:
        return (bounded_utility, gate_utility, -coverage_penalty, edge_ap, row_acc, -val_loss)
    if metric_name == "commit_viability_utility_cov":
        return (
            viable_commit,
            commit_f1,
            min(commit_precision, commit_recall),
            commit_precision,
            commit_recall,
            bounded_utility,
            gate_utility,
            gate_coverage,
            -coverage_penalty,
            edge_ap,
            row_acc,
            -val_loss,
        )
    if metric_name == "hybrid_gate_f0_5_loss":
        return (gate_f0_5, gate_utility, -val_loss, edge_ap, row_acc)
    if metric_name == "hybrid_gate_utility_loss":
        return (gate_utility, gate_f0_5, -val_loss, edge_ap, row_acc)
    raise ValueError(f"Unsupported model selection metric: {metric_name}")


def _margin_loss(
    dense_logits: torch.Tensor,
    feasible_mask: torch.Tensor,
    target: torch.Tensor,
    num_tracks: int,
    *,
    margin_commit: float,
    margin_row: float,
    margin_defer: float,
    row_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    losses = []
    for det_idx in range(int(dense_logits.shape[0])):
        if row_mask is not None:
            if det_idx >= int(row_mask.numel()) or float(row_mask[det_idx].item()) <= 0.5:
                continue
        row_track_logits = dense_logits[det_idx, :num_tracks]
        defer_logit = dense_logits[det_idx, num_tracks]
        feasible = feasible_mask[det_idx] if feasible_mask.numel() > 0 else torch.zeros((num_tracks,), device=dense_logits.device, dtype=torch.bool)
        target_idx = int(target[det_idx].item())
        if target_idx < num_tracks:
            pos_logit = row_track_logits[target_idx]
            losses.append(F.relu(float(margin_commit) - (pos_logit - defer_logit)))
            neg_mask = feasible.clone()
            if target_idx >= 0 and target_idx < num_tracks:
                neg_mask[target_idx] = False
            if bool(neg_mask.any().item()):
                hardest_negative = row_track_logits[neg_mask].max()
                losses.append(F.relu(float(margin_row) - (pos_logit - hardest_negative)))
        else:
            if bool(feasible.any().item()):
                best_edge = row_track_logits[feasible].max()
                losses.append(F.relu(float(margin_defer) - (defer_logit - best_edge)))
    if not losses:
        return dense_logits.new_zeros(())
    return torch.stack(losses).mean()


def _host_relative_edit_margin_loss(
    dense_logits: torch.Tensor,
    target: torch.Tensor,
    host_action_by_det_runtime: torch.Tensor | None,
    num_tracks: int,
    *,
    margin_host_edit: float,
    row_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if float(margin_host_edit) <= 0.0 or host_action_by_det_runtime is None:
        return dense_logits.new_zeros(())
    losses = []
    host_actions = host_action_by_det_runtime.view(-1).to(device=dense_logits.device, dtype=torch.long)
    for det_idx in range(int(dense_logits.shape[0])):
        if row_mask is not None:
            if det_idx >= int(row_mask.numel()) or float(row_mask[det_idx].item()) <= 0.5:
                continue
        target_idx = int(target[det_idx].item())
        if not (0 <= target_idx < int(num_tracks)):
            continue
        pos_logit = dense_logits[det_idx, target_idx]
        host_idx = int(host_actions[det_idx].item()) if det_idx < int(host_actions.numel()) else -1
        if 0 <= host_idx < int(num_tracks) and host_idx != target_idx:
            host_logit = dense_logits[det_idx, host_idx]
            losses.append(F.relu(float(margin_host_edit) - (pos_logit - host_logit)))
        else:
            defer_logit = dense_logits[det_idx, num_tracks]
            losses.append(F.relu(float(margin_host_edit) - (pos_logit - defer_logit)))
    if not losses:
        return dense_logits.new_zeros(())
    return torch.stack(losses).mean()


def _tensor_is_finite(tensor: torch.Tensor) -> bool:
    if tensor.numel() == 0:
        return True
    return bool(torch.isfinite(tensor).all().item())


def _model_parameters_are_finite(model: torch.nn.Module) -> bool:
    for param in model.parameters():
        if param.numel() == 0:
            continue
        if not bool(torch.isfinite(param).all().item()):
            return False
    return True


def _coverage_penalty_tensor(
    coverage: torch.Tensor,
    *,
    target_coverage_min: float,
    target_coverage_max: float,
) -> torch.Tensor:
    lower = F.relu(coverage.new_tensor(float(target_coverage_min)) - coverage)
    upper = F.relu(coverage - coverage.new_tensor(float(target_coverage_max)))
    return lower + upper


def _run_epoch(
    *,
    model: HostConditionedLocalConflictSetPredictor,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    cluster_gate_thresh: float,
    cluster_gate_temp: float,
    cluster_gate_bias: float,
    cluster_gate_loss_mode: str,
    cluster_gate_positive_weight: float,
    cluster_gate_negative_weight: float,
    assignment_target_key: str,
    assignment_row_mask_key: str,
    edge_loss_mask_key: str,
    edge_target_key: str,
    cluster_target_key: str,
    target_gate_coverage_min: float,
    target_gate_coverage_max: float,
    coverage_penalty_weight: float,
    keep_row_loss_weight: float,
    edit_row_loss_weight: float,
    edit_edge_positive_weight: float,
    loss_assign_weight: float,
    loss_edge_weight: float,
    loss_cluster_weight: float,
    loss_margin_weight: float,
    margin_commit: float,
    margin_row: float,
    margin_defer: float,
    margin_host_edit: float,
    edge_focal_alpha: float,
    edge_focal_gamma: float,
    score_jitter_std: float,
    grad_clip_norm: float,
    skip_step_grad_norm: float,
) -> dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    totals = {
        "loss": 0.0,
        "assign_loss": 0.0,
        "edge_loss": 0.0,
        "cluster_loss": 0.0,
        "margin_loss": 0.0,
        "clusters": 0.0,
        "detections": 0.0,
        "metric_rows": 0.0,
        "row_correct": 0.0,
        "pred_commits": 0.0,
        "target_commits": 0.0,
        "tp_commits": 0.0,
        "gate_pass_clusters": 0.0,
        "skipped_nonfinite_samples": 0.0,
        "skipped_nonfinite_batches": 0.0,
    }
    edge_logits_acc: list[torch.Tensor] = []
    edge_targets_acc: list[torch.Tensor] = []
    cluster_logits_acc: list[torch.Tensor] = []
    cluster_targets_acc: list[torch.Tensor] = []

    for batch in loader:
        losses = []
        batch_gate_probs: list[torch.Tensor] = []
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        for sample in batch:
            det_features = sample["det_features"].to(device)
            track_features = sample["track_features"].to(device)
            edge_features = sample["edge_features"].to(device)
            if train_mode and float(score_jitter_std) > 0.0 and edge_features.numel() > 0:
                noise = torch.zeros_like(edge_features)
                raw_dims = min(int(getattr(model, "raw_edge_score_dims", 3)), int(edge_features.shape[1]))
                noise[:, :raw_dims] = torch.randn_like(edge_features[:, :raw_dims]) * float(score_jitter_std)
                edge_features = edge_features + noise
            edge_det_index = sample["edge_det_index"].to(device)
            edge_track_index = sample["edge_track_index"].to(device)
            cluster_features = sample["cluster_features"].to(device)
            assignment_target = sample.get(str(assignment_target_key))
            if not isinstance(assignment_target, torch.Tensor) or int(assignment_target.numel()) <= 0:
                assignment_target = sample["target_by_det"]
            target_by_det = assignment_target.to(device=device, dtype=torch.long).view(-1)
            row_mask = sample.get(str(assignment_row_mask_key))
            if isinstance(row_mask, torch.Tensor):
                row_mask = row_mask.to(device=device, dtype=torch.float32).view(-1)
            else:
                row_mask = None
            edge_is_oracle_commit = sample["edge_is_oracle_commit"].to(device)
            edge_is_delta_commit = sample["edge_is_delta_commit"].to(device)
            edge_is_edit_commit = sample["edge_is_edit_commit"].to(device)
            host_action_by_det_runtime = sample.get("host_action_by_det_runtime")
            if isinstance(host_action_by_det_runtime, torch.Tensor):
                host_action_by_det_runtime = host_action_by_det_runtime.to(device=device, dtype=torch.long).view(-1)
            else:
                host_action_by_det_runtime = None
            host_variant_id = torch.tensor([int(sample["host_variant_id"])], device=device, dtype=torch.long)

            outputs = model(
                det_features=det_features,
                track_features=track_features,
                edge_features=edge_features,
                edge_det_index=edge_det_index,
                edge_track_index=edge_track_index,
                cluster_features=cluster_features,
                host_variant_id=host_variant_id,
            )
            if not (
                _tensor_is_finite(outputs["edge_logits"])
                and _tensor_is_finite(outputs["defer_logits"])
                and _tensor_is_finite(outputs["cluster_commit_logit"].view(1))
            ):
                totals["skipped_nonfinite_samples"] += 1.0
                continue
            num_tracks = int(track_features.shape[0])
            dense_logits = HostConditionedLocalConflictSetPredictor.build_dense_assignment_logits(
                num_detections=int(det_features.shape[0]),
                num_tracks=num_tracks,
                edge_logits=outputs["edge_logits"],
                edge_det_index=edge_det_index,
                edge_track_index=edge_track_index,
                defer_logits=outputs["defer_logits"],
            )
            if not _tensor_is_finite(dense_logits):
                totals["skipped_nonfinite_samples"] += 1.0
                continue
            target = target_by_det.clone()
            target[target < 0] = num_tracks
            if row_mask is None or int(row_mask.numel()) != int(target.numel()):
                row_mask = torch.ones((int(target.numel()),), device=device, dtype=torch.float32)

            feasible_mask = torch.zeros((int(det_features.shape[0]), num_tracks), device=device, dtype=torch.bool)
            if edge_det_index.numel() > 0:
                feasible_mask[edge_det_index, edge_track_index] = True

            row_loss = F.cross_entropy(dense_logits, target, reduction="none")
            row_weights = torch.where(
                row_mask > 0.5,
                row_loss.new_full(row_loss.shape, float(edit_row_loss_weight)),
                row_loss.new_full(row_loss.shape, float(keep_row_loss_weight)),
            )
            if float(row_weights.sum().item()) <= 0.0:
                row_weights = torch.ones_like(row_loss)
            assign_loss = (row_loss * row_weights).sum() / row_weights.sum().clamp(min=1.0)
            edge_targets = sample.get(str(edge_target_key))
            if not isinstance(edge_targets, torch.Tensor) or edge_targets.numel() != outputs["edge_logits"].numel():
                edge_targets = edge_is_edit_commit
            if not isinstance(edge_targets, torch.Tensor) or edge_targets.numel() != outputs["edge_logits"].numel():
                edge_targets = (
                    edge_is_delta_commit
                    if edge_is_delta_commit.numel() == outputs["edge_logits"].numel()
                    else edge_is_oracle_commit
                )
            edge_targets = edge_targets.to(device=device, dtype=torch.float32)
            edge_mask = None
            edge_mask_source: torch.Tensor | None = None
            edge_loss_mask_key_normalized = str(edge_loss_mask_key or "").strip().lower()
            if edge_loss_mask_key_normalized in {"", "none", "all"}:
                edge_mask_source = None
            elif edge_loss_mask_key_normalized == "assignment_row_mask":
                edge_mask_source = row_mask
            else:
                sample_edge_mask = sample.get(str(edge_loss_mask_key))
                if isinstance(sample_edge_mask, torch.Tensor):
                    edge_mask_source = sample_edge_mask.to(device=device, dtype=torch.float32).view(-1)
            if (
                edge_mask_source is not None
                and edge_det_index.numel() == outputs["edge_logits"].numel()
                and edge_mask_source.numel() == int(det_features.shape[0])
            ):
                edge_mask = edge_mask_source.index_select(0, edge_det_index) > 0.5
            edge_loss = _binary_focal_with_logits(
                outputs["edge_logits"],
                edge_targets,
                alpha=float(edge_focal_alpha),
                gamma=float(edge_focal_gamma),
                positive_weight=float(edit_edge_positive_weight),
                mask=edge_mask,
            )
            cluster_target_value = sample.get(str(cluster_target_key), sample.get("cluster_should_intervene", sample["trigger_pass"]))
            cluster_target = torch.tensor(
                float(cluster_target_value),
                device=device,
                dtype=torch.float32,
            )
            cluster_loss = _cluster_gate_loss(
                outputs["cluster_commit_logit"].view(()),
                cluster_target,
                mode=str(cluster_gate_loss_mode),
                positive_weight=float(cluster_gate_positive_weight),
                negative_weight=float(cluster_gate_negative_weight),
            )
            margin_loss = _margin_loss(
                dense_logits,
                feasible_mask,
                target,
                num_tracks,
                margin_commit=float(margin_commit),
                margin_row=float(margin_row),
                margin_defer=float(margin_defer),
                row_mask=row_mask,
            )
            margin_loss = margin_loss + _host_relative_edit_margin_loss(
                dense_logits,
                target,
                host_action_by_det_runtime,
                num_tracks,
                margin_host_edit=float(margin_host_edit),
                row_mask=row_mask,
            )
            total_loss = (
                float(loss_assign_weight) * assign_loss
                + float(loss_edge_weight) * edge_loss
                + float(loss_cluster_weight) * cluster_loss
                + float(loss_margin_weight) * margin_loss
            )
            if not bool(torch.isfinite(total_loss).item()):
                totals["skipped_nonfinite_samples"] += 1.0
                continue
            losses.append(total_loss)

            gate_logit = outputs["cluster_commit_logit"].view(())
            gate_score_tensor = torch.sigmoid(
                gate_logit / max(float(cluster_gate_temp), 1e-6) + float(cluster_gate_bias)
            )
            batch_gate_probs.append(gate_score_tensor)
            gate_score = gate_score_tensor.detach()
            gate_pred = bool(float(gate_score.item()) >= float(cluster_gate_thresh))
            pred = dense_logits.argmax(dim=-1)
            if not gate_pred:
                pred = torch.full_like(pred, fill_value=num_tracks)
            metric_mask = row_mask > 0.5
            if not bool(metric_mask.any().item()):
                metric_mask = torch.ones_like(metric_mask, dtype=torch.bool)

            totals["clusters"] += 1.0
            totals["detections"] += float(target.numel())
            totals["metric_rows"] += float(metric_mask.sum().item())
            totals["loss"] += float(total_loss.detach().item()) * float(target.numel())
            totals["assign_loss"] += float(assign_loss.detach().item()) * float(target.numel())
            totals["edge_loss"] += float(edge_loss.detach().item()) * float(target.numel())
            totals["cluster_loss"] += float(cluster_loss.detach().item()) * float(target.numel())
            totals["margin_loss"] += float(margin_loss.detach().item()) * float(target.numel())
            totals["row_correct"] += float(((pred == target) & metric_mask).sum().item())
            pred_commit = (pred < num_tracks) & metric_mask
            target_commit = (target < num_tracks) & metric_mask
            totals["pred_commits"] += float(pred_commit.sum().item())
            totals["target_commits"] += float(target_commit.sum().item())
            totals["tp_commits"] += float((pred_commit & target_commit).sum().item())
            totals["gate_pass_clusters"] += float(int(gate_pred))

            if outputs["edge_logits"].numel() > 0:
                if edge_mask is not None and bool(edge_mask.any().item()):
                    edge_logits_acc.append(outputs["edge_logits"].detach().cpu()[edge_mask.detach().cpu()])
                    edge_targets_acc.append(edge_targets.detach().cpu().to(dtype=torch.float32)[edge_mask.detach().cpu()])
                elif edge_mask is None:
                    edge_logits_acc.append(outputs["edge_logits"].detach().cpu())
                    edge_targets_acc.append(edge_targets.detach().cpu().to(dtype=torch.float32))
            cluster_logits_acc.append(gate_logit.detach().cpu().view(1))
            cluster_targets_acc.append(cluster_target.detach().cpu().view(1))

        if losses and optimizer is not None:
            batch_loss = torch.stack(losses).mean()
            if batch_gate_probs:
                mean_gate_coverage = torch.stack(batch_gate_probs).mean()
                batch_loss = batch_loss + float(coverage_penalty_weight) * _coverage_penalty_tensor(
                    mean_gate_coverage,
                    target_coverage_min=float(target_gate_coverage_min),
                    target_coverage_max=float(target_gate_coverage_max),
                )
            if not bool(torch.isfinite(batch_loss).item()):
                totals["skipped_nonfinite_batches"] += 1.0
                optimizer.zero_grad(set_to_none=True)
                continue
            batch_loss.backward()
            if not _model_parameters_are_finite(model):
                totals["skipped_nonfinite_batches"] += 1.0
                optimizer.zero_grad(set_to_none=True)
                continue
            if float(grad_clip_norm) > 0.0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
            else:
                grad_norm = torch.tensor(0.0, device=device)
            if not bool(torch.isfinite(grad_norm).item()) or float(grad_norm.item()) > float(skip_step_grad_norm):
                totals["skipped_nonfinite_batches"] += 1.0
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.step()
            if not _model_parameters_are_finite(model):
                totals["skipped_nonfinite_batches"] += 1.0
                optimizer.zero_grad(set_to_none=True)
                return {
                    "loss": float("inf"),
                    "assign_loss": float("inf"),
                    "edge_loss": float("inf"),
                    "cluster_loss": float("inf"),
                    "margin_loss": float("inf"),
                    "row_acc": 0.0,
                    "commit_precision": 0.0,
                    "commit_recall": 0.0,
                    "edge_ap": 0.0,
                    "cluster_precision": 0.0,
                    "cluster_recall": 0.0,
                    "cluster_f1": 0.0,
                    "clusters": totals["clusters"],
                    "detections": totals["detections"],
                    "gate_pass_clusters": totals["gate_pass_clusters"],
                    "skipped_nonfinite_samples": totals["skipped_nonfinite_samples"],
                    "skipped_nonfinite_batches": totals["skipped_nonfinite_batches"],
                    "cluster_logits": torch.zeros((0,), dtype=torch.float32),
                    "cluster_targets": torch.zeros((0,), dtype=torch.float32),
                    "pred_commits": totals["pred_commits"],
                    "target_commits": totals["target_commits"],
                    "tp_commits": totals["tp_commits"],
                }
        elif optimizer is not None:
            totals["skipped_nonfinite_batches"] += 1.0

    denom = max(totals["detections"], 1.0)
    row_denom = max(totals["metric_rows"], 1.0)
    pred_commits = max(totals["pred_commits"], 1.0)
    target_commits = max(totals["target_commits"], 1.0)
    edge_logits_cat = torch.cat(edge_logits_acc, dim=0) if edge_logits_acc else torch.zeros((0,), dtype=torch.float32)
    edge_targets_cat = torch.cat(edge_targets_acc, dim=0) if edge_targets_acc else torch.zeros((0,), dtype=torch.float32)
    cluster_logits_cat = torch.cat(cluster_logits_acc, dim=0) if cluster_logits_acc else torch.zeros((0,), dtype=torch.float32)
    cluster_targets_cat = torch.cat(cluster_targets_acc, dim=0) if cluster_targets_acc else torch.zeros((0,), dtype=torch.float32)
    cluster_scores_cat = torch.sigmoid(
        cluster_logits_cat / max(float(cluster_gate_temp), 1e-6) + float(cluster_gate_bias)
    )
    cluster_precision, cluster_recall, cluster_f1 = _cluster_f1(
        cluster_scores_cat,
        cluster_targets_cat,
        float(cluster_gate_thresh),
    )
    if totals["detections"] <= 0.0:
        return {
            "loss": float("inf"),
            "assign_loss": float("inf"),
            "edge_loss": float("inf"),
            "cluster_loss": float("inf"),
            "margin_loss": float("inf"),
            "row_acc": 0.0,
            "commit_precision": 0.0,
            "commit_recall": 0.0,
            "edge_ap": 0.0,
            "cluster_precision": cluster_precision,
            "cluster_recall": cluster_recall,
            "cluster_f1": cluster_f1,
            "clusters": totals["clusters"],
            "detections": totals["detections"],
            "gate_pass_clusters": totals["gate_pass_clusters"],
            "skipped_nonfinite_samples": totals["skipped_nonfinite_samples"],
            "skipped_nonfinite_batches": totals["skipped_nonfinite_batches"],
            "cluster_logits": cluster_logits_cat,
            "cluster_targets": cluster_targets_cat,
            "pred_commits": totals["pred_commits"],
            "target_commits": totals["target_commits"],
            "tp_commits": totals["tp_commits"],
        }
    return {
        "loss": totals["loss"] / denom,
        "assign_loss": totals["assign_loss"] / denom,
        "edge_loss": totals["edge_loss"] / denom,
        "cluster_loss": totals["cluster_loss"] / denom,
        "margin_loss": totals["margin_loss"] / denom,
        "row_acc": totals["row_correct"] / row_denom,
        "commit_precision": totals["tp_commits"] / pred_commits,
        "commit_recall": totals["tp_commits"] / target_commits,
        "edge_ap": _average_precision(edge_logits_cat, edge_targets_cat),
        "cluster_precision": cluster_precision,
        "cluster_recall": cluster_recall,
        "cluster_f1": cluster_f1,
        "clusters": totals["clusters"],
        "detections": totals["detections"],
        "gate_pass_clusters": totals["gate_pass_clusters"],
        "skipped_nonfinite_samples": totals["skipped_nonfinite_samples"],
        "skipped_nonfinite_batches": totals["skipped_nonfinite_batches"],
        "cluster_logits": cluster_logits_cat,
        "cluster_targets": cluster_targets_cat,
        "pred_commits": totals["pred_commits"],
        "target_commits": totals["target_commits"],
        "tp_commits": totals["tp_commits"],
    }


def main() -> int:
    args = parse_args()
    _set_seed(int(args.seed))

    data_jsonl = Path(args.data_jsonl).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    result_csv = out_dir / "result.csv"
    summary_csv = out_dir / "summary.csv"
    metrics_jsonl = out_dir / "metrics.jsonl"
    best_ckpt = out_dir / "best.pt"

    running_row = {
        "exp_name": "local_conflict_set_predictor_stage1",
        "module_family": "set_predictor_v2",
        "data_jsonl": str(data_jsonl),
        "source_manifest": str(args.source_manifest or ""),
        "dataset_tag": str(args.dataset_tag or ""),
        "feature_version": str(args.feature_version or ""),
        "teacher_mode": "",
        "checkpoint": "",
        "train_examples": "",
        "val_examples": "",
        "train_sequences": "",
        "val_sequences": "",
        "train_host_variants": "",
        "val_host_variants": "",
        "train_target_positives": "",
        "val_target_positives": "",
        "host_vocab": "",
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "hidden_dim": int(args.hidden_dim),
        "num_heads": int(args.num_heads),
        "num_conflict_blocks": int(args.num_conflict_blocks),
        "dropout": float(args.dropout),
        "strict_sequence_split": int(bool(args.strict_sequence_split)),
        "split_strategy": str(args.split_strategy),
        "split_target_key": str(args.split_target_key),
        "val_fraction": float(args.val_fraction),
        "min_val_examples": int(args.min_val_examples),
        "cluster_gate_thresh": float(args.cluster_gate_thresh),
        "cluster_gate_calibration": str(args.cluster_gate_calibration),
        "cluster_gate_select_metric": str(args.cluster_gate_select_metric),
        "cluster_gate_beta": float(args.cluster_gate_beta),
        "cluster_gate_fp_weight": float(args.cluster_gate_fp_weight),
        "cluster_gate_search_min": float(args.cluster_gate_search_min),
        "cluster_gate_search_max": float(args.cluster_gate_search_max),
        "cluster_gate_search_steps": int(args.cluster_gate_search_steps),
        "cluster_gate_loss_mode": str(args.cluster_gate_loss_mode),
        "cluster_gate_positive_weight": float(args.cluster_gate_positive_weight),
        "cluster_gate_negative_weight": float(args.cluster_gate_negative_weight),
        "train_positive_cluster_oversample": float(args.train_positive_cluster_oversample),
        "model_selection_metric": str(args.model_selection_metric),
        "assignment_target_key": str(args.assignment_target_key),
        "assignment_row_mask_key": str(args.assignment_row_mask_key),
        "edge_loss_mask_key": str(args.edge_loss_mask_key),
        "edge_target_key": str(args.edge_target_key),
        "cluster_target_key": str(args.cluster_target_key),
        "target_gate_coverage_min": float(args.target_gate_coverage_min),
        "target_gate_coverage_max": float(args.target_gate_coverage_max),
        "coverage_penalty_weight": float(args.coverage_penalty_weight),
        "keep_row_loss_weight": float(args.keep_row_loss_weight),
        "edit_row_loss_weight": float(args.edit_row_loss_weight),
        "edit_edge_positive_weight": float(args.edit_edge_positive_weight),
        "loss_assign_weight": float(args.loss_assign_weight),
        "loss_edge_weight": float(args.loss_edge_weight),
        "loss_cluster_weight": float(args.loss_cluster_weight),
        "loss_margin_weight": float(args.loss_margin_weight),
        "margin_host_edit": float(args.margin_host_edit),
        "score_jitter_std": float(args.score_jitter_std),
        "split_mode": "",
        "best_epoch": "",
        "train_loss": "",
        "val_loss": "",
        "train_row_acc": "",
        "val_row_acc": "",
        "val_commit_precision": "",
        "val_commit_recall": "",
        "val_edge_ap": "",
        "val_cluster_f1": "",
        "val_cluster_gate_temp": "",
        "val_cluster_gate_bias": "",
        "val_cluster_gate_thresh_calibrated": "",
        "val_cluster_gate_precision_cal": "",
        "val_cluster_gate_recall_cal": "",
        "val_cluster_gate_f0_5": "",
        "val_cluster_gate_utility_cal": "",
        "val_cluster_gate_coverage_cal": "",
        "val_cluster_gate_bounded_utility": "",
        "val_cluster_gate_coverage_penalty": "",
        "val_selective_utility_targetcov": "",
        "train_skipped_nonfinite_samples": "",
        "train_skipped_nonfinite_batches": "",
        "val_skipped_nonfinite_samples": "",
        "val_skipped_nonfinite_batches": "",
        "status": "running",
        "error": "",
    }
    fieldnames = list(running_row.keys())
    _write_single_row_csv(result_csv, fieldnames, running_row)
    _write_single_row_csv(summary_csv, fieldnames, running_row)
    metrics_jsonl.write_text("", encoding="utf-8")

    try:
        examples = _load_examples(data_jsonl)
        train_rows, val_rows, train_sequences, val_sequences, split_mode = _split_examples(
            examples=examples,
            train_sequences_arg=str(args.train_sequences or ""),
            val_sequences_arg=str(args.val_sequences or ""),
            strict_sequence_split=bool(args.strict_sequence_split),
            split_strategy=str(args.split_strategy),
            split_target_key=str(args.split_target_key),
            val_fraction=float(args.val_fraction),
            min_val_examples=int(args.min_val_examples),
            seed=int(args.seed),
        )
        host_vocab = normalize_host_vocab(
            sorted(
                {
                    str(row.get("host_variant", "")).strip()
                    for row in examples
                    if str(row.get("host_variant", "")).strip()
                }
            )
        )
        train_dataset = LocalConflictSetDataset(train_rows, host_vocab)
        train_sampler = _build_positive_cluster_sampler(
            train_rows,
            positive_key=str(args.cluster_target_key),
            positive_oversample=float(args.train_positive_cluster_oversample),
            seed=int(args.seed),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=max(int(args.batch_size), 1),
            shuffle=bool(train_sampler is None),
            sampler=train_sampler,
            collate_fn=_collate_samples,
        )
        val_loader = DataLoader(
            LocalConflictSetDataset(val_rows, host_vocab),
            batch_size=max(int(args.batch_size), 1),
            shuffle=False,
            collate_fn=_collate_samples,
        )

        feature_stats = _compute_feature_stats(train_rows)
        teacher_mode = _join_unique(examples, "teacher_mode")
        device = torch.device(args.device)
        model = HostConditionedLocalConflictSetPredictor(
            hidden_dim=int(args.hidden_dim),
            num_heads=int(args.num_heads),
            num_conflict_blocks=int(args.num_conflict_blocks),
            dropout=float(args.dropout),
            num_host_variants=len(host_vocab),
        ).to(device)
        model.host_vocab = list(host_vocab)
        model.feature_stats = dict(feature_stats)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
        )

        best_val_loss = float("inf")
        best_row: dict[str, Any] | None = None
        with metrics_jsonl.open("a", encoding="utf-8") as metrics_fp:
            for epoch in range(1, int(args.epochs) + 1):
                train_metrics = _run_epoch(
                    model=model,
                    loader=train_loader,
                    optimizer=optimizer,
                    device=device,
                    cluster_gate_thresh=float(args.cluster_gate_thresh),
                    cluster_gate_temp=1.0,
                    cluster_gate_bias=0.0,
                    cluster_gate_loss_mode=str(args.cluster_gate_loss_mode),
                    cluster_gate_positive_weight=float(args.cluster_gate_positive_weight),
                    cluster_gate_negative_weight=float(args.cluster_gate_negative_weight),
                    assignment_target_key=str(args.assignment_target_key),
                    assignment_row_mask_key=str(args.assignment_row_mask_key),
                    edge_loss_mask_key=str(args.edge_loss_mask_key),
                    edge_target_key=str(args.edge_target_key),
                    cluster_target_key=str(args.cluster_target_key),
                    target_gate_coverage_min=float(args.target_gate_coverage_min),
                    target_gate_coverage_max=float(args.target_gate_coverage_max),
                    coverage_penalty_weight=float(args.coverage_penalty_weight),
                    keep_row_loss_weight=float(args.keep_row_loss_weight),
                    edit_row_loss_weight=float(args.edit_row_loss_weight),
                    edit_edge_positive_weight=float(args.edit_edge_positive_weight),
                    loss_assign_weight=float(args.loss_assign_weight),
                    loss_edge_weight=float(args.loss_edge_weight),
                    loss_cluster_weight=float(args.loss_cluster_weight),
                    loss_margin_weight=float(args.loss_margin_weight),
                    margin_commit=float(args.margin_commit),
                    margin_row=float(args.margin_row),
                    margin_defer=float(args.margin_defer),
                    margin_host_edit=float(args.margin_host_edit),
                    edge_focal_alpha=float(args.edge_focal_alpha),
                    edge_focal_gamma=float(args.edge_focal_gamma),
                    score_jitter_std=float(args.score_jitter_std),
                    grad_clip_norm=float(args.grad_clip_norm),
                    skip_step_grad_norm=float(args.skip_step_grad_norm),
                )
                with torch.inference_mode():
                    val_metrics = _run_epoch(
                        model=model,
                        loader=val_loader,
                        optimizer=None,
                        device=device,
                        cluster_gate_thresh=float(args.cluster_gate_thresh),
                        cluster_gate_temp=1.0,
                        cluster_gate_bias=0.0,
                        cluster_gate_loss_mode=str(args.cluster_gate_loss_mode),
                        cluster_gate_positive_weight=float(args.cluster_gate_positive_weight),
                        cluster_gate_negative_weight=float(args.cluster_gate_negative_weight),
                        assignment_target_key=str(args.assignment_target_key),
                        assignment_row_mask_key=str(args.assignment_row_mask_key),
                        edge_loss_mask_key=str(args.edge_loss_mask_key),
                        edge_target_key=str(args.edge_target_key),
                        cluster_target_key=str(args.cluster_target_key),
                        target_gate_coverage_min=float(args.target_gate_coverage_min),
                        target_gate_coverage_max=float(args.target_gate_coverage_max),
                        coverage_penalty_weight=float(args.coverage_penalty_weight),
                        keep_row_loss_weight=float(args.keep_row_loss_weight),
                        edit_row_loss_weight=float(args.edit_row_loss_weight),
                        edit_edge_positive_weight=float(args.edit_edge_positive_weight),
                        loss_assign_weight=float(args.loss_assign_weight),
                        loss_edge_weight=float(args.loss_edge_weight),
                        loss_cluster_weight=float(args.loss_cluster_weight),
                        loss_margin_weight=float(args.loss_margin_weight),
                        margin_commit=float(args.margin_commit),
                        margin_row=float(args.margin_row),
                        margin_defer=float(args.margin_defer),
                        margin_host_edit=float(args.margin_host_edit),
                        edge_focal_alpha=float(args.edge_focal_alpha),
                        edge_focal_gamma=float(args.edge_focal_gamma),
                        score_jitter_std=0.0,
                        grad_clip_norm=float(args.grad_clip_norm),
                        skip_step_grad_norm=float(args.skip_step_grad_norm),
                    )
                gate_calibration = fit_cluster_gate_calibration(
                    val_metrics["cluster_logits"],
                    val_metrics["cluster_targets"],
                    mode=str(args.cluster_gate_calibration),
                )
                calibrated_gate = select_cluster_gate_threshold(
                    gate_calibration["scores"],
                    val_metrics["cluster_targets"],
                    metric=str(args.cluster_gate_select_metric),
                    beta=float(args.cluster_gate_beta),
                    fp_weight=float(args.cluster_gate_fp_weight),
                    search_min=float(args.cluster_gate_search_min),
                    search_max=float(args.cluster_gate_search_max),
                    search_steps=int(args.cluster_gate_search_steps),
                    default_thresh=float(args.cluster_gate_thresh),
                    enable_search=str(args.cluster_gate_calibration) != "none",
                    target_coverage_min=float(args.target_gate_coverage_min),
                    target_coverage_max=float(args.target_gate_coverage_max),
                    coverage_penalty_weight=float(args.coverage_penalty_weight),
                )
                with torch.inference_mode():
                    # Re-evaluate val commit behavior under the calibrated gate, not the raw 0.5 gate.
                    val_metrics_calibrated = _run_epoch(
                        model=model,
                        loader=val_loader,
                        optimizer=None,
                        device=device,
                        cluster_gate_thresh=float(calibrated_gate["thresh"]),
                        cluster_gate_temp=float(gate_calibration["temp"]),
                        cluster_gate_bias=float(gate_calibration["bias"]),
                        cluster_gate_loss_mode=str(args.cluster_gate_loss_mode),
                        cluster_gate_positive_weight=float(args.cluster_gate_positive_weight),
                        cluster_gate_negative_weight=float(args.cluster_gate_negative_weight),
                        assignment_target_key=str(args.assignment_target_key),
                        assignment_row_mask_key=str(args.assignment_row_mask_key),
                        edge_loss_mask_key=str(args.edge_loss_mask_key),
                        edge_target_key=str(args.edge_target_key),
                        cluster_target_key=str(args.cluster_target_key),
                        target_gate_coverage_min=float(args.target_gate_coverage_min),
                        target_gate_coverage_max=float(args.target_gate_coverage_max),
                        coverage_penalty_weight=float(args.coverage_penalty_weight),
                        keep_row_loss_weight=float(args.keep_row_loss_weight),
                        edit_row_loss_weight=float(args.edit_row_loss_weight),
                        edit_edge_positive_weight=float(args.edit_edge_positive_weight),
                        loss_assign_weight=float(args.loss_assign_weight),
                        loss_edge_weight=float(args.loss_edge_weight),
                        loss_cluster_weight=float(args.loss_cluster_weight),
                        loss_margin_weight=float(args.loss_margin_weight),
                        margin_commit=float(args.margin_commit),
                        margin_row=float(args.margin_row),
                        margin_defer=float(args.margin_defer),
                        margin_host_edit=float(args.margin_host_edit),
                        edge_focal_alpha=float(args.edge_focal_alpha),
                        edge_focal_gamma=float(args.edge_focal_gamma),
                        score_jitter_std=0.0,
                        grad_clip_norm=float(args.grad_clip_norm),
                        skip_step_grad_norm=float(args.skip_step_grad_norm),
                    )
                val_metrics_selected = dict(val_metrics)
                val_metrics_selected.update(
                    {
                        "row_acc": float(val_metrics_calibrated["row_acc"]),
                        "commit_precision": float(val_metrics_calibrated["commit_precision"]),
                        "commit_recall": float(val_metrics_calibrated["commit_recall"]),
                        "cluster_precision": float(val_metrics_calibrated["cluster_precision"]),
                        "cluster_recall": float(val_metrics_calibrated["cluster_recall"]),
                        "cluster_f1": float(val_metrics_calibrated["cluster_f1"]),
                        "gate_pass_clusters": float(val_metrics_calibrated["gate_pass_clusters"]),
                        "pred_commits": float(val_metrics_calibrated.get("pred_commits", 0.0)),
                        "target_commits": float(val_metrics_calibrated.get("target_commits", 0.0)),
                        "tp_commits": float(val_metrics_calibrated.get("tp_commits", 0.0)),
                    }
                )

                metrics_row = {
                    "epoch": epoch,
                    "teacher_mode": teacher_mode,
                    "train_loss": train_metrics["loss"],
                    "train_assign_loss": train_metrics["assign_loss"],
                    "train_edge_loss": train_metrics["edge_loss"],
                    "train_cluster_loss": train_metrics["cluster_loss"],
                    "train_margin_loss": train_metrics["margin_loss"],
                    "train_row_acc": train_metrics["row_acc"],
                    "train_commit_precision": train_metrics["commit_precision"],
                    "train_commit_recall": train_metrics["commit_recall"],
                    "train_edge_ap": train_metrics["edge_ap"],
                    "train_cluster_f1": train_metrics["cluster_f1"],
                    "val_loss": val_metrics["loss"],
                    "val_assign_loss": val_metrics["assign_loss"],
                    "val_edge_loss": val_metrics["edge_loss"],
                    "val_cluster_loss": val_metrics["cluster_loss"],
                    "val_margin_loss": val_metrics["margin_loss"],
                    "val_row_acc": val_metrics_selected["row_acc"],
                    "val_commit_precision": val_metrics_selected["commit_precision"],
                    "val_commit_recall": val_metrics_selected["commit_recall"],
                    "val_edge_ap": val_metrics["edge_ap"],
                    "val_cluster_precision": val_metrics_selected["cluster_precision"],
                    "val_cluster_recall": val_metrics_selected["cluster_recall"],
                    "val_cluster_f1": val_metrics_selected["cluster_f1"],
                    "val_cluster_gate_temp": float(gate_calibration["temp"]),
                    "val_cluster_gate_bias": float(gate_calibration["bias"]),
                    "val_cluster_gate_thresh_calibrated": float(calibrated_gate["thresh"]),
                    "val_cluster_gate_precision_cal": float(calibrated_gate["precision"]),
                    "val_cluster_gate_recall_cal": float(calibrated_gate["recall"]),
                    "val_cluster_gate_f0_5": float(calibrated_gate["f_beta"]),
                    "val_cluster_gate_utility_cal": float(calibrated_gate["utility"]),
                    "val_cluster_gate_coverage_cal": float(calibrated_gate["coverage"]),
                    "val_cluster_gate_bounded_utility": float(calibrated_gate["bounded_utility"]),
                    "val_cluster_gate_coverage_penalty": float(calibrated_gate["coverage_penalty"]),
                    "val_selective_utility_targetcov": float(calibrated_gate["bounded_utility"]),
                    "train_skipped_nonfinite_samples": train_metrics["skipped_nonfinite_samples"],
                    "train_skipped_nonfinite_batches": train_metrics["skipped_nonfinite_batches"],
                    "val_skipped_nonfinite_samples": val_metrics["skipped_nonfinite_samples"],
                    "val_skipped_nonfinite_batches": val_metrics["skipped_nonfinite_batches"],
                    "split_mode": split_mode,
                    "split_strategy": str(args.split_strategy),
                    "split_target_key": str(args.split_target_key),
                    "val_fraction": float(args.val_fraction),
                    "train_examples": len(train_rows),
                    "val_examples": len(val_rows),
                    "train_target_positives": _count_positive(train_rows, str(args.split_target_key)),
                    "val_target_positives": _count_positive(val_rows, str(args.split_target_key)),
                    "cluster_gate_thresh": float(args.cluster_gate_thresh),
                    "cluster_gate_loss_mode": str(args.cluster_gate_loss_mode),
                    "cluster_gate_positive_weight": float(args.cluster_gate_positive_weight),
                    "cluster_gate_negative_weight": float(args.cluster_gate_negative_weight),
                    "train_positive_cluster_oversample": float(args.train_positive_cluster_oversample),
                    "model_selection_metric": str(args.model_selection_metric),
                    "assignment_target_key": str(args.assignment_target_key),
                    "assignment_row_mask_key": str(args.assignment_row_mask_key),
                    "edge_loss_mask_key": str(args.edge_loss_mask_key),
                    "edge_target_key": str(args.edge_target_key),
                    "cluster_target_key": str(args.cluster_target_key),
                    "target_gate_coverage_min": float(args.target_gate_coverage_min),
                    "target_gate_coverage_max": float(args.target_gate_coverage_max),
                    "coverage_penalty_weight": float(args.coverage_penalty_weight),
                    "keep_row_loss_weight": float(args.keep_row_loss_weight),
                    "edit_row_loss_weight": float(args.edit_row_loss_weight),
                    "edit_edge_positive_weight": float(args.edit_edge_positive_weight),
                    "margin_host_edit": float(args.margin_host_edit),
                }
                metrics_fp.write(json.dumps(metrics_row) + "\n")
                metrics_fp.flush()

                selection_key = _selection_key(
                    metric_name=str(args.model_selection_metric),
                    val_metrics=val_metrics_selected,
                    calibrated_gate=calibrated_gate,
                )
                if best_row is None:
                    best_val_loss = float(val_metrics["loss"])
                    should_update = True
                else:
                    previous_key = tuple(best_row.get("_selection_key", ()))
                    should_update = selection_key > previous_key
                    if np.isfinite(val_metrics["loss"]) and val_metrics["loss"] < best_val_loss:
                        best_val_loss = float(val_metrics["loss"])
                if should_update:
                    best_val_loss = min(best_val_loss, float(val_metrics["loss"])) if np.isfinite(val_metrics["loss"]) else best_val_loss
                    train_metrics_ckpt = {
                        key: value
                        for key, value in train_metrics.items()
                        if key not in {"cluster_logits", "cluster_targets"}
                    }
                    val_metrics_ckpt = {
                        key: value
                        for key, value in val_metrics_selected.items()
                        if key not in {"cluster_logits", "cluster_targets"}
                    }
                    best_row = {
                        "exp_name": "local_conflict_set_predictor_stage1",
                        "module_family": "set_predictor_v2",
                        "data_jsonl": str(data_jsonl),
                        "source_manifest": str(args.source_manifest or ""),
                        "dataset_tag": str(args.dataset_tag or ""),
                        "feature_version": str(args.feature_version or ""),
                        "teacher_mode": teacher_mode,
                        "checkpoint": str(best_ckpt),
                        "train_examples": int(len(train_rows)),
                        "val_examples": int(len(val_rows)),
                        "train_sequences": ",".join(train_sequences),
                        "val_sequences": ",".join(val_sequences),
                        "train_host_variants": _join_unique(train_rows, "host_variant"),
                        "val_host_variants": _join_unique(val_rows, "host_variant"),
                        "train_target_positives": _count_positive(train_rows, str(args.split_target_key)),
                        "val_target_positives": _count_positive(val_rows, str(args.split_target_key)),
                        "host_vocab": ",".join(host_vocab),
                        "epochs": int(args.epochs),
                        "batch_size": int(args.batch_size),
                        "hidden_dim": int(args.hidden_dim),
                        "num_heads": int(args.num_heads),
                        "num_conflict_blocks": int(args.num_conflict_blocks),
                        "dropout": float(args.dropout),
                        "strict_sequence_split": int(bool(args.strict_sequence_split)),
                        "split_strategy": str(args.split_strategy),
                        "split_target_key": str(args.split_target_key),
                        "val_fraction": float(args.val_fraction),
                        "min_val_examples": int(args.min_val_examples),
                        "cluster_gate_thresh": float(args.cluster_gate_thresh),
                        "cluster_gate_calibration": str(args.cluster_gate_calibration),
                        "cluster_gate_select_metric": str(args.cluster_gate_select_metric),
                        "cluster_gate_beta": float(args.cluster_gate_beta),
                        "cluster_gate_fp_weight": float(args.cluster_gate_fp_weight),
                        "cluster_gate_search_min": float(args.cluster_gate_search_min),
                        "cluster_gate_search_max": float(args.cluster_gate_search_max),
                        "cluster_gate_search_steps": int(args.cluster_gate_search_steps),
                        "cluster_gate_loss_mode": str(args.cluster_gate_loss_mode),
                        "cluster_gate_positive_weight": float(args.cluster_gate_positive_weight),
                        "cluster_gate_negative_weight": float(args.cluster_gate_negative_weight),
                        "train_positive_cluster_oversample": float(args.train_positive_cluster_oversample),
                        "model_selection_metric": str(args.model_selection_metric),
                        "assignment_target_key": str(args.assignment_target_key),
                        "assignment_row_mask_key": str(args.assignment_row_mask_key),
                        "edge_loss_mask_key": str(args.edge_loss_mask_key),
                        "edge_target_key": str(args.edge_target_key),
                        "cluster_target_key": str(args.cluster_target_key),
                        "target_gate_coverage_min": float(args.target_gate_coverage_min),
                        "target_gate_coverage_max": float(args.target_gate_coverage_max),
                        "coverage_penalty_weight": float(args.coverage_penalty_weight),
                        "keep_row_loss_weight": float(args.keep_row_loss_weight),
                        "edit_row_loss_weight": float(args.edit_row_loss_weight),
                        "edit_edge_positive_weight": float(args.edit_edge_positive_weight),
                        "margin_host_edit": float(args.margin_host_edit),
                        "loss_assign_weight": float(args.loss_assign_weight),
                        "loss_edge_weight": float(args.loss_edge_weight),
                        "loss_cluster_weight": float(args.loss_cluster_weight),
                        "loss_margin_weight": float(args.loss_margin_weight),
                        "score_jitter_std": float(args.score_jitter_std),
                        "split_mode": split_mode,
                        "best_epoch": int(epoch),
                        "train_loss": float(train_metrics["loss"]),
                        "val_loss": float(val_metrics["loss"]),
                        "train_row_acc": float(train_metrics["row_acc"]),
                        "val_row_acc": float(val_metrics_selected["row_acc"]),
                        "val_commit_precision": float(val_metrics_selected["commit_precision"]),
                        "val_commit_recall": float(val_metrics_selected["commit_recall"]),
                        "val_edge_ap": float(val_metrics["edge_ap"]),
                        "val_cluster_f1": float(val_metrics_selected["cluster_f1"]),
                        "val_cluster_gate_temp": float(gate_calibration["temp"]),
                        "val_cluster_gate_bias": float(gate_calibration["bias"]),
                        "val_cluster_gate_thresh_calibrated": float(calibrated_gate["thresh"]),
                        "val_cluster_gate_precision_cal": float(calibrated_gate["precision"]),
                        "val_cluster_gate_recall_cal": float(calibrated_gate["recall"]),
                        "val_cluster_gate_f0_5": float(calibrated_gate["f_beta"]),
                        "val_cluster_gate_utility_cal": float(calibrated_gate["utility"]),
                        "val_cluster_gate_coverage_cal": float(calibrated_gate["coverage"]),
                        "val_cluster_gate_bounded_utility": float(calibrated_gate["bounded_utility"]),
                        "val_cluster_gate_coverage_penalty": float(calibrated_gate["coverage_penalty"]),
                        "val_selective_utility_targetcov": float(calibrated_gate["bounded_utility"]),
                        "train_skipped_nonfinite_samples": float(train_metrics["skipped_nonfinite_samples"]),
                        "train_skipped_nonfinite_batches": float(train_metrics["skipped_nonfinite_batches"]),
                        "val_skipped_nonfinite_samples": float(val_metrics_selected["skipped_nonfinite_samples"]),
                        "val_skipped_nonfinite_batches": float(val_metrics_selected["skipped_nonfinite_batches"]),
                        "_selection_key": selection_key,
                        "status": "running",
                        "error": "",
                    }
                    torch.save(
                        model.checkpoint_payload(
                            epoch=int(epoch),
                            train_metrics=train_metrics_ckpt,
                            val_metrics=val_metrics_ckpt,
                            data_jsonl=str(data_jsonl),
                            source_manifest=str(args.source_manifest or ""),
                            dataset_tag=str(args.dataset_tag or ""),
                            feature_version=str(args.feature_version or ""),
                            teacher_mode=teacher_mode,
                            train_examples=int(len(train_rows)),
                            val_examples=int(len(val_rows)),
                            train_sequences=list(train_sequences),
                            val_sequences=list(val_sequences),
                            split_mode=split_mode,
                            split_strategy=str(args.split_strategy),
                            split_target_key=str(args.split_target_key),
                            val_fraction=float(args.val_fraction),
                            train_host_variants=_join_unique(train_rows, "host_variant"),
                            val_host_variants=_join_unique(val_rows, "host_variant"),
                            cluster_gate_thresh=float(args.cluster_gate_thresh),
                            cluster_gate_calibration=str(args.cluster_gate_calibration),
                            cluster_gate_select_metric=str(args.cluster_gate_select_metric),
                            cluster_gate_beta=float(args.cluster_gate_beta),
                            cluster_gate_fp_weight=float(args.cluster_gate_fp_weight),
                            cluster_gate_search_min=float(args.cluster_gate_search_min),
                            cluster_gate_search_max=float(args.cluster_gate_search_max),
                            cluster_gate_search_steps=int(args.cluster_gate_search_steps),
                            cluster_gate_loss_mode=str(args.cluster_gate_loss_mode),
                            cluster_gate_positive_weight=float(args.cluster_gate_positive_weight),
                            cluster_gate_negative_weight=float(args.cluster_gate_negative_weight),
                            train_positive_cluster_oversample=float(args.train_positive_cluster_oversample),
                            model_selection_metric=str(args.model_selection_metric),
                            assignment_target_key=str(args.assignment_target_key),
                            assignment_row_mask_key=str(args.assignment_row_mask_key),
                            edge_loss_mask_key=str(args.edge_loss_mask_key),
                            edge_target_key=str(args.edge_target_key),
                            cluster_target_key=str(args.cluster_target_key),
                            target_gate_coverage_min=float(args.target_gate_coverage_min),
                            target_gate_coverage_max=float(args.target_gate_coverage_max),
                            coverage_penalty_weight=float(args.coverage_penalty_weight),
                            keep_row_loss_weight=float(args.keep_row_loss_weight),
                            edit_row_loss_weight=float(args.edit_row_loss_weight),
                            edit_edge_positive_weight=float(args.edit_edge_positive_weight),
                            margin_host_edit=float(args.margin_host_edit),
                            cluster_gate_temp=float(gate_calibration["temp"]),
                            cluster_gate_bias=float(gate_calibration["bias"]),
                            cluster_gate_thresh_calibrated=float(calibrated_gate["thresh"]),
                            cluster_gate_precision_cal=float(calibrated_gate["precision"]),
                            cluster_gate_recall_cal=float(calibrated_gate["recall"]),
                            cluster_gate_f0_5=float(calibrated_gate["f_beta"]),
                            cluster_gate_utility_cal=float(calibrated_gate["utility"]),
                            cluster_gate_coverage_cal=float(calibrated_gate["coverage"]),
                            cluster_gate_bounded_utility=float(calibrated_gate["bounded_utility"]),
                            cluster_gate_coverage_penalty=float(calibrated_gate["coverage_penalty"]),
                            loss_assign_weight=float(args.loss_assign_weight),
                            loss_edge_weight=float(args.loss_edge_weight),
                            loss_cluster_weight=float(args.loss_cluster_weight),
                            loss_margin_weight=float(args.loss_margin_weight),
                        ),
                        best_ckpt,
                    )
                    _write_single_row_csv(summary_csv, fieldnames, best_row)

        if best_row is None:
            best_row = dict(running_row)
            best_row.update(
                {
                    "train_examples": int(len(train_rows)),
                    "val_examples": int(len(val_rows)),
                    "train_sequences": ",".join(train_sequences),
                    "val_sequences": ",".join(val_sequences),
                    "train_host_variants": _join_unique(train_rows, "host_variant"),
                    "val_host_variants": _join_unique(val_rows, "host_variant"),
                    "train_target_positives": _count_positive(train_rows, str(args.split_target_key)),
                    "val_target_positives": _count_positive(val_rows, str(args.split_target_key)),
                    "host_vocab": ",".join(host_vocab),
                    "teacher_mode": teacher_mode,
                    "split_mode": split_mode,
                    "status": "failed",
                    "error": "no_best_epoch",
                }
            )

        if str(best_row.get("status", "")) != "failed":
            best_row["status"] = "ok"
            best_row["error"] = ""
        if "_selection_key" in best_row:
            best_row.pop("_selection_key", None)

        _write_single_row_csv(result_csv, fieldnames, best_row)
        _write_single_row_csv(summary_csv, fieldnames, best_row)
        return 0 if str(best_row.get("status", "")) == "ok" else 1
    except Exception as exc:
        failed_row = dict(running_row)
        failed_row.update(
            {
                "status": "failed",
                "error": str(exc),
            }
        )
        _write_single_row_csv(result_csv, fieldnames, failed_row)
        _write_single_row_csv(summary_csv, fieldnames, failed_row)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
