#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path("/gemini/code/FMtrack-main/FM-Track")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.posthost_one_edit_hierarchical import MODEL_FAMILY, PosthostOneEditHierarchical


REGISTRY_CSV = REPO_ROOT / "outputs" / "experiment_registry.csv"
ACTION_TYPE_TO_ID = {"keep": 0, "add": 1, "swap": 2, "defer": 3}


SUMMARY_FIELDS = [
    "run_name",
    "dataset_jsonl",
    "train_examples",
    "val_examples",
    "cluster_feature_dim",
    "candidate_feature_dim",
    "swap_oversample",
    "gate_positive_weight",
    "swap_selector_weight",
    "rank_pairwise_margin",
    "rank_positive_eps",
    "hidden_dim",
    "num_layers",
    "dropout",
    "batch_size",
    "epochs",
    "lr",
    "weight_decay",
    "seed",
    "device",
    "best_epoch",
    "best_metric_name",
    "best_metric_value",
    "train_gate_acc",
    "train_action_type_acc",
    "train_keep_vs_edit_acc",
    "train_swap_action_recall",
    "train_exact_top1_acc",
    "train_nonkeep_exact_top1_acc",
    "train_swap_exact_top1_acc",
    "train_defer_exact_top1_acc",
    "train_positive_utility_gate_recall",
    "train_positive_utility_hit_rate",
    "train_utility_capture",
    "train_mean_pred_adjusted_utility",
    "val_gate_acc",
    "val_action_type_acc",
    "val_keep_vs_edit_acc",
    "val_swap_action_recall",
    "val_exact_top1_acc",
    "val_nonkeep_exact_top1_acc",
    "val_swap_exact_top1_acc",
    "val_defer_exact_top1_acc",
    "val_positive_utility_gate_recall",
    "val_positive_utility_hit_rate",
    "val_utility_capture",
    "val_mean_pred_adjusted_utility",
    "checkpoint",
    "metrics_jsonl",
    "status",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a hierarchical post-host one-edit learner.")
    parser.add_argument("--dataset-jsonl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run-name", default="official_bytetrack_posthost_one_edit_hierarchical")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--swap-oversample", type=float, default=1.0)
    parser.add_argument("--gate-positive-weight", type=float, default=2.0)
    parser.add_argument("--swap-selector-weight", type=float, default=2.0)
    parser.add_argument("--rank-pairwise-margin", type=float, default=0.0)
    parser.add_argument("--rank-positive-eps", type=float, default=1e-6)
    parser.add_argument(
        "--best-metric",
        choices=[
            "exact_top1_acc",
            "nonkeep_exact_top1_acc",
            "swap_exact_top1_acc",
            "swap_focus",
            "balanced_gate_swap",
            "utility_capture",
            "utility_capture_hit_rate",
        ],
        default="utility_capture_hit_rate",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--registry-csv", default=str(REGISTRY_CSV))
    return parser.parse_args()


def write_rows(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def zero_like(row: Sequence[float]) -> list[float]:
    return [0.0 for _ in row]


def mean_pool(rows: Sequence[Sequence[float]], dim: int) -> list[float]:
    if not rows:
        return [0.0 for _ in range(dim)]
    tensor = torch.tensor(rows, dtype=torch.float32)
    return [float(x) for x in tensor.mean(dim=0).tolist()]


def max_pool(rows: Sequence[Sequence[float]], dim: int) -> list[float]:
    if not rows:
        return [0.0 for _ in range(dim)]
    tensor = torch.tensor(rows, dtype=torch.float32)
    return [float(x) for x in tensor.max(dim=0).values.tolist()]


def build_cluster_feature(row: Dict[str, Any]) -> list[float]:
    candidate_features = row["candidate_features"]
    feature_dim = int(len(candidate_features[0]))
    keep_feature = list(candidate_features[0])
    defer_rows = [candidate_features[idx] for idx, action in enumerate(row["candidate_action_types"]) if action == "defer"]
    swap_rows = [candidate_features[idx] for idx, action in enumerate(row["candidate_action_types"]) if action == "swap"]
    add_rows = [candidate_features[idx] for idx, action in enumerate(row["candidate_action_types"]) if action == "add"]
    cluster_feature = (
        keep_feature
        + mean_pool(defer_rows, feature_dim)
        + max_pool(defer_rows, feature_dim)
        + mean_pool(swap_rows, feature_dim)
        + max_pool(swap_rows, feature_dim)
        + [
            float(len(defer_rows)),
            float(len(swap_rows)),
            float(len(add_rows)),
            float(len(candidate_features)),
        ]
    )
    return [float(x) for x in cluster_feature]


def prepare_rows(rows: Sequence[Dict[str, Any]]) -> list[Dict[str, Any]]:
    prepared: list[Dict[str, Any]] = []
    action_priority = {"keep": 0, "defer": 1, "add": 2, "swap": 3}
    for row in rows:
        candidate_types = list(row["candidate_action_types"])
        candidate_raw_utilities = [float(x) for x in row.get("candidate_utility_deltas", [])]
        candidate_adjusted_utilities = [
            float(x)
            for x in row.get("candidate_adjusted_utility_deltas", row.get("candidate_utility_deltas", []))
        ]
        defer_indices = [idx for idx, action in enumerate(candidate_types) if action == "defer"]
        swap_indices = [idx for idx, action in enumerate(candidate_types) if action == "swap"]
        utility_target_index = 0
        best_key = None
        for idx in range(1, len(candidate_types)):
            adjusted_utility = float(candidate_adjusted_utilities[idx])
            raw_utility = float(candidate_raw_utilities[idx]) if idx < len(candidate_raw_utilities) else adjusted_utility
            if adjusted_utility <= 0.0:
                continue
            key = (
                adjusted_utility,
                raw_utility,
                int(action_priority.get(str(candidate_types[idx]), -1)),
            )
            if best_key is None or key > best_key:
                best_key = key
                utility_target_index = int(idx)
        utility_target_type = str(candidate_types[utility_target_index]) if candidate_types else "keep"
        best_positive_nonkeep_adjusted_utility = max(
            0.0,
            max((float(x) for x in candidate_adjusted_utilities[1:]), default=0.0),
        )
        prepared.append(
            {
                **row,
                "cluster_features_hier": build_cluster_feature(row),
                "candidate_utility_deltas": candidate_raw_utilities,
                "candidate_adjusted_utility_deltas": candidate_adjusted_utilities,
                "utility_target_index": int(utility_target_index),
                "utility_target_action_type": str(utility_target_type),
                "best_positive_nonkeep_adjusted_utility": float(best_positive_nonkeep_adjusted_utility),
                "gate_target": int(int(utility_target_index) > 0),
                "defer_indices": defer_indices,
                "swap_indices": swap_indices,
                "target_type_id": int(ACTION_TYPE_TO_ID.get(utility_target_type, 0)),
            }
        )
    return prepared


class ClusterDataset(Dataset):
    def __init__(self, rows: Sequence[Dict[str, Any]]) -> None:
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.rows[idx]


def collate_cluster_batch(batch: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    cluster_features = torch.tensor([row["cluster_features_hier"] for row in batch], dtype=torch.float32)
    gate_target = torch.tensor([int(row["gate_target"]) for row in batch], dtype=torch.float32)
    target_type_id = torch.tensor([int(row["target_type_id"]) for row in batch], dtype=torch.long)
    return {
        "cluster_features": cluster_features,
        "gate_target": gate_target,
        "target_type_id": target_type_id,
    }


def collate_ranker_batch(batch: Sequence[Dict[str, Any]], action: str) -> Dict[str, torch.Tensor]:
    subset_key = f"{action}_indices"
    candidate_lists = []
    utility_lists = []
    targets = []
    for row in batch:
        subset = [row["candidate_features"][idx] for idx in row[subset_key]]
        utilities = [float(row["candidate_adjusted_utility_deltas"][idx]) for idx in row[subset_key]]
        target_global = int(row["utility_target_index"])
        target_local = int(row[subset_key].index(target_global))
        candidate_lists.append(subset)
        utility_lists.append(utilities)
        targets.append(target_local)
    max_candidates = max(len(rows) for rows in candidate_lists)
    feat_dim = len(candidate_lists[0][0])
    features = torch.zeros((len(batch), max_candidates, feat_dim), dtype=torch.float32)
    mask = torch.zeros((len(batch), max_candidates), dtype=torch.bool)
    utilities = torch.zeros((len(batch), max_candidates), dtype=torch.float32)
    for idx, subset in enumerate(candidate_lists):
        subset_tensor = torch.tensor(subset, dtype=torch.float32)
        count = int(subset_tensor.shape[0])
        features[idx, :count] = subset_tensor
        mask[idx, :count] = True
        utilities[idx, :count] = torch.tensor(utility_lists[idx], dtype=torch.float32)
    return {
        "candidate_features": features,
        "candidate_mask": mask,
        "candidate_utilities": utilities,
        "target_index": torch.tensor(targets, dtype=torch.long),
    }


def masked_argmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = logits.masked_fill(~mask.to(dtype=torch.bool), float("-inf"))
    return masked.argmax(dim=-1)


def bce_logits_loss(logits: torch.Tensor, targets: torch.Tensor, positive_weight: float) -> torch.Tensor:
    weight = torch.where(
        targets > 0.5,
        torch.full_like(targets, float(positive_weight)),
        torch.ones_like(targets),
    )
    return nn.functional.binary_cross_entropy_with_logits(logits, targets, weight=weight)


def utility_pairwise_ranking_loss(
    logits: torch.Tensor,
    utilities: torch.Tensor,
    mask: torch.Tensor,
    *,
    margin: float,
    positive_eps: float,
) -> torch.Tensor:
    total_loss = logits.sum() * 0.0
    pair_groups = 0
    valid_mask = mask.to(dtype=torch.bool)
    for row_logits, row_utilities, row_mask in zip(logits, utilities, valid_mask):
        active_logits = row_logits[row_mask]
        active_utilities = row_utilities[row_mask]
        if int(active_logits.numel()) <= 1:
            continue
        utility_diff = active_utilities.unsqueeze(1) - active_utilities.unsqueeze(0)
        pair_mask = utility_diff > float(positive_eps)
        if not bool(pair_mask.any().item()):
            continue
        score_diff = active_logits.unsqueeze(1) - active_logits.unsqueeze(0)
        pair_weights = utility_diff[pair_mask].detach().abs().clamp_min(float(positive_eps))
        pair_loss = nn.functional.softplus(-(score_diff[pair_mask] - float(margin)))
        total_loss = total_loss + (pair_loss * pair_weights).sum() / pair_weights.sum()
        pair_groups += 1
    if pair_groups <= 0:
        return total_loss
    return total_loss / float(pair_groups)


def load_rows(path: Path) -> list[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


@torch.no_grad()
def hierarchical_predict(
    model: PosthostOneEditHierarchical,
    row: Dict[str, Any],
    *,
    device: torch.device,
) -> Dict[str, Any]:
    cluster_features = torch.tensor(row["cluster_features_hier"], dtype=torch.float32, device=device).unsqueeze(0)
    gate_logit = model.keep_edit_gate(cluster_features)
    gate_pred = int(torch.sigmoid(gate_logit).item() >= 0.5)
    if gate_pred <= 0:
        pred_index = 0
        pred_action = "keep"
    else:
        action_logit = model.defer_swap_selector(cluster_features)
        action_is_swap = int(torch.sigmoid(action_logit).item() >= 0.5)
        if action_is_swap > 0 and row["swap_indices"]:
            candidate_features = torch.tensor(
                [row["candidate_features"][idx] for idx in row["swap_indices"]],
                dtype=torch.float32,
                device=device,
            )
            local_pred = int(model.swap_ranker(candidate_features).argmax().item())
            pred_index = int(row["swap_indices"][local_pred])
            pred_action = "swap"
        elif row["defer_indices"]:
            candidate_features = torch.tensor(
                [row["candidate_features"][idx] for idx in row["defer_indices"]],
                dtype=torch.float32,
                device=device,
            )
            local_pred = int(model.defer_ranker(candidate_features).argmax().item())
            pred_index = int(row["defer_indices"][local_pred])
            pred_action = "defer"
        elif row["swap_indices"]:
            candidate_features = torch.tensor(
                [row["candidate_features"][idx] for idx in row["swap_indices"]],
                dtype=torch.float32,
                device=device,
            )
            local_pred = int(model.swap_ranker(candidate_features).argmax().item())
            pred_index = int(row["swap_indices"][local_pred])
            pred_action = "swap"
        else:
            pred_index = 0
            pred_action = "keep"
    return {
        "pred_index": int(pred_index),
        "pred_action_type": str(pred_action),
        "gate_pred": int(gate_pred),
    }


@torch.no_grad()
def evaluate_hierarchical(
    model: PosthostOneEditHierarchical,
    rows: Sequence[Dict[str, Any]],
    *,
    device: torch.device,
) -> Dict[str, float]:
    total = len(rows)
    gate_correct = 0
    action_type_correct = 0
    keep_vs_edit_correct = 0
    exact_top1_correct = 0
    nonkeep_total = 0
    nonkeep_exact_top1_correct = 0
    target_swap = 0
    correct_swap_action = 0
    swap_exact_top1_correct = 0
    target_defer = 0
    defer_exact_top1_correct = 0
    positive_utility_total = 0
    positive_utility_gate_recall = 0
    positive_utility_hit = 0
    utility_capture_numer = 0.0
    utility_capture_denom = 0.0
    pred_adjusted_utility_sum = 0.0
    for row in rows:
        pred = hierarchical_predict(model, row, device=device)
        target_index = int(row["utility_target_index"])
        target_action = str(row["utility_target_action_type"])
        pred_adjusted_utility = float(row["candidate_adjusted_utility_deltas"][int(pred["pred_index"])])
        pred_adjusted_utility_sum += pred_adjusted_utility
        best_positive_utility = float(row.get("best_positive_nonkeep_adjusted_utility", 0.0))
        gate_correct += int(int(pred["gate_pred"]) == int(row["gate_target"]))
        action_type_correct += int(str(pred["pred_action_type"]) == target_action)
        keep_vs_edit_correct += int((int(pred["pred_index"] > 0)) == int(row["gate_target"]))
        exact_top1_correct += int(int(pred["pred_index"]) == target_index)
        if target_index > 0:
            nonkeep_total += 1
            nonkeep_exact_top1_correct += int(int(pred["pred_index"]) == target_index)
        if target_action == "swap":
            target_swap += 1
            correct_swap_action += int(str(pred["pred_action_type"]) == "swap")
            swap_exact_top1_correct += int(int(pred["pred_index"]) == target_index)
        if target_action == "defer":
            target_defer += 1
            defer_exact_top1_correct += int(int(pred["pred_index"]) == target_index)
        if best_positive_utility > 0.0:
            positive_utility_total += 1
            positive_utility_gate_recall += int(int(pred["gate_pred"]) > 0)
            positive_utility_hit += int(int(pred["pred_index"]) > 0 and pred_adjusted_utility > 0.0)
            utility_capture_numer += float(max(pred_adjusted_utility, 0.0))
            utility_capture_denom += float(best_positive_utility)
    return {
        "gate_acc": float(gate_correct / max(total, 1)),
        "action_type_acc": float(action_type_correct / max(total, 1)),
        "keep_vs_edit_acc": float(keep_vs_edit_correct / max(total, 1)),
        "swap_action_recall": float(correct_swap_action / max(target_swap, 1)),
        "exact_top1_acc": float(exact_top1_correct / max(total, 1)),
        "nonkeep_exact_top1_acc": float(nonkeep_exact_top1_correct / max(nonkeep_total, 1)),
        "swap_exact_top1_acc": float(swap_exact_top1_correct / max(target_swap, 1)),
        "defer_exact_top1_acc": float(defer_exact_top1_correct / max(target_defer, 1)),
        "positive_utility_gate_recall": float(positive_utility_gate_recall / max(positive_utility_total, 1)),
        "positive_utility_hit_rate": float(positive_utility_hit / max(positive_utility_total, 1)),
        "utility_capture": float(utility_capture_numer / max(utility_capture_denom, 1e-6)),
        "mean_pred_adjusted_utility": float(pred_adjusted_utility_sum / max(total, 1)),
    }


def select_metric_value(metrics: Dict[str, float], *, best_metric: str) -> float:
    name = str(best_metric)
    if name == "exact_top1_acc":
        return float(metrics["exact_top1_acc"])
    if name == "nonkeep_exact_top1_acc":
        return float(metrics["nonkeep_exact_top1_acc"])
    if name == "swap_exact_top1_acc":
        return float(metrics["swap_exact_top1_acc"])
    if name == "swap_focus":
        return float(
            0.50 * float(metrics["swap_exact_top1_acc"])
            + 0.35 * float(metrics["nonkeep_exact_top1_acc"])
            + 0.15 * float(metrics["swap_action_recall"])
        )
    if name == "balanced_gate_swap":
        # Online behavior depends on both opening the right clusters and ranking swaps safely.
        return float(
            0.40 * float(metrics["gate_acc"])
            + 0.30 * float(metrics["swap_exact_top1_acc"])
            + 0.20 * float(metrics["nonkeep_exact_top1_acc"])
            + 0.10 * float(metrics["swap_action_recall"])
        )
    if name == "utility_capture":
        return float(metrics["utility_capture"])
    if name == "utility_capture_hit_rate":
        return float(
            0.70 * float(metrics["utility_capture"])
            + 0.30 * float(metrics["positive_utility_hit_rate"])
        )
    raise ValueError(f"Unsupported best metric: {best_metric}")


def append_registry(args: argparse.Namespace, *, out_dir: Path, checkpoint: Path, status: str) -> None:
    import subprocess

    cmd = [
        "python",
        str(REPO_ROOT / "scripts" / "append_experiment_record.py"),
        "--csv",
        str(Path(args.registry_csv).resolve()),
        "--kind",
        "train",
        "--status",
        status,
        "--script",
        "scripts/train_posthost_one_edit_hierarchical.py",
        "--dataset",
        "MOT17",
        "--split",
        "official_trainhalf_manifest_split",
        "--tracker-family",
        "official_bytetrack",
        "--variant",
        "posthost_one_edit_hierarchical",
        "--tag",
        args.run_name,
        "--run-root",
        str(out_dir.resolve()),
        "--summary-csv",
        str((out_dir / "summary.csv").resolve()),
        "--checkpoint",
        str(checkpoint.resolve()) if checkpoint.is_file() else "",
        "--notes",
        "offline train hierarchical post-host one-edit learner",
        "--extra",
        f"dataset_jsonl={str(Path(args.dataset_jsonl).resolve())}",
        f"model_family={MODEL_FAMILY}",
        f"swap_oversample={float(args.swap_oversample)}",
        f"gate_positive_weight={float(args.gate_positive_weight)}",
        f"swap_selector_weight={float(args.swap_selector_weight)}",
        f"rank_pairwise_margin={float(args.rank_pairwise_margin)}",
        f"rank_positive_eps={float(args.rank_positive_eps)}",
        f"best_metric={str(args.best_metric)}",
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(args.seed))

    metrics_jsonl = out_dir / "metrics.jsonl"
    checkpoint_path = out_dir / "best.pt"
    summary_row: Dict[str, Any] = {
        "run_name": args.run_name,
        "dataset_jsonl": str(Path(args.dataset_jsonl).resolve()),
        "swap_oversample": float(args.swap_oversample),
        "gate_positive_weight": float(args.gate_positive_weight),
        "swap_selector_weight": float(args.swap_selector_weight),
        "rank_pairwise_margin": float(args.rank_pairwise_margin),
        "rank_positive_eps": float(args.rank_positive_eps),
        "hidden_dim": int(args.hidden_dim),
        "num_layers": int(args.num_layers),
        "dropout": float(args.dropout),
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "seed": int(args.seed),
        "checkpoint": str(checkpoint_path.resolve()),
        "metrics_jsonl": str(metrics_jsonl.resolve()),
        "status": "running",
        "error": "",
    }
    write_rows(out_dir / "summary.csv", SUMMARY_FIELDS, [summary_row])
    write_rows(out_dir / "result.csv", SUMMARY_FIELDS, [summary_row])

    try:
        rows = prepare_rows(load_rows(Path(args.dataset_jsonl).resolve()))
        train_rows = [row for row in rows if str(row.get("split_tag", "train")) == "train"]
        val_rows = [row for row in rows if str(row.get("split_tag", "train")) == "val"]
        if not train_rows or not val_rows:
            raise RuntimeError("Need non-empty train and val splits")

        swap_train = [row for row in train_rows if str(row.get("utility_target_action_type")) == "swap"]
        swap_repeat = max(int(math.ceil(float(args.swap_oversample))) - 1, 0)
        if swap_repeat > 0 and swap_train:
            train_rows = list(train_rows) + swap_train * swap_repeat

        cluster_feature_dim = len(train_rows[0]["cluster_features_hier"])
        candidate_feature_dim = len(train_rows[0]["candidate_features"][0])
        summary_row.update(
            {
                "train_examples": int(len(train_rows)),
                "val_examples": int(len(val_rows)),
                "cluster_feature_dim": int(cluster_feature_dim),
                "candidate_feature_dim": int(candidate_feature_dim),
                "device": "cuda" if torch.cuda.is_available() else "cpu",
            }
        )
        write_rows(out_dir / "summary.csv", SUMMARY_FIELDS, [summary_row])
        write_rows(out_dir / "result.csv", SUMMARY_FIELDS, [summary_row])

        device = torch.device(summary_row["device"])
        model = PosthostOneEditHierarchical(
            cluster_feature_dim=int(cluster_feature_dim),
            candidate_feature_dim=int(candidate_feature_dim),
            hidden_dim=int(args.hidden_dim),
            num_layers=int(args.num_layers),
            dropout=float(args.dropout),
        ).to(device)

        gate_opt = torch.optim.AdamW(model.keep_edit_gate.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
        action_opt = torch.optim.AdamW(model.defer_swap_selector.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
        defer_opt = torch.optim.AdamW(model.defer_ranker.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
        swap_opt = torch.optim.AdamW(model.swap_ranker.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

        cluster_loader = DataLoader(
            ClusterDataset(train_rows),
            batch_size=int(args.batch_size),
            shuffle=True,
            num_workers=int(args.num_workers),
            collate_fn=collate_cluster_batch,
            drop_last=False,
        )
        edit_rows_train = [
            row
            for row in train_rows
            if int(row["gate_target"]) > 0 and str(row.get("utility_target_action_type")) in {"defer", "swap"}
        ]
        action_loader = (
            DataLoader(
                ClusterDataset(edit_rows_train),
                batch_size=int(args.batch_size),
                shuffle=True,
                num_workers=int(args.num_workers),
                collate_fn=collate_cluster_batch,
                drop_last=False,
            )
            if edit_rows_train
            else None
        )
        defer_rows_train = [
            row
            for row in train_rows
            if str(row.get("utility_target_action_type")) == "defer" and row["defer_indices"]
        ]
        swap_rows_train = [
            row
            for row in train_rows
            if str(row.get("utility_target_action_type")) == "swap" and row["swap_indices"]
        ]

        best_epoch = -1
        best_metric = float("-inf")
        best_metrics: Dict[str, float] | None = None
        metrics_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with metrics_jsonl.open("w", encoding="utf-8") as metrics_fp:
            for epoch in range(1, int(args.epochs) + 1):
                model.train()

                for batch in cluster_loader:
                    feats = batch["cluster_features"].to(device)
                    gate_target = batch["gate_target"].to(device)
                    logits = model.keep_edit_gate(feats)
                    loss = bce_logits_loss(logits, gate_target, positive_weight=float(args.gate_positive_weight))
                    gate_opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.keep_edit_gate.parameters(), max_norm=5.0)
                    gate_opt.step()

                if action_loader is not None:
                    for batch in action_loader:
                        feats = batch["cluster_features"].to(device)
                        target_type_id = batch["target_type_id"].to(device)
                        action_target = (target_type_id == 2).to(dtype=torch.float32)
                        logits = model.defer_swap_selector(feats)
                        loss = bce_logits_loss(logits, action_target, positive_weight=float(args.swap_selector_weight))
                        action_opt.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.defer_swap_selector.parameters(), max_norm=5.0)
                        action_opt.step()

                if defer_rows_train:
                    random.shuffle(defer_rows_train)
                    for start in range(0, len(defer_rows_train), int(args.batch_size)):
                        batch_rows = defer_rows_train[start : start + int(args.batch_size)]
                        batch = collate_ranker_batch(batch_rows, action="defer")
                        feats = batch["candidate_features"].to(device)
                        mask = batch["candidate_mask"].to(device)
                        utilities = batch["candidate_utilities"].to(device)
                        logits = model.defer_ranker(feats)
                        loss = utility_pairwise_ranking_loss(
                            logits,
                            utilities,
                            mask,
                            margin=float(args.rank_pairwise_margin),
                            positive_eps=float(args.rank_positive_eps),
                        )
                        defer_opt.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.defer_ranker.parameters(), max_norm=5.0)
                        defer_opt.step()

                if swap_rows_train:
                    random.shuffle(swap_rows_train)
                    for start in range(0, len(swap_rows_train), int(args.batch_size)):
                        batch_rows = swap_rows_train[start : start + int(args.batch_size)]
                        batch = collate_ranker_batch(batch_rows, action="swap")
                        feats = batch["candidate_features"].to(device)
                        mask = batch["candidate_mask"].to(device)
                        utilities = batch["candidate_utilities"].to(device)
                        logits = model.swap_ranker(feats)
                        loss = utility_pairwise_ranking_loss(
                            logits,
                            utilities,
                            mask,
                            margin=float(args.rank_pairwise_margin),
                            positive_eps=float(args.rank_positive_eps),
                        )
                        swap_opt.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.swap_ranker.parameters(), max_norm=5.0)
                        swap_opt.step()

                train_metrics = evaluate_hierarchical(model, train_rows, device=device)
                val_metrics = evaluate_hierarchical(model, val_rows, device=device)
                current_metric = select_metric_value(val_metrics, best_metric=str(args.best_metric))
                if current_metric > best_metric:
                    best_metric = current_metric
                    best_epoch = int(epoch)
                    best_metrics = {"train": dict(train_metrics), "val": dict(val_metrics)}
                    model.save_checkpoint(
                        checkpoint_path,
                        extra={
                            "epoch": int(epoch),
                            "best_metric_name": str(args.best_metric),
                            "best_metric_value": float(best_metric),
                            "train_metrics": dict(train_metrics),
                            "val_metrics": dict(val_metrics),
                        },
                    )

                metrics_fp.write(
                    json.dumps(
                        {
                            "epoch": int(epoch),
                            "train_gate_acc": float(train_metrics["gate_acc"]),
                            "train_action_type_acc": float(train_metrics["action_type_acc"]),
                            "train_keep_vs_edit_acc": float(train_metrics["keep_vs_edit_acc"]),
                            "train_swap_action_recall": float(train_metrics["swap_action_recall"]),
                            "train_exact_top1_acc": float(train_metrics["exact_top1_acc"]),
                            "train_nonkeep_exact_top1_acc": float(train_metrics["nonkeep_exact_top1_acc"]),
                            "train_swap_exact_top1_acc": float(train_metrics["swap_exact_top1_acc"]),
                            "train_defer_exact_top1_acc": float(train_metrics["defer_exact_top1_acc"]),
                            "train_positive_utility_gate_recall": float(train_metrics["positive_utility_gate_recall"]),
                            "train_positive_utility_hit_rate": float(train_metrics["positive_utility_hit_rate"]),
                            "train_utility_capture": float(train_metrics["utility_capture"]),
                            "train_mean_pred_adjusted_utility": float(train_metrics["mean_pred_adjusted_utility"]),
                            "val_gate_acc": float(val_metrics["gate_acc"]),
                            "val_action_type_acc": float(val_metrics["action_type_acc"]),
                            "val_keep_vs_edit_acc": float(val_metrics["keep_vs_edit_acc"]),
                            "val_swap_action_recall": float(val_metrics["swap_action_recall"]),
                            "val_exact_top1_acc": float(val_metrics["exact_top1_acc"]),
                            "val_nonkeep_exact_top1_acc": float(val_metrics["nonkeep_exact_top1_acc"]),
                            "val_swap_exact_top1_acc": float(val_metrics["swap_exact_top1_acc"]),
                            "val_defer_exact_top1_acc": float(val_metrics["defer_exact_top1_acc"]),
                            "val_positive_utility_gate_recall": float(val_metrics["positive_utility_gate_recall"]),
                            "val_positive_utility_hit_rate": float(val_metrics["positive_utility_hit_rate"]),
                            "val_utility_capture": float(val_metrics["utility_capture"]),
                            "val_mean_pred_adjusted_utility": float(val_metrics["mean_pred_adjusted_utility"]),
                            "best_metric_name": str(args.best_metric),
                            "best_metric_value_so_far": float(best_metric),
                            "is_best": int(epoch == best_epoch),
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )
                metrics_fp.flush()

        if best_metrics is None:
            raise RuntimeError("No best hierarchical checkpoint produced")

        summary_row.update(
            {
                "best_epoch": int(best_epoch),
                "best_metric_name": str(args.best_metric),
                "best_metric_value": float(best_metric),
                "train_gate_acc": float(best_metrics["train"]["gate_acc"]),
                "train_action_type_acc": float(best_metrics["train"]["action_type_acc"]),
                "train_keep_vs_edit_acc": float(best_metrics["train"]["keep_vs_edit_acc"]),
                "train_swap_action_recall": float(best_metrics["train"]["swap_action_recall"]),
                "train_exact_top1_acc": float(best_metrics["train"]["exact_top1_acc"]),
                "train_nonkeep_exact_top1_acc": float(best_metrics["train"]["nonkeep_exact_top1_acc"]),
                "train_swap_exact_top1_acc": float(best_metrics["train"]["swap_exact_top1_acc"]),
                "train_defer_exact_top1_acc": float(best_metrics["train"]["defer_exact_top1_acc"]),
                "train_positive_utility_gate_recall": float(best_metrics["train"]["positive_utility_gate_recall"]),
                "train_positive_utility_hit_rate": float(best_metrics["train"]["positive_utility_hit_rate"]),
                "train_utility_capture": float(best_metrics["train"]["utility_capture"]),
                "train_mean_pred_adjusted_utility": float(best_metrics["train"]["mean_pred_adjusted_utility"]),
                "val_gate_acc": float(best_metrics["val"]["gate_acc"]),
                "val_action_type_acc": float(best_metrics["val"]["action_type_acc"]),
                "val_keep_vs_edit_acc": float(best_metrics["val"]["keep_vs_edit_acc"]),
                "val_swap_action_recall": float(best_metrics["val"]["swap_action_recall"]),
                "val_exact_top1_acc": float(best_metrics["val"]["exact_top1_acc"]),
                "val_nonkeep_exact_top1_acc": float(best_metrics["val"]["nonkeep_exact_top1_acc"]),
                "val_swap_exact_top1_acc": float(best_metrics["val"]["swap_exact_top1_acc"]),
                "val_defer_exact_top1_acc": float(best_metrics["val"]["defer_exact_top1_acc"]),
                "val_positive_utility_gate_recall": float(best_metrics["val"]["positive_utility_gate_recall"]),
                "val_positive_utility_hit_rate": float(best_metrics["val"]["positive_utility_hit_rate"]),
                "val_utility_capture": float(best_metrics["val"]["utility_capture"]),
                "val_mean_pred_adjusted_utility": float(best_metrics["val"]["mean_pred_adjusted_utility"]),
                "status": "success",
                "error": "",
            }
        )
        write_rows(out_dir / "summary.csv", SUMMARY_FIELDS, [summary_row])
        write_rows(out_dir / "result.csv", SUMMARY_FIELDS, [summary_row])
        append_registry(args, out_dir=out_dir, checkpoint=checkpoint_path, status="success")
        return 0
    except Exception as exc:
        summary_row.update({"status": "failed", "error": str(exc)})
        write_rows(out_dir / "summary.csv", SUMMARY_FIELDS, [summary_row])
        write_rows(out_dir / "result.csv", SUMMARY_FIELDS, [summary_row])
        append_registry(args, out_dir=out_dir, checkpoint=checkpoint_path, status="failed")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
