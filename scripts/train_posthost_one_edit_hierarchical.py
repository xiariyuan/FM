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
    "val_gate_acc",
    "val_action_type_acc",
    "val_keep_vs_edit_acc",
    "val_swap_action_recall",
    "val_exact_top1_acc",
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
    parser.add_argument("--swap-oversample", type=float, default=8.0)
    parser.add_argument("--gate-positive-weight", type=float, default=2.0)
    parser.add_argument("--swap-selector-weight", type=float, default=8.0)
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
    for row in rows:
        candidate_types = list(row["candidate_action_types"])
        defer_indices = [idx for idx, action in enumerate(candidate_types) if action == "defer"]
        swap_indices = [idx for idx, action in enumerate(candidate_types) if action == "swap"]
        target_type = str(row["target_action_type"])
        prepared.append(
            {
                **row,
                "cluster_features_hier": build_cluster_feature(row),
                "gate_target": int(int(row["target_index"]) > 0),
                "defer_indices": defer_indices,
                "swap_indices": swap_indices,
                "target_type_id": int(ACTION_TYPE_TO_ID.get(target_type, 0)),
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
    targets = []
    for row in batch:
        subset = [row["candidate_features"][idx] for idx in row[subset_key]]
        target_global = int(row["target_index"])
        target_local = int(row[subset_key].index(target_global))
        candidate_lists.append(subset)
        targets.append(target_local)
    max_candidates = max(len(rows) for rows in candidate_lists)
    feat_dim = len(candidate_lists[0][0])
    features = torch.zeros((len(batch), max_candidates, feat_dim), dtype=torch.float32)
    mask = torch.zeros((len(batch), max_candidates), dtype=torch.bool)
    for idx, subset in enumerate(candidate_lists):
        subset_tensor = torch.tensor(subset, dtype=torch.float32)
        count = int(subset_tensor.shape[0])
        features[idx, :count] = subset_tensor
        mask[idx, :count] = True
    return {
        "candidate_features": features,
        "candidate_mask": mask,
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
    target_swap = 0
    correct_swap_action = 0
    for row in rows:
        pred = hierarchical_predict(model, row, device=device)
        target_index = int(row["target_index"])
        target_action = str(row["target_action_type"])
        gate_correct += int(int(pred["gate_pred"]) == int(row["gate_target"]))
        action_type_correct += int(str(pred["pred_action_type"]) == target_action)
        keep_vs_edit_correct += int((int(pred["pred_index"] > 0)) == int(row["gate_target"]))
        exact_top1_correct += int(int(pred["pred_index"]) == target_index)
        if target_action == "swap":
            target_swap += 1
            correct_swap_action += int(str(pred["pred_action_type"]) == "swap")
    return {
        "gate_acc": float(gate_correct / max(total, 1)),
        "action_type_acc": float(action_type_correct / max(total, 1)),
        "keep_vs_edit_acc": float(keep_vs_edit_correct / max(total, 1)),
        "swap_action_recall": float(correct_swap_action / max(target_swap, 1)),
        "exact_top1_acc": float(exact_top1_correct / max(total, 1)),
    }


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

        swap_train = [row for row in train_rows if str(row.get("target_action_type")) == "swap"]
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
        edit_rows_train = [row for row in train_rows if int(row["gate_target"]) > 0]
        action_loader = DataLoader(
            ClusterDataset(edit_rows_train),
            batch_size=int(args.batch_size),
            shuffle=True,
            num_workers=int(args.num_workers),
            collate_fn=collate_cluster_batch,
            drop_last=False,
        )
        defer_rows_train = [row for row in train_rows if str(row.get("target_action_type")) == "defer" and row["defer_indices"]]
        swap_rows_train = [row for row in train_rows if str(row.get("target_action_type")) == "swap" and row["swap_indices"]]

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
                        target = batch["target_index"].to(device)
                        logits = model.defer_ranker(feats).masked_fill(~mask, float("-inf"))
                        loss = nn.functional.cross_entropy(logits, target)
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
                        target = batch["target_index"].to(device)
                        logits = model.swap_ranker(feats).masked_fill(~mask, float("-inf"))
                        loss = nn.functional.cross_entropy(logits, target)
                        swap_opt.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.swap_ranker.parameters(), max_norm=5.0)
                        swap_opt.step()

                train_metrics = evaluate_hierarchical(model, train_rows, device=device)
                val_metrics = evaluate_hierarchical(model, val_rows, device=device)
                current_metric = float(val_metrics["exact_top1_acc"])
                if current_metric > best_metric:
                    best_metric = current_metric
                    best_epoch = int(epoch)
                    best_metrics = {"train": dict(train_metrics), "val": dict(val_metrics)}
                    model.save_checkpoint(
                        checkpoint_path,
                        extra={
                            "epoch": int(epoch),
                            "best_metric_name": "exact_top1_acc",
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
                            "val_gate_acc": float(val_metrics["gate_acc"]),
                            "val_action_type_acc": float(val_metrics["action_type_acc"]),
                            "val_keep_vs_edit_acc": float(val_metrics["keep_vs_edit_acc"]),
                            "val_swap_action_recall": float(val_metrics["swap_action_recall"]),
                            "val_exact_top1_acc": float(val_metrics["exact_top1_acc"]),
                            "best_metric_name": "exact_top1_acc",
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
                "best_metric_name": "exact_top1_acc",
                "best_metric_value": float(best_metric),
                "train_gate_acc": float(best_metrics["train"]["gate_acc"]),
                "train_action_type_acc": float(best_metrics["train"]["action_type_acc"]),
                "train_keep_vs_edit_acc": float(best_metrics["train"]["keep_vs_edit_acc"]),
                "train_swap_action_recall": float(best_metrics["train"]["swap_action_recall"]),
                "train_exact_top1_acc": float(best_metrics["train"]["exact_top1_acc"]),
                "val_gate_acc": float(best_metrics["val"]["gate_acc"]),
                "val_action_type_acc": float(best_metrics["val"]["action_type_acc"]),
                "val_keep_vs_edit_acc": float(best_metrics["val"]["keep_vs_edit_acc"]),
                "val_swap_action_recall": float(best_metrics["val"]["swap_action_recall"]),
                "val_exact_top1_acc": float(best_metrics["val"]["exact_top1_acc"]),
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
