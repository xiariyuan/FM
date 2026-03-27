#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path("/gemini/code/FMtrack-main/FM-Track")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.posthost_one_edit_scorer import (
    MODEL_FAMILY,
    CANDIDATE_FEATURE_NAMES,
    PosthostOneEditScorer,
    masked_argmax,
)


REGISTRY_CSV = REPO_ROOT / "outputs" / "experiment_registry.csv"


SUMMARY_FIELDS = [
    "run_name",
    "dataset_jsonl",
    "train_examples",
    "val_examples",
    "candidate_feature_dim",
    "positive_cluster_oversample",
    "positive_cluster_weight",
    "swap_cluster_oversample",
    "swap_cluster_weight",
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
    "train_loss",
    "train_top1_acc",
    "train_action_type_acc",
    "train_keep_vs_edit_acc",
    "train_swap_action_recall",
    "train_nonkeep_precision",
    "train_nonkeep_recall",
    "train_nonkeep_coverage",
    "train_nonkeep_f0_5",
    "val_loss",
    "val_top1_acc",
    "val_action_type_acc",
    "val_keep_vs_edit_acc",
    "val_swap_action_recall",
    "val_nonkeep_precision",
    "val_nonkeep_recall",
    "val_nonkeep_coverage",
    "val_nonkeep_f0_5",
    "val_target_keep_clusters",
    "val_target_nonkeep_clusters",
    "checkpoint",
    "metrics_jsonl",
    "status",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a standalone post-host one-edit action scorer.")
    parser.add_argument("--dataset-jsonl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run-name", default="official_bytetrack_posthost_one_edit_scorer")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--positive-cluster-oversample", type=float, default=4.0)
    parser.add_argument("--positive-cluster-weight", type=float, default=4.0)
    parser.add_argument("--swap-cluster-oversample", type=float, default=1.0)
    parser.add_argument("--swap-cluster-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--registry-csv", default=str(REGISTRY_CSV))
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def write_rows(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


class ActionDataset(Dataset):
    def __init__(self, rows: Sequence[Dict[str, Any]]) -> None:
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.rows[idx]


def collate_batch(batch: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    max_candidates = max(int(len(row["candidate_features"])) for row in batch)
    feat_dim = len(batch[0]["candidate_features"][0])
    features = torch.zeros((len(batch), max_candidates, feat_dim), dtype=torch.float32)
    mask = torch.zeros((len(batch), max_candidates), dtype=torch.bool)
    targets = torch.zeros((len(batch),), dtype=torch.long)
    target_is_nonkeep = torch.zeros((len(batch),), dtype=torch.bool)
    candidate_action_type_id = torch.zeros((len(batch), max_candidates), dtype=torch.long)
    target_action_type_id = torch.zeros((len(batch),), dtype=torch.long)
    action_vocab = {"keep": 0, "add": 1, "swap": 2, "defer": 3}
    for idx, row in enumerate(batch):
        candidate_features = torch.tensor(row["candidate_features"], dtype=torch.float32)
        count = int(candidate_features.shape[0])
        features[idx, :count] = candidate_features
        mask[idx, :count] = True
        targets[idx] = int(row["target_index"])
        target_is_nonkeep[idx] = bool(int(row.get("target_is_nonkeep", 0)))
        target_action_type_id[idx] = int(action_vocab.get(str(row.get("target_action_type", "keep")), 0))
        type_ids = [int(action_vocab.get(str(action_type), 0)) for action_type in row["candidate_action_types"]]
        candidate_action_type_id[idx, :count] = torch.tensor(type_ids, dtype=torch.long)
    return {
        "candidate_features": features,
        "candidate_mask": mask,
        "target_index": targets,
        "target_is_nonkeep": target_is_nonkeep,
        "candidate_action_type_id": candidate_action_type_id,
        "target_action_type_id": target_action_type_id,
    }


def nonkeep_f_beta(precision: float, recall: float, *, beta: float = 0.5) -> float:
    if precision <= 0.0 or recall <= 0.0:
        return 0.0
    beta_sq = float(beta * beta)
    denom = beta_sq * precision + recall
    if denom <= 0.0:
        return 0.0
    return float((1.0 + beta_sq) * precision * recall / denom)


@torch.no_grad()
def evaluate(
    model: PosthostOneEditScorer,
    loader: DataLoader,
    *,
    device: torch.device,
    positive_cluster_weight: float,
    swap_cluster_weight: float,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    total_correct = 0
    total_action_type_correct = 0
    total_keep_vs_edit_correct = 0
    total_pred_nonkeep = 0
    total_target_nonkeep = 0
    total_correct_nonkeep = 0
    total_target_swap = 0
    total_correct_swap_action = 0
    for batch in loader:
        features = batch["candidate_features"].to(device)
        mask = batch["candidate_mask"].to(device)
        targets = batch["target_index"].to(device)
        target_is_nonkeep = batch["target_is_nonkeep"].to(device)
        candidate_action_type_id = batch["candidate_action_type_id"].to(device)
        target_action_type_id = batch["target_action_type_id"].to(device)

        logits = model(features)
        logits = logits.masked_fill(~mask, float("-inf"))
        per_sample_loss = nn.functional.cross_entropy(logits, targets, reduction="none")
        sample_weight = torch.ones_like(per_sample_loss)
        sample_weight = torch.where(
            target_is_nonkeep,
            torch.full_like(per_sample_loss, float(positive_cluster_weight)),
            sample_weight,
        )
        sample_weight = torch.where(
            target_action_type_id == 2,
            torch.full_like(per_sample_loss, float(max(swap_cluster_weight, 1.0))),
            sample_weight,
        )
        loss = (per_sample_loss * sample_weight).sum() / sample_weight.sum().clamp(min=1.0)

        pred = masked_argmax(logits, mask)
        pred_action_type_id = candidate_action_type_id.gather(1, pred.view(-1, 1)).view(-1)
        total_examples += int(targets.numel())
        total_loss += float(loss.item()) * float(targets.numel())
        total_correct += int((pred == targets).sum().item())
        total_action_type_correct += int((pred_action_type_id == target_action_type_id).sum().item())
        total_keep_vs_edit_correct += int(((pred != 0) == target_is_nonkeep.to(dtype=torch.bool)).sum().item())

        pred_nonkeep = pred != 0
        target_nonkeep_bool = target_is_nonkeep.to(dtype=torch.bool)
        total_pred_nonkeep += int(pred_nonkeep.sum().item())
        total_target_nonkeep += int(target_nonkeep_bool.sum().item())
        total_correct_nonkeep += int(((pred == targets) & target_nonkeep_bool).sum().item())
        total_target_swap += int((target_action_type_id == 2).sum().item())
        total_correct_swap_action += int(((pred_action_type_id == 2) & (target_action_type_id == 2)).sum().item())

    top1_acc = float(total_correct / max(total_examples, 1))
    nonkeep_precision = float(total_correct_nonkeep / max(total_pred_nonkeep, 1))
    nonkeep_recall = float(total_correct_nonkeep / max(total_target_nonkeep, 1))
    nonkeep_coverage = float(total_pred_nonkeep / max(total_examples, 1))
    return {
        "loss": float(total_loss / max(total_examples, 1)),
        "top1_acc": top1_acc,
        "action_type_acc": float(total_action_type_correct / max(total_examples, 1)),
        "keep_vs_edit_acc": float(total_keep_vs_edit_correct / max(total_examples, 1)),
        "swap_action_recall": float(total_correct_swap_action / max(total_target_swap, 1)),
        "nonkeep_precision": nonkeep_precision,
        "nonkeep_recall": nonkeep_recall,
        "nonkeep_coverage": nonkeep_coverage,
        "nonkeep_f0_5": nonkeep_f_beta(nonkeep_precision, nonkeep_recall, beta=0.5),
        "target_keep_clusters": int(total_examples - total_target_nonkeep),
        "target_nonkeep_clusters": int(total_target_nonkeep),
    }


def metric_value(metrics: Dict[str, float], metric_name: str) -> float:
    if metric_name == "nonkeep_f0.5":
        return float(metrics["nonkeep_f0_5"])
    if metric_name == "top1_acc":
        return float(metrics["top1_acc"])
    return float(metrics["nonkeep_f0_5"])


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
        "scripts/train_posthost_one_edit_scorer.py",
        "--dataset",
        "MOT17",
        "--split",
        "official_trainhalf_manifest_split",
        "--tracker-family",
        "official_bytetrack",
        "--variant",
        "posthost_one_edit_scorer",
        "--tag",
        args.run_name,
        "--run-root",
        str(out_dir.resolve()),
        "--summary-csv",
        str((out_dir / "summary.csv").resolve()),
        "--checkpoint",
        str(checkpoint.resolve()) if checkpoint.is_file() else "",
        "--notes",
        "offline train standalone post-host one-edit action scorer",
        "--extra",
        f"dataset_jsonl={str(Path(args.dataset_jsonl).resolve())}",
        f"model_family={MODEL_FAMILY}",
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
        "candidate_feature_dim": int(len(CANDIDATE_FEATURE_NAMES)),
        "positive_cluster_oversample": float(args.positive_cluster_oversample),
        "positive_cluster_weight": float(args.positive_cluster_weight),
        "swap_cluster_oversample": float(args.swap_cluster_oversample),
        "swap_cluster_weight": float(args.swap_cluster_weight),
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
        rows: List[Dict[str, Any]] = []
        with Path(args.dataset_jsonl).resolve().open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        train_rows = [row for row in rows if str(row.get("split_tag", "train")) == "train"]
        val_rows = [row for row in rows if str(row.get("split_tag", "train")) == "val"]
        if not train_rows or not val_rows:
            raise RuntimeError("Need non-empty train and val splits in dataset_jsonl")

        positive_train = [row for row in train_rows if int(row.get("target_is_nonkeep", 0)) == 1]
        swap_train = [row for row in train_rows if str(row.get("target_action_type", "keep")) == "swap"]
        repeat_count = max(int(math.ceil(float(args.positive_cluster_oversample))) - 1, 0)
        if positive_train and repeat_count > 0:
            train_rows = list(train_rows) + positive_train * repeat_count
        swap_repeat_count = max(int(math.ceil(float(args.swap_cluster_oversample))) - 1, 0)
        if swap_train and swap_repeat_count > 0:
            train_rows = list(train_rows) + swap_train * swap_repeat_count

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        summary_row["device"] = str(device)
        summary_row["train_examples"] = int(len(train_rows))
        summary_row["val_examples"] = int(len(val_rows))
        write_rows(out_dir / "summary.csv", SUMMARY_FIELDS, [summary_row])
        write_rows(out_dir / "result.csv", SUMMARY_FIELDS, [summary_row])

        train_loader = DataLoader(
            ActionDataset(train_rows),
            batch_size=int(args.batch_size),
            shuffle=True,
            num_workers=int(args.num_workers),
            collate_fn=collate_batch,
            drop_last=False,
        )
        val_loader = DataLoader(
            ActionDataset(val_rows),
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            collate_fn=collate_batch,
            drop_last=False,
        )

        model = PosthostOneEditScorer(
            input_dim=len(CANDIDATE_FEATURE_NAMES),
            hidden_dim=int(args.hidden_dim),
            dropout=float(args.dropout),
            num_layers=int(args.num_layers),
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
        )

        best_state: Dict[str, Any] | None = None
        best_metric_name = "nonkeep_f0.5"
        best_metric = float("-inf")
        best_epoch = -1
        metrics_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with metrics_jsonl.open("w", encoding="utf-8") as metrics_fp:
            for epoch in range(1, int(args.epochs) + 1):
                model.train()
                total_weight = 0.0
                total_loss_weighted = 0.0
                total_examples = 0
                total_correct = 0
                total_action_type_correct = 0
                total_keep_vs_edit_correct = 0
                total_pred_nonkeep = 0
                total_target_nonkeep = 0
                total_correct_nonkeep = 0
                total_target_swap = 0
                total_correct_swap_action = 0

                for batch in train_loader:
                    features = batch["candidate_features"].to(device)
                    mask = batch["candidate_mask"].to(device)
                    targets = batch["target_index"].to(device)
                    target_is_nonkeep = batch["target_is_nonkeep"].to(device)
                    candidate_action_type_id = batch["candidate_action_type_id"].to(device)
                    target_action_type_id = batch["target_action_type_id"].to(device)

                    logits = model(features)
                    logits = logits.masked_fill(~mask, float("-inf"))
                    per_sample_loss = nn.functional.cross_entropy(logits, targets, reduction="none")
                    sample_weight = torch.ones_like(per_sample_loss)
                    sample_weight = torch.where(
                        target_is_nonkeep,
                        torch.full_like(per_sample_loss, float(args.positive_cluster_weight)),
                        sample_weight,
                    )
                    sample_weight = torch.where(
                        target_action_type_id == 2,
                        torch.full_like(per_sample_loss, float(max(args.swap_cluster_weight, 1.0))),
                        sample_weight,
                    )
                    loss = (per_sample_loss * sample_weight).sum() / sample_weight.sum().clamp(min=1.0)

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()

                    with torch.no_grad():
                        pred = masked_argmax(logits, mask)
                        pred_action_type_id = candidate_action_type_id.gather(1, pred.view(-1, 1)).view(-1)
                        total_examples += int(targets.numel())
                        total_weight += float(sample_weight.sum().item())
                        total_loss_weighted += float((per_sample_loss * sample_weight).sum().item())
                        total_correct += int((pred == targets).sum().item())
                        total_action_type_correct += int((pred_action_type_id == target_action_type_id).sum().item())
                        total_keep_vs_edit_correct += int(((pred != 0) == target_is_nonkeep.to(dtype=torch.bool)).sum().item())
                        pred_nonkeep = pred != 0
                        target_nonkeep_bool = target_is_nonkeep.to(dtype=torch.bool)
                        total_pred_nonkeep += int(pred_nonkeep.sum().item())
                        total_target_nonkeep += int(target_nonkeep_bool.sum().item())
                        total_correct_nonkeep += int(((pred == targets) & target_nonkeep_bool).sum().item())
                        total_target_swap += int((target_action_type_id == 2).sum().item())
                        total_correct_swap_action += int(((pred_action_type_id == 2) & (target_action_type_id == 2)).sum().item())

                train_precision = float(total_correct_nonkeep / max(total_pred_nonkeep, 1))
                train_recall = float(total_correct_nonkeep / max(total_target_nonkeep, 1))
                train_metrics = {
                    "loss": float(total_loss_weighted / max(total_weight, 1.0)),
                    "top1_acc": float(total_correct / max(total_examples, 1)),
                    "action_type_acc": float(total_action_type_correct / max(total_examples, 1)),
                    "keep_vs_edit_acc": float(total_keep_vs_edit_correct / max(total_examples, 1)),
                    "swap_action_recall": float(total_correct_swap_action / max(total_target_swap, 1)),
                    "nonkeep_precision": train_precision,
                    "nonkeep_recall": train_recall,
                    "nonkeep_coverage": float(total_pred_nonkeep / max(total_examples, 1)),
                    "nonkeep_f0_5": nonkeep_f_beta(train_precision, train_recall, beta=0.5),
                    "target_keep_clusters": int(total_examples - total_target_nonkeep),
                    "target_nonkeep_clusters": int(total_target_nonkeep),
                }
                val_metrics = evaluate(
                    model,
                    val_loader,
                    device=device,
                    positive_cluster_weight=float(args.positive_cluster_weight),
                    swap_cluster_weight=float(args.swap_cluster_weight),
                )

                current_metric = metric_value(val_metrics, best_metric_name)
                if current_metric > best_metric:
                    best_metric = float(current_metric)
                    best_epoch = int(epoch)
                    best_state = {
                        "epoch": int(epoch),
                        "best_metric_name": best_metric_name,
                        "best_metric_value": float(best_metric),
                        "train_metrics": dict(train_metrics),
                        "val_metrics": dict(val_metrics),
                        "feature_names": list(CANDIDATE_FEATURE_NAMES),
                    }
                    model.save_checkpoint(
                        checkpoint_path,
                        extra={
                            "epoch": int(epoch),
                            "best_metric_name": best_metric_name,
                            "best_metric_value": float(best_metric),
                            "train_metrics": dict(train_metrics),
                            "val_metrics": dict(val_metrics),
                        },
                    )

                epoch_row = {
                    "epoch": int(epoch),
                    "train_loss": float(train_metrics["loss"]),
                    "train_top1_acc": float(train_metrics["top1_acc"]),
                    "train_action_type_acc": float(train_metrics["action_type_acc"]),
                    "train_keep_vs_edit_acc": float(train_metrics["keep_vs_edit_acc"]),
                    "train_swap_action_recall": float(train_metrics["swap_action_recall"]),
                    "train_nonkeep_precision": float(train_metrics["nonkeep_precision"]),
                    "train_nonkeep_recall": float(train_metrics["nonkeep_recall"]),
                    "train_nonkeep_coverage": float(train_metrics["nonkeep_coverage"]),
                    "train_nonkeep_f0_5": float(train_metrics["nonkeep_f0_5"]),
                    "val_loss": float(val_metrics["loss"]),
                    "val_top1_acc": float(val_metrics["top1_acc"]),
                    "val_action_type_acc": float(val_metrics["action_type_acc"]),
                    "val_keep_vs_edit_acc": float(val_metrics["keep_vs_edit_acc"]),
                    "val_swap_action_recall": float(val_metrics["swap_action_recall"]),
                    "val_nonkeep_precision": float(val_metrics["nonkeep_precision"]),
                    "val_nonkeep_recall": float(val_metrics["nonkeep_recall"]),
                    "val_nonkeep_coverage": float(val_metrics["nonkeep_coverage"]),
                    "val_nonkeep_f0_5": float(val_metrics["nonkeep_f0_5"]),
                    "val_target_keep_clusters": int(val_metrics["target_keep_clusters"]),
                    "val_target_nonkeep_clusters": int(val_metrics["target_nonkeep_clusters"]),
                    "best_metric_name": best_metric_name,
                    "best_metric_value_so_far": float(best_metric),
                    "is_best": int(epoch == best_epoch),
                }
                metrics_fp.write(json.dumps(epoch_row, ensure_ascii=True) + "\n")
                metrics_fp.flush()

        if best_state is None:
            raise RuntimeError("Training finished without producing a best checkpoint")

        summary_row.update(
            {
                "best_epoch": int(best_state["epoch"]),
                "best_metric_name": str(best_state["best_metric_name"]),
                "best_metric_value": float(best_state["best_metric_value"]),
                "train_loss": float(best_state["train_metrics"]["loss"]),
                "train_top1_acc": float(best_state["train_metrics"]["top1_acc"]),
                "train_action_type_acc": float(best_state["train_metrics"]["action_type_acc"]),
                "train_keep_vs_edit_acc": float(best_state["train_metrics"]["keep_vs_edit_acc"]),
                "train_swap_action_recall": float(best_state["train_metrics"]["swap_action_recall"]),
                "train_nonkeep_precision": float(best_state["train_metrics"]["nonkeep_precision"]),
                "train_nonkeep_recall": float(best_state["train_metrics"]["nonkeep_recall"]),
                "train_nonkeep_coverage": float(best_state["train_metrics"]["nonkeep_coverage"]),
                "train_nonkeep_f0_5": float(best_state["train_metrics"]["nonkeep_f0_5"]),
                "val_loss": float(best_state["val_metrics"]["loss"]),
                "val_top1_acc": float(best_state["val_metrics"]["top1_acc"]),
                "val_action_type_acc": float(best_state["val_metrics"]["action_type_acc"]),
                "val_keep_vs_edit_acc": float(best_state["val_metrics"]["keep_vs_edit_acc"]),
                "val_swap_action_recall": float(best_state["val_metrics"]["swap_action_recall"]),
                "val_nonkeep_precision": float(best_state["val_metrics"]["nonkeep_precision"]),
                "val_nonkeep_recall": float(best_state["val_metrics"]["nonkeep_recall"]),
                "val_nonkeep_coverage": float(best_state["val_metrics"]["nonkeep_coverage"]),
                "val_nonkeep_f0_5": float(best_state["val_metrics"]["nonkeep_f0_5"]),
                "val_target_keep_clusters": int(best_state["val_metrics"]["target_keep_clusters"]),
                "val_target_nonkeep_clusters": int(best_state["val_metrics"]["target_nonkeep_clusters"]),
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
