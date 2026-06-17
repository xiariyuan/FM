#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

REPO_ROOT = Path("/gemini/code/FMtrack-main/FM-Track")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.fcaa.fcaa.data.pairbank_dataset import PairGroupDataset, collate_pair_groups, group_pairbank_rows, load_pairbank_rows
from projects.fcaa.fcaa.metrics.offline_metrics import binary_auc, edit_flip_stats, flatten_scores, group_top1_accuracy
from projects.fcaa.fcaa.model.pair_scorer import FCAAPairScorer, feature_names


REGISTRY_CSV = REPO_ROOT / "outputs" / "experiment_registry.csv"

SUMMARY_FIELDS = [
    "run_name",
    "pairbank_jsonl",
    "val_pairbank_jsonl",
    "mode",
    "optimizer",
    "train_groups",
    "val_groups",
    "ambiguous_train_groups",
    "ambiguous_val_groups",
    "batch_size",
    "epochs",
    "lr",
    "weight_decay",
    "positive_weight",
    "ambiguous_oversample",
    "seed",
    "best_epoch",
    "best_metric_name",
    "best_metric_value",
    "val_loss",
    "val_auc",
    "val_ambiguous_auc",
    "val_top1",
    "val_ambiguous_top1",
    "val_wrong_to_right_rate",
    "val_right_to_wrong_rate",
    "checkpoint",
    "metrics_jsonl",
    "status",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the minimal FCAA pair scorer.")
    parser.add_argument("--pairbank-jsonl", required=True)
    parser.add_argument("--val-pairbank-jsonl", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run-name", default="fcaa_pair_scorer")
    parser.add_argument("--mode", choices=["control", "freq"], default="freq")
    parser.add_argument("--optimizer", choices=["adamw", "lbfgs"], default="adamw")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--train-seq-names", nargs="*", default=[])
    parser.add_argument("--val-seq-names", nargs="*", default=[])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--positive-weight", type=float, default=3.0)
    parser.add_argument("--ambiguous-oversample", type=float, default=2.0)
    parser.add_argument("--lbfgs-max-iter", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--registry-csv", default=str(REGISTRY_CSV))
    return parser.parse_args()


def write_rows(path: Path, fieldnames: Iterable[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def split_groups(groups: List[object], train_ratio: float, seed: int) -> tuple[List[object], List[object]]:
    indices = list(range(len(groups)))
    rng = random.Random(int(seed))
    rng.shuffle(indices)
    cut = max(1, min(len(indices) - 1, int(round(len(indices) * float(train_ratio)))))
    train = [groups[idx] for idx in indices[:cut]]
    val = [groups[idx] for idx in indices[cut:]]
    return train, val


def split_groups_by_seq(groups: List[object], train_seq_names: List[str], val_seq_names: List[str]) -> tuple[List[object], List[object]]:
    train_set = set(str(name) for name in train_seq_names)
    val_set = set(str(name) for name in val_seq_names)
    train_groups = []
    val_groups = []
    for group in groups:
        raw_seq = ""
        if group.candidates:
            raw_seq = str(group.candidates[0].raw.get("seq_name", ""))
        if raw_seq in val_set:
            val_groups.append(group)
        elif raw_seq in train_set:
            train_groups.append(group)
    return train_groups, val_groups


def baseline_tensor(batch: Dict[str, torch.Tensor | List[str]]) -> torch.Tensor:
    features = batch["features"]
    assert isinstance(features, torch.Tensor)
    return features[..., 0]


def pad_group_rows(batches: List[torch.Tensor], *, padding_value: float = 0.0) -> torch.Tensor:
    rows: List[torch.Tensor] = []
    for batch in batches:
        for idx in range(batch.shape[0]):
            rows.append(batch[idx])
    if not rows:
        return torch.zeros((0, 0), dtype=torch.float32)
    return pad_sequence(rows, batch_first=True, padding_value=padding_value)


def evaluate(model: FCAAPairScorer, loader: DataLoader, device: torch.device, positive_weight: float) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_groups = 0
    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    all_masks: List[torch.Tensor] = []
    all_ambiguous: List[torch.Tensor] = []
    all_baseline: List[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            features = batch["features"]
            labels = batch["labels"]
            mask = batch["mask"]
            ambiguous = batch["ambiguous"]
            assert isinstance(features, torch.Tensor)
            assert isinstance(labels, torch.Tensor)
            assert isinstance(mask, torch.Tensor)
            assert isinstance(ambiguous, torch.Tensor)
            logits = model(features.to(device))
            labels_dev = labels.to(device)
            mask_dev = mask.to(device)
            pos_weight = torch.full_like(labels_dev, float(positive_weight))
            weight = torch.where(labels_dev > 0.5, pos_weight, torch.ones_like(labels_dev))
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels_dev, reduction="none")
            loss = (loss * weight * mask_dev.to(dtype=loss.dtype)).sum() / mask_dev.to(dtype=loss.dtype).sum().clamp(min=1.0)
            total_loss += float(loss.item()) * float(features.shape[0])
            total_groups += int(features.shape[0])
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            all_masks.append(mask.cpu())
            all_ambiguous.append(ambiguous.cpu())
            all_baseline.append(baseline_tensor(batch).cpu())
    logits = pad_group_rows(all_logits, padding_value=0.0)
    labels = pad_group_rows(all_labels, padding_value=0.0)
    mask = pad_group_rows([batch.to(dtype=torch.float32) for batch in all_masks], padding_value=0.0).to(dtype=torch.bool)
    ambiguous = torch.cat(all_ambiguous, dim=0) if all_ambiguous else torch.zeros((0,), dtype=torch.bool)
    baseline_scores = pad_group_rows(all_baseline, padding_value=0.0)
    flat = flatten_scores(logits, labels, mask)
    ambiguous_scores: List[float] = []
    ambiguous_labels: List[int] = []
    for idx in range(logits.shape[0]):
        if not bool(ambiguous[idx].item()):
            continue
        for cand_idx in torch.nonzero(mask[idx], as_tuple=False).view(-1).tolist():
            ambiguous_scores.append(float(logits[idx, cand_idx].item()))
            ambiguous_labels.append(int(labels[idx, cand_idx].item()))
    flips = edit_flip_stats(logits, labels, mask, baseline_scores)
    return {
        "loss": float(total_loss / max(total_groups, 1)),
        "auc": binary_auc(flat["scores"], flat["labels"]),
        "ambiguous_auc": binary_auc(ambiguous_scores, ambiguous_labels),
        "top1": group_top1_accuracy(logits, labels, mask),
        "ambiguous_top1": group_top1_accuracy(logits, labels, mask, ambiguous_only=True, ambiguous=ambiguous),
        "wrong_to_right_rate": flips["wrong_to_right_rate"],
        "right_to_wrong_rate": flips["right_to_wrong_rate"],
    }


def batch_loss(
    model: FCAAPairScorer,
    batch: Dict[str, torch.Tensor | List[str]],
    device: torch.device,
    positive_weight: float,
) -> torch.Tensor:
    features = batch["features"]
    labels = batch["labels"]
    mask = batch["mask"]
    assert isinstance(features, torch.Tensor)
    assert isinstance(labels, torch.Tensor)
    assert isinstance(mask, torch.Tensor)
    logits = model(features.to(device))
    labels_dev = labels.to(device)
    mask_dev = mask.to(device)
    pos_weight = torch.full_like(labels_dev, float(positive_weight))
    weight = torch.where(labels_dev > 0.5, pos_weight, torch.ones_like(labels_dev))
    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels_dev, reduction="none")
    return (loss * weight * mask_dev.to(dtype=loss.dtype)).sum() / mask_dev.to(dtype=loss.dtype).sum().clamp(min=1.0)


def append_registry(args: argparse.Namespace, summary_csv: Path, run_root: Path, checkpoint: Path, status: str) -> None:
    import subprocess

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/append_experiment_record.py"),
        "--csv",
        str(args.registry_csv),
        "--kind",
        "train",
        "--status",
        status,
        "--script",
        "scripts/train_fcaa_pair_scorer.py",
        "--dataset",
        "MOT17",
        "--split",
        "pairbank",
        "--tracker-family",
        "botsort_fcaa",
        "--variant",
        args.run_name,
        "--tag",
        args.mode,
        "--run-root",
        str(run_root),
        "--summary-csv",
        str(summary_csv),
        "--checkpoint",
        str(checkpoint),
        "--notes",
        f"fcaa pair scorer mode={args.mode}",
    ]
    subprocess.run(cmd, check=False)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "summary.csv"
    metrics_jsonl = out_dir / "metrics.jsonl"
    checkpoint = out_dir / "best.pt"
    summary_row = {
        "run_name": args.run_name,
        "pairbank_jsonl": args.pairbank_jsonl,
        "val_pairbank_jsonl": args.val_pairbank_jsonl,
        "mode": args.mode,
        "optimizer": args.optimizer,
        "train_groups": 0,
        "val_groups": 0,
        "ambiguous_train_groups": 0,
        "ambiguous_val_groups": 0,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "positive_weight": args.positive_weight,
        "ambiguous_oversample": args.ambiguous_oversample,
        "seed": args.seed,
        "best_epoch": "",
        "best_metric_name": "ambiguous_top1",
        "best_metric_value": "",
        "val_loss": "",
        "val_auc": "",
        "val_ambiguous_auc": "",
        "val_top1": "",
        "val_ambiguous_top1": "",
        "val_wrong_to_right_rate": "",
        "val_right_to_wrong_rate": "",
        "checkpoint": "",
        "metrics_jsonl": str(metrics_jsonl),
        "status": "running",
        "error": "",
    }
    write_rows(summary_csv, SUMMARY_FIELDS, [summary_row])

    set_seed(int(args.seed))
    train_rows = load_pairbank_rows(Path(args.pairbank_jsonl))
    if args.val_pairbank_jsonl and (args.train_seq_names or args.val_seq_names):
        raise ValueError("Cannot combine --val-pairbank-jsonl with sequence-based splitting flags.")
    if args.val_pairbank_jsonl:
        train_groups = group_pairbank_rows(train_rows, args.mode)
        val_rows = load_pairbank_rows(Path(args.val_pairbank_jsonl))
        val_groups = group_pairbank_rows(val_rows, args.mode)
        if not train_groups or not val_groups:
            raise ValueError("Explicit train/val pair-bank inputs produced an empty train or val set.")
    else:
        groups = group_pairbank_rows(train_rows, args.mode)
        if args.train_seq_names or args.val_seq_names:
            train_groups, val_groups = split_groups_by_seq(groups, list(args.train_seq_names), list(args.val_seq_names))
            if not train_groups or not val_groups:
                raise ValueError("Sequence-based split produced an empty train or val set.")
        else:
            train_groups, val_groups = split_groups(groups, float(args.train_ratio), int(args.seed))
    summary_row["train_groups"] = len(train_groups)
    summary_row["val_groups"] = len(val_groups)
    summary_row["ambiguous_train_groups"] = sum(1 for group in train_groups if group.ambiguous)
    summary_row["ambiguous_val_groups"] = sum(1 for group in val_groups if group.ambiguous)
    write_rows(summary_csv, SUMMARY_FIELDS, [summary_row])

    train_dataset = PairGroupDataset(train_groups, ambiguous_oversample=float(args.ambiguous_oversample), seed=int(args.seed))
    val_dataset = PairGroupDataset(val_groups, ambiguous_oversample=1.0, seed=int(args.seed))
    train_batch_size = int(args.batch_size)
    train_shuffle = True
    if str(args.optimizer) == "lbfgs":
        train_batch_size = max(1, len(train_dataset))
        train_shuffle = False
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=train_shuffle, num_workers=int(args.num_workers), collate_fn=collate_pair_groups)
    val_loader = DataLoader(val_dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers), collate_fn=collate_pair_groups)

    input_dim = len(feature_names(args.mode))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FCAAPairScorer(input_dim=input_dim).to(device)
    if str(args.optimizer) == "lbfgs":
        optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=float(args.lr),
            max_iter=int(args.lbfgs_max_iter),
            history_size=10,
            line_search_fn="strong_wolfe",
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    best_metric = float("-inf")
    best_epoch = -1

    try:
        with metrics_jsonl.open("w", encoding="utf-8") as metrics_handle:
            for epoch in range(int(args.epochs)):
                model.train()
                running_loss = 0.0
                running_groups = 0
                if str(args.optimizer) == "lbfgs":
                    batch = next(iter(train_loader))

                    def closure() -> torch.Tensor:
                        optimizer.zero_grad(set_to_none=True)
                        loss = batch_loss(model, batch, device, float(args.positive_weight))
                        loss.backward()
                        return loss

                    loss = optimizer.step(closure)
                    features = batch["features"]
                    assert isinstance(features, torch.Tensor)
                    running_loss += float(loss.item()) * float(features.shape[0])
                    running_groups += int(features.shape[0])
                else:
                    for batch in train_loader:
                        loss = batch_loss(model, batch, device, float(args.positive_weight))
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                        features = batch["features"]
                        assert isinstance(features, torch.Tensor)
                        running_loss += float(loss.item()) * float(features.shape[0])
                        running_groups += int(features.shape[0])
                val_metrics = evaluate(model, val_loader, device, float(args.positive_weight))
                train_loss = float(running_loss / max(running_groups, 1))
                payload = {
                    "epoch": int(epoch),
                    "train_loss": train_loss,
                    **{f"val_{key}": value for key, value in val_metrics.items()},
                }
                metrics_handle.write(json.dumps(payload) + "\n")
                metrics_handle.flush()
                metric = float(val_metrics["ambiguous_top1"])
                if metric > best_metric:
                    best_metric = metric
                    best_epoch = int(epoch)
                    torch.save(
                        {
                            "model_state": model.state_dict(),
                            "input_dim": input_dim,
                            "mode": args.mode,
                            "feature_names": feature_names(args.mode),
                            "epoch": int(epoch),
                            "metric": metric,
                        },
                        checkpoint,
                    )
                    summary_row.update(
                        {
                            "best_epoch": best_epoch,
                            "best_metric_value": metric,
                            "val_loss": val_metrics["loss"],
                            "val_auc": val_metrics["auc"],
                            "val_ambiguous_auc": val_metrics["ambiguous_auc"],
                            "val_top1": val_metrics["top1"],
                            "val_ambiguous_top1": val_metrics["ambiguous_top1"],
                            "val_wrong_to_right_rate": val_metrics["wrong_to_right_rate"],
                            "val_right_to_wrong_rate": val_metrics["right_to_wrong_rate"],
                            "checkpoint": str(checkpoint),
                        }
                    )
                    write_rows(summary_csv, SUMMARY_FIELDS, [summary_row])
        summary_row["status"] = "success"
        write_rows(summary_csv, SUMMARY_FIELDS, [summary_row])
        append_registry(args, summary_csv, out_dir, checkpoint, "success")
    except Exception as exc:
        summary_row["status"] = "failed"
        summary_row["error"] = str(exc)
        write_rows(summary_csv, SUMMARY_FIELDS, [summary_row])
        append_registry(args, summary_csv, out_dir, checkpoint, "failed")
        raise


if __name__ == "__main__":
    main()
