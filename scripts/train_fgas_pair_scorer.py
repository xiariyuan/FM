#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
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

from projects.fcaa.fcaa.metrics.offline_metrics import binary_auc, edit_flip_stats, flatten_scores, group_top1_accuracy
from projects.fgas.fgas.data.blockbank_io import load_blockbank_jsonl
from projects.fgas.fgas.data.pairgroup_dataset import PairGroupDataset, blockbank_to_pair_groups, collate_pair_groups, group_feature_names
from projects.fgas.fgas.model.pair_scorer import FGASPairScorer


REGISTRY_CSV = REPO_ROOT / "outputs" / "experiment_registry.csv"
SUMMARY_FIELDS = [
    "run_name",
    "train_blockbank_jsonl",
    "val_blockbank_jsonl",
    "mode",
    "train_blocks",
    "val_blocks",
    "train_groups",
    "val_groups",
    "ambiguous_train_groups",
    "ambiguous_val_groups",
    "batch_size",
    "epochs",
    "lr",
    "weight_decay",
    "hidden_dim",
    "dropout",
    "positive_weight",
    "ambiguous_oversample",
    "seed",
    "input_dim",
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
    parser = argparse.ArgumentParser(description="Train the detector-aware FGAS ambiguity pair scorer.")
    parser.add_argument("--train-blockbank-jsonl", required=True)
    parser.add_argument("--val-blockbank-jsonl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run-name", default="fgas_pair_scorer")
    parser.add_argument("--mode", choices=["nofreq", "freq"], default="nofreq")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--positive-weight", type=float, default=3.0)
    parser.add_argument("--ambiguous-oversample", type=float, default=3.0)
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


def pad_group_rows(batches: List[torch.Tensor], *, padding_value: float = 0.0) -> torch.Tensor:
    rows: List[torch.Tensor] = []
    for batch in batches:
        for idx in range(batch.shape[0]):
            rows.append(batch[idx])
    if not rows:
        return torch.zeros((0, 0), dtype=torch.float32)
    return pad_sequence(rows, batch_first=True, padding_value=padding_value)


def evaluate(model: FGASPairScorer, loader: DataLoader, device: torch.device, positive_weight: float) -> Dict[str, float]:
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
            baseline = batch["baseline"]
            assert isinstance(features, torch.Tensor)
            assert isinstance(labels, torch.Tensor)
            assert isinstance(mask, torch.Tensor)
            assert isinstance(ambiguous, torch.Tensor)
            assert isinstance(baseline, torch.Tensor)
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
            all_baseline.append(baseline.cpu())
    logits = pad_group_rows(all_logits, padding_value=0.0)
    labels = pad_group_rows(all_labels, padding_value=0.0)
    mask = pad_group_rows([item.to(dtype=torch.float32) for item in all_masks], padding_value=0.0).to(dtype=torch.bool)
    ambiguous = torch.cat(all_ambiguous, dim=0) if all_ambiguous else torch.zeros((0,), dtype=torch.bool)
    baseline = pad_group_rows(all_baseline, padding_value=0.0)
    flat = flatten_scores(logits, labels, mask)
    ambiguous_scores: List[float] = []
    ambiguous_labels: List[int] = []
    for idx in range(logits.shape[0]):
        if not bool(ambiguous[idx].item()):
            continue
        for cand_idx in torch.nonzero(mask[idx], as_tuple=False).view(-1).tolist():
            ambiguous_scores.append(float(logits[idx, cand_idx].item()))
            ambiguous_labels.append(int(labels[idx, cand_idx].item()))
    flips = edit_flip_stats(logits, labels, mask, baseline)
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
    model: FGASPairScorer,
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
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "append_experiment_record.py"),
        "--csv",
        str(args.registry_csv),
        "--kind",
        "train",
        "--status",
        status,
        "--script",
        "scripts/train_fgas_pair_scorer.py",
        "--dataset",
        "MOT17",
        "--split",
        "blockbank",
        "--tracker-family",
        "deep_ocsort_fgas",
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
        f"FGAS pair scorer mode={args.mode}",
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=False)


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "summary.csv"
    metrics_jsonl = out_dir / "metrics.jsonl"
    checkpoint = out_dir / "best.pt"
    summary_row: Dict[str, object] = {
        "run_name": args.run_name,
        "train_blockbank_jsonl": args.train_blockbank_jsonl,
        "val_blockbank_jsonl": args.val_blockbank_jsonl,
        "mode": args.mode,
        "train_blocks": 0,
        "val_blocks": 0,
        "train_groups": 0,
        "val_groups": 0,
        "ambiguous_train_groups": 0,
        "ambiguous_val_groups": 0,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "positive_weight": args.positive_weight,
        "ambiguous_oversample": args.ambiguous_oversample,
        "seed": args.seed,
        "input_dim": 0,
        "best_epoch": -1,
        "best_metric_name": "ambiguous_top1",
        "best_metric_value": 0.0,
        "val_loss": 0.0,
        "val_auc": 0.0,
        "val_ambiguous_auc": 0.0,
        "val_top1": 0.0,
        "val_ambiguous_top1": 0.0,
        "val_wrong_to_right_rate": 0.0,
        "val_right_to_wrong_rate": 0.0,
        "checkpoint": "",
        "metrics_jsonl": str(metrics_jsonl),
        "status": "running",
        "error": "",
    }
    write_rows(summary_csv, SUMMARY_FIELDS, [summary_row])
    append_registry(args, summary_csv, out_dir, checkpoint, "running")

    try:
        include_frequency = bool(str(args.mode) == "freq")
        train_blocks = load_blockbank_jsonl(Path(args.train_blockbank_jsonl))
        val_blocks = load_blockbank_jsonl(Path(args.val_blockbank_jsonl))
        train_groups = blockbank_to_pair_groups(train_blocks, include_frequency=include_frequency)
        val_groups = blockbank_to_pair_groups(val_blocks, include_frequency=include_frequency)
        if not train_groups or not val_groups:
            raise ValueError("Empty train or val groups after blockbank conversion.")

        feature_spec = group_feature_names(train_blocks, include_frequency=include_frequency)
        edge_feature_names = list(feature_spec["edge_feature_names"])
        row_feature_names = list(feature_spec["row_feature_names"])
        col_feature_names = list(feature_spec["col_feature_names"])
        domain_feature_names = list(feature_spec["domain_feature_names"])
        input_dim = len(edge_feature_names) + len(row_feature_names) + len(col_feature_names) + len(domain_feature_names)

        summary_row.update(
            {
                "train_blocks": int(len(train_blocks)),
                "val_blocks": int(len(val_blocks)),
                "train_groups": int(len(train_groups)),
                "val_groups": int(len(val_groups)),
                "ambiguous_train_groups": int(sum(1 for group in train_groups if group.ambiguous)),
                "ambiguous_val_groups": int(sum(1 for group in val_groups if group.ambiguous)),
                "input_dim": int(input_dim),
            }
        )
        write_rows(summary_csv, SUMMARY_FIELDS, [summary_row])

        train_dataset = PairGroupDataset(train_groups, ambiguous_oversample=float(args.ambiguous_oversample), seed=int(args.seed))
        val_dataset = PairGroupDataset(val_groups, ambiguous_oversample=1.0, seed=int(args.seed))
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(args.batch_size),
            shuffle=True,
            num_workers=int(args.num_workers),
            collate_fn=collate_pair_groups,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            collate_fn=collate_pair_groups,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = FGASPairScorer(input_dim=input_dim, hidden_dim=int(args.hidden_dim), dropout=float(args.dropout)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

        best_metric = float("-inf")
        best_state = None
        best_epoch = -1
        with metrics_jsonl.open("w", encoding="utf-8") as metrics_handle:
            for epoch in range(1, int(args.epochs) + 1):
                model.train()
                train_loss_sum = 0.0
                train_batches = 0
                for batch in train_loader:
                    optimizer.zero_grad(set_to_none=True)
                    loss = batch_loss(model, batch, device, float(args.positive_weight))
                    loss.backward()
                    optimizer.step()
                    train_loss_sum += float(loss.item())
                    train_batches += 1
                val_metrics = evaluate(model, val_loader, device, float(args.positive_weight))
                metric_value = float(val_metrics["ambiguous_top1"])
                record = {
                    "epoch": int(epoch),
                    "train_loss": float(train_loss_sum / max(train_batches, 1)),
                    "val_loss": float(val_metrics["loss"]),
                    "val_auc": float(val_metrics["auc"]),
                    "val_ambiguous_auc": float(val_metrics["ambiguous_auc"]),
                    "val_top1": float(val_metrics["top1"]),
                    "val_ambiguous_top1": float(val_metrics["ambiguous_top1"]),
                    "val_wrong_to_right_rate": float(val_metrics["wrong_to_right_rate"]),
                    "val_right_to_wrong_rate": float(val_metrics["right_to_wrong_rate"]),
                }
                metrics_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                metrics_handle.flush()
                if metric_value > best_metric:
                    best_metric = metric_value
                    best_epoch = int(epoch)
                    best_state = {
                        "model_state": model.state_dict(),
                        "input_dim": int(input_dim),
                        "hidden_dim": int(args.hidden_dim),
                        "dropout": float(args.dropout),
                        "mode": str(args.mode),
                        "edge_feature_names": edge_feature_names,
                        "row_feature_names": row_feature_names,
                        "col_feature_names": col_feature_names,
                        "domain_feature_names": domain_feature_names,
                        "best_metric_name": "ambiguous_top1",
                        "best_metric_value": float(best_metric),
                    }
                    torch.save(best_state, checkpoint)
                    summary_row.update(
                        {
                            "best_epoch": int(best_epoch),
                            "best_metric_value": float(best_metric),
                            "val_loss": float(val_metrics["loss"]),
                            "val_auc": float(val_metrics["auc"]),
                            "val_ambiguous_auc": float(val_metrics["ambiguous_auc"]),
                            "val_top1": float(val_metrics["top1"]),
                            "val_ambiguous_top1": float(val_metrics["ambiguous_top1"]),
                            "val_wrong_to_right_rate": float(val_metrics["wrong_to_right_rate"]),
                            "val_right_to_wrong_rate": float(val_metrics["right_to_wrong_rate"]),
                            "checkpoint": str(checkpoint),
                        }
                    )
                    write_rows(summary_csv, SUMMARY_FIELDS, [summary_row])
        if best_state is None:
            raise RuntimeError("Training produced no checkpoint.")

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
