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
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path("/gemini/code/FMtrack-main/FM-Track")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.fgas.fgas.features.block_gate_features import BLOCK_GATE_FEATURE_NAMES
from projects.fgas.fgas.model.acceptance_gate import FGASAcceptanceGate


REGISTRY_CSV = REPO_ROOT / "outputs" / "experiment_registry.csv"
SUMMARY_FIELDS = [
    "run_name",
    "train_jsonl",
    "val_jsonl",
    "feature_dim",
    "epochs",
    "batch_size",
    "lr",
    "weight_decay",
    "hidden_dim",
    "dropout",
    "pos_weight",
    "seed",
    "best_epoch",
    "best_metric_name",
    "best_metric_value",
    "val_loss",
    "val_accuracy",
    "val_balanced_accuracy",
    "val_precision",
    "val_recall",
    "val_f1",
    "train_rows",
    "val_rows",
    "checkpoint",
    "metrics_jsonl",
    "status",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a block-level FGAS gate for pair-scorer intervention filtering.")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--pos-weight", type=float, default=0.0, help="<=0 enables automatic negative/positive ratio weighting")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--registry-csv", default=str(REGISTRY_CSV))
    return parser.parse_args()


def write_rows(path: Path, fieldnames: Iterable[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def append_registry(args: argparse.Namespace, summary_csv: Path, checkpoint: Path, status: str) -> None:
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
        "scripts/train_fgas_block_gate.py",
        "--dataset",
        "MOT17",
        "--split",
        "block_gate_jsonl",
        "--tracker-family",
        "deep_ocsort_fgas",
        "--variant",
        Path(args.out_dir).name,
        "--tag",
        Path(args.out_dir).name,
        "--run-root",
        str(Path(args.out_dir)),
        "--summary-csv",
        str(summary_csv),
        "--checkpoint",
        str(checkpoint),
        "--notes",
        "FGAS block gate training",
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=False)


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


class BlockGateDataset(Dataset):
    def __init__(self, rows: List[Dict[str, object]]) -> None:
        self.rows = rows

    @staticmethod
    def from_jsonl(path: str) -> "BlockGateDataset":
        rows: List[Dict[str, object]] = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return BlockGateDataset(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]
        return {
            "features": torch.tensor(row["features"], dtype=torch.float32),
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
        }


def collate_rows(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "features": torch.stack([row["features"] for row in batch], dim=0),
        "label": torch.stack([row["label"] for row in batch], dim=0),
    }


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    tp = float(((preds == 1) & (labels == 1)).sum().item())
    tn = float(((preds == 0) & (labels == 0)).sum().item())
    fp = float(((preds == 1) & (labels == 0)).sum().item())
    fn = float(((preds == 0) & (labels == 1)).sum().item())
    total = max(tp + tn + fp + fn, 1.0)
    accuracy = (tp + tn) / total
    recall = tp / max(tp + fn, 1.0)
    tnr = tn / max(tn + fp, 1.0)
    precision = tp / max(tp + fp, 1.0)
    f1 = (2.0 * precision * recall) / max(precision + recall, 1e-8)
    balanced_accuracy = 0.5 * (recall + tnr)
    return {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def evaluate(model: FGASAcceptanceGate, loader: DataLoader, device: torch.device, criterion: nn.Module) -> Dict[str, float]:
    model.eval()
    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    total_loss = 0.0
    batches = 0
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            labels = batch["label"].to(device)
            logits = model(features)
            loss = criterion(logits, labels)
            total_loss += float(loss.item())
            batches += 1
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
    logits = torch.cat(all_logits, dim=0) if all_logits else torch.zeros((0,), dtype=torch.float32)
    labels = torch.cat(all_labels, dim=0) if all_labels else torch.zeros((0,), dtype=torch.float32)
    metrics = compute_metrics(logits, labels)
    metrics["loss"] = float(total_loss / batches) if batches else 0.0
    return metrics


def main() -> int:
    args = parse_args()
    set_seed(int(args.seed))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "summary.csv"
    metrics_jsonl = out_dir / "metrics.jsonl"
    best_ckpt = out_dir / "best.pt"
    summary_row: Dict[str, object] = {
        "run_name": out_dir.name,
        "train_jsonl": str(Path(args.train_jsonl)),
        "val_jsonl": str(Path(args.val_jsonl)),
        "feature_dim": int(len(BLOCK_GATE_FEATURE_NAMES)),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "hidden_dim": int(args.hidden_dim),
        "dropout": float(args.dropout),
        "pos_weight": 0.0,
        "seed": int(args.seed),
        "best_epoch": -1,
        "best_metric_name": "balanced_accuracy",
        "best_metric_value": 0.0,
        "val_loss": 0.0,
        "val_accuracy": 0.0,
        "val_balanced_accuracy": 0.0,
        "val_precision": 0.0,
        "val_recall": 0.0,
        "val_f1": 0.0,
        "train_rows": 0,
        "val_rows": 0,
        "checkpoint": "",
        "metrics_jsonl": str(metrics_jsonl),
        "status": "running",
        "error": "",
    }
    write_rows(summary_csv, SUMMARY_FIELDS, [summary_row])
    append_registry(args, summary_csv, best_ckpt, "running")

    try:
        train_dataset = BlockGateDataset.from_jsonl(args.train_jsonl)
        val_dataset = BlockGateDataset.from_jsonl(args.val_jsonl)
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            raise ValueError("Empty block-gate dataset.")
        summary_row["train_rows"] = int(len(train_dataset))
        summary_row["val_rows"] = int(len(val_dataset))

        pos_count = sum(int(row["label"]) for row in train_dataset.rows)
        neg_count = len(train_dataset.rows) - pos_count
        pos_weight_value = float(args.pos_weight)
        if pos_weight_value <= 0.0:
            pos_weight_value = float(neg_count / max(pos_count, 1))
        summary_row["pos_weight"] = float(pos_weight_value)
        write_rows(summary_csv, SUMMARY_FIELDS, [summary_row])

        train_loader = DataLoader(
            train_dataset,
            batch_size=int(args.batch_size),
            shuffle=True,
            num_workers=0,
            collate_fn=collate_rows,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=0,
            collate_fn=collate_rows,
        )

        device = torch.device(str(args.device))
        model = FGASAcceptanceGate(
            input_dim=int(len(BLOCK_GATE_FEATURE_NAMES)),
            hidden_dim=int(args.hidden_dim),
            dropout=float(args.dropout),
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32, device=device))

        best_metric = -1.0
        with metrics_jsonl.open("w", encoding="utf-8") as handle:
            for epoch in range(1, int(args.epochs) + 1):
                model.train()
                for batch in train_loader:
                    features = batch["features"].to(device)
                    labels = batch["label"].to(device)
                    logits = model(features)
                    loss = criterion(logits, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                val_metrics = evaluate(model, val_loader, device, criterion)
                handle.write(json.dumps({"epoch": int(epoch), **val_metrics}))
                handle.write("\n")
                handle.flush()

                metric = float(val_metrics["balanced_accuracy"])
                if metric >= best_metric:
                    best_metric = metric
                    torch.save(
                        {
                            "model_state": model.state_dict(),
                            "feature_names": list(BLOCK_GATE_FEATURE_NAMES),
                            "input_dim": int(len(BLOCK_GATE_FEATURE_NAMES)),
                            "hidden_dim": int(args.hidden_dim),
                            "dropout": float(args.dropout),
                        },
                        best_ckpt,
                    )
                    summary_row.update(
                        {
                            "best_epoch": int(epoch),
                            "best_metric_value": float(metric),
                            "val_loss": float(val_metrics["loss"]),
                            "val_accuracy": float(val_metrics["accuracy"]),
                            "val_balanced_accuracy": float(val_metrics["balanced_accuracy"]),
                            "val_precision": float(val_metrics["precision"]),
                            "val_recall": float(val_metrics["recall"]),
                            "val_f1": float(val_metrics["f1"]),
                            "checkpoint": str(best_ckpt),
                        }
                    )
                    write_rows(summary_csv, SUMMARY_FIELDS, [summary_row])

        summary_row["status"] = "success"
        write_rows(summary_csv, SUMMARY_FIELDS, [summary_row])
        append_registry(args, summary_csv, best_ckpt, "success")
        return 0
    except Exception as exc:
        summary_row["status"] = "failed"
        summary_row["error"] = repr(exc)
        write_rows(summary_csv, SUMMARY_FIELDS, [summary_row])
        append_registry(args, summary_csv, best_ckpt, "failed")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
