#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path("/gemini/code/FMtrack-main/FM-Track")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.fgas.fgas.features.acceptance_features import ACCEPTANCE_FEATURE_NAMES
from projects.fgas.fgas.model.acceptance_gate import FGASAcceptanceGate


REGISTRY_CSV = REPO_ROOT / "outputs" / "experiment_registry.csv"
SUMMARY_FIELDS = [
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
    "avg_train_target",
    "avg_train_weight",
    "avg_val_target",
    "avg_val_weight",
    "best_epoch",
    "best_metric",
    "best_threshold",
    "val_accuracy",
    "val_balanced_accuracy",
    "val_precision",
    "val_recall",
    "val_f1",
    "val_positive_recall",
    "train_rows",
    "val_rows",
    "status",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a changed-row acceptance gate for FGAS runtime filtering.")
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
    parser.add_argument("--registry-kind", default="train")
    parser.add_argument("--registry-script", default="scripts/train_fgas_acceptance_gate.py")
    parser.add_argument("--registry-dataset", default="MOT17")
    parser.add_argument("--registry-split", default="acceptance_jsonl")
    parser.add_argument("--registry-tracker-family", default="deep_ocsort_fgas")
    parser.add_argument("--registry-variant", default="")
    parser.add_argument("--registry-tag", default="")
    parser.add_argument("--registry-notes", default="FGAS acceptance gate training")
    return parser.parse_args()


def write_single_row_csv(path: Path, fieldnames: List[str], row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def append_registry(args: argparse.Namespace, summary_csv: Path, checkpoint: Path | None, status: str, notes: str) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "append_experiment_record.py"),
        "--csv",
        str(args.registry_csv),
        "--kind",
        str(args.registry_kind),
        "--status",
        status,
        "--script",
        str(args.registry_script),
        "--dataset",
        str(args.registry_dataset),
        "--split",
        str(args.registry_split),
        "--tracker-family",
        str(args.registry_tracker_family),
        "--variant",
        str(args.registry_variant) or Path(args.out_dir).name,
        "--tag",
        str(args.registry_tag) or Path(args.out_dir).name,
        "--run-root",
        str(Path(args.out_dir)),
        "--summary-csv",
        str(summary_csv),
        "--checkpoint",
        str(checkpoint) if checkpoint else "",
        "--notes",
        notes,
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=False)


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


class AcceptanceDataset(Dataset):
    def __init__(self, rows: List[Dict[str, object]], feature_names: List[str]) -> None:
        self.rows = rows
        self.feature_names = list(feature_names)

    @staticmethod
    def from_jsonl(path: str) -> "AcceptanceDataset":
        rows: List[Dict[str, object]] = []
        feature_names: List[str] = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                row_feature_names = [str(name) for name in list(row.get("feature_names", []) or [])]
                if not row_feature_names:
                    row_feature_names = list(ACCEPTANCE_FEATURE_NAMES)
                    row["feature_names"] = list(row_feature_names)
                row_features = list(row.get("features", []) or [])
                if not feature_names:
                    feature_names = list(row_feature_names)
                if row_feature_names != feature_names:
                    raise ValueError(f"Inconsistent feature_names in {path}")
                if len(row_features) != len(feature_names):
                    raise ValueError(f"Feature length mismatch in {path}: expected {len(feature_names)} got {len(row_features)}")
                rows.append(row)
        return AcceptanceDataset(rows, feature_names or list(ACCEPTANCE_FEATURE_NAMES))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.rows[idx]
        return {
            "features": torch.tensor(row["features"], dtype=torch.float32),
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "train_target": torch.tensor(float(row.get("train_target", row["label"])), dtype=torch.float32),
            "sample_weight": torch.tensor(float(row.get("sample_weight", 1.0)), dtype=torch.float32),
        }


def collate_acceptance(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "features": torch.stack([row["features"] for row in batch], dim=0),
        "label": torch.stack([row["label"] for row in batch], dim=0),
        "train_target": torch.stack([row["train_target"] for row in batch], dim=0),
        "sample_weight": torch.stack([row["sample_weight"] for row in batch], dim=0),
    }


def weighted_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    sample_weight: torch.Tensor,
    pos_weight_value: float,
) -> torch.Tensor:
    pos_weight = torch.tensor([float(pos_weight_value)], dtype=torch.float32, device=logits.device)
    losses = nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
        pos_weight=pos_weight,
    )
    return (losses * sample_weight).mean()


def compute_metrics_from_probs(probs: torch.Tensor, labels: torch.Tensor, threshold: float) -> Dict[str, float]:
    preds = (probs >= float(threshold)).float()
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
        "positive_recall": float(recall),
    }


def select_best_threshold(probs: torch.Tensor, labels: torch.Tensor) -> tuple[float, Dict[str, float]]:
    best_threshold = 0.5
    best_metrics = compute_metrics_from_probs(probs, labels, best_threshold)
    best_key = (
        float(best_metrics["balanced_accuracy"]),
        float(best_metrics["f1"]),
        float(best_metrics["precision"]),
        -abs(best_threshold - 0.5),
    )
    for threshold_step in range(5, 96):
        threshold = float(threshold_step) / 100.0
        metrics = compute_metrics_from_probs(probs, labels, threshold)
        key = (
            float(metrics["balanced_accuracy"]),
            float(metrics["f1"]),
            float(metrics["precision"]),
            -abs(threshold - 0.5),
        )
        if key > best_key:
            best_threshold = float(threshold)
            best_metrics = metrics
            best_key = key
    return best_threshold, best_metrics


def evaluate(model: FGASAcceptanceGate, loader: DataLoader, device: torch.device, pos_weight_value: float) -> Dict[str, float]:
    model.eval()
    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    total_loss = 0.0
    batches = 0
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            labels = batch["label"].to(device)
            sample_weight = torch.ones_like(labels, device=device)
            logits = model(features)
            loss = weighted_bce_loss(logits, labels, sample_weight, pos_weight_value)
            total_loss += float(loss.item())
            batches += 1
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
    logits = torch.cat(all_logits, dim=0) if all_logits else torch.zeros((0,), dtype=torch.float32)
    labels = torch.cat(all_labels, dim=0) if all_labels else torch.zeros((0,), dtype=torch.float32)
    probs = torch.sigmoid(logits)
    best_threshold, metrics = select_best_threshold(probs, labels)
    metrics["loss"] = float(total_loss / batches) if batches else 0.0
    metrics["best_threshold"] = float(best_threshold)
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
        "train_jsonl": str(Path(args.train_jsonl)),
        "val_jsonl": str(Path(args.val_jsonl)),
        "feature_dim": 0,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "hidden_dim": int(args.hidden_dim),
        "dropout": float(args.dropout),
        "pos_weight": 0.0,
        "avg_train_target": 0.0,
        "avg_train_weight": 0.0,
        "avg_val_target": 0.0,
        "avg_val_weight": 0.0,
        "best_epoch": -1,
        "best_metric": 0.0,
        "best_threshold": 0.5,
        "val_accuracy": 0.0,
        "val_balanced_accuracy": 0.0,
        "val_precision": 0.0,
        "val_recall": 0.0,
        "val_f1": 0.0,
        "val_positive_recall": 0.0,
        "train_rows": 0,
        "val_rows": 0,
        "status": "running",
        "error": "",
    }
    write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)
    append_registry(args, summary_csv, best_ckpt, "running", str(args.registry_notes))

    try:
        train_dataset = AcceptanceDataset.from_jsonl(args.train_jsonl)
        val_dataset = AcceptanceDataset.from_jsonl(args.val_jsonl)
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            raise ValueError("Empty acceptance dataset.")
        if list(train_dataset.feature_names) != list(val_dataset.feature_names):
            raise ValueError("Train/val acceptance datasets have different feature_names.")
        feature_names = list(train_dataset.feature_names)
        summary_row["feature_dim"] = int(len(feature_names))
        summary_row["train_rows"] = int(len(train_dataset))
        summary_row["val_rows"] = int(len(val_dataset))
        summary_row["avg_train_target"] = float(
            sum(float(row.get("train_target", row["label"])) for row in train_dataset.rows)
            / float(max(len(train_dataset.rows), 1))
        )
        summary_row["avg_train_weight"] = float(
            sum(float(row.get("sample_weight", 1.0)) for row in train_dataset.rows)
            / float(max(len(train_dataset.rows), 1))
        )
        summary_row["avg_val_target"] = float(
            sum(float(row.get("train_target", row["label"])) for row in val_dataset.rows)
            / float(max(len(val_dataset.rows), 1))
        )
        summary_row["avg_val_weight"] = float(
            sum(float(row.get("sample_weight", 1.0)) for row in val_dataset.rows)
            / float(max(len(val_dataset.rows), 1))
        )
        has_soft_supervision = any(
            abs(float(row.get("train_target", row["label"])) - float(row["label"])) > 1e-6
            or abs(float(row.get("sample_weight", 1.0)) - 1.0) > 1e-6
            for row in train_dataset.rows
        )

        pos_count = sum(int(row["label"]) for row in train_dataset.rows)
        neg_count = len(train_dataset.rows) - pos_count
        pos_weight_value = float(args.pos_weight)
        if pos_weight_value <= 0.0:
            pos_weight_value = float(neg_count / max(pos_count, 1))
        summary_row["pos_weight"] = float(pos_weight_value)
        write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)

        train_loader = DataLoader(
            train_dataset,
            batch_size=int(args.batch_size),
            shuffle=True,
            num_workers=0,
            collate_fn=collate_acceptance,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=0,
            collate_fn=collate_acceptance,
        )

        device = torch.device(str(args.device))
        model = FGASAcceptanceGate(
            input_dim=len(feature_names),
            hidden_dim=int(args.hidden_dim),
            dropout=float(args.dropout),
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

        best_metric = -1.0
        with metrics_jsonl.open("w", encoding="utf-8") as handle:
            for epoch in range(1, int(args.epochs) + 1):
                model.train()
                for batch in train_loader:
                    features = batch["features"].to(device)
                    train_target = batch["train_target"].to(device)
                    sample_weight = batch["sample_weight"].to(device)
                    logits = model(features)
                    loss = weighted_bce_loss(logits, train_target, sample_weight, pos_weight_value)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                val_metrics = evaluate(model, val_loader, device, pos_weight_value)
                handle.write(json.dumps({"epoch": int(epoch), **val_metrics}))
                handle.write("\n")
                handle.flush()

                metric = float(val_metrics["balanced_accuracy"])
                if metric >= best_metric:
                    best_metric = metric
                    torch.save(
                        {
                            "model_state": model.state_dict(),
                            "feature_names": list(feature_names),
                            "input_dim": int(len(feature_names)),
                            "hidden_dim": int(args.hidden_dim),
                            "dropout": float(args.dropout),
                            "train_target_mode": "soft_weighted" if has_soft_supervision else "hard",
                            "acceptance_gate_thresh": float(val_metrics["best_threshold"]),
                        },
                        best_ckpt,
                    )
                    summary_row.update(
                        {
                            "best_epoch": int(epoch),
                            "best_metric": float(metric),
                            "best_threshold": float(val_metrics["best_threshold"]),
                            "val_accuracy": float(val_metrics["accuracy"]),
                            "val_balanced_accuracy": float(val_metrics["balanced_accuracy"]),
                            "val_precision": float(val_metrics["precision"]),
                            "val_recall": float(val_metrics["recall"]),
                            "val_f1": float(val_metrics["f1"]),
                            "val_positive_recall": float(val_metrics["positive_recall"]),
                        }
                    )
                    write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)

        summary_row["status"] = "success"
        write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)
        append_registry(args, summary_csv, best_ckpt, "success", str(args.registry_notes))
        return 0
    except Exception as exc:
        summary_row["status"] = "failed"
        summary_row["error"] = repr(exc)
        write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)
        append_registry(args, summary_csv, best_ckpt, "failed", str(args.registry_notes))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
