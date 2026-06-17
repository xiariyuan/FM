#!/usr/bin/env python3
"""Train RGSA Stage 1 deferral head.

Reads Stage 1 labels from outputs/rgsa_labels/ and trains a small MLP
to predict accept/defer/reject for each det-track pair.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from models.rgsa_contract import STAGE1_FEATURE_DIM, STAGE1_FEATURE_NAMES
from models.rgsa_losses import WeightedCrossEntropyLoss
from models.rgsa_stage1_deferral import Stage1DeferralHead


def validate_split(train_seqs, val_seqs):
    overlap = sorted(set(train_seqs) & set(val_seqs))
    if overlap:
        raise ValueError(f"train/val overlap detected: {overlap}")


def load_labels_and_features(labels_dir: str, seqs: list):
    """Load Stage 1 labels CSVs and extract features + labels."""
    all_features = []
    all_labels = []

    for seq in seqs:
        csv_path = os.path.join(labels_dir, seq, "stage1_labels.csv")
        if not os.path.exists(csv_path):
            print(f"[warn] missing {csv_path}, skipping")
            continue
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                feats = [float(row.get(fn, 0.0)) for fn in STAGE1_FEATURE_NAMES]
                label = int(row["label"])  # 0=accept, 1=defer, 2=reject
                all_features.append(feats)
                all_labels.append(label)

    if not all_features:
        raise RuntimeError(f"No data found in {labels_dir} for sequences {seqs}")

    X = np.array(all_features, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.array(all_labels, dtype=np.int64)
    return X, y


def compute_class_weights(y: np.ndarray, num_classes: int = 3) -> list:
    """Inverse frequency class weights."""
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (num_classes * counts)
    return weights.tolist()


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    class_correct = [0, 0, 0]
    class_total = [0, 0, 0]
    all_preds = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(X_batch)
            preds = logits.argmax(dim=-1)
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)
            for c in range(3):
                mask = y_batch == c
                class_correct[c] += (preds[mask] == c).sum().item()
                class_total[c] += mask.sum().item()
            all_preds.extend(preds.cpu().tolist())

    acc = correct / max(total, 1)
    class_accs = [class_correct[c] / max(class_total[c], 1) for c in range(3)]
    pred_counts = [all_preds.count(c) for c in range(3)]
    return {
        "accuracy": acc,
        "accept_acc": class_accs[0],
        "defer_acc": class_accs[1],
        "reject_acc": class_accs[2],
        "accept_count": class_total[0],
        "defer_count": class_total[1],
        "reject_count": class_total[2],
        "pred_accept": pred_counts[0],
        "pred_defer": pred_counts[1],
        "pred_reject": pred_counts[2],
        "total": total,
    }


def main():
    parser = argparse.ArgumentParser(description="Train RGSA Stage 1 deferral head")
    parser.add_argument("--labels-dir", required=True, help="Root of rgsa_labels directory")
    parser.add_argument("--train-seqs", nargs="+", default=["MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-09-FRCNN"])
    parser.add_argument("--val-seqs", nargs="+", default=["MOT17-05-FRCNN"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--focal-gamma", type=float, default=0.0)
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[32, 16])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    validate_split(args.train_seqs, args.val_seqs)

    # Load data
    print(f"[load] training data from {args.labels_dir}")
    X_train, y_train = load_labels_and_features(args.labels_dir, args.train_seqs)
    X_val, y_val = load_labels_and_features(args.labels_dir, args.val_seqs)

    print(f"[data] train: {len(X_train)} samples, val: {len(X_val)} samples")
    print(f"[data] train distribution: accept={sum(y_train==0)}, defer={sum(y_train==1)}, reject={sum(y_train==2)}")
    print(f"[data] val distribution: accept={sum(y_val==0)}, defer={sum(y_val==1)}, reject={sum(y_val==2)}")

    class_weights = compute_class_weights(y_train)
    print(f"[loss] class weights: {class_weights}")

    device = args.device
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # Model
    model = Stage1DeferralHead(
        input_dim=STAGE1_FEATURE_DIM,
        hidden_dims=tuple(args.hidden_dims),
        dropout=args.dropout,
    ).to(device)

    criterion = WeightedCrossEntropyLoss(
        class_weights=class_weights, focal_gamma=args.focal_gamma
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    # Training
    best_val_acc = 0.0
    patience_counter = 0
    history = []

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step(1.0 - val_metrics["accuracy"])

        record = {
            "epoch": epoch,
            "train_loss": avg_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(record)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"[epoch {epoch+1:3d}] loss={avg_loss:.4f} "
                f"val_acc={val_metrics['accuracy']:.4f} "
                f"accept_acc={val_metrics['accept_acc']:.4f} "
                f"defer_acc={val_metrics['defer_acc']:.4f} "
                f"reject_acc={val_metrics['reject_acc']:.4f} "
                f"defer_rate={val_metrics['pred_defer']/max(val_metrics['total'],1):.3f}"
            )

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            patience_counter = 0
            model.save_checkpoint(
                os.path.join(args.out_dir, "stage1_best.pt"),
                metadata={
                    "epoch": epoch,
                    "val_accuracy": val_metrics["accuracy"],
                    "class_weights": class_weights,
                    "train_seqs": args.train_seqs,
                    "val_seqs": args.val_seqs,
                    "feature_names": list(STAGE1_FEATURE_NAMES),
                },
            )
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"[early stop] at epoch {epoch+1}, best val_acc={best_val_acc:.4f}")
                break

    # Save final
    model.save_checkpoint(
        os.path.join(args.out_dir, "stage1_final.pt"),
        metadata={"epochs_trained": epoch + 1, "best_val_acc": best_val_acc},
    )

    # Save history
    with open(os.path.join(args.out_dir, "train_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Save summary CSV
    summary_path = os.path.join(args.out_dir, "summary.csv")
    best_record = max(history, key=lambda r: r.get("val_accuracy", 0))
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(best_record.keys()))
        writer.writeheader()
        writer.writerow(best_record)

    print(f"[done] best val_acc={best_val_acc:.4f}")
    print(f"[output] {args.out_dir}")


if __name__ == "__main__":
    main()
