#!/usr/bin/env python3
"""Train RGSA Stage 2 local recovery head.

Reads Stage 2 labels from outputs/rgsa_labels/ and trains a per-candidate
scoring + action classification model for deferred detections.
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

from models.rgsa_contract import HACA_PAIR_FEATURE_DIM, HACA_PAIR_FEATURE_NAMES
from models.rgsa_losses import StagePairLoss
from models.rgsa_stage2_recovery import Stage2RecoveryHead


def safe_float(value, default=0.0):
    if value in ("", None):
        return float(default)
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return out


def validate_split(train_seqs, val_seqs):
    overlap = sorted(set(train_seqs) & set(val_seqs))
    if overlap:
        raise ValueError(f"train/val overlap detected: {overlap}")


def compute_capped_class_weights(labels, num_classes=3, cap=20.0):
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (num_classes * counts)
    weights = np.minimum(weights, float(cap))
    return weights.tolist()


def load_stage2_data(labels_dir, seqs, max_k=5):
    """Load Stage 2 labels grouped by (seq, frame, det).

    Returns:
        features: (N, max_k, pair_dim) padded candidate features
        masks: (N, max_k) bool mask
        target_ranks: (N,) index of correct candidate (-1 if none)
        target_actions: (N,) action label (0=rewrite, 1=defer, 2=reject)
    """
    # Group by (seq, frame, det)
    from collections import defaultdict
    groups = defaultdict(list)

    for seq in seqs:
        csv_path = os.path.join(labels_dir, seq, "stage2_labels.csv")
        if not os.path.exists(csv_path):
            continue
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                key = (row["seq_name"], int(row["frame_id"]), int(row["det_id"]))
                feats = [safe_float(row.get(fn, 0.0), 0.0) for fn in HACA_PAIR_FEATURE_NAMES]
                group_label = int(row["label"])
                correct_candidate_rank = int(row.get("correct_candidate_rank", -1) or -1)
                groups[key].append({
                    "features": feats,
                    "rank": int(row.get("topk_rank", 0)),
                    "label": group_label,
                    "correct_candidate_rank": correct_candidate_rank,
                })

    if not groups:
        raise RuntimeError(f"No Stage 2 data found in {labels_dir}")

    all_features = []
    all_masks = []
    all_ranks = []
    all_actions = []

    for key, candidates in groups.items():
        candidates.sort(key=lambda c: c["rank"])
        k = min(len(candidates), max_k)
        feat = np.zeros((max_k, HACA_PAIR_FEATURE_DIM), dtype=np.float32)
        mask = np.zeros((max_k,), dtype=bool)
        for i in range(k):
            feat[i] = candidates[i]["features"]
            mask[i] = True

        # Target rank comes from group metadata, not per-candidate label.
        target_rank = int(candidates[0].get("correct_candidate_rank", -1))
        if target_rank >= k:
            target_rank = -1

        # Action label is group-level and should be identical within the group.
        action = int(candidates[0]["label"])

        all_features.append(feat)
        all_masks.append(mask)
        all_ranks.append(target_rank)
        all_actions.append(action)

    return (
        np.nan_to_num(np.stack(all_features), nan=0.0, posinf=0.0, neginf=0.0),
        np.stack(all_masks),
        np.array(all_ranks, dtype=np.int64),
        np.array(all_actions, dtype=np.int64),
    )


def evaluate(model, loader, device):
    model.eval()
    action_correct = 0
    total = 0
    action_preds = []
    action_targets = []

    with torch.no_grad():
        for feats, masks, ranks, actions in loader:
            feats, masks = feats.to(device), masks.to(device)
            ranks, actions = ranks.to(device), actions.to(device)
            scores, action_logits = model(feats, masks)
            preds = action_logits.argmax(dim=-1)
            action_correct += (preds == actions).sum().item()
            total += len(actions)
            action_preds.extend(preds.cpu().tolist())
            action_targets.extend(actions.cpu().tolist())

    acc = action_correct / max(total, 1)

    # Per-class accuracy
    names = ["rewrite", "defer", "reject"]
    per_class = {}
    for c in range(3):
        mask = np.array(action_targets) == c
        if mask.sum() > 0:
            per_class[f"{names[c]}_acc"] = round(float((np.array(action_preds)[mask] == c).mean()), 4)
        per_class[f"{names[c]}_count"] = int(mask.sum())

    return {"action_accuracy": acc, "total": total, **per_class}


def main():
    parser = argparse.ArgumentParser(description="Train RGSA Stage 2")
    parser.add_argument("--labels-dir", required=True)
    parser.add_argument("--train-seqs", nargs="+", default=["MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-09-FRCNN"])
    parser.add_argument("--val-seqs", nargs="+", default=["MOT17-05-FRCNN"])
    parser.add_argument("--max-k", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    validate_split(args.train_seqs, args.val_seqs)

    print(f"[load] Stage 2 training data from {args.labels_dir}")
    X_train, M_train, R_train, A_train = load_stage2_data(args.labels_dir, args.train_seqs, args.max_k)
    X_val, M_val, R_val, A_val = load_stage2_data(args.labels_dir, args.val_seqs, args.max_k)

    print(f"[data] train: {len(X_train)}, val: {len(X_val)}")
    print(f"[data] train actions: rewrite={sum(A_train==0)} defer={sum(A_train==1)} reject={sum(A_train==2)}")
    print(f"[data] val actions: rewrite={sum(A_val==0)} defer={sum(A_val==1)} reject={sum(A_val==2)}")
    print(f"[data] train valid ranks: {(R_train >= 0).sum()}, val valid ranks: {(R_val >= 0).sum()}")

    device = args.device

    # Datasets
    train_ds = TensorDataset(
        torch.tensor(X_train), torch.tensor(M_train),
        torch.tensor(R_train), torch.tensor(A_train)
    )
    val_ds = TensorDataset(
        torch.tensor(X_val), torch.tensor(M_val),
        torch.tensor(R_val), torch.tensor(A_val)
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = Stage2RecoveryHead(
        pair_dim=HACA_PAIR_FEATURE_DIM,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    # Compute class weights
    weights = compute_capped_class_weights(A_train)

    criterion = StagePairLoss(action_class_weights=weights).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = 0.0
    patience_counter = 0
    history = []

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for feats, masks, ranks, actions in train_loader:
            feats, masks = feats.to(device), masks.to(device)
            ranks, actions = ranks.to(device), actions.to(device)
            scores, action_logits = model(feats, masks)
            losses = criterion(scores, action_logits, ranks, actions)
            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += losses["total"].item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        val_metrics = evaluate(model, val_loader, device)

        record = {"epoch": epoch, "train_loss": avg_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(record)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"[epoch {epoch+1:3d}] loss={avg_loss:.4f} val_acc={val_metrics['action_accuracy']:.4f}")

        if val_metrics["action_accuracy"] > best_val_acc:
            best_val_acc = val_metrics["action_accuracy"]
            patience_counter = 0
            model.save_checkpoint(
                os.path.join(args.out_dir, "stage2_best.pt"),
                metadata={
                    "epoch": epoch,
                    "val_action_accuracy": val_metrics["action_accuracy"],
                    "train_seqs": args.train_seqs,
                    "val_seqs": args.val_seqs,
                    "feature_names": list(HACA_PAIR_FEATURE_NAMES),
                    "max_k": args.max_k,
                    "action_class_weights": weights,
                },
            )
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"[early stop] at epoch {epoch+1}, best val_acc={best_val_acc:.4f}")
                break

    model.save_checkpoint(
        os.path.join(args.out_dir, "stage2_final.pt"),
        metadata={
            "epochs_trained": epoch + 1,
            "best_val_action_acc": best_val_acc,
            "train_seqs": args.train_seqs,
            "val_seqs": args.val_seqs,
            "feature_names": list(HACA_PAIR_FEATURE_NAMES),
            "max_k": args.max_k,
            "action_class_weights": weights,
        },
    )
    with open(os.path.join(args.out_dir, "train_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    best_record = max(history, key=lambda r: r.get("val_action_accuracy", 0))
    with open(os.path.join(args.out_dir, "summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(best_record.keys()))
        writer.writeheader()
        writer.writerow(best_record)

    print(f"[done] best val_action_acc={best_val_acc:.4f}")


if __name__ == "__main__":
    main()
