#!/usr/bin/env python3
"""Evaluate RGSA Stage 1 deferral head offline.

Loads a trained Stage 1 checkpoint and evaluates on held-out data,
producing structured metrics: accept precision/recall, defer rate,
reject accuracy, and confusion matrix.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from models.rgsa_contract import STAGE1_FEATURE_DIM, STAGE1_FEATURE_NAMES
from models.rgsa_stage1_deferral import Stage1DeferralHead


def load_data(labels_dir, seqs):
    all_features, all_labels = [], []
    for seq in seqs:
        csv_path = os.path.join(labels_dir, seq, "stage1_labels.csv")
        if not os.path.exists(csv_path):
            continue
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                feats = [float(row.get(fn, 0.0)) for fn in STAGE1_FEATURE_NAMES]
                all_features.append(feats)
                all_labels.append(int(row["label"]))
    return np.array(all_features, dtype=np.float32), np.array(all_labels, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--labels-dir", required=True)
    parser.add_argument("--seqs", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    model = Stage1DeferralHead.from_checkpoint(args.checkpoint, device=args.device)
    X, y = load_data(args.labels_dir, args.seqs)
    print(f"[data] {len(X)} samples from {args.seqs}")

    x = torch.tensor(X, device=args.device)
    actions, probs = model.predict(x)
    preds = actions.cpu().numpy()

    # Confusion matrix
    cm = np.zeros((3, 3), dtype=int)
    for true, pred in zip(y, preds):
        cm[true][pred] += 1

    # Per-class metrics
    names = ["accept", "defer", "reject"]
    metrics = {}
    for c in range(3):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        metrics[f"{names[c]}_precision"] = round(prec, 4)
        metrics[f"{names[c]}_recall"] = round(rec, 4)
        metrics[f"{names[c]}_f1"] = round(f1, 4)
        metrics[f"{names[c]}_count"] = int(cm[c, :].sum())

    total = len(y)
    metrics["accuracy"] = round((preds == y).mean(), 4)
    metrics["defer_rate"] = round((preds == 1).mean(), 4)
    metrics["reject_rate"] = round((preds == 2).mean(), 4)
    metrics["accept_rate"] = round((preds == 0).mean(), 4)
    metrics["total"] = total
    metrics["confusion_matrix"] = cm.tolist()

    # Save
    with open(os.path.join(args.out_dir, "eval_results.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(args.out_dir, "summary.csv"), "w", newline="") as f:
        flat = {k: v for k, v in metrics.items() if k != "confusion_matrix"}
        writer = csv.DictWriter(f, fieldnames=list(flat.keys()))
        writer.writeheader()
        writer.writerow(flat)

    print(f"[results] accuracy={metrics['accuracy']}")
    for c in range(3):
        print(f"  {names[c]}: P={metrics[f'{names[c]}_precision']} R={metrics[f'{names[c]}_recall']} F1={metrics[f'{names[c]}_f1']} n={metrics[f'{names[c]}_count']}")
    print(f"  defer_rate={metrics['defer_rate']} reject_rate={metrics['reject_rate']}")
    print(f"[output] {args.out_dir}")


if __name__ == "__main__":
    main()
