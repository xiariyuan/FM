#!/usr/bin/env python3
"""Evaluate RGSA Stage 2 local recovery head offline."""

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

from models.rgsa_contract import HACA_PAIR_FEATURE_DIM, HACA_PAIR_FEATURE_NAMES
from models.rgsa_stage2_recovery import Stage2RecoveryHead
from scripts.train_rgsa_stage2 import load_stage2_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--labels-dir", required=True)
    parser.add_argument("--seqs", nargs="+", required=True)
    parser.add_argument("--max-k", type=int, default=5)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    model = Stage2RecoveryHead.from_checkpoint(args.checkpoint, device=args.device)
    X, M, R, A = load_stage2_data(args.labels_dir, args.seqs, args.max_k)
    print(f"[data] {len(X)} samples")

    x = torch.tensor(X, device=args.device)
    m = torch.tensor(M, device=device if (device := args.device) else "cpu")
    scores, actions, probs = model.predict(x, m)
    preds = actions.cpu().numpy()

    names = ["rewrite", "defer", "reject"]
    metrics = {"accuracy": round(float((preds == A).mean()), 4), "total": len(A)}
    for c in range(3):
        mask = A == c
        if mask.sum() > 0:
            metrics[f"{names[c]}_precision"] = round(float((preds[mask] == c).mean()), 4)
        metrics[f"{names[c]}_count"] = int(mask.sum())
        metrics[f"{names[c]}_pred_count"] = int((preds == c).sum())

    with open(os.path.join(args.out_dir, "eval_results.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(args.out_dir, "summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)

    print(f"[results] accuracy={metrics['accuracy']}")
    for c in range(3):
        print(f"  {names[c]}: n={metrics[f'{names[c]}_count']} pred={metrics[f'{names[c]}_pred_count']}")
    print(f"[output] {args.out_dir}")


if __name__ == "__main__":
    main()
