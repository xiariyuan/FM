#!/usr/bin/env python3
"""Offline evaluation of heuristic verifier for RGSA Stage 2.

Tests heuristic rules against verifier labels and reports:
  - verifier_precision: confirm_local where top-1 is actually correct
  - confirm_coverage: confirm_local / total deferred
  - veto_correct_rate: veto_local where top-1 is indeed wrong
  - threshold sweep table
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from models.rgsa_contract import VERIFIER_FEATURE_NAMES
from models.rgsa_stage2_verifier import HeuristicVerifier


def load_verifier_data(labels_dir, seqs):
    """Load verifier labels + features."""
    all_features = []
    all_labels = []
    for seq in seqs:
        csv_path = os.path.join(labels_dir, seq, "verifier_labels.csv")
        if not os.path.exists(csv_path):
            continue
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                feats = []
                for fn in VERIFIER_FEATURE_NAMES:
                    v = row.get(fn, "")
                    try:
                        feats.append(float(v) if v != "" else 0.0)
                    except (ValueError, TypeError):
                        feats.append(0.0)
                all_features.append(feats)
                all_labels.append(int(row["label"]))
    return np.array(all_features, dtype=np.float32), np.array(all_labels, dtype=np.int64)


def evaluate_verifier(features, labels, verifier):
    """Evaluate verifier on data. Returns metrics dict."""
    confirm_correct = 0
    confirm_total = 0
    veto_correct = 0
    veto_total = 0

    for i in range(len(features)):
        signals = {fn: float(features[i, j]) for j, fn in enumerate(VERIFIER_FEATURE_NAMES)}
        action, _ = verifier.verify_single(signals)
        true_label = int(labels[i])  # 0=should confirm, 1=should veto

        if action == 0:  # confirmed
            confirm_total += 1
            if true_label == 0:
                confirm_correct += 1
        else:  # vetoed
            veto_total += 1
            if true_label == 1:
                veto_correct += 1

    total = len(labels)
    precision = confirm_correct / max(confirm_total, 1)
    coverage = confirm_total / max(total, 1)
    veto_correct_rate = veto_correct / max(veto_total, 1)

    return {
        "verifier_precision": round(precision, 4),
        "confirm_coverage": round(coverage, 4),
        "veto_correct_rate": round(veto_correct_rate, 4),
        "confirm_total": confirm_total,
        "veto_total": veto_total,
        "confirm_correct": confirm_correct,
        "veto_correct": veto_correct,
        "total": total,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate heuristic verifier offline")
    parser.add_argument("--labels-dir", required=True)
    parser.add_argument("--seqs", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sweep", action="store_true", help="Run threshold sweep")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    features, labels = load_verifier_data(args.labels_dir, args.seqs)
    print(f"[data] {len(features)} samples, confirm={sum(labels==0)}, veto={sum(labels==1)}")

    # Default verifier
    verifier = HeuristicVerifier()
    default_metrics = evaluate_verifier(features, labels, verifier)
    print(f"\n[default] precision={default_metrics['verifier_precision']} "
          f"coverage={default_metrics['confirm_coverage']} "
          f"veto_correct={default_metrics['veto_correct_rate']}")

    # Threshold sweep
    if args.sweep:
        sweep_results = []
        for s1_min in [0.6, 0.7, 0.8, 0.9]:
            for m1_min in [0.05, 0.1, 0.15, 0.2]:
                for e_max in [0.3, 0.5, 0.8]:
                    v = HeuristicVerifier(
                        rule1_s_final_min=s1_min,
                        rule1_margin_min=m1_min,
                        rule1_entropy_max=e_max,
                    )
                    metrics = evaluate_verifier(features, labels, v)
                    metrics["s1_min"] = s1_min
                    metrics["m1_min"] = m1_min
                    metrics["e_max"] = e_max
                    sweep_results.append(metrics)

        # Filter for precision >= 0.90
        good = [r for r in sweep_results if r["verifier_precision"] >= 0.90]
        good.sort(key=lambda r: -r["confirm_coverage"])

        print(f"\n[sweep] {len(sweep_results)} configs tested, {len(good)} with precision >= 0.90")
        if good:
            print(f"\n{'s1_min':>6} {'m1_min':>6} {'e_max':>6} {'precision':>10} {'coverage':>10} {'veto_cr':>8} {'confirm':>8} {'veto':>6}")
            for r in good[:20]:
                print(f"{r['s1_min']:>6.2f} {r['m1_min']:>6.2f} {r['e_max']:>6.2f} "
                      f"{r['verifier_precision']:>10.4f} {r['confirm_coverage']:>10.4f} "
                      f"{r['veto_correct_rate']:>8.4f} {r['confirm_total']:>8} {r['veto_total']:>6}")

        # Save sweep
        sweep_path = os.path.join(args.out_dir, "threshold_sweep.csv")
        if sweep_results:
            with open(sweep_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(sweep_results[0].keys()))
                writer.writeheader()
                writer.writerows(sweep_results)
            print(f"[saved] {sweep_path}")

    # Save default result
    with open(os.path.join(args.out_dir, "summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(default_metrics.keys()))
        writer.writeheader()
        writer.writerow(default_metrics)

    with open(os.path.join(args.out_dir, "eval_results.json"), "w") as f:
        json.dump({"default": default_metrics, "thresholds": verifier.get_thresholds()}, f, indent=2)

    print(f"[output] {args.out_dir}")


if __name__ == "__main__":
    main()
