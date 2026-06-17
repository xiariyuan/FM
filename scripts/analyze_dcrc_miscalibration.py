#!/usr/bin/env python3
"""Phase 3.5: Conflict/ambiguity-conditioned miscalibration analysis.

Tests whether s_final reliability varies under different conflict complexity,
NOT just raw density. Focuses on features that capture local association ambiguity.

Key hypothesis: same s_final → different correctness under different conflict/ambiguity.
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def load_diagnostic_csv(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def safe_float(v, default=0.0):
    try:
        return float(v) if v not in ("", None) else default
    except (ValueError, TypeError):
        return default


def compute_calibration_metrics(scores, labels, n_bins=10):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (scores >= bin_edges[i]) & (scores < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        avg_score = float(scores[mask].mean())
        avg_label = float(labels[mask].mean())
        ece += abs(avg_score - avg_label) * len(scores[mask]) / len(scores)

    eps = 1e-7
    scores_clipped = np.clip(scores, eps, 1 - eps)
    brier = float(np.mean((scores - labels) ** 2))
    nll = float(-np.mean(labels * np.log(scores_clipped) + (1 - labels) * np.log(1 - scores_clipped)))
    return {"ece": round(ece, 6), "brier": round(brier, 6), "nll": round(nll, 6)}


def split_by_quantiles(values, n_buckets=4):
    """Split values into n_buckets by quantile, return bucket labels and edges."""
    values = np.asarray(values, dtype=np.float64)
    quantiles = [100 * i / n_buckets for i in range(1, n_buckets)]
    edges = np.percentile(values, quantiles)
    labels = np.digitize(values, edges)
    return labels, edges.tolist()


def analyze_feature(rows, feature_name, score_field="s_final", n_buckets=4):
    """Analyze whether a feature stably explains calibration quality."""
    scores = np.array([safe_float(r.get(score_field, 0)) for r in rows])
    labels = np.array([int(r.get("label_commit_ok", 0)) for r in rows])
    feature_vals = np.array([safe_float(r.get(feature_name, 0)) for r in rows])

    # Split by feature quantiles
    bucket_labels, edges = split_by_quantiles(feature_vals, n_buckets)

    results = []
    for b in range(n_buckets):
        mask = bucket_labels == b
        if mask.sum() < 20:
            continue
        b_scores = scores[mask]
        b_labels = labels[mask]
        metrics = compute_calibration_metrics(b_scores, b_labels)
        results.append({
            "feature": feature_name,
            "bucket": b,
            "edge_lo": round(edges[b - 1] if b > 0 else float(feature_vals.min()), 4),
            "edge_hi": round(edges[b] if b < len(edges) else float(feature_vals.max()), 4),
            "n": int(mask.sum()),
            "accuracy": round(float(b_labels.mean()), 4),
            **metrics,
        })

    # Compute spread: max accuracy - min accuracy across buckets
    if results:
        accs = [r["accuracy"] for r in results]
        eces = [r["ece"] for r in results]
        spread = {
            "feature": feature_name,
            "acc_spread": round(max(accs) - min(accs), 4),
            "ece_spread": round(max(eces) - min(eces), 4),
            "acc_max_bucket": int(np.argmax(accs)),
            "acc_min_bucket": int(np.argmin(accs)),
            "n_buckets": len(results),
        }
    else:
        spread = {"feature": feature_name, "acc_spread": 0, "ece_spread": 0}

    return results, spread


def main():
    parser = argparse.ArgumentParser(description="Phase 3.5: Conflict/ambiguity-conditioned analysis")
    parser.add_argument("--diagnostic-dir", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-buckets", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Features to analyze (conflict/ambiguity, not raw density)
    features = [
        "top1_top2_gap",
        "candidate_count",
        "local_iou_max",
        "local_iou_mean",
        "nearby_track_count",
        "margin",
        "entropy",
        "activation",
        "ambiguity_score",
        "bg_prob",
        "beta_hist",
        "det_score",
        "track_gap",
    ]

    all_spreads = []
    all_details = []

    for split in args.splits:
        csv_path = os.path.join(args.diagnostic_dir, f"{split}_diagnostic_edges.csv")
        if not os.path.exists(csv_path):
            print(f"[skip] {split}")
            continue

        rows = load_diagnostic_csv(csv_path)
        if not rows:
            continue

        print(f"\n{'='*60}")
        print(f"[{split}] n={len(rows)}")
        print(f"{'='*60}")

        global_acc = float(np.mean([int(r.get("label_commit_ok", 0)) for r in rows]))
        global_ece = compute_calibration_metrics(
            [safe_float(r.get("s_final", 0)) for r in rows],
            [int(r.get("label_commit_ok", 0)) for r in rows]
        )["ece"]
        print(f"Global: accuracy={global_acc:.4f} ECE={global_ece:.4f}")

        split_spreads = []
        for feat in features:
            details, spread = analyze_feature(rows, feat, n_buckets=args.n_buckets)
            spread["split"] = split
            all_spreads.append(spread)
            split_spreads.append(spread)

            for d in details:
                d["split"] = split
            all_details.extend(details)

            if details:
                print(f"\n  {feat}:")
                print(f"    acc_spread={spread['acc_spread']:.4f} ece_spread={spread['ece_spread']:.4f}")
                for d in details:
                    print(f"    bucket {d['bucket']}: [{d['edge_lo']:.2f}, {d['edge_hi']:.2f}] "
                          f"n={d['n']} acc={d['accuracy']:.4f} ECE={d['ece']:.4f}")

        # Rank features by acc_spread
        split_spreads.sort(key=lambda x: -x["acc_spread"])
        print(f"\n  [{split}] Top features by accuracy spread:")
        for s in split_spreads[:5]:
            print(f"    {s['feature']}: acc_spread={s['acc_spread']:.4f} ece_spread={s['ece_spread']:.4f}")

    # Save all results
    if all_spreads:
        # Write spread summary
        spread_path = os.path.join(args.out_dir, "feature_spread_summary.csv")
        with open(spread_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_spreads[0].keys()))
            w.writeheader()
            w.writerows(all_spreads)
        print(f"\n[saved] {spread_path}")

    if all_details:
        detail_path = os.path.join(args.out_dir, "by_feature_bucket.csv")
        with open(detail_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_details[0].keys()))
            w.writeheader()
            w.writerows(all_details)
        print(f"[saved] {detail_path}")

    # Write summary: which features are stable across splits?
    feature_stability = {}
    for s in all_spreads:
        feat = s["feature"]
        if feat not in feature_stability:
            feature_stability[feat] = {"splits": [], "acc_spreads": [], "ece_spreads": []}
        feature_stability[feat]["splits"].append(s.get("split", ""))
        feature_stability[feat]["acc_spreads"].append(s.get("acc_spread", 0))
        feature_stability[feat]["ece_spreads"].append(s.get("ece_spread", 0))

    stability_path = os.path.join(args.out_dir, "feature_stability.csv")
    with open(stability_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["feature", "mean_acc_spread", "mean_ece_spread", "min_acc_spread", "splits"])
        for feat, info in sorted(feature_stability.items(), key=lambda x: -np.mean(x[1]["acc_spreads"])):
            w.writerow([
                feat,
                round(np.mean(info["acc_spreads"]), 4),
                round(np.mean(info["ece_spreads"]), 4),
                round(min(info["acc_spreads"]), 4),
                "+".join(info["splits"]),
            ])
    print(f"[saved] {stability_path}")

    # Print cross-split stability verdict
    print(f"\n{'='*60}")
    print("[VERDICT] Features with stable acc_spread > 0.05 across all splits:")
    for feat, info in sorted(feature_stability.items(), key=lambda x: -np.mean(x[1]["acc_spreads"])):
        min_spread = min(info["acc_spreads"])
        mean_spread = np.mean(info["acc_spreads"])
        if min_spread > 0.03 and mean_spread > 0.05:
            print(f"  STABLE: {feat} (mean={mean_spread:.4f}, min={min_spread:.4f})")
        elif mean_spread > 0.05:
            print(f"  PARTIAL: {feat} (mean={mean_spread:.4f}, min={min_spread:.4f})")

    print(f"\n[output] {args.out_dir}")


if __name__ == "__main__":
    main()
