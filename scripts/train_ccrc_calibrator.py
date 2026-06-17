#!/usr/bin/env python3
"""Train and evaluate CCRC calibrators.

Trains three calibrators in order:
1. Global temperature scaling
2. Feature-binned calibration (per stable feature)
3. Small MLP calibrator

Validates with within-score residual gap analysis:
  In fixed s_final bucket, check if conflict/ambiguity bucket
  calibration gap is reduced after calibration.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from models.ccrc_calibrators import (
    CCRC_FEATURE_DIM,
    CCRC_FEATURE_NAMES,
    GlobalTemperatureScaling,
    PlattScaling,
    FeatureBinnedCalibrator,
    MLP_Calibrator,
    MultiFeatureBinnedCalibrator,
)


def load_diagnostic_csv(path, score_field="s_final", label_field="label_commit_ok"):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    scores = np.array([float(r.get(score_field, 0) or 0) for r in rows], dtype=np.float32)
    labels = np.array([int(r.get(label_field, 0)) for r in rows], dtype=np.int64)
    features = np.array([[float(r.get(fn, 0) or 0) for fn in CCRC_FEATURE_NAMES] for r in rows], dtype=np.float32)
    return scores, labels, features, rows


def compute_metrics(scores, labels, n_bins=10):
    """ECE, Brier, NLL."""
    scores = np.clip(scores.astype(np.float64), 1e-7, 1 - 1e-7)
    labels = labels.astype(np.float64)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (scores >= bin_edges[i]) & (scores < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        ece += abs(scores[mask].mean() - labels[mask].mean()) * mask.sum() / len(scores)
    brier = float(np.mean((scores - labels) ** 2))
    nll = float(-np.mean(labels * np.log(scores) + (1 - labels) * np.log(1 - scores)))
    return {"ece": round(ece, 6), "brier": round(brier, 6), "nll": round(nll, 6)}


def within_score_residual_gap(scores_before, scores_after, labels, feature_vals,
                               n_score_bins=4, n_feat_bins=4):
    """Compute within-score-bucket residual calibration gap before/after.

    For each s_final bucket, split by feature, compute per-bucket ECE.
    Return the mean and max reduction in ECE spread across feature buckets.
    """
    score_edges = np.percentile(scores_before, [25, 50, 75])
    score_bins = np.digitize(scores_before, score_edges)

    results = []
    for sb in range(n_score_bins):
        smask = score_bins == sb
        if smask.sum() < 20:
            continue
        feat_edges = np.percentile(feature_vals[smask], [25, 50, 75])
        feat_bins = np.digitize(feature_vals[smask], feat_edges)

        ece_before_list = []
        ece_after_list = []
        for fb in range(n_feat_bins):
            fmask = feat_bins == fb
            if fmask.sum() < 10:
                continue
            s_b = scores_before[smask][fmask]
            s_a = scores_after[smask][fmask]
            l = labels[smask][fmask]
            eb = compute_metrics(s_b, l, n_bins=5)["ece"]
            ea = compute_metrics(s_a, l, n_bins=5)["ece"]
            ece_before_list.append(eb)
            ece_after_list.append(ea)

        if ece_before_list:
            spread_before = max(ece_before_list) - min(ece_before_list)
            spread_after = max(ece_after_list) - min(ece_after_list)
            results.append({
                "score_bin": sb,
                "spread_before": round(spread_before, 6),
                "spread_after": round(spread_after, 6),
                "spread_reduction": round(spread_before - spread_after, 6),
            })

    return results


def train_platt_scaling(scores, labels, lr=0.01, epochs=200):
    model = PlattScaling()
    optimizer = torch.optim.LBFGS([model.a, model.b], lr=lr, max_iter=20)
    s = torch.tensor(scores, dtype=torch.float32)
    l = torch.tensor(labels, dtype=torch.float32)

    def closure():
        optimizer.zero_grad()
        pred = model(s)
        loss = nn.functional.binary_cross_entropy(pred, l)
        loss.backward()
        return loss

    for epoch in range(epochs):
        optimizer.step(closure)

    a, b = float(model.a), float(model.b)
    print(f"  Platt: a={a:.4f} b={b:.4f}")
    return model


def train_global_temperature(scores, labels, lr=0.01, epochs=200):
    model = GlobalTemperatureScaling()
    optimizer = torch.optim.LBFGS([model.temperature], lr=lr, max_iter=20)
    s = torch.tensor(scores, dtype=torch.float32)
    l = torch.tensor(labels, dtype=torch.float32)

    def closure():
        optimizer.zero_grad()
        pred = model(s)
        loss = nn.functional.binary_cross_entropy(pred, l)
        loss.backward()
        return loss

    for epoch in range(epochs):
        optimizer.step(closure)

    t = float(model.temperature)
    print(f"  Global TS: temperature={t:.4f}")
    return model


def train_mlp_calibrator(features, labels, val_features, val_labels,
                          epochs=50, batch_size=512, lr=1e-3, device="cpu"):
    model = MLP_Calibrator(input_dim=features.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    X = torch.tensor(features, dtype=torch.float32).to(device)
    y = torch.tensor(labels, dtype=torch.float32).to(device)
    X_val = torch.tensor(val_features, dtype=torch.float32).to(device)
    y_val = torch.tensor(val_labels, dtype=torch.float32).to(device)

    best_val_loss = float("inf")
    best_state = None
    patience = 5
    bad = 0

    for epoch in range(epochs):
        model.train()
        # Simple batched training
        perm = torch.randperm(len(X))
        total_loss = 0
        n = 0
        for i in range(0, len(X), batch_size):
            idx = perm[i:i+batch_size]
            pred = model(X[idx])
            loss = nn.functional.binary_cross_entropy(pred, y[idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
            n += 1

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = nn.functional.binary_cross_entropy(val_pred, y_val).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"  MLP: early stop at epoch {epoch+1}, best val_loss={best_val_loss:.4f}")
                break

    if best_state:
        model.load_state_dict(best_state)
    print(f"  MLP: trained {epoch+1} epochs, best val_loss={best_val_loss:.4f}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train CCRC calibrators")
    parser.add_argument("--train-csv", required=True, help="Training diagnostic_edges.csv")
    parser.add_argument("--val-csv", required=True, help="Validation diagnostic_edges.csv")
    parser.add_argument("--test-csv", required=True, help="Test diagnostic_edges.csv")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load data
    print("[load] data...")
    train_scores, train_labels, train_feats, _ = load_diagnostic_csv(args.train_csv)
    val_scores, val_labels, val_feats, _ = load_diagnostic_csv(args.val_csv)
    test_scores, test_labels, test_feats, _ = load_diagnostic_csv(args.test_csv)

    print(f"  train: {len(train_labels)} (acc={train_labels.mean():.4f})")
    print(f"  val: {len(val_labels)} (acc={val_labels.mean():.4f})")
    print(f"  test: {len(test_labels)} (acc={test_labels.mean():.4f})")

    # Baseline (uncalibrated)
    baseline_train = compute_metrics(train_scores, train_labels)
    baseline_val = compute_metrics(val_scores, val_labels)
    baseline_test = compute_metrics(test_scores, test_labels)
    print(f"\n[baseline] train ECE={baseline_train['ece']:.4f} val ECE={baseline_val['ece']:.4f} test ECE={baseline_test['ece']:.4f}")

    all_results = [{"method": "uncalibrated", "split": "train", **baseline_train},
                   {"method": "uncalibrated", "split": "val", **baseline_val},
                   {"method": "uncalibrated", "split": "test", **baseline_test}]

    # 1. Global temperature scaling (logit-space)
    print("\n[1] Global temperature scaling (logit-space)...")
    global_ts = train_global_temperature(train_scores, train_labels)
    for split, scores, labels in [("train", train_scores, train_labels),
                                   ("val", val_scores, val_labels),
                                   ("test", test_scores, test_labels)]:
        calibrated = global_ts.predict_numpy(scores)
        m = compute_metrics(calibrated, labels)
        all_results.append({"method": "global_ts", "split": split, **m})
        print(f"  {split}: ECE={m['ece']:.4f} Brier={m['brier']:.4f} NLL={m['nll']:.4f}")

    # 1b. Platt scaling
    print("\n[1b] Platt scaling...")
    platt = train_platt_scaling(train_scores, train_labels)
    for split, scores, labels in [("train", train_scores, train_labels),
                                   ("val", val_scores, val_labels),
                                   ("test", test_scores, test_labels)]:
        calibrated = platt.predict_numpy(scores)
        m = compute_metrics(calibrated, labels)
        all_results.append({"method": "platt", "split": split, **m})
        print(f"  {split}: ECE={m['ece']:.4f} Brier={m['brier']:.4f} NLL={m['nll']:.4f}")

    # 2. Feature-binned calibration (for top stable features)
    stable_features = ["top1_top2_gap", "margin", "candidate_count", "beta_hist", "ambiguity_score"]
    feat_idx = {fn: i for i, fn in enumerate(CCRC_FEATURE_NAMES)}

    for feat_name in stable_features:
        if feat_name not in feat_idx:
            continue
        print(f"\n[2] Feature-binned calibration: {feat_name}...")
        fbc = FeatureBinnedCalibrator(feat_name, n_buckets=4)
        fbc.fit(train_scores, train_feats[:, feat_idx[feat_name]], train_labels)
        print(f"  temperatures: {fbc.temperatures}")
        print(f"  counts: {fbc.bucket_counts}")

        for split, scores, feats, labels in [("train", train_scores, train_feats, train_labels),
                                              ("val", val_scores, val_feats, val_labels),
                                              ("test", test_scores, test_feats, test_labels)]:
            calibrated = fbc.predict(scores, feats[:, feat_idx[feat_name]])
            m = compute_metrics(calibrated, labels)
            all_results.append({"method": f"fbc_{feat_name}", "split": split, **m})
            print(f"  {split}: ECE={m['ece']:.4f} Brier={m['brier']:.4f}")

    # 3. Multi-feature binned calibration
    print("\n[3] Multi-feature binned calibration (margin + top1_top2_gap + beta_hist)...")
    mfbc = MultiFeatureBinnedCalibrator(
        feature_names=["margin", "top1_top2_gap", "beta_hist"],
        n_buckets_per_feature=3,
    )
    mfbc.fit(train_scores, train_feats, list(CCRC_FEATURE_NAMES), train_labels)
    print(f"  n_buckets_filled: {mfbc.get_params()['n_buckets_filled']}")

    for split, scores, feats, labels in [("train", train_scores, train_feats, train_labels),
                                          ("val", val_scores, val_feats, val_labels),
                                          ("test", test_scores, test_feats, test_labels)]:
        calibrated = mfbc.predict(scores, feats, list(CCRC_FEATURE_NAMES))
        m = compute_metrics(calibrated, labels)
        all_results.append({"method": "multi_fbc", "split": split, **m})
        print(f"  {split}: ECE={m['ece']:.4f} Brier={m['brier']:.4f}")

    # 4. MLP calibrator
    print(f"\n[4] MLP calibrator (input_dim={CCRC_FEATURE_DIM})...")
    mlp = train_mlp_calibrator(
        train_feats, train_labels,
        val_feats, val_labels,
        epochs=args.epochs, device=args.device,
    )
    for split, feats, labels in [("train", train_feats, train_labels),
                                  ("val", val_feats, val_labels),
                                  ("test", test_feats, test_labels)]:
        calibrated = mlp.predict_numpy(feats)
        m = compute_metrics(calibrated, labels)
        all_results.append({"method": "mlp", "split": split, **m})
        print(f"  {split}: ECE={m['ece']:.4f} Brier={m['brier']:.4f}")

    mlp.save_checkpoint(os.path.join(args.out_dir, "mlp_calibrator.pt"),
                        metadata={"feature_names": list(CCRC_FEATURE_NAMES)})

    # 4b. MLP_no_score ablation (conflict features only, no s_final)
    # Features without s_final (index 0): all others
    no_score_indices = list(range(1, CCRC_FEATURE_DIM))
    no_score_dim = len(no_score_indices)
    print(f"\n[4b] MLP_no_score (input_dim={no_score_dim}, conflict features only)...")
    mlp_ns = train_mlp_calibrator(
        train_feats[:, no_score_indices], train_labels,
        val_feats[:, no_score_indices], val_labels,
        epochs=args.epochs, device=args.device,
    )
    for split, feats, labels in [("train", train_feats, train_labels),
                                  ("val", val_feats, val_labels),
                                  ("test", test_feats, test_labels)]:
        calibrated = mlp_ns.predict_numpy(feats[:, no_score_indices])
        m = compute_metrics(calibrated, labels)
        all_results.append({"method": "mlp_no_score", "split": split, **m})
        print(f"  {split}: ECE={m['ece']:.4f} Brier={m['brier']:.4f}")

    # 5. Within-score residual gap analysis (core CCRC validation)
    print("\n[5] Within-score residual gap analysis...")
    gap_results = []

    # Test with best calibrators
    test_cal_global = global_ts.predict_numpy(test_scores)
    test_cal_platt = platt.predict_numpy(test_scores)
    test_cal_mlp = mlp.predict_numpy(test_feats)
    test_cal_mlp_ns = mlp_ns.predict_numpy(test_feats[:, no_score_indices])

    for feat_name in ["margin", "top1_top2_gap", "beta_hist", "candidate_count", "ambiguity_score"]:
        if feat_name not in feat_idx:
            continue
        feat_vals = test_feats[:, feat_idx[feat_name]]

        # Global TS
        gaps_global = within_score_residual_gap(test_scores, test_cal_global, test_labels, feat_vals)
        for g in gaps_global:
            g["method"] = "global_ts"
            g["conditioning_feature"] = feat_name
            gap_results.append(g)

        # Platt
        gaps_platt = within_score_residual_gap(test_scores, test_cal_platt, test_labels, feat_vals)
        for g in gaps_platt:
            g["method"] = "platt"
            g["conditioning_feature"] = feat_name
            gap_results.append(g)

        # MLP full
        gaps_mlp = within_score_residual_gap(test_scores, test_cal_mlp, test_labels, feat_vals)
        for g in gaps_mlp:
            g["method"] = "mlp"
            g["conditioning_feature"] = feat_name
            gap_results.append(g)

        # MLP no_score
        gaps_mlp_ns = within_score_residual_gap(test_scores, test_cal_mlp_ns, test_labels, feat_vals)
        for g in gaps_mlp_ns:
            g["method"] = "mlp_no_score"
            g["conditioning_feature"] = feat_name
            gap_results.append(g)

        # Uncalibrated baseline
        gaps_base = within_score_residual_gap(test_scores, test_scores, test_labels, feat_vals)
        for g in gaps_base:
            g["method"] = "uncalibrated"
            g["conditioning_feature"] = feat_name
            gap_results.append(g)

        mean_red_global = np.mean([g["spread_reduction"] for g in gaps_global])
        mean_red_platt = np.mean([g["spread_reduction"] for g in gaps_platt])
        mean_red_mlp = np.mean([g["spread_reduction"] for g in gaps_mlp])
        mean_red_mlp_ns = np.mean([g["spread_reduction"] for g in gaps_mlp_ns])
        print(f"  {feat_name}: global_ts={mean_red_global:.6f} platt={mean_red_platt:.6f} mlp={mean_red_mlp:.6f} mlp_no_score={mean_red_mlp_ns:.6f}")

    # Save all results
    if all_results:
        summary_path = os.path.join(args.out_dir, "summary.csv")
        with open(summary_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            w.writeheader()
            w.writerows(all_results)
        print(f"\n[saved] {summary_path}")

    if gap_results:
        gap_path = os.path.join(args.out_dir, "within_score_residual_gap.csv")
        with open(gap_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(gap_results[0].keys()))
            w.writeheader()
            w.writerows(gap_results)
        print(f"[saved] {gap_path}")

    # Overall verdict
    print("\n" + "=" * 60)
    print("[VERDICT]")
    # Compare best method vs uncalibrated on test
    test_uncal = [r for r in all_results if r["method"] == "uncalibrated" and r["split"] == "test"][0]
    test_methods = [r for r in all_results if r["split"] == "test" and r["method"] != "uncalibrated"]
    best = min(test_methods, key=lambda r: r["ece"])
    print(f"  Best method on test: {best['method']}")
    print(f"    ECE: {test_uncal['ece']:.4f} -> {best['ece']:.4f} (delta={best['ece']-test_uncal['ece']:.4f})")
    print(f"    Brier: {test_uncal['brier']:.4f} -> {best['brier']:.4f} (delta={best['brier']-test_uncal['brier']:.4f})")
    print(f"    NLL: {test_uncal['nll']:.4f} -> {best['nll']:.4f} (delta={best['nll']-test_uncal['nll']:.4f})")

    # Within-score gap improvement
    if gap_results:
        mlp_gaps = [g for g in gap_results if g["method"] == "mlp"]
        base_gaps = [g for g in gap_results if g["method"] == "uncalibrated"]
        mlp_mean = np.mean([g["spread_reduction"] for g in mlp_gaps]) if mlp_gaps else 0
        print(f"\n  Within-score residual gap (MLP): mean_reduction={mlp_mean:.6f}")
        if mlp_mean > 0:
            print(f"  MLP calibrator REDUCES within-score conflict variance")
        else:
            print(f"  MLP calibrator does NOT reduce within-score conflict variance")

    print(f"\n[output] {args.out_dir}")


if __name__ == "__main__":
    main()
