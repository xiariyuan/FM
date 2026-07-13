#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

EXCLUDE_EXACT = {
    "seq",
    "track_a",
    "track_b",
    "same_gt",
    "quality_label_a",
    "quality_label_b",
    "validator_score",
    "combined_score",
    "base_score_saved",
    "aux_score_saved",
    "mul_score_saved",
}
EXCLUDE_PREFIX = ("dominant_gt",)
EXCLUDE_CONTAINS = ("matched_ratio", "quality_label")


def as_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def label(row):
    return int(float(row.get("same_gt", 0) or 0))


def read_rows(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [dict(r) for r in csv.DictReader(f)]


def infer_features(rows):
    if not rows:
        return []
    keys = list(rows[0].keys())
    feats = []
    for k in keys:
        if k in EXCLUDE_EXACT:
            continue
        if any(k.startswith(p) for p in EXCLUDE_PREFIX):
            continue
        if any(p in k for p in EXCLUDE_CONTAINS):
            continue
        ok = True
        seen = 0
        for r in rows[:200]:
            v = r.get(k, "")
            if v == "":
                continue
            seen += 1
            try:
                float(v)
            except Exception:
                ok = False
                break
        if ok and seen > 0:
            feats.append(k)
    return feats


def xy(rows, features):
    X = np.array([[as_float(r.get(k, 0.0)) for k in features] for r in rows], dtype=float)
    y = np.array([label(r) for r in rows], dtype=int)
    return X, y


def model_factory(kind):
    if kind == "logistic":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0)),
        ])
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=6,
            class_weight="balanced_subsample",
            random_state=31,
            n_jobs=-1,
        )
    return HistGradientBoostingClassifier(
        max_iter=350,
        learning_rate=0.04,
        max_leaf_nodes=15,
        l2_regularization=0.1,
        random_state=31,
    )


def proba(model, X):
    if len(X) == 0:
        return np.array([], dtype=float)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    raw = model.decision_function(X)
    return 1.0 / (1.0 + np.exp(-raw))


def topk(y, p, ks=(20, 50, 100, 200, 500)):
    out = {}
    order = np.argsort(-p)
    for k in ks:
        n = min(k, len(order))
        idx = order[:n]
        out[f"precision_at_{k}"] = float(y[idx].mean()) if n else 0.0
        out[f"tp_at_{k}"] = int(y[idx].sum()) if n else 0
    return out


def thresholds(y, p, ths=(0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)):
    out = {}
    for t in ths:
        m = p >= t
        n = int(m.sum())
        tp = int(y[m].sum()) if n else 0
        out[f"n_ge_{t}"] = n
        out[f"tp_ge_{t}"] = tp
        out[f"precision_ge_{t}"] = float(tp / n) if n else 0.0
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--base-min-score", type=float, default=0.05)
    ap.add_argument("--model-kind", choices=["hgb", "rf", "logistic"], default="hgb")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_all = read_rows(Path(args.input_csv))
    rows_region = [r for r in rows_all if as_float(r.get("aflink_score", 0.0)) >= args.base_min_score]
    features = infer_features(rows_region)
    seqs = sorted(set(r["seq"] for r in rows_all))

    oof_rows = []
    fold_reports = []
    for seq in seqs:
        train_rows = [r for r in rows_region if r["seq"] != seq]
        val_rows_all = [r for r in rows_all if r["seq"] == seq]
        val_rows_region = [r for r in val_rows_all if as_float(r.get("aflink_score", 0.0)) >= args.base_min_score]
        Xtr, ytr = xy(train_rows, features)
        Xva_all, _ = xy(val_rows_all, features)
        model = model_factory(args.model_kind)
        model.fit(Xtr, ytr)
        scores_all = proba(model, Xva_all)
        for row, s in zip(val_rows_all, scores_all):
            rr = dict(row)
            base = as_float(rr.get("aflink_score", 0.0))
            rr["validator_score"] = float(s)
            rr["combined_score"] = float(base * s)
            oof_rows.append(rr)
        region_scores = np.array([as_float(r["validator_score"]) for r in oof_rows[-len(val_rows_all):] if as_float(r.get("aflink_score", 0.0)) >= args.base_min_score], dtype=float)
        region_y = np.array([label(r) for r in val_rows_region], dtype=int)
        rep = {
            "val_seq": seq,
            "train_rows": len(train_rows),
            "train_pos": int(ytr.sum()),
            "val_region_rows": len(val_rows_region),
            "val_region_pos": int(region_y.sum()) if len(region_y) else 0,
        }
        if len(region_y) and len(set(region_y)) > 1:
            rep["roc_auc"] = float(roc_auc_score(region_y, region_scores))
            rep["pr_auc"] = float(average_precision_score(region_y, region_scores))
        else:
            rep["roc_auc"] = None
            rep["pr_auc"] = None
        rep.update(topk(region_y, region_scores))
        rep.update(thresholds(region_y, region_scores))
        fold_reports.append(rep)

    with (out_dir / "oof_validator_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(oof_rows[0].keys()))
        writer.writeheader()
        writer.writerows(oof_rows)

    X, y = xy(rows_region, features)
    final = model_factory(args.model_kind)
    final.fit(X, y)
    with (out_dir / "pair_validator_model.pkl").open("wb") as f:
        pickle.dump({"model": final, "features": features, "model_kind": args.model_kind, "base_min_score": args.base_min_score}, f)

    eval_rows = [r for r in oof_rows if as_float(r.get("aflink_score", 0.0)) >= args.base_min_score]
    y_o = np.array([label(r) for r in eval_rows], dtype=int)
    p_o = np.array([as_float(r.get("validator_score", 0.0)) for r in eval_rows], dtype=float)
    summary = {
        "model_kind": args.model_kind,
        "base_min_score": args.base_min_score,
        "rows_all": len(rows_all),
        "rows_region": len(rows_region),
        "positives_region": int(y.sum()),
        "positive_rate_region": float(y.mean()) if len(y) else 0.0,
        "num_features": len(features),
        "features": features,
        "folds": fold_reports,
    }
    if len(y_o) and len(set(y_o)) > 1:
        summary["oof_roc_auc"] = float(roc_auc_score(y_o, p_o))
        summary["oof_pr_auc"] = float(average_precision_score(y_o, p_o))
    summary.update(topk(y_o, p_o))
    summary.update(thresholds(y_o, p_o))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Pair Validator Auto Summary", "", f"model_kind: {args.model_kind}", f"base_min_score: {args.base_min_score}", f"rows_region: {len(rows_region)}", f"positives_region: {int(y.sum())}", f"positive_rate_region: {float(y.mean()) if len(y) else 0.0:.4f}", f"num_features: {len(features)}", f"oof_pr_auc: {summary.get('oof_pr_auc')}", f"oof_roc_auc: {summary.get('oof_roc_auc')}", "", "## Top-k / thresholds", "| metric | value |", "|---|---:|"]
    for key in ["precision_at_20", "tp_at_20", "precision_at_50", "tp_at_50", "precision_at_100", "tp_at_100", "precision_at_200", "tp_at_200", "n_ge_0.2", "tp_ge_0.2", "precision_ge_0.2", "n_ge_0.5", "tp_ge_0.5", "precision_ge_0.5", "n_ge_0.7", "tp_ge_0.7", "precision_ge_0.7"]:
        lines.append(f"| {key} | {summary.get(key)} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
