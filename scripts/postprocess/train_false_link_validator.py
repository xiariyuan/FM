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

FEATURES = [
    "aflink_score",
    "gap",
    "center_distance",
    "center_distance_per_frame",
    "predicted_distance",
    "predicted_distance_per_frame",
    "velocity_cosine",
    "height_ratio",
    "area_ratio",
    "bottom_y_gap",
    "len_a",
    "len_b",
    "duration_a",
    "duration_b",
    "avg_score_a",
    "avg_score_b",
    "last_score_a",
    "first_score_b",
    "cos_end_start",
    "cos_end_global",
    "cos_global_start",
    "cos_global_global",
    "cos_high_high",
    "cos_start_start",
    "cos_end_end",
    "appearance_mean",
    "appearance_max",
    "appearance_min",
    "appearance_std",
    "appearance_gap_consistency",
    "has_reid",
    "rank_as_successor",
    "rank_as_predecessor",
    "score_margin_for_a",
    "score_margin_for_b",
    "min_score_margin",
    "is_mutual_top1",
    "is_mutual_top2",
]


def read_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(dict(row))
    return rows


def f(row, key):
    return float(row.get(key, 0.0) or 0.0)


def yval(row):
    return int(float(row.get("same_gt", 0) or 0))


def make_xy(rows):
    X = np.array([[f(r, key) for key in FEATURES] for r in rows], dtype=float)
    y = np.array([yval(r) for r in rows], dtype=int)
    return X, y


def model_factory(kind: str):
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
            random_state=23,
            n_jobs=-1,
        )
    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=0.1,
        random_state=23,
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


def th_metrics(y, p, ths=(0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)):
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
    rows = [r for r in rows_all if f(r, "aflink_score") >= args.base_min_score]
    seqs = sorted(set(r["seq"] for r in rows_all))

    fold_reports = []
    scored_rows = []
    for seq in seqs:
        train_rows = [r for r in rows if r["seq"] != seq]
        val_rows_all = [r for r in rows_all if r["seq"] == seq]
        val_rows_eval = [r for r in val_rows_all if f(r, "aflink_score") >= args.base_min_score]
        Xtr, ytr = make_xy(train_rows)
        Xva_all, _ = make_xy(val_rows_all)
        model = model_factory(args.model_kind)
        model.fit(Xtr, ytr)
        p_all = proba(model, Xva_all)
        for row, score in zip(val_rows_all, p_all):
            rr = dict(row)
            rr["validator_score"] = float(score)
            rr["combined_score"] = float(score) * f(row, "aflink_score")
            scored_rows.append(rr)
        if val_rows_eval:
            p_eval = np.array([float(scored_rows[-len(val_rows_all)+i]["validator_score"]) for i, r in enumerate(val_rows_all) if f(r, "aflink_score") >= args.base_min_score], dtype=float)
            y_eval = np.array([yval(r) for r in val_rows_eval], dtype=int)
        else:
            p_eval = np.array([], dtype=float)
            y_eval = np.array([], dtype=int)
        rep = {
            "val_seq": seq,
            "train_rows": len(train_rows),
            "train_pos": int(ytr.sum()),
            "val_eval_rows": len(val_rows_eval),
            "val_eval_pos": int(y_eval.sum()) if len(y_eval) else 0,
        }
        if len(y_eval) and len(set(y_eval)) > 1:
            rep["roc_auc"] = float(roc_auc_score(y_eval, p_eval))
            rep["pr_auc"] = float(average_precision_score(y_eval, p_eval))
        else:
            rep["roc_auc"] = None
            rep["pr_auc"] = None
        rep.update(topk(y_eval, p_eval))
        rep.update(th_metrics(y_eval, p_eval))
        fold_reports.append(rep)

    fields = list(scored_rows[0].keys()) if scored_rows else []
    with (out_dir / "oof_validator_predictions.csv").open("w", newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fields)
        writer.writeheader()
        writer.writerows(scored_rows)

    X, y = make_xy(rows)
    final = model_factory(args.model_kind)
    final.fit(X, y)
    with (out_dir / "false_link_validator_model.pkl").open("wb") as fmodel:
        pickle.dump({"model": final, "features": FEATURES, "model_kind": args.model_kind, "base_min_score": args.base_min_score}, fmodel)

    eval_rows = [r for r in scored_rows if f(r, "aflink_score") >= args.base_min_score]
    y_o = np.array([yval(r) for r in eval_rows], dtype=int)
    p_o = np.array([f(r, "validator_score") for r in eval_rows], dtype=float)
    summary = {
        "model_kind": args.model_kind,
        "base_min_score": args.base_min_score,
        "rows_all": len(rows_all),
        "rows_train_region": len(rows),
        "positives_train_region": int(y.sum()),
        "positive_rate_train_region": float(y.mean()) if len(y) else 0.0,
        "features": FEATURES,
        "folds": fold_reports,
    }
    if len(y_o) and len(set(y_o)) > 1:
        summary["oof_roc_auc"] = float(roc_auc_score(y_o, p_o))
        summary["oof_pr_auc"] = float(average_precision_score(y_o, p_o))
    summary.update(topk(y_o, p_o))
    summary.update(th_metrics(y_o, p_o))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# False-link Validator Summary",
        "",
        f"model_kind: {args.model_kind}",
        f"base_min_score: {args.base_min_score}",
        f"rows_train_region: {len(rows)}",
        f"positives_train_region: {int(y.sum())}",
        f"positive_rate_train_region: {float(y.mean()) if len(y) else 0.0:.4f}",
        f"oof_pr_auc: {summary.get('oof_pr_auc')}",
        f"oof_roc_auc: {summary.get('oof_roc_auc')}",
        "",
        "## Top-k / thresholds",
        "| metric | value |",
        "|---|---:|",
    ]
    for key in ["precision_at_20", "tp_at_20", "precision_at_50", "tp_at_50", "precision_at_100", "tp_at_100", "precision_at_200", "tp_at_200", "n_ge_0.5", "tp_ge_0.5", "precision_ge_0.5", "n_ge_0.7", "tp_ge_0.7", "precision_ge_0.7", "n_ge_0.9", "tp_ge_0.9", "precision_ge_0.9"]:
        lines.append(f"| {key} | {summary.get(key)} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
