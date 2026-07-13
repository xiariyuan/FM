#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--summary-json", default=None)
    args = ap.parse_args()

    with open(args.model, "rb") as f:
        payload = pickle.load(f)
    model = payload["model"]
    features = payload["features"]

    rows = []
    with open(args.pairs, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(dict(row))

    X = np.array([[float(r.get(k, 0.0) or 0.0) for k in features] for r in rows], dtype=float)
    if len(rows) == 0:
        scores = np.array([], dtype=float)
    elif hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)[:, 1]
    else:
        raw = model.decision_function(X)
        scores = 1.0 / (1.0 + np.exp(-raw))

    for r, s in zip(rows, scores):
        r["aflink_score"] = float(s)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        out_csv.write_text("", encoding="utf-8")

    summary = {
        "rows": len(rows),
        "model": args.model,
        "features": features,
        "score_mean": float(scores.mean()) if len(scores) else 0.0,
        "score_max": float(scores.max()) if len(scores) else 0.0,
        "score_min": float(scores.min()) if len(scores) else 0.0,
        "n_ge_0_2": int((scores >= 0.2).sum()) if len(scores) else 0,
        "n_ge_0_15": int((scores >= 0.15).sum()) if len(scores) else 0,
        "n_ge_0_1": int((scores >= 0.1).sum()) if len(scores) else 0,
        "n_ge_0_05": int((scores >= 0.05).sum()) if len(scores) else 0,
    }
    if args.summary_json:
        Path(args.summary_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
