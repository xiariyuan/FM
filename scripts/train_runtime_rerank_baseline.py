#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train a simple runtime replay rerank baseline on labeled candidate rows.")
    ap.add_argument("--input-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="logistic", choices=["logistic", "gbdt"])
    ap.add_argument("--rank-score-col", default="refined_score", choices=["base_score", "refined_score"])
    ap.add_argument("--train-seqs", default="", help="Comma-separated sequence names for training.")
    ap.add_argument("--val-seqs", default="", help="Comma-separated sequence names for validation.")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--valid-only", action="store_true", help="Use only rows with valid_train_row=1.")
    ap.add_argument("--apply-on", default="all", choices=["all", "ambiguous"], help="Apply rerank on all positive groups or only on ambiguous groups.")
    return ap.parse_args()


def _parse_seq_set(raw: str) -> set[str]:
    items = [x.strip() for x in str(raw or "").split(",") if x.strip()]
    return set(items)


def _safe_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return float(default)


def _safe_int(row: dict[str, str], key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default)))
    except Exception:
        return int(default)


def _box_iou_cxcywh(det: tuple[float, float, float, float], trk: tuple[float, float, float, float]) -> float:
    dcx, dcy, dw, dh = det
    tcx, tcy, tw, th = trk
    dx1, dy1, dx2, dy2 = dcx - 0.5 * dw, dcy - 0.5 * dh, dcx + 0.5 * dw, dcy + 0.5 * dh
    tx1, ty1, tx2, ty2 = tcx - 0.5 * tw, tcy - 0.5 * th, tcx + 0.5 * tw, tcy + 0.5 * th
    ix1, iy1 = max(dx1, tx1), max(dy1, ty1)
    ix2, iy2 = min(dx2, tx2), min(dy2, ty2)
    iw, ih = max(ix2 - ix1, 0.0), max(iy2 - iy1, 0.0)
    inter = iw * ih
    union = max(dw * dh, 0.0) + max(tw * th, 0.0) - inter
    if union <= 1e-8:
        return 0.0
    return float(inter / union)


def _build_feature(row: dict[str, str], rank_score_col: str) -> list[float]:
    det_cx = _safe_float(row, "det_cx")
    det_cy = _safe_float(row, "det_cy")
    det_w = max(_safe_float(row, "det_w"), 1e-6)
    det_h = max(_safe_float(row, "det_h"), 1e-6)
    trk_cx = _safe_float(row, "track_cx")
    trk_cy = _safe_float(row, "track_cy")
    trk_w = max(_safe_float(row, "track_w"), 1e-6)
    trk_h = max(_safe_float(row, "track_h"), 1e-6)
    dx = (det_cx - trk_cx) / det_w
    dy = (det_cy - trk_cy) / det_h
    dw = math.log(det_w / trk_w)
    dh = math.log(det_h / trk_h)
    area_ratio = math.log((det_w * det_h) / max(trk_w * trk_h, 1e-6))
    group_size = max(_safe_int(row, "group_size", 1), 1)
    track_rank = _safe_int(row, "track_rank", 1)
    rank_frac = 1.0 - float(track_rank - 1) / float(max(group_size - 1, 1))
    return [
        _safe_float(row, rank_score_col),
        _safe_float(row, "base_score"),
        _safe_float(row, "refined_score"),
        _safe_float(row, "motion_score"),
        _safe_float(row, "det_score"),
        math.log1p(max(_safe_float(row, "track_gap"), 0.0)),
        math.log1p(max(_safe_float(row, "track_hist_len"), 0.0)),
        _safe_float(row, "base_margin"),
        _safe_float(row, "refined_margin"),
        _safe_float(row, "rank_margin"),
        _safe_float(row, "rank_entropy"),
        rank_frac,
        dx,
        dy,
        dw,
        dh,
        area_ratio,
        _box_iou_cxcywh((det_cx, det_cy, det_w, det_h), (trk_cx, trk_cy, trk_w, trk_h)),
    ]


def _load_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "group_id" not in row or not str(row.get("group_id", "")).strip():
                row["group_id"] = f"{row.get('seq', '')}:{row.get('frame', '')}:{row.get('det_index', '')}"
            rows.append(row)
    return rows


def _split_groups(group_rows: dict[str, list[dict[str, str]]], train_seqs: set[str], val_seqs: set[str], val_ratio: float, seed: int) -> tuple[list[str], list[str]]:
    group_ids = sorted(group_rows.keys())
    if train_seqs or val_seqs:
        train_ids: list[str] = []
        val_ids: list[str] = []
        for gid in group_ids:
            seq = str(group_rows[gid][0].get("seq", ""))
            if val_seqs and seq in val_seqs:
                val_ids.append(gid)
            elif train_seqs and seq in train_seqs:
                train_ids.append(gid)
            elif val_seqs:
                train_ids.append(gid)
            else:
                val_ids.append(gid)
        return train_ids, val_ids

    rng = np.random.default_rng(seed)
    group_ids_arr = np.asarray(group_ids, dtype=object)
    rng.shuffle(group_ids_arr)
    split = int(round((1.0 - val_ratio) * float(group_ids_arr.size)))
    split = max(1, min(split, int(group_ids_arr.size) - 1))
    return list(group_ids_arr[:split]), list(group_ids_arr[split:])


def _build_dataset(group_ids: Iterable[str], group_rows: dict[str, list[dict[str, str]]], rank_score_col: str, valid_only: bool) -> tuple[np.ndarray, np.ndarray]:
    feats: list[list[float]] = []
    labels: list[int] = []
    for gid in group_ids:
        for row in group_rows[gid]:
            if valid_only and _safe_int(row, "valid_train_row", 1) == 0:
                continue
            feats.append(_build_feature(row, rank_score_col))
            labels.append(_safe_int(row, "label", 0))
    if not feats:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.asarray(feats, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def _fit_model(model_name: str, x: np.ndarray, y: np.ndarray, seed: int):
    if model_name == "gbdt":
        model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=3,
            max_iter=200,
            random_state=seed,
        )
    else:
        model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=seed,
            solver="lbfgs",
        )
    model.fit(x, y)
    return model


def _predict_scores(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(x)
        if prob.ndim == 2 and prob.shape[1] >= 2:
            return prob[:, 1].astype(np.float32)
    if hasattr(model, "decision_function"):
        logits = model.decision_function(x)
        logits = np.asarray(logits, dtype=np.float32)
        return 1.0 / (1.0 + np.exp(-logits))
    pred = model.predict(x)
    return np.asarray(pred, dtype=np.float32)


def _evaluate(
    group_ids: Iterable[str],
    group_rows: dict[str, list[dict[str, str]]],
    model,
    rank_score_col: str,
    valid_only: bool,
    apply_on: str,
) -> dict[str, float]:
    stats = {
        "groups": 0,
        "positive_groups": 0,
        "ambiguous_groups": 0,
        "easy_groups": 0,
        "recoverable_groups": 0,
        "recovered_groups": 0,
        "base_top1": 0.0,
        "final_top1": 0.0,
        "amb_base_top1": 0.0,
        "amb_final_top1": 0.0,
        "easy_base_top1": 0.0,
        "easy_final_top1": 0.0,
    }
    pos_base = pos_final = 0
    amb_base = amb_final = 0
    easy_base = easy_final = 0

    for gid in group_ids:
        rows = group_rows[gid]
        eval_rows = [r for r in rows if (not valid_only) or _safe_int(r, "valid_train_row", 1) == 1]
        if not eval_rows:
            continue
        positive_indices = [idx for idx, r in enumerate(eval_rows) if _safe_int(r, "label", 0) == 1]
        if not positive_indices:
            stats["groups"] += 1
            continue

        x = np.asarray([_build_feature(r, rank_score_col) for r in eval_rows], dtype=np.float32)
        pred_scores = _predict_scores(model, x)
        anchor_scores = np.asarray([_safe_float(r, rank_score_col) for r in eval_rows], dtype=np.float32)
        base_idx = int(np.argmax(anchor_scores))
        group_is_ambiguous = _safe_int(eval_rows[0], "group_is_ambiguous", 0) == 1
        group_is_recoverable = _safe_int(eval_rows[0], "group_is_recoverable", 0) == 1
        final_scores = pred_scores if (apply_on == "all" or group_is_ambiguous) else anchor_scores
        final_idx = int(np.argmax(final_scores))
        base_correct = 1 if base_idx in positive_indices else 0
        final_correct = 1 if final_idx in positive_indices else 0

        stats["groups"] += 1
        stats["positive_groups"] += 1
        pos_base += base_correct
        pos_final += final_correct
        if group_is_ambiguous:
            stats["ambiguous_groups"] += 1
            amb_base += base_correct
            amb_final += final_correct
        else:
            stats["easy_groups"] += 1
            easy_base += base_correct
            easy_final += final_correct
        if group_is_recoverable:
            stats["recoverable_groups"] += 1
            stats["recovered_groups"] += final_correct

    if stats["positive_groups"] > 0:
        stats["base_top1"] = pos_base / stats["positive_groups"]
        stats["final_top1"] = pos_final / stats["positive_groups"]
    if stats["ambiguous_groups"] > 0:
        stats["amb_base_top1"] = amb_base / stats["ambiguous_groups"]
        stats["amb_final_top1"] = amb_final / stats["ambiguous_groups"]
    if stats["easy_groups"] > 0:
        stats["easy_base_top1"] = easy_base / stats["easy_groups"]
        stats["easy_final_top1"] = easy_final / stats["easy_groups"]
    stats["top1_gain"] = stats["final_top1"] - stats["base_top1"]
    stats["amb_top1_gain"] = stats["amb_final_top1"] - stats["amb_base_top1"]
    stats["easy_top1_gain"] = stats["easy_final_top1"] - stats["easy_base_top1"]
    stats["recovered_rate"] = (
        stats["recovered_groups"] / stats["recoverable_groups"] if stats["recoverable_groups"] > 0 else 0.0
    )
    return stats


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(Path(args.input_csv).resolve())
    group_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        group_rows[str(row["group_id"])].append(row)

    train_ids, val_ids = _split_groups(
        group_rows=group_rows,
        train_seqs=_parse_seq_set(args.train_seqs),
        val_seqs=_parse_seq_set(args.val_seqs),
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
    )
    x_train, y_train = _build_dataset(train_ids, group_rows, args.rank_score_col, args.valid_only)
    x_val, y_val = _build_dataset(val_ids, group_rows, args.rank_score_col, args.valid_only)
    if x_train.size == 0 or x_train.shape[0] == 0:
        raise RuntimeError("No training rows were built from the labeled CSV.")

    model = _fit_model(args.model, x_train, y_train, args.seed)

    metrics = {
        "model": args.model,
        "rank_score_col": args.rank_score_col,
        "apply_on": args.apply_on,
        "seed": int(args.seed),
        "train_groups": len(train_ids),
        "val_groups": len(val_ids),
        "train_rows": int(x_train.shape[0]),
        "val_rows": int(x_val.shape[0]) if x_val.ndim == 2 else 0,
        "train_positive_rate": float(np.mean(y_train)) if y_train.size else 0.0,
        "val_positive_rate": float(np.mean(y_val)) if y_val.size else 0.0,
        "train_eval": _evaluate(train_ids, group_rows, model, args.rank_score_col, args.valid_only, args.apply_on),
        "val_eval": _evaluate(val_ids, group_rows, model, args.rank_score_col, args.valid_only, args.apply_on),
        "feature_names": [
            args.rank_score_col,
            "base_score",
            "refined_score",
            "motion_score",
            "det_score",
            "log1p(track_gap)",
            "log1p(track_hist_len)",
            "base_margin",
            "refined_margin",
            "rank_margin",
            "rank_entropy",
            "rank_frac",
            "dx_norm",
            "dy_norm",
            "log_w_ratio",
            "log_h_ratio",
            "log_area_ratio",
            "det_track_iou",
        ],
    }

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (out_dir / "model.pkl").open("wb") as f:
        pickle.dump(model, f)
    print(json.dumps(metrics, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
