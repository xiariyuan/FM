from __future__ import annotations

# Research artifact for the MOT20 M23 sequence-LOSO audit.

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score


SEQS = ["MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05"]
ROOT = Path("outputs/mot20_m23_20260718/micrograph_chunk30_v1")
OUT = Path("outputs/mot20_m23_20260718/micrograph_chunk30_role_loso_v1")
FEATURES = [
    "gap",
    "log_gap",
    "appearance_cos",
    "forward_motion_error",
    "backward_motion_error",
    "motion_error_min",
    "motion_error_mean",
    "endpoint_displacement",
    "velocity_cos",
    "log_height_ratio",
    "src_rows",
    "dst_rows",
    "src_mapping_rate",
    "dst_mapping_rate",
    "mapping_rate_min",
    "src_consistency",
    "dst_consistency",
    "consistency_min",
    "src_match_iou",
    "dst_match_iou",
    "out_rank",
    "in_rank",
    "max_rank",
    "out_margin",
    "in_margin",
    "max_margin",
]
LABEL_FILTER = (
    "modal GT present at both endpoints and endpoint modal purity >= 0.7; "
    "GT columns are excluded from model features"
)


def sequence_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.seq.value_counts()
    mapping = {seq: len(frame) / (len(counts) * count) for seq, count in counts.items()}
    return frame.seq.map(mapping).to_numpy(float)


def capped_role_weights(frame: pd.DataFrame, rare_label: int, cap: float = 25.0) -> np.ndarray:
    y = frame.same_gt.to_numpy(int)
    rare = int((y == rare_label).sum())
    common = int((y != rare_label).sum())
    multiplier = min(cap, common / max(rare, 1))
    role = np.where(y == rare_label, multiplier, 1.0)
    confidence = frame.label_confidence.clip(0.2, 1.0).to_numpy(float)
    return sequence_weights(frame) * role * confidence


def safe_metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float | None]:
    if len(np.unique(y)) < 2:
        return {"auc": None, "ap": None}
    return {
        "auc": float(roc_auc_score(y, score)),
        "ap": float(average_precision_score(y, score)),
    }


def threshold_audit(
    y: np.ndarray,
    score: np.ndarray,
    thresholds: list[float],
    select_high: bool,
    prefix: str,
) -> dict[str, float | int | None]:
    out: dict[str, float | int | None] = {}
    positives = max(int(y.sum()), 1)
    for threshold in thresholds:
        selected = score >= threshold if select_high else score < threshold
        key = f"{prefix}_t{int(round(100 * threshold)):03d}"
        out[f"{key}_selected"] = int(selected.sum())
        out[f"{key}_precision"] = float(y[selected].mean()) if selected.any() else None
        out[f"{key}_recall"] = float(y[selected].sum() / positives)
    return out


def topk_audit(y: np.ndarray, score: np.ndarray, prefix: str) -> dict[str, float | int]:
    order = np.argsort(-score)
    out: dict[str, float | int] = {}
    for k in [10, 25, 50, 100, 250, 500]:
        n = min(k, len(order))
        chosen = order[:n]
        out[f"{prefix}_top{k}_selected"] = n
        out[f"{prefix}_top{k}_positive"] = int(y[chosen].sum())
        out[f"{prefix}_top{k}_precision"] = float(y[chosen].mean()) if n else 0.0
    return out


def clean(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[
        (frame.src_modal_gt > 0)
        & (frame.dst_modal_gt > 0)
        & (frame.src_purity >= 0.7)
        & (frame.dst_purity >= 0.7)
    ].copy()


def fit_model(frame: pd.DataFrame, rare_label: int, seed: int, min_samples_leaf: int):
    model = HistGradientBoostingClassifier(
        max_iter=260,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=6.0,
        max_bins=255,
        random_state=seed,
        early_stopping=False,
    )
    model.fit(
        frame[FEATURES],
        frame.same_gt.astype(int),
        sample_weight=capped_role_weights(frame, rare_label=rare_label),
    )
    return model


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frames = []
    for seq in SEQS:
        frame = pd.read_parquet(ROOT / seq / "candidate_edges.parquet")
        frame.insert(0, "seq", seq)
        frames.append(frame)
    all_data = pd.concat(frames, ignore_index=True, sort=False)
    all_clean = clean(all_data)

    folds = []
    for fold_index, held in enumerate(SEQS):
        train_pool = all_clean[all_clean.seq != held]
        boundary_train = train_pool[train_pool.source_adjacent == 1].copy()
        cross_train = train_pool[train_pool.same_source == 0].copy()
        boundary_model = fit_model(
            boundary_train,
            rare_label=0,
            seed=10200 + fold_index,
            min_samples_leaf=20,
        )
        cross_model = fit_model(
            cross_train,
            rare_label=1,
            seed=10300 + fold_index,
            min_samples_leaf=40,
        )

        held_all = all_data[all_data.seq == held].copy()
        boundary_all = held_all[held_all.source_adjacent == 1].copy()
        cross_all = held_all[held_all.same_source == 0].copy()
        boundary_all["pred_keep_prob"] = boundary_model.predict_proba(boundary_all[FEATURES])[:, 1]
        cross_all["pred_link_prob"] = cross_model.predict_proba(cross_all[FEATURES])[:, 1]
        boundary_all.to_parquet(OUT / f"{held}_boundary_predictions.parquet", index=False)
        cross_all.to_parquet(OUT / f"{held}_cross_predictions.parquet", index=False)

        boundary_eval = clean(boundary_all)
        cross_eval = clean(cross_all)
        boundary_break_y = 1 - boundary_eval.same_gt.to_numpy(int)
        boundary_break_score = 1.0 - boundary_eval.pred_keep_prob.to_numpy(float)
        cross_y = cross_eval.same_gt.to_numpy(int)
        cross_score = cross_eval.pred_link_prob.to_numpy(float)

        record = {
            "held_out_seq": held,
            "boundary_train_rows": len(boundary_train),
            "boundary_train_breaks": int((boundary_train.same_gt == 0).sum()),
            "boundary_test_rows": len(boundary_eval),
            "boundary_test_breaks": int(boundary_break_y.sum()),
            "boundary_break_weight_cap": 25.0,
            **{f"boundary_break_{k}": v for k, v in safe_metrics(boundary_break_y, boundary_break_score).items()},
            **threshold_audit(
                boundary_break_y,
                boundary_eval.pred_keep_prob.to_numpy(float),
                [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99],
                select_high=False,
                prefix="predicted_break",
            ),
            **topk_audit(boundary_break_y, boundary_break_score, "break"),
            "cross_train_rows": len(cross_train),
            "cross_train_positive": int(cross_train.same_gt.sum()),
            "cross_test_rows": len(cross_eval),
            "cross_test_positive": int(cross_y.sum()),
            "cross_positive_weight_cap": 25.0,
            **{f"cross_{k}": v for k, v in safe_metrics(cross_y, cross_score).items()},
            **threshold_audit(
                cross_y,
                cross_score,
                [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99],
                select_high=True,
                prefix="predicted_link",
            ),
            **topk_audit(cross_y, cross_score, "link"),
        }
        folds.append(record)
        print(json.dumps(record), flush=True)

    report = {
        "protocol": {
            "validation": "strict leave-one-sequence-out; held sequence excluded from both role models",
            "boundary_role": "classify whether consecutive chunks from one source track preserve identity",
            "cross_role": "classify whether chunks from different source tracks share identity",
            "features": FEATURES,
            "label_filter": LABEL_FILTER,
            "class_weighting": "sequence-balanced, label-confidence weighted, rare-class multiplier capped at 25",
            "inference": "GT-free",
            "status": "diagnostic ranking only; no TrackEval policy selected here",
        },
        "folds": folds,
        "dataset": {
            "edges": len(all_data),
            "clean_edges": len(all_clean),
            "clean_boundary_edges": int((all_clean.source_adjacent == 1).sum()),
            "clean_boundary_breaks": int(((all_clean.source_adjacent == 1) & (all_clean.same_gt == 0)).sum()),
            "clean_cross_edges": int((all_clean.same_source == 0).sum()),
            "clean_cross_positive": int(((all_clean.same_source == 0) & (all_clean.same_gt == 1)).sum()),
        },
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
