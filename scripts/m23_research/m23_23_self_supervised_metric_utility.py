from __future__ import annotations

# Research artifact for the MOT20 M23 GT-free sequence-adaptive metric audit.

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


OUT = Path(
    "outputs/mot20_m23_20260718/m23_23_self_supervised_metric_utility_v1"
)
MAX_HARD_NEGATIVES = 3
METRIC_REPORTS: dict[str, dict] = {}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def normalized_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def positive_pairs(meta: pd.DataFrame) -> np.ndarray:
    pairs = []
    ordered = meta.sort_values(["source_track_id", "source_ordinal"])
    for _, part in ordered.groupby("source_track_id", sort=False):
        ids = part.chunk_id.to_numpy(int)
        if len(ids) > 1:
            pairs.append(np.column_stack([ids[:-1], ids[1:]]))
    return np.concatenate(pairs, axis=0) if pairs else np.empty((0, 2), dtype=int)


def hard_negative_pairs(meta: pd.DataFrame, prototypes: np.ndarray) -> np.ndarray:
    midpoint = ((meta.first_frame.to_numpy(int) + meta.last_frame.to_numpy(int)) // 2)
    bucket = midpoint // 30
    first = meta.first_frame.to_numpy(int)
    last = meta.last_frame.to_numpy(int)
    tracks = meta.source_track_id.to_numpy(int)
    pairs = []
    for _, indices in pd.Series(np.arange(len(meta))).groupby(bucket, sort=False):
        ids = indices.to_numpy(int)
        if len(ids) < 2:
            continue
        similarity = prototypes[ids] @ prototypes[ids].T
        overlap = (first[ids, None] <= last[ids][None, :]) & (
            first[ids][None, :] <= last[ids, None]
        )
        valid = overlap & (tracks[ids, None] != tracks[ids][None, :])
        np.fill_diagonal(valid, False)
        similarity[~valid] = -np.inf
        for local_index, chunk_id in enumerate(ids):
            candidates = np.flatnonzero(np.isfinite(similarity[local_index]))
            if not len(candidates):
                continue
            order = candidates[
                np.argsort(similarity[local_index, candidates], kind="mergesort")[::-1]
            ]
            for other in order[:MAX_HARD_NEGATIVES]:
                left, right = sorted((int(chunk_id), int(ids[other])))
                pairs.append((left, right))
    if not pairs:
        return np.empty((0, 2), dtype=int)
    return np.asarray(sorted(set(pairs)), dtype=int)


def metric_weights(
    prototypes: np.ndarray,
    positives: np.ndarray,
    negatives: np.ndarray,
) -> tuple[np.ndarray, dict]:
    if not len(positives) or not len(negatives):
        weights = np.ones(prototypes.shape[1], dtype=np.float32)
        return weights, {"fallback_unit_weights": True}
    positive_sq = (
        prototypes[positives[:, 0]] - prototypes[positives[:, 1]]
    ) ** 2
    negative_sq = (
        prototypes[negatives[:, 0]] - prototypes[negatives[:, 1]]
    ) ** 2
    positive_scale = np.median(positive_sq, axis=0)
    negative_scale = np.median(negative_sq, axis=0)
    floor = max(float(np.median(positive_scale)), 1e-8)
    fisher = np.maximum(negative_scale - positive_scale, 0.0) / (
        positive_scale + 0.25 * floor
    )
    positive_fisher = fisher[fisher > 0]
    cap = float(np.quantile(positive_fisher, 0.95)) if len(positive_fisher) else 1.0
    weights = np.clip(fisher, 0.0, max(cap, 1e-6))
    if not np.any(weights > 0):
        weights = np.ones_like(weights)
    weights = weights / max(float(weights.mean()), 1e-8)
    weights = np.clip(weights, 0.05, 8.0).astype(np.float32)
    return weights, {
        "fallback_unit_weights": False,
        "positive_sq_median_mean": float(positive_scale.mean()),
        "negative_sq_median_mean": float(negative_scale.mean()),
        "weight_min": float(weights.min()),
        "weight_mean": float(weights.mean()),
        "weight_max": float(weights.max()),
        "active_dimensions": int((weights > 0.05).sum()),
    }


def track_part_vectors(meta: pd.DataFrame, prototypes: np.ndarray):
    prefix = np.zeros_like(prototypes, dtype=np.float32)
    suffix = np.zeros_like(prototypes, dtype=np.float32)
    track = np.zeros_like(prototypes, dtype=np.float32)
    ordered = meta.sort_values(["source_track_id", "source_ordinal"])
    for _, part in ordered.groupby("source_track_id", sort=False):
        ids = part.chunk_id.to_numpy(int)
        weights = part.rows.to_numpy(np.float32)
        weighted = prototypes[ids] * weights[:, None]
        prefix_values = np.cumsum(weighted, axis=0)
        suffix_values = np.cumsum(weighted[::-1], axis=0)[::-1]
        prefix[ids] = prefix_values
        suffix[ids] = suffix_values
        track[ids] = prefix_values[-1]
    return prefix, suffix, track


def add_self_supervised_metric_features(
    seq: str,
    frame: pd.DataFrame,
    features: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    root = Path("outputs/mot20_m23_20260718/micrograph_chunk30_v1") / seq
    meta = pd.read_parquet(root / "microtracklets.parquet").sort_values("chunk_id")
    if meta.chunk_id.tolist() != list(range(len(meta))):
        raise RuntimeError(f"{seq}: chunk ids are not dense")
    prototypes = normalized_rows(
        np.load(root / "prototypes.f16.npy").astype(np.float32)
    )
    positives = positive_pairs(meta)
    negatives = hard_negative_pairs(meta, prototypes)
    coordinate_weights, diagnostics = metric_weights(prototypes, positives, negatives)
    scale = np.sqrt(coordinate_weights)[None, :]
    prefix, suffix, track = track_part_vectors(meta, prototypes)
    weighted_prefix = normalized_rows(prefix * scale)
    weighted_suffix = normalized_rows(suffix * scale)
    weighted_track = normalized_rows(track * scale)
    src = frame.src_chunk.to_numpy(int)
    dst = frame.dst_chunk.to_numpy(int)
    segment_cos = np.einsum(
        "ij,ij->i", weighted_prefix[src], weighted_suffix[dst]
    )
    track_cos = np.einsum("ij,ij->i", weighted_track[src], weighted_track[dst])
    output = frame.copy()
    output["selfsup_segment_cos"] = segment_cos.astype(np.float32)
    output["selfsup_track_cos"] = track_cos.astype(np.float32)
    output["selfsup_cos_mean"] = ((segment_cos + track_cos) / 2).astype(np.float32)
    output["selfsup_segment_cos_gain"] = (
        segment_cos - output.segment_appearance_cos.to_numpy(float)
    ).astype(np.float32)
    output["selfsup_track_cos_gain"] = (
        track_cos - output.track_appearance_cos.to_numpy(float)
    ).astype(np.float32)
    src_rank = output.groupby("transaction_src_track_id", sort=False)[
        "selfsup_segment_cos"
    ].rank(method="average", pct=True, ascending=False)
    dst_rank = output.groupby("transaction_dst_track_id", sort=False)[
        "selfsup_segment_cos"
    ].rank(method="average", pct=True, ascending=False)
    output["selfsup_src_rank"] = src_rank.astype(np.float32)
    output["selfsup_dst_rank"] = dst_rank.astype(np.float32)
    output["selfsup_mutual_rank_max"] = np.maximum(src_rank, dst_rank).astype(np.float32)
    added = [
        "selfsup_segment_cos",
        "selfsup_track_cos",
        "selfsup_cos_mean",
        "selfsup_segment_cos_gain",
        "selfsup_track_cos_gain",
        "selfsup_src_rank",
        "selfsup_dst_rank",
        "selfsup_mutual_rank_max",
    ]
    METRIC_REPORTS[seq] = {
        "seq": seq,
        "gt_use": "none",
        "positive_pairs": int(len(positives)),
        "positive_definition": "consecutive microtracklets within a parent track",
        "hard_negative_pairs": int(len(negatives)),
        "hard_negative_definition": (
            "up to three most appearance-similar temporally overlapping "
            "microtracklets from different parent tracks per chunk"
        ),
        "coordinate_weights_sha256": hashlib.sha256(
            coordinate_weights.tobytes()
        ).hexdigest(),
        **diagnostics,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "self_supervised_metric_adaptation.json").write_text(
        json.dumps(
            {
                "status": "frozen GT-free sequence-specific metric adaptation",
                "max_hard_negatives_per_chunk": MAX_HARD_NEGATIVES,
                "sequences": [METRIC_REPORTS[key] for key in sorted(METRIC_REPORTS)],
            },
            indent=2,
        )
        + "\n"
    )
    return output, [*features, *added]


def main() -> None:
    base = load_module(
        "m23_17_selfsup_metric_base",
        Path("scripts/m23_research/m23_17_sequence_normalized_utility.py"),
    )
    base.OUT = OUT
    base.NAME = "self_supervised_metric_nested_budget_policy_v1"
    base.SCORE = "pred_self_supervised_metric_utility"
    base.TOP_K_GRID = [10, 25, 50, 100, 250, 500, 1000, 2500]
    base.USE_FROZEN_M14_OOD_GATE = False
    base.TRAINING_GT_USE = (
        "sequence-normalized transaction utility target on each fold's fit sequences "
        "only; held-sequence metric adaptation is GT-free"
    )
    base.FEATURE_TRANSFORM_DESCRIPTION = (
        "M23-17 sequence ranks plus a test-time GT-free diagonal appearance metric "
        "estimated from parent-track consecutive positives and overlapping-track hard "
        "negatives within each sequence"
    )
    base.TARGET_TRANSFORM_DESCRIPTION = (
        "M23-17 sequence-normalized asinh transaction utility"
    )
    base.STATUS_DESCRIPTION = (
        "strict nested sequence-adaptive self-supervised metric audit on reused "
        "development sequences; fixed-parent provenance remains exploratory"
    )
    base.augment_model_features = add_self_supervised_metric_features
    base.main()


if __name__ == "__main__":
    main()
