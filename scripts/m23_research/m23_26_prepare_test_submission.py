#!/usr/bin/env python3
from __future__ import annotations

"""Prepare one GT-free MOT20 test submission from the frozen M23-25 evidence.

Deployment protocol
-------------------
1. Average the four strict-LOSO FastReID checkpoints over all compatible model
   tensors.  The identity-classification tensor ``heads.weight`` is copied from
   one checkpoint because its shape is fold-specific and it is not used during
   inference.
2. Re-embed the official MOT20 test detections.  No test GT path is opened.
3. Build test microtracklet graphs from the frozen A43/A42-family parent test
   tracker.
4. Build an out-of-fold train graph by taking each sequence from its own outer
   held fold.  Train transaction models and calibrate the policy using train GT
   only.
5. Apply the frozen policy to MOT20-04/06/07/08, validate the four root-level
   MOTChallenge text files, and create a zip archive.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import networkx as nx
import numpy as np
import pandas as pd
import torch


REPO = Path(__file__).resolve().parents[2]
BOT_ROOT = REPO / "external" / "BoT-SORT-main"
TRAIN_SEQUENCES = ("MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05")
TEST_SEQUENCES = ("MOT20-04", "MOT20-06", "MOT20-07", "MOT20-08")

DEFAULT_RUN_ROOT = REPO / "outputs" / "mot20_m23_20260718" / "m23_26_test_deploy_oof_ensemble_v1"
DEFAULT_TEST_PHASE0 = REPO / "outputs" / "dmm_phase0_mot20_test_parambest"
DEFAULT_TEST_PARENT = (
    REPO
    / "outputs"
    / "assocriskbench_p15_20260713"
    / "identity_debt_a43_test_submission"
    / "track_results"
)
DEFAULT_TRAIN_PARENT = (
    REPO
    / "outputs"
    / "assocriskbench_p15_20260714"
    / "fused_a42_adaptive_risk_eval"
    / "l1_r2_q75_adaptive_priority"
    / "track_results"
)
LOSO_ROOT = REPO / "outputs" / "mot20_m23_20260718" / "m23_24_micrograph_loso"
CHECKPOINT_ROOT = REPO / "outputs" / "mot20_m23_20260718" / "m23_24_fastreid_loso_full"

TARGET = "chain_transaction_delta_proxy"
POSITIVE = "chain_transaction_positive"
GRAPH_FEATURES = [
    "src_conflict_degree",
    "dst_conflict_degree",
    "conflict_degree",
    "log_src_conflict_degree",
    "log_dst_conflict_degree",
    "log_conflict_degree",
    "pair_candidate_count",
    "src_candidate_fraction",
    "dst_candidate_fraction",
    "sequence_appearance_percentile",
    "sequence_segment_appearance_percentile",
    "sequence_motion_percentile",
    "sequence_gap_percentile",
    "src_appearance_percentile",
    "dst_appearance_percentile",
    "src_segment_appearance_percentile",
    "dst_segment_appearance_percentile",
    "src_motion_percentile",
    "dst_motion_percentile",
    "rank_consensus",
    "appearance_rank_consensus",
    "motion_rank_consensus",
    "appearance_upgrade",
    "segment_endpoint_gain",
    "coherence_floor",
    "coherence_imbalance",
    "motion_appearance_joint",
    "action_complexity",
    "conflict_per_merged_row",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    print(f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] {message}", flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name: str, relative_path: str):
    path = REPO / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parse_float_grid(raw: str) -> List[float]:
    values = sorted({float(item) for item in raw.split(",") if item.strip()})
    if not values:
        raise ValueError("empty grid")
    return values


def robust_percentile(values: pd.Series, higher_is_better: bool = True) -> pd.Series:
    ranked = values.rank(method="average", pct=True)
    if higher_is_better:
        return ranked
    return 1.0 - ranked + 1.0 / max(len(values), 1)


def grouped_percentile(
    frame: pd.DataFrame, group: str, value: str, higher_is_better: bool = True
) -> pd.Series:
    ranked = frame.groupby(group, sort=False)[value].rank(method="average", pct=True)
    if higher_is_better:
        return ranked
    sizes = frame.groupby(group, sort=False)[value].transform("size").clip(lower=1)
    return 1.0 - ranked + 1.0 / sizes


def add_conflict_graph_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    required = {
        "transaction_src_track_id",
        "transaction_dst_track_id",
        "appearance_cos",
        "segment_appearance_cos",
        "motion_error_min",
    }
    missing = sorted(required - set(output.columns))
    if missing:
        raise RuntimeError(f"missing transaction graph columns: {missing}")
    src = output.transaction_src_track_id.astype(np.int64)
    dst = output.transaction_dst_track_id.astype(np.int64)
    all_tracks = pd.concat([src, dst], ignore_index=True)
    all_counts = all_tracks.value_counts()
    src_degree = src.map(all_counts).to_numpy(float) - 1.0
    dst_degree = dst.map(all_counts).to_numpy(float) - 1.0
    output["src_conflict_degree"] = np.maximum(src_degree, 0.0)
    output["dst_conflict_degree"] = np.maximum(dst_degree, 0.0)
    output["conflict_degree"] = np.maximum(src_degree + dst_degree, 0.0)
    output["log_src_conflict_degree"] = np.log1p(output.src_conflict_degree)
    output["log_dst_conflict_degree"] = np.log1p(output.dst_conflict_degree)
    output["log_conflict_degree"] = np.log1p(output.conflict_degree)
    output["pair_candidate_count"] = output.groupby(
        ["transaction_src_track_id", "transaction_dst_track_id"], sort=False
    ).src_chunk.transform("size")
    denominator = max(len(output), 1)
    output["src_candidate_fraction"] = (output.src_conflict_degree + 1.0) / denominator
    output["dst_candidate_fraction"] = (output.dst_conflict_degree + 1.0) / denominator
    output["sequence_appearance_percentile"] = robust_percentile(output.appearance_cos)
    output["sequence_segment_appearance_percentile"] = robust_percentile(
        output.segment_appearance_cos
    )
    output["sequence_motion_percentile"] = robust_percentile(
        output.motion_error_min, higher_is_better=False
    )
    output["sequence_gap_percentile"] = robust_percentile(
        output.gap, higher_is_better=False
    )
    output["src_appearance_percentile"] = grouped_percentile(
        output, "transaction_src_track_id", "appearance_cos"
    )
    output["dst_appearance_percentile"] = grouped_percentile(
        output, "transaction_dst_track_id", "appearance_cos"
    )
    output["src_segment_appearance_percentile"] = grouped_percentile(
        output, "transaction_src_track_id", "segment_appearance_cos"
    )
    output["dst_segment_appearance_percentile"] = grouped_percentile(
        output, "transaction_dst_track_id", "segment_appearance_cos"
    )
    output["src_motion_percentile"] = grouped_percentile(
        output,
        "transaction_src_track_id",
        "motion_error_min",
        higher_is_better=False,
    )
    output["dst_motion_percentile"] = grouped_percentile(
        output,
        "transaction_dst_track_id",
        "motion_error_min",
        higher_is_better=False,
    )
    output["appearance_rank_consensus"] = np.minimum(
        output.src_segment_appearance_percentile,
        output.dst_segment_appearance_percentile,
    )
    output["motion_rank_consensus"] = np.minimum(
        output.src_motion_percentile, output.dst_motion_percentile
    )
    output["rank_consensus"] = np.minimum(
        output.appearance_rank_consensus, output.motion_rank_consensus
    )
    output["appearance_upgrade"] = (
        output.segment_appearance_cos.to_numpy(float)
        - output.track_appearance_cos.to_numpy(float)
    )
    endpoint_floor = np.minimum(
        output.src_prefix_endpoint_cos.to_numpy(float),
        output.dst_suffix_endpoint_cos.to_numpy(float),
    )
    output["segment_endpoint_gain"] = (
        output.segment_appearance_cos.to_numpy(float) - endpoint_floor
    )
    output["coherence_floor"] = np.minimum(
        output.src_prefix_coherence.to_numpy(float),
        output.dst_suffix_coherence.to_numpy(float),
    )
    output["coherence_imbalance"] = np.abs(
        output.src_prefix_coherence.to_numpy(float)
        - output.dst_suffix_coherence.to_numpy(float)
    )
    output["motion_appearance_joint"] = (
        output.segment_appearance_cos.to_numpy(float)
        - np.log1p(np.maximum(output.motion_error_min.to_numpy(float), 0.0))
    )
    output["action_complexity"] = (
        output.transaction_removes_source_out.to_numpy(float)
        + output.transaction_removes_source_in.to_numpy(float)
    )
    output["conflict_per_merged_row"] = output.conflict_degree.to_numpy(float) / np.maximum(
        output.merged_segment_rows.to_numpy(float), 1.0
    )
    return output


def structural_transactions(meta: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    cross = edges[edges.same_source.to_numpy(int) == 0].copy()
    meta = meta.sort_values("chunk_id")
    if meta.chunk_id.tolist() != list(range(len(meta))):
        raise RuntimeError("chunk ids are not dense")
    track_for_chunk = meta.source_track_id.to_numpy(int)
    ordinal_for_chunk = meta.source_ordinal.to_numpy(int)
    chunks_per_track = meta.groupby("source_track_id").source_ordinal.max().add(1).to_dict()
    src = cross.src_chunk.to_numpy(int)
    dst = cross.dst_chunk.to_numpy(int)
    src_track = track_for_chunk[src]
    dst_track = track_for_chunk[dst]
    src_ordinal = ordinal_for_chunk[src]
    dst_ordinal = ordinal_for_chunk[dst]
    cross["transaction_src_track_id"] = src_track
    cross["transaction_dst_track_id"] = dst_track
    cross["transaction_removes_source_out"] = [
        int(ordinal + 1 < chunks_per_track[int(track)])
        for ordinal, track in zip(src_ordinal, src_track)
    ]
    cross["transaction_removes_source_in"] = (dst_ordinal > 0).astype(np.int8)
    return cross


def sequence_class_weights(frame: pd.DataFrame) -> np.ndarray:
    groups = frame.groupby(["seq", POSITIVE], sort=False).size().to_dict()
    present = max(len(groups), 1)
    return np.asarray(
        [len(frame) / (present * groups[(seq, value)]) for seq, value in zip(frame.seq, frame[POSITIVE])],
        dtype=np.float64,
    )


def sequence_sign_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.seq.value_counts().to_dict()
    return np.asarray(
        [len(frame) / (len(counts) * counts[seq]) for seq in frame.seq], dtype=np.float64
    )


def sequence_target_scale(frame: pd.DataFrame) -> np.ndarray:
    scale_by_sequence: Dict[str, float] = {}
    for seq, part in frame.groupby("seq", sort=False):
        positive = part.loc[part[TARGET] > 0, TARGET].to_numpy(float)
        scale_by_sequence[str(seq)] = max(float(np.median(positive)) if len(positive) else 1.0, 1e-3)
    return frame.seq.map(scale_by_sequence).to_numpy(float)


def fit_distributional_models(
    frame: pd.DataFrame, features: Sequence[str], seed: int, max_iter: int
):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    classifier = HistGradientBoostingClassifier(
        max_iter=max_iter,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=60,
        l2_regularization=14.0,
        max_bins=255,
        random_state=seed,
        early_stopping=False,
    )
    classifier.fit(
        frame[list(features)],
        frame[POSITIVE].astype(int),
        sample_weight=sequence_class_weights(frame),
    )
    scales = sequence_target_scale(frame)
    positive = frame[TARGET].to_numpy(float) > 0
    negative = frame[TARGET].to_numpy(float) < 0
    if positive.sum() < 10 or negative.sum() < 10:
        raise RuntimeError("insufficient signed transaction labels")

    def fit_magnitude(mask: np.ndarray, target: np.ndarray, offset: int):
        model = HistGradientBoostingRegressor(
            loss="absolute_error",
            max_iter=max_iter,
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            l2_regularization=14.0,
            max_bins=255,
            random_state=seed + offset,
            early_stopping=False,
        )
        part = frame.loc[mask].copy()
        model.fit(
            part[list(features)],
            target,
            sample_weight=sequence_sign_weights(part),
        )
        return model

    gain_target = np.log1p(frame.loc[positive, TARGET].to_numpy(float) / scales[positive])
    loss_target = np.log1p(-frame.loc[negative, TARGET].to_numpy(float) / scales[negative])
    return (
        classifier,
        fit_magnitude(positive, gain_target, 1000),
        fit_magnitude(negative, loss_target, 2000),
    )


def predict_distributional(
    frame: pd.DataFrame, models, features: Sequence[str]
) -> pd.DataFrame:
    classifier, gain_model, loss_model = models
    output = frame.copy()
    probability = classifier.predict_proba(output[list(features)])[:, 1]
    gain = np.expm1(gain_model.predict(output[list(features)]))
    loss = np.expm1(loss_model.predict(output[list(features)]))
    output["pred_positive_probability"] = probability
    output["pred_normalized_gain"] = np.maximum(gain, 0.0)
    output["pred_normalized_loss"] = np.maximum(loss, 0.0)
    output["pred_entropy"] = -(
        probability * np.log(np.maximum(probability, 1e-8))
        + (1.0 - probability) * np.log(np.maximum(1.0 - probability, 1e-8))
    )
    return output


def add_policy_score(frame: pd.DataFrame, loss_multiplier: float) -> pd.DataFrame:
    output = frame.copy()
    probability = output.pred_positive_probability.to_numpy(float)
    gain = output.pred_normalized_gain.to_numpy(float)
    loss = output.pred_normalized_loss.to_numpy(float)
    output["policy_score"] = (
        probability * gain - loss_multiplier * (1.0 - probability) * loss
    )
    output["policy_score_percentile"] = robust_percentile(output.policy_score)
    return output


def maximum_weight_transaction_matching(
    frame: pd.DataFrame,
    loss_multiplier: float,
    min_probability: float,
    score_quantile: float,
) -> pd.DataFrame:
    scored = add_policy_score(frame, loss_multiplier)
    eligible = scored[
        (scored.policy_score.to_numpy(float) > 0.0)
        & (scored.pred_positive_probability.to_numpy(float) >= min_probability)
        & (scored.policy_score_percentile.to_numpy(float) >= score_quantile)
    ].copy()
    if eligible.empty:
        return eligible
    eligible.sort_values(
        [
            "transaction_src_track_id",
            "transaction_dst_track_id",
            "policy_score",
            "src_chunk",
            "dst_chunk",
        ],
        ascending=[True, True, False, True, True],
        inplace=True,
    )
    best_by_pair: Dict[Tuple[int, int], Tuple[float, int]] = {}
    for index, row in eligible.iterrows():
        left = int(row.transaction_src_track_id)
        right = int(row.transaction_dst_track_id)
        key = (min(left, right), max(left, right))
        weight = float(row.policy_score)
        current = best_by_pair.get(key)
        if current is None or weight > current[0]:
            best_by_pair[key] = (weight, int(index))
    graph = nx.Graph()
    for (left, right), (weight, index) in sorted(best_by_pair.items()):
        if left != right:
            graph.add_edge(left, right, weight=weight, row_index=index)
    if graph.number_of_edges() == 0:
        return eligible.iloc[:0].copy()
    matching = nx.algorithms.matching.max_weight_matching(
        graph, maxcardinality=False, weight="weight"
    )
    indices = [int(graph[left][right]["row_index"]) for left, right in matching]
    selected = scored.loc[sorted(indices)].copy()
    selected.sort_values(
        ["policy_score", "src_chunk", "dst_chunk"],
        ascending=[False, True, True],
        inplace=True,
    )
    return selected


def policy_diagnostics(
    predictions: Dict[str, pd.DataFrame],
    loss_multiplier: float,
    min_probability: float,
    score_quantile: float,
) -> Dict[str, object]:
    by_sequence = []
    normalized_utilities = []
    for seq, frame in predictions.items():
        selected = maximum_weight_transaction_matching(
            frame, loss_multiplier, min_probability, score_quantile
        )
        actual = selected[TARGET].to_numpy(float)
        positive_total = float(frame.loc[frame[TARGET] > 0, TARGET].sum())
        utility = float(actual.sum())
        normalized = utility / max(positive_total, 1e-9)
        normalized_utilities.append(normalized)
        by_sequence.append(
            {
                "seq": seq,
                "actions": int(len(selected)),
                "actual_proxy_sum": utility,
                "normalized_utility": normalized,
                "positive": int((actual > 0).sum()),
                "negative": int((actual < 0).sum()),
                "precision": float((actual > 0).mean()) if len(actual) else None,
            }
        )
    values = np.asarray(normalized_utilities, dtype=float)
    return {
        "loss_multiplier": loss_multiplier,
        "min_probability": min_probability,
        "score_quantile": score_quantile,
        "worst_normalized_utility": float(values.min()) if len(values) else 0.0,
        "median_normalized_utility": float(np.median(values)) if len(values) else 0.0,
        "sum_normalized_utility": float(values.sum()),
        "positive_sequences": int((values > 0).sum()),
        "negative_sequences": int((values < 0).sum()),
        "actions": int(sum(item["actions"] for item in by_sequence)),
        "by_sequence": by_sequence,
    }


def deployment_policy_key(report: Dict[str, object]) -> Tuple[int, float, float, float, int]:
    """Train-only deployment objective.

    Prefer policies that improve more OOF sequences.  Among equal coverage,
    maximize total and median normalized utility, then the worst sequence, and
    finally prefer fewer actions.  The no-op candidate remains available.
    """

    return (
        int(report["positive_sequences"]),
        float(report["sum_normalized_utility"]),
        float(report["median_normalized_utility"]),
        float(report["worst_normalized_utility"]),
        -int(report["actions"]),
    )


def choose_deployment_policy(
    predictions: Dict[str, pd.DataFrame],
    loss_grid: Iterable[float],
    probability_grid: Iterable[float],
    quantile_grid: Iterable[float],
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    reports: List[Dict[str, object]] = [
        {
            "loss_multiplier": None,
            "min_probability": None,
            "score_quantile": None,
            "worst_normalized_utility": 0.0,
            "median_normalized_utility": 0.0,
            "sum_normalized_utility": 0.0,
            "positive_sequences": 0,
            "negative_sequences": 0,
            "actions": 0,
            "by_sequence": [
                {
                    "seq": seq,
                    "actions": 0,
                    "actual_proxy_sum": 0.0,
                    "normalized_utility": 0.0,
                    "positive": 0,
                    "negative": 0,
                    "precision": None,
                }
                for seq in predictions
            ],
        }
    ]
    for loss_multiplier in loss_grid:
        for min_probability in probability_grid:
            for score_quantile in quantile_grid:
                reports.append(
                    policy_diagnostics(
                        predictions,
                        loss_multiplier,
                        min_probability,
                        score_quantile,
                    )
                )
    return max(reports, key=deployment_policy_key), reports


def average_checkpoints(run_root: Path) -> Tuple[Path, Dict[str, object]]:
    output = run_root / "checkpoints" / "m23_26_loso_weight_average.pth"
    protocol_path = output.with_suffix(".json")
    sources = [CHECKPOINT_ROOT / seq / "model_best.pth" for seq in TRAIN_SEQUENCES]
    for path in sources:
        if not path.is_file():
            raise FileNotFoundError(path)
    source_hashes = {path.parent.name: sha256(path) for path in sources}
    if output.is_file() and protocol_path.is_file():
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        if protocol.get("source_sha256") == source_hashes:
            log(f"skip compatible averaged checkpoint: {output}")
            return output, protocol
    log("averaging four strict-LOSO FastReID checkpoints")
    objects = [torch.load(path, map_location="cpu") for path in sources]
    states = [obj["model"] for obj in objects]
    keys = list(states[0])
    if any(set(state) != set(keys) for state in states[1:]):
        raise RuntimeError("checkpoint model key sets differ")
    averaged: Dict[str, torch.Tensor] = {}
    copied_shape_mismatch = []
    averaged_keys = []
    for key in keys:
        tensors = [state[key] for state in states]
        shapes = [tuple(tensor.shape) for tensor in tensors]
        if len(set(shapes)) != 1:
            averaged[key] = tensors[0].clone()
            copied_shape_mismatch.append({"key": key, "shapes": shapes, "source": sources[0].parent.name})
        elif tensors[0].is_floating_point():
            accumulator = torch.zeros_like(tensors[0], dtype=torch.float32)
            for tensor in tensors:
                accumulator.add_(tensor.float())
            averaged[key] = (accumulator / len(tensors)).to(dtype=tensors[0].dtype)
            averaged_keys.append(key)
        else:
            averaged[key] = tensors[0].clone()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(
        {
            "model": averaged,
            "epoch": -1,
            "held_sequence": "MOT20_TEST_DEPLOYMENT",
            "val_map": None,
            "deployment": "uniform compatible-tensor average of four strict-LOSO checkpoints",
        },
        temporary,
    )
    os.replace(temporary, output)
    protocol = {
        "status": "completed",
        "created_at": now_iso(),
        "checkpoint": str(output.relative_to(REPO)),
        "checkpoint_sha256": sha256(output),
        "source_checkpoints": [str(path.relative_to(REPO)) for path in sources],
        "source_sha256": source_hashes,
        "averaged_tensor_count": len(averaged_keys),
        "copied_shape_mismatch": copied_shape_mismatch,
        "classifier_head_used_for_inference": False,
        "test_gt_read": False,
    }
    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output, protocol


def reembed_test(
    run_root: Path,
    checkpoint: Path,
    source_root: Path,
    batch_size: int,
    overwrite: bool,
) -> List[Dict[str, object]]:
    phase_root = run_root / "test_phase_reembed"
    config = BOT_ROOT / "fast_reid" / "configs" / "MOT20" / "sbs_S50.yml"
    if str(BOT_ROOT) not in sys.path:
        sys.path.insert(0, str(BOT_ROOT))
    from fast_reid.fast_reid_interfece import FastReIDInterface

    checkpoint_hash = sha256(checkpoint)
    encoder = None
    reports = []
    for seq in TEST_SEQUENCES:
        source = source_root / seq / "dump_yolox_reid.npz"
        destination_dir = phase_root / seq
        destination = destination_dir / "dump_yolox_reid.npz"
        protocol_path = destination_dir / "m23_26_reembed_protocol.json"
        if destination.is_file() and protocol_path.is_file() and not overwrite:
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
            if protocol.get("complete") and protocol.get("checkpoint_sha256") == checkpoint_hash:
                log(f"skip completed test re-embedding: {seq}")
                reports.append(protocol)
                continue
        if not source.is_file():
            raise FileNotFoundError(source)
        if encoder is None:
            log("initializing averaged FastReID inference model")
            encoder = FastReIDInterface(str(config), str(checkpoint), "gpu", batch_size=batch_size)
        destination_dir.mkdir(parents=True, exist_ok=True)
        log(f"re-embedding {seq}")
        with np.load(source, allow_pickle=False) as archive:
            detections = archive["detections"]
            frame_offsets = archive["frame_offsets"]
            columns = archive["columns"]
            image_files = archive["image_files"]
            image_wh = archive["image_wh"]
        column_index = {str(name): index for index, name in enumerate(columns.tolist())}
        required = {"x1", "y1", "x2", "y2", "has_reid"}
        if not required.issubset(column_index):
            raise RuntimeError(f"missing columns in {source}: {sorted(required - set(column_index))}")
        total_frames = len(frame_offsets) - 1
        features = np.zeros((len(detections), 2048), dtype=np.float16)
        encoded = 0
        failed_images = 0
        for zero_frame in range(total_frames):
            start = int(frame_offsets[zero_frame])
            end = int(frame_offsets[zero_frame + 1])
            if start == end:
                continue
            image_path = Path(str(image_files[zero_frame]))
            image = cv2.imread(str(image_path))
            if image is None:
                failed_images += 1
                continue
            rows = detections[start:end]
            mask = rows[:, column_index["has_reid"]] > 0.5
            if not np.any(mask):
                continue
            boxes = rows[mask][:, [
                column_index["x1"],
                column_index["y1"],
                column_index["x2"],
                column_index["y2"],
            ]].astype(np.float32)
            embedding = np.asarray(encoder.inference(image, boxes), dtype=np.float32)
            if embedding.shape != (int(mask.sum()), 2048):
                raise RuntimeError(
                    f"unexpected embedding shape {embedding.shape} seq={seq} frame={zero_frame + 1}"
                )
            row_indices = np.flatnonzero(mask) + start
            features[row_indices] = embedding.astype(np.float16)
            encoded += len(row_indices)
            if zero_frame == 0 or (zero_frame + 1) % 50 == 0 or zero_frame + 1 == total_frames:
                log(f"reembed {seq} frame={zero_frame + 1}/{total_frames} encoded={encoded}")
        temporary = destination.with_name(destination.name + ".tmp")
        with temporary.open("wb") as stream:
            np.savez(
                stream,
                detections=detections,
                features=features,
                frame_offsets=frame_offsets,
                columns=columns,
                image_files=image_files,
                image_wh=image_wh,
            )
        os.replace(temporary, destination)
        protocol = {
            "status": "completed",
            "sequence": seq,
            "source": str(source.relative_to(REPO)),
            "destination": str(destination.relative_to(REPO)),
            "detector_rows_preserved": int(len(detections)),
            "frames_total": int(total_frames),
            "features_encoded": int(encoded),
            "failed_images": int(failed_images),
            "feature_dim": 2048,
            "feature_dtype": "float16",
            "checkpoint": str(checkpoint.relative_to(REPO)),
            "checkpoint_sha256": checkpoint_hash,
            "test_gt_read": False,
            "complete": bool(failed_images == 0),
        }
        protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if not protocol["complete"]:
            raise RuntimeError(f"incomplete re-embedding for {seq}")
        reports.append(protocol)
    (phase_root / "m23_26_reembed_summary.json").write_text(
        json.dumps({"reports": reports}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return reports


def build_test_graphs(run_root: Path, parent: Path, overwrite: bool) -> List[Dict[str, object]]:
    phase_root = run_root / "test_phase_reembed"
    graph_root = run_root / "test_micrograph"
    base = load_module("m23_26_micrograph_base", "scripts/m23_research/m23_10_build_micrograph.py")
    base.PHASE = phase_root
    base.PARENT = parent
    graph_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(base.SEED)
    projection = rng.normal(size=(2048, base.DIM)).astype(np.float32) / math.sqrt(base.DIM)
    reports = []
    for seq in TEST_SEQUENCES:
        seq_dir = graph_root / seq
        expected = [
            seq_dir / "microtracklets.parquet",
            seq_dir / "prototypes.f16.npy",
            seq_dir / "candidate_edges.parquet",
            seq_dir / "protocol.json",
        ]
        if all(path.is_file() for path in expected) and not overwrite:
            protocol = json.loads(expected[-1].read_text(encoding="utf-8"))
            log(f"skip completed test micrograph: {seq}")
            reports.append(protocol)
            continue
        log(f"building GT-free test micrograph: {seq}")
        tracker_rows = base.read_tracker(parent / f"{seq}.txt")
        phase, match_iou, embedding, position = base.map_phase(tracker_rows, seq, projection)
        gt = np.zeros(len(tracker_rows), dtype=np.int32)
        metadata, prototypes = base.build_chunks(
            tracker_rows, phase, match_iou, embedding, position, gt
        )
        edges = base.build_edges(metadata, prototypes)
        seq_dir.mkdir(parents=True, exist_ok=True)
        metadata.to_parquet(seq_dir / "microtracklets.parquet", index=False)
        np.save(seq_dir / "prototypes.f16.npy", prototypes.astype(np.float16))
        edges.to_parquet(seq_dir / "candidate_edges.parquet", index=False)
        protocol = {
            "status": "completed",
            "sequence": seq,
            "parent": str(parent.relative_to(REPO)),
            "parent_sha256": sha256(parent / f"{seq}.txt"),
            "tracker_rows": int(len(tracker_rows)),
            "mapped_rows": int((phase >= 0).sum()),
            "mapping_rate": float((phase >= 0).mean()) if len(phase) else 0.0,
            "chunks": int(len(metadata)),
            "edges": int(len(edges)),
            "cross_edges": int((edges.same_source.to_numpy(int) == 0).sum()),
            "test_gt_read": False,
            "candidate_generation_gt_use": "none",
            "projection_seed": int(base.SEED),
            "projected_dim": int(base.DIM),
        }
        (seq_dir / "protocol.json").write_text(
            json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        reports.append(protocol)
        log(json.dumps(protocol, sort_keys=True))
    (graph_root / "report.json").write_text(
        json.dumps({"sequences": reports, "test_gt_read": False}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return reports


def prepare_oof_graph_root(run_root: Path) -> Path:
    oof_root = run_root / "train_oof_micrograph"
    for seq in TRAIN_SEQUENCES:
        source = LOSO_ROOT / seq / seq
        destination = oof_root / seq
        destination.mkdir(parents=True, exist_ok=True)
        for name in ("microtracklets.parquet", "prototypes.f16.npy", "candidate_edges.parquet"):
            source_file = source / name
            destination_file = destination / name
            if not source_file.is_file():
                raise FileNotFoundError(source_file)
            if not destination_file.is_file() or sha256(destination_file) != sha256(source_file):
                shutil.copy2(source_file, destination_file)
        (destination / "protocol.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "sequence": seq,
                    "source_outer_held_graph": str(source.relative_to(REPO)),
                    "feature_role": "out-of-fold train feature graph",
                    "test_gt_read": False,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return oof_root


def train_apply_and_package(
    run_root: Path,
    train_parent: Path,
    test_parent: Path,
    max_iter: int,
    loss_grid: List[float],
    probability_grid: List[float],
    quantile_grid: List[float],
) -> Dict[str, object]:
    oof_root = prepare_oof_graph_root(run_root)
    test_graph_root = run_root / "test_micrograph"
    utility_root = run_root / "train_edge_utility"
    label_root = run_root / "train_transaction_labels"
    prediction_root = run_root / "predictions"
    selected_root = run_root / "selected_transactions"
    applied_root = run_root / "applied_edges"
    track_root = run_root / "track_results"
    package_root = run_root / "package_root"
    for path in (
        utility_root,
        label_root,
        prediction_root,
        selected_root,
        applied_root,
        track_root,
        package_root,
    ):
        path.mkdir(parents=True, exist_ok=True)

    utility = load_module("m23_26_utility", "scripts/m23_research/m23_11_add_micrograph_utility.py")
    chain = load_module("m23_26_chain", "scripts/m23_research/m23_12_chain_transaction_oracle.py")
    base = load_module("m23_26_chain_features", "scripts/m23_research/m23_12_train_chain_transaction_loso.py")
    evaluator = load_module("m23_26_evaluator", "scripts/m23_research/m23_11_eval_utility_graph.py")

    utility.ROOT = oof_root
    utility.OUT = utility_root
    utility.PARENT = train_parent
    chain.DATA = oof_root
    chain.UTILITY = utility_root
    chain.PARENT = train_parent
    base.META = oof_root
    features = list(base.FEATURES) + GRAPH_FEATURES

    m23 = utility.load_m23()
    training_frames: Dict[str, pd.DataFrame] = {}
    label_reports = []
    for seq in TRAIN_SEQUENCES:
        label_path = label_root / seq / "cross_chain_transaction_utility.parquet"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        if label_path.is_file():
            transactions = pd.read_parquet(label_path)
            label_report = {"seq": seq, "reused": str(label_path.relative_to(REPO)), "rows": int(len(transactions))}
        else:
            log(f"labeling OOF train transactions: {seq}")
            edge_report = utility.label_sequence(seq, m23)
            _, _, transactions = chain.label_transactions(seq, utility)
            transactions.to_parquet(label_path, index=False)
            label_report = {"seq": seq, "edge_report": edge_report, "rows": int(len(transactions))}
        label_reports.append(label_report)
        frame = add_conflict_graph_features(base.add_chain_features(seq, transactions))
        training_frames[seq] = frame
        log(f"OOF train frame {seq}: rows={len(frame)} positive={int(frame[POSITIVE].sum())}")

    inner_predictions: Dict[str, pd.DataFrame] = {}
    for fold_index, pseudo_held in enumerate(TRAIN_SEQUENCES):
        output_path = prediction_root / f"oof_{pseudo_held}.parquet"
        if output_path.is_file():
            predicted = pd.read_parquet(output_path)
        else:
            log(f"fitting OOF transaction fold: held={pseudo_held}")
            inner_train = pd.concat(
                [training_frames[seq] for seq in TRAIN_SEQUENCES if seq != pseudo_held],
                ignore_index=True,
                sort=False,
            )
            models = fit_distributional_models(
                inner_train, features, 26000 + fold_index, max_iter
            )
            predicted = predict_distributional(training_frames[pseudo_held], models, features)
            predicted.to_parquet(output_path, index=False)
        inner_predictions[pseudo_held] = predicted

    chosen, calibration_reports = choose_deployment_policy(
        inner_predictions, loss_grid, probability_grid, quantile_grid
    )
    calibration_path = run_root / "deployment_calibration.json"
    calibration_path.write_text(
        json.dumps(
            {
                "objective": (
                    "maximize number of positive OOF train sequences, then total, median, "
                    "and worst normalized transaction utility; train GT only"
                ),
                "chosen": chosen,
                "candidates": calibration_reports,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if chosen["loss_multiplier"] is None:
        chosen_policy = None
    else:
        chosen_policy = {
            "loss_multiplier": float(chosen["loss_multiplier"]),
            "min_probability": float(chosen["min_probability"]),
            "score_quantile": float(chosen["score_quantile"]),
        }
    log(f"deployment policy: {json.dumps(chosen_policy, sort_keys=True)}")

    outer_train = pd.concat(list(training_frames.values()), ignore_index=True, sort=False)
    log(f"fitting final transaction model on all OOF train rows={len(outer_train)}")
    outer_models = fit_distributional_models(outer_train, features, 26999, max_iter)

    # Chain features for deployment must be read from the GT-free test graph,
    # not from the OOF train graph used above.
    base.META = test_graph_root
    evaluator.DATA = test_graph_root
    evaluator.PARENT = test_parent
    evaluator.SEQS = list(TEST_SEQUENCES)
    sequence_reports = []
    for seq in TEST_SEQUENCES:
        log(f"applying frozen M23-26 policy to {seq}")
        meta = pd.read_parquet(test_graph_root / seq / "microtracklets.parquet")
        edges = pd.read_parquet(test_graph_root / seq / "candidate_edges.parquet")
        structural = structural_transactions(meta, edges)
        test_features = add_conflict_graph_features(
            base.add_chain_features(seq, structural)
        )
        predictions = predict_distributional(test_features, outer_models, features)
        predictions.to_parquet(prediction_root / f"{seq}.parquet", index=False)
        if chosen_policy is None:
            selected = predictions.iloc[:0].copy()
            selected["policy_score"] = np.asarray([], dtype=float)
        else:
            selected = maximum_weight_transaction_matching(predictions, **chosen_policy)
        selected.to_parquet(selected_root / f"{seq}.parquet", index=False)
        applied = chain.apply_transactions(
            edges,
            selected.assign(**{TARGET: selected.policy_score.to_numpy(float)}),
        )
        for column, default in (
            ("assa_edge_delta_proxy", 0.0),
            ("assa_edge_positive", 0),
            ("assa_edge_negative", 0),
        ):
            if column not in applied:
                applied[column] = default
        applied.to_parquet(applied_root / f"{seq}.parquet", index=False)
        output_txt = track_root / f"{seq}.txt"
        tracker_report = evaluator.write_tracker(seq, meta, applied, output_txt)
        package_file = package_root / f"{seq}.txt"
        shutil.copy2(output_txt, package_file)
        report = {
            "sequence": seq,
            "candidates": int(len(predictions)),
            "selected_actions": int(len(selected)),
            "selected_score_sum": float(selected.policy_score.sum()),
            "parent_sha256": sha256(test_parent / f"{seq}.txt"),
            "output_sha256": sha256(output_txt),
            "output_bytes": output_txt.stat().st_size,
            **tracker_report,
        }
        sequence_reports.append(report)
        log(json.dumps(report, sort_keys=True))

    validation_dir_log = run_root / "precheck_results_dir.log"
    validation_zip_log = run_root / "precheck_zip.log"
    command = [
        sys.executable,
        str(REPO / "scripts" / "check_mot20_submission.py"),
        "--results-dir",
        str(package_root),
        "--profile",
        "mot20_test_4",
    ]
    completed = subprocess.run(command, cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    validation_dir_log.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(completed.stdout[-5000:])

    zip_path = run_root / "MOT20_M23_26_OOF_ensemble_test_submission.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for seq in TEST_SEQUENCES:
            archive.write(package_root / f"{seq}.txt", arcname=f"{seq}.txt")
    command = [
        sys.executable,
        str(REPO / "scripts" / "check_mot20_submission.py"),
        "--zip-path",
        str(zip_path),
        "--profile",
        "mot20_test_4",
    ]
    completed = subprocess.run(command, cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    validation_zip_log.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(completed.stdout[-5000:])

    return {
        "status": "completed",
        "created_at": now_iso(),
        "protocol": "OOF train transaction model plus LOSO-weight-averaged FastReID; test GT-free",
        "test_gt_read": False,
        "train_parent": str(train_parent.relative_to(REPO)),
        "test_parent": str(test_parent.relative_to(REPO)),
        "features": features,
        "label_reports": label_reports,
        "calibration": chosen,
        "chosen_policy": chosen_policy,
        "sequences": sequence_reports,
        "package_root": str(package_root.relative_to(REPO)),
        "zip_path": str(zip_path.relative_to(REPO)),
        "zip_sha256": sha256(zip_path),
        "zip_bytes": zip_path.stat().st_size,
        "precheck_results_dir": str(validation_dir_log.relative_to(REPO)),
        "precheck_zip": str(validation_zip_log.relative_to(REPO)),
        "files": [
            {
                "name": f"{seq}.txt",
                "sha256": sha256(package_root / f"{seq}.txt"),
                "bytes": (package_root / f"{seq}.txt").stat().st_size,
            }
            for seq in TEST_SEQUENCES
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument("--test-phase0", default=str(DEFAULT_TEST_PHASE0))
    parser.add_argument("--test-parent", default=str(DEFAULT_TEST_PARENT))
    parser.add_argument("--train-parent", default=str(DEFAULT_TRAIN_PARENT))
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--max-iter", type=int, default=220)
    parser.add_argument("--loss-grid", default="1,2,4,8,16,32")
    parser.add_argument("--probability-grid", default="0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--quantile-grid", default="0.99,0.995,0.9975,0.999,0.9995")
    parser.add_argument("--overwrite-reembed", action="store_true")
    parser.add_argument("--overwrite-graph", action="store_true")
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    test_phase0 = Path(args.test_phase0).resolve()
    test_parent = Path(args.test_parent).resolve()
    train_parent = Path(args.train_parent).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    running_protocol = {
        "status": "running",
        "started_at": now_iso(),
        "script": str(Path(__file__).relative_to(REPO)),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "test_gt_read": False,
    }
    protocol_path = run_root / "submission_manifest.json"
    protocol_path.write_text(json.dumps(running_protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        checkpoint, checkpoint_protocol = average_checkpoints(run_root)
        reembed_reports = reembed_test(
            run_root,
            checkpoint,
            test_phase0,
            args.batch_size,
            args.overwrite_reembed,
        )
        graph_reports = build_test_graphs(run_root, test_parent, args.overwrite_graph)
        manifest = train_apply_and_package(
            run_root,
            train_parent,
            test_parent,
            args.max_iter,
            parse_float_grid(args.loss_grid),
            parse_float_grid(args.probability_grid),
            parse_float_grid(args.quantile_grid),
        )
        manifest.update(
            {
                "script": str(Path(__file__).relative_to(REPO)),
                "git_commit_at_start": running_protocol["git_commit"],
                "checkpoint_protocol": checkpoint_protocol,
                "reembed_reports": reembed_reports,
                "graph_reports": graph_reports,
            }
        )
        protocol_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    except Exception as exc:
        running_protocol["status"] = "failed"
        running_protocol["failed_at"] = now_iso()
        running_protocol["error"] = f"{type(exc).__name__}: {exc}"
        protocol_path.write_text(json.dumps(running_protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
