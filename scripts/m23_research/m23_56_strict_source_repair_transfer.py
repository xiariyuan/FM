#!/usr/bin/env python3
from __future__ import annotations

"""M23-56 strict nested-LOSO source-repair structured transfer.

The immutable M23-55 graph is converted to observable source-anchored repair
transactions before any M23-56 label is opened.  Each outer run reads teacher
labels only for its three training sequences.  Inner and outer trackers are
written and hashed before the corresponding validation/held labels are opened.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import resource
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from scipy.stats import beta
from sklearn.metrics import average_precision_score, precision_recall_curve

REPO = Path(__file__).resolve().parents[2]
SEQUENCES = ("MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05")
SEQ_INDEX = {seq: index for index, seq in enumerate(SEQUENCES)}
SEQ_SHORT = {"MOT20-01": "m01", "MOT20-02": "m02", "MOT20-03": "m03", "MOT20-05": "m05"}
RUN_ROOT = Path("outputs/mot20_m23_20260718/m23_56_strict_source_repair_transfer_v1")
PREREG_SHA256 = "8ba1c85b791614bae498afe30f2d2e8441c11d8f6340412438644370c7d71583"
M55_ROOTS = {
    seq: Path(f"outputs/mot20_m23_20260718/m23_55_stratified_gap_candidates_{SEQ_SHORT[seq]}_v1")
    for seq in SEQUENCES
}
BASELINE_CACHE = Path("outputs/mot20_m23_20260718/m23_47_m23_46_deployable_baseline_cache_v1")
SOURCE_PARENT = Path(
    "outputs/assocriskbench_p15_20260714/fused_a42_adaptive_risk_eval/"
    "l1_r2_q75_adaptive_priority/track_results"
)
M46_REPORTS = {
    seq: Path(f"outputs/mot20_m23_20260718/m23_46_pure_hgb_topk_{SEQ_SHORT[seq]}_full_v1/report.json")
    for seq in SEQUENCES
}
M46_COMBINED = Path("outputs/mot20_m23_20260718/m23_46_pure_hgb_topk_combined_oof_v1/report.json")

EDGE_BASE = (
    "appearance_cos", "recall_score", "multi_cos", "whole_cos", "motion_min",
    "endpoint_disp", "velocity_cos", "abs_log_height", "log_gap",
    "consistency_min", "out_rank", "in_rank", "reciprocal_out",
    "reciprocal_in", "mutual_app", "mutual_motion",
)
PREFIXES = ("source", "cross", "incoming")
PAIR_FEATURES = (
    "pair_exists", "pair_recall_score", "pair_appearance_cos",
    "pair_motion_min", "pair_rank_min",
)
RELATIVE_FEATURES = (
    "rel_recall", "rel_multi", "rel_whole", "rel_appearance",
    "rel_motion_improvement", "rel_disp_improvement", "rel_velocity",
    "rel_rank_improvement", "rel_consistency", "rel_incoming_recall",
)
CONTEXT_FEATURES = (
    "cross_gap", "src_rows_log", "dst_rows_log", "src_node_consistency",
    "dst_node_consistency", "out_degree_log", "in_degree_log", "out_entropy",
    "in_entropy", "out_best_margin", "in_best_margin", "conflict_degree_log",
    "crowd_src", "crowd_dst", "crowd_delta", "has_source_out",
    "has_source_in", "transaction_single", "transaction_double",
    "transaction_paired", "origin_stratified", "origin_legacy",
)
FEATURES = tuple(f"{prefix}_{name}" for prefix in PREFIXES for name in EDGE_BASE) + PAIR_FEATURES + RELATIVE_FEATURES + CONTEXT_FEATURES
BINARY_FEATURES = {
    *(f"{prefix}_{name}" for prefix in PREFIXES for name in ("mutual_app", "mutual_motion")),
    "pair_exists", "has_source_out", "has_source_in", "transaction_single",
    "transaction_double", "transaction_paired", "origin_stratified", "origin_legacy",
}
FORBIDDEN_TOKENS = (
    "same_gt", "modal_gt", "purity", "label_confidence", "actual_assa",
    "delta_hota", "matched_gt", "teacher",
)
POLICIES = {"P0": None, "P1": 0.02, "P2": 0.05}
EPOCHS = 20
SMOKE_EPOCHS = 2
HIDDEN = 64
BASE_SEED = 235600
FOCAL_GAMMA = 2.0
LOSS_WEIGHTS = {
    "source": 1.0, "continuity": 1.0, "risk": 1.5, "imitation": 1.0,
    "out_listwise": 0.5, "in_listwise": 0.5, "group_dro": 0.5,
    "sparsity": 0.02,
}
SOURCE_ANCHOR = 2.0
EDIT_PENALTY = 1.0
CUT_PENALTY = 1.5
RISK_LAMBDA = 4.0
INCOMING_DISPLACEMENT_PENALTY = 1.0
RELATIVE_COEFFICIENT = 0.5
PAIR_BONUS = 0.25
CONFIDENCE_ALPHA = 0.05
LCB_MINIMUM = 0.5
INNER_MIN_FOLD_DELTA = 0.05
INNER_MIN_MEAN_DELTA = 0.20


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def json_write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_event(root: Path, event: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with (root / "protocol_events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time": utc_now(), **event}, sort_keys=True) + "\n")


def load_module(name: str, relative: str):
    path = REPO / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_preregistration(root: Path) -> dict:
    path = root / "preregistered_protocol.json"
    if not path.exists():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != PREREG_SHA256:
        raise RuntimeError(f"preregistration hash changed: {actual} != {PREREG_SHA256}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("frozen_before_training"):
        raise RuntimeError("preregistration is not frozen")
    return payload


def audit_feature_names(columns: Iterable[str]) -> list[str]:
    return sorted(column for column in columns if any(token in column.lower() for token in FORBIDDEN_TOKENS))


def detector_payload(path: Path) -> str:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split(",")
            if len(fields) >= 6:
                rows.append((int(float(fields[0])), tuple(fields[2:])))
    rows.sort()
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def baseline_tracker(seq: str) -> Path:
    return BASELINE_CACHE / seq / "track_results" / f"{seq}.txt"


def m46_metrics(seq: str) -> dict:
    report = json.loads(M46_REPORTS[seq].read_text(encoding="utf-8"))
    return report["eval"]


def verify_m55_graph(seq: str) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    root = M55_ROOTS[seq]
    manifest_path = root / "frozen_candidate_graph" / "freeze_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("candidate_graph_frozen") is not True or manifest.get("gt_opened") is not False:
        raise RuntimeError(f"invalid M23-55 freeze state: {seq}")
    artifacts = manifest["frozen_artifacts"]
    for path_key, hash_key in (("nodes", "nodes_sha256"), ("edges", "edges_sha256")):
        path = Path(artifacts[path_key])
        if sha256_file(path) != artifacts[hash_key]:
            raise RuntimeError(f"M23-55 hash mismatch: {path}")
    nodes = pd.read_parquet(artifacts["nodes"])
    edges = pd.read_parquet(artifacts["edges"])
    forbidden = audit_feature_names(list(nodes.columns) + list(edges.columns))
    if forbidden:
        raise RuntimeError(f"forbidden frozen observable columns: {forbidden}")
    first = nodes.first_frame.to_numpy(int)
    last = nodes.last_frame.to_numpy(int)
    if np.any(first[edges.dst_chunk.to_numpy(int)] <= last[edges.src_chunk.to_numpy(int)]):
        raise RuntimeError("M23-55 graph contains non-forward edge")
    return manifest, nodes, edges


def numeric(frame: pd.DataFrame, name: str, default: float = 0.0) -> np.ndarray:
    if name not in frame:
        return np.full(len(frame), default, np.float32)
    values = pd.to_numeric(frame[name], errors="coerce").to_numpy(np.float64)
    values[~np.isfinite(values)] = default
    return values.astype(np.float32)


def add_edge_base(frame: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame(index=frame.index)
    appearance = numeric(frame, "appearance_cos")
    recall = numeric(frame, "m23_55_recall_score", np.nan)
    recall = np.where(np.isfinite(recall), recall, appearance).astype(np.float32)
    multi = numeric(frame, "m23_55_multi_appearance_cos", np.nan)
    multi = np.where(np.isfinite(multi), multi, appearance).astype(np.float32)
    whole = numeric(frame, "m23_55_whole_appearance_cos", np.nan)
    whole = np.where(np.isfinite(whole), whole, appearance).astype(np.float32)
    out_rank = numeric(frame, "appearance_out_rank", np.nan)
    fallback_out = numeric(frame, "out_rank", 1e6)
    out_rank = np.where(np.isfinite(out_rank), out_rank, fallback_out)
    in_rank = numeric(frame, "appearance_in_rank", np.nan)
    fallback_in = numeric(frame, "in_rank", 1e6)
    in_rank = np.where(np.isfinite(in_rank), in_rank, fallback_in)
    output["appearance_cos"] = appearance
    output["recall_score"] = recall
    output["multi_cos"] = multi
    output["whole_cos"] = whole
    output["motion_min"] = numeric(frame, "motion_error_min")
    output["endpoint_disp"] = numeric(frame, "endpoint_displacement")
    output["velocity_cos"] = numeric(frame, "velocity_cos")
    output["abs_log_height"] = np.abs(numeric(frame, "log_height_ratio"))
    output["log_gap"] = numeric(frame, "log_gap")
    output["consistency_min"] = numeric(frame, "consistency_min")
    output["out_rank"] = np.clip(out_rank, 0, 1e6)
    output["in_rank"] = np.clip(in_rank, 0, 1e6)
    output["reciprocal_out"] = 1.0 / (1.0 + output.out_rank.to_numpy(float))
    output["reciprocal_in"] = 1.0 / (1.0 + output.in_rank.to_numpy(float))
    output["mutual_app"] = numeric(frame, "mutual_appearance_topk")
    output["mutual_motion"] = numeric(frame, "mutual_motion_topk")
    return output.astype(np.float32)


def active_counts(nodes: pd.DataFrame, query: np.ndarray) -> np.ndarray:
    starts = np.sort(nodes.first_frame.to_numpy(int))
    ends = np.sort(nodes.last_frame.to_numpy(int))
    return (
        np.searchsorted(starts, query, side="right")
        - np.searchsorted(ends, query, side="left")
    ).astype(np.float32)


def group_entropy(frame: pd.DataFrame, group: str, score: str) -> pd.Series:
    maximum = frame.groupby(group, sort=False)[score].transform("max")
    exponent = np.exp(np.clip(frame[score].to_numpy(float) - maximum.to_numpy(float), -30, 0))
    total = pd.Series(exponent, index=frame.index).groupby(frame[group], sort=False).transform("sum")
    probability = exponent / np.maximum(total.to_numpy(float), 1e-12)
    contribution = -probability * np.log(np.maximum(probability, 1e-12))
    return pd.Series(contribution, index=frame.index).groupby(frame[group], sort=False).transform("sum")


def freeze_observable_sequence(seq: str, root: Path) -> dict:
    verify_preregistration(root)
    output = root / "observable_transactions" / seq
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite observable freeze: {manifest_path}")
    started = time.time()
    m55_manifest, nodes, edges = verify_m55_graph(seq)
    parent_mask = edges.parent_edge.fillna(0).to_numpy(int) == 1
    parent = edges.loc[parent_mask].copy().reset_index(drop=True)
    cross = edges.loc[~parent_mask].copy().reset_index(drop=True)
    if parent.src_chunk.duplicated().any() or parent.dst_chunk.duplicated().any():
        raise RuntimeError("M23-46 parent graph is not one-to-one")
    parent_base = add_edge_base(parent)
    cross_base = add_edge_base(cross)
    for name in EDGE_BASE:
        cross[f"cross_{name}"] = cross_base[name].to_numpy(np.float32)
    cross["obs_out_order"] = (
        cross.groupby("src_chunk", sort=False)["cross_recall_score"]
        .rank(method="first", ascending=False).astype(np.int32)
    )
    cross["obs_in_order"] = (
        cross.groupby("dst_chunk", sort=False)["cross_recall_score"]
        .rank(method="first", ascending=False).astype(np.int32)
    )
    source_lookup = pd.DataFrame({
        "src_chunk": parent.src_chunk.astype(np.int32),
        "source_out_dst": parent.dst_chunk.astype(np.int32),
    })
    incoming_lookup = pd.DataFrame({
        "dst_chunk": parent.dst_chunk.astype(np.int32),
        "incoming_src": parent.src_chunk.astype(np.int32),
    })
    for name in EDGE_BASE:
        source_lookup[f"source_{name}"] = parent_base[name].to_numpy(np.float32)
        incoming_lookup[f"incoming_{name}"] = parent_base[name].to_numpy(np.float32)
    tx = cross[[
        "src_chunk", "dst_chunk", "src_track", "dst_track", "gap", "candidate_origin",
        "obs_out_order", "obs_in_order", *[f"cross_{name}" for name in EDGE_BASE]
    ]].copy()
    tx = tx.merge(source_lookup, on="src_chunk", how="left", sort=False)
    tx = tx.merge(incoming_lookup, on="dst_chunk", how="left", sort=False)
    tx["has_source_out"] = tx.source_out_dst.notna().astype(np.int8)
    tx["has_source_in"] = tx.incoming_src.notna().astype(np.int8)
    tx["source_out_dst"] = tx.source_out_dst.fillna(-1).astype(np.int32)
    tx["incoming_src"] = tx.incoming_src.fillna(-1).astype(np.int32)
    for prefix in ("source", "incoming"):
        for name in EDGE_BASE:
            tx[f"{prefix}_{name}"] = pd.to_numeric(tx[f"{prefix}_{name}"], errors="coerce").fillna(0.0).astype(np.float32)

    node_count = len(nodes)
    edge_keys = edges.src_chunk.to_numpy(np.int64) * node_count + edges.dst_chunk.to_numpy(np.int64)
    order = np.argsort(edge_keys, kind="mergesort")
    sorted_keys = edge_keys[order]
    query_valid = (tx.incoming_src.to_numpy(int) >= 0) & (tx.source_out_dst.to_numpy(int) >= 0)
    query_keys = np.full(len(tx), -1, np.int64)
    query_keys[query_valid] = (
        tx.incoming_src.to_numpy(np.int64)[query_valid] * node_count
        + tx.source_out_dst.to_numpy(np.int64)[query_valid]
    )
    positions = np.searchsorted(sorted_keys, query_keys)
    pair_exists = query_valid & (positions < len(sorted_keys))
    safe_positions = np.clip(positions, 0, max(len(sorted_keys) - 1, 0))
    pair_exists &= sorted_keys[safe_positions] == query_keys
    pair_rows = np.zeros(len(tx), np.int64)
    pair_rows[pair_exists] = order[safe_positions[pair_exists]]
    pair_base = add_edge_base(edges.iloc[pair_rows].reset_index(drop=True)) if len(edges) else pd.DataFrame()
    tx["pair_exists"] = pair_exists.astype(np.int8)
    tx["pair_recall_score"] = np.where(pair_exists, pair_base.recall_score, 0.0).astype(np.float32)
    tx["pair_appearance_cos"] = np.where(pair_exists, pair_base.appearance_cos, 0.0).astype(np.float32)
    tx["pair_motion_min"] = np.where(pair_exists, pair_base.motion_min, 0.0).astype(np.float32)
    tx["pair_rank_min"] = np.where(
        pair_exists, np.minimum(pair_base.out_rank, pair_base.in_rank), 0.0
    ).astype(np.float32)

    tx["rel_recall"] = tx.cross_recall_score - tx.source_recall_score
    tx["rel_multi"] = tx.cross_multi_cos - tx.source_multi_cos
    tx["rel_whole"] = tx.cross_whole_cos - tx.source_whole_cos
    tx["rel_appearance"] = tx.cross_appearance_cos - tx.source_appearance_cos
    tx["rel_motion_improvement"] = tx.source_motion_min - tx.cross_motion_min
    tx["rel_disp_improvement"] = tx.source_endpoint_disp - tx.cross_endpoint_disp
    tx["rel_velocity"] = tx.cross_velocity_cos - tx.source_velocity_cos
    tx["rel_rank_improvement"] = (
        np.minimum(tx.source_out_rank, tx.source_in_rank)
        - np.minimum(tx.cross_out_rank, tx.cross_in_rank)
    )
    tx["rel_consistency"] = tx.cross_consistency_min - tx.source_consistency_min
    tx["rel_incoming_recall"] = tx.cross_recall_score - tx.incoming_recall_score

    src = tx.src_chunk.to_numpy(int)
    dst = tx.dst_chunk.to_numpy(int)
    tx["cross_gap"] = tx.gap.astype(np.float32)
    tx["src_rows_log"] = np.log1p(nodes.rows.to_numpy(float)[src]).astype(np.float32)
    tx["dst_rows_log"] = np.log1p(nodes.rows.to_numpy(float)[dst]).astype(np.float32)
    tx["src_node_consistency"] = nodes.appearance_consistency.to_numpy(float)[src].astype(np.float32)
    tx["dst_node_consistency"] = nodes.appearance_consistency.to_numpy(float)[dst].astype(np.float32)
    out_degree = tx.groupby("src_chunk", sort=False).src_chunk.transform("size").to_numpy(float)
    in_degree = tx.groupby("dst_chunk", sort=False).dst_chunk.transform("size").to_numpy(float)
    tx["out_degree_log"] = np.log1p(out_degree).astype(np.float32)
    tx["in_degree_log"] = np.log1p(in_degree).astype(np.float32)
    tx["out_entropy"] = group_entropy(tx, "src_chunk", "cross_recall_score").astype(np.float32)
    tx["in_entropy"] = group_entropy(tx, "dst_chunk", "cross_recall_score").astype(np.float32)
    out_best = tx.groupby("src_chunk", sort=False).cross_recall_score.transform("max")
    in_best = tx.groupby("dst_chunk", sort=False).cross_recall_score.transform("max")
    tx["out_best_margin"] = (out_best - tx.cross_recall_score).astype(np.float32)
    tx["in_best_margin"] = (in_best - tx.cross_recall_score).astype(np.float32)
    tx["conflict_degree_log"] = np.log1p(out_degree + in_degree).astype(np.float32)
    crowd_src = active_counts(nodes, nodes.last_frame.to_numpy(int)[src])
    crowd_dst = active_counts(nodes, nodes.first_frame.to_numpy(int)[dst])
    tx["crowd_src"] = crowd_src
    tx["crowd_dst"] = crowd_dst
    tx["crowd_delta"] = crowd_dst - crowd_src
    both = (tx.has_source_out.to_numpy(int) == 1) & (tx.has_source_in.to_numpy(int) == 1)
    tx["transaction_paired"] = (both & pair_exists).astype(np.int8)
    tx["transaction_double"] = (both & ~pair_exists).astype(np.int8)
    tx["transaction_single"] = (~both).astype(np.int8)
    origin = tx.candidate_origin.astype(str)
    tx["origin_stratified"] = origin.str.contains("m23_55_stratified", regex=False).astype(np.int8)
    tx["origin_legacy"] = (1 - tx.origin_stratified).astype(np.int8)

    identifiers = [
        "src_chunk", "dst_chunk", "source_out_dst", "incoming_src", "obs_out_order",
        "obs_in_order",
    ]
    tx = tx[identifiers + list(FEATURES)].copy()
    forbidden = audit_feature_names(tx.columns)
    if forbidden:
        raise RuntimeError(f"forbidden transaction columns: {forbidden}")
    for column in FEATURES:
        if column in BINARY_FEATURES:
            tx[column] = tx[column].fillna(0).astype(np.int8)
        else:
            tx[column] = pd.to_numeric(tx[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    tx["src_chunk"] = tx.src_chunk.astype(np.int32)
    tx["dst_chunk"] = tx.dst_chunk.astype(np.int32)

    parent_sources = source_lookup[["src_chunk", "source_out_dst", *[f"source_{name}" for name in EDGE_BASE]]].copy()
    best = tx.sort_values(
        ["src_chunk", "cross_recall_score", "dst_chunk"],
        ascending=[True, False, True], kind="mergesort",
    ).drop_duplicates("src_chunk", keep="first")
    best = best.set_index("src_chunk")
    source_rows = pd.DataFrame(index=parent_sources.src_chunk.astype(int))
    for column in FEATURES:
        source_rows[column] = best[column].reindex(source_rows.index).to_numpy()
    source_rows.insert(0, "src_chunk", source_rows.index.astype(np.int32))
    source_rows.insert(1, "parent_dst", parent_sources.set_index("src_chunk").source_out_dst.reindex(source_rows.index).to_numpy(np.int32))
    for name in EDGE_BASE:
        column = f"source_{name}"
        source_rows[column] = parent_sources.set_index("src_chunk")[column].reindex(source_rows.src_chunk).to_numpy(np.float32)
    for column in FEATURES:
        if column in BINARY_FEATURES:
            source_rows[column] = pd.to_numeric(source_rows[column], errors="coerce").fillna(0).astype(np.int8)
        else:
            source_rows[column] = pd.to_numeric(source_rows[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    source_rows.reset_index(drop=True, inplace=True)

    output.mkdir(parents=True, exist_ok=True)
    cross_path = output / "cross_transactions.parquet"
    source_path = output / "source_transactions.parquet"
    cross_features_path = output / "cross_features.f16.npy"
    source_features_path = output / "source_features.f16.npy"
    tx.to_parquet(cross_path, index=False)
    source_rows.to_parquet(source_path, index=False)
    np.save(cross_features_path, normalize_frame(tx).astype(np.float16))
    np.save(source_features_path, normalize_frame(source_rows).astype(np.float16))
    manifest = {
        "experiment": "M23-56 GT-free observable transaction freeze",
        "seq": seq,
        "frozen_before_m23_56_labels": True,
        "teacher_labels_opened": False,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "preregistration_sha256": PREREG_SHA256,
        "m23_55_candidate_graph_sha256": m55_manifest["frozen_artifacts"]["edges_sha256"],
        "feature_columns": list(FEATURES),
        "binary_features": sorted(BINARY_FEATURES),
        "forbidden_columns": forbidden,
        "artifacts": {
            "cross_transactions": str(cross_path),
            "cross_transactions_sha256": sha256_file(cross_path),
            "cross_rows": len(tx),
            "source_transactions": str(source_path),
            "source_transactions_sha256": sha256_file(source_path),
            "source_rows": len(source_rows),
            "cross_features": str(cross_features_path),
            "cross_features_sha256": sha256_file(cross_features_path),
            "source_features": str(source_features_path),
            "source_features_sha256": sha256_file(source_features_path),
        },
        "runtime_seconds": time.time() - started,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }
    json_write(manifest_path, manifest)
    append_event(root, {"event": "observable_transactions_frozen", "seq": seq, "manifest": str(manifest_path), "manifest_sha256": sha256_file(manifest_path), "teacher_labels_opened": False})
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return manifest


def verify_observable(seq: str, root: Path) -> tuple[dict, Path, Path, Path, Path]:
    path = root / "observable_transactions" / seq / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("teacher_labels_opened") is not False or manifest.get("preregistration_sha256") != PREREG_SHA256:
        raise RuntimeError(f"invalid observable transaction manifest: {seq}")
    artifacts = manifest["artifacts"]
    cross_path = Path(artifacts["cross_transactions"])
    source_path = Path(artifacts["source_transactions"])
    cross_features_path = Path(artifacts["cross_features"])
    source_features_path = Path(artifacts["source_features"])
    if sha256_file(cross_path) != artifacts["cross_transactions_sha256"]:
        raise RuntimeError(f"cross transaction hash mismatch: {seq}")
    if sha256_file(source_path) != artifacts["source_transactions_sha256"]:
        raise RuntimeError(f"source transaction hash mismatch: {seq}")
    if sha256_file(cross_features_path) != artifacts["cross_features_sha256"]:
        raise RuntimeError(f"cross feature hash mismatch: {seq}")
    if sha256_file(source_features_path) != artifacts["source_features_sha256"]:
        raise RuntimeError(f"source feature hash mismatch: {seq}")
    if manifest["feature_columns"] != list(FEATURES):
        raise RuntimeError("feature allowlist changed")
    return manifest, cross_path, source_path, cross_features_path, source_features_path


def normalize_frame(frame: pd.DataFrame) -> np.ndarray:
    matrix = np.empty((len(frame), len(FEATURES)), np.float32)
    for index, column in enumerate(FEATURES):
        values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        fill = float(values.median()) if values.notna().any() else 0.0
        values = values.fillna(fill)
        if column in BINARY_FEATURES:
            matrix[:, index] = values.to_numpy(np.float32)
        elif len(values) == 0 or values.nunique(dropna=False) <= 1:
            matrix[:, index] = 0.0
        else:
            matrix[:, index] = (2.0 * values.rank(method="average", pct=True).to_numpy(np.float32) - 1.0)
    return matrix


def frozen_matrix(frame: pd.DataFrame) -> np.ndarray:
    return frame.loc[:, FEATURES].to_numpy(np.float32, copy=False)


def teacher_labels(seq: str, outer_held: str) -> tuple[pd.DataFrame, set[tuple[int, int]]]:
    if seq == outer_held:
        raise RuntimeError(f"attempted to open outer-held teacher labels: {seq}")
    root = M55_ROOTS[seq]
    utility_path = root / "teacher_identity_flow" / "teacher_edge_utilities.parquet"
    selected_path = root / "teacher_identity_flow" / "selected_path_cover_edges.parquet"
    utility = pd.read_parquet(
        utility_path,
        columns=["src_chunk", "dst_chunk", "parent_edge", "teacher_same_identity_forward"],
    )
    selected = pd.read_parquet(selected_path, columns=["src_chunk", "dst_chunk"])
    selected_keys = set(zip(selected.src_chunk.astype(int), selected.dst_chunk.astype(int)))
    return utility, selected_keys


def calibration_mask(src_chunk: np.ndarray, seq: str) -> np.ndarray:
    values = (src_chunk.astype(np.int64) * 2654435761 + SEQ_INDEX[seq] * 97) & 0x7FFFFFFF
    return values % 5 == 0


def labeled_sequence(seq: str, outer_held: str, root: Path) -> dict:
    _, cross_path, source_path, cross_features_path, source_features_path = verify_observable(seq, root)
    cross = pd.read_parquet(cross_path)
    source = pd.read_parquet(source_path)
    cross["_row_index"] = np.arange(len(cross), dtype=np.int64)
    source_matrix = np.load(source_features_path, mmap_mode="r")
    cross_matrix = np.load(cross_features_path, mmap_mode="r")
    utility, selected_keys = teacher_labels(seq, outer_held)
    utility_cross = utility[utility.parent_edge.to_numpy(int) == 0][
        ["src_chunk", "dst_chunk", "teacher_same_identity_forward"]
    ].copy()
    utility_cross.rename(columns={"teacher_same_identity_forward": "cross_valid"}, inplace=True)
    cross = cross.merge(utility_cross, on=["src_chunk", "dst_chunk"], how="left", validate="one_to_one")
    cross["cross_valid"] = cross.cross_valid.fillna(0).astype(np.int8)
    cross["imitation"] = [
        int((int(src), int(dst)) in selected_keys)
        for src, dst in zip(cross.src_chunk, cross.dst_chunk)
    ]
    valid_rank = (
        cross[cross.cross_valid == 1]
        .groupby("src_chunk", sort=False)["cross_recall_score"]
        .rank(method="first", ascending=False)
    )
    cross["valid_rank"] = np.inf
    cross.loc[cross.cross_valid == 1, "valid_rank"] = valid_rank.to_numpy(float)
    sample = (
        (cross.obs_out_order.to_numpy(int) <= 16)
        | (cross.obs_in_order.to_numpy(int) <= 4)
        | (cross.imitation.to_numpy(int) == 1)
        | ((cross.cross_valid.to_numpy(int) == 1) & (cross.valid_rank.to_numpy(float) <= 4))
    )
    selected_rows = cross.loc[sample, "_row_index"].to_numpy(np.int64)
    cross = cross.loc[sample].copy().reset_index(drop=True)
    for index, column in enumerate(FEATURES):
        cross[column] = np.asarray(cross_matrix[selected_rows, index], np.float32)
    selected_parent = set(
        (int(row.src_chunk), int(row.dst_chunk))
        for row in utility[utility.parent_edge.to_numpy(int) == 1].itertuples(index=False)
        if (int(row.src_chunk), int(row.dst_chunk)) in selected_keys
    )
    source["source_keep"] = [
        int((int(src), int(dst)) in selected_parent)
        for src, dst in zip(source.src_chunk, source.parent_dst)
    ]
    source["calibration"] = calibration_mask(source.src_chunk.to_numpy(int), seq)
    cross["calibration"] = calibration_mask(cross.src_chunk.to_numpy(int), seq)
    for index, column in enumerate(FEATURES):
        source[column] = np.asarray(source_matrix[:, index], np.float32)
    cross.drop(columns=["_row_index"], inplace=True)
    source["seq"] = seq
    cross["seq"] = seq
    return {"source": source, "cross": cross}


class RepairMLP(torch.nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(input_dim, HIDDEN),
            torch.nn.GELU(),
            torch.nn.Linear(HIDDEN, HIDDEN),
            torch.nn.GELU(),
        )
        self.source_keep = torch.nn.Linear(HIDDEN, 1)
        self.cross_continuity = torch.nn.Linear(HIDDEN, 1)
        self.catastrophic_risk = torch.nn.Linear(HIDDEN, 1)
        self.outgoing = torch.nn.Linear(HIDDEN, 1)
        self.incoming = torch.nn.Linear(HIDDEN, 1)
        self.imitation = torch.nn.Linear(HIDDEN, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.trunk(x)
        return {
            "source_keep": self.source_keep(hidden).squeeze(-1),
            "cross_continuity": self.cross_continuity(hidden).squeeze(-1),
            "catastrophic_risk": self.catastrophic_risk(hidden).squeeze(-1),
            "outgoing": self.outgoing(hidden).squeeze(-1),
            "incoming": self.incoming(hidden).squeeze(-1),
            "imitation": self.imitation(hidden).squeeze(-1),
        }


def class_balanced_focal(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = target.float()
    positive = torch.clamp(target.sum(), min=1.0)
    negative = torch.clamp((1.0 - target).sum(), min=1.0)
    total = positive + negative
    weight_positive = total / (2.0 * positive)
    weight_negative = total / (2.0 * negative)
    probability = torch.sigmoid(logits)
    probability_target = torch.where(target > 0.5, probability, 1.0 - probability)
    class_weight = torch.where(target > 0.5, weight_positive, weight_negative)
    loss = -class_weight * torch.pow(1.0 - probability_target, FOCAL_GAMMA) * torch.log(torch.clamp(probability_target, min=1e-7))
    return loss.mean()


def listwise_loss(scores: torch.Tensor, groups: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
    positive = selected > 0.5
    if not bool(positive.any()):
        return scores.sum() * 0.0
    group_count = int(groups.max().item()) + 1
    maximum = torch.full((group_count,), -torch.inf, device=scores.device)
    maximum.scatter_reduce_(0, groups, scores, reduce="amax", include_self=True)
    exponential = torch.exp(scores - maximum[groups])
    denominator = torch.zeros(group_count, device=scores.device)
    denominator.scatter_add_(0, groups, exponential)
    log_denominator = maximum[groups] + torch.log(torch.clamp(denominator[groups], min=1e-12))
    return (log_denominator[positive] - scores[positive]).mean()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def factorized(values: Sequence[tuple[str, int]]) -> np.ndarray:
    _, codes = np.unique(np.asarray([f"{seq}:{value}" for seq, value in values], object), return_inverse=True)
    return codes.astype(np.int64)


def combine_labeled(parts: list[dict], calibration: bool) -> dict:
    source = pd.concat(
        [part["source"][part["source"].calibration == calibration] for part in parts],
        ignore_index=True, sort=False,
    )
    cross = pd.concat(
        [part["cross"][part["cross"].calibration == calibration] for part in parts],
        ignore_index=True, sort=False,
    )
    return {"source": source, "cross": cross}


def train_model(parts: list[dict], seed: int, output: Path, epochs: int = EPOCHS, smoke_limit: int = 0) -> tuple[RepairMLP, dict]:
    started = time.time()
    set_seed(seed)
    train = combine_labeled(parts, calibration=False)
    if smoke_limit > 0:
        train["source"] = train["source"].sort_values(["seq", "src_chunk"]).head(smoke_limit).copy()
        train["cross"] = train["cross"].sort_values(["seq", "src_chunk", "dst_chunk"]).head(smoke_limit * 8).copy()
    source_frame, cross_frame = train["source"], train["cross"]
    source_x = frozen_matrix(source_frame)
    cross_x = frozen_matrix(cross_frame)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = RepairMLP(len(FEATURES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    sx = torch.from_numpy(source_x).to(device)
    cx = torch.from_numpy(cross_x).to(device)
    source_target = torch.from_numpy(source_frame.source_keep.to_numpy(np.float32)).to(device)
    valid_target = torch.from_numpy(cross_frame.cross_valid.to_numpy(np.float32)).to(device)
    risk_target = 1.0 - valid_target
    imitation_target = torch.from_numpy(cross_frame.imitation.to_numpy(np.float32)).to(device)
    out_groups = torch.from_numpy(factorized(list(zip(cross_frame.seq.astype(str), cross_frame.src_chunk.astype(int))))).to(device)
    in_groups = torch.from_numpy(factorized(list(zip(cross_frame.seq.astype(str), cross_frame.dst_chunk.astype(int))))).to(device)
    source_seq = source_frame.seq.astype(str).to_numpy()
    cross_seq = cross_frame.seq.astype(str).to_numpy()
    history = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        source_output = model(sx)
        cross_output = model(cx)
        source_loss = class_balanced_focal(source_output["source_keep"], source_target)
        continuity_loss = class_balanced_focal(cross_output["cross_continuity"], valid_target)
        risk_loss = class_balanced_focal(cross_output["catastrophic_risk"], risk_target)
        imitation_loss = class_balanced_focal(cross_output["imitation"], imitation_target)
        outgoing_loss = listwise_loss(cross_output["outgoing"], out_groups, imitation_target)
        incoming_loss = listwise_loss(cross_output["incoming"], in_groups, imitation_target)
        per_sequence = []
        for seq in sorted(set(source_seq) | set(cross_seq)):
            smask = torch.from_numpy(source_seq == seq).to(device)
            cmask = torch.from_numpy(cross_seq == seq).to(device)
            seq_loss = class_balanced_focal(source_output["source_keep"][smask], source_target[smask])
            seq_loss = seq_loss + class_balanced_focal(cross_output["cross_continuity"][cmask], valid_target[cmask])
            seq_loss = seq_loss + class_balanced_focal(cross_output["catastrophic_risk"][cmask], risk_target[cmask])
            seq_loss = seq_loss + class_balanced_focal(cross_output["imitation"][cmask], imitation_target[cmask])
            per_sequence.append(seq_loss)
        group_dro = torch.stack(per_sequence).max() if per_sequence else source_loss * 0.0
        sparsity = 0.5 * (1.0 - torch.sigmoid(source_output["source_keep"])).mean() + 0.5 * torch.sigmoid(cross_output["imitation"]).mean()
        total = (
            LOSS_WEIGHTS["source"] * source_loss
            + LOSS_WEIGHTS["continuity"] * continuity_loss
            + LOSS_WEIGHTS["risk"] * risk_loss
            + LOSS_WEIGHTS["imitation"] * imitation_loss
            + LOSS_WEIGHTS["out_listwise"] * outgoing_loss
            + LOSS_WEIGHTS["in_listwise"] * incoming_loss
            + LOSS_WEIGHTS["group_dro"] * group_dro
            + LOSS_WEIGHTS["sparsity"] * sparsity
        )
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        history.append({
            "epoch": epoch + 1, "loss": float(total.detach().cpu()),
            "source": float(source_loss.detach().cpu()),
            "continuity": float(continuity_loss.detach().cpu()),
            "risk": float(risk_loss.detach().cpu()),
            "imitation": float(imitation_loss.detach().cpu()),
            "out_listwise": float(outgoing_loss.detach().cpu()),
            "in_listwise": float(incoming_loss.detach().cpu()),
            "group_dro": float(group_dro.detach().cpu()),
            "sparsity": float(sparsity.detach().cpu()),
        })
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "model.pt"
    torch.save({
        "state_dict": model.state_dict(), "features": list(FEATURES), "hidden": HIDDEN,
        "seed": seed, "epochs": epochs,
    }, model_path)
    history_path = output / "training_history.json"
    json_write(history_path, history)
    report = {
        "model_path": str(model_path), "model_sha256": sha256_file(model_path),
        "history_path": str(history_path), "history_sha256": sha256_file(history_path),
        "source_training_rows": len(source_frame), "cross_training_rows": len(cross_frame),
        "source_positive_rate": float(source_frame.source_keep.mean()) if len(source_frame) else None,
        "cross_valid_rate": float(cross_frame.cross_valid.mean()) if len(cross_frame) else None,
        "imitation_rate": float(cross_frame.imitation.mean()) if len(cross_frame) else None,
        "epochs": epochs, "seed": seed, "device": str(device),
        "runtime_seconds": time.time() - started,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "peak_gpu_mb": torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else 0.0,
    }
    json_write(output / "training_report.json", report)
    return model, report


def predict_matrix(model: RepairMLP, matrix: np.ndarray, batch_size: int = 131072) -> dict[str, np.ndarray]:
    device = next(model.parameters()).device
    storage: dict[str, list[np.ndarray]] = defaultdict(list)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(matrix), batch_size):
            tensor = torch.from_numpy(matrix[start:start + batch_size]).to(device)
            output = model(tensor)
            for name, logits in output.items():
                storage[name].append(logits.detach().cpu().numpy().astype(np.float32))
    return {name: np.concatenate(values) if values else np.asarray([], np.float32) for name, values in storage.items()}


def cp_lower(successes: np.ndarray, totals: np.ndarray) -> np.ndarray:
    output = np.zeros(len(totals), np.float64)
    mask = successes > 0
    output[mask] = beta.ppf(CONFIDENCE_ALPHA / 2.0, successes[mask], totals[mask] - successes[mask] + 1)
    return np.nan_to_num(output, nan=0.0)


def cp_upper(failures: np.ndarray, totals: np.ndarray) -> np.ndarray:
    output = np.ones(len(totals), np.float64)
    mask = failures < totals
    output[mask] = beta.ppf(1.0 - CONFIDENCE_ALPHA / 2.0, failures[mask] + 1, totals[mask] - failures[mask])
    return np.nan_to_num(output, nan=1.0)


def calibration_curve_exact(scores: np.ndarray, labels: np.ndarray, direction: str) -> dict[str, np.ndarray]:
    scores = np.asarray(scores, np.float64)
    labels = np.asarray(labels, np.int64)
    order = np.argsort(scores if direction == "ascending" else -scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    totals = np.arange(1, len(scores) + 1, dtype=np.int64)
    successes = np.cumsum(sorted_labels)
    if direction == "ascending":
        bounds = cp_upper(successes, totals)
    else:
        bounds = cp_lower(successes, totals)
    return {"scores": sorted_scores.astype(np.float32), "bounds": bounds.astype(np.float32), "direction": np.asarray([1 if direction == "ascending" else -1], np.int8)}


def apply_curve(curve: dict[str, np.ndarray], scores: np.ndarray) -> np.ndarray:
    calibration_scores = curve["scores"]
    ascending = int(curve["direction"][0]) == 1
    if ascending:
        counts = np.searchsorted(calibration_scores, scores, side="right")
    else:
        counts = np.searchsorted(-calibration_scores, -scores, side="right")
    output = np.ones(len(scores), np.float32) if ascending else np.zeros(len(scores), np.float32)
    valid = counts > 0
    output[valid] = curve["bounds"][counts[valid] - 1]
    return output


def logit_probability(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(probability, 1e-5, 1.0 - 1e-5)
    return np.log(probability / (1.0 - probability))


def fit_calibration(model: RepairMLP, parts: list[dict], output: Path, smoke_limit: int = 0) -> tuple[dict, dict]:
    calibration = combine_labeled(parts, calibration=True)
    source, cross = calibration["source"], calibration["cross"]
    if smoke_limit > 0:
        source = source.sort_values(["seq", "src_chunk"]).head(smoke_limit).copy()
        cross = cross.sort_values(["seq", "src_chunk", "dst_chunk"]).head(smoke_limit * 8).copy()
    source_predictions = predict_matrix(model, frozen_matrix(source))
    cross_predictions = predict_matrix(model, frozen_matrix(cross))
    source_invalid = 1.0 - torch.sigmoid(torch.from_numpy(source_predictions["source_keep"])).numpy()
    cross_valid = torch.sigmoid(torch.from_numpy(cross_predictions["cross_continuity"])).numpy()
    risk = torch.sigmoid(torch.from_numpy(cross_predictions["catastrophic_risk"])).numpy()
    curves = {
        "source_invalid_lcb": calibration_curve_exact(source_invalid, 1 - source.source_keep.to_numpy(int), "descending"),
        "cross_valid_lcb": calibration_curve_exact(cross_valid, cross.cross_valid.to_numpy(int), "descending"),
        "risk_ucb": calibration_curve_exact(risk, 1 - cross.cross_valid.to_numpy(int), "ascending"),
    }
    output.mkdir(parents=True, exist_ok=True)
    calibration_path = output / "calibration.npz"
    np.savez_compressed(
        calibration_path,
        source_invalid_scores=curves["source_invalid_lcb"]["scores"],
        source_invalid_bounds=curves["source_invalid_lcb"]["bounds"],
        source_invalid_direction=curves["source_invalid_lcb"]["direction"],
        cross_valid_scores=curves["cross_valid_lcb"]["scores"],
        cross_valid_bounds=curves["cross_valid_lcb"]["bounds"],
        cross_valid_direction=curves["cross_valid_lcb"]["direction"],
        risk_scores=curves["risk_ucb"]["scores"],
        risk_bounds=curves["risk_ucb"]["bounds"],
        risk_direction=curves["risk_ucb"]["direction"],
    )
    report = {
        "source_calibration_rows": len(source), "cross_calibration_rows": len(cross),
        "calibration_path": str(calibration_path), "calibration_sha256": sha256_file(calibration_path),
        "confidence": 0.95, "method": "monotone exact Clopper-Pearson score strata",
    }
    json_write(output / "calibration_report.json", report)
    return curves, {**report, "source_frame": source, "cross_frame": cross, "source_predictions": source_predictions, "cross_predictions": cross_predictions}


def score_sequence(model: RepairMLP, curves: dict, seq: str, root: Path, smoke_cross_limit: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    started = time.time()
    _, cross_path, source_path, cross_features_path, source_features_path = verify_observable(seq, root)
    source = pd.read_parquet(source_path)
    cross = pd.read_parquet(cross_path)
    source_matrix = np.asarray(np.load(source_features_path, mmap_mode="r"), np.float32)
    cross_matrix = np.load(cross_features_path, mmap_mode="r")
    if smoke_cross_limit > 0:
        row_order = cross.sort_values(["src_chunk", "dst_chunk"]).head(smoke_cross_limit).index.to_numpy(np.int64)
        cross = cross.loc[row_order].copy().reset_index(drop=True)
        cross_matrix = np.asarray(cross_matrix[row_order], np.float32)
    else:
        cross_matrix = np.asarray(cross_matrix, np.float32)
    source_output = predict_matrix(model, source_matrix)
    cross_output = predict_matrix(model, cross_matrix)
    source["p_keep"] = torch.sigmoid(torch.from_numpy(source_output["source_keep"])).numpy()
    source["p_invalid"] = 1.0 - source.p_keep
    source["invalid_lcb"] = apply_curve(curves["source_invalid_lcb"], source.p_invalid.to_numpy(float))
    source["keep_energy"] = SOURCE_ANCHOR + logit_probability(source.p_keep.to_numpy(float)) - source.p_invalid.to_numpy(float)
    source["cut_energy"] = logit_probability(source.p_invalid.to_numpy(float)) - CUT_PENALTY
    confident_invalid = source.invalid_lcb.to_numpy(float) >= LCB_MINIMUM
    source["parent_weight"] = np.where(
        confident_invalid,
        source.keep_energy.to_numpy(float) - source.cut_energy.to_numpy(float),
        np.maximum(source.keep_energy.to_numpy(float), 1e-6),
    ).astype(np.float32)
    source_index = source.set_index("src_chunk")
    source_invalid = source_index.p_invalid.reindex(cross.src_chunk).fillna(1.0).to_numpy(float)
    source_invalid_lcb = source_index.invalid_lcb.reindex(cross.src_chunk).fillna(0.0).to_numpy(float)
    source_keep_energy = source_index.keep_energy.reindex(cross.src_chunk).fillna(0.0).to_numpy(float)
    source_cut_energy = source_index.cut_energy.reindex(cross.src_chunk).fillna(0.0).to_numpy(float)
    incoming_keep_energy = source_index.keep_energy.reindex(cross.incoming_src).fillna(0.0).to_numpy(float)
    p_valid = torch.sigmoid(torch.from_numpy(cross_output["cross_continuity"])).numpy()
    p_risk = torch.sigmoid(torch.from_numpy(cross_output["catastrophic_risk"])).numpy()
    p_imitation = torch.sigmoid(torch.from_numpy(cross_output["imitation"])).numpy()
    valid_lcb = apply_curve(curves["cross_valid_lcb"], p_valid)
    risk_ucb = apply_curve(curves["risk_ucb"], p_risk)
    relative_index = FEATURES.index("rel_recall")
    pair_index = FEATURES.index("pair_recall_score")
    relative = cross_matrix[:, relative_index]
    pair_score = cross_matrix[:, pair_index]
    out_logit = np.clip(cross_output["outgoing"], -4.0, 4.0)
    in_logit = np.clip(cross_output["incoming"], -4.0, 4.0)
    imitation_logit = np.clip(cross_output["imitation"], -4.0, 4.0)
    replace_energy = (
        logit_probability(source_invalid)
        + logit_probability(p_valid)
        + 0.5 * out_logit
        + 0.5 * in_logit
        + 0.5 * imitation_logit
        + RELATIVE_COEFFICIENT * relative
        + PAIR_BONUS * cross.pair_exists.to_numpy(float) * pair_score
        - RISK_LAMBDA * p_risk
        - EDIT_PENALTY
        - INCOMING_DISPLACEMENT_PENALTY * np.maximum(incoming_keep_energy, 0.0)
    )
    cross["p_valid"] = p_valid.astype(np.float32)
    cross["p_risk"] = p_risk.astype(np.float32)
    cross["p_imitation"] = p_imitation.astype(np.float32)
    cross["source_invalid_lcb"] = source_invalid_lcb.astype(np.float32)
    cross["valid_lcb"] = valid_lcb.astype(np.float32)
    cross["risk_ucb"] = risk_ucb.astype(np.float32)
    cross["keep_energy"] = source_keep_energy.astype(np.float32)
    cross["cut_energy"] = source_cut_energy.astype(np.float32)
    cross["replace_energy"] = replace_energy.astype(np.float32)
    cross["outgoing_logit"] = out_logit.astype(np.float32)
    cross["incoming_logit"] = in_logit.astype(np.float32)
    cross["imitation_logit"] = imitation_logit.astype(np.float32)
    report = {
        "seq": seq, "source_edges_scored": len(source), "cross_edges_scored": len(cross),
        "runtime_seconds": time.time() - started,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "peak_gpu_mb": torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0,
    }
    return source, cross, report


def aggregate_policy_risk(curves: dict, calibration_report: dict, policy: str) -> dict:
    source = calibration_report["source_frame"].copy()
    cross = calibration_report["cross_frame"].copy()
    source_predictions = calibration_report["source_predictions"]
    cross_predictions = calibration_report["cross_predictions"]
    source["p_keep"] = torch.sigmoid(torch.from_numpy(source_predictions["source_keep"])).numpy()
    source["p_invalid"] = 1.0 - source.p_keep
    source["invalid_lcb"] = apply_curve(curves["source_invalid_lcb"], source.p_invalid.to_numpy(float))
    source["keep_energy"] = SOURCE_ANCHOR + logit_probability(source.p_keep.to_numpy(float)) - source.p_invalid.to_numpy(float)
    source["cut_energy"] = logit_probability(source.p_invalid.to_numpy(float)) - CUT_PENALTY
    source_index = source.set_index(["seq", "src_chunk"])
    keys = pd.MultiIndex.from_arrays([cross.seq.astype(str), cross.src_chunk.astype(int)])
    source_invalid = source_index.p_invalid.reindex(keys).fillna(1.0).to_numpy(float)
    source_lcb = source_index.invalid_lcb.reindex(keys).fillna(0.0).to_numpy(float)
    keep_energy = source_index.keep_energy.reindex(keys).fillna(0.0).to_numpy(float)
    p_valid = torch.sigmoid(torch.from_numpy(cross_predictions["cross_continuity"])).numpy()
    p_risk = torch.sigmoid(torch.from_numpy(cross_predictions["catastrophic_risk"])).numpy()
    valid_lcb = apply_curve(curves["cross_valid_lcb"], p_valid)
    risk_ucb = apply_curve(curves["risk_ucb"], p_risk)
    matrix = frozen_matrix(cross)
    relative = matrix[:, FEATURES.index("rel_recall")]
    pair_score = matrix[:, FEATURES.index("pair_recall_score")]
    out_logit = np.clip(cross_predictions["outgoing"], -4, 4)
    in_logit = np.clip(cross_predictions["incoming"], -4, 4)
    imitation_logit = np.clip(cross_predictions["imitation"], -4, 4)
    incoming_keys = pd.MultiIndex.from_arrays([cross.seq.astype(str), cross.incoming_src.astype(int)])
    incoming_keep_energy = source_index.keep_energy.reindex(incoming_keys).fillna(0.0).to_numpy(float)
    replace = (
        logit_probability(source_invalid) + logit_probability(p_valid)
        + 0.5 * out_logit + 0.5 * in_logit + 0.5 * imitation_logit
        + RELATIVE_COEFFICIENT * relative
        + PAIR_BONUS * cross.pair_exists.to_numpy(float) * pair_score
        - RISK_LAMBDA * p_risk - EDIT_PENALTY
        - INCOMING_DISPLACEMENT_PENALTY * np.maximum(incoming_keep_energy, 0.0)
    )
    limit = POLICIES[policy]
    eligible = (
        (source_lcb >= LCB_MINIMUM) & (valid_lcb >= LCB_MINIMUM)
        & (risk_ucb <= float(limit)) & (replace > keep_energy) & (replace > 0.0)
        & ((cross.has_source_out.to_numpy(int) == 1) | (cross.has_source_in.to_numpy(int) == 1))
    )
    failures = int((1 - cross.loc[eligible, "cross_valid"]).sum())
    count = int(eligible.sum())
    upper = float(cp_upper(np.asarray([failures]), np.asarray([count]))[0]) if count else 1.0
    return {
        "policy": policy, "preliminary_selected": count,
        "catastrophic_failures": failures, "catastrophic_rate": failures / count if count else None,
        "catastrophic_ucb": upper, "limit": limit, "enabled": bool(count > 0 and upper <= float(limit)),
    }


def solve_policy(seq: str, source: pd.DataFrame, cross: pd.DataFrame, policy: str, calibration_policy: dict) -> tuple[pd.DataFrame, dict]:
    nodes = pd.read_parquet(M55_ROOTS[seq] / "frozen_candidate_graph" / "nodes.parquet")
    edges = pd.read_parquet(M55_ROOTS[seq] / "frozen_candidate_graph" / "edges.parquet", columns=["src_chunk", "dst_chunk", "parent_edge"])
    parent = edges[edges.parent_edge.to_numpy(int) == 1][["src_chunk", "dst_chunk", "parent_edge"]].copy()
    if policy == "P0" or not calibration_policy.get("enabled", False):
        selected = parent.copy()
        reason = "P0" if policy == "P0" else "policy_disabled_by_training_calibration_ucb"
    else:
        source_weight = source.set_index("src_chunk").parent_weight
        parent["weight"] = parent.src_chunk.map(source_weight).fillna(1e-6).astype(float)
        limit = float(POLICIES[policy])
        eligible = (
            (cross.source_invalid_lcb.to_numpy(float) >= LCB_MINIMUM)
            & (cross.valid_lcb.to_numpy(float) >= LCB_MINIMUM)
            & (cross.risk_ucb.to_numpy(float) <= limit)
            & (cross.replace_energy.to_numpy(float) > cross.keep_energy.to_numpy(float))
            & (cross.replace_energy.to_numpy(float) > 0.0)
            & ((cross.has_source_out.to_numpy(int) == 1) | (cross.has_source_in.to_numpy(int) == 1))
        )
        candidate_cross = cross.loc[eligible, ["src_chunk", "dst_chunk", "replace_energy"]].copy()
        candidate_cross.rename(columns={"replace_energy": "weight"}, inplace=True)
        real = pd.concat([
            parent[["src_chunk", "dst_chunk", "weight"]], candidate_cross,
        ], ignore_index=True)
        real.sort_values(["src_chunk", "dst_chunk", "weight"], ascending=[True, True, False], kind="mergesort", inplace=True)
        real.drop_duplicates(["src_chunk", "dst_chunk"], keep="first", inplace=True)
        real = real[real.weight.to_numpy(float) > 0.0].copy()
        n = len(nodes)
        if len(real):
            weights = real.weight.to_numpy(float)
            offset = float(weights.max()) + 1.0
            row_index = np.concatenate([real.src_chunk.to_numpy(int), np.arange(n, dtype=int)])
            col_index = np.concatenate([real.dst_chunk.to_numpy(int), n + np.arange(n, dtype=int)])
            costs = np.concatenate([offset - weights, np.full(n, offset, float)])
            matrix = coo_matrix((costs, (row_index, col_index)), shape=(n, 2 * n)).tocsr()
            matched_rows, matched_cols = min_weight_full_bipartite_matching(matrix)
            real_match = matched_cols < n
            selected_keys = set(zip(matched_rows[real_match].tolist(), matched_cols[real_match].tolist()))
            selected = edges[[
                (int(src), int(dst)) in selected_keys
                for src, dst in zip(edges.src_chunk, edges.dst_chunk)
            ]].copy()
        else:
            selected = edges.iloc[:0].copy()
        reason = "structured_flow"
    m53 = load_module(f"m23_56_m53_solve_{SEQ_SHORT[seq]}_{policy}", "scripts/m23_research/m23_53_global_identity_flow_capacity.py")
    m53.graph_chains(selected, len(nodes))
    if selected.src_chunk.duplicated().any() or selected.dst_chunk.duplicated().any():
        raise RuntimeError("selected policy graph is not one-to-one")
    first = nodes.first_frame.to_numpy(int)
    last = nodes.last_frame.to_numpy(int)
    if len(selected) and np.any(first[selected.dst_chunk.to_numpy(int)] <= last[selected.src_chunk.to_numpy(int)]):
        raise RuntimeError("selected policy graph is not time-forward")
    parent_keys = set(zip(parent.src_chunk.astype(int), parent.dst_chunk.astype(int)))
    selected_keys = set(zip(selected.src_chunk.astype(int), selected.dst_chunk.astype(int)))
    changed = parent_keys.symmetric_difference(selected_keys)
    affected = {chunk for pair in changed for chunk in pair}
    report = {
        "policy": policy, "reason": reason, "calibration_policy": calibration_policy,
        "keep_parent": len(parent_keys & selected_keys), "cut_parent": len(parent_keys - selected_keys),
        "cross": len(selected_keys - parent_keys), "dummy_terminate": len(nodes) - len(selected),
        "dummy_restart": len(nodes) - len(selected), "selected_edges": len(selected),
        "affected_chunks": len(affected),
        "affected_rows": int(nodes.iloc[sorted(affected)].rows.sum()) if affected else 0,
        "one_to_one": True, "acyclic": True, "time_forward": True,
    }
    return selected, report


def write_tracker(seq: str, selected: pd.DataFrame, output_path: Path) -> dict:
    m53 = load_module(f"m23_56_writer_{SEQ_SHORT[seq]}_{hash(str(output_path)) & 0xffff}", "scripts/m23_research/m23_53_global_identity_flow_capacity.py")
    nodes = pd.read_parquet(M55_ROOTS[seq] / "frozen_candidate_graph" / "nodes.parquet")
    report = m53.write_tracker(seq, SOURCE_PARENT / f"{seq}.txt", nodes, selected, output_path)
    report["sha256"] = sha256_file(output_path)
    source_payload = detector_payload(SOURCE_PARENT / f"{seq}.txt")
    output_payload = detector_payload(output_path)
    if source_payload != output_payload:
        raise RuntimeError("detection rows/boxes/scores changed")
    report["detection_payload_unchanged"] = True
    report["detection_payload_sha256"] = output_payload
    return report


def exact_eval(seq: str, policy_root: Path, tracker_name: str, cache_root: Path) -> tuple[dict, dict]:
    tracker_path = policy_root / "track_results" / f"{seq}.txt"
    tracker_sha = sha256_file(tracker_path)
    cache_path = cache_root / "exact_eval_cache.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    if tracker_sha in cache:
        return cache[tracker_sha]["metrics"], {"reused_exact_tracker_sha": tracker_sha, "source": cache[tracker_sha]["source"]}
    m53 = load_module(f"m23_56_eval_{SEQ_SHORT[seq]}_{len(cache)}", "scripts/m23_research/m23_53_global_identity_flow_capacity.py")
    started = time.time()
    metrics = m53.run_official_trackeval(seq=seq, output_root=policy_root, tracker_name=tracker_name)
    cache[tracker_sha] = {"seq": seq, "metrics": metrics, "source": str(policy_root), "evaluated_at": utc_now()}
    json_write(cache_path, cache)
    return metrics, {"official_trackeval_runtime_seconds": time.time() - started, "tracker_sha": tracker_sha}


def diagnostic_metrics(source: pd.DataFrame, cross: pd.DataFrame, selected: pd.DataFrame, teacher_selected: set[tuple[int, int]]) -> dict:
    source_error = 1 - source.source_keep.to_numpy(int)
    source_score = source.p_invalid.to_numpy(float)
    source_pr_auc = float(average_precision_score(source_error, source_score)) if len(np.unique(source_error)) > 1 else None
    threshold = 0.5
    predicted_error = source_score >= threshold
    precision = float(source_error[predicted_error].mean()) if predicted_error.any() else None
    recall = float(source_error[predicted_error].sum() / max(1, source_error.sum()))
    actual_count = int(source_error.sum())
    top = np.argsort(-source_score, kind="mergesort")[:actual_count]
    precision_actual = float(source_error[top].mean()) if actual_count else None
    p, r, _ = precision_recall_curve(source_error, source_score)
    recall95 = float(np.max(r[p >= 0.95])) if np.any(p >= 0.95) else 0.0
    recall99 = float(np.max(r[p >= 0.99])) if np.any(p >= 0.99) else 0.0
    selected_keys = set(zip(selected.src_chunk.astype(int), selected.dst_chunk.astype(int)))
    parent_keys = set(zip(source.src_chunk.astype(int), source.parent_dst.astype(int)))
    selected_cross = selected_keys - parent_keys
    cross_index = cross.set_index(["src_chunk", "dst_chunk"])
    valid_selected = []
    imitation_selected = []
    for key in selected_cross:
        if key in cross_index.index:
            row = cross_index.loc[key]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            valid_selected.append(int(row.cross_valid))
            imitation_selected.append(int(key in teacher_selected))
    correct_parent = {key for key, keep in zip(parent_keys, source.source_keep) if int(keep) == 1}
    false_cuts = len(correct_parent - selected_keys)
    false_cut_rate = false_cuts / len(correct_parent) if correct_parent else None
    ranks = []
    for src, group in cross.groupby("src_chunk", sort=False):
        positives = group[group.imitation == 1]
        if len(positives):
            ordered = group.sort_values(["replace_energy", "dst_chunk"], ascending=[False, True], kind="mergesort")
            position = {int(dst): index + 1 for index, dst in enumerate(ordered.dst_chunk)}
            ranks.extend(position[int(dst)] for dst in positives.dst_chunk)
    return {
        "source_error_pr_auc": source_pr_auc,
        "source_error_precision_at_0p5": precision,
        "source_error_recall_at_0p5": recall,
        "precision_at_actual_edit_count": precision_actual,
        "recall_at_95_precision": recall95,
        "recall_at_99_precision": recall99,
        "false_cut_rate": false_cut_rate,
        "catastrophic_false_repair_rate": float(1.0 - np.mean(valid_selected)) if valid_selected else 0.0,
        "repair_transaction_precision": float(np.mean(imitation_selected)) if imitation_selected else 0.0,
        "cross_alternative_mrr": float(np.mean(1.0 / np.asarray(ranks))) if ranks else 0.0,
        "selected_cross_transactions": len(selected_cross),
    }


def inner_fold(
    outer: str,
    validation: str,
    train_sequences: list[str],
    root: Path,
    fold_root: Path,
    seed: int,
    smoke: bool = False,
) -> dict:
    if validation == outer or outer in train_sequences:
        raise RuntimeError("invalid nested split")
    parts = [labeled_sequence(seq, outer, root) for seq in train_sequences]
    model, training = train_model(parts, seed, fold_root / "model", epochs=SMOKE_EPOCHS if smoke else EPOCHS, smoke_limit=1000 if smoke else 0)
    curves, calibration = fit_calibration(model, parts, fold_root / "calibration", smoke_limit=1000 if smoke else 0)
    calibration_policies = {
        policy: aggregate_policy_risk(curves, calibration, policy) for policy in ("P1", "P2")
    }
    source, cross, scoring = score_sequence(model, curves, validation, root, smoke_cross_limit=50000 if smoke else 0)
    trackers = {}
    for policy in POLICIES:
        policy_calibration = {"enabled": True, "policy": "P0"} if policy == "P0" else calibration_policies[policy]
        selected, flow = solve_policy(validation, source, cross, policy, policy_calibration)
        policy_root = fold_root / "policies" / policy
        tracker_path = policy_root / "track_results" / f"{validation}.txt"
        tracker = write_tracker(validation, selected, tracker_path)
        if policy == "P0" and tracker_path.read_bytes() != baseline_tracker(validation).read_bytes():
            raise RuntimeError("P0 is not byte-exact M23-46")
        manifest = {
            "outer": outer, "validation": validation, "train_sequences": train_sequences,
            "policy": policy, "outer_gt_read": False, "validation_gt_read": False,
            "tracker": tracker, "flow": flow, "training_model_sha256": training["model_sha256"],
            "calibration_sha256": calibration["calibration_sha256"],
        }
        json_write(policy_root / "tracker_frozen_before_validation_gt.json", manifest)
        trackers[policy] = {"selected": selected, "root": policy_root, "manifest": manifest}
    append_event(root, {"event": "inner_trackers_frozen_before_validation_gt", "outer": outer, "validation": validation, "policies": {policy: value["manifest"]["tracker"]["sha256"] for policy, value in trackers.items()}, "smoke": smoke})
    if smoke:
        report = {
            "status": "inner_only_smoke_completed", "outer": outer, "validation": validation,
            "train_sequences": train_sequences, "outer_teacher_read": False,
            "training": training, "calibration_policies": calibration_policies,
            "scoring": scoring, "trackers": {policy: data["manifest"] for policy, data in trackers.items()},
        }
        json_write(fold_root / "report.json", report)
        return report

    validation_labels = labeled_sequence(validation, outer, root)
    source_label = validation_labels["source"]
    cross_label = validation_labels["cross"]
    source = source.merge(source_label[["src_chunk", "parent_dst", "source_keep"]], on=["src_chunk", "parent_dst"], how="left", validate="one_to_one")
    cross = cross.merge(cross_label[["src_chunk", "dst_chunk", "cross_valid", "imitation"]], on=["src_chunk", "dst_chunk"], how="left", validate="one_to_one")
    cross["cross_valid"] = cross.cross_valid.fillna(0).astype(np.int8)
    cross["imitation"] = cross.imitation.fillna(0).astype(np.int8)
    utility, teacher_selected = teacher_labels(validation, outer)
    del utility
    exact = {}
    diagnostics = {}
    cache_root = root / "exact_eval_cache"
    for policy, data in trackers.items():
        metrics, provenance = exact_eval(validation, data["root"], f"m23_56_inner_{SEQ_SHORT[outer]}_{SEQ_SHORT[validation]}_{policy}", cache_root)
        exact[policy] = {**metrics, "provenance": provenance}
        diagnostics[policy] = diagnostic_metrics(source, cross, data["selected"], teacher_selected)
    baseline_hota = exact["P0"]["HOTA"]
    for policy in exact:
        exact[policy]["delta_HOTA"] = exact[policy]["HOTA"] - baseline_hota
    report = {
        "status": "completed", "outer": outer, "validation": validation,
        "train_sequences": train_sequences, "outer_teacher_read": False,
        "validation_gt_opened_after_tracker_freeze": True,
        "training": training, "calibration_policies": calibration_policies,
        "scoring": scoring, "exact_trackeval": exact, "diagnostics": diagnostics,
        "trackers": {policy: data["manifest"] for policy, data in trackers.items()},
    }
    json_write(fold_root / "report.json", report)
    print(json.dumps({"outer": outer, "validation": validation, "exact": exact, "calibration": calibration_policies}, indent=2, sort_keys=True), flush=True)
    return report


def policy_key(policy: str, rows: list[dict]) -> tuple[float, float, int, int]:
    deltas = [row["exact_trackeval"][policy]["delta_HOTA"] for row in rows]
    edits = sum(row["trackers"][policy]["flow"]["cross"] + row["trackers"][policy]["flow"]["cut_parent"] for row in rows)
    return (min(deltas), float(np.mean(deltas)), -edits, 1 if policy == "P1" else 0)


def run_outer_inner(outer: str, root: Path) -> dict:
    outer_root = root / "outers" / outer
    manifest_path = outer_root / "outer_policy_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite frozen outer policy: {manifest_path}")
    training_sequences = [seq for seq in SEQUENCES if seq != outer]
    rows = []
    for inner_index, validation in enumerate(training_sequences):
        train_sequences = [seq for seq in training_sequences if seq != validation]
        rows.append(inner_fold(
            outer, validation, train_sequences, root,
            outer_root / "inner" / validation,
            BASE_SEED + SEQ_INDEX[outer] * 100 + inner_index,
            smoke=False,
        ))
    policy_summary = {}
    passing = []
    for policy in ("P1", "P2"):
        deltas = [row["exact_trackeval"][policy]["delta_HOTA"] for row in rows]
        summary = {
            "policy": policy, "deltas": dict(zip(training_sequences, deltas)),
            "worst_fold_delta_HOTA": min(deltas), "mean_delta_HOTA": float(np.mean(deltas)),
            "all_folds_at_least_0p05": bool(all(delta >= INNER_MIN_FOLD_DELTA for delta in deltas)),
            "mean_at_least_0p20": bool(np.mean(deltas) >= INNER_MIN_MEAN_DELTA),
            "modified_transactions": int(sum(row["trackers"][policy]["flow"]["cross"] + row["trackers"][policy]["flow"]["cut_parent"] for row in rows)),
        }
        summary["gate_pass"] = summary["all_folds_at_least_0p05"] and summary["mean_at_least_0p20"]
        policy_summary[policy] = summary
        if summary["gate_pass"]:
            passing.append(policy)
    chosen = max(passing, key=lambda policy: policy_key(policy, rows)) if passing else "P0"
    final_training = None
    final_calibration = None
    if chosen == "P0":
        edges = pd.read_parquet(M55_ROOTS[outer] / "frozen_candidate_graph" / "edges.parquet", columns=["src_chunk", "dst_chunk", "parent_edge"])
        selected = edges[edges.parent_edge.to_numpy(int) == 1].copy()
        flow = {"policy": "P0", "reason": "inner_gate_failed", "keep_parent": len(selected), "cut_parent": 0, "cross": 0}
        calibration_policy = {"policy": "P0", "enabled": True}
    else:
        parts = [labeled_sequence(seq, outer, root) for seq in training_sequences]
        model, final_training = train_model(parts, BASE_SEED + SEQ_INDEX[outer] * 100 + 99, outer_root / "final_model")
        curves, calibration = fit_calibration(model, parts, outer_root / "final_calibration")
        final_calibration = aggregate_policy_risk(curves, calibration, chosen)
        source, cross, final_scoring = score_sequence(model, curves, outer, root)
        selected, flow = solve_policy(outer, source, cross, chosen, final_calibration)
        final_training["held_scoring"] = final_scoring
        calibration_policy = final_calibration
    tracker_path = outer_root / "outer_frozen" / "track_results" / f"{outer}.txt"
    tracker = write_tracker(outer, selected, tracker_path)
    if chosen == "P0" and tracker_path.read_bytes() != baseline_tracker(outer).read_bytes():
        raise RuntimeError("frozen P0 outer tracker is not byte-exact M23-46")
    manifest = {
        "experiment": "M23-56 strict source-repair outer policy",
        "outer": outer, "training_sequences": training_sequences,
        "inner_gate": {"minimum_each": INNER_MIN_FOLD_DELTA, "minimum_mean": INNER_MIN_MEAN_DELTA},
        "policy_summary": policy_summary, "selected_policy": chosen,
        "selection_rule": "max worst delta, max mean delta, fewer transactions, P1",
        "outer_gt_read": False, "outer_teacher_action_read": False,
        "all_inner_validation_trackers_frozen_before_validation_gt": True,
        "training": final_training, "calibration_policy": calibration_policy,
        "flow": flow, "tracker": tracker,
        "candidate_graph_sha256": json.loads((M55_ROOTS[outer] / "frozen_candidate_graph" / "freeze_manifest.json").read_text())["frozen_artifacts"]["edges_sha256"],
        "preregistration_sha256": PREREG_SHA256,
        "status": "outer_policy_and_tracker_frozen_before_outer_gt",
    }
    json_write(manifest_path, manifest)
    append_event(root, {"event": "outer_policy_tracker_frozen_before_outer_gt", "outer": outer, "selected_policy": chosen, "tracker_sha256": tracker["sha256"], "manifest": str(manifest_path), "manifest_sha256": sha256_file(manifest_path)})
    print(json.dumps({"outer": outer, "selected_policy": chosen, "policy_summary": policy_summary, "tracker_sha256": tracker["sha256"]}, indent=2, sort_keys=True), flush=True)
    return manifest


def verify_all_outer_manifests(root: Path) -> dict[str, dict]:
    manifests = {}
    for seq in SEQUENCES:
        path = root / "outers" / seq / "outer_policy_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("outer_gt_read") is not False or manifest.get("status") != "outer_policy_and_tracker_frozen_before_outer_gt":
            raise RuntimeError(f"outer manifest not frozen: {seq}")
        tracker = root / "outers" / seq / "outer_frozen" / "track_results" / f"{seq}.txt"
        if sha256_file(tracker) != manifest["tracker"]["sha256"]:
            raise RuntimeError(f"outer tracker hash mismatch: {seq}")
        manifests[seq] = manifest
    return manifests


def outer_evaluate(root: Path) -> dict:
    manifests = verify_all_outer_manifests(root)
    all_noop = all(manifest["selected_policy"] == "P0" for manifest in manifests.values())
    result_root = root / "strict_outer_evaluation"
    if (result_root / "report.json").exists():
        raise FileExistsError("refusing to repeat outer evaluation")
    result_root.mkdir(parents=True, exist_ok=True)
    if all_noop:
        folds = {seq: m46_metrics(seq) for seq in SEQUENCES}
        combined = json.loads(M46_COMBINED.read_text(encoding="utf-8"))["metrics"]
        report = {
            "experiment": "M23-56 strict outer evaluation",
            "protocol_valid": True, "all_outer_policies": "P0/no-op",
            "outer_trackeval_runs": 0, "combined_trackeval_runs": 0,
            "exact_metric_provenance": "all frozen trackers byte-exact M23-46; reuse previously official exact metrics",
            "folds": folds, "COMBINED": combined, "deployable": True,
            "strict_best_remains": "M23-46 79.123193",
            "decision": "close M23-56; observable source-integrity transfer insufficient",
        }
        json_write(result_root / "report.json", report)
        append_event(root, {"event": "outer_evaluation_closed_all_noop", "outer_trackeval_runs": 0, "combined_trackeval_runs": 0})
        return report
    folds = {}
    combined_track_root = result_root / "combined" / "track_results"
    combined_track_root.mkdir(parents=True, exist_ok=True)
    for seq in SEQUENCES:
        source = root / "outers" / seq / "outer_frozen" / "track_results" / f"{seq}.txt"
        destination = combined_track_root / f"{seq}.txt"
        destination.write_bytes(source.read_bytes())
        fold_root = result_root / "folds" / seq
        (fold_root / "track_results").mkdir(parents=True, exist_ok=True)
        (fold_root / "track_results" / f"{seq}.txt").write_bytes(source.read_bytes())
        metrics, _ = exact_eval(seq, fold_root, f"m23_56_outer_{SEQ_SHORT[seq]}", root / "outer_exact_eval_cache")
        folds[seq] = metrics
    m45 = load_module("m23_56_combined_eval", "scripts/m23_research/m23_45_domain_robust_source_cut_student.py")
    combined_detail = m45.evaluate_detailed(
        combined_track_root, result_root / "combined_eval", "m23_56_strict_combined", SEQUENCES,
    )
    combined = combined_detail["COMBINED"]
    report = {
        "experiment": "M23-56 strict outer evaluation", "protocol_valid": True,
        "all_outer_policies": {seq: manifest["selected_policy"] for seq, manifest in manifests.items()},
        "outer_trackeval_runs": 4, "combined_trackeval_runs": 1,
        "folds": folds, "COMBINED": combined, "deployable": True,
        "delta_vs_m23_46": combined["HOTA"] - 79.123193,
        "decision": "new strict deployable best" if combined["HOTA"] > 80.0 else "report without post-hoc tuning",
        "mot20_test_submitted": False,
    }
    json_write(result_root / "report.json", report)
    append_event(root, {"event": "outer_evaluation_completed", "outer_trackeval_runs": 4, "combined_trackeval_runs": 1, "HOTA": combined["HOTA"]})
    return report


def aggregate(root: Path) -> dict:
    prereg = verify_preregistration(root)
    manifests = verify_all_outer_manifests(root)
    evaluation_path = root / "strict_outer_evaluation" / "report.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8")) if evaluation_path.exists() else None
    generated = {
        "experiment": "M23-56 Strict Source-Repair Structured Transfer Feasibility",
        "protocol_valid": True, "deployable": True,
        "preregistration_sha256": PREREG_SHA256,
        "outer_policies": {seq: manifest["selected_policy"] for seq, manifest in manifests.items()},
        "inner_gates": {seq: manifest["policy_summary"] for seq, manifest in manifests.items()},
        "outer_tracker_sha256": {seq: manifest["tracker"]["sha256"] for seq, manifest in manifests.items()},
        "evaluation": evaluation,
        "prohibited_actions_respected": {
            "m23_54_started": False, "large_model": False, "candidate_scan": False,
            "mot20_test_submission": False,
        },
    }
    generated_path = Path("docs/generated/M23_56_STRICT_SOURCE_REPAIR_TRANSFER_20260720.json")
    json_write(generated_path, generated)
    lines = [
        "# M23-56 Strict Source-Repair Structured Transfer Feasibility", "", "Date: 2026-07-20", "",
        "## Protocol", "", f"- Preregistration SHA-256: `{PREREG_SHA256}`.",
        "- Fixed two-layer shared MLP, hidden dimension 64; P0/P1/P2 only.",
        "- Every outer used strict three-fold inner sequence-LOSO; held teacher labels were blocked before tracker freeze.",
        "- M23-55 candidate graphs, K and gap quotas were immutable.", "", "## Outer inner gates", "",
        "| Outer | P1 worst | P1 mean | P1 pass | P2 worst | P2 mean | P2 pass | Frozen policy |",
        "|---|---:|---:|:---:|---:|---:|:---:|:---:|",
    ]
    for seq in SEQUENCES:
        summary = manifests[seq]["policy_summary"]
        lines.append(
            f"| {seq} | {summary['P1']['worst_fold_delta_HOTA']:.6f} | {summary['P1']['mean_delta_HOTA']:.6f} | {summary['P1']['gate_pass']} | "
            f"{summary['P2']['worst_fold_delta_HOTA']:.6f} | {summary['P2']['mean_delta_HOTA']:.6f} | {summary['P2']['gate_pass']} | {manifests[seq]['selected_policy']} |"
        )
    lines.extend(["", "## Strict result", ""])
    if evaluation:
        combined = evaluation["COMBINED"]
        lines.extend([
            f"- COMBINED HOTA: **{combined['HOTA']:.6f}**",
            f"- DetA: {combined['DetA']:.6f}; AssA: {combined['AssA']:.6f}; IDSW: {combined['IDSW']}",
            f"- Decision: {evaluation['decision']}",
        ])
    else:
        lines.append("Outer evaluation has not been unlocked.")
    lines.extend(["", "## Restrictions", "", "- M23-54 was not started.", "- No candidate/threshold/risk-level scan was performed.", "- No MOT20 test submission was made."])
    doc_path = Path("docs/m23_56_strict_source_repair_transfer_20260720.md")
    doc_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"generated": str(generated_path), "document": str(doc_path), "outer_policies": generated["outer_policies"], "evaluation": evaluation}, indent=2, sort_keys=True), flush=True)
    return generated


def implementation_manifest(root: Path) -> dict:
    prereg = verify_preregistration(root)
    script_path = Path(__file__)
    payload = {
        "experiment": "M23-56 implementation freeze",
        "preregistration_sha256": PREREG_SHA256,
        "script": str(script_path.relative_to(REPO)),
        "script_sha256": sha256_file(script_path),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "features": list(FEATURES), "hidden": HIDDEN, "epochs": EPOCHS,
        "policies": POLICIES, "loss_weights": LOSS_WEIGHTS,
        "architecture": "shared Linear-64-GELU-Linear-64-GELU with six scalar heads",
        "outer_gt_read": False,
        "environment": {
            "cpu_count": os.cpu_count(), "python": sys.version,
            "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    path = root / "implementation_manifest.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError("implementation changed after freeze")
    else:
        json_write(path, payload)
        append_event(root, {"event": "implementation_frozen", "manifest": str(path), "manifest_sha256": sha256_file(path), "outer_gt_read": False})
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(RUN_ROOT))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze-implementation")
    p = sub.add_parser("freeze-observable")
    p.add_argument("--seq", action="append", choices=SEQUENCES)
    p = sub.add_parser("smoke")
    p.add_argument("--outer", default="MOT20-02", choices=SEQUENCES)
    p = sub.add_parser("run-outer")
    p.add_argument("--outer", required=True, choices=SEQUENCES)
    sub.add_parser("outer-eval")
    sub.add_parser("aggregate")
    args = parser.parse_args()
    root = Path(args.root)
    verify_preregistration(root)
    if args.command != "freeze-implementation":
        implementation_manifest(root)
    if args.command == "freeze-implementation":
        print(json.dumps(implementation_manifest(root), indent=2, sort_keys=True))
    elif args.command == "freeze-observable":
        for seq in args.seq or SEQUENCES:
            freeze_observable_sequence(seq, root)
    elif args.command == "smoke":
        if args.outer != "MOT20-02":
            raise ValueError("preregistered first smoke is outer=MOT20-02")
        fold_root = root / "smoke_outer_m02" / "train_m01_m03_validate_m05"
        if (fold_root / "report.json").exists():
            raise FileExistsError("refusing to repeat smoke")
        report = inner_fold(
            "MOT20-02", "MOT20-05", ["MOT20-01", "MOT20-03"], root,
            fold_root, BASE_SEED + 201, smoke=True,
        )
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    elif args.command == "run-outer":
        run_outer_inner(args.outer, root)
    elif args.command == "outer-eval":
        print(json.dumps(outer_evaluate(root), indent=2, sort_keys=True), flush=True)
    else:
        aggregate(root)


if __name__ == "__main__":
    main()
