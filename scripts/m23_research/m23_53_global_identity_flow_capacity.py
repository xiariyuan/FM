#!/usr/bin/env python3
from __future__ import annotations

"""M23-53 teacher-only global identity-flow/path-cover capacity audit.

Protocol invariants
-------------------
1. The strict M23-46 tracker and applied graph are immutable parents.
2. Nodes and candidate edges are built from an explicit GT-free allowlist.
3. The frozen node/edge files and SHA-256 manifest are written before any GT
   loader is invoked.
4. GT is used only after freeze to construct structural edge utilities for a
   global one-to-one, time-forward path cover.
5. The generated tracker is teacher-only (deployable=false) and is verified by
   official TrackEval.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching


REPO = Path(__file__).resolve().parents[2]
SEQUENCES = ("MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05")
SEQUENCE_INDEX = {seq: index for index, seq in enumerate(SEQUENCES)}
DEFAULT_GRAPH_ROOT = Path("outputs/mot20_m23_20260718/micrograph_chunk30_v1")
DEFAULT_BASELINE_CACHE = Path(
    "outputs/mot20_m23_20260718/m23_47_m23_46_deployable_baseline_cache_v1"
)
DEFAULT_SOURCE_PARENT = Path(
    "outputs/assocriskbench_p15_20260714/"
    "fused_a42_adaptive_risk_eval/l1_r2_q75_adaptive_priority/track_results"
)
SYNTHETIC_ID_BASE = 10_000_000
SYNTHETIC_ID_STRIDE = 1_000_000
CHUNK_SPAN = 30
GAP_BREAK = 1

NODE_ALLOWLIST = (
    "chunk_id",
    "source_track_id",
    "source_ordinal",
    "first_frame",
    "last_frame",
    "span_frames",
    "rows",
    "first_line",
    "last_line",
    "appearance_consistency",
    "first_cx",
    "first_cy",
    "last_cx",
    "last_cy",
    "first_h",
    "last_h",
    "start_vx",
    "start_vy",
    "end_vx",
    "end_vy",
)

EDGE_ALLOWLIST = (
    "src_chunk",
    "dst_chunk",
    "src_track",
    "dst_track",
    "gap",
    "log_gap",
    "appearance_cos",
    "same_source",
    "source_adjacent",
    "forward_motion_error",
    "backward_motion_error",
    "motion_error_min",
    "motion_error_mean",
    "endpoint_displacement",
    "velocity_cos",
    "log_height_ratio",
    "src_rows",
    "dst_rows",
    "src_consistency",
    "dst_consistency",
    "consistency_min",
    "out_rank",
    "in_rank",
    "max_rank",
    "out_margin",
    "in_margin",
    "max_margin",
    "edge_role",
)

FORBIDDEN_FROZEN_TOKENS = (
    "same_gt",
    "modal_gt",
    "purity",
    "label_confidence",
    "actual_assa",
    "assa_edge",
    "delta_proxy",
    "matched_gt",
    "match_iou",
    "mapping_rate",
    "mapped_rows",
    "teacher",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def append_event(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time": utc_now(), **event}, sort_keys=True) + "\n")


def parquet_columns(path: Path) -> list[str]:
    return list(pq.read_schema(path).names)


def read_allowlisted_parquet(path: Path, allowlist: Iterable[str]) -> pd.DataFrame:
    available = set(parquet_columns(path))
    columns = [column for column in allowlist if column in available]
    return pd.read_parquet(path, columns=columns)


def audit_frozen_columns(columns: Iterable[str]) -> list[str]:
    forbidden = []
    for column in columns:
        lower = column.lower()
        if any(token in lower for token in FORBIDDEN_FROZEN_TOKENS):
            forbidden.append(column)
    return forbidden


def load_module(name: str, relative_path: str):
    path = REPO / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_tracker_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            fields = line.rstrip("\n").split(",")
            if len(fields) < 6:
                continue
            rows.append(
                {
                    "line": line_index,
                    "frame": int(float(fields[0])),
                    "track_id": int(float(fields[1])),
                    "fields": fields,
                }
            )
    return rows


def line_chunks(rows: list[dict], nodes: pd.DataFrame) -> dict[int, int]:
    by_track: dict[int, list[int]] = defaultdict(list)
    for row_index, row in enumerate(rows):
        by_track[int(row["track_id"])].append(row_index)
    mapping: dict[int, int] = {}
    chunk_id = 0
    for track_id, indices in sorted(by_track.items()):
        indices.sort(key=lambda index: (rows[index]["frame"], index))
        start = 0
        ordinal = 0
        for end in range(1, len(indices) + 1):
            boundary = (
                end == len(indices)
                or rows[indices[end]]["frame"] - rows[indices[end - 1]]["frame"]
                > GAP_BREAK
                or rows[indices[end]]["frame"] - rows[indices[start]]["frame"]
                >= CHUNK_SPAN
            )
            if not boundary:
                continue
            record = nodes.iloc[chunk_id]
            if (
                int(record.source_track_id) != track_id
                or int(record.source_ordinal) != ordinal
                or int(record.first_frame) != rows[indices[start]]["frame"]
                or int(record.last_frame) != rows[indices[end - 1]]["frame"]
            ):
                raise RuntimeError(f"chunk reconstruction mismatch at {chunk_id}")
            for index in indices[start:end]:
                mapping[index] = chunk_id
            chunk_id += 1
            ordinal += 1
            start = end
    if chunk_id != len(nodes) or len(mapping) != len(rows):
        raise RuntimeError(
            f"chunk reconstruction incomplete: chunks={chunk_id}/{len(nodes)}, "
            f"rows={len(mapping)}/{len(rows)}"
        )
    return mapping


def graph_chains(selected: pd.DataFrame, chunk_count: int) -> dict[int, int]:
    successor = {
        int(row.src_chunk): int(row.dst_chunk) for row in selected.itertuples()
    }
    predecessor = {dst: src for src, dst in successor.items()}
    if len(successor) != len(selected) or len(predecessor) != len(selected):
        raise RuntimeError("selected graph is not one-to-one")
    roots = [chunk_id for chunk_id in range(chunk_count) if chunk_id not in predecessor]
    assignment: dict[int, int] = {}
    for root in roots:
        current = root
        seen: set[int] = set()
        while current not in seen:
            seen.add(current)
            assignment[current] = root
            if current not in successor:
                break
            current = successor[current]
        else:
            raise RuntimeError("cycle in selected identity flow")
    if len(assignment) != chunk_count:
        raise RuntimeError(f"unassigned chunks: {chunk_count - len(assignment)}")
    return assignment


def write_tracker(
    seq: str,
    source_parent: Path,
    nodes: pd.DataFrame,
    selected: pd.DataFrame,
    output_path: Path,
) -> dict:
    rows = read_tracker_rows(source_parent)
    mapping = line_chunks(rows, nodes)
    assignment = graph_chains(selected, len(nodes))
    base = SYNTHETIC_ID_BASE + SEQUENCE_INDEX[seq] * SYNTHETIC_ID_STRIDE
    output: list[list[str]] = []
    seen: set[tuple[int, int]] = set()
    for row_index, row in enumerate(rows):
        chunk_id = mapping[row_index]
        new_id = base + assignment[chunk_id]
        if new_id >= (1 << 24):
            raise RuntimeError("synthetic ID exceeds exact float32 integer range")
        fields = list(row["fields"])
        fields[1] = str(new_id)
        key = (int(row["frame"]), int(new_id))
        if key in seen:
            raise RuntimeError(f"{seq}: duplicate frame/id {key}")
        seen.add(key)
        output.append(fields)
    output.sort(
        key=lambda fields: (
            int(float(fields[0])),
            int(float(fields[1])),
            float(fields[2]),
            float(fields[3]),
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for fields in output:
            handle.write(",".join(fields) + "\n")
    return {
        "rows": len(rows),
        "chunks": len(nodes),
        "chains": len(set(assignment.values())),
        "selected_edges": len(selected),
    }


def rank_edges(edges: pd.DataFrame) -> pd.DataFrame:
    ranked = edges.sort_values(
        ["src_chunk", "dst_chunk", "parent_edge"],
        ascending=[True, True, False],
        kind="mergesort",
    ).copy()
    ranked["appearance_out_rank"] = (
        ranked.groupby("src_chunk")["appearance_cos"]
        .rank(method="first", ascending=False)
        .astype(np.int32)
    )
    ranked["appearance_in_rank"] = (
        ranked.groupby("dst_chunk")["appearance_cos"]
        .rank(method="first", ascending=False)
        .astype(np.int32)
    )
    ranked["motion_out_rank"] = (
        ranked.groupby("src_chunk")["motion_error_min"]
        .rank(method="first", ascending=True)
        .astype(np.int32)
    )
    ranked["motion_in_rank"] = (
        ranked.groupby("dst_chunk")["motion_error_min"]
        .rank(method="first", ascending=True)
        .astype(np.int32)
    )
    return ranked


def build_and_freeze_candidates(
    *,
    seq: str,
    graph_root: Path,
    baseline_cache: Path,
    source_parent_root: Path,
    output_root: Path,
    top_k: int,
    max_gap: int,
    max_motion_error: float,
    max_endpoint_displacement: float,
    max_abs_log_height_ratio: float,
    smoke_active_chunks: int,
    smoke_max_cross: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    seq_graph = graph_root / seq
    nodes_path = seq_graph / "microtracklets.parquet"
    bank_path = seq_graph / "candidate_edges.parquet"
    seq_cache = baseline_cache / seq
    parent_applied_path = seq_cache / "frozen_applied_edges.parquet"
    parent_tracker = seq_cache / "track_results" / f"{seq}.txt"
    source_parent = source_parent_root / f"{seq}.txt"
    for path in (
        nodes_path,
        bank_path,
        parent_applied_path,
        parent_tracker,
        source_parent,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    nodes = read_allowlisted_parquet(nodes_path, NODE_ALLOWLIST)
    missing_nodes = set(NODE_ALLOWLIST) - set(nodes.columns)
    if missing_nodes:
        raise RuntimeError(f"missing node observables: {sorted(missing_nodes)}")
    nodes = nodes.sort_values("chunk_id", kind="mergesort").reset_index(drop=True)
    if not np.array_equal(nodes.chunk_id.to_numpy(int), np.arange(len(nodes))):
        raise RuntimeError("chunk IDs are not dense and ordered")

    parent_applied = read_allowlisted_parquet(parent_applied_path, EDGE_ALLOWLIST)
    parent_applied["parent_edge"] = np.int8(1)
    parent_applied["candidate_origin"] = "m23_46_parent"

    # Byte-exact no-op reconstruction is a pre-GT protocol gate.
    reconstructed = output_root / "baseline_reconstruction" / "track_results" / f"{seq}.txt"
    baseline_report = write_tracker(
        seq, source_parent, nodes, parent_applied, reconstructed
    )
    parent_sha = sha256_file(parent_tracker)
    reconstructed_sha = sha256_file(reconstructed)
    baseline_exact = parent_tracker.read_bytes() == reconstructed.read_bytes()
    if not baseline_exact:
        raise RuntimeError("M23-46 baseline reconstruction is not byte-exact")

    bank = read_allowlisted_parquet(bank_path, EDGE_ALLOWLIST)
    missing_edges = {
        "src_chunk",
        "dst_chunk",
        "gap",
        "appearance_cos",
        "same_source",
        "motion_error_min",
        "endpoint_displacement",
        "log_height_ratio",
    } - set(bank.columns)
    if missing_edges:
        raise RuntimeError(f"missing edge observables: {sorted(missing_edges)}")

    # Only cross edges are taken from the generic bank. Current parent edges are
    # supplied exclusively by the frozen M23-46 applied graph.
    cross = bank[bank.same_source.to_numpy(int) == 0].copy()
    first_frame = nodes.first_frame.to_numpy(int)
    last_frame = nodes.last_frame.to_numpy(int)
    src = cross.src_chunk.to_numpy(int)
    dst = cross.dst_chunk.to_numpy(int)
    valid_index = (
        (src >= 0)
        & (src < len(nodes))
        & (dst >= 0)
        & (dst < len(nodes))
        & (src != dst)
    )
    cross = cross.loc[valid_index].copy()
    src = cross.src_chunk.to_numpy(int)
    dst = cross.dst_chunk.to_numpy(int)
    temporal = first_frame[dst] > last_frame[src]
    reachable = (
        (cross.gap.to_numpy(float) >= 0)
        & (cross.gap.to_numpy(float) <= max_gap)
        & (cross.motion_error_min.to_numpy(float) <= max_motion_error)
        & (
            cross.endpoint_displacement.to_numpy(float)
            <= max_endpoint_displacement
        )
        & (
            np.abs(cross.log_height_ratio.to_numpy(float))
            <= max_abs_log_height_ratio
        )
    )
    cross = cross.loc[temporal & reachable].copy()
    cross["parent_edge"] = np.int8(0)
    cross["candidate_origin"] = "gt_free_cross_bank"

    candidates = pd.concat([parent_applied, cross], ignore_index=True, sort=False)
    candidates.sort_values(
        ["parent_edge", "src_chunk", "dst_chunk"],
        ascending=[False, True, True],
        kind="mergesort",
        inplace=True,
    )
    candidates.drop_duplicates(
        ["src_chunk", "dst_chunk"], keep="first", inplace=True
    )
    candidates.reset_index(drop=True, inplace=True)
    candidates = rank_edges(candidates)
    candidates["mutual_appearance_topk"] = (
        (candidates.appearance_out_rank <= top_k)
        & (candidates.appearance_in_rank <= top_k)
    ).astype(np.int8)
    candidates["mutual_motion_topk"] = (
        (candidates.motion_out_rank <= top_k)
        & (candidates.motion_in_rank <= top_k)
    ).astype(np.int8)
    keep = (
        (candidates.parent_edge.to_numpy(int) == 1)
        | (candidates.appearance_out_rank.to_numpy(int) <= top_k)
        | (candidates.appearance_in_rank.to_numpy(int) <= top_k)
        | (candidates.motion_out_rank.to_numpy(int) <= top_k)
        | (candidates.motion_in_rank.to_numpy(int) <= top_k)
    )
    candidates = candidates.loc[keep].copy()

    if smoke_active_chunks > 0:
        active = set(
            nodes.sort_values(["first_frame", "chunk_id"], kind="mergesort")
            .head(smoke_active_chunks)
            .chunk_id.astype(int)
        )
        parent_mask = candidates.parent_edge.to_numpy(int) == 1
        active_cross = (
            candidates.src_chunk.astype(int).isin(active)
            & candidates.dst_chunk.astype(int).isin(active)
        ).to_numpy()
        candidates = candidates.loc[parent_mask | active_cross].copy()

    if smoke_max_cross > 0:
        parent = candidates[candidates.parent_edge.to_numpy(int) == 1].copy()
        nonparent = candidates[candidates.parent_edge.to_numpy(int) == 0].copy()
        nonparent["best_rank"] = nonparent[
            [
                "appearance_out_rank",
                "appearance_in_rank",
                "motion_out_rank",
                "motion_in_rank",
            ]
        ].min(axis=1)
        nonparent.sort_values(
            [
                "mutual_appearance_topk",
                "mutual_motion_topk",
                "best_rank",
                "src_chunk",
                "dst_chunk",
            ],
            ascending=[False, False, True, True, True],
            kind="mergesort",
            inplace=True,
        )
        nonparent = nonparent.head(smoke_max_cross).drop(columns=["best_rank"])
        candidates = pd.concat([parent, nonparent], ignore_index=True, sort=False)

    candidates.sort_values(
        ["src_chunk", "dst_chunk", "parent_edge"],
        ascending=[True, True, False],
        kind="mergesort",
        inplace=True,
    )
    candidates.drop_duplicates(["src_chunk", "dst_chunk"], keep="first", inplace=True)
    candidates.reset_index(drop=True, inplace=True)

    forbidden_nodes = audit_frozen_columns(nodes.columns)
    forbidden_edges = audit_frozen_columns(candidates.columns)
    if forbidden_nodes or forbidden_edges:
        raise RuntimeError(
            f"GT-derived frozen columns: nodes={forbidden_nodes}, edges={forbidden_edges}"
        )
    if np.any(
        nodes.first_frame.to_numpy(int)[candidates.dst_chunk.to_numpy(int)]
        <= nodes.last_frame.to_numpy(int)[candidates.src_chunk.to_numpy(int)]
    ):
        raise RuntimeError("frozen graph contains a non-forward edge")

    frozen_dir = output_root / "frozen_candidate_graph"
    frozen_dir.mkdir(parents=True, exist_ok=True)
    frozen_nodes = frozen_dir / "nodes.parquet"
    frozen_edges = frozen_dir / "edges.parquet"
    nodes.to_parquet(frozen_nodes, index=False)
    candidates.to_parquet(frozen_edges, index=False)
    freeze_manifest = {
        "experiment": "M23-53 global identity-flow capacity",
        "stage": "candidate_graph_frozen_before_gt",
        "teacher_only": True,
        "deployable": False,
        "seq": seq,
        "frozen_at": utc_now(),
        "protocol": {
            "parent": "strict frozen M23-46 tracker/applied graph",
            "nodes": "fixed chunk-30 microtracklets; detection rows unchanged",
            "cross_candidates": "GT-free observable allowlist; temporal and motion reachability",
            "top_k": top_k,
            "selection": "union of top-k outgoing/incoming appearance and motion ranks plus every M23-46 parent edge",
            "dummy_no_link": "implicit zero-weight terminate/restart option in bipartite path cover",
            "max_gap": max_gap,
            "max_motion_error": max_motion_error,
            "max_endpoint_displacement": max_endpoint_displacement,
            "max_abs_log_height_ratio": max_abs_log_height_ratio,
            "smoke_active_chunks": smoke_active_chunks,
            "smoke_max_cross": smoke_max_cross,
        },
        "baseline_reconstruction": {
            **baseline_report,
            "byte_exact": baseline_exact,
            "cached_tracker_sha256": parent_sha,
            "reconstructed_tracker_sha256": reconstructed_sha,
        },
        "inputs": {
            "nodes": str(nodes_path),
            "candidate_bank": str(bank_path),
            "m23_46_applied": str(parent_applied_path),
            "m23_46_tracker": str(parent_tracker),
            "source_parent": str(source_parent),
        },
        "frozen_artifacts": {
            "nodes": str(frozen_nodes),
            "nodes_sha256": sha256_file(frozen_nodes),
            "edges": str(frozen_edges),
            "edges_sha256": sha256_file(frozen_edges),
            "node_rows": len(nodes),
            "edge_rows": len(candidates),
            "parent_edges": int(candidates.parent_edge.sum()),
            "cross_edges": int((candidates.parent_edge == 0).sum()),
            "node_columns": list(nodes.columns),
            "edge_columns": list(candidates.columns),
            "forbidden_node_columns": forbidden_nodes,
            "forbidden_edge_columns": forbidden_edges,
        },
        "gt_opened": False,
    }
    freeze_path = frozen_dir / "freeze_manifest.json"
    json_write(freeze_path, freeze_manifest)
    append_event(
        output_root / "protocol_events.jsonl",
        {
            "event": "candidate_graph_frozen",
            "freeze_manifest": str(freeze_path),
            "freeze_manifest_sha256": sha256_file(freeze_path),
            "gt_opened": False,
        },
    )
    return nodes, candidates, freeze_manifest


def association_contribution(
    gt_total: Counter, tracker_rows: int, matches: Counter
) -> float:
    return float(
        sum(
            (matched * matched)
            / max(1, gt_total[int(gt_id)] + tracker_rows - matched)
            for gt_id, matched in matches.items()
        )
    )


def build_teacher_utilities(
    *,
    seq: str,
    source_parent_root: Path,
    output_root: Path,
    freeze_manifest: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    frozen_dir = output_root / "frozen_candidate_graph"
    nodes_path = frozen_dir / "nodes.parquet"
    edges_path = frozen_dir / "edges.parquet"
    if sha256_file(nodes_path) != freeze_manifest["frozen_artifacts"]["nodes_sha256"]:
        raise RuntimeError("frozen node hash changed before GT opening")
    if sha256_file(edges_path) != freeze_manifest["frozen_artifacts"]["edges_sha256"]:
        raise RuntimeError("frozen edge hash changed before GT opening")

    append_event(
        output_root / "protocol_events.jsonl",
        {
            "event": "gt_teacher_opened_after_candidate_freeze",
            "freeze_verified": True,
            "teacher_only": True,
            "deployable": False,
        },
    )

    nodes = pd.read_parquet(nodes_path)
    edges = pd.read_parquet(edges_path)
    labeler = load_module(
        "m23_53_gt_labeler",
        "scripts/m23_research/m23_11_add_micrograph_utility.py",
    )
    labeler.PARENT = source_parent_root
    m23 = labeler.load_m23()
    rows = labeler.read_tracker(source_parent_root / f"{seq}.txt")
    row_chunk = labeler.line_chunks(rows, nodes)
    row_gt = labeler.matched_gt_per_row(rows, seq, m23)
    totals = labeler.gt_counts(seq)

    chunk_matches = [Counter() for _ in range(len(nodes))]
    for chunk_id, gt_id in zip(row_chunk, row_gt):
        if int(gt_id) > 0:
            chunk_matches[int(chunk_id)][int(gt_id)] += 1

    dominant_gt = np.full(len(nodes), -1, np.int32)
    dominant_count = np.zeros(len(nodes), np.int32)
    matched_rows = np.zeros(len(nodes), np.int32)
    dominant_purity = np.zeros(len(nodes), np.float64)
    for chunk_id, counts in enumerate(chunk_matches):
        matched_rows[chunk_id] = int(sum(counts.values()))
        if counts:
            gt_id, count = counts.most_common(1)[0]
            dominant_gt[chunk_id] = int(gt_id)
            dominant_count[chunk_id] = int(count)
            dominant_purity[chunk_id] = float(count / max(1, matched_rows[chunk_id]))

    # Rank chunks within each teacher identity only after the candidate graph is frozen.
    teacher_order = np.full(len(nodes), -1, np.int32)
    by_identity: dict[int, list[int]] = defaultdict(list)
    for chunk_id, gt_id in enumerate(dominant_gt):
        if int(gt_id) > 0:
            by_identity[int(gt_id)].append(chunk_id)
    for gt_id, chunk_ids in by_identity.items():
        chunk_ids.sort(
            key=lambda chunk_id: (
                int(nodes.iloc[chunk_id].first_frame),
                int(nodes.iloc[chunk_id].last_frame),
                int(chunk_id),
            )
        )
        for order, chunk_id in enumerate(chunk_ids):
            teacher_order[chunk_id] = order

    chunk_rows = nodes.rows.to_numpy(int)
    separate_contribution = np.asarray(
        [
            association_contribution(
                totals, int(chunk_rows[chunk_id]), chunk_matches[chunk_id]
            )
            for chunk_id in range(len(nodes))
        ],
        dtype=np.float64,
    )

    weights = np.empty(len(edges), np.float64)
    same_identity = np.zeros(len(edges), np.int8)
    order_delta = np.zeros(len(edges), np.int32)
    recoverable = np.zeros(len(edges), np.float64)
    chain_coverage = np.zeros(len(edges), np.float64)
    for edge_index, edge in enumerate(edges.itertuples(index=False)):
        src = int(edge.src_chunk)
        dst = int(edge.dst_chunk)
        src_gt = int(dominant_gt[src])
        dst_gt = int(dominant_gt[dst])
        same = src_gt > 0 and src_gt == dst_gt
        delta_order = (
            int(teacher_order[dst] - teacher_order[src]) if same else 0
        )
        merged = chunk_matches[src] + chunk_matches[dst]
        merged_contribution = association_contribution(
            totals, int(chunk_rows[src] + chunk_rows[dst]), merged
        )
        structural_gain = (
            merged_contribution
            - separate_contribution[src]
            - separate_contribution[dst]
        )
        recoverable[edge_index] = structural_gain
        same_identity[edge_index] = int(same and delta_order > 0)
        order_delta[edge_index] = delta_order
        if same and delta_order > 0:
            coverage = float(
                (dominant_count[src] + dominant_count[dst])
                / max(1, totals[src_gt])
            )
            chain_coverage[edge_index] = coverage
            length_term = math.sqrt(max(1.0, float(chunk_rows[src] * chunk_rows[dst])))
            purity_term = float(min(dominant_purity[src], dominant_purity[dst]))
            continuity_term = 1000.0 / float(delta_order)
            weights[edge_index] = (
                continuity_term
                + 10.0 * max(0.0, structural_gain)
                + length_term
                + 10.0 * coverage
                + purity_term
                + 1e-6
            )
        else:
            # Explicitly worse than the zero-cost dummy terminate/restart edge.
            length_term = math.sqrt(max(1.0, float(chunk_rows[src] * chunk_rows[dst])))
            weights[edge_index] = -(
                1.0 + abs(min(0.0, structural_gain)) + 0.01 * length_term
            )

    teacher_edges = edges.copy()
    teacher_edges["teacher_src_dominant_gt"] = dominant_gt[
        teacher_edges.src_chunk.to_numpy(int)
    ]
    teacher_edges["teacher_dst_dominant_gt"] = dominant_gt[
        teacher_edges.dst_chunk.to_numpy(int)
    ]
    teacher_edges["teacher_same_identity_forward"] = same_identity
    teacher_edges["teacher_identity_order_delta"] = order_delta
    teacher_edges["teacher_recoverable_association"] = recoverable
    teacher_edges["teacher_chain_coverage"] = chain_coverage
    teacher_edges["teacher_weight"] = weights

    positive = teacher_edges[teacher_edges.teacher_weight > 0.0].copy()
    if positive.empty:
        selected = positive.copy()
    else:
        src = positive.src_chunk.to_numpy(int)
        dst = positive.dst_chunk.to_numpy(int)
        utility = positive.teacher_weight.to_numpy(float)
        offset = float(utility.max()) + 1.0
        rows_index = np.concatenate([src, np.arange(len(nodes), dtype=int)])
        cols_index = np.concatenate(
            [dst, len(nodes) + np.arange(len(nodes), dtype=int)]
        )
        costs = np.concatenate(
            [offset - utility, np.full(len(nodes), offset, dtype=float)]
        )
        matrix = coo_matrix(
            (costs, (rows_index, cols_index)),
            shape=(len(nodes), 2 * len(nodes)),
        ).tocsr()
        matched_rows_index, matched_cols_index = min_weight_full_bipartite_matching(
            matrix
        )
        real = matched_cols_index < len(nodes)
        selected_pairs = set(
            zip(
                matched_rows_index[real].tolist(),
                matched_cols_index[real].tolist(),
            )
        )
        selected = positive[
            [
                (int(src_chunk), int(dst_chunk)) in selected_pairs
                for src_chunk, dst_chunk in zip(
                    positive.src_chunk, positive.dst_chunk
                )
            ]
        ].copy()

    # Time-forward candidate construction makes acyclicity structural; verify it.
    graph_chains(selected, len(nodes))
    if selected.src_chunk.duplicated().any() or selected.dst_chunk.duplicated().any():
        raise RuntimeError("teacher path cover is not one-to-one")
    first_frame = nodes.first_frame.to_numpy(int)
    last_frame = nodes.last_frame.to_numpy(int)
    if len(selected) and np.any(
        first_frame[selected.dst_chunk.to_numpy(int)]
        <= last_frame[selected.src_chunk.to_numpy(int)]
    ):
        raise RuntimeError("teacher path cover is not time-forward")

    teacher_dir = output_root / "teacher_identity_flow"
    teacher_dir.mkdir(parents=True, exist_ok=True)
    teacher_edges.to_parquet(teacher_dir / "teacher_edge_utilities.parquet", index=False)
    selected.to_parquet(teacher_dir / "selected_path_cover_edges.parquet", index=False)
    teacher_nodes = nodes.copy()
    teacher_nodes["teacher_dominant_gt"] = dominant_gt
    teacher_nodes["teacher_dominant_count"] = dominant_count
    teacher_nodes["teacher_matched_rows"] = matched_rows
    teacher_nodes["teacher_dominant_purity"] = dominant_purity
    teacher_nodes["teacher_identity_order"] = teacher_order
    teacher_nodes.to_parquet(teacher_dir / "teacher_node_labels.parquet", index=False)

    report = {
        "gt_opened_after_freeze": True,
        "teacher_only": True,
        "deployable": False,
        "matched_tracker_rows": int((row_gt > 0).sum()),
        "gt_identities": len(totals),
        "candidate_edges": len(teacher_edges),
        "positive_same_identity_edges": int((weights > 0.0).sum()),
        "negative_or_unmatched_edges": int((weights < 0.0).sum()),
        "selected_edges": len(selected),
        "selected_parent_edges": int(selected.parent_edge.sum()) if len(selected) else 0,
        "selected_cross_edges": int((selected.parent_edge == 0).sum()) if len(selected) else 0,
        "one_to_one": True,
        "acyclic": True,
        "time_forward": True,
        "objective": (
            "maximum-weight global path cover over frozen candidates; same dominant "
            "GT forward edges positive and continuity-weighted; different/unmatched "
            "identity edges negative; dummy terminate/restart has zero weight"
        ),
    }
    json_write(teacher_dir / "teacher_manifest.json", report)
    return nodes, selected, report


def run_official_trackeval(
    *, seq: str, output_root: Path, tracker_name: str
) -> dict:
    work_dir = output_root / "official_eval"
    command = [
        sys.executable,
        "scripts/eval_motstyle_trackeval.py",
        "--benchmark-name",
        "MOT20",
        "--split-to-eval",
        "train",
        "--gt-root",
        "datasets/MOT20/train",
        "--results-dir",
        str(output_root / "track_results"),
        "--tracker-name",
        tracker_name,
        "--work-dir",
        str(work_dir),
        "--keep-workdir",
        "--seqs",
        seq,
    ]
    completed = subprocess.run(
        command,
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    (output_root / "official_trackeval.log").write_text(
        completed.stdout, encoding="utf-8"
    )
    if completed.returncode:
        raise RuntimeError(completed.stdout[-8000:])
    detailed = work_dir / "eval" / tracker_name / "pedestrian_detailed.csv"
    rows = list(csv.DictReader(detailed.open(encoding="utf-8")))
    row = next(item for item in rows if item["seq"] == seq)
    return {
        "HOTA": 100.0 * float(row["HOTA___AUC"]),
        "DetA": 100.0 * float(row["DetA___AUC"]),
        "AssA": 100.0 * float(row["AssA___AUC"]),
        "IDSW": int(float(row["IDSW"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", required=True, choices=SEQUENCES)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--graph-root", default=str(DEFAULT_GRAPH_ROOT))
    parser.add_argument("--baseline-cache", default=str(DEFAULT_BASELINE_CACHE))
    parser.add_argument("--source-parent", default=str(DEFAULT_SOURCE_PARENT))
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--max-gap", type=int, default=600)
    parser.add_argument("--max-motion-error", type=float, default=5.0)
    parser.add_argument("--max-endpoint-displacement", type=float, default=6.0)
    parser.add_argument("--max-abs-log-height-ratio", type=float, default=1.2)
    parser.add_argument("--smoke-active-chunks", type=int, default=0)
    parser.add_argument("--smoke-max-cross", type=int, default=0)
    parser.add_argument("--tracker-name", default=None)
    parser.add_argument("--skip-official-trackeval", action="store_true")
    args = parser.parse_args()

    if args.top_k <= 0:
        raise ValueError("top-k must be positive")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    tracker_name = args.tracker_name or output_root.name

    nodes, _, freeze_manifest = build_and_freeze_candidates(
        seq=args.seq,
        graph_root=Path(args.graph_root),
        baseline_cache=Path(args.baseline_cache),
        source_parent_root=Path(args.source_parent),
        output_root=output_root,
        top_k=args.top_k,
        max_gap=args.max_gap,
        max_motion_error=args.max_motion_error,
        max_endpoint_displacement=args.max_endpoint_displacement,
        max_abs_log_height_ratio=args.max_abs_log_height_ratio,
        smoke_active_chunks=args.smoke_active_chunks,
        smoke_max_cross=args.smoke_max_cross,
    )
    nodes, selected, teacher_report = build_teacher_utilities(
        seq=args.seq,
        source_parent_root=Path(args.source_parent),
        output_root=output_root,
        freeze_manifest=freeze_manifest,
    )
    tracker_path = output_root / "track_results" / f"{args.seq}.txt"
    tracker_report = write_tracker(
        args.seq,
        Path(args.source_parent) / f"{args.seq}.txt",
        nodes,
        selected,
        tracker_path,
    )
    tracker_report["sha256"] = sha256_file(tracker_path)

    official = None
    if not args.skip_official_trackeval:
        official = run_official_trackeval(
            seq=args.seq, output_root=output_root, tracker_name=tracker_name
        )

    report = {
        "experiment": "M23-53 Global Identity Flow Capacity",
        "seq": args.seq,
        "status": "completed",
        "teacher_only": True,
        "deployable": False,
        "strict_parent": "M23-46 frozen deployable tracker/applied graph",
        "candidate_freeze_manifest": str(
            output_root / "frozen_candidate_graph" / "freeze_manifest.json"
        ),
        "candidate_freeze_manifest_sha256": sha256_file(
            output_root / "frozen_candidate_graph" / "freeze_manifest.json"
        ),
        "freeze": freeze_manifest,
        "teacher": teacher_report,
        "tracker": tracker_report,
        "official_trackeval": official,
        "stopping_rule": {
            "combined_below_80_3": "do not train M23-54; fixed chunk-30 graph lacks margin",
            "combined_80_3_to_80_7": "mechanism evidence only; expand GT-free node/candidate capacity",
            "combined_at_least_80_7": "eligible for strict M23-54",
        },
    }
    json_write(output_root / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
