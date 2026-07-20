#!/usr/bin/env python3
from __future__ import annotations

"""M23-57 optional intra-node change-point teacher-capacity audit.

Protocol boundary:
1. freeze every legal adjacent boundary and every observable feature before GT;
2. verify hashes, then open the existing M23-53/M23-55 canonical row matching;
3. split at teacher ownership transitions, regenerate the fixed M23-55-style
   candidate graph, and run the unchanged M23-53 global path-cover teacher.

This is teacher-only and never creates a deployable tracker.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import resource
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
SEQUENCES = ("MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05")
ROOT = Path("outputs/mot20_m23_20260718/m23_57_intra_node_change_point_capacity_v2")
PREREG = ROOT / "preregistered_protocol.json"
PREREG_SHA256 = "06ef9d70f2fe319de8e787e7aedcf74895c39736b31da1146c84cec24a5778d5"
FIXED_GRAPH_ROOT = Path("outputs/mot20_m23_20260718/micrograph_chunk30_v1")
BASELINE_CACHE = Path("outputs/mot20_m23_20260718/m23_47_m23_46_deployable_baseline_cache_v1")
SOURCE_PARENT = Path(
    "outputs/assocriskbench_p15_20260714/"
    "fused_a42_adaptive_risk_eval/l1_r2_q75_adaptive_priority/track_results"
)
M55_ROOTS = {
    "MOT20-01": Path("outputs/mot20_m23_20260718/m23_55_stratified_gap_candidates_m01_v1"),
    "MOT20-02": Path("outputs/mot20_m23_20260718/m23_55_stratified_gap_candidates_m02_v1"),
    "MOT20-03": Path("outputs/mot20_m23_20260718/m23_55_stratified_gap_candidates_m03_v1"),
    "MOT20-05": Path("outputs/mot20_m23_20260718/m23_55_stratified_gap_candidates_m05_v1"),
}
M55_COMBINED = Path("outputs/mot20_m23_20260718/m23_55_stratified_gap_capacity_combined_v1/report.json")
FORBIDDEN_TOKENS = (
    "same_gt",
    "modal_gt",
    "purity",
    "label_confidence",
    "actual_assa",
    "matched_gt",
    "teacher",
    "delta_hota",
)
WINDOWS = (1, 3, 5, 8)
BINARY_FEATURES = {
    "left_embedding_valid",
    "right_embedding_valid",
    "both_embedding_valid",
    "parent_id_transition",
    "near_node_start",
    "near_node_end",
}
ID_COLUMNS = {
    "boundary_id",
    "fixed_chunk_id",
    "source_track_id",
    "source_ordinal",
    "position",
    "left_row_index",
    "right_row_index",
    "left_line",
    "right_line",
    "left_frame",
    "right_frame",
    "chunk_first_frame",
    "chunk_last_frame",
    "chunk_rows",
    "source_predecessor_chunk",
    "source_successor_chunk",
    "left_parent_tracker_id",
    "right_parent_tracker_id",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_event(root: Path, payload: dict[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    record = {"time": utc_now(), **payload}
    with (root / "protocol_events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def load_module(name: str, relative_path: str):
    path = REPO / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def peak_rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)


def verify_preregistration(root: Path) -> dict[str, Any]:
    path = root / "preregistered_protocol.json"
    if not path.exists():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != PREREG_SHA256:
        raise RuntimeError(f"preregistration SHA mismatch: {actual} != {PREREG_SHA256}")
    return json.loads(path.read_text(encoding="utf-8"))


def audit_columns(columns: Iterable[str]) -> list[str]:
    return [str(column) for column in columns if any(token in str(column).lower() for token in FORBIDDEN_TOKENS)]


def detector_payload(path: Path) -> str:
    rows: list[tuple[Any, ...]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split(",")
            if len(fields) < 6:
                continue
            payload = (
                int(float(fields[0])),
                round(float(fields[2]), 8),
                round(float(fields[3]), 8),
                round(float(fields[4]), 8),
                round(float(fields[5]), 8),
                tuple(fields[6:]),
            )
            rows.append(payload)
    rows.sort()
    digest = hashlib.sha256()
    for row in rows:
        digest.update(repr(row).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def empirical_percentile(values: np.ndarray) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=float))
    return series.rank(method="average", pct=True).fillna(0.5).to_numpy(np.float32)


def normalized_mean(vectors: np.ndarray) -> np.ndarray | None:
    if len(vectors) == 0:
        return None
    value = vectors.mean(axis=0)
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        return None
    return (value / norm).astype(np.float32)


def embedding_stats(vectors: np.ndarray) -> tuple[np.ndarray | None, float, np.ndarray | None]:
    prototype = normalized_mean(vectors)
    if prototype is None:
        return None, 1.0, None
    similarity = vectors @ prototype
    variance = float(np.mean(1.0 - similarity))
    medoid = vectors[int(np.argmax(similarity))]
    return prototype, variance, medoid


def safe_cosine_distance(left: np.ndarray | None, right: np.ndarray | None) -> float:
    if left is None or right is None:
        return 1.0
    return float(np.clip(1.0 - float(left @ right), 0.0, 2.0))


def velocity(frames: np.ndarray, cx: np.ndarray, cy: np.ndarray) -> tuple[float, float]:
    if len(frames) < 2 or int(frames[-1]) == int(frames[0]):
        return 0.0, 0.0
    x = frames.astype(float)
    x -= x.mean()
    denominator = max(float(x @ x), 1e-12)
    return float((x @ cx.astype(float)) / denominator), float((x @ cy.astype(float)) / denominator)


def acceleration(frames: np.ndarray, cx: np.ndarray, cy: np.ndarray) -> tuple[float, float]:
    if len(frames) < 3:
        return 0.0, 0.0
    dt1 = max(float(frames[-2] - frames[-3]), 1.0)
    dt2 = max(float(frames[-1] - frames[-2]), 1.0)
    vx1 = float((cx[-2] - cx[-3]) / dt1)
    vy1 = float((cy[-2] - cy[-3]) / dt1)
    vx2 = float((cx[-1] - cx[-2]) / dt2)
    vy2 = float((cy[-1] - cy[-2]) / dt2)
    return vx2 - vx1, vy2 - vy1


def side_motion_consistency(frames: np.ndarray, cx: np.ndarray, cy: np.ndarray, height: np.ndarray) -> float:
    if len(frames) < 3:
        return 0.0
    vx, vy = velocity(frames, cx, cy)
    residuals = []
    for index in range(1, len(frames)):
        dt = max(float(frames[index] - frames[index - 1]), 1.0)
        px = float(cx[index - 1] + vx * dt)
        py = float(cy[index - 1] + vy * dt)
        scale = max(float(0.5 * (height[index - 1] + height[index])), 1.0)
        residuals.append(math.hypot(float(cx[index]) - px, float(cy[index]) - py) / scale)
    return float(np.mean(residuals)) if residuals else 0.0


def source_neighbors(fixed_nodes: pd.DataFrame) -> tuple[dict[int, int], dict[int, int]]:
    predecessor: dict[int, int] = {}
    successor: dict[int, int] = {}
    for _, group in fixed_nodes.groupby("source_track_id", sort=True):
        ordered = group.sort_values(["source_ordinal", "first_frame", "chunk_id"], kind="mergesort")
        ids = ordered.chunk_id.astype(int).tolist()
        for left, right in zip(ids, ids[1:]):
            successor[left] = right
            predecessor[right] = left
    return predecessor, successor


def prepare_observable_rows(seq: str):
    m10 = load_module(f"m23_57_m10_{seq[-2:]}", "scripts/m23_research/m23_10_build_micrograph.py")
    m53 = load_module(f"m23_57_m53_{seq[-2:]}", "scripts/m23_research/m23_53_global_identity_flow_capacity.py")
    m53b = load_module(f"m23_57_m53b_{seq[-2:]}", "scripts/m23_research/m23_53b_build_adaptive_micrograph.py")
    source_path = SOURCE_PARENT / f"{seq}.txt"
    baseline_path = BASELINE_CACHE / seq / "track_results" / f"{seq}.txt"
    nodes_path = FIXED_GRAPH_ROOT / seq / "microtracklets.parquet"
    for path in (source_path, baseline_path, nodes_path):
        if not path.exists():
            raise FileNotFoundError(path)
    source_rows = m53b.read_source_rows(source_path)
    compatible_rows = m10.read_tracker(source_path)
    if len(source_rows) != len(compatible_rows):
        raise RuntimeError("source reader mismatch")
    fixed_nodes = pd.read_parquet(nodes_path).sort_values("chunk_id", kind="mergesort").reset_index(drop=True)
    chunks = m53b.fixed_chunk_rows(source_rows, fixed_nodes, m53)
    parent_ids = m53b.align_parent_tracker_ids(source_rows, baseline_path)
    crowd = m53b.local_crowd_density(source_rows)
    rng = np.random.default_rng(m10.SEED)
    projection = rng.normal(size=(2048, m10.DIM)).astype(np.float32) / math.sqrt(m10.DIM)
    phase, match_iou, unique_embeddings, positions = m10.map_phase(compatible_rows, seq, projection)
    mapped = phase >= 0
    row_embeddings = np.zeros((len(source_rows), m10.DIM), np.float32)
    if np.any(mapped):
        row_embeddings[mapped] = unique_embeddings[positions[mapped]]
    return m10, m53, m53b, source_path, baseline_path, source_rows, fixed_nodes, chunks, parent_ids, crowd, mapped, match_iou, row_embeddings


def freeze_boundaries(seq: str, root: Path) -> dict[str, Any]:
    verify_preregistration(root)
    implementation = root / "implementation_manifest.json"
    if not implementation.exists():
        raise RuntimeError("freeze implementation before boundary actions")
    sequence_root = root / "boundary_universe" / seq
    manifest_path = sequence_root / "freeze_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite frozen boundary universe: {sequence_root}")
    started = time.time()
    (
        _,
        _,
        _,
        source_path,
        baseline_path,
        source_rows,
        fixed_nodes,
        chunks,
        parent_ids,
        crowd,
        mapped,
        match_iou,
        row_embeddings,
    ) = prepare_observable_rows(seq)
    predecessor, successor = source_neighbors(fixed_nodes)
    records: list[dict[str, Any]] = []
    membership_records: list[dict[str, Any]] = []
    boundary_id = 0
    for node in fixed_nodes.itertuples(index=False):
        chunk_id = int(node.chunk_id)
        indices = chunks[chunk_id]
        membership_records.append(
            {
                "fixed_chunk_id": chunk_id,
                "source_track_id": int(node.source_track_id),
                "source_ordinal": int(node.source_ordinal),
                "row_indices": [int(value) for value in indices],
                "line_indices": [int(source_rows[value]["line"]) for value in indices],
                "frames": [int(source_rows[value]["frame"]) for value in indices],
            }
        )
        count = len(indices)
        if count < 2:
            continue
        frames = np.asarray([source_rows[index]["frame"] for index in indices], np.int32)
        cx = np.asarray([source_rows[index]["cx"] for index in indices], np.float32)
        cy = np.asarray([source_rows[index]["cy"] for index in indices], np.float32)
        width = np.asarray([source_rows[index]["x2"] - source_rows[index]["x1"] for index in indices], np.float32)
        height = np.asarray([source_rows[index]["height"] for index in indices], np.float32)
        score = np.asarray([source_rows[index]["score"] for index in indices], np.float32)
        local_crowd = crowd[np.asarray(indices, np.int64)]
        local_parent = parent_ids[np.asarray(indices, np.int64)]
        local_mapped = mapped[np.asarray(indices, np.int64)]
        local_iou = match_iou[np.asarray(indices, np.int64)]
        local_embedding = row_embeddings[np.asarray(indices, np.int64)]
        for position in range(1, count):
            left_global = indices[position - 1]
            right_global = indices[position]
            if int(frames[position]) <= int(frames[position - 1]):
                raise RuntimeError(f"{seq} chunk {chunk_id} contains non-forward adjacent rows")
            record: dict[str, Any] = {
                "boundary_id": boundary_id,
                "fixed_chunk_id": chunk_id,
                "source_track_id": int(node.source_track_id),
                "source_ordinal": int(node.source_ordinal),
                "position": position,
                "left_row_index": int(left_global),
                "right_row_index": int(right_global),
                "left_line": int(source_rows[left_global]["line"]),
                "right_line": int(source_rows[right_global]["line"]),
                "left_frame": int(frames[position - 1]),
                "right_frame": int(frames[position]),
                "chunk_first_frame": int(frames[0]),
                "chunk_last_frame": int(frames[-1]),
                "chunk_rows": count,
                "source_predecessor_chunk": int(predecessor.get(chunk_id, -1)),
                "source_successor_chunk": int(successor.get(chunk_id, -1)),
                "left_parent_tracker_id": int(local_parent[position - 1]),
                "right_parent_tracker_id": int(local_parent[position]),
                "parent_id_transition": int(local_parent[position - 1] != local_parent[position]),
                "boundary_relative_position": float(position / count),
                "node_age_left": float(position),
                "node_age_right": float(count - position),
                "near_node_start": int(position <= 2),
                "near_node_end": int(count - position <= 2),
                "frame_gap": float(frames[position] - frames[position - 1] - 1),
                "left_embedding_valid": int(local_mapped[position - 1]),
                "right_embedding_valid": int(local_mapped[position]),
                "both_embedding_valid": int(local_mapped[position - 1] and local_mapped[position]),
            }
            for window in WINDOWS:
                left_positions = np.arange(max(0, position - window), position, dtype=np.int64)
                right_positions = np.arange(position, min(count, position + window), dtype=np.int64)
                left_valid = left_positions[local_mapped[left_positions]]
                right_valid = right_positions[local_mapped[right_positions]]
                left_vectors = local_embedding[left_valid]
                right_vectors = local_embedding[right_valid]
                left_proto, left_variance, left_medoid = embedding_stats(left_vectors)
                right_proto, right_variance, right_medoid = embedding_stats(right_vectors)
                record[f"appearance_jump_w{window}"] = safe_cosine_distance(left_proto, right_proto)
                record[f"left_prototype_variance_w{window}"] = left_variance
                record[f"right_prototype_variance_w{window}"] = right_variance
                record[f"bilateral_medoid_distance_w{window}"] = safe_cosine_distance(left_medoid, right_medoid)
                record[f"left_valid_embeddings_w{window}"] = float(len(left_valid))
                record[f"right_valid_embeddings_w{window}"] = float(len(right_valid))
            record["single_minus_window_jump"] = float(record["appearance_jump_w1"] - record["appearance_jump_w8"])
            record["embedding_change_cumulative"] = float(record["appearance_jump_w8"] + 0.5 * record["appearance_jump_w5"] + 0.25 * record["appearance_jump_w3"])
            left_motion = np.arange(max(0, position - 5), position, dtype=np.int64)
            right_motion = np.arange(position, min(count, position + 5), dtype=np.int64)
            lvx, lvy = velocity(frames[left_motion], cx[left_motion], cy[left_motion])
            rvx, rvy = velocity(frames[right_motion], cx[right_motion], cy[right_motion])
            normalization = max(float(0.5 * (height[position - 1] + height[position])), 1.0)
            record["velocity_innovation"] = float(math.hypot(rvx - lvx, rvy - lvy) / normalization)
            lax, lay = acceleration(frames[left_motion], cx[left_motion], cy[left_motion])
            reverse_right = right_motion[::-1]
            rax_reverse, ray_reverse = acceleration(-frames[reverse_right], cx[reverse_right], cy[reverse_right])
            rax, ray = -rax_reverse, -ray_reverse
            record["acceleration_jump"] = float(math.hypot(rax - lax, ray - lay) / normalization)
            dt = max(float(frames[position] - frames[position - 1]), 1.0)
            displacement = math.hypot(float(cx[position] - cx[position - 1]), float(cy[position] - cy[position - 1]))
            record["normalized_center_displacement"] = float(displacement / (normalization * dt))
            record["width_jump"] = float(abs(math.log(max(float(width[position]), 1e-3) / max(float(width[position - 1]), 1e-3))))
            record["height_jump"] = float(abs(math.log(max(float(height[position]), 1e-3) / max(float(height[position - 1]), 1e-3))))
            record["scale_jump"] = float(abs(math.log(max(float(width[position] * height[position]), 1e-3) / max(float(width[position - 1] * height[position - 1]), 1e-3))))
            predicted_x = float(cx[position - 1] + lvx * dt)
            predicted_y = float(cy[position - 1] + lvy * dt)
            record["predicted_position_residual"] = float(math.hypot(float(cx[position]) - predicted_x, float(cy[position]) - predicted_y) / normalization)
            record["left_motion_consistency"] = side_motion_consistency(frames[left_motion], cx[left_motion], cy[left_motion], height[left_motion])
            record["right_motion_consistency"] = side_motion_consistency(frames[right_motion], cx[right_motion], cy[right_motion], height[right_motion])
            record["left_confidence"] = float(score[position - 1])
            record["right_confidence"] = float(score[position])
            record["confidence_jump"] = float(abs(score[position] - score[position - 1]))
            record["left_match_quality"] = float(local_iou[position - 1])
            record["right_match_quality"] = float(local_iou[position])
            record["endpoint_quality"] = float(min(score[position - 1] * local_iou[position - 1], score[position] * local_iou[position]))
            record["left_crowd_density"] = float(local_crowd[position - 1])
            record["right_crowd_density"] = float(local_crowd[position])
            record["crowd_density_jump"] = float(abs(local_crowd[position] - local_crowd[position - 1]))
            record["occlusion_proxy"] = float(max(local_crowd[position - 1], local_crowd[position]) / max(min(score[position - 1], score[position]), 1e-3))
            record["neighbor_configuration_change"] = float(abs(math.log1p(local_crowd[position]) - math.log1p(local_crowd[position - 1])))
            record["competing_detection_ambiguity"] = float(max(local_crowd[position - 1], local_crowd[position]) / (1.0 + max(local_crowd[position - 1], local_crowd[position])))
            records.append(record)
            boundary_id += 1
    boundaries = pd.DataFrame(records)
    membership = pd.DataFrame(membership_records)
    if boundaries.empty:
        raise RuntimeError(f"no internal boundaries for {seq}")
    if not np.array_equal(boundaries.boundary_id.to_numpy(np.int64), np.arange(len(boundaries), dtype=np.int64)):
        raise RuntimeError("boundary IDs are not dense")
    forbidden = audit_columns(boundaries.columns)
    if forbidden:
        raise RuntimeError(f"forbidden frozen input columns: {forbidden}")
    feature_columns = [
        column
        for column in boundaries.columns
        if column not in ID_COLUMNS and column not in {"boundary_id"}
    ]
    continuous = [column for column in feature_columns if column not in BINARY_FEATURES]
    for column in continuous:
        boundaries[f"{column}_pct"] = empirical_percentile(boundaries[column].to_numpy(float))
    feature_columns = [column for column in boundaries.columns if column not in ID_COLUMNS and column != "boundary_id"]
    forbidden = audit_columns(feature_columns)
    if forbidden:
        raise RuntimeError(f"forbidden feature columns after percentile expansion: {forbidden}")
    sequence_root.mkdir(parents=True, exist_ok=True)
    boundaries_path = sequence_root / "boundary_features.parquet"
    membership_path = sequence_root / "chunk_membership.parquet"
    matrix_path = sequence_root / "features.f16.npy"
    boundaries.to_parquet(boundaries_path, index=False)
    membership.to_parquet(membership_path, index=False)
    matrix = boundaries[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float32)
    np.save(matrix_path, matrix.astype(np.float16))
    manifest = {
        "experiment": "M23-57 GT-free internal-boundary action universe",
        "seq": seq,
        "stage": "boundary_actions_and_observables_frozen_before_gt",
        "gt_opened": False,
        "teacher_only": True,
        "deployable": False,
        "git_head": git_head(),
        "preregistration_sha256": PREREG_SHA256,
        "implementation_manifest_sha256": sha256_file(implementation),
        "source_tracker": str(source_path),
        "source_tracker_sha256": sha256_file(source_path),
        "m23_46_tracker": str(baseline_path),
        "m23_46_tracker_sha256": sha256_file(baseline_path),
        "fixed_nodes": str(FIXED_GRAPH_ROOT / seq / "microtracklets.parquet"),
        "fixed_nodes_sha256": sha256_file(FIXED_GRAPH_ROOT / seq / "microtracklets.parquet"),
        "rows": len(source_rows),
        "fixed_nodes_count": len(fixed_nodes),
        "boundary_actions": len(boundaries),
        "mandatory_parent_transition_boundaries": int(boundaries.parent_id_transition.sum()),
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "forbidden_columns": forbidden,
        "artifacts": {
            "boundary_features": str(boundaries_path),
            "boundary_features_sha256": sha256_file(boundaries_path),
            "chunk_membership": str(membership_path),
            "chunk_membership_sha256": sha256_file(membership_path),
            "feature_matrix": str(matrix_path),
            "feature_matrix_sha256": sha256_file(matrix_path),
        },
        "protocol": {
            "all_adjacent_internal_boundaries": True,
            "minimum_segment_rows": 1,
            "deterministic_compression": "none",
            "normalization": "sequence-local empirical percentile frozen before GT",
            "teacher_labels_opened": False,
        },
        "runtime_seconds": time.time() - started,
        "peak_rss_mb": peak_rss_mb(),
    }
    json_write(manifest_path, manifest)
    append_event(root, {"event": "boundary_universe_frozen", "seq": seq, "manifest": str(manifest_path), "manifest_sha256": sha256_file(manifest_path), "gt_opened": False})
    return manifest


def verify_boundary_freeze(seq: str, root: Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    manifest_path = root / "boundary_universe" / seq / "freeze_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest["artifacts"]
    for name, key in (("boundary_features", "boundary_features_sha256"), ("chunk_membership", "chunk_membership_sha256"), ("feature_matrix", "feature_matrix_sha256")):
        path = Path(artifacts[name])
        if sha256_file(path) != artifacts[key]:
            raise RuntimeError(f"{seq} frozen {name} SHA changed")
    boundaries = pd.read_parquet(artifacts["boundary_features"])
    membership = pd.read_parquet(artifacts["chunk_membership"])
    forbidden = audit_columns(manifest["feature_columns"])
    if forbidden:
        raise RuntimeError(f"{seq} frozen manifest has forbidden features: {forbidden}")
    return manifest, boundaries, membership


def ownership_runs(gt_values: np.ndarray, positions: np.ndarray) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for gt_id, position in zip(gt_values, positions):
        value = int(gt_id)
        if value <= 0:
            continue
        if not runs or int(runs[-1]["identity"]) != value:
            runs.append({"identity": value, "positions": [int(position)]})
        else:
            runs[-1]["positions"].append(int(position))
    return runs


def build_teacher_labels(seq: str, root: Path) -> tuple[pd.DataFrame, dict[str, Any], dict[int, list[int]]]:
    freeze, boundaries, membership = verify_boundary_freeze(seq, root)
    labels_path = root / "teacher_labels" / seq / "boundary_labels.parquet"
    if labels_path.exists():
        raise FileExistsError(f"refusing to overwrite teacher labels: {labels_path}")
    append_event(root, {"event": "teacher_labels_opened_after_boundary_freeze", "seq": seq, "freeze_manifest_sha256": sha256_file(root / "boundary_universe" / seq / "freeze_manifest.json"), "teacher_only": True, "deployable": False})
    labeler = load_module(f"m23_57_labeler_{seq[-2:]}", "scripts/m23_research/m23_11_add_micrograph_utility.py")
    labeler.PARENT = SOURCE_PARENT
    m23 = labeler.load_m23()
    rows = labeler.read_tracker(SOURCE_PARENT / f"{seq}.txt")
    row_gt = labeler.matched_gt_per_row(rows, seq, m23)
    if len(row_gt) != int(freeze["rows"]):
        raise RuntimeError("teacher row count mismatch")
    label = np.full(len(boundaries), -1, np.int8)
    capacity_split = np.zeros(len(boundaries), np.int8)
    ambiguous = np.zeros(len(boundaries), np.int8)
    supported = np.zeros(len(boundaries), np.int8)
    left_support = np.zeros(len(boundaries), np.int16)
    right_support = np.zeros(len(boundaries), np.int16)
    transition_index = np.full(len(boundaries), -1, np.int16)
    node_identity_count = np.zeros(len(boundaries), np.int16)
    pure_node = np.zeros(len(boundaries), np.int8)
    by_key = {(int(row.fixed_chunk_id), int(row.position)): int(row.boundary_id) for row in boundaries.itertuples(index=False)}
    boundary_ids_by_chunk = {
        int(chunk_id): group.boundary_id.to_numpy(np.int64)
        for chunk_id, group in boundaries.groupby("fixed_chunk_id", sort=False)
    }
    teacher_cuts: dict[int, list[int]] = defaultdict(list)
    impure_nodes = 0
    pure_nodes = 0
    unsupported_nodes = 0
    transition_count = 0
    supported_transition_count = 0
    unsupported_transition_count = 0
    aba_patterns = 0
    equivalent_ambiguous = 0
    transition_type_counts: Counter[str] = Counter()
    affected_rows = 0
    for member in membership.itertuples(index=False):
        chunk_id = int(member.fixed_chunk_id)
        indices = np.asarray(member.row_indices, np.int64)
        ownership = row_gt[indices]
        matched_positions = np.flatnonzero(ownership > 0)
        runs = ownership_runs(ownership[matched_positions], matched_positions)
        unique = {int(run["identity"]) for run in runs}
        chunk_boundary_ids = boundary_ids_by_chunk.get(chunk_id, np.empty(0, np.int64))
        node_identity_count[chunk_boundary_ids] = len(unique)
        pure_node[chunk_boundary_ids] = int(len(unique) == 1)
        if len(unique) > 1:
            impure_nodes += 1
        elif len(unique) == 1:
            pure_nodes += 1
        else:
            unsupported_nodes += 1
        identities = [int(run["identity"]) for run in runs]
        aba_patterns += sum(int(identities[index] == identities[index + 2] and identities[index] != identities[index + 1]) for index in range(max(0, len(identities) - 2)))
        chosen_positions: set[int] = set()
        ambiguous_positions: set[int] = set()
        for local_transition, (left_run, right_run) in enumerate(zip(runs, runs[1:])):
            left_last = int(left_run["positions"][-1])
            right_first = int(right_run["positions"][0])
            chosen = left_last + 1
            if chosen < 1 or chosen >= len(indices):
                raise RuntimeError(f"invalid teacher boundary {seq} chunk={chunk_id} position={chosen}")
            chosen_positions.add(chosen)
            teacher_cuts[chunk_id].append(chosen)
            transition_count += 1
            transition_type_counts[f"{left_run['identity']}->{right_run['identity']}"] += 1
            left_count = len(left_run["positions"])
            right_count = len(right_run["positions"])
            is_supported = left_count >= 2 and right_count >= 2
            supported_transition_count += int(is_supported)
            unsupported_transition_count += int(not is_supported)
            boundary_id = by_key[(chunk_id, chosen)]
            capacity_split[boundary_id] = 1
            supported[boundary_id] = int(is_supported)
            left_support[boundary_id] = left_count
            right_support[boundary_id] = right_count
            transition_index[boundary_id] = local_transition
            if is_supported:
                label[boundary_id] = 1
            for position in range(chosen + 1, right_first + 1):
                if 1 <= position < len(indices):
                    ambiguous_positions.add(position)
        equivalent_ambiguous += len(ambiguous_positions - chosen_positions)
        for position in range(1, len(indices)):
            boundary_id = by_key[(chunk_id, position)]
            if position in chosen_positions:
                continue
            if position in ambiguous_positions:
                ambiguous[boundary_id] = 1
                label[boundary_id] = -1
                continue
            left_candidates = matched_positions[matched_positions < position]
            right_candidates = matched_positions[matched_positions >= position]
            if not len(left_candidates) or not len(right_candidates):
                label[boundary_id] = -1
                continue
            left_position = int(left_candidates[-1])
            right_position = int(right_candidates[0])
            left_identity = int(ownership[left_position])
            right_identity = int(ownership[right_position])
            if left_identity != right_identity:
                label[boundary_id] = -1
                ambiguous[boundary_id] = 1
                continue
            containing = next((run for run in runs if left_position in run["positions"] and right_position in run["positions"]), None)
            if containing is not None and len(containing["positions"]) >= 2:
                label[boundary_id] = 0
                supported[boundary_id] = 1
                left_support[boundary_id] = len(containing["positions"])
                right_support[boundary_id] = len(containing["positions"])
        if chosen_positions:
            affected_rows += len(indices)
    labels = boundaries[["boundary_id", "fixed_chunk_id", "position"]].copy()
    labels["audit_label"] = label
    labels["capacity_split"] = capacity_split
    labels["ambiguous_or_unsupported"] = ((label < 0) | (ambiguous > 0)).astype(np.int8)
    labels["supported_boundary"] = supported
    labels["left_run_support"] = left_support
    labels["right_run_support"] = right_support
    labels["transition_index"] = transition_index
    labels["node_identity_count"] = node_identity_count
    labels["pure_node"] = pure_node
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(labels_path, index=False)
    report = {
        "experiment": "M23-57 teacher-only intra-node ownership taxonomy",
        "seq": seq,
        "teacher_only": True,
        "deployable": False,
        "boundary_freeze_verified": True,
        "boundary_freeze_manifest_sha256": sha256_file(root / "boundary_universe" / seq / "freeze_manifest.json"),
        "matched_rows": int((row_gt > 0).sum()),
        "fixed_nodes": len(membership),
        "pure_nodes": pure_nodes,
        "impure_nodes": impure_nodes,
        "unsupported_nodes": unsupported_nodes,
        "change_points": transition_count,
        "supported_change_points": supported_transition_count,
        "unsupported_change_points": unsupported_transition_count,
        "a_to_b_transitions": transition_count,
        "a_to_b_to_a_patterns": aba_patterns,
        "equivalent_or_ambiguous_boundaries": int((labels.ambiguous_or_unsupported > 0).sum()),
        "equivalent_unmatched_span_boundaries": equivalent_ambiguous,
        "audit_positive_boundaries": int((labels.audit_label == 1).sum()),
        "audit_negative_boundaries": int((labels.audit_label == 0).sum()),
        "audit_ignored_boundaries": int((labels.audit_label < 0).sum()),
        "teacher_split_affected_detection_rows": affected_rows,
        "transition_type_count": len(transition_type_counts),
        "artifacts": {
            "labels": str(labels_path),
            "labels_sha256": sha256_file(labels_path),
        },
    }
    json_write(labels_path.parent / "report.json", report)
    append_event(root, {"event": "teacher_boundary_labels_frozen", "seq": seq, "labels_sha256": sha256_file(labels_path), "change_points": transition_count})
    return labels, report, {key: sorted(set(value)) for key, value in teacher_cuts.items()}


def add_legacy_columns(edges: pd.DataFrame) -> pd.DataFrame:
    result = edges.copy()
    result["parent_edge"] = (result.edge_role == "m23_46_parent").astype(np.int8)
    result["candidate_origin"] = np.where(result.parent_edge > 0, "m23_46_parent", "m23_57_regenerated_legacy")
    result["appearance_out_rank"] = result.out_rank.astype(np.float32)
    result["appearance_in_rank"] = result.in_rank.astype(np.float32)
    result["motion_out_rank"] = result.out_rank.astype(np.float32)
    result["motion_in_rank"] = result.in_rank.astype(np.float32)
    result["mutual_appearance_topk"] = ((result.out_rank <= 32) & (result.in_rank <= 32)).astype(np.int8)
    result["mutual_motion_topk"] = result.mutual_appearance_topk.astype(np.int8)
    result["m23_55_recall_score"] = np.nan
    result["m23_55_multi_appearance_cos"] = np.nan
    result["m23_55_whole_appearance_cos"] = np.nan
    result["m23_55_best_view"] = np.int8(-1)
    result["m23_55_gap_bucket"] = np.where(result.gap.to_numpy(int) == 0, "0", "legacy_nonzero")
    return result


def canonical_summary(root: Path, seq: str, nodes: pd.DataFrame, edges: pd.DataFrame, selected: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    labels = pd.read_parquet(root / "capacity" / seq / "teacher_identity_flow" / "teacher_node_labels.parquet")
    labeled = labels[labels.teacher_dominant_gt > 0].copy()
    edge_lookup = (
        edges.sort_values(["parent_edge", "src_chunk", "dst_chunk"], ascending=[False, True, True])
        .drop_duplicates(["src_chunk", "dst_chunk"], keep="first")
        .set_index(["src_chunk", "dst_chunk"], drop=False)
    )
    selected_pairs = set(zip(selected.src_chunk.astype(int), selected.dst_chunk.astype(int)))
    rank_columns = [
        "out_rank", "in_rank", "max_rank", "appearance_out_rank",
        "appearance_in_rank", "motion_out_rank", "motion_in_rank",
    ]
    rows: list[dict[str, Any]] = []
    for gt_id, group in labeled.groupby("teacher_dominant_gt"):
        ordered = group.sort_values(["first_frame", "last_frame", "chunk_id"], kind="mergesort").reset_index(drop=True)
        for source in ordered.itertuples(index=False):
            later = ordered[ordered.first_frame > int(source.last_frame)]
            if later.empty:
                continue
            destination = later.iloc[0]
            src = int(source.chunk_id)
            dst = int(destination.chunk_id)
            pair = (src, dst)
            present = pair in edge_lookup.index
            entry: dict[str, Any] = {
                "seq": seq,
                "teacher_gt": int(gt_id),
                "src_chunk": src,
                "dst_chunk": dst,
                "src_first_frame": int(source.first_frame),
                "src_last_frame": int(source.last_frame),
                "dst_first_frame": int(destination.first_frame),
                "dst_last_frame": int(destination.last_frame),
                "gap": int(destination.first_frame) - int(source.last_frame) - 1,
                "candidate_present": int(present),
                "selected_by_teacher": int(pair in selected_pairs),
                "candidate_is_parent": 0,
                "candidate_is_cross": 0,
            }
            for rank in rank_columns:
                entry[rank] = None
            if present:
                edge = edge_lookup.loc[pair]
                if isinstance(edge, pd.DataFrame):
                    edge = edge.iloc[0]
                entry["candidate_is_parent"] = int(edge.parent_edge)
                entry["candidate_is_cross"] = int(not bool(edge.parent_edge))
                for rank in rank_columns:
                    value = float(edge[rank])
                    entry[rank] = int(value) if math.isfinite(value) else None
            rows.append(entry)
    events = pd.DataFrame(rows)
    present = int(events.candidate_present.sum()) if len(events) else 0
    selected_count = int(events.selected_by_teacher.sum()) if len(events) else 0
    internal = events[nodes.fixed_chunk_id.to_numpy(int)[events.src_chunk.to_numpy(int)] == nodes.fixed_chunk_id.to_numpy(int)[events.dst_chunk.to_numpy(int)]].copy() if len(events) else events.copy()
    return {
        "canonical_successors": len(events),
        "canonical_present_in_flow_graph": present,
        "candidate_recall": present / len(events) if len(events) else None,
        "canonical_selected": selected_count,
        "teacher_flow_conversion_when_present": selected_count / present if present else None,
        "new_internal_canonical_successors": len(internal),
        "new_internal_canonical_present": int(internal.candidate_present.sum()) if len(internal) else 0,
        "new_internal_canonical_selected": int(internal.selected_by_teacher.sum()) if len(internal) else 0,
    }, events


def flow_actions(nodes: pd.DataFrame, edges: pd.DataFrame, selected: pd.DataFrame) -> dict[str, Any]:
    parent = set(zip(edges.loc[edges.parent_edge > 0, "src_chunk"].astype(int), edges.loc[edges.parent_edge > 0, "dst_chunk"].astype(int)))
    chosen = set(zip(selected.src_chunk.astype(int), selected.dst_chunk.astype(int)))
    keep = parent & chosen
    cut = parent - chosen
    cross = chosen - parent
    affected_chunks = {value for pair in cut | cross for value in pair}
    rows = nodes.rows.to_numpy(int)
    affected_rows = int(sum(rows[index] for index in affected_chunks))
    return {
        "keep_parent": len(keep),
        "cut_parent": len(cut),
        "cross": len(cross),
        "dummy_terminate": len(nodes) - len(set(selected.src_chunk.astype(int))),
        "dummy_restart": len(nodes) - len(set(selected.dst_chunk.astype(int))),
        "affected_chunks": len(affected_chunks),
        "affected_detection_rows": affected_rows,
        "affected_detection_row_rate": affected_rows / max(int(rows.sum()), 1),
    }


def build_capacity_sequence(seq: str, root: Path, skip_trackeval: bool = False) -> dict[str, Any]:
    verify_preregistration(root)
    freeze, boundaries, membership = verify_boundary_freeze(seq, root)
    output_root = root / "capacity" / seq
    if (output_root / "report.json").exists():
        raise FileExistsError(f"refusing to overwrite completed capacity fold: {output_root}")
    started = time.time()
    labels, taxonomy, teacher_cuts = build_teacher_labels(seq, root)
    (
        _,
        m53,
        m53b,
        source_path,
        baseline_path,
        source_rows,
        fixed_nodes,
        chunks,
        parent_ids,
        crowd,
        mapped,
        _,
        row_embeddings,
    ) = prepare_observable_rows(seq)
    mandatory = boundaries.loc[boundaries.parent_id_transition > 0].groupby("fixed_chunk_id").position.apply(lambda values: sorted(set(map(int, values)))).to_dict()
    cuts: dict[int, list[int]] = {}
    for chunk_id in set(mandatory) | set(teacher_cuts):
        cuts[int(chunk_id)] = sorted(set(mandatory.get(chunk_id, [])) | set(teacher_cuts.get(chunk_id, [])))
    nodes, prototypes = m53b.build_adaptive_nodes(
        source_rows=source_rows,
        fixed_nodes=fixed_nodes,
        chunk_rows=chunks,
        selected_boundaries=cuts,
        row_embeddings=row_embeddings,
        mapped=mapped,
        parent_ids=parent_ids,
        crowd_density=crowd,
    )
    split_graph_root = output_root / "oracle_split_graph_root"
    split_graph = split_graph_root / seq
    split_graph.mkdir(parents=True, exist_ok=True)
    nodes_path = split_graph / "microtracklets.parquet"
    prototype_path = split_graph / "prototypes.f16.npy"
    nodes.to_parquet(nodes_path, index=False)
    np.save(prototype_path, prototypes.astype(np.float16))
    legacy_edges, parent_edges = m53b.build_candidate_edges(nodes=nodes, prototypes=prototypes, max_gap=600, appearance_bank_k=32, motion_bank_k=8)
    legacy_edges = add_legacy_columns(legacy_edges)
    parent_edges = legacy_edges[legacy_edges.parent_edge > 0].copy()
    reconstructed = output_root / "baseline_reconstruction" / "track_results" / f"{seq}.txt"
    baseline_report = m53.write_tracker(seq, source_path, nodes, parent_edges, reconstructed, preserve_parent_ids=True)
    baseline_exact = reconstructed.read_bytes() == baseline_path.read_bytes()
    if not baseline_exact:
        raise RuntimeError(f"{seq}: oracle split parent graph does not reproduce M23-46")
    m55 = load_module(f"m23_57_m55_{seq[-2:]}", "scripts/m23_research/m23_55_stratified_gap_candidate_expansion.py")
    descriptor, descriptor_report = m55.build_descriptors(seq, nodes, SOURCE_PARENT, split_graph_root, m53, load_module(f"m23_57_m10_desc_{seq[-2:]}", "scripts/m23_research/m23_10_build_micrograph.py"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outgoing, outgoing_report = m55.generate_direction_pool(nodes, descriptor, "out", 256, 256, device)
    incoming, incoming_report = m55.generate_direction_pool(nodes, descriptor, "in", 256, 256, device)
    out_flow = outgoing[outgoing["rank"] <= 32][["src_chunk", "dst_chunk", "gap_bucket", "rank"]].rename(columns={"rank": "out_rank"})
    in_flow = incoming[incoming["rank"] <= 32][["src_chunk", "dst_chunk", "gap_bucket", "rank"]].rename(columns={"rank": "in_rank"})
    flow_pairs = out_flow.merge(in_flow, on=["src_chunk", "dst_chunk", "gap_bucket"], how="outer")
    flow_pairs["out_rank"] = flow_pairs.out_rank.fillna(np.inf)
    flow_pairs["in_rank"] = flow_pairs.in_rank.fillna(np.inf)
    flow_pairs.drop_duplicates(["src_chunk", "dst_chunk"], keep="first", inplace=True)
    expanded = m55.vectorized_edge_features(nodes, flow_pairs, descriptor)
    edges = pd.concat([legacy_edges, expanded], ignore_index=True, sort=False)
    edges.sort_values(["parent_edge", "src_chunk", "dst_chunk"], ascending=[False, True, True], kind="mergesort", inplace=True)
    edges.drop_duplicates(["src_chunk", "dst_chunk"], keep="first", inplace=True)
    edges.sort_values(["src_chunk", "dst_chunk"], kind="mergesort", inplace=True)
    edges.reset_index(drop=True, inplace=True)
    forbidden_edges = audit_columns(edges.columns)
    if forbidden_edges:
        raise RuntimeError(f"unexpected forbidden pre-teacher edge fields: {forbidden_edges}")
    frozen = output_root / "frozen_candidate_graph"
    frozen.mkdir(parents=True, exist_ok=True)
    frozen_nodes = frozen / "nodes.parquet"
    frozen_edges = frozen / "edges.parquet"
    nodes.to_parquet(frozen_nodes, index=False)
    edges.to_parquet(frozen_edges, index=False)
    descriptor_path = frozen / "observable_descriptors.npz"
    np.savez_compressed(descriptor_path, **{name: value.astype(np.float16) for name, value in descriptor.items()})
    outgoing_path = frozen / "outgoing_ranking_pool.parquet"
    incoming_path = frozen / "incoming_ranking_pool.parquet"
    outgoing.to_parquet(outgoing_path, index=False)
    incoming.to_parquet(incoming_path, index=False)
    graph_manifest = {
        "experiment": "M23-57 oracle-split M23-55-style candidate graph",
        "seq": seq,
        "teacher_only": True,
        "deployable": False,
        "boundary_universe_manifest_sha256": sha256_file(root / "boundary_universe" / seq / "freeze_manifest.json"),
        "teacher_labels_sha256": taxonomy["artifacts"]["labels_sha256"],
        "oracle_change_points": taxonomy["change_points"],
        "mandatory_parent_transition_boundaries": int(boundaries.parent_id_transition.sum()),
        "baseline_reconstruction": {**baseline_report, "byte_exact": baseline_exact, "cached_sha256": sha256_file(baseline_path), "reconstructed_sha256": sha256_file(reconstructed)},
        "protocol": {
            "legacy_max_gap": 600,
            "legacy_appearance_bank_k": 32,
            "legacy_motion_bank_k": 8,
            "m23_55_ranking_k": 256,
            "m23_55_flow_k": 32,
            "gap_buckets": [list(value) for value in m55.GAP_BUCKETS],
            "teacher_objective": "unchanged M23-53/M23-55 global path cover",
        },
        "descriptor_report": descriptor_report,
        "ranking_reports": {"outgoing": outgoing_report, "incoming": incoming_report},
        "frozen_artifacts": {
            "nodes": str(frozen_nodes),
            "nodes_sha256": sha256_file(frozen_nodes),
            "node_rows": len(nodes),
            "edges": str(frozen_edges),
            "edges_sha256": sha256_file(frozen_edges),
            "edge_rows": len(edges),
            "descriptors": str(descriptor_path),
            "descriptors_sha256": sha256_file(descriptor_path),
            "outgoing_pool": str(outgoing_path),
            "outgoing_pool_sha256": sha256_file(outgoing_path),
            "incoming_pool": str(incoming_path),
            "incoming_pool_sha256": sha256_file(incoming_path),
            "parent_edges": int(edges.parent_edge.sum()),
            "legacy_edges": int((edges.candidate_origin != "m23_55_stratified_nonzero_gap").sum()),
            "stratified_edges": int((edges.candidate_origin == "m23_55_stratified_nonzero_gap").sum()),
            "forbidden_columns": forbidden_edges,
        },
    }
    graph_manifest_path = frozen / "freeze_manifest.json"
    json_write(graph_manifest_path, graph_manifest)
    nodes, selected, teacher_report = m53.build_teacher_utilities(seq=seq, source_parent_root=SOURCE_PARENT, output_root=output_root, freeze_manifest=graph_manifest)
    tracker_path = output_root / "track_results" / f"{seq}.txt"
    tracker_report = m53.write_tracker(seq, source_path, nodes, selected, tracker_path)
    tracker_report["sha256"] = sha256_file(tracker_path)
    payload_unchanged = detector_payload(source_path) == detector_payload(tracker_path)
    if not payload_unchanged:
        raise RuntimeError(f"{seq}: detection payload changed")
    official = None
    if not skip_trackeval:
        official = m53.run_official_trackeval(seq=seq, output_root=output_root, tracker_name=f"m23_57_{seq[-2:]}")
    canonical, events = canonical_summary(root, seq, nodes, edges, selected)
    events_path = output_root / "postfreeze_audit" / "successor_events.parquet"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    events.to_parquet(events_path, index=False)
    actions = flow_actions(nodes, edges, selected)
    old_report = json.loads((M55_ROOTS[seq] / "report.json").read_text(encoding="utf-8"))
    report = {
        "experiment": "M23-57 oracle internal split plus M23-55-style flow capacity",
        "seq": seq,
        "status": "completed",
        "teacher_only": True,
        "deployable": False,
        "equivalent_to_m23_53b": False,
        "boundary_freeze_verified": True,
        "taxonomy": taxonomy,
        "node_counts": {
            "fixed_chunk30_nodes": len(fixed_nodes),
            "oracle_split_nodes": len(nodes),
            "added_nodes": len(nodes) - len(fixed_nodes),
            "impure_fixed_nodes": taxonomy["impure_nodes"],
        },
        "split_actions": {
            "teacher_change_points": taxonomy["change_points"],
            "mandatory_parent_transition_boundaries": int(boundaries.parent_id_transition.sum()),
            "chunks_with_teacher_split": len(teacher_cuts),
            "teacher_split_affected_detection_rows": taxonomy["teacher_split_affected_detection_rows"],
        },
        "candidate_graph": {
            "nodes": len(nodes),
            "edges": len(edges),
            "parent_edges": int(edges.parent_edge.sum()),
            "cross_edges": int((edges.parent_edge == 0).sum()),
            "manifest": str(graph_manifest_path),
            "manifest_sha256": sha256_file(graph_manifest_path),
        },
        "canonical_flow": canonical,
        "flow_actions": actions,
        "teacher": teacher_report,
        "integrity": {
            "baseline_m23_46_byte_exact": baseline_exact,
            "detection_payload_unchanged": payload_unchanged,
            "one_to_one": teacher_report["one_to_one"],
            "time_forward": teacher_report["time_forward"],
            "acyclic": teacher_report["acyclic"],
        },
        "official_trackeval": official,
        "delta_vs_m23_55": ({key: official[key] - old_report["official_trackeval"][key] for key in ("HOTA", "DetA", "AssA")} | {"IDSW": official["IDSW"] - old_report["official_trackeval"]["IDSW"]}) if official else None,
        "tracker": tracker_report,
        "artifacts": {
            "boundary_labels_sha256": taxonomy["artifacts"]["labels_sha256"],
            "nodes_sha256": sha256_file(frozen_nodes),
            "candidate_edges_sha256": sha256_file(frozen_edges),
            "tracker_sha256": sha256_file(tracker_path),
            "successor_events": str(events_path),
            "successor_events_sha256": sha256_file(events_path),
        },
        "runtime_seconds": time.time() - started,
        "peak_rss_mb": peak_rss_mb(),
        "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated() / 1024**2) if torch.cuda.is_available() else 0.0,
        "commands": {
            "freeze": f"python {Path(__file__).as_posix()} freeze-boundaries --seq {seq}",
            "capacity": f"python {Path(__file__).as_posix()} capacity --seq {seq}",
        },
    }
    json_write(output_root / "report.json", report)
    append_event(root, {"event": "capacity_fold_completed", "seq": seq, "tracker_sha256": tracker_report["sha256"], "HOTA": official["HOTA"] if official else None})
    return report


def combine_capacity(root: Path) -> dict[str, Any]:
    verify_preregistration(root)
    combined_root = root / "capacity_combined"
    report_path = combined_root / "report.json"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite combined capacity: {combined_root}")
    reports = {}
    track_results = combined_root / "track_results"
    track_results.mkdir(parents=True, exist_ok=True)
    tracker_hashes = {}
    for seq in SEQUENCES:
        path = root / "capacity" / seq / "report.json"
        if not path.exists():
            raise FileNotFoundError(path)
        report = json.loads(path.read_text(encoding="utf-8"))
        reports[seq] = report
        source = root / "capacity" / seq / "track_results" / f"{seq}.txt"
        destination = track_results / f"{seq}.txt"
        shutil.copy2(source, destination)
        tracker_hashes[f"{seq}.txt"] = sha256_file(destination)
    evaluator = load_module("m23_57_combined_eval", "scripts/m23_research/m23_45_domain_robust_source_cut_student.py")
    metrics = evaluator.evaluate_detailed(track_results, combined_root / "official_eval", "m23_57_intra_node_change_point_capacity", SEQUENCES)
    previous = json.loads(M55_COMBINED.read_text(encoding="utf-8"))["official_trackeval"]
    combined = metrics["COMBINED"]
    decision = "D_or_C_capacity_pass_run_observability" if combined["HOTA"] >= 80.7 else "B_capacity_below_80_7_close_without_observability"
    payload = {
        "experiment": "M23-57 Optional Intra-node Change-point Capacity COMBINED",
        "status": "completed",
        "teacher_only": True,
        "deployable": False,
        "equivalent_to_m23_53b": False,
        "official_trackeval": metrics,
        "delta_vs_m23_55": {
            seq: {
                "HOTA": metrics[seq]["HOTA"] - previous[seq]["HOTA"],
                "DetA": metrics[seq]["DetA"] - previous[seq]["DetA"],
                "AssA": metrics[seq]["AssA"] - previous[seq]["AssA"],
                "IDSW": metrics[seq]["IDSW"] - previous[seq]["IDSW"],
            }
            for seq in (*SEQUENCES, "COMBINED")
        },
        "capacity_gate": {
            "threshold_HOTA": 80.7,
            "pass": combined["HOTA"] >= 80.7,
            "margin": combined["HOTA"] - 80.7,
        },
        "decision": decision,
        "trackeval_run_count": {"fold": 4, "combined": 1},
        "tracker_sha256": tracker_hashes,
        "fold_reports_sha256": {seq: sha256_file(root / "capacity" / seq / "report.json") for seq in SEQUENCES},
        "observability_unlocked": combined["HOTA"] >= 80.7,
        "m23_54_started": False,
        "m23_58_started": False,
        "mot20_test_submission": False,
    }
    json_write(report_path, payload)
    append_event(root, {"event": "combined_capacity_completed", "HOTA": combined["HOTA"], "capacity_gate_pass": payload["capacity_gate"]["pass"], "decision": decision})
    return payload


def freeze_implementation(root: Path) -> dict[str, Any]:
    verify_preregistration(root)
    output = root / "implementation_manifest.json"
    if output.exists():
        raise FileExistsError(output)
    capacity_script = Path("scripts/m23_research/m23_57_intra_node_change_point_capacity.py")
    observability_script = Path("scripts/m23_research/m23_57_change_point_observability_audit.py")
    for path in (capacity_script, observability_script):
        if not path.exists():
            raise FileNotFoundError(path)
    environment = {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cpu_count": os.cpu_count(),
    }
    payload = {
        "experiment": "M23-57 implementation freeze",
        "git_head": git_head(),
        "preregistration_sha256": PREREG_SHA256,
        "capacity_script": str(capacity_script),
        "capacity_script_sha256": sha256_file(capacity_script),
        "observability_script": str(observability_script),
        "observability_script_sha256": sha256_file(observability_script),
        "environment": environment,
        "outer_gt_opened": False,
        "teacher_labels_opened": False,
    }
    json_write(output, payload)
    append_event(root, {"event": "implementation_frozen", "manifest_sha256": sha256_file(output), "teacher_labels_opened": False})
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze-implementation")
    freeze = sub.add_parser("freeze-boundaries")
    freeze.add_argument("--seq", required=True, choices=SEQUENCES)
    capacity = sub.add_parser("capacity")
    capacity.add_argument("--seq", required=True, choices=SEQUENCES)
    capacity.add_argument("--skip-trackeval", action="store_true")
    sub.add_parser("combine")
    args = parser.parse_args()
    if args.command == "freeze-implementation":
        result = freeze_implementation(args.root)
    elif args.command == "freeze-boundaries":
        result = freeze_boundaries(args.seq, args.root)
    elif args.command == "capacity":
        result = build_capacity_sequence(args.seq, args.root, args.skip_trackeval)
    elif args.command == "combine":
        result = combine_capacity(args.root)
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
