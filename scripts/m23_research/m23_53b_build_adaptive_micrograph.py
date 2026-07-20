#!/usr/bin/env python3
from __future__ import annotations

"""Build the GT-free adaptive microtracklet graph for M23-53B.

The builder preserves every fixed chunk-30 boundary and adds at most one
sequence-normalized, signal-driven boundary inside each fixed chunk. Signals are
computed only from frozen tracker rows, phase-0 ReID embeddings, motion, box
scale/confidence, ReID recovery, and local crowd density. The strict M23-46
tracker is used only as a frozen GT-free parent state so the adaptive graph can
reconstruct it byte-exactly before any teacher GT is opened by M23-53.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


REPO = Path(__file__).resolve().parents[2]
SEQUENCES = ("MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05")
SOURCE_PARENT = Path(
    "outputs/assocriskbench_p15_20260714/"
    "fused_a42_adaptive_risk_eval/l1_r2_q75_adaptive_priority/track_results"
)
FIXED_GRAPH_ROOT = Path("outputs/mot20_m23_20260718/micrograph_chunk30_v1")
M23_46_CACHE = Path(
    "outputs/mot20_m23_20260718/m23_47_m23_46_deployable_baseline_cache_v1"
)
DEFAULT_GRAPH_OUT = Path(
    "outputs/mot20_m23_20260718/m23_53b_adaptive_micrograph_v1"
)
DEFAULT_CACHE_OUT = Path(
    "outputs/mot20_m23_20260718/m23_53b_m23_46_adaptive_baseline_cache_v1"
)
PROJECTION_DIM = 128
PROJECTION_SEED = 2353
GRID_SIZE = 128.0


def load_module(name: str, relative_path: str):
    path = REPO / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def read_source_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            fields = line.rstrip("\n").split(",")
            if len(fields) < 6:
                continue
            frame = int(float(fields[0]))
            track_id = int(float(fields[1]))
            x, y, width, height = map(float, fields[2:6])
            score = float(fields[6]) if len(fields) > 6 else 1.0
            rows.append(
                {
                    "line": line_index,
                    "frame": frame,
                    "track_id": track_id,
                    "x1": x,
                    "y1": y,
                    "x2": x + width,
                    "y2": y + height,
                    "cx": x + 0.5 * width,
                    "cy": y + 0.5 * height,
                    "height": height,
                    "score": score,
                    "fields": fields,
                }
            )
    return rows


def align_parent_tracker_ids(source_rows: list[dict], parent_path: Path) -> np.ndarray:
    parent_by_frame: dict[int, list[tuple[np.ndarray, int]]] = defaultdict(list)
    with parent_path.open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split(",")
            if len(fields) < 6:
                continue
            frame = int(float(fields[0]))
            box = np.asarray([float(value) for value in fields[2:6]], dtype=float)
            parent_by_frame[frame].append((box, int(float(fields[1]))))
    source_by_frame: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(source_rows):
        source_by_frame[int(row["frame"])].append(index)
    output = np.full(len(source_rows), -1, np.int64)
    for frame, indices in source_by_frame.items():
        parent = parent_by_frame.get(frame, [])
        if len(parent) != len(indices):
            raise RuntimeError(
                f"frame {frame}: source/parent row mismatch {len(indices)} != {len(parent)}"
            )
        source_boxes = np.asarray(
            [
                [
                    source_rows[index]["x1"],
                    source_rows[index]["y1"],
                    source_rows[index]["x2"] - source_rows[index]["x1"],
                    source_rows[index]["y2"] - source_rows[index]["y1"],
                ]
                for index in indices
            ],
            dtype=float,
        )
        parent_boxes = np.asarray([item[0] for item in parent], dtype=float)
        cost = np.max(
            np.abs(source_boxes[:, np.newaxis, :] - parent_boxes[np.newaxis, :, :]),
            axis=2,
        )
        row_index, col_index = linear_sum_assignment(cost)
        if len(row_index) != len(indices) or np.max(cost[row_index, col_index]) > 1e-4:
            raise RuntimeError(f"frame {frame}: parent tracker box alignment failed")
        for local_source, local_parent in zip(row_index, col_index):
            output[indices[int(local_source)]] = int(parent[int(local_parent)][1])
    if np.any(output < 0):
        raise RuntimeError("parent tracker ID alignment is incomplete")
    return output


def local_crowd_density(rows: list[dict]) -> np.ndarray:
    by_frame: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_frame[int(row["frame"])].append(index)
    density = np.zeros(len(rows), np.float32)
    for indices in by_frame.values():
        cells: dict[tuple[int, int], int] = defaultdict(int)
        row_cells: list[tuple[int, int]] = []
        for index in indices:
            row = rows[index]
            cell = (
                int(math.floor(float(row["cx"]) / GRID_SIZE)),
                int(math.floor(float(row["cy"]) / GRID_SIZE)),
            )
            row_cells.append(cell)
            cells[cell] += 1
        for index, cell in zip(indices, row_cells):
            count = 0
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    count += cells.get((cell[0] + dx, cell[1] + dy), 0)
            density[index] = max(0, count - 1)
    return density


def robust_positive_z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    median = float(np.median(values)) if len(values) else 0.0
    mad = float(np.median(np.abs(values - median))) if len(values) else 0.0
    scale = 1.4826 * mad
    if scale < 1e-8:
        q75 = float(np.quantile(values, 0.75)) if len(values) else 0.0
        scale = max(q75 - median, 1e-6)
    return np.maximum(0.0, (values - median) / scale)


def slope(frames: list[int], values: list[float]) -> float:
    if len(frames) < 2 or frames[-1] == frames[0]:
        return 0.0
    x = np.asarray(frames, dtype=float)
    y = np.asarray(values, dtype=float)
    x -= x.mean()
    return float((x @ y) / max(float(x @ x), 1e-12))


def fixed_chunk_rows(
    source_rows: list[dict], fixed_nodes: pd.DataFrame, m53
) -> dict[int, list[int]]:
    compatible_rows = [
        {
            "frame": int(row["frame"]),
            "track_id": int(row["track_id"]),
            "fields": row["fields"],
        }
        for row in source_rows
    ]
    mapping = m53.line_chunks(compatible_rows, fixed_nodes)
    output: dict[int, list[int]] = defaultdict(list)
    for row_index, chunk_id in mapping.items():
        output[int(chunk_id)].append(int(row_index))
    for indices in output.values():
        indices.sort(key=lambda index: (source_rows[index]["frame"], index))
    return output


def build_boundary_table(
    *,
    source_rows: list[dict],
    chunk_rows: dict[int, list[int]],
    row_embeddings: np.ndarray,
    mapped: np.ndarray,
    crowd_density: np.ndarray,
    parent_ids: np.ndarray,
    min_segment_rows: int,
    cut_quantile: float,
) -> tuple[pd.DataFrame, dict[int, list[int]], dict]:
    records: list[dict] = []
    for fixed_chunk_id, indices in sorted(chunk_rows.items()):
        count = len(indices)
        if count < 2 * min_segment_rows:
            continue
        for position in range(min_segment_rows, count - min_segment_rows + 1):
            left = indices[position - 1]
            right = indices[position]
            left_row = source_rows[left]
            right_row = source_rows[right]
            appearance_jump = 0.0
            if mapped[left] and mapped[right]:
                appearance_jump = float(
                    max(
                        0.0,
                        1.0
                        - float(
                            row_embeddings[left].astype(float)
                            @ row_embeddings[right].astype(float)
                        ),
                    )
                )
            motion_error = 0.0
            if position >= 2:
                previous = indices[position - 2]
                previous_row = source_rows[previous]
                dt_history = max(
                    1, int(left_row["frame"]) - int(previous_row["frame"])
                )
                vx = (float(left_row["cx"]) - float(previous_row["cx"])) / dt_history
                vy = (float(left_row["cy"]) - float(previous_row["cy"])) / dt_history
                dt = max(1, int(right_row["frame"]) - int(left_row["frame"]))
                predicted_x = float(left_row["cx"]) + vx * dt
                predicted_y = float(left_row["cy"]) + vy * dt
                normalization = max(
                    1.0, 0.5 * (float(left_row["height"]) + float(right_row["height"]))
                )
                motion_error = float(
                    math.hypot(
                        float(right_row["cx"]) - predicted_x,
                        float(right_row["cy"]) - predicted_y,
                    )
                    / normalization
                )
            crowd_jump = float(
                abs(
                    math.log1p(float(crowd_density[right]))
                    - math.log1p(float(crowd_density[left]))
                )
            )
            scale_jump = float(
                abs(
                    math.log(
                        max(float(right_row["height"]), 1e-3)
                        / max(float(left_row["height"]), 1e-3)
                    )
                )
            )
            confidence_jump = float(
                abs(float(right_row["score"]) - float(left_row["score"]))
            )
            reid_recovery = int(mapped[right] and not mapped[left])
            parent_transition = int(parent_ids[left] != parent_ids[right])
            records.append(
                {
                    "fixed_chunk_id": int(fixed_chunk_id),
                    "position": int(position),
                    "left_row": int(left),
                    "right_row": int(right),
                    "left_frame": int(left_row["frame"]),
                    "right_frame": int(right_row["frame"]),
                    "appearance_jump": appearance_jump,
                    "motion_error": motion_error,
                    "crowd_jump": crowd_jump,
                    "scale_jump": scale_jump,
                    "confidence_jump": confidence_jump,
                    "reid_recovery": reid_recovery,
                    "parent_transition": parent_transition,
                }
            )
    table = pd.DataFrame(records)
    selected: dict[int, list[int]] = defaultdict(list)
    if table.empty:
        return table, selected, {
            "eligible_boundaries": 0,
            "selected_boundaries": 0,
            "score_threshold": None,
        }
    signal_columns = (
        "appearance_jump",
        "motion_error",
        "crowd_jump",
        "scale_jump",
        "confidence_jump",
    )
    for column in signal_columns:
        table[f"{column}_z"] = robust_positive_z(table[column].to_numpy(float))
    z_columns = [f"{column}_z" for column in signal_columns]
    z = table[z_columns].to_numpy(float)
    ordered = np.sort(z, axis=1)
    primary = ordered[:, -1]
    secondary = ordered[:, -2] if z.shape[1] >= 2 else np.zeros(len(z))
    table["primary_signal_z"] = primary
    table["secondary_signal_z"] = secondary
    table["change_score"] = (
        primary
        + 0.35 * secondary
        + 1.5 * table.reid_recovery.to_numpy(float)
        + 4.0 * table.parent_transition.to_numpy(float)
    )
    threshold = float(np.quantile(table.change_score.to_numpy(float), cut_quantile))
    table["selected"] = np.int8(0)
    for fixed_chunk_id, part in table.groupby("fixed_chunk_id", sort=True):
        eligible = part[
            (part.change_score >= threshold)
            & (
                (part.primary_signal_z >= 1.0)
                | (part.reid_recovery > 0)
                | (part.parent_transition > 0)
            )
        ].copy()
        if eligible.empty:
            continue
        eligible.sort_values(
            ["parent_transition", "change_score", "position"],
            ascending=[False, False, True],
            kind="mergesort",
            inplace=True,
        )
        chosen = int(eligible.iloc[0].position)
        selected[int(fixed_chunk_id)].append(chosen)
        table.loc[eligible.index[0], "selected"] = np.int8(1)
    diagnostics = {
        "eligible_boundaries": int(len(table)),
        "selected_boundaries": int(table.selected.sum()),
        "score_threshold": threshold,
        "cut_quantile": cut_quantile,
        "min_segment_rows": min_segment_rows,
        "maximum_adaptive_cuts_per_fixed_chunk": 1,
        "selected_parent_transitions": int(
            table.loc[table.selected > 0, "parent_transition"].sum()
        ),
        "selected_reid_recoveries": int(
            table.loc[table.selected > 0, "reid_recovery"].sum()
        ),
    }
    return table, selected, diagnostics


def build_adaptive_nodes(
    *,
    source_rows: list[dict],
    fixed_nodes: pd.DataFrame,
    chunk_rows: dict[int, list[int]],
    selected_boundaries: dict[int, list[int]],
    row_embeddings: np.ndarray,
    mapped: np.ndarray,
    parent_ids: np.ndarray,
    crowd_density: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    fixed_by_track: dict[int, list] = defaultdict(list)
    for node in fixed_nodes.itertuples(index=False):
        fixed_by_track[int(node.source_track_id)].append(node)
    node_records: list[dict] = []
    prototypes: list[np.ndarray] = []
    chunk_id = 0
    for source_track_id, track_nodes in sorted(fixed_by_track.items()):
        track_nodes.sort(key=lambda node: int(node.source_ordinal))
        source_ordinal = 0
        for fixed_node in track_nodes:
            fixed_chunk_id = int(fixed_node.chunk_id)
            indices = chunk_rows[fixed_chunk_id]
            cuts = sorted(selected_boundaries.get(fixed_chunk_id, []))
            boundaries = [0, *cuts, len(indices)]
            for part_index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
                part = indices[start:end]
                if not part:
                    raise RuntimeError("empty adaptive segment")
                identities = np.unique(parent_ids[np.asarray(part, dtype=np.int64)])
                if len(identities) != 1:
                    raise RuntimeError(
                        f"fixed chunk {fixed_chunk_id}: adaptive segment crosses M23-46 IDs"
                    )
                mapped_part = [index for index in part if mapped[index]]
                prototype = np.zeros(PROJECTION_DIM, np.float32)
                consistency = 0.0
                if mapped_part:
                    embeddings = row_embeddings[np.asarray(mapped_part, dtype=np.int64)]
                    prototype = embeddings.mean(axis=0)
                    prototype /= max(float(np.linalg.norm(prototype)), 1e-12)
                    consistency = float(np.mean(embeddings @ prototype))
                first = source_rows[part[0]]
                last = source_rows[part[-1]]
                head = part[: min(5, len(part))]
                tail = part[max(0, len(part) - 5) :]
                head_frames = [int(source_rows[index]["frame"]) for index in head]
                tail_frames = [int(source_rows[index]["frame"]) for index in tail]
                head_cx = [float(source_rows[index]["cx"]) for index in head]
                head_cy = [float(source_rows[index]["cy"]) for index in head]
                tail_cx = [float(source_rows[index]["cx"]) for index in tail]
                tail_cy = [float(source_rows[index]["cy"]) for index in tail]
                node_records.append(
                    {
                        "chunk_id": chunk_id,
                        "source_track_id": int(source_track_id),
                        "source_ordinal": source_ordinal,
                        "first_frame": int(first["frame"]),
                        "last_frame": int(last["frame"]),
                        "span_frames": int(last["frame"] - first["frame"] + 1),
                        "rows": len(part),
                        "first_line": int(part[0]),
                        "last_line": int(part[-1]),
                        "appearance_consistency": consistency,
                        "first_cx": float(first["cx"]),
                        "first_cy": float(first["cy"]),
                        "last_cx": float(last["cx"]),
                        "last_cy": float(last["cy"]),
                        "first_h": float(first["height"]),
                        "last_h": float(last["height"]),
                        "start_vx": slope(head_frames, head_cx),
                        "start_vy": slope(head_frames, head_cy),
                        "end_vx": slope(tail_frames, tail_cx),
                        "end_vy": slope(tail_frames, tail_cy),
                        "parent_tracker_id": int(identities[0]),
                        "fixed_chunk_id": fixed_chunk_id,
                        "fixed_subsegment": part_index,
                        "mapped_rows_gt_free": len(mapped_part),
                        "mean_crowd_density": float(
                            np.mean(crowd_density[np.asarray(part, dtype=np.int64)])
                        ),
                    }
                )
                prototypes.append(prototype)
                chunk_id += 1
                source_ordinal += 1
    nodes = pd.DataFrame(node_records)
    if not np.array_equal(nodes.chunk_id.to_numpy(int), np.arange(len(nodes))):
        raise RuntimeError("adaptive chunk IDs are not dense")
    return nodes, np.asarray(prototypes, dtype=np.float32)


def edge_arrays(nodes: pd.DataFrame, prototypes: np.ndarray, keys: np.ndarray) -> pd.DataFrame:
    count = len(nodes)
    src = (keys // count).astype(np.int32)
    dst = (keys % count).astype(np.int32)
    first_frame = nodes.first_frame.to_numpy(np.int32)
    last_frame = nodes.last_frame.to_numpy(np.int32)
    first_cx = nodes.first_cx.to_numpy(np.float32)
    first_cy = nodes.first_cy.to_numpy(np.float32)
    last_cx = nodes.last_cx.to_numpy(np.float32)
    last_cy = nodes.last_cy.to_numpy(np.float32)
    first_h = nodes.first_h.to_numpy(np.float32)
    last_h = nodes.last_h.to_numpy(np.float32)
    start_vx = nodes.start_vx.to_numpy(np.float32)
    start_vy = nodes.start_vy.to_numpy(np.float32)
    end_vx = nodes.end_vx.to_numpy(np.float32)
    end_vy = nodes.end_vy.to_numpy(np.float32)
    appearance = np.empty(len(keys), np.float32)
    for start in range(0, len(keys), 200_000):
        stop = min(len(keys), start + 200_000)
        appearance[start:stop] = np.einsum(
            "ij,ij->i", prototypes[src[start:stop]], prototypes[dst[start:stop]]
        )
    gap = first_frame[dst] - last_frame[src] - 1
    dt = np.maximum(first_frame[dst] - last_frame[src], 1).astype(np.float32)
    normalization = np.maximum(0.5 * (last_h[src] + first_h[dst]), 1.0)
    predicted_x = last_cx[src] + end_vx[src] * dt
    predicted_y = last_cy[src] + end_vy[src] * dt
    backward_x = first_cx[dst] - start_vx[dst] * dt
    backward_y = first_cy[dst] - start_vy[dst] * dt
    forward_error = np.hypot(first_cx[dst] - predicted_x, first_cy[dst] - predicted_y) / normalization
    backward_error = np.hypot(last_cx[src] - backward_x, last_cy[src] - backward_y) / normalization
    endpoint_displacement = np.hypot(first_cx[dst] - last_cx[src], first_cy[dst] - last_cy[src]) / normalization
    velocity_dot = end_vx[src] * start_vx[dst] + end_vy[src] * start_vy[dst]
    velocity_norm = np.maximum(
        np.hypot(end_vx[src], end_vy[src])
        * np.hypot(start_vx[dst], start_vy[dst]),
        1e-8,
    )
    source_track = nodes.source_track_id.to_numpy(np.int64)
    source_ordinal = nodes.source_ordinal.to_numpy(np.int32)
    same_source = source_track[src] == source_track[dst]
    source_adjacent = same_source & (source_ordinal[dst] == source_ordinal[src] + 1)
    data = pd.DataFrame(
        {
            "src_chunk": src,
            "dst_chunk": dst,
            "src_track": source_track[src],
            "dst_track": source_track[dst],
            "gap": gap.astype(np.int32),
            "log_gap": np.log1p(np.maximum(gap, 0)).astype(np.float32),
            "appearance_cos": appearance,
            "same_source": same_source.astype(np.int8),
            "source_adjacent": source_adjacent.astype(np.int8),
            "forward_motion_error": forward_error.astype(np.float32),
            "backward_motion_error": backward_error.astype(np.float32),
            "motion_error_min": np.minimum(forward_error, backward_error).astype(np.float32),
            "motion_error_mean": (0.5 * (forward_error + backward_error)).astype(np.float32),
            "endpoint_displacement": endpoint_displacement.astype(np.float32),
            "velocity_cos": (velocity_dot / velocity_norm).astype(np.float32),
            "log_height_ratio": np.log(
                np.maximum(first_h[dst], 1e-3) / np.maximum(last_h[src], 1e-3)
            ).astype(np.float32),
            "src_rows": nodes.rows.to_numpy(np.int32)[src],
            "dst_rows": nodes.rows.to_numpy(np.int32)[dst],
            "src_consistency": nodes.appearance_consistency.to_numpy(np.float32)[src],
            "dst_consistency": nodes.appearance_consistency.to_numpy(np.float32)[dst],
        }
    )
    data["consistency_min"] = data[["src_consistency", "dst_consistency"]].min(axis=1)
    return data


def build_candidate_edges(
    *,
    nodes: pd.DataFrame,
    prototypes: np.ndarray,
    max_gap: int,
    appearance_bank_k: int,
    motion_bank_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    count = len(nodes)
    starts = nodes.first_frame.to_numpy(np.int32)
    order = np.argsort(starts, kind="mergesort")
    sorted_starts = starts[order]
    first_cx = nodes.first_cx.to_numpy(np.float32)
    first_cy = nodes.first_cy.to_numpy(np.float32)
    first_h = nodes.first_h.to_numpy(np.float32)
    last_frame = nodes.last_frame.to_numpy(np.int32)
    last_cx = nodes.last_cx.to_numpy(np.float32)
    last_cy = nodes.last_cy.to_numpy(np.float32)
    last_h = nodes.last_h.to_numpy(np.float32)
    start_vx = nodes.start_vx.to_numpy(np.float32)
    start_vy = nodes.start_vy.to_numpy(np.float32)
    end_vx = nodes.end_vx.to_numpy(np.float32)
    end_vy = nodes.end_vy.to_numpy(np.float32)
    candidate_keys: list[np.ndarray] = []
    for src in range(count):
        low = int(np.searchsorted(sorted_starts, last_frame[src] + 1, side="left"))
        high = int(
            np.searchsorted(
                sorted_starts, last_frame[src] + max_gap + 2, side="left"
            )
        )
        destinations = order[low:high]
        if not len(destinations):
            continue
        similarities = prototypes[destinations] @ prototypes[src]
        app_k = min(appearance_bank_k, len(destinations))
        if app_k:
            app_local = np.argpartition(-similarities, app_k - 1)[:app_k]
        else:
            app_local = np.empty(0, dtype=np.int64)
        dt = np.maximum(starts[destinations] - last_frame[src], 1).astype(np.float32)
        normalization = np.maximum(0.5 * (last_h[src] + first_h[destinations]), 1.0)
        predicted_x = last_cx[src] + end_vx[src] * dt
        predicted_y = last_cy[src] + end_vy[src] * dt
        backward_x = first_cx[destinations] - start_vx[destinations] * dt
        backward_y = first_cy[destinations] - start_vy[destinations] * dt
        forward = np.hypot(first_cx[destinations] - predicted_x, first_cy[destinations] - predicted_y) / normalization
        backward = np.hypot(last_cx[src] - backward_x, last_cy[src] - backward_y) / normalization
        motion = np.minimum(forward, backward)
        mot_k = min(motion_bank_k, len(destinations))
        if mot_k:
            mot_local = np.argpartition(motion, mot_k - 1)[:mot_k]
        else:
            mot_local = np.empty(0, dtype=np.int64)
        chosen = np.unique(np.concatenate([app_local, mot_local]))
        selected_dst = destinations[chosen].astype(np.int64)
        candidate_keys.append(src * count + selected_dst)

    parent_pairs: list[tuple[int, int]] = []
    by_parent: dict[int, list[int]] = defaultdict(list)
    for node in nodes.itertuples(index=False):
        by_parent[int(node.parent_tracker_id)].append(int(node.chunk_id))
    for parent_id, chunk_ids in by_parent.items():
        chunk_ids.sort(
            key=lambda chunk_id: (
                int(nodes.iloc[chunk_id].first_frame),
                int(nodes.iloc[chunk_id].last_frame),
                int(chunk_id),
            )
        )
        for src, dst in zip(chunk_ids, chunk_ids[1:]):
            if int(nodes.iloc[dst].first_frame) <= int(nodes.iloc[src].last_frame):
                raise RuntimeError(f"M23-46 identity {parent_id} is not time-forward")
            parent_pairs.append((src, dst))
    parent_keys = np.asarray(
        [src * count + dst for src, dst in parent_pairs], dtype=np.int64
    )
    all_keys = np.unique(
        np.concatenate([*candidate_keys, parent_keys])
        if candidate_keys
        else parent_keys
    )
    edges = edge_arrays(nodes, prototypes, all_keys)
    parent_mask = np.isin(all_keys, parent_keys, assume_unique=False)
    edges["edge_role"] = np.where(parent_mask, "m23_46_parent", "cross")
    edges["out_rank"] = (
        edges.groupby("src_chunk")["appearance_cos"]
        .rank(method="first", ascending=False)
        .astype(np.int32)
    )
    edges["in_rank"] = (
        edges.groupby("dst_chunk")["appearance_cos"]
        .rank(method="first", ascending=False)
        .astype(np.int32)
    )
    edges["max_rank"] = edges[["out_rank", "in_rank"]].max(axis=1).astype(np.int32)
    best_out = edges.groupby("src_chunk")["appearance_cos"].transform("max")
    best_in = edges.groupby("dst_chunk")["appearance_cos"].transform("max")
    edges["out_margin"] = (best_out - edges.appearance_cos).astype(np.float32)
    edges["in_margin"] = (best_in - edges.appearance_cos).astype(np.float32)
    edges["max_margin"] = edges[["out_margin", "in_margin"]].max(axis=1).astype(np.float32)
    parent_edges = edges.loc[parent_mask].copy()
    parent_edges["utility"] = np.float32(1.0)
    if parent_edges.src_chunk.duplicated().any() or parent_edges.dst_chunk.duplicated().any():
        raise RuntimeError("adaptive M23-46 parent graph is not one-to-one")
    return edges.reset_index(drop=True), parent_edges.reset_index(drop=True)


def build_sequence(args, seq: str) -> dict:
    m10 = load_module(
        f"m23_53b_m10_{seq[-2:]}",
        "scripts/m23_research/m23_10_build_micrograph.py",
    )
    m53 = load_module(
        f"m23_53b_m53_{seq[-2:]}",
        "scripts/m23_research/m23_53_global_identity_flow_capacity.py",
    )
    source_path = Path(args.source_parent) / f"{seq}.txt"
    parent_tracker = Path(args.m23_46_cache) / seq / "track_results" / f"{seq}.txt"
    fixed_path = Path(args.fixed_graph_root) / seq / "microtracklets.parquet"
    for path in (source_path, parent_tracker, fixed_path):
        if not path.exists():
            raise FileNotFoundError(path)
    source_rows = read_source_rows(source_path)
    m10_rows = m10.read_tracker(source_path)
    if len(m10_rows) != len(source_rows):
        raise RuntimeError("source row parser mismatch")
    rng = np.random.default_rng(PROJECTION_SEED)
    projection = (
        rng.normal(size=(2048, PROJECTION_DIM)).astype(np.float32)
        / math.sqrt(PROJECTION_DIM)
    )
    m10.DIM = PROJECTION_DIM
    phase, _, unique_embeddings, positions = m10.map_phase(
        m10_rows, seq, projection
    )
    mapped = phase >= 0
    row_embeddings = np.zeros((len(source_rows), PROJECTION_DIM), np.float32)
    if np.any(mapped):
        row_embeddings[mapped] = unique_embeddings[positions[mapped]]
    parent_ids = align_parent_tracker_ids(source_rows, parent_tracker)
    crowd = local_crowd_density(source_rows)
    fixed_nodes = pd.read_parquet(
        fixed_path,
        columns=[
            "chunk_id",
            "source_track_id",
            "source_ordinal",
            "first_frame",
            "last_frame",
            "rows",
        ],
    )
    chunks = fixed_chunk_rows(source_rows, fixed_nodes, m53)
    boundaries, selected, boundary_report = build_boundary_table(
        source_rows=source_rows,
        chunk_rows=chunks,
        row_embeddings=row_embeddings,
        mapped=mapped,
        crowd_density=crowd,
        parent_ids=parent_ids,
        min_segment_rows=args.min_segment_rows,
        cut_quantile=args.cut_quantile,
    )
    nodes, prototypes = build_adaptive_nodes(
        source_rows=source_rows,
        fixed_nodes=fixed_nodes,
        chunk_rows=chunks,
        selected_boundaries=selected,
        row_embeddings=row_embeddings,
        mapped=mapped,
        parent_ids=parent_ids,
        crowd_density=crowd,
    )
    edges, parent_edges = build_candidate_edges(
        nodes=nodes,
        prototypes=prototypes,
        max_gap=args.max_gap,
        appearance_bank_k=args.appearance_bank_k,
        motion_bank_k=args.motion_bank_k,
    )

    graph_dir = Path(args.output_graph_root) / seq
    cache_dir = Path(args.output_baseline_cache) / seq
    graph_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    nodes_path = graph_dir / "microtracklets.parquet"
    proto_path = graph_dir / "prototypes.f16.npy"
    edges_path = graph_dir / "candidate_edges.parquet"
    boundary_path = graph_dir / "adaptive_boundaries.parquet"
    nodes.to_parquet(nodes_path, index=False)
    np.save(proto_path, prototypes.astype(np.float16))
    edges.to_parquet(edges_path, index=False)
    boundaries.to_parquet(boundary_path, index=False)
    parent_edges.to_parquet(cache_dir / "frozen_applied_edges.parquet", index=False)
    tracker_destination = cache_dir / "track_results" / f"{seq}.txt"
    tracker_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(parent_tracker, tracker_destination)

    # The adaptive parent graph must reproduce the exact frozen M23-46 tracker.
    reconstructed = cache_dir / "adaptive_parent_reconstruction.txt"
    compatible_source = Path(args.source_parent) / f"{seq}.txt"
    reconstruction = m53.write_tracker(
        seq,
        compatible_source,
        nodes,
        parent_edges,
        reconstructed,
        preserve_parent_ids=True,
    )
    baseline_exact = reconstructed.read_bytes() == parent_tracker.read_bytes()
    if not baseline_exact:
        raise RuntimeError(f"{seq}: adaptive parent is not byte-exact M23-46")

    report = {
        "experiment": "M23-53B GT-free adaptive microtracklet graph",
        "seq": seq,
        "status": "completed",
        "gt_used": False,
        "strict_parent": "frozen M23-46 tracker",
        "protocol": {
            "fixed_chunk30_boundaries_preserved": True,
            "adaptive_signals": [
                "phase0_reid_discontinuity",
                "motion_residual",
                "local_crowd_density_change",
                "box_scale_change",
                "confidence_change",
                "reid_recovery",
            ],
            "projection_dim": PROJECTION_DIM,
            "projection_seed": PROJECTION_SEED,
            "min_segment_rows": args.min_segment_rows,
            "cut_quantile": args.cut_quantile,
            "max_adaptive_cuts_per_fixed_chunk": 1,
            "max_gap": args.max_gap,
            "appearance_bank_k": args.appearance_bank_k,
            "motion_bank_k": args.motion_bank_k,
        },
        "rows": len(source_rows),
        "mapped_reid_rows": int(mapped.sum()),
        "fixed_chunks": len(fixed_nodes),
        "adaptive_chunks": len(nodes),
        "added_boundaries": len(nodes) - len(fixed_nodes),
        "candidate_edges": len(edges),
        "parent_edges": len(parent_edges),
        "boundary_report": boundary_report,
        "baseline_reconstruction": {
            **reconstruction,
            "byte_exact": baseline_exact,
            "cached_sha256": sha256_file(parent_tracker),
            "reconstructed_sha256": sha256_file(reconstructed),
        },
        "artifacts": {
            "nodes": str(nodes_path),
            "nodes_sha256": sha256_file(nodes_path),
            "prototypes": str(proto_path),
            "prototypes_sha256": sha256_file(proto_path),
            "candidate_edges": str(edges_path),
            "candidate_edges_sha256": sha256_file(edges_path),
            "adaptive_boundaries": str(boundary_path),
            "adaptive_boundaries_sha256": sha256_file(boundary_path),
            "parent_applied": str(cache_dir / "frozen_applied_edges.parquet"),
            "parent_applied_sha256": sha256_file(
                cache_dir / "frozen_applied_edges.parquet"
            ),
            "parent_tracker": str(tracker_destination),
            "parent_tracker_sha256": sha256_file(tracker_destination),
        },
        "frozen_at": utc_now(),
    }
    write_json(graph_dir / "freeze_manifest.json", report)
    print(json.dumps(report, sort_keys=True), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", action="append", choices=SEQUENCES)
    parser.add_argument("--source-parent", default=str(SOURCE_PARENT))
    parser.add_argument("--fixed-graph-root", default=str(FIXED_GRAPH_ROOT))
    parser.add_argument("--m23-46-cache", default=str(M23_46_CACHE))
    parser.add_argument("--output-graph-root", default=str(DEFAULT_GRAPH_OUT))
    parser.add_argument("--output-baseline-cache", default=str(DEFAULT_CACHE_OUT))
    parser.add_argument("--min-segment-rows", type=int, default=5)
    parser.add_argument("--cut-quantile", type=float, default=0.75)
    parser.add_argument("--max-gap", type=int, default=600)
    parser.add_argument("--appearance-bank-k", type=int, default=32)
    parser.add_argument("--motion-bank-k", type=int, default=8)
    args = parser.parse_args()
    if not 0.0 < args.cut_quantile < 1.0:
        raise ValueError("cut quantile must be in (0, 1)")
    if args.min_segment_rows < 2:
        raise ValueError("min segment rows must be at least 2")
    sequences = args.seq or list(SEQUENCES)
    reports = [build_sequence(args, seq) for seq in sequences]
    root_report = {
        "experiment": "M23-53B GT-free adaptive microtracklet graph",
        "status": "completed",
        "gt_used": False,
        "sequences": reports,
    }
    write_json(Path(args.output_graph_root) / "report.json", root_report)


if __name__ == "__main__":
    main()
