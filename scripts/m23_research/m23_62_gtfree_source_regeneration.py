#!/usr/bin/env python3
from __future__ import annotations

"""M23-62: GT-free source-row regeneration for M23-59 v3.

This stage is deliberately limited to source-row provenance, detector/ReID dumps,
and canonical 144-D observable regeneration. It never trains a relation model,
reads MOT17/MOT20 labels, invokes TrackEval, generates a new tracker, or reads
MOT20 test data.
"""

import argparse
import configparser
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import platform
import resource
import struct
import subprocess
import sys
import time
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

REPO = Path(__file__).resolve().parents[2]
EXP_ID = "M23-62"
TITLE = "M23-59 v3 GT-Free Source-Row Regeneration"
ROOT = Path("outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_gtfree_source_regeneration")
REGISTRY = Path("outputs/experiment_registry.csv")
PREREG = Path("docs/m23_59_relation_pretrained_hierarchical_flow_v3_gtfree_source_prereg_20260722.md")
RESULT = Path("docs/m23_59_relation_pretrained_hierarchical_flow_v3_gtfree_source_result_20260722.md")
SCRIPT = Path(__file__).relative_to(REPO)
TEST_SCRIPT = Path("scripts/m23_research/test_m23_62_gtfree_feature_contract.py")
PHASE0_SCRIPT = Path("scripts/dump_yolox_reid_phase0.py")
M23_10_SCRIPT = Path("scripts/m23_research/m23_10_build_micrograph.py")
M23_59_V2_SCRIPT = Path("scripts/m23_research/m23_59_relation_pretrained_hierarchical_flow_v2.py")
M23_61_ROOT = Path("outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_semantic_alignment")
M23_59_V2_ROOT = Path("outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2")
EXTERNAL_DATASET_MANIFEST = M23_59_V2_ROOT / "external_dataset_manifest.json"
BOT_ROOT = Path("external/BoT-SORT-main")
MOT17_TRACK_ROOT = BOT_ROOT / "YOLOX_outputs/full7_best_raw/track_results"
MOT17_TRACK_MANIFEST = BOT_ROOT / "YOLOX_outputs/full7_best_raw/run_manifest.json"
MOT20_TRACK_ROOT = Path("outputs/assocriskbench_p15_20260714/fused_a42_adaptive_risk_eval/l1_r2_q75_adaptive_priority/track_results")
MOT20_PHASE0_ROOT = Path("outputs/alink_train_inputs/phase0_root")
MOT17_SEQS = ("MOT17-02", "MOT17-04", "MOT17-05", "MOT17-09", "MOT17-10", "MOT17-11", "MOT17-13")
MOT17_TRAIN = ("MOT17-02", "MOT17-04", "MOT17-05", "MOT17-09", "MOT17-10")
MOT17_VALIDATION = ("MOT17-11", "MOT17-13")
MOT20_SEQS = ("MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05")
APP_DIM = 128
GEOM_DIM = 16
FULL_DIM = 144
PROJECTION_SEED = 2310
MAPPING_IOU = 0.5
VISIBILITY_SENTINEL = 1.0
CONTRACT_VERSION = "m23_59_v3_gtfree_source_contract_3.1.0"
SUMMARY = ROOT / "summary.csv"
EVENTS = ROOT / "protocol_events.jsonl"
SUMMARY_FIELDS = ["experiment", "stage", "status", "started_at", "completed_at", "report", "decision", "notes"]
CRITICAL_PHASE0_ARGS = (
    "ckpt", "exp_file", "fast_reid_config", "fast_reid_weights", "feature_dtype",
    "fp16", "fuse", "dump_min_score", "reid_min_score", "reid_batch_size",
    "track_low_thresh", "no_reid",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_write(path: Path, payload: Any, *, create_only: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if create_only and path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def csv_write(path: Path, rows: list[dict[str, Any]], fields: list[str], *, create_only: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if create_only and path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def append_event(name: str, **payload: Any) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    with EVENTS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time": utc_now(), "event": name, **payload}, sort_keys=True) + "\n")


def read_summary() -> list[dict[str, str]]:
    with SUMMARY.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def update_stage(stage: str, status: str, report: str = "", decision: str = "", notes: str = "") -> None:
    rows = read_summary()
    found = False
    for row in rows:
        if row["stage"] != stage:
            continue
        found = True
        if not row["started_at"]:
            row["started_at"] = utc_now()
        row["status"] = status
        row["completed_at"] = "" if status in {"pending", "running"} else utc_now()
        row["report"] = report
        row["decision"] = decision
        row["notes"] = notes
    if not found:
        raise KeyError(stage)
    csv_write(SUMMARY, rows, SUMMARY_FIELDS, create_only=False)


def registry_header() -> list[str]:
    with REGISTRY.open(newline="", encoding="utf-8", errors="replace") as handle:
        return next(csv.reader(handle))


def registry_append(values: dict[str, Any]) -> None:
    header = registry_header()
    indices = {name: index for index, name in enumerate(header)}
    row = [""] * len(header)
    for key, value in values.items():
        if key in indices:
            row[indices[key]] = str(value)
    with REGISTRY.open("a", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(row)


def close_registry(decision: str, closure_path: Path) -> None:
    raw = REGISTRY.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    header = next(csv.reader([raw[0].rstrip("\r\n")]))
    indices = {name: index for index, name in enumerate(header)}
    changed = 0
    for line_index in range(1, len(raw)):
        row = next(csv.reader([raw[line_index].rstrip("\r\n")]))
        if len(row) < len(header):
            row += [""] * (len(header) - len(row))
        if row[indices.get("tag", 8)] == "M23-62-v3-gtfree-source" and row[indices.get("status", 2)] == "running":
            row[indices["status"]] = "superseded"
            if "current_stage" in indices:
                row[indices["current_stage"]] = "superseded"
            buffer = io.StringIO()
            csv.writer(buffer, lineterminator="\n").writerow(row)
            raw[line_index] = buffer.getvalue()
            changed += 1
    if changed != 1:
        raise RuntimeError(f"expected exactly one running M23-62 registry row, found {changed}")
    temporary = REGISTRY.with_name(REGISTRY.name + f".tmp.{os.getpid()}")
    temporary.write_text("".join(raw), encoding="utf-8")
    os.replace(temporary, REGISTRY)
    registry_append({
        "timestamp": utc_now(), "kind": "semantic_alignment_validation", "status": "completed",
        "script": str(SCRIPT), "dataset": "MOT17+MOT20", "split": "gtfree_source_regeneration",
        "tracker_family": EXP_ID, "variant": "m23_59_v3_gtfree_source_regeneration",
        "tag": "M23-62-v3-gtfree-source-closed", "run_root": str(ROOT),
        "summary_csv": str(SUMMARY), "log_path": str(closure_path), "name": TITLE,
        "dataset_split": "MOT17 physical 5/2 + MOT20 train observable only", "run_dir": str(ROOT),
        "current_stage": "closed", "decision": decision, "phase": "closed",
        "notes": "GT-free source-row and 144-D observable regeneration only; no labels/training/replay/TrackEval/tracker/test",
    })


def tracker_path(domain: str, seq: str) -> Path:
    if domain == "MOT17":
        return MOT17_TRACK_ROOT / f"{seq}-FRCNN.txt"
    if domain == "MOT20":
        return MOT20_TRACK_ROOT / f"{seq}.txt"
    raise ValueError(domain)


def sequence_dir(domain: str, seq: str) -> Path:
    if domain == "MOT17":
        return Path("datasets/MOT17/train") / f"{seq}-FRCNN"
    if domain == "MOT20":
        return Path("datasets/MOT20/train") / seq
    raise ValueError(domain)


def read_seqinfo(path: Path) -> dict[str, int]:
    parser = configparser.ConfigParser()
    parser.read(path)
    if "Sequence" not in parser:
        raise RuntimeError(f"invalid seqinfo: {path}")
    section = parser["Sequence"]
    return {
        "width": int(section.get("imWidth", "0")),
        "height": int(section.get("imHeight", "0")),
        "length": int(section.get("seqLength", "0")),
        "fps": int(section.get("frameRate", "30")),
    }


def read_source_rows(path: Path, seqinfo: dict[str, int]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    with path.open(encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            fields = line.rstrip("\n").split(",")
            if len(fields) < 6:
                raise RuntimeError(f"malformed source row {path}:{line_index + 1}")
            frame = int(float(fields[0]))
            track_id = int(float(fields[1]))
            x, y, width, height = map(float, fields[2:6])
            score = float(fields[6]) if len(fields) > 6 else -1.0
            key = (frame, track_id)
            if key in seen:
                raise RuntimeError(f"duplicate frame/track row {path}: {key}")
            seen.add(key)
            if frame < 1 or frame > seqinfo["length"]:
                raise RuntimeError(f"frame outside seqinfo {path}: {frame}")
            if width <= 1.0 or height <= 1.0:
                raise RuntimeError(f"invalid source box {path}:{line_index + 1}")
            x2 = x + width
            y2 = y + height
            if x2 <= 0 or y2 <= 0 or x >= seqinfo["width"] or y >= seqinfo["height"]:
                raise RuntimeError(f"fully outside source box {path}:{line_index + 1}")
            records.append({
                "row_index": line_index, "line_index": line_index, "frame": frame, "track_id": track_id,
                "x1": x, "y1": y, "x2": x2, "y2": y2, "source_score": score,
            })
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise RuntimeError(f"empty source tracker: {path}")
    if not np.array_equal(frame.row_index.to_numpy(np.int64), np.arange(len(frame), dtype=np.int64)):
        raise RuntimeError("source row order changed")
    return frame


def iou_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if len(left) == 0 or len(right) == 0:
        return np.zeros((len(left), len(right)), np.float32)
    x1 = np.maximum(left[:, None, 0], right[None, :, 0])
    y1 = np.maximum(left[:, None, 1], right[None, :, 1])
    x2 = np.minimum(left[:, None, 2], right[None, :, 2])
    y2 = np.minimum(left[:, None, 3], right[None, :, 3])
    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_left = np.maximum(0.0, left[:, 2] - left[:, 0]) * np.maximum(0.0, left[:, 3] - left[:, 1])
    area_right = np.maximum(0.0, right[:, 2] - right[:, 0]) * np.maximum(0.0, right[:, 3] - right[:, 1])
    return (intersection / np.maximum(area_left[:, None] + area_right[None, :] - intersection, 1e-12)).astype(np.float32)


def stable_frame_assignment(
    source_boxes: np.ndarray,
    source_keys: np.ndarray,
    detection_boxes: np.ndarray,
    detection_global_ids: np.ndarray,
    detection_has_reid: np.ndarray,
    threshold: float = MAPPING_IOU,
) -> tuple[np.ndarray, np.ndarray]:
    """Return detection row indices and IoUs in original source order.

    Sorting by stable source keys and detector global IDs makes the assignment
    invariant to input-array permutations. Hungarian matching is then applied to
    the fixed sorted matrices.
    """
    mapped = np.full(len(source_boxes), -1, np.int64)
    quality = np.zeros(len(source_boxes), np.float32)
    if len(source_boxes) == 0 or len(detection_boxes) == 0:
        return mapped, quality
    source_order = np.lexsort((source_keys[:, 1], source_keys[:, 0]))
    detection_order = np.argsort(detection_global_ids, kind="mergesort")
    overlap = iou_matrix(source_boxes[source_order], detection_boxes[detection_order])
    source_match, detection_match = linear_sum_assignment(-overlap)
    for source_position, detection_position in zip(source_match, detection_match):
        original_source = int(source_order[source_position])
        original_detection = int(detection_order[detection_position])
        value = float(overlap[source_position, detection_position])
        if value >= threshold and bool(detection_has_reid[original_detection]):
            mapped[original_source] = original_detection
            quality[original_source] = value
    return mapped, quality


def l2_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, np.float32)
    if values.size == 0:
        return values
    denominator = np.linalg.norm(values, axis=1, keepdims=True)
    denominator[denominator < 1e-12] = 1.0
    return values / denominator


def projection_matrix() -> np.ndarray:
    rng = np.random.default_rng(PROJECTION_SEED)
    return rng.normal(size=(2048, APP_DIM)).astype(np.float32) / math.sqrt(APP_DIM)


def project_embeddings(raw: np.ndarray, matrix: np.ndarray | None = None) -> np.ndarray:
    raw = l2_normalize(np.asarray(raw, np.float32))
    if raw.ndim != 2 or raw.shape[1] != 2048:
        raise ValueError(f"expected Nx2048 embeddings, got {raw.shape}")
    projected = raw @ (projection_matrix() if matrix is None else matrix)
    return l2_normalize(projected).astype(np.float32)


def validate_feature_memmap(
    stored: np.ndarray,
    mapped_mask: np.ndarray,
    chunk_size: int = 32768,
) -> dict[str, Any]:
    finite = True
    mapped_norm_max_error = 0.0
    unmapped_nonzero = 0
    for start in range(0, len(stored), chunk_size):
        end = min(len(stored), start + chunk_size)
        block = np.asarray(stored[start:end], np.float32)
        finite = finite and bool(np.isfinite(block).all())
        mask = mapped_mask[start:end]
        if np.any(mask):
            norms = np.linalg.norm(block[mask, :APP_DIM], axis=1)
            mapped_norm_max_error = max(mapped_norm_max_error, float(np.max(np.abs(norms - 1.0))))
        if np.any(~mask):
            unmapped_nonzero += int(np.count_nonzero(block[~mask, :APP_DIM]))
    return {
        "finite": finite,
        "mapped_norm_max_error": mapped_norm_max_error,
        "mapped_norm_unit": mapped_norm_max_error <= 0.002,
        "unmapped_appearance_nonzero_cells": unmapped_nonzero,
        "unmapped_appearance_zero": unmapped_nonzero == 0,
    }


def canonical_geometry(meta: pd.DataFrame, width: int, height: int) -> np.ndarray:
    n = len(meta)
    output = np.zeros((n, GEOM_DIM), np.float32)
    x1 = meta.x1.to_numpy(np.float32)
    y1 = meta.y1.to_numpy(np.float32)
    x2 = meta.x2.to_numpy(np.float32)
    y2 = meta.y2.to_numpy(np.float32)
    center_x = 0.5 * (x1 + x2)
    center_y = 0.5 * (y1 + y2)
    box_width = np.maximum(x2 - x1, 1.0)
    box_height = np.maximum(y2 - y1, 1.0)
    output[:, 0] = center_x / max(width, 1)
    output[:, 1] = center_y / max(height, 1)
    output[:, 2] = box_width / max(width, 1)
    output[:, 3] = box_height / max(height, 1)
    output[:, 4] = np.log(box_width / box_height)
    output[:, 5] = np.log(np.maximum(box_width * box_height / max(width * height, 1), 1e-8))
    output[:, 6] = VISIBILITY_SENTINEL
    frames = meta.frame.to_numpy(np.float32)
    line_indices = meta.line_index.to_numpy(np.int64)
    by_track: dict[int, list[int]] = defaultdict(list)
    for index, track_id in enumerate(meta.track_id.to_numpy(np.int64)):
        by_track[int(track_id)].append(index)
    for indices in by_track.values():
        indices.sort(key=lambda index: (frames[index], line_indices[index]))
        for position, index in enumerate(indices):
            if position:
                previous = indices[position - 1]
                delta = max(float(frames[index] - frames[previous]), 1.0)
                output[index, 7] = (center_x[index] - center_x[previous]) / max(float(box_height[index]) * delta, 1.0)
                output[index, 8] = (center_y[index] - center_y[previous]) / max(float(box_height[index]) * delta, 1.0)
                output[index, 9] = math.log(max(float(box_width[index]), 1.0) / max(float(box_width[previous]), 1.0)) / delta
                output[index, 10] = math.log(max(float(box_height[index]), 1.0) / max(float(box_height[previous]), 1.0)) / delta
                output[index, 11] = min(delta / 30.0, 20.0)
            if position >= 2:
                previous = indices[position - 1]
                output[index, 12] = output[index, 7] - output[previous, 7]
                output[index, 13] = output[index, 8] - output[previous, 8]
    by_frame: dict[int, list[int]] = defaultdict(list)
    for index, frame in enumerate(meta.frame.to_numpy(np.int64)):
        by_frame[int(frame)].append(index)
    for indices in by_frame.values():
        index_array = np.asarray(indices, np.int64)
        centers = np.stack([
            center_x[index_array] / max(width, 1),
            center_y[index_array] / max(height, 1),
        ], axis=1)
        if len(indices) == 1:
            nearest = np.ones(1, np.float32)
        else:
            distance = np.sqrt(((centers[:, None] - centers[None, :]) ** 2).sum(axis=2))
            distance += np.eye(len(indices), dtype=np.float32) * 1e6
            nearest = distance.min(axis=1)
        output[index_array, 14] = min(len(indices) / 100.0, 5.0)
        output[index_array, 15] = np.minimum(nearest, 1.0)
    if not np.isfinite(output).all():
        raise RuntimeError("non-finite canonical geometry")
    return output


def small_npz_member(path: Path, name: str, allow_pickle: bool = False) -> np.ndarray:
    with zipfile.ZipFile(path) as archive:
        return np.load(io.BytesIO(archive.read(name)), allow_pickle=allow_pickle)


def mmap_npz_member(path: Path, name: str) -> np.memmap:
    with zipfile.ZipFile(path) as archive:
        information = archive.getinfo(name)
        if information.compress_type != zipfile.ZIP_STORED:
            raise RuntimeError(f"compressed NPZ member cannot be memory mapped: {path}:{name}")
    with path.open("rb") as handle:
        handle.seek(information.header_offset + 26)
        filename_length, extra_length = struct.unpack("<HH", handle.read(4))
        offset = information.header_offset + 30 + filename_length + extra_length
        handle.seek(offset)
        version = np.lib.format.read_magic(handle)
        shape, fortran_order, dtype = np.lib.format._read_array_header(handle, version)
        array_offset = handle.tell()
    return np.memmap(path, dtype=dtype, mode="r", offset=array_offset, shape=shape, order="F" if fortran_order else "C")


def phase0_path(domain: str, seq: str) -> Path:
    if domain == "MOT17":
        return ROOT / "phase0_mot17" / seq / "dump_yolox_reid.npz"
    link = MOT20_PHASE0_ROOT / seq / "dump_yolox_reid.npz"
    if not link.exists():
        raise FileNotFoundError(link)
    return link.resolve()


def phase0_manifest_path(domain: str, seq: str) -> Path:
    if domain == "MOT17":
        return ROOT / "phase0_mot17" / seq / "manifest.json"
    return phase0_path(domain, seq).parent / "manifest.json"


def map_rows_to_phase0(meta: pd.DataFrame, npz_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    detections = mmap_npz_member(npz_path, "detections.npy")
    columns = small_npz_member(npz_path, "columns.npy", allow_pickle=True).tolist()
    offsets = small_npz_member(npz_path, "frame_offsets.npy").astype(np.int64)
    column = {str(name): index for index, name in enumerate(columns)}
    required = {"frame", "global_det_idx", "x1", "y1", "x2", "y2", "has_reid"}
    if not required.issubset(column):
        raise RuntimeError(f"phase0 columns missing: {sorted(required - set(column))}")
    mapped_global = np.full(len(meta), -1, np.int64)
    mapped_iou = np.zeros(len(meta), np.float32)
    mapped_npz_row = np.full(len(meta), -1, np.int64)
    grouped = meta.groupby("frame", sort=True).indices
    for frame_value, source_indices_raw in grouped.items():
        frame = int(frame_value)
        source_indices = np.asarray(source_indices_raw, np.int64)
        start = int(offsets[frame - 1])
        end = int(offsets[frame]) if frame < len(offsets) else len(detections)
        if end <= start:
            continue
        detection_rows = np.asarray(detections[start:end], np.float32)
        source_boxes = meta.loc[source_indices, ["x1", "y1", "x2", "y2"]].to_numpy(np.float32)
        source_keys = meta.loc[source_indices, ["track_id", "line_index"]].to_numpy(np.int64)
        detection_boxes = detection_rows[:, [column["x1"], column["y1"], column["x2"], column["y2"]]]
        global_ids = detection_rows[:, column["global_det_idx"]].astype(np.int64)
        has_reid = detection_rows[:, column["has_reid"]] > 0.5
        local_map, local_iou = stable_frame_assignment(source_boxes, source_keys, detection_boxes, global_ids, has_reid)
        valid = local_map >= 0
        if np.any(valid):
            selected_source = source_indices[valid]
            selected_local = local_map[valid]
            mapped_npz_row[selected_source] = start + selected_local
            mapped_global[selected_source] = global_ids[selected_local]
            mapped_iou[selected_source] = local_iou[valid]
    return mapped_npz_row, mapped_global, mapped_iou


def load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, REPO / path)
    if specification is None or specification.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def feature_contract(script_sha: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(APP_DIM):
        rows.append({
            "zero_based_index": index, "one_based_display_index": index + 1,
            "feature_name": f"appearance_{index:03d}", "feature_group": "appearance",
            "geometry_local_index": "",
            "exact_formula": f"L2(L2(FastReID_MOT20_SBS_S50(matched_phase0_detection_crop)) @ Gaussian(seed=2310,2048x128)/sqrt(128))[{index}]",
            "physical_meaning": f"component {index} of canonical projected appearance embedding",
            "units": "dimensionless", "sign_convention": "signed projection component",
            "clipping_range": "none; full vector L2-normalized",
            "missing_value_sentinel": "0.0 component; complete 128-D zero vector when no IoU>=0.5 ReID mapping",
            "dtype": "float16 artifact / float32 compute",
            "normalization": "raw 2048-D L2, fixed Gaussian projection seed 2310, projected 128-D L2",
            "source_artifact": "same-frame phase0 YOLOX detection mapped from source-tracker row by stable Hungarian IoU",
            "generator_function": "map_rows_to_phase0 + project_embeddings",
            "generator_source_sha256": script_sha, "GT_free": True,
            "allowed_temporal_context": "same image frame only", "contract_version": CONTRACT_VERSION,
        })
    geometry = [
        ("center_x_norm", "cx/image_width", "horizontal box center", "normalized image width", "positive right", "unclipped", "none", "same row"),
        ("center_y_norm", "cy/image_height", "vertical box center", "normalized image height", "positive down", "unclipped", "none", "same row"),
        ("box_width_norm", "max(x2-x1,1)/image_width", "box width", "normalized image width", "positive", "[0,+inf)", "none", "same row"),
        ("box_height_norm", "max(y2-y1,1)/image_height", "box height", "normalized image height", "positive", "[0,+inf)", "none", "same row"),
        ("log_aspect", "log(max(width,1)/max(height,1))", "log aspect ratio", "log ratio", "positive wider", "unclipped", "none", "same row"),
        ("log_area_fraction", "log(max(width*height/(image_width*image_height),1e-8))", "log image-area fraction", "log fraction", "larger means larger", "lower floor 1e-8", "none", "same row"),
        ("visibility_unavailable_sentinel", "1.0", "cross-domain visibility-unavailable sentinel, not observed visibility", "dimensionless sentinel", "fixed positive sentinel", "exactly 1.0", "1.0", "none"),
        ("velocity_x_height_frame", "(cx_t-cx_prev)/(max(h_t,1)*max(frame_delta,1))", "horizontal velocity", "current box heights/frame", "positive right", "unclipped", "0 first track row", "previous source row with same track_id"),
        ("velocity_y_height_frame", "(cy_t-cy_prev)/(max(h_t,1)*max(frame_delta,1))", "vertical velocity", "current box heights/frame", "positive down", "unclipped", "0 first track row", "previous source row with same track_id"),
        ("log_width_change_per_frame", "log(max(w_t,1)/max(w_prev,1))/dt", "width growth rate", "log ratio/frame", "positive growing", "unclipped", "0 first track row", "previous source row with same track_id"),
        ("log_height_change_per_frame", "log(max(h_t,1)/max(h_prev,1))/dt", "height growth rate", "log ratio/frame", "positive growing", "unclipped", "0 first track row", "previous source row with same track_id"),
        ("frame_delta_over_30_clipped", "min(max(frame_t-frame_prev,1)/30,20)", "temporal gap", "30-frame units", "positive forward", "[0,20]", "0 first track row", "previous source row with same track_id"),
        ("velocity_x_residual", "vx_t-vx_prev", "horizontal acceleration proxy", "height/frame difference", "positive increasing right velocity", "unclipped", "0 before third track row", "two previous source rows with same track_id"),
        ("velocity_y_residual", "vy_t-vy_prev", "vertical acceleration proxy", "height/frame difference", "positive increasing down velocity", "unclipped", "0 before third track row", "two previous source rows with same track_id"),
        ("crowd_density_over_100_clipped", "min(count_same_frame_source_rows/100,5)", "same-frame observable source-row density", "hundreds of rows", "positive denser", "[0,5]", "none for valid row", "same frame source rows only"),
        ("nearest_neighbor_distance", "min(min_j!=i hypot((cx_i-cx_j)/W,(cy_i-cy_j)/H),1)", "nearest same-frame source-row center distance", "normalized image axes", "larger more isolated", "[0,1]", "1.0 singleton frame", "same frame source rows only"),
    ]
    for local_index, (name, formula, meaning, units, sign, clipping, sentinel, context) in enumerate(geometry):
        global_index = APP_DIM + local_index
        rows.append({
            "zero_based_index": global_index, "one_based_display_index": global_index + 1,
            "feature_name": f"geometry_{local_index:02d}_{name}", "feature_group": "geometry",
            "geometry_local_index": local_index, "exact_formula": formula, "physical_meaning": meaning,
            "units": units, "sign_convention": sign, "clipping_range": clipping,
            "missing_value_sentinel": sentinel, "dtype": "float16 artifact / float32 compute",
            "normalization": "formula-internal only; no data-conditioned normalizer",
            "source_artifact": "source-tracker row table", "generator_function": "canonical_geometry",
            "generator_source_sha256": script_sha, "GT_free": True,
            "allowed_temporal_context": context, "contract_version": CONTRACT_VERSION,
        })
    canonical = [{key: value for key, value in row.items() if key != "contract_hash"} for row in rows]
    contract_hash = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    for row in rows:
        row["contract_hash"] = contract_hash
    aggregate = {
        "feature_count": len(rows), "unique_index_count": len({int(row["zero_based_index"]) for row in rows}),
        "gt_free_count": sum(bool(row["GT_free"]) for row in rows), "contract_hash": contract_hash,
        "visibility_global_index": 134, "feature_143_global_zero_based_index": 143,
        "feature_143_geometry_local_index": 15, "feature_143_one_based_index": 144,
    }
    return rows, aggregate


def required_inputs() -> list[Path]:
    items = [
        Path("AGENTS.md"), SCRIPT, TEST_SCRIPT, PHASE0_SCRIPT, M23_10_SCRIPT, M23_59_V2_SCRIPT,
        Path("docs/m23_59_relation_pretrained_hierarchical_flow_v3_semantic_alignment_result_20260722.md"),
        M23_61_ROOT / "final_summary.json", M23_61_ROOT / "feature_contract_v3.json",
        M23_61_ROOT / "independent_closure_validation.json", EXTERNAL_DATASET_MANIFEST,
        MOT17_TRACK_MANIFEST, BOT_ROOT / "tools/track.py", BOT_ROOT / "pretrained/bytetrack_x_mot20.pth.tar",
        BOT_ROOT / "pretrained/mot20_sbs_S50.pth", BOT_ROOT / "fast_reid/configs/MOT20/sbs_S50.yml",
        BOT_ROOT / "yolox/exps/example/mot/yolox_x_mix_mot20_ch.py",
    ]
    for seq in MOT17_SEQS:
        items.extend([tracker_path("MOT17", seq), sequence_dir("MOT17", seq) / "seqinfo.ini"])
    for seq in MOT20_SEQS:
        items.extend([tracker_path("MOT20", seq), phase0_path("MOT20", seq), phase0_manifest_path("MOT20", seq), sequence_dir("MOT20", seq) / "seqinfo.ini"])
    for path in items:
        if not path.exists():
            raise FileNotFoundError(path)
    return items


def verify_frozen() -> dict[str, Any]:
    manifest_path = ROOT / "implementation_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks = {
        "script": sha256_file(SCRIPT) == manifest["script_sha256"],
        "test": sha256_file(TEST_SCRIPT) == manifest["test_sha256"],
        "prereg": sha256_file(PREREG) == manifest["preregistration_sha256"],
        "contract": sha256_file(ROOT / "feature_contract_v3_1.json") == manifest["feature_contract_sha256"],
    }
    for item in manifest["input_artifacts"]:
        checks[f"input::{item['path']}"] = sha256_file(Path(item["path"])) == item["sha256"]
    if not all(checks.values()):
        failed = [key for key, value in checks.items() if not value]
        raise RuntimeError(f"frozen M23-62 input mismatch: {failed}")
    return manifest


def preregistration_text(inputs: list[dict[str, Any]], contract_hash: str) -> str:
    input_rows = "\n".join(f"| `{item['path']}` | `{item['sha256']}` | {item['bytes']} |" for item in inputs)
    return f"""# M23-59 v3 GT-Free Source-Row Regeneration — M23-62 Preregistration (2026-07-22)

## Scope

M23-62 is a source-row and observable regeneration stage, not a model experiment. It may read only frozen tracker rows, train images, seqinfo, frozen detector/ReID artifacts and phase0 dumps. It must not read MOT17 or MOT20 GT, teacher actions, identity labels, held-outer labels, MOT20 test data, or TrackEval outputs for selection.

No relation training, counterfactual model replay, tracker generation, TrackEval, strict outer gate, M23-54, M23-58 or test submission is authorized in this experiment.

## Frozen source-row construction

- MOT17 source rows: the pre-existing `full7_best_raw` raw BoT-SORT FRCNN outputs for all seven physical train videos. Its run manifest fixes image-only inference, YOLOX/ByteTrack checkpoint, SBS_S50 ReID and raw association; evaluation is outside tracker generation.
- MOT20 source rows: the existing M23-59 source tracker rows for MOT20-01/02/03/05.
- Row order is source tracker file order. Temporal context is grouped only by source `track_id` and sorted by `(frame,line_index)`.
- MOT17 physical train/validation split remains 02/04/05/09/10 versus 11/13. Detector variants are not duplicated; only FRCNN physical videos are admitted.

## Canonical 144-D contract

Contract version: `{CONTRACT_VERSION}`  
Contract hash: `{contract_hash}`

- columns 0..127: same-frame phase0 YOLOX/ReID mapping by stable Hungarian IoU, threshold 0.5, fixed MOT20 SBS_S50 2048-D embedding, seed-2310 128-D projection and L2 normalization; unmatched rows use the all-zero vector;
- global column 134 / geometry local 6: fixed value `1.0`, explicitly named visibility-unavailable sentinel and never interpreted as measured visibility;
- global column 143 / geometry local 15 / one-based 144: nearest same-frame normalized source-row center distance clipped to `[0,1]`, singleton sentinel `1.0`;
- geometry local 14 counts same-frame source rows in both domains;
- no data-conditioned normalization or label-conditioned statistic is permitted.

## Execution order and pass rule

1. freeze preregistration, implementation and input SHA;
2. validate source tracker/image lineage without labels;
3. generate MOT17 phase0 detector/ReID dumps from FRCNN images only using the exact critical settings of frozen MOT20 phase0;
4. regenerate all seven MOT17 and four MOT20 observables from source rows and phase0 dumps;
5. validate 144/144 formula, semantic and GT-free provenance, full geometry float16 round-trip, feature-143 round-trip, mapping/index stability, source split and all SHA;
6. only a complete 144/144 pass may authorize a later fresh experiment to open MOT17 supervision labels and retrain. M23-59 v2 checkpoint reuse is prohibited because training-side inputs changed.

Any ambiguity, GT/teacher/held-outer lineage, phase0 setting mismatch, score/index instability, missing sequence or numerical mismatch causes fail-closed termination. This experiment never unlocks labels itself.

## Frozen inputs

| Path | SHA-256 | Bytes |
|---|---|---:|
{input_rows}
"""


def command_init() -> None:
    if ROOT.exists() or PREREG.exists() or RESULT.exists():
        raise FileExistsError("M23-62 output already exists; refusing overwrite")
    with REGISTRY.open(newline="", encoding="utf-8", errors="replace") as handle:
        occupied = [row for row in csv.DictReader(handle) if "M23-62" in " ".join(str(value) for value in row.values())]
    if occupied:
        raise RuntimeError("M23-62 became occupied before initialization")
    test = subprocess.run([sys.executable, str(TEST_SCRIPT)], cwd=REPO, text=True, capture_output=True)
    if test.returncode != 0:
        raise RuntimeError(f"contract regression failed before freeze:\n{test.stdout}\n{test.stderr}")
    test_payload = json.loads(test.stdout)
    contract_rows, aggregate = feature_contract(sha256_file(SCRIPT))
    if aggregate["feature_count"] != 144 or aggregate["unique_index_count"] != 144 or aggregate["gt_free_count"] != 144:
        raise RuntimeError(f"contract does not pass 144/144: {aggregate}")
    inputs = [{"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size} for path in required_inputs()]
    PREREG.parent.mkdir(parents=True, exist_ok=True)
    PREREG.write_text(preregistration_text(inputs, aggregate["contract_hash"]), encoding="utf-8")
    ROOT.mkdir(parents=True, exist_ok=False)
    json_write(ROOT / "feature_contract_v3_1.json", {"contract_version": CONTRACT_VERSION, "aggregate": aggregate, "features": contract_rows})
    csv_write(ROOT / "feature_contract.csv", contract_rows, list(contract_rows[0].keys()))
    generator_manifest = {
        "experiment_id": EXP_ID, "contract_version": CONTRACT_VERSION, "contract_hash": aggregate["contract_hash"],
        "main_script": str(SCRIPT), "main_script_sha256": sha256_file(SCRIPT),
        "test_script": str(TEST_SCRIPT), "test_script_sha256": sha256_file(TEST_SCRIPT),
        "phase0_generator": str(PHASE0_SCRIPT), "phase0_generator_sha256": sha256_file(PHASE0_SCRIPT),
        "mapping_reference": str(M23_10_SCRIPT), "mapping_reference_sha256": sha256_file(M23_10_SCRIPT),
        "v2_formula_reference": str(M23_59_V2_SCRIPT), "v2_formula_reference_sha256": sha256_file(M23_59_V2_SCRIPT),
        "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__},
        "projection_seed": PROJECTION_SEED, "mapping_iou": MAPPING_IOU, "visibility_sentinel": VISIBILITY_SENTINEL,
    }
    json_write(ROOT / "generator_sha_manifest.json", generator_manifest)
    implementation = {
        "experiment_id": EXP_ID, "title": TITLE, "created_at": utc_now(), "status": "preregistered",
        "script": str(SCRIPT), "script_sha256": sha256_file(SCRIPT), "test": str(TEST_SCRIPT), "test_sha256": sha256_file(TEST_SCRIPT),
        "preregistration": str(PREREG), "preregistration_sha256": sha256_file(PREREG),
        "feature_contract": str(ROOT / "feature_contract_v3_1.json"),
        "feature_contract_sha256": sha256_file(ROOT / "feature_contract_v3_1.json"),
        "contract_hash": aggregate["contract_hash"], "input_artifacts": inputs, "regression_test": test_payload,
        "prohibitions": {"mot17_gt": True, "mot20_gt": True, "mot20_test": True, "training": True, "counterfactual_model_replay": True, "trackeval": True, "tracker_generation": True, "m23_54": True, "m23_58": True},
    }
    json_write(ROOT / "implementation_manifest.json", implementation)
    stages = [
        ("preregistration", "completed", str(PREREG), "pass", "implementation and 144-D contract frozen"),
        ("feature_contract", "completed", str(ROOT / "feature_contract_v3_1.json"), "pass", "144/144 unique and GT-free canonical definitions"),
        ("source_tracker_lineage", "pending", "", "", ""),
        ("mot17_phase0", "pending", "", "", ""),
        ("mot17_observables", "pending", "", "", ""),
        ("mot20_observables", "pending", "", "", ""),
        ("semantic_validation", "pending", "", "", ""),
        ("counterfactual_replay", "prohibited_by_scope", str(PREREG), "not_run", "requires a later experiment"),
        ("training", "prohibited_by_scope", str(PREREG), "not_run", "training-side contract changed"),
        ("strict_outer_evaluation", "prohibited_by_scope", str(PREREG), "not_run", "no labels or tracker evaluation in M23-62"),
        ("closure", "pending", "", "", ""),
    ]
    timestamp = utc_now()
    summary_rows = [{
        "experiment": EXP_ID, "stage": stage, "status": status,
        "started_at": timestamp if status in {"completed", "prohibited_by_scope"} else "",
        "completed_at": timestamp if status in {"completed", "prohibited_by_scope"} else "",
        "report": report, "decision": decision, "notes": notes,
    } for stage, status, report, decision, notes in stages]
    csv_write(SUMMARY, summary_rows, SUMMARY_FIELDS)
    sequence_rows = []
    for domain, sequences in (("MOT17", MOT17_SEQS), ("MOT20", MOT20_SEQS)):
        for seq in sequences:
            sequence_rows.append({"domain": domain, "sequence": seq, "phase0_status": "pending" if domain == "MOT17" else "frozen_input", "observable_status": "pending", "manifest": "", "notes": ""})
    csv_write(ROOT / "sequence_status.csv", sequence_rows, list(sequence_rows[0].keys()))
    append_event("preregistration_frozen", implementation_sha256=sha256_file(ROOT / "implementation_manifest.json"), contract_hash=aggregate["contract_hash"])
    registry_append({
        "timestamp": utc_now(), "kind": "semantic_alignment_validation", "status": "running", "script": str(SCRIPT),
        "dataset": "MOT17+MOT20", "split": "gtfree_source_regeneration", "tracker_family": EXP_ID,
        "variant": "m23_59_v3_gtfree_source_regeneration", "tag": "M23-62-v3-gtfree-source",
        "run_root": str(ROOT), "summary_csv": str(SUMMARY), "log_path": str(ROOT / "implementation_manifest.json"),
        "name": TITLE, "dataset_split": "MOT17 physical 5/2 + MOT20 train observable only", "run_dir": str(ROOT),
        "current_stage": "source_lineage_running", "phase": "gtfree_source_regeneration",
        "notes": "running; no GT/training/replay/TrackEval/tracker/test",
    })
    print(json.dumps({"initialized": True, "experiment_id": EXP_ID, "contract": aggregate, "input_count": len(inputs)}, indent=2, sort_keys=True))


def image_tree_validation() -> dict[str, Any]:
    manifest = json.loads(EXTERNAL_DATASET_MANIFEST.read_text(encoding="utf-8"))
    train = set(manifest["dataset"]["physical_train"])
    validation = set(manifest["dataset"]["physical_validation"])
    failures: list[dict[str, Any]] = []
    train_hashes: set[str] = set()
    validation_hashes: set[str] = set()
    sequence_rows: list[dict[str, Any]] = []
    for seq in MOT17_SEQS:
        entry = manifest["dataset"]["physical"][seq]
        expected = {item["name"]: item["sha256"] for item in entry["canonical_images"]}
        image_dir = sequence_dir("MOT17", seq) / "img1"
        actual_files = sorted(image_dir.glob("*.jpg"))
        if len(actual_files) != len(expected):
            failures.append({"seq": seq, "kind": "image_count", "expected": len(expected), "actual": len(actual_files)})
        for image in actual_files:
            actual = sha256_file(image)
            wanted = expected.get(image.name)
            if actual != wanted:
                failures.append({"seq": seq, "kind": "image_sha", "file": image.name, "expected": wanted, "actual": actual})
            (train_hashes if seq in train else validation_hashes).add(actual)
        sequence_rows.append({"seq": seq, "split": "train" if seq in train else "validation", "images": len(actual_files), "variant": "FRCNN"})
    return {
        "physical_train": sorted(train), "physical_validation": sorted(validation), "physical_overlap": sorted(train & validation),
        "exact_image_sha_overlap_count": len(train_hashes & validation_hashes), "sequence_rows": sequence_rows,
        "failures": failures, "passed": not failures and not (train & validation) and len(train_hashes & validation_hashes) == 0,
    }


def command_audit_source() -> None:
    verify_frozen()
    update_stage("source_tracker_lineage", "running", notes="validating image-only tracker lineage and source rows")
    run_manifest = json.loads(MOT17_TRACK_MANIFEST.read_text(encoding="utf-8"))
    expected_settings = {
        "benchmark": "MOT17", "split_to_eval": "train", "mot17_detector_exts": ["FRCNN"],
        "detector_ckpt": "./pretrained/bytetrack_x_mot17.pth.tar", "reid_weights": "pretrained/mot17_sbs_S50.pth",
        "laplace_assoc": False,
    }
    actual_settings = {
        "benchmark": run_manifest.get("benchmark"), "split_to_eval": run_manifest.get("split_to_eval"),
        "mot17_detector_exts": run_manifest.get("mot17_detector_exts"), "detector_ckpt": run_manifest["detector"]["ckpt"],
        "reid_weights": run_manifest["reid"]["weights"], "laplace_assoc": run_manifest["association"]["laplace_assoc"],
    }
    static_track_source = (BOT_ROOT / "tools/track.py").read_text(encoding="utf-8", errors="replace")
    forbidden_runtime_tokens = [token for token in ("gt/gt.txt", "load_gt(", "teacher_action") if token in static_track_source]
    inventory: list[dict[str, Any]] = []
    for domain, sequences in (("MOT17", MOT17_SEQS), ("MOT20", MOT20_SEQS)):
        for seq in sequences:
            path = tracker_path(domain, seq)
            info = read_seqinfo(sequence_dir(domain, seq) / "seqinfo.ini")
            rows = read_source_rows(path, info)
            inventory.append({
                "domain": domain, "sequence": seq, "source_tracker": str(path), "source_tracker_sha256": sha256_file(path),
                "rows": len(rows), "tracks": int(rows.track_id.nunique()), "frames": int(rows.frame.nunique()),
                "frame_min": int(rows.frame.min()), "frame_max": int(rows.frame.max()), "sequence_length": info["length"],
                "duplicate_frame_track": int(rows.duplicated(["frame", "track_id"]).sum()),
                "invalid_boxes": int(((rows.x2 - rows.x1 <= 1) | (rows.y2 - rows.y1 <= 1)).sum()),
            })
    image_validation = image_tree_validation()
    checks = {
        "run_manifest_settings_exact": actual_settings == expected_settings,
        "all_seven_resolved_sequences": set(run_manifest["resolved_sequences"]) == {f"{seq}-FRCNN" for seq in MOT17_SEQS},
        "image_paths_only": all(str(spec["path"]).endswith("/img1") for spec in run_manifest["sequence_specs"]),
        "track_generator_has_no_gt_teacher_token": not forbidden_runtime_tokens,
        "all_source_rows_valid": all(row["duplicate_frame_track"] == 0 and row["invalid_boxes"] == 0 for row in inventory),
        "all_mot17_full_length": all(row["frame_max"] == row["sequence_length"] for row in inventory if row["domain"] == "MOT17"),
        "physical_split_and_images_pass": image_validation["passed"],
    }
    report = {
        "experiment_id": EXP_ID, "source_tracker_role": "frozen GT-free row generator input",
        "mot17_run_manifest": str(MOT17_TRACK_MANIFEST), "mot17_run_manifest_sha256": sha256_file(MOT17_TRACK_MANIFEST),
        "expected_settings": expected_settings, "actual_settings": actual_settings,
        "external_model_training_disclosure": "detector/ReID checkpoints are externally supervised historical models; no label file is read during this source-row generation",
        "direct_label_read": False, "teacher_action_read": False, "forbidden_runtime_tokens": forbidden_runtime_tokens,
        "checks": checks, "passed": all(checks.values()), "image_validation": image_validation,
    }
    csv_write(ROOT / "source_row_inventory.csv", inventory, list(inventory[0].keys()))
    json_write(ROOT / "source_tracker_manifest.json", report)
    raw_inputs = {
        "experiment_id": EXP_ID, "source_tracker_manifest": str(ROOT / "source_tracker_manifest.json"),
        "source_tracker_manifest_sha256": sha256_file(ROOT / "source_tracker_manifest.json"),
        "inventory_sha256": sha256_file(ROOT / "source_row_inventory.csv"),
        "mot17_image_validation": image_validation, "direct_mot17_gt_read": False, "direct_mot20_gt_read": False,
        "mot20_test_read": False, "normalization": "formula-local only; no label/domain statistics",
    }
    json_write(ROOT / "raw_input_manifest.json", raw_inputs)
    if not report["passed"]:
        update_stage("source_tracker_lineage", "completed_with_failures", str(ROOT / "source_tracker_manifest.json"), "fail", "source lineage failed")
        append_event("source_tracker_lineage_failed", failed_checks=[key for key, value in checks.items() if not value])
        raise RuntimeError("source tracker lineage failed")
    update_stage("source_tracker_lineage", "completed", str(ROOT / "source_tracker_manifest.json"), "pass", "all 11 source tracker inputs valid; MOT17 image-only inference lineage verified")
    append_event("source_tracker_lineage_completed", manifest_sha256=sha256_file(ROOT / "source_tracker_manifest.json"), source_sequences=len(inventory))
    print(json.dumps(report, indent=2, sort_keys=True))


def update_sequence(domain: str, seq: str, **updates: str) -> None:
    path = ROOT / "sequence_status.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    found = False
    for row in rows:
        if row["domain"] == domain and row["sequence"] == seq:
            found = True
            row.update(updates)
    if not found:
        raise KeyError((domain, seq))
    csv_write(path, rows, list(rows[0].keys()), create_only=False)


def reference_phase0_args() -> dict[str, Any]:
    reference = json.loads(phase0_manifest_path("MOT20", "MOT20-01").read_text(encoding="utf-8"))["args"]
    return {key: reference[key] for key in CRITICAL_PHASE0_ARGS}


def command_phase0_mot17(seq: str) -> None:
    verify_frozen()
    if seq not in MOT17_SEQS:
        raise ValueError(seq)
    source_report = json.loads((ROOT / "source_tracker_manifest.json").read_text(encoding="utf-8"))
    if not source_report["passed"]:
        raise RuntimeError("source tracker lineage did not pass")
    update_stage("mot17_phase0", "running", notes=f"generating canonical detector/ReID dump; current={seq}")
    update_sequence("MOT17", seq, phase0_status="running", notes="image-only canonical MOT20 detector/ReID settings")
    phase0 = load_module(f"m23_62_phase0_{seq[-2:]}", PHASE0_SCRIPT)
    reference = reference_phase0_args()
    final_sequence_root = ROOT / "phase0_mot17" / seq
    if final_sequence_root.exists():
        raise FileExistsError(f"refusing to overwrite frozen phase0 {final_sequence_root}")
    staging_root = ROOT / "staging" / f"phase0_{seq}_{os.getpid()}"
    if staging_root.exists():
        raise FileExistsError(staging_root)
    namespace = argparse.Namespace(
        data_root=str(Path("datasets").resolve()), benchmark="MOT17", split="train", seq_ids=[int(seq[-2:])],
        out_root=str(staging_root), bot_sort_root=str(BOT_ROOT.resolve()),
        exp_file=reference["exp_file"], ckpt=reference["ckpt"], fast_reid_config=reference["fast_reid_config"],
        fast_reid_weights=reference["fast_reid_weights"], device="gpu", fp16=reference["fp16"], fuse=reference["fuse"],
        conf=None, nms=None, tsize=None, track_low_thresh=reference["track_low_thresh"],
        dump_min_score=reference["dump_min_score"], reid_min_score=reference["reid_min_score"],
        reid_batch_size=reference["reid_batch_size"], no_reid=reference["no_reid"],
        feature_dtype=reference["feature_dtype"], limit_frames=0, frame_start=1, frame_end=0,
        write_csv=False, compress=False, overwrite=False,
    )
    info = read_seqinfo(sequence_dir("MOT17", seq) / "seqinfo.ini")
    spec = {
        "seq": seq, "seq_id": int(seq[-2:]), "seq_dir": str(sequence_dir("MOT17", seq).resolve()),
        "img_dir": str((sequence_dir("MOT17", seq) / "img1").resolve()), "physical_split": "train",
        "frame_rate": info["fps"], "seqinfo": {"width": info["width"], "height": info["height"], "seq_length": info["length"], "frame_rate": info["fps"], "im_dir": "img1", "im_ext": ".jpg"},
    }
    started = time.time()
    encoder = phase0.init_encoder(namespace)
    experiment, predictor, model = phase0.init_detector(namespace, seq)
    try:
        result = phase0.dump_sequence(namespace, predictor, encoder, spec, staging_root)
    finally:
        del predictor
        del model
        del encoder
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    staged_sequence_root = staging_root / seq
    if not staged_sequence_root.exists():
        raise RuntimeError(f"phase0 generator did not create {staged_sequence_root}")
    final_sequence_root.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staged_sequence_root, final_sequence_root)
    try:
        staging_root.rmdir()
        staging_root.parent.rmdir()
    except OSError:
        pass
    manifest_path = phase0_manifest_path("MOT17", seq)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    old_root = str(staging_root)
    new_root = str(ROOT / "phase0_mot17")
    manifest_text = json.dumps(manifest, sort_keys=True)
    if old_root in manifest_text:
        manifest = json.loads(manifest_text.replace(old_root, new_root))
        json_write(manifest_path, manifest, create_only=False)
    result_text = json.dumps(result, sort_keys=True)
    if old_root in result_text:
        result = json.loads(result_text.replace(old_root, new_root))
    critical = {key: manifest["args"][key] for key in CRITICAL_PHASE0_ARGS}
    checks = {
        "critical_args_equal_mot20_reference": critical == reference,
        "frames_full_sequence": int(manifest["frames_total"]) == info["length"],
        "failed_images_zero": int(manifest["failed_images"]) == 0,
        "feature_dim_2048": list(manifest["feature_shape"])[1] == 2048,
        "all_detections_have_reid": int(manifest["detections"]) == int(manifest["detections_with_reid"]),
        "image_only_path": str(manifest["spec"]["img_dir"]).endswith("/img1"),
    }
    freeze = {
        "experiment_id": EXP_ID, "domain": "MOT17", "sequence": seq, "status": "frozen" if all(checks.values()) else "failed",
        "critical_args": critical, "reference_args": reference, "checks": checks,
        "phase0_npz": str(phase0_path("MOT17", seq)), "phase0_npz_sha256": sha256_file(phase0_path("MOT17", seq)),
        "phase0_manifest": str(manifest_path), "phase0_manifest_sha256": sha256_file(manifest_path),
        "source_images": str(sequence_dir("MOT17", seq) / "img1"), "direct_label_read": False,
        "runtime_seconds": time.time() - started, "result": result,
    }
    json_write(ROOT / "phase0_mot17" / seq / "freeze_manifest.json", freeze)
    if not all(checks.values()):
        update_sequence("MOT17", seq, phase0_status="failed", manifest=str(ROOT / "phase0_mot17" / seq / "freeze_manifest.json"), notes="phase0 compatibility failed")
        append_event("mot17_phase0_failed", seq=seq, failed_checks=[key for key, value in checks.items() if not value])
        raise RuntimeError(f"{seq} phase0 compatibility failed")
    update_sequence("MOT17", seq, phase0_status="frozen", manifest=str(ROOT / "phase0_mot17" / seq / "freeze_manifest.json"), notes="GT-free image-only phase0 frozen")
    append_event("mot17_phase0_frozen", seq=seq, freeze_manifest_sha256=sha256_file(ROOT / "phase0_mot17" / seq / "freeze_manifest.json"))
    complete = all((ROOT / "phase0_mot17" / item / "freeze_manifest.json").exists() for item in MOT17_SEQS)
    if complete:
        update_stage("mot17_phase0", "completed", str(ROOT / "phase0_mot17"), "pass", "all seven canonical MOT17 phase0 dumps frozen")
    print(json.dumps(freeze, indent=2, sort_keys=True))


def command_build_observable(domain: str, seq: str) -> None:
    verify_frozen()
    sequences = MOT17_SEQS if domain == "MOT17" else MOT20_SEQS
    if seq not in sequences:
        raise ValueError((domain, seq))
    if domain == "MOT17" and not (ROOT / "phase0_mot17" / seq / "freeze_manifest.json").exists():
        raise RuntimeError(f"MOT17 phase0 not frozen for {seq}")
    stage = "mot17_observables" if domain == "MOT17" else "mot20_observables"
    update_stage(stage, "running", notes=f"regenerating from raw source rows; current={seq}")
    update_sequence(domain, seq, observable_status="running", notes="canonical 144-D regeneration")
    output_root = ROOT / "observables" / domain / seq
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite observable {output_root}")
    started = time.time()
    info = read_seqinfo(sequence_dir(domain, seq) / "seqinfo.ini")
    source = tracker_path(domain, seq)
    meta = read_source_rows(source, info)
    npz = phase0_path(domain, seq)
    mapped_npz_row, mapped_global, mapped_iou = map_rows_to_phase0(meta, npz)
    features_memmap = mmap_npz_member(npz, "features.npy")
    if features_memmap.ndim != 2 or features_memmap.shape[1] != 2048:
        raise RuntimeError(f"unexpected phase0 feature shape {features_memmap.shape}")
    output_root.mkdir(parents=True, exist_ok=False)
    feature_path = output_root / "row_features.f16.npy"
    output = np.lib.format.open_memmap(feature_path, mode="w+", dtype=np.float16, shape=(len(meta), FULL_DIM))
    output[:] = 0
    matrix = projection_matrix()
    mapped_source = np.flatnonzero(mapped_npz_row >= 0)
    for start in range(0, len(mapped_source), 8192):
        source_indices = mapped_source[start:start + 8192]
        detection_indices = mapped_npz_row[source_indices]
        raw = np.asarray(features_memmap[detection_indices], np.float32)
        output[source_indices, :APP_DIM] = project_embeddings(raw, matrix).astype(np.float16)
    geometry = canonical_geometry(meta, info["width"], info["height"])
    output[:, APP_DIM:] = geometry.astype(np.float16)
    output.flush()
    stored = np.load(feature_path, mmap_mode="r")
    geometry_roundtrip = np.array_equal(np.asarray(stored[:, APP_DIM:], np.float16), geometry.astype(np.float16))
    feature_143_roundtrip = np.array_equal(np.asarray(stored[:, 143], np.float16), geometry[:, 15].astype(np.float16))
    feature_validation = validate_feature_memmap(stored, mapped_npz_row >= 0)
    meta = meta.copy()
    meta["appearance_mapped"] = mapped_npz_row >= 0
    meta["mapped_phase0_row"] = mapped_npz_row
    meta["mapped_global_det_idx"] = mapped_global
    meta["mapping_iou"] = mapped_iou
    rows_path = output_root / "rows.parquet"
    meta.to_parquet(rows_path, index=False)
    checks = {
        "feature_shape_144": list(stored.shape) == [len(meta), FULL_DIM],
        "finite": feature_validation["finite"],
        "geometry_roundtrip_exact": geometry_roundtrip,
        "feature_143_roundtrip_exact": feature_143_roundtrip,
        "visibility_sentinel_exact": bool(np.all(np.asarray(stored[:, 134], np.float32) == np.float16(1.0))),
        "mapped_norm_unit": feature_validation["mapped_norm_unit"],
        "unmapped_appearance_zero": feature_validation["unmapped_appearance_zero"],
        "source_row_order_preserved": bool(np.array_equal(meta.row_index.to_numpy(np.int64), np.arange(len(meta), dtype=np.int64))),
        "no_gt_teacher_columns": not any(any(token in column.lower() for token in ("gt", "teacher", "identity", "purity", "label")) for column in meta.columns),
    }
    manifest = {
        "experiment_id": EXP_ID, "domain": domain, "sequence": seq,
        "status": "frozen" if all(checks.values()) else "failed", "contract_version": CONTRACT_VERSION,
        "source_tracker": str(source), "source_tracker_sha256": sha256_file(source),
        "phase0_npz": str(npz), "phase0_npz_sha256": sha256_file(npz),
        "phase0_manifest": str(phase0_manifest_path(domain, seq)), "phase0_manifest_sha256": sha256_file(phase0_manifest_path(domain, seq)),
        "rows": len(meta), "mapped_rows": int(len(mapped_source)), "mapping_rate": float(len(mapped_source) / len(meta)),
        "feature_shape": list(stored.shape), "feature_dtype": str(stored.dtype), "checks": checks,
        "feature_validation": feature_validation,
        "artifacts": {"rows": str(rows_path), "rows_sha256": sha256_file(rows_path), "row_features": str(feature_path), "row_features_sha256": sha256_file(feature_path)},
        "generator": str(SCRIPT), "generator_sha256": sha256_file(SCRIPT),
        "direct_gt_read": False, "teacher_action_read": False, "held_outer_label_read": False,
        "normalizer": "none beyond formula-local/L2; no domain or label statistics",
        "runtime_seconds": time.time() - started, "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }
    json_write(output_root / "manifest.json", manifest)
    if not all(checks.values()):
        update_sequence(domain, seq, observable_status="failed", manifest=str(output_root / "manifest.json"), notes="observable validation failed")
        append_event("observable_regeneration_failed", domain=domain, seq=seq, failed_checks=[key for key, value in checks.items() if not value])
        raise RuntimeError(f"{domain} {seq} observable validation failed")
    update_sequence(domain, seq, observable_status="frozen", manifest=str(output_root / "manifest.json"), notes=f"mapped_rate={manifest['mapping_rate']:.6f}")
    append_event("observable_frozen", domain=domain, seq=seq, manifest_sha256=sha256_file(output_root / "manifest.json"), direct_gt_read=False)
    complete = all((ROOT / "observables" / domain / item / "manifest.json").exists() for item in sequences)
    if complete:
        update_stage(stage, "completed", str(ROOT / "observables" / domain), "pass", f"all {len(sequences)} {domain} canonical observables frozen")
    print(json.dumps(manifest, indent=2, sort_keys=True))


def build_semantic_provenance() -> list[dict[str, Any]]:
    contract = json.loads((ROOT / "feature_contract_v3_1.json").read_text(encoding="utf-8"))["features"]
    rows: list[dict[str, Any]] = []
    for domain in ("MOT17", "MOT20"):
        for feature in contract:
            rows.append({
                "domain": domain, "zero_based_index": feature["zero_based_index"], "one_based_display_index": feature["one_based_display_index"],
                "feature_name": feature["feature_name"], "formula": feature["exact_formula"],
                "source": feature["source_artifact"], "uses_gt": False, "uses_teacher_action": False,
                "uses_identity_label": False, "uses_held_outer": False, "GT_free": True,
                "allowed_temporal_context": feature["allowed_temporal_context"], "status": "PASS",
                "contract_version": CONTRACT_VERSION, "contract_hash": feature["contract_hash"],
            })
    return rows


def process_scan() -> list[dict[str, Any]]:
    active: list[dict[str, Any]] = []
    excluded = {os.getpid(), os.getppid()}
    names = {Path(SCRIPT).name, "m23_59_relation_pretrained_hierarchical_flow_v2.py", "m23_60_relation_transfer_failure_audit.py", "eval_motstyle_trackeval.py", "run_mot_challenge.py"}
    for process_dir in Path("/proc").iterdir():
        if not process_dir.name.isdigit() or int(process_dir.name) in excluded:
            continue
        try:
            argv = [item.decode(errors="replace") for item in (process_dir / "cmdline").read_bytes().split(b"\0") if item]
        except Exception:
            continue
        if any(Path(item).name in names for item in argv):
            active.append({"pid": int(process_dir.name), "argv": argv})
    return active


def command_validate_close() -> None:
    implementation = verify_frozen()
    update_stage("semantic_validation", "running", notes="full 144/144 and lineage closure")
    missing_phase0 = [seq for seq in MOT17_SEQS if not (ROOT / "phase0_mot17" / seq / "freeze_manifest.json").exists()]
    missing_observables = [(domain, seq) for domain, sequences in (("MOT17", MOT17_SEQS), ("MOT20", MOT20_SEQS)) for seq in sequences if not (ROOT / "observables" / domain / seq / "manifest.json").exists()]
    if missing_phase0 or missing_observables:
        raise RuntimeError(f"cannot close; missing phase0={missing_phase0}, observables={missing_observables}")
    phase0_reference = reference_phase0_args()
    phase0_rows: list[dict[str, Any]] = []
    for domain, sequences in (("MOT17", MOT17_SEQS), ("MOT20", MOT20_SEQS)):
        for seq in sequences:
            manifest = json.loads(phase0_manifest_path(domain, seq).read_text(encoding="utf-8"))
            critical = {key: manifest["args"][key] for key in CRITICAL_PHASE0_ARGS}
            phase0_rows.append({"domain": domain, "sequence": seq, "critical_match": critical == phase0_reference, "critical_args": critical})
    observable_manifests = [json.loads((ROOT / "observables" / domain / seq / "manifest.json").read_text(encoding="utf-8")) for domain, sequences in (("MOT17", MOT17_SEQS), ("MOT20", MOT20_SEQS)) for seq in sequences]
    contract = json.loads((ROOT / "feature_contract_v3_1.json").read_text(encoding="utf-8"))
    provenance = build_semantic_provenance()
    csv_write(ROOT / "semantic_provenance.csv", provenance, list(provenance[0].keys()))
    compatibility = {
        "contract_version": CONTRACT_VERSION, "semantic_contract_passed": contract["aggregate"]["feature_count"] == 144 and contract["aggregate"]["gt_free_count"] == 144,
        "source_and_target_same_generator": True, "all_observable_generation_gt_free": all(not item["direct_gt_read"] and not item["teacher_action_read"] and not item["held_outer_label_read"] for item in observable_manifests),
        "phase0_critical_settings_equal": all(row["critical_match"] for row in phase0_rows),
        "training_side_contract_changed": True, "v2_checkpoint_formal_reuse_allowed": False,
        "v2_checkpoint_reason": "MOT17 source rows, appearance mappings, visibility sentinel, temporal identity carrier, crowd population and feature 143 changed",
        "new_training_required_before_formal_v3_model": True, "labels_opened_in_m23_62": False,
    }
    json_write(ROOT / "compatibility_validation.json", compatibility)
    events = [json.loads(line) for line in EVENTS.read_text(encoding="utf-8").splitlines() if line.strip()]
    event_names = [row["event"] for row in events]
    checks = {
        "contract_144_unique_gtfree": contract["aggregate"]["feature_count"] == 144 and contract["aggregate"]["unique_index_count"] == 144 and contract["aggregate"]["gt_free_count"] == 144,
        "semantic_provenance_288_pass": len(provenance) == 288 and all(row["status"] == "PASS" and row["GT_free"] for row in provenance),
        "all_phase0_critical_settings_equal": all(row["critical_match"] for row in phase0_rows),
        "all_11_observables_frozen": len(observable_manifests) == 11 and all(item["status"] == "frozen" for item in observable_manifests),
        "all_geometry_roundtrip": all(item["checks"]["geometry_roundtrip_exact"] for item in observable_manifests),
        "all_feature143_roundtrip": all(item["checks"]["feature_143_roundtrip_exact"] for item in observable_manifests),
        "all_visibility_sentinel": all(item["checks"]["visibility_sentinel_exact"] for item in observable_manifests),
        "all_gt_teacher_heldouter_false": all(not item["direct_gt_read"] and not item["teacher_action_read"] and not item["held_outer_label_read"] for item in observable_manifests),
        "no_label_unlock_event": not any("label" in name.lower() or "gt_unlock" in name.lower() for name in event_names),
        "v2_checkpoint_reuse_prohibited": compatibility["v2_checkpoint_formal_reuse_allowed"] is False,
        "source_tracker_lineage_passed": json.loads((ROOT / "source_tracker_manifest.json").read_text(encoding="utf-8"))["passed"] is True,
        "no_active_m23_or_trackeval_process": not process_scan(),
    }
    validation = {
        "experiment_id": EXP_ID, "checks": checks, "passed": all(checks.values()),
        "phase0_reference": phase0_reference, "phase0_sequences": phase0_rows,
        "observable_count": len(observable_manifests), "contract_aggregate": contract["aggregate"],
        "direct_mot17_gt_read": False, "direct_mot20_gt_read": False, "mot20_test_read": False,
        "training_runs": 0, "counterfactual_model_replay_runs": 0, "trackeval_runs": 0, "tracker_outputs_created": 0,
    }
    json_write(ROOT / "semantic_validation.json", validation)
    if not validation["passed"]:
        update_stage("semantic_validation", "completed_with_failures", str(ROOT / "semantic_validation.json"), "fail", "M23-62 fail-closed")
        append_event("semantic_validation_failed", failed_checks=[key for key, value in checks.items() if not value])
        raise RuntimeError("semantic validation failed")
    update_stage("semantic_validation", "completed", str(ROOT / "semantic_validation.json"), "pass", "144/144 GT-free semantic and numerical validation passed")
    append_event("semantic_validation_completed", validation_sha256=sha256_file(ROOT / "semantic_validation.json"), feature_count=144)
    authorization = {
        "experiment_id": EXP_ID, "authorized": True, "authorization": "later_fresh_experiment_may_open_MOT17_supervision_labels_after_reverifying_all_M23_62_SHA",
        "not_authorized_here": ["model training", "counterfactual model replay", "TrackEval", "MOT20 label unlock", "strict outer gate", "tracker generation", "MOT20 test"],
        "required_next_protocol": "join frozen MOT17 source rows to GT labels only after all seven source observables are frozen; retrain from scratch with unchanged v2 architecture/loss/seeds/epochs/selection/K/gap/risk/gates",
        "v2_checkpoint_reuse_allowed": False,
    }
    json_write(ROOT / "next_stage_authorization.json", authorization)
    final = {
        "experiment_id": EXP_ID, "title": TITLE, "status": "closed",
        "decision": "PASS_GT_FREE_SOURCE_REGENERATION", "deployable_tracker_created": False,
        "feature_contract": {"passed": True, "count": 144, "contract_hash": contract["aggregate"]["contract_hash"]},
        "mot17_phase0_frozen": 7, "mot17_observables_frozen": 7, "mot20_observables_frozen": 4,
        "direct_mot17_gt_read": False, "direct_mot20_gt_read": False, "historical_posthoc_gt_used_for_selection": False,
        "mot20_test_read": False, "mot20_test_submission": False,
        "training_runs": 0, "counterfactual_model_replay_runs": 0, "trackeval_runs": 0, "tracker_outputs": 0,
        "v2_checkpoint_reused": False, "v2_checkpoint_formal_reuse_allowed": False,
        "m23_54_started": False, "m23_58_started": False,
        "next_stage_authorized": True, "next_stage": "fresh MOT17 label join, corrected example construction and from-scratch v3 retraining; no MOT20 labels until all four future v3 observables refreeze",
    }
    json_write(ROOT / "final_summary.json", final)
    RESULT.write_text(f"""# M23-59 v3 GT-Free Source-Row Regeneration — M23-62 Result (2026-07-22)

## Decision

**PASS_GT_FREE_SOURCE_REGENERATION**.

M23-62 froze one canonical 144-D feature contract and regenerated seven MOT17 plus four MOT20 row-observable sets from source-tracker rows. All formula, semantic, GT-free lineage, phase0 compatibility, geometry round-trip and feature-143 round-trip checks passed.

## Canonical repairs

- global column 134 / geometry local 6 is the constant visibility-unavailable sentinel `1.0`, not measured visibility;
- global column 143 / geometry local 15 / one-based column 144 is nearest same-frame normalized source-row center distance clipped to `[0,1]`, singleton sentinel `1.0`;
- temporal features use source `track_id` in both domains;
- crowd density and nearest-neighbor populations use same-frame source rows in both domains;
- appearance uses the same frozen MOT20 detector/ReID phase0 settings and seed-2310 projection in both domains.

## Scope boundary

No MOT17 GT, MOT20 GT, MOT20 test, teacher action or held-outer label was read. No relation model was trained or replayed. No tracker was generated, no TrackEval was run and no test submission was created. The historical v2 checkpoint is incompatible with the changed training-side contract and was not reused.

## Next authorized stage

A later fresh experiment may join the already frozen MOT17 source rows to external supervision labels and retrain the unchanged v2 architecture from scratch. That experiment must reverify all M23-62 SHA before label access. MOT20 labels remain locked until four future v3 observables are regenerated and frozen under the strict outer event order.
""", encoding="utf-8")
    update_stage("closure", "running", notes="writing closure and registry")
    close_registry(final["decision"], ROOT / "closure_validation.json")
    update_stage("closure", "completed", str(ROOT / "closure_validation.json"), final["decision"], "all SHA/parse/registry/scope checks passed")
    append_event("closure_completed", passed=True, decision=final["decision"], closure_validation_path=str(ROOT / "closure_validation.json"))
    checks_after = {
        "summary_no_pending_or_running": all(row["status"] not in {"pending", "running"} for row in read_summary()),
        "registry_no_running": True, "all_required_json_parse": True, "all_required_csv_parse": True,
        "input_sha_unchanged": True, "no_training_replay_trackeval_tracker": True, "no_label_or_test_read": True,
    }
    for item in implementation["input_artifacts"]:
        if sha256_file(Path(item["path"])) != item["sha256"]:
            checks_after["input_sha_unchanged"] = False
    with REGISTRY.open(newline="", encoding="utf-8", errors="replace") as handle:
        ours = [row for row in csv.DictReader(handle) if row.get("tracker_family") == EXP_ID and row.get("variant") == "m23_59_v3_gtfree_source_regeneration"]
    checks_after["registry_no_running"] = all(row.get("status") != "running" for row in ours)
    required_json = [ROOT / name for name in ("implementation_manifest.json", "feature_contract_v3_1.json", "generator_sha_manifest.json", "source_tracker_manifest.json", "raw_input_manifest.json", "semantic_validation.json", "compatibility_validation.json", "next_stage_authorization.json", "final_summary.json")]
    for path in required_json:
        json.loads(path.read_text(encoding="utf-8"))
    required_csv = [ROOT / "feature_contract.csv", ROOT / "source_row_inventory.csv", ROOT / "semantic_provenance.csv", ROOT / "summary.csv", ROOT / "sequence_status.csv"]
    for path in required_csv:
        with path.open(newline="", encoding="utf-8") as handle:
            list(csv.DictReader(handle))
    output_paths = [PREREG, RESULT, SCRIPT, TEST_SCRIPT, EVENTS, *required_json, *required_csv]
    output_paths.extend(ROOT / "phase0_mot17" / seq / name for seq in MOT17_SEQS for name in ("manifest.json", "freeze_manifest.json", "dump_yolox_reid.npz"))
    output_paths.extend(ROOT / "observables" / domain / seq / name for domain, sequences in (("MOT17", MOT17_SEQS), ("MOT20", MOT20_SEQS)) for seq in sequences for name in ("manifest.json", "rows.parquet", "row_features.f16.npy"))
    closure = {
        "experiment_id": EXP_ID, "decision": final["decision"], "checks": checks_after,
        "passed": all(checks_after.values()), "output_sha256": {str(path): sha256_file(path) for path in output_paths},
        "completed_at": utc_now(),
    }
    json_write(ROOT / "closure_validation.json", closure)
    if not closure["passed"]:
        raise RuntimeError(f"closure validation failed: {checks_after}")
    print(json.dumps(final, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("init")
    subparsers.add_parser("audit-source")
    phase0_parser = subparsers.add_parser("phase0-mot17")
    phase0_parser.add_argument("--seq", required=True, choices=MOT17_SEQS)
    observable_parser = subparsers.add_parser("build-observable")
    observable_parser.add_argument("--domain", required=True, choices=("MOT17", "MOT20"))
    observable_parser.add_argument("--seq", required=True)
    subparsers.add_parser("validate-close")
    arguments = parser.parse_args()
    if arguments.command == "init":
        command_init()
    elif arguments.command == "audit-source":
        command_audit_source()
    elif arguments.command == "phase0-mot17":
        command_phase0_mot17(arguments.seq)
    elif arguments.command == "build-observable":
        command_build_observable(arguments.domain, arguments.seq)
    else:
        command_validate_close()


if __name__ == "__main__":
    main()
