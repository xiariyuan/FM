#!/usr/bin/env python3
from __future__ import annotations

"""M23-55 GT-free stratified long-gap candidate expansion.

The script has one irreversible protocol boundary: every descriptor, ranking pool,
flow graph, manifest and SHA is written before any GT loader is imported or
called. It never reads held-sequence GT. Post-freeze GT diagnostics and teacher
capacity are implemented in the companion audit script.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
SEQUENCES = ("MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05")
GRAPH_ROOT = Path("outputs/mot20_m23_20260718/micrograph_chunk30_v1")
BASELINE_CACHE = Path("outputs/mot20_m23_20260718/m23_47_m23_46_deployable_baseline_cache_v1")
SOURCE_PARENT = Path("outputs/assocriskbench_p15_20260714/fused_a42_adaptive_risk_eval/l1_r2_q75_adaptive_priority/track_results")
M23_53_ROOTS = {
    "MOT20-01": Path("outputs/mot20_m23_20260718/m23_53_identity_flow_capacity_m01_v1"),
    "MOT20-02": Path("outputs/mot20_m23_20260718/m23_53_identity_flow_capacity_m02_v1"),
    "MOT20-03": Path("outputs/mot20_m23_20260718/m23_53_identity_flow_capacity_m03_v1"),
    "MOT20-05": Path("outputs/mot20_m23_20260718/m23_53_identity_flow_capacity_m05_v1"),
}
GAP_BUCKETS = (("1-30", 1, 30), ("31-90", 31, 90), ("91-180", 91, 180), ("181-600", 181, 600))
VIEW_NAMES = ("tail1_head1", "tail3_head3", "tail8_head8", "quality_endpoint", "robust_whole", "medoid", "whole_mean")
FORBIDDEN_TOKENS = (
    "same_gt", "modal_gt", "purity", "label_confidence", "actual_assa",
    "delta_hota", "matched_gt", "teacher", "mapping_rate", "match_iou",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_module(name: str, rel: str):
    path = REPO / rel
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, 1e-12)


def normalized_mean(x: np.ndarray, weights: np.ndarray | None, fallback: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return fallback.copy()
    if weights is None:
        value = x.mean(axis=0)
    else:
        value = (x * weights[:, None]).sum(axis=0) / max(float(weights.sum()), 1e-12)
    norm = float(np.linalg.norm(value))
    return fallback.copy() if norm <= 1e-12 else (value / norm).astype(np.float32)


def tracker_scores(path: Path) -> np.ndarray:
    values = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split(",")
            if len(fields) >= 7:
                values.append(float(fields[6]))
    return np.asarray(values, np.float32)


def jump_quality(rows: list[dict]) -> np.ndarray:
    jump = np.zeros(len(rows), np.float32)
    by_track: dict[int, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_track[int(row["track_id"])].append(i)
    for ids in by_track.values():
        ids.sort(key=lambda i: (rows[i]["frame"], i))
        for previous, current in zip(ids, ids[1:]):
            a, b = rows[previous], rows[current]
            acx, acy = 0.5 * (a["x1"] + a["x2"]), 0.5 * (a["y1"] + a["y2"])
            bcx, bcy = 0.5 * (b["x1"] + b["x2"]), 0.5 * (b["y1"] + b["y2"])
            h = max(0.5 * ((a["y2"] - a["y1"]) + (b["y2"] - b["y1"])), 1.0)
            dt = max(int(b["frame"]) - int(a["frame"]), 1)
            jump[current] = math.hypot(bcx - acx, bcy - acy) / (h * dt)
    return jump


def build_descriptors(seq: str, nodes: pd.DataFrame, source_parent: Path, graph_root: Path, m53, m10) -> tuple[dict[str, np.ndarray], dict]:
    started = time.time()
    tracker_path = source_parent / f"{seq}.txt"
    rows = m10.read_tracker(tracker_path)
    rows53 = m53.read_tracker_rows(tracker_path)
    if len(rows) != len(rows53):
        raise RuntimeError("tracker reader disagreement")
    rng = np.random.default_rng(m10.SEED)
    projection = rng.normal(size=(2048, m10.DIM)).astype(np.float32) / math.sqrt(m10.DIM)
    phase, match_iou, embedding, position = m10.map_phase(rows, seq, projection)
    row_to_chunk = m53.line_chunks(rows53, nodes)
    scores = tracker_scores(tracker_path)
    if len(scores) != len(rows):
        raise RuntimeError("tracker score count mismatch")
    jump = jump_quality(rows)
    mapped = phase >= 0
    score_floor = float(np.quantile(scores[mapped], 0.10)) if mapped.any() else 0.0
    quality_floor = float(np.quantile((scores[mapped] * match_iou[mapped]), 0.20)) if mapped.any() else 0.0
    jump_ceiling = float(np.quantile(jump[mapped], 0.99)) if mapped.any() else float("inf")
    base = normalize_rows(np.load(graph_root / seq / "prototypes.f16.npy").astype(np.float32))
    n, dim = base.shape
    descriptor = {name: np.zeros((n, dim), np.float32) for name in (
        "head1", "head3", "head8", "tail1", "tail3", "tail8", "qhead", "qtail", "robust", "medoid", "whole"
    )}
    descriptor["whole"][:] = base
    quality_count = np.zeros(n, np.int32)
    mapped_count = np.zeros(n, np.int32)
    by_chunk: dict[int, list[int]] = defaultdict(list)
    for row_index, chunk_id in row_to_chunk.items():
        by_chunk[int(chunk_id)].append(int(row_index))
    for chunk_id in range(n):
        ids = sorted(by_chunk[chunk_id], key=lambda i: (rows[i]["frame"], i))
        valid = [i for i in ids if phase[i] >= 0]
        mapped_count[chunk_id] = len(valid)
        fallback = base[chunk_id]
        if not valid:
            for key in descriptor:
                if key != "whole":
                    descriptor[key][chunk_id] = fallback
            continue
        em = embedding[position[np.asarray(valid, np.int64)]]
        weights = np.clip(scores[valid] * match_iou[valid], 1e-4, None).astype(np.float32)
        good_mask = (scores[valid] >= score_floor) & (weights >= quality_floor) & (jump[valid] <= jump_ceiling)
        good_ids = np.flatnonzero(good_mask)
        quality_count[chunk_id] = int(len(good_ids))
        robust_em = em[good_ids] if len(good_ids) else em
        robust_w = weights[good_ids] if len(good_ids) else weights
        robust = normalized_mean(robust_em, robust_w, fallback)
        descriptor["robust"][chunk_id] = robust
        medoid_scores = em @ robust
        descriptor["medoid"][chunk_id] = em[int(np.argmax(medoid_scores))]
        for k in (1, 3, 8):
            descriptor[f"head{k}"][chunk_id] = normalized_mean(em[:k], weights[:k], fallback)
            descriptor[f"tail{k}"][chunk_id] = normalized_mean(em[-k:], weights[-k:], fallback)
        q = good_ids if len(good_ids) else np.arange(len(valid))
        qhead = q[: min(8, len(q))]
        qtail = q[max(0, len(q) - 8):]
        descriptor["qhead"][chunk_id] = normalized_mean(em[qhead], weights[qhead], fallback)
        descriptor["qtail"][chunk_id] = normalized_mean(em[qtail], weights[qtail], fallback)
    for key in descriptor:
        descriptor[key] = normalize_rows(descriptor[key])
    report = {
        "mapped_rows": int(mapped.sum()),
        "tracker_rows": len(rows),
        "nodes": len(nodes),
        "dimension": int(dim),
        "score_floor_p10": score_floor,
        "quality_floor_p20": quality_floor,
        "jump_ceiling_p99": jump_ceiling,
        "nodes_without_mapped_embedding": int((mapped_count == 0).sum()),
        "nodes_without_quality_filtered_embedding": int((quality_count == 0).sum()),
        "runtime_seconds": time.time() - started,
        "protocol": "existing fixed 128-D phase0 projection; endpoint 1/3/8, quality endpoint, robust whole and medoid; no GT",
    }
    return descriptor, report


def geometry_arrays(nodes: pd.DataFrame) -> dict[str, np.ndarray]:
    names = (
        "first_frame", "last_frame", "first_cx", "first_cy", "last_cx", "last_cy",
        "first_h", "last_h", "start_vx", "start_vy", "end_vx", "end_vy", "source_track_id",
    )
    result = {}
    for name in names:
        dtype = np.int64 if name in ("first_frame", "last_frame", "source_track_id") else np.float32
        result[name] = nodes[name].to_numpy(dtype)
    return result


def torch_tensors(descriptor: dict[str, np.ndarray], geometry: dict[str, np.ndarray], device: torch.device):
    desc = {key: torch.from_numpy(value).to(device=device, dtype=torch.float32) for key, value in descriptor.items()}
    geo = {key: torch.from_numpy(value).to(device=device) for key, value in geometry.items()}
    return desc, geo


def score_matrix(src: torch.Tensor, dst: torch.Tensor, desc: dict[str, torch.Tensor], geo: dict[str, torch.Tensor]):
    views = torch.stack((
        desc["tail1"][src] @ desc["head1"][dst].T,
        desc["tail3"][src] @ desc["head3"][dst].T,
        desc["tail8"][src] @ desc["head8"][dst].T,
        desc["qtail"][src] @ desc["qhead"][dst].T,
        desc["robust"][src] @ desc["robust"][dst].T,
        desc["medoid"][src] @ desc["medoid"][dst].T,
        desc["whole"][src] @ desc["whole"][dst].T,
    ), dim=0)
    multi, best_view = views.max(dim=0)
    whole = views[-1]
    sf = geo["first_frame"][dst][None, :]
    sl = geo["last_frame"][src][:, None]
    gap = sf - sl - 1
    dt = torch.clamp(sf - sl, min=1).float()
    h = torch.clamp(0.5 * (geo["last_h"][src][:, None] + geo["first_h"][dst][None, :]), min=1.0)
    pred_x = geo["last_cx"][src][:, None] + geo["end_vx"][src][:, None] * dt
    pred_y = geo["last_cy"][src][:, None] + geo["end_vy"][src][:, None] * dt
    back_x = geo["first_cx"][dst][None, :] - geo["start_vx"][dst][None, :] * dt
    back_y = geo["first_cy"][dst][None, :] - geo["start_vy"][dst][None, :] * dt
    ferr = torch.sqrt((geo["first_cx"][dst][None, :] - pred_x) ** 2 + (geo["first_cy"][dst][None, :] - pred_y) ** 2) / h
    berr = torch.sqrt((geo["last_cx"][src][:, None] - back_x) ** 2 + (geo["last_cy"][src][:, None] - back_y) ** 2) / h
    motion = torch.minimum(ferr, berr)
    disp = torch.sqrt((geo["first_cx"][dst][None, :] - geo["last_cx"][src][:, None]) ** 2 + (geo["first_cy"][dst][None, :] - geo["last_cy"][src][:, None]) ** 2) / h
    avx, avy = geo["end_vx"][src][:, None], geo["end_vy"][src][:, None]
    bvx, bvy = geo["start_vx"][dst][None, :], geo["start_vy"][dst][None, :]
    velocity = (avx * bvx + avy * bvy) / torch.clamp(torch.sqrt(avx ** 2 + avy ** 2) * torch.sqrt(bvx ** 2 + bvy ** 2), min=1e-8)
    log_height = torch.log(torch.clamp(geo["first_h"][dst][None, :], min=1e-3) / torch.clamp(geo["last_h"][src][:, None], min=1e-3))
    motion_quality = torch.exp(-0.5 * torch.clamp(motion, min=0.0, max=20.0))
    recall_score = multi + 0.15 * motion_quality + 0.03 * velocity - 0.02 * torch.abs(log_height) - 0.005 * torch.log1p(torch.clamp(gap.float(), min=0.0))
    return {
        "gap": gap, "multi": multi, "whole": whole, "best_view": best_view,
        "motion": motion, "disp": disp, "velocity": velocity, "log_height": log_height,
        "score": recall_score,
    }


def append_topk(storage: dict[str, list[np.ndarray]], scores: dict[str, torch.Tensor], src_ids: torch.Tensor, dst_ids: torch.Tensor, mask: torch.Tensor, k: int, direction: str, bucket: str):
    values = scores["score"].masked_fill(~mask, -torch.inf)
    if direction == "out":
        kk = min(k, values.shape[1])
        top_value, top_index = torch.topk(values, kk, dim=1, largest=True, sorted=True)
        valid = torch.isfinite(top_value)
        row = torch.arange(values.shape[0], device=values.device)[:, None].expand_as(top_index)
        src = src_ids[row[valid]]
        dst = dst_ids[top_index[valid]]
        rank = torch.arange(1, kk + 1, device=values.device)[None, :].expand_as(top_index)[valid]
        gather_row, gather_col = row[valid], top_index[valid]
    else:
        kk = min(k, values.shape[0])
        top_value, top_index = torch.topk(values, kk, dim=0, largest=True, sorted=True)
        valid = torch.isfinite(top_value)
        col = torch.arange(values.shape[1], device=values.device)[None, :].expand_as(top_index)
        src = src_ids[top_index[valid]]
        dst = dst_ids[col[valid]]
        rank = torch.arange(1, kk + 1, device=values.device)[:, None].expand_as(top_index)[valid]
        gather_row, gather_col = top_index[valid], col[valid]
    if not valid.any():
        return
    fields = {
        "src_chunk": src, "dst_chunk": dst, "gap": scores["gap"][gather_row, gather_col],
        "rank": rank, "recall_score": scores["score"][gather_row, gather_col],
        "multi_appearance_cos": scores["multi"][gather_row, gather_col],
        "whole_appearance_cos": scores["whole"][gather_row, gather_col],
        "motion_error_min": scores["motion"][gather_row, gather_col],
        "endpoint_displacement": scores["disp"][gather_row, gather_col],
        "velocity_cos": scores["velocity"][gather_row, gather_col],
        "log_height_ratio": scores["log_height"][gather_row, gather_col],
        "best_view": scores["best_view"][gather_row, gather_col],
    }
    for name, tensor in fields.items():
        storage[name].append(tensor.detach().cpu().numpy())
    storage["gap_bucket"].append(np.full(int(valid.sum().item()), bucket, object))


def generate_direction_pool(nodes: pd.DataFrame, descriptor: dict[str, np.ndarray], direction: str, k: int, block_size: int, device: torch.device) -> tuple[pd.DataFrame, dict]:
    started = time.time()
    geometry = geometry_arrays(nodes)
    desc_t, geo_t = torch_tensors(descriptor, geometry, device)
    storage: dict[str, list[np.ndarray]] = defaultdict(list)
    if direction == "out":
        ordered = np.argsort(geometry["last_frame"], kind="mergesort")
        counterpart_order = np.argsort(geometry["first_frame"], kind="mergesort")
        counterpart_time = geometry["first_frame"][counterpart_order]
    else:
        ordered = np.argsort(geometry["first_frame"], kind="mergesort")
        counterpart_order = np.argsort(geometry["last_frame"], kind="mergesort")
        counterpart_time = geometry["last_frame"][counterpart_order]
    peak_candidates = 0
    blocks = 0
    with torch.inference_mode():
        for start in range(0, len(ordered), block_size):
            block_np = ordered[start:start + block_size]
            if direction == "out":
                lo_time = int(geometry["last_frame"][block_np].min()) + 2
                hi_time = int(geometry["last_frame"][block_np].max()) + 601
            else:
                lo_time = int(geometry["first_frame"][block_np].min()) - 601
                hi_time = int(geometry["first_frame"][block_np].max()) - 2
            lo = int(np.searchsorted(counterpart_time, lo_time, side="left"))
            hi = int(np.searchsorted(counterpart_time, hi_time, side="right"))
            candidate_np = counterpart_order[lo:hi]
            if len(candidate_np) == 0:
                continue
            peak_candidates = max(peak_candidates, len(candidate_np))
            if direction == "out":
                src_ids = torch.from_numpy(block_np.astype(np.int64)).to(device)
                dst_ids = torch.from_numpy(candidate_np.astype(np.int64)).to(device)
            else:
                src_ids = torch.from_numpy(candidate_np.astype(np.int64)).to(device)
                dst_ids = torch.from_numpy(block_np.astype(np.int64)).to(device)
            scores = score_matrix(src_ids, dst_ids, desc_t, geo_t)
            cross = geo_t["source_track_id"][src_ids][:, None] != geo_t["source_track_id"][dst_ids][None, :]
            for bucket, minimum, maximum in GAP_BUCKETS:
                mask = cross & (scores["gap"] >= minimum) & (scores["gap"] <= maximum)
                append_topk(storage, scores, src_ids, dst_ids, mask, k, direction, bucket)
            blocks += 1
    result = {}
    for name, values in storage.items():
        result[name] = np.concatenate(values) if values else np.asarray([])
    frame = pd.DataFrame(result)
    if len(frame):
        frame["src_chunk"] = frame.src_chunk.astype(np.int32)
        frame["dst_chunk"] = frame.dst_chunk.astype(np.int32)
        frame["gap"] = frame.gap.astype(np.int16)
        frame["rank"] = frame["rank"].astype(np.int16)
        frame["best_view"] = frame.best_view.astype(np.int8)
        frame.sort_values(["src_chunk" if direction == "out" else "dst_chunk", "gap_bucket", "rank", "src_chunk", "dst_chunk"], kind="mergesort", inplace=True)
        frame.drop_duplicates(["src_chunk", "dst_chunk"], keep="first", inplace=True)
        frame.reset_index(drop=True, inplace=True)
    report = {
        "direction": direction, "rows": len(frame), "top_k": k, "blocks": blocks,
        "peak_counterpart_candidates": peak_candidates, "runtime_seconds": time.time() - started,
    }
    return frame, report


def vectorized_edge_features(nodes: pd.DataFrame, pairs: pd.DataFrame, descriptor: dict[str, np.ndarray]) -> pd.DataFrame:
    src = pairs.src_chunk.to_numpy(np.int64)
    dst = pairs.dst_chunk.to_numpy(np.int64)
    a = nodes.iloc[src]
    b = nodes.iloc[dst]
    gap = b.first_frame.to_numpy(np.int64) - a.last_frame.to_numpy(np.int64) - 1
    dt = np.maximum(b.first_frame.to_numpy(np.float32) - a.last_frame.to_numpy(np.float32), 1.0)
    h = np.maximum(0.5 * (a.last_h.to_numpy(np.float32) + b.first_h.to_numpy(np.float32)), 1.0)
    predx = a.last_cx.to_numpy(np.float32) + a.end_vx.to_numpy(np.float32) * dt
    predy = a.last_cy.to_numpy(np.float32) + a.end_vy.to_numpy(np.float32) * dt
    backx = b.first_cx.to_numpy(np.float32) - b.start_vx.to_numpy(np.float32) * dt
    backy = b.first_cy.to_numpy(np.float32) - b.start_vy.to_numpy(np.float32) * dt
    ferr = np.hypot(b.first_cx.to_numpy(np.float32) - predx, b.first_cy.to_numpy(np.float32) - predy) / h
    berr = np.hypot(a.last_cx.to_numpy(np.float32) - backx, a.last_cy.to_numpy(np.float32) - backy) / h
    disp = np.hypot(b.first_cx.to_numpy(np.float32) - a.last_cx.to_numpy(np.float32), b.first_cy.to_numpy(np.float32) - a.last_cy.to_numpy(np.float32)) / h
    avx, avy = a.end_vx.to_numpy(np.float32), a.end_vy.to_numpy(np.float32)
    bvx, bvy = b.start_vx.to_numpy(np.float32), b.start_vy.to_numpy(np.float32)
    velocity = (avx * bvx + avy * bvy) / np.maximum(np.hypot(avx, avy) * np.hypot(bvx, bvy), 1e-8)
    log_height = np.log(np.maximum(b.first_h.to_numpy(np.float32), 1e-3) / np.maximum(a.last_h.to_numpy(np.float32), 1e-3))
    view_scores = np.stack((
        np.einsum("ij,ij->i", descriptor["tail1"][src], descriptor["head1"][dst]),
        np.einsum("ij,ij->i", descriptor["tail3"][src], descriptor["head3"][dst]),
        np.einsum("ij,ij->i", descriptor["tail8"][src], descriptor["head8"][dst]),
        np.einsum("ij,ij->i", descriptor["qtail"][src], descriptor["qhead"][dst]),
        np.einsum("ij,ij->i", descriptor["robust"][src], descriptor["robust"][dst]),
        np.einsum("ij,ij->i", descriptor["medoid"][src], descriptor["medoid"][dst]),
        np.einsum("ij,ij->i", descriptor["whole"][src], descriptor["whole"][dst]),
    ), axis=1)
    multi = view_scores.max(axis=1)
    whole = view_scores[:, -1]
    score = multi + 0.15 * np.exp(-0.5 * np.clip(np.minimum(ferr, berr), 0, 20)) + 0.03 * velocity - 0.02 * np.abs(log_height) - 0.005 * np.log1p(np.maximum(gap, 0))
    output = pd.DataFrame({
        "src_chunk": src.astype(np.int64), "dst_chunk": dst.astype(np.int64),
        "src_track": a.source_track_id.to_numpy(np.int64), "dst_track": b.source_track_id.to_numpy(np.int64),
        "gap": gap.astype(np.int64), "log_gap": np.log1p(np.maximum(gap, 0)),
        "appearance_cos": multi.astype(np.float32), "same_source": (a.source_track_id.to_numpy(np.int64) == b.source_track_id.to_numpy(np.int64)).astype(np.int8),
        "source_adjacent": np.zeros(len(pairs), np.int8), "forward_motion_error": ferr.astype(np.float32),
        "backward_motion_error": berr.astype(np.float32), "motion_error_min": np.minimum(ferr, berr).astype(np.float32),
        "motion_error_mean": (0.5 * (ferr + berr)).astype(np.float32), "endpoint_displacement": disp.astype(np.float32),
        "velocity_cos": velocity.astype(np.float32), "log_height_ratio": log_height.astype(np.float32),
        "src_rows": a.rows.to_numpy(np.int64), "dst_rows": b.rows.to_numpy(np.int64),
        "src_consistency": a.appearance_consistency.to_numpy(np.float32), "dst_consistency": b.appearance_consistency.to_numpy(np.float32),
        "consistency_min": np.minimum(a.appearance_consistency.to_numpy(np.float32), b.appearance_consistency.to_numpy(np.float32)),
        "out_rank": pairs.out_rank.to_numpy(np.float32), "in_rank": pairs.in_rank.to_numpy(np.float32),
        "max_rank": np.fmax(pairs.out_rank.to_numpy(np.float32), pairs.in_rank.to_numpy(np.float32)),
        "out_margin": np.zeros(len(pairs), np.float32), "in_margin": np.zeros(len(pairs), np.float32), "max_margin": np.zeros(len(pairs), np.float32),
        "edge_role": "cross", "parent_edge": np.zeros(len(pairs), np.int8), "candidate_origin": "m23_55_stratified_nonzero_gap",
        "appearance_out_rank": pairs.out_rank.to_numpy(np.float32), "appearance_in_rank": pairs.in_rank.to_numpy(np.float32),
        "motion_out_rank": pairs.out_rank.to_numpy(np.float32), "motion_in_rank": pairs.in_rank.to_numpy(np.float32),
        "mutual_appearance_topk": ((pairs.out_rank <= 32) & (pairs.in_rank <= 32)).astype(np.int8),
        "mutual_motion_topk": ((pairs.out_rank <= 32) & (pairs.in_rank <= 32)).astype(np.int8),
        "m23_55_recall_score": score.astype(np.float32), "m23_55_multi_appearance_cos": multi.astype(np.float32),
        "m23_55_whole_appearance_cos": whole.astype(np.float32), "m23_55_best_view": view_scores.argmax(axis=1).astype(np.int8),
        "m23_55_gap_bucket": pairs.gap_bucket.astype(str).to_numpy(),
    })
    return output


def audit_columns(columns) -> list[str]:
    return [column for column in columns if any(token in column.lower() for token in FORBIDDEN_TOKENS)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", required=True, choices=SEQUENCES)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--graph-root", default=str(GRAPH_ROOT))
    parser.add_argument("--baseline-cache", default=str(BASELINE_CACHE))
    parser.add_argument("--source-parent", default=str(SOURCE_PARENT))
    parser.add_argument("--ranking-k", type=int, default=256)
    parser.add_argument("--flow-k", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.ranking_k != 256 or args.flow_k != 32:
        raise ValueError("M23-55 v1 is preregistered at ranking_k=256 and flow_k=32")
    output_root = Path(args.output_root)
    if (output_root / "frozen_candidate_graph" / "freeze_manifest.json").exists():
        raise FileExistsError(f"refusing to overwrite frozen run: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    m53 = load_module("m23_55_m53", "scripts/m23_research/m23_53_global_identity_flow_capacity.py")
    m10 = load_module("m23_55_m10", "scripts/m23_research/m23_10_build_micrograph.py")
    graph_root, baseline_cache, source_parent = Path(args.graph_root), Path(args.baseline_cache), Path(args.source_parent)
    nodes_path = graph_root / args.seq / "microtracklets.parquet"
    source_tracker = source_parent / f"{args.seq}.txt"
    cache_tracker = baseline_cache / args.seq / "track_results" / f"{args.seq}.txt"
    parent_applied_path = baseline_cache / args.seq / "frozen_applied_edges.parquet"
    old_root = M23_53_ROOTS[args.seq]
    old_edges_path = old_root / "frozen_candidate_graph" / "edges.parquet"
    for path in (nodes_path, source_tracker, cache_tracker, parent_applied_path, old_edges_path, graph_root / args.seq / "prototypes.f16.npy"):
        if not path.exists():
            raise FileNotFoundError(path)
    nodes = m53.read_allowlisted_parquet(nodes_path, m53.NODE_ALLOWLIST)
    nodes = nodes.sort_values("chunk_id", kind="mergesort").reset_index(drop=True)
    forbidden_nodes = audit_columns(nodes.columns)
    if forbidden_nodes:
        raise RuntimeError(f"forbidden node columns: {forbidden_nodes}")
    parent_applied = m53.read_allowlisted_parquet(parent_applied_path, m53.EDGE_ALLOWLIST)
    reconstructed = output_root / "baseline_reconstruction" / "track_results" / f"{args.seq}.txt"
    baseline_report = m53.write_tracker(args.seq, source_tracker, nodes, parent_applied, reconstructed, preserve_parent_ids="parent_tracker_id" in nodes.columns)
    byte_exact = cache_tracker.read_bytes() == reconstructed.read_bytes()
    if not byte_exact:
        raise RuntimeError("M23-46 baseline reconstruction is not byte-exact")
    descriptor, descriptor_report = build_descriptors(args.seq, nodes, source_parent, graph_root, m53, m10)
    frozen_dir = output_root / "frozen_candidate_graph"
    frozen_dir.mkdir(parents=True, exist_ok=True)
    descriptor_path = frozen_dir / "observable_descriptors.npz"
    np.savez_compressed(descriptor_path, **{key: value.astype(np.float16) for key, value in descriptor.items()})
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    outgoing, outgoing_report = generate_direction_pool(nodes, descriptor, "out", args.ranking_k, args.block_size, device)
    incoming, incoming_report = generate_direction_pool(nodes, descriptor, "in", args.ranking_k, args.block_size, device)
    outgoing_path = frozen_dir / "outgoing_ranking_pool.parquet"
    incoming_path = frozen_dir / "incoming_ranking_pool.parquet"
    outgoing.to_parquet(outgoing_path, index=False)
    incoming.to_parquet(incoming_path, index=False)
    out_flow = outgoing[outgoing["rank"] <= args.flow_k][["src_chunk", "dst_chunk", "gap_bucket", "rank"]].rename(columns={"rank": "out_rank"})
    in_flow = incoming[incoming["rank"] <= args.flow_k][["src_chunk", "dst_chunk", "gap_bucket", "rank"]].rename(columns={"rank": "in_rank"})
    flow_pairs = out_flow.merge(in_flow, on=["src_chunk", "dst_chunk", "gap_bucket"], how="outer")
    flow_pairs["out_rank"] = flow_pairs.out_rank.fillna(np.inf)
    flow_pairs["in_rank"] = flow_pairs.in_rank.fillna(np.inf)
    flow_pairs.drop_duplicates(["src_chunk", "dst_chunk"], keep="first", inplace=True)
    new_edges = vectorized_edge_features(nodes, flow_pairs, descriptor)
    old_edges = pd.read_parquet(old_edges_path)
    forbidden_old = audit_columns(old_edges.columns)
    if forbidden_old:
        raise RuntimeError(f"old frozen graph unexpectedly contains forbidden columns: {forbidden_old}")
    old_edges["m23_55_recall_score"] = np.nan
    old_edges["m23_55_multi_appearance_cos"] = np.nan
    old_edges["m23_55_whole_appearance_cos"] = np.nan
    old_edges["m23_55_best_view"] = np.int8(-1)
    old_edges["m23_55_gap_bucket"] = np.where(old_edges.gap.to_numpy(int) == 0, "0", "legacy_nonzero")
    edges = pd.concat([old_edges, new_edges], ignore_index=True, sort=False)
    edges.sort_values(["parent_edge", "src_chunk", "dst_chunk"], ascending=[False, True, True], kind="mergesort", inplace=True)
    edges.drop_duplicates(["src_chunk", "dst_chunk"], keep="first", inplace=True)
    edges.sort_values(["src_chunk", "dst_chunk"], kind="mergesort", inplace=True)
    edges.reset_index(drop=True, inplace=True)
    forbidden_edges = audit_columns(edges.columns)
    if forbidden_edges:
        raise RuntimeError(f"forbidden edge columns: {forbidden_edges}")
    first = nodes.first_frame.to_numpy(int)
    last = nodes.last_frame.to_numpy(int)
    if np.any(first[edges.dst_chunk.to_numpy(int)] <= last[edges.src_chunk.to_numpy(int)]):
        raise RuntimeError("non-forward edge in frozen flow graph")
    nodes_frozen = frozen_dir / "nodes.parquet"
    edges_frozen = frozen_dir / "edges.parquet"
    nodes.to_parquet(nodes_frozen, index=False)
    edges.to_parquet(edges_frozen, index=False)
    freeze_manifest = {
        "experiment": "M23-55 Stratified Long-Gap Candidate Recall Expansion",
        "stage": "candidate_graph_frozen_before_gt",
        "seq": args.seq,
        "teacher_only": True,
        "deployable": False,
        "candidate_graph_frozen": True,
        "gt_opened": False,
        "frozen_at": utc_now(),
        "git_head_at_freeze": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "protocol": {
            "parent": "all frozen M23-53 source/cross edges retained; M23-46 parent retained",
            "gap_buckets": [{"name": name, "minimum": lo, "maximum": hi} for name, lo, hi in GAP_BUCKETS],
            "gap0_consumes_nonzero_quota": False,
            "source_edges_consume_cross_quota": False,
            "directions": ["outgoing", "incoming"],
            "ranking_k": args.ranking_k,
            "flow_k": args.flow_k,
            "ranking_score": "max observable multi-view cosine + fixed motion continuity terms; no learned or GT-derived threshold",
            "views": list(VIEW_NAMES),
            "motion": "constant-velocity forward/backward residual, endpoint displacement, velocity consistency, height ratio; no hard held-fold gate",
            "dummy": "implicit zero-weight terminate/restart inherited from M23-53",
            "teacher": "unchanged M23-53 teacher weights, source/cross/dummy semantics, solver and reconstruction",
        },
        "baseline_reconstruction": {
            **baseline_report,
            "byte_exact": byte_exact,
            "cached_tracker_sha256": sha256_file(cache_tracker),
            "reconstructed_tracker_sha256": sha256_file(reconstructed),
        },
        "descriptor_report": descriptor_report,
        "ranking_reports": {"outgoing": outgoing_report, "incoming": incoming_report},
        "inputs": {
            "nodes": str(nodes_path), "prototypes": str(graph_root / args.seq / "prototypes.f16.npy"),
            "source_parent": str(source_tracker), "m23_46_tracker": str(cache_tracker),
            "m23_46_applied": str(parent_applied_path), "m23_53_edges": str(old_edges_path),
        },
        "frozen_artifacts": {
            "nodes": str(nodes_frozen), "nodes_sha256": sha256_file(nodes_frozen), "node_rows": len(nodes),
            "edges": str(edges_frozen), "edges_sha256": sha256_file(edges_frozen), "edge_rows": len(edges),
            "outgoing_pool": str(outgoing_path), "outgoing_pool_sha256": sha256_file(outgoing_path), "outgoing_rows": len(outgoing),
            "incoming_pool": str(incoming_path), "incoming_pool_sha256": sha256_file(incoming_path), "incoming_rows": len(incoming),
            "descriptors": str(descriptor_path), "descriptors_sha256": sha256_file(descriptor_path),
            "parent_edges": int(edges.parent_edge.fillna(0).sum()),
            "new_stratified_edges": int((edges.candidate_origin == "m23_55_stratified_nonzero_gap").sum()),
            "node_columns": list(nodes.columns), "edge_columns": list(edges.columns),
            "forbidden_node_columns": forbidden_nodes, "forbidden_edge_columns": forbidden_edges,
        },
        "runtime_seconds": time.time() - started,
    }
    manifest_path = frozen_dir / "freeze_manifest.json"
    json_write(manifest_path, freeze_manifest)
    with (output_root / "protocol_events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time": utc_now(), "event": "candidate_graph_frozen", "gt_opened": False, "manifest": str(manifest_path), "manifest_sha256": sha256_file(manifest_path)}, sort_keys=True) + "\n")
    print(json.dumps({
        "seq": args.seq, "manifest": str(manifest_path), "nodes": len(nodes), "flow_edges": len(edges),
        "outgoing_pool": len(outgoing), "incoming_pool": len(incoming), "byte_exact": byte_exact,
        "runtime_seconds": freeze_manifest["runtime_seconds"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
