#!/usr/bin/env python3
from __future__ import annotations

"""M23-47 exact teacher for GT-free two-boundary suffix swaps.

A swap chooses two selected source boundaries from different current chains,
removes both source edges, and cross-connects the two suffixes.  Candidate
construction uses only GT-free structural features.  Exact HOTA is opened only
after the shortlist is frozen.  Results are capacity diagnostics only and are
never deployable.
"""

import argparse
import csv
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
SEQUENCES = ("MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05")
DEFAULT_GRAPH_ROOT = Path("outputs/mot20_m23_20260718/micrograph_chunk30_v1")
DEFAULT_PARENT = Path(
    "outputs/assocriskbench_p15_20260714/fused_a42_adaptive_risk_eval/"
    "l1_r2_q75_adaptive_priority/track_results"
)
SAFE_CROSS_COLUMNS = (
    "src_chunk", "dst_chunk", "src_track", "dst_track", "gap", "log_gap",
    "appearance_cos", "same_source", "source_adjacent",
    "forward_motion_error", "backward_motion_error", "motion_error_min",
    "motion_error_mean", "endpoint_displacement", "velocity_cos",
    "log_height_ratio", "src_rows", "dst_rows", "src_mapping_rate",
    "dst_mapping_rate", "mapping_rate_min", "src_consistency",
    "dst_consistency", "consistency_min", "src_match_iou", "dst_match_iou",
    "out_rank", "in_rank", "max_rank", "out_margin", "in_margin",
    "max_margin",
)
FORBIDDEN_SUBSTRINGS = (
    "same_gt", "modal_gt", "purity", "label_confidence", "ground_truth",
    "exact_hota", "delta_hota", "teacher_seconds",
)


def load_module(name: str, relative_path: str):
    path = REPO / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def percentile(values: pd.Series, higher_is_good: bool = True) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    fill = float(numeric.median()) if numeric.notna().any() else 0.0
    ranked = numeric.fillna(fill).rank(method="average", pct=True)
    return ranked if higher_is_good else 1.0 - ranked


def safe_edge_lookup(edges: pd.DataFrame) -> Dict[Tuple[int, int], pd.Series]:
    safe = edges[(edges.same_source.astype(int) == 0)].copy()
    safe.sort_values(
        ["src_chunk", "dst_chunk", "max_rank", "appearance_cos", "motion_error_min"],
        ascending=[True, True, True, False, True],
        inplace=True,
    )
    lookup: Dict[Tuple[int, int], pd.Series] = {}
    for row in safe.itertuples(index=False):
        key = (int(row.src_chunk), int(row.dst_chunk))
        if key not in lookup:
            lookup[key] = pd.Series(row._asdict())
    return lookup


def edge_rank_context(edges: pd.DataFrame) -> Dict[str, Dict[int, np.ndarray | float]]:
    out_values = {
        int(key): np.sort(group.appearance_cos.to_numpy(float))[::-1]
        for key, group in edges.groupby("src_chunk", sort=False)
    }
    in_values = {
        int(key): np.sort(group.appearance_cos.to_numpy(float))[::-1]
        for key, group in edges.groupby("dst_chunk", sort=False)
    }
    return {
        "out_values": out_values,
        "in_values": in_values,
        "best_out": {key: float(values[0]) for key, values in out_values.items() if len(values)},
        "best_in": {key: float(values[0]) for key, values in in_values.items() if len(values)},
    }


def gt_free_edge_features(a: pd.Series, b: pd.Series, cosine: float) -> pd.Series:
    gap = int(b.first_frame - a.last_frame - 1)
    height = max(0.5 * (float(a.last_h) + float(b.first_h)), 1.0)
    dt = max(int(b.first_frame - a.last_frame), 1)
    predicted_x = float(a.last_cx) + float(a.end_vx) * dt
    predicted_y = float(a.last_cy) + float(a.end_vy) * dt
    backward_x = float(b.first_cx) - float(b.start_vx) * dt
    backward_y = float(b.first_cy) - float(b.start_vy) * dt
    forward_error = math.hypot(float(b.first_cx) - predicted_x, float(b.first_cy) - predicted_y) / height
    backward_error = math.hypot(float(a.last_cx) - backward_x, float(a.last_cy) - backward_y) / height
    displacement = math.hypot(float(b.first_cx) - float(a.last_cx), float(b.first_cy) - float(a.last_cy)) / height
    left_velocity = np.asarray([float(a.end_vx), float(a.end_vy)])
    right_velocity = np.asarray([float(b.start_vx), float(b.start_vy)])
    velocity_cosine = float(
        left_velocity @ right_velocity
        / max(np.linalg.norm(left_velocity) * np.linalg.norm(right_velocity), 1e-8)
    )
    same_source = int(int(a.source_track_id) == int(b.source_track_id))
    source_adjacent = int(
        same_source and int(b.source_ordinal) == int(a.source_ordinal) + 1
    )
    return pd.Series({
        "src_chunk": int(a.chunk_id),
        "dst_chunk": int(b.chunk_id),
        "src_track": int(a.source_track_id),
        "dst_track": int(b.source_track_id),
        "gap": gap,
        "log_gap": math.log1p(max(gap, 0)),
        "appearance_cos": float(cosine),
        "same_source": same_source,
        "source_adjacent": source_adjacent,
        "forward_motion_error": forward_error,
        "backward_motion_error": backward_error,
        "motion_error_min": min(forward_error, backward_error),
        "motion_error_mean": 0.5 * (forward_error + backward_error),
        "endpoint_displacement": displacement,
        "velocity_cos": velocity_cosine,
        "log_height_ratio": math.log(max(float(b.first_h), 1e-3) / max(float(a.last_h), 1e-3)),
        "src_rows": int(a.rows),
        "dst_rows": int(b.rows),
        "src_mapping_rate": float(a.mapping_rate),
        "dst_mapping_rate": float(b.mapping_rate),
        "mapping_rate_min": min(float(a.mapping_rate), float(b.mapping_rate)),
        "src_consistency": float(a.appearance_consistency),
        "dst_consistency": float(b.appearance_consistency),
        "consistency_min": min(float(a.appearance_consistency), float(b.appearance_consistency)),
        "src_match_iou": float(a.mean_match_iou),
        "dst_match_iou": float(b.mean_match_iou),
    })


def get_or_synthesize_edge(
    src_chunk: int,
    dst_chunk: int,
    edge_lookup: Dict[Tuple[int, int], pd.Series],
    meta_by_id: pd.DataFrame,
    prototypes: np.ndarray,
    rank_context: Dict[str, Dict[int, np.ndarray | float]],
    max_cross_gap: int,
) -> pd.Series | None:
    key = (src_chunk, dst_chunk)
    existing = edge_lookup.get(key)
    if existing is not None:
        return existing
    source = meta_by_id.loc[src_chunk]
    destination = meta_by_id.loc[dst_chunk]
    gap = int(destination.first_frame - source.last_frame - 1)
    if gap < 0 or gap > max_cross_gap:
        return None
    cosine = float(prototypes[src_chunk] @ prototypes[dst_chunk])
    edge = gt_free_edge_features(source, destination, cosine)
    out_values = rank_context["out_values"].get(src_chunk, np.asarray([], dtype=float))
    in_values = rank_context["in_values"].get(dst_chunk, np.asarray([], dtype=float))
    edge["out_rank"] = int(1 + np.count_nonzero(out_values > cosine))
    edge["in_rank"] = int(1 + np.count_nonzero(in_values > cosine))
    edge["max_rank"] = max(int(edge.out_rank), int(edge.in_rank))
    best_out = float(rank_context["best_out"].get(src_chunk, cosine))
    best_in = float(rank_context["best_in"].get(dst_chunk, cosine))
    edge["out_margin"] = max(best_out - cosine, 0.0)
    edge["in_margin"] = max(best_in - cosine, 0.0)
    edge["max_margin"] = max(float(edge.out_margin), float(edge.in_margin))
    edge_lookup[key] = edge
    return edge


def current_chain_assignment(applied: pd.DataFrame, chunk_count: int, evaluator) -> Dict[int, int]:
    return evaluator.chains(applied, chunk_count)


def build_swap_candidates(
    applied: pd.DataFrame,
    meta: pd.DataFrame,
    edge_lookup: Dict[Tuple[int, int], pd.Series],
    rank_context: Dict[str, Dict[int, np.ndarray | float]],
    prototypes: np.ndarray,
    max_boundaries: int,
    max_swaps: int,
    max_boundary_offset: int,
    max_cross_gap: int,
    m43,
    evaluator,
) -> pd.DataFrame:
    boundaries = m43.candidate_source_cuts(applied, meta, max_boundaries).copy()
    if len(boundaries) < 2:
        return pd.DataFrame()
    assignment = current_chain_assignment(applied, len(meta), evaluator)
    meta_by_id = meta.set_index("chunk_id", drop=False)
    records: List[Dict[str, object]] = []

    for left_pos in range(len(boundaries)):
        left = boundaries.iloc[left_pos]
        a = int(left.src_chunk)
        b = int(left.dst_chunk)
        left_root = int(assignment[a])
        left_boundary_frame = int(meta_by_id.loc[b, "first_frame"])
        for right_pos in range(left_pos + 1, len(boundaries)):
            right = boundaries.iloc[right_pos]
            c = int(right.src_chunk)
            d = int(right.dst_chunk)
            if left_root == int(assignment[c]):
                continue
            right_boundary_frame = int(meta_by_id.loc[d, "first_frame"])
            boundary_offset = abs(left_boundary_frame - right_boundary_frame)
            if boundary_offset > max_boundary_offset:
                continue
            cross_ad = get_or_synthesize_edge(
                a, d, edge_lookup, meta_by_id, prototypes, rank_context, max_cross_gap
            )
            cross_cb = get_or_synthesize_edge(
                c, b, edge_lookup, meta_by_id, prototypes, rank_context, max_cross_gap
            )
            if cross_ad is None or cross_cb is None:
                continue
            if int(meta_by_id.loc[a, "last_frame"]) >= int(meta_by_id.loc[d, "first_frame"]):
                continue
            if int(meta_by_id.loc[c, "last_frame"]) >= int(meta_by_id.loc[b, "first_frame"]):
                continue

            original_appearance = 0.5 * (float(left.appearance_cos) + float(right.appearance_cos))
            cross_appearance = 0.5 * (float(cross_ad.appearance_cos) + float(cross_cb.appearance_cos))
            original_motion = 0.5 * (float(left.motion_error_mean) + float(right.motion_error_mean))
            cross_motion = 0.5 * (float(cross_ad.motion_error_mean) + float(cross_cb.motion_error_mean))
            original_rank = 0.5 * (float(left.max_rank) + float(right.max_rank))
            cross_rank = 0.5 * (float(cross_ad.max_rank) + float(cross_cb.max_rank))
            original_consistency = min(float(left.consistency_min), float(right.consistency_min))
            cross_consistency = min(float(cross_ad.consistency_min), float(cross_cb.consistency_min))
            original_iou = 0.5 * (float(left.src_match_iou) + float(right.src_match_iou))
            cross_iou = 0.25 * (
                float(cross_ad.src_match_iou) + float(cross_ad.dst_match_iou)
                + float(cross_cb.src_match_iou) + float(cross_cb.dst_match_iou)
            )
            records.append({
                "source_index_a": int(left.source_index),
                "source_index_b": int(right.source_index),
                "a_prefix_chunk": a,
                "a_suffix_chunk": b,
                "b_prefix_chunk": c,
                "b_suffix_chunk": d,
                "cross_ad_src_chunk": a,
                "cross_ad_dst_chunk": d,
                "cross_cb_src_chunk": c,
                "cross_cb_dst_chunk": b,
                "boundary_frame_a": left_boundary_frame,
                "boundary_frame_b": right_boundary_frame,
                "boundary_offset": boundary_offset,
                "left_source_cut_score": float(left.source_cut_policy_score),
                "right_source_cut_score": float(right.source_cut_policy_score),
                "boundary_risk_mean": 0.5 * (
                    float(left.source_cut_policy_score) + float(right.source_cut_policy_score)
                ),
                "boundary_risk_min": min(
                    float(left.source_cut_policy_score), float(right.source_cut_policy_score)
                ),
                "min_side_rows": min(float(left.min_side_rows), float(right.min_side_rows)),
                "impact_rows_sum": float(left.min_side_rows) + float(right.min_side_rows),
                "original_appearance": original_appearance,
                "cross_appearance": cross_appearance,
                "appearance_gain": cross_appearance - original_appearance,
                "original_motion": original_motion,
                "cross_motion": cross_motion,
                "motion_gain": original_motion - cross_motion,
                "original_rank": original_rank,
                "cross_rank": cross_rank,
                "rank_gain": original_rank - cross_rank,
                "original_consistency": original_consistency,
                "cross_consistency": cross_consistency,
                "consistency_gain": cross_consistency - original_consistency,
                "original_match_iou": original_iou,
                "cross_match_iou": cross_iou,
                "match_iou_gain": cross_iou - original_iou,
                "cross_ad_appearance": float(cross_ad.appearance_cos),
                "cross_cb_appearance": float(cross_cb.appearance_cos),
                "cross_ad_motion": float(cross_ad.motion_error_mean),
                "cross_cb_motion": float(cross_cb.motion_error_mean),
                "cross_ad_rank": float(cross_ad.max_rank),
                "cross_cb_rank": float(cross_cb.max_rank),
            })

    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records)
    frame["appearance_gain_pct"] = percentile(frame.appearance_gain, True)
    frame["motion_gain_pct"] = percentile(frame.motion_gain, True)
    frame["rank_gain_pct"] = percentile(frame.rank_gain, True)
    frame["consistency_gain_pct"] = percentile(frame.consistency_gain, True)
    frame["match_iou_gain_pct"] = percentile(frame.match_iou_gain, True)
    frame["boundary_risk_pct"] = percentile(frame.boundary_risk_mean, True)
    frame["impact_pct"] = percentile(frame.min_side_rows, True)
    frame["temporal_alignment_pct"] = percentile(frame.boundary_offset, False)
    frame["cross_floor_appearance"] = frame[["cross_ad_appearance", "cross_cb_appearance"]].min(axis=1)
    frame["cross_floor_appearance_pct"] = percentile(frame.cross_floor_appearance, True)
    frame["swap_policy_score"] = (
        0.23 * frame.appearance_gain_pct
        + 0.18 * frame.motion_gain_pct
        + 0.12 * frame.rank_gain_pct
        + 0.10 * frame.consistency_gain_pct
        + 0.07 * frame.match_iou_gain_pct
        + 0.12 * frame.boundary_risk_pct
        + 0.08 * frame.impact_pct
        + 0.05 * frame.temporal_alignment_pct
        + 0.05 * frame.cross_floor_appearance_pct
    )

    orders = {
        "composite": frame.sort_values(["swap_policy_score", "boundary_offset"], ascending=[False, True]).index.tolist(),
        "appearance_gain": frame.sort_values(["appearance_gain", "cross_floor_appearance"], ascending=[False, False]).index.tolist(),
        "motion_gain": frame.sort_values(["motion_gain", "boundary_risk_mean"], ascending=[False, False]).index.tolist(),
        "rank_gain": frame.sort_values(["rank_gain", "appearance_gain"], ascending=[False, False]).index.tolist(),
        "boundary_risk": frame.sort_values(["boundary_risk_mean", "appearance_gain"], ascending=[False, False]).index.tolist(),
        "temporal": frame.sort_values(["boundary_offset", "swap_policy_score"], ascending=[True, False]).index.tolist(),
        "impact": frame.sort_values(["min_side_rows", "swap_policy_score"], ascending=[False, False]).index.tolist(),
    }
    chosen: List[int] = []
    chosen_set = set()
    channel_rank: Dict[int, Tuple[str, int]] = {}
    cursors = {name: 0 for name in orders}
    active = list(orders)
    while active and len(chosen) < max_swaps:
        progressed = False
        for name in list(active):
            order = orders[name]
            cursor = cursors[name]
            while cursor < len(order) and int(order[cursor]) in chosen_set:
                cursor += 1
            if cursor >= len(order):
                active.remove(name)
                continue
            index = int(order[cursor])
            cursor += 1
            cursors[name] = cursor
            chosen.append(index)
            chosen_set.add(index)
            channel_rank[index] = (name, cursor)
            progressed = True
            if len(chosen) >= max_swaps:
                break
        if not progressed:
            break
    result = frame.loc[chosen].copy()
    result["selection_channel"] = [channel_rank[int(index)][0] for index in result.index]
    result["selection_rank"] = [channel_rank[int(index)][1] for index in result.index]
    result.reset_index(drop=True, inplace=True)
    forbidden = [
        column for column in result.columns
        if any(token in column.lower() for token in FORBIDDEN_SUBSTRINGS)
    ]
    if forbidden:
        raise RuntimeError(f"GT-derived swap candidate columns: {forbidden}")
    return result


def aligned_cross_row(
    applied: pd.DataFrame,
    edge: pd.Series,
    utility: float,
) -> Dict[str, object]:
    row: Dict[str, object] = {column: np.nan for column in applied.columns}
    for column in SAFE_CROSS_COLUMNS:
        if column in edge:
            value = edge[column]
            row[column] = value.item() if isinstance(value, np.generic) else value
    row["edge_role"] = "cross"
    row["utility"] = float(utility)
    for column, value in (
        ("same_gt", 0), ("src_modal_gt", 0), ("dst_modal_gt", 0),
        ("src_purity", 0.0), ("dst_purity", 0.0), ("label_confidence", 0.0),
        ("chain_transaction_delta_proxy", 0.0), ("assa_edge_delta_proxy", 0.0),
        ("assa_edge_positive", 0), ("assa_edge_negative", 0),
    ):
        if column in row:
            row[column] = value
    return row


def apply_swap(
    applied: pd.DataFrame,
    candidate: Mapping[str, object],
    edge_lookup: Dict[Tuple[int, int], pd.Series],
    evaluator,
    chunk_count: int,
) -> pd.DataFrame:
    source_indices = [int(candidate["source_index_a"]), int(candidate["source_index_b"])]
    if source_indices[0] == source_indices[1]:
        raise RuntimeError("same source edge twice")
    retained = applied.drop(index=source_indices).copy()
    key_ad = (int(candidate["cross_ad_src_chunk"]), int(candidate["cross_ad_dst_chunk"]))
    key_cb = (int(candidate["cross_cb_src_chunk"]), int(candidate["cross_cb_dst_chunk"]))
    edge_ad = edge_lookup[key_ad]
    edge_cb = edge_lookup[key_cb]
    utility = max(float(candidate.get("swap_policy_score", 1.0)), 1e-6)
    additions = pd.DataFrame([
        aligned_cross_row(applied, edge_ad, utility),
        aligned_cross_row(applied, edge_cb, utility),
    ], columns=applied.columns)
    modified = pd.concat([retained, additions], ignore_index=True, sort=False)
    evaluator.chains(modified, chunk_count)
    return modified


def compatible_greedy_swaps(
    applied: pd.DataFrame,
    chosen: pd.DataFrame,
    edge_lookup: Dict[Tuple[int, int], pd.Series],
    evaluator,
    chunk_count: int,
) -> Tuple[pd.DataFrame, List[int]]:
    current = applied.copy()
    used_source_indices = set()
    accepted: List[int] = []
    for index, row in chosen.sort_values(["delta_HOTA", "swap_policy_score"], ascending=[False, False]).iterrows():
        indices = {int(row.source_index_a), int(row.source_index_b)}
        if used_source_indices.intersection(indices):
            continue
        try:
            current = apply_swap(current, row, edge_lookup, evaluator, chunk_count)
        except Exception:
            continue
        used_source_indices.update(indices)
        accepted.append(int(index))
    return current, accepted


def metric_delta(metrics: Mapping[str, float], baseline: Mapping[str, float]) -> Dict[str, float]:
    return {
        "exact_HOTA": float(metrics["HOTA"]),
        "exact_DetA": float(metrics["DetA"]),
        "exact_AssA": float(metrics["AssA"]),
        "delta_HOTA": float(metrics["HOTA"] - baseline["HOTA"]),
        "delta_DetA": float(metrics["DetA"] - baseline["DetA"]),
        "delta_AssA": float(metrics["AssA"] - baseline["AssA"]),
        "affected_frames": int(metrics["affected_frames"]),
        "changed_processed_detections": int(metrics["changed_processed_detections"]),
        "teacher_seconds": float(metrics["incremental_seconds"]),
    }


def write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", required=True, choices=SEQUENCES)
    parser.add_argument("--baseline-tracker", required=True)
    parser.add_argument("--baseline-applied", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--graph-root", default=str(DEFAULT_GRAPH_ROOT))
    parser.add_argument("--source-parent", default=str(DEFAULT_PARENT))
    parser.add_argument("--max-boundaries", type=int, default=256)
    parser.add_argument("--max-swaps", type=int, default=64)
    parser.add_argument("--max-boundary-offset", type=int, default=90)
    parser.add_argument("--max-cross-gap", type=int, default=300)
    args = parser.parse_args()

    seq = args.seq
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    protocol = {
        "experiment": "M23-47 exact two-boundary suffix-swap teacher",
        "status": "running",
        "teacher_only": True,
        "deployable": False,
        "sequence": seq,
        "candidate_construction": "GT-free two-boundary and reciprocal cross-edge structural features only",
        "gt_use": "exact HOTA teacher after swap shortlist freeze",
        "action_space": "remove two source edges and cross-connect both suffixes",
        "constraints": "different current chains; reciprocal candidate edges; time-forward; one-to-one; acyclic",
        "max_boundaries": args.max_boundaries,
        "max_swaps": args.max_swaps,
        "max_boundary_offset": args.max_boundary_offset,
        "max_cross_gap": args.max_cross_gap,
        "baseline_tracker": args.baseline_tracker,
        "baseline_applied": args.baseline_applied,
    }
    (output_root / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")

    m37 = load_module("m23_47_exact", "scripts/m23_research/m23_37_fast_exact_hota_teacher.py")
    m38 = load_module("m23_47_helpers", "scripts/m23_research/m23_38_exact_hota_advantage_labels.py")
    m43 = load_module("m23_47_boundaries", "scripts/m23_research/m23_43_source_boundary_cut_teacher.py")
    evaluator = load_module("m23_47_evaluator", "scripts/m23_research/m23_11_eval_utility_graph.py")

    tracker = Path(args.baseline_tracker)
    applied = pd.read_parquet(args.baseline_applied)
    graph_root = Path(args.graph_root)
    meta = pd.read_parquet(graph_root / seq / "microtracklets.parquet")
    candidate_edges = pd.read_parquet(graph_root / seq / "candidate_edges.parquet")
    safe_candidate_edges = candidate_edges[list(SAFE_CROSS_COLUMNS)].copy()
    edge_lookup = safe_edge_lookup(safe_candidate_edges)
    rank_context = edge_rank_context(safe_candidate_edges)
    prototypes = np.load(graph_root / seq / "prototypes.f16.npy").astype(np.float32)
    source_parent = Path(args.source_parent)

    prepared = m37.PreparedExactHOTA(seq, tracker, output_root / "cache")
    source_rows = evaluator.read_parent(source_parent / f"{seq}.txt")
    line_to_chunk = evaluator.line_chunks(source_rows, meta)
    parent_to_source = m38.parent_to_source_indices(prepared, source_rows)
    baseline_ids = np.asarray([int(float(row[1])) for row in prepared.parent_rows], dtype=np.int64)
    reconstructed = m38.row_ids_from_applied(
        applied, meta, line_to_chunk, parent_to_source, SEQUENCES.index(seq), evaluator
    )
    if not np.array_equal(baseline_ids, reconstructed):
        raise RuntimeError(
            f"baseline reconstruction mismatch: {np.count_nonzero(baseline_ids != reconstructed)}"
        )
    baseline_metrics = prepared.evaluate_row_ids_incremental(baseline_ids)

    candidates = build_swap_candidates(
        applied, meta, edge_lookup, rank_context, prototypes,
        args.max_boundaries, args.max_swaps, args.max_boundary_offset,
        args.max_cross_gap, m43, evaluator,
    )
    if candidates.empty:
        raise RuntimeError("no feasible GT-free suffix-swap candidates")
    candidates.to_parquet(output_root / "gt_free_swap_candidates.parquet", index=False)
    print(json.dumps({
        "stage": "shortlist_frozen", "seq": seq, "candidates": len(candidates),
        "gt_fields_used": False,
    }), flush=True)

    labels: List[Dict[str, object]] = []
    for ordinal, row in candidates.iterrows():
        started = time.perf_counter()
        try:
            modified = apply_swap(applied, row, edge_lookup, evaluator, len(meta))
            ids = m38.row_ids_from_applied(
                modified, meta, line_to_chunk, parent_to_source, SEQUENCES.index(seq), evaluator
            )
            metrics = prepared.evaluate_row_ids_incremental(ids)
            record = {
                "status": "success", "seq": seq, "action_type": "two_boundary_suffix_swap",
                "action_ordinal": int(ordinal),
                **{
                    key: (value.item() if isinstance(value, np.generic) else value)
                    for key, value in row.items()
                },
                "changed_raw_rows": int(np.count_nonzero(ids != baseline_ids)),
                **metric_delta(metrics, baseline_metrics),
            }
        except Exception as error:
            record = {
                "status": "failed", "seq": seq, "action_type": "two_boundary_suffix_swap",
                "action_ordinal": int(ordinal), "error": repr(error),
                "teacher_seconds": time.perf_counter() - started,
            }
        labels.append(record)
        pd.DataFrame(labels).to_parquet(output_root / "exact_suffix_swap_labels.parquet", index=False)
        print(json.dumps({
            "stage": "swap_labeled", "ordinal": ordinal + 1, "total": len(candidates),
            "delta_HOTA": record.get("delta_HOTA"), "status": record["status"],
        }), flush=True)

    successful = pd.DataFrame([row for row in labels if row["status"] == "success"])
    policies: List[Dict[str, object]] = []
    best_metrics = baseline_metrics
    best_name = "noop"
    best_applied = applied.copy()
    for threshold in (float("inf"), 0.0, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1):
        chosen = (
            successful[successful.delta_HOTA > threshold]
            if np.isfinite(threshold) else successful.iloc[:0]
        )
        modified, accepted = compatible_greedy_swaps(
            applied, chosen, edge_lookup, evaluator, len(meta)
        )
        ids = m38.row_ids_from_applied(
            modified, meta, line_to_chunk, parent_to_source, SEQUENCES.index(seq), evaluator
        )
        metrics = prepared.evaluate_row_ids_incremental(ids)
        name = "noop" if not np.isfinite(threshold) else f"positive_t{threshold:g}"
        policy = {
            "policy": name,
            "threshold": None if not np.isfinite(threshold) else threshold,
            "candidate_swaps": int(len(chosen)),
            "accepted_swaps": int(len(accepted)),
            "HOTA": float(metrics["HOTA"]),
            "DetA": float(metrics["DetA"]),
            "AssA": float(metrics["AssA"]),
            "delta_HOTA": float(metrics["HOTA"] - baseline_metrics["HOTA"]),
        }
        policies.append(policy)
        print(json.dumps({"stage": "policy_evaluated", **policy}), flush=True)
        if float(metrics["HOTA"]) > float(best_metrics["HOTA"]):
            best_metrics = metrics
            best_name = name
            best_applied = modified

    best_applied.to_parquet(output_root / "best_applied_edges.parquet", index=False)
    evaluator.DATA = graph_root
    evaluator.PARENT = source_parent
    best_tracker = output_root / "best_tracker" / "track_results" / f"{seq}.txt"
    tracker_report = evaluator.write_tracker(seq, meta, best_applied, best_tracker)
    write_csv(output_root / "policy_metrics.csv", policies)
    report = {
        **protocol,
        "status": "completed",
        "stage": "completed",
        "baseline_metrics": {
            key: float(value) for key, value in baseline_metrics.items()
            if isinstance(value, (int, float, np.number))
        },
        "successful_actions": int(len(successful)),
        "positive_actions": int((successful.delta_HOTA > 0.0).sum()),
        "negative_actions": int((successful.delta_HOTA < 0.0).sum()),
        "zero_actions": int((successful.delta_HOTA == 0.0).sum()),
        "best_policy": best_name,
        "best_metrics": {
            key: float(value) for key, value in best_metrics.items()
            if isinstance(value, (int, float, np.number))
        },
        "best_tracker": str(best_tracker),
        "tracker_report": tracker_report,
    }
    (output_root / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
