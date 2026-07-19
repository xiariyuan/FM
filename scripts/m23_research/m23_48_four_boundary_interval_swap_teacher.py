#!/usr/bin/env python3
from __future__ import annotations

"""Exact-HOTA capacity teacher for GT-free four-boundary interval exchange.

Each action chooses two time-aligned intervals from two different current chains.
It removes the two source boundaries around each interval and exchanges the two
middle segments. Candidate construction uses only GT-free graph, appearance,
motion, timing, and chain-structure fields. GT is opened only after the
shortlist has been frozen, solely for exact HOTA teacher labels.
"""

import argparse
import json
import time
from collections import defaultdict
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
FORBIDDEN_SUBSTRINGS = (
    "same_gt", "modal_gt", "purity", "label_confidence", "teacher",
    "exact_hota", "delta_hota", "assa_edge", "actual_assa",
)


def chain_paths(applied: pd.DataFrame, chunk_count: int) -> Tuple[Dict[int, List[int]], Dict[int, int], Dict[int, int]]:
    successor = {int(row.src_chunk): int(row.dst_chunk) for row in applied.itertuples()}
    predecessor = {dst: src for src, dst in successor.items()}
    if len(successor) != len(applied) or len(predecessor) != len(applied):
        raise RuntimeError("selected graph is not one-to-one")
    roots = [chunk for chunk in range(chunk_count) if chunk not in predecessor]
    paths: Dict[int, List[int]] = {}
    root_of: Dict[int, int] = {}
    position: Dict[int, int] = {}
    for root in roots:
        path: List[int] = []
        current = root
        seen = set()
        while current not in seen:
            seen.add(current)
            position[current] = len(path)
            root_of[current] = root
            path.append(current)
            if current not in successor:
                break
            current = successor[current]
        else:
            raise RuntimeError("cycle in selected graph")
        paths[root] = path
    if len(root_of) != chunk_count:
        raise RuntimeError(f"unassigned chunks: {chunk_count-len(root_of)}")
    return paths, root_of, position


def percentile(values: pd.Series, higher_is_good: bool = True) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    fill = float(numeric.median()) if numeric.notna().any() else 0.0
    rank = numeric.fillna(fill).rank(method="average", pct=True)
    return rank if higher_is_good else 1.0 - rank


def interval_bank(
    applied: pd.DataFrame,
    meta: pd.DataFrame,
    max_boundaries: int,
    max_intervals: int,
    max_interval_chunks: int,
    max_interval_frames: int,
    min_interval_rows: int,
    m43,
) -> pd.DataFrame:
    boundaries = m43.candidate_source_cuts(applied, meta, max_boundaries).copy()
    if len(boundaries) < 4:
        return pd.DataFrame()
    paths, root_of, position = chain_paths(applied, len(meta))
    meta_by_id = meta.set_index("chunk_id", drop=False)
    groups: Dict[int, List[pd.Series]] = defaultdict(list)
    for _, row in boundaries.iterrows():
        groups[int(root_of[int(row.src_chunk)])].append(row)

    records: List[Dict[str, object]] = []
    for root, rows in groups.items():
        rows.sort(key=lambda row: position[int(row.dst_chunk)])
        path = paths[root]
        for left_index, left in enumerate(rows):
            left_dst_pos = position[int(left.dst_chunk)]
            for right in rows[left_index + 1:]:
                right_src_pos = position[int(right.src_chunk)]
                if right_src_pos < left_dst_pos:
                    continue
                interval_chunks = right_src_pos - left_dst_pos + 1
                if interval_chunks > max_interval_chunks:
                    break
                chunks = path[left_dst_pos:right_src_pos + 1]
                start_frame = int(meta_by_id.loc[chunks[0], "first_frame"])
                end_frame = int(meta_by_id.loc[chunks[-1], "last_frame"])
                interval_frames = end_frame - start_frame + 1
                if interval_frames > max_interval_frames:
                    continue
                interval_rows = int(meta_by_id.loc[chunks, "rows"].sum())
                if interval_rows < min_interval_rows:
                    continue
                records.append({
                    "chain_root": int(root),
                    "left_source_index": int(left.source_index),
                    "right_source_index": int(right.source_index),
                    "prefix_chunk": int(left.src_chunk),
                    "interval_start_chunk": int(left.dst_chunk),
                    "interval_end_chunk": int(right.src_chunk),
                    "suffix_chunk": int(right.dst_chunk),
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "interval_frames": interval_frames,
                    "interval_rows": interval_rows,
                    "interval_chunks": interval_chunks,
                    "left_boundary_risk": float(left.source_cut_policy_score),
                    "right_boundary_risk": float(right.source_cut_policy_score),
                    "boundary_risk_mean": 0.5 * (
                        float(left.source_cut_policy_score) + float(right.source_cut_policy_score)
                    ),
                    "boundary_risk_min": min(
                        float(left.source_cut_policy_score), float(right.source_cut_policy_score)
                    ),
                    "left_appearance": float(left.appearance_cos),
                    "right_appearance": float(right.appearance_cos),
                    "original_appearance": 0.5 * (float(left.appearance_cos) + float(right.appearance_cos)),
                    "original_motion": 0.5 * (float(left.motion_error_mean) + float(right.motion_error_mean)),
                    "original_rank": 0.5 * (float(left.max_rank) + float(right.max_rank)),
                    "original_consistency": min(float(left.consistency_min), float(right.consistency_min)),
                    "original_match_iou": 0.25 * (
                        float(left.src_match_iou) + float(left.dst_match_iou)
                        + float(right.src_match_iou) + float(right.dst_match_iou)
                    ),
                })
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records)
    frame["risk_pct"] = percentile(frame.boundary_risk_mean, True)
    frame["rows_pct"] = percentile(frame.interval_rows, True)
    frame["compact_pct"] = percentile(frame.interval_frames, False)
    frame["interval_policy_score"] = 0.55 * frame.risk_pct + 0.25 * frame.rows_pct + 0.20 * frame.compact_pct

    orders = {
        "risk": frame.sort_values(["boundary_risk_mean", "interval_rows"], ascending=[False, False]).index.tolist(),
        "impact": frame.sort_values(["interval_rows", "boundary_risk_mean"], ascending=[False, False]).index.tolist(),
        "compact": frame.sort_values(["interval_frames", "boundary_risk_mean"], ascending=[True, False]).index.tolist(),
        "composite": frame.sort_values(["interval_policy_score", "interval_frames"], ascending=[False, True]).index.tolist(),
    }
    chosen: List[int] = []
    chosen_set = set()
    cursors = {name: 0 for name in orders}
    active = list(orders)
    while active and len(chosen) < max_intervals:
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
            cursors[name] = cursor + 1
            chosen.append(index)
            chosen_set.add(index)
            progressed = True
            if len(chosen) >= max_intervals:
                break
        if not progressed:
            break
    return frame.loc[chosen].reset_index(drop=True)


def build_candidates(
    applied: pd.DataFrame,
    meta: pd.DataFrame,
    edge_lookup: Dict[Tuple[int, int], pd.Series],
    rank_context,
    prototypes: np.ndarray,
    max_boundaries: int,
    max_intervals: int,
    max_actions: int,
    max_interval_chunks: int,
    max_interval_frames: int,
    min_interval_rows: int,
    max_boundary_offset: int,
    min_span_ratio: float,
    max_cross_gap: int,
    m43,
    m47,
) -> pd.DataFrame:
    intervals = interval_bank(
        applied, meta, max_boundaries, max_intervals, max_interval_chunks,
        max_interval_frames, min_interval_rows, m43,
    )
    if len(intervals) < 2:
        return pd.DataFrame()
    meta_by_id = meta.set_index("chunk_id", drop=False)
    records: List[Dict[str, object]] = []
    for left_index in range(len(intervals)):
        left = intervals.iloc[left_index]
        for right_index in range(left_index + 1, len(intervals)):
            right = intervals.iloc[right_index]
            if int(left.chain_root) == int(right.chain_root):
                continue
            start_offset = abs(int(left.start_frame) - int(right.start_frame))
            end_offset = abs(int(left.end_frame) - int(right.end_frame))
            if start_offset > max_boundary_offset or end_offset > max_boundary_offset:
                continue
            span_ratio = min(float(left.interval_frames), float(right.interval_frames)) / max(
                float(left.interval_frames), float(right.interval_frames), 1.0
            )
            if span_ratio < min_span_ratio:
                continue

            # A: a -> [b ... c] -> d; B: e -> [f ... g] -> h.
            a, b, c, d = (int(left.prefix_chunk), int(left.interval_start_chunk), int(left.interval_end_chunk), int(left.suffix_chunk))
            e, f, g, h = (int(right.prefix_chunk), int(right.interval_start_chunk), int(right.interval_end_chunk), int(right.suffix_chunk))
            keys = ((a, f), (g, d), (e, b), (c, h))
            cross = [
                m47.get_or_synthesize_edge(src, dst, edge_lookup, meta_by_id, prototypes, rank_context, max_cross_gap)
                for src, dst in keys
            ]
            if any(edge is None for edge in cross):
                continue
            cross_edges = [edge for edge in cross if edge is not None]
            cross_appearance = float(np.mean([float(edge.appearance_cos) for edge in cross_edges]))
            cross_motion = float(np.mean([float(edge.motion_error_mean) for edge in cross_edges]))
            cross_rank = float(np.mean([float(edge.max_rank) for edge in cross_edges]))
            cross_consistency = float(min(float(edge.consistency_min) for edge in cross_edges))
            cross_match_iou = float(np.mean([
                0.5 * (float(edge.src_match_iou) + float(edge.dst_match_iou))
                for edge in cross_edges
            ]))
            original_appearance = 0.5 * (float(left.original_appearance) + float(right.original_appearance))
            original_motion = 0.5 * (float(left.original_motion) + float(right.original_motion))
            original_rank = 0.5 * (float(left.original_rank) + float(right.original_rank))
            original_consistency = min(float(left.original_consistency), float(right.original_consistency))
            original_match_iou = 0.5 * (float(left.original_match_iou) + float(right.original_match_iou))
            source_indices = (
                int(left.left_source_index), int(left.right_source_index),
                int(right.left_source_index), int(right.right_source_index),
            )
            if len(set(source_indices)) != 4:
                continue
            records.append({
                "a_left_source_index": source_indices[0],
                "a_right_source_index": source_indices[1],
                "b_left_source_index": source_indices[2],
                "b_right_source_index": source_indices[3],
                "a_chain_root": int(left.chain_root),
                "b_chain_root": int(right.chain_root),
                "a_prefix_chunk": a,
                "a_interval_start_chunk": b,
                "a_interval_end_chunk": c,
                "a_suffix_chunk": d,
                "b_prefix_chunk": e,
                "b_interval_start_chunk": f,
                "b_interval_end_chunk": g,
                "b_suffix_chunk": h,
                "cross_1_src": a, "cross_1_dst": f,
                "cross_2_src": g, "cross_2_dst": d,
                "cross_3_src": e, "cross_3_dst": b,
                "cross_4_src": c, "cross_4_dst": h,
                "start_offset": start_offset,
                "end_offset": end_offset,
                "offset_mean": 0.5 * (start_offset + end_offset),
                "span_ratio": span_ratio,
                "row_ratio": min(float(left.interval_rows), float(right.interval_rows)) / max(
                    float(left.interval_rows), float(right.interval_rows), 1.0
                ),
                "interval_rows_sum": float(left.interval_rows) + float(right.interval_rows),
                "interval_frames_mean": 0.5 * (float(left.interval_frames) + float(right.interval_frames)),
                "boundary_risk_mean": 0.5 * (float(left.boundary_risk_mean) + float(right.boundary_risk_mean)),
                "boundary_risk_min": min(float(left.boundary_risk_min), float(right.boundary_risk_min)),
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
                "original_match_iou": original_match_iou,
                "cross_match_iou": cross_match_iou,
                "match_iou_gain": cross_match_iou - original_match_iou,
                "cross_floor_appearance": min(float(edge.appearance_cos) for edge in cross_edges),
                "cross_max_motion": max(float(edge.motion_error_mean) for edge in cross_edges),
            })
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records)
    for column, higher in (
        ("appearance_gain", True), ("motion_gain", True), ("rank_gain", True),
        ("consistency_gain", True), ("match_iou_gain", True),
        ("boundary_risk_mean", True), ("interval_rows_sum", True),
        ("offset_mean", False), ("span_ratio", True), ("row_ratio", True),
        ("cross_floor_appearance", True), ("cross_max_motion", False),
    ):
        frame[column + "_pct"] = percentile(frame[column], higher)
    frame["interval_swap_policy_score"] = (
        0.18 * frame.appearance_gain_pct
        + 0.15 * frame.motion_gain_pct
        + 0.08 * frame.rank_gain_pct
        + 0.07 * frame.consistency_gain_pct
        + 0.05 * frame.match_iou_gain_pct
        + 0.13 * frame.boundary_risk_mean_pct
        + 0.09 * frame.interval_rows_sum_pct
        + 0.07 * frame.offset_mean_pct
        + 0.05 * frame.span_ratio_pct
        + 0.04 * frame.row_ratio_pct
        + 0.05 * frame.cross_floor_appearance_pct
        + 0.04 * frame.cross_max_motion_pct
    )
    orders = {
        "composite": frame.sort_values(["interval_swap_policy_score", "offset_mean"], ascending=[False, True]).index.tolist(),
        "appearance": frame.sort_values(["appearance_gain", "cross_floor_appearance"], ascending=[False, False]).index.tolist(),
        "motion": frame.sort_values(["motion_gain", "cross_max_motion"], ascending=[False, True]).index.tolist(),
        "risk": frame.sort_values(["boundary_risk_mean", "interval_rows_sum"], ascending=[False, False]).index.tolist(),
        "alignment": frame.sort_values(["offset_mean", "span_ratio"], ascending=[True, False]).index.tolist(),
        "impact": frame.sort_values(["interval_rows_sum", "interval_swap_policy_score"], ascending=[False, False]).index.tolist(),
    }
    chosen: List[int] = []
    chosen_set = set()
    channel: Dict[int, Tuple[str, int]] = {}
    cursors = {name: 0 for name in orders}
    active = list(orders)
    while active and len(chosen) < max_actions:
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
            cursors[name] = cursor + 1
            chosen.append(index)
            chosen_set.add(index)
            channel[index] = (name, cursor + 1)
            progressed = True
            if len(chosen) >= max_actions:
                break
        if not progressed:
            break
    result = frame.loc[chosen].copy()
    result["selection_channel"] = [channel[int(index)][0] for index in result.index]
    result["selection_rank"] = [channel[int(index)][1] for index in result.index]
    result.reset_index(drop=True, inplace=True)
    forbidden = [column for column in result.columns if any(token in column.lower() for token in FORBIDDEN_SUBSTRINGS)]
    if forbidden:
        raise RuntimeError(f"GT-derived interval-swap candidate columns: {forbidden}")
    return result


def apply_action(applied: pd.DataFrame, candidate: Mapping[str, object], edge_lookup, evaluator, chunk_count: int, m47) -> pd.DataFrame:
    source_indices = [
        int(candidate["a_left_source_index"]), int(candidate["a_right_source_index"]),
        int(candidate["b_left_source_index"]), int(candidate["b_right_source_index"]),
    ]
    if len(set(source_indices)) != 4:
        raise RuntimeError("interval swap requires four distinct source edges")
    retained = applied.drop(index=source_indices).copy()
    keys = [
        (int(candidate["cross_1_src"]), int(candidate["cross_1_dst"])),
        (int(candidate["cross_2_src"]), int(candidate["cross_2_dst"])),
        (int(candidate["cross_3_src"]), int(candidate["cross_3_dst"])),
        (int(candidate["cross_4_src"]), int(candidate["cross_4_dst"])),
    ]
    utility = max(float(candidate.get("interval_swap_policy_score", 1.0)), 1e-6)
    additions = pd.DataFrame(
        [m47.aligned_cross_row(applied, edge_lookup[key], utility) for key in keys],
        columns=applied.columns,
    )
    modified = pd.concat([retained, additions], ignore_index=True, sort=False)
    evaluator.chains(modified, chunk_count)
    return modified


def compatible_greedy(applied: pd.DataFrame, chosen: pd.DataFrame, edge_lookup, evaluator, chunk_count: int, m47):
    current = applied.copy()
    used_indices = set()
    used_endpoints = set()
    accepted: List[int] = []
    for index, row in chosen.sort_values(["delta_HOTA", "interval_swap_policy_score"], ascending=[False, False]).iterrows():
        indices = {
            int(row.a_left_source_index), int(row.a_right_source_index),
            int(row.b_left_source_index), int(row.b_right_source_index),
        }
        endpoints = {
            int(row.a_prefix_chunk), int(row.a_interval_start_chunk), int(row.a_interval_end_chunk), int(row.a_suffix_chunk),
            int(row.b_prefix_chunk), int(row.b_interval_start_chunk), int(row.b_interval_end_chunk), int(row.b_suffix_chunk),
        }
        if used_indices.intersection(indices) or used_endpoints.intersection(endpoints):
            continue
        try:
            current = apply_action(current, row, edge_lookup, evaluator, chunk_count, m47)
        except Exception:
            continue
        used_indices.update(indices)
        used_endpoints.update(endpoints)
        accepted.append(int(index))
    return current, accepted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", required=True, choices=SEQUENCES)
    parser.add_argument("--baseline-tracker", required=True)
    parser.add_argument("--baseline-applied", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--graph-root", default=str(DEFAULT_GRAPH_ROOT))
    parser.add_argument("--source-parent", default=str(DEFAULT_PARENT))
    parser.add_argument("--max-boundaries", type=int, default=256)
    parser.add_argument("--max-intervals", type=int, default=256)
    parser.add_argument("--max-actions", type=int, default=64)
    parser.add_argument("--max-interval-chunks", type=int, default=20)
    parser.add_argument("--max-interval-frames", type=int, default=600)
    parser.add_argument("--min-interval-rows", type=int, default=30)
    parser.add_argument("--max-boundary-offset", type=int, default=60)
    parser.add_argument("--min-span-ratio", type=float, default=0.5)
    parser.add_argument("--max-cross-gap", type=int, default=300)
    args = parser.parse_args()

    seq = args.seq
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    protocol = {
        "experiment": "M23-48 exact four-boundary interval-swap teacher",
        "status": "running",
        "teacher_only": True,
        "deployable": False,
        "sequence": seq,
        "candidate_construction": "GT-free chain intervals, projected appearance, motion, timing and structural features only",
        "gt_use": "exact HOTA teacher after interval-swap shortlist freeze",
        "action_space": "remove four source boundaries and exchange two aligned middle intervals",
        "constraints": "different chains; four synthesized time-forward edges; one-to-one; acyclic",
        **{key: value for key, value in vars(args).items() if key not in {"seq", "baseline_tracker", "baseline_applied", "output_root", "graph_root", "source_parent"}},
        "baseline_tracker": args.baseline_tracker,
        "baseline_applied": args.baseline_applied,
    }
    (output_root / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")

    import importlib.util
    import sys

    def load(name: str, relative: str):
        spec = importlib.util.spec_from_file_location(name, REPO / relative)
        if spec is None or spec.loader is None:
            raise RuntimeError(relative)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    m37 = load("m23_48_exact", "scripts/m23_research/m23_37_fast_exact_hota_teacher.py")
    m38 = load("m23_48_helpers", "scripts/m23_research/m23_38_exact_hota_advantage_labels.py")
    m43 = load("m23_48_boundaries", "scripts/m23_research/m23_43_source_boundary_cut_teacher.py")
    m47 = load("m23_48_swap_helpers", "scripts/m23_research/m23_47_two_boundary_suffix_swap_teacher.py")
    evaluator = load("m23_48_evaluator", "scripts/m23_research/m23_11_eval_utility_graph.py")

    tracker = Path(args.baseline_tracker)
    applied = pd.read_parquet(args.baseline_applied)
    graph_root = Path(args.graph_root)
    meta = pd.read_parquet(graph_root / seq / "microtracklets.parquet")
    raw_edges = pd.read_parquet(graph_root / seq / "candidate_edges.parquet")
    safe_edges = raw_edges[list(m47.SAFE_CROSS_COLUMNS)].copy()
    edge_lookup = m47.safe_edge_lookup(safe_edges)
    rank_context = m47.edge_rank_context(safe_edges)
    prototypes = np.load(graph_root / seq / "prototypes.f16.npy").astype(np.float32)
    source_parent = Path(args.source_parent)

    prepared = m37.PreparedExactHOTA(seq, tracker, output_root / "cache")
    source_rows = evaluator.read_parent(source_parent / f"{seq}.txt")
    line_to_chunk = evaluator.line_chunks(source_rows, meta)
    parent_to_source = m38.parent_to_source_indices(prepared, source_rows)
    baseline_ids = np.asarray([int(float(row[1])) for row in prepared.parent_rows], dtype=np.int64)
    reconstructed = m38.row_ids_from_applied(applied, meta, line_to_chunk, parent_to_source, SEQUENCES.index(seq), evaluator)
    if not np.array_equal(baseline_ids, reconstructed):
        raise RuntimeError(f"baseline reconstruction mismatch: {np.count_nonzero(baseline_ids != reconstructed)}")
    baseline_metrics = prepared.evaluate_row_ids_incremental(baseline_ids)

    candidates = build_candidates(
        applied, meta, edge_lookup, rank_context, prototypes,
        args.max_boundaries, args.max_intervals, args.max_actions,
        args.max_interval_chunks, args.max_interval_frames, args.min_interval_rows,
        args.max_boundary_offset, args.min_span_ratio, args.max_cross_gap,
        m43, m47,
    )
    if candidates.empty:
        raise RuntimeError("no feasible GT-free four-boundary interval-swap candidates")
    candidates.to_parquet(output_root / "gt_free_interval_swap_candidates.parquet", index=False)
    print(json.dumps({"stage": "shortlist_frozen", "seq": seq, "candidates": len(candidates), "gt_fields_used": False}), flush=True)

    labels: List[Dict[str, object]] = []
    for ordinal, row in candidates.iterrows():
        started = time.perf_counter()
        try:
            modified = apply_action(applied, row, edge_lookup, evaluator, len(meta), m47)
            ids = m38.row_ids_from_applied(modified, meta, line_to_chunk, parent_to_source, SEQUENCES.index(seq), evaluator)
            metrics = prepared.evaluate_row_ids_incremental(ids)
            record = {
                "status": "success", "seq": seq, "action_type": "four_boundary_interval_swap",
                "action_ordinal": int(ordinal),
                **{key: (value.item() if isinstance(value, np.generic) else value) for key, value in row.items()},
                "changed_raw_rows": int(np.count_nonzero(ids != baseline_ids)),
                **m47.metric_delta(metrics, baseline_metrics),
            }
        except Exception as error:
            record = {
                "status": "failed", "seq": seq, "action_type": "four_boundary_interval_swap",
                "action_ordinal": int(ordinal), "error": repr(error),
                "teacher_seconds": time.perf_counter() - started,
            }
        labels.append(record)
        pd.DataFrame(labels).to_parquet(output_root / "exact_interval_swap_labels.parquet", index=False)
        print(json.dumps({
            "stage": "interval_swap_labeled", "ordinal": ordinal + 1, "total": len(candidates),
            "delta_HOTA": record.get("delta_HOTA"), "status": record["status"],
        }), flush=True)

    successful = pd.DataFrame([row for row in labels if row["status"] == "success"])
    policies: List[Dict[str, object]] = []
    best_metrics = baseline_metrics
    best_name = "noop"
    best_applied = applied.copy()
    for threshold in (float("inf"), 0.0, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1):
        chosen = successful[successful.delta_HOTA > threshold] if np.isfinite(threshold) else successful.iloc[:0]
        modified, accepted = compatible_greedy(applied, chosen, edge_lookup, evaluator, len(meta), m47)
        ids = m38.row_ids_from_applied(modified, meta, line_to_chunk, parent_to_source, SEQUENCES.index(seq), evaluator)
        metrics = prepared.evaluate_row_ids_incremental(ids)
        name = "noop" if not np.isfinite(threshold) else f"positive_t{threshold:g}"
        policy = {
            "policy": name,
            "threshold": None if not np.isfinite(threshold) else threshold,
            "candidate_actions": int(len(chosen)),
            "accepted_actions": int(len(accepted)),
            "HOTA": float(metrics["HOTA"]),
            "DetA": float(metrics["DetA"]),
            "AssA": float(metrics["AssA"]),
            "delta_HOTA": float(metrics["HOTA"] - baseline_metrics["HOTA"]),
        }
        policies.append(policy)
        print(json.dumps({"stage": "policy_evaluated", **policy}), flush=True)
        if float(metrics["HOTA"]) > float(best_metrics["HOTA"]):
            best_metrics, best_name, best_applied = metrics, name, modified

    best_applied.to_parquet(output_root / "best_applied_edges.parquet", index=False)
    evaluator.DATA = graph_root
    evaluator.PARENT = source_parent
    best_tracker = output_root / "best_tracker" / "track_results" / f"{seq}.txt"
    tracker_report = evaluator.write_tracker(seq, meta, best_applied, best_tracker)
    m47.write_csv(output_root / "policy_metrics.csv", policies)
    report = {
        **protocol,
        "status": "completed", "stage": "completed",
        "baseline_metrics": {key: float(value) for key, value in baseline_metrics.items() if isinstance(value, (int, float, np.number))},
        "successful_actions": int(len(successful)),
        "positive_actions": int((successful.delta_HOTA > 0.0).sum()),
        "negative_actions": int((successful.delta_HOTA < 0.0).sum()),
        "zero_actions": int((successful.delta_HOTA == 0.0).sum()),
        "best_policy": best_name,
        "best_metrics": {key: float(value) for key, value in best_metrics.items() if isinstance(value, (int, float, np.number))},
        "best_tracker": str(best_tracker),
        "tracker_report": tracker_report,
    }
    (output_root / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
