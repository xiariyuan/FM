#!/usr/bin/env python3
from __future__ import annotations

"""Exact teacher labels for GT-free shortlisted cuts of selected source edges.

M23-41 can only change cross-track transactions.  This probe targets a
different failure mode: an identity switch already embedded inside one parent
track.  A cut removes one selected adjacent source edge and therefore splits a
chain at that boundary.  Candidate construction uses no GT columns; GT is
opened only by the exact HOTA teacher after the candidate bank is frozen.
"""

import argparse
import csv
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

# Deliberately excludes same_gt, modal_gt, purity, and every GT diagnostic.
SAFE_EDGE_COLUMNS = (
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


def percentile(values: pd.Series, higher_is_risk: bool = True) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    fill = float(numeric.median()) if numeric.notna().any() else 0.0
    ranked = numeric.fillna(fill).rank(method="average", pct=True)
    return ranked if higher_is_risk else 1.0 - ranked


def path_boundary_stats(applied: pd.DataFrame, meta: pd.DataFrame) -> Dict[Tuple[int, int], Dict[str, float]]:
    successor = {int(row.src_chunk): int(row.dst_chunk) for row in applied.itertuples()}
    predecessor = {dst: src for src, dst in successor.items()}
    if len(successor) != len(applied) or len(predecessor) != len(applied):
        raise RuntimeError("selected graph is not one-to-one")
    rows = meta.rows.to_numpy(float)
    roots = [chunk for chunk in range(len(meta)) if chunk not in predecessor]
    output: Dict[Tuple[int, int], Dict[str, float]] = {}
    for root in roots:
        chain = [root]
        while chain[-1] in successor:
            nxt = successor[chain[-1]]
            if nxt in chain:
                raise RuntimeError("cycle in selected graph")
            chain.append(nxt)
        weights = np.asarray([rows[chunk] for chunk in chain], dtype=float)
        cumulative = np.cumsum(weights)
        total = float(cumulative[-1])
        for offset, (left, right) in enumerate(zip(chain[:-1], chain[1:])):
            prefix = float(cumulative[offset])
            suffix = float(total - prefix)
            minimum = min(prefix, suffix)
            output[(left, right)] = {
                "chain_rows": total,
                "chain_chunks": float(len(chain)),
                "prefix_rows": prefix,
                "suffix_rows": suffix,
                "min_side_rows": minimum,
                "cut_fraction": prefix / max(total, 1.0),
                "cut_balance": minimum / max(total, 1.0),
                "log_chain_rows": float(np.log1p(total)),
                "log_min_side_rows": float(np.log1p(minimum)),
            }
    if len(output) != len(applied):
        raise RuntimeError("failed to describe every selected edge")
    return output


def candidate_source_cuts(applied: pd.DataFrame, meta: pd.DataFrame, max_cuts: int) -> pd.DataFrame:
    path_stats = path_boundary_stats(applied, meta)
    source = applied[applied.edge_role.astype(str) == "source"].copy()
    if source.empty:
        raise RuntimeError("no selected source edges")
    source["source_index"] = source.index.astype(int)
    safe = pd.DataFrame(index=source.index)
    for column in SAFE_EDGE_COLUMNS:
        safe[column] = source[column] if column in source else np.nan
    for key, values in path_stats.items():
        mask = (safe.src_chunk.astype(int) == key[0]) & (safe.dst_chunk.astype(int) == key[1])
        if not mask.any():
            continue
        for column, value in values.items():
            safe.loc[mask, column] = value
    safe["source_index"] = source.source_index.to_numpy(int)
    safe = safe[safe.min_side_rows.fillna(0.0) >= 15.0].copy()

    safe["risk_low_appearance"] = percentile(safe.appearance_cos, higher_is_risk=False)
    safe["risk_high_motion"] = percentile(safe.motion_error_mean, higher_is_risk=True)
    safe["risk_high_rank"] = percentile(safe.max_rank, higher_is_risk=True)
    safe["risk_low_consistency"] = percentile(safe.consistency_min, higher_is_risk=False)
    safe["risk_low_mapping"] = percentile(safe.mapping_rate_min, higher_is_risk=False)
    safe["impact_percentile"] = percentile(safe.min_side_rows, higher_is_risk=True)
    safe["boundary_disagreement"] = (
        safe.risk_low_appearance * safe.risk_high_motion
    )
    safe["source_cut_policy_score"] = (
        0.30 * safe.risk_low_appearance
        + 0.22 * safe.risk_high_motion
        + 0.13 * safe.risk_high_rank
        + 0.10 * safe.risk_low_consistency
        + 0.05 * safe.risk_low_mapping
        + 0.12 * safe.impact_percentile
        + 0.08 * safe.boundary_disagreement
    )

    orders = {
        "composite": safe.sort_values(["source_cut_policy_score", "src_chunk"], ascending=[False, True]).index.tolist(),
        "low_appearance": safe.sort_values(["risk_low_appearance", "impact_percentile"], ascending=[False, False]).index.tolist(),
        "high_motion": safe.sort_values(["risk_high_motion", "impact_percentile"], ascending=[False, False]).index.tolist(),
        "high_rank": safe.sort_values(["risk_high_rank", "impact_percentile"], ascending=[False, False]).index.tolist(),
        "low_consistency": safe.sort_values(["risk_low_consistency", "impact_percentile"], ascending=[False, False]).index.tolist(),
        "large_impact": safe.sort_values(["impact_percentile", "source_cut_policy_score"], ascending=[False, False]).index.tolist(),
        "disagreement": safe.sort_values(["boundary_disagreement", "impact_percentile"], ascending=[False, False]).index.tolist(),
    }
    chosen: List[int] = []
    chosen_set = set()
    cursors = {name: 0 for name in orders}
    channel_names = list(orders)
    chosen_channel: Dict[int, str] = {}
    chosen_rank: Dict[int, int] = {}
    while channel_names and len(chosen) < max_cuts:
        progressed = False
        for channel in list(channel_names):
            order = orders[channel]
            cursor = cursors[channel]
            while cursor < len(order):
                index = int(order[cursor])
                cursor += 1
                if index in chosen_set:
                    continue
                chosen.append(index)
                chosen_set.add(index)
                chosen_channel[index] = channel
                chosen_rank[index] = cursor
                progressed = True
                break
            cursors[channel] = cursor
            if cursor >= len(order):
                channel_names.remove(channel)
            if len(chosen) >= max_cuts:
                break
        if not progressed:
            break
    result = safe.loc[chosen].copy()
    result["selection_channel"] = [chosen_channel[int(index)] for index in result.index]
    result["selection_rank"] = [chosen_rank[int(index)] for index in result.index]
    return result.reset_index(drop=True)


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
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", required=True, choices=SEQUENCES)
    parser.add_argument("--baseline-tracker", required=True)
    parser.add_argument("--baseline-applied", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--graph-root", default=str(DEFAULT_GRAPH_ROOT))
    parser.add_argument("--source-parent", default=str(DEFAULT_PARENT))
    parser.add_argument("--max-cuts", type=int, default=96)
    args = parser.parse_args()

    seq = args.seq
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    protocol = {
        "experiment": "M23-43 exact source-boundary cut teacher",
        "status": "running",
        "teacher_only": True,
        "deployable": False,
        "sequence": seq,
        "candidate_construction": "GT-free source-edge structural features only",
        "gt_use": "exact HOTA teacher after candidate shortlist freeze",
        "action_space": "remove one selected adjacent source edge",
        "max_cuts": args.max_cuts,
        "baseline_tracker": args.baseline_tracker,
        "baseline_applied": args.baseline_applied,
    }
    (output_root / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")

    import importlib.util
    import sys

    def load_module(name: str, path: str):
        spec = importlib.util.spec_from_file_location(name, REPO / path)
        if spec is None or spec.loader is None:
            raise RuntimeError(path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    m37 = load_module("m23_43_exact", "scripts/m23_research/m23_37_fast_exact_hota_teacher.py")
    m38 = load_module("m23_43_helpers", "scripts/m23_research/m23_38_exact_hota_advantage_labels.py")
    evaluator = load_module("m23_43_evaluator", "scripts/m23_research/m23_11_eval_utility_graph.py")

    tracker = Path(args.baseline_tracker)
    applied = pd.read_parquet(args.baseline_applied)
    graph_root = Path(args.graph_root)
    meta = pd.read_parquet(graph_root / seq / "microtracklets.parquet")
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
        raise RuntimeError(f"baseline reconstruction mismatch: {np.count_nonzero(baseline_ids != reconstructed)}")
    baseline_metrics = prepared.evaluate_row_ids_incremental(baseline_ids)

    candidates = candidate_source_cuts(applied, meta, args.max_cuts)
    candidates.to_parquet(output_root / "gt_free_cut_candidates.parquet", index=False)
    print(json.dumps({"stage": "shortlist_frozen", "seq": seq, "candidates": len(candidates), "gt_fields_used": False}), flush=True)

    labels: List[Dict[str, object]] = []
    for ordinal, row in candidates.iterrows():
        started = time.perf_counter()
        try:
            source_index = int(row.source_index)
            modified = applied.drop(index=source_index).copy()
            candidate_ids = m38.row_ids_from_applied(
                modified, meta, line_to_chunk, parent_to_source, SEQUENCES.index(seq), evaluator
            )
            metrics = prepared.evaluate_row_ids_incremental(candidate_ids)
            record = {
                "status": "success", "seq": seq, "action_type": "source_cut",
                "action_ordinal": int(ordinal),
                **{key: (value.item() if isinstance(value, np.generic) else value) for key, value in row.items()},
                "changed_raw_rows": int(np.count_nonzero(candidate_ids != baseline_ids)),
                **metric_delta(metrics, baseline_metrics),
            }
        except Exception as error:
            record = {
                "status": "failed", "seq": seq, "action_type": "source_cut",
                "action_ordinal": int(ordinal), "source_index": int(row.source_index),
                "error": repr(error), "teacher_seconds": time.perf_counter() - started,
            }
        labels.append(record)
        pd.DataFrame(labels).to_parquet(output_root / "exact_source_cut_labels.parquet", index=False)
        print(json.dumps({"stage": "cut_labeled", "ordinal": ordinal + 1, "total": len(candidates), "delta_HOTA": record.get("delta_HOTA"), "status": record["status"]}), flush=True)

    successful = pd.DataFrame([row for row in labels if row["status"] == "success"])
    policies = []
    best_metrics = baseline_metrics
    best_name = "noop"
    best_applied = applied.copy()
    for threshold in (float("inf"), 0.0, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1):
        chosen = successful[successful.delta_HOTA > threshold] if np.isfinite(threshold) else successful.iloc[:0]
        modified = applied.drop(index=chosen.source_index.astype(int).tolist()).copy()
        ids = m38.row_ids_from_applied(
            modified, meta, line_to_chunk, parent_to_source, SEQUENCES.index(seq), evaluator
        )
        metrics = prepared.evaluate_row_ids_incremental(ids)
        name = "noop" if not np.isfinite(threshold) else f"positive_t{threshold:g}"
        policy = {
            "policy": name, "threshold": None if not np.isfinite(threshold) else threshold,
            "cuts": int(len(chosen)), "HOTA": float(metrics["HOTA"]),
            "DetA": float(metrics["DetA"]), "AssA": float(metrics["AssA"]),
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
    write_csv(output_root / "policy_metrics.csv", policies)
    report = {
        **protocol, "status": "completed", "stage": "completed",
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
