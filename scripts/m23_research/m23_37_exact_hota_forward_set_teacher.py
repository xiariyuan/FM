#!/usr/bin/env python3
from __future__ import annotations

"""Exact-HOTA forward set teacher over a frozen GT-free transaction shortlist.

This is a non-deployable teacher. It uses GT only after the action shortlist has
been frozen, and greedily chooses the next disjoint transaction by its exact
HOTA marginal under the already-selected action set. The resulting decision
trajectory is intended for structured imitation on outer-training sequences.
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
TARGET = "chain_transaction_delta_proxy"


def load_module(name: str, relative_path: str):
    path = REPO / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", required=True)
    parser.add_argument("--graph-root", required=True)
    parser.add_argument("--teacher-bank", required=True)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--min-delta-hota", type=float, default=1e-6)
    parser.add_argument("--official-final-check", action="store_true")
    args = parser.parse_args()

    seq = args.seq
    graph_root = Path(args.graph_root)
    parent_root = Path(args.parent)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    chain = load_module(
        "m23_37_set_chain", "scripts/m23_research/m23_12_chain_transaction_oracle.py"
    )
    evaluator = load_module(
        "m23_37_set_evaluator", "scripts/m23_research/m23_11_eval_utility_graph.py"
    )
    m36 = load_module(
        "m23_37_set_m36", "scripts/m23_research/m23_36_robust_fold_component_matcher.py"
    )
    teacher_module = load_module(
        "m23_37_set_teacher", "scripts/m23_research/m23_37_fast_exact_hota_teacher.py"
    )

    candidates = pd.read_parquet(args.teacher_bank).copy()
    candidates["teacher_action_id"] = np.arange(len(candidates), dtype=np.int64)
    required = {
        "src_chunk",
        "dst_chunk",
        "transaction_src_track_id",
        "transaction_dst_track_id",
    }
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise RuntimeError(f"teacher bank missing columns: {missing}")

    meta = pd.read_parquet(graph_root / seq / "microtracklets.parquet")
    edges = pd.read_parquet(graph_root / seq / "candidate_edges.parquet")
    parent_tracker = parent_root / f"{seq}.txt"
    teacher = teacher_module.PreparedExactHOTA(
        seq,
        parent_tracker,
        output_root / "teacher_cache",
    )
    baseline_ids = teacher.ids_from_tracker_file(parent_tracker)
    current_metrics = teacher.evaluate_row_ids_incremental(baseline_ids)

    chain.DATA = graph_root
    chain.PARENT = parent_root
    evaluator.DATA = graph_root
    evaluator.PARENT = parent_root

    selected_ids: List[int] = []
    used_tracks: Set[int] = set()
    trajectory: List[Dict[str, object]] = []
    evaluation_rows: List[Dict[str, object]] = []

    for step in range(args.max_steps):
        eligible = candidates[
            ~candidates.teacher_action_id.isin(selected_ids)
            & ~candidates.transaction_src_track_id.astype(int).isin(used_tracks)
            & ~candidates.transaction_dst_track_id.astype(int).isin(used_tracks)
        ].copy()
        if eligible.empty:
            break

        best_record: Dict[str, object] | None = None
        for candidate in eligible.itertuples(index=False):
            trial_ids = selected_ids + [int(candidate.teacher_action_id)]
            trial = candidates[
                candidates.teacher_action_id.isin(trial_ids)
            ].copy()
            trial[TARGET] = 1.0
            applied = chain.apply_transactions(edges, trial)
            row_ids = teacher.row_ids_from_selected_graph(meta, applied, evaluator)
            metrics = teacher.evaluate_row_ids_incremental(row_ids)
            record: Dict[str, object] = {
                "step": int(step),
                "candidate_action_id": int(candidate.teacher_action_id),
                "src_chunk": int(candidate.src_chunk),
                "dst_chunk": int(candidate.dst_chunk),
                "transaction_src_track_id": int(candidate.transaction_src_track_id),
                "transaction_dst_track_id": int(candidate.transaction_dst_track_id),
                "selected_before": int(len(selected_ids)),
                "trial_HOTA": float(metrics["HOTA"]),
                "trial_DetA": float(metrics["DetA"]),
                "trial_AssA": float(metrics["AssA"]),
                "marginal_HOTA": float(metrics["HOTA"] - current_metrics["HOTA"]),
                "marginal_DetA": float(metrics["DetA"] - current_metrics["DetA"]),
                "marginal_AssA": float(metrics["AssA"] - current_metrics["AssA"]),
                "affected_frames": int(metrics["affected_frames"]),
                "incremental_seconds": float(metrics["incremental_seconds"]),
                "shortlist_sources": str(getattr(candidate, "shortlist_sources", "")),
                "individual_delta_HOTA": float(
                    getattr(candidate, "exact_delta_HOTA", np.nan)
                ),
                "proxy_transaction_utility": float(
                    getattr(candidate, TARGET, np.nan)
                ),
            }
            evaluation_rows.append(record)
            if best_record is None:
                best_record = record
            else:
                candidate_key = (
                    float(record["trial_HOTA"]),
                    float(record["trial_AssA"]),
                    float(record["trial_DetA"]),
                    -int(record["affected_frames"]),
                    -int(record["candidate_action_id"]),
                )
                best_key = (
                    float(best_record["trial_HOTA"]),
                    float(best_record["trial_AssA"]),
                    float(best_record["trial_DetA"]),
                    -int(best_record["affected_frames"]),
                    -int(best_record["candidate_action_id"]),
                )
                if candidate_key > best_key:
                    best_record = record

        if best_record is None or float(best_record["marginal_HOTA"]) <= args.min_delta_hota:
            trajectory.append(
                {
                    "step": int(step),
                    "decision": "stop",
                    "current_HOTA": float(current_metrics["HOTA"]),
                    "best_available_marginal_HOTA": (
                        float(best_record["marginal_HOTA"])
                        if best_record is not None
                        else None
                    ),
                    "eligible_actions": int(len(eligible)),
                }
            )
            break

        selected_action_id = int(best_record["candidate_action_id"])
        selected_ids.append(selected_action_id)
        selected_row = candidates.loc[
            candidates.teacher_action_id == selected_action_id
        ].iloc[0]
        used_tracks.add(int(selected_row.transaction_src_track_id))
        used_tracks.add(int(selected_row.transaction_dst_track_id))
        current_metrics = {
            "HOTA": float(best_record["trial_HOTA"]),
            "DetA": float(best_record["trial_DetA"]),
            "AssA": float(best_record["trial_AssA"]),
        }
        trajectory.append(
            {
                **best_record,
                "decision": "select",
                "selected_after": int(len(selected_ids)),
                "used_tracks_after": int(len(used_tracks)),
            }
        )
        print(
            json.dumps(
                {
                    "stage": "set_teacher_step",
                    "seq": seq,
                    "step": int(step) + 1,
                    "action_id": selected_action_id,
                    "marginal_HOTA": best_record["marginal_HOTA"],
                    "HOTA": best_record["trial_HOTA"],
                    "DetA": best_record["trial_DetA"],
                    "AssA": best_record["trial_AssA"],
                    "eligible": len(eligible),
                }
            ),
            flush=True,
        )

    selected = candidates[candidates.teacher_action_id.isin(selected_ids)].copy()
    selected["set_teacher_order"] = selected.teacher_action_id.map(
        {action_id: order for order, action_id in enumerate(selected_ids)}
    )
    selected.sort_values("set_teacher_order", inplace=True)
    selected.to_parquet(output_root / "selected_set_teacher_actions.parquet", index=False)
    pd.DataFrame(trajectory).to_csv(output_root / "trajectory.csv", index=False)
    pd.DataFrame(evaluation_rows).to_csv(
        output_root / "candidate_marginal_evaluations.csv", index=False
    )

    final_official = None
    if args.official_final_check:
        final_selected = selected.copy()
        final_selected[TARGET] = 1.0
        applied = chain.apply_transactions(edges, final_selected)
        for column, default in [
            ("assa_edge_delta_proxy", 0.0),
            ("assa_edge_positive", 0),
            ("assa_edge_negative", 0),
        ]:
            if column not in applied:
                applied[column] = default
        tracker_root = output_root / "official_final" / "track_results"
        tracker_root.mkdir(parents=True, exist_ok=True)
        tracker_report = evaluator.write_tracker(
            seq, meta, applied, tracker_root / f"{seq}.txt"
        )
        final_official = m36.evaluate_single(
            tracker_root,
            output_root / "official_final" / "eval_work",
            "m23_37_exact_hota_set_teacher",
            seq,
        )
        final_official["tracker_report"] = tracker_report

    report = {
        "status": "completed",
        "teacher_only": True,
        "deployment_allowed": False,
        "seq": seq,
        "shortlist_gt_use": "none; inherited frozen teacher bank shortlist",
        "teacher_gt_use": "exact HOTA forward marginal on outer-training sequence only",
        "baseline": teacher.evaluate_row_ids_incremental(baseline_ids),
        "selected_actions": int(len(selected_ids)),
        "selected_action_ids": selected_ids,
        "final_incremental_metrics": current_metrics,
        "delta_HOTA": float(current_metrics["HOTA"] - teacher.evaluate_row_ids_incremental(baseline_ids)["HOTA"]),
        "trajectory_steps": int(len(trajectory)),
        "candidate_evaluations": int(len(evaluation_rows)),
        "official_final_check": final_official,
    }
    (output_root / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"stage": "completed", **report}, indent=2), flush=True)


if __name__ == "__main__":
    main()
