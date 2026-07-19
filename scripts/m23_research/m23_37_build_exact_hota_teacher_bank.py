#!/usr/bin/env python3
from __future__ import annotations

"""Build a GT-free-shortlisted, exact-HOTA transaction teacher bank.

The shortlist is constructed only from frozen model predictions and structural
features. GT is opened afterwards by the exact HOTA teacher on a training or
already-exposed diagnostic sequence. The output labels are teacher-only and may
be consumed only inside a strict outer-training fold.
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

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


def build_shortlist(
    predictions: pd.DataFrame,
    m36,
    per_score: int,
    max_actions: int,
) -> pd.DataFrame:
    frame = predictions.copy()
    criteria: Dict[str, np.ndarray] = {}
    representative_policies = {}
    for policy in m36.default_policies():
        if policy.policy_id == "noop":
            continue
        family = policy.policy_id.split("_q", 1)[0]
        representative_policies.setdefault(family, policy)
    for family, policy in representative_policies.items():
        scored = m36.add_policy_scores(frame, policy)
        column = f"shortlist_{family}_lcb"
        frame[column] = scored.component_lcb_score.to_numpy(float)
        criteria[column] = frame[column].to_numpy(float)

    direct_columns = [
        "pred_component_selection_probability",
        "pred_normalized_gain",
        "rank_consensus",
        "appearance_rank_consensus",
        "sequence_segment_appearance_percentile",
        "segment_appearance_cos",
        "merged_coherence_gain",
        "coherence_floor",
    ]
    for column in direct_columns:
        if column in frame:
            criteria[column] = frame[column].to_numpy(float)
    if "pred_catastrophic_probability" in frame:
        criteria["negative_pred_catastrophic_probability"] = -frame[
            "pred_catastrophic_probability"
        ].to_numpy(float)
    if "detached_fraction" in frame:
        criteria["negative_detached_fraction"] = -frame.detached_fraction.to_numpy(float)
    if "conflict_per_merged_row" in frame:
        criteria["negative_conflict_per_merged_row"] = -frame[
            "conflict_per_merged_row"
        ].to_numpy(float)

    selected_sources: Dict[int, List[str]] = {}
    selected_ranks: Dict[int, List[int]] = {}
    for name, raw_values in criteria.items():
        values = np.nan_to_num(raw_values, nan=-np.inf, neginf=-np.inf, posinf=np.inf)
        order = np.lexsort(
            (
                frame.dst_chunk.to_numpy(int),
                frame.src_chunk.to_numpy(int),
                -values,
            )
        )
        for rank, index in enumerate(order[:per_score], start=1):
            index = int(index)
            selected_sources.setdefault(index, []).append(name)
            selected_ranks.setdefault(index, []).append(rank)

    candidate_indices = list(selected_sources)
    candidate_indices.sort(
        key=lambda index: (
            -len(selected_sources[index]),
            min(selected_ranks[index]),
            float(frame.loc[index, "pred_catastrophic_probability"])
            if "pred_catastrophic_probability" in frame
            else 0.0,
            int(frame.loc[index, "src_chunk"]),
            int(frame.loc[index, "dst_chunk"]),
        )
    )
    candidate_indices = candidate_indices[:max_actions]
    shortlist = frame.loc[candidate_indices].copy()
    shortlist["shortlist_sources"] = [
        ";".join(selected_sources[int(index)]) for index in shortlist.index
    ]
    shortlist["shortlist_source_count"] = [
        len(selected_sources[int(index)]) for index in shortlist.index
    ]
    shortlist["shortlist_best_rank"] = [
        min(selected_ranks[int(index)]) for index in shortlist.index
    ]
    shortlist["prediction_row_index"] = shortlist.index.astype(np.int64)
    shortlist.sort_values(
        ["shortlist_source_count", "shortlist_best_rank", "src_chunk", "dst_chunk"],
        ascending=[False, True, True, True],
        inplace=True,
    )
    shortlist.reset_index(drop=True, inplace=True)
    return shortlist


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", required=True)
    parser.add_argument("--graph-root", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--per-score", type=int, default=32)
    parser.add_argument("--max-actions", type=int, default=128)
    args = parser.parse_args()

    seq = args.seq
    graph_root = Path(args.graph_root)
    predictions_path = Path(args.predictions)
    parent_root = Path(args.parent)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    m36 = load_module(
        "m23_37_m36", "scripts/m23_research/m23_36_robust_fold_component_matcher.py"
    )
    chain = load_module(
        "m23_37_chain", "scripts/m23_research/m23_12_chain_transaction_oracle.py"
    )
    evaluator = load_module(
        "m23_37_evaluator", "scripts/m23_research/m23_11_eval_utility_graph.py"
    )
    teacher_module = load_module(
        "m23_37_teacher", "scripts/m23_research/m23_37_fast_exact_hota_teacher.py"
    )

    predictions = pd.read_parquet(predictions_path)
    shortlist = build_shortlist(
        predictions,
        m36,
        per_score=args.per_score,
        max_actions=args.max_actions,
    )
    shortlist.to_parquet(output_root / "gt_free_shortlist.parquet", index=False)
    shortlist[[
        "src_chunk",
        "dst_chunk",
        "shortlist_sources",
        "shortlist_source_count",
        "shortlist_best_rank",
        "prediction_row_index",
    ]].to_csv(output_root / "gt_free_shortlist.csv", index=False)

    parent_tracker = parent_root / f"{seq}.txt"
    prepare_start = time.perf_counter()
    teacher = teacher_module.PreparedExactHOTA(
        seq,
        parent_tracker,
        output_root / "teacher_cache",
    )
    teacher_prepare_seconds = time.perf_counter() - prepare_start
    parent_ids = teacher.ids_from_tracker_file(parent_tracker)
    baseline = teacher.evaluate_row_ids_incremental(parent_ids)

    meta = pd.read_parquet(graph_root / seq / "microtracklets.parquet")
    edges = pd.read_parquet(graph_root / seq / "candidate_edges.parquet")
    chain.DATA = graph_root
    chain.PARENT = parent_root
    evaluator.DATA = graph_root
    evaluator.PARENT = parent_root

    metric_rows: List[Dict[str, object]] = []
    for action_index, row in shortlist.iterrows():
        selected = row.to_frame().T.copy()
        selected[TARGET] = 1.0
        applied = chain.apply_transactions(edges, selected)
        row_ids = teacher.row_ids_from_selected_graph(
            meta,
            applied,
            evaluator,
        )
        metrics = teacher.evaluate_row_ids_incremental(row_ids)
        record: Dict[str, object] = {
            "action_index": int(action_index),
            "src_chunk": int(row.src_chunk),
            "dst_chunk": int(row.dst_chunk),
            "transaction_src_track_id": int(row.transaction_src_track_id),
            "transaction_dst_track_id": int(row.transaction_dst_track_id),
            "shortlist_sources": str(row.shortlist_sources),
            "shortlist_source_count": int(row.shortlist_source_count),
            "shortlist_best_rank": int(row.shortlist_best_rank),
            "prediction_row_index": int(row.prediction_row_index),
            "exact_HOTA": float(metrics["HOTA"]),
            "exact_DetA": float(metrics["DetA"]),
            "exact_AssA": float(metrics["AssA"]),
            "exact_delta_HOTA": float(metrics["HOTA"] - baseline["HOTA"]),
            "exact_delta_DetA": float(metrics["DetA"] - baseline["DetA"]),
            "exact_delta_AssA": float(metrics["AssA"] - baseline["AssA"]),
            "exact_HOTA_positive": int(metrics["HOTA"] > baseline["HOTA"] + 1e-12),
            "affected_frames": int(metrics["affected_frames"]),
            "changed_processed_detections": int(
                metrics["changed_processed_detections"]
            ),
            "incremental_seconds": float(metrics["incremental_seconds"]),
        }
        if TARGET in row:
            record["proxy_transaction_utility"] = float(row[TARGET])
        for column in [
            "pred_component_selection_probability",
            "pred_catastrophic_probability",
            "pred_normalized_gain",
            "pred_normalized_severity",
            "rank_consensus",
            "segment_appearance_cos",
            "detached_fraction",
            "dst_cut_fraction",
            "src_cut_fraction",
            "conflict_per_merged_row",
            "merged_coherence_gain",
        ]:
            if column in row:
                record[column] = float(row[column])
        metric_rows.append(record)
        print(
            json.dumps(
                {
                    "stage": "exact_action",
                    "seq": seq,
                    "action": int(action_index) + 1,
                    "total": len(shortlist),
                    "src_chunk": int(row.src_chunk),
                    "dst_chunk": int(row.dst_chunk),
                    "delta_HOTA": record["exact_delta_HOTA"],
                    "delta_AssA": record["exact_delta_AssA"],
                    "affected_frames": record["affected_frames"],
                    "seconds": record["incremental_seconds"],
                }
            ),
            flush=True,
        )

    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_csv(output_root / "exact_hota_teacher_labels.csv", index=False)
    enriched = shortlist.merge(
        metrics_frame,
        on=[
            "src_chunk",
            "dst_chunk",
            "transaction_src_track_id",
            "transaction_dst_track_id",
            "shortlist_sources",
            "shortlist_source_count",
            "shortlist_best_rank",
            "prediction_row_index",
        ],
        how="left",
        validate="one_to_one",
        suffixes=("", "_teacher"),
    )
    enriched.to_parquet(output_root / "exact_hota_teacher_bank.parquet", index=False)

    correlations: Dict[str, object] = {}
    for column in [
        TARGET,
        "pred_component_selection_probability",
        "pred_catastrophic_probability",
        "pred_normalized_gain",
        "rank_consensus",
        "segment_appearance_cos",
    ]:
        if column in enriched and enriched[column].nunique(dropna=True) > 1:
            correlation = spearmanr(
                enriched[column].to_numpy(float),
                enriched.exact_delta_HOTA.to_numpy(float),
                nan_policy="omit",
            )
            correlations[column] = {
                "spearman": float(correlation.statistic),
                "pvalue": float(correlation.pvalue),
            }

    report = {
        "status": "completed",
        "teacher_only": True,
        "deployment_allowed": False,
        "seq": seq,
        "graph_root": str(graph_root),
        "predictions": str(predictions_path),
        "parent_tracker": str(parent_tracker),
        "shortlist_gt_use": "none",
        "teacher_gt_use": "exact HOTA labels after GT-free shortlist freeze",
        "teacher_prepare_seconds": teacher_prepare_seconds,
        "incremental_cache_seconds": teacher.incremental_prepare_seconds,
        "baseline": baseline,
        "shortlist_actions": int(len(shortlist)),
        "positive_exact_HOTA_actions": int(metrics_frame.exact_HOTA_positive.sum()),
        "positive_rate": float(metrics_frame.exact_HOTA_positive.mean())
        if len(metrics_frame)
        else 0.0,
        "delta_HOTA_sum": float(metrics_frame.exact_delta_HOTA.sum()),
        "delta_HOTA_max": float(metrics_frame.exact_delta_HOTA.max())
        if len(metrics_frame)
        else None,
        "delta_HOTA_min": float(metrics_frame.exact_delta_HOTA.min())
        if len(metrics_frame)
        else None,
        "mean_action_seconds": float(metrics_frame.incremental_seconds.mean())
        if len(metrics_frame)
        else None,
        "correlations": correlations,
    }
    (output_root / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"stage": "completed", **report}, indent=2), flush=True)


if __name__ == "__main__":
    main()
