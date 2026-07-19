#!/usr/bin/env python3
from __future__ import annotations

"""Exact diagnostic policy probe over M23-38 HOTA-advantage actions.

This is a GT teacher/oracle diagnostic.  It selects actions using exact labels
from an allowed training or already-exposed sequence, then measures interaction
with the affected-frame exact HOTA evaluator.  Results are ceilings and must
not be reported as deployable LOSO policies.
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import networkx as nx
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SEQUENCES = ("MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05")
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


def nodes(row) -> set[int]:
    return {
        int(row.transaction_src_track_id),
        int(row.transaction_dst_track_id),
    }


def maximum_weight_replacements(labels: pd.DataFrame) -> pd.DataFrame:
    if labels.empty:
        return labels.copy()
    best_by_pair: Dict[tuple[int, int], tuple[float, int]] = {}
    for index, row in labels.iterrows():
        left = int(row.transaction_src_track_id)
        right = int(row.transaction_dst_track_id)
        pair = (min(left, right), max(left, right))
        weight = float(row.delta_HOTA)
        current = best_by_pair.get(pair)
        if current is None or weight > current[0]:
            best_by_pair[pair] = (weight, int(index))
    graph = nx.Graph()
    for (left, right), (weight, index) in best_by_pair.items():
        if left != right and weight > 0.0:
            graph.add_edge(left, right, weight=weight, row_index=index)
    if graph.number_of_edges() == 0:
        return labels.iloc[:0].copy()
    matching = nx.algorithms.matching.max_weight_matching(
        graph, maxcardinality=False, weight="weight"
    )
    indices = [int(graph[left][right]["row_index"]) for left, right in matching]
    return labels.loc[sorted(indices)].copy()


def build_selected_transactions(
    baseline_selected: pd.DataFrame,
    predictions: pd.DataFrame,
    drop_labels: pd.DataFrame,
    replacement_labels: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    drop_indices = set(drop_labels.source_index.astype(int))
    replacement_nodes = set()
    for row in replacement_labels.itertuples():
        replacement_nodes.update(nodes(row))
    keep = []
    for index, row in baseline_selected.iterrows():
        keep.append(
            int(index) not in drop_indices and nodes(row).isdisjoint(replacement_nodes)
        )
    retained = baseline_selected[np.asarray(keep, dtype=bool)].copy()
    replacement_rows = []
    for label in replacement_labels.itertuples():
        candidate = predictions.loc[int(label.source_index)].copy()
        candidate[TARGET] = float(label.delta_HOTA)
        replacement_rows.append(candidate)
    if replacement_rows:
        replacement_frame = pd.DataFrame(replacement_rows)
        output = pd.concat([retained, replacement_frame], ignore_index=True, sort=False)
    else:
        output = retained.reset_index(drop=True)
    removed = int(len(baseline_selected) - len(retained))
    return output, removed


def mean_metric_fields(metrics: Dict[str, object]) -> Dict[str, float]:
    return {
        key: float(metrics[key])
        for key in ("HOTA", "DetA", "AssA", "DetRe", "DetPr", "AssRe", "AssPr", "LocA")
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", required=True, choices=SEQUENCES)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--graph-root", required=True)
    parser.add_argument("--source-parent", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--thresholds", default="0,0.005,0.01,0.025,0.05,0.1")
    args = parser.parse_args()

    seq = args.seq
    baseline_root = Path(args.baseline_root)
    graph_root = Path(args.graph_root)
    source_parent = Path(args.source_parent)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    thresholds = sorted({float(value) for value in args.thresholds.split(",")})

    label_module = load_module(
        "m23_38_policy_labels", "scripts/m23_research/m23_38_exact_hota_advantage_labels.py"
    )
    exact_module = load_module(
        "m23_38_policy_exact", "scripts/m23_research/m23_37_fast_exact_hota_teacher.py"
    )
    chain = load_module(
        "m23_38_policy_chain", "scripts/m23_research/m23_12_chain_transaction_oracle.py"
    )
    evaluator = load_module(
        "m23_38_policy_eval", "scripts/m23_research/m23_11_eval_utility_graph.py"
    )

    tracker = baseline_root / "track_results" / f"{seq}.txt"
    prepare_start = time.perf_counter()
    prepared = exact_module.PreparedExactHOTA(seq, tracker, output_root / "cache")
    prepare_seconds = time.perf_counter() - prepare_start
    meta = pd.read_parquet(graph_root / seq / "microtracklets.parquet")
    edges = pd.read_parquet(graph_root / seq / "candidate_edges.parquet")
    predictions = pd.read_parquet(
        baseline_root / "predictions" / f"{seq}_predictions.parquet"
    )
    baseline_selected = pd.read_parquet(
        baseline_root / f"{seq}_selected_transactions.parquet"
    )
    if TARGET not in baseline_selected:
        baseline_selected[TARGET] = (
            baseline_selected.policy_score.to_numpy(float)
            if "policy_score" in baseline_selected
            else 1.0
        )
    baseline_selected[TARGET] = baseline_selected[TARGET].fillna(1.0).astype(float)
    source_rows = evaluator.read_parent(source_parent / f"{seq}.txt")
    line_to_chunk = evaluator.line_chunks(source_rows, meta)
    parent_to_source = label_module.parent_to_source_indices(prepared, source_rows)
    baseline_ids = np.asarray(
        [int(float(fields[1])) for fields in prepared.parent_rows], dtype=np.int64
    )
    baseline_metrics = prepared.evaluate_row_ids_incremental(baseline_ids)

    labels = pd.read_parquet(args.labels)
    labels = labels[labels.status == "success"].copy()
    positive_drops = labels[
        (labels.action_type == "drop") & (labels.delta_HOTA > 0.0)
    ].copy()
    replacements = labels[labels.action_type == "replace"].copy()
    policy_specs: List[tuple[str, pd.DataFrame, pd.DataFrame]] = [
        ("noop", positive_drops.iloc[:0], replacements.iloc[:0]),
        ("drop_positive", positive_drops, replacements.iloc[:0]),
    ]
    for threshold in thresholds:
        eligible = replacements[replacements.delta_HOTA > threshold].copy()
        matched = maximum_weight_replacements(eligible)
        tag = str(threshold).replace(".", "p")
        policy_specs.append((f"replace_t{tag}", positive_drops.iloc[:0], matched))
        policy_specs.append((f"combo_t{tag}", positive_drops, matched))

    rows = []
    best_payload = None
    for policy_index, (name, drops, replacement_selection) in enumerate(policy_specs):
        selected, removed = build_selected_transactions(
            baseline_selected, predictions, drops, replacement_selection
        )
        applied = chain.apply_transactions(edges, selected)
        candidate_ids = label_module.row_ids_from_applied(
            applied,
            meta,
            line_to_chunk,
            parent_to_source,
            SEQUENCES.index(seq),
            evaluator,
        )
        metrics = prepared.evaluate_row_ids_incremental(candidate_ids)
        row = {
            "policy_index": policy_index,
            "policy": name,
            **mean_metric_fields(metrics),
            "delta_HOTA": float(metrics["HOTA"] - baseline_metrics["HOTA"]),
            "delta_AssA": float(metrics["AssA"] - baseline_metrics["AssA"]),
            "delta_DetA": float(metrics["DetA"] - baseline_metrics["DetA"]),
            "drop_actions": int(len(drops)),
            "replacement_actions": int(len(replacement_selection)),
            "removed_baseline_actions": removed,
            "resulting_selected_actions": int(len(selected)),
            "sum_single_action_delta_HOTA": float(
                drops.delta_HOTA.sum() + replacement_selection.delta_HOTA.sum()
            ),
            "changed_raw_rows": int(np.count_nonzero(candidate_ids != baseline_ids)),
            "affected_frames": int(metrics["affected_frames"]),
            "teacher_seconds": float(metrics["incremental_seconds"]),
        }
        rows.append(row)
        if best_payload is None or row["HOTA"] > best_payload[0]["HOTA"]:
            best_payload = (row, selected.copy(), applied.copy())
        pd.DataFrame(rows).to_csv(output_root / "policy_metrics.csv", index=False)
        print(json.dumps({"stage": "policy_evaluated", **row}), flush=True)

    assert best_payload is not None
    best_row, best_selected, best_applied = best_payload
    best_selected.to_parquet(output_root / "best_selected_transactions.parquet", index=False)
    for column, default in (
        ("assa_edge_delta_proxy", 0.0),
        ("assa_edge_positive", 0),
        ("assa_edge_negative", 0),
    ):
        if column not in best_applied:
            best_applied[column] = default
    best_applied.to_parquet(output_root / "best_applied_edges.parquet", index=False)
    evaluator.DATA = graph_root
    evaluator.PARENT = source_parent
    tracker_root = output_root / "best_tracker" / "track_results"
    tracker_root.mkdir(parents=True, exist_ok=True)
    tracker_report = evaluator.write_tracker(
        seq, meta, best_applied, tracker_root / f"{seq}.txt"
    )
    report = {
        "status": "completed",
        "teacher_only": True,
        "deployable": False,
        "protocol": "posthoc exact-label interaction probe on allowed diagnostic/training sequence",
        "seq": seq,
        "prepare_seconds": prepare_seconds,
        "baseline": mean_metric_fields(baseline_metrics),
        "policies": rows,
        "best_policy": best_row,
        "tracker_report": tracker_report,
        "best_tracker": str(tracker_root / f"{seq}.txt"),
    }
    (output_root / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"stage": "completed", **report}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
