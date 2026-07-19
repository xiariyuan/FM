#!/usr/bin/env python3
from __future__ import annotations

"""Generate exact HOTA-advantage labels around a frozen M23-25 tracker.

Actions are local counterfactual edits of the frozen selected transaction set:
drop one selected transaction, or replace every selected transaction that
conflicts with a residual candidate.  Official TrackEval preprocessing is
cached and only affected frames rerun Hungarian matching.

GT use is teacher-only.  Run this only on outer-training sequences or already
exposed diagnostic sequences; never use held-sequence labels for model or
policy selection.
"""

import argparse
import importlib.util
import json
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SEQUENCES = ("MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05")
TARGET = "chain_transaction_delta_proxy"
DEFAULT_GRAPH_ROOT = Path("outputs/mot20_m23_20260718/micrograph_chunk30_v1")
DEFAULT_SOURCE_PARENT = Path(
    "outputs/assocriskbench_p15_20260714/fused_a42_adaptive_risk_eval/"
    "l1_r2_q75_adaptive_priority/track_results"
)


def load_module(name: str, relative_path: str):
    path = REPO / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def geometry_key(fields: Sequence[str]):
    return (
        int(float(fields[0])),
        *(int(round(float(value) * 10000.0)) for value in fields[2:6]),
    )


def parent_to_source_indices(prepared, source_rows: Sequence[Dict[str, object]]):
    source_by_geometry = defaultdict(deque)
    for source_index, row in enumerate(source_rows):
        source_by_geometry[geometry_key(row["fields"])].append(source_index)
    output = np.empty(prepared.num_parent_rows, dtype=np.int64)
    for parent_index, fields in enumerate(prepared.parent_rows):
        queue = source_by_geometry.get(geometry_key(fields))
        if not queue:
            raise RuntimeError(f"cannot align baseline row {parent_index} to source parent")
        output[parent_index] = queue.popleft()
    if any(source_by_geometry.values()):
        raise RuntimeError("unconsumed source-parent rows after geometry alignment")
    return output


def action_tracks(row) -> set[int]:
    return {
        int(row.transaction_src_track_id),
        int(row.transaction_dst_track_id),
    }


def transaction_key(row) -> tuple[int, int]:
    return int(row.src_chunk), int(row.dst_chunk)


def policy_score(frame: pd.DataFrame) -> np.ndarray:
    probability = frame.pred_positive_probability.to_numpy(float)
    gain = frame.pred_normalized_gain.to_numpy(float)
    loss = frame.pred_normalized_loss.to_numpy(float)
    return probability * gain - (1.0 - probability) * loss


def residual_replacements(
    predictions: pd.DataFrame,
    selected: pd.DataFrame,
    max_replacements: int,
    selection_mode: str = "policy",
) -> pd.DataFrame:
    scored = predictions.copy()
    scored["m23_38_parent_policy_score"] = policy_score(scored)
    selected_keys = {transaction_key(row) for row in selected.itertuples()}
    scored = scored[
        [transaction_key(row) not in selected_keys for row in scored.itertuples()]
    ].copy()
    scored = scored[
        scored.transaction_src_track_id.to_numpy(int)
        != scored.transaction_dst_track_id.to_numpy(int)
    ].copy()
    scored.sort_values(
        [
            "m23_38_parent_policy_score",
            "pred_positive_probability",
            "src_chunk",
            "dst_chunk",
        ],
        ascending=[False, False, True, True],
        inplace=True,
    )
    def pair_key(row) -> tuple[int, int]:
        left = int(row.transaction_src_track_id)
        right = int(row.transaction_dst_track_id)
        return min(left, right), max(left, right)

    if selection_mode == "policy":
        pair_seen = set()
        indices: List[int] = []
        for index, row in scored.iterrows():
            pair = pair_key(row)
            if pair in pair_seen:
                continue
            pair_seen.add(pair)
            indices.append(int(index))
            if len(indices) >= max_replacements:
                break
        output = scored.loc[indices].copy()
        output["m23_41_selection_channel"] = "parent_policy"
        output["m23_41_selection_rank"] = np.arange(1, len(output) + 1)
        return output

    # M23-41: retain the strongest parent-policy candidates, then fill the
    # remaining budget round-robin from deliberately different GT-free views.
    # This avoids spending a larger teacher budget on near-duplicates of the
    # same old score while preserving backward coverage of the M23-38 top set.
    criteria: Dict[str, tuple[str, bool]] = {
        "positive_probability": ("pred_positive_probability", True),
        "predicted_gain": ("pred_normalized_gain", True),
        "motion_error_mean": ("motion_error_mean", False),
        "motion_error_min": ("motion_error_min", False),
        "motion_consensus": ("sequence_motion_percentile", True),
        "rank_consensus": ("rank_consensus", True),
        "appearance_consensus": ("sequence_segment_appearance_percentile", True),
        "track_appearance": ("track_appearance_cos", True),
        "low_detachment": ("detached_fraction", False),
        "low_conflict_density": ("conflict_per_merged_row", False),
        "large_merged_segment": ("merged_segment_rows", True),
        "large_detached_segment": ("detached_segment_rows", True),
        "high_entropy": ("pred_entropy", True),
    }

    channel_orders: Dict[str, List[int]] = {}
    for channel, (column, higher_is_better) in criteria.items():
        if column not in scored or scored[column].notna().sum() == 0:
            continue
        values = scored[column].to_numpy(float)
        finite = np.isfinite(values)
        fill = np.nanmin(values[finite]) if finite.any() else 0.0
        values = np.nan_to_num(values, nan=fill, posinf=fill, neginf=fill)
        primary = -values if higher_is_better else values
        order = np.lexsort(
            (
                scored.dst_chunk.to_numpy(int),
                scored.src_chunk.to_numpy(int),
                primary,
            )
        )
        channel_orders[channel] = [int(scored.index[position]) for position in order]

    # Near the old decision boundary is a separate active-learning channel.
    neutral_order = np.lexsort(
        (
            scored.dst_chunk.to_numpy(int),
            scored.src_chunk.to_numpy(int),
            np.abs(scored.m23_38_parent_policy_score.to_numpy(float)),
        )
    )
    channel_orders["parent_boundary"] = [
        int(scored.index[position]) for position in neutral_order
    ]

    chosen: List[int] = []
    chosen_channel: Dict[int, str] = {}
    chosen_rank: Dict[int, int] = {}
    pair_seen = set()

    exploit_budget = min(max_replacements, max(32, min(64, max_replacements // 4)))
    for rank, (index, row) in enumerate(scored.iterrows(), start=1):
        pair = pair_key(row)
        if pair in pair_seen:
            continue
        pair_seen.add(pair)
        index = int(index)
        chosen.append(index)
        chosen_channel[index] = "parent_policy"
        chosen_rank[index] = rank
        if len(chosen) >= exploit_budget:
            break

    cursors = {channel: 0 for channel in channel_orders}
    channel_names = list(channel_orders)
    while len(chosen) < max_replacements and channel_names:
        progressed = False
        for channel in list(channel_names):
            order = channel_orders[channel]
            cursor = cursors[channel]
            while cursor < len(order):
                index = order[cursor]
                cursor += 1
                row = scored.loc[index]
                pair = pair_key(row)
                if pair in pair_seen:
                    continue
                pair_seen.add(pair)
                chosen.append(index)
                chosen_channel[index] = channel
                chosen_rank[index] = cursor
                progressed = True
                break
            cursors[channel] = cursor
            if cursor >= len(order):
                channel_names.remove(channel)
            if len(chosen) >= max_replacements:
                break
        if not progressed:
            break

    output = scored.loc[chosen].copy()
    output["m23_41_selection_channel"] = [chosen_channel[int(index)] for index in output.index]
    output["m23_41_selection_rank"] = [chosen_rank[int(index)] for index in output.index]
    return output


def replace_action(selected: pd.DataFrame, candidate: pd.Series):
    nodes = action_tracks(candidate)
    keep = [
        action_tracks(row).isdisjoint(nodes) for row in selected.itertuples()
    ]
    retained = selected[np.asarray(keep, dtype=bool)].copy()
    candidate_frame = candidate.to_frame().T.copy()
    candidate_frame[TARGET] = float(candidate.m23_38_parent_policy_score)
    output = pd.concat([retained, candidate_frame], ignore_index=True, sort=False)
    return output, int(len(selected) - len(retained))


def row_ids_from_applied(
    applied: pd.DataFrame,
    meta: pd.DataFrame,
    line_to_chunk: Dict[int, int],
    parent_to_source: np.ndarray,
    sequence_index: int,
    evaluator,
) -> np.ndarray:
    assignment = evaluator.chains(applied, len(meta))
    base = int(evaluator.SYN_BASE + sequence_index * evaluator.STRIDE)
    source_ids = np.asarray(
        [base + int(assignment[int(line_to_chunk[index])]) for index in range(len(line_to_chunk))],
        dtype=np.int64,
    )
    return source_ids[parent_to_source]


def metric_delta(metrics: Dict[str, float], baseline: Dict[str, float]):
    return {
        "exact_HOTA": float(metrics["HOTA"]),
        "exact_DetA": float(metrics["DetA"]),
        "exact_AssA": float(metrics["AssA"]),
        "delta_HOTA": float(metrics["HOTA"] - baseline["HOTA"]),
        "delta_DetA": float(metrics["DetA"] - baseline["DetA"]),
        "delta_AssA": float(metrics["AssA"] - baseline["AssA"]),
        "affected_frames": int(metrics["affected_frames"]),
        "affected_processed_detections": int(metrics["affected_processed_detections"]),
        "changed_processed_detections": int(metrics["changed_processed_detections"]),
        "teacher_seconds": float(metrics["incremental_seconds"]),
    }


def compact_feature_row(row, columns: Iterable[str]) -> Dict[str, object]:
    output: Dict[str, object] = {}
    for column in columns:
        if column not in row.index:
            continue
        value = row[column]
        if isinstance(value, np.generic):
            value = value.item()
        output[column] = value
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", required=True, choices=SEQUENCES)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--graph-root", default=str(DEFAULT_GRAPH_ROOT))
    parser.add_argument("--source-parent", default=str(DEFAULT_SOURCE_PARENT))
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-replacements", type=int, default=32)
    parser.add_argument(
        "--replacement-selection",
        choices=("policy", "diverse"),
        default="policy",
    )
    parser.add_argument("--skip-drops", action="store_true")
    args = parser.parse_args()

    seq = args.seq
    baseline_root = Path(args.baseline_root)
    graph_root = Path(args.graph_root)
    source_parent_root = Path(args.source_parent)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    protocol = {
        "experiment": "M23-38 exact HOTA advantage labels",
        "status": "running",
        "teacher_only": True,
        "sequence": seq,
        "gt_use": "teacher label generation only",
        "allowed_use": "outer-training or already-exposed diagnostic sequence only",
        "baseline_root": str(baseline_root),
        "graph_root": str(graph_root),
        "action_space": "drop one frozen transaction or replace all conflicts with one residual transaction",
        "max_replacements": args.max_replacements,
        "replacement_selection": args.replacement_selection,
    }
    (output_root / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    m23_37 = load_module(
        "m23_38_label_exact", "scripts/m23_research/m23_37_fast_exact_hota_teacher.py"
    )
    chain = load_module(
        "m23_38_label_chain", "scripts/m23_research/m23_12_chain_transaction_oracle.py"
    )
    evaluator = load_module(
        "m23_38_label_evaluator", "scripts/m23_research/m23_11_eval_utility_graph.py"
    )

    tracker = baseline_root / "track_results" / f"{seq}.txt"
    prepare_start = time.perf_counter()
    prepared = m23_37.PreparedExactHOTA(
        seq, tracker, output_root / "cache"
    )
    prepare_seconds = time.perf_counter() - prepare_start
    meta = pd.read_parquet(graph_root / seq / "microtracklets.parquet")
    edges = pd.read_parquet(graph_root / seq / "candidate_edges.parquet")
    predictions = pd.read_parquet(
        baseline_root / "predictions" / f"{seq}_predictions.parquet"
    )
    selected_path = baseline_root / f"{seq}_selected_transactions.parquet"
    applied_path = baseline_root / f"{seq}_applied_edges.parquet"
    selected = pd.read_parquet(selected_path)
    baseline_applied = pd.read_parquet(applied_path)
    if TARGET not in selected:
        if "policy_score" in selected:
            selected[TARGET] = selected.policy_score.to_numpy(float)
        else:
            selected[TARGET] = 1.0
    selected[TARGET] = selected[TARGET].fillna(1.0).astype(float)

    source_rows = evaluator.read_parent(source_parent_root / f"{seq}.txt")
    line_to_chunk = evaluator.line_chunks(source_rows, meta)
    parent_to_source = parent_to_source_indices(prepared, source_rows)
    baseline_ids = np.asarray(
        [int(float(fields[1])) for fields in prepared.parent_rows], dtype=np.int64
    )
    reconstructed_ids = row_ids_from_applied(
        baseline_applied,
        meta,
        line_to_chunk,
        parent_to_source,
        SEQUENCES.index(seq),
        evaluator,
    )
    if not np.array_equal(baseline_ids, reconstructed_ids):
        mismatch = int(np.count_nonzero(baseline_ids != reconstructed_ids))
        raise RuntimeError(f"baseline graph reconstruction mismatch: {mismatch} rows")
    baseline_metrics = prepared.evaluate_row_ids_incremental(baseline_ids)

    feature_columns = [
        column
        for column in predictions.columns
        if column
        not in {
            "same_gt",
            "src_modal_gt",
            "dst_modal_gt",
            "src_purity",
            "dst_purity",
            "label_confidence",
        }
    ]
    actions: List[Dict[str, object]] = []
    replacement_frame = residual_replacements(
        predictions,
        selected,
        args.max_replacements,
        selection_mode=args.replacement_selection,
    )
    feature_columns.extend(
        column
        for column in replacement_frame.columns
        if column not in feature_columns
        and column not in {
            "same_gt",
            "src_modal_gt",
            "dst_modal_gt",
            "src_purity",
            "dst_purity",
            "label_confidence",
        }
    )
    action_specs = []
    if not args.skip_drops:
        action_specs.extend(("drop", index, row) for index, row in selected.iterrows())
    action_specs.extend(
        ("replace", index, row) for index, row in replacement_frame.iterrows()
    )

    label_start = time.perf_counter()
    for action_ordinal, (action_type, source_index, row) in enumerate(action_specs):
        try:
            if action_type == "drop":
                modified = selected.drop(index=source_index).copy()
                removed_actions = 1
            else:
                modified, removed_actions = replace_action(selected, row)
            applied = chain.apply_transactions(edges, modified)
            candidate_ids = row_ids_from_applied(
                applied,
                meta,
                line_to_chunk,
                parent_to_source,
                SEQUENCES.index(seq),
                evaluator,
            )
            metrics = prepared.evaluate_row_ids_incremental(candidate_ids)
            record = {
                "status": "success",
                "seq": seq,
                "action_ordinal": action_ordinal,
                "action_type": action_type,
                "source_index": int(source_index),
                "removed_baseline_actions": removed_actions,
                "resulting_selected_actions": int(len(modified)),
                "changed_raw_rows": int(np.count_nonzero(candidate_ids != baseline_ids)),
                **compact_feature_row(row, feature_columns),
                **metric_delta(metrics, baseline_metrics),
            }
        except Exception as error:
            record = {
                "status": "failed",
                "seq": seq,
                "action_ordinal": action_ordinal,
                "action_type": action_type,
                "source_index": int(source_index),
                "error": repr(error),
            }
        actions.append(record)
        pd.DataFrame(actions).to_parquet(output_root / "exact_action_labels.parquet", index=False)
        print(
            json.dumps(
                {
                    "stage": "action_labeled",
                    "ordinal": action_ordinal,
                    "total": len(action_specs),
                    "type": action_type,
                    "status": record["status"],
                    "delta_HOTA": record.get("delta_HOTA"),
                    "teacher_seconds": record.get("teacher_seconds"),
                }
            ),
            flush=True,
        )

    successful = [row for row in actions if row["status"] == "success"]
    positive = [row for row in successful if float(row["delta_HOTA"]) > 0.0]
    report = {
        **protocol,
        "status": "completed",
        "prepare_seconds": prepare_seconds,
        "label_seconds": time.perf_counter() - label_start,
        "baseline_metrics": {
            key: value
            for key, value in baseline_metrics.items()
            if key in {"HOTA", "DetA", "AssA", "DetRe", "DetPr", "AssRe", "AssPr", "LocA"}
        },
        "baseline_selected_actions": int(len(selected)),
        "candidate_replacements": int(len(replacement_frame)),
        "attempted_actions": int(len(actions)),
        "successful_actions": int(len(successful)),
        "positive_actions": int(len(positive)),
        "negative_actions": int(
            sum(float(row["delta_HOTA"]) < 0.0 for row in successful)
        ),
        "zero_actions": int(
            sum(float(row["delta_HOTA"]) == 0.0 for row in successful)
        ),
        "best_action": (
            max(successful, key=lambda row: float(row["delta_HOTA"]))
            if successful
            else None
        ),
        "mean_teacher_seconds": (
            float(np.mean([row["teacher_seconds"] for row in successful]))
            if successful
            else None
        ),
    }
    (output_root / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_root / "protocol.json").write_text(
        json.dumps({**protocol, "status": "completed"}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"stage": "completed", **report}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
