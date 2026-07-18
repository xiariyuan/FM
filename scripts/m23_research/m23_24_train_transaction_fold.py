#!/usr/bin/env python3
from __future__ import annotations

"""Train and execute one strict M23-24 outer transaction fold.

All utility labels, inner policy selection and model fitting use only the three
outer-training sequences.  The outer-held graph is structural/appearance-only;
its GT is opened only by the final TrackEval subprocess.
"""

import argparse
import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
SEQUENCES = ("MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05")
DEFAULT_PARENT = "outputs/assocriskbench_p15_20260714/fused_a42_adaptive_risk_eval/l1_r2_q75_adaptive_priority/track_results"


def load_module(name: str, relative_path: str):
    path = REPO / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def structural_transactions(meta: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    cross = edges[edges.same_source.to_numpy(int) == 0].copy()
    meta = meta.sort_values("chunk_id")
    if meta.chunk_id.tolist() != list(range(len(meta))):
        raise RuntimeError("chunk ids are not dense")
    track_for_chunk = meta.source_track_id.to_numpy(int)
    ordinal_for_chunk = meta.source_ordinal.to_numpy(int)
    chunks_per_track = meta.groupby("source_track_id").source_ordinal.max().add(1).to_dict()
    src = cross.src_chunk.to_numpy(int)
    dst = cross.dst_chunk.to_numpy(int)
    src_track = track_for_chunk[src]
    dst_track = track_for_chunk[dst]
    src_ordinal = ordinal_for_chunk[src]
    dst_ordinal = ordinal_for_chunk[dst]
    cross["transaction_src_track_id"] = src_track
    cross["transaction_dst_track_id"] = dst_track
    cross["transaction_removes_source_out"] = [
        int(ordinal + 1 < chunks_per_track[int(track)])
        for ordinal, track in zip(src_ordinal, src_track)
    ]
    cross["transaction_removes_source_in"] = (dst_ordinal > 0).astype(np.int8)
    return cross


def select_disjoint(frame: pd.DataFrame, maximum_actions: int, score_column: str) -> pd.DataFrame:
    if maximum_actions <= 0 or frame.empty:
        return frame.iloc[:0].copy()
    ordered = frame.sort_values(
        [score_column, "src_chunk", "dst_chunk"], ascending=[False, True, True]
    )
    used_tracks = set()
    selected = []
    for index, row in ordered.iterrows():
        if not np.isfinite(float(row[score_column])) or float(row[score_column]) <= 0:
            continue
        src_track = int(row.transaction_src_track_id)
        dst_track = int(row.transaction_dst_track_id)
        if src_track in used_tracks or dst_track in used_tracks:
            continue
        used_tracks.add(src_track)
        used_tracks.add(dst_track)
        selected.append(index)
        if len(selected) >= maximum_actions:
            break
    return frame.loc[selected].copy()


def evaluate_held(track_results: Path, held: str, tracker_name: str, work_dir: Path) -> Dict[str, float]:
    command = [
        sys.executable,
        "scripts/eval_motstyle_trackeval.py",
        "--benchmark-name",
        "MOT20",
        "--split-to-eval",
        "train",
        "--gt-root",
        "datasets/MOT20/train",
        "--results-dir",
        str(track_results),
        "--tracker-name",
        tracker_name,
        "--work-dir",
        str(work_dir),
        "--seqs",
        held,
    ]
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (work_dir.parent / "eval.log").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(completed.stdout[-5000:])
    detailed = work_dir / "eval" / tracker_name / "pedestrian_detailed.csv"
    rows = list(csv.DictReader(detailed.open(encoding="utf-8")))
    row = next(item for item in rows if item["seq"] in {held, "COMBINED"})
    return {
        "HOTA": 100.0 * float(row["HOTA___AUC"]),
        "DetA": 100.0 * float(row["DetA___AUC"]),
        "AssA": 100.0 * float(row["AssA___AUC"]),
        "IDSW": int(float(row["IDSW"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--held-seq", required=True, choices=SEQUENCES)
    parser.add_argument("--graph-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--parent", default=DEFAULT_PARENT)
    parser.add_argument("--budget-grid", default="0,1,2,4,8,16,32,64,128,256")
    args = parser.parse_args()

    held = args.held_seq
    training_sequences = [seq for seq in SEQUENCES if seq != held]
    graph_root = Path(args.graph_root)
    output_root = Path(args.output_root)
    parent = Path(args.parent)
    utility_root = output_root / "training_edge_utility"
    label_root = output_root / "training_transaction_labels"
    prediction_root = output_root / "predictions"
    track_root = output_root / "track_results"
    for path in [utility_root, label_root, prediction_root, track_root]:
        path.mkdir(parents=True, exist_ok=True)

    utility = load_module("m23_24_utility", "scripts/m23_research/m23_11_add_micrograph_utility.py")
    chain = load_module("m23_24_chain", "scripts/m23_research/m23_12_chain_transaction_oracle.py")
    trainer = load_module("m23_24_trainer", "scripts/m23_research/m23_12_train_chain_transaction_loso.py")
    evaluator = load_module("m23_24_evaluator", "scripts/m23_research/m23_11_eval_utility_graph.py")
    utility.ROOT = graph_root
    utility.OUT = utility_root
    utility.PARENT = parent
    chain.DATA = graph_root
    chain.UTILITY = utility_root
    chain.OUT = output_root / "unused_oracle_output"
    chain.PARENT = parent
    trainer.META = graph_root
    evaluator.DATA = graph_root
    evaluator.PARENT = parent

    # Labels are generated only for outer-training sequences.
    m23 = utility.load_m23()
    label_reports = []
    training_frames: Dict[str, pd.DataFrame] = {}
    for seq in training_sequences:
        label_reports.append(utility.label_sequence(seq, m23))
        _, _, transactions = chain.label_transactions(seq, utility)
        sequence_dir = label_root / seq
        sequence_dir.mkdir(parents=True, exist_ok=True)
        transactions.to_parquet(sequence_dir / "cross_chain_transaction_utility.parquet", index=False)
        training_frames[seq] = trainer.add_chain_features(seq, transactions)

    # Strict inner leave-one-training-sequence-out prediction for budget selection.
    score_column = "pred_risk_adjusted_transaction_utility"
    inner_predictions: Dict[str, pd.DataFrame] = {}
    for pseudo_index, pseudo_held in enumerate(training_sequences):
        inner_train = pd.concat(
            [training_frames[seq] for seq in training_sequences if seq != pseudo_held],
            ignore_index=True,
            sort=False,
        )
        classifier, regressor = trainer.fit_models(inner_train, 2400 + pseudo_index)
        predicted = trainer.predict(training_frames[pseudo_held], classifier, regressor)
        predicted.to_parquet(prediction_root / f"inner_{pseudo_held}.parquet", index=False)
        inner_predictions[pseudo_held] = predicted

    budgets = sorted({int(value) for value in args.budget_grid.split(",") if value.strip()})
    budget_reports = []
    for budget in budgets:
        by_sequence = []
        for seq in training_sequences:
            selected = select_disjoint(inner_predictions[seq], budget, score_column)
            by_sequence.append(
                {
                    "seq": seq,
                    "actions": len(selected),
                    "actual_proxy_sum": float(selected.chain_transaction_delta_proxy.sum()),
                    "positive": int((selected.chain_transaction_delta_proxy > 0).sum()),
                    "negative": int((selected.chain_transaction_delta_proxy < 0).sum()),
                }
            )
        budget_reports.append(
            {
                "budget": budget,
                "actual_proxy_sum": float(sum(row["actual_proxy_sum"] for row in by_sequence)),
                "actions": int(sum(row["actions"] for row in by_sequence)),
                "by_sequence": by_sequence,
            }
        )
    # Deterministic tie break prefers the smaller action budget.
    chosen = max(budget_reports, key=lambda row: (row["actual_proxy_sum"], -row["budget"]))
    chosen_budget = int(chosen["budget"])

    # Fit on all three outer-training sequences and infer the held graph.
    outer_train = pd.concat(list(training_frames.values()), ignore_index=True, sort=False)
    classifier, regressor = trainer.fit_models(outer_train, 2499)
    held_meta = pd.read_parquet(graph_root / held / "microtracklets.parquet")
    held_edges = pd.read_parquet(graph_root / held / "candidate_edges.parquet")
    held_structural = structural_transactions(held_meta, held_edges)
    held_features = trainer.add_chain_features(held, held_structural)
    held_predictions = trainer.predict(held_features, classifier, regressor)
    held_predictions.to_parquet(prediction_root / f"{held}_predictions.parquet", index=False)
    selected = select_disjoint(held_predictions, chosen_budget, score_column)
    selected = selected.copy()
    selected["chain_transaction_delta_proxy"] = selected[score_column].to_numpy(float)
    selected.to_parquet(output_root / f"{held}_selected_transactions.parquet", index=False)
    applied = chain.apply_transactions(held_edges, selected)
    for column, default in [
        ("assa_edge_delta_proxy", 0.0),
        ("assa_edge_positive", 0),
        ("assa_edge_negative", 0),
    ]:
        if column not in applied:
            applied[column] = default
    applied.to_parquet(output_root / f"{held}_applied_edges.parquet", index=False)
    tracker_report = evaluator.write_tracker(held, held_meta, applied, track_root / f"{held}.txt")

    tracker_name = "m23_24_finetuned_transaction"
    evaluation = evaluate_held(track_root, held, tracker_name, output_root / "eval_work")
    report = {
        "experiment": "M23-24 strict outer-fold fine-tuned appearance transaction policy",
        "held_sequence": held,
        "training_sequences": training_sequences,
        "held_gt_read_before_frozen_tracker": False,
        "held_gt_use": "final TrackEval only",
        "training_gt_use": "outer-training edge/transaction utility labels and inner budget selection",
        "inference": "GT-free",
        "budget_selection": {
            "protocol": "inner leave-one-training-sequence-out; maximize summed disjoint transaction proxy",
            "grid": budgets,
            "chosen_budget": chosen_budget,
            "candidates": budget_reports,
        },
        "label_reports": label_reports,
        "held_candidates": len(held_predictions),
        "held_selected_actions": len(selected),
        "held_predicted_utility_sum": float(selected[score_column].sum()),
        "tracker_report": tracker_report,
        "eval": evaluation,
    }
    (output_root / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
