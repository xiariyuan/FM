#!/usr/bin/env python3
from __future__ import annotations

"""Evaluate rank ensembles of strict-OOF M23-28 base and stack predictions.

All candidate features and model predictions are GT-free. Held train GT is used
only by final TrackEval and post-hoc diagnostics, so these results are intended
for train-set deployment policy selection rather than unbiased validation.
"""

import argparse
import csv
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
SEQUENCES = ("MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05")
ROOT = REPO / "outputs" / "mot20_m23_20260718"
DEFAULT_SOURCE = ROOT / "m23_28_structured_oracle_base"
DEFAULT_STACK = ROOT / "m23_28_structured_oracle_stack"
DEFAULT_GRAPH = ROOT / "m23_26_test_deploy_oof_ensemble_v1" / "train_oof_micrograph"
DEFAULT_PARENT = (
    REPO
    / "outputs"
    / "assocriskbench_p15_20260714"
    / "fused_a42_adaptive_risk_eval"
    / "l1_r2_q75_adaptive_priority"
    / "track_results"
)
TARGET = "chain_transaction_delta_proxy"
ORACLE_TARGET = "structured_oracle_selected"


def load_module(name: str, relative_path: str):
    path = REPO / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True).to_numpy(float)


def parse_specs(raw: str) -> List[Tuple[str, float]]:
    output = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        name, quantile = item.rsplit(":", 1)
        output.append((name, float(quantile)))
    if not output:
        raise ValueError("empty candidate specs")
    return output


def score_from_name(
    name: str,
    base_rank: np.ndarray,
    stack_rank: np.ndarray,
    utility_rank: np.ndarray,
) -> np.ndarray:
    match = re.fullmatch(r"b([0-9.]+)_s([0-9.]+)_u([0-9.]+)", name)
    if match:
        wb, ws, wu = (float(value) for value in match.groups())
        denominator = wb + ws + wu
        if denominator <= 0:
            raise ValueError(f"invalid zero-weight ensemble: {name}")
        return (wb * base_rank + ws * stack_rank + wu * utility_rank) / denominator
    if name == "min_bs":
        return np.minimum(base_rank, stack_rank)
    if name == "geom_bs":
        return np.sqrt(base_rank * stack_rank)
    if name == "prod_bsu":
        return np.cbrt(base_rank * stack_rank * utility_rank)
    if name == "min_bsu":
        return np.minimum(np.minimum(base_rank, stack_rank), utility_rank)
    raise ValueError(f"unknown ensemble name: {name}")


def evaluate_held(
    track_results: Path,
    held: str,
    tracker_name: str,
    candidate_root: Path,
) -> Dict[str, float]:
    work_dir = candidate_root / "eval_work"
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
    completed = subprocess.run(
        command,
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    (candidate_root / "eval.log").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(completed.stdout[-5000:])
    detailed = work_dir / "eval" / tracker_name / "pedestrian_detailed.csv"
    with detailed.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
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
    parser.add_argument("--base-root", default=str(DEFAULT_SOURCE))
    parser.add_argument("--stack-root", default=str(DEFAULT_STACK))
    parser.add_argument("--graph-root", default=str(DEFAULT_GRAPH))
    parser.add_argument("--parent", default=str(DEFAULT_PARENT))
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--candidate-specs", required=True)
    args = parser.parse_args()

    held = args.held_seq
    base_root = Path(args.base_root).resolve()
    stack_root = Path(args.stack_root).resolve()
    graph_root = Path(args.graph_root).resolve()
    parent = Path(args.parent).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    m28 = load_module("m23_30_m28", "scripts/m23_research/m23_28_structured_oracle_imitation_loso.py")
    chain = load_module("m23_30_chain", "scripts/m23_research/m23_12_chain_transaction_oracle.py")
    evaluator = load_module("m23_30_evaluator", "scripts/m23_research/m23_11_eval_utility_graph.py")

    base = pd.read_parquet(base_root / held / "held_predictions.parquet")
    stack = pd.read_parquet(stack_root / held / "held_predictions.parquet")
    for column in ("src_chunk", "dst_chunk"):
        if not np.array_equal(base[column].to_numpy(), stack[column].to_numpy()):
            raise RuntimeError(f"base/stack row mismatch: {column}")

    base_rank = rank(base.oracle_selection_probability.to_numpy(float))
    stack_rank = rank(stack.oracle_selection_probability.to_numpy(float))
    probability = base.pred_positive_probability.to_numpy(float)
    utility_score = (
        probability * base.pred_normalized_gain.to_numpy(float)
        - (1.0 - probability) * base.pred_normalized_loss.to_numpy(float)
    )
    utility_rank = rank(utility_score)

    evaluator.DATA = graph_root
    evaluator.PARENT = parent
    evaluator.SEQS = [held]
    meta = pd.read_parquet(graph_root / held / "microtracklets.parquet")
    edges = pd.read_parquet(graph_root / held / "candidate_edges.parquet")

    rows = []
    for ensemble_name, quantile in parse_specs(args.candidate_specs):
        candidate_id = f"{ensemble_name}_q{quantile:.5f}".replace(".", "p")
        candidate_root = output_root / "candidates" / candidate_id
        track_root = candidate_root / "track_results"
        selected_root = candidate_root / "selected_transactions"
        track_root.mkdir(parents=True, exist_ok=True)
        selected_root.mkdir(parents=True, exist_ok=True)

        frame = base.copy()
        score_column = "ensemble_score"
        frame[score_column] = score_from_name(
            ensemble_name, base_rank, stack_rank, utility_rank
        )
        selected = m28.maximum_weight_matching(frame, score_column, quantile)
        selected.to_parquet(selected_root / f"{held}.parquet", index=False)
        applied = chain.apply_transactions(
            edges,
            selected.assign(**{TARGET: selected.policy_score.to_numpy(float)}),
        )
        for column, default in (
            ("assa_edge_delta_proxy", 0.0),
            ("assa_edge_positive", 0),
            ("assa_edge_negative", 0),
        ):
            if column not in applied:
                applied[column] = default
        tracker_report = evaluator.write_tracker(
            held,
            meta,
            applied,
            track_root / f"{held}.txt",
        )
        metrics = evaluate_held(
            track_root,
            held,
            f"m23_30_{candidate_id}",
            candidate_root,
        )
        oracle = selected[ORACLE_TARGET].to_numpy(int)
        row = {
            "candidate_id": candidate_id,
            "held_sequence": held,
            "ensemble_name": ensemble_name,
            "score_quantile": quantile,
            "selected_actions": int(len(selected)),
            "selected_oracle_actions": int(oracle.sum()),
            "selected_oracle_precision": float(oracle.mean()) if len(oracle) else 0.0,
            "selected_proxy_sum": float(selected[TARGET].sum()) if len(selected) else 0.0,
            "selected_edge_proxy_sum": (
                float(selected.assa_edge_delta_proxy.sum()) if len(selected) else 0.0
            ),
            **metrics,
            **tracker_report,
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    fields = sorted({key for row in rows for key in row})
    with (output_root / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "status": "completed",
        "held_sequence": held,
        "protocol": "strict-OOF prediction rank ensemble; held GT only for TrackEval/model selection",
        "best": max(rows, key=lambda row: float(row["HOTA"])),
        "candidates": rows,
    }
    (output_root / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
