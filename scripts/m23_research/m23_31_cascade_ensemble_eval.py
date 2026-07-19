#!/usr/bin/env python3
from __future__ import annotations

"""Evaluate two-stage core-plus-residual strict-OOF rank ensembles."""

import argparse
import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
SEQUENCES = ("MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05")
ROOT = REPO / "outputs" / "mot20_m23_20260718"
DEFAULT_BASE = ROOT / "m23_28_structured_oracle_base"
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


def parse_specs(raw: str) -> List[Tuple[str, float, str, float]]:
    output = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        core_name, core_quantile, broad_name, broad_quantile = item.split(":")
        output.append(
            (core_name, float(core_quantile), broad_name, float(broad_quantile))
        )
    if not output:
        raise ValueError("empty cascade specs")
    return output


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
    parser.add_argument("--base-root", default=str(DEFAULT_BASE))
    parser.add_argument("--stack-root", default=str(DEFAULT_STACK))
    parser.add_argument("--graph-root", default=str(DEFAULT_GRAPH))
    parser.add_argument("--parent", default=str(DEFAULT_PARENT))
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--cascade-specs", required=True)
    args = parser.parse_args()

    held = args.held_seq
    base_root = Path(args.base_root).resolve()
    stack_root = Path(args.stack_root).resolve()
    graph_root = Path(args.graph_root).resolve()
    parent = Path(args.parent).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    m28 = load_module("m23_31_m28", "scripts/m23_research/m23_28_structured_oracle_imitation_loso.py")
    m30 = load_module("m23_31_m30", "scripts/m23_research/m23_30_rank_ensemble_eval.py")
    chain = load_module("m23_31_chain", "scripts/m23_research/m23_12_chain_transaction_oracle.py")
    evaluator = load_module("m23_31_evaluator", "scripts/m23_research/m23_11_eval_utility_graph.py")

    base = pd.read_parquet(base_root / held / "held_predictions.parquet")
    stack = pd.read_parquet(stack_root / held / "held_predictions.parquet")
    for column in ("src_chunk", "dst_chunk"):
        if not np.array_equal(base[column].to_numpy(), stack[column].to_numpy()):
            raise RuntimeError(f"base/stack row mismatch: {column}")
    base_rank = m30.rank(base.oracle_selection_probability.to_numpy(float))
    stack_rank = m30.rank(stack.oracle_selection_probability.to_numpy(float))
    probability = base.pred_positive_probability.to_numpy(float)
    utility_rank = m30.rank(
        probability * base.pred_normalized_gain.to_numpy(float)
        - (1.0 - probability) * base.pred_normalized_loss.to_numpy(float)
    )

    evaluator.DATA = graph_root
    evaluator.PARENT = parent
    evaluator.SEQS = [held]
    meta = pd.read_parquet(graph_root / held / "microtracklets.parquet")
    edges = pd.read_parquet(graph_root / held / "candidate_edges.parquet")

    rows = []
    for core_name, core_quantile, broad_name, broad_quantile in parse_specs(
        args.cascade_specs
    ):
        candidate_id = (
            f"core_{core_name}_q{core_quantile:.5f}__"
            f"broad_{broad_name}_q{broad_quantile:.5f}"
        ).replace(".", "p")
        candidate_root = output_root / "candidates" / candidate_id
        track_root = candidate_root / "track_results"
        selected_root = candidate_root / "selected_transactions"
        track_root.mkdir(parents=True, exist_ok=True)
        selected_root.mkdir(parents=True, exist_ok=True)

        frame = base.copy()
        frame["core_score"] = m30.score_from_name(
            core_name, base_rank, stack_rank, utility_rank
        )
        frame["broad_score"] = m30.score_from_name(
            broad_name, base_rank, stack_rank, utility_rank
        )
        core = m28.maximum_weight_matching(frame, "core_score", core_quantile)
        used_tracks = set(core.transaction_src_track_id.astype(int)) | set(
            core.transaction_dst_track_id.astype(int)
        )
        residual = frame[
            ~frame.transaction_src_track_id.isin(used_tracks)
            & ~frame.transaction_dst_track_id.isin(used_tracks)
        ].copy()
        additional = m28.maximum_weight_matching(
            residual, "broad_score", broad_quantile
        )
        selected = pd.concat([core, additional], ignore_index=False).drop_duplicates(
            subset=["src_chunk", "dst_chunk"]
        )
        selected["policy_score"] = np.concatenate(
            [
                2.0 + core.core_score.to_numpy(float),
                additional.broad_score.to_numpy(float),
            ]
        )
        selected.sort_values("policy_score", ascending=False, inplace=True)
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
            f"m23_31_{candidate_id}",
            candidate_root,
        )
        oracle = selected[ORACLE_TARGET].to_numpy(int)
        row = {
            "candidate_id": candidate_id,
            "held_sequence": held,
            "core_name": core_name,
            "core_quantile": core_quantile,
            "broad_name": broad_name,
            "broad_quantile": broad_quantile,
            "core_actions": int(len(core)),
            "additional_actions": int(len(additional)),
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
        "protocol": "strict-OOF core-plus-residual rank ensemble; held GT for TrackEval/model selection only",
        "best": max(rows, key=lambda row: float(row["HOTA"])),
        "candidates": rows,
    }
    (output_root / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
