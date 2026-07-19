#!/usr/bin/env python3
from __future__ import annotations

"""M23-50 strict transaction student with expert lower-bound abstention.

The pooled five-leaf HGB retains M23-49's ranking.  One additional five-leaf
expert is fit per outer-training sequence.  A candidate is eligible only when
the minimum source-domain expert probability exceeds a pre-registered
threshold.  Threshold and top-k are selected only by inner exact TrackEval,
maximizing the worst per-inner-sequence HOTA delta.  Held labels and GT remain
unread until the held tracker has been frozen.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd


SEQUENCES = ("MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05")
THRESHOLDS = (0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
TOP_K = (1, 2)


@dataclass(frozen=True)
class Policy:
    policy_id: str
    top_k: int
    min_expert_probability: float


def policies() -> List[Policy]:
    output = [Policy("noop", 0, 1.0)]
    for k in TOP_K:
        for threshold in THRESHOLDS:
            tag = f"p{threshold:.2f}".replace(".", "p")
            output.append(Policy(f"replace_top{k}_{tag}", k, threshold))
    return output


def fit_bundle(frame: pd.DataFrame, seed: int, m49):
    pooled = m49.fit_model(frame, seed)
    experts = []
    for offset, (seq, local) in enumerate(frame.groupby(frame.seq.astype(str))):
        experts.append((str(seq), m49.fit_model(local.reset_index(drop=True), seed + 100 + offset)))
    if not experts:
        raise RuntimeError("no source-domain experts")
    return {"pooled": pooled, "experts": experts}


def predict_bundle(frame: pd.DataFrame, bundle, m49) -> pd.DataFrame:
    output = m49.predict_replacements(frame, bundle["pooled"])
    output.rename(columns={"pure_hgb_probability": "pooled_probability"}, inplace=True)
    expert_columns = []
    expert_values = []
    for seq, model in bundle["experts"]:
        predicted = m49.predict_replacements(frame, model)
        column = f"expert_{seq.replace('MOT20-', 'M')}_probability"
        output[column] = predicted.pure_hgb_probability.to_numpy(float)
        expert_columns.append(column)
        expert_values.append(output[column].to_numpy(float))
    matrix = np.stack(expert_values, axis=1)
    output["expert_min_probability"] = np.min(matrix, axis=1)
    output["expert_mean_probability"] = np.mean(matrix, axis=1)
    output["expert_probability_std"] = np.std(matrix, axis=1)
    output["pure_hgb_probability"] = output.pooled_probability.to_numpy(float)
    output["student_value"] = output.pooled_probability.to_numpy(float)
    output.attrs["expert_columns"] = expert_columns
    return output


def select_policy(frame: pd.DataFrame, policy: Policy, m49) -> pd.DataFrame:
    if policy.top_k <= 0:
        return frame.iloc[:0].copy()
    eligible = frame[
        frame.expert_min_probability.to_numpy(float) >= policy.min_expert_probability
    ].copy()
    if eligible.empty:
        return frame.iloc[:0].copy()
    return m49.select_topk_replacements(eligible, policy.top_k)


def write_tracker(
    seq: str,
    predictions: pd.DataFrame,
    policy: Policy,
    graph_root: Path,
    parent: Path,
    output_path: Path,
    m39,
    m49,
    chain,
    evaluator,
):
    baseline_root = m39.BASELINE_ROOTS[seq]
    original_predictions = pd.read_parquet(
        baseline_root / "predictions" / f"{seq}_predictions.parquet"
    )
    baseline_selected = pd.read_parquet(
        baseline_root / f"{seq}_selected_transactions.parquet"
    )
    replacements = select_policy(predictions, policy, m49)
    drops = predictions.iloc[:0].copy()
    selected, removed = m39.build_selected_transactions(
        baseline_selected, original_predictions, drops, replacements
    )
    edges = pd.read_parquet(graph_root / seq / "candidate_edges.parquet")
    meta = pd.read_parquet(graph_root / seq / "microtracklets.parquet")
    applied = chain.apply_transactions(edges, selected)
    for column, default in (
        ("assa_edge_delta_proxy", 0.0),
        ("assa_edge_positive", 0),
        ("assa_edge_negative", 0),
    ):
        if column not in applied:
            applied[column] = default
    evaluator.DATA = graph_root
    evaluator.PARENT = parent
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tracker_report = evaluator.write_tracker(seq, meta, applied, output_path)
    return replacements, {
        "seq": seq,
        "replacement_actions": int(len(replacements)),
        "eligible_actions": int((predictions.expert_min_probability >= policy.min_expert_probability).sum())
        if policy.top_k > 0 else 0,
        "removed_baseline_actions": int(removed),
        "resulting_selected_actions": int(len(selected)),
        "selected_source_indices": replacements.source_index.astype(int).tolist()
        if len(replacements) else [],
        "mean_pooled_probability": float(replacements.pooled_probability.mean())
        if len(replacements) else 0.0,
        "minimum_selected_expert_probability": float(replacements.expert_min_probability.min())
        if len(replacements) else None,
        "tracker_report": tracker_report,
    }


def robust_key(row: Mapping[str, object]) -> Tuple[float, float, float, float, int, float]:
    return (
        float(row["worst_fold_delta_HOTA"]),
        float(row["mean_fold_delta_HOTA"]),
        float(row["HOTA"]),
        float(row["AssA"]),
        -int(row["selected_actions"]),
        float(row["min_expert_probability"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--held-seq", required=True, choices=SEQUENCES)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--graph-root", default=None)
    parser.add_argument("--parent", default=None)
    parser.add_argument("--max-replacements", type=int, default=256)
    parser.add_argument("--policy-limit", type=int, default=0)
    parser.add_argument("--skip-outer-trackeval", action="store_true")
    args = parser.parse_args()

    import importlib.util
    import sys

    repo = Path(__file__).resolve().parents[2]

    def load(name: str, relative: str):
        spec = importlib.util.spec_from_file_location(name, repo / relative)
        if spec is None or spec.loader is None:
            raise RuntimeError(relative)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    m39 = load("m23_50_m39", "scripts/m23_research/m23_39_exact_advantage_student_nested_loso.py")
    m49 = load("m23_50_m49", "scripts/m23_research/m23_49_pure_hgb_topk_transaction_student.py")
    eval_helper = load("m23_50_eval", "scripts/m23_research/m23_45_domain_robust_source_cut_student.py")
    label_module = load("m23_50_labels", "scripts/m23_research/m23_38_exact_hota_advantage_labels.py")
    chain = load("m23_50_chain", "scripts/m23_research/m23_12_chain_transaction_oracle.py")
    evaluator = load("m23_50_evaluator", "scripts/m23_research/m23_11_eval_utility_graph.py")
    original_within = m39.within_sequence_features

    held = args.held_seq
    training_sequences = [seq for seq in SEQUENCES if seq != held]
    output_root = Path(args.output_root)
    graph_root = Path(args.graph_root) if args.graph_root else m39.DEFAULT_GRAPH_ROOT
    parent = Path(args.parent) if args.parent else m39.DEFAULT_PARENT
    output_root.mkdir(parents=True, exist_ok=True)
    candidate_policies = policies()
    if args.policy_limit > 0:
        candidate_policies = candidate_policies[:args.policy_limit]

    protocol = {
        "experiment": "M23-50 expert lower-bound transaction student",
        "protocol": "strict nested sequence LOSO",
        "deployable": True,
        "outer_held_sequence": held,
        "outer_training_sequences": training_sequences,
        "training_gt_use": "existing M23-41 exact transaction labels from outer-training sequences only",
        "held_candidate_gt_use": "none",
        "label_regeneration": False,
        "candidate_space": "M23-41 diverse GT-free transaction shortlist",
        "ranking": "pooled five-leaf HGB probability",
        "abstention": "minimum single-training-sequence HGB probability",
        "threshold_grid": list(THRESHOLDS),
        "top_k_grid": list(TOP_K),
        "policy_selection": "maximum worst per-inner-sequence HOTA delta; then mean delta and stitched HOTA",
        "features": list(m49.COMPACT_FEATURES),
        "held_label_file_read": False,
        "held_gt_read_before_tracker_freeze": False,
        "status": "running",
    }
    (output_root / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n"
    )

    training_frames = {
        seq: m49.load_training_frame(seq, m49.LABEL_PATHS[seq], original_within)
        for seq in training_sequences
    }
    print(json.dumps({
        "stage": "training_labels_ready",
        "frames": {
            seq: {
                "rows": int(len(frame)),
                "replace_rows": int((frame.action_type.astype(str) == "replace").sum()),
                "replace_positive": int(((frame.action_type.astype(str) == "replace") & (frame.delta_HOTA > 0.0)).sum()),
            }
            for seq, frame in training_frames.items()
        },
    }), flush=True)

    inner_predictions: Dict[str, pd.DataFrame] = {}
    for fold_index, pseudo_held in enumerate(training_sequences):
        train = pd.concat(
            [training_frames[seq] for seq in training_sequences if seq != pseudo_held],
            ignore_index=True,
            sort=False,
        )
        bundle = fit_bundle(train, 50000 + fold_index, m49)
        prediction = predict_bundle(training_frames[pseudo_held], bundle, m49)
        inner_predictions[pseudo_held] = prediction
        path = output_root / "inner_predictions" / f"{pseudo_held}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        prediction.to_parquet(path, index=False)
        print(json.dumps({
            "stage": "inner_prediction",
            "pseudo_held": pseudo_held,
            "replacement_rows": int(len(prediction)),
            "experts": [seq for seq, _model in bundle["experts"]],
        }), flush=True)

    inner_rows: List[Dict[str, object]] = []
    baseline_metrics = None
    for policy_index, policy in enumerate(candidate_policies):
        candidate_root = output_root / "inner_candidates" / policy.policy_id
        track_root = candidate_root / "track_results"
        reports = []
        selected_actions = 0
        for seq in training_sequences:
            selected, tracker_report = write_tracker(
                seq, inner_predictions[seq], policy, graph_root, parent,
                track_root / f"{seq}.txt", m39, m49, chain, evaluator,
            )
            selected_actions += int(len(selected))
            reports.append(tracker_report)
        metrics = eval_helper.evaluate_detailed(
            track_root, candidate_root,
            f"m23_50_inner_{held}_{policy_index}", training_sequences,
        )
        if policy.top_k == 0:
            baseline_metrics = {seq: metrics[seq] for seq in training_sequences}
        if baseline_metrics is None:
            raise RuntimeError("no-op must be first")
        deltas = {
            seq: float(metrics[seq]["HOTA"] - baseline_metrics[seq]["HOTA"])
            for seq in training_sequences
        }
        row: Dict[str, object] = {
            "policy_id": policy.policy_id,
            "top_k": int(policy.top_k),
            "min_expert_probability": float(policy.min_expert_probability),
            **metrics["COMBINED"],
            "selected_actions": int(selected_actions),
            "worst_fold_delta_HOTA": float(min(deltas.values())),
            "mean_fold_delta_HOTA": float(np.mean(list(deltas.values()))),
            "positive_inner_folds": int(sum(value > 0.0 for value in deltas.values())),
        }
        for seq in training_sequences:
            tag = seq.replace("MOT20-", "M")
            row[f"{tag}_HOTA"] = float(metrics[seq]["HOTA"])
            row[f"{tag}_delta_HOTA"] = float(deltas[seq])
        inner_rows.append(row)
        (candidate_root / "tracker_reports.json").write_text(
            json.dumps(reports, indent=2, sort_keys=True) + "\n"
        )
        eval_helper.write_csv(output_root / "inner_metrics.csv", inner_rows)
        print(json.dumps({"stage": "inner_trackeval", **row}), flush=True)

    chosen_row = max(inner_rows, key=robust_key)
    chosen_policy = next(
        policy for policy in candidate_policies
        if policy.policy_id == chosen_row["policy_id"]
    )
    frozen_selection = {
        "selected_policy": chosen_policy.__dict__,
        "selected_inner_metrics": chosen_row,
        "selection_rule": "maximum worst-fold exact HOTA delta; then mean delta, stitched HOTA, AssA, fewer actions",
        "held_label_file_read": False,
        "outer_held_gt_read": False,
    }
    (output_root / "frozen_inner_selection.json").write_text(
        json.dumps(frozen_selection, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"stage": "policy_frozen", **chosen_row}), flush=True)

    outer_train = pd.concat(list(training_frames.values()), ignore_index=True, sort=False)
    held_frame = m49.diverse_held_frame(
        held, args.max_replacements, m39, label_module, original_within
    )
    outer_bundle = fit_bundle(outer_train, 50999, m49)
    held_predictions = predict_bundle(held_frame, outer_bundle, m49)
    held_predictions.to_parquet(
        output_root / "outer_action_predictions.parquet", index=False
    )
    outer_track_root = output_root / "outer_final" / "track_results"
    selected, tracker_report = write_tracker(
        held, held_predictions, chosen_policy, graph_root, parent,
        outer_track_root / f"{held}.txt", m39, m49, chain, evaluator,
    )
    selected.to_parquet(
        output_root / "outer_selected_replacements.parquet", index=False
    )
    forbidden = [
        column for column in held_predictions.columns
        if any(token in column.lower() for token in m49.FORBIDDEN_HELD_TOKENS)
    ]
    if forbidden:
        raise RuntimeError(f"held prediction leakage columns: {forbidden}")
    frozen_tracker = {
        "held_sequence": held,
        "held_label_file_read": False,
        "held_gt_read": False,
        "held_candidates": int(len(held_predictions)),
        "held_candidate_forbidden_columns": [],
        "source_domain_experts": [seq for seq, _model in outer_bundle["experts"]],
        "selected_policy": chosen_policy.__dict__,
        "selected_replacements": int(len(selected)),
        "selected_source_indices": selected.source_index.astype(int).tolist()
        if len(selected) else [],
        "tracker_report": tracker_report,
        "tracker_path": str(outer_track_root / f"{held}.txt"),
    }
    (output_root / "outer_tracker_frozen.json").write_text(
        json.dumps(frozen_tracker, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"stage": "outer_tracker_frozen", **frozen_tracker}), flush=True)

    evaluation = None
    if not args.skip_outer_trackeval:
        evaluation = eval_helper.evaluate_detailed(
            outer_track_root, output_root / "outer_eval",
            f"m23_50_outer_{held}", (held,),
        )["COMBINED"]
        print(json.dumps({"stage": "outer_trackeval", **evaluation}), flush=True)

    report = {
        **protocol,
        "status": "completed" if evaluation is not None else "tracker_frozen",
        "training_rows": {seq: int(len(frame)) for seq, frame in training_frames.items()},
        "inner_candidates": inner_rows,
        "frozen_inner_selection": frozen_selection,
        "held_candidate_actions": int(len(held_predictions)),
        "held_selected_replacements": int(len(selected)),
        "tracker_report": tracker_report,
        "eval": evaluation,
        "strict_m23_25_baseline": m39.STRICT_M23_25[held],
        "strict_m23_39_baseline": m49.STRICT_M23_39[held],
        "strict_m23_46_baseline": m49.STRICT_M23_46[held],
        "delta_vs_strict_m23_25": m49.metric_delta(evaluation, m39.STRICT_M23_25[held])
        if evaluation else None,
        "delta_vs_strict_m23_39": m49.metric_delta(evaluation, m49.STRICT_M23_39[held])
        if evaluation else None,
        "delta_vs_strict_m23_46": m49.metric_delta(evaluation, m49.STRICT_M23_46[held])
        if evaluation else None,
    }
    (output_root / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    protocol["status"] = report["status"]
    (output_root / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
