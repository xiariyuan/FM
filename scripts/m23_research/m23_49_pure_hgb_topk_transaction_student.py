#!/usr/bin/env python3
from __future__ import annotations

"""M23-49 strict transaction student: compact pure-HGB + robust top-k.

This reuses the already-completed M23-41 diverse exact-HOTA transaction label
banks.  No labels are regenerated.  A separate five-leaf HGB replacement head
is trained only on outer-training sequences.  Candidate policies are no-op and
replacement top-k for k in {1,2,4,8}; replacements are resolved by maximum-
weight matching.  Policy selection maximizes the worst per-inner-sequence exact
HOTA delta, then mean delta and stitched HOTA.  The outer-held label file and GT
are not read before the tracker is frozen.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


REPO = Path(__file__).resolve().parents[2]
SEQUENCES = ("MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05")
LABEL_PATHS = {
    seq: Path(
        f"outputs/mot20_m23_20260718/m23_41_diverse_exact_advantage_{seq[-2:]}_v1/"
        "exact_action_labels.parquet"
    )
    for seq in SEQUENCES
}
STRICT_M23_39 = {
    "MOT20-01": {"HOTA": 78.805125, "DetA": 81.837410, "AssA": 76.043165, "IDSW": 46},
    "MOT20-02": {"HOTA": 73.098150, "DetA": 80.584820, "AssA": 66.407293, "IDSW": 325},
    "MOT20-03": {"HOTA": 80.603280, "DetA": 81.174386, "AssA": 80.068130, "IDSW": 146},
    "MOT20-05": {"HOTA": 79.732850, "DetA": 81.954850, "AssA": 77.612920, "IDSW": 478},
}
STRICT_M23_46 = {
    **STRICT_M23_39,
    "MOT20-05": {"HOTA": 79.770327, "DetA": 81.954810, "AssA": 77.685820, "IDSW": 479},
}
COMPACT_FEATURES = (
    "m23_38_parent_policy_score",
    "pred_positive_probability",
    "motion_appearance_joint",
    "max_margin",
    "backward_motion_error",
    "motion_error_mean",
    "motion_error_min",
    "sequence_motion_percentile",
    "motion_rank_consensus",
    "rank_consensus",
    "appearance_cos",
    "segment_appearance_cos",
    "segment_endpoint_gain",
    "max_rank",
    "channel_parent_policy",
)
CHANNEL_FEATURES = (
    "channel_parent_policy",
    "channel_motion",
    "channel_appearance",
    "channel_structure",
    "channel_exploration",
)
FORBIDDEN_HELD_TOKENS = (
    "same_gt", "modal_gt", "purity", "label_confidence", "exact_hota",
    "delta_hota", "teacher", "actual_assa",
)
POLICY_K = (0, 1, 2, 4, 8)


def load_module(name: str, relative_path: str):
    path = REPO / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def add_channel_features(frame: pd.DataFrame, original_within) -> pd.DataFrame:
    output = original_within(frame)
    channels = output.get(
        "m23_41_selection_channel", pd.Series("drop", index=output.index)
    ).fillna("drop").astype(str)
    groups = {
        "channel_parent_policy": {"parent_policy"},
        "channel_motion": {
            "motion_error_mean", "motion_error_min", "motion_consensus", "rank_consensus"
        },
        "channel_appearance": {"appearance_consensus", "track_appearance"},
        "channel_structure": {
            "low_detachment", "low_conflict_density", "large_merged_segment",
            "large_detached_segment",
        },
        "channel_exploration": {
            "parent_boundary", "high_entropy", "positive_probability", "predicted_gain"
        },
    }
    additions = {
        feature: channels.isin(names).astype(float)
        for feature, names in groups.items()
    }
    return pd.concat([output, pd.DataFrame(additions, index=output.index)], axis=1).copy()


def load_training_frame(seq: str, path: Path, original_within) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame = frame[frame.status.astype(str) == "success"].copy()
    if frame.empty:
        raise RuntimeError(f"empty labels: {path}")
    if frame.seq.astype(str).nunique() != 1 or str(frame.seq.iloc[0]) != seq:
        raise RuntimeError(f"label sequence mismatch: {seq} {path}")
    if "delta_HOTA" not in frame:
        raise RuntimeError(f"missing exact target: {path}")
    output = add_channel_features(frame, original_within)
    missing = [feature for feature in COMPACT_FEATURES if feature not in output]
    if missing:
        raise RuntimeError(f"missing compact features for {seq}: {missing}")
    return output


def fit_model(frame: pd.DataFrame, seed: int):
    replace = frame[frame.action_type.astype(str) == "replace"].copy()
    if replace.empty:
        raise RuntimeError("no replacement training actions")
    labels = (replace.delta_HOTA.to_numpy(float) > 0.0).astype(int)
    if np.unique(labels).size < 2:
        return {"kind": "constant", "probability": float(labels.mean())}
    model = HistGradientBoostingClassifier(
        learning_rate=0.04,
        max_iter=180,
        max_leaf_nodes=5,
        min_samples_leaf=8,
        l2_regularization=12.0,
        early_stopping=False,
        random_state=seed,
    )
    model.fit(replace[list(COMPACT_FEATURES)].to_numpy(float), labels)
    return {"kind": "hgb", "model": model}


def predict_replacements(frame: pd.DataFrame, bundle) -> pd.DataFrame:
    output = frame[frame.action_type.astype(str) == "replace"].copy()
    if output.empty:
        raise RuntimeError("no replacement candidates")
    if bundle["kind"] == "constant":
        probability = np.full(len(output), float(bundle["probability"]), dtype=float)
    else:
        probability = bundle["model"].predict_proba(
            output[list(COMPACT_FEATURES)].to_numpy(float)
        )[:, 1]
    output["pure_hgb_probability"] = probability
    output["student_value"] = probability
    return output


def action_nodes(row) -> set[int]:
    return {int(row.transaction_src_track_id), int(row.transaction_dst_track_id)}


def select_topk_replacements(frame: pd.DataFrame, k: int) -> pd.DataFrame:
    if k <= 0:
        return frame.iloc[:0].copy()
    best_by_pair: Dict[Tuple[int, int], Tuple[float, int]] = {}
    for index, row in frame.iterrows():
        left, right = sorted(action_nodes(row))
        if left == right:
            continue
        candidate = (float(row.pure_hgb_probability), int(index))
        current = best_by_pair.get((left, right))
        if current is None or candidate[0] > current[0]:
            best_by_pair[(left, right)] = candidate
    graph = nx.Graph()
    for (left, right), (weight, index) in best_by_pair.items():
        graph.add_edge(left, right, weight=weight, row_index=index)
    matching = nx.algorithms.matching.max_weight_matching(
        graph, maxcardinality=False, weight="weight"
    )
    indices = [int(graph[left][right]["row_index"]) for left, right in matching]
    if not indices:
        return frame.iloc[:0].copy()
    return frame.loc[indices].nlargest(
        k, ["pure_hgb_probability", "m23_38_parent_policy_score"]
    ).copy()


def write_tracker(
    seq: str,
    predictions: pd.DataFrame,
    k: int,
    graph_root: Path,
    parent: Path,
    output_path: Path,
    m39,
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
    replacements = select_topk_replacements(predictions, k)
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
        "removed_baseline_actions": int(removed),
        "resulting_selected_actions": int(len(selected)),
        "selected_source_indices": replacements.source_index.astype(int).tolist()
        if len(replacements) else [],
        "mean_probability": float(replacements.pure_hgb_probability.mean())
        if len(replacements) else 0.0,
        "tracker_report": tracker_report,
    }


def diverse_held_frame(seq: str, max_replacements: int, m39, label_module, original_within):
    original_residual = label_module.residual_replacements
    original_transform = m39.within_sequence_features

    def diverse(predictions, selected, maximum):
        return original_residual(
            predictions, selected, maximum, selection_mode="diverse"
        )

    def transformed(frame: pd.DataFrame) -> pd.DataFrame:
        return add_channel_features(frame, original_within)

    label_module.residual_replacements = diverse
    m39.within_sequence_features = transformed
    try:
        frame = m39.make_held_action_frame(
            seq, m39.BASELINE_ROOTS[seq], label_module, max_replacements
        )
    finally:
        label_module.residual_replacements = original_residual
        m39.within_sequence_features = original_transform
    forbidden = [
        column for column in frame.columns
        if any(token in column.lower() for token in FORBIDDEN_HELD_TOKENS)
    ]
    if forbidden:
        raise RuntimeError(f"held candidate leakage columns: {forbidden}")
    return frame


def robust_key(row: Mapping[str, object]) -> Tuple[float, float, float, float, int]:
    return (
        float(row["worst_fold_delta_HOTA"]),
        float(row["mean_fold_delta_HOTA"]),
        float(row["HOTA"]),
        float(row["AssA"]),
        -int(row["selected_actions"]),
    )


def metric_delta(metrics: Mapping[str, float], baseline: Mapping[str, float]):
    return {
        "HOTA": float(metrics["HOTA"] - baseline["HOTA"]),
        "DetA": float(metrics["DetA"] - baseline["DetA"]),
        "AssA": float(metrics["AssA"] - baseline["AssA"]),
        "IDSW": int(metrics["IDSW"] - baseline["IDSW"]),
    }


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

    m39 = load_module(
        "m23_49_m39", "scripts/m23_research/m23_39_exact_advantage_student_nested_loso.py"
    )
    eval_helper = load_module(
        "m23_49_eval", "scripts/m23_research/m23_45_domain_robust_source_cut_student.py"
    )
    label_module = load_module(
        "m23_49_labels", "scripts/m23_research/m23_38_exact_hota_advantage_labels.py"
    )
    chain = load_module(
        "m23_49_chain", "scripts/m23_research/m23_12_chain_transaction_oracle.py"
    )
    evaluator = load_module(
        "m23_49_evaluator", "scripts/m23_research/m23_11_eval_utility_graph.py"
    )
    original_within = m39.within_sequence_features

    held = args.held_seq
    training_sequences = [seq for seq in SEQUENCES if seq != held]
    output_root = Path(args.output_root)
    graph_root = Path(args.graph_root) if args.graph_root else m39.DEFAULT_GRAPH_ROOT
    parent = Path(args.parent) if args.parent else m39.DEFAULT_PARENT
    output_root.mkdir(parents=True, exist_ok=True)
    policy_k = list(POLICY_K)
    if args.policy_limit > 0:
        policy_k = policy_k[:args.policy_limit]
    protocol = {
        "experiment": "M23-49 compact pure-HGB top-k transaction student",
        "protocol": "strict nested sequence LOSO",
        "deployable": True,
        "outer_held_sequence": held,
        "outer_training_sequences": training_sequences,
        "training_gt_use": "existing M23-41 exact transaction labels from outer-training sequences only",
        "held_candidate_gt_use": "none",
        "label_regeneration": False,
        "candidate_space": "M23-41 diverse GT-free transaction shortlist",
        "action_policy": "replacement-only maximum-weight matching, top-k",
        "model": "single replacement HistGradientBoostingClassifier, max_leaf_nodes=5",
        "features": list(COMPACT_FEATURES),
        "policies": policy_k,
        "policy_selection": "maximum worst per-inner-sequence HOTA delta; then mean delta and stitched HOTA",
        "held_label_file_read": False,
        "held_gt_read_before_tracker_freeze": False,
        "status": "running",
    }
    (output_root / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n"
    )

    training_frames = {
        seq: load_training_frame(seq, LABEL_PATHS[seq], original_within)
        for seq in training_sequences
    }
    print(json.dumps({
        "stage": "training_labels_ready",
        "frames": {
            seq: {
                "rows": int(len(frame)),
                "replace_rows": int((frame.action_type.astype(str) == "replace").sum()),
                "replace_positive": int(((frame.action_type.astype(str) == "replace") & (frame.delta_HOTA > 0.0)).sum()),
                "replace_negative": int(((frame.action_type.astype(str) == "replace") & (frame.delta_HOTA < 0.0)).sum()),
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
        model = fit_model(train, 49000 + fold_index)
        prediction = predict_replacements(training_frames[pseudo_held], model)
        inner_predictions[pseudo_held] = prediction
        path = output_root / "inner_predictions" / f"{pseudo_held}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        prediction.to_parquet(path, index=False)
        print(json.dumps({
            "stage": "inner_prediction",
            "pseudo_held": pseudo_held,
            "replacement_rows": int(len(prediction)),
        }), flush=True)

    inner_rows: List[Dict[str, object]] = []
    baseline_metrics = None
    for policy_index, k in enumerate(policy_k):
        policy_id = "noop" if k == 0 else f"replace_top{k}"
        candidate_root = output_root / "inner_candidates" / policy_id
        track_root = candidate_root / "track_results"
        reports = []
        selected_actions = 0
        for seq in training_sequences:
            chosen, tracker_report = write_tracker(
                seq, inner_predictions[seq], k, graph_root, parent,
                track_root / f"{seq}.txt", m39, chain, evaluator,
            )
            selected_actions += int(len(chosen))
            reports.append(tracker_report)
        metrics = eval_helper.evaluate_detailed(
            track_root, candidate_root,
            f"m23_49_inner_{held}_{policy_index}", training_sequences,
        )
        if k == 0:
            baseline_metrics = {seq: metrics[seq] for seq in training_sequences}
        if baseline_metrics is None:
            raise RuntimeError("no-op must be evaluated first")
        deltas = {
            seq: float(metrics[seq]["HOTA"] - baseline_metrics[seq]["HOTA"])
            for seq in training_sequences
        }
        row: Dict[str, object] = {
            "policy_id": policy_id,
            "top_k": int(k),
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
    chosen_k = int(chosen_row["top_k"])
    frozen_selection = {
        "selected_policy": {
            "policy_id": str(chosen_row["policy_id"]),
            "top_k": chosen_k,
            "action_type": "replace",
        },
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
    outer_model = fit_model(outer_train, 49999)
    held_frame = diverse_held_frame(
        held, args.max_replacements, m39, label_module, original_within
    )
    held_predictions = predict_replacements(held_frame, outer_model)
    held_predictions.to_parquet(
        output_root / "outer_action_predictions.parquet", index=False
    )
    outer_track_root = output_root / "outer_final" / "track_results"
    selected, tracker_report = write_tracker(
        held, held_predictions, chosen_k, graph_root, parent,
        outer_track_root / f"{held}.txt", m39, chain, evaluator,
    )
    selected.to_parquet(output_root / "outer_selected_replacements.parquet", index=False)
    frozen_tracker = {
        "held_sequence": held,
        "held_label_file_read": False,
        "held_gt_read": False,
        "held_candidates": int(len(held_predictions)),
        "held_candidate_forbidden_columns": [],
        "selected_policy": frozen_selection["selected_policy"],
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
            f"m23_49_outer_{held}", (held,),
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
        "strict_m23_39_baseline": STRICT_M23_39[held],
        "strict_m23_46_baseline": STRICT_M23_46[held],
        "delta_vs_strict_m23_25": metric_delta(evaluation, m39.STRICT_M23_25[held])
        if evaluation else None,
        "delta_vs_strict_m23_39": metric_delta(evaluation, STRICT_M23_39[held])
        if evaluation else None,
        "delta_vs_strict_m23_46": metric_delta(evaluation, STRICT_M23_46[held])
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
