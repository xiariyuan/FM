#!/usr/bin/env python3
from __future__ import annotations

"""M23-46 strict source-cut student: pure low-capacity HGB + robust top-k.

This removes M23-45's probability ensemble and magnitude heads.  Each inner
model is a five-leaf HistGradientBoostingClassifier trained on two
outer-training sequences.  Candidate policies are only no-op and top-k for
k=1..4.  Policy selection maximizes the worst per-inner-sequence exact HOTA
delta, then mean delta and stitched HOTA.  Outer-held labels are never opened
before the held tracker is frozen.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


REPO = Path(__file__).resolve().parents[2]
SEQUENCES = ("MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05")


def load_module(name: str, relative_path: str):
    path = REPO / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def fit_model(frame: pd.DataFrame, features: Sequence[str], seed: int):
    model = HistGradientBoostingClassifier(
        learning_rate=0.04,
        max_iter=180,
        max_leaf_nodes=5,
        min_samples_leaf=8,
        l2_regularization=12.0,
        early_stopping=False,
        random_state=seed,
    )
    model.fit(frame[list(features)], (frame.delta_HOTA > 0.0).astype(int))
    return model


def predict(frame: pd.DataFrame, model, features: Sequence[str]) -> pd.DataFrame:
    output = frame.copy()
    output["pure_hgb_probability"] = model.predict_proba(output[list(features)])[:, 1]
    return output


def select_topk(frame: pd.DataFrame, k: int) -> pd.DataFrame:
    if k <= 0:
        return frame.iloc[:0].copy()
    return frame.nlargest(k, ["pure_hgb_probability", "source_cut_policy_score"])


def write_tracker(
    seq: str,
    predictions: pd.DataFrame,
    k: int,
    baseline_cache: Path,
    graph_root: Path,
    parent: Path,
    output_path: Path,
    evaluator,
):
    applied = pd.read_parquet(baseline_cache / seq / "frozen_applied_edges.parquet")
    meta = pd.read_parquet(graph_root / seq / "microtracklets.parquet")
    chosen = select_topk(predictions, k)
    source_indices = chosen.source_index.astype(int).tolist() if len(chosen) else []
    if len(source_indices) != len(set(source_indices)):
        raise RuntimeError("duplicate source-cut indices")
    modified = applied.drop(index=source_indices).copy()
    evaluator.DATA = graph_root
    evaluator.PARENT = parent
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = evaluator.write_tracker(seq, meta, modified, output_path)
    return chosen, {
        "seq": seq,
        "selected_cuts": int(len(chosen)),
        "selected_source_indices": source_indices,
        "remaining_applied_edges": int(len(modified)),
        "mean_probability": float(chosen.pure_hgb_probability.mean()) if len(chosen) else 0.0,
        "tracker_report": report,
    }


def robust_key(row: Mapping[str, object]) -> Tuple[float, float, float, float, int]:
    return (
        float(row["worst_fold_delta_HOTA"]),
        float(row["mean_fold_delta_HOTA"]),
        float(row["HOTA"]),
        float(row["AssA"]),
        -int(row["selected_actions"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--held-seq", required=True, choices=SEQUENCES)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--baseline-cache", default=None)
    parser.add_argument("--graph-root", default=None)
    parser.add_argument("--parent", default=None)
    parser.add_argument("--skip-outer-trackeval", action="store_true")
    args = parser.parse_args()

    base = load_module("m23_46_base", "scripts/m23_research/m23_45_domain_robust_source_cut_student.py")
    candidates = load_module("m23_46_candidates", "scripts/m23_research/m23_43_source_boundary_cut_teacher.py")
    evaluator = load_module("m23_46_evaluator", "scripts/m23_research/m23_11_eval_utility_graph.py")

    held = args.held_seq
    training_sequences = [seq for seq in SEQUENCES if seq != held]
    output_root = Path(args.output_root)
    baseline_cache = Path(args.baseline_cache) if args.baseline_cache else base.DEFAULT_BASELINE_CACHE
    graph_root = Path(args.graph_root) if args.graph_root else base.DEFAULT_GRAPH_ROOT
    parent = Path(args.parent) if args.parent else base.DEFAULT_PARENT
    output_root.mkdir(parents=True, exist_ok=True)

    protocol = {
        "experiment": "M23-46 pure-HGB top-k source-cut student",
        "protocol": "strict nested sequence LOSO",
        "deployable": True,
        "outer_held_sequence": held,
        "outer_training_sequences": training_sequences,
        "training_gt_use": "exact source-cut labels from outer-training sequences only",
        "held_candidate_gt_use": "none",
        "model": "single HistGradientBoostingClassifier, max_leaf_nodes=5",
        "policies": [0, 1, 2, 3, 4],
        "policy_selection": "maximum worst-fold exact HOTA delta; then mean delta and stitched HOTA",
        "held_label_file_read": False,
        "held_gt_read_before_tracker_freeze": False,
        "status": "running",
    }
    (output_root / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")

    training_frames = {
        seq: base.load_training_frame(seq, base.LABEL_PATHS[seq])
        for seq in training_sequences
    }
    print(json.dumps({
        "stage": "training_labels_ready",
        "frames": {
            seq: {
                "rows": int(len(frame)),
                "positive": int((frame.delta_HOTA > 0.0).sum()),
                "negative": int((frame.delta_HOTA < 0.0).sum()),
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
        model = fit_model(train, base.FEATURES, 46000 + fold_index)
        predicted = predict(training_frames[pseudo_held], model, base.FEATURES)
        inner_predictions[pseudo_held] = predicted
        path = output_root / "inner_predictions" / f"{pseudo_held}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        predicted.to_parquet(path, index=False)
        print(json.dumps({"stage": "inner_prediction", "pseudo_held": pseudo_held, "rows": len(predicted)}), flush=True)

    rows: List[Dict[str, object]] = []
    baseline_metrics = None
    for policy_index, k in enumerate((0, 1, 2, 3, 4)):
        policy_name = "noop" if k == 0 else f"top{k}"
        candidate_root = output_root / "inner_candidates" / policy_name
        track_root = candidate_root / "track_results"
        selected_actions = 0
        reports = []
        for seq in training_sequences:
            chosen, report = write_tracker(
                seq, inner_predictions[seq], k, baseline_cache, graph_root, parent,
                track_root / f"{seq}.txt", evaluator,
            )
            selected_actions += len(chosen)
            reports.append(report)
        metrics = base.evaluate_detailed(
            track_root, candidate_root, f"m23_46_inner_{held}_{policy_index}", training_sequences,
        )
        if k == 0:
            baseline_metrics = {seq: metrics[seq] for seq in training_sequences}
        if baseline_metrics is None:
            raise RuntimeError("no-op must be first")
        deltas = {
            seq: metrics[seq]["HOTA"] - baseline_metrics[seq]["HOTA"]
            for seq in training_sequences
        }
        row: Dict[str, object] = {
            "policy_id": policy_name,
            "top_k": k,
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
        rows.append(row)
        (candidate_root / "tracker_reports.json").write_text(json.dumps(reports, indent=2, sort_keys=True) + "\n")
        base.write_csv(output_root / "inner_metrics.csv", rows)
        print(json.dumps({"stage": "inner_trackeval", **row}), flush=True)

    chosen_row = max(rows, key=robust_key)
    chosen_k = int(chosen_row["top_k"])
    frozen_selection = {
        "selected_policy": {"policy_id": chosen_row["policy_id"], "top_k": chosen_k},
        "selected_inner_metrics": chosen_row,
        "selection_rule": "maximum worst-fold exact HOTA delta; then mean delta, stitched HOTA, AssA, fewer actions",
        "held_label_file_read": False,
        "outer_held_gt_read": False,
    }
    (output_root / "frozen_inner_selection.json").write_text(json.dumps(frozen_selection, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"stage": "policy_frozen", **chosen_row}), flush=True)

    outer_train = pd.concat(list(training_frames.values()), ignore_index=True, sort=False)
    outer_model = fit_model(outer_train, base.FEATURES, 46999)
    held_frame = base.held_candidate_frame(held, baseline_cache, graph_root, candidates)
    held_predictions = predict(held_frame, outer_model, base.FEATURES)
    held_predictions.to_parquet(output_root / "outer_cut_predictions.parquet", index=False)

    outer_track_root = output_root / "outer_final" / "track_results"
    selected, tracker_report = write_tracker(
        held, held_predictions, chosen_k, baseline_cache, graph_root, parent,
        outer_track_root / f"{held}.txt", evaluator,
    )
    selected.to_parquet(output_root / "outer_selected_cuts.parquet", index=False)
    frozen_tracker = {
        "held_sequence": held,
        "held_label_file_read": False,
        "held_gt_read": False,
        "held_candidates": int(len(held_predictions)),
        "selected_policy": frozen_selection["selected_policy"],
        "selected_cuts": int(len(selected)),
        "selected_source_indices": selected.source_index.astype(int).tolist() if len(selected) else [],
        "tracker_report": tracker_report,
        "tracker_path": str(outer_track_root / f"{held}.txt"),
    }
    (output_root / "outer_tracker_frozen.json").write_text(json.dumps(frozen_tracker, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"stage": "outer_tracker_frozen", **frozen_tracker}), flush=True)

    evaluation = None
    if not args.skip_outer_trackeval:
        evaluation = base.evaluate_detailed(
            outer_track_root, output_root / "outer_eval", f"m23_46_outer_{held}", (held,),
        )["COMBINED"]
        print(json.dumps({"stage": "outer_trackeval", **evaluation}), flush=True)

    report = {
        **protocol,
        "status": "completed" if evaluation is not None else "tracker_frozen",
        "inner_candidates": rows,
        "frozen_inner_selection": frozen_selection,
        "held_candidates": int(len(held_predictions)),
        "held_selected_cuts": int(len(selected)),
        "tracker_report": tracker_report,
        "eval": evaluation,
    }
    (output_root / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    protocol["status"] = report["status"]
    (output_root / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
