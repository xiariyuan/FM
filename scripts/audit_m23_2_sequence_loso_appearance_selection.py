"""Strict sequence-LOSO audit for M23-2 appearance-conditioned selection.

The candidate pool and deployable features are frozen before this script reads
MOT20 training GT. Two fixed HistGradientBoosting classifiers are compared:
geometry-only and all appearance features. Outer-sequence predictions are
selected at a fixed top-10% rate; no outer threshold or parameter sweep is
performed. A separate identity-attribution audit compares appearance margin
against a geometry-IoU rule on unambiguous oracle-positive events.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

SEQUENCES = ("MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05")
SEQUENCE_TO_ID = {name: index for index, name in enumerate(SEQUENCES)}
TOP_RATE = 0.10
MODEL_PARAMS = {
    "learning_rate": 0.06,
    "max_iter": 160,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 64,
    "l2_regularization": 1.0,
    "early_stopping": False,
    "random_state": 2302,
}
MIN_AP_GAIN = 0.02
MIN_TOP_PRECISION = 0.20
MIN_TOP_PRECISION_GAIN = 0.02
MIN_TOP_RECALL = 0.40
MIN_WORST_SEQUENCE_PRECISION = 0.10
MIN_WORST_SEQUENCE_RECALL = 0.20
MIN_WORST_SEQUENCE_AP_GAIN = 0.0
MIN_WORST_SEQUENCE_PRECISION_GAIN = 0.0
MIN_IDENTITY_CASES = 100
MIN_IDENTITY_ACCURACY_GAIN = 0.05

PREDICTION_DTYPE = np.dtype([
    ("sequence_id", "u1"),
    ("frame", "<i4"),
    ("global_pre_idx", "<i8"),
    ("label", "u1"),
    ("geometry_score", "<f4"),
    ("appearance_score", "<f4"),
    ("geometry_selected", "u1"),
    ("appearance_selected", "u1"),
])


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_dump(obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def resolve(repo: Path, value: str) -> Path:
    p = Path(value)
    return p.resolve() if p.is_absolute() else (repo / p).resolve()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def balanced_sequence_class_weights(sequence_ids: np.ndarray, labels: np.ndarray, train_sequences: Sequence[int]) -> np.ndarray:
    weights = np.zeros((len(labels),), dtype=np.float64)
    sequence_mass = 1.0 / len(train_sequences)
    for sequence_id in train_sequences:
        mask = sequence_ids == sequence_id
        positive = mask & (labels == 1)
        negative = mask & (labels == 0)
        n_positive = int(positive.sum())
        n_negative = int(negative.sum())
        if n_positive == 0 or n_negative == 0:
            raise RuntimeError(f"sequence {sequence_id}: both classes are required")
        weights[positive] = sequence_mass * 0.5 / n_positive
        weights[negative] = sequence_mass * 0.5 / n_negative
    return weights


def binary_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    if len(np.unique(labels)) < 2:
        return {"auc": None, "ap": None}
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "ap": float(average_precision_score(labels, scores)),
    }


def fixed_top_selection(scores: np.ndarray, global_pre_idx: np.ndarray, rate: float) -> np.ndarray:
    count = max(1, int(math.ceil(len(scores) * rate)))
    order = np.lexsort((global_pre_idx, -scores.astype(np.float64)))
    selected = np.zeros((len(scores),), dtype=bool)
    selected[order[:count]] = True
    return selected


def selection_metrics(labels: np.ndarray, selected: np.ndarray) -> dict:
    tp = int(np.sum(selected & (labels == 1)))
    fp = int(np.sum(selected & (labels == 0)))
    fn = int(np.sum((~selected) & (labels == 1)))
    tn = int(np.sum((~selected) & (labels == 0)))
    return {
        "selected": int(selected.sum()),
        "positives_selected": tp,
        "negatives_selected": fp,
        "positives_total": int(np.sum(labels == 1)),
        "negatives_total": int(np.sum(labels == 0)),
        "precision": float(tp / max(1, tp + fp)),
        "recall": float(tp / max(1, tp + fn)),
        "negative_reduction": float(1.0 - fp / max(1, fp + tn)),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", default=".")
    p.add_argument("--feature-dir", required=True)
    p.add_argument("--appearance-manifest-dir", required=True)
    p.add_argument("--oracle-events", required=True)
    p.add_argument("--preregister", required=True)
    p.add_argument("--baseline-dir", default="outputs/trusttrack_review_20260710/all4_baseline_oracle/baseline/eval_work/trackers/all4_baseline/data")
    p.add_argument("--gt-root", default="/gemini/code/datasets/MOT20/train")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(args.repo).resolve()
    feature_dir = resolve(repo, args.feature_dir)
    appearance_dir = resolve(repo, args.appearance_manifest_dir)
    oracle_path = resolve(repo, args.oracle_events)
    preregister_path = resolve(repo, args.preregister)
    baseline_dir = resolve(repo, args.baseline_dir)
    gt_root = resolve(repo, args.gt_root)
    output_dir = resolve(repo, args.out_dir)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(output_dir)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    preregister = json.loads(preregister_path.read_text(encoding="utf-8"))
    if not bool(preregister.get("created_before_predictions")):
        raise RuntimeError("LOSO preregistration was not marked as pre-prediction")
    preregister_sources = preregister["sources"]
    current_script_sha = sha256_file(Path(__file__).resolve())
    if preregister_sources["audit_script_sha256"] != current_script_sha:
        raise RuntimeError("LOSO audit script differs from preregistration")

    feature_report = json.loads((feature_dir / "report.json").read_text(encoding="utf-8"))
    if bool(feature_report["protocol"]["ground_truth_read"]):
        raise RuntimeError("deployable feature bank used GT")
    if not bool(feature_report["decision"]["sequence_loso_audit_ready"]):
        raise RuntimeError("feature bank is not ready for sequence LOSO")
    if int(feature_report["validation"].get("duplicate_candidate_keys", -1)) != 0:
        raise RuntimeError("feature bank candidate keys are not uniquely verified")
    feature_columns = json.loads((feature_dir / "feature_columns.json").read_text(encoding="utf-8"))
    if sha256_file(feature_dir / "appearance_features.f32") != feature_report["files"]["appearance_features.f32"]["sha256"]:
        raise RuntimeError("appearance feature hash differs from feature report")
    if sha256_file(feature_dir / "candidate_keys.npy") != feature_report["files"]["candidate_keys.npy"]["sha256"]:
        raise RuntimeError("candidate key hash differs from feature report")
    preregister_expected = {
        "feature_report_sha256": sha256_file(feature_dir / "report.json"),
        "feature_manifest_sha256": sha256_file(feature_dir / "manifest.json"),
        "candidate_keys_sha256": sha256_file(feature_dir / "candidate_keys.npy"),
        "appearance_features_sha256": sha256_file(feature_dir / "appearance_features.f32"),
        "oracle_events_sha256": sha256_file(oracle_path),
    }
    for name, actual in preregister_expected.items():
        if preregister_sources[name] != actual:
            raise RuntimeError(f"{name} differs from preregistration")
    keys = np.load(feature_dir / "candidate_keys.npy", allow_pickle=False, mmap_mode="r")
    n_rows = len(keys)
    features = np.memmap(
        feature_dir / "appearance_features.f32",
        dtype="<f4",
        mode="r",
        shape=(n_rows, len(feature_columns)),
    )
    candidate_manifest = np.load(appearance_dir / "candidate_manifest.npy", allow_pickle=False, mmap_mode="r")
    if len(candidate_manifest) != n_rows:
        raise RuntimeError("appearance manifest and compact feature rows differ")

    additive_positive_gt: Dict[tuple[int, int], int] = {}
    replace_positive_gt: Dict[tuple[int, int], int] = {}
    with oracle_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["source"] != "prenms_suppressed":
                continue
            if row["variant"] == "prenms_budget_additive_oracle":
                target = additive_positive_gt
            elif row["variant"] == "prenms_budget_replace_add_oracle":
                target = replace_positive_gt
            else:
                continue
            sequence_id = SEQUENCE_TO_ID[row["sequence"]]
            key = (sequence_id, int(row["global_pre_idx"]))
            gt_id = int(row["gt_id"])
            previous = target.get(key)
            if previous is not None and previous != gt_id:
                raise RuntimeError(f"oracle key {key} maps to multiple GT identities")
            target[key] = gt_id

    labels = np.fromiter(
        ((int(sequence_id), int(global_pre_idx)) in additive_positive_gt for sequence_id, global_pre_idx in zip(keys["sequence_id"], keys["global_pre_idx"])),
        dtype=np.uint8,
        count=n_rows,
    )
    key_set = {(int(s), int(g)) for s, g in zip(keys["sequence_id"], keys["global_pre_idx"])}
    if len(key_set) != n_rows:
        raise RuntimeError(f"candidate key table has {n_rows - len(key_set)} duplicates")
    missing_additive_keys = sorted(set(additive_positive_gt) - key_set)
    missing_replace_keys = sorted(set(replace_positive_gt) - key_set)
    if missing_additive_keys:
        raise RuntimeError(f"{len(missing_additive_keys)} additive oracle-positive keys are outside the frozen budget")
    if missing_replace_keys:
        raise RuntimeError(f"{len(missing_replace_keys)} replace/add oracle-positive keys are outside the frozen budget")
    if int(labels.sum()) != len(additive_positive_gt):
        raise RuntimeError("additive oracle-positive label count differs from unique positive keys")
    selected_count = sum(
        max(1, int(math.ceil(int(np.sum(keys["sequence_id"] == sequence_id)) * TOP_RATE)))
        for sequence_id in range(len(SEQUENCES))
    )
    maximum_top_recall = min(1.0, selected_count / max(1, int(labels.sum())))
    if maximum_top_recall < MIN_TOP_RECALL:
        raise RuntimeError(
            f"fixed top-{TOP_RATE:.2f} selector cannot meet recall gate: "
            f"maximum={maximum_top_recall:.6f} required={MIN_TOP_RECALL:.6f}"
        )
    replace_positive_mask = np.fromiter(
        ((int(sequence_id), int(global_pre_idx)) in replace_positive_gt for sequence_id, global_pre_idx in zip(keys["sequence_id"], keys["global_pre_idx"])),
        dtype=bool,
        count=n_rows,
    )

    geometry_columns = [
        name for index, name in enumerate(feature_columns)
        if index < 14 or index >= 28
    ]
    geometry_indices = np.asarray([feature_columns.index(name) for name in geometry_columns], dtype=np.int64)
    appearance_indices = np.arange(len(feature_columns), dtype=np.int64)
    sequence_ids = np.asarray(keys["sequence_id"], dtype=np.int64)
    geometry_scores = np.zeros((n_rows,), dtype=np.float32)
    appearance_scores = np.zeros((n_rows,), dtype=np.float32)
    fold_rows = []

    for outer_id, outer_sequence in enumerate(SEQUENCES):
        print(f"[M23-2 LOSO] outer={outer_sequence}", flush=True)
        test = sequence_ids == outer_id
        train_sequences = [value for value in range(len(SEQUENCES)) if value != outer_id]
        train = ~test
        weights = balanced_sequence_class_weights(sequence_ids, labels, train_sequences)
        for model_name, indices, destination in (
            ("geometry", geometry_indices, geometry_scores),
            ("appearance", appearance_indices, appearance_scores),
        ):
            model = HistGradientBoostingClassifier(**MODEL_PARAMS)
            model.fit(np.asarray(features[train][:, indices], dtype=np.float32), labels[train], sample_weight=weights[train])
            score = model.predict_proba(np.asarray(features[test][:, indices], dtype=np.float32))[:, 1]
            destination[test] = score.astype(np.float32)
            metrics = binary_metrics(labels[test], score)
            fold_rows.append({
                "outer_sequence": outer_sequence,
                "model": model_name,
                "rows": int(test.sum()),
                "positives": int(labels[test].sum()),
                **metrics,
            })

    geometry_selected = np.zeros((n_rows,), dtype=bool)
    appearance_selected = np.zeros((n_rows,), dtype=bool)
    selection_rows = []
    for sequence_id, sequence in enumerate(SEQUENCES):
        mask = np.flatnonzero(sequence_ids == sequence_id)
        for model_name, scores, selected_global in (
            ("geometry", geometry_scores, geometry_selected),
            ("appearance", appearance_scores, appearance_selected),
        ):
            local_selected = fixed_top_selection(scores[mask], keys["global_pre_idx"][mask], TOP_RATE)
            selected_global[mask] = local_selected
            metrics = selection_metrics(labels[mask], local_selected)
            selection_rows.append({"sequence": sequence, "model": model_name, **metrics})

    combined_fold_rows = []
    for model_name, scores in (("geometry", geometry_scores), ("appearance", appearance_scores)):
        metrics = binary_metrics(labels, scores)
        combined_fold_rows.append({
            "outer_sequence": "COMBINED",
            "model": model_name,
            "rows": n_rows,
            "positives": int(labels.sum()),
            **metrics,
        })
    fold_rows.extend(combined_fold_rows)
    for model_name, selected in (("geometry", geometry_selected), ("appearance", appearance_selected)):
        selection_rows.append({"sequence": "COMBINED", "model": model_name, **selection_metrics(labels, selected)})

    # Four fixed temporal blocks per sequence; no block threshold tuning.
    block_rows = []
    for sequence_id, sequence in enumerate(SEQUENCES):
        mask = np.flatnonzero(sequence_ids == sequence_id)
        maximum_frame = int(np.max(keys["frame"][mask]))
        block_id = np.minimum(3, ((keys["frame"][mask].astype(np.int64) - 1) * 4) // max(1, maximum_frame))
        for block in range(4):
            local = mask[block_id == block]
            positives = int(labels[local].sum())
            for model_name, selected in (("geometry", geometry_selected), ("appearance", appearance_selected)):
                block_rows.append({
                    "sequence": sequence,
                    "block": block,
                    "model": model_name,
                    "rows": int(len(local)),
                    "positives": positives,
                    "selected": int(selected[local].sum()),
                    "positive_covered": bool(positives > 0 and np.any(selected[local] & (labels[local] == 1))),
                })

    # Identity attribution on oracle-positive events with different observable tracks.
    m23 = load_module(repo / "scripts/audit_m23_mot20_expanded_evidence_oracle.py", "m23_0_for_m23_2_loso")
    feature_index = {name: index for index, name in enumerate(feature_columns)}
    identity_rows = []
    identity_summary = []
    for sequence_id, sequence in enumerate(SEQUENCES):
        baseline = m23.load_baseline(baseline_dir / f"{sequence}.txt")
        gt = m23.load_gt(gt_root / sequence / "gt" / "gt.txt")
        positive_different = np.flatnonzero(
            (sequence_ids == sequence_id)
            & replace_positive_mask
            & (candidate_manifest["candidate_track_id"] > 0)
            & (candidate_manifest["suppressor_track_id"] > 0)
            & (candidate_manifest["candidate_track_id"] != candidate_manifest["suppressor_track_id"])
        )
        frames = np.unique(keys["frame"][positive_different])
        frame_track_to_gt: Dict[int, Dict[int, int]] = {}
        for frame in frames.tolist():
            baseline_rows = baseline.get(int(frame), [])
            valid_gt = [item for item in gt.get(int(frame), []) if item.marked != 0 and item.cls == 1]
            mapping: Dict[int, int] = {}
            matches = m23.thresholded_hungarian(
                m23.iou_matrix([item.box for item in baseline_rows], [item.box for item in valid_gt]),
                0.5,
            )
            for baseline_index, gt_index, _ in matches:
                mapping[int(baseline_rows[baseline_index].original_id)] = int(valid_gt[gt_index].gt_id)
            frame_track_to_gt[int(frame)] = mapping

        cases = 0
        appearance_correct = 0
        geometry_correct = 0
        for row_index in positive_different.tolist():
            frame = int(keys[row_index]["frame"])
            global_pre_idx = int(keys[row_index]["global_pre_idx"])
            target_gt = replace_positive_gt[(sequence_id, global_pre_idx)]
            candidate_track = int(candidate_manifest[row_index]["candidate_track_id"])
            suppressor_track = int(candidate_manifest[row_index]["suppressor_track_id"])
            mapping = frame_track_to_gt.get(frame, {})
            candidate_is_correct = mapping.get(candidate_track) == target_gt
            suppressor_is_correct = mapping.get(suppressor_track) == target_gt
            if candidate_is_correct == suppressor_is_correct:
                continue
            appearance_margin = float(features[row_index, feature_index["candidate_track_margin"]])
            geometry_margin = float(candidate_manifest[row_index]["candidate_track_iou"] - candidate_manifest[row_index]["suppressor_track_iou"])
            appearance_choice_candidate = appearance_margin > 0.0
            geometry_choice_candidate = geometry_margin > 0.0
            appearance_ok = appearance_choice_candidate == candidate_is_correct
            geometry_ok = geometry_choice_candidate == candidate_is_correct
            cases += 1
            appearance_correct += int(appearance_ok)
            geometry_correct += int(geometry_ok)
            identity_rows.append({
                "sequence": sequence,
                "frame": frame,
                "global_pre_idx": global_pre_idx,
                "target_gt_id": target_gt,
                "candidate_track_id": candidate_track,
                "suppressor_track_id": suppressor_track,
                "oracle_choice": "candidate" if candidate_is_correct else "suppressor",
                "appearance_margin": appearance_margin,
                "geometry_margin": geometry_margin,
                "appearance_correct": appearance_ok,
                "geometry_correct": geometry_ok,
            })
        identity_summary.append({
            "sequence": sequence,
            "cases": cases,
            "appearance_correct": appearance_correct,
            "geometry_correct": geometry_correct,
            "appearance_accuracy": float(appearance_correct / max(1, cases)),
            "geometry_accuracy": float(geometry_correct / max(1, cases)),
            "accuracy_gain": float((appearance_correct - geometry_correct) / max(1, cases)),
        })

    total_identity_cases = sum(row["cases"] for row in identity_summary)
    total_appearance_correct = sum(row["appearance_correct"] for row in identity_summary)
    total_geometry_correct = sum(row["geometry_correct"] for row in identity_summary)
    identity_summary.append({
        "sequence": "COMBINED",
        "cases": total_identity_cases,
        "appearance_correct": total_appearance_correct,
        "geometry_correct": total_geometry_correct,
        "appearance_accuracy": float(total_appearance_correct / max(1, total_identity_cases)),
        "geometry_accuracy": float(total_geometry_correct / max(1, total_identity_cases)),
        "accuracy_gain": float((total_appearance_correct - total_geometry_correct) / max(1, total_identity_cases)),
    })

    prediction = np.empty((n_rows,), dtype=PREDICTION_DTYPE)
    prediction["sequence_id"] = keys["sequence_id"]
    prediction["frame"] = keys["frame"]
    prediction["global_pre_idx"] = keys["global_pre_idx"]
    prediction["label"] = labels
    prediction["geometry_score"] = geometry_scores
    prediction["appearance_score"] = appearance_scores
    prediction["geometry_selected"] = geometry_selected.astype(np.uint8)
    prediction["appearance_selected"] = appearance_selected.astype(np.uint8)
    prediction_path = output_dir / "oof_predictions.npy"
    np.save(prediction_path, prediction, allow_pickle=False)

    write_csv(output_dir / "fold_metrics.csv", fold_rows, ["outer_sequence", "model", "rows", "positives", "auc", "ap"])
    write_csv(
        output_dir / "selection_metrics.csv",
        selection_rows,
        ["sequence", "model", "selected", "positives_selected", "negatives_selected", "positives_total", "negatives_total", "precision", "recall", "negative_reduction"],
    )
    write_csv(output_dir / "block_coverage.csv", block_rows, ["sequence", "block", "model", "rows", "positives", "selected", "positive_covered"])
    write_csv(
        output_dir / "identity_attribution_events.csv",
        identity_rows,
        ["sequence", "frame", "global_pre_idx", "target_gt_id", "candidate_track_id", "suppressor_track_id", "oracle_choice", "appearance_margin", "geometry_margin", "appearance_correct", "geometry_correct"],
    )
    write_csv(
        output_dir / "identity_attribution_summary.csv",
        identity_summary,
        ["sequence", "cases", "appearance_correct", "geometry_correct", "appearance_accuracy", "geometry_accuracy", "accuracy_gain"],
    )

    combined_metrics = {row["model"]: row for row in fold_rows if row["outer_sequence"] == "COMBINED"}
    combined_selection = {row["model"]: row for row in selection_rows if row["sequence"] == "COMBINED"}
    appearance_sequence_precision = [row["precision"] for row in selection_rows if row["model"] == "appearance" and row["sequence"] != "COMBINED"]
    appearance_sequence_recall = [row["recall"] for row in selection_rows if row["model"] == "appearance" and row["sequence"] != "COMBINED"]
    fold_lookup = {(row["outer_sequence"], row["model"]): row for row in fold_rows if row["outer_sequence"] != "COMBINED"}
    selection_lookup = {(row["sequence"], row["model"]): row for row in selection_rows if row["sequence"] != "COMBINED"}
    sequence_ap_gains = {
        sequence: float(fold_lookup[(sequence, "appearance")]["ap"] - fold_lookup[(sequence, "geometry")]["ap"])
        for sequence in SEQUENCES
    }
    sequence_precision_gains = {
        sequence: float(selection_lookup[(sequence, "appearance")]["precision"] - selection_lookup[(sequence, "geometry")]["precision"])
        for sequence in SEQUENCES
    }
    identity_combined = identity_summary[-1]
    decision = {
        "appearance_ap_gain": float(combined_metrics["appearance"]["ap"] - combined_metrics["geometry"]["ap"]),
        "appearance_ap_gain_passed": bool(combined_metrics["appearance"]["ap"] - combined_metrics["geometry"]["ap"] >= MIN_AP_GAIN),
        "worst_sequence_ap_gain": float(min(sequence_ap_gains.values())),
        "worst_sequence_ap_gain_passed": bool(min(sequence_ap_gains.values()) >= MIN_WORST_SEQUENCE_AP_GAIN),
        "appearance_top10_precision": float(combined_selection["appearance"]["precision"]),
        "appearance_top10_precision_passed": bool(combined_selection["appearance"]["precision"] >= MIN_TOP_PRECISION),
        "appearance_top10_precision_gain": float(combined_selection["appearance"]["precision"] - combined_selection["geometry"]["precision"]),
        "appearance_top10_precision_gain_passed": bool(combined_selection["appearance"]["precision"] - combined_selection["geometry"]["precision"] >= MIN_TOP_PRECISION_GAIN),
        "appearance_top10_recall": float(combined_selection["appearance"]["recall"]),
        "appearance_top10_recall_passed": bool(combined_selection["appearance"]["recall"] >= MIN_TOP_RECALL),
        "worst_sequence_top10_precision": float(min(appearance_sequence_precision)),
        "worst_sequence_precision_passed": bool(min(appearance_sequence_precision) >= MIN_WORST_SEQUENCE_PRECISION),
        "worst_sequence_top10_recall": float(min(appearance_sequence_recall)),
        "worst_sequence_recall_passed": bool(min(appearance_sequence_recall) >= MIN_WORST_SEQUENCE_RECALL),
        "worst_sequence_precision_gain": float(min(sequence_precision_gains.values())),
        "worst_sequence_precision_gain_passed": bool(min(sequence_precision_gains.values()) >= MIN_WORST_SEQUENCE_PRECISION_GAIN),
        "sequence_ap_gains": sequence_ap_gains,
        "sequence_precision_gains": sequence_precision_gains,
        "identity_cases": int(identity_combined["cases"]),
        "identity_case_count_passed": bool(identity_combined["cases"] >= MIN_IDENTITY_CASES),
        "identity_accuracy_gain": float(identity_combined["accuracy_gain"]),
        "identity_accuracy_gain_passed": bool(identity_combined["cases"] >= MIN_IDENTITY_CASES and identity_combined["accuracy_gain"] >= MIN_IDENTITY_ACCURACY_GAIN),
    }
    decision["appearance_selector_retained"] = bool(
        decision["appearance_ap_gain_passed"]
        and decision["worst_sequence_ap_gain_passed"]
        and decision["appearance_top10_precision_passed"]
        and decision["appearance_top10_precision_gain_passed"]
        and decision["appearance_top10_recall_passed"]
        and decision["worst_sequence_precision_passed"]
        and decision["worst_sequence_recall_passed"]
        and decision["worst_sequence_precision_gain_passed"]
    )
    decision["appearance_identity_router_retained"] = bool(
        decision["identity_case_count_passed"] and decision["identity_accuracy_gain_passed"]
    )
    decision.update({
        "deployment_allowed": False,
        "locked_manifest_created": False,
        "next_stage": "global tracklet graph integration only for retained appearance components",
    })

    report = {
        "schema": "fmtrack.m23_2.sequence_loso_appearance_selection.v1",
        "protocol": {
            "outer_split": "MOT20 sequence LOSO",
            "models": ["geometry", "appearance"],
            "model_class": "sklearn HistGradientBoostingClassifier",
            "model_parameters": MODEL_PARAMS,
            "sequence_and_class_balanced_training_weight": True,
            "fixed_outer_selection_rate": TOP_RATE,
            "selector_target": "prenms_budget_additive_oracle suppressed candidates that recover baseline-unmatched GT",
            "identity_attribution_target": "prenms_budget_replace_add_oracle suppressed candidates",
            "outer_threshold_sweep": False,
            "parameter_sweep": False,
            "trackeval_calls": 0,
            "locked_label_reads": 0,
            "identity_attribution_scope": "oracle-positive different-track cases with exactly one same-frame baseline track mapped to target GT",
            "preregistered_before_predictions": True,
            "preregister_sha256": sha256_file(preregister_path),
        },
        "counts": {
            "candidates": n_rows,
            "additive_oracle_positive_candidates": int(labels.sum()),
            "additive_oracle_negative_candidates": int(n_rows - labels.sum()),
            "replace_add_oracle_positive_candidates": int(replace_positive_mask.sum()),
            "identity_cases": total_identity_cases,
        },
        "feasibility": {
            "fixed_top_selection_count": selected_count,
            "additive_positive_prevalence": float(labels.mean()),
            "maximum_possible_top_recall": float(maximum_top_recall),
            "recall_gate_mathematically_feasible": bool(maximum_top_recall >= MIN_TOP_RECALL),
        },
        "feature_sets": {
            "geometry": geometry_columns,
            "appearance": feature_columns,
        },
        "acceptance": {
            "minimum_ap_gain": MIN_AP_GAIN,
            "minimum_worst_sequence_ap_gain": MIN_WORST_SEQUENCE_AP_GAIN,
            "minimum_top10_precision": MIN_TOP_PRECISION,
            "minimum_top10_precision_gain": MIN_TOP_PRECISION_GAIN,
            "minimum_top10_recall": MIN_TOP_RECALL,
            "minimum_worst_sequence_precision": MIN_WORST_SEQUENCE_PRECISION,
            "minimum_worst_sequence_recall": MIN_WORST_SEQUENCE_RECALL,
            "minimum_worst_sequence_precision_gain": MIN_WORST_SEQUENCE_PRECISION_GAIN,
            "minimum_identity_cases": MIN_IDENTITY_CASES,
            "minimum_identity_accuracy_gain": MIN_IDENTITY_ACCURACY_GAIN,
        },
        "combined_metrics": combined_metrics,
        "combined_selection": combined_selection,
        "identity_combined": identity_combined,
        "decision": decision,
        "sources": {
            "feature_report_sha256": sha256_file(feature_dir / "report.json"),
            "feature_manifest_sha256": sha256_file(feature_dir / "manifest.json"),
            "appearance_manifest_sha256": sha256_file(appearance_dir / "manifest.json"),
            "oracle_events_sha256": sha256_file(oracle_path),
            "preregister_sha256": sha256_file(preregister_path),
        },
        "locked_state": {
            "p15_policy": "no_op",
            "locked_label_reads": 0,
            "locked_trackeval_calls": 0,
            "remaining_locked_rows_untouched": 156,
        },
    }
    canonical_json_dump(report, output_dir / "report.json")
    compact_files = (
        "oof_predictions.npy", "fold_metrics.csv", "selection_metrics.csv", "block_coverage.csv",
        "identity_attribution_events.csv", "identity_attribution_summary.csv", "report.json",
    )
    manifest = {
        "schema": "fmtrack.m23_2.sequence_loso_appearance_selection.manifest.v1",
        "report_sha256": sha256_file(output_dir / "report.json"),
        "file_hashes": {name: sha256_file(output_dir / name) for name in compact_files},
    }
    canonical_json_dump(manifest, output_dir / "manifest.json")
    print(json.dumps(decision, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
