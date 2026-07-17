"""Post-hoc strong-baseline correction for the M23-2 identity attribution audit.

The preregistered audit compared an appearance-margin router with a weak
geometry-IoU router. After predictions were inspected, a construction-aware
candidate-only prior was recognized as a necessary stronger baseline. This
script preserves the original preregistered outputs, explicitly labels the
analysis post-hoc, and compares appearance, geometry, candidate-only, and
suppressor-only decisions on the exact same unambiguous events.

No raw GT is read and TrackEval is never called. The derived oracle labels are
read only from the frozen identity attribution event table.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import binomtest

SEQUENCES = ("MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05")
COMPARISONS = (
    ("appearance", "geometry"),
    ("appearance", "candidate_only"),
    ("appearance", "suppressor_only"),
    ("candidate_only", "geometry"),
)
MARGIN_BINS = (
    (0.0, 0.02, "[0,0.02)"),
    (0.02, 0.05, "[0.02,0.05)"),
    (0.05, 0.10, "[0.05,0.10)"),
    (0.10, 0.20, "[0.10,0.20)"),
    (0.20, float("inf"), "[0.20,inf)"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--loso-dir", required=True)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(repo: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_dump(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fields})


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def wilson_interval(correct: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    p = correct / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    spread = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return (float(max(0.0, center - spread)), float(min(1.0, center + spread)))


def exact_p_values(a_only: int, b_only: int) -> dict:
    discordant = a_only + b_only
    if discordant == 0:
        return {"discordant": 0, "two_sided_p": 1.0, "a_greater_p": 1.0, "b_greater_p": 1.0}
    return {
        "discordant": discordant,
        "two_sided_p": float(binomtest(a_only, discordant, 0.5, alternative="two-sided").pvalue),
        "a_greater_p": float(binomtest(a_only, discordant, 0.5, alternative="greater").pvalue),
        "b_greater_p": float(binomtest(b_only, discordant, 0.5, alternative="greater").pvalue),
    }


def main() -> None:
    args = parse_args()
    repo = Path(args.repo).resolve()
    loso_dir = resolve(repo, args.loso_dir)
    feature_dir = resolve(repo, args.feature_dir)
    output_dir = resolve(repo, args.out_dir)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(output_dir)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loso_report_path = loso_dir / "report.json"
    loso_manifest_path = loso_dir / "manifest.json"
    event_path = loso_dir / "identity_attribution_events.csv"
    original_summary_path = loso_dir / "identity_attribution_summary.csv"
    feature_report_path = feature_dir / "report.json"
    key_path = feature_dir / "candidate_keys.npy"
    feature_path = feature_dir / "appearance_features.f32"
    columns_path = feature_dir / "feature_columns.json"

    loso_report = json.loads(loso_report_path.read_text(encoding="utf-8"))
    loso_manifest = json.loads(loso_manifest_path.read_text(encoding="utf-8"))
    feature_report = json.loads(feature_report_path.read_text(encoding="utf-8"))
    if sha256_file(event_path) != loso_manifest["file_hashes"]["identity_attribution_events.csv"]:
        raise RuntimeError("identity event file differs from frozen LOSO manifest")
    if sha256_file(original_summary_path) != loso_manifest["file_hashes"]["identity_attribution_summary.csv"]:
        raise RuntimeError("identity summary differs from frozen LOSO manifest")
    if sha256_file(key_path) != feature_report["files"]["candidate_keys.npy"]["sha256"]:
        raise RuntimeError("candidate keys differ from feature report")
    if sha256_file(feature_path) != feature_report["files"]["appearance_features.f32"]["sha256"]:
        raise RuntimeError("appearance features differ from feature report")

    feature_columns = json.loads(columns_path.read_text(encoding="utf-8"))
    keys = np.load(key_path, allow_pickle=False, mmap_mode="r")
    features = np.memmap(
        feature_path,
        dtype="<f4",
        mode="r",
        shape=(len(keys), len(feature_columns)),
    )
    feature_index = {name: index for index, name in enumerate(feature_columns)}
    required_features = {
        "candidate_track_margin",
        "candidate_track_count_log1p",
        "suppressor_track_count_log1p",
        "candidate_track_coherence",
        "suppressor_track_coherence",
    }
    missing_features = sorted(required_features - set(feature_index))
    if missing_features:
        raise RuntimeError(f"required support features missing: {missing_features}")

    key_to_row = {
        (int(sequence_id), int(global_pre_idx)): row
        for row, (sequence_id, global_pre_idx) in enumerate(zip(keys["sequence_id"], keys["global_pre_idx"]))
    }
    if len(key_to_row) != len(keys):
        raise RuntimeError("candidate keys are not unique")
    sequence_to_id = {name: index for index, name in enumerate(SEQUENCES)}

    events = []
    seen = set()
    with event_path.open("r", encoding="utf-8", newline="") as handle:
        for source_row in csv.DictReader(handle):
            sequence = source_row["sequence"]
            key = (sequence_to_id[sequence], int(source_row["global_pre_idx"]))
            if key in seen:
                raise RuntimeError(f"duplicate identity event {key}")
            seen.add(key)
            feature_row = key_to_row.get(key)
            if feature_row is None:
                raise RuntimeError(f"identity event outside compact feature bank: {key}")
            oracle_candidate = source_row["oracle_choice"] == "candidate"
            appearance_correct = as_bool(source_row["appearance_correct"])
            geometry_correct = as_bool(source_row["geometry_correct"])
            appearance_margin = float(source_row["appearance_margin"])
            geometry_margin = float(source_row["geometry_margin"])
            recomputed_appearance_choice = appearance_margin > 0.0
            recomputed_geometry_choice = geometry_margin > 0.0
            if appearance_correct != (recomputed_appearance_choice == oracle_candidate):
                raise RuntimeError(f"appearance correctness inconsistent for {key}")
            if geometry_correct != (recomputed_geometry_choice == oracle_candidate):
                raise RuntimeError(f"geometry correctness inconsistent for {key}")
            candidate_only_correct = oracle_candidate
            suppressor_only_correct = not oracle_candidate
            events.append({
                "sequence": sequence,
                "frame": int(source_row["frame"]),
                "global_pre_idx": key[1],
                "oracle_choice": source_row["oracle_choice"],
                "appearance_margin": appearance_margin,
                "geometry_margin": geometry_margin,
                "appearance_correct": appearance_correct,
                "geometry_correct": geometry_correct,
                "candidate_only_correct": candidate_only_correct,
                "suppressor_only_correct": suppressor_only_correct,
                "candidate_track_samples": float(np.expm1(features[feature_row, feature_index["candidate_track_count_log1p"]])),
                "suppressor_track_samples": float(np.expm1(features[feature_row, feature_index["suppressor_track_count_log1p"]])),
                "candidate_track_coherence": float(features[feature_row, feature_index["candidate_track_coherence"]]),
                "suppressor_track_coherence": float(features[feature_row, feature_index["suppressor_track_coherence"]]),
            })
    if len(events) != int(loso_report["counts"]["identity_cases"]):
        raise RuntimeError("identity event count differs from frozen LOSO report")

    model_names = ("appearance", "geometry", "candidate_only", "suppressor_only")
    summary_rows = []
    comparison_rows = []
    support_rows = []
    margin_rows = []
    scopes = list(SEQUENCES) + ["COMBINED"]
    for scope in scopes:
        rows = events if scope == "COMBINED" else [row for row in events if row["sequence"] == scope]
        total = len(rows)
        if total == 0:
            raise RuntimeError(f"identity scope {scope} has zero rows")
        oracle_candidate_count = sum(row["oracle_choice"] == "candidate" for row in rows)
        for model in model_names:
            field = f"{model}_correct"
            correct = sum(bool(row[field]) for row in rows)
            lower, upper = wilson_interval(correct, total)
            summary_rows.append({
                "sequence": scope,
                "model": model,
                "cases": total,
                "correct": correct,
                "accuracy": float(correct / total),
                "wilson95_lower": lower,
                "wilson95_upper": upper,
                "oracle_candidate_cases": oracle_candidate_count,
                "oracle_suppressor_cases": total - oracle_candidate_count,
            })
        for a, b in COMPARISONS:
            a_field = f"{a}_correct"
            b_field = f"{b}_correct"
            both_correct = sum(bool(row[a_field]) and bool(row[b_field]) for row in rows)
            a_only = sum(bool(row[a_field]) and not bool(row[b_field]) for row in rows)
            b_only = sum(not bool(row[a_field]) and bool(row[b_field]) for row in rows)
            both_wrong = total - both_correct - a_only - b_only
            p_values = exact_p_values(a_only, b_only)
            a_accuracy = (both_correct + a_only) / total
            b_accuracy = (both_correct + b_only) / total
            comparison_rows.append({
                "sequence": scope,
                "model_a": a,
                "model_b": b,
                "cases": total,
                "both_correct": both_correct,
                "a_only_correct": a_only,
                "b_only_correct": b_only,
                "both_wrong": both_wrong,
                "accuracy_a": float(a_accuracy),
                "accuracy_b": float(b_accuracy),
                "accuracy_gain_a_minus_b": float(a_accuracy - b_accuracy),
                **p_values,
            })

        support_groups = (
            ("both_prototypes", lambda row: row["candidate_track_samples"] > 0 and row["suppressor_track_samples"] > 0),
            ("candidate_missing", lambda row: row["candidate_track_samples"] <= 0 < row["suppressor_track_samples"]),
            ("suppressor_missing", lambda row: row["suppressor_track_samples"] <= 0 < row["candidate_track_samples"]),
            ("both_missing", lambda row: row["candidate_track_samples"] <= 0 and row["suppressor_track_samples"] <= 0),
        )
        for support, predicate in support_groups:
            selected = [row for row in rows if predicate(row)]
            if not selected:
                continue
            support_rows.append({
                "sequence": scope,
                "support": support,
                "cases": len(selected),
                "appearance_accuracy": float(sum(row["appearance_correct"] for row in selected) / len(selected)),
                "candidate_only_accuracy": float(sum(row["candidate_only_correct"] for row in selected) / len(selected)),
                "geometry_accuracy": float(sum(row["geometry_correct"] for row in selected) / len(selected)),
                "mean_candidate_samples": float(np.mean([row["candidate_track_samples"] for row in selected])),
                "mean_suppressor_samples": float(np.mean([row["suppressor_track_samples"] for row in selected])),
            })
        for lower, upper, label in MARGIN_BINS:
            selected = [row for row in rows if abs(row["appearance_margin"]) >= lower and abs(row["appearance_margin"]) < upper]
            if not selected:
                continue
            margin_rows.append({
                "sequence": scope,
                "absolute_margin_bin": label,
                "cases": len(selected),
                "appearance_accuracy": float(sum(row["appearance_correct"] for row in selected) / len(selected)),
                "candidate_only_accuracy": float(sum(row["candidate_only_correct"] for row in selected) / len(selected)),
                "oracle_candidate_rate": float(sum(row["oracle_choice"] == "candidate" for row in selected) / len(selected)),
            })

    combined = {
        row["model"]: row
        for row in summary_rows
        if row["sequence"] == "COMBINED"
    }
    appearance_accuracy = float(combined["appearance"]["accuracy"])
    geometry_accuracy = float(combined["geometry"]["accuracy"])
    candidate_accuracy = float(combined["candidate_only"]["accuracy"])
    suppressor_accuracy = float(combined["suppressor_only"]["accuracy"])
    best_deterministic_baseline = max(
        ("candidate_only", candidate_accuracy),
        ("suppressor_only", suppressor_accuracy),
        key=lambda pair: pair[1],
    )
    per_sequence_candidate_dominance = all(
        next(row for row in summary_rows if row["sequence"] == sequence and row["model"] == "candidate_only")["accuracy"]
        >= next(row for row in summary_rows if row["sequence"] == sequence and row["model"] == "appearance")["accuracy"]
        for sequence in SEQUENCES
    )
    corrected_decision = {
        "original_preregistered_selector_retained": bool(loso_report["decision"]["appearance_selector_retained"]),
        "appearance_selector_retained": False,
        "appearance_selector_reason": "preregistered sequence-LOSO authenticity ranking failed",
        "original_preregistered_identity_router_retained": bool(loso_report["decision"]["appearance_identity_router_retained"]),
        "appearance_identity_router_retained": False,
        "appearance_identity_router_reason": "appearance margin beats geometry but underperforms the construction-aware candidate-only prior",
        "appearance_accuracy": appearance_accuracy,
        "geometry_accuracy": geometry_accuracy,
        "candidate_only_accuracy": candidate_accuracy,
        "suppressor_only_accuracy": suppressor_accuracy,
        "appearance_gain_vs_geometry": float(appearance_accuracy - geometry_accuracy),
        "appearance_gain_vs_candidate_only": float(appearance_accuracy - candidate_accuracy),
        "best_deterministic_baseline": best_deterministic_baseline[0],
        "best_deterministic_baseline_accuracy": float(best_deterministic_baseline[1]),
        "candidate_only_at_least_appearance_in_every_sequence": bool(per_sequence_candidate_dominance),
        "appearance_feature_bank_retained_as_auxiliary_evidence": True,
        "appearance_feature_bank_role": "auxiliary evidence for calibrated selective override or global graph; not a direct selector/router",
        "deployment_allowed": False,
        "locked_manifest_created": False,
        "next_stage": "counterfactual global tracklet graph with candidate-default edges and learned evidence only for selective overrides",
    }

    write_csv(
        output_dir / "identity_baseline_summary.csv",
        summary_rows,
        ["sequence", "model", "cases", "correct", "accuracy", "wilson95_lower", "wilson95_upper", "oracle_candidate_cases", "oracle_suppressor_cases"],
    )
    write_csv(
        output_dir / "paired_comparisons.csv",
        comparison_rows,
        ["sequence", "model_a", "model_b", "cases", "both_correct", "a_only_correct", "b_only_correct", "both_wrong", "accuracy_a", "accuracy_b", "accuracy_gain_a_minus_b", "discordant", "two_sided_p", "a_greater_p", "b_greater_p"],
    )
    write_csv(
        output_dir / "prototype_support_summary.csv",
        support_rows,
        ["sequence", "support", "cases", "appearance_accuracy", "candidate_only_accuracy", "geometry_accuracy", "mean_candidate_samples", "mean_suppressor_samples"],
    )
    write_csv(
        output_dir / "appearance_margin_bins.csv",
        margin_rows,
        ["sequence", "absolute_margin_bin", "cases", "appearance_accuracy", "candidate_only_accuracy", "oracle_candidate_rate"],
    )

    report = {
        "schema": "fmtrack.m23_2.identity_baseline_correction.v1",
        "protocol": {
            "analysis_status": "post_hoc_strong_baseline_correction",
            "post_hoc_disclosed": True,
            "original_preregistered_outputs_modified": False,
            "direct_ground_truth_read": False,
            "derived_oracle_labels_read": True,
            "trackeval_calls": 0,
            "models": list(model_names),
            "candidate_only_definition": "always assign the suppressed observation to its observable candidate_track_id",
            "suppressor_only_definition": "always assign the suppressed observation to its observable suppressor_track_id",
            "paired_test": "exact McNemar/binomial test on discordant correctness",
        },
        "counts": {
            "identity_cases": len(events),
            "unique_identity_keys": len(seen),
            "oracle_candidate_cases": sum(row["oracle_choice"] == "candidate" for row in events),
            "oracle_suppressor_cases": sum(row["oracle_choice"] == "suppressor" for row in events),
        },
        "combined_summary": combined,
        "corrected_decision": corrected_decision,
        "sources": {
            "original_loso_report_sha256": sha256_file(loso_report_path),
            "original_loso_manifest_sha256": sha256_file(loso_manifest_path),
            "identity_events_sha256": sha256_file(event_path),
            "identity_summary_sha256": sha256_file(original_summary_path),
            "feature_report_sha256": sha256_file(feature_report_path),
            "candidate_keys_sha256": sha256_file(key_path),
            "appearance_features_sha256": sha256_file(feature_path),
            "correction_script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "locked_state": {
            "p15_policy": "no_op",
            "locked_label_reads": 0,
            "locked_trackeval_calls": 0,
            "remaining_locked_rows_untouched": 156,
        },
    }
    canonical_json_dump(report, output_dir / "report.json")
    formal_files = (
        "identity_baseline_summary.csv",
        "paired_comparisons.csv",
        "prototype_support_summary.csv",
        "appearance_margin_bins.csv",
        "report.json",
    )
    manifest = {
        "schema": "fmtrack.m23_2.identity_baseline_correction.manifest.v1",
        "report_sha256": sha256_file(output_dir / "report.json"),
        "file_hashes": {name: sha256_file(output_dir / name) for name in formal_files},
    }
    canonical_json_dump(manifest, output_dir / "manifest.json")
    print(json.dumps(corrected_decision, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
