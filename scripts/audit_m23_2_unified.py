"""Unified fail-closed audit for the complete M23-2 appearance evidence stage."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def compare_files(formal: Path, repro: Path, names: list[str]) -> dict:
    result = {}
    for name in names:
        formal_hash = sha256_file(formal / name)
        repro_hash = sha256_file(repro / name)
        result[name] = {
            "formal_sha256": formal_hash,
            "repro_sha256": repro_hash,
            "byte_identical": formal_hash == repro_hash,
        }
    return result


def all_identical(result: dict) -> bool:
    return all(bool(row["byte_identical"]) for row in result.values())


def main() -> None:
    args = parse_args()
    repo = Path(args.repo).resolve()
    root = repo / "outputs/mot20_m23_20260717"
    appearance = root / "appearance_manifest_v1"
    cache = root / "candidate_reid_sharded_cache_v1"
    shard_plan = root / "candidate_reid_shards_v2/shard_plan.json"
    compact = root / "compact_appearance_features_v1"
    key_repro = root / "compact_appearance_features_v1_keyrepro"
    preregister = root / "sequence_loso_appearance_selection_preregister_v1.json"
    loso = root / "sequence_loso_appearance_selection_v1"
    loso_repro = root / "sequence_loso_appearance_selection_v1_repro"
    correction = root / "identity_baseline_correction_v1"
    correction_repro = root / "identity_baseline_correction_v1_repro"

    appearance_report = load_json(appearance / "report.json")
    cache_report = load_json(cache / "report.json")
    compact_report = load_json(compact / "report.json")
    key_report = load_json(compact / "key_rebuild_report.json")
    preregister_report = load_json(preregister)
    loso_report = load_json(loso / "report.json")
    correction_report = load_json(correction / "report.json")

    key_reproduction = compare_files(
        compact,
        key_repro,
        ["candidate_keys.npy", "key_rebuild_report.json", "report.json", "manifest.json"],
    )
    loso_reproduction = compare_files(
        loso,
        loso_repro,
        [
            "oof_predictions.npy",
            "fold_metrics.csv",
            "selection_metrics.csv",
            "block_coverage.csv",
            "identity_attribution_events.csv",
            "identity_attribution_summary.csv",
            "report.json",
            "manifest.json",
        ],
    )
    correction_reproduction = compare_files(
        correction,
        correction_repro,
        [
            "identity_baseline_summary.csv",
            "paired_comparisons.csv",
            "prototype_support_summary.csv",
            "appearance_margin_bins.csv",
            "report.json",
            "manifest.json",
        ],
    )

    original_decision = loso_report["decision"]
    corrected_decision = correction_report["corrected_decision"]
    checks = {
        "appearance_manifest_ready": bool(appearance_report["decision"]["appearance_manifest_ready"]),
        "appearance_manifest_gt_free": not bool(appearance_report["protocol"]["ground_truth_read"]),
        "appearance_manifest_no_trackeval": int(appearance_report["protocol"]["trackeval_calls"]) == 0,
        "candidate_reid_cache_ready": bool(cache_report["decision"]["candidate_embeddings_ready"]),
        "candidate_reid_cache_gt_free": not bool(cache_report["protocol"]["ground_truth_read"]),
        "candidate_reid_cache_no_trackeval": int(cache_report["protocol"]["trackeval_calls"]) == 0,
        "candidate_reid_crop_count_matches": int(cache_report["counts"]["candidate_crops"]) == 459891,
        "compact_features_ready": bool(compact_report["decision"]["compact_appearance_features_ready"]),
        "compact_features_gt_free": not bool(compact_report["protocol"]["ground_truth_read"]),
        "compact_features_finite": bool(compact_report["validation"]["all_features_finite"]),
        "compact_candidate_count_matches": int(compact_report["counts"]["candidates"]) == 459899,
        "candidate_keys_unique": int(key_report["counts"]["duplicate_candidate_keys"]) == 0,
        "candidate_keys_complete": int(key_report["counts"]["unique_candidate_keys"]) == 459899,
        "feature_hash_unchanged_by_key_repair": bool(key_report["validation"]["feature_sha_unchanged"]),
        "key_reproduction_all_identical": all_identical(key_reproduction),
        "preregister_created_before_predictions": bool(preregister_report["created_before_predictions"]),
        "preregister_bound_to_audit_script": preregister_report["sources"]["audit_script_sha256"] == sha256_file(repo / "scripts/audit_m23_2_sequence_loso_appearance_selection.py"),
        "loso_reproduction_all_identical": all_identical(loso_reproduction),
        "correction_reproduction_all_identical": all_identical(correction_reproduction),
        "preregistered_selector_rejected": not bool(original_decision["appearance_selector_retained"]),
        "corrected_identity_router_rejected": not bool(corrected_decision["appearance_identity_router_retained"]),
        "candidate_only_beats_appearance": float(corrected_decision["candidate_only_accuracy"]) > float(corrected_decision["appearance_accuracy"]),
        "candidate_only_dominates_each_sequence": bool(corrected_decision["candidate_only_at_least_appearance_in_every_sequence"]),
        "final_deployment_false": not bool(corrected_decision["deployment_allowed"]),
        "final_locked_manifest_false": not bool(corrected_decision["locked_manifest_created"]),
        "p15_no_op": correction_report["locked_state"]["p15_policy"] == "no_op",
        "locked_label_reads_zero": int(correction_report["locked_state"]["locked_label_reads"]) == 0,
        "locked_trackeval_calls_zero": int(correction_report["locked_state"]["locked_trackeval_calls"]) == 0,
        "locked_rows_untouched": int(correction_report["locked_state"]["remaining_locked_rows_untouched"]) == 156,
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise RuntimeError(f"unified M23-2 audit failed: {failed}")

    audit = {
        "schema": "fmtrack.m23_2.unified_audit.v1",
        "base_commit": "899040ee2429ee226473811c42f2638be9838a07",
        "checks": checks,
        "counts": {
            "budget_candidates": int(appearance_report["counts"]["budget_candidates"]),
            "unique_candidate_crops": int(appearance_report["counts"]["unique_candidate_crops_added"]),
            "reid_frames": int(cache_report["counts"]["frames_processed"]),
            "reid_shards": int(cache_report["counts"]["shards"]),
            "reid_raw_bytes": int(cache_report["counts"]["raw_bytes"]),
            "compact_features": int(compact_report["counts"]["features"]),
            "additive_oracle_positives": int(loso_report["counts"]["additive_oracle_positive_candidates"]),
            "identity_cases": int(correction_report["counts"]["identity_cases"]),
        },
        "selector_metrics": {
            "geometry_ap": float(loso_report["combined_metrics"]["geometry"]["ap"]),
            "appearance_ap": float(loso_report["combined_metrics"]["appearance"]["ap"]),
            "appearance_ap_gain": float(original_decision["appearance_ap_gain"]),
            "geometry_top10_precision": float(loso_report["combined_selection"]["geometry"]["precision"]),
            "appearance_top10_precision": float(loso_report["combined_selection"]["appearance"]["precision"]),
            "geometry_top10_recall": float(loso_report["combined_selection"]["geometry"]["recall"]),
            "appearance_top10_recall": float(loso_report["combined_selection"]["appearance"]["recall"]),
            "worst_sequence_ap_gain": float(original_decision["worst_sequence_ap_gain"]),
            "worst_sequence_precision_gain": float(original_decision["worst_sequence_precision_gain"]),
        },
        "identity_metrics": {
            "appearance_accuracy": float(corrected_decision["appearance_accuracy"]),
            "geometry_accuracy": float(corrected_decision["geometry_accuracy"]),
            "candidate_only_accuracy": float(corrected_decision["candidate_only_accuracy"]),
            "suppressor_only_accuracy": float(corrected_decision["suppressor_only_accuracy"]),
            "appearance_gain_vs_geometry": float(corrected_decision["appearance_gain_vs_geometry"]),
            "appearance_gain_vs_candidate_only": float(corrected_decision["appearance_gain_vs_candidate_only"]),
        },
        "scientific_correction": {
            "original_preregistered_identity_router_retained": bool(corrected_decision["original_preregistered_identity_router_retained"]),
            "authoritative_corrected_identity_router_retained": bool(corrected_decision["appearance_identity_router_retained"]),
            "reason": corrected_decision["appearance_identity_router_reason"],
            "post_hoc_disclosed": bool(correction_report["protocol"]["post_hoc_disclosed"]),
            "original_outputs_modified": bool(correction_report["protocol"]["original_preregistered_outputs_modified"]),
        },
        "final_decision": {
            "appearance_authenticity_selector_retained": False,
            "appearance_identity_router_retained": False,
            "appearance_feature_bank_retained_as_auxiliary_evidence": True,
            "deployment_allowed": False,
            "locked_manifest_created": False,
            "next_stage": corrected_decision["next_stage"],
        },
        "reproduction": {
            "candidate_key_repair": key_reproduction,
            "sequence_loso": loso_reproduction,
            "identity_baseline_correction": correction_reproduction,
        },
        "artifacts": {
            "appearance_manifest": {
                "report_sha256": sha256_file(appearance / "report.json"),
                "manifest_sha256": sha256_file(appearance / "manifest.json"),
            },
            "candidate_reid_cache": {
                "report_sha256": sha256_file(cache / "report.json"),
                "manifest_sha256": sha256_file(cache / "manifest.json"),
                "shard_plan_sha256": sha256_file(shard_plan),
                "content_merkle_sha256": cache_report["validation"]["sharded_content_merkle_sha256"],
            },
            "compact_features": {
                "report_sha256": sha256_file(compact / "report.json"),
                "manifest_sha256": sha256_file(compact / "manifest.json"),
                "feature_matrix_sha256": compact_report["files"]["appearance_features.f32"]["sha256"],
                "candidate_keys_sha256": compact_report["files"]["candidate_keys.npy"]["sha256"],
            },
            "preregister_sha256": sha256_file(preregister),
            "loso_report_sha256": sha256_file(loso / "report.json"),
            "correction_report_sha256": sha256_file(correction / "report.json"),
        },
        "git_policy": {
            "large_raw_embeddings_committed": False,
            "large_feature_matrix_committed": False,
            "formal_manifests_reports_metrics_and_content_hashes_committed": True,
            "reproduction_directories_committed": False,
        },
        "locked_state": correction_report["locked_state"],
    }
    output_path = (repo / args.out).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"checks": len(checks), "all_passed": True, "output_sha256": sha256_file(output_path)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
