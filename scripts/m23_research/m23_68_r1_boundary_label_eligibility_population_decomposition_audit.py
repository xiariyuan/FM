"""M23-68-R1 single-predicate repair over immutable M23-68."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE_ROOT = ROOT / "outputs/mot20_m23_20260718"
BASE_SCRIPT = ROOT / "scripts/m23_research/m23_68_boundary_label_eligibility_population_decomposition_audit.py"
SCRIPT = ROOT / "scripts/m23_research/m23_68_r1_boundary_label_eligibility_population_decomposition_audit.py"
TEST_SCRIPT = ROOT / "scripts/m23_research/test_m23_68_r1_boundary_label_eligibility_population_decomposition_audit.py"
PREREG = ROOT / "docs/m23_68_r1_boundary_label_eligibility_population_decomposition_audit_prereg_20260725.md"
RESULT = ROOT / "docs/m23_68_r1_boundary_label_eligibility_population_decomposition_audit_result_20260725.md"
PREDECESSOR = BASE_ROOT / "m23_68_boundary_label_eligibility_population_decomposition_audit"
RUN = BASE_ROOT / "m23_68_r1_boundary_label_eligibility_population_decomposition_audit"
RECONCILER = ROOT / "scripts/m23_research/m23_68_failure_closure_reconcile.py"

SPEC = importlib.util.spec_from_file_location("m23_68_immutable_base", BASE_SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load immutable M23-68 implementation")
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)

original_collect_input_paths = base.collect_input_paths
original_predecessor_r2_check = base.predecessor_r2_check
original_self_test = base.self_test
original_command_init = base.command_init


def join_counts_exact(item: dict[str, Any]) -> bool:
    counts = item["join_counts"]
    both = int(counts.get("both", 0))
    left_only = int(counts.get("left_only", 0))
    right_only = int(counts.get("right_only", 0))
    return bool(
        both == int(item["source_window_count"])
        and both == int(item["node_example_count"])
        and left_only == 0
        and right_only == 0
    )


def self_test() -> dict[str, bool]:
    checks = dict(original_self_test())
    checks.update(
        {
            "categorical_zero_count_categories_pass": join_counts_exact(
                {
                    "join_counts": {"both": 3, "left_only": 0, "right_only": 0},
                    "source_window_count": 3,
                    "node_example_count": 3,
                }
            ),
            "missing_zero_categories_pass": join_counts_exact(
                {
                    "join_counts": {"both": 3},
                    "source_window_count": 3,
                    "node_example_count": 3,
                }
            ),
            "left_only_fails": not join_counts_exact(
                {
                    "join_counts": {"both": 3, "left_only": 1, "right_only": 0},
                    "source_window_count": 4,
                    "node_example_count": 3,
                }
            ),
            "right_only_fails": not join_counts_exact(
                {
                    "join_counts": {"both": 3, "left_only": 0, "right_only": 1},
                    "source_window_count": 3,
                    "node_example_count": 4,
                }
            ),
            "deficient_both_fails": not join_counts_exact(
                {
                    "join_counts": {"both": 2, "left_only": 0, "right_only": 0},
                    "source_window_count": 3,
                    "node_example_count": 3,
                }
            ),
        }
    )
    if not all(checks.values()):
        raise AssertionError({key: value for key, value in checks.items() if not value})
    return checks


def predecessor_m23_68_check() -> dict[str, Any]:
    final = base.read_json(PREDECESSOR / "final_summary.json", {}) or {}
    failure = base.read_json(PREDECESSOR / "implementation_failure.json", {}) or {}
    closure = base.read_json(PREDECESSOR / "closure_validation.json", {}) or {}
    independent = base.read_json(PREDECESSOR / "independent_closure_validation.json", {}) or {}
    reconciliation = base.read_json(PREDECESSOR / "closure_reconciliation.json", {}) or {}
    artifact_validation = base.read_json(PREDECESSOR / "artifact_manifest_validation.json", {}) or {}
    summary = pd.read_csv(PREDECESSOR / "summary.csv", keep_default_na=False)
    _, registry = base.registry_rows()
    matching = [row for row in registry if row.get("name") == "M23-68" or row.get("tag") == "M23-68"]
    artifact_manifest = base.read_json(PREDECESSOR / "artifact_sha256_manifest.json", {}) or {}
    artifact_checks = []
    for record in artifact_manifest.get("records", []):
        path = Path(record["path"])
        path = path if path.is_absolute() else ROOT / path
        actual = base.sha256(path) if path.exists() else None
        artifact_checks.append(actual == record.get("sha256"))
    checks = {
        "failed_closed": final.get("status") == "failed" and final.get("current_stage") == "closed" and final.get("decision") == "FAIL_IMPLEMENTATION",
        "failure_record_closed": failure.get("status") == "closed" and failure.get("same_root_repair_performed") is False,
        "closure_integrity_passed": closure.get("closure_integrity_passed") is True,
        "independent_closure_passed": independent.get("independent_closure_passed") is True,
        "reconciliation_exact_bug": reconciliation.get("implementation_failure_exactly_identified") is True,
        "same_root_scientific_repair_absent": reconciliation.get("same_root_scientific_repair_performed") is False,
        "artifact_validation_passed": artifact_validation.get("all_match") is True,
        "artifact_manifest_all_match": bool(artifact_checks) and all(artifact_checks),
        "summary_no_stale": not summary.status.astype(str).isin(["running", "pending"]).any(),
        "registry_failed_closed": bool(matching and matching[-1].get("status") == "failed" and matching[-1].get("current_stage") == "closed"),
        "hota_null": final.get("hota") is None,
        "next_policy_not_authorized": final.get("next_policy_authorized") is False,
    }
    return {"experiment_id": "M23-68", "checked_at": base.now(), "checks": checks, "all_passed": all(checks.values())}


def combined_predecessor_check() -> dict[str, Any]:
    r2 = original_predecessor_r2_check()
    m23_68 = predecessor_m23_68_check()
    return {
        "checked_at": base.now(),
        "m23_67_r2": r2,
        "m23_68": m23_68,
        "all_passed": bool(r2["all_passed"] and m23_68["all_passed"]),
    }


def collect_input_paths() -> list[Path]:
    paths = list(original_collect_input_paths())
    paths.extend([BASE_SCRIPT, RECONCILER])
    for name in [
        "final_summary.json",
        "implementation_failure.json",
        "implementation_manifest.json",
        "input_manifest.json",
        "input_reverification_final.json",
        "closure_reconciliation.json",
        "closure_validation.json",
        "independent_closure_validation.json",
        "artifact_sha256_manifest.json",
        "artifact_manifest_validation.json",
        "population_identity_validation.json",
        "label_reconstruction_validation.json",
        "summary.csv",
        "protocol_events.jsonl",
    ]:
        paths.append(PREDECESSOR / name)
    artifact_manifest = base.read_json(PREDECESSOR / "artifact_sha256_manifest.json", {}) or {}
    for record in artifact_manifest.get("records", []):
        path = Path(record["path"])
        paths.append(path if path.is_absolute() else ROOT / path)
    input_manifest = base.read_json(PREDECESSOR / "input_manifest.json", {}) or {}
    for item in input_manifest.get("sha256", {}):
        path = Path(item)
        paths.append(path if path.is_absolute() else ROOT / path)
    return sorted(set(paths), key=lambda path: str(path))


def command_population_identity() -> None:
    started, rchar = base.stage_started("audit-population-identity")
    windows = pd.read_parquet(base.R63 / "source_windows.parquet")
    observations = pd.read_parquet(RUN / "reconstructed_boundary_observations.parquet")
    per_split = {}
    for split in ["train", "validation"]:
        metadata = pd.read_parquet(base.R64 / f"node_examples_{split}.parquet").sort_values("tensor_index", kind="mergesort").reset_index(drop=True)
        selected_windows = windows[windows.split == split][["sequence", "window_id", "source_track_key", "row_indices"]].rename(columns={"row_indices": "window_row_indices"})
        joined = selected_windows.merge(
            metadata[["sequence", "window_id", "source_track_key", "source_row_indices", "tensor_index"]],
            on=["sequence", "window_id", "source_track_key"],
            how="outer",
            indicator=True,
            validate="one_to_one",
        )
        rows_equal = [
            base.parse_ids(first) == base.parse_ids(second)
            if isinstance(first, str) and isinstance(second, str)
            else False
            for first, second in zip(joined.window_row_indices, joined.source_row_indices)
        ]
        tensor_indices_exact = sorted(metadata.tensor_index.astype(int).tolist()) == list(range(len(metadata)))
        with np.load(base.R64 / f"examples_{split}.npz", allow_pickle=False) as archive:
            node_x = np.asarray(archive["node_x"], dtype=np.float16)
            node_mask = np.asarray(archive["node_mask"], dtype=np.uint8)
        feature_cache = {
            sequence: np.load(base.R62 / "observables/MOT17" / sequence / "row_features.f16.npy", mmap_mode="r")
            for sequence in (base.TRAIN_SEQUENCES if split == "train" else base.VAL_SEQUENCES)
        }
        exact_tensors = True
        exact_masks = True
        exact_padding = True
        for row in metadata.itertuples(index=False):
            ids = base.parse_ids(row.source_row_indices)
            index = int(row.tensor_index)
            expected = np.asarray(feature_cache[str(row.sequence)][ids], dtype=np.float16)
            exact_tensors = exact_tensors and np.array_equal(node_x[index, : len(ids)], expected)
            expected_mask = np.zeros(node_mask.shape[1], dtype=np.uint8)
            expected_mask[: len(ids)] = 1
            exact_masks = exact_masks and np.array_equal(node_mask[index], expected_mask)
            exact_padding = exact_padding and bool(np.all(node_x[index, len(ids) :] == 0))
        expected_observations = int(sum(len(base.parse_ids(value)) - 1 for value in metadata.source_row_indices))
        actual_observations = int((observations.split == split).sum())
        counts = {str(key): int(value) for key, value in joined._merge.value_counts().items()}
        per_split[split] = {
            "source_window_count": int(len(selected_windows)),
            "node_example_count": int(len(metadata)),
            "join_counts": counts,
            "join_semantics_exact": join_counts_exact(
                {
                    "join_counts": counts,
                    "source_window_count": len(selected_windows),
                    "node_example_count": len(metadata),
                }
            ),
            "row_sequences_exact": bool(all(rows_equal)),
            "tensor_indices_exact": bool(tensor_indices_exact),
            "node_features_exact_r62": bool(exact_tensors),
            "node_masks_exact": bool(exact_masks),
            "padding_zero": bool(exact_padding),
            "expected_boundary_observations": expected_observations,
            "actual_boundary_observations": actual_observations,
            "optimizer_eligible_windows": int((metadata.node_label >= 0).sum()),
            "optimizer_excluded_windows": int((metadata.node_label < 0).sum()),
        }
    checks = {
        "all_windows_join_both": all(item["join_semantics_exact"] for item in per_split.values()),
        "all_window_rows_exact": all(item["row_sequences_exact"] for item in per_split.values()),
        "all_tensor_indices_exact": all(item["tensor_indices_exact"] for item in per_split.values()),
        "all_node_features_exact_r62": all(item["node_features_exact_r62"] for item in per_split.values()),
        "all_masks_exact": all(item["node_masks_exact"] for item in per_split.values()),
        "all_padding_zero": all(item["padding_zero"] for item in per_split.values()),
        "all_observation_counts_exact": all(item["expected_boundary_observations"] == item["actual_boundary_observations"] for item in per_split.values()),
    }
    report = {
        "experiment_id": "M23-68-R1",
        "checked_at": base.now(),
        "splits": per_split,
        "checks": checks,
        "all_passed": all(checks.values()),
        "frozen_example_construction_subset_absent": checks["all_windows_join_both"] and checks["all_window_rows_exact"],
        "optimizer_adapter_filter_reconstructed": True,
        "sole_repair": "explicit both/left_only/right_only semantic counts; no len(mapping) predicate",
    }
    base.write_json(RUN / "population_identity_validation.json", report)
    if not report["all_passed"]:
        raise RuntimeError(f"population identity hard checks failed: {checks}")
    base.append_event("population_identity_audited", checks=checks, sole_repair=True)
    base.stage_finished("audit-population-identity", started, rchar, {"checks": checks})


def command_init() -> None:
    original_command_init()
    implementation_path = RUN / "implementation_manifest.json"
    implementation = base.read_json(implementation_path, {}) or {}
    implementation.update(
        {
            "immutable_base_script_sha256": base.sha256(BASE_SCRIPT),
            "predecessor_m23_68_artifact_manifest_sha256": base.sha256(PREDECESSOR / "artifact_sha256_manifest.json"),
            "sole_repair": "join_counts_exact explicit semantic counts",
        }
    )
    base.write_json(implementation_path, implementation)
    base.write_json(RUN / "predecessor_m23_68_reverification.json", predecessor_m23_68_check())
    base.append_event(
        "predecessor_m23_68_verified",
        predecessor_artifact_manifest_sha256=implementation["predecessor_m23_68_artifact_manifest_sha256"],
    )


base.R68 = RUN
base.SCRIPT = SCRIPT
base.TEST_SCRIPT = TEST_SCRIPT
base.PREREG = PREREG
base.RESULT = RESULT
base.EXP_ID = "M23-68-R1"
base.EXP_NAME = "M23-68-R1 Boundary Label-Eligibility and Population Decomposition Audit Repair"
base.collect_input_paths = collect_input_paths
base.predecessor_r2_check = combined_predecessor_check
base.self_test = self_test
base.command_population_identity = command_population_identity
base.command_init = command_init


def main() -> None:
    base.main()


if __name__ == "__main__":
    main()
