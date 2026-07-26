#!/usr/bin/env python3
"""Reconcile immutable failed M23-69-R1 after numerical reproduction hard-fail."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "outputs/mot20_m23_20260718/m23_69_r1_boundary_objective_gate_population_alignment_audit"
SCRIPT = ROOT / "scripts/m23_research/m23_69_r1_boundary_objective_gate_population_alignment_audit.py"
TEST = ROOT / "scripts/m23_research/test_m23_69_r1_boundary_objective_gate_population_alignment_audit.py"
PREREG = ROOT / "docs/m23_69_r1_boundary_objective_gate_population_alignment_audit_prereg_20260725.md"
RESULT = ROOT / "docs/m23_69_r1_boundary_objective_gate_population_alignment_audit_result_20260725.md"
REGISTRY = ROOT / "outputs/experiment_registry.csv"
PROCESS_HELPER = ROOT / "scripts/m23_research/m23_67_r2_source_boundary_failure_root_cause_audit.py"
RECONCILER = ROOT / "scripts/m23_research/m23_69_r1_failure_closure_reconcile.py"
EXP_ID = "M23-69-R1"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def process_snapshot() -> dict[str, Any]:
    specification = importlib.util.spec_from_file_location("m23_69_r1_reconcile_process", PROCESS_HELPER)
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot load process helper")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module.process_gpu_snapshot(exclude_self=True)


def process_idle(snapshot: dict[str, Any]) -> bool:
    gpu = snapshot.get("gpu", {})
    return bool(
        not snapshot.get("blocking_processes")
        and not snapshot.get("relevant_processes")
        and not gpu.get("compute_processes")
        and gpu.get("memory_used_mib") in (None, 0)
    )


def verify_inputs() -> tuple[list[dict[str, Any]], bool]:
    manifest = read_json(RUN / "input_manifest.json")
    records = []
    for candidate, expected in sorted(manifest["sha256"].items()):
        path = ROOT / candidate
        actual = sha256(path) if path.is_file() else None
        records.append(
            {
                "path": candidate,
                "expected": expected,
                "actual": actual,
                "match": actual == expected,
            }
        )
    return records, bool(records) and all(record["match"] for record in records)


def registry_check() -> tuple[int | None, bool]:
    with REGISTRY.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    matching = [
        (index + 2, row)
        for index, row in enumerate(rows)
        if row.get("name") == EXP_ID or row.get("tag") == EXP_ID
    ]
    passed = bool(
        len(matching) == 1
        and matching[0][1].get("status") == "failed"
        and matching[0][1].get("current_stage") == "closed"
        and matching[0][1].get("decision") == "FAIL_IMPLEMENTATION"
    )
    return (matching[0][0] if len(matching) == 1 else None), passed


def main() -> None:
    if (RUN / "closure_reconciliation.json").exists():
        raise RuntimeError("M23-69-R1 failure closure already reconciled")
    failure = read_json(RUN / "implementation_failure.json")
    final = read_json(RUN / "final_summary.json")
    objective = read_json(RUN / "objective_population.json")
    implementation = read_json(RUN / "implementation_manifest.json")
    summary = pd.read_csv(RUN / "summary.csv", keep_default_na=False)
    inputs, inputs_unchanged = verify_inputs()
    registry_line, registry_passed = registry_check()
    reproduction = objective["selected_checkpoint_performance"]["historical_conditional_reproduction"]
    reconstructed = next(
        row
        for row in objective["selected_checkpoint_performance"]["records"]
        if row["split"] == "validation"
        and row["scope"] == "pooled"
        and row["definition"] == "native"
        and row["population"] == "conditional"
        and row["view"] == "raw_observation"
    )
    historical = read_json(
        ROOT
        / "outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_pair_reconstruction_training_repair1/frozen_checkpoint/checkpoint_manifest.json"
    )["validation_metrics"]["conditional_boundary"]
    pr_auc_difference = abs(float(reconstructed["pr_auc"]) - float(historical["pr_auc"]))
    non_pr_checks = {key: value for key, value in reproduction.items() if key != "pr_auc"}
    objective_checks = objective["checks"]
    only_failed_objective_check = {
        key: value for key, value in objective_checks.items() if not value
    } == {"historical_conditional_metrics_reproduced": False}
    exact_failure = bool(
        failure.get("decision") == "FAIL_IMPLEMENTATION"
        and failure.get("same_root_repair_performed") is False
        and only_failed_objective_check
        and all(non_pr_checks.values())
        and reproduction["pr_auc"] is False
        and pr_auc_difference < 1e-4
        and not (RUN / "checkpoint_inventory.json").exists()
        and not (RUN / "retained_checkpoint_evaluation.json").exists()
        and not (RUN / "gate_score_semantics.json").exists()
    )
    snapshot = process_snapshot()
    implementation_checks = {
        "script": sha256(SCRIPT) == implementation.get("script_sha256"),
        "test": sha256(TEST) == implementation.get("test_script_sha256"),
        "prereg": sha256(PREREG) == implementation.get("prereg_sha256"),
    }
    forbidden = final.get("forbidden_scope_counts", {})
    checks = {
        "exact_failure_identified": exact_failure,
        "inputs_unchanged": inputs_unchanged,
        "input_record_count_295": len(inputs) == 295,
        "implementation_sha_guard": all(implementation_checks.values()),
        "summary_no_running_pending": not summary.status.astype(str).isin(["running", "pending"]).any(),
        "registry_failed_closed": registry_passed,
        "objective_population_partial_only": (RUN / "objective_population.json").exists()
        and not (RUN / "checkpoint_inventory.json").exists(),
        "checkpoint_loads_zero": not (RUN / "retained_checkpoint_validation_scores.parquet").exists(),
        "forbidden_scope_all_zero": bool(forbidden) and all(value == 0 for value in forbidden.values()),
        "process_gpu_idle": process_idle(snapshot),
        "hota_null": final.get("hota") is None,
        "next_policy_not_authorized": final.get("next_policy_authorized") is False,
        "m23_70_not_started": final.get("m23_70_started") is False,
        "same_root_repair_absent": failure.get("same_root_repair_performed") is False,
    }
    reconciliation = {
        "experiment_id": EXP_ID,
        "reconciled_at": now(),
        "decision": "FAIL_IMPLEMENTATION",
        "registry_line": registry_line,
        "implementation_failure_exactly_identified": exact_failure,
        "failure_cause": (
            "historical conditional PR-AUC was required within 1e-12, while the frozen R66/R68 selected-score "
            "inference path differed from the M64 training-time metric by less than 1e-4"
        ),
        "historical_pr_auc": historical["pr_auc"],
        "reconstructed_pr_auc": reconstructed["pr_auc"],
        "absolute_pr_auc_difference": pr_auc_difference,
        "rows_exact": reproduction["rows"],
        "positives_exact": reproduction["positives"],
        "precision_at_actual_exact": reproduction["precision_at_actual"],
        "recall_at_95_precision_exact": reproduction["recall_at_95_precision"],
        "partial_objective_artifacts_are_not_final_scientific_results": True,
        "checkpoint_inference_performed": False,
        "scientific_rerun_performed": False,
        "same_root_scientific_repair_performed": False,
        "original_script_modified": False,
        "checks": checks,
        "all_passed": all(checks.values()),
    }
    write_json(RUN / "closure_reconciliation.json", reconciliation)
    write_json(
        RUN / "input_reverification_final.json",
        {"checked_at": now(), "records": inputs, "inputs_unchanged": inputs_unchanged},
    )
    with (RUN / "protocol_events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "timestamp": now(),
                    "experiment_id": EXP_ID,
                    "event": "failure_closure_reconciled",
                    "registry_line": registry_line,
                    "scientific_rerun_performed": False,
                    "absolute_pr_auc_difference": pr_auc_difference,
                },
                sort_keys=True,
            )
            + "\n"
        )
    closure = {
        "experiment_id": EXP_ID,
        "validated_at": now(),
        "status": "closed",
        "decision": "FAIL_IMPLEMENTATION",
        **checks,
    }
    closure["closure_integrity_passed"] = all(checks.values())
    write_json(RUN / "closure_validation.json", closure)
    independent = {
        "experiment_id": EXP_ID,
        "validated_at": now(),
        "failed_closed": closure["status"] == "closed" and closure["decision"] == "FAIL_IMPLEMENTATION",
        **{key: value for key, value in checks.items()},
        "closure_integrity_passed": closure["closure_integrity_passed"],
    }
    independent["independent_closure_passed"] = all(
        value for key, value in independent.items() if key not in {"experiment_id", "validated_at"}
    )
    write_json(RUN / "independent_closure_validation.json", independent)
    excluded = {"artifact_sha256_manifest.json", "artifact_manifest_validation.json"}
    paths = [SCRIPT, TEST, PREREG, RESULT, RECONCILER]
    paths.extend(path for path in RUN.iterdir() if path.is_file() and path.name not in excluded)
    records = [
        {"path": relative(path), "sha256": sha256(path), "size": path.stat().st_size}
        for path in sorted(set(paths), key=lambda item: str(item))
    ]
    write_json(
        RUN / "artifact_sha256_manifest.json",
        {"experiment_id": EXP_ID, "created_at": now(), "records": records},
    )
    artifact_checks = []
    for record in records:
        path = ROOT / record["path"]
        actual = sha256(path) if path.is_file() else None
        artifact_checks.append({**record, "actual": actual, "match": actual == record["sha256"]})
    write_json(
        RUN / "artifact_manifest_validation.json",
        {
            "experiment_id": EXP_ID,
            "checked_at": now(),
            "records": artifact_checks,
            "all_match": bool(artifact_checks) and all(record["match"] for record in artifact_checks),
        },
    )
    if not reconciliation["all_passed"]:
        raise RuntimeError(f"reconciliation checks failed: {checks}")
    if not independent["independent_closure_passed"]:
        raise RuntimeError("independent closure failed")
    if not all(record["match"] for record in artifact_checks):
        raise RuntimeError("artifact validation failed")
    print(
        json.dumps(
            {
                "status": "closed",
                "decision": "FAIL_IMPLEMENTATION",
                "absolute_pr_auc_difference": pr_auc_difference,
            }
        )
    )


if __name__ == "__main__":
    main()
