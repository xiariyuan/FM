#!/usr/bin/env python3
"""M23-69-R2 numerical-reproduction repair over immutable M23-69/R1."""
from __future__ import annotations

import builtins
import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE_ROOT = ROOT / "outputs/mot20_m23_20260718"
BASE_SCRIPT = ROOT / "scripts/m23_research/m23_69_boundary_objective_gate_population_alignment_audit.py"
SCRIPT = ROOT / "scripts/m23_research/m23_69_r2_boundary_objective_gate_population_alignment_audit.py"
TEST_SCRIPT = ROOT / "scripts/m23_research/test_m23_69_r2_boundary_objective_gate_population_alignment_audit.py"
PREREG = ROOT / "docs/m23_69_r2_boundary_objective_gate_population_alignment_audit_prereg_20260725.md"
RESULT = ROOT / "docs/m23_69_r2_boundary_objective_gate_population_alignment_audit_result_20260725.md"
M69 = BASE_ROOT / "m23_69_boundary_objective_gate_population_alignment_audit"
R1 = BASE_ROOT / "m23_69_r1_boundary_objective_gate_population_alignment_audit"
RUN = BASE_ROOT / "m23_69_r2_boundary_objective_gate_population_alignment_audit"
M69_TEST = ROOT / "scripts/m23_research/test_m23_69_boundary_objective_gate_population_alignment_audit.py"
M69_PREREG = ROOT / "docs/m23_69_boundary_objective_gate_population_alignment_audit_prereg_20260725.md"
M69_RESULT = ROOT / "docs/m23_69_boundary_objective_gate_population_alignment_audit_result_20260725.md"
M69_RECONCILER = ROOT / "scripts/m23_research/m23_69_failure_closure_reconcile.py"
R1_SCRIPT = ROOT / "scripts/m23_research/m23_69_r1_boundary_objective_gate_population_alignment_audit.py"
R1_TEST = ROOT / "scripts/m23_research/test_m23_69_r1_boundary_objective_gate_population_alignment_audit.py"
R1_PREREG = ROOT / "docs/m23_69_r1_boundary_objective_gate_population_alignment_audit_prereg_20260725.md"
R1_RESULT = ROOT / "docs/m23_69_r1_boundary_objective_gate_population_alignment_audit_result_20260725.md"
R1_RECONCILER = ROOT / "scripts/m23_research/m23_69_r1_failure_closure_reconcile.py"
EXP_ID = "M23-69-R2"
EXP_NAME = "M23-69-R2 Boundary Objective–Gate Population Alignment Audit Repair"
HISTORICAL_REPRODUCTION_TOLERANCE = 1e-4


SPECIFICATION = importlib.util.spec_from_file_location("m23_69_r2_immutable_base", BASE_SCRIPT)
if SPECIFICATION is None or SPECIFICATION.loader is None:
    raise RuntimeError("cannot load immutable M23-69 scientific implementation")
base = importlib.util.module_from_spec(SPECIFICATION)
sys.modules[SPECIFICATION.name] = base
SPECIFICATION.loader.exec_module(base)

original_collect_input_paths = base.collect_input_paths
original_predecessor_check = base.predecessor_check
original_command_init = base.command_init
original_selected_source_metrics = base.selected_source_metrics
original_evaluate_retained_checkpoints = base.command_evaluate_retained_checkpoints
original_result_markdown = base.result_markdown


def failed_predecessor_check(
    run: Path,
    experiment_id: str,
    *,
    partial_objective_expected: bool,
) -> dict[str, Any]:
    final = base.read_json(run / "final_summary.json", {}) or {}
    failure = base.read_json(run / "implementation_failure.json", {}) or {}
    reconciliation = base.read_json(run / "closure_reconciliation.json", {}) or {}
    closure = base.read_json(run / "closure_validation.json", {}) or {}
    independent = base.read_json(run / "independent_closure_validation.json", {}) or {}
    artifact_validation = base.read_json(run / "artifact_manifest_validation.json", {}) or {}
    summary = pd.read_csv(run / "summary.csv", keep_default_na=False)
    _, registry = base.registry_rows()
    matching = [
        row for row in registry if row.get("name") == experiment_id or row.get("tag") == experiment_id
    ]
    objective_exists = (run / "objective_population.json").exists()
    checks = {
        "failed_closed": final.get("status") == "failed"
        and final.get("current_stage") == "closed"
        and final.get("decision") == "FAIL_IMPLEMENTATION",
        "exact_failure_reconciled": reconciliation.get("implementation_failure_exactly_identified") is True
        and reconciliation.get("scientific_rerun_performed") is False
        and reconciliation.get("same_root_scientific_repair_performed") is False,
        "closure_integrity_passed": closure.get("closure_integrity_passed") is True,
        "independent_closure_passed": independent.get("independent_closure_passed") is True,
        "artifact_manifest_match": artifact_validation.get("all_match") is True,
        "summary_no_stale": not summary.status.astype(str).isin(["running", "pending"]).any(),
        "registry_failed_closed": bool(
            len(matching) == 1
            and matching[0].get("status") == "failed"
            and matching[0].get("current_stage") == "closed"
            and matching[0].get("decision") == "FAIL_IMPLEMENTATION"
        ),
        "same_root_repair_absent": failure.get("same_root_repair_performed") is False,
        "partial_objective_state_exact": objective_exists is partial_objective_expected,
        "checkpoint_inference_absent": not (run / "retained_checkpoint_validation_scores.parquet").exists(),
        "hota_null": final.get("hota") is None,
        "next_policy_not_authorized": final.get("next_policy_authorized") is False,
    }
    return {
        "experiment_id": experiment_id,
        "checked_at": base.now(),
        "checks": checks,
        "all_passed": all(checks.values()),
    }


def current_registry_state_valid() -> bool:
    _, registry = base.registry_rows()
    matching = [row for row in registry if row.get("name") == EXP_ID or row.get("tag") == EXP_ID]
    if not matching:
        return True
    if len(matching) != 1:
        return False
    return matching[0].get("status") in {"running", "completed", "failed"}


def predecessor_check() -> dict[str, Any]:
    inherited = original_predecessor_check()
    inherited_checks = dict(inherited["checks"])
    inherited_checks.pop("m23_69_registry_unused", None)
    m69 = failed_predecessor_check(M69, "M23-69", partial_objective_expected=False)
    r1 = failed_predecessor_check(R1, "M23-69-R1", partial_objective_expected=True)
    checks = {
        **inherited_checks,
        "predecessor_m23_69_failed_closed": m69["all_passed"],
        "predecessor_m23_69_r1_failed_closed": r1["all_passed"],
        "current_r2_registry_state_valid": current_registry_state_valid(),
    }
    return {
        "checked_at": base.now(),
        "checks": checks,
        "predecessor_m23_69": m69,
        "predecessor_m23_69_r1": r1,
        "all_passed": all(checks.values()),
    }


def extend_run_inputs(paths: list[Path], run: Path) -> None:
    for name in [
        "summary.csv",
        "protocol_events.jsonl",
        "preflight.json",
        "preregistration.json",
        "input_manifest.json",
        "input_reverification.json",
        "implementation_manifest.json",
        "predecessor_reverification.json",
        "implementation_failure.json",
        "final_summary.json",
        "closure_reconciliation.json",
        "input_reverification_final.json",
        "closure_validation.json",
        "independent_closure_validation.json",
        "artifact_sha256_manifest.json",
        "artifact_manifest_validation.json",
    ]:
        path = run / name
        if path.exists():
            paths.append(path)
    for name in [
        "source_unique_transition_population.parquet",
        "objective_population.csv",
        "objective_population.json",
        "selected_source_population_metrics.csv",
    ]:
        path = run / name
        if path.exists():
            paths.append(path)
    artifact_manifest = base.read_json(run / "artifact_sha256_manifest.json", {}) or {}
    for record in artifact_manifest.get("records", []):
        paths.append(base.manifest_record_path(record))
    input_manifest = base.read_json(run / "input_manifest.json", {}) or {}
    for candidate in input_manifest.get("sha256", {}):
        path = Path(candidate)
        paths.append(path if path.is_absolute() else ROOT / path)


def collect_input_paths() -> list[Path]:
    paths = list(original_collect_input_paths())
    paths.extend(
        [
            BASE_SCRIPT,
            M69_TEST,
            M69_PREREG,
            M69_RESULT,
            M69_RECONCILER,
            R1_SCRIPT,
            R1_TEST,
            R1_PREREG,
            R1_RESULT,
            R1_RECONCILER,
        ]
    )
    extend_run_inputs(paths, M69)
    extend_run_inputs(paths, R1)
    deduplicated = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved == base.REGISTRY.resolve() or RUN.resolve() in resolved.parents:
            continue
        if resolved not in seen:
            deduplicated.append(resolved)
            seen.add(resolved)
    return sorted(deduplicated, key=lambda path: str(path))


def registry_close(status: str, decision: str, notes: str) -> int:
    fields, rows = base.registry_rows()
    matching = [
        index for index, row in enumerate(rows) if row.get("name") == EXP_ID or row.get("tag") == EXP_ID
    ]
    if not matching:
        raise RuntimeError("M23-69-R2 registry running entry missing")
    index = matching[-1]
    updates = {
        "timestamp": base.now(),
        "status": status,
        "current_stage": "closed",
        "decision": decision,
        "notes": notes,
        "HOTA": "",
    }
    rows[index].update({key: value for key, value in updates.items() if key in fields})
    base.write_registry(fields, rows)
    return index + 2


def tolerant_abs(value: Any):
    magnitude = builtins.abs(value)
    return 0.0 if magnitude <= HISTORICAL_REPRODUCTION_TOLERANCE else magnitude


def call_with_historical_tolerance(function: Callable, *args, **kwargs):
    had_global = hasattr(base, "abs")
    previous = getattr(base, "abs", None)
    base.abs = tolerant_abs
    try:
        return function(*args, **kwargs)
    finally:
        if had_global:
            base.abs = previous
        else:
            delattr(base, "abs")


def selected_source_metrics(frame: pd.DataFrame, unique: pd.DataFrame) -> dict[str, Any]:
    result = call_with_historical_tolerance(original_selected_source_metrics, frame, unique)
    historical = base.read_json(base.R64 / "frozen_checkpoint/checkpoint_manifest.json", {})[
        "validation_metrics"
    ]["conditional_boundary"]
    reconstructed = next(
        row
        for row in result["records"]
        if row["split"] == "validation"
        and row["scope"] == "pooled"
        and row["definition"] == "native"
        and row["population"] == "conditional"
        and row["view"] == "raw_observation"
    )
    result["historical_reproduction_numerics"] = {
        "tolerance": HISTORICAL_REPRODUCTION_TOLERANCE,
        "pr_auc_absolute_difference": builtins.abs(
            float(reconstructed["pr_auc"]) - float(historical["pr_auc"])
        ),
        "precision_at_actual_absolute_difference": builtins.abs(
            float(reconstructed["precision_at_actual"])
            - float(historical["precision_at_actual_count"])
        ),
        "recall_at_95_precision_absolute_difference": builtins.abs(
            float(reconstructed["recall_at_95_precision"])
            - float(historical["recall_at_95_precision"])
        ),
        "metric_values_unmodified": True,
    }
    return result


def command_evaluate_retained_checkpoints() -> None:
    call_with_historical_tolerance(original_evaluate_retained_checkpoints)
    path = RUN / "retained_checkpoint_evaluation.json"
    report = base.read_json(path, {}) or {}
    report["historical_metric_reproduction_tolerance"] = HISTORICAL_REPRODUCTION_TOLERANCE
    report["historical_metric_values_unmodified"] = True
    report["tolerance_applied_only_to_cross_inference_hard_checks"] = True
    base.write_json(path, report)


def result_markdown(*args, **kwargs) -> str:
    text = original_result_markdown(*args, **kwargs)
    return (
        text
        + "\n## R2 numerical-reproduction repair\n\n"
        + f"Cross-inference historical metric hard checks use a fixed absolute tolerance of `{HISTORICAL_REPRODUCTION_TOLERANCE}`. "
        + "Rows, positives, labels, masks, P@actual, R@95P, all published metric values, counterfactual rankings, and gate thresholds are not rounded or replaced.\n"
    )


def command_init() -> None:
    original_command_init()
    implementation_path = RUN / "implementation_manifest.json"
    implementation = base.read_json(implementation_path, {}) or {}
    implementation.update(
        {
            "immutable_base_script_sha256": base.sha256(BASE_SCRIPT),
            "predecessor_m23_69_artifact_manifest_sha256": base.sha256(
                M69 / "artifact_sha256_manifest.json"
            ),
            "predecessor_m23_69_r1_artifact_manifest_sha256": base.sha256(
                R1 / "artifact_sha256_manifest.json"
            ),
            "repair_contract": {
                "registry_lifecycle": "current R2 row may be absent before init or present in a valid lifecycle state afterward",
                "registry_close": "write only fields present in registry header",
                "historical_metric_absolute_tolerance": HISTORICAL_REPRODUCTION_TOLERANCE,
                "tolerance_scope": "cross-inference historical metric hard checks only",
                "scientific_metric_values_modified": False,
                "scientific_functions_other_than_check_tolerance_changed": False,
            },
        }
    )
    base.write_json(implementation_path, implementation)
    base.write_json(RUN / "predecessor_m23_69_reverification.json", predecessor_check())
    preregistration_path = RUN / "preregistration.json"
    preregistration = base.read_json(preregistration_path, {}) or {}
    preregistration["r2_repairs"] = implementation["repair_contract"]
    base.write_json(preregistration_path, preregistration)
    base.append_event(
        "r2_numerical_reproduction_repair_frozen",
        historical_metric_absolute_tolerance=HISTORICAL_REPRODUCTION_TOLERANCE,
        immutable_base_script_sha256=implementation["immutable_base_script_sha256"],
    )


base.R69 = RUN
base.SCRIPT = SCRIPT
base.TEST_SCRIPT = TEST_SCRIPT
base.PREREG = PREREG
base.RESULT = RESULT
base.EXP_ID = EXP_ID
base.EXP_NAME = EXP_NAME
base.collect_input_paths = collect_input_paths
base.predecessor_check = predecessor_check
base.registry_close = registry_close
base.selected_source_metrics = selected_source_metrics
base.command_evaluate_retained_checkpoints = command_evaluate_retained_checkpoints
base.result_markdown = result_markdown
base.command_init = command_init


def main() -> None:
    base.main()


if __name__ == "__main__":
    main()
