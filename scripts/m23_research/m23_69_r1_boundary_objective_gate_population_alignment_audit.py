#!/usr/bin/env python3
"""M23-69-R1 registry-lifecycle repair over immutable M23-69."""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE_ROOT = ROOT / "outputs/mot20_m23_20260718"
BASE_SCRIPT = ROOT / "scripts/m23_research/m23_69_boundary_objective_gate_population_alignment_audit.py"
SCRIPT = ROOT / "scripts/m23_research/m23_69_r1_boundary_objective_gate_population_alignment_audit.py"
TEST_SCRIPT = ROOT / "scripts/m23_research/test_m23_69_r1_boundary_objective_gate_population_alignment_audit.py"
PREREG = ROOT / "docs/m23_69_r1_boundary_objective_gate_population_alignment_audit_prereg_20260725.md"
RESULT = ROOT / "docs/m23_69_r1_boundary_objective_gate_population_alignment_audit_result_20260725.md"
PREDECESSOR = BASE_ROOT / "m23_69_boundary_objective_gate_population_alignment_audit"
RUN = BASE_ROOT / "m23_69_r1_boundary_objective_gate_population_alignment_audit"
RECONCILER = ROOT / "scripts/m23_research/m23_69_failure_closure_reconcile.py"
PREDECESSOR_TEST = ROOT / "scripts/m23_research/test_m23_69_boundary_objective_gate_population_alignment_audit.py"
PREDECESSOR_PREREG = ROOT / "docs/m23_69_boundary_objective_gate_population_alignment_audit_prereg_20260725.md"
PREDECESSOR_RESULT = ROOT / "docs/m23_69_boundary_objective_gate_population_alignment_audit_result_20260725.md"
EXP_ID = "M23-69-R1"
EXP_NAME = "M23-69-R1 Boundary Objective–Gate Population Alignment Audit Repair"


SPECIFICATION = importlib.util.spec_from_file_location("m23_69_immutable_base", BASE_SCRIPT)
if SPECIFICATION is None or SPECIFICATION.loader is None:
    raise RuntimeError("cannot load immutable M23-69 implementation")
base = importlib.util.module_from_spec(SPECIFICATION)
sys.modules[SPECIFICATION.name] = base
SPECIFICATION.loader.exec_module(base)

original_collect_input_paths = base.collect_input_paths
original_predecessor_check = base.predecessor_check
original_command_init = base.command_init


def predecessor_m23_69_check() -> dict[str, Any]:
    final = base.read_json(PREDECESSOR / "final_summary.json", {}) or {}
    failure = base.read_json(PREDECESSOR / "implementation_failure.json", {}) or {}
    reconciliation = base.read_json(PREDECESSOR / "closure_reconciliation.json", {}) or {}
    closure = base.read_json(PREDECESSOR / "closure_validation.json", {}) or {}
    independent = base.read_json(PREDECESSOR / "independent_closure_validation.json", {}) or {}
    artifact_validation = base.read_json(PREDECESSOR / "artifact_manifest_validation.json", {}) or {}
    summary = pd.read_csv(PREDECESSOR / "summary.csv", keep_default_na=False)
    _, registry = base.registry_rows()
    matching = [row for row in registry if row.get("name") == "M23-69" or row.get("tag") == "M23-69"]
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
        "scientific_artifacts_absent": not any(
            (PREDECESSOR / name).exists()
            for name in [
                "objective_population.json",
                "checkpoint_inventory.json",
                "retained_checkpoint_evaluation.json",
                "gate_score_semantics.json",
                "final_diagnosis.json",
            ]
        ),
        "hota_null": final.get("hota") is None,
        "next_policy_not_authorized": final.get("next_policy_authorized") is False,
    }
    return {"experiment_id": "M23-69", "checked_at": base.now(), "checks": checks, "all_passed": all(checks.values())}


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
    predecessor = predecessor_m23_69_check()
    checks = {
        **inherited_checks,
        "predecessor_m23_69_failed_closed": predecessor["all_passed"],
        "current_r1_registry_state_valid": current_registry_state_valid(),
    }
    return {
        "checked_at": base.now(),
        "checks": checks,
        "predecessor_m23_69": predecessor,
        "all_passed": all(checks.values()),
    }


def collect_input_paths() -> list[Path]:
    paths = list(original_collect_input_paths())
    paths.extend(
        [
            BASE_SCRIPT,
            PREDECESSOR_TEST,
            PREDECESSOR_PREREG,
            PREDECESSOR_RESULT,
            RECONCILER,
        ]
    )
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
        paths.append(PREDECESSOR / name)
    artifact_manifest = base.read_json(PREDECESSOR / "artifact_sha256_manifest.json", {}) or {}
    for record in artifact_manifest.get("records", []):
        paths.append(base.manifest_record_path(record))
    input_manifest = base.read_json(PREDECESSOR / "input_manifest.json", {}) or {}
    for candidate in input_manifest.get("sha256", {}):
        path = Path(candidate)
        paths.append(path if path.is_absolute() else ROOT / path)
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
        raise RuntimeError("M23-69-R1 registry running entry missing")
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


def command_init() -> None:
    original_command_init()
    implementation_path = RUN / "implementation_manifest.json"
    implementation = base.read_json(implementation_path, {}) or {}
    implementation.update(
        {
            "immutable_base_script_sha256": base.sha256(BASE_SCRIPT),
            "predecessor_m23_69_artifact_manifest_sha256": base.sha256(
                PREDECESSOR / "artifact_sha256_manifest.json"
            ),
            "repair_contract": {
                "registry_preflight": "current R1 row may be absent before init or present in a valid lifecycle state afterward",
                "registry_close": "write only fields present in the central registry header",
                "scientific_functions_changed": False,
            },
        }
    )
    base.write_json(implementation_path, implementation)
    base.write_json(RUN / "predecessor_m23_69_reverification.json", predecessor_m23_69_check())
    preregistration_path = RUN / "preregistration.json"
    preregistration = base.read_json(preregistration_path, {}) or {}
    preregistration["r1_repairs"] = implementation["repair_contract"]
    base.write_json(preregistration_path, preregistration)
    base.append_event(
        "r1_registry_lifecycle_repair_frozen",
        immutable_base_script_sha256=implementation["immutable_base_script_sha256"],
        predecessor_artifact_manifest_sha256=implementation[
            "predecessor_m23_69_artifact_manifest_sha256"
        ],
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
base.command_init = command_init


def main() -> None:
    base.main()


if __name__ == "__main__":
    main()
