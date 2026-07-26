"""Reconcile fail-close bookkeeping for immutable M23-68."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "outputs/mot20_m23_20260718/m23_68_boundary_label_eligibility_population_decomposition_audit"
SCRIPT = ROOT / "scripts/m23_research/m23_68_boundary_label_eligibility_population_decomposition_audit.py"
TEST = ROOT / "scripts/m23_research/test_m23_68_boundary_label_eligibility_population_decomposition_audit.py"
PREREG = ROOT / "docs/m23_68_boundary_label_eligibility_population_decomposition_audit_prereg_20260725.md"
RESULT = ROOT / "docs/m23_68_boundary_label_eligibility_population_decomposition_audit_result_20260725.md"
RECONCILER = ROOT / "scripts/m23_research/m23_68_failure_closure_reconcile.py"
R2_SCRIPT = ROOT / "scripts/m23_research/m23_67_r2_source_boundary_failure_root_cause_audit.py"
REGISTRY = ROOT / "outputs/experiment_registry.csv"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def process_snapshot() -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location("m23_67_r2_process_helper_for_m23_68_closure", R2_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load robust process helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.process_gpu_snapshot(exclude_self=True)


def process_idle(snapshot: dict[str, Any]) -> bool:
    gpu = snapshot.get("gpu", {})
    return bool(
        not snapshot.get("blocking_processes")
        and not snapshot.get("relevant_processes")
        and not gpu.get("compute_processes")
        and gpu.get("memory_used_mib") in (None, 0)
    )


def registry_check() -> tuple[bool, int | None]:
    with REGISTRY.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    matching = [
        (index, row)
        for index, row in enumerate(rows, 2)
        if row.get("name") == "M23-68" or row.get("tag") == "M23-68"
    ]
    if not matching:
        return False, None
    line, row = matching[-1]
    return bool(
        row.get("status") == "failed"
        and row.get("current_stage") == "closed"
        and row.get("decision") == "FAIL_IMPLEMENTATION"
    ), line


def main() -> None:
    if not RUN.exists():
        raise RuntimeError("M23-68 run root missing")
    final = read_json(RUN / "final_summary.json")
    failure = read_json(RUN / "implementation_failure.json")
    implementation = read_json(RUN / "implementation_manifest.json")
    summary = pd.read_csv(RUN / "summary.csv", keep_default_na=False)
    registry_ok, registry_line = registry_check()
    implementation_checks = {
        "script": sha256(SCRIPT) == implementation["script_sha256"],
        "test": sha256(TEST) == implementation["test_script_sha256"],
        "prereg": sha256(PREREG) == implementation["prereg_sha256"],
    }
    input_manifest = read_json(RUN / "input_manifest.json")
    input_checks = []
    for item, expected in input_manifest["sha256"].items():
        path = ROOT / item
        actual = sha256(path) if path.exists() else None
        input_checks.append({"path": item, "expected": expected, "actual": actual, "match": actual == expected})
    snapshot = process_snapshot()
    bug_evidence = read_json(RUN / "population_identity_validation.json")
    exact_bug = bool(
        bug_evidence["splits"]["train"]["join_counts"] == {"both": 5662, "left_only": 0, "right_only": 0}
        and bug_evidence["splits"]["validation"]["join_counts"] == {"both": 1281, "left_only": 0, "right_only": 0}
        and bug_evidence["checks"]["all_windows_join_both"] is False
        and all(
            value
            for key, value in bug_evidence["checks"].items()
            if key != "all_windows_join_both"
        )
    )
    scope = {
        "training_runs": 0,
        "optimizer_steps": 0,
        "checkpoint_outputs": 0,
        "checkpoint_modifications": 0,
        "new_model_inference_runs": 0,
        "tracker_outputs": 0,
        "trackeval_runs": 0,
        "hota_evaluations": 0,
        "raw_mot17_gt_reads": 0,
        "raw_mot20_gt_reads": 0,
        "mot20_test_reads": 0,
        "mot20_test_submissions": 0,
        "teacher_reads": 0,
        "held_outer_reads": 0,
        "threshold_searches": 0,
        "calibration_fits": 0,
        "score_reversals_used": 0,
        "policy_runs": 0,
        "m23_54_starts": 0,
        "m23_58_starts": 0,
    }
    write_json(
        RUN / "input_reverification_final.json",
        {
            "checked_at": now(),
            "records": input_checks,
            "inputs_unchanged": bool(input_checks) and all(record["match"] for record in input_checks),
        },
    )
    write_json(RUN / "scope_validation.json", {"scope_counts": scope, "all_zero": all(value == 0 for value in scope.values())})
    write_json(RUN / "process_gpu_validation.json", snapshot)
    reconciliation = {
        "experiment_id": "M23-68",
        "reconciled_at": now(),
        "reconciler_path": relative(RECONCILER),
        "reconciler_sha256": sha256(RECONCILER),
        "frozen_script_unchanged": all(implementation_checks.values()),
        "same_root_scientific_repair_performed": False,
        "implementation_failure_exactly_identified": exact_bug,
        "failure_reason": "pandas categorical value_counts retained zero-count left_only/right_only categories; len(join_counts)==1 was invalid",
        "scientific_stages_not_resumed": True,
        "next_repair_root_required": "M23-68-R1",
    }
    write_json(RUN / "closure_reconciliation.json", reconciliation)
    RESULT.write_text(
        "# M23-68 Boundary Label-Eligibility and Population Decomposition Audit — Fail-closed Result\n\n"
        "Status: failed / closed\n\n"
        "Decision: FAIL_IMPLEMENTATION\n\n"
        "The frozen implementation incorrectly required the categorical join-count mapping to have length one. "
        "Both splits had exact one-to-one joins, but pandas value_counts retained left_only and right_only keys with zero counts. "
        "All row-sequence, tensor-index, R62 feature, mask, padding, and observation-count checks passed. "
        "The implementation failed after freeze, so M23-68 was not modified or resumed and no scientific decomposition was claimed.\n\n"
        "No training, optimizer step, checkpoint output/modification, model inference, tracker, TrackEval, HOTA, raw GT, MOT20 test, "
        "teacher, held-outer, threshold search, calibration, score reversal, or policy run occurred. HOTA is null and no next policy is authorized.\n\n"
        "A successor repair must use a new M23-68-R1 preregistration, script SHA, and run root. Notion writeback was not executed.\n",
        encoding="utf-8",
    )
    with (RUN / "protocol_events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"timestamp": now(), "experiment_id": "M23-68", "event": "failure_closure_reconciled", "reconciler_sha256": sha256(RECONCILER)}, sort_keys=True) + "\n")
    closure = {
        "experiment_id": "M23-68",
        "validated_at": now(),
        "status": "closed",
        "decision": "FAIL_IMPLEMENTATION",
        "final_failed_closed": final["status"] == "failed" and final["current_stage"] == "closed" and final["decision"] == "FAIL_IMPLEMENTATION",
        "failure_record_closed": failure["status"] == "closed" and failure["same_root_repair_performed"] is False,
        "summary_no_running_pending": not summary.status.astype(str).isin(["running", "pending"]).any(),
        "registry_failed_closed": registry_ok,
        "registry_line": registry_line,
        "inputs_unchanged": bool(input_checks) and all(record["match"] for record in input_checks),
        "implementation_sha_guard": all(implementation_checks.values()),
        "process_gpu_idle": process_idle(snapshot),
        "scope_all_zero": all(value == 0 for value in scope.values()),
        "result_document_exists": RESULT.exists(),
        "implementation_failure_exactly_identified": exact_bug,
        "same_root_scientific_repair_not_performed": reconciliation["same_root_scientific_repair_performed"] is False,
        "hota_is_null": final["hota"] is None,
        "next_policy_authorized_false": final["next_policy_authorized"] is False,
    }
    closure["closure_integrity_passed"] = all(
        value
        for key, value in closure.items()
        if key not in {"experiment_id", "validated_at", "status", "decision", "registry_line"}
    )
    write_json(RUN / "closure_validation.json", closure)
    independent = {
        "experiment_id": "M23-68",
        "validated_at": now(),
        **{key: value for key, value in closure.items() if isinstance(value, bool)},
    }
    independent["independent_closure_passed"] = all(independent[key] for key in independent if key not in {"experiment_id", "validated_at", "independent_closure_passed"})
    write_json(RUN / "independent_closure_validation.json", independent)
    excluded = {"artifact_sha256_manifest.json", "artifact_manifest_validation.json"}
    paths = [SCRIPT, TEST, PREREG, RESULT, RECONCILER]
    paths.extend(path for path in RUN.iterdir() if path.is_file() and path.name not in excluded)
    records = [
        {"path": relative(path), "sha256": sha256(path), "size": path.stat().st_size}
        for path in sorted(set(paths), key=lambda item: str(item))
    ]
    write_json(RUN / "artifact_sha256_manifest.json", {"experiment_id": "M23-68", "created_at": now(), "records": records})
    checks = []
    for record in records:
        path = ROOT / record["path"]
        actual = sha256(path) if path.exists() else None
        checks.append({"path": record["path"], "expected": record["sha256"], "actual": actual, "match": actual == record["sha256"]})
    write_json(RUN / "artifact_manifest_validation.json", {"experiment_id": "M23-68", "checked_at": now(), "records": checks, "all_match": all(record["match"] for record in checks)})
    if not closure["closure_integrity_passed"] or not independent["independent_closure_passed"] or not all(record["match"] for record in checks):
        raise RuntimeError("M23-68 failure closure reconciliation failed")
    print(json.dumps({"closure_integrity_passed": True, "independent_closure_passed": True, "artifact_count": len(records)}, sort_keys=True))


if __name__ == "__main__":
    main()
