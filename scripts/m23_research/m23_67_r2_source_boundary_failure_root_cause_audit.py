"""M23-67-R2 process-identity repair and deterministic reproduction audit."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import inspect
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

if len(sys.argv) == 2 and sys.argv[1] == "_process-fixture-sleep":
    time.sleep(30)
    raise SystemExit(0)

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/mot20_m23_20260718"
R1_SCRIPT = ROOT / "scripts/m23_research/m23_67_r1_source_boundary_failure_root_cause_audit.py"
R1_TEST_SCRIPT = ROOT / "scripts/m23_research/test_m23_67_r1_source_boundary_audit.py"
R1_PREREG = ROOT / "docs/m23_67_r1_source_boundary_failure_root_cause_audit_prereg_20260724.md"
R1_RESULT = ROOT / "docs/m23_67_r1_source_boundary_failure_root_cause_audit_result_20260724.md"
R1 = BASE / "m23_67_r1_source_boundary_failure_root_cause_audit"
R2 = BASE / "m23_67_r2_source_boundary_failure_root_cause_audit"
SCRIPT = ROOT / "scripts/m23_research/m23_67_r2_source_boundary_failure_root_cause_audit.py"
TEST_SCRIPT = ROOT / "scripts/m23_research/test_m23_67_r2_source_boundary_audit.py"
PREREG = ROOT / "docs/m23_67_r2_source_boundary_failure_root_cause_audit_prereg_20260724.md"
RESULT = ROOT / "docs/m23_67_r2_source_boundary_failure_root_cause_audit_result_20260724.md"
EXP_ID = "M23-67-R2"
EXP_NAME = "M23-67-R2 — Process Identity Repair and Deterministic Reproduction"
STAGES = [
    "init",
    "verify-inputs",
    "run-mapping-audit",
    "run-label-audit",
    "run-population-audit",
    "run-score-distribution-audit",
    "run-stratified-boundary-audit",
    "diagnose",
    "reproduce-r1-scientific-payload",
    "validate",
    "summarize",
    "closed",
]

legacy_spec = importlib.util.spec_from_file_location("m23_67_r1_frozen_base", R1_SCRIPT)
if legacy_spec is None or legacy_spec.loader is None:
    raise RuntimeError("unable to load frozen M23-67-R1 implementation")
legacy = importlib.util.module_from_spec(legacy_spec)
sys.modules[legacy_spec.name] = legacy
legacy_spec.loader.exec_module(legacy)

SCIENTIFIC_FUNCTION_NAMES = [
    "bucket_value",
    "appearance_bucket",
    "gap_bucket",
    "safe_ap",
    "safe_roc",
    "precision_at_actual",
    "recall_at_precision",
    "binary_metrics",
    "quantile_summary",
    "synthetic_join_fixture",
    "mapping_join_self_test",
    "synthetic_self_test",
    "observable",
    "supervision",
    "_semantic_boundary_join_core",
    "semantic_boundary_join",
    "build_validation_observations",
    "deterministic_samples",
    "command_mapping_audit",
    "aggregate_transitions",
    "command_label_audit",
    "track_purity_table",
    "score_frozen_examples",
    "annotate_observations",
    "population_summary",
    "command_population_audit",
    "command_score_distribution_audit",
    "stratum_metrics",
    "command_stratified_audit",
    "command_diagnose",
]

REFERENCE_JSON = [
    "boundary_row_mapping_validation.json",
    "boundary_row_mapping_examples.json",
    "boundary_label_semantics_validation.json",
    "boundary_label_trace_samples.json",
    "boundary_aggregation_audit.json",
    "training_vs_audit_boundary_population.json",
    "boundary_score_distribution.json",
    "score_orientation_validation.json",
    "score_saturation_validation.json",
    "stratified_boundary_metrics.json",
    "gap_convention_reconciliation.json",
    "final_diagnosis.json",
]
REFERENCE_CSV = [
    "boundary_aggregation_audit.csv",
    "boundary_label_distribution.csv",
    "training_vs_audit_boundary_population.csv",
    "boundary_score_distribution.csv",
    "stratified_boundary_metrics.csv",
]
REFERENCE_PARQUET = [
    "validation_boundary_observations.parquet",
    "validation_unique_transitions.parquet",
    "boundary_observations_all_populations.parquet",
    "boundary_unique_transitions_all_populations.parquet",
]
VOLATILE_JSON_KEYS = {
    "experiment_id",
    "created_at",
    "checked_at",
    "validated_at",
    "timestamp",
    "closed_at",
}
PROCESS_TEXT_KEYS = (
    "m23_59",
    "m23_60",
    "m23_61",
    "m23_62",
    "m23_63",
    "m23_64",
    "m23_65",
    "m23_66",
    "m23_67",
    "m23_68",
    "m23_69",
    "trackeval",
    "eval_motstyle_trackeval",
    "tracker",
)
SHELL_EXECUTABLES = {"bash", "sh", "dash", "zsh", "fish", "timeout"}


class ReproductionFailure(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_function_hash(function: Any) -> str:
    source = inspect.getsource(function)
    tree = ast.parse(source)
    normalized = ast.dump(tree, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def scientific_function_hashes() -> dict[str, str]:
    return {
        name: normalized_function_hash(getattr(legacy, name))
        for name in SCIENTIFIC_FUNCTION_NAMES
    }


def _path_from_record(record: dict[str, Any]) -> Path:
    candidate = record.get("absolute_path") or record.get("path")
    path = Path(candidate)
    return path if path.is_absolute() else ROOT / path


def _extend_manifest_paths(paths: list[Path], manifest_path: Path) -> None:
    manifest = legacy.read_json(manifest_path, {}) or {}
    for record in manifest.get("records", []):
        path = _path_from_record(record)
        if path.resolve() != legacy.REGISTRY.resolve():
            paths.append(path)


def _extend_input_paths(paths: list[Path], manifest_path: Path) -> None:
    manifest = legacy.read_json(manifest_path, {}) or {}
    for relative in manifest.get("sha256", {}):
        path = Path(relative)
        paths.append(path if path.is_absolute() else ROOT / path)


core_inputs = list(legacy.CORE_INPUTS)
core_inputs.extend([
    R1_SCRIPT,
    R1_TEST_SCRIPT,
    R1_PREREG,
    R1_RESULT,
    R1 / "input_manifest.json",
    R1 / "implementation_manifest.json",
    R1 / "final_summary.json",
    R1 / "failure_record.json",
    R1 / "closure_validation.json",
    R1 / "independent_closure_validation.json",
    R1 / "artifact_sha256_manifest.json",
    R1 / "artifact_manifest_validation.json",
    R1 / "summary.csv",
    R1 / "protocol_events.jsonl",
    R1 / "final_diagnosis.json",
])
_extend_manifest_paths(core_inputs, R1 / "artifact_sha256_manifest.json")
_extend_input_paths(core_inputs, R1 / "input_manifest.json")
deduplicated_inputs: list[Path] = []
seen_inputs: set[Path] = set()
for input_path in core_inputs:
    resolved = input_path.resolve()
    if resolved not in seen_inputs and resolved != legacy.REGISTRY.resolve():
        deduplicated_inputs.append(resolved)
        seen_inputs.add(resolved)
CORE_INPUTS = deduplicated_inputs

legacy.R67_PREDECESSOR = R1
legacy.R67 = R2
legacy.SCRIPT = SCRIPT
legacy.TEST_SCRIPT = TEST_SCRIPT
legacy.PREREG = PREREG
legacy.RESULT = RESULT
legacy.EXP_ID = EXP_ID
legacy.EXP_NAME = EXP_NAME
legacy.STAGES = STAGES
legacy.CORE_INPUTS = CORE_INPUTS


def _read_process_record(pid: int) -> dict[str, Any] | None:
    process_root = Path("/proc") / str(pid)
    try:
        stat_text = (process_root / "stat").read_text(encoding="utf-8")
        right_paren = stat_text.rfind(")")
        if right_paren < 0:
            return None
        comm = stat_text[stat_text.find("(") + 1:right_paren]
        stat_fields = stat_text[right_paren + 2:].split()
        state = stat_fields[0]
        ppid = int(stat_fields[1])
        starttime_ticks = int(stat_fields[19])
        cmdline_raw = (process_root / "cmdline").read_bytes()
        argv = [
            token.decode("utf-8", errors="replace")
            for token in cmdline_raw.split(b"\0")
            if token
        ]
        try:
            executable = str((process_root / "exe").resolve(strict=True))
        except (FileNotFoundError, PermissionError, OSError):
            executable = ""
        return {
            "pid": pid,
            "ppid": ppid,
            "state": state,
            "comm": comm,
            "exe": executable,
            "argv": argv,
            "starttime_ticks": starttime_ticks,
        }
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError, ValueError, IndexError):
        return None


def _ancestor_pids(pid: int) -> set[int]:
    ancestors: set[int] = set()
    current = pid
    while current > 1 and current not in ancestors:
        ancestors.add(current)
        record = _read_process_record(current)
        if record is None:
            break
        parent = int(record["ppid"])
        if parent <= 0 or parent == current:
            break
        current = parent
    return ancestors


def _python_target(argv: list[str]) -> str | None:
    if not argv:
        return None
    if argv[0].lower().endswith(".py"):
        return argv[0]
    index = 1
    while index < len(argv):
        token = argv[index]
        if token == "-c":
            return None
        if token == "-m":
            return f"module:{argv[index + 1]}" if index + 1 < len(argv) else None
        if token in {"-W", "-X"}:
            index += 2
            continue
        if token.startswith("-"):
            index += 1
            continue
        return token
    return None


def _target_is_blocking(target: str | None) -> bool:
    if not target:
        return False
    lower = target.lower()
    if lower.startswith("module:"):
        return "trackeval" in lower or "tracker" in lower
    basename = Path(target).name.lower()
    if re.match(r"^m23_(?:59|6[0-9])(?:_|-).+\.py$", basename):
        return True
    if basename == "eval_motstyle_trackeval.py":
        return True
    return basename.endswith(".py") and ("trackeval" in basename or "tracker" in basename)


def _blocking_reason(record: dict[str, Any]) -> str | None:
    if record.get("state") in {"Z", "X"}:
        return None
    argv = list(record.get("argv") or [])
    executable_name = Path(str(record.get("exe") or record.get("comm") or "")).name.lower()
    argv_zero_name = Path(argv[0]).name.lower() if argv else executable_name
    if executable_name in SHELL_EXECUTABLES or argv_zero_name in SHELL_EXECUTABLES:
        return None
    is_python = executable_name.startswith("python") or argv_zero_name.startswith("python")
    if is_python:
        target = _python_target(argv)
        return f"python_target:{target}" if _target_is_blocking(target) else None
    direct_target = argv[0] if argv else str(record.get("exe") or "")
    return f"direct_target:{direct_target}" if _target_is_blocking(direct_target) else None


def classify_process_records(
    records: list[dict[str, Any]],
    *,
    current_pid: int,
    ancestor_pids: set[int],
) -> dict[str, list[dict[str, Any]]]:
    blocking: list[dict[str, Any]] = []
    ignored: list[dict[str, Any]] = []
    mentions: list[dict[str, Any]] = []
    for record in records:
        pid = int(record["pid"])
        argv_text = " ".join(record.get("argv") or []).lower()
        text_mention = any(key in argv_text for key in PROCESS_TEXT_KEYS)
        reason = _blocking_reason(record)
        enriched = {**record, "classification_reason": reason}
        if pid == current_pid or pid in ancestor_pids:
            if reason or text_mention:
                ignored.append({**enriched, "ignored_reason": "self_or_ancestor"})
        elif reason:
            blocking.append(enriched)
        elif text_mention:
            mentions.append({**enriched, "ignored_reason": "text_mention_without_executable_identity"})
    return {
        "blocking_processes": blocking,
        "ignored_orchestrator_processes": ignored,
        "nonblocking_text_mentions": mentions,
    }


def _scan_processes(exclude_self: bool = True) -> dict[str, list[dict[str, Any]]]:
    current_pid = os.getpid()
    ancestors = _ancestor_pids(current_pid) if exclude_self else set()
    records = []
    for path in Path("/proc").iterdir():
        if path.name.isdigit():
            record = _read_process_record(int(path.name))
            if record is not None:
                records.append(record)
    return classify_process_records(records, current_pid=current_pid, ancestor_pids=ancestors)


def _parse_gpu_snapshot(gpu_output: str, compute_output: str) -> dict[str, Any]:
    memory_values: list[int] = []
    utilization_values: list[int] = []
    for line in gpu_output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) >= 2:
            try:
                memory_values.append(int(fields[0]))
                utilization_values.append(int(fields[1]))
            except ValueError:
                continue
    compute_processes = []
    for line in compute_output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if not fields or fields[0] in {"", "0"}:
            continue
        try:
            used_memory = int(fields[-1])
        except ValueError:
            used_memory = -1
        if used_memory != 0:
            compute_processes.append(line.strip())
    return {
        "available": bool(memory_values),
        "memory_used_mib": max(memory_values) if memory_values else None,
        "utilization_pct": max(utilization_values) if utilization_values else None,
        "compute_processes": compute_processes,
    }


def _single_process_gpu_snapshot(exclude_self: bool = True) -> dict[str, Any]:
    classified = _scan_processes(exclude_self=exclude_self)
    gpu_query = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True,
        capture_output=True,
    )
    compute_query = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
        text=True,
        capture_output=True,
    )
    gpu = _parse_gpu_snapshot(
        gpu_query.stdout if gpu_query.returncode == 0 else "",
        compute_query.stdout if compute_query.returncode == 0 else "",
    )
    return {
        "timestamp": legacy.now(),
        **classified,
        "relevant_processes": classified["blocking_processes"],
        "gpu": gpu,
    }


def _deduplicate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in records:
        key = (
            record.get("pid"),
            record.get("starttime_ticks"),
            record.get("classification_reason"),
            record.get("ignored_reason"),
        )
        unique[key] = record
    return list(unique.values())


def process_gpu_snapshot(exclude_self: bool = True) -> dict[str, Any]:
    first = _single_process_gpu_snapshot(exclude_self=exclude_self)
    time.sleep(1.0)
    second = _single_process_gpu_snapshot(exclude_self=exclude_self)
    blocking = _deduplicate_records(first["blocking_processes"] + second["blocking_processes"])
    ignored = _deduplicate_records(first["ignored_orchestrator_processes"] + second["ignored_orchestrator_processes"])
    mentions = _deduplicate_records(first["nonblocking_text_mentions"] + second["nonblocking_text_mentions"])
    compute_processes = sorted(set(first["gpu"]["compute_processes"] + second["gpu"]["compute_processes"]))
    memory_values = [value for value in [first["gpu"]["memory_used_mib"], second["gpu"]["memory_used_mib"]] if value is not None]
    utilization_values = [value for value in [first["gpu"]["utilization_pct"], second["gpu"]["utilization_pct"]] if value is not None]
    gpu = {
        "available": first["gpu"]["available"] or second["gpu"]["available"],
        "memory_used_mib": max(memory_values) if memory_values else None,
        "utilization_pct": max(utilization_values) if utilization_values else None,
        "compute_processes": compute_processes,
    }
    return {
        "timestamp": legacy.now(),
        "samples": [first, second],
        "blocking_processes": blocking,
        "ignored_orchestrator_processes": ignored,
        "nonblocking_text_mentions": mentions,
        "relevant_processes": blocking,
        "gpu": gpu,
    }


def process_classifier_self_test() -> dict[str, bool]:
    shell_record = {
        "pid": 101,
        "ppid": 1,
        "state": "S",
        "comm": "bash",
        "exe": "/bin/bash",
        "argv": ["bash", "-lc", "python scripts/m23_research/m23_67_r2_source_boundary_failure_root_cause_audit.py validate"],
        "starttime_ticks": 1,
    }
    runner_record = {
        "pid": 102,
        "ppid": 1,
        "state": "S",
        "comm": "python",
        "exe": sys.executable,
        "argv": [sys.executable, "-B", str(SCRIPT), "validate"],
        "starttime_ticks": 2,
    }
    tracker_record = {
        "pid": 103,
        "ppid": 1,
        "state": "S",
        "comm": "python",
        "exe": sys.executable,
        "argv": [sys.executable, "/tmp/eval_motstyle_trackeval.py"],
        "starttime_ticks": 3,
    }
    zombie_record = {**runner_record, "pid": 104, "state": "Z", "starttime_ticks": 4}
    unrelated_record = {
        "pid": 105,
        "ppid": 1,
        "state": "S",
        "comm": "python",
        "exe": sys.executable,
        "argv": [sys.executable, "/tmp/unrelated.py"],
        "starttime_ticks": 5,
    }
    classified = classify_process_records(
        [shell_record, runner_record, tracker_record, zombie_record, unrelated_record],
        current_pid=999,
        ancestor_pids={101},
    )
    blocking_pids = {int(record["pid"]) for record in classified["blocking_processes"]}
    ignored_pids = {int(record["pid"]) for record in classified["ignored_orchestrator_processes"]}
    checks = {
        "parent_shell_command_text_ignored": 101 in ignored_pids and 101 not in blocking_pids,
        "independent_python_runner_detected": 102 in blocking_pids,
        "trackeval_runner_detected": 103 in blocking_pids,
        "zombie_runner_ignored": 104 not in blocking_pids,
        "unrelated_python_ignored": 105 not in blocking_pids,
        "gpu_zero_placeholder_ignored": not _parse_gpu_snapshot("0, 0\n", "0, _, 0\n")["compute_processes"],
        "gpu_nonzero_compute_detected": bool(_parse_gpu_snapshot("128, 5\n", "321, python, 128\n")["compute_processes"]),
    }
    live_process = subprocess.Popen([sys.executable, "-B", str(SCRIPT), "_process-fixture-sleep"])
    try:
        live_seen = False
        for _ in range(30):
            time.sleep(0.1)
            scan = _scan_processes(exclude_self=True)
            if any(int(record["pid"]) == live_process.pid for record in scan["blocking_processes"]):
                live_seen = True
                break
        checks["live_sibling_runner_detected"] = live_seen
    finally:
        live_process.terminate()
        try:
            live_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            live_process.kill()
            live_process.wait(timeout=5)
    post_scan = _scan_processes(exclude_self=True)
    checks["live_fixture_reaped"] = not any(
        int(record["pid"]) == live_process.pid
        for record in post_scan["blocking_processes"]
    )
    clean_snapshot = _single_process_gpu_snapshot(exclude_self=True)
    ancestor_ids = _ancestor_pids(os.getpid())
    checks["self_and_ancestors_not_blocking"] = not any(
        int(record["pid"]) in ancestor_ids
        for record in clean_snapshot["blocking_processes"]
    )
    if not all(checks.values()):
        raise AssertionError({name: passed for name, passed in checks.items() if not passed})
    return checks


def _manifest_records_match(manifest_path: Path) -> tuple[list[dict[str, Any]], bool]:
    manifest = legacy.read_json(manifest_path, {}) or {}
    records = []
    for record in manifest.get("records", []):
        path = _path_from_record(record)
        actual = sha256(path) if path.exists() else None
        records.append({
            "path": str(record.get("path")),
            "expected": record.get("sha256"),
            "actual": actual,
            "match": actual == record.get("sha256"),
        })
    return records, bool(records) and all(record["match"] for record in records)


def _input_records_match(manifest_path: Path) -> tuple[list[dict[str, Any]], bool]:
    manifest = legacy.read_json(manifest_path, {}) or {}
    records = []
    for relative, expected in manifest.get("sha256", {}).items():
        path = Path(relative)
        path = path if path.is_absolute() else ROOT / path
        actual = sha256(path) if path.exists() else None
        records.append({"path": relative, "expected": expected, "actual": actual, "match": actual == expected})
    return records, bool(records) and all(record["match"] for record in records)


def verify_predecessor_r1() -> dict[str, Any]:
    final = legacy.read_json(R1 / "final_summary.json", {}) or {}
    failure = legacy.read_json(R1 / "failure_record.json", {}) or {}
    closure = legacy.read_json(R1 / "closure_validation.json", {}) or {}
    independent = legacy.read_json(R1 / "independent_closure_validation.json", {}) or {}
    artifact_validation = legacy.read_json(R1 / "artifact_manifest_validation.json", {}) or {}
    input_reverification = legacy.read_json(R1 / "input_reverification_at_failure.json", {}) or {}
    summary = pd.read_csv(R1 / "summary.csv", keep_default_na=False)
    artifact_checks, artifacts_match = _manifest_records_match(R1 / "artifact_sha256_manifest.json")
    input_checks, inputs_match = _input_records_match(R1 / "input_manifest.json")
    _, registry_rows = legacy.registry_rows()
    matching = [row for row in registry_rows if row.get("name") == "M23-67-R1" or row.get("tag") == "M23-67-R1"]
    checks = {
        "final_failed_implementation": final.get("status") == "failed" and final.get("decision") == "FAIL_IMPLEMENTATION",
        "failure_record_closed": failure.get("status") == "closed" and failure.get("same_root_repair_performed") is False,
        "closure_integrity_passed": closure.get("closure_integrity_passed") is True,
        "independent_closure_passed": independent.get("independent_closure_passed") is True,
        "artifact_validation_passed": artifact_validation.get("all_match") is True and artifacts_match,
        "input_reverification_passed": input_reverification.get("inputs_unchanged") is True and inputs_match,
        "summary_no_running_pending": not summary.status.astype(str).isin(["running", "pending"]).any(),
        "registry_failed_closed": bool(matching and matching[-1].get("status") == "failed" and matching[-1].get("current_stage") == "closed"),
        "next_policy_authorized_false": final.get("next_policy_authorized") is False,
        "m23_68_not_started": final.get("m23_68_started") is False,
    }
    return {
        "experiment_id": "M23-67-R1",
        "checked_at": legacy.now(),
        "checks": checks,
        "artifact_checks": artifact_checks,
        "input_checks": input_checks,
        "all_passed": all(checks.values()),
    }


def write_registry(fields: list[str], rows: list[dict[str, str]]) -> None:
    temporary = legacy.REGISTRY.with_suffix(".m23_67_r2.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(legacy.REGISTRY)


original_implementation_guard = legacy.implementation_guard


def implementation_guard() -> None:
    original_implementation_guard()
    manifest = legacy.read_json(R2 / "implementation_manifest.json", {}) or {}
    if not manifest:
        return
    checks = {
        "r1_scientific_base": sha256(R1_SCRIPT) == manifest.get("scientific_base_script_sha256"),
        "scientific_function_hashes": scientific_function_hashes() == manifest.get("scientific_function_hashes"),
    }
    if not all(checks.values()):
        raise RuntimeError(f"scientific implementation guard failed: {checks}")


legacy.process_gpu_snapshot = process_gpu_snapshot
legacy.verify_predecessor_r67 = verify_predecessor_r1
legacy.write_registry = write_registry
legacy.implementation_guard = implementation_guard


def command_init() -> None:
    if R2.exists():
        raise RuntimeError("M23-67-R2 output root already exists")
    if RESULT.exists():
        raise RuntimeError("M23-67-R2 result document already exists")
    if not PREREG.exists() or not TEST_SCRIPT.exists():
        raise RuntimeError("R2 preregistration or test script is missing")
    fields, registry = legacy.registry_rows()
    if any(row.get("name") == EXP_ID or row.get("tag") == EXP_ID for row in registry):
        raise RuntimeError("M23-67-R2 registry conflict")
    missing_inputs = [str(path) for path in CORE_INPUTS if not path.exists()]
    if missing_inputs:
        raise RuntimeError(f"missing frozen inputs: {missing_inputs[:10]}")
    snapshot = process_gpu_snapshot(exclude_self=True)
    gpu_memory = snapshot["gpu"].get("memory_used_mib")
    if snapshot["relevant_processes"] or snapshot["gpu"].get("compute_processes") or gpu_memory not in (None, 0):
        raise RuntimeError(f"process/GPU precondition failed: {snapshot}")
    r66_final = legacy.read_json(legacy.R66 / "final_summary.json", {}) or {}
    r66_closure = legacy.read_json(legacy.R66 / "closure_validation.json", {}) or {}
    r66_independent = legacy.read_json(legacy.R66 / "independent_closure_validation.json", {}) or {}
    if not (
        r66_final.get("status") == "completed"
        and r66_final.get("next_policy_authorized") is False
        and r66_closure.get("closure_integrity_passed") is True
        and r66_independent.get("independent_closure_passed") is True
    ):
        raise RuntimeError("M23-66 is not validly closed")
    predecessor_check = verify_predecessor_r1()
    if not predecessor_check["all_passed"]:
        raise RuntimeError(f"M23-67-R1 predecessor validation failed: {predecessor_check['checks']}")
    R2.mkdir(parents=True)
    legacy.initialize_summary()
    (R2 / "protocol_events.jsonl").touch()
    legacy.set_stage("init", "running", started_at=legacy.now())
    started = time.perf_counter()
    scientific_checks = legacy.synthetic_self_test()
    process_checks = process_classifier_self_test()
    fail_close_checks = legacy.fail_close_end_to_end_self_test()
    registry_line = legacy.registry_start()
    frozen_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in CORE_INPUTS}
    legacy.write_json(R2 / "preregistration.json", {
        "experiment_id": EXP_ID,
        "created_at": legacy.now(),
        "prereg_sha256": sha256(PREREG),
        "fixed_sequences": legacy.SEQUENCES,
        "fixed_gap_buckets": legacy.GAP_BUCKETS,
        "fixed_crowd_buckets": legacy.CROWD_BUCKETS,
        "fixed_purity_buckets": legacy.PURITY_BUCKETS,
        "fixed_decision_priority": [
            "boundary_label_mapping_failure",
            "boundary_population_mismatch",
            "boundary_score_collapse",
            "source_boundary_capacity_failure",
        ],
        "reproduction_reference": "M23-67-R1 provisional scientific payload",
        "reproduction_float_atol": 1e-12,
        **legacy.FIXED_DECLARATIONS,
    })
    legacy.write_json(R2 / "input_manifest.json", {
        "experiment_id": EXP_ID,
        "frozen_at": legacy.now(),
        "git_head": legacy.git_head(),
        "sha256": frozen_hashes,
        "r66_closed": True,
        "r67_failed_closed": True,
        "r1_failed_closed": True,
        "all_inputs_read_only": True,
    })
    legacy.write_json(R2 / "predecessor_r1_reverification.json", predecessor_check)
    implementation = {
        "experiment_id": EXP_ID,
        "created_at": legacy.now(),
        "git_head": legacy.git_head(),
        "script_sha256": sha256(SCRIPT),
        "test_script_sha256": sha256(TEST_SCRIPT),
        "prereg_sha256": sha256(PREREG),
        "scientific_base_script_sha256": sha256(R1_SCRIPT),
        "scientific_function_hashes": scientific_function_hashes(),
        "versions": legacy.software_versions(),
        "model_source_sha256": sha256(legacy.MODEL_SOURCE),
        "checkpoint_sha256": sha256(legacy.CHECKPOINT),
        "parameter_count": legacy.EXPECTED_PARAMETER_COUNT,
        "synthetic_scientific_checks": scientific_checks,
        "process_classifier_checks": process_checks,
        "fail_close_checks": fail_close_checks,
        "optimizer_constructed": False,
        "implementation_frozen": True,
    }
    legacy.write_json(R2 / "implementation_manifest.json", implementation)
    legacy.append_event("initialized", registry_running_line=registry_line, git_head=legacy.git_head())
    legacy.append_event("synthetic_fixture_passed", scientific_checks=scientific_checks, process_checks=process_checks, fail_close_checks=fail_close_checks)
    legacy.append_event(
        "implementation_frozen",
        script_sha256=implementation["script_sha256"],
        test_script_sha256=implementation["test_script_sha256"],
        prereg_sha256=implementation["prereg_sha256"],
        scientific_base_script_sha256=implementation["scientific_base_script_sha256"],
    )
    legacy.set_stage(
        "init",
        "completed",
        finished_at=legacy.now(),
        wall_seconds=time.perf_counter() - started,
        peak_rss_kb=legacy.resource.getrusage(legacy.resource.RUSAGE_SELF).ru_maxrss,
        rchar_delta=0,
        notes={
            "registry_running_line": registry_line,
            "scientific_checks": scientific_checks,
            "process_checks": process_checks,
            "fail_close_checks": fail_close_checks,
        },
    )
    print(json.dumps({"stage": "init", "status": "completed", "registry_line": registry_line}, sort_keys=True))


def command_verify_inputs() -> None:
    implementation_guard()
    legacy.set_stage("verify-inputs", "running", started_at=legacy.now())
    started = time.perf_counter()
    manifest = legacy.read_json(R2 / "input_manifest.json")
    hash_checks = []
    for relative, expected in manifest["sha256"].items():
        path = ROOT / relative
        actual = sha256(path) if path.exists() else None
        hash_checks.append({"path": relative, "expected": expected, "actual": actual, "match": actual == expected})
    contract = legacy.read_json(legacy.R62 / "feature_contract_v3_1.json")
    feature_142 = contract["features"][142]
    feature_143 = contract["features"][143]
    r66_final = legacy.read_json(legacy.R66 / "final_summary.json", {}) or {}
    predecessor = verify_predecessor_r1()
    implementation = legacy.read_json(R2 / "implementation_manifest.json", {}) or {}
    semantic_checks = {
        "feature_contract_hash": feature_142.get("contract_hash") == legacy.CONTRACT_HASH,
        "feature_142_is_crowd_density": feature_142.get("feature_name") == "geometry_14_crowd_density_over_100_clipped",
        "feature_143_is_nearest_neighbor": feature_143.get("feature_name") == "geometry_15_nearest_neighbor_distance",
        "appearance_mapped_is_separate_sidecar": True,
        "r66_closed": r66_final.get("status") == "completed" and r66_final.get("next_policy_authorized") is False,
        "r1_failed_closed_and_unchanged": predecessor.get("all_passed") is True,
        "scientific_base_sha_unchanged": sha256(R1_SCRIPT) == implementation.get("scientific_base_script_sha256"),
        "scientific_function_hashes_unchanged": scientific_function_hashes() == implementation.get("scientific_function_hashes"),
    }
    all_passed = all(item["match"] for item in hash_checks) and all(semantic_checks.values())
    result = {
        "experiment_id": EXP_ID,
        "checked_at": legacy.now(),
        "sha_checks": hash_checks,
        "semantic_checks": semantic_checks,
        "all_passed": all_passed,
        "first_mismatch": next((item for item in hash_checks if not item["match"]), None),
    }
    legacy.write_json(R2 / "input_reverification.json", result)
    legacy.write_json(R2 / "predecessor_r1_reverification.json", predecessor)
    if not all_passed:
        raise RuntimeError(f"input reverification failed: {result['first_mismatch']} {semantic_checks}")
    legacy.append_event("inputs_reverified", input_reverification_sha256=sha256(R2 / "input_reverification.json"))
    legacy.set_stage(
        "verify-inputs",
        "completed",
        finished_at=legacy.now(),
        wall_seconds=time.perf_counter() - started,
        peak_rss_kb=legacy.resource.getrusage(legacy.resource.RUSAGE_SELF).ru_maxrss,
        rchar_delta=0,
    )
    print(json.dumps({"stage": "verify-inputs", "status": "completed", "all_passed": True}, sort_keys=True))


def _normalize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _normalize_json(item)
            for key, item in sorted(value.items())
            if key not in VOLATILE_JSON_KEYS
        }
    if isinstance(value, list):
        return [_normalize_json(item) for item in value]
    return value


def _compare_nested(reference: Any, candidate: Any, path: str = "root") -> list[str]:
    differences: list[str] = []
    if isinstance(reference, bool) or isinstance(candidate, bool):
        if reference is not candidate:
            differences.append(f"{path}: {reference!r} != {candidate!r}")
        return differences
    if isinstance(reference, (int, float)) and isinstance(candidate, (int, float)):
        if np.isnan(reference) and np.isnan(candidate):
            return differences
        if abs(float(reference) - float(candidate)) > 1e-12:
            differences.append(f"{path}: {reference!r} != {candidate!r}")
        return differences
    if type(reference) is not type(candidate):
        differences.append(f"{path}: type {type(reference).__name__} != {type(candidate).__name__}")
        return differences
    if isinstance(reference, dict):
        if set(reference) != set(candidate):
            differences.append(f"{path}: keys {sorted(reference)} != {sorted(candidate)}")
            return differences
        for key in reference:
            differences.extend(_compare_nested(reference[key], candidate[key], f"{path}.{key}"))
            if len(differences) >= 20:
                break
        return differences
    if isinstance(reference, list):
        if len(reference) != len(candidate):
            differences.append(f"{path}: length {len(reference)} != {len(candidate)}")
            return differences
        for index, (reference_item, candidate_item) in enumerate(zip(reference, candidate)):
            differences.extend(_compare_nested(reference_item, candidate_item, f"{path}[{index}]"))
            if len(differences) >= 20:
                break
        return differences
    if reference != candidate:
        differences.append(f"{path}: {reference!r} != {candidate!r}")
    return differences


def _compare_json_artifact(name: str) -> dict[str, Any]:
    reference_path = R1 / name
    candidate_path = R2 / name
    if not reference_path.exists() or not candidate_path.exists():
        return {"artifact": name, "kind": "json", "match": False, "detail": "missing artifact"}
    reference = _normalize_json(legacy.read_json(reference_path))
    candidate = _normalize_json(legacy.read_json(candidate_path))
    differences = _compare_nested(reference, candidate)
    return {
        "artifact": name,
        "kind": "json",
        "match": not differences,
        "differences": differences,
        "reference_sha256": sha256(reference_path),
        "candidate_sha256": sha256(candidate_path),
    }


def _compare_frame_artifact(name: str, kind: str) -> dict[str, Any]:
    reference_path = R1 / name
    candidate_path = R2 / name
    if not reference_path.exists() or not candidate_path.exists():
        return {"artifact": name, "kind": kind, "match": False, "detail": "missing artifact"}
    reference = pd.read_csv(reference_path) if kind == "csv" else pd.read_parquet(reference_path)
    candidate = pd.read_csv(candidate_path) if kind == "csv" else pd.read_parquet(candidate_path)
    for frame in (reference, candidate):
        if "experiment_id" in frame.columns:
            frame.drop(columns=["experiment_id"], inplace=True)
    detail = ""
    try:
        pd.testing.assert_frame_equal(
            reference,
            candidate,
            check_dtype=False,
            check_exact=False,
            check_like=False,
            atol=1e-12,
            rtol=0,
        )
        match = True
    except AssertionError as error:
        match = False
        detail = str(error)[:4000]
    return {
        "artifact": name,
        "kind": kind,
        "match": match,
        "detail": detail,
        "reference_rows": int(len(reference)),
        "candidate_rows": int(len(candidate)),
        "reference_sha256": sha256(reference_path),
        "candidate_sha256": sha256(candidate_path),
    }


def command_reproduce() -> None:
    implementation_guard()
    legacy.set_stage("reproduce-r1-scientific-payload", "running", started_at=legacy.now())
    started = time.perf_counter()
    records = [_compare_json_artifact(name) for name in REFERENCE_JSON]
    records.extend(_compare_frame_artifact(name, "csv") for name in REFERENCE_CSV)
    records.extend(_compare_frame_artifact(name, "parquet") for name in REFERENCE_PARQUET)
    all_passed = bool(records) and all(record["match"] for record in records)
    report = {
        "experiment_id": EXP_ID,
        "checked_at": legacy.now(),
        "reference_experiment": "M23-67-R1",
        "float_atol": 1e-12,
        "records": records,
        "all_passed": all_passed,
        "first_mismatch": next((record for record in records if not record["match"]), None),
    }
    legacy.write_json(R2 / "r1_scientific_reproduction.json", report)
    if not all_passed:
        raise ReproductionFailure(f"R1 scientific payload reproduction failed: {report['first_mismatch']}")
    legacy.append_event("r1_scientific_payload_reproduced", record_count=len(records), float_atol=1e-12)
    legacy.set_stage(
        "reproduce-r1-scientific-payload",
        "completed",
        finished_at=legacy.now(),
        wall_seconds=time.perf_counter() - started,
        peak_rss_kb=legacy.resource.getrusage(legacy.resource.RUSAGE_SELF).ru_maxrss,
        rchar_delta=0,
    )
    print(json.dumps({"stage": "reproduce-r1-scientific-payload", "status": "completed", "records": len(records)}, sort_keys=True))


def command_validate() -> None:
    implementation_guard()
    legacy.set_stage("validate", "running", started_at=legacy.now())
    started = time.perf_counter()
    input_checks = legacy._manifest_input_checks(R2 / "input_manifest.json", ROOT)
    required = [
        "summary.csv",
        "protocol_events.jsonl",
        "preregistration.json",
        "input_manifest.json",
        "implementation_manifest.json",
        "input_reverification.json",
        "predecessor_r1_reverification.json",
        "boundary_row_mapping_validation.json",
        "boundary_row_mapping_examples.json",
        "boundary_label_semantics_validation.json",
        "boundary_label_distribution.csv",
        "boundary_label_trace_samples.json",
        "boundary_aggregation_audit.csv",
        "boundary_aggregation_audit.json",
        "training_vs_audit_boundary_population.csv",
        "training_vs_audit_boundary_population.json",
        "boundary_score_distribution.csv",
        "boundary_score_distribution.json",
        "score_orientation_validation.json",
        "score_saturation_validation.json",
        "stratified_boundary_metrics.csv",
        "stratified_boundary_metrics.json",
        "gap_convention_reconciliation.json",
        "final_diagnosis.json",
        "r1_scientific_reproduction.json",
    ]
    present = {name: (R2 / name).exists() for name in required}
    scope = {
        "experiment_id": EXP_ID,
        "scope_counts": legacy.SCOPE_COUNTS,
        "all_zero": all(value == 0 for value in legacy.SCOPE_COUNTS.values()),
        "new_raw_mot17_gt_reads": 0,
        "new_raw_mot20_gt_reads": 0,
        "frozen_label_sidecar_reads": True,
        "m23_68_started": False,
        **legacy.FIXED_DECLARATIONS,
    }
    process_gpu = process_gpu_snapshot(exclude_self=True)
    legacy.write_json(R2 / "scope_validation.json", scope)
    legacy.write_json(R2 / "process_gpu_validation.json", process_gpu)
    gpu_memory = process_gpu.get("gpu", {}).get("memory_used_mib")
    process_gpu_idle = bool(
        not process_gpu.get("relevant_processes")
        and not process_gpu.get("gpu", {}).get("compute_processes")
        and gpu_memory in (None, 0)
    )
    predecessor = verify_predecessor_r1()
    reproduction = legacy.read_json(R2 / "r1_scientific_reproduction.json", {}) or {}
    validation = {
        "experiment_id": EXP_ID,
        "validated_at": legacy.now(),
        "inputs_unchanged": bool(input_checks) and all(item["match"] for item in input_checks),
        "input_checks": input_checks,
        "required_artifacts": present,
        "required_artifacts_present": all(present.values()),
        "scope_passed": scope["all_zero"],
        "implementation_guard_passed": True,
        "scientific_reproduction_passed": reproduction.get("all_passed") is True,
        "process_gpu_idle": process_gpu_idle,
        "predecessor_r1_unchanged_failed_closed": predecessor.get("all_passed") is True,
    }
    validation["all_passed"] = all(
        validation[key]
        for key in [
            "inputs_unchanged",
            "required_artifacts_present",
            "scope_passed",
            "implementation_guard_passed",
            "scientific_reproduction_passed",
            "process_gpu_idle",
            "predecessor_r1_unchanged_failed_closed",
        ]
    )
    legacy.write_json(R2 / "validation_report.json", validation)
    if not validation["all_passed"]:
        raise RuntimeError("validation failed")
    legacy.append_event("validation_completed", all_passed=True, scientific_reproduction_passed=True)
    legacy.set_stage(
        "validate",
        "completed",
        finished_at=legacy.now(),
        wall_seconds=time.perf_counter() - started,
        peak_rss_kb=legacy.resource.getrusage(legacy.resource.RUSAGE_SELF).ru_maxrss,
        rchar_delta=0,
    )
    print(json.dumps({"stage": "validate", "status": "completed", "all_passed": True}, sort_keys=True))


original_result_markdown = legacy.result_markdown


def result_markdown(final: dict[str, Any], diagnosis: dict[str, Any], population: dict[str, Any], orientation: dict[str, Any], saturation: dict[str, Any]) -> str:
    markdown = original_result_markdown(final, diagnosis, population, orientation, saturation)
    return markdown.replace("M23-67-R1", "M23-67-R2")


legacy.result_markdown = result_markdown

synthetic_join_fixture = legacy.synthetic_join_fixture
semantic_boundary_join = legacy.semantic_boundary_join
synthetic_self_test = legacy.synthetic_self_test
fail_close_end_to_end_self_test = legacy.fail_close_end_to_end_self_test


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=[
        "init",
        "verify-inputs",
        "run-mapping-audit",
        "run-label-audit",
        "run-population-audit",
        "run-score-distribution-audit",
        "run-stratified-boundary-audit",
        "diagnose",
        "reproduce-r1-scientific-payload",
        "validate",
        "summarize",
    ])
    arguments = parser.parse_args()
    commands = {
        "init": command_init,
        "verify-inputs": command_verify_inputs,
        "run-mapping-audit": legacy.command_mapping_audit,
        "run-label-audit": legacy.command_label_audit,
        "run-population-audit": legacy.command_population_audit,
        "run-score-distribution-audit": legacy.command_score_distribution_audit,
        "run-stratified-boundary-audit": legacy.command_stratified_audit,
        "diagnose": legacy.command_diagnose,
        "reproduce-r1-scientific-payload": command_reproduce,
        "validate": command_validate,
        "summarize": legacy.command_summarize,
    }
    try:
        commands[arguments.command]()
    except Exception as error:
        if R2.exists():
            legacy.append_event("stage_failed", command=arguments.command, error=repr(error), traceback=traceback.format_exc())
            if isinstance(error, ReproductionFailure):
                decision = "FAIL_REPRODUCTION"
            else:
                decision = "FAIL_IMPLEMENTATION" if (R2 / "implementation_manifest.json").exists() else "FAIL_INITIALIZATION"
            legacy.fail_close(decision, repr(error))
        raise


if __name__ == "__main__":
    main()
