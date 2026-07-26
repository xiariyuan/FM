"""M23-68 boundary label-eligibility and population decomposition audit."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/mot20_m23_20260718"
R62 = BASE / "m23_59_relation_pretrained_hierarchical_flow_v3_gtfree_source_regeneration"
R63 = BASE / "m23_59_relation_pretrained_hierarchical_flow_v3_supervision_join"
R64 = BASE / "m23_59_relation_pretrained_hierarchical_flow_v3_pair_reconstruction_training_repair1"
R66 = BASE / "m23_66_v3_metric_correctness_source_target_decomposition_audit"
R2 = BASE / "m23_67_r2_source_boundary_failure_root_cause_audit"
R68 = BASE / "m23_68_boundary_label_eligibility_population_decomposition_audit"
SCRIPT = ROOT / "scripts/m23_research/m23_68_boundary_label_eligibility_population_decomposition_audit.py"
TEST_SCRIPT = ROOT / "scripts/m23_research/test_m23_68_boundary_label_eligibility_population_decomposition_audit.py"
R2_SCRIPT = ROOT / "scripts/m23_research/m23_67_r2_source_boundary_failure_root_cause_audit.py"
PREREG = ROOT / "docs/m23_68_boundary_label_eligibility_population_decomposition_audit_prereg_20260725.md"
RESULT = ROOT / "docs/m23_68_boundary_label_eligibility_population_decomposition_audit_result_20260725.md"
REGISTRY = ROOT / "outputs/experiment_registry.csv"
EXP_ID = "M23-68"
EXP_NAME = "M23-68 Boundary Label-Eligibility and Population Decomposition Audit"
SEQUENCES = ["MOT17-02", "MOT17-04", "MOT17-05", "MOT17-09", "MOT17-10", "MOT17-11", "MOT17-13"]
TRAIN_SEQUENCES = SEQUENCES[:5]
VAL_SEQUENCES = SEQUENCES[5:]
STAGES = ["init", "verify-inputs", "reconstruct-labels", "audit-population-identity", "decompose-population", "audit-score-capacity", "diagnose", "validate", "summarize", "closed"]
SUMMARY_FIELDS = ["experiment_id", "stage", "status", "started_at", "finished_at", "decision", "error", "wall_seconds", "peak_rss_kb", "notes"]
SCOPE_COUNTS = {
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
FIXED = {
    "post_hoc_diagnostic_only": True,
    "uses_frozen_gt_derived_label_sidecars": True,
    "not_deployable": True,
    "not_a_strict_result": True,
    "training_authorized": False,
    "next_policy_authorized": False,
    "hota": None,
}
THRESHOLDS = {
    "composition_ratio_low": 0.5,
    "composition_ratio_high": 2.0,
    "composition_absolute_difference": 0.001,
    "composition_fisher_p": 0.01,
    "composition_min_positives_per_group": 20,
    "composition_min_train_sequences_same_direction": 4,
    "component_ratio_low": 0.8,
    "component_ratio_high": 1.25,
    "component_absolute_difference": 0.0005,
    "sampling_min_excluded_window_fraction": 0.01,
    "component_min_sequences_same_direction": 3,
    "r2_severe_ratio_low": 0.2,
    "r2_severe_ratio_high": 5.0,
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_default(value: Any):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def parse_ids(value: Any) -> list[int]:
    if isinstance(value, str):
        return [int(item) for item in json.loads(value)]
    return [int(item) for item in value]


def sigmoid(value: float | np.ndarray):
    return 1.0 / (1.0 + np.exp(-np.clip(value, -80.0, 80.0)))


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


_R2_MODULE = None


def r2_module():
    global _R2_MODULE
    if _R2_MODULE is None:
        spec = importlib.util.spec_from_file_location("m23_67_r2_process_helper", R2_SCRIPT)
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot load M23-67-R2 process helper")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _R2_MODULE = module
    return _R2_MODULE


def process_gpu_snapshot() -> dict[str, Any]:
    return r2_module().process_gpu_snapshot(exclude_self=True)


def process_gpu_idle(snapshot: dict[str, Any]) -> bool:
    gpu = snapshot.get("gpu", {})
    return bool(
        not snapshot.get("blocking_processes")
        and not snapshot.get("relevant_processes")
        and not gpu.get("compute_processes")
        and gpu.get("memory_used_mib") in (None, 0)
    )


def registry_rows() -> tuple[list[str], list[dict[str, str]]]:
    with REGISTRY.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_registry(fields: list[str], rows: list[dict[str, str]]) -> None:
    temporary = REGISTRY.with_suffix(".m23_68.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(REGISTRY)


def registry_start() -> int:
    fields, rows = registry_rows()
    if any(row.get("name") == EXP_ID or row.get("tag") == EXP_ID for row in rows):
        raise RuntimeError("M23-68 registry entry already exists")
    record = {field: "" for field in fields}
    record.update(
        {
            "timestamp": now(),
            "kind": "post_hoc_diagnostic",
            "status": "running",
            "script": relative(SCRIPT),
            "dataset": "MOT17",
            "split": "frozen_sidecars",
            "tracker_family": "FM-Track/M23-59-v3",
            "variant": "boundary_label_eligibility_population_decomposition",
            "tag": EXP_ID,
            "run_root": relative(R68),
            "summary_csv": relative(R68 / "summary.csv"),
            "notes": "M23-68 initialized; diagnostic only; no training/tracker/TrackEval/HOTA",
            "name": EXP_ID,
            "current_stage": "init",
        }
    )
    rows.append(record)
    write_registry(fields, rows)
    return len(rows) + 1


def registry_close(status: str, decision: str, notes: str) -> int:
    fields, rows = registry_rows()
    matching = [index for index, row in enumerate(rows) if row.get("name") == EXP_ID or row.get("tag") == EXP_ID]
    if not matching:
        raise RuntimeError("M23-68 registry running entry missing")
    index = matching[-1]
    rows[index].update(
        {
            "timestamp": now(),
            "status": status,
            "current_stage": "closed",
            "decision": decision,
            "notes": notes,
            "HOTA": "",
        }
    )
    write_registry(fields, rows)
    return index + 2


def initialize_summary() -> None:
    rows = [{field: "" for field in SUMMARY_FIELDS} for _ in STAGES]
    for row, stage in zip(rows, STAGES):
        row.update(experiment_id=EXP_ID, stage=stage, status="pending")
    with (R68 / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def set_stage(stage: str, status: str, **updates) -> None:
    path = R68 / "summary.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    found = False
    for row in rows:
        if row["stage"] == stage:
            found = True
            row["status"] = status
            for key, value in updates.items():
                row[key] = json.dumps(value, sort_keys=True, default=json_default) if key == "notes" and not isinstance(value, str) else value
    if not found:
        raise KeyError(stage)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def append_event(event: str, **updates) -> None:
    record = {"timestamp": now(), "experiment_id": EXP_ID, "event": event, **updates}
    with (R68 / "protocol_events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, default=json_default) + "\n")


def collect_input_paths() -> list[Path]:
    paths = [
        R2_SCRIPT,
        ROOT / "scripts/m23_research/m23_67_r1_source_boundary_failure_root_cause_audit.py",
        ROOT / "scripts/m23_research/m23_63_v3_supervision_join_example_audit.py",
        ROOT / "scripts/m23_research/m23_64_v3_pair_reconstruction_training_repair.py",
        ROOT / "scripts/m23_research/m23_66_v3_metric_correctness_source_target_decomposition_audit.py",
        ROOT / "docs/m23_59_relation_pretrained_hierarchical_flow_v3_supervision_join_prereg_20260722.md",
        ROOT / "docs/m23_59_relation_pretrained_hierarchical_flow_v3_pair_reconstruction_training_repair1_prereg_20260723.md",
        ROOT / "docs/m23_66_v3_metric_correctness_source_target_decomposition_audit_prereg_20260724.md",
        ROOT / "docs/m23_67_r2_source_boundary_failure_root_cause_audit_prereg_20260724.md",
        ROOT / "docs/m23_67_r2_source_boundary_failure_root_cause_audit_result_20260724.md",
        R63 / "source_windows.parquet",
        R63 / "row_supervision.parquet",
        R63 / "example_manifest_train.json",
        R63 / "example_manifest_validation.json",
        R64 / "examples_train.npz",
        R64 / "examples_validation.npz",
        R64 / "node_examples_train.parquet",
        R64 / "node_examples_validation.parquet",
        R64 / "training_data_adapter.json",
        R64 / "frozen_checkpoint/relation_v3_frozen.pt",
        R64 / "final_summary.json",
        R64 / "closure_validation.json",
        R2 / "boundary_observations_all_populations.parquet",
        R2 / "training_vs_audit_boundary_population.json",
        R2 / "final_diagnosis.json",
        R2 / "final_summary.json",
        R2 / "closure_validation.json",
        R2 / "independent_closure_validation.json",
        R2 / "artifact_sha256_manifest.json",
        R2 / "artifact_manifest_validation.json",
        R2 / "input_manifest.json",
        R2 / "implementation_manifest.json",
        R2 / "summary.csv",
    ]
    artifact_manifest = read_json(R2 / "artifact_sha256_manifest.json", {}) or {}
    for record in artifact_manifest.get("records", []):
        path = Path(record["path"])
        paths.append(path if path.is_absolute() else ROOT / path)
    input_manifest = read_json(R2 / "input_manifest.json", {}) or {}
    for item in input_manifest.get("sha256", {}):
        path = Path(item)
        paths.append(path if path.is_absolute() else ROOT / path)
    for sequence in SEQUENCES:
        paths.extend(
            [
                R62 / "observables/MOT17" / sequence / "rows.parquet",
                R62 / "observables/MOT17" / sequence / "row_features.f16.npy",
                R62 / "observables/MOT17" / sequence / "manifest.json",
            ]
        )
    return sorted(set(paths), key=lambda path: str(path))


def hash_paths(paths: list[Path]) -> dict[str, str]:
    missing = [relative(path) if path.is_relative_to(ROOT) else str(path) for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(f"missing frozen inputs: {missing[:10]}")
    return {relative(path): sha256(path) for path in paths}


def verify_hash_map(expected: dict[str, str]) -> list[dict[str, Any]]:
    records = []
    for item, digest in expected.items():
        path = ROOT / item
        actual = sha256(path) if path.exists() else None
        records.append({"path": item, "expected": digest, "actual": actual, "match": actual == digest})
    return records


def predecessor_r2_check() -> dict[str, Any]:
    final = read_json(R2 / "final_summary.json", {}) or {}
    closure = read_json(R2 / "closure_validation.json", {}) or {}
    independent = read_json(R2 / "independent_closure_validation.json", {}) or {}
    artifact_validation = read_json(R2 / "artifact_manifest_validation.json", {}) or {}
    summary = pd.read_csv(R2 / "summary.csv", keep_default_na=False)
    _, registry = registry_rows()
    matching = [row for row in registry if row.get("name") == "M23-67-R2" or row.get("tag") == "M23-67-R2"]
    artifact_manifest = read_json(R2 / "artifact_sha256_manifest.json", {}) or {}
    artifact_checks = []
    for record in artifact_manifest.get("records", []):
        path = Path(record["path"])
        path = path if path.is_absolute() else ROOT / path
        actual = sha256(path) if path.exists() else None
        artifact_checks.append(actual == record.get("sha256"))
    checks = {
        "completed_closed": final.get("status") == "completed" and final.get("current_stage") == "closed",
        "decision_exact": final.get("decision") == "FAIL_BOUNDARY_POPULATION_SHIFT",
        "closure_passed": closure.get("closure_integrity_passed") is True,
        "independent_passed": independent.get("independent_closure_passed") is True,
        "artifact_validation_passed": artifact_validation.get("all_match") is True,
        "artifact_manifest_all_match": bool(artifact_checks) and all(artifact_checks),
        "summary_no_stale": not summary.status.astype(str).isin(["running", "pending"]).any(),
        "registry_completed_closed": bool(matching and matching[-1].get("status") == "completed" and matching[-1].get("current_stage") == "closed"),
        "next_policy_not_authorized": final.get("next_policy_authorized") is False,
    }
    return {"checked_at": now(), "checks": checks, "all_passed": all(checks.values())}


def implementation_guard() -> None:
    manifest = read_json(R68 / "implementation_manifest.json", {}) or {}
    if not manifest:
        return
    checks = {
        "script": sha256(SCRIPT) == manifest.get("script_sha256"),
        "test": sha256(TEST_SCRIPT) == manifest.get("test_script_sha256"),
        "prereg": sha256(PREREG) == manifest.get("prereg_sha256"),
        "r2_process_helper": sha256(R2_SCRIPT) == manifest.get("r2_process_helper_sha256"),
    }
    if not all(checks.values()):
        raise RuntimeError(f"implementation changed after freeze: {checks}")


def endpoint_labels(source: Any, destination: Any) -> tuple[int, int, str]:
    source_matched = source["supervision_status"] == "matched"
    destination_matched = destination["supervision_status"] == "matched"
    native_eligible = source_matched and destination_matched
    strict_eligible = bool(
        native_eligible
        and not source["distractor_removed"]
        and not destination["distractor_removed"]
        and not source["ambiguity_flag"]
        and not destination["ambiguity_flag"]
        and not source["tie_flag"]
        and not destination["tie_flag"]
    )
    native = int(source["gt_identity_key"] != destination["gt_identity_key"]) if native_eligible else -1
    strict = int(source["gt_identity_key"] != destination["gt_identity_key"]) if strict_eligible else -1
    if strict_eligible:
        reason = "strict_eligible"
    elif not native_eligible:
        reason = "unmatched_or_distractor_status"
    elif source["ambiguity_flag"] or destination["ambiguity_flag"]:
        reason = "ambiguity_endpoint"
    elif source["tie_flag"] or destination["tie_flag"]:
        reason = "tie_endpoint"
    elif source["distractor_removed"] or destination["distractor_removed"]:
        reason = "distractor_endpoint"
    else:
        reason = "strict_excluded_other"
    return native, strict, reason


def risk_ratio(first_rate: float | None, second_rate: float | None) -> float | None:
    if first_rate is None or second_rate is None or second_rate == 0:
        return None
    return float(first_rate / second_rate)


def rate_summary(labels: np.ndarray) -> dict[str, Any]:
    known = labels[np.isin(labels, [0, 1])]
    positives = int(np.sum(known == 1))
    negatives = int(np.sum(known == 0))
    return {
        "rows": int(len(labels)),
        "known": int(len(known)),
        "positives": positives,
        "negatives": negatives,
        "positive_rate": float(positives / len(known)) if len(known) else None,
    }


def material_ratio_shift(first_rate: float | None, second_rate: float | None, absolute_threshold: float) -> bool:
    ratio = risk_ratio(first_rate, second_rate)
    if ratio is None:
        return False
    return bool(
        (ratio < THRESHOLDS["component_ratio_low"] or ratio > THRESHOLDS["component_ratio_high"])
        and abs(float(first_rate) - float(second_rate)) >= absolute_threshold
    )


def classify_components(composition: bool, sampling: bool, weighting: bool, support: bool = True) -> str:
    if not support:
        return "inconclusive_population_decomposition"
    enabled = [name for name, flag in [
        ("sequence_composition_shift", composition),
        ("example_sampling_shift", sampling),
        ("observation_weighting_shift", weighting),
    ] if flag]
    if len(enabled) > 1:
        return "multiple_population_components"
    if len(enabled) == 1:
        return enabled[0]
    return "no_material_population_component"


def self_test() -> dict[str, bool]:
    base = {
        "supervision_status": "matched",
        "gt_identity_key": "a",
        "distractor_removed": False,
        "ambiguity_flag": False,
        "tie_flag": False,
    }
    different = {**base, "gt_identity_key": "b"}
    ambiguous = {**different, "ambiguity_flag": True}
    unknown = {**different, "supervision_status": "unknown"}
    checks = {
        "same_identity_negative": endpoint_labels(base, base)[:2] == (0, 0),
        "different_identity_positive": endpoint_labels(base, different)[:2] == (1, 1),
        "ambiguity_native_but_strict_excluded": endpoint_labels(base, ambiguous)[:2] == (1, -1),
        "unknown_ignored_both": endpoint_labels(base, unknown)[:2] == (-1, -1),
        "component_none": classify_components(False, False, False) == "no_material_population_component",
        "component_single": classify_components(True, False, False) == "sequence_composition_shift",
        "component_multiple": classify_components(True, True, False) == "multiple_population_components",
        "component_inconclusive": classify_components(False, False, False, False) == "inconclusive_population_decomposition",
    }
    if not all(checks.values()):
        raise AssertionError({key: value for key, value in checks.items() if not value})
    return checks


def stage_started(stage: str) -> tuple[float, int]:
    implementation_guard()
    set_stage(stage, "running", started_at=now())
    usage_path = Path(f"/proc/{os.getpid()}/io")
    rchar = 0
    if usage_path.exists():
        for line in usage_path.read_text().splitlines():
            if line.startswith("rchar:"):
                rchar = int(line.split()[1])
    return time.perf_counter(), rchar


def stage_finished(stage: str, started: float, rchar_before: int, notes=None) -> None:
    usage_path = Path(f"/proc/{os.getpid()}/io")
    rchar_after = rchar_before
    if usage_path.exists():
        for line in usage_path.read_text().splitlines():
            if line.startswith("rchar:"):
                rchar_after = int(line.split()[1])
    set_stage(
        stage,
        "completed",
        finished_at=now(),
        wall_seconds=time.perf_counter() - started,
        peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        notes={"rchar_delta": max(0, rchar_after - rchar_before), **(notes or {})},
    )


def command_init() -> None:
    if R68.exists() or RESULT.exists():
        raise RuntimeError("M23-68 output or result already exists")
    if not PREREG.exists() or not TEST_SCRIPT.exists():
        raise RuntimeError("M23-68 preregistration or test is missing")
    _, registry = registry_rows()
    if any(row.get("name") == EXP_ID or row.get("tag") == EXP_ID for row in registry):
        raise RuntimeError("M23-68 registry conflict")
    snapshot = process_gpu_snapshot()
    if not process_gpu_idle(snapshot):
        raise RuntimeError(f"process/GPU precondition failed: {snapshot}")
    predecessor = predecessor_r2_check()
    if not predecessor["all_passed"]:
        raise RuntimeError(f"M23-67-R2 predecessor check failed: {predecessor['checks']}")
    checks = self_test()
    R68.mkdir(parents=True)
    initialize_summary()
    (R68 / "protocol_events.jsonl").touch()
    set_stage("init", "running", started_at=now())
    started = time.perf_counter()
    registry_line = registry_start()
    frozen = hash_paths(collect_input_paths())
    write_json(
        R68 / "input_manifest.json",
        {
            "experiment_id": EXP_ID,
            "frozen_at": now(),
            "git_head": git_head(),
            "sha256": frozen,
            "input_count": len(frozen),
            "all_inputs_read_only": True,
        },
    )
    write_json(
        R68 / "preregistration.json",
        {
            "experiment_id": EXP_ID,
            "created_at": now(),
            "user_explicit_authorization": True,
            "fixed_sequences": SEQUENCES,
            "train_sequences": TRAIN_SEQUENCES,
            "validation_sequences": VAL_SEQUENCES,
            "label_definitions": ["m23_63_native", "m23_66_strict"],
            "thresholds": THRESHOLDS,
            "classification_priority": [
                "implementation_failure",
                "cross_protocol_label_eligibility_mismatch",
                "harmonized_population_component",
            ],
            **FIXED,
        },
    )
    implementation = {
        "experiment_id": EXP_ID,
        "created_at": now(),
        "git_head": git_head(),
        "script_sha256": sha256(SCRIPT),
        "test_script_sha256": sha256(TEST_SCRIPT),
        "prereg_sha256": sha256(PREREG),
        "r2_process_helper_sha256": sha256(R2_SCRIPT),
        "self_test_checks": checks,
        "implementation_frozen": True,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    write_json(R68 / "implementation_manifest.json", implementation)
    write_json(R68 / "predecessor_r2_reverification.json", predecessor)
    append_event("initialized", registry_running_line=registry_line, input_count=len(frozen))
    append_event(
        "implementation_frozen",
        script_sha256=implementation["script_sha256"],
        test_script_sha256=implementation["test_script_sha256"],
        prereg_sha256=implementation["prereg_sha256"],
    )
    set_stage(
        "init",
        "completed",
        finished_at=now(),
        wall_seconds=time.perf_counter() - started,
        peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        notes={"registry_line": registry_line, "self_test_checks": checks},
    )


def command_verify_inputs() -> None:
    started, rchar = stage_started("verify-inputs")
    manifest = read_json(R68 / "input_manifest.json", {}) or {}
    records = verify_hash_map(manifest.get("sha256", {}))
    predecessor = predecessor_r2_check()
    report = {
        "experiment_id": EXP_ID,
        "checked_at": now(),
        "records": records,
        "inputs_unchanged": bool(records) and all(record["match"] for record in records),
        "predecessor_r2": predecessor,
        "all_passed": bool(records) and all(record["match"] for record in records) and predecessor["all_passed"],
        "first_mismatch": next((record for record in records if not record["match"]), None),
    }
    write_json(R68 / "input_reverification.json", report)
    if not report["all_passed"]:
        raise RuntimeError(f"input reverification failed: {report['first_mismatch']}")
    append_event("inputs_reverified", input_count=len(records))
    stage_finished("verify-inputs", started, rchar, {"input_count": len(records)})


def supervision_lookup() -> dict[tuple[str, int], dict[str, Any]]:
    frame = pd.read_parquet(R63 / "row_supervision.parquet")
    columns = [
        "sequence",
        "row_index",
        "supervision_status",
        "gt_identity_key",
        "distractor_removed",
        "ambiguity_flag",
        "tie_flag",
    ]
    return {
        (str(row.sequence), int(row.row_index)): {
            "supervision_status": str(row.supervision_status),
            "gt_identity_key": str(row.gt_identity_key),
            "distractor_removed": bool(row.distractor_removed),
            "ambiguity_flag": bool(row.ambiguity_flag),
            "tie_flag": bool(row.tie_flag),
        }
        for row in frame[columns].itertuples(index=False)
    }


def command_reconstruct_labels() -> None:
    started, rchar = stage_started("reconstruct-labels")
    labels = supervision_lookup()
    r2_observations = pd.read_parquet(R2 / "boundary_observations_all_populations.parquet")
    scored = r2_observations[r2_observations.population.isin(["train", "validation"])].copy()
    score_key = ["sequence", "window_id", "src_row_index", "dst_row_index"]
    if scored.duplicated(score_key).any():
        raise RuntimeError("R2 frozen score observations are not unique by window endpoint key")
    score_lookup = {
        (str(row.sequence), str(row.window_id), int(row.src_row_index), int(row.dst_row_index)): {
            "label": int(row.label),
            "boundary_logit": float(row.boundary_logit),
            "boundary_probability": float(row.boundary_probability),
        }
        for row in scored.itertuples(index=False)
    }
    audit = r2_observations[r2_observations.population == "audit"].copy()
    if audit.duplicated(score_key).any():
        raise RuntimeError("R2 audit observations are not unique by window endpoint key")
    audit_lookup = {
        (str(row.sequence), str(row.window_id), int(row.src_row_index), int(row.dst_row_index)): {
            "label": int(row.label),
            "boundary_logit": float(row.boundary_logit),
        }
        for row in audit.itertuples(index=False)
    }
    records = []
    missing_scores = []
    for split, population in [("train", "train"), ("validation", "validation")]:
        metadata = pd.read_parquet(R64 / f"node_examples_{split}.parquet").sort_values("tensor_index", kind="mergesort")
        with np.load(R64 / f"examples_{split}.npz", allow_pickle=False) as archive:
            boundary_y = np.asarray(archive["boundary_y"], dtype=np.int8)
            node_y = np.asarray(archive["node_y"], dtype=np.int8)
            node_mask = np.asarray(archive["node_mask"], dtype=np.uint8)
        for row in metadata.itertuples(index=False):
            tensor_index = int(row.tensor_index)
            ids = parse_ids(row.source_row_indices)
            for position, (source_row, destination_row) in enumerate(zip(ids, ids[1:])):
                key = (str(row.sequence), str(row.window_id), int(source_row), int(destination_row))
                score = score_lookup.get(key)
                if score is None:
                    missing_scores.append(key)
                    continue
                native, strict, reason = endpoint_labels(
                    labels[(str(row.sequence), int(source_row))],
                    labels[(str(row.sequence), int(destination_row))],
                )
                frozen = int(boundary_y[tensor_index, position])
                audit_record = audit_lookup.get(key)
                records.append(
                    {
                        "split": split,
                        "population": population,
                        "sequence": str(row.sequence),
                        "window_id": str(row.window_id),
                        "source_track_key": str(row.source_track_key),
                        "tensor_index": tensor_index,
                        "position": position,
                        "node_label": int(node_y[tensor_index]),
                        "optimizer_eligible_window": bool(node_y[tensor_index] >= 0),
                        "src_row_index": int(source_row),
                        "dst_row_index": int(destination_row),
                        "transition_key": f"{row.sequence}:{source_row}:{destination_row}",
                        "frozen_boundary_y": frozen,
                        "m23_63_native_label": native,
                        "m23_66_strict_label": strict,
                        "strict_eligibility_reason": reason,
                        "r2_population_label": score["label"],
                        "r2_audit_label": int(audit_record["label"]) if audit_record is not None else -99,
                        "boundary_logit": score["boundary_logit"],
                        "boundary_probability": score["boundary_probability"],
                        "node_mask_pair_valid": bool(node_mask[tensor_index, position] and node_mask[tensor_index, position + 1]),
                    }
                )
    frame = pd.DataFrame(records)
    if missing_scores:
        raise RuntimeError(f"missing R2 scores: {missing_scores[:10]}")
    expected_score_count = int(len(scored))
    finite = bool(np.isfinite(frame.boundary_logit).all() and np.isfinite(frame.boundary_probability).all())
    sigmoid_exact = bool(np.allclose(frame.boundary_probability.to_numpy(), sigmoid(frame.boundary_logit.to_numpy()), atol=1e-12, rtol=0))
    audit_expected = frame.sequence.isin(VAL_SEQUENCES)
    checks = {
        "record_count_exact_r2_score_count": len(frame) == expected_score_count,
        "all_mask_pairs_valid": bool(frame.node_mask_pair_valid.all()),
        "frozen_boundary_y_exact_m23_63_native": bool((frame.frozen_boundary_y == frame.m23_63_native_label).all()),
        "r2_population_label_exact_frozen": bool((frame.r2_population_label == frame.frozen_boundary_y).all()),
        "r2_audit_present_for_all_validation": bool((frame.loc[audit_expected, "r2_audit_label"] != -99).all()),
        "r2_audit_absent_for_train": bool((frame.loc[~audit_expected, "r2_audit_label"] == -99).all()),
        "r2_audit_label_exact_m23_66_strict": bool((frame.loc[audit_expected, "r2_audit_label"] == frame.loc[audit_expected, "m23_66_strict_label"]).all()),
        "finite_scores": finite,
        "sigmoid_probability_exact": sigmoid_exact,
    }
    native_group = frame.groupby("transition_key", sort=False).m23_63_native_label.nunique()
    strict_group = frame.groupby("transition_key", sort=False).m23_66_strict_label.nunique()
    checks["duplicate_native_labels_consistent"] = bool((native_group == 1).all())
    checks["duplicate_strict_labels_consistent"] = bool((strict_group == 1).all())
    label_effect = (
        frame.groupby(
            ["split", "sequence", "frozen_boundary_y", "m23_66_strict_label", "strict_eligibility_reason"],
            dropna=False,
            sort=True,
        )
        .size()
        .rename("observation_count")
        .reset_index()
    )
    label_effect.to_csv(R68 / "label_eligibility_confusion.csv", index=False)
    frame.to_parquet(R68 / "reconstructed_boundary_observations.parquet", index=False)
    report = {
        "experiment_id": EXP_ID,
        "checked_at": now(),
        "observation_count": int(len(frame)),
        "checks": checks,
        "all_passed": all(checks.values()),
        "native_known_strict_excluded_observations": int(((frame.m23_63_native_label.isin([0, 1])) & (frame.m23_66_strict_label == -1)).sum()),
        "native_positive_strict_excluded_observations": int(((frame.m23_63_native_label == 1) & (frame.m23_66_strict_label == -1)).sum()),
        "native_negative_strict_excluded_observations": int(((frame.m23_63_native_label == 0) & (frame.m23_66_strict_label == -1)).sum()),
    }
    write_json(R68 / "label_reconstruction_validation.json", report)
    if not report["all_passed"]:
        raise RuntimeError(f"label reconstruction hard checks failed: {checks}")
    append_event("labels_reconstructed", observation_count=len(frame), mixed_eligibility_count=report["native_known_strict_excluded_observations"])
    stage_finished("reconstruct-labels", started, rchar, {"observation_count": len(frame)})


def command_population_identity() -> None:
    started, rchar = stage_started("audit-population-identity")
    windows = pd.read_parquet(R63 / "source_windows.parquet")
    observations = pd.read_parquet(R68 / "reconstructed_boundary_observations.parquet")
    per_split = {}
    feature_checks = []
    for split in ["train", "validation"]:
        metadata = pd.read_parquet(R64 / f"node_examples_{split}.parquet").sort_values("tensor_index", kind="mergesort").reset_index(drop=True)
        selected_windows = windows[windows.split == split][["sequence", "window_id", "source_track_key", "row_indices"]].rename(columns={"row_indices": "window_row_indices"})
        joined = selected_windows.merge(
            metadata[["sequence", "window_id", "source_track_key", "source_row_indices", "tensor_index"]],
            on=["sequence", "window_id", "source_track_key"],
            how="outer",
            indicator=True,
            validate="one_to_one",
        )
        rows_equal = [
            parse_ids(first) == parse_ids(second)
            if isinstance(first, str) and isinstance(second, str)
            else False
            for first, second in zip(joined.window_row_indices, joined.source_row_indices)
        ]
        tensor_indices_exact = sorted(metadata.tensor_index.astype(int).tolist()) == list(range(len(metadata)))
        with np.load(R64 / f"examples_{split}.npz", allow_pickle=False) as archive:
            node_x = np.asarray(archive["node_x"], dtype=np.float16)
            node_mask = np.asarray(archive["node_mask"], dtype=np.uint8)
        feature_cache = {
            sequence: np.load(R62 / "observables/MOT17" / sequence / "row_features.f16.npy", mmap_mode="r")
            for sequence in (TRAIN_SEQUENCES if split == "train" else VAL_SEQUENCES)
        }
        exact_tensors = True
        exact_masks = True
        exact_padding = True
        for row in metadata.itertuples(index=False):
            ids = parse_ids(row.source_row_indices)
            index = int(row.tensor_index)
            expected = np.asarray(feature_cache[str(row.sequence)][ids], dtype=np.float16)
            exact_tensors = exact_tensors and np.array_equal(node_x[index, : len(ids)], expected)
            expected_mask = np.zeros(node_mask.shape[1], dtype=np.uint8)
            expected_mask[: len(ids)] = 1
            exact_masks = exact_masks and np.array_equal(node_mask[index], expected_mask)
            exact_padding = exact_padding and bool(np.all(node_x[index, len(ids) :] == 0))
        feature_checks.append(exact_tensors and exact_masks and exact_padding)
        expected_observations = int(sum(len(parse_ids(value)) - 1 for value in metadata.source_row_indices))
        actual_observations = int((observations.split == split).sum())
        per_split[split] = {
            "source_window_count": int(len(selected_windows)),
            "node_example_count": int(len(metadata)),
            "join_counts": {str(key): int(value) for key, value in joined._merge.value_counts().items()},
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
        "all_windows_join_both": all(item["join_counts"].get("both", 0) == item["source_window_count"] and len(item["join_counts"]) == 1 for item in per_split.values()),
        "all_window_rows_exact": all(item["row_sequences_exact"] for item in per_split.values()),
        "all_tensor_indices_exact": all(item["tensor_indices_exact"] for item in per_split.values()),
        "all_node_features_exact_r62": all(item["node_features_exact_r62"] for item in per_split.values()),
        "all_masks_exact": all(item["node_masks_exact"] for item in per_split.values()),
        "all_padding_zero": all(item["padding_zero"] for item in per_split.values()),
        "all_observation_counts_exact": all(item["expected_boundary_observations"] == item["actual_boundary_observations"] for item in per_split.values()),
    }
    report = {
        "experiment_id": EXP_ID,
        "checked_at": now(),
        "splits": per_split,
        "checks": checks,
        "all_passed": all(checks.values()),
        "frozen_example_construction_subset_absent": checks["all_windows_join_both"] and checks["all_window_rows_exact"],
        "optimizer_adapter_filter_reconstructed": True,
    }
    write_json(R68 / "population_identity_validation.json", report)
    if not report["all_passed"]:
        raise RuntimeError(f"population identity hard checks failed: {checks}")
    append_event("population_identity_audited", checks=checks)
    stage_finished("audit-population-identity", started, rchar, {"checks": checks})


def aggregate_unique(observations: pd.DataFrame) -> pd.DataFrame:
    grouped = observations.groupby(
        ["split", "sequence", "source_track_key", "src_row_index", "dst_row_index", "transition_key"],
        sort=True,
    ).agg(
        m23_63_native_label=("m23_63_native_label", "first"),
        m23_63_native_label_nunique=("m23_63_native_label", "nunique"),
        m23_66_strict_label=("m23_66_strict_label", "first"),
        m23_66_strict_label_nunique=("m23_66_strict_label", "nunique"),
        mean_logit=("boundary_logit", "mean"),
        mean_probability=("boundary_probability", "mean"),
        observation_count=("boundary_logit", "size"),
        optimizer_observation_count=("optimizer_eligible_window", "sum"),
    ).reset_index()
    grouped["boundary_probability"] = sigmoid(grouped.mean_logit.to_numpy())
    grouped["optimizer_eligible_transition"] = grouped.optimizer_observation_count > 0
    return grouped


def population_summary_row(
    data: pd.DataFrame,
    *,
    definition: str,
    view: str,
    selection: str,
    scope: str,
) -> dict[str, Any]:
    label_column = f"{definition}_label"
    labels = data[label_column].to_numpy(np.int8)
    summary = rate_summary(labels)
    return {
        "definition": definition,
        "view": view,
        "selection": selection,
        "scope": scope,
        **summary,
    }


def scope_frame(data: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope == "train":
        return data[data.sequence.isin(TRAIN_SEQUENCES)]
    if scope == "validation":
        return data[data.sequence.isin(VAL_SEQUENCES)]
    if scope == "all":
        return data
    return data[data.sequence == scope]


def lookup_summary(records: list[dict[str, Any]], definition: str, view: str, selection: str, scope: str) -> dict[str, Any]:
    for record in records:
        if (
            record["definition"] == definition
            and record["view"] == view
            and record["selection"] == selection
            and record["scope"] == scope
        ):
            return record
    raise KeyError((definition, view, selection, scope))


def sequence_direction_count(
    records: list[dict[str, Any]],
    *,
    definition: str,
    view: str,
    first_selection: str,
    second_selection: str,
    sequences: list[str],
    target_sign: int,
) -> int:
    count = 0
    for sequence in sequences:
        first = lookup_summary(records, definition, view, first_selection, sequence)["positive_rate"]
        second = lookup_summary(records, definition, view, second_selection, sequence)["positive_rate"]
        if first is None or second is None:
            continue
        difference = float(first - second)
        if (difference > 0 and target_sign > 0) or (difference < 0 and target_sign < 0):
            count += 1
    return count


def command_decompose_population() -> None:
    started, rchar = stage_started("decompose-population")
    observations = pd.read_parquet(R68 / "reconstructed_boundary_observations.parquet")
    unique = aggregate_unique(observations)
    consistency = bool(
        (unique.m23_63_native_label_nunique == 1).all()
        and (unique.m23_66_strict_label_nunique == 1).all()
    )
    if not consistency:
        raise RuntimeError("unique physical transition labels are inconsistent")
    unique.to_parquet(R68 / "unique_boundary_transitions.parquet", index=False)
    summaries = []
    scopes = SEQUENCES + ["train", "validation", "all"]
    for definition in ["m23_63_native", "m23_66_strict"]:
        for selection in ["full", "optimizer"]:
            raw_selected = observations if selection == "full" else observations[observations.optimizer_eligible_window]
            unique_selected = unique if selection == "full" else unique[unique.optimizer_eligible_transition]
            for scope in scopes:
                summaries.append(
                    population_summary_row(
                        scope_frame(raw_selected, scope),
                        definition=definition,
                        view="raw_observation",
                        selection=selection,
                        scope=scope,
                    )
                )
                summaries.append(
                    population_summary_row(
                        scope_frame(unique_selected, scope),
                        definition=definition,
                        view="unique_transition_primary",
                        selection=selection,
                        scope=scope,
                    )
                )
    pd.DataFrame(summaries).to_csv(R68 / "population_decomposition.csv", index=False)
    native_train_raw = lookup_summary(summaries, "m23_63_native", "raw_observation", "full", "train")
    strict_validation_raw = lookup_summary(summaries, "m23_66_strict", "raw_observation", "full", "validation")
    strict_train_raw = lookup_summary(summaries, "m23_66_strict", "raw_observation", "full", "train")
    native_validation_raw = lookup_summary(summaries, "m23_63_native", "raw_observation", "full", "validation")
    strict_train_unique = lookup_summary(summaries, "m23_66_strict", "unique_transition_primary", "full", "train")
    strict_validation_unique = lookup_summary(summaries, "m23_66_strict", "unique_transition_primary", "full", "validation")
    strict_train_optimizer_unique = lookup_summary(summaries, "m23_66_strict", "unique_transition_primary", "optimizer", "train")
    original = read_json(R2 / "training_vs_audit_boundary_population.json", {}) or {}
    original_ratio = float(original["ratios"]["train_to_audit_positive_ratio"])
    reconstructed_mixed_ratio = risk_ratio(native_train_raw["positive_rate"], strict_validation_raw["positive_rate"])
    if reconstructed_mixed_ratio is None or abs(reconstructed_mixed_ratio - original_ratio) > 1e-15:
        raise RuntimeError(f"R2 mixed ratio reproduction failed: {reconstructed_mixed_ratio} versus {original_ratio}")
    native_ratio = risk_ratio(native_train_raw["positive_rate"], native_validation_raw["positive_rate"])
    strict_raw_ratio = risk_ratio(strict_train_raw["positive_rate"], strict_validation_raw["positive_rate"])
    strict_unique_ratio = risk_ratio(strict_train_unique["positive_rate"], strict_validation_unique["positive_rate"])
    contingency = [
        [strict_train_unique["positives"], strict_train_unique["negatives"]],
        [strict_validation_unique["positives"], strict_validation_unique["negatives"]],
    ]
    _, fisher_p = fisher_exact(contingency, alternative="two-sided")
    validation_rate = strict_validation_unique["positive_rate"]
    train_above = 0
    train_below = 0
    for sequence in TRAIN_SEQUENCES:
        rate = lookup_summary(summaries, "m23_66_strict", "unique_transition_primary", "full", sequence)["positive_rate"]
        if rate is not None and validation_rate is not None:
            train_above += int(rate > validation_rate)
            train_below += int(rate < validation_rate)
    composition_support = bool(
        strict_train_unique["positives"] >= THRESHOLDS["composition_min_positives_per_group"]
        and strict_validation_unique["positives"] >= THRESHOLDS["composition_min_positives_per_group"]
    )
    composition_direction_count = max(train_above, train_below)
    composition_shift = bool(
        composition_support
        and strict_unique_ratio is not None
        and (
            strict_unique_ratio < THRESHOLDS["composition_ratio_low"]
            or strict_unique_ratio > THRESHOLDS["composition_ratio_high"]
        )
        and abs(strict_train_unique["positive_rate"] - strict_validation_unique["positive_rate"])
        >= THRESHOLDS["composition_absolute_difference"]
        and fisher_p < THRESHOLDS["composition_fisher_p"]
        and composition_direction_count >= THRESHOLDS["composition_min_train_sequences_same_direction"]
    )
    identity = read_json(R68 / "population_identity_validation.json", {}) or {}
    excluded_windows = int(identity["splits"]["train"]["optimizer_excluded_windows"])
    total_windows = int(identity["splits"]["train"]["node_example_count"])
    excluded_fraction = float(excluded_windows / total_windows)
    sampling_ratio = risk_ratio(
        strict_train_optimizer_unique["positive_rate"],
        strict_train_unique["positive_rate"],
    )
    sampling_difference = (
        None
        if strict_train_optimizer_unique["positive_rate"] is None or strict_train_unique["positive_rate"] is None
        else float(strict_train_optimizer_unique["positive_rate"] - strict_train_unique["positive_rate"])
    )
    sampling_sign = 1 if sampling_difference is not None and sampling_difference > 0 else -1
    sampling_direction_count = sequence_direction_count(
        summaries,
        definition="m23_66_strict",
        view="unique_transition_primary",
        first_selection="optimizer",
        second_selection="full",
        sequences=TRAIN_SEQUENCES,
        target_sign=sampling_sign,
    )
    sampling_shift = bool(
        material_ratio_shift(
            strict_train_optimizer_unique["positive_rate"],
            strict_train_unique["positive_rate"],
            THRESHOLDS["component_absolute_difference"],
        )
        and excluded_fraction >= THRESHOLDS["sampling_min_excluded_window_fraction"]
        and sampling_direction_count >= THRESHOLDS["component_min_sequences_same_direction"]
    )
    weighting_records = []
    weighting_shift = False
    for composition in ["train", "validation"]:
        raw = lookup_summary(summaries, "m23_66_strict", "raw_observation", "full", composition)
        primary = lookup_summary(summaries, "m23_66_strict", "unique_transition_primary", "full", composition)
        difference = None if raw["positive_rate"] is None or primary["positive_rate"] is None else float(raw["positive_rate"] - primary["positive_rate"])
        sign = 1 if difference is not None and difference > 0 else -1
        direction_count = 0
        for sequence in SEQUENCES:
            sequence_raw = lookup_summary(summaries, "m23_66_strict", "raw_observation", "full", sequence)["positive_rate"]
            sequence_unique = lookup_summary(summaries, "m23_66_strict", "unique_transition_primary", "full", sequence)["positive_rate"]
            if sequence_raw is None or sequence_unique is None:
                continue
            sequence_difference = sequence_raw - sequence_unique
            direction_count += int((sequence_difference > 0 and sign > 0) or (sequence_difference < 0 and sign < 0))
        material = bool(
            material_ratio_shift(
                raw["positive_rate"],
                primary["positive_rate"],
                THRESHOLDS["component_absolute_difference"],
            )
            and direction_count >= THRESHOLDS["component_min_sequences_same_direction"]
        )
        weighting_shift = weighting_shift or material
        weighting_records.append(
            {
                "composition": composition,
                "raw_positive_rate": raw["positive_rate"],
                "unique_positive_rate": primary["positive_rate"],
                "raw_to_unique_ratio": risk_ratio(raw["positive_rate"], primary["positive_rate"]),
                "absolute_difference": difference,
                "same_direction_sequence_count": direction_count,
                "material": material,
            }
        )
    reconstruction = read_json(R68 / "label_reconstruction_validation.json", {}) or {}
    mixed_definitions = reconstruction.get("native_known_strict_excluded_observations", 0) > 0
    original_severe = bool(original_ratio > THRESHOLDS["r2_severe_ratio_high"] or original_ratio < THRESHOLDS["r2_severe_ratio_low"])
    strict_severe = bool(
        strict_raw_ratio is not None
        and (strict_raw_ratio > THRESHOLDS["r2_severe_ratio_high"] or strict_raw_ratio < THRESHOLDS["r2_severe_ratio_low"])
    )
    if original_severe and strict_severe:
        stability = "STABLE_FAIL"
    elif original_severe and not strict_severe:
        stability = "UNSTABLE_FAIL"
    elif not original_severe and strict_severe:
        stability = "REVERSED"
    else:
        stability = "STABLE_PASS"
    scientific_component = classify_components(
        composition_shift,
        sampling_shift,
        weighting_shift,
        composition_support,
    )
    result = {
        "experiment_id": EXP_ID,
        "created_at": now(),
        "summaries": summaries,
        "r2_mixed_comparison": {
            "original_ratio": original_ratio,
            "reconstructed_ratio": reconstructed_mixed_ratio,
            "absolute_error": abs(reconstructed_mixed_ratio - original_ratio),
            "native_train_rate": native_train_raw["positive_rate"],
            "strict_validation_rate": strict_validation_raw["positive_rate"],
        },
        "harmonized_ratios": {
            "native_raw_train_to_validation": native_ratio,
            "strict_raw_train_to_validation": strict_raw_ratio,
            "strict_unique_train_to_validation": strict_unique_ratio,
        },
        "mixed_label_eligibility_definitions": mixed_definitions,
        "measurement_integrity_decision": "FAIL_MIXED_LABEL_ELIGIBILITY_DEFINITIONS" if mixed_definitions else "PASS_COMMON_LABEL_ELIGIBILITY",
        "r2_population_failure_stability": stability,
        "sequence_composition": {
            "material": composition_shift,
            "support_passed": composition_support,
            "risk_ratio": strict_unique_ratio,
            "absolute_difference": abs(strict_train_unique["positive_rate"] - strict_validation_unique["positive_rate"]),
            "fisher_p": float(fisher_p),
            "train_sequences_above_validation_rate": train_above,
            "train_sequences_below_validation_rate": train_below,
            "same_direction_count": composition_direction_count,
            "contingency": contingency,
        },
        "optimizer_adapter_sampling": {
            "material": sampling_shift,
            "risk_ratio": sampling_ratio,
            "absolute_difference": sampling_difference,
            "excluded_windows": excluded_windows,
            "total_windows": total_windows,
            "excluded_window_fraction": excluded_fraction,
            "same_direction_sequence_count": sampling_direction_count,
        },
        "duplicate_observation_weighting": {
            "material": weighting_shift,
            "records": weighting_records,
        },
        "harmonized_scientific_population_component": scientific_component,
        "physical_transition_label_consistency": consistency,
        "thresholds": THRESHOLDS,
    }
    write_json(R68 / "population_decomposition.json", result)
    append_event(
        "population_decomposed",
        measurement_integrity_decision=result["measurement_integrity_decision"],
        r2_stability=stability,
        scientific_component=scientific_component,
    )
    stage_finished("decompose-population", started, rchar, {"scientific_component": scientific_component})


def precision_at_actual(labels: np.ndarray, scores: np.ndarray, keys: np.ndarray) -> float:
    positives = int(np.sum(labels == 1))
    if positives <= 0:
        return 0.0
    order = np.lexsort((keys.astype(str), -scores))
    return float(np.mean(labels[order[:positives]] == 1))


def recall_at_precision(labels: np.ndarray, scores: np.ndarray, target: float) -> float:
    if int(np.sum(labels == 1)) == 0:
        return 0.0
    precision, recall, _ = precision_recall_curve(labels, scores)
    eligible = recall[precision >= target]
    return float(np.max(eligible)) if len(eligible) else 0.0


def binary_score_metrics(data: pd.DataFrame, label_column: str, key_column: str = "transition_key") -> dict[str, Any]:
    selected = data[data[label_column].isin([0, 1]) & np.isfinite(data.boundary_probability)].copy()
    labels = selected[label_column].to_numpy(np.int8)
    scores = selected.boundary_probability.to_numpy(float)
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if positives and negatives:
        pr_auc = float(average_precision_score(labels, scores))
        roc_auc = float(roc_auc_score(labels, scores))
    else:
        pr_auc = None
        roc_auc = None
    return {
        "rows": int(len(selected)),
        "positives": positives,
        "negatives": negatives,
        "base_rate": float(positives / len(selected)) if len(selected) else None,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "precision_at_actual": precision_at_actual(labels, scores, selected[key_column].to_numpy()) if len(selected) else 0.0,
        "recall_at_95_precision": recall_at_precision(labels, scores, 0.95) if len(selected) else 0.0,
        "positive_score_mean": float(scores[labels == 1].mean()) if positives else None,
        "negative_score_mean": float(scores[labels == 0].mean()) if negatives else None,
    }


def command_score_capacity() -> None:
    started, rchar = stage_started("audit-score-capacity")
    observations = pd.read_parquet(R68 / "reconstructed_boundary_observations.parquet")
    unique = pd.read_parquet(R68 / "unique_boundary_transitions.parquet")
    records = []
    per_definition_view: dict[str, Any] = {}
    for definition in ["m23_63_native", "m23_66_strict"]:
        label_column = f"{definition}_label"
        for view, data in [("raw_observation", observations), ("unique_transition_primary", unique)]:
            view_records = []
            for sequence in SEQUENCES:
                metrics = binary_score_metrics(data[data.sequence == sequence], label_column)
                record = {"definition": definition, "view": view, "scope": sequence, **metrics}
                records.append(record)
                view_records.append(record)
            for scope, sequence_set in [("train_pooled", TRAIN_SEQUENCES), ("validation_pooled", VAL_SEQUENCES), ("all_pooled", SEQUENCES)]:
                record = {
                    "definition": definition,
                    "view": view,
                    "scope": scope,
                    **binary_score_metrics(data[data.sequence.isin(sequence_set)], label_column),
                }
                records.append(record)
            macro = {}
            for metric in ["pr_auc", "roc_auc", "precision_at_actual", "recall_at_95_precision"]:
                values = [record[metric] for record in view_records if record[metric] is not None]
                macro[metric] = float(np.mean(values)) if values else None
            minimum_precision = float(min(record["precision_at_actual"] for record in view_records))
            per_definition_view[f"{definition}:{view}"] = {
                "per_sequence": view_records,
                "macro": macro,
                "minimum_sequence_precision_at_actual": minimum_precision,
            }
    pd.DataFrame(records).to_csv(R68 / "score_capacity.csv", index=False)
    strict_validation_records = [
        record
        for record in records
        if record["definition"] == "m23_66_strict"
        and record["view"] == "unique_transition_primary"
        and record["scope"] in VAL_SEQUENCES
    ]
    native_validation_records = [
        record
        for record in records
        if record["definition"] == "m23_63_native"
        and record["view"] == "unique_transition_primary"
        and record["scope"] in VAL_SEQUENCES
    ]
    def capacity_gate(selected: list[dict[str, Any]]) -> dict[str, Any]:
        macro = {
            metric: float(np.mean([record[metric] for record in selected if record[metric] is not None]))
            for metric in ["pr_auc", "precision_at_actual", "recall_at_95_precision"]
        }
        minimum = float(min(record["precision_at_actual"] for record in selected))
        passed = bool(
            macro["pr_auc"] >= 0.283
            and macro["precision_at_actual"] >= 0.35
            and macro["recall_at_95_precision"] >= 0.05
            and minimum >= 0.20
        )
        return {"macro": macro, "minimum_sequence_precision_at_actual": minimum, "passed": passed}
    report = {
        "experiment_id": EXP_ID,
        "created_at": now(),
        "records": records,
        "views": per_definition_view,
        "strict_validation_reference": capacity_gate(strict_validation_records),
        "native_validation_sensitivity": capacity_gate(native_validation_records),
        "reference_thresholds": {
            "macro_pr_auc": 0.283,
            "macro_precision_at_actual": 0.35,
            "macro_recall_at_95_precision": 0.05,
            "minimum_sequence_precision_at_actual": 0.20,
        },
        "diagnostic_only": True,
        "gate_or_policy_authorized": False,
        "checkpoint_executed": False,
    }
    write_json(R68 / "score_capacity.json", report)
    append_event(
        "score_capacity_audited",
        strict_passed=report["strict_validation_reference"]["passed"],
        native_passed=report["native_validation_sensitivity"]["passed"],
    )
    stage_finished("audit-score-capacity", started, rchar)


def command_diagnose() -> None:
    started, rchar = stage_started("diagnose")
    labels = read_json(R68 / "label_reconstruction_validation.json", {}) or {}
    identity = read_json(R68 / "population_identity_validation.json", {}) or {}
    decomposition = read_json(R68 / "population_decomposition.json", {}) or {}
    capacity = read_json(R68 / "score_capacity.json", {}) or {}
    hard_checks_passed = bool(labels.get("all_passed") and identity.get("all_passed"))
    mixed = bool(decomposition.get("mixed_label_eligibility_definitions"))
    scientific = decomposition.get("harmonized_scientific_population_component")
    if not hard_checks_passed:
        measurement = "FAIL_IMPLEMENTATION"
        overall = "implementation_failure"
        decision = "FAIL_IMPLEMENTATION"
    elif mixed:
        measurement = "FAIL_MIXED_LABEL_ELIGIBILITY_DEFINITIONS"
        overall = "cross_protocol_label_eligibility_mismatch"
        decision = "COMPLETED_POST_HOC_DIAGNOSTIC"
    else:
        measurement = "PASS_COMMON_LABEL_ELIGIBILITY"
        overall = scientific
        decision = "COMPLETED_POST_HOC_DIAGNOSTIC"
    diagnosis = {
        "experiment_id": EXP_ID,
        "status": "diagnosed",
        "measurement_integrity_decision": measurement,
        "r2_population_failure_stability": decomposition.get("r2_population_failure_stability"),
        "harmonized_scientific_population_component": scientific,
        "overall_primary_classification": overall,
        "experiment_decision": decision,
        "label_reconstruction_hard_checks_passed": hard_checks_passed,
        "mixed_label_eligibility_definitions": mixed,
        "sequence_composition_shift": decomposition.get("sequence_composition", {}).get("material"),
        "example_sampling_shift": decomposition.get("optimizer_adapter_sampling", {}).get("material"),
        "observation_weighting_shift": decomposition.get("duplicate_observation_weighting", {}).get("material"),
        "strict_validation_capacity_passed": capacity.get("strict_validation_reference", {}).get("passed"),
        "native_validation_capacity_passed": capacity.get("native_validation_sensitivity", {}).get("passed"),
        "m23_67_r2_modified": False,
        "m23_69_started": False,
        **FIXED,
    }
    write_json(R68 / "final_diagnosis.json", diagnosis)
    append_event(
        "diagnosis_completed",
        measurement=measurement,
        scientific=scientific,
        overall=overall,
    )
    stage_finished("diagnose", started, rchar, {"overall": overall})


def scoped_git_status() -> list[str]:
    paths = [SCRIPT, TEST_SCRIPT, PREREG, RESULT, R68, REGISTRY]
    process = subprocess.run(
        ["git", "status", "--short", "--", *[relative(path) for path in paths]],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    return [line for line in process.stdout.splitlines() if line.strip()]


def command_validate() -> None:
    started, rchar = stage_started("validate")
    input_manifest = read_json(R68 / "input_manifest.json", {}) or {}
    input_checks = verify_hash_map(input_manifest.get("sha256", {}))
    required = [
        "summary.csv",
        "protocol_events.jsonl",
        "preregistration.json",
        "input_manifest.json",
        "implementation_manifest.json",
        "input_reverification.json",
        "predecessor_r2_reverification.json",
        "label_reconstruction_validation.json",
        "label_eligibility_confusion.csv",
        "reconstructed_boundary_observations.parquet",
        "population_identity_validation.json",
        "population_decomposition.csv",
        "population_decomposition.json",
        "unique_boundary_transitions.parquet",
        "score_capacity.csv",
        "score_capacity.json",
        "final_diagnosis.json",
    ]
    present = {name: (R68 / name).exists() for name in required}
    process_gpu = process_gpu_snapshot()
    scope = {
        "experiment_id": EXP_ID,
        "scope_counts": SCOPE_COUNTS,
        "all_zero": all(value == 0 for value in SCOPE_COUNTS.values()),
        "frozen_label_sidecar_reads": True,
        "frozen_score_artifact_reads": True,
        **FIXED,
    }
    write_json(R68 / "scope_validation.json", scope)
    write_json(R68 / "process_gpu_validation.json", process_gpu)
    validation = {
        "experiment_id": EXP_ID,
        "validated_at": now(),
        "inputs_unchanged": bool(input_checks) and all(record["match"] for record in input_checks),
        "input_checks": input_checks,
        "required_artifacts": present,
        "required_artifacts_present": all(present.values()),
        "scope_all_zero": scope["all_zero"],
        "process_gpu_idle": process_gpu_idle(process_gpu),
        "implementation_guard_passed": True,
        "predecessor_r2_unchanged": predecessor_r2_check()["all_passed"],
        "label_reconstruction_passed": read_json(R68 / "label_reconstruction_validation.json", {}).get("all_passed") is True,
        "population_identity_passed": read_json(R68 / "population_identity_validation.json", {}).get("all_passed") is True,
        "hota_is_null": read_json(R68 / "final_diagnosis.json", {}).get("hota") is None,
        "next_policy_authorized_false": read_json(R68 / "final_diagnosis.json", {}).get("next_policy_authorized") is False,
    }
    validation["all_passed"] = all(
        validation[key]
        for key in [
            "inputs_unchanged",
            "required_artifacts_present",
            "scope_all_zero",
            "process_gpu_idle",
            "implementation_guard_passed",
            "predecessor_r2_unchanged",
            "label_reconstruction_passed",
            "population_identity_passed",
            "hota_is_null",
            "next_policy_authorized_false",
        ]
    )
    write_json(R68 / "validation_report.json", validation)
    if not validation["all_passed"]:
        raise RuntimeError("M23-68 validation failed")
    append_event("validation_completed", all_passed=True)
    stage_finished("validate", started, rchar)


def result_markdown(diagnosis: dict[str, Any], decomposition: dict[str, Any], capacity: dict[str, Any]) -> str:
    composition = decomposition["sequence_composition"]
    sampling = decomposition["optimizer_adapter_sampling"]
    weighting = decomposition["duplicate_observation_weighting"]
    strict_capacity = capacity["strict_validation_reference"]
    native_capacity = capacity["native_validation_sensitivity"]
    lines = [
        f"# {EXP_NAME} — Result",
        "",
        f"Status: completed / closed",
        f"Decision: {diagnosis['experiment_decision']}",
        f"Overall primary classification: {diagnosis['overall_primary_classification']}",
        "",
        "## Measurement integrity",
        "",
        f"- measurement_integrity_decision: {diagnosis['measurement_integrity_decision']}",
        f"- mixed_label_eligibility_definitions: {str(diagnosis['mixed_label_eligibility_definitions']).lower()}",
        f"- M23-67-R2 population failure stability: {diagnosis['r2_population_failure_stability']}",
        f"- original mixed ratio: {decomposition['r2_mixed_comparison']['original_ratio']:.15g}",
        f"- reconstructed mixed ratio: {decomposition['r2_mixed_comparison']['reconstructed_ratio']:.15g}",
        f"- native/native raw ratio: {decomposition['harmonized_ratios']['native_raw_train_to_validation']:.15g}",
        f"- strict/strict raw ratio: {decomposition['harmonized_ratios']['strict_raw_train_to_validation']:.15g}",
        f"- strict/strict unique ratio: {decomposition['harmonized_ratios']['strict_unique_train_to_validation']:.15g}",
        "",
        "The frozen M23-63 boundary tensor follows its native matched-endpoint definition exactly. The later M23-66/M23-67 audit excludes ambiguity/tie endpoints. M23-67-R2 compared a native-label train numerator to a strict-label audit denominator; M23-68 reports this as a cross-protocol estimand mismatch rather than an M23-63 tensor implementation defect.",
        "",
        "## Harmonized population decomposition",
        "",
        f"- scientific component: {diagnosis['harmonized_scientific_population_component']}",
        f"- sequence composition material: {str(composition['material']).lower()}",
        f"- strict unique train/validation risk ratio: {composition['risk_ratio']:.15g}",
        f"- absolute rate difference: {composition['absolute_difference']:.15g}",
        f"- Fisher exact p: {composition['fisher_p']:.15g}",
        f"- train sequences in common direction: {composition['same_direction_count']}",
        f"- optimizer-adapter sampling material: {str(sampling['material']).lower()}",
        f"- optimizer/full risk ratio: {sampling['risk_ratio']:.15g}",
        f"- excluded train-window fraction: {sampling['excluded_window_fraction']:.15g}",
        f"- duplicate weighting material: {str(weighting['material']).lower()}",
        "",
        "## Score capacity",
        "",
        f"- strict validation macro PR-AUC: {strict_capacity['macro']['pr_auc']:.15g}",
        f"- strict validation macro precision@actual: {strict_capacity['macro']['precision_at_actual']:.15g}",
        f"- strict validation macro recall@95P: {strict_capacity['macro']['recall_at_95_precision']:.15g}",
        f"- strict validation reference passed: {str(strict_capacity['passed']).lower()}",
        f"- native validation macro PR-AUC: {native_capacity['macro']['pr_auc']:.15g}",
        f"- native validation macro precision@actual: {native_capacity['macro']['precision_at_actual']:.15g}",
        f"- native validation macro recall@95P: {native_capacity['macro']['recall_at_95_precision']:.15g}",
        f"- native validation sensitivity passed: {str(native_capacity['passed']).lower()}",
        "",
        "## Scope",
        "",
        "No training, optimizer step, checkpoint output/modification, new model inference, tracker, TrackEval, HOTA, raw GT, MOT20 test, teacher, held-outer, threshold search, calibration, score reversal, policy, M23-54, or M23-58 run occurred. HOTA is null. No next policy is authorized.",
        "",
        "Notion writeback was not executed.",
        "",
    ]
    return "\n".join(lines)


def artifact_manifest() -> dict[str, Any]:
    excluded = {"artifact_sha256_manifest.json", "artifact_manifest_validation.json"}
    paths = [SCRIPT, TEST_SCRIPT, PREREG, RESULT]
    paths.extend(path for path in R68.iterdir() if path.is_file() and path.name not in excluded)
    records = [
        {"path": relative(path), "sha256": sha256(path), "size": path.stat().st_size}
        for path in sorted(set(paths), key=lambda item: str(item))
    ]
    return {"experiment_id": EXP_ID, "created_at": now(), "records": records}


def command_summarize() -> None:
    started, rchar = stage_started("summarize")
    diagnosis = read_json(R68 / "final_diagnosis.json", {}) or {}
    decomposition = read_json(R68 / "population_decomposition.json", {}) or {}
    capacity = read_json(R68 / "score_capacity.json", {}) or {}
    RESULT.write_text(result_markdown(diagnosis, decomposition, capacity), encoding="utf-8")
    stage_finished("summarize", started, rchar)
    decision = str(diagnosis["experiment_decision"])
    set_stage("closed", "closed", started_at=now(), finished_at=now(), decision=decision)
    registry_line = registry_close(
        "completed",
        decision,
        "measurement="
        + str(diagnosis["measurement_integrity_decision"])
        + "; stability="
        + str(diagnosis["r2_population_failure_stability"])
        + "; scientific="
        + str(diagnosis["harmonized_scientific_population_component"])
        + "; overall="
        + str(diagnosis["overall_primary_classification"])
        + "; HOTA=null; next_policy_authorized=false; result="
        + relative(RESULT),
    )
    final_input_checks = verify_hash_map(read_json(R68 / "input_manifest.json", {})["sha256"])
    write_json(
        R68 / "input_reverification_final.json",
        {
            "checked_at": now(),
            "records": final_input_checks,
            "inputs_unchanged": bool(final_input_checks) and all(record["match"] for record in final_input_checks),
        },
    )
    process_gpu = process_gpu_snapshot()
    summary = pd.read_csv(R68 / "summary.csv", keep_default_na=False)
    final = {
        "experiment_id": EXP_ID,
        "name": EXP_NAME,
        "status": "completed",
        "current_stage": "closed",
        "decision": decision,
        "closed_at": now(),
        "registry_line": registry_line,
        "diagnosis": diagnosis,
        "process_gpu": process_gpu,
        "scope_counts": SCOPE_COUNTS,
        "git_head": git_head(),
        "git_scoped_status": scoped_git_status(),
        "notion_writeback": "未执行Notion写回",
        "m23_69_started": False,
        **FIXED,
    }
    write_json(R68 / "final_summary.json", final)
    append_event(
        "experiment_closed",
        decision=decision,
        overall=diagnosis["overall_primary_classification"],
        registry_line=registry_line,
    )
    implementation = read_json(R68 / "implementation_manifest.json", {}) or {}
    implementation_checks = {
        "script": sha256(SCRIPT) == implementation.get("script_sha256"),
        "test": sha256(TEST_SCRIPT) == implementation.get("test_script_sha256"),
        "prereg": sha256(PREREG) == implementation.get("prereg_sha256"),
    }
    closure = {
        "experiment_id": EXP_ID,
        "validated_at": now(),
        "status": "closed",
        "decision": decision,
        "inputs_unchanged": all(record["match"] for record in final_input_checks),
        "implementation_checks": implementation_checks,
        "implementation_sha_guard": all(implementation_checks.values()),
        "summary_no_running_pending": not summary.status.astype(str).isin(["running", "pending"]).any(),
        "registry_completed_closed": True,
        "process_gpu_idle": process_gpu_idle(process_gpu),
        "scope_all_zero": all(value == 0 for value in SCOPE_COUNTS.values()),
        "result_document_exists": RESULT.exists(),
        "final_summary_exists": True,
        "hota_is_null": final["hota"] is None,
        "next_policy_authorized_false": final["next_policy_authorized"] is False,
        "predecessor_r2_unchanged": predecessor_r2_check()["all_passed"],
    }
    closure["closure_integrity_passed"] = all(
        closure[key]
        for key in [
            "inputs_unchanged",
            "implementation_sha_guard",
            "summary_no_running_pending",
            "registry_completed_closed",
            "process_gpu_idle",
            "scope_all_zero",
            "result_document_exists",
            "final_summary_exists",
            "hota_is_null",
            "next_policy_authorized_false",
            "predecessor_r2_unchanged",
        ]
    )
    write_json(R68 / "closure_validation.json", closure)
    independent = {
        "experiment_id": EXP_ID,
        "validated_at": now(),
        "final_status_completed_closed": final["status"] == "completed" and final["current_stage"] == "closed",
        "inputs_unchanged": closure["inputs_unchanged"],
        "implementation_sha_guard": closure["implementation_sha_guard"],
        "summary_no_running_pending": closure["summary_no_running_pending"],
        "registry_completed_closed": closure["registry_completed_closed"],
        "process_gpu_idle": closure["process_gpu_idle"],
        "scope_all_zero": closure["scope_all_zero"],
        "result_document_exists": closure["result_document_exists"],
        "hota_is_null": closure["hota_is_null"],
        "next_policy_authorized_false": closure["next_policy_authorized_false"],
        "predecessor_r2_unchanged": closure["predecessor_r2_unchanged"],
        "closure_integrity_passed": closure["closure_integrity_passed"],
    }
    independent["independent_closure_passed"] = all(
        value for key, value in independent.items() if key not in {"experiment_id", "validated_at"}
    )
    write_json(R68 / "independent_closure_validation.json", independent)
    manifest = artifact_manifest()
    write_json(R68 / "artifact_sha256_manifest.json", manifest)
    artifact_checks = []
    for record in manifest["records"]:
        path = ROOT / record["path"]
        actual = sha256(path) if path.exists() else None
        artifact_checks.append(
            {
                "path": record["path"],
                "expected": record["sha256"],
                "actual": actual,
                "match": actual == record["sha256"],
            }
        )
    write_json(
        R68 / "artifact_manifest_validation.json",
        {
            "experiment_id": EXP_ID,
            "checked_at": now(),
            "records": artifact_checks,
            "all_match": bool(artifact_checks) and all(record["match"] for record in artifact_checks),
        },
    )
    if not closure["closure_integrity_passed"] or not independent["independent_closure_passed"] or not all(record["match"] for record in artifact_checks):
        raise RuntimeError("post-summary closure validation failed")


def fail_close(error: BaseException) -> None:
    if not R68.exists() or not (R68 / "summary.csv").exists():
        return
    summary = pd.read_csv(R68 / "summary.csv", keep_default_na=False)
    running = summary[summary.status == "running"]
    if len(running):
        set_stage(str(running.iloc[0].stage), "failed", finished_at=now(), error=repr(error))
    summary = pd.read_csv(R68 / "summary.csv", keep_default_na=False)
    for stage in summary.loc[summary.status == "pending", "stage"].tolist():
        if stage != "closed":
            set_stage(stage, "skipped", finished_at=now(), error="fail_closed")
    set_stage("closed", "closed", started_at=now(), finished_at=now(), decision="FAIL_IMPLEMENTATION", error=repr(error))
    try:
        registry_line = registry_close("failed", "FAIL_IMPLEMENTATION", f"fail_closed; {repr(error)}; HOTA=null; next_policy_authorized=false")
    except Exception:
        registry_line = None
    snapshot = process_gpu_snapshot()
    write_json(
        R68 / "implementation_failure.json",
        {
            "experiment_id": EXP_ID,
            "status": "closed",
            "decision": "FAIL_IMPLEMENTATION",
            "error": repr(error),
            "traceback": traceback.format_exc(),
            "same_root_repair_performed": False,
            **FIXED,
        },
    )
    final = {
        "experiment_id": EXP_ID,
        "status": "failed",
        "current_stage": "closed",
        "decision": "FAIL_IMPLEMENTATION",
        "registry_line": registry_line,
        "process_gpu": snapshot,
        "scope_counts": SCOPE_COUNTS,
        "m23_69_started": False,
        **FIXED,
    }
    write_json(R68 / "final_summary.json", final)
    RESULT.write_text(
        f"# {EXP_NAME} — Fail-closed result\n\nDecision: FAIL_IMPLEMENTATION\n\nError: {repr(error)}\n\nNo training, tracker, TrackEval, HOTA, raw GT, or next-policy run occurred.\n\nNotion writeback was not executed.\n",
        encoding="utf-8",
    )
    append_event("experiment_failed_closed", error=repr(error))


def run_all() -> None:
    command_init()
    command_verify_inputs()
    command_reconstruct_labels()
    command_population_identity()
    command_decompose_population()
    command_score_capacity()
    command_diagnose()
    command_validate()
    command_summarize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["self-test", "preflight", "run"])
    args = parser.parse_args()
    if args.command == "self-test":
        print(json.dumps(self_test(), sort_keys=True))
        return
    if args.command == "preflight":
        occupied = R68.exists() or RESULT.exists()
        _, registry = registry_rows()
        registry_conflict = any(row.get("name") == EXP_ID or row.get("tag") == EXP_ID for row in registry)
        report = {
            "occupied": occupied,
            "registry_conflict": registry_conflict,
            "prereg_exists": PREREG.exists(),
            "test_exists": TEST_SCRIPT.exists(),
            "input_count": len(collect_input_paths()),
            "missing_inputs": [relative(path) for path in collect_input_paths() if not path.exists()],
            "predecessor_r2": predecessor_r2_check(),
            "process_gpu": process_gpu_snapshot(),
        }
        report["passed"] = bool(
            not occupied
            and not registry_conflict
            and report["prereg_exists"]
            and report["test_exists"]
            and not report["missing_inputs"]
            and report["predecessor_r2"]["all_passed"]
            and process_gpu_idle(report["process_gpu"])
        )
        print(json.dumps(report, indent=2, sort_keys=True, default=json_default))
        if not report["passed"]:
            raise SystemExit(1)
        return
    try:
        run_all()
    except BaseException as error:
        fail_close(error)
        raise


if __name__ == "__main__":
    main()
