#!/usr/bin/env python3
"""M23-69 boundary objective, checkpoint-selection, and gate alignment audit."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import inspect
import json
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
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/mot20_m23_20260718"
R63 = BASE / "m23_59_relation_pretrained_hierarchical_flow_v3_supervision_join"
R64 = BASE / "m23_59_relation_pretrained_hierarchical_flow_v3_pair_reconstruction_training_repair1"
R65 = BASE / "m23_59_relation_pretrained_hierarchical_flow_v3_mot20_representation_gate"
R66 = BASE / "m23_66_v3_metric_correctness_source_target_decomposition_audit"
R68 = BASE / "m23_68_r1_boundary_label_eligibility_population_decomposition_audit"
R69 = BASE / "m23_69_boundary_objective_gate_population_alignment_audit"
SCRIPT = ROOT / "scripts/m23_research/m23_69_boundary_objective_gate_population_alignment_audit.py"
TEST_SCRIPT = ROOT / "scripts/m23_research/test_m23_69_boundary_objective_gate_population_alignment_audit.py"
V2_SCRIPT = ROOT / "scripts/m23_research/m23_59_relation_pretrained_hierarchical_flow_v2.py"
M64_SCRIPT = ROOT / "scripts/m23_research/m23_64_v3_pair_reconstruction_training.py"
M64_REPAIR_SCRIPT = ROOT / "scripts/m23_research/m23_64_v3_pair_reconstruction_training_repair.py"
M65_SCRIPT = ROOT / "scripts/m23_research/m23_65_v3_mot20_representation_gate.py"
M66_SCRIPT = ROOT / "scripts/m23_research/m23_66_v3_metric_correctness_source_target_decomposition_audit.py"
M67_R2_SCRIPT = ROOT / "scripts/m23_research/m23_67_r2_source_boundary_failure_root_cause_audit.py"
M68_SCRIPT = ROOT / "scripts/m23_research/m23_68_boundary_label_eligibility_population_decomposition_audit.py"
M68_R1_SCRIPT = ROOT / "scripts/m23_research/m23_68_r1_boundary_label_eligibility_population_decomposition_audit.py"
PREREG = ROOT / "docs/m23_69_boundary_objective_gate_population_alignment_audit_prereg_20260725.md"
RESULT = ROOT / "docs/m23_69_boundary_objective_gate_population_alignment_audit_result_20260725.md"
REGISTRY = ROOT / "outputs/experiment_registry.csv"

EXP_ID = "M23-69"
EXP_NAME = "M23-69 Boundary Objective–Gate Population Alignment Audit"
TRAIN_SEQUENCES = ["MOT17-02", "MOT17-04", "MOT17-05", "MOT17-09", "MOT17-10"]
VALIDATION_SEQUENCES = ["MOT17-11", "MOT17-13"]
TARGET_SEQUENCES = ["MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05"]
ALL_SOURCE_SEQUENCES = TRAIN_SEQUENCES + VALIDATION_SEQUENCES
SEEDS = [2359001, 2359002, 2359003]
MAX_NODE_ROWS = 30
NODE_LABEL_MIN_KNOWN = 5
STAGES = [
    "init",
    "verify-inputs",
    "audit-objective-population",
    "inventory-checkpoints",
    "evaluate-retained-checkpoints",
    "audit-gate-score-semantics",
    "diagnose",
    "validate",
    "summarize",
    "closed",
]
SUMMARY_FIELDS = [
    "experiment_id",
    "stage",
    "status",
    "started_at",
    "finished_at",
    "decision",
    "error",
    "report",
    "wall_seconds",
    "peak_rss_kb",
    "notes",
]
GATE_THRESHOLDS = {
    "macro_pr_auc": 0.283,
    "macro_precision_at_actual": 0.35,
    "macro_recall_at_95_precision": 0.05,
    "minimum_sequence_precision_at_actual": 0.20,
}
DIAGNOSTIC_THRESHOLDS = {
    "maximum_conditional_row_coverage_for_material_mismatch": 0.50,
    "minimum_full_population_outside_conditional": 0.10,
    "minimum_positive_coverage_for_negative_support_diagnosis": 0.95,
    "minimum_train_validation_pr_auc_drop": 0.20,
    "maximum_validation_to_train_pr_auc_ratio": 0.50,
    "minimum_hierarchical_pr_auc_absolute_gain": 0.02,
}
FORBIDDEN_SCOPE_COUNTS = {
    "training_runs": 0,
    "optimizer_steps": 0,
    "checkpoint_outputs": 0,
    "checkpoint_modifications": 0,
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
}
EXPECTED_ALLOWED_COUNTS = {
    "retained_checkpoint_loads": 3,
    "retained_checkpoint_inference_runs": 3,
    "retained_checkpoint_validation_windows": 1281 * 3,
}
FIXED_STATUS = {
    "post_hoc_diagnostic_only": True,
    "uses_frozen_gt_derived_label_sidecars": True,
    "not_deployable": True,
    "not_a_strict_result": True,
    "training_authorized": False,
    "next_policy_authorized": False,
    "hota": None,
    "m23_70_started": False,
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_default(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def read_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=json_default) + "\n",
        encoding="utf-8",
    )


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True
    ).stdout.strip()


def parse_ids(value: Any) -> list[int]:
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, np.ndarray):
        value = value.tolist()
    return [int(item) for item in value]


def sigmoid(value: float | np.ndarray) -> float | np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    result = 1.0 / (1.0 + np.exp(-np.clip(array, -80.0, 80.0)))
    return float(result) if result.ndim == 0 else result


def logit_probability(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(value, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    return np.log(clipped) - np.log1p(-clipped)


_PROCESS_MODULE = None


def process_module():
    global _PROCESS_MODULE
    if _PROCESS_MODULE is None:
        specification = importlib.util.spec_from_file_location("m23_69_process_helper", M67_R2_SCRIPT)
        if specification is None or specification.loader is None:
            raise RuntimeError("cannot load M23-67-R2 process helper")
        module = importlib.util.module_from_spec(specification)
        sys.modules[specification.name] = module
        specification.loader.exec_module(module)
        _PROCESS_MODULE = module
    return _PROCESS_MODULE


def process_gpu_snapshot() -> dict[str, Any]:
    return process_module().process_gpu_snapshot(exclude_self=True)


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
    temporary = REGISTRY.with_suffix(".m23_69.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(REGISTRY)


def registry_start() -> int:
    fields, rows = registry_rows()
    if any(row.get("name") == EXP_ID or row.get("tag") == EXP_ID for row in rows):
        raise RuntimeError("M23-69 registry entry already exists")
    record = {field: "" for field in fields}
    record.update(
        {
            "timestamp": now(),
            "kind": "post_hoc_diagnostic",
            "status": "running",
            "script": relative(SCRIPT),
            "dataset": "MOT17+MOT20",
            "split": "frozen_sidecars",
            "tracker_family": "FM-Track/M23-59-v3",
            "variant": "boundary_objective_gate_population_alignment",
            "tag": EXP_ID,
            "run_root": relative(R69),
            "summary_csv": relative(R69 / "summary.csv"),
            "notes": "M23-69 initialized; diagnostic only; no training/tracker/TrackEval/HOTA",
            "name": EXP_ID,
            "current_stage": "init",
        }
    )
    rows.append(record)
    write_registry(fields, rows)
    return len(rows) + 1


def registry_stage(stage: str) -> None:
    fields, rows = registry_rows()
    matching = [index for index, row in enumerate(rows) if row.get("name") == EXP_ID or row.get("tag") == EXP_ID]
    if not matching:
        raise RuntimeError("M23-69 registry running entry missing")
    rows[matching[-1]]["current_stage"] = stage
    rows[matching[-1]]["timestamp"] = now()
    write_registry(fields, rows)


def registry_close(status: str, decision: str, notes: str) -> int:
    fields, rows = registry_rows()
    matching = [index for index, row in enumerate(rows) if row.get("name") == EXP_ID or row.get("tag") == EXP_ID]
    if not matching:
        raise RuntimeError("M23-69 registry running entry missing")
    index = matching[-1]
    rows[index].update(
        {
            "timestamp": now(),
            "status": status,
            "current_stage": "closed",
            "decision": decision,
            "notes": notes,
            "HOTA": "",
            "hota": "",
            "result": relative(RESULT),
        }
    )
    write_registry(fields, rows)
    return index + 2


def initialize_summary() -> None:
    rows = [{field: "" for field in SUMMARY_FIELDS} for _ in STAGES]
    for row, stage in zip(rows, STAGES):
        row.update(experiment_id=EXP_ID, stage=stage, status="pending")
    with (R69 / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def set_stage(stage: str, status: str, **updates) -> None:
    path = R69 / "summary.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    found = False
    for row in rows:
        if row["stage"] == stage:
            found = True
            row["status"] = status
            for key, value in updates.items():
                row[key] = (
                    json.dumps(value, sort_keys=True, default=json_default)
                    if key == "notes" and not isinstance(value, str)
                    else value
                )
    if not found:
        raise KeyError(stage)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def append_event(event: str, **updates) -> None:
    record = {"timestamp": now(), "experiment_id": EXP_ID, "event": event, **updates}
    with (R69 / "protocol_events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, default=json_default) + "\n")


def implementation_guard() -> dict[str, bool]:
    manifest = read_json(R69 / "implementation_manifest.json", {}) or {}
    checks = {
        "script": sha256(SCRIPT) == manifest.get("script_sha256"),
        "test": sha256(TEST_SCRIPT) == manifest.get("test_script_sha256"),
        "prereg": sha256(PREREG) == manifest.get("prereg_sha256"),
    }
    if not all(checks.values()):
        raise RuntimeError(f"implementation SHA guard failed: {checks}")
    return checks


def stage_started(stage: str) -> tuple[float, int]:
    implementation_guard()
    set_stage(stage, "running", started_at=now())
    registry_stage(stage)
    append_event("stage_started", stage=stage)
    rchar = int(Path("/proc/self/io").read_text().split("rchar:", 1)[1].splitlines()[0].strip())
    return time.perf_counter(), rchar


def stage_finished(stage: str, started: float, rchar_before: int, notes=None, report: str = "") -> None:
    current = int(Path("/proc/self/io").read_text().split("rchar:", 1)[1].splitlines()[0].strip())
    set_stage(
        stage,
        "completed",
        finished_at=now(),
        wall_seconds=time.perf_counter() - started,
        peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        report=report,
        notes={"rchar_delta": current - rchar_before, **(notes or {})},
    )
    append_event("stage_completed", stage=stage, report=report)


def precision_at_actual(labels: np.ndarray, scores: np.ndarray, keys: np.ndarray | None = None) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int((labels == 1).sum())
    if positives <= 0:
        return 0.0
    tie_keys = np.arange(len(scores)) if keys is None else np.asarray(keys).astype(str)
    order = np.lexsort((tie_keys, -scores))
    return float((labels[order[:positives]] == 1).mean())


def recall_at_precision(labels: np.ndarray, scores: np.ndarray, target: float) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    if int((labels == 1).sum()) == 0 or len(np.unique(labels)) < 2:
        return 0.0
    precision, recall, _ = precision_recall_curve(labels, scores)
    valid = recall[precision >= target]
    return float(valid.max()) if len(valid) else 0.0


def binary_metrics(labels: np.ndarray, scores: np.ndarray, keys: np.ndarray | None = None) -> dict[str, Any]:
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    keep = np.isin(labels, [0, 1]) & np.isfinite(scores)
    labels = labels[keep].astype(np.int8)
    scores = scores[keep]
    if keys is not None:
        keys = np.asarray(keys)[keep]
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    return {
        "rows": int(len(labels)),
        "positives": positives,
        "negatives": negatives,
        "base_rate": float(labels.mean()) if len(labels) else None,
        "pr_auc": float(average_precision_score(labels, scores)) if positives and negatives else None,
        "roc_auc": float(roc_auc_score(labels, scores)) if positives and negatives else None,
        "precision_at_actual": precision_at_actual(labels, scores, keys),
        "recall_at_95_precision": recall_at_precision(labels, scores, 0.95),
        "positive_score_mean": float(scores[labels == 1].mean()) if positives else None,
        "negative_score_mean": float(scores[labels == 0].mean()) if negatives else None,
    }


def aggregate_unique(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby(
        ["domain", "split", "sequence", "src_row_index", "dst_row_index", "transition_key"],
        sort=True,
    ).agg(
        native_label=("native_label", "first"),
        native_label_nunique=("native_label", "nunique"),
        strict_label=("strict_label", "first"),
        strict_label_nunique=("strict_label", "nunique"),
        mean_boundary_logit=("boundary_logit", "mean"),
        mean_joint_logit=("joint_logit", "mean"),
        mean_joint_probability=("joint_probability", "mean"),
        observation_count=("boundary_logit", "size"),
        conditional_observation_count=("conditional_population", "sum"),
        auxiliary_observation_count=("auxiliary_population", "sum"),
    ).reset_index()
    grouped["boundary_probability"] = sigmoid(grouped.mean_boundary_logit.to_numpy())
    grouped["joint_probability"] = sigmoid(grouped.mean_joint_logit.to_numpy())
    grouped["conditional_population"] = grouped.conditional_observation_count > 0
    grouped["auxiliary_population"] = grouped.auxiliary_observation_count > 0
    return grouped


def population_material_mismatch(conditional_rows: int, full_rows: int) -> bool:
    if full_rows <= 0:
        return False
    coverage = conditional_rows / full_rows
    outside = 1.0 - coverage
    return bool(
        coverage < DIAGNOSTIC_THRESHOLDS["maximum_conditional_row_coverage_for_material_mismatch"]
        and outside >= DIAGNOSTIC_THRESHOLDS["minimum_full_population_outside_conditional"]
    )


def counterfactual_composite(validation: dict[str, Any], boundary_pr_auc: float) -> float:
    node = float(validation["node"]["pr_auc"] or 0.0)
    relation = 0.5 * (
        float(validation["outgoing_successor_R_at_1_pairwise"])
        + float(validation["incoming_predecessor_R_at_1_pairwise"])
    )
    catastrophic = float(validation["catastrophic_false_link_rate"])
    return 0.30 * node + 0.35 * boundary_pr_auc + 0.25 * relation + 0.10 * (1.0 - catastrophic)


def rank_candidates(records: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    ordered = sorted(records, key=lambda row: (-float(row[metric]), int(row["seed"]), int(row["epoch"])))
    return [{**row, "rank": index + 1} for index, row in enumerate(ordered)]


def classify_diagnosis(
    hard_checks_passed: bool,
    material_population_mismatch: bool,
    retained_selection_changed: bool,
    source_generalization_failure: bool,
) -> str:
    if not hard_checks_passed:
        return "implementation_or_measurement_failure"
    if material_population_mismatch:
        return "boundary_objective_gate_population_mismatch"
    if retained_selection_changed:
        return "checkpoint_selection_population_mismatch"
    if source_generalization_failure:
        return "source_sequence_generalization_failure"
    return "no_fixed_failure_component_identified"


def self_test() -> dict[str, bool]:
    labels = np.asarray([[0, 1, -1], [0, 0, 0], [1, 0, 1]], dtype=np.int8)
    node_y = np.asarray([0, 1, -1], dtype=np.int8)
    valid = np.ones_like(labels, dtype=bool)
    focal = valid & (labels >= 0) & (node_y[:, None] == 0)
    boundary = np.asarray([-2.0, 2.0], dtype=np.float64)
    node = np.asarray([0.25, 0.5], dtype=np.float64)
    joint = sigmoid(boundary) * node
    frame = pd.DataFrame(
        {
            "domain": ["source", "source"],
            "split": ["validation", "validation"],
            "sequence": ["S", "S"],
            "src_row_index": [1, 1],
            "dst_row_index": [2, 2],
            "transition_key": ["S:1:2", "S:1:2"],
            "native_label": [1, 1],
            "strict_label": [1, 1],
            "boundary_logit": boundary,
            "joint_logit": logit_probability(joint),
            "joint_probability": joint,
            "conditional_population": [True, False],
            "auxiliary_population": [True, True],
        }
    )
    unique = aggregate_unique(frame)
    candidates = [
        {"seed": 2, "epoch": 1, "metric": 0.5},
        {"seed": 1, "epoch": 2, "metric": 0.5},
        {"seed": 3, "epoch": 1, "metric": 0.4},
    ]
    checks = {
        "impure_native_known_mask": focal.tolist() == [[True, True, False], [False, False, False], [False, False, False]],
        "joint_probability_chain": bool(np.allclose(joint, sigmoid(boundary) * node, atol=0, rtol=0)),
        "duplicate_mean_boundary_logit": abs(float(unique.iloc[0].boundary_probability) - 0.5) < 1e-12,
        "duplicate_conditional_any": bool(unique.iloc[0].conditional_population),
        "material_mismatch_true": population_material_mismatch(10, 100),
        "material_mismatch_false": not population_material_mismatch(80, 100),
        "ranking_tie_lower_seed": rank_candidates(candidates, "metric")[0]["seed"] == 1,
        "classification_priority": classify_diagnosis(True, True, True, True) == "boundary_objective_gate_population_mismatch",
    }
    if not all(checks.values()):
        raise AssertionError({key: value for key, value in checks.items() if not value})
    return checks


def manifest_record_path(record: dict[str, Any]) -> Path:
    candidate = record.get("absolute_path") or record.get("path")
    path = Path(str(candidate))
    return path if path.is_absolute() else ROOT / path


def collect_input_paths() -> list[Path]:
    paths = [
        SCRIPT,
        TEST_SCRIPT,
        PREREG,
        V2_SCRIPT,
        M64_SCRIPT,
        M64_REPAIR_SCRIPT,
        M65_SCRIPT,
        M66_SCRIPT,
        M67_R2_SCRIPT,
        M68_SCRIPT,
        M68_R1_SCRIPT,
        R63 / "source_windows.parquet",
        R63 / "row_supervision.parquet",
        R64 / "examples_train.npz",
        R64 / "examples_validation.npz",
        R64 / "node_examples_train.parquet",
        R64 / "node_examples_validation.parquet",
        R64 / "training_config.json",
        R64 / "training_data_adapter.json",
        R64 / "final_summary.json",
        R64 / "closure_validation.json",
        R64 / "frozen_checkpoint/checkpoint_manifest.json",
        R64 / "frozen_checkpoint/checkpoint_selection.json",
        R64 / "frozen_checkpoint/relation_v3_frozen.pt",
        R65 / "final_summary.json",
        R65 / "closure_validation.json",
        R65 / "representation_gate.json",
        R65 / "representation_metrics.json",
        R66 / "final_summary.json",
        R66 / "closure_validation.json",
        R66 / "boundary_metrics.json",
        R66 / "metric_definition_validation.json",
        R68 / "final_summary.json",
        R68 / "closure_validation.json",
        R68 / "independent_closure_validation.json",
        R68 / "artifact_sha256_manifest.json",
        R68 / "artifact_manifest_validation.json",
        R68 / "input_manifest.json",
        R68 / "reconstructed_boundary_observations.parquet",
        R68 / "unique_boundary_transitions.parquet",
        R68 / "population_decomposition.json",
        R68 / "score_capacity.json",
    ]
    for seed in SEEDS:
        root = R64 / "training" / f"seed_{seed}"
        paths.extend(
            [
                root / "best_checkpoint.pt",
                root / "checkpoint_manifest.json",
                root / "training_history.json",
                root / "metrics.jsonl",
            ]
        )
    for sequence in TARGET_SEQUENCES:
        root = R65 / sequence
        paths.extend(
            [
                root / "scores/node_scores.parquet",
                root / "scores/boundary_scores.parquet",
                root / "scores/score_manifest.json",
                root / "labels/row_labels.parquet",
                root / "topology/windows.parquet",
                root / "metrics.json",
            ]
        )
    for sequence in VALIDATION_SEQUENCES:
        root = R66 / "source_scores" / sequence
        paths.extend(
            [
                root / "node_scores.parquet",
                root / "boundary_scores.parquet",
                root / "score_manifest.json",
            ]
        )
    r68_manifest = read_json(R68 / "artifact_sha256_manifest.json", {}) or {}
    for record in r68_manifest.get("records", []):
        paths.append(manifest_record_path(record))
    r68_inputs = read_json(R68 / "input_manifest.json", {}) or {}
    for candidate in r68_inputs.get("sha256", {}):
        path = Path(candidate)
        paths.append(path if path.is_absolute() else ROOT / path)
    deduplicated = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved == REGISTRY.resolve() or R69.resolve() in resolved.parents:
            continue
        if resolved not in seen:
            deduplicated.append(resolved)
            seen.add(resolved)
    return sorted(deduplicated, key=lambda path: str(path))


def hash_paths(paths: list[Path]) -> dict[str, str]:
    missing = [relative(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing frozen inputs: {missing[:20]}")
    return {relative(path): sha256(path) for path in paths}


def verify_hash_map(expected: dict[str, str]) -> list[dict[str, Any]]:
    records = []
    for candidate, digest in sorted(expected.items()):
        path = Path(candidate)
        path = path if path.is_absolute() else ROOT / path
        actual = sha256(path) if path.is_file() else None
        records.append(
            {
                "path": relative(path),
                "expected": digest,
                "actual": actual,
                "match": actual == digest,
            }
        )
    return records


def predecessor_check() -> dict[str, Any]:
    r64_final = read_json(R64 / "final_summary.json", {}) or {}
    r64_closure = read_json(R64 / "closure_validation.json", {}) or {}
    r65_final = read_json(R65 / "final_summary.json", {}) or {}
    r65_closure = read_json(R65 / "closure_validation.json", {}) or {}
    r66_final = read_json(R66 / "final_summary.json", {}) or {}
    r66_closure = read_json(R66 / "closure_validation.json", {}) or {}
    r68_final = read_json(R68 / "final_summary.json", {}) or {}
    r68_closure = read_json(R68 / "closure_validation.json", {}) or {}
    r68_independent = read_json(R68 / "independent_closure_validation.json", {}) or {}
    r68_artifacts = read_json(R68 / "artifact_manifest_validation.json", {}) or {}
    r68_summary = pd.read_csv(R68 / "summary.csv", keep_default_na=False)
    selection = read_json(R64 / "frozen_checkpoint/checkpoint_selection.json", {}) or {}
    checkpoint_manifest = read_json(R64 / "frozen_checkpoint/checkpoint_manifest.json", {}) or {}
    expected_selected = R64 / "training/seed_2359003/best_checkpoint.pt"
    fields, registry = registry_rows()
    del fields
    r68_registry = [row for row in registry if row.get("name") == "M23-68-R1" or row.get("tag") == "M23-68-R1"]
    checks = {
        "r64_training_passed": r64_final.get("decision") == "PASS_V3_FROM_SCRATCH_RELATION_TRAINING"
        and r64_final.get("status") == "closed",
        "r64_closure_passed": r64_closure.get("closure_integrity_passed") is True,
        "r64_selected_checkpoint_exact": selection.get("selected_seed") == 2359003
        and selection.get("selected_epoch") == 19
        and sha256(expected_selected) == selection.get("source_checkpoint_sha256")
        and sha256(R64 / "frozen_checkpoint/relation_v3_frozen.pt") == checkpoint_manifest.get("checkpoint_sha256"),
        "r65_failed_gate_closed": r65_final.get("decision") == "FAIL_MOT20_REPRESENTATION_GATE"
        and r65_final.get("status") == "closed",
        "r65_closure_passed": r65_closure.get("closure_integrity_passed") is True,
        "r66_completed_closed": r66_final.get("status") == "completed"
        and r66_final.get("decision") == "COMPLETED_POST_HOC_DIAGNOSTIC"
        and r66_closure.get("status") == "closed",
        "r66_closure_passed": r66_closure.get("closure_integrity_passed") is True,
        "r68_completed_closed": r68_final.get("status") == "completed"
        and r68_final.get("current_stage") == "closed",
        "r68_closure_passed": r68_closure.get("closure_integrity_passed") is True,
        "r68_independent_passed": r68_independent.get("independent_closure_passed") is True,
        "r68_artifacts_match": r68_artifacts.get("all_match") is True,
        "r68_summary_no_stale": not r68_summary.status.astype(str).isin(["running", "pending"]).any(),
        "r68_registry_closed": bool(
            r68_registry
            and r68_registry[-1].get("status") == "completed"
            and r68_registry[-1].get("current_stage") == "closed"
        ),
        "m23_69_registry_unused": not any(
            row.get("name") == EXP_ID or row.get("tag") == EXP_ID for row in registry
        ),
    }
    return {"checked_at": now(), "checks": checks, "all_passed": all(checks.values())}


def scoped_git_status() -> list[str]:
    paths = [SCRIPT, TEST_SCRIPT, PREREG, RESULT, R69, REGISTRY]
    result = subprocess.run(
        ["git", "status", "--short", "--", *[relative(path) for path in paths]],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def command_preflight() -> dict[str, Any]:
    _, registry = registry_rows()
    snapshot = process_gpu_snapshot()
    checkpoint_files = sorted(
        path for path in (R64 / "training").rglob("*") if path.suffix.lower() in {".pt", ".pth", ".ckpt"}
    )
    report = {
        "run_root_unused": not R69.exists(),
        "result_path_unused": not RESULT.exists(),
        "registry_unused": not any(row.get("name") == EXP_ID or row.get("tag") == EXP_ID for row in registry),
        "process_gpu_idle": process_gpu_idle(snapshot),
        "predecessors": predecessor_check(),
        "retained_training_checkpoint_files": [relative(path) for path in checkpoint_files],
        "retained_training_checkpoint_count": len(checkpoint_files),
    }
    report["all_passed"] = bool(
        report["run_root_unused"]
        and report["result_path_unused"]
        and report["registry_unused"]
        and report["process_gpu_idle"]
        and report["predecessors"]["all_passed"]
        and report["retained_training_checkpoint_count"] == 3
    )
    return report


def command_init() -> None:
    if not PREREG.is_file() or not TEST_SCRIPT.is_file():
        raise RuntimeError("M23-69 preregistration or test script is missing")
    preflight = command_preflight()
    if not preflight["all_passed"]:
        raise RuntimeError(f"M23-69 preflight failed: {preflight}")
    checks = self_test()
    process_checks = process_module().process_classifier_self_test()
    if not all(process_checks.values()):
        raise RuntimeError(f"process classifier self-test failed: {process_checks}")
    R69.mkdir(parents=True)
    initialize_summary()
    (R69 / "protocol_events.jsonl").touch()
    set_stage("init", "running", started_at=now())
    started = time.perf_counter()
    registry_line = registry_start()
    frozen = hash_paths(collect_input_paths())
    write_json(
        R69 / "input_manifest.json",
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
        R69 / "preregistration.json",
        {
            "experiment_id": EXP_ID,
            "created_at": now(),
            "user_explicit_authorization": True,
            "source_train_sequences": TRAIN_SEQUENCES,
            "source_validation_sequences": VALIDATION_SEQUENCES,
            "target_sequences": TARGET_SEQUENCES,
            "gate_thresholds": GATE_THRESHOLDS,
            "diagnostic_thresholds": DIAGNOSTIC_THRESHOLDS,
            "retained_checkpoint_scope": "three persisted seed-best checkpoints only",
            "all_epoch_counterfactual": "unavailable unless all 90 epoch weights are present",
            "classification_priority": [
                "implementation_or_measurement_failure",
                "boundary_objective_gate_population_mismatch",
                "checkpoint_selection_population_mismatch",
                "source_sequence_generalization_failure",
                "no_fixed_failure_component_identified",
            ],
            "hierarchical_score": "sigmoid(node_impurity_logit) * sigmoid(conditional_boundary_logit)",
            "unique_transition_aggregation": "sigmoid(mean(logit(score)))",
            "primary_label_definition": "M23-63-native matched endpoints",
            "strict_label_sensitivity": "exclude ambiguity, tie, distractor, or nonmatched endpoints",
            "forbidden_scope_counts": FORBIDDEN_SCOPE_COUNTS,
            "expected_allowed_counts": EXPECTED_ALLOWED_COUNTS,
            **FIXED_STATUS,
        },
    )
    implementation = {
        "experiment_id": EXP_ID,
        "created_at": now(),
        "git_head": git_head(),
        "script_sha256": sha256(SCRIPT),
        "test_script_sha256": sha256(TEST_SCRIPT),
        "prereg_sha256": sha256(PREREG),
        "v2_script_sha256": sha256(V2_SCRIPT),
        "m64_script_sha256": sha256(M64_SCRIPT),
        "m64_repair_script_sha256": sha256(M64_REPAIR_SCRIPT),
        "m65_script_sha256": sha256(M65_SCRIPT),
        "process_helper_sha256": sha256(M67_R2_SCRIPT),
        "self_test_checks": checks,
        "process_classifier_checks": process_checks,
        "implementation_frozen": True,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
        },
    }
    write_json(R69 / "implementation_manifest.json", implementation)
    write_json(R69 / "preflight.json", preflight)
    write_json(R69 / "predecessor_reverification.json", predecessor_check())
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
        notes={"registry_line": registry_line, "input_count": len(frozen)},
    )


def command_verify_inputs() -> None:
    started, rchar = stage_started("verify-inputs")
    manifest = read_json(R69 / "input_manifest.json", {}) or {}
    records = verify_hash_map(manifest.get("sha256", {}))
    predecessor = predecessor_check()
    report = {
        "experiment_id": EXP_ID,
        "checked_at": now(),
        "records": records,
        "inputs_unchanged": bool(records) and all(record["match"] for record in records),
        "predecessors": predecessor,
        "all_passed": bool(records) and all(record["match"] for record in records) and predecessor["all_passed"],
        "first_mismatch": next((record for record in records if not record["match"]), None),
    }
    write_json(R69 / "input_reverification.json", report)
    if not report["all_passed"]:
        raise RuntimeError(f"input reverification failed: {report['first_mismatch']}")
    stage_finished(
        "verify-inputs",
        started,
        rchar,
        {"input_count": len(records)},
        relative(R69 / "input_reverification.json"),
    )


_V2_MODULE = None


def v2_module():
    global _V2_MODULE
    if _V2_MODULE is None:
        specification = importlib.util.spec_from_file_location("m23_69_v2_model_source", V2_SCRIPT)
        if specification is None or specification.loader is None:
            raise RuntimeError("cannot load frozen v2 model source")
        module = importlib.util.module_from_spec(specification)
        sys.modules[specification.name] = module
        specification.loader.exec_module(module)
        _V2_MODULE = module
    return _V2_MODULE


def source_contract_checks() -> dict[str, bool]:
    v2_text = V2_SCRIPT.read_text(encoding="utf-8")
    m64_text = M64_SCRIPT.read_text(encoding="utf-8")
    m65_text = M65_SCRIPT.read_text(encoding="utf-8")
    configuration = read_json(R64 / "training_config.json", {}) or {}
    module = v2_module()
    function_hashes = {
        name: hashlib.sha256(inspect.getsource(getattr(module, name)).encode("utf-8")).hexdigest()
        for name in ["training_objective", "validation_metrics"]
    }
    frozen_hashes = configuration.get("source_function_sha256", {})
    return {
        "training_focal_requires_valid": "valid_boundary = (bv > 0)" in v2_text,
        "training_focal_requires_known_label": "& (by >= 0)" in v2_text,
        "training_focal_requires_positive_adapted_node": "& (ny[:, None] > 0)" in v2_text,
        "validation_conditional_requires_positive_adapted_node": "conditional = (np.asarray(data[\"node_label\"][ids])[:, None] > 0)" in v2_text,
        "adapter_inverts_native_node_label": 'node_label=(1-arrays["node_y"][keep]).astype(np.int8)' in m64_text,
        "adapter_excludes_unknown_node_windows": 'keep=np.flatnonzero(arrays["node_y"]>=0)' in m64_text,
        "boundary_head_is_named_conditional": "self.conditional_boundary_head" in v2_text,
        "node_head_is_named_impurity": "self.node_impurity_head" in v2_text,
        "count_consistency_uses_all_valid_positions": "(torch.sigmoid(bl) * bv).sum(dim=1)" in v2_text,
        "count_consistency_clamps_unknown_to_zero": "by.clamp_min(0).sum(dim=1)" in v2_text,
        "sparsity_uses_all_valid_positions": "sparsity = (torch.sigmoid(bl) * bv).mean()" in v2_text,
        "m65_gate_publishes_raw_boundary_probability": "by.append((int(a.gt_identity_key!=b.gt_identity_key),float(r.boundary_probability)))" in m65_text,
        "m65_gate_does_not_multiply_node_probability": "node_probability *" not in m65_text
        and "node_probability)*" not in m65_text,
        "training_objective_hash_matches_config": function_hashes["training_objective"]
        == frozen_hashes.get("training_objective"),
        "validation_metrics_hash_matches_config": function_hashes["validation_metrics"]
        == frozen_hashes.get("validation_metrics"),
    }


def source_population_frame() -> pd.DataFrame:
    frame = pd.read_parquet(R68 / "reconstructed_boundary_observations.parquet").copy()
    required = {
        "split",
        "sequence",
        "window_id",
        "tensor_index",
        "position",
        "node_label",
        "src_row_index",
        "dst_row_index",
        "transition_key",
        "m23_63_native_label",
        "m23_66_strict_label",
        "boundary_logit",
        "boundary_probability",
    }
    if not required.issubset(frame.columns):
        raise RuntimeError(f"R68 source observation columns missing: {sorted(required - set(frame.columns))}")
    frame = frame.rename(
        columns={
            "m23_63_native_label": "native_label",
            "m23_66_strict_label": "strict_label",
        }
    )
    frame["domain"] = "source"
    frame["node_known_window"] = frame.node_label >= 0
    frame["impure_window"] = frame.node_label == 0
    frame["pure_window"] = frame.node_label == 1
    frame["conditional_population"] = frame.impure_window
    frame["auxiliary_population"] = frame.node_known_window
    frame["native_known"] = frame.native_label.isin([0, 1])
    frame["strict_known"] = frame.strict_label.isin([0, 1])
    frame["boundary_focal_eligible"] = frame.conditional_population & frame.native_known
    frame["checkpoint_selection_eligible"] = frame.boundary_focal_eligible
    frame["auxiliary_unknown_target_clamped_zero"] = frame.auxiliary_population & ~frame.native_known
    frame["joint_probability"] = frame.boundary_probability.astype(np.float64)
    frame["joint_logit"] = frame.boundary_logit.astype(np.float64)
    return frame


def direct_array_population(split: str) -> dict[str, Any]:
    with np.load(R64 / f"examples_{split}.npz", allow_pickle=False) as archive:
        node_y = np.asarray(archive["node_y"], dtype=np.int8)
        boundary_y = np.asarray(archive["boundary_y"], dtype=np.int8)
        node_mask = np.asarray(archive["node_mask"], dtype=np.uint8)
    valid = (node_mask[:, :-1] > 0) & (node_mask[:, 1:] > 0)
    native_known = np.isin(boundary_y, [0, 1]) & valid
    focal = native_known & (node_y[:, None] == 0)
    auxiliary = valid & (node_y[:, None] >= 0)
    pure = native_known & (node_y[:, None] == 1)
    unknown_node = native_known & (node_y[:, None] < 0)
    return {
        "split": split,
        "windows": int(len(node_y)),
        "node_label_counts": {str(int(value)): int((node_y == value).sum()) for value in np.unique(node_y)},
        "valid_observations": int(valid.sum()),
        "native_known_observations": int(native_known.sum()),
        "native_positive_observations": int(((boundary_y == 1) & valid).sum()),
        "boundary_focal_rows": int(focal.sum()),
        "boundary_focal_positives": int(((boundary_y == 1) & focal).sum()),
        "auxiliary_valid_rows": int(auxiliary.sum()),
        "auxiliary_unknown_labels_clamped_zero": int((auxiliary & ~np.isin(boundary_y, [0, 1])).sum()),
        "pure_known_rows": int(pure.sum()),
        "pure_positive_rows": int(((boundary_y == 1) & pure).sum()),
        "unknown_node_known_rows": int(unknown_node.sum()),
        "node_y": node_y,
        "boundary_y": boundary_y,
        "node_mask": node_mask,
    }


def population_summary(
    frame: pd.DataFrame,
    label_column: str,
    population: str,
    view: str,
    split: str,
    sequence: str,
) -> dict[str, Any]:
    data = frame
    if split != "all":
        data = data[data.split == split]
    if sequence != "all":
        data = data[data.sequence == sequence]
    if population == "conditional":
        data = data[data.conditional_population]
    elif population == "auxiliary":
        data = data[data.auxiliary_population]
    elif population == "pure":
        data = data[data.pure_window if "pure_window" in data else ~data.conditional_population]
    elif population == "node_unknown":
        data = data[~data.auxiliary_population]
    labels = data[label_column].to_numpy(np.int8)
    known = np.isin(labels, [0, 1])
    return {
        "definition": label_column.removesuffix("_label"),
        "population": population,
        "view": view,
        "split": split,
        "sequence": sequence,
        "rows": int(known.sum()),
        "positives": int((labels[known] == 1).sum()),
        "negatives": int((labels[known] == 0).sum()),
        "positive_rate": float(labels[known].mean()) if known.any() else None,
    }


def selected_source_metrics(frame: pd.DataFrame, unique: pd.DataFrame) -> dict[str, Any]:
    records = []
    for split, sequences in [("train", TRAIN_SEQUENCES), ("validation", VALIDATION_SEQUENCES)]:
        for definition in ["native", "strict"]:
            label = f"{definition}_label"
            for population in ["conditional", "full"]:
                raw = frame[frame.split == split]
                if population == "conditional":
                    raw = raw[raw.conditional_population]
                raw_metrics = binary_metrics(
                    raw[label].to_numpy(),
                    raw.boundary_probability.to_numpy(),
                    raw.index.to_numpy(),
                )
                records.append(
                    {
                        "split": split,
                        "scope": "pooled",
                        "definition": definition,
                        "population": population,
                        "view": "raw_observation",
                        **raw_metrics,
                    }
                )
                chosen_unique = unique[unique.split == split]
                if population == "conditional":
                    chosen_unique = chosen_unique[chosen_unique.conditional_population]
                unique_metrics = binary_metrics(
                    chosen_unique[label].to_numpy(),
                    chosen_unique.boundary_probability.to_numpy(),
                    chosen_unique.transition_key.to_numpy(),
                )
                records.append(
                    {
                        "split": split,
                        "scope": "pooled",
                        "definition": definition,
                        "population": population,
                        "view": "unique_transition",
                        **unique_metrics,
                    }
                )
                for sequence in sequences:
                    sequence_unique = chosen_unique[chosen_unique.sequence == sequence]
                    sequence_metrics = binary_metrics(
                        sequence_unique[label].to_numpy(),
                        sequence_unique.boundary_probability.to_numpy(),
                        sequence_unique.transition_key.to_numpy(),
                    )
                    records.append(
                        {
                            "split": split,
                            "scope": sequence,
                            "definition": definition,
                            "population": population,
                            "view": "unique_transition",
                            **sequence_metrics,
                        }
                    )
    validation_manifest = read_json(R64 / "frozen_checkpoint/checkpoint_manifest.json", {}) or {}
    historical = validation_manifest.get("validation_metrics", {}).get("conditional_boundary", {})
    reconstructed = next(
        row
        for row in records
        if row["split"] == "validation"
        and row["scope"] == "pooled"
        and row["definition"] == "native"
        and row["population"] == "conditional"
        and row["view"] == "raw_observation"
    )
    reproduction = {
        "rows": reconstructed["rows"] == historical.get("rows"),
        "positives": reconstructed["positives"] == historical.get("positives"),
        "pr_auc": abs(float(reconstructed["pr_auc"]) - float(historical.get("pr_auc"))) <= 1e-12,
        "precision_at_actual": abs(float(reconstructed["precision_at_actual"]) - float(historical.get("precision_at_actual_count"))) <= 1e-12,
        "recall_at_95_precision": abs(float(reconstructed["recall_at_95_precision"]) - float(historical.get("recall_at_95_precision"))) <= 1e-12,
    }
    train = next(
        row
        for row in records
        if row["split"] == "train"
        and row["scope"] == "pooled"
        and row["definition"] == "native"
        and row["population"] == "conditional"
        and row["view"] == "raw_observation"
    )
    validation = reconstructed
    ratio = None if not train["pr_auc"] else float(validation["pr_auc"] / train["pr_auc"])
    drop = None if train["pr_auc"] is None or validation["pr_auc"] is None else float(train["pr_auc"] - validation["pr_auc"])
    generalization_failure = bool(
        drop is not None
        and ratio is not None
        and drop >= DIAGNOSTIC_THRESHOLDS["minimum_train_validation_pr_auc_drop"]
        and ratio <= DIAGNOSTIC_THRESHOLDS["maximum_validation_to_train_pr_auc_ratio"]
    )
    return {
        "records": records,
        "historical_conditional_reproduction": reproduction,
        "historical_conditional_reproduction_passed": all(reproduction.values()),
        "conditional_native_train_to_validation": {
            "train_pr_auc": train["pr_auc"],
            "validation_pr_auc": validation["pr_auc"],
            "absolute_drop": drop,
            "validation_to_train_ratio": ratio,
            "fixed_generalization_failure": generalization_failure,
        },
    }


def command_audit_objective_population() -> None:
    started, rchar = stage_started("audit-objective-population")
    frame = source_population_frame()
    direct = {split: direct_array_population(split) for split in ["train", "validation"]}
    checks = source_contract_checks()
    for split in ["train", "validation"]:
        selected = frame[frame.split == split]
        tensor_index = selected.tensor_index.to_numpy(np.int64)
        position = selected.position.to_numpy(np.int64)
        node_y = direct[split]["node_y"]
        boundary_y = direct[split]["boundary_y"]
        mask = direct[split]["node_mask"]
        checks[f"{split}_observation_count_exact"] = len(selected) == direct[split]["valid_observations"]
        checks[f"{split}_node_label_exact"] = bool(
            np.array_equal(selected.node_label.to_numpy(np.int8), node_y[tensor_index])
        )
        checks[f"{split}_native_boundary_label_exact"] = bool(
            np.array_equal(selected.native_label.to_numpy(np.int8), boundary_y[tensor_index, position])
        )
        checks[f"{split}_adjacent_mask_valid"] = bool(
            np.all(mask[tensor_index, position] > 0) and np.all(mask[tensor_index, position + 1] > 0)
        )
        checks[f"{split}_focal_count_exact"] = int(selected.boundary_focal_eligible.sum()) == direct[split]["boundary_focal_rows"]
        checks[f"{split}_focal_positive_exact"] = int(
            ((selected.native_label == 1) & selected.boundary_focal_eligible).sum()
        ) == direct[split]["boundary_focal_positives"]
        checks[f"{split}_auxiliary_count_exact"] = int(selected.auxiliary_population.sum()) == direct[split]["auxiliary_valid_rows"]
        checks[f"{split}_auxiliary_unknown_exact"] = int(
            selected.auxiliary_unknown_target_clamped_zero.sum()
        ) == direct[split]["auxiliary_unknown_labels_clamped_zero"]
        checks[f"{split}_optimizer_selection_population_identical"] = bool(
            (selected.boundary_focal_eligible == selected.checkpoint_selection_eligible).all()
        )
    unique = aggregate_unique(frame)
    checks["native_duplicate_labels_consistent"] = bool((unique.native_label_nunique == 1).all())
    checks["strict_duplicate_labels_consistent"] = bool((unique.strict_label_nunique == 1).all())
    unique.to_parquet(R69 / "source_unique_transition_population.parquet", index=False)
    summary_rows = []
    for definition in ["native", "strict"]:
        label = f"{definition}_label"
        for view, data in [("raw_observation", frame), ("unique_transition", unique)]:
            for population in ["full", "conditional", "auxiliary", "pure", "node_unknown"]:
                for split, sequences in [
                    ("train", TRAIN_SEQUENCES),
                    ("validation", VALIDATION_SEQUENCES),
                    ("all", ALL_SOURCE_SEQUENCES),
                ]:
                    summary_rows.append(population_summary(data, label, population, view, split, "all"))
                    if split != "all":
                        for sequence in sequences:
                            summary_rows.append(population_summary(data, label, population, view, split, sequence))
    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(R69 / "objective_population.csv", index=False)
    selected_metrics = selected_source_metrics(frame, unique)
    pd.DataFrame(selected_metrics["records"]).to_csv(R69 / "selected_source_population_metrics.csv", index=False)
    checks["historical_conditional_metrics_reproduced"] = selected_metrics[
        "historical_conditional_reproduction_passed"
    ]
    split_alignment = {}
    material_splits = []
    for split in ["train", "validation"]:
        full = frame[(frame.split == split) & frame.native_known]
        conditional = full[full.conditional_population]
        full_positive = int((full.native_label == 1).sum())
        conditional_positive = int((conditional.native_label == 1).sum())
        row_coverage = len(conditional) / len(full)
        positive_coverage = conditional_positive / full_positive if full_positive else None
        negative_coverage = int((conditional.native_label == 0).sum()) / max(int((full.native_label == 0).sum()), 1)
        material = population_material_mismatch(len(conditional), len(full))
        material_splits.append(material)
        split_alignment[split] = {
            "full_native_known_rows": int(len(full)),
            "conditional_native_known_rows": int(len(conditional)),
            "conditional_row_coverage": float(row_coverage),
            "full_native_positives": full_positive,
            "conditional_native_positives": conditional_positive,
            "conditional_positive_coverage": positive_coverage,
            "conditional_negative_coverage": float(negative_coverage),
            "material_population_mismatch": material,
        }
    positive_support_diagnosis = all(
        record["conditional_positive_coverage"] is not None
        and record["conditional_positive_coverage"]
        >= DIAGNOSTIC_THRESHOLDS["minimum_positive_coverage_for_negative_support_diagnosis"]
        for record in split_alignment.values()
    )
    hard_checks = all(checks.values())
    report = {
        "experiment_id": EXP_ID,
        "created_at": now(),
        "checks": checks,
        "hard_checks_passed": hard_checks,
        "direct_array_population": {
            split: {key: value for key, value in values.items() if not isinstance(value, np.ndarray)}
            for split, values in direct.items()
        },
        "split_alignment": split_alignment,
        "material_population_mismatch": bool(all(material_splits)),
        "positive_support_diagnosis": positive_support_diagnosis,
        "interpretation": (
            "The conditional focal/checkpoint population contains nearly all observed positives but only a small "
            "fraction of full-population negatives; count consistency and sparsity see a broader population, while "
            "unknown boundary labels are clamped to zero in the count target."
        ),
        "selected_checkpoint_performance": selected_metrics,
        "diagnostic_thresholds": DIAGNOSTIC_THRESHOLDS,
    }
    write_json(R69 / "objective_population.json", report)
    if not hard_checks:
        raise RuntimeError(f"objective population hard checks failed: {checks}")
    append_event(
        "objective_population_audited",
        material_population_mismatch=report["material_population_mismatch"],
        split_alignment=split_alignment,
    )
    stage_finished(
        "audit-objective-population",
        started,
        rchar,
        {"material_population_mismatch": report["material_population_mismatch"]},
        relative(R69 / "objective_population.json"),
    )


def command_inventory_checkpoints() -> None:
    started, rchar = stage_started("inventory-checkpoints")
    training_files = sorted(
        path for path in (R64 / "training").rglob("*") if path.suffix.lower() in {".pt", ".pth", ".ckpt"}
    )
    all_weight_files = sorted(
        path for path in R64.rglob("*") if path.suffix.lower() in {".pt", ".pth", ".ckpt"}
    )
    selection = read_json(R64 / "frozen_checkpoint/checkpoint_selection.json", {}) or {}
    records = []
    history_count = 0
    checks = {}
    for seed in SEEDS:
        root = R64 / "training" / f"seed_{seed}"
        manifest = read_json(root / "checkpoint_manifest.json", {}) or {}
        history = read_json(root / "training_history.json", []) or []
        metric_rows = [
            json.loads(line)
            for line in (root / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        checkpoint = root / "best_checkpoint.pt"
        history_count += len(history)
        checks[f"seed_{seed}_history_30"] = len(history) == 30 and len(metric_rows) == 30
        checks[f"seed_{seed}_all_finite"] = all(bool(row.get("finite")) for row in metric_rows)
        checks[f"seed_{seed}_manifest_completed"] = manifest.get("status") == "completed"
        checks[f"seed_{seed}_checkpoint_sha"] = sha256(checkpoint) == manifest.get("checkpoint_sha256")
        checks[f"seed_{seed}_single_final_selection"] = sum(
            int(bool(row.get("checkpoint_selected_final"))) for row in metric_rows
        ) == 1
        records.append(
            {
                "seed": seed,
                "epoch": int(manifest.get("best_epoch")),
                "historical_composite": float(manifest.get("best_composite")),
                "checkpoint": relative(checkpoint),
                "checkpoint_sha256": sha256(checkpoint),
                "history_rows": len(history),
                "metric_rows": len(metric_rows),
            }
        )
    expected_training = {
        (R64 / "training" / f"seed_{seed}" / "best_checkpoint.pt").resolve() for seed in SEEDS
    }
    checks["exact_three_training_weight_files"] = {path.resolve() for path in training_files} == expected_training
    checks["frozen_checkpoint_is_selected_duplicate"] = sha256(
        R64 / "frozen_checkpoint/relation_v3_frozen.pt"
    ) == sha256(R64 / "training/seed_2359003/best_checkpoint.pt")
    checks["selection_candidates_exact"] = {
        (int(record["seed"]), int(record["epoch"])) for record in selection.get("candidates", [])
    } == {(record["seed"], record["epoch"]) for record in records}
    checks["ninety_metric_records"] = history_count == 90
    unique_weight_shas = {sha256(path) for path in all_weight_files}
    all_epoch_available = len(unique_weight_shas) >= 90
    report = {
        "experiment_id": EXP_ID,
        "created_at": now(),
        "checks": checks,
        "hard_checks_passed": all(checks.values()),
        "retained_seed_best_checkpoints": records,
        "training_weight_files": [relative(path) for path in training_files],
        "all_weight_files_including_frozen_duplicate": [relative(path) for path in all_weight_files],
        "unique_weight_sha_count": len(unique_weight_shas),
        "historical_epoch_metric_records": history_count,
        "historical_epoch_weight_records": len(unique_weight_shas),
        "all_epoch_counterfactual_available": all_epoch_available,
        "all_epoch_counterfactual_status": (
            "available" if all_epoch_available else "unavailable_only_seed_best_weights_retained"
        ),
        "counterfactual_scope": "three retained seed-best checkpoints",
        "original_selected_seed": selection.get("selected_seed"),
        "original_selected_epoch": selection.get("selected_epoch"),
    }
    write_json(R69 / "checkpoint_inventory.json", report)
    if not report["hard_checks_passed"]:
        raise RuntimeError(f"checkpoint inventory hard checks failed: {checks}")
    append_event(
        "checkpoint_inventory_frozen",
        retained_checkpoint_count=len(records),
        all_epoch_counterfactual_available=all_epoch_available,
    )
    stage_finished(
        "inventory-checkpoints",
        started,
        rchar,
        {"retained_checkpoint_count": len(records), "all_epoch_counterfactual_available": all_epoch_available},
        relative(R69 / "checkpoint_inventory.json"),
    )


def infer_node_boundary(model: torch.nn.Module, node_x: np.ndarray, node_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    module = v2_module()
    node_parts = []
    boundary_parts = []
    valid_parts = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(node_x), 128):
            stop = min(start + 128, len(node_x))
            x = torch.as_tensor(np.asarray(node_x[start:stop]), dtype=torch.float32, device="cpu")
            mask = torch.as_tensor(np.asarray(node_mask[start:stop]), dtype=torch.float32, device="cpu")
            node_logit, boundary_logit, valid = model.node_and_boundary(x, mask)
            node_parts.append(node_logit.detach().cpu().numpy())
            boundary_parts.append(boundary_logit.detach().cpu().numpy())
            valid_parts.append(valid.detach().cpu().numpy())
    node = np.concatenate(node_parts).astype(np.float64)
    boundary = np.concatenate(boundary_parts).astype(np.float64)
    valid = np.concatenate(valid_parts).astype(np.uint8)
    if not np.isfinite(node).all() or not np.isfinite(boundary).all():
        raise RuntimeError("nonfinite retained checkpoint inference")
    if boundary.shape != (len(node_x), MAX_NODE_ROWS - 1):
        raise RuntimeError(f"unexpected boundary shape: {boundary.shape}")
    if module.parameter_count(model) != 881124:
        raise RuntimeError("retained checkpoint parameter count mismatch")
    return node, boundary, valid


def metric_records_for_checkpoint(frame: pd.DataFrame) -> list[dict[str, Any]]:
    unique = aggregate_unique(frame)
    records = []
    score_columns = {
        "raw_boundary": "boundary_probability",
        "hierarchical_joint": "joint_probability",
    }
    for definition in ["native", "strict"]:
        label = f"{definition}_label"
        for population in ["conditional", "full"]:
            raw = frame if population == "full" else frame[frame.conditional_population]
            chosen_unique = unique if population == "full" else unique[unique.conditional_population]
            for score_name, score_column in score_columns.items():
                for view, data in [("raw_observation", raw), ("unique_transition", chosen_unique)]:
                    for scope in ["pooled", *VALIDATION_SEQUENCES]:
                        selected = data if scope == "pooled" else data[data.sequence == scope]
                        metrics = binary_metrics(
                            selected[label].to_numpy(),
                            selected[score_column].to_numpy(),
                            selected.index.to_numpy() if view == "raw_observation" else selected.transition_key.to_numpy(),
                        )
                        records.append(
                            {
                                "definition": definition,
                                "population": population,
                                "score": score_name,
                                "view": view,
                                "scope": scope,
                                **metrics,
                            }
                        )
    return records


def find_metric(
    records: list[dict[str, Any]],
    *,
    definition: str,
    population: str,
    score: str,
    view: str,
    scope: str,
) -> dict[str, Any]:
    for record in records:
        if (
            record["definition"] == definition
            and record["population"] == population
            and record["score"] == score
            and record["view"] == view
            and record["scope"] == scope
        ):
            return record
    raise KeyError((definition, population, score, view, scope))


def macro_from_records(
    records: list[dict[str, Any]],
    *,
    definition: str,
    population: str,
    score: str,
    view: str,
) -> dict[str, Any]:
    selected = [
        find_metric(
            records,
            definition=definition,
            population=population,
            score=score,
            view=view,
            scope=sequence,
        )
        for sequence in VALIDATION_SEQUENCES
    ]
    macro = {
        metric: float(np.mean([record[metric] for record in selected]))
        for metric in ["pr_auc", "precision_at_actual", "recall_at_95_precision"]
    }
    minimum = float(min(record["precision_at_actual"] for record in selected))
    passed = bool(
        macro["pr_auc"] >= GATE_THRESHOLDS["macro_pr_auc"]
        and macro["precision_at_actual"] >= GATE_THRESHOLDS["macro_precision_at_actual"]
        and macro["recall_at_95_precision"] >= GATE_THRESHOLDS["macro_recall_at_95_precision"]
        and minimum >= GATE_THRESHOLDS["minimum_sequence_precision_at_actual"]
    )
    return {
        "macro": macro,
        "minimum_sequence_precision_at_actual": minimum,
        "passed": passed,
        "per_sequence": {sequence: record for sequence, record in zip(VALIDATION_SEQUENCES, selected)},
    }


def selected_frozen_node_lookup() -> pd.DataFrame:
    frames = []
    for sequence in VALIDATION_SEQUENCES:
        frame = pd.read_parquet(R66 / "source_scores" / sequence / "node_scores.parquet")
        frames.append(frame[["sequence", "window_id", "node_logit", "node_probability"]])
    result = pd.concat(frames, ignore_index=True)
    if result.duplicated(["sequence", "window_id"]).any():
        raise RuntimeError("selected frozen node score key duplicate")
    return result


def command_evaluate_retained_checkpoints() -> None:
    started, rchar = stage_started("evaluate-retained-checkpoints")
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    with np.load(R64 / "examples_validation.npz", allow_pickle=False) as archive:
        node_x = np.asarray(archive["node_x"], dtype=np.float16)
        node_mask = np.asarray(archive["node_mask"], dtype=np.uint8)
    base_frame = source_population_frame()
    base_frame = base_frame[base_frame.split == "validation"].copy()
    base_frame = base_frame.sort_values(["tensor_index", "position"], kind="mergesort").reset_index(drop=True)
    expected_valid = (node_mask[:, :-1] > 0) & (node_mask[:, 1:] > 0)
    selected_nodes = selected_frozen_node_lookup()
    metadata = pd.read_parquet(R64 / "node_examples_validation.parquet")[
        ["sequence", "window_id", "tensor_index"]
    ]
    metadata = metadata.sort_values("tensor_index", kind="mergesort").reset_index(drop=True)
    selected_nodes = metadata.merge(
        selected_nodes,
        on=["sequence", "window_id"],
        how="left",
        validate="one_to_one",
    )
    if selected_nodes.node_logit.isna().any():
        raise RuntimeError("selected frozen node lookup incomplete")
    inventory = read_json(R69 / "checkpoint_inventory.json", {}) or {}
    score_frames = []
    metric_rows = []
    candidate_rows = []
    checks = {
        "validation_window_count_exact": len(node_x) == 1281,
        "base_observation_count_exact": len(base_frame) == int(expected_valid.sum()),
    }
    max_selected_boundary_difference = None
    max_selected_node_difference = None
    for record in inventory.get("retained_seed_best_checkpoints", []):
        seed = int(record["seed"])
        checkpoint_path = ROOT / record["checkpoint"]
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checks[f"seed_{seed}_metadata"] = int(state.get("seed")) == seed and int(state.get("epoch")) == int(record["epoch"])
        model = v2_module().HierarchicalRelationEncoder().to(torch.device("cpu"))
        model.load_state_dict(state["model"], strict=True)
        node_logit, boundary_logit, valid = infer_node_boundary(model, node_x, node_mask)
        checks[f"seed_{seed}_valid_mask_exact"] = bool(np.array_equal(valid > 0, expected_valid))
        tensor_index = base_frame.tensor_index.to_numpy(np.int64)
        position = base_frame.position.to_numpy(np.int64)
        scored = base_frame.copy()
        scored["seed"] = seed
        scored["epoch"] = int(record["epoch"])
        scored["node_logit"] = node_logit[tensor_index]
        scored["node_probability"] = sigmoid(scored.node_logit.to_numpy())
        scored["boundary_logit"] = boundary_logit[tensor_index, position]
        scored["boundary_probability"] = sigmoid(scored.boundary_logit.to_numpy())
        scored["joint_probability"] = scored.node_probability * scored.boundary_probability
        scored["joint_logit"] = logit_probability(scored.joint_probability.to_numpy())
        scored["domain"] = "source"
        scored["conditional_population"] = scored.node_label == 0
        scored["auxiliary_population"] = scored.node_label >= 0
        records_for_checkpoint = metric_records_for_checkpoint(scored)
        for metric in records_for_checkpoint:
            metric_rows.append({"seed": seed, "epoch": int(record["epoch"]), **metric})
        historical = state["validation"]
        conditional = find_metric(
            records_for_checkpoint,
            definition="native",
            population="conditional",
            score="raw_boundary",
            view="raw_observation",
            scope="pooled",
        )
        reproduction = {
            "rows": conditional["rows"] == historical["conditional_boundary"]["rows"],
            "positives": conditional["positives"] == historical["conditional_boundary"]["positives"],
            "pr_auc": abs(float(conditional["pr_auc"]) - float(historical["conditional_boundary"]["pr_auc"])) <= 1e-12,
            "precision_at_actual": abs(
                float(conditional["precision_at_actual"])
                - float(historical["conditional_boundary"]["precision_at_actual_count"])
            ) <= 1e-12,
            "recall_at_95_precision": abs(
                float(conditional["recall_at_95_precision"])
                - float(historical["conditional_boundary"]["recall_at_95_precision"])
            ) <= 1e-12,
        }
        checks[f"seed_{seed}_historical_conditional_reproduced"] = all(reproduction.values())
        native_raw_macro = macro_from_records(
            records_for_checkpoint,
            definition="native",
            population="full",
            score="raw_boundary",
            view="unique_transition",
        )
        native_joint_macro = macro_from_records(
            records_for_checkpoint,
            definition="native",
            population="full",
            score="hierarchical_joint",
            view="unique_transition",
        )
        strict_raw_macro = macro_from_records(
            records_for_checkpoint,
            definition="strict",
            population="full",
            score="raw_boundary",
            view="unique_transition",
        )
        strict_joint_macro = macro_from_records(
            records_for_checkpoint,
            definition="strict",
            population="full",
            score="hierarchical_joint",
            view="unique_transition",
        )
        native_raw_pooled = find_metric(
            records_for_checkpoint,
            definition="native",
            population="full",
            score="raw_boundary",
            view="raw_observation",
            scope="pooled",
        )
        candidate = {
            "seed": seed,
            "epoch": int(record["epoch"]),
            "checkpoint_sha256": record["checkpoint_sha256"],
            "historical_composite": float(historical["checkpoint_selection_composite"]),
            "native_full_raw_pooled_boundary_pr_auc": float(native_raw_pooled["pr_auc"]),
            "native_full_unique_macro_raw_boundary_pr_auc": native_raw_macro["macro"]["pr_auc"],
            "native_full_unique_macro_joint_pr_auc": native_joint_macro["macro"]["pr_auc"],
            "strict_full_unique_macro_raw_boundary_pr_auc": strict_raw_macro["macro"]["pr_auc"],
            "strict_full_unique_macro_joint_pr_auc": strict_joint_macro["macro"]["pr_auc"],
            "native_full_raw_pooled_composite": counterfactual_composite(
                historical, float(native_raw_pooled["pr_auc"])
            ),
            "native_full_unique_macro_raw_composite": counterfactual_composite(
                historical, native_raw_macro["macro"]["pr_auc"]
            ),
            "native_full_unique_macro_joint_composite": counterfactual_composite(
                historical, native_joint_macro["macro"]["pr_auc"]
            ),
            "strict_full_unique_macro_joint_composite": counterfactual_composite(
                historical, strict_joint_macro["macro"]["pr_auc"]
            ),
            "native_raw_gate": native_raw_macro,
            "native_joint_gate": native_joint_macro,
            "strict_raw_gate": strict_raw_macro,
            "strict_joint_gate": strict_joint_macro,
            "historical_conditional_reproduction": reproduction,
        }
        candidate_rows.append(candidate)
        if seed == 2359003:
            max_selected_boundary_difference = float(
                np.max(np.abs(scored.boundary_logit.to_numpy() - base_frame.boundary_logit.to_numpy()))
            )
            max_selected_node_difference = float(
                np.max(np.abs(node_logit - selected_nodes.node_logit.to_numpy(np.float64)))
            )
            checks["selected_boundary_scores_match_frozen"] = max_selected_boundary_difference <= 2e-5
            checks["selected_node_scores_match_frozen"] = max_selected_node_difference <= 2e-5
        score_frames.append(
            scored[
                [
                    "seed",
                    "epoch",
                    "sequence",
                    "window_id",
                    "tensor_index",
                    "position",
                    "src_row_index",
                    "dst_row_index",
                    "transition_key",
                    "node_label",
                    "native_label",
                    "strict_label",
                    "node_logit",
                    "node_probability",
                    "boundary_logit",
                    "boundary_probability",
                    "joint_logit",
                    "joint_probability",
                    "conditional_population",
                ]
            ]
        )
        del model, state, node_logit, boundary_logit, valid, scored
        gc.collect()
    scores = pd.concat(score_frames, ignore_index=True)
    scores.to_parquet(R69 / "retained_checkpoint_validation_scores.parquet", index=False)
    pd.DataFrame(metric_rows).to_csv(R69 / "retained_checkpoint_metrics.csv", index=False)
    ranking_metrics = [
        "historical_composite",
        "native_full_raw_pooled_composite",
        "native_full_unique_macro_raw_composite",
        "native_full_unique_macro_joint_composite",
        "strict_full_unique_macro_joint_composite",
    ]
    rankings = {metric: rank_candidates(candidate_rows, metric) for metric in ranking_metrics}
    primary_metric = "native_full_unique_macro_joint_composite"
    primary_winner = int(rankings[primary_metric][0]["seed"])
    retained_selection_changed = primary_winner != 2359003
    checks["all_candidate_metrics_finite"] = all(
        np.isfinite(float(row[metric])) for row in candidate_rows for metric in ranking_metrics
    )
    checks["three_checkpoint_inference_runs"] = len(candidate_rows) == 3
    report = {
        "experiment_id": EXP_ID,
        "created_at": now(),
        "device": "cpu",
        "new_training": False,
        "optimizer_steps": 0,
        "checkpoint_modifications": 0,
        "retained_checkpoint_loads": len(candidate_rows),
        "retained_checkpoint_inference_runs": len(candidate_rows),
        "retained_checkpoint_validation_windows": len(node_x) * len(candidate_rows),
        "checks": checks,
        "hard_checks_passed": all(checks.values()),
        "selected_frozen_reproduction": {
            "maximum_boundary_logit_absolute_difference": max_selected_boundary_difference,
            "maximum_node_logit_absolute_difference": max_selected_node_difference,
            "tolerance": 2e-5,
        },
        "candidates": candidate_rows,
        "rankings": rankings,
        "primary_counterfactual_metric": primary_metric,
        "original_selected_seed": 2359003,
        "primary_counterfactual_winner_seed": primary_winner,
        "retained_selection_changed": retained_selection_changed,
        "counterfactual_scope": "three retained seed-best checkpoints only",
        "all_epoch_counterfactual_status": "unavailable_only_seed_best_weights_retained",
        "gate_thresholds": GATE_THRESHOLDS,
    }
    write_json(R69 / "retained_checkpoint_evaluation.json", report)
    if not report["hard_checks_passed"]:
        raise RuntimeError(f"retained checkpoint evaluation hard checks failed: {checks}")
    append_event(
        "retained_checkpoints_evaluated",
        checkpoint_count=len(candidate_rows),
        primary_counterfactual_winner_seed=primary_winner,
        retained_selection_changed=retained_selection_changed,
    )
    stage_finished(
        "evaluate-retained-checkpoints",
        started,
        rchar,
        {
            "primary_counterfactual_winner_seed": primary_winner,
            "retained_selection_changed": retained_selection_changed,
        },
        relative(R69 / "retained_checkpoint_evaluation.json"),
    )


def strict_endpoint_eligible(frame: pd.DataFrame) -> np.ndarray:
    return (
        (frame.supervision_status.to_numpy(dtype=str) == "matched")
        & ~frame.distractor_removed.to_numpy(bool)
        & ~frame.ambiguity_flag.to_numpy(bool)
        & ~frame.tie_flag.to_numpy(bool)
    )


def target_node_labels(node_scores: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    label_index = labels.set_index("row_index", drop=False)
    records = []
    for row in node_scores.itertuples(index=False):
        ids = parse_ids(row.row_indices)
        selected = label_index.reindex(ids)
        matched = selected[selected.supervision_status.astype(str) == "matched"]
        if len(matched) < NODE_LABEL_MIN_KNOWN:
            node_label = -1
        else:
            node_label = 0 if matched.gt_identity_key.astype(str).nunique() > 1 else 1
        strict = selected[strict_endpoint_eligible(selected)]
        if len(strict) < NODE_LABEL_MIN_KNOWN:
            strict_node_label = -1
        else:
            strict_node_label = 0 if strict.gt_identity_key.astype(str).nunique() > 1 else 1
        records.append(
            {
                "sequence": str(row.sequence),
                "window_id": str(row.window_id),
                "node_logit": float(row.node_logit),
                "node_probability": float(row.node_probability),
                "node_label": node_label,
                "strict_node_label": strict_node_label,
                "known_rows": int(len(matched)),
                "strict_known_rows": int(len(strict)),
            }
        )
    result = pd.DataFrame(records)
    if result.duplicated(["sequence", "window_id"]).any():
        raise RuntimeError("target node score key duplicate")
    return result


def target_sequence_semantic_frame(sequence: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = R65 / sequence
    boundary = pd.read_parquet(root / "scores/boundary_scores.parquet").reset_index(drop=True)
    node = pd.read_parquet(root / "scores/node_scores.parquet")
    labels = pd.read_parquet(root / "labels/row_labels.parquet").sort_values(
        "row_index", kind="mergesort"
    )
    if labels.row_index.duplicated().any():
        raise RuntimeError(f"{sequence}: target label row key duplicate")
    node_labeled = target_node_labels(node, labels)
    frame = boundary.merge(
        node_labeled,
        on=["sequence", "window_id"],
        how="left",
        validate="many_to_one",
    )
    if frame.node_probability.isna().any():
        raise RuntimeError(f"{sequence}: target node score join incomplete")
    label_index = labels.set_index("row_index", drop=False)
    source = label_index.reindex(frame.src_row_index.to_numpy(np.int64))
    destination = label_index.reindex(frame.dst_row_index.to_numpy(np.int64))
    native_known = (
        (source.supervision_status.to_numpy(dtype=str) == "matched")
        & (destination.supervision_status.to_numpy(dtype=str) == "matched")
    )
    strict_known = strict_endpoint_eligible(source) & strict_endpoint_eligible(destination)
    source_identity = source.gt_identity_key.to_numpy(dtype=str)
    destination_identity = destination.gt_identity_key.to_numpy(dtype=str)
    native_label = np.full(len(frame), -1, dtype=np.int8)
    strict_label = np.full(len(frame), -1, dtype=np.int8)
    native_label[native_known] = (source_identity[native_known] != destination_identity[native_known]).astype(
        np.int8
    )
    strict_label[strict_known] = (source_identity[strict_known] != destination_identity[strict_known]).astype(
        np.int8
    )
    boundary_logit = frame.boundary_logit.to_numpy(np.float64)
    boundary_probability = frame.boundary_probability.to_numpy(np.float64)
    node_probability = frame.node_probability.to_numpy(np.float64)
    joint_probability = node_probability * boundary_probability
    checks = {
        "boundary_finite": bool(np.isfinite(boundary_logit).all() and np.isfinite(boundary_probability).all()),
        "node_finite": bool(np.isfinite(node_probability).all()),
        "boundary_sigmoid_exact": bool(
            np.allclose(boundary_probability, sigmoid(boundary_logit), atol=1e-12, rtol=1e-12)
        ),
        "forward_row_order": bool((frame.dst_row_index.to_numpy() > frame.src_row_index.to_numpy()).all()),
        "native_labels_defined_only_matched": bool(np.all(native_label[~native_known] == -1)),
        "strict_subset_native": bool(np.all(native_known[strict_known])),
    }
    frame["domain"] = "target"
    frame["split"] = "target"
    frame["transition_key"] = (
        frame.sequence.astype(str)
        + ":"
        + frame.src_row_index.astype(str)
        + ":"
        + frame.dst_row_index.astype(str)
    )
    frame["native_label"] = native_label
    frame["strict_label"] = strict_label
    frame["conditional_population"] = frame.node_label == 0
    frame["auxiliary_population"] = frame.node_label >= 0
    frame["pure_window"] = frame.node_label == 1
    frame["joint_probability"] = joint_probability
    frame["joint_logit"] = logit_probability(joint_probability)
    audit = {
        "sequence": sequence,
        "raw_boundary_observations": int(len(frame)),
        "native_known_observations": int(native_known.sum()),
        "strict_known_observations": int(strict_known.sum()),
        "conditional_native_known_observations": int(
            (frame.conditional_population & frame.native_label.isin([0, 1])).sum()
        ),
        "node_window_counts": {
            str(int(value)): int((node_labeled.node_label == value).sum())
            for value in sorted(node_labeled.node_label.unique())
        },
        "checks": checks,
        "all_passed": all(checks.values()),
    }
    return frame, audit


def source_frozen_semantic_frame(sequence: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = source_population_frame()
    frame = frame[(frame.split == "validation") & (frame.sequence == sequence)].copy()
    node = pd.read_parquet(R66 / "source_scores" / sequence / "node_scores.parquet")
    node = node[["sequence", "window_id", "node_logit", "node_probability"]]
    frame = frame.drop(columns=["joint_probability", "joint_logit"]).merge(
        node,
        on=["sequence", "window_id"],
        how="left",
        validate="many_to_one",
    )
    if frame.node_probability.isna().any():
        raise RuntimeError(f"{sequence}: source frozen node join incomplete")
    frozen_boundary = pd.read_parquet(R66 / "source_scores" / sequence / "boundary_scores.parquet")
    key = ["sequence", "window_id", "src_row_index", "dst_row_index"]
    comparison = frame[key + ["boundary_logit"]].merge(
        frozen_boundary[key + ["boundary_logit"]],
        on=key,
        how="outer",
        suffixes=("_r68", "_r66"),
        indicator=True,
        validate="one_to_one",
    )
    max_difference = float(
        np.max(np.abs(comparison.boundary_logit_r68.to_numpy() - comparison.boundary_logit_r66.to_numpy()))
    )
    frame["joint_probability"] = frame.node_probability * frame.boundary_probability
    frame["joint_logit"] = logit_probability(frame.joint_probability.to_numpy())
    frame["pure_window"] = frame.node_label == 1
    audit = {
        "sequence": sequence,
        "raw_boundary_observations": int(len(frame)),
        "native_known_observations": int(frame.native_label.isin([0, 1]).sum()),
        "strict_known_observations": int(frame.strict_label.isin([0, 1]).sum()),
        "conditional_native_known_observations": int(
            (frame.conditional_population & frame.native_label.isin([0, 1])).sum()
        ),
        "r66_r68_key_join_all_both": bool((comparison._merge == "both").all()),
        "maximum_r66_r68_boundary_logit_difference": max_difference,
        "all_passed": bool((comparison._merge == "both").all() and max_difference <= 1e-12),
    }
    return frame, audit


def semantic_records_for_sequence(frame: pd.DataFrame) -> list[dict[str, Any]]:
    unique = aggregate_unique(frame)
    records = []
    for definition in ["native", "strict"]:
        label = f"{definition}_label"
        for population in ["full", "conditional"]:
            raw = frame if population == "full" else frame[frame.conditional_population]
            primary = unique if population == "full" else unique[unique.conditional_population]
            for score, score_column in [
                ("raw_boundary", "boundary_probability"),
                ("hierarchical_joint", "joint_probability"),
            ]:
                for view, data in [("raw_observation", raw), ("unique_transition", primary)]:
                    metrics = binary_metrics(
                        data[label].to_numpy(),
                        data[score_column].to_numpy(),
                        data.index.to_numpy() if view == "raw_observation" else data.transition_key.to_numpy(),
                    )
                    records.append(
                        {
                            "domain": str(frame.domain.iloc[0]),
                            "sequence": str(frame.sequence.iloc[0]),
                            "definition": definition,
                            "population": population,
                            "score": score,
                            "view": view,
                            **metrics,
                        }
                    )
    return records


def domain_gate(
    records: list[dict[str, Any]],
    sequences: list[str],
    *,
    definition: str,
    population: str,
    score: str,
    view: str,
) -> dict[str, Any]:
    selected = []
    for sequence in sequences:
        matching = [
            record
            for record in records
            if record["sequence"] == sequence
            and record["definition"] == definition
            and record["population"] == population
            and record["score"] == score
            and record["view"] == view
        ]
        if len(matching) != 1:
            raise RuntimeError((sequence, definition, population, score, view, len(matching)))
        selected.append(matching[0])
    macro = {
        metric: float(np.mean([record[metric] for record in selected]))
        for metric in ["pr_auc", "precision_at_actual", "recall_at_95_precision"]
    }
    minimum = float(min(record["precision_at_actual"] for record in selected))
    return {
        "definition": definition,
        "population": population,
        "score": score,
        "view": view,
        "macro": macro,
        "minimum_sequence_precision_at_actual": minimum,
        "passed": bool(
            macro["pr_auc"] >= GATE_THRESHOLDS["macro_pr_auc"]
            and macro["precision_at_actual"] >= GATE_THRESHOLDS["macro_precision_at_actual"]
            and macro["recall_at_95_precision"] >= GATE_THRESHOLDS["macro_recall_at_95_precision"]
            and minimum >= GATE_THRESHOLDS["minimum_sequence_precision_at_actual"]
        ),
        "per_sequence": {record["sequence"]: record for record in selected},
    }


def command_audit_gate_score_semantics() -> None:
    started, rchar = stage_started("audit-gate-score-semantics")
    records = []
    audits = {}
    for sequence in VALIDATION_SEQUENCES:
        frame, audit = source_frozen_semantic_frame(sequence)
        records.extend(semantic_records_for_sequence(frame))
        audits[sequence] = audit
        del frame
        gc.collect()
    for sequence in TARGET_SEQUENCES:
        frame, audit = target_sequence_semantic_frame(sequence)
        records.extend(semantic_records_for_sequence(frame))
        audits[sequence] = audit
        del frame
        gc.collect()
    metrics_frame = pd.DataFrame(records)
    metrics_frame.to_csv(R69 / "gate_score_semantics_metrics.csv", index=False)
    source_gates = {}
    target_gates = {}
    for definition in ["native", "strict"]:
        for population in ["full", "conditional"]:
            for score in ["raw_boundary", "hierarchical_joint"]:
                name = f"{definition}_{population}_{score}_unique"
                source_gates[name] = domain_gate(
                    records,
                    VALIDATION_SEQUENCES,
                    definition=definition,
                    population=population,
                    score=score,
                    view="unique_transition",
                )
                target_gates[name] = domain_gate(
                    records,
                    TARGET_SEQUENCES,
                    definition=definition,
                    population=population,
                    score=score,
                    view="unique_transition",
                )
    published = read_json(R65 / "representation_metrics.json", {}) or {}
    published_by_sequence = {item["sequence"]: item["boundary"] for item in published.get("sequences", [])}
    reproduction = {}
    for sequence in TARGET_SEQUENCES:
        matching = [
            record
            for record in records
            if record["sequence"] == sequence
            and record["definition"] == "native"
            and record["population"] == "full"
            and record["score"] == "raw_boundary"
            and record["view"] == "raw_observation"
        ][0]
        expected = published_by_sequence[sequence]
        reproduction[sequence] = {
            "rows": matching["rows"] == expected["rows"],
            "positives": matching["positives"] == expected["positives"],
            "pr_auc": abs(float(matching["pr_auc"]) - float(expected["pr_auc"])) <= 1e-12,
            "precision_at_actual": abs(
                float(matching["precision_at_actual"]) - float(expected["precision_at_actual"])
            ) <= 1e-12,
            "recall_at_95_precision": abs(
                float(matching["recall_at_95_precision"]) - float(expected["recall_at_95_precision"])
            ) <= 1e-12,
        }
    raw_source = source_gates["native_full_raw_boundary_unique"]
    joint_source = source_gates["native_full_hierarchical_joint_unique"]
    raw_target = target_gates["native_full_raw_boundary_unique"]
    joint_target = target_gates["native_full_hierarchical_joint_unique"]
    source_gain = float(joint_source["macro"]["pr_auc"] - raw_source["macro"]["pr_auc"])
    target_gain = float(joint_target["macro"]["pr_auc"] - raw_target["macro"]["pr_auc"])
    material_hierarchical_gain = bool(
        source_gain >= DIAGNOSTIC_THRESHOLDS["minimum_hierarchical_pr_auc_absolute_gain"]
        and target_gain >= DIAGNOSTIC_THRESHOLDS["minimum_hierarchical_pr_auc_absolute_gain"]
    )
    conditional_target = target_gates["native_conditional_raw_boundary_unique"]
    report = {
        "experiment_id": EXP_ID,
        "created_at": now(),
        "sequence_audits": audits,
        "all_sequence_audits_passed": all(audit["all_passed"] for audit in audits.values()),
        "published_m23_65_raw_metric_reproduction": reproduction,
        "published_m23_65_raw_metric_reproduction_passed": all(
            all(checks.values()) for checks in reproduction.values()
        ),
        "source_gates": source_gates,
        "target_gates": target_gates,
        "native_full_unique_primary": {
            "source_raw_macro_pr_auc": raw_source["macro"]["pr_auc"],
            "source_hierarchical_macro_pr_auc": joint_source["macro"]["pr_auc"],
            "source_absolute_gain": source_gain,
            "target_raw_macro_pr_auc": raw_target["macro"]["pr_auc"],
            "target_hierarchical_macro_pr_auc": joint_target["macro"]["pr_auc"],
            "target_absolute_gain": target_gain,
            "material_hierarchical_gain": material_hierarchical_gain,
            "source_gate_conclusion_changed": raw_source["passed"] != joint_source["passed"],
            "target_gate_conclusion_changed": raw_target["passed"] != joint_target["passed"],
        },
        "target_conditional_native_raw_gate": conditional_target,
        "fixed_hierarchical_score": "sigmoid(node_impurity_logit) * sigmoid(conditional_boundary_logit)",
        "no_threshold_search": True,
        "no_calibration_fit": True,
        "no_score_reversal": True,
        "frozen_score_only": True,
        "raw_gt_reads": 0,
    }
    hard_checks = bool(
        report["all_sequence_audits_passed"] and report["published_m23_65_raw_metric_reproduction_passed"]
    )
    report["hard_checks_passed"] = hard_checks
    write_json(R69 / "gate_score_semantics.json", report)
    if not hard_checks:
        raise RuntimeError("gate score semantic hard checks failed")
    append_event(
        "gate_score_semantics_audited",
        source_raw_macro_pr_auc=raw_source["macro"]["pr_auc"],
        source_hierarchical_macro_pr_auc=joint_source["macro"]["pr_auc"],
        target_raw_macro_pr_auc=raw_target["macro"]["pr_auc"],
        target_hierarchical_macro_pr_auc=joint_target["macro"]["pr_auc"],
    )
    stage_finished(
        "audit-gate-score-semantics",
        started,
        rchar,
        {
            "source_hierarchical_pr_auc_gain": source_gain,
            "target_hierarchical_pr_auc_gain": target_gain,
        },
        relative(R69 / "gate_score_semantics.json"),
    )


def command_diagnose() -> None:
    started, rchar = stage_started("diagnose")
    population = read_json(R69 / "objective_population.json", {}) or {}
    checkpoints = read_json(R69 / "retained_checkpoint_evaluation.json", {}) or {}
    semantics = read_json(R69 / "gate_score_semantics.json", {}) or {}
    inventory = read_json(R69 / "checkpoint_inventory.json", {}) or {}
    hard_checks = bool(
        population.get("hard_checks_passed")
        and checkpoints.get("hard_checks_passed")
        and semantics.get("hard_checks_passed")
        and inventory.get("hard_checks_passed")
    )
    material_population_mismatch = bool(population.get("material_population_mismatch"))
    retained_selection_changed = bool(checkpoints.get("retained_selection_changed"))
    source_generalization_failure = bool(
        population.get("selected_checkpoint_performance", {})
        .get("conditional_native_train_to_validation", {})
        .get("fixed_generalization_failure")
    )
    overall = classify_diagnosis(
        hard_checks,
        material_population_mismatch,
        retained_selection_changed,
        source_generalization_failure,
    )
    if not hard_checks:
        measurement = "FAIL_MEASUREMENT_INTEGRITY"
        decision = "FAIL_IMPLEMENTATION"
    else:
        measurement = "PASS_OBJECTIVE_GATE_RECONSTRUCTION"
        decision = "COMPLETED_POST_HOC_DIAGNOSTIC"
    alignment = (
        "FAIL_CONDITIONAL_BOUNDARY_POPULATION_ALIGNMENT"
        if material_population_mismatch
        else "PASS_BOUNDARY_POPULATION_ALIGNMENT"
    )
    retained_stability = (
        "UNSTABLE_RETAINED_SEED_BEST_SELECTION"
        if retained_selection_changed
        else "STABLE_RETAINED_SEED_BEST_SELECTION"
    )
    hierarchy = semantics.get("native_full_unique_primary", {})
    secondary = []
    if retained_selection_changed:
        secondary.append("retained_checkpoint_selection_population_sensitivity")
    if source_generalization_failure:
        secondary.append("source_sequence_generalization_failure")
    if hierarchy.get("material_hierarchical_gain"):
        secondary.append("fixed_hierarchical_score_material_gain")
    if not inventory.get("all_epoch_counterfactual_available"):
        secondary.append("all_epoch_counterfactual_unavailable")
    diagnosis = {
        "experiment_id": EXP_ID,
        "status": "diagnosed",
        "experiment_decision": decision,
        "measurement_integrity_decision": measurement,
        "objective_gate_alignment_decision": alignment,
        "retained_checkpoint_selection_stability": retained_stability,
        "all_epoch_checkpoint_selection_status": inventory.get("all_epoch_counterfactual_status"),
        "source_sequence_generalization_status": (
            "FAIL_SOURCE_SEQUENCE_GENERALIZATION"
            if source_generalization_failure
            else "NO_FIXED_SOURCE_GENERALIZATION_FAILURE"
        ),
        "overall_primary_classification": overall,
        "secondary_findings": secondary,
        "hard_checks_passed": hard_checks,
        "material_population_mismatch": material_population_mismatch,
        "retained_selection_changed": retained_selection_changed,
        "source_generalization_failure": source_generalization_failure,
        "primary_counterfactual_metric": checkpoints.get("primary_counterfactual_metric"),
        "original_selected_seed": checkpoints.get("original_selected_seed"),
        "primary_counterfactual_winner_seed": checkpoints.get("primary_counterfactual_winner_seed"),
        "hierarchical_score_sensitivity": hierarchy,
        "m23_65_decision_unchanged": "FAIL_MOT20_REPRESENTATION_GATE",
        "same_experiment_retraining_authorized": False,
        "same_experiment_gate_rerun_authorized": False,
        "all_epoch_counterfactual_claim_forbidden": not inventory.get("all_epoch_counterfactual_available"),
        **FIXED_STATUS,
    }
    write_json(R69 / "final_diagnosis.json", diagnosis)
    if not hard_checks:
        raise RuntimeError("M23-69 diagnosis integrity failed")
    append_event(
        "diagnosis_completed",
        alignment=alignment,
        retained_stability=retained_stability,
        overall=overall,
    )
    stage_finished(
        "diagnose",
        started,
        rchar,
        {"overall_primary_classification": overall},
        relative(R69 / "final_diagnosis.json"),
    )


def scope_report() -> dict[str, Any]:
    evaluation = read_json(R69 / "retained_checkpoint_evaluation.json", {}) or {}
    allowed = {
        "retained_checkpoint_loads": int(evaluation.get("retained_checkpoint_loads", -1)),
        "retained_checkpoint_inference_runs": int(evaluation.get("retained_checkpoint_inference_runs", -1)),
        "retained_checkpoint_validation_windows": int(
            evaluation.get("retained_checkpoint_validation_windows", -1)
        ),
    }
    return {
        "experiment_id": EXP_ID,
        "created_at": now(),
        "forbidden_scope_counts": FORBIDDEN_SCOPE_COUNTS,
        "forbidden_scope_all_zero": all(value == 0 for value in FORBIDDEN_SCOPE_COUNTS.values()),
        "allowed_scope_counts": allowed,
        "expected_allowed_scope_counts": EXPECTED_ALLOWED_COUNTS,
        "allowed_scope_exact": allowed == EXPECTED_ALLOWED_COUNTS,
        "frozen_mot17_label_sidecar_reads": True,
        "frozen_mot20_label_sidecar_reads": True,
        "frozen_score_artifact_reads": True,
        "new_model_inference_device": "cpu",
        "raw_gt_reads": 0,
        **FIXED_STATUS,
    }


def command_validate() -> None:
    started, rchar = stage_started("validate")
    input_manifest = read_json(R69 / "input_manifest.json", {}) or {}
    input_checks = verify_hash_map(input_manifest.get("sha256", {}))
    required = [
        "summary.csv",
        "protocol_events.jsonl",
        "preflight.json",
        "preregistration.json",
        "input_manifest.json",
        "implementation_manifest.json",
        "input_reverification.json",
        "predecessor_reverification.json",
        "source_unique_transition_population.parquet",
        "objective_population.csv",
        "objective_population.json",
        "selected_source_population_metrics.csv",
        "checkpoint_inventory.json",
        "retained_checkpoint_validation_scores.parquet",
        "retained_checkpoint_metrics.csv",
        "retained_checkpoint_evaluation.json",
        "gate_score_semantics_metrics.csv",
        "gate_score_semantics.json",
        "final_diagnosis.json",
    ]
    present = {name: (R69 / name).is_file() for name in required}
    scope = scope_report()
    write_json(R69 / "scope_validation.json", scope)
    process_gpu = process_gpu_snapshot()
    write_json(R69 / "process_gpu_validation.json", process_gpu)
    diagnosis = read_json(R69 / "final_diagnosis.json", {}) or {}
    predecessor = predecessor_check()
    validation = {
        "experiment_id": EXP_ID,
        "validated_at": now(),
        "inputs_unchanged": bool(input_checks) and all(record["match"] for record in input_checks),
        "input_checks": input_checks,
        "required_artifacts": present,
        "required_artifacts_present": all(present.values()),
        "implementation_guard_passed": all(implementation_guard().values()),
        "predecessors_unchanged": predecessor["all_passed"],
        "objective_population_passed": read_json(R69 / "objective_population.json", {}).get(
            "hard_checks_passed"
        )
        is True,
        "checkpoint_inventory_passed": read_json(R69 / "checkpoint_inventory.json", {}).get(
            "hard_checks_passed"
        )
        is True,
        "retained_checkpoint_evaluation_passed": read_json(
            R69 / "retained_checkpoint_evaluation.json", {}
        ).get("hard_checks_passed")
        is True,
        "gate_score_semantics_passed": read_json(R69 / "gate_score_semantics.json", {}).get(
            "hard_checks_passed"
        )
        is True,
        "forbidden_scope_all_zero": scope["forbidden_scope_all_zero"],
        "allowed_scope_exact": scope["allowed_scope_exact"],
        "process_gpu_idle": process_gpu_idle(process_gpu),
        "hota_is_null": diagnosis.get("hota") is None,
        "next_policy_authorized_false": diagnosis.get("next_policy_authorized") is False,
        "m23_70_not_started": diagnosis.get("m23_70_started") is False,
    }
    validation["all_passed"] = all(
        validation[key]
        for key in [
            "inputs_unchanged",
            "required_artifacts_present",
            "implementation_guard_passed",
            "predecessors_unchanged",
            "objective_population_passed",
            "checkpoint_inventory_passed",
            "retained_checkpoint_evaluation_passed",
            "gate_score_semantics_passed",
            "forbidden_scope_all_zero",
            "allowed_scope_exact",
            "process_gpu_idle",
            "hota_is_null",
            "next_policy_authorized_false",
            "m23_70_not_started",
        ]
    )
    write_json(R69 / "validation_report.json", validation)
    if not validation["all_passed"]:
        raise RuntimeError("M23-69 validation failed")
    append_event("validation_completed", all_passed=True)
    stage_finished(
        "validate",
        started,
        rchar,
        {"input_count": len(input_checks)},
        relative(R69 / "validation_report.json"),
    )


def format_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def result_markdown(
    diagnosis: dict[str, Any],
    population: dict[str, Any],
    checkpoints: dict[str, Any],
    semantics: dict[str, Any],
    inventory: dict[str, Any],
) -> str:
    source_generalization = population["selected_checkpoint_performance"][
        "conditional_native_train_to_validation"
    ]
    hierarchy = semantics["native_full_unique_primary"]
    lines = [
        f"# {EXP_NAME} — Result",
        "",
        "## Final diagnosis",
        "",
        f"- status: `completed`",
        f"- experiment decision: `{diagnosis['experiment_decision']}`",
        f"- measurement integrity: `{diagnosis['measurement_integrity_decision']}`",
        f"- objective/gate alignment: `{diagnosis['objective_gate_alignment_decision']}`",
        f"- retained checkpoint selection: `{diagnosis['retained_checkpoint_selection_stability']}`",
        f"- all-epoch checkpoint status: `{diagnosis['all_epoch_checkpoint_selection_status']}`",
        f"- source generalization: `{diagnosis['source_sequence_generalization_status']}`",
        f"- overall primary classification: `{diagnosis['overall_primary_classification']}`",
        "",
        "M23-69 is a post-hoc diagnostic using frozen GT-derived label sidecars. It is not deployable, not a strict result, does not modify M23-64/M23-65, and does not authorize a tracker, TrackEval, HOTA evaluation, retraining, or next policy.",
        "",
        "## Objective population reconstruction",
        "",
    ]
    for split in ["train", "validation"]:
        item = population["split_alignment"][split]
        lines.extend(
            [
                f"### {split}",
                "",
                f"- full native-known observations: {item['full_native_known_rows']}",
                f"- conditional focal/selection observations: {item['conditional_native_known_rows']}",
                f"- conditional row coverage: {format_value(item['conditional_row_coverage'])}",
                f"- full/conditional positives: {item['full_native_positives']} / {item['conditional_native_positives']}",
                f"- positive coverage: {format_value(item['conditional_positive_coverage'])}",
                f"- negative coverage: {format_value(item['conditional_negative_coverage'])}",
                "",
            ]
        )
    lines.extend(
        [
            "The exact source code and tensor reconstruction show that the class-balanced boundary focal loss and checkpoint-selection boundary metric use native-known transitions only inside oracle-impure windows. The count-consistency and sparsity terms act on a broader node-known valid population; count consistency clamps unknown boundary labels to zero. M23-65 instead publishes the raw conditional-boundary probability over all matched full-window transitions without multiplying by the node-impurity probability.",
            "",
            "## Source generalization",
            "",
            f"- conditional native train PR-AUC: {format_value(source_generalization['train_pr_auc'])}",
            f"- conditional native validation PR-AUC: {format_value(source_generalization['validation_pr_auc'])}",
            f"- absolute drop: {format_value(source_generalization['absolute_drop'])}",
            f"- validation/train ratio: {format_value(source_generalization['validation_to_train_ratio'])}",
            f"- fixed generalization failure: `{source_generalization['fixed_generalization_failure']}`",
            "",
            "## Retained checkpoint sensitivity",
            "",
            f"- historical epoch metric records: {inventory['historical_epoch_metric_records']}",
            f"- unique retained weight states: {inventory['unique_weight_sha_count']}",
            f"- original selected seed/epoch: {checkpoints['original_selected_seed']} / 19",
            f"- primary counterfactual metric: `{checkpoints['primary_counterfactual_metric']}`",
            f"- primary retained winner seed: {checkpoints['primary_counterfactual_winner_seed']}",
            f"- retained selection changed: `{checkpoints['retained_selection_changed']}`",
            "",
            "Only the three seed-best weights survive. The 90 epoch metric rows do not contain 90 restorable model states, so M23-69 does not claim an all-epoch counterfactual winner.",
            "",
            "## Fixed hierarchical score sensitivity",
            "",
            f"- source raw unique macro PR-AUC: {format_value(hierarchy['source_raw_macro_pr_auc'])}",
            f"- source hierarchical unique macro PR-AUC: {format_value(hierarchy['source_hierarchical_macro_pr_auc'])}",
            f"- source absolute gain: {format_value(hierarchy['source_absolute_gain'])}",
            f"- target raw unique macro PR-AUC: {format_value(hierarchy['target_raw_macro_pr_auc'])}",
            f"- target hierarchical unique macro PR-AUC: {format_value(hierarchy['target_hierarchical_macro_pr_auc'])}",
            f"- target absolute gain: {format_value(hierarchy['target_absolute_gain'])}",
            f"- target gate conclusion changed: `{hierarchy['target_gate_conclusion_changed']}`",
            "",
            "The hierarchical score was fixed before evaluation as `sigmoid(node_impurity_logit) * sigmoid(conditional_boundary_logit)`. No threshold search, calibration, score reversal, policy construction, tracker generation, TrackEval, or HOTA evaluation occurred.",
            "",
            "## Scope and closure",
            "",
            "- training runs: 0",
            "- optimizer steps: 0",
            "- checkpoint outputs/modifications: 0 / 0",
            "- retained checkpoint inference runs: 3 (CPU, MOT17 validation only)",
            "- tracker outputs / TrackEval / HOTA: 0 / 0 / 0",
            "- raw MOT17 GT / raw MOT20 GT / MOT20 test reads: 0 / 0 / 0",
            "- HOTA: null",
            "- next_policy_authorized: false",
            "- M23-70 started: false",
            "",
            "Structured records are under `outputs/mot20_m23_20260718/m23_69_boundary_objective_gate_population_alignment_audit/`. Notion writeback was not executed.",
            "",
        ]
    )
    return "\n".join(lines)


def artifact_manifest() -> dict[str, Any]:
    excluded = {"artifact_sha256_manifest.json", "artifact_manifest_validation.json"}
    paths = [SCRIPT, TEST_SCRIPT, PREREG, RESULT]
    paths.extend(path for path in R69.iterdir() if path.is_file() and path.name not in excluded)
    records = [
        {"path": relative(path), "sha256": sha256(path), "size": path.stat().st_size}
        for path in sorted(set(paths), key=lambda item: str(item))
    ]
    return {"experiment_id": EXP_ID, "created_at": now(), "records": records}


def command_summarize() -> None:
    started, rchar = stage_started("summarize")
    diagnosis = read_json(R69 / "final_diagnosis.json", {}) or {}
    population = read_json(R69 / "objective_population.json", {}) or {}
    checkpoints = read_json(R69 / "retained_checkpoint_evaluation.json", {}) or {}
    semantics = read_json(R69 / "gate_score_semantics.json", {}) or {}
    inventory = read_json(R69 / "checkpoint_inventory.json", {}) or {}
    RESULT.write_text(
        result_markdown(diagnosis, population, checkpoints, semantics, inventory), encoding="utf-8"
    )
    stage_finished("summarize", started, rchar, report=relative(RESULT))
    decision = str(diagnosis["experiment_decision"])
    set_stage("closed", "closed", started_at=now(), finished_at=now(), decision=decision)
    registry_line = registry_close(
        "completed",
        decision,
        "alignment="
        + str(diagnosis["objective_gate_alignment_decision"])
        + "; retained_selection="
        + str(diagnosis["retained_checkpoint_selection_stability"])
        + "; all_epoch="
        + str(diagnosis["all_epoch_checkpoint_selection_status"])
        + "; overall="
        + str(diagnosis["overall_primary_classification"])
        + "; HOTA=null; next_policy_authorized=false; result="
        + relative(RESULT),
    )
    final_input_checks = verify_hash_map(read_json(R69 / "input_manifest.json", {})["sha256"])
    write_json(
        R69 / "input_reverification_final.json",
        {
            "checked_at": now(),
            "records": final_input_checks,
            "inputs_unchanged": bool(final_input_checks)
            and all(record["match"] for record in final_input_checks),
        },
    )
    process_gpu = process_gpu_snapshot()
    scope = scope_report()
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
        "scope": scope,
        "git_head": git_head(),
        "git_scoped_status": scoped_git_status(),
        "notion_writeback": "未执行Notion写回",
        **FIXED_STATUS,
    }
    write_json(R69 / "final_summary.json", final)
    append_event(
        "experiment_closed",
        decision=decision,
        overall=diagnosis["overall_primary_classification"],
        registry_line=registry_line,
    )
    summary = pd.read_csv(R69 / "summary.csv", keep_default_na=False)
    implementation_checks = implementation_guard()
    closure = {
        "experiment_id": EXP_ID,
        "validated_at": now(),
        "status": "closed",
        "decision": decision,
        "inputs_unchanged": bool(final_input_checks)
        and all(record["match"] for record in final_input_checks),
        "implementation_checks": implementation_checks,
        "implementation_sha_guard": all(implementation_checks.values()),
        "summary_no_running_pending": not summary.status.astype(str).isin(["running", "pending"]).any(),
        "registry_completed_closed": True,
        "process_gpu_idle": process_gpu_idle(process_gpu),
        "forbidden_scope_all_zero": scope["forbidden_scope_all_zero"],
        "allowed_scope_exact": scope["allowed_scope_exact"],
        "result_document_exists": RESULT.is_file(),
        "final_summary_exists": True,
        "hota_is_null": final["hota"] is None,
        "next_policy_authorized_false": final["next_policy_authorized"] is False,
        "m23_70_not_started": final["m23_70_started"] is False,
        "predecessors_unchanged": predecessor_check()["all_passed"],
    }
    closure["closure_integrity_passed"] = all(
        closure[key]
        for key in [
            "inputs_unchanged",
            "implementation_sha_guard",
            "summary_no_running_pending",
            "registry_completed_closed",
            "process_gpu_idle",
            "forbidden_scope_all_zero",
            "allowed_scope_exact",
            "result_document_exists",
            "final_summary_exists",
            "hota_is_null",
            "next_policy_authorized_false",
            "m23_70_not_started",
            "predecessors_unchanged",
        ]
    )
    write_json(R69 / "closure_validation.json", closure)
    independent = {
        "experiment_id": EXP_ID,
        "validated_at": now(),
        "final_status_completed_closed": final["status"] == "completed"
        and final["current_stage"] == "closed",
        "inputs_unchanged": closure["inputs_unchanged"],
        "implementation_sha_guard": closure["implementation_sha_guard"],
        "summary_no_running_pending": closure["summary_no_running_pending"],
        "registry_completed_closed": closure["registry_completed_closed"],
        "process_gpu_idle": closure["process_gpu_idle"],
        "forbidden_scope_all_zero": closure["forbidden_scope_all_zero"],
        "allowed_scope_exact": closure["allowed_scope_exact"],
        "result_document_exists": closure["result_document_exists"],
        "hota_is_null": closure["hota_is_null"],
        "next_policy_authorized_false": closure["next_policy_authorized_false"],
        "m23_70_not_started": closure["m23_70_not_started"],
        "predecessors_unchanged": closure["predecessors_unchanged"],
        "closure_integrity_passed": closure["closure_integrity_passed"],
    }
    independent["independent_closure_passed"] = all(
        value for key, value in independent.items() if key not in {"experiment_id", "validated_at"}
    )
    write_json(R69 / "independent_closure_validation.json", independent)
    manifest = artifact_manifest()
    write_json(R69 / "artifact_sha256_manifest.json", manifest)
    artifact_checks = []
    for record in manifest["records"]:
        path = ROOT / record["path"]
        actual = sha256(path) if path.is_file() else None
        artifact_checks.append(
            {
                "path": record["path"],
                "expected": record["sha256"],
                "actual": actual,
                "match": actual == record["sha256"],
            }
        )
    write_json(
        R69 / "artifact_manifest_validation.json",
        {
            "experiment_id": EXP_ID,
            "checked_at": now(),
            "records": artifact_checks,
            "all_match": bool(artifact_checks) and all(record["match"] for record in artifact_checks),
        },
    )
    if not closure["closure_integrity_passed"]:
        raise RuntimeError("M23-69 closure integrity failed")
    if not independent["independent_closure_passed"]:
        raise RuntimeError("M23-69 independent closure failed")
    if not all(record["match"] for record in artifact_checks):
        raise RuntimeError("M23-69 artifact manifest validation failed")


def fail_close(error: BaseException) -> None:
    if not R69.exists() or not (R69 / "summary.csv").exists():
        return
    summary = pd.read_csv(R69 / "summary.csv", keep_default_na=False)
    running = summary[summary.status == "running"]
    if len(running):
        set_stage(str(running.iloc[0].stage), "failed", finished_at=now(), error=repr(error))
    summary = pd.read_csv(R69 / "summary.csv", keep_default_na=False)
    for stage in summary.loc[summary.status == "pending", "stage"].tolist():
        if stage != "closed":
            set_stage(stage, "skipped", finished_at=now(), error="fail_closed")
    set_stage(
        "closed",
        "closed",
        started_at=now(),
        finished_at=now(),
        decision="FAIL_IMPLEMENTATION",
        error=repr(error),
    )
    try:
        registry_line = registry_close(
            "failed",
            "FAIL_IMPLEMENTATION",
            f"fail_closed; {repr(error)}; HOTA=null; next_policy_authorized=false",
        )
    except Exception:
        registry_line = None
    snapshot = process_gpu_snapshot()
    write_json(
        R69 / "implementation_failure.json",
        {
            "experiment_id": EXP_ID,
            "status": "closed",
            "decision": "FAIL_IMPLEMENTATION",
            "error": repr(error),
            "traceback": traceback.format_exc(),
            "same_root_repair_performed": False,
            **FIXED_STATUS,
        },
    )
    final = {
        "experiment_id": EXP_ID,
        "status": "failed",
        "current_stage": "closed",
        "decision": "FAIL_IMPLEMENTATION",
        "registry_line": registry_line,
        "process_gpu": snapshot,
        "forbidden_scope_counts": FORBIDDEN_SCOPE_COUNTS,
        **FIXED_STATUS,
    }
    write_json(R69 / "final_summary.json", final)
    RESULT.write_text(
        f"# {EXP_NAME} — Fail-closed result\n\n"
        f"Decision: **FAIL_IMPLEMENTATION**\n\n"
        f"Error: `{repr(error)}`\n\n"
        "No training, checkpoint modification, tracker, TrackEval, HOTA, raw GT, MOT20 test, or next-policy run occurred. HOTA is null and next_policy_authorized=false.\n\n"
        "Notion writeback was not executed.\n",
        encoding="utf-8",
    )
    append_event("experiment_failed_closed", error=repr(error))


def run_all() -> None:
    command_init()
    command_verify_inputs()
    command_audit_objective_population()
    command_inventory_checkpoints()
    command_evaluate_retained_checkpoints()
    command_audit_gate_score_semantics()
    command_diagnose()
    command_validate()
    command_summarize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["self-test", "preflight", "run"])
    arguments = parser.parse_args()
    if arguments.command == "self-test":
        print(json.dumps(self_test(), indent=2, sort_keys=True))
        return
    if arguments.command == "preflight":
        report = command_preflight()
        print(json.dumps(report, indent=2, sort_keys=True, default=json_default))
        if not report["all_passed"]:
            raise SystemExit(2)
        return
    try:
        run_all()
    except BaseException as error:
        fail_close(error)
        raise


if __name__ == "__main__":
    main()
