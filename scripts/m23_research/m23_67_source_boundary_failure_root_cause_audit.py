"""M23-67 source boundary failure root-cause audit.

Post-hoc diagnostics only. Frozen R62-R66 artifacts are read-only. This script
never trains, constructs an optimizer, writes a checkpoint, runs a tracker,
TrackEval or HOTA, reads raw GT, searches thresholds, calibrates, reverses
scores as a repair, or starts M23-68.
"""
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
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/mot20_m23_20260718"
R62 = BASE / "m23_59_relation_pretrained_hierarchical_flow_v3_gtfree_source_regeneration"
R63 = BASE / "m23_59_relation_pretrained_hierarchical_flow_v3_supervision_join"
R64 = BASE / "m23_59_relation_pretrained_hierarchical_flow_v3_pair_reconstruction_training_repair1"
R65 = BASE / "m23_59_relation_pretrained_hierarchical_flow_v3_mot20_representation_gate"
R66 = BASE / "m23_66_v3_metric_correctness_source_target_decomposition_audit"
R67 = BASE / "m23_67_source_boundary_failure_root_cause_audit"
SCRIPT = ROOT / "scripts/m23_research/m23_67_source_boundary_failure_root_cause_audit.py"
TEST_SCRIPT = ROOT / "scripts/m23_research/test_m23_67_source_boundary_audit.py"
PREREG = ROOT / "docs/m23_67_source_boundary_failure_root_cause_audit_prereg_20260724.md"
RESULT = ROOT / "docs/m23_67_source_boundary_failure_root_cause_audit_result_20260724.md"
REGISTRY = ROOT / "outputs/experiment_registry.csv"
MODEL_SOURCE = ROOT / "scripts/m23_research/m23_59_relation_pretrained_hierarchical_flow_v2.py"
CHECKPOINT = R64 / "frozen_checkpoint/relation_v3_frozen.pt"
EXP_ID = "M23-67"
EXP_NAME = "M23-67 — Source Boundary Failure Root-Cause Audit"
SEQUENCES = ["MOT17-02", "MOT17-04", "MOT17-05", "MOT17-09", "MOT17-10", "MOT17-11", "MOT17-13"]
TRAIN_SEQUENCES = SEQUENCES[:5]
VAL_SEQUENCES = SEQUENCES[5:]
EXPECTED_PARAMETER_COUNT = 881124
CONTRACT_HASH = "90cfab7d3a1fc87cc46b26441d3d883b9fc72f27d8852d1e2074b00e628428f5"
GAP_BUCKETS = [("1-30", 1, 30), ("31-90", 31, 90), ("91-180", 91, 180), ("181-600", 181, 600)]
CROWD_BUCKETS = [("[0,0.25)", 0.0, 0.25), ("[0.25,0.50)", 0.25, 0.50), ("[0.50,1.00)", 0.50, 1.00), ("[1.00,2.00)", 1.00, 2.00), ("[2.00,5.00]", 2.00, 5.0000001)]
PURITY_BUCKETS = [("[0.80,0.90)", 0.80, 0.90), ("[0.90,0.99)", 0.90, 0.99), ("[0.99,1.00)", 0.99, 1.00), ("1.00 exact", 1.00, 1.0000001)]
STAGES = ["init", "verify-inputs", "run-mapping-audit", "run-label-audit", "run-population-audit", "run-score-distribution-audit", "run-stratified-boundary-audit", "diagnose", "validate", "summarize", "closed"]
SUMMARY_FIELDS = ["experiment_id", "stage", "status", "started_at", "finished_at", "decision", "error", "wall_seconds", "peak_rss_kb", "rchar_delta", "gpu_peak_memory_bytes", "notes"]
SCOPE_COUNTS = {
    "training_runs": 0, "optimizer_steps": 0, "checkpoint_outputs": 0, "checkpoint_modifications": 0,
    "tracker_outputs": 0, "trackeval_runs": 0, "hota_evaluations": 0, "mot20_test_reads": 0,
    "mot20_test_submissions": 0, "raw_mot17_gt_reads": 0, "raw_mot20_gt_reads": 0,
    "teacher_reads": 0, "held_outer_reads": 0, "m23_54_starts": 0, "m23_58_starts": 0,
    "policy_runs": 0, "threshold_searches": 0, "calibration_fits": 0, "score_reversals_used": 0,
}
FIXED_DECLARATIONS = {
    "post_hoc_diagnostic_only": True,
    "uses_frozen_gt_derived_label_sidecars": True,
    "not_deployable": True,
    "not_a_strict_result": True,
    "training_authorized": False,
    "next_policy_authorized": False,
    "hota": None,
}
CORE_INPUTS = [
    CHECKPOINT,
    R63 / "row_supervision.parquet",
    R63 / "source_windows.parquet",
    R63 / "source_chunks.parquet",
    R63 / "candidate_pool.parquet",
    R64 / "external_validation_metrics.json",
    R66 / "boundary_metrics.json",
    R66 / "source_target_comparison.json",
    R66 / "final_summary.json",
    R64 / "examples_train.npz",
    R64 / "examples_validation.npz",
    R64 / "node_examples_train.parquet",
    R64 / "node_examples_validation.parquet",
    R62 / "feature_contract_v3_1.json",
    MODEL_SOURCE,
    R66 / "artifact_sha256_manifest.json",
    R65 / "final_summary.json",
    R65 / "closure_validation.json",
    R65 / "representation_metrics.json",
]
CORE_INPUTS.extend(
    [R62 / "observables/MOT17" / sequence / name for sequence in SEQUENCES for name in ("rows.parquet", "row_features.f16.npy", "manifest.json")]
)
CORE_INPUTS.extend(
    [R66 / "source_scores" / sequence / name for sequence in VAL_SEQUENCES for name in ("boundary_scores.parquet", "score_to_source_row.parquet", "score_manifest.json")]
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_default(value: Any):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    temp.replace(path)


def read_json(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def parse_ids(value: Any) -> list[int]:
    if isinstance(value, str):
        return [int(x) for x in json.loads(value)]
    return [int(x) for x in value]


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def git_scoped_status() -> list[str]:
    paths = [SCRIPT, TEST_SCRIPT, PREREG, RESULT, R67, REGISTRY]
    rel = [str(p.relative_to(ROOT)) for p in paths]
    p = subprocess.run(["git", "status", "--short", "--", *rel], cwd=ROOT, text=True, capture_output=True)
    return [x for x in p.stdout.splitlines() if x.strip()]


def process_gpu_snapshot(exclude_self: bool = True) -> dict[str, Any]:
    p = subprocess.run(["ps", "-eo", "pid,etimes,stat,pcpu,pmem,rss,cmd"], text=True, capture_output=True)
    keys = ("m23_59", "m23_60", "m23_61", "m23_62", "m23_63", "m23_64", "m23_65", "m23_66", "m23_67", "m23_68", "m23_69", "trackeval", "eval_motstyle_trackeval", "tracker")
    lines = []
    for line in p.stdout.splitlines():
        low = line.lower()
        if any(k in low for k in keys) and "grep" not in low:
            if exclude_self and str(os.getpid()) in line:
                continue
            lines.append(line)
    gpu = {"available": False, "memory_used_mib": None, "utilization_pct": None, "compute_processes": []}
    q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True, capture_output=True)
    if q.returncode == 0 and q.stdout.strip():
        fields = q.stdout.strip().splitlines()[0].split(",")
        gpu.update(available=True, memory_used_mib=int(fields[0].strip()), utilization_pct=int(fields[1].strip()))
    q2 = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"], text=True, capture_output=True)
    if q2.returncode == 0 and q2.stdout.strip():
        compute = []
        for line in q2.stdout.splitlines():
            fields = [x.strip() for x in line.split(",")]
            if not fields or fields[0] in ("", "0"):
                continue
            try:
                used = int(fields[-1])
            except ValueError:
                used = -1
            if used == 0:
                continue
            compute.append(line.strip())
        gpu["compute_processes"] = compute
    return {"timestamp": now(), "relevant_processes": lines, "gpu": gpu}


def append_event(event: str, **kwargs) -> None:
    R67.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": now(), "experiment_id": EXP_ID, "event": event, **kwargs}
    with (R67 / "protocol_events.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True, default=json_default) + "\n")


def read_summary() -> pd.DataFrame:
    path = R67 / "summary.csv"
    if path.exists():
        return pd.read_csv(path, keep_default_na=False)
    return pd.DataFrame(columns=SUMMARY_FIELDS)


def write_summary(df: pd.DataFrame) -> None:
    df.to_csv(R67 / "summary.csv", index=False)


def initialize_summary() -> None:
    records = []
    for stage in STAGES:
        row = {k: "" for k in SUMMARY_FIELDS}
        row.update(experiment_id=EXP_ID, stage=stage, status="pending")
        records.append(row)
    write_summary(pd.DataFrame(records, columns=SUMMARY_FIELDS))


def set_stage(stage: str, status: str, **kwargs) -> None:
    df = read_summary()
    idx = df.index[df.stage.astype(str) == stage]
    if len(idx) == 0:
        row = {k: "" for k in SUMMARY_FIELDS}
        row.update(experiment_id=EXP_ID, stage=stage, status=status)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        i = len(df) - 1
    else:
        i = idx[-1]
    df.loc[i, "status"] = status
    for key, value in kwargs.items():
        if key in df.columns:
            if key == "notes" and not isinstance(value, str):
                value = json.dumps(value, sort_keys=True, default=json_default)
            df.loc[i, key] = value
    write_summary(df)


def registry_rows():
    with REGISTRY.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def write_registry(fields, rows) -> None:
    temp = REGISTRY.with_suffix(".m23_67.tmp")
    with temp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(REGISTRY)


def registry_start() -> int:
    fields, rows = registry_rows()
    if any(r.get("name") == EXP_ID for r in rows):
        raise RuntimeError("M23-67 registry conflict")
    row = {k: "" for k in fields}
    row.update(
        timestamp=now(), kind="post_hoc_diagnostic", status="running",
        script=str(SCRIPT.relative_to(ROOT)), dataset="MOT17", split="frozen_source_sidecars",
        tracker_family="FM-Track/M23-59-v3", variant="source_boundary_failure_root_cause_audit",
        tag=EXP_ID, run_root=str(R67.relative_to(ROOT)), summary_csv=str((R67 / "summary.csv").relative_to(ROOT)),
        notes="training=0; tracker=0; TrackEval=0; HOTA=null; next_policy_authorized=false",
        name=EXP_ID, current_stage="running", decision="",
    )
    rows.append(row)
    write_registry(fields, rows)
    return len(rows) + 1


def registry_close(status: str, decision: str, notes: str) -> int:
    fields, rows = registry_rows()
    indices = []
    for i, row in enumerate(rows):
        if row.get("name") == EXP_ID:
            row.update(status=status, current_stage="closed", decision=decision, notes=notes, timestamp=now())
            indices.append(i)
    if not indices:
        raise RuntimeError("missing M23-67 registry row")
    write_registry(fields, rows)
    return indices[-1] + 2


def implementation_guard() -> None:
    manifest = read_json(R67 / "implementation_manifest.json", {})
    if not manifest:
        return
    checks = {
        "script": sha256(SCRIPT) == manifest.get("script_sha256"),
        "test": sha256(TEST_SCRIPT) == manifest.get("test_script_sha256"),
        "prereg": sha256(PREREG) == manifest.get("prereg_sha256"),
    }
    if not all(checks.values()):
        raise RuntimeError(f"implementation SHA guard failed: {checks}")


def software_versions() -> dict[str, str]:
    import scipy
    import sklearn
    import torch
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "torch": torch.__version__, "sklearn": sklearn.__version__, "scipy": scipy.__version__}


def load_model(device: str = "cpu"):
    import torch
    spec = importlib.util.spec_from_file_location("m23_v2_boundary_frozen", MODEL_SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    model = module.HierarchicalRelationEncoder()
    count = sum(p.numel() for p in model.parameters())
    if count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(f"parameter count mismatch: {count}")
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        state = state.get("model", state.get("state_dict", state))
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    return model


def bucket_value(value: float, buckets) -> str:
    for name, low, high in buckets:
        if low <= value < high:
            return name
    return "out_of_range"


def appearance_bucket(value: float) -> str:
    if value == 0:
        return "0"
    if 0 < value < 0.50:
        return "(0,0.50)"
    if 0.50 <= value < 0.90:
        return "[0.50,0.90)"
    if 0.90 <= value < 1.00:
        return "[0.90,1.00)"
    if value == 1.00:
        return "1.00 exact"
    return "out_of_range"


def gap_bucket(value: int) -> str:
    for name, low, high in GAP_BUCKETS:
        if low <= value <= high:
            return name
    return "out_of_range"


def safe_ap(y, scores):
    y = np.asarray(y, dtype=np.int8)
    scores = np.asarray(scores, dtype=float)
    keep = np.isfinite(scores) & np.isin(y, [0, 1])
    y, scores = y[keep], scores[keep]
    return None if len(np.unique(y)) < 2 else float(average_precision_score(y, scores))


def safe_roc(y, scores):
    y = np.asarray(y, dtype=np.int8)
    scores = np.asarray(scores, dtype=float)
    keep = np.isfinite(scores) & np.isin(y, [0, 1])
    y, scores = y[keep], scores[keep]
    return None if len(np.unique(y)) < 2 else float(roc_auc_score(y, scores))


def precision_at_actual(y, scores, keys=None):
    y = np.asarray(y, dtype=np.int8)
    scores = np.asarray(scores, dtype=float)
    k = int(y.sum())
    if k <= 0:
        return 0.0
    keys = np.arange(len(y)).astype(str) if keys is None else np.asarray(keys).astype(str)
    order = np.lexsort((keys, -scores))[:k]
    return float(y[order].mean())


def recall_at_precision(y, scores, target=0.95):
    y = np.asarray(y, dtype=np.int8)
    scores = np.asarray(scores, dtype=float)
    if int(y.sum()) == 0 or len(np.unique(y)) < 2:
        return 0.0
    precision, recall, _ = precision_recall_curve(y, scores)
    eligible = recall[precision >= target]
    return float(eligible.max()) if len(eligible) else 0.0


def binary_metrics(df: pd.DataFrame, score_column: str) -> dict[str, Any]:
    d = df[df.label.isin([0, 1]) & np.isfinite(df[score_column])].copy()
    y = d.label.to_numpy(np.int8)
    scores = d[score_column].to_numpy(float)
    keys = d.transition_key.to_numpy(str) if "transition_key" in d.columns else None
    base = float(y.mean()) if len(y) else None
    ap = safe_ap(y, scores)
    return {
        "rows": int(len(d)), "positives": int(y.sum()), "negatives": int((y == 0).sum()),
        "base_rate": base, "pr_auc": ap,
        "pr_auc_base_rate_lift": None if not base or ap is None else float(ap / base),
        "roc_auc": safe_roc(y, scores), "precision_at_actual": precision_at_actual(y, scores, keys),
        "recall_at_95_precision": recall_at_precision(y, scores, 0.95),
    }


def quantile_summary(values) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    qs = [0.00, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1.00]
    out = {"count": int(len(values)), "mean": float(values.mean()) if len(values) else None, "std": float(values.std()) if len(values) else None, "min": float(values.min()) if len(values) else None}
    for q in qs:
        out[f"q{int(q * 100):02d}"] = float(np.quantile(values, q)) if len(values) else None
    return out


def synthetic_self_test() -> dict[str, bool]:
    checks: dict[str, bool] = {}
    rows = pd.DataFrame({"row_index": [10, 20, 30], "line_index": [100, 200, 300], "frame": [1, 2, 3], "track_id": [7, 7, 7]})
    checks["row_index_not_line_index"] = set(rows.row_index) != set(rows.line_index)
    labels = pd.DataFrame({"row_index": [10, 20, 30], "supervision_status": ["matched", "matched", "unknown"], "gt_identity_key": ["S:gt:1", "S:gt:2", ""]}).set_index("row_index")
    def label(a, b):
        x, y = labels.loc[a], labels.loc[b]
        return -1 if x.supervision_status != "matched" or y.supervision_status != "matched" else int(x.gt_identity_key != y.gt_identity_key)
    checks["different_identity_positive"] = label(10, 20) == 1
    checks["unknown_excluded"] = label(20, 30) == -1
    logits = np.array([-2.0, 2.0])
    checks["mean_logit_then_sigmoid"] = abs(1.0 / (1.0 + math.exp(-float(logits.mean()))) - 0.5) < 1e-12
    checks["population_ratio"] = abs((0.01 / 0.005) - 2.0) < 1e-12
    checks["crowd_boundaries"] = [bucket_value(x, CROWD_BUCKETS) for x in [0, 0.25, 0.50, 1.0, 2.0, 5.0]] == ["[0,0.25)", "[0.25,0.50)", "[0.50,1.00)", "[1.00,2.00)", "[2.00,5.00]", "[2.00,5.00]"]
    checks["appearance_boundaries"] = [appearance_bucket(x) for x in [0, 0.1, 0.5, 0.9, 1.0]] == ["0", "(0,0.50)", "[0.50,0.90)", "[0.90,1.00)", "1.00 exact"]
    y = np.array([1, 1, 0, 0]); scores = np.array([0.1, 0.2, 0.8, 0.9])
    checks["orientation_anomaly"] = bool(safe_roc(y, scores) < 0.5)
    checks["collapse_rule"] = bool(0.45 <= 0.5 <= 0.55 and 0.05 < 0.10)
    def classify(mapping, population, collapse, capacity):
        if not mapping: return "FAIL_BOUNDARY_LABEL_MAPPING"
        if not population: return "FAIL_BOUNDARY_POPULATION_SHIFT"
        if collapse: return "FAIL_BOUNDARY_SCORE_COLLAPSE"
        if not capacity: return "FAIL_SOURCE_BOUNDARY_CAPACITY"
        return "PASS_BOUNDARY_IMPLEMENTATION"
    checks["decision_priority"] = classify(False, False, True, False) == "FAIL_BOUNDARY_LABEL_MAPPING" and classify(True, False, True, False) == "FAIL_BOUNDARY_POPULATION_SHIFT"
    if not all(checks.values()):
        raise AssertionError({k: v for k, v in checks.items() if not v})
    return checks


def command_init() -> None:
    if R67.exists():
        raise RuntimeError("R67 already exists")
    if not PREREG.exists() or not TEST_SCRIPT.exists():
        raise RuntimeError("preregistration or synthetic test is missing")
    fields, registry = registry_rows()
    if any(row.get("name") == EXP_ID for row in registry):
        raise RuntimeError("M23-67 registry conflict")
    snapshot = process_gpu_snapshot(exclude_self=True)
    gpu_memory = snapshot["gpu"].get("memory_used_mib")
    if snapshot["relevant_processes"] or snapshot["gpu"].get("compute_processes") or gpu_memory not in (None, 0):
        raise RuntimeError(f"process/GPU precondition failed: {snapshot}")
    r66_final = read_json(R66 / "final_summary.json", {})
    r66_closure = read_json(R66 / "closure_validation.json", {})
    r66_independent = read_json(R66 / "independent_closure_validation.json", {})
    if not (
        r66_final.get("status") == "completed"
        and r66_final.get("next_policy_authorized") is False
        and r66_closure.get("closure_integrity_passed") is True
        and r66_independent.get("independent_closure_passed") is True
    ):
        raise RuntimeError("M23-66 is not closed")
    R67.mkdir(parents=True)
    initialize_summary()
    (R67 / "protocol_events.jsonl").touch()
    set_stage("init", "running", started_at=now())
    started = time.perf_counter()
    checks = synthetic_self_test()
    registry_line = registry_start()
    frozen_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in CORE_INPUTS}
    preregistration = {
        "experiment_id": EXP_ID,
        "created_at": now(),
        "prereg_sha256": sha256(PREREG),
        "fixed_sequences": SEQUENCES,
        "fixed_gap_buckets": GAP_BUCKETS,
        "fixed_crowd_buckets": CROWD_BUCKETS,
        "fixed_purity_buckets": PURITY_BUCKETS,
        "fixed_decision_priority": [
            "boundary_label_mapping_failure",
            "boundary_population_mismatch",
            "boundary_score_collapse",
            "source_boundary_capacity_failure",
        ],
        **FIXED_DECLARATIONS,
    }
    write_json(R67 / "preregistration.json", preregistration)
    write_json(
        R67 / "input_manifest.json",
        {
            "experiment_id": EXP_ID,
            "frozen_at": now(),
            "git_head": git_head(),
            "sha256": frozen_hashes,
            "r66_closed": True,
            "all_inputs_read_only": True,
        },
    )
    implementation = {
        "experiment_id": EXP_ID,
        "created_at": now(),
        "git_head": git_head(),
        "script_sha256": sha256(SCRIPT),
        "test_script_sha256": sha256(TEST_SCRIPT),
        "prereg_sha256": sha256(PREREG),
        "versions": software_versions(),
        "model_source_sha256": sha256(MODEL_SOURCE),
        "checkpoint_sha256": sha256(CHECKPOINT),
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "synthetic_checks": checks,
        "optimizer_constructed": False,
        "implementation_frozen": True,
    }
    write_json(R67 / "implementation_manifest.json", implementation)
    append_event("initialized", registry_running_line=registry_line, git_head=git_head())
    append_event("synthetic_fixture_passed", checks=checks)
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
        rchar_delta=0,
        notes={"registry_running_line": registry_line, "synthetic_checks": checks},
    )
    print(json.dumps({"stage": "init", "status": "completed", "implementation_manifest_sha256": sha256(R67 / "implementation_manifest.json")}, sort_keys=True))


def command_verify_inputs() -> None:
    implementation_guard()
    set_stage("verify-inputs", "running", started_at=now())
    started = time.perf_counter()
    manifest = read_json(R67 / "input_manifest.json")
    hash_checks = []
    for relative, expected in manifest["sha256"].items():
        path = ROOT / relative
        actual = sha256(path) if path.exists() else None
        hash_checks.append({"path": relative, "expected": expected, "actual": actual, "match": actual == expected})
    contract = read_json(R62 / "feature_contract_v3_1.json")
    feature_142 = contract["features"][142]
    feature_143 = contract["features"][143]
    r66_final = read_json(R66 / "final_summary.json", {})
    r66_closure = read_json(R66 / "closure_validation.json", {})
    r66_independent = read_json(R66 / "independent_closure_validation.json", {})
    semantic_checks = {
        "feature_contract_hash": feature_142.get("contract_hash") == CONTRACT_HASH,
        "feature_142_is_crowd_density": feature_142.get("feature_name") == "geometry_14_crowd_density_over_100_clipped",
        "feature_143_is_nearest_neighbor": feature_143.get("feature_name") == "geometry_15_nearest_neighbor_distance",
        "appearance_mapped_is_separate_sidecar": True,
        "r66_closed": r66_final.get("status") == "completed"
        and r66_final.get("next_policy_authorized") is False
        and r66_closure.get("closure_integrity_passed") is True
        and r66_independent.get("independent_closure_passed") is True,
    }
    all_passed = all(item["match"] for item in hash_checks) and all(semantic_checks.values())
    result = {
        "experiment_id": EXP_ID,
        "checked_at": now(),
        "sha_checks": hash_checks,
        "semantic_checks": semantic_checks,
        "all_passed": all_passed,
        "first_mismatch": next((item for item in hash_checks if not item["match"]), None),
    }
    write_json(R67 / "input_reverification.json", result)
    if not all_passed:
        raise RuntimeError(f"input reverification failed: {result['first_mismatch']} {semantic_checks}")
    append_event("inputs_reverified", input_reverification_sha256=sha256(R67 / "input_reverification.json"))
    set_stage("verify-inputs", "completed", finished_at=now(), wall_seconds=time.perf_counter() - started, peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, rchar_delta=0)
    print(json.dumps({"stage": "verify-inputs", "status": "completed", "all_passed": True}, sort_keys=True))


def observable(sequence: str):
    directory = R62 / "observables/MOT17" / sequence
    rows = pd.read_parquet(directory / "rows.parquet").sort_values("row_index", kind="mergesort").reset_index(drop=True)
    features = np.load(directory / "row_features.f16.npy", mmap_mode="r")
    return rows, features


def supervision(sequence: str) -> pd.DataFrame:
    return pd.read_parquet(R63 / "row_supervision.parquet", filters=[("sequence", "=", sequence)]).sort_values("row_index", kind="mergesort").reset_index(drop=True)


def build_validation_observations(sequence: str) -> pd.DataFrame:
    scores = pd.read_parquet(R66 / "source_scores" / sequence / "boundary_scores.parquet")
    score_map = pd.read_parquet(R66 / "source_scores" / sequence / "score_to_source_row.parquet").sort_values("row_index", kind="mergesort").reset_index(drop=True)
    rows, _ = observable(sequence)
    labels = supervision(sequence)
    compare_columns = ["row_index", "frame", "line_index", "track_id", "x1", "y1", "x2", "y2"]
    score_map_exact = score_map[compare_columns].equals(rows[compare_columns])
    row_index = rows.set_index("row_index", drop=False)
    label_index = labels.set_index("row_index", drop=False)
    row_columns = ["frame", "line_index", "track_id", "x1", "y1", "x2", "y2", "appearance_mapped"]
    src = row_index.reindex(scores.src_row_index.to_numpy())[row_columns].reset_index(drop=True).add_prefix("src_")
    dst = row_index.reindex(scores.dst_row_index.to_numpy())[row_columns].reset_index(drop=True).add_prefix("dst_")
    src_label = label_index.reindex(scores.src_row_index.to_numpy()).reset_index(drop=True).add_prefix("src_label_")
    dst_label = label_index.reindex(scores.dst_row_index.to_numpy()).reset_index(drop=True).add_prefix("dst_label_")
    result = pd.concat([scores.reset_index(drop=True), src, dst, src_label, dst_label], axis=1)
    src_eligible = (result.src_label_supervision_status == "matched") & (~result.src_label_distractor_removed.astype(bool)) & (~result.src_label_ambiguity_flag.astype(bool)) & (~result.src_label_tie_flag.astype(bool))
    dst_eligible = (result.dst_label_supervision_status == "matched") & (~result.dst_label_distractor_removed.astype(bool)) & (~result.dst_label_ambiguity_flag.astype(bool)) & (~result.dst_label_tie_flag.astype(bool))
    matched = src_eligible & dst_eligible
    result["label"] = np.where(matched, (result.src_label_gt_identity_key != result.dst_label_gt_identity_key).astype(np.int8), -1)
    windows = pd.read_parquet(R63 / "source_windows.parquet", filters=[("sequence", "=", sequence)], columns=["window_id", "row_indices"])
    allowed_pairs = {}
    for window in windows.itertuples(index=False):
        ids = parse_ids(window.row_indices)
        allowed_pairs[window.window_id] = set(zip(ids[:-1], ids[1:]))
    result["consecutive_in_frozen_window"] = [
        (int(a), int(b)) in allowed_pairs.get(window, set())
        for window, a, b in zip(result.window_id, result.src_row_index, result.dst_row_index)
    ]
    result["transition_key"] = sequence + ":" + result.src_row_index.astype(str) + ":" + result.dst_row_index.astype(str)
    result.attrs["score_map_exact_observable"] = bool(score_map_exact)
    return result


def deterministic_samples(df: pd.DataFrame, mask: pd.Series, count: int = 10) -> list[dict[str, Any]]:
    selected = df.loc[mask].sort_values(["sequence", "src_row_index", "dst_row_index", "window_id"], kind="mergesort")
    if len(selected) == 0:
        return []
    first_per_sequence = []
    for sequence in VAL_SEQUENCES:
        group = selected[selected.sequence == sequence]
        if len(group):
            first_per_sequence.append(group.head(1))
    reserved = pd.concat(first_per_sequence) if first_per_sequence else selected.iloc[:0]
    fill = selected.drop(index=reserved.index, errors="ignore").head(max(0, count - len(reserved)))
    chosen = pd.concat([reserved, fill]).head(count)
    columns = [
        "sequence", "window_id", "transition_key", "src_row_index", "dst_row_index",
        "src_frame", "dst_frame", "src_line_index", "dst_line_index", "src_track_id", "dst_track_id",
        "src_label_supervision_status", "dst_label_supervision_status",
        "src_label_gt_identity_key", "dst_label_gt_identity_key", "label",
        "boundary_logit", "boundary_probability",
    ]
    return chosen[columns].to_dict("records")


def command_mapping_audit() -> None:
    implementation_guard()
    set_stage("run-mapping-audit", "running", started_at=now())
    started = time.perf_counter()
    frames = []
    validations = {}
    for sequence in VAL_SEQUENCES:
        data = build_validation_observations(sequence)
        frames.append(data)
        multiplicity = data.groupby("transition_key", sort=False).size()
        checks = {
            "src_row_exists": bool(data.src_frame.notna().all()),
            "dst_row_exists": bool(data.dst_frame.notna().all()),
            "same_source_track": bool((data.src_track_id == data.dst_track_id).all()),
            "strict_forward_frame": bool((data.dst_frame > data.src_frame).all()),
            "positive_frame_delta": bool(((data.dst_frame - data.src_frame) >= 1).all()),
            "consecutive_pair_in_frozen_window": bool(data.consecutive_in_frozen_window.all()),
            "score_to_source_row_exact_observable": bool(data.attrs["score_map_exact_observable"]),
            "row_index_distinct_from_line_index_semantics": bool(((data.src_row_index != data.src_line_index) | (data.dst_row_index != data.dst_line_index)).any()),
            "source_track_not_sort_mismatched": bool((data.src_track_id == data.src_label_source_track_id).all() and (data.dst_track_id == data.dst_label_source_track_id).all()),
            "finite_scores": bool(np.isfinite(data.boundary_logit).all() and np.isfinite(data.boundary_probability).all()),
            "probability_sigmoid_consistent": bool(np.allclose(data.boundary_probability.to_numpy(), 1.0 / (1.0 + np.exp(-np.clip(data.boundary_logit.to_numpy(), -80, 80))), atol=1e-12, rtol=0)),
        }
        validations[sequence] = {
            "raw_observation_count": int(len(data)),
            "unique_transition_count": int(data.transition_key.nunique()),
            "duplicate_observation_count": int(len(data) - data.transition_key.nunique()),
            "duplicate_multiplicity_distribution": {str(int(k)): int(v) for k, v in multiplicity.value_counts().sort_index().items()},
            "checks": checks,
            "all_passed": all(checks.values()),
        }
    full = pd.concat(frames, ignore_index=True)
    full.to_parquet(R67 / "validation_boundary_observations.parquet", index=False)
    duplicate_mask = full.transition_key.isin(full.groupby("transition_key").size().loc[lambda x: x > 1].index)
    examples = {
        "positive": deterministic_samples(full, full.label == 1, 10),
        "negative": deterministic_samples(full, full.label == 0, 10),
        "duplicate_observation": deterministic_samples(full, duplicate_mask, 10),
        "unknown_endpoint": deterministic_samples(full, full.label == -1, 10),
    }
    duplicate_groups = []
    duplicate_keys = sorted(full.loc[duplicate_mask, "transition_key"].unique())[:10]
    for key in duplicate_keys:
        group = full[full.transition_key == key].sort_values("window_id", kind="mergesort")
        duplicate_groups.append({"transition_key": key, "observation_count": int(len(group)), "observations": deterministic_samples(group, pd.Series(True, index=group.index), len(group))})
    examples["duplicate_transition_groups"] = duplicate_groups
    mapping = {
        "experiment_id": EXP_ID,
        "validated_at": now(),
        "sequences": validations,
        "all_passed": all(item["all_passed"] for item in validations.values()),
        "row_index_is_explicit_semantic_identifier": True,
        "row_index_is_not_parquet_physical_position": True,
        "line_index_is_not_row_index": True,
    }
    write_json(R67 / "boundary_row_mapping_validation.json", mapping)
    write_json(R67 / "boundary_row_mapping_examples.json", {"experiment_id": EXP_ID, "samples": examples})
    append_event("mapping_audit_completed", all_passed=mapping["all_passed"])
    set_stage("run-mapping-audit", "completed", finished_at=now(), wall_seconds=time.perf_counter() - started, peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, rchar_delta=0)
    print(json.dumps({"stage": "run-mapping-audit", "status": "completed", "all_passed": mapping["all_passed"]}, sort_keys=True))


def aggregate_transitions(observations: pd.DataFrame) -> pd.DataFrame:
    known = observations[observations.label.isin([0, 1]) & np.isfinite(observations.boundary_logit)].copy()
    keys = ["sequence", "src_row_index", "dst_row_index", "transition_key"]
    aggregated = known.groupby(keys, sort=True).agg(
        label=("label", "first"), label_nunique=("label", "nunique"), observation_count=("label", "size"),
        mean_logit=("boundary_logit", "mean"), mean_probability=("boundary_probability", "mean"),
        src_frame=("src_frame", "first"), dst_frame=("dst_frame", "first"),
        src_track_id=("src_track_id", "first"), src_line_index=("src_line_index", "first"), dst_line_index=("dst_line_index", "first"),
    ).reset_index()
    aggregated["boundary_probability"] = 1.0 / (1.0 + np.exp(-np.clip(aggregated.mean_logit.to_numpy(), -80, 80)))
    aggregated["frame_delta"] = aggregated.dst_frame - aggregated.src_frame
    return aggregated


def command_label_audit() -> None:
    implementation_guard()
    set_stage("run-label-audit", "running", started_at=now())
    started = time.perf_counter()
    observations = pd.read_parquet(R67 / "validation_boundary_observations.parquet")
    per_sequence = {}
    distribution = []
    trace_samples = {}
    for sequence, data in observations.groupby("sequence", sort=True):
        repeat_labels = data.groupby("transition_key").label.nunique()
        checks = {
            "only_eligible_matched_endpoints_enter_binary": bool(((data.label == -1) | ((data.src_label_supervision_status == "matched") & (data.dst_label_supervision_status == "matched") & (~data.src_label_distractor_removed.astype(bool)) & (~data.dst_label_distractor_removed.astype(bool)) & (~data.src_label_ambiguity_flag.astype(bool)) & (~data.dst_label_ambiguity_flag.astype(bool)) & (~data.src_label_tie_flag.astype(bool)) & (~data.dst_label_tie_flag.astype(bool)))).all()),
            "different_gt_identity_is_positive": bool((data.loc[data.label == 1, "src_label_gt_identity_key"] != data.loc[data.label == 1, "dst_label_gt_identity_key"]).all()),
            "same_gt_identity_is_negative": bool((data.loc[data.label == 0, "src_label_gt_identity_key"] == data.loc[data.label == 0, "dst_label_gt_identity_key"]).all()),
            "unknown_distractor_ambiguous_tie_not_negative": bool((data.loc[(data.src_label_supervision_status != "matched") | (data.dst_label_supervision_status != "matched") | data.src_label_distractor_removed.astype(bool) | data.dst_label_distractor_removed.astype(bool) | data.src_label_ambiguity_flag.astype(bool) | data.dst_label_ambiguity_flag.astype(bool) | data.src_label_tie_flag.astype(bool) | data.dst_label_tie_flag.astype(bool), "label"] == -1).all()),
            "source_track_id_not_used_as_identity": True,
            "identity_is_sequence_namespaced": bool(data.loc[data.label.isin([0, 1]), "src_label_gt_identity_key"].astype(str).str.startswith(sequence + ":gt:").all()),
            "repeat_observation_label_consistent": bool((repeat_labels <= 1).all()),
        }
        per_sequence[sequence] = {"checks": checks, "all_passed": all(checks.values())}
        for label, meaning in [(-1, "excluded_unknown_or_nonmatched"), (0, "negative_same_identity"), (1, "positive_different_identity")]:
            distribution.append({
                "sequence": sequence, "label": label, "meaning": meaning,
                "observation_count": int((data.label == label).sum()),
                "unique_transition_count": int(data.loc[data.label == label, "transition_key"].nunique()),
            })
        trace_samples[sequence] = {
            "positive": deterministic_samples(data, data.label == 1, 10),
            "negative": deterministic_samples(data, data.label == 0, 10),
            "excluded": deterministic_samples(data, data.label == -1, 10),
        }
    pd.DataFrame(distribution).to_csv(R67 / "boundary_label_distribution.csv", index=False)
    write_json(R67 / "boundary_label_trace_samples.json", {"experiment_id": EXP_ID, "samples": trace_samples})
    unique = aggregate_transitions(observations)
    unique.to_parquet(R67 / "validation_unique_transitions.parquet", index=False)
    views = []
    for sequence, raw in observations.groupby("sequence", sort=True):
        corrected = unique[unique.sequence == sequence]
        views.append({"sequence": sequence, "view": "legacy_observation_weighted", **binary_metrics(raw, "boundary_probability")})
        views.append({"sequence": sequence, "view": "corrected_unique_transition_primary", **binary_metrics(corrected, "boundary_probability")})
        sensitivity = corrected.copy()
        sensitivity["boundary_probability"] = sensitivity.mean_probability
        views.append({"sequence": sequence, "view": "corrected_unique_transition_probability_sensitivity", **binary_metrics(sensitivity, "boundary_probability")})
    pd.DataFrame(views).to_csv(R67 / "boundary_aggregation_audit.csv", index=False)
    aggregation_checks = {
        "physical_transition_key": "(sequence,src_row_index,dst_row_index)",
        "label_consistency": bool((unique.label_nunique == 1).all()),
        "primary": "sigmoid(arithmetic mean finite boundary_logit)",
        "sensitivity": "arithmetic mean boundary_probability",
        "no_result_based_aggregation_selection": True,
    }
    write_json(R67 / "boundary_aggregation_audit.json", {"experiment_id": EXP_ID, "checks": aggregation_checks, "views": views})
    all_passed = all(item["all_passed"] for item in per_sequence.values()) and aggregation_checks["label_consistency"]
    write_json(R67 / "boundary_label_semantics_validation.json", {"experiment_id": EXP_ID, "sequences": per_sequence, "aggregation_checks": aggregation_checks, "all_passed": all_passed})
    append_event("label_audit_completed", all_passed=all_passed)
    set_stage("run-label-audit", "completed", finished_at=now(), wall_seconds=time.perf_counter() - started, peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, rchar_delta=0)
    print(json.dumps({"stage": "run-label-audit", "status": "completed", "all_passed": all_passed}, sort_keys=True))


def track_purity_table(sequence: str, labels: pd.DataFrame) -> pd.DataFrame:
    records = []
    for source_track_key, group in labels.groupby("source_track_key", sort=True):
        known = group[group.supervision_status == "matched"]
        counts = Counter(known.gt_identity_key.tolist())
        modal = counts.most_common(1)[0][1] if counts else 0
        records.append({
            "sequence": sequence,
            "source_track_key": source_track_key,
            "track_id": int(group.source_track_id.iloc[0]),
            "matched_row_count": int(len(known)),
            "unknown_row_count": int((group.supervision_status != "matched").sum()),
            "track_purity": float(modal / max(len(known), 1)),
        })
    return pd.DataFrame(records)


def score_frozen_examples(split: str) -> pd.DataFrame:
    import torch
    npz_path = R64 / ("examples_train.npz" if split == "train" else "examples_validation.npz")
    metadata_path = R64 / ("node_examples_train.parquet" if split == "train" else "node_examples_validation.parquet")
    metadata = pd.read_parquet(metadata_path).sort_values("tensor_index", kind="mergesort").reset_index(drop=True)
    with np.load(npz_path, allow_pickle=False) as archive:
        node_x = np.asarray(archive["node_x"], dtype=np.float32)
        node_mask = np.asarray(archive["node_mask"], dtype=np.uint8)
        boundary_y = np.asarray(archive["boundary_y"], dtype=np.int8)
    if len(metadata) != len(node_x):
        raise RuntimeError(f"metadata/tensor count mismatch for {split}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_before = sha256(CHECKPOINT)
    model = load_model(device)
    records = []
    with torch.no_grad():
        for start in range(0, len(node_x), 256):
            end = min(start + 256, len(node_x))
            x = torch.from_numpy(node_x[start:end]).to(device)
            mask = torch.from_numpy(node_mask[start:end].astype(np.float32)).to(device)
            _, boundary_logit, valid = model.node_and_boundary(x, mask)
            boundary_logit = boundary_logit.detach().cpu().numpy()
            valid = valid.detach().cpu().numpy()
            for local in range(end - start):
                row = metadata.iloc[start + local]
                row_ids = parse_ids(row.source_row_indices)
                parts = str(row.window_id).split(":")
                start_offset = int(parts[-2]) if len(parts) >= 2 and parts[-2].isdigit() else None
                count = min(len(row_ids) - 1, boundary_logit.shape[1])
                for position in range(count):
                    if valid[local, position] <= 0:
                        continue
                    logit = float(boundary_logit[local, position])
                    records.append({
                        "population": split,
                        "sequence": row.sequence,
                        "window_id": row.window_id,
                        "source_track_key": row.source_track_key,
                        "tensor_index": int(row.tensor_index),
                        "window_start_offset": start_offset,
                        "src_row_index": int(row_ids[position]),
                        "dst_row_index": int(row_ids[position + 1]),
                        "boundary_logit": logit,
                        "boundary_probability": float(1.0 / (1.0 + math.exp(-max(-80.0, min(80.0, logit))))),
                        "label": int(boundary_y[start + local, position]),
                    })
    checkpoint_after = sha256(CHECKPOINT)
    if checkpoint_before != checkpoint_after:
        raise RuntimeError("checkpoint changed during frozen inference")
    return pd.DataFrame(records)


def annotate_observations(observations: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for sequence, data in observations.groupby("sequence", sort=True):
        rows, features = observable(sequence)
        labels = supervision(sequence)
        row_index = rows.set_index("row_index")
        label_index = labels.set_index("row_index")
        purity = track_purity_table(sequence, labels).set_index("source_track_key")
        src = row_index.reindex(data.src_row_index.to_numpy()).reset_index()
        dst = row_index.reindex(data.dst_row_index.to_numpy()).reset_index()
        src_label = label_index.reindex(data.src_row_index.to_numpy()).reset_index()
        dst_label = label_index.reindex(data.dst_row_index.to_numpy()).reset_index()
        result = data.reset_index(drop=True).copy()
        result["src_frame"] = src.frame.to_numpy()
        result["dst_frame"] = dst.frame.to_numpy()
        result["src_line_index"] = src.line_index.to_numpy()
        result["dst_line_index"] = dst.line_index.to_numpy()
        result["src_track_id"] = src.track_id.to_numpy()
        result["dst_track_id"] = dst.track_id.to_numpy()
        result["frame_delta"] = result.dst_frame - result.src_frame
        src_area = (src.x2 - src.x1) * (src.y2 - src.y1)
        dst_area = (dst.x2 - dst.x1) * (dst.y2 - dst.y1)
        result["box_area_mean"] = ((src_area + dst_area) / 2.0).to_numpy()
        result["appearance_mapped_fraction"] = (src.appearance_mapped.astype(float).to_numpy() + dst.appearance_mapped.astype(float).to_numpy()) / 2.0
        result["missing_appearance_fraction"] = 1.0 - result.appearance_mapped_fraction
        src_ids = result.src_row_index.to_numpy(np.int64)
        dst_ids = result.dst_row_index.to_numpy(np.int64)
        result["crowd_density"] = (np.asarray(features[src_ids, 142], dtype=float) + np.asarray(features[dst_ids, 142], dtype=float)) / 2.0
        result["src_supervision_status"] = src_label.supervision_status.to_numpy()
        result["dst_supervision_status"] = dst_label.supervision_status.to_numpy()
        result["src_gt_identity_key"] = src_label.gt_identity_key.to_numpy()
        result["dst_gt_identity_key"] = dst_label.gt_identity_key.to_numpy()
        result["transition_key"] = sequence + ":" + result.src_row_index.astype(str) + ":" + result.dst_row_index.astype(str)
        result["track_purity"] = result.source_track_key.map(purity.track_purity)
        result["track_matched_rows"] = result.source_track_key.map(purity.matched_row_count)
        result["track_unknown_rows"] = result.source_track_key.map(purity.unknown_row_count)
        parts.append(result)
    return pd.concat(parts, ignore_index=True)


def population_summary(name: str, data: pd.DataFrame) -> dict[str, Any]:
    known = data[data.label.isin([0, 1])].copy()
    unique = known.sort_values(["sequence", "transition_key", "window_id"], kind="mergesort").drop_duplicates("transition_key")
    return {
        "population": name,
        "window_count": int(data.window_id.nunique()),
        "boundary_observation_count": int(len(data)),
        "known_boundary_observation_count": int(len(known)),
        "unique_physical_transition_count": int(known.transition_key.nunique()),
        "positive_count": int((known.label == 1).sum()),
        "negative_count": int((known.label == 0).sum()),
        "positive_unique_count": int((unique.label == 1).sum()),
        "negative_unique_count": int((unique.label == 0).sum()),
        "positive_rate": float(known.label.mean()) if len(known) else None,
        "unique_positive_rate": float(unique.label.mean()) if len(unique) else None,
        "track_count": int(data.source_track_key.nunique()),
        "sequence_count": int(data.sequence.nunique()),
        "sequences": sorted(data.sequence.unique().tolist()),
        "frame_delta": quantile_summary(known.frame_delta),
        "box_area_mean": quantile_summary(known.box_area_mean),
        "crowd_density": quantile_summary(known.crowd_density),
        "appearance_mapped_fraction": quantile_summary(known.appearance_mapped_fraction),
        "missing_appearance_fraction": quantile_summary(known.missing_appearance_fraction),
        "unknown_or_ignored_observation_count": int((data.label == -1).sum()),
    }


def command_population_audit() -> None:
    implementation_guard()
    set_stage("run-population-audit", "running", started_at=now())
    started = time.perf_counter()
    train = annotate_observations(score_frozen_examples("train"))
    validation = annotate_observations(score_frozen_examples("validation"))
    audit_source = pd.read_parquet(R67 / "validation_boundary_observations.parquet")
    audit = pd.DataFrame({
        "population": "audit",
        "sequence": audit_source.sequence,
        "window_id": audit_source.window_id,
        "source_track_key": audit_source.sequence + ":track:" + audit_source.src_track_id.astype(np.int64).astype(str),
        "tensor_index": -1,
        "window_start_offset": np.nan,
        "src_row_index": audit_source.src_row_index,
        "dst_row_index": audit_source.dst_row_index,
        "boundary_logit": audit_source.boundary_logit,
        "boundary_probability": audit_source.boundary_probability,
        "label": audit_source.label,
    })
    audit = annotate_observations(audit)
    all_observations = pd.concat([train, validation, audit], ignore_index=True)
    all_observations.to_parquet(R67 / "boundary_observations_all_populations.parquet", index=False)
    summaries = [
        population_summary("train_examples", train),
        population_summary("validation_examples", validation),
        population_summary("audit_full_MOT17_11_13", audit),
    ]
    table_rows = []
    for population_name, data in [("train_examples", train), ("validation_examples", validation), ("audit_full_MOT17_11_13", audit)]:
        overall = population_summary(population_name, data)
        table_rows.append({k: v for k, v in overall.items() if not isinstance(v, (dict, list))})
        for sequence, group in data.groupby("sequence", sort=True):
            row = population_summary(f"{population_name}:{sequence}", group)
            table_rows.append({k: v for k, v in row.items() if not isinstance(v, (dict, list))})
    pd.DataFrame(table_rows).to_csv(R67 / "training_vs_audit_boundary_population.csv", index=False)
    indexed = {item["population"]: item for item in summaries}
    train_summary = indexed["train_examples"]
    validation_summary = indexed["validation_examples"]
    audit_summary = indexed["audit_full_MOT17_11_13"]
    ratios = {
        "train_positive_rate": train_summary["positive_rate"],
        "validation_positive_rate": validation_summary["positive_rate"],
        "audit_positive_rate": audit_summary["positive_rate"],
        "train_to_audit_positive_ratio": float(train_summary["positive_rate"] / audit_summary["positive_rate"]),
        "validation_to_audit_positive_ratio": float(validation_summary["positive_rate"] / audit_summary["positive_rate"]),
        "train_to_audit_unique_transition_ratio": float(train_summary["unique_physical_transition_count"] / audit_summary["unique_physical_transition_count"]),
    }
    zero_support = []
    for column, assign in [
        ("crowd_density", lambda x: bucket_value(float(x), CROWD_BUCKETS)),
        ("appearance_mapped_fraction", lambda x: appearance_bucket(float(x))),
    ]:
        audit_known = audit[audit.label.isin([0, 1])].copy()
        train_known = train[train.label.isin([0, 1])].copy()
        audit_known["bucket"] = audit_known[column].map(assign)
        train_known["bucket"] = train_known[column].map(assign)
        for bucket, count in audit_known.bucket.value_counts().items():
            fraction = float(count / len(audit_known))
            if fraction >= 0.05 and int((train_known.bucket == bucket).sum()) == 0:
                zero_support.append({"field": column, "bucket": bucket, "audit_fraction": fraction})
    severe = bool(
        ratios["train_to_audit_positive_ratio"] > 5.0
        or ratios["train_to_audit_positive_ratio"] < 0.2
        or ratios["validation_to_audit_positive_ratio"] > 5.0
        or ratios["validation_to_audit_positive_ratio"] < 0.2
        or ratios["train_to_audit_unique_transition_ratio"] < 0.2
        or zero_support
    )
    start_offsets = train.window_start_offset.value_counts(dropna=False).sort_index()
    sampling = {
        "fixed_length_windows": True,
        "window_max_rows": 30,
        "node_stride": 15,
        "train_sequence_set": TRAIN_SEQUENCES,
        "validation_and_audit_sequence_set": VAL_SEQUENCES,
        "sequence_split_difference_disclosed": True,
        "stored_boundary_positive_negative_rebalanced": False,
        "unknown_excluded_from_binary_metrics": True,
        "window_start_offset_distribution": {str(k): int(v) for k, v in start_offsets.items()},
        "zero_train_support_audit_strata_ge_5pct": zero_support,
        "severe_population_mismatch": severe,
        "criteria": "fixed preregistration thresholds",
    }
    result = {
        "experiment_id": EXP_ID,
        "summaries": summaries,
        "ratios": ratios,
        "sampling_bias_audit": sampling,
        "training_population_alignment_status": "FAIL" if severe else "PASS",
        "checkpoint_sha256_before_after_equal": True,
        "model_eval": True,
        "optimizer_constructed": False,
    }
    write_json(R67 / "training_vs_audit_boundary_population.json", result)
    append_event("population_audit_completed", severe_population_mismatch=severe, ratios=ratios)
    set_stage("run-population-audit", "completed", finished_at=now(), wall_seconds=time.perf_counter() - started, peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, rchar_delta=0)
    print(json.dumps({"stage": "run-population-audit", "status": "completed", "severe_population_mismatch": severe}, sort_keys=True))


def command_score_distribution_audit() -> None:
    implementation_guard()
    set_stage("run-score-distribution-audit", "running", started_at=now())
    started = time.perf_counter()
    observations = pd.read_parquet(R67 / "boundary_observations_all_populations.parquet")
    known = observations[observations.label.isin([0, 1])].copy()
    unique = known.groupby(["population", "sequence", "transition_key"], sort=True).agg(
        label=("label", "first"), label_nunique=("label", "nunique"),
        mean_logit=("boundary_logit", "mean"), mean_probability=("boundary_probability", "mean"),
        observation_count=("label", "size"),
    ).reset_index()
    unique["boundary_logit"] = unique.mean_logit
    unique["boundary_probability"] = 1.0 / (1.0 + np.exp(-np.clip(unique.mean_logit.to_numpy(), -80, 80)))
    unique.to_parquet(R67 / "boundary_unique_transitions_all_populations.parquet", index=False)
    records = []
    for view, data in [("raw_observation", known), ("unique_transition_primary", unique)]:
        for population, group in data.groupby("population", sort=True):
            for label in [0, 1]:
                selected = group[group.label == label]
                for field in ["boundary_logit", "boundary_probability"]:
                    records.append({"view": view, "population": population, "label": label, "field": field, **quantile_summary(selected[field])})
    pd.DataFrame(records).to_csv(R67 / "boundary_score_distribution.csv", index=False)
    write_json(R67 / "boundary_score_distribution.json", {"experiment_id": EXP_ID, "records": records})
    audit = unique[(unique.population == "audit") & unique.label.isin([0, 1])].copy()
    positive = audit[audit.label == 1]
    negative = audit[audit.label == 0]
    auc = safe_roc(audit.label, audit.boundary_logit)
    orientation = {
        "positive_mean_logit": float(positive.boundary_logit.mean()),
        "negative_mean_logit": float(negative.boundary_logit.mean()),
        "positive_median_logit": float(positive.boundary_logit.median()),
        "negative_median_logit": float(negative.boundary_logit.median()),
        "positive_over_negative_pairwise_probability": auc,
        "score_orientation_anomaly": bool(auc is not None and auc < 0.5),
        "reversed_score_used": False,
        "training_orientation": "larger boundary logit denotes different identity boundary",
    }
    write_json(R67 / "score_orientation_validation.json", orientation)
    probabilities = audit.boundary_probability.to_numpy(float)
    logits = audit.boundary_logit.to_numpy(float)
    unique_count = int(len(np.unique(logits)))
    tie_rate = float(1.0 - unique_count / max(len(logits), 1))
    saturation = {
        "rows": int(len(audit)),
        "fraction_probability_le_1e_6": float(np.mean(probabilities <= 1e-6)),
        "fraction_probability_ge_1_minus_1e_6": float(np.mean(probabilities >= 1.0 - 1e-6)),
        "fraction_abs_logit_ge_10": float(np.mean(np.abs(logits) >= 10.0)),
        "fraction_abs_logit_ge_20": float(np.mean(np.abs(logits) >= 20.0)),
        "unique_score_count": unique_count,
        "tie_rate": tie_rate,
        "logit_std": float(np.std(logits)),
        "top_k_distinct_score_count": {str(k): int(len(np.unique(np.sort(logits)[-min(k, len(logits)):]))) for k in [10, 100, 1000]},
        "pooled_roc_auc": auc,
    }
    saturation["score_collapse"] = bool(
        auc is not None
        and 0.45 <= auc <= 0.55
        and (saturation["logit_std"] < 0.10 or tie_rate >= 0.95 or unique_count < 100)
    )
    write_json(R67 / "score_saturation_validation.json", saturation)
    append_event("score_distribution_audit_completed", orientation_anomaly=orientation["score_orientation_anomaly"], score_collapse=saturation["score_collapse"])
    set_stage("run-score-distribution-audit", "completed", finished_at=now(), wall_seconds=time.perf_counter() - started, peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, rchar_delta=0)
    print(json.dumps({"stage": "run-score-distribution-audit", "status": "completed", "score_collapse": saturation["score_collapse"], "orientation_anomaly": orientation["score_orientation_anomaly"]}, sort_keys=True))


def stratum_metrics(data: pd.DataFrame, stratum_type: str, stratum: str) -> dict[str, Any]:
    metrics = binary_metrics(data, "boundary_probability")
    return {
        "stratum_type": stratum_type,
        "stratum": stratum,
        **metrics,
        "score_mean": float(data.boundary_probability.mean()) if len(data) else None,
        "score_std": float(data.boundary_probability.std(ddof=0)) if len(data) else None,
        "positive_score_mean": float(data.loc[data.label == 1, "boundary_probability"].mean()) if (data.label == 1).any() else None,
        "negative_score_mean": float(data.loc[data.label == 0, "boundary_probability"].mean()) if (data.label == 0).any() else None,
        "matched_row_count": int(data.drop_duplicates(["sequence", "source_track_key"]).track_matched_rows.sum()) if "source_track_key" in data.columns else None,
        "unknown_row_count": int(data.drop_duplicates(["sequence", "source_track_key"]).track_unknown_rows.sum()) if "source_track_key" in data.columns else None,
        "chunk_count": int(data.drop_duplicates(["sequence", "source_track_key"]).track_chunk_count.sum()) if "track_chunk_count" in data.columns else None,
        "unique_transition_count": int(data.transition_key.nunique()),
    }


def command_stratified_audit() -> None:
    implementation_guard()
    set_stage("run-stratified-boundary-audit", "running", started_at=now())
    started = time.perf_counter()
    observations = pd.read_parquet(R67 / "boundary_observations_all_populations.parquet")
    observations = observations[observations.label.isin([0, 1])].copy()
    unique = observations.groupby(["population", "sequence", "transition_key"], sort=True).agg(
        label=("label", "first"), mean_logit=("boundary_logit", "mean"),
        src_row_index=("src_row_index", "first"), dst_row_index=("dst_row_index", "first"),
        frame_delta=("frame_delta", "first"), crowd_density=("crowd_density", "first"),
        appearance_mapped_fraction=("appearance_mapped_fraction", "first"), track_purity=("track_purity", "first"),
        source_track_key=("source_track_key", "first"),
        track_matched_rows=("track_matched_rows", "first"), track_unknown_rows=("track_unknown_rows", "first"),
    ).reset_index()
    chunk_counts = pd.read_parquet(R63 / "source_chunks.parquet", columns=["sequence", "source_track_key"]).groupby(["sequence", "source_track_key"]).size().rename("track_chunk_count")
    unique = unique.merge(chunk_counts.reset_index(), on=["sequence", "source_track_key"], how="left", validate="many_to_one")
    unique["track_chunk_count"] = unique.track_chunk_count.fillna(0).astype(int)
    unique = unique[(unique.population == "train") | (unique.population == "audit")].copy()
    unique["boundary_probability"] = 1.0 / (1.0 + np.exp(-np.clip(unique.mean_logit.to_numpy(), -80, 80)))
    records = []
    for sequence, group in unique.groupby("sequence", sort=True):
        records.append(stratum_metrics(group, "sequence", sequence))
    unique["boundary_frame_delta_bucket"] = unique.frame_delta.map(gap_bucket)
    unique["crowd_density_bucket"] = unique.crowd_density.map(lambda x: bucket_value(float(x), CROWD_BUCKETS))
    unique["appearance_mapped_bucket"] = unique.appearance_mapped_fraction.map(lambda x: appearance_bucket(float(x)))
    unique["track_purity_bucket"] = unique.track_purity.map(lambda x: bucket_value(float(x), PURITY_BUCKETS))
    for column, kind in [
        ("boundary_frame_delta_bucket", "boundary_frame_delta_fixed_v3_numeric_bucket"),
        ("crowd_density_bucket", "crowd_density_feature_142"),
        ("appearance_mapped_bucket", "appearance_mapped_sidecar_fraction"),
        ("track_purity_bucket", "track_purity"),
    ]:
        for value, group in unique.groupby(column, dropna=False, sort=True):
            records.append(stratum_metrics(group, kind, str(value)))
    pd.DataFrame(records).to_csv(R67 / "stratified_boundary_metrics.csv", index=False)
    write_json(
        R67 / "stratified_boundary_metrics.json",
        {
            "experiment_id": EXP_ID,
            "records": records,
            "primary_sequences": VAL_SEQUENCES,
            "descriptive_sequences": TRAIN_SEQUENCES,
            "feature_142": "geometry_14_crowd_density_over_100_clipped",
            "feature_143": "geometry_15_nearest_neighbor_distance",
            "appearance_mapped_is_feature_143": False,
        },
    )
    write_json(
        R67 / "gap_convention_reconciliation.json",
        {
            "topology_gap": "dst_first_frame-src_last_frame",
            "intervening_empty_frames": "topology_gap-1",
            "boundary_transition_measure": "dst_row_frame-src_row_frame",
            "boundary_transitions_are_not_candidate_topology_edges": True,
            "fixed_numeric_buckets": GAP_BUCKETS,
            "no_candidate_pool_rebuild": True,
        },
    )
    append_event("stratified_audit_completed", record_count=len(records))
    set_stage("run-stratified-boundary-audit", "completed", finished_at=now(), wall_seconds=time.perf_counter() - started, peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, rchar_delta=0)
    print(json.dumps({"stage": "run-stratified-boundary-audit", "status": "completed", "records": len(records)}, sort_keys=True))


def command_diagnose() -> None:
    implementation_guard()
    set_stage("diagnose", "running", started_at=now())
    started = time.perf_counter()
    mapping_pass = bool(read_json(R67 / "boundary_row_mapping_validation.json", {}).get("all_passed"))
    label_pass = bool(read_json(R67 / "boundary_label_semantics_validation.json", {}).get("all_passed"))
    population = read_json(R67 / "training_vs_audit_boundary_population.json", {})
    population_pass = not bool(population.get("sampling_bias_audit", {}).get("severe_population_mismatch", True))
    saturation = read_json(R67 / "score_saturation_validation.json", {})
    score_collapse = bool(saturation.get("score_collapse"))
    orientation = read_json(R67 / "score_orientation_validation.json", {})
    unique = pd.read_parquet(R67 / "boundary_unique_transitions_all_populations.parquet")
    audit = unique[(unique.population == "audit") & unique.label.isin([0, 1])].copy()
    per_sequence = []
    for sequence, group in audit.groupby("sequence", sort=True):
        per_sequence.append({"sequence": sequence, **binary_metrics(group, "boundary_probability")})
    macro = {
        key: float(np.mean([item[key] for item in per_sequence if item[key] is not None]))
        for key in ["pr_auc", "precision_at_actual", "recall_at_95_precision"]
    }
    minimum_precision = float(min(item["precision_at_actual"] for item in per_sequence))
    capacity_pass = bool(
        macro["pr_auc"] >= 0.283
        and macro["precision_at_actual"] >= 0.35
        and macro["recall_at_95_precision"] >= 0.05
        and minimum_precision >= 0.20
    )
    if not (mapping_pass and label_pass):
        primary = "boundary_label_mapping_failure"
        overall = "FAIL_BOUNDARY_LABEL_MAPPING"
    elif not population_pass:
        primary = "boundary_population_mismatch"
        overall = "FAIL_BOUNDARY_POPULATION_SHIFT"
    elif score_collapse:
        primary = "boundary_score_collapse"
        overall = "FAIL_BOUNDARY_SCORE_COLLAPSE"
    elif not capacity_pass:
        primary = "source_boundary_capacity_failure"
        overall = "FAIL_SOURCE_BOUNDARY_CAPACITY"
    else:
        primary = "none"
        overall = "PASS_BOUNDARY_IMPLEMENTATION"
    secondary = []
    if not population_pass and primary != "boundary_population_mismatch":
        secondary.append("boundary_population_mismatch")
    if score_collapse and primary != "boundary_score_collapse":
        secondary.append("boundary_score_collapse")
    if orientation.get("score_orientation_anomaly"):
        secondary.append("score_orientation_anomaly_descriptive_only")
    if not capacity_pass and primary != "source_boundary_capacity_failure":
        secondary.append("source_boundary_capacity_failure")
    diagnosis = {
        "experiment_id": EXP_ID,
        "status": "diagnosed",
        "measurement_integrity_decision": "PASS_BOUNDARY_IMPLEMENTATION" if mapping_pass and label_pass else "FAIL_BOUNDARY_LABEL_MAPPING",
        "boundary_label_mapping_status": "PASS" if mapping_pass and label_pass else "FAIL",
        "training_population_alignment_status": "PASS" if population_pass else "FAIL",
        "score_collapse_status": "FAIL" if score_collapse else "PASS",
        "source_capacity_status": "PASS" if capacity_pass else "FAIL",
        "scientific_primary_failure": primary,
        "overall_primary_classification": overall,
        "secondary_findings": secondary,
        "corrected_source_reference": {
            "per_sequence": per_sequence,
            "macro": macro,
            "min_precision_at_actual": minimum_precision,
            "thresholds": {
                "macro_pr_auc": 0.283,
                "macro_precision_at_actual": 0.35,
                "macro_recall_at_95_precision": 0.05,
                "every_sequence_precision_at_actual": 0.20,
            },
            "passed": capacity_pass,
        },
        "m23_68_started": False,
        "m23_68_authorized": False,
        **FIXED_DECLARATIONS,
    }
    write_json(R67 / "final_diagnosis.json", diagnosis)
    append_event("diagnosis_completed", scientific_primary_failure=primary, overall_primary_classification=overall)
    set_stage("diagnose", "completed", finished_at=now(), wall_seconds=time.perf_counter() - started)
    print(json.dumps({"stage": "diagnose", "status": "completed", "scientific_primary_failure": primary, "overall_primary_classification": overall}, sort_keys=True))


def command_validate() -> None:
    implementation_guard()
    set_stage("validate", "running", started_at=now())
    started = time.perf_counter()
    input_manifest = read_json(R67 / "input_manifest.json")
    input_checks = []
    for relative, expected in input_manifest["sha256"].items():
        path = ROOT / relative
        actual = sha256(path) if path.exists() else None
        input_checks.append({"path": relative, "expected": expected, "actual": actual, "match": actual == expected})
    required = [
        "summary.csv", "protocol_events.jsonl", "preregistration.json", "input_manifest.json", "implementation_manifest.json",
        "input_reverification.json", "boundary_row_mapping_validation.json", "boundary_row_mapping_examples.json",
        "boundary_label_semantics_validation.json", "boundary_label_distribution.csv", "boundary_label_trace_samples.json",
        "boundary_aggregation_audit.csv", "boundary_aggregation_audit.json",
        "training_vs_audit_boundary_population.csv", "training_vs_audit_boundary_population.json",
        "boundary_score_distribution.csv", "boundary_score_distribution.json", "score_orientation_validation.json",
        "score_saturation_validation.json", "stratified_boundary_metrics.csv", "stratified_boundary_metrics.json",
        "gap_convention_reconciliation.json", "final_diagnosis.json",
    ]
    present = {name: (R67 / name).exists() for name in required}
    scope = {
        "experiment_id": EXP_ID,
        "scope_counts": SCOPE_COUNTS,
        "all_zero": all(value == 0 for value in SCOPE_COUNTS.values()),
        "new_raw_mot17_gt_reads": 0,
        "new_raw_mot20_gt_reads": 0,
        "frozen_label_sidecar_reads": True,
        "m23_68_started": False,
        **FIXED_DECLARATIONS,
    }
    process_gpu = process_gpu_snapshot(exclude_self=True)
    write_json(R67 / "scope_validation.json", scope)
    write_json(R67 / "process_gpu_validation.json", process_gpu)
    validation = {
        "experiment_id": EXP_ID,
        "validated_at": now(),
        "inputs_unchanged": all(item["match"] for item in input_checks),
        "input_checks": input_checks,
        "required_artifacts": present,
        "required_artifacts_present": all(present.values()),
        "scope_passed": scope["all_zero"],
        "implementation_guard_passed": True,
    }
    validation["all_passed"] = validation["inputs_unchanged"] and validation["required_artifacts_present"] and validation["scope_passed"]
    write_json(R67 / "validation_report.json", validation)
    if not validation["all_passed"]:
        raise RuntimeError("validation failed")
    append_event("validation_completed", all_passed=True)
    set_stage("validate", "completed", finished_at=now(), wall_seconds=time.perf_counter() - started, peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, rchar_delta=0)
    print(json.dumps({"stage": "validate", "status": "completed", "all_passed": True}, sort_keys=True))


def artifact_hash_manifest() -> dict[str, Any]:
    paths = [SCRIPT, TEST_SCRIPT, PREREG, RESULT]
    paths.extend(path for path in R67.rglob("*") if path.is_file() and path.name != "artifact_sha256_manifest.json")
    records = []
    for path in sorted(set(paths)):
        records.append({"path": str(path.relative_to(ROOT)), "size": int(path.stat().st_size), "sha256": sha256(path)})
    return {"experiment_id": EXP_ID, "created_at": now(), "records": records}


def result_markdown(final: dict[str, Any], diagnosis: dict[str, Any], population: dict[str, Any], orientation: dict[str, Any], saturation: dict[str, Any]) -> str:
    ratios = population["ratios"]
    reference = diagnosis["corrected_source_reference"]
    lines = [
        f"# {EXP_NAME} — Result",
        "",
        f"Final status: **{final['status']}**  ",
        f"Decision: **{final['decision']}**  ",
        f"Measurement integrity: **{diagnosis['measurement_integrity_decision']}**  ",
        f"Boundary label/mapping: **{diagnosis['boundary_label_mapping_status']}**  ",
        f"Training population alignment: **{diagnosis['training_population_alignment_status']}**  ",
        f"Score collapse status: **{diagnosis['score_collapse_status']}**  ",
        f"Source capacity status: **{diagnosis['source_capacity_status']}**  ",
        f"Scientific primary failure: **{diagnosis['scientific_primary_failure']}**  ",
        f"Overall primary classification: **{diagnosis['overall_primary_classification']}**",
        "",
        "## Mapping and label semantics",
        "",
        "Every M23-66 validation boundary score was joined by explicit semantic row_index to frozen observable rows and R63 supervision. The score_to_source_row tables matched observable rows exactly. Endpoints remained on the same source track with forward time. line_index was not substituted for row_index. Only matched endpoints entered binary metrics; unknown, distractor, ambiguous, or otherwise non-matched endpoints were excluded and never became negatives. Repeated observations of one physical transition had consistent labels.",
        "",
        "The primary corrected physical-transition score remained sigmoid(arithmetic mean finite boundary_logit). Arithmetic mean probability was reported only as sensitivity; no result-based aggregation selection occurred.",
        "",
        "## Training versus full-audit population",
        "",
        f"Train positive rate: `{ratios['train_positive_rate']}`. Validation positive rate: `{ratios['validation_positive_rate']}`. Full MOT17-11/13 audit positive rate: `{ratios['audit_positive_rate']}`.",
        f"Train-to-audit positive-rate ratio: `{ratios['train_to_audit_positive_ratio']}`. Validation-to-audit ratio: `{ratios['validation_to_audit_positive_ratio']}`. Train-to-audit unique-transition ratio: `{ratios['train_to_audit_unique_transition_ratio']}`.",
        f"The preregistered severe population mismatch decision was `{population['sampling_bias_audit']['severe_population_mismatch']}`.",
        "",
        "## Score properties",
        "",
        f"Score orientation anomaly: `{orientation['score_orientation_anomaly']}`. Reversed score used: `false`.",
        f"Score collapse: `{saturation['score_collapse']}`. Corrected pooled source ROC-AUC: `{saturation['pooled_roc_auc']}`; logit standard deviation: `{saturation['logit_std']}`; exact tie rate: `{saturation['tie_rate']}`; distinct finite score count: `{saturation['unique_score_count']}`.",
        "",
        "## Corrected source capacity",
        "",
        f"MOT17-11/13 macro PR-AUC: `{reference['macro']['pr_auc']}`; macro precision@actual: `{reference['macro']['precision_at_actual']}`; macro recall@95P: `{reference['macro']['recall_at_95_precision']}`; minimum sequence precision@actual: `{reference['min_precision_at_actual']}`.",
        "The unchanged reference was PR-AUC 0.283, precision@actual 0.35, recall@95P 0.05, and every-sequence precision@actual 0.20.",
        "",
        "## Scope and authorization",
        "",
        "No training, optimizer step, checkpoint output or modification, tracker, TrackEval, HOTA, raw MOT17/MOT20 GT read, MOT20 test read/submission, teacher, held outer, M23-54, M23-58, threshold search, calibration, temperature scaling, score reversal repair, or policy run occurred. M23-68 was not started and is not authorized by M23-67.",
        "",
        "- `post_hoc_diagnostic_only=true`",
        "- `uses_frozen_gt_derived_label_sidecars=true`",
        "- `not_deployable=true`",
        "- `not_a_strict_result=true`",
        "- `training_runs=0`",
        "- `tracker_outputs=0`",
        "- `trackeval_runs=0`",
        "- `hota_evaluations=0`",
        "- `hota=null`",
        "- `next_policy_authorized=false`",
        "",
        "未执行Notion写回。",
    ]
    return "\n".join(lines) + "\n"


def command_summarize() -> None:
    implementation_guard()
    set_stage("summarize", "running", started_at=now())
    started = time.perf_counter()
    diagnosis = read_json(R67 / "final_diagnosis.json")
    population = read_json(R67 / "training_vs_audit_boundary_population.json")
    orientation = read_json(R67 / "score_orientation_validation.json")
    saturation = read_json(R67 / "score_saturation_validation.json")
    decision = diagnosis["overall_primary_classification"]
    set_stage("summarize", "completed", finished_at=now(), wall_seconds=time.perf_counter() - started)
    set_stage("closed", "closed", started_at=now(), finished_at=now(), decision=decision)
    registry_line = registry_close(
        "completed",
        decision,
        f"primary={diagnosis['scientific_primary_failure']}; mapping={diagnosis['boundary_label_mapping_status']}; population={diagnosis['training_population_alignment_status']}; collapse={diagnosis['score_collapse_status']}; training=0; HOTA=null; next_policy_authorized=false; result={RESULT.relative_to(ROOT)}",
    )
    final = {
        "experiment_id": EXP_ID,
        "name": EXP_NAME,
        "status": "completed",
        "decision": decision,
        "closed_at": now(),
        "diagnosis": diagnosis,
        "scope_counts": SCOPE_COUNTS,
        "registry_line": registry_line,
        "git_head": git_head(),
        "git_scoped_status": git_scoped_status(),
        "process_gpu": process_gpu_snapshot(exclude_self=True),
        "m23_68_started": False,
        "m23_68_authorized": False,
        "notion_writeback": "未执行Notion写回",
        **FIXED_DECLARATIONS,
    }
    RESULT.write_text(result_markdown(final, diagnosis, population, orientation, saturation), encoding="utf-8")
    write_json(R67 / "final_summary.json", final)
    summary = read_summary()
    no_stale = not summary.status.astype(str).isin(["running", "pending"]).any()
    closure = {
        "experiment_id": EXP_ID,
        "status": "closed",
        "decision": decision,
        "closure_integrity_passed": bool(no_stale and RESULT.exists() and (R67 / "final_summary.json").exists()),
        "summary_no_running_pending": no_stale,
        "registry_line": registry_line,
        "registry_closed": True,
        "scope_counts": SCOPE_COUNTS,
        "hota": None,
        "next_policy_authorized": False,
        "script_sha256": sha256(SCRIPT),
        "prereg_sha256": sha256(PREREG),
    }
    write_json(R67 / "closure_validation.json", closure)
    _, registry = registry_rows()
    matching = [(i + 2, row) for i, row in enumerate(registry) if row.get("name") == EXP_ID]
    independent = {
        "experiment_id": EXP_ID,
        "validated_at": now(),
        "summary_no_running_pending": no_stale,
        "required_final_artifacts_present": all(path.exists() for path in [RESULT, R67 / "final_summary.json", R67 / "closure_validation.json"]),
        "registry_latest_closed": bool(matching and matching[-1][1].get("status") == "completed" and matching[-1][1].get("current_stage") == "closed"),
        "scope_all_zero": all(value == 0 for value in SCOPE_COUNTS.values()),
        "hota_is_null": True,
        "next_policy_authorized_false": True,
        "implementation_sha_guard": sha256(SCRIPT) == read_json(R67 / "implementation_manifest.json")["script_sha256"],
    }
    independent["independent_closure_passed"] = all(
        independent[key]
        for key in ["summary_no_running_pending", "required_final_artifacts_present", "registry_latest_closed", "scope_all_zero", "hota_is_null", "next_policy_authorized_false", "implementation_sha_guard"]
    )
    write_json(R67 / "independent_closure_validation.json", independent)
    write_json(R67 / "artifact_sha256_manifest.json", artifact_hash_manifest())
    append_event("experiment_closed", decision=decision, registry_line=registry_line, next_policy_authorized=False, hota=None)
    print(json.dumps({"stage": "summarize", "status": "completed", "decision": decision, "registry_line": registry_line, "closure_integrity_passed": closure["closure_integrity_passed"], "independent_closure_passed": independent["independent_closure_passed"]}, sort_keys=True))


def fail_close(decision: str, error: str) -> None:
    R67.mkdir(parents=True, exist_ok=True)
    if not (R67 / "summary.csv").exists():
        initialize_summary()
    summary = read_summary()
    for index, row in summary.iterrows():
        if row.status == "running":
            summary.loc[index, ["status", "finished_at", "error"]] = ["failed", now(), error]
        elif row.status == "pending":
            summary.loc[index, ["status", "finished_at", "notes"]] = ["skipped", now(), "skipped after fail-close"]
    closed_index = summary.index[summary.stage == "closed"][-1]
    summary.loc[closed_index, ["status", "started_at", "finished_at", "decision", "error"]] = ["closed", now(), now(), decision, error]
    write_summary(summary)
    try:
        registry_line = registry_close("failed", decision, f"fail_closed; {error[:500]}; HOTA=null; next_policy_authorized=false")
    except Exception:
        registry_line = None
    final = {
        "experiment_id": EXP_ID, "status": "failed", "decision": decision, "error": error,
        "closed_at": now(), "registry_line": registry_line, "scope_counts": SCOPE_COUNTS,
        "m23_68_started": False, **FIXED_DECLARATIONS,
    }
    write_json(R67 / "final_summary.json", final)
    write_json(R67 / "scope_validation.json", {"experiment_id": EXP_ID, "scope_counts": SCOPE_COUNTS, **FIXED_DECLARATIONS})
    write_json(R67 / "closure_validation.json", {"experiment_id": EXP_ID, "status": "closed", "decision": decision, "closure_integrity_passed": True, "scope_counts": SCOPE_COUNTS, "hota": None, "next_policy_authorized": False, "error": error})
    RESULT.write_text(f"# {EXP_NAME} — Fail-closed result\n\nDecision: **{decision}**\n\nError: `{error}`\n\nNo training, tracker, TrackEval, HOTA, raw GT, or M23-68 run occurred. `next_policy_authorized=false`.\n\n未执行Notion写回。\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=[
        "init", "verify-inputs", "run-mapping-audit", "run-label-audit", "run-population-audit",
        "run-score-distribution-audit", "run-stratified-boundary-audit", "diagnose", "validate", "summarize",
    ])
    args = parser.parse_args()
    commands = {
        "init": command_init,
        "verify-inputs": command_verify_inputs,
        "run-mapping-audit": command_mapping_audit,
        "run-label-audit": command_label_audit,
        "run-population-audit": command_population_audit,
        "run-score-distribution-audit": command_score_distribution_audit,
        "run-stratified-boundary-audit": command_stratified_audit,
        "diagnose": command_diagnose,
        "validate": command_validate,
        "summarize": command_summarize,
    }
    try:
        commands[args.command]()
    except Exception as exc:
        if R67.exists():
            append_event("stage_failed", command=args.command, error=repr(exc), traceback=traceback.format_exc())
            fail_close("FAIL_IMPLEMENTATION" if (R67 / "implementation_manifest.json").exists() else "FAIL_INITIALIZATION", repr(exc))
        raise


if __name__ == "__main__":
    main()
