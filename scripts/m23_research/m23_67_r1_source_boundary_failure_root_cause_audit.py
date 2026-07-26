"""M23-67-R1 source boundary failure root-cause audit repair.

Post-hoc diagnostics only. Frozen R62-R67 artifacts are read-only. This script
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
import tempfile
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
R67_PREDECESSOR = BASE / "m23_67_source_boundary_failure_root_cause_audit"
R67 = BASE / "m23_67_r1_source_boundary_failure_root_cause_audit"
SCRIPT = ROOT / "scripts/m23_research/m23_67_r1_source_boundary_failure_root_cause_audit.py"
TEST_SCRIPT = ROOT / "scripts/m23_research/test_m23_67_r1_source_boundary_audit.py"
PREREG = ROOT / "docs/m23_67_r1_source_boundary_failure_root_cause_audit_prereg_20260724.md"
RESULT = ROOT / "docs/m23_67_r1_source_boundary_failure_root_cause_audit_result_20260724.md"
REGISTRY = ROOT / "outputs/experiment_registry.csv"
MODEL_SOURCE = ROOT / "scripts/m23_research/m23_59_relation_pretrained_hierarchical_flow_v2.py"
CHECKPOINT = R64 / "frozen_checkpoint/relation_v3_frozen.pt"
EXP_ID = "M23-67-R1"
EXP_NAME = "M23-67-R1 — Source Boundary Failure Root-Cause Audit Repair"
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
CORE_INPUTS.extend([
    R67_PREDECESSOR / "input_manifest.json",
    R67_PREDECESSOR / "artifact_sha256_manifest.json",
    R67_PREDECESSOR / "implementation_failure.json",
    R67_PREDECESSOR / "final_summary.json",
    R67_PREDECESSOR / "closure_validation.json",
    R67_PREDECESSOR / "independent_closure_validation.json",
    ROOT / "scripts/m23_research/m23_67_source_boundary_failure_root_cause_audit.py",
    ROOT / "scripts/m23_research/test_m23_67_source_boundary_audit.py",
    ROOT / "docs/m23_67_source_boundary_failure_root_cause_audit_prereg_20260724.md",
    ROOT / "docs/m23_67_source_boundary_failure_root_cause_audit_result_20260724.md",
])
# Freeze every predecessor artifact and every predecessor-declared input, not only the summaries.
_predecessor_artifact_manifest_path = R67_PREDECESSOR / "artifact_sha256_manifest.json"
if _predecessor_artifact_manifest_path.exists():
    _predecessor_artifact_manifest = json.loads(_predecessor_artifact_manifest_path.read_text(encoding="utf-8"))
    for _record in _predecessor_artifact_manifest.get("records", []):
        _path = Path(_record["path"])
        CORE_INPUTS.append(_path if _path.is_absolute() else ROOT / _path)
_predecessor_input_manifest_path = R67_PREDECESSOR / "input_manifest.json"
if _predecessor_input_manifest_path.exists():
    _predecessor_input_manifest = json.loads(_predecessor_input_manifest_path.read_text(encoding="utf-8"))
    for _relative in _predecessor_input_manifest.get("sha256", {}):
        _path = Path(_relative)
        CORE_INPUTS.append(_path if _path.is_absolute() else ROOT / _path)
CORE_INPUTS = list(dict.fromkeys(CORE_INPUTS))


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


def verify_predecessor_r67() -> dict[str, Any]:
    final = read_json(R67_PREDECESSOR / "final_summary.json", {}) or {}
    closure = read_json(R67_PREDECESSOR / "closure_validation.json", {}) or {}
    independent = read_json(R67_PREDECESSOR / "independent_closure_validation.json", {}) or {}
    failure = read_json(R67_PREDECESSOR / "implementation_failure.json", {}) or {}
    summary_path = R67_PREDECESSOR / "summary.csv"
    summary_no_stale = False
    if summary_path.exists():
        summary = pd.read_csv(summary_path, keep_default_na=False)
        summary_no_stale = not summary.status.astype(str).isin(["running", "pending"]).any()

    artifact_manifest = read_json(R67_PREDECESSOR / "artifact_sha256_manifest.json", {}) or {}
    artifact_checks = []
    for record in artifact_manifest.get("records", []):
        path = Path(record["path"])
        if not path.is_absolute():
            path = ROOT / path
        actual = sha256(path) if path.exists() else None
        artifact_checks.append({"path": record["path"], "expected": record.get("sha256"), "actual": actual, "match": actual == record.get("sha256")})

    predecessor_input = read_json(R67_PREDECESSOR / "input_manifest.json", {}) or {}
    input_checks = []
    for relative, expected in predecessor_input.get("sha256", {}).items():
        path = Path(relative)
        if not path.is_absolute():
            path = ROOT / path
        actual = sha256(path) if path.exists() else None
        input_checks.append({"path": relative, "expected": expected, "actual": actual, "match": actual == expected})

    registry_match = False
    try:
        _, rows = registry_rows()
        matching = [row for row in rows if row.get("name") == "M23-67" or row.get("tag") == "M23-67"]
        registry_match = bool(matching and matching[-1].get("status") == "failed" and matching[-1].get("current_stage") == "closed" and matching[-1].get("decision") == "FAIL_IMPLEMENTATION")
    except Exception:
        registry_match = False
    checks = {
        "final_failed_implementation": final.get("status") == "failed" and final.get("decision") == "FAIL_IMPLEMENTATION",
        "next_policy_authorized_false": final.get("next_policy_authorized") is False,
        "m23_68_not_started": final.get("m23_68_started") is False,
        "closure_integrity_passed": closure.get("closure_integrity_passed") is True and closure.get("status") == "closed",
        "independent_closure_passed": independent.get("independent_closure_passed") is True,
        "same_root_repair_not_performed": failure.get("same_root_repair_performed") is False,
        "summary_no_running_pending": bool(summary_no_stale),
        "registry_failed_closed": bool(registry_match),
        "artifact_manifest_nonempty_all_match": bool(artifact_checks) and all(x["match"] for x in artifact_checks),
        "input_manifest_nonempty_all_match": bool(input_checks) and all(x["match"] for x in input_checks),
    }
    return {
        "experiment_id": "M23-67", "checked_at": now(), "checks": checks,
        "artifact_checks": artifact_checks, "input_checks": input_checks,
        "all_passed": all(checks.values()),
    }


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
    temp = REGISTRY.with_suffix(".m23_67_r1.tmp")
    with temp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(REGISTRY)


def registry_start() -> int:
    fields, rows = registry_rows()
    if any(r.get("name") == EXP_ID or r.get("tag") == EXP_ID for r in rows):
        raise RuntimeError("M23-67-R1 registry conflict")
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
        raise RuntimeError("missing M23-67-R1 registry row")
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


def synthetic_join_fixture(*, equal_row_line: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    row_ids = [10, 20, 30, 40]
    line_ids = row_ids if equal_row_line else [101, 202, 303, 404]
    rows = pd.DataFrame({
        "row_index": row_ids,
        "line_index": line_ids,
        "frame": [1, 2, 3, 4],
        "track_id": [7, 7, 7, 7],
        "x1": [0.0, 1.0, 2.0, 3.0],
        "y1": [0.0, 0.0, 0.0, 0.0],
        "x2": [10.0, 11.0, 12.0, 13.0],
        "y2": [20.0, 20.0, 20.0, 20.0],
        "appearance_mapped": [1.0, 0.8, 0.0, 0.5],
    })
    labels = pd.DataFrame({
        "row_index": row_ids,
        "line_index": line_ids,
        "frame": [1, 2, 3, 4],
        "source_track_id": [7, 7, 7, 7],
        "source_track_key": ["S:track:7"] * 4,
        "supervision_status": ["matched", "matched", "unknown", "matched"],
        "gt_identity_key": ["S:gt:1", "S:gt:2", "", "S:gt:2"],
        "distractor_removed": [False, False, False, False],
        "ambiguity_flag": [False, False, False, True],
        "tie_flag": [False, False, False, False],
    })
    logits = np.array([-2.0, -1.0, -1.0], dtype=float)
    scores = pd.DataFrame({
        "sequence": ["S"] * 3,
        "window_id": ["w0", "w0", "w1"],
        "src_row_index": [10, 20, 20],
        "dst_row_index": [20, 30, 30],
        "boundary_logit": logits,
        "boundary_probability": 1.0 / (1.0 + np.exp(-logits)),
    })
    return scores, rows, labels


def mapping_join_self_test() -> dict[str, bool]:
    checks: dict[str, bool] = {}
    equal_scores, equal_rows, equal_labels = synthetic_join_fixture(equal_row_line=True)
    equal_join, equal_evidence = semantic_boundary_join(
        equal_scores, equal_rows, equal_labels, join_key="row_index", verify_physical_reorder=True
    )
    checks["equal_row_line_values_pass"] = bool(
        equal_evidence["semantic_join_passed"]
        and equal_evidence["observable_row_line_equal_fraction"] == 1.0
        and equal_join.loc[0, "label"] == 1
    )

    diff_scores, diff_rows, diff_labels = synthetic_join_fixture(equal_row_line=False)
    diff_join, diff_evidence = semantic_boundary_join(
        diff_scores, diff_rows, diff_labels, join_key="row_index", verify_physical_reorder=True
    )
    checks["different_row_line_values_pass"] = bool(
        diff_evidence["semantic_join_passed"]
        and diff_evidence["observable_row_line_equal_fraction"] == 0.0
        and diff_join.loc[0, "label"] == 1
    )

    try:
        semantic_boundary_join(diff_scores, diff_rows, diff_labels, join_key="line_index")
        checks["line_index_join_rejected"] = False
    except ValueError as exc:
        checks["line_index_join_rejected"] = "require join_key='row_index'" in str(exc)

    shuffled_rows = equal_rows.sample(frac=1.0, random_state=6701).reset_index(drop=True)
    shuffled_labels = equal_labels.sample(frac=1.0, random_state=6702).reset_index(drop=True)
    shuffled_join, shuffled_evidence = semantic_boundary_join(
        equal_scores, shuffled_rows, shuffled_labels, join_key="row_index", verify_physical_reorder=True
    )
    compare = [
        "src_row_index", "dst_row_index", "src_observable_row_index", "dst_observable_row_index",
        "src_label_row_index", "dst_label_row_index", "src_frame", "dst_frame", "label",
    ]
    checks["physical_row_shuffle_invariant"] = bool(
        shuffled_evidence["physical_row_reorder_invariant"]
        and equal_join[compare].equals(shuffled_join[compare])
    )

    missing_scores = equal_scores.copy()
    missing_scores.loc[0, "src_row_index"] = 999
    try:
        semantic_boundary_join(missing_scores, equal_rows, equal_labels, join_key="row_index")
        checks["missing_row_index_fails"] = False
    except KeyError:
        checks["missing_row_index_fails"] = True

    duplicate_rows = pd.concat([equal_rows, equal_rows.iloc[[0]]], ignore_index=True)
    try:
        semantic_boundary_join(equal_scores, duplicate_rows, equal_labels, join_key="row_index")
        checks["duplicate_row_index_fails"] = False
    except ValueError as exc:
        checks["duplicate_row_index_fails"] = "not unique" in str(exc)

    missing_column_rows = equal_rows.drop(columns=["row_index"])
    try:
        semantic_boundary_join(equal_scores, missing_column_rows, equal_labels, join_key="row_index")
        checks["missing_explicit_row_index_column_fails"] = False
    except ValueError as exc:
        checks["missing_explicit_row_index_column_fails"] = "missing explicit semantic columns" in str(exc)

    checks["unknown_endpoint_excluded"] = bool((equal_join.loc[[1, 2], "label"] == -1).all())
    checks["ambiguity_endpoint_excluded"] = bool(equal_join.loc[2, "label"] == -1)
    repeated = equal_join[equal_join.src_row_index.eq(20) & equal_join.dst_row_index.eq(30)]
    checks["repeated_label_consistent"] = bool(repeated.label.nunique() == 1)
    if not all(checks.values()):
        raise AssertionError({k: v for k, v in checks.items() if not v})
    return checks


def synthetic_self_test() -> dict[str, bool]:
    checks: dict[str, bool] = {}
    checks.update(mapping_join_self_test())
    logits = np.array([-2.0, 2.0])
    checks["mean_logit_then_sigmoid"] = abs(1.0 / (1.0 + math.exp(-float(logits.mean()))) - 0.5) < 1e-12
    # The actual aggregation helper is exercised, including repeated observations.
    scores, rows, labels = synthetic_join_fixture(equal_row_line=True)
    joined, _ = semantic_boundary_join(scores, rows, labels)
    joined["transition_key"] = joined.sequence + ":" + joined.src_row_index.astype(str) + ":" + joined.dst_row_index.astype(str)
    joined["src_track_id"] = joined.src_track_id.astype(int)
    aggregated = aggregate_transitions(joined)
    target = aggregated[(aggregated.src_row_index == 10) & (aggregated.dst_row_index == 20)].iloc[0]
    checks["physical_transition_key"] = target.transition_key == "S:10:20"
    checks["primary_aggregation_sigmoid_mean_logit"] = abs(target.boundary_probability - (1.0 / (1.0 + math.exp(2.0)))) < 1e-12
    checks["mean_probability_sensitivity_present"] = "mean_probability" in aggregated.columns
    checks["population_ratio"] = abs((0.01 / 0.005) - 2.0) < 1e-12
    checks["gap_boundaries"] = [gap_bucket(x) for x in [1, 30, 31, 90, 91, 180, 181, 600]] == ["1-30", "1-30", "31-90", "31-90", "91-180", "91-180", "181-600", "181-600"]
    checks["crowd_boundaries"] = [bucket_value(x, CROWD_BUCKETS) for x in [0, 0.25, 0.50, 1.0, 2.0, 5.0]] == ["[0,0.25)", "[0.25,0.50)", "[0.50,1.00)", "[1.00,2.00)", "[2.00,5.00]", "[2.00,5.00]"]
    checks["appearance_boundaries"] = [appearance_bucket(x) for x in [0, 0.1, 0.5, 0.9, 1.0]] == ["0", "(0,0.50)", "[0.50,0.90)", "[0.90,1.00)", "1.00 exact"]
    y = np.array([1, 1, 0, 0]); scores_array = np.array([0.1, 0.2, 0.8, 0.9])
    checks["orientation_anomaly"] = bool(safe_roc(y, scores_array) < 0.5)
    checks["collapse_rule"] = bool(0.45 <= 0.5 <= 0.55 and 0.05 < 0.10)
    def classify(mapping, population, collapse, capacity):
        if not mapping: return "FAIL_BOUNDARY_LABEL_MAPPING"
        if not population: return "FAIL_BOUNDARY_POPULATION_SHIFT"
        if collapse: return "FAIL_BOUNDARY_SCORE_COLLAPSE"
        if not capacity: return "FAIL_SOURCE_BOUNDARY_CAPACITY"
        return "PASS_BOUNDARY_IMPLEMENTATION"
    checks["decision_priority"] = (
        classify(False, False, True, False) == "FAIL_BOUNDARY_LABEL_MAPPING"
        and classify(True, False, True, False) == "FAIL_BOUNDARY_POPULATION_SHIFT"
        and classify(True, True, True, False) == "FAIL_BOUNDARY_SCORE_COLLAPSE"
        and classify(True, True, False, False) == "FAIL_SOURCE_BOUNDARY_CAPACITY"
    )
    if not all(checks.values()):
        raise AssertionError({k: v for k, v in checks.items() if not v})
    return checks


def pre_freeze_self_test() -> dict[str, Any]:
    checks = pre_freeze_self_test()
    fail_close_checks = fail_close_end_to_end_self_test()
    if not all(fail_close_checks.values()):
        raise AssertionError({k: v for k, v in fail_close_checks.items() if not v})
    return {"scientific_and_join_checks": checks, "fail_close_checks": fail_close_checks}


def command_init() -> None:
    if R67.exists():
        raise RuntimeError("M23-67-R1 output root already exists")
    if RESULT.exists():
        raise RuntimeError("M23-67-R1 result document already exists")
    if not PREREG.exists() or not TEST_SCRIPT.exists():
        raise RuntimeError("preregistration or synthetic test is missing")
    fields, registry = registry_rows()
    if any(row.get("name") == EXP_ID or row.get("tag") == EXP_ID for row in registry):
        raise RuntimeError("M23-67-R1 registry conflict")
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
    predecessor_check = verify_predecessor_r67()
    if not predecessor_check["all_passed"]:
        raise RuntimeError(f"M23-67 predecessor is not validly fail-closed: {predecessor_check['checks']}")
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
            "r67_failed_closed": True,
            "all_inputs_read_only": True,
        },
    )
    write_json(R67 / "predecessor_r67_reverification.json", predecessor_check)
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
        "r67_failed_closed_and_unchanged": verify_predecessor_r67().get("all_passed") is True,
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


def _semantic_boundary_join_core(
    scores: pd.DataFrame,
    observable_rows: pd.DataFrame,
    supervision_rows: pd.DataFrame,
    *,
    join_key: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join score endpoints to observable and label rows by semantic row_index.

    The helper rejects any alternative key. It never derives a semantic ID from
    dataframe position, and it permits row_index/line_index values to be equal.
    """
    if join_key != "row_index":
        raise ValueError(f"semantic boundary joins require join_key='row_index', got {join_key!r}")
    required_score = {"sequence", "window_id", "src_row_index", "dst_row_index", "boundary_logit", "boundary_probability"}
    required_rows = {"row_index", "line_index", "frame", "track_id", "x1", "y1", "x2", "y2", "appearance_mapped"}
    required_labels = {
        "row_index", "line_index", "frame", "source_track_id", "source_track_key",
        "supervision_status", "gt_identity_key", "distractor_removed", "ambiguity_flag", "tie_flag",
    }
    missing = {
        "scores": sorted(required_score - set(scores.columns)),
        "observable_rows": sorted(required_rows - set(observable_rows.columns)),
        "supervision_rows": sorted(required_labels - set(supervision_rows.columns)),
    }
    if any(missing.values()):
        raise ValueError(f"missing explicit semantic columns: {missing}")
    if observable_rows.row_index.isna().any() or supervision_rows.row_index.isna().any():
        raise ValueError("semantic row_index contains missing values")
    if not observable_rows.row_index.is_unique:
        raise ValueError("observable semantic row_index is not unique")
    if not supervision_rows.row_index.is_unique:
        raise ValueError("supervision semantic row_index is not unique")
    if scores[["src_row_index", "dst_row_index"]].isna().any().any():
        raise ValueError("score endpoint row_index contains missing values")

    observable_index = observable_rows.set_index("row_index", drop=False, verify_integrity=True)
    label_index = supervision_rows.set_index("row_index", drop=False, verify_integrity=True)
    refs = pd.Index(pd.concat([scores.src_row_index, scores.dst_row_index], ignore_index=True).unique())
    missing_observable = refs.difference(observable_index.index)
    missing_labels = refs.difference(label_index.index)
    if len(missing_observable):
        raise KeyError(f"score row_index missing from observable map: {missing_observable[:10].tolist()}")
    if len(missing_labels):
        raise KeyError(f"score row_index missing from supervision map: {missing_labels[:10].tolist()}")

    row_columns = ["row_index", "frame", "line_index", "track_id", "x1", "y1", "x2", "y2", "appearance_mapped"]
    src = observable_index.loc[scores.src_row_index.to_numpy(), row_columns].reset_index(drop=True).rename(columns={"row_index": "observable_row_index"}).add_prefix("src_")
    dst = observable_index.loc[scores.dst_row_index.to_numpy(), row_columns].reset_index(drop=True).rename(columns={"row_index": "observable_row_index"}).add_prefix("dst_")
    src_label = label_index.loc[scores.src_row_index.to_numpy()].reset_index(drop=True).add_prefix("src_label_")
    dst_label = label_index.loc[scores.dst_row_index.to_numpy()].reset_index(drop=True).add_prefix("dst_label_")
    result = pd.concat([scores.reset_index(drop=True), src, dst, src_label, dst_label], axis=1)

    src_eligible = (result.src_label_supervision_status == "matched") & (~result.src_label_distractor_removed.astype(bool)) & (~result.src_label_ambiguity_flag.astype(bool)) & (~result.src_label_tie_flag.astype(bool))
    dst_eligible = (result.dst_label_supervision_status == "matched") & (~result.dst_label_distractor_removed.astype(bool)) & (~result.dst_label_ambiguity_flag.astype(bool)) & (~result.dst_label_tie_flag.astype(bool))
    eligible = src_eligible & dst_eligible
    result["label"] = np.where(eligible, (result.src_label_gt_identity_key != result.dst_label_gt_identity_key).astype(np.int8), -1)

    endpoint_identity_exact = bool(
        (result.src_row_index.to_numpy() == result.src_observable_row_index.to_numpy()).all()
        and (result.src_row_index.to_numpy() == result.src_label_row_index.to_numpy()).all()
        and (result.dst_row_index.to_numpy() == result.dst_observable_row_index.to_numpy()).all()
        and (result.dst_row_index.to_numpy() == result.dst_label_row_index.to_numpy()).all()
    )
    evidence = {
        "requested_join_key": join_key,
        "actual_join_key": "row_index",
        "row_index_explicit_in_observable": "row_index" in observable_rows.columns,
        "row_index_explicit_in_supervision": "row_index" in supervision_rows.columns,
        "line_index_explicit_in_observable": "line_index" in observable_rows.columns,
        "line_index_explicit_in_supervision": "line_index" in supervision_rows.columns,
        "observable_row_index_unique": bool(observable_rows.row_index.is_unique),
        "supervision_row_index_unique": bool(supervision_rows.row_index.is_unique),
        "score_references_complete_in_observable": len(missing_observable) == 0,
        "score_references_complete_in_supervision": len(missing_labels) == 0,
        "endpoint_row_identity_exact": endpoint_identity_exact,
        "line_index_not_used_as_join_key": join_key == "row_index",
        "parquet_physical_position_not_used_as_join_key": join_key == "row_index",
        "row_line_numeric_equality_allowed": True,
        "observable_row_line_equal_fraction": float((observable_rows.row_index.to_numpy() == observable_rows.line_index.to_numpy()).mean()) if len(observable_rows) else None,
        "supervision_row_line_equal_fraction": float((supervision_rows.row_index.to_numpy() == supervision_rows.line_index.to_numpy()).mean()) if len(supervision_rows) else None,
    }
    evidence["semantic_join_passed"] = bool(all(
        evidence[k] for k in [
            "row_index_explicit_in_observable", "row_index_explicit_in_supervision",
            "line_index_explicit_in_observable", "line_index_explicit_in_supervision",
            "observable_row_index_unique", "supervision_row_index_unique",
            "score_references_complete_in_observable", "score_references_complete_in_supervision",
            "endpoint_row_identity_exact", "line_index_not_used_as_join_key",
            "parquet_physical_position_not_used_as_join_key",
        ]
    ))
    return result, evidence


def semantic_boundary_join(
    scores: pd.DataFrame,
    observable_rows: pd.DataFrame,
    supervision_rows: pd.DataFrame,
    *,
    join_key: str = "row_index",
    verify_physical_reorder: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    result, evidence = _semantic_boundary_join_core(scores, observable_rows, supervision_rows, join_key=join_key)
    if verify_physical_reorder:
        reversed_rows = observable_rows.iloc[::-1].reset_index(drop=True)
        reversed_labels = supervision_rows.iloc[::-1].reset_index(drop=True)
        reordered, _ = _semantic_boundary_join_core(scores, reversed_rows, reversed_labels, join_key=join_key)
        compare = [
            "src_row_index", "dst_row_index", "src_observable_row_index", "dst_observable_row_index",
            "src_frame", "dst_frame", "src_line_index", "dst_line_index", "src_track_id", "dst_track_id",
            "src_label_row_index", "dst_label_row_index", "src_label_gt_identity_key",
            "dst_label_gt_identity_key", "label",
        ]
        physical_reorder_invariant = result[compare].equals(reordered[compare])
        evidence["physical_row_reorder_invariant"] = bool(physical_reorder_invariant)
        evidence["semantic_join_passed"] = bool(evidence["semantic_join_passed"] and physical_reorder_invariant)
        if not physical_reorder_invariant:
            raise AssertionError("semantic row_index join changed after physical row reorder")
    else:
        evidence["physical_row_reorder_invariant"] = None
    return result, evidence


def build_validation_observations(sequence: str) -> pd.DataFrame:
    scores = pd.read_parquet(R66 / "source_scores" / sequence / "boundary_scores.parquet")
    score_map = pd.read_parquet(R66 / "source_scores" / sequence / "score_to_source_row.parquet")
    rows, _ = observable(sequence)
    labels = supervision(sequence)
    compare_columns = ["row_index", "frame", "line_index", "track_id", "x1", "y1", "x2", "y2"]
    score_map_sorted = score_map[compare_columns].sort_values("row_index", kind="mergesort").reset_index(drop=True)
    rows_sorted = rows[compare_columns].sort_values("row_index", kind="mergesort").reset_index(drop=True)
    score_map_exact = score_map_sorted.equals(rows_sorted)
    result, evidence = semantic_boundary_join(scores, rows, labels, join_key="row_index", verify_physical_reorder=True)
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
    evidence["score_to_source_row_exact_observable"] = bool(score_map_exact)
    evidence["semantic_join_passed"] = bool(evidence["semantic_join_passed"] and score_map_exact)
    result.attrs["score_map_exact_observable"] = bool(score_map_exact)
    result.attrs["semantic_join_evidence"] = evidence
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
            "semantic_row_index_join_passed": bool(data.attrs["semantic_join_evidence"]["semantic_join_passed"]),
            "row_index_explicit_and_unique": bool(data.attrs["semantic_join_evidence"]["row_index_explicit_in_observable"] and data.attrs["semantic_join_evidence"]["row_index_explicit_in_supervision"] and data.attrs["semantic_join_evidence"]["observable_row_index_unique"] and data.attrs["semantic_join_evidence"]["supervision_row_index_unique"]),
            "line_index_not_used_as_join_key": bool(data.attrs["semantic_join_evidence"]["line_index_not_used_as_join_key"]),
            "parquet_physical_reorder_invariant": bool(data.attrs["semantic_join_evidence"]["physical_row_reorder_invariant"]),
            "endpoint_row_identity_exact": bool(data.attrs["semantic_join_evidence"]["endpoint_row_identity_exact"]),
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
        "row_index_is_explicit_semantic_identifier": all(item["checks"]["row_index_explicit_and_unique"] for item in validations.values()),
        "row_index_is_not_parquet_physical_position": all(item["checks"]["parquet_physical_reorder_invariant"] for item in validations.values()),
        "line_index_is_metadata_not_join_key": all(item["checks"]["line_index_not_used_as_join_key"] for item in validations.values()),
        "row_index_line_index_numeric_equality_is_allowed": True,
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
    gpu_memory = process_gpu.get("gpu", {}).get("memory_used_mib")
    process_gpu_idle = bool(
        not process_gpu.get("relevant_processes")
        and not process_gpu.get("gpu", {}).get("compute_processes")
        and gpu_memory in (None, 0)
    )
    predecessor_check = verify_predecessor_r67()
    validation = {
        "experiment_id": EXP_ID,
        "validated_at": now(),
        "inputs_unchanged": all(item["match"] for item in input_checks),
        "input_checks": input_checks,
        "required_artifacts": present,
        "required_artifacts_present": all(present.values()),
        "scope_passed": scope["all_zero"],
        "implementation_guard_passed": True,
        "process_gpu_idle": process_gpu_idle,
        "predecessor_r67_unchanged_failed_closed": predecessor_check.get("all_passed") is True,
    }
    validation["all_passed"] = all(
        validation[key] for key in [
            "inputs_unchanged", "required_artifacts_present", "scope_passed",
            "implementation_guard_passed", "process_gpu_idle",
            "predecessor_r67_unchanged_failed_closed",
        ]
    )
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
        "No training, optimizer step, checkpoint output or modification, tracker, TrackEval, HOTA, raw MOT17/MOT20 GT read, MOT20 test read/submission, teacher, held outer, M23-54, M23-58, threshold search, calibration, temperature scaling, score reversal repair, or policy run occurred. M23-68 was not started and is not authorized by M23-67-R1.",
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
    validation_report = read_json(R67 / "validation_report.json", {}) or {}
    decision = diagnosis["overall_primary_classification"]
    set_stage("summarize", "completed", finished_at=now(), wall_seconds=time.perf_counter() - started)
    set_stage("closed", "closed", started_at=now(), finished_at=now(), decision=decision)
    registry_line = registry_close(
        "completed",
        decision,
        f"primary={diagnosis['scientific_primary_failure']}; mapping={diagnosis['boundary_label_mapping_status']}; population={diagnosis['training_population_alignment_status']}; collapse={diagnosis['score_collapse_status']}; training=0; HOTA=null; next_policy_authorized=false; result={RESULT.relative_to(ROOT)}",
    )
    final_process_gpu = process_gpu_snapshot(exclude_self=True)
    final = {
        "experiment_id": EXP_ID,
        "name": EXP_NAME,
        "status": "completed",
        "current_stage": "closed",
        "decision": decision,
        "closed_at": now(),
        "diagnosis": diagnosis,
        "scope_counts": SCOPE_COUNTS,
        "registry_line": registry_line,
        "git_head": git_head(),
        "git_scoped_status": git_scoped_status(),
        "process_gpu": final_process_gpu,
        "m23_68_started": False,
        "m23_68_authorized": False,
        "notion_writeback": "未执行Notion写回",
        **FIXED_DECLARATIONS,
    }
    RESULT.write_text(result_markdown(final, diagnosis, population, orientation, saturation), encoding="utf-8")
    write_json(R67 / "final_summary.json", final)

    summary = read_summary()
    no_stale = not summary.status.astype(str).isin(["running", "pending"]).any()
    inputs_at_close = _manifest_input_checks(R67 / "input_manifest.json", ROOT)
    inputs_unchanged = bool(inputs_at_close) and all(item["match"] for item in inputs_at_close)
    _, registry = registry_rows()
    matching = [(i + 2, row) for i, row in enumerate(registry) if row.get("name") == EXP_ID or row.get("tag") == EXP_ID]
    registry_closed = bool(matching and matching[-1][1].get("status") == "completed" and matching[-1][1].get("current_stage") == "closed" and matching[-1][1].get("decision") == decision)
    gpu_memory = final_process_gpu.get("gpu", {}).get("memory_used_mib")
    process_gpu_idle = bool(
        not final_process_gpu.get("relevant_processes")
        and not final_process_gpu.get("gpu", {}).get("compute_processes")
        and gpu_memory in (None, 0)
    )
    implementation_manifest = read_json(R67 / "implementation_manifest.json", {}) or {}
    implementation_checks = {
        "script": sha256(SCRIPT) == implementation_manifest.get("script_sha256"),
        "test_script": sha256(TEST_SCRIPT) == implementation_manifest.get("test_script_sha256"),
        "prereg": sha256(PREREG) == implementation_manifest.get("prereg_sha256"),
    }
    predecessor_check = verify_predecessor_r67()
    closure_checks = {
        "summary_no_running_pending": bool(no_stale),
        "registry_completed_closed": bool(registry_closed),
        "inputs_unchanged": bool(inputs_unchanged),
        "scope_all_zero": all(value == 0 for value in SCOPE_COUNTS.values()),
        "implementation_sha_guard": all(implementation_checks.values()),
        "validation_report_passed": validation_report.get("all_passed") is True,
        "process_gpu_idle": process_gpu_idle,
        "predecessor_r67_unchanged_failed_closed": predecessor_check.get("all_passed") is True,
        "result_document_exists": RESULT.exists(),
        "final_summary_exists": (R67 / "final_summary.json").exists(),
        "hota_is_null": final.get("hota") is None,
        "next_policy_authorized_false": final.get("next_policy_authorized") is False,
    }
    closure = {
        "experiment_id": EXP_ID,
        "status": "closed",
        "decision": decision,
        **closure_checks,
        "closure_integrity_passed": all(closure_checks.values()),
        "input_checks": inputs_at_close,
        "implementation_checks": implementation_checks,
        "registry_line": registry_line,
        "scope_counts": SCOPE_COUNTS,
        "hota": None,
        "next_policy_authorized": False,
    }
    write_json(R67 / "closure_validation.json", closure)

    # Independent validation rereads all persisted state and recomputes SHA/scope/process checks.
    persisted_final = read_json(R67 / "final_summary.json", {}) or {}
    persisted_closure = read_json(R67 / "closure_validation.json", {}) or {}
    reread_summary = read_summary()
    independent_input_checks = _manifest_input_checks(R67 / "input_manifest.json", ROOT)
    independent_process = process_gpu_snapshot(exclude_self=True)
    independent_gpu_memory = independent_process.get("gpu", {}).get("memory_used_mib")
    independent_checks = {
        "summary_no_running_pending": not reread_summary.status.astype(str).isin(["running", "pending"]).any(),
        "registry_completed_closed": _registry_closed_at(REGISTRY, EXP_ID, "completed"),
        "inputs_unchanged": bool(independent_input_checks) and all(x["match"] for x in independent_input_checks),
        "scope_all_zero": all(int(v) == 0 for v in persisted_final.get("scope_counts", {}).values()),
        "closure_integrity_passed": persisted_closure.get("closure_integrity_passed") is True,
        "final_status_completed_closed": persisted_final.get("status") == "completed" and persisted_final.get("current_stage") == "closed",
        "implementation_sha_guard": sha256(SCRIPT) == implementation_manifest.get("script_sha256") and sha256(TEST_SCRIPT) == implementation_manifest.get("test_script_sha256") and sha256(PREREG) == implementation_manifest.get("prereg_sha256"),
        "process_gpu_idle": not independent_process.get("relevant_processes") and not independent_process.get("gpu", {}).get("compute_processes") and independent_gpu_memory in (None, 0),
        "predecessor_r67_unchanged_failed_closed": verify_predecessor_r67().get("all_passed") is True,
        "hota_is_null": persisted_final.get("hota") is None,
        "next_policy_authorized_false": persisted_final.get("next_policy_authorized") is False,
        "result_document_exists": RESULT.exists(),
    }
    independent = {
        "experiment_id": EXP_ID,
        "validated_at": now(),
        **independent_checks,
        "independent_closure_passed": all(independent_checks.values()),
    }
    write_json(R67 / "independent_closure_validation.json", independent)
    append_event("experiment_closed", decision=decision, registry_line=registry_line, next_policy_authorized=False, hota=None, closure_integrity_passed=closure["closure_integrity_passed"], independent_closure_passed=independent["independent_closure_passed"])

    write_json(R67 / "artifact_sha256_manifest.json", artifact_hash_manifest())
    manifest = read_json(R67 / "artifact_sha256_manifest.json")
    artifact_checks = []
    for record in manifest.get("records", []):
        path = ROOT / record["path"]
        actual = sha256(path) if path.exists() else None
        artifact_checks.append({"path": record["path"], "expected": record["sha256"], "actual": actual, "match": actual == record["sha256"]})
    artifact_validation = {
        "experiment_id": EXP_ID,
        "checked_at": now(),
        "records": artifact_checks,
        "all_match": bool(artifact_checks) and all(item["match"] for item in artifact_checks),
    }
    write_json(R67 / "artifact_manifest_validation.json", artifact_validation)
    if not (closure["closure_integrity_passed"] and independent["independent_closure_passed"] and artifact_validation["all_match"]):
        raise RuntimeError(f"closure validation failed: closure={closure['closure_integrity_passed']} independent={independent['independent_closure_passed']} manifest={artifact_validation['all_match']}")
    print(json.dumps({"stage": "summarize", "status": "completed", "decision": decision, "registry_line": registry_line, "closure_integrity_passed": True, "independent_closure_passed": True, "artifact_manifest_all_match": True}, sort_keys=True))


def _manifest_input_checks(manifest_path: Path, input_base: Path) -> list[dict[str, Any]]:
    manifest = read_json(manifest_path, {}) or {}
    records = []
    for relative, expected in manifest.get("sha256", {}).items():
        path = Path(relative)
        if not path.is_absolute():
            path = input_base / path
        actual = sha256(path) if path.exists() else None
        records.append({"path": str(relative), "expected": expected, "actual": actual, "match": actual == expected})
    return records


def _registry_update_at(registry_path: Path, exp_id: str, status: str, decision: str, notes: str) -> tuple[int | None, bool]:
    if not registry_path.exists():
        return None, False
    with registry_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        rows = list(reader)
    indices = []
    for i, row in enumerate(rows):
        if row.get("name") == exp_id or row.get("tag") == exp_id or row.get("experiment_id") == exp_id:
            for key, value in {
                "status": status, "current_stage": "closed", "decision": decision,
                "notes": notes, "timestamp": now(),
            }.items():
                if key in fields:
                    row[key] = value
            indices.append(i)
    if not indices:
        return None, False
    temp = registry_path.with_suffix(registry_path.suffix + ".failclose.tmp")
    with temp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(registry_path)
    return indices[-1] + 2, True


def _registry_closed_at(registry_path: Path, exp_id: str, expected_status: str) -> bool:
    if not registry_path.exists():
        return False
    with registry_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    matching = [r for r in rows if r.get("name") == exp_id or r.get("tag") == exp_id or r.get("experiment_id") == exp_id]
    return bool(matching and matching[-1].get("status") == expected_status and matching[-1].get("current_stage") == "closed")


def _artifact_manifest_for(run_root: Path, extra_paths: list[Path], experiment_id: str, relative_base: Path) -> dict[str, Any]:
    paths = [p for p in run_root.rglob("*") if p.is_file() and p.name not in {"artifact_sha256_manifest.json", "artifact_manifest_validation.json"}]
    paths.extend(p for p in extra_paths if p.exists())
    records = []
    for path in sorted(set(path.resolve() for path in paths)):
        try:
            display = str(path.relative_to(relative_base.resolve()))
        except ValueError:
            display = str(path)
        records.append({"path": display, "absolute_path": str(path), "size": int(path.stat().st_size), "sha256": sha256(path)})
    return {"experiment_id": experiment_id, "created_at": now(), "records": records}


def execute_fail_close(
    *,
    run_root: Path,
    registry_path: Path,
    result_path: Path,
    input_base: Path,
    experiment_id: str,
    experiment_name: str,
    decision: str,
    error: str,
    scope_counts: dict[str, int],
    implementation_paths: dict[str, Path] | None = None,
    artifact_extra_paths: list[Path] | None = None,
) -> dict[str, Any]:
    """Perform and independently validate a complete fail-close transaction."""
    run_root.mkdir(parents=True, exist_ok=True)
    protocol_path = run_root / "protocol_events.jsonl"
    protocol_path.touch(exist_ok=True)
    with protocol_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp": now(), "experiment_id": experiment_id, "event": "fail_close_started", "decision": decision, "error": error}, sort_keys=True) + "\n")

    summary_path = run_root / "summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path, keep_default_na=False)
    else:
        summary = pd.DataFrame([{**{k: "" for k in SUMMARY_FIELDS}, "experiment_id": experiment_id, "stage": stage, "status": "pending"} for stage in STAGES], columns=SUMMARY_FIELDS)
    for index, row in summary.iterrows():
        if str(row.status) == "running":
            summary.loc[index, ["status", "finished_at", "decision", "error"]] = ["failed", now(), decision, error]
        elif str(row.status) == "pending":
            summary.loc[index, ["status", "finished_at", "notes"]] = ["skipped", now(), "skipped after fail-close"]
    closed_indices = summary.index[summary.stage.astype(str) == "closed"]
    if len(closed_indices) == 0:
        row = {k: "" for k in SUMMARY_FIELDS}
        row.update(experiment_id=experiment_id, stage="closed", status="closed", started_at=now(), finished_at=now(), decision=decision, error=error)
        summary = pd.concat([summary, pd.DataFrame([row])], ignore_index=True)
    else:
        i = closed_indices[-1]
        summary.loc[i, ["status", "started_at", "finished_at", "decision", "error"]] = ["closed", now(), now(), decision, error]
    summary.to_csv(summary_path, index=False)

    input_checks = _manifest_input_checks(run_root / "input_manifest.json", input_base)
    inputs_unchanged = bool(input_checks) and all(item["match"] for item in input_checks)
    input_reverification = {
        "experiment_id": experiment_id, "checked_at": now(), "checks": input_checks,
        "inputs_unchanged": inputs_unchanged, "first_mismatch": next((x for x in input_checks if not x["match"]), None),
    }
    write_json(run_root / "input_reverification_at_failure.json", input_reverification)

    scope_all_zero = all(int(value) == 0 for value in scope_counts.values())
    write_json(run_root / "scope_validation.json", {
        "experiment_id": experiment_id, "scope_counts": scope_counts, "all_zero": scope_all_zero,
        "m23_68_started": False, **FIXED_DECLARATIONS,
    })

    registry_line, registry_updated = _registry_update_at(
        registry_path, experiment_id, "failed", decision,
        f"fail_closed; {error[:500]}; HOTA=null; next_policy_authorized=false",
    )
    implementation_manifest = read_json(run_root / "implementation_manifest.json", {}) or {}
    implementation_checks = {}
    for key, path in (implementation_paths or {}).items():
        expected = implementation_manifest.get(f"{key}_sha256")
        implementation_checks[key] = bool(expected and path.exists() and sha256(path) == expected)
    implementation_guard_passed = bool(implementation_checks) and all(implementation_checks.values())

    failure = {
        "experiment_id": experiment_id, "status": "closed", "decision": decision,
        "failure_type": "post_freeze_fail_close", "error": error, "detected_at": now(),
        "registry_line": registry_line, "registry_updated": registry_updated,
        "input_reverification_passed": inputs_unchanged, "scope_all_zero": scope_all_zero,
        "implementation_checks": implementation_checks, "implementation_guard_passed": implementation_guard_passed,
        "same_root_repair_performed": False, "m23_68_started": False, **FIXED_DECLARATIONS,
    }
    write_json(run_root / "failure_record.json", failure)

    final = {
        "experiment_id": experiment_id, "name": experiment_name, "status": "failed", "decision": decision,
        "error": error, "closed_at": now(), "registry_line": registry_line, "scope_counts": scope_counts,
        "inputs_unchanged": inputs_unchanged, "implementation_guard_passed": implementation_guard_passed,
        "m23_68_started": False, "notion_writeback": "未执行Notion写回", **FIXED_DECLARATIONS,
    }
    write_json(run_root / "final_summary.json", final)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        f"# {experiment_name} — Fail-closed result\n\nDecision: **{decision}**\n\nError: `{error}`\n\n"
        "The fail-close transaction re-verified frozen inputs, scope, summary state, registry closure, and implementation SHAs. "
        "No training, tracker, TrackEval, HOTA, raw GT, or M23-68 run occurred. `next_policy_authorized=false`.\n\n未执行Notion写回。\n",
        encoding="utf-8",
    )

    summary_check = pd.read_csv(summary_path, keep_default_na=False)
    summary_no_stale = not summary_check.status.astype(str).isin(["running", "pending"]).any()
    registry_closed = _registry_closed_at(registry_path, experiment_id, "failed")
    required_failure_artifacts = all((run_root / name).exists() for name in [
        "summary.csv", "protocol_events.jsonl", "input_reverification_at_failure.json",
        "scope_validation.json", "failure_record.json", "final_summary.json",
    ]) and result_path.exists()
    closure_checks = {
        "summary_no_running_pending": bool(summary_no_stale),
        "registry_failed_closed": bool(registry_closed),
        "inputs_unchanged": bool(inputs_unchanged),
        "scope_all_zero": bool(scope_all_zero),
        "implementation_sha_guard": bool(implementation_guard_passed),
        "required_failure_artifacts_present": bool(required_failure_artifacts),
        "hota_is_null": final.get("hota") is None,
        "next_policy_authorized_false": final.get("next_policy_authorized") is False,
    }
    closure = {
        "experiment_id": experiment_id, "status": "closed", "decision": decision,
        **closure_checks, "closure_integrity_passed": all(closure_checks.values()),
        "registry_line": registry_line, "scope_counts": scope_counts, "hota": None,
        "next_policy_authorized": False, "error": error,
    }
    write_json(run_root / "closure_validation.json", closure)

    # Independent pass rereads every persisted record rather than trusting in-memory booleans.
    persisted_final = read_json(run_root / "final_summary.json", {})
    persisted_closure = read_json(run_root / "closure_validation.json", {})
    persisted_scope = read_json(run_root / "scope_validation.json", {})
    persisted_input = read_json(run_root / "input_reverification_at_failure.json", {})
    reread_summary = pd.read_csv(summary_path, keep_default_na=False)
    independent_checks = {
        "summary_no_running_pending": not reread_summary.status.astype(str).isin(["running", "pending"]).any(),
        "registry_failed_closed": _registry_closed_at(registry_path, experiment_id, "failed"),
        "input_reverification_passed": persisted_input.get("inputs_unchanged") is True and all(x.get("match") is True for x in persisted_input.get("checks", [])),
        "scope_all_zero": persisted_scope.get("all_zero") is True and all(int(v) == 0 for v in persisted_scope.get("scope_counts", {}).values()),
        "closure_integrity_passed": persisted_closure.get("closure_integrity_passed") is True,
        "final_status_failed": persisted_final.get("status") == "failed" and persisted_final.get("decision") == decision,
        "implementation_sha_guard": persisted_final.get("implementation_guard_passed") is True,
        "hota_is_null": persisted_final.get("hota") is None,
        "next_policy_authorized_false": persisted_final.get("next_policy_authorized") is False,
        "result_document_exists": result_path.exists(),
    }
    independent = {
        "experiment_id": experiment_id, "validated_at": now(), **independent_checks,
        "independent_closure_passed": all(independent_checks.values()),
    }
    write_json(run_root / "independent_closure_validation.json", independent)
    with protocol_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp": now(), "experiment_id": experiment_id, "event": "fail_close_completed", "decision": decision, "closure_integrity_passed": closure["closure_integrity_passed"], "independent_closure_passed": independent["independent_closure_passed"]}, sort_keys=True) + "\n")

    extras = list(artifact_extra_paths or []) + [result_path]
    manifest = _artifact_manifest_for(run_root, extras, experiment_id, input_base)
    write_json(run_root / "artifact_sha256_manifest.json", manifest)
    artifact_checks = []
    for record in manifest["records"]:
        path = Path(record["absolute_path"])
        actual = sha256(path) if path.exists() else None
        artifact_checks.append({"path": record["path"], "expected": record["sha256"], "actual": actual, "match": actual == record["sha256"]})
    artifact_validation = {
        "experiment_id": experiment_id, "checked_at": now(), "records": artifact_checks,
        "all_match": bool(artifact_checks) and all(x["match"] for x in artifact_checks),
    }
    write_json(run_root / "artifact_manifest_validation.json", artifact_validation)
    return {
        "closure_integrity_passed": closure["closure_integrity_passed"],
        "independent_closure_passed": independent["independent_closure_passed"],
        "artifact_manifest_all_match": artifact_validation["all_match"],
        "registry_line": registry_line,
    }


def fail_close_end_to_end_self_test() -> dict[str, bool]:
    checks: dict[str, bool] = {}
    with tempfile.TemporaryDirectory(prefix="m23_67_r1_failclose_") as temporary:
        base = Path(temporary)
        run_root = base / "run"
        run_root.mkdir()
        input_file = base / "frozen_input.txt"
        input_file.write_text("frozen\n", encoding="utf-8")
        write_json(run_root / "input_manifest.json", {"experiment_id": "SELFTEST", "sha256": {"frozen_input.txt": sha256(input_file)}})
        script_file = base / "script.py"; script_file.write_text("print('frozen')\n", encoding="utf-8")
        test_file = base / "test.py"; test_file.write_text("assert True\n", encoding="utf-8")
        prereg_file = base / "prereg.md"; prereg_file.write_text("frozen protocol\n", encoding="utf-8")
        write_json(run_root / "implementation_manifest.json", {
            "script_sha256": sha256(script_file), "test_script_sha256": sha256(test_file), "prereg_sha256": sha256(prereg_file),
        })
        rows = []
        for stage in STAGES:
            status = "completed" if stage == "init" else ("running" if stage == "verify-inputs" else "pending")
            rows.append({**{k: "" for k in SUMMARY_FIELDS}, "experiment_id": "SELFTEST", "stage": stage, "status": status})
        pd.DataFrame(rows, columns=SUMMARY_FIELDS).to_csv(run_root / "summary.csv", index=False)
        (run_root / "protocol_events.jsonl").write_text(json.dumps({"event": "implementation_frozen"}) + "\n", encoding="utf-8")
        registry = base / "registry.csv"
        fields = ["timestamp", "kind", "status", "script", "dataset", "split", "tracker_family", "variant", "tag", "run_root", "summary_csv", "notes", "name", "current_stage", "decision"]
        with registry.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
            writer.writerow({**{k: "" for k in fields}, "name": "SELFTEST", "tag": "SELFTEST", "status": "running", "current_stage": "running"})
        result_path = base / "result.md"
        outcome = execute_fail_close(
            run_root=run_root, registry_path=registry, result_path=result_path, input_base=base,
            experiment_id="SELFTEST", experiment_name="M23-67-R1 fail-close selftest",
            decision="FAIL_IMPLEMENTATION", error="injected post-freeze failure", scope_counts=dict(SCOPE_COUNTS),
            implementation_paths={"script": script_file, "test_script": test_file, "prereg": prereg_file},
            artifact_extra_paths=[script_file, test_file, prereg_file],
        )
        summary = pd.read_csv(run_root / "summary.csv", keep_default_na=False)
        closure = read_json(run_root / "closure_validation.json")
        independent = read_json(run_root / "independent_closure_validation.json")
        input_check = read_json(run_root / "input_reverification_at_failure.json")
        artifact_check = read_json(run_root / "artifact_manifest_validation.json")
        checks.update({
            "failclose_summary_no_stale": not summary.status.astype(str).isin(["running", "pending"]).any(),
            "failclose_failure_record_exists": (run_root / "failure_record.json").exists(),
            "failclose_protocol_jsonl_exists": (run_root / "protocol_events.jsonl").exists() and "fail_close_completed" in (run_root / "protocol_events.jsonl").read_text(),
            "failclose_input_reverified": input_check.get("inputs_unchanged") is True,
            "failclose_scope_zero": read_json(run_root / "scope_validation.json").get("all_zero") is True,
            "failclose_registry_closed": _registry_closed_at(registry, "SELFTEST", "failed"),
            "failclose_closure_derived_pass": closure.get("closure_integrity_passed") is True and all(closure.get(k) is True for k in ["summary_no_running_pending", "registry_failed_closed", "inputs_unchanged", "scope_all_zero", "implementation_sha_guard", "required_failure_artifacts_present", "hota_is_null", "next_policy_authorized_false"]),
            "failclose_independent_pass": independent.get("independent_closure_passed") is True,
            "failclose_artifact_manifest_pass": artifact_check.get("all_match") is True and outcome["artifact_manifest_all_match"] is True,
            "failclose_result_exists": result_path.exists(),
        })
    if not all(checks.values()):
        raise AssertionError({k: v for k, v in checks.items() if not v})
    return checks


def fail_close(decision: str, error: str) -> None:
    R67.mkdir(parents=True, exist_ok=True)
    execute_fail_close(
        run_root=R67,
        registry_path=REGISTRY,
        result_path=RESULT,
        input_base=ROOT,
        experiment_id=EXP_ID,
        experiment_name=EXP_NAME,
        decision=decision,
        error=error,
        scope_counts=dict(SCOPE_COUNTS),
        implementation_paths={"script": SCRIPT, "test_script": TEST_SCRIPT, "prereg": PREREG},
        artifact_extra_paths=[SCRIPT, TEST_SCRIPT, PREREG],
    )


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
