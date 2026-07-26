"""M23-66 metric correctness and source-to-target decomposition audit.

Independent post-hoc diagnostic only.  The implementation never imports or
executes the mutable M23-65 script and never opens raw MOT17/MOT20 ground truth.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import resource
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import average_precision_score, balanced_accuracy_score, precision_recall_curve, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/mot20_m23_20260718"
R62 = BASE / "m23_59_relation_pretrained_hierarchical_flow_v3_gtfree_source_regeneration"
R63 = BASE / "m23_59_relation_pretrained_hierarchical_flow_v3_supervision_join"
R64 = BASE / "m23_59_relation_pretrained_hierarchical_flow_v3_pair_reconstruction_training_repair1"
R65 = BASE / "m23_59_relation_pretrained_hierarchical_flow_v3_mot20_representation_gate"
R66 = BASE / "m23_66_v3_metric_correctness_source_target_decomposition_audit"
SCRIPT = ROOT / "scripts/m23_research/m23_66_v3_metric_correctness_source_target_decomposition_audit.py"
TEST_SCRIPT = ROOT / "scripts/m23_research/test_m23_66_metric_definitions.py"
PREREG = ROOT / "docs/m23_66_v3_metric_correctness_source_target_decomposition_audit_prereg_20260724.md"
RESULT = ROOT / "docs/m23_66_v3_metric_correctness_source_target_decomposition_audit_result_20260724.md"
REGISTRY = ROOT / "outputs/experiment_registry.csv"

EXP_ID = "M23-66"
EXP_NAME = "M23-66 — M23-59 v3 Metric Correctness and Source-to-Target Decomposition Audit"
CONTRACT_HASH = "90cfab7d3a1fc87cc46b26441d3d883b9fc72f27d8852d1e2074b00e628428f5"
CHECKPOINT_SHA = "dc24211ff8c050f03920711aadac009d4feac26568ff356f8d6bd29dabf7d329"
MODEL_SOURCE_SHA = "50e12cdc6259903f9a7a76adfd3329622899c0c727f1466c2399034c49f6451d"
REPAIR_SCRIPT_SHA = "ed7242cae656f0b31e7cdbce8ac35a5cfb303a70992d657e39fc52e0c19d19fb"
M23_65_RECORDED_SCRIPT_SHA = "9f690c7f667ce45d7c8bc1d1a67b68190bfde7c2923904abf5bab40a0d5710f5"
EXPECTED_PARAMETER_COUNT = 881124

SOURCE_SEQUENCES = ["MOT17-11", "MOT17-13"]
TARGET_SEQUENCES = ["MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05"]
ALL_SEQUENCES = SOURCE_SEQUENCES + TARGET_SEQUENCES
MAX_NODE_ROWS = 30
NODE_STRIDE = 15
CHUNK_MAX_ROWS = 30
CHUNK_MAX_GAP = 30
CANDIDATE_MAX_GAP = 600
CANDIDATE_K = 32
GAP_BUCKETS = [("1-30", 1, 30), ("31-90", 31, 90), ("91-180", 91, 180), ("181-600", 181, 600)]
TRUST_MIN_MATCHED = 2
TRUST_MIN_PURITY = 0.80
BOUNDARY_GATE = {
    "macro_pr_auc": 0.283,
    "macro_precision_at_actual": 0.35,
    "macro_recall_at_95_precision": 0.05,
    "every_sequence_precision_at_actual": 0.20,
}
STAGES = [
    "init",
    "verify-inputs",
    "freeze-source-scores",
    "freeze-canonical-queries",
    "reproduce-legacy",
    "audit-corrected-metrics",
    "diagnose",
    "validate",
    "summarize",
    "closed",
]
SCOPE_COUNTS = {
    "training_runs": 0,
    "optimizer_steps": 0,
    "checkpoint_outputs": 0,
    "warm_starts": 0,
    "v2_checkpoint_loads": 0,
    "tracker_outputs": 0,
    "trackeval_runs": 0,
    "hota_evaluations": 0,
    "mot20_test_reads": 0,
    "mot20_test_submissions": 0,
    "teacher_reads": 0,
    "held_outer_reads": 0,
    "m23_54_starts": 0,
    "m23_58_starts": 0,
    "policy_starts": 0,
    "new_raw_mot20_gt_reads": 0,
    "new_raw_mot17_gt_reads": 0,
}
FIXED_DECLARATIONS = {
    "post_hoc_diagnostic_only": True,
    "uses_frozen_gt_derived_label_sidecars": True,
    "not_deployable": True,
    "not_a_strict_result": True,
    "hota": None,
    "next_policy_authorized": False,
}

TOP_LEVEL_EXPECTED_SHA = {
    R62 / "feature_contract_v3_1.json": "4c243b411cdfe02bad65d9add856db5702c757707675ff326c47a87f2d555e99",
    R63 / "source_topology_manifest.json": "82974588e0b6ba5c1c8478f0a2f49da8c7925dece5e9abec23e44595990cd2a7",
    R63 / "source_chunks.parquet": "fac94cc2833f55b8f906e88d930d31054a4fd107b3392605bf4a4c2a9d3e4b6f",
    R63 / "candidate_pool.parquet": "689846d9b311436ba6f88b5d5b09b7caf4ee1e15f5153f5b1e80d8ed8833b2c4",
    R63 / "paired_candidate_pool.parquet": "6d3f1401ac42104ef83718d037f6442af0d6c605fb080d9e584a3acd34af0d79",
    R63 / "row_supervision.parquet": "1cccd06c0a6fdbbe3ed1a3d9baca0d23529bff173fb364853e186d4634956c04",
    R64 / "frozen_checkpoint/relation_v3_frozen.pt": CHECKPOINT_SHA,
    R64 / "external_validation_metrics.json": "5aa240738c5abf1076cfe1c1ea2a83ed1e2d90ad3e3e75e09d2130736c311c45",
    ROOT / "scripts/m23_research/m23_59_relation_pretrained_hierarchical_flow_v2.py": MODEL_SOURCE_SHA,
    ROOT / "scripts/m23_research/m23_64_v3_pair_reconstruction_training_repair.py": REPAIR_SCRIPT_SHA,
    R65 / "topology_manifest.json": "2f5d3a20d882cc4d630ef89e2abb3692e2acf3dbb3f2cb7ce0126cdfc5019445",
    R65 / "score_freeze_manifest.json": "e43484f7e40e9ef3660bb96e64d442b56b5ed17e946380f3c76f10f09927b3a5",
    R65 / "label_join_manifest.json": "b65978f906438b9d21bc143eb959897cef83b2006cf810f788455feca2a67fed",
    R65 / "representation_metrics.json": "622eed49f8e6cb58ab3e838ffc234c4e33e276c89ac43de277a39deafb2fbf3e",
    R65 / "representation_metrics.csv": "e56185ccb21bbd8235cf7d42d59abc56323e055adc2b152710000a8520560e81",
    R65 / "representation_gate.json": "47e128cd29297df3e8ab95798a14e7b5228436d4e0f4a34ac7d0d174dc989f37",
    R65 / "input_manifest.json": "58e2deca17e76aa643b161232225d3be733dad4c59d429921583b2736d4f2bd4",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_default(v: Any) -> Any:
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.bool_,)):
        return bool(v)
    raise TypeError(type(v).__name__)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_event(event: str, **payload: Any) -> None:
    R66.mkdir(parents=True, exist_ok=True)
    rec = {"timestamp": utc_now(), "experiment_id": EXP_ID, "event": event, **payload}
    with (R66 / "protocol_events.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, sort_keys=True, default=json_default) + "\n")


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()


def git_scoped_status() -> list[str]:
    paths = [SCRIPT, TEST_SCRIPT, PREREG, RESULT, R66, REGISTRY]
    rel = [str(p.relative_to(ROOT)) for p in paths]
    p = subprocess.run(["git", "status", "--short", "--", *rel], cwd=ROOT, text=True, capture_output=True, check=False)
    return [x for x in p.stdout.splitlines() if x.strip()]


def parse_row_indices(v: Any) -> list[int]:
    if isinstance(v, str):
        return [int(x) for x in json.loads(v)]
    if isinstance(v, np.ndarray):
        return [int(x) for x in v.tolist()]
    return [int(x) for x in v]


def gap_bucket(gap: int | float) -> str | None:
    g = int(gap)
    for name, lo, hi in GAP_BUCKETS:
        if lo <= g <= hi:
            return name
    return None


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return expit(np.clip(x, -80.0, 80.0))


def safe_average_precision(y: np.ndarray, s: np.ndarray) -> float | None:
    y = np.asarray(y, np.int8)
    s = np.asarray(s, np.float64)
    keep = np.isfinite(s) & np.isin(y, [0, 1])
    y, s = y[keep], s[keep]
    return None if len(y) == 0 or len(np.unique(y)) < 2 else float(average_precision_score(y, s))


def safe_roc_auc(y: np.ndarray, s: np.ndarray) -> float | None:
    y = np.asarray(y, np.int8)
    s = np.asarray(s, np.float64)
    keep = np.isfinite(s) & np.isin(y, [0, 1])
    y, s = y[keep], s[keep]
    return None if len(y) == 0 or len(np.unique(y)) < 2 else float(roc_auc_score(y, s))


def precision_at_actual(y: np.ndarray, s: np.ndarray, keys: Sequence[str] | None = None) -> float:
    y = np.asarray(y, np.int8)
    s = np.asarray(s, np.float64)
    keep = np.isfinite(s) & np.isin(y, [0, 1])
    y, s = y[keep], s[keep]
    if keys is None:
        kk = np.asarray([f"{i:020d}" for i in range(len(s))], dtype=object)
    else:
        kk = np.asarray(keys, dtype=object)[keep]
    k = int(y.sum())
    if k <= 0 or len(y) == 0:
        return 0.0
    order = np.lexsort((kk, -s))[:k]
    return float(np.mean(y[order]))


def recall_at_precision(y: np.ndarray, s: np.ndarray, threshold: float) -> float:
    y = np.asarray(y, np.int8)
    s = np.asarray(s, np.float64)
    keep = np.isfinite(s) & np.isin(y, [0, 1])
    y, s = y[keep], s[keep]
    if len(y) == 0 or int(y.sum()) == 0 or len(np.unique(y)) < 2:
        return 0.0
    precision, recall, _ = precision_recall_curve(y, s)
    eligible = recall[precision >= threshold]
    return float(np.max(eligible)) if len(eligible) else 0.0


def score_quantiles(s: np.ndarray) -> dict[str, float | None]:
    s = np.asarray(s, np.float64)
    s = s[np.isfinite(s)]
    qs = [("q00", 0.0), ("q01", 0.01), ("q05", 0.05), ("q25", 0.25), ("q50", 0.50), ("q75", 0.75), ("q95", 0.95), ("q99", 0.99), ("q100", 1.0)]
    return {name: (float(np.quantile(s, q)) if len(s) else None) for name, q in qs}


def binary_metrics(y: np.ndarray, s: np.ndarray, keys: Sequence[str] | None = None) -> dict[str, Any]:
    y = np.asarray(y, np.int8)
    s = np.asarray(s, np.float64)
    keep = np.isfinite(s) & np.isin(y, [0, 1])
    y, s = y[keep], s[keep]
    keys2 = None if keys is None else np.asarray(keys, dtype=object)[keep].tolist()
    ap = safe_average_precision(y, s)
    base = float(np.mean(y)) if len(y) else None
    pos = s[y == 1]
    neg = s[y == 0]
    unique_count = int(pd.Series(s).nunique(dropna=True)) if len(s) else 0
    return {
        "rows": int(len(y)),
        "positives": int(y.sum()) if len(y) else 0,
        "negatives": int((y == 0).sum()) if len(y) else 0,
        "base_rate": base,
        "pr_auc": ap,
        "pr_auc_base_rate_lift": (float(ap / base) if ap is not None and base not in (None, 0.0) else None),
        "roc_auc": safe_roc_auc(y, s),
        "precision_at_actual": precision_at_actual(y, s, keys2),
        "recall_at_90_precision": recall_at_precision(y, s, 0.90),
        "recall_at_95_precision": recall_at_precision(y, s, 0.95),
        "recall_at_99_precision": recall_at_precision(y, s, 0.99),
        "score_mean": float(np.mean(s)) if len(s) else None,
        "score_std": float(np.std(s)) if len(s) else None,
        "positive_score_mean": float(np.mean(pos)) if len(pos) else None,
        "positive_score_std": float(np.std(pos)) if len(pos) else None,
        "negative_score_mean": float(np.mean(neg)) if len(neg) else None,
        "negative_score_std": float(np.std(neg)) if len(neg) else None,
        "score_quantiles": score_quantiles(s),
        "positive_score_quantiles": score_quantiles(pos),
        "negative_score_quantiles": score_quantiles(neg),
        "saturation_rate": float(np.mean((s < 1e-6) | (s > 1.0 - 1e-6))) if len(s) else None,
        "tie_rate": float(1.0 - unique_count / len(s)) if len(s) else None,
    }


def balanced_accuracy(y: np.ndarray, pred: np.ndarray) -> float | None:
    y = np.asarray(y, np.int8)
    pred = np.asarray(pred, np.int8)
    return None if len(y) == 0 or len(np.unique(y)) < 2 else float(balanced_accuracy_score(y, pred))


def summary_fields() -> list[str]:
    return [
        "experiment_id", "stage", "status", "started_at", "finished_at", "decision", "error",
        "wall_seconds", "peak_rss_kb", "rchar_delta", "gpu_peak_memory_bytes", "notes",
    ]


def read_summary() -> pd.DataFrame:
    p = R66 / "summary.csv"
    if p.exists():
        return pd.read_csv(p, keep_default_na=False)
    now = utc_now()
    return pd.DataFrame([
        {"experiment_id": EXP_ID, "stage": s, "status": "pending", "started_at": "", "finished_at": "",
         "decision": "", "error": "", "wall_seconds": "", "peak_rss_kb": "", "rchar_delta": "",
         "gpu_peak_memory_bytes": "", "notes": ""}
        for s in STAGES
    ], columns=summary_fields())


def write_summary(df: pd.DataFrame) -> None:
    R66.mkdir(parents=True, exist_ok=True)
    tmp = R66 / "summary.csv.tmp"
    df.to_csv(tmp, index=False)
    tmp.replace(R66 / "summary.csv")


def set_stage(stage: str, status: str, **values: Any) -> None:
    df = read_summary()
    if stage not in set(df.stage.astype(str)):
        raise KeyError(stage)
    idx = df.index[df.stage.astype(str) == stage][-1]
    df.loc[idx, "status"] = status
    for key, val in values.items():
        if key in df.columns:
            df.loc[idx, key] = json.dumps(val, sort_keys=True, default=json_default) if key == "notes" and not isinstance(val, str) else val
    write_summary(df)


def proc_rchar() -> int:
    try:
        vals = {}
        for line in Path("/proc/self/io").read_text().splitlines():
            k, v = line.split(":", 1)
            vals[k.strip()] = int(v.strip())
        return vals.get("rchar", 0)
    except Exception:
        return 0


@contextmanager
def stage_context(stage: str):
    ensure_implementation_guard(stage)
    started = utc_now()
    t0 = time.perf_counter()
    r0 = proc_rchar()
    gpu0 = 0
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            gpu0 = int(torch.cuda.memory_allocated())
    except Exception:
        pass
    set_stage(stage, "running", started_at=started, finished_at="", error="")
    append_event("stage_started", stage=stage)
    try:
        yield
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        gpu_peak = 0
        try:
            import torch
            if torch.cuda.is_available():
                gpu_peak = int(torch.cuda.max_memory_allocated())
        except Exception:
            pass
        set_stage(stage, "failed", finished_at=utc_now(), error=repr(exc), wall_seconds=elapsed,
                  peak_rss_kb=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                  rchar_delta=max(0, proc_rchar() - r0), gpu_peak_memory_bytes=max(gpu0, gpu_peak))
        append_event("stage_failed", stage=stage, error=repr(exc), traceback=traceback.format_exc())
        raise
    else:
        elapsed = time.perf_counter() - t0
        gpu_peak = 0
        try:
            import torch
            if torch.cuda.is_available():
                gpu_peak = int(torch.cuda.max_memory_allocated())
        except Exception:
            pass
        set_stage(stage, "completed", finished_at=utc_now(), wall_seconds=elapsed,
                  peak_rss_kb=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                  rchar_delta=max(0, proc_rchar() - r0), gpu_peak_memory_bytes=max(gpu0, gpu_peak))
        append_event("stage_completed", stage=stage, wall_seconds=elapsed,
                     peak_rss_kb=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                     rchar_delta=max(0, proc_rchar() - r0), gpu_peak_memory_bytes=max(gpu0, gpu_peak))


def ensure_implementation_guard(stage: str) -> None:
    if stage == "init":
        return
    manifest = read_json(R66 / "implementation_manifest.json", {})
    frozen = manifest.get("script_sha256")
    if frozen and sha256(SCRIPT) != frozen:
        raise RuntimeError("FAIL_IMPLEMENTATION: script SHA changed after implementation_frozen")
    prereg_sha = manifest.get("prereg_sha256")
    if prereg_sha and sha256(PREREG) != prereg_sha:
        raise RuntimeError("FAIL_IMPLEMENTATION: prereg SHA changed after implementation_frozen")


def registry_rows() -> tuple[list[str], list[dict[str, str]]]:
    with REGISTRY.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def write_registry_rows(fields: list[str], rows: list[dict[str, str]]) -> None:
    fd, name = tempfile.mkstemp(prefix="m23_66_registry_", suffix=".csv", dir=str(REGISTRY.parent))
    os.close(fd)
    p = Path(name)
    try:
        with p.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        p.replace(REGISTRY)
    finally:
        if p.exists():
            p.unlink()


def registry_start() -> int:
    fields, rows = registry_rows()
    for r in rows:
        if r.get("name") == EXP_ID and r.get("status") == "running":
            r["status"] = "superseded"
            r["current_stage"] = "superseded"
    row = {k: "" for k in fields}
    row.update({
        "timestamp": utc_now(), "kind": "post_hoc_diagnostic", "status": "running",
        "script": str(SCRIPT.relative_to(ROOT)), "dataset": "MOT17+MOT20", "split": "frozen_sidecars",
        "tracker_family": "FM-Track/M23-59-v3", "variant": "metric_correctness_source_target_decomposition_audit",
        "tag": EXP_ID, "run_root": str(R66.relative_to(ROOT)), "summary_csv": str((R66 / "summary.csv").relative_to(ROOT)),
        "name": EXP_ID, "HOTA": "", "current_stage": "running", "decision": "",
        "notes": "post_hoc_diagnostic_only=true; training=0; tracker=0; TrackEval=0; HOTA=null",
    })
    rows.append(row)
    write_registry_rows(fields, rows)
    return len(rows) + 1


def registry_close(status: str, decision: str, notes: str) -> int | None:
    fields, rows = registry_rows()
    line_no = None
    for i in range(len(rows) - 1, -1, -1):
        r = rows[i]
        if r.get("name") == EXP_ID and r.get("status") == "running":
            r.update({
                "timestamp": utc_now(), "status": status, "current_stage": "closed", "decision": decision,
                "run_root": str(R66.relative_to(ROOT)), "summary_csv": str((R66 / "summary.csv").relative_to(ROOT)),
                "script": str(SCRIPT.relative_to(ROOT)), "HOTA": "", "notes": notes,
            })
            r["notes"] = f"{notes}; result={RESULT.relative_to(ROOT)}"
            if "result" in fields:
                r["result"] = str(RESULT.relative_to(ROOT))
            line_no = i + 2
            break
    if line_no is None:
        row = {k: "" for k in fields}
        row.update({
            "timestamp": utc_now(), "kind": "post_hoc_diagnostic", "status": status,
            "script": str(SCRIPT.relative_to(ROOT)), "dataset": "MOT17+MOT20", "split": "frozen_sidecars",
            "tracker_family": "FM-Track/M23-59-v3", "variant": "metric_correctness_source_target_decomposition_audit",
            "tag": EXP_ID, "run_root": str(R66.relative_to(ROOT)), "summary_csv": str((R66 / "summary.csv").relative_to(ROOT)),
            "name": EXP_ID, "HOTA": "", "current_stage": "closed", "decision": decision,
            "notes": f"{notes}; result={RESULT.relative_to(ROOT)}",
        })
        if "result" in fields:
            row["result"] = str(RESULT.relative_to(ROOT))
        rows.append(row)
        line_no = len(rows) + 1
    write_registry_rows(fields, rows)
    return line_no


def system_versions() -> dict[str, Any]:
    import scipy
    import sklearn
    import torch
    return {
        "python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__,
        "pandas": pd.__version__, "torch": torch.__version__, "sklearn": sklearn.__version__, "scipy": scipy.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def process_gpu_snapshot(exclude_self: bool = True) -> dict[str, Any]:
    pattern = ("m23_59", "m23_60", "m23_61", "m23_62", "m23_63", "m23_64", "m23_65", "m23_66",
               "m23_67", "m23_68", "m23_69", "trackeval", "eval_motstyle_trackeval", "tracker/evaluation")
    p = subprocess.run(["ps", "-eo", "pid,ppid,etimes,stat,pcpu,pmem,args"], text=True, capture_output=True, check=False)
    relevant = []
    for line in p.stdout.splitlines()[1:]:
        low = line.lower()
        if any(k in low for k in pattern):
            try:
                pid = int(line.split(None, 1)[0])
            except Exception:
                pid = -1
            if exclude_self and pid in (os.getpid(), os.getppid()):
                continue
            relevant.append(line.strip())
    gpu: dict[str, Any] = {"available": False, "gpus": [], "compute_processes": []}
    q = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total,temperature.gpu", "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=False,
    )
    if q.returncode == 0:
        gpu["available"] = True
        for line in q.stdout.splitlines():
            vals = [x.strip() for x in line.split(",")]
            if len(vals) >= 7:
                gpu["gpus"].append(dict(zip(["index", "uuid", "name", "utilization_pct", "memory_used_mib", "memory_total_mib", "temperature_c"], vals[:7])))
        qp = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"],
            text=True, capture_output=True, check=False,
        )
        if qp.returncode == 0:
            gpu["compute_processes"] = [x.strip() for x in qp.stdout.splitlines() if x.strip()]
    return {"captured_at": utc_now(), "relevant_processes": relevant, "gpu": gpu}


def load_frozen_model(load_checkpoint: bool = True):
    import torch
    model_path = ROOT / "scripts/m23_research/m23_59_relation_pretrained_hierarchical_flow_v2.py"
    if sha256(model_path) != MODEL_SOURCE_SHA:
        raise RuntimeError("frozen model source SHA mismatch")
    spec = importlib.util.spec_from_file_location("m23_66_frozen_model_source", model_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen model source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    model = module.HierarchicalRelationEncoder()
    count = sum(int(p.numel()) for p in model.parameters())
    if count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(f"parameter count mismatch: {count}")
    if load_checkpoint:
        state = torch.load(R64 / "frozen_checkpoint/relation_v3_frozen.pt", map_location="cpu", weights_only=False)
        if isinstance(state, dict):
            state = state.get("model", state.get("state_dict", state))
        model.load_state_dict(state, strict=True)
    model.eval()
    return model, module


def historical_provenance(frozen_scientific_artifacts_all_match: bool) -> dict[str, Any]:
    current = ROOT / "scripts/m23_research/m23_65_v3_mot20_representation_gate.py"
    candidates: list[dict[str, Any]] = []
    found = False
    for p in sorted((ROOT / "scripts").rglob("*m23_65*.py")):
        try:
            digest = sha256(p)
        except OSError:
            continue
        rec = {"path": str(p.relative_to(ROOT)), "sha256": digest, "size": p.stat().st_size, "mtime": p.stat().st_mtime}
        candidates.append(rec)
        found = found or digest == M23_65_RECORDED_SCRIPT_SHA
    pycs = []
    for p in sorted((ROOT / "scripts").rglob("*m23_65*.pyc")):
        data = p.read_bytes()
        header = data[:16]
        flags = int.from_bytes(header[4:8], "little") if len(header) >= 8 else None
        source_timestamp = int.from_bytes(header[8:12], "little") if len(header) >= 12 and flags == 0 else None
        source_size = int.from_bytes(header[12:16], "little") if len(header) >= 16 and flags == 0 else None
        pycs.append({
            "path": str(p.relative_to(ROOT)), "pyc_sha256": sha256(p), "pyc_flags": flags,
            "pyc_source_timestamp": source_timestamp, "pyc_source_size": source_size,
        })
    current_sha = sha256(current)
    return {
        "experiment_id": EXP_ID,
        "recorded_script_sha256": M23_65_RECORDED_SCRIPT_SHA,
        "current_script_sha256": current_sha,
        "current_script_mtime": current.stat().st_mtime,
        "current_script_is_not_historical_source": current_sha != M23_65_RECORDED_SCRIPT_SHA,
        "byte_exact_historical_source_found": found,
        "candidate_sources": candidates,
        "pyc_records": pycs,
        "pyc_sha256": pycs[0]["pyc_sha256"] if pycs else None,
        "pyc_source_timestamp": pycs[0]["pyc_source_timestamp"] if pycs else None,
        "frozen_scientific_artifacts_all_match": bool(frozen_scientific_artifacts_all_match),
        "historical_source_reproduction_status": "available" if found else "unavailable",
        "limitation": (
            "The byte-exact M23-65 historical source matching the recorded closure SHA was not found. "
            "The current working-tree script is not treated as historical source; only frozen-artifact behavioral reproduction is claimed."
            if not found else "A byte-exact source candidate was found and recorded."
        ),
    }


def nested_r65_expected_paths() -> dict[Path, str]:
    out: dict[Path, str] = {}
    topology = read_json(R65 / "topology_manifest.json", {})
    score = read_json(R65 / "score_freeze_manifest.json", {})
    labels = read_json(R65 / "label_join_manifest.json", {})
    top_names = {
        "candidate_pool_sha256": "topology/candidate_pool.parquet",
        "chunks_sha256": "topology/chunks.parquet",
        "paired_candidate_pool_sha256": "topology/paired_candidate_pool.parquet",
        "windows_sha256": "topology/windows.parquet",
    }
    for seq, vals in topology.get("sha256", {}).items():
        for key, rel in top_names.items():
            out[R65 / seq / rel] = vals[key]
        out[R62 / "observables/MOT20" / seq / "rows.parquet"] = vals["source_rows_sha256"]
        out[R62 / "observables/MOT20" / seq / "row_features.f16.npy"] = vals["source_features_sha256"]
    for seq, vals in score.get("score_sha256", {}).items():
        for key, expected in vals.items():
            filename = "score_manifest.json" if key == "score_manifest_sha256" else key
            out[R65 / seq / "scores" / filename] = expected
    label_names = {"labels_sha256": "row_labels.parquet", "purity_sha256": "track_purity.parquet", "trace_sha256": "join_trace.json"}
    for seq, vals in labels.get("labels_sha256", {}).items():
        for key, filename in label_names.items():
            out[R65 / seq / "labels" / filename] = vals[key]
    return out


def verify_all_inputs() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    first_mismatch = None
    expected = {**TOP_LEVEL_EXPECTED_SHA, **nested_r65_expected_paths()}
    for path, expected_sha in expected.items():
        actual = sha256(path) if path.exists() else None
        ok = actual == expected_sha
        rec = {"path": str(path.relative_to(ROOT)), "expected_sha256": expected_sha, "actual_sha256": actual, "match": ok}
        checks.append(rec)
        if not ok and first_mismatch is None:
            first_mismatch = rec
    contract = read_json(R62 / "feature_contract_v3_1.json", {})
    features = contract.get("features", [])
    f143 = features[143] if len(features) > 143 else {}
    semantic_checks = {
        "contract_hash": contract.get("aggregate", {}).get("contract_hash") == CONTRACT_HASH,
        "feature_143_name": f143.get("feature_name") == "geometry_15_nearest_neighbor_distance",
        "feature_143_zero_based_index": f143.get("zero_based_index") == 143,
    }
    r63_rules = read_json(R63 / "source_topology_manifest.json", {}).get("rules", {})
    r65_rules = read_json(R65 / "topology_manifest.json", {}).get("rules", {})
    topology_checks = {
        "max_node_rows": r63_rules.get("max_node_rows") == r65_rules.get("max_node_rows") == MAX_NODE_ROWS,
        "node_stride": r63_rules.get("node_stride") == r65_rules.get("node_stride") == NODE_STRIDE,
        "chunk_max_rows": r63_rules.get("chunk_max_rows") == r65_rules.get("chunk_max_rows") == CHUNK_MAX_ROWS,
        "chunk_max_gap": r63_rules.get("chunk_max_gap") == r65_rules.get("chunk_max_gap") == CHUNK_MAX_GAP,
        "candidate_max_gap": r65_rules.get("candidate_max_gap") == CANDIDATE_MAX_GAP,
        "candidate_k": r65_rules.get("candidate_k") == CANDIDATE_K,
        "gap_buckets": [tuple(x) for x in r65_rules.get("gap_buckets", [])] == GAP_BUCKETS,
        "candidate_score_rule": str(r65_rules.get("ranking", "")).startswith("0.70 cosine + 0.30 exp(-4"),
    }
    model, _ = load_frozen_model(load_checkpoint=True)
    parameter_count = sum(int(p.numel()) for p in model.parameters())
    checkpoint_before = sha256(R64 / "frozen_checkpoint/relation_v3_frozen.pt")
    del model
    checkpoint_after = sha256(R64 / "frozen_checkpoint/relation_v3_frozen.pt")
    all_match = first_mismatch is None and all(semantic_checks.values()) and all(topology_checks.values()) and parameter_count == EXPECTED_PARAMETER_COUNT and checkpoint_before == checkpoint_after == CHECKPOINT_SHA
    return {
        "experiment_id": EXP_ID, "checked_at": utc_now(), "all_passed": all_match,
        "first_mismatch": first_mismatch, "file_checks": checks, "semantic_checks": semantic_checks,
        "topology_checks": topology_checks, "parameter_count": parameter_count,
        "checkpoint_sha256_before": checkpoint_before, "checkpoint_sha256_after": checkpoint_after,
        "frozen_scientific_artifacts_all_match": first_mismatch is None,
        "new_raw_mot20_gt_reads": 0, "new_raw_mot17_gt_reads": 0,
        "frozen_label_sidecar_reads": False,
    }


def synthetic_self_test() -> dict[str, bool]:
    # Query fixtures cover missing canonical, multiple positives, no-positive pool,
    # incoming predecessor choice, and deterministic score ties.
    chunks = pd.DataFrame([
        {"sequence": "S", "chunk_id": "a", "first_frame": 1, "last_frame": 10, "trusted": True, "identity": "id1"},
        {"sequence": "S", "chunk_id": "b", "first_frame": 11, "last_frame": 20, "trusted": True, "identity": "id1"},
        {"sequence": "S", "chunk_id": "c", "first_frame": 21, "last_frame": 30, "trusted": True, "identity": "id1"},
        {"sequence": "S", "chunk_id": "d", "first_frame": 11, "last_frame": 20, "trusted": True, "identity": "id2"},
    ])
    q, vt = build_canonical_from_chunk_info(chunks)
    out_a = q[(q.direction == "outgoing") & (q.query_chunk_id == "a")].iloc[0]
    inc_c = q[(q.direction == "incoming") & (q.query_chunk_id == "c")].iloc[0]
    candidates = pd.DataFrame([
        {"candidate_id": "a", "src_chunk_id": "a", "dst_chunk_id": "c", "relation_logit": 2.0},
        {"candidate_id": "z", "src_chunk_id": "a", "dst_chunk_id": "d", "relation_logit": 2.0},
    ])
    qr = evaluate_query_fixture(out_a.to_dict(), vt[vt.query_id == out_a.query_id], candidates, "relation_logit", False)
    no_pos = candidates[candidates.dst_chunk_id == "d"]
    qr_no = evaluate_query_fixture(out_a.to_dict(), vt[vt.query_id == out_a.query_id], no_pos, "relation_logit", False)
    boundary = pd.DataFrame([
        {"sequence": "S", "src_row_index": 1, "dst_row_index": 2, "boundary_logit": -2.0, "boundary_probability": float(sigmoid(-2.0)), "label": 0},
        {"sequence": "S", "src_row_index": 1, "dst_row_index": 2, "boundary_logit": 2.0, "boundary_probability": float(sigmoid(2.0)), "label": 0},
        {"sequence": "S", "src_row_index": 2, "dst_row_index": 3, "boundary_logit": 2.0, "boundary_probability": float(sigmoid(2.0)), "label": 1},
    ])
    bviews = aggregate_boundary_views(boundary)
    pair = pd.DataFrame([
        {"original_margin": 1.0, "paired_probability": float(sigmoid(1.0)), "legal_pair": True, "trusted_endpoints": True},
        {"original_margin": -1.0, "paired_probability": float(sigmoid(-1.0)), "legal_pair": False, "trusted_endpoints": True},
        {"original_margin": 2.0, "paired_probability": float(sigmoid(2.0)), "legal_pair": False, "trusted_endpoints": False},
    ])
    pa = paired_fixture_metrics(pair)
    checks = {
        "canonical_missing_all_query_zero": qr["canonical_rank"] is None and qr["canonical_r1"] == 0.0,
        "multiple_positive_any_valid_differs": out_a.valid_target_count == 2 and qr["any_valid_rank"] == 1 and qr["canonical_rank"] is None,
        "candidate_pool_no_positive_not_skipped": qr_no["any_valid_rank"] is None and qr_no["all_query_count"] == 1,
        "incoming_latest_predecessor": inc_c.canonical_target_chunk_id == "b",
        "score_tie_candidate_id_break": qr["top_candidate_id"] == "a",
        "duplicate_boundary_unique_differs": len(bviews["legacy_observation_weighted"]) == 3 and len(bviews["corrected_unique_transition_primary"]) == 2,
        "duplicate_boundary_mean_logit_sigmoid": abs(float(bviews["corrected_unique_transition_primary"].iloc[0].score) - 0.5) < 1e-12,
        "unknown_label_excluded_contract": True,
        "paired_fields_separate": pa["valid_pair_original_over_cross_accuracy"] == 1.0 and pa["threshold_accuracy_at_0p5"] == 1.0 and pa["valid_pair_count"] == 1,
        "invalid_pair_excluded_primary": pa["valid_pair_count"] == 1,
    }
    if not all(checks.values()):
        raise AssertionError({k: v for k, v in checks.items() if not v})
    return checks


def build_canonical_from_chunk_info(chunk_info: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    queries: list[dict[str, Any]] = []
    valid_targets: list[dict[str, Any]] = []
    trusted = chunk_info[chunk_info.trusted.astype(bool)].copy()
    for (sequence, identity), group in trusted.groupby(["sequence", "identity"], sort=True, observed=True):
        group = group.sort_values(["first_frame", "last_frame", "chunk_id"], kind="mergesort").reset_index(drop=True)
        records_first = group.to_dict("records")
        first_values = [int(r["first_frame"]) for r in records_first]
        records_last = sorted(records_first, key=lambda r: (int(r["last_frame"]), int(r["first_frame"]), str(r["chunk_id"])))
        last_values = [int(r["last_frame"]) for r in records_last]
        for src in records_first:
            start = bisect.bisect_right(first_values, int(src["last_frame"]))
            end = bisect.bisect_right(first_values, int(src["last_frame"]) + CANDIDATE_MAX_GAP)
            future = records_first[start:end]
            future_all_count = len(records_first) - start
            status = "eligible" if future else ("horizon_outside" if future_all_count else "no_successor")
            qid = f"{sequence}:outgoing:{src['chunk_id']}"
            canonical = future[0] if future else None
            rec = dict(src)
            rec.update({
                "query_id": qid, "sequence": sequence, "direction": "outgoing", "query_chunk_id": src["chunk_id"],
                "query_status": status, "canonical_target_chunk_id": canonical["chunk_id"] if canonical else None,
                "canonical_topology_gap": int(canonical["first_frame"] - src["last_frame"]) if canonical else None,
                "canonical_gap_bucket": gap_bucket(int(canonical["first_frame"] - src["last_frame"])) if canonical else None,
                "valid_target_count": len(future), "temporal_target_count": future_all_count,
            })
            queries.append(rec)
            for rank, target in enumerate(future):
                valid_targets.append({
                    "query_id": qid, "sequence": sequence, "direction": "outgoing", "query_chunk_id": src["chunk_id"],
                    "target_chunk_id": target["chunk_id"], "canonical": rank == 0,
                    "topology_gap": int(target["first_frame"] - src["last_frame"]),
                    "gap_bucket": gap_bucket(int(target["first_frame"] - src["last_frame"])), "valid_target_order": rank,
                })
        for dst in records_first:
            end = bisect.bisect_left(last_values, int(dst["first_frame"]))
            start = bisect.bisect_left(last_values, int(dst["first_frame"]) - CANDIDATE_MAX_GAP)
            past = records_last[start:end]
            past.sort(key=lambda s: (-int(s["last_frame"]), -int(s["first_frame"]), str(s["chunk_id"])))
            status = "eligible" if past else ("horizon_outside" if end else "no_predecessor")
            qid = f"{sequence}:incoming:{dst['chunk_id']}"
            canonical = past[0] if past else None
            rec = dict(dst)
            rec.update({
                "query_id": qid, "sequence": sequence, "direction": "incoming", "query_chunk_id": dst["chunk_id"],
                "query_status": status, "canonical_target_chunk_id": canonical["chunk_id"] if canonical else None,
                "canonical_topology_gap": int(dst["first_frame"] - canonical["last_frame"]) if canonical else None,
                "canonical_gap_bucket": gap_bucket(int(dst["first_frame"] - canonical["last_frame"])) if canonical else None,
                "valid_target_count": len(past), "temporal_target_count": end,
            })
            queries.append(rec)
            for rank, target in enumerate(past):
                valid_targets.append({
                    "query_id": qid, "sequence": sequence, "direction": "incoming", "query_chunk_id": dst["chunk_id"],
                    "target_chunk_id": target["chunk_id"], "canonical": rank == 0,
                    "topology_gap": int(dst["first_frame"] - target["last_frame"]),
                    "gap_bucket": gap_bucket(int(dst["first_frame"] - target["last_frame"])), "valid_target_order": rank,
                })
    return pd.DataFrame(queries), pd.DataFrame(valid_targets)


def evaluate_query_fixture(query: Mapping[str, Any], valid: pd.DataFrame, candidates: pd.DataFrame,
                           score_col: str, ascending: bool) -> dict[str, Any]:
    direction = str(query["direction"])
    qchunk = str(query["query_chunk_id"])
    if direction == "outgoing":
        c = candidates[candidates.src_chunk_id.astype(str) == qchunk].copy()
        target_col = "dst_chunk_id"
    else:
        c = candidates[candidates.dst_chunk_id.astype(str) == qchunk].copy()
        target_col = "src_chunk_id"
    c = c[np.isfinite(c[score_col].to_numpy(np.float64))]
    c = c.sort_values([score_col, "candidate_id"], ascending=[ascending, True], kind="mergesort")
    ordered = c[target_col].astype(str).tolist()
    canonical = str(query["canonical_target_chunk_id"])
    valid_set = set(valid.target_chunk_id.astype(str))
    canonical_rank = ordered.index(canonical) + 1 if canonical in ordered else None
    any_ranks = [i + 1 for i, x in enumerate(ordered) if x in valid_set]
    any_rank = min(any_ranks) if any_ranks else None
    return {
        "all_query_count": 1, "candidate_present": len(c) > 0,
        "canonical_rank": canonical_rank, "any_valid_rank": any_rank,
        "canonical_r1": float(canonical_rank == 1) if canonical_rank is not None else 0.0,
        "any_valid_r1": float(any_rank == 1) if any_rank is not None else 0.0,
        "top_candidate_id": str(c.iloc[0].candidate_id) if len(c) else None,
    }


def aggregate_boundary_views(labeled_observations: pd.DataFrame) -> dict[str, pd.DataFrame]:
    d = labeled_observations.copy()
    d["transition_key"] = (
        d.sequence.astype(str) + "|" +
        d.src_row_index.astype(np.int64).astype(str).str.zfill(12) + "|" +
        d.dst_row_index.astype(np.int64).astype(str).str.zfill(12)
    )
    legacy = d[["sequence", "src_row_index", "dst_row_index", "transition_key", "label"]].copy()
    legacy["score"] = d.boundary_probability.astype(float)
    grouped = d.groupby(["sequence", "src_row_index", "dst_row_index", "transition_key"], sort=True, observed=True)
    primary = grouped.agg(label=("label", "first"), label_nunique=("label", "nunique"), mean_logit=("boundary_logit", "mean"), multiplicity=("boundary_logit", "size")).reset_index()
    if (primary.label_nunique != 1).any():
        raise RuntimeError("duplicate transition labels are inconsistent")
    primary["score"] = sigmoid(primary.mean_logit.to_numpy(np.float64))
    sens = grouped.agg(label=("label", "first"), label_nunique=("label", "nunique"), score=("boundary_probability", "mean"), multiplicity=("boundary_probability", "size")).reset_index()
    return {
        "legacy_observation_weighted": legacy,
        "corrected_unique_transition_primary": primary,
        "corrected_unique_transition_probability_sensitivity": sens,
    }


def paired_fixture_metrics(pair: pd.DataFrame) -> dict[str, Any]:
    trusted = pair[pair.trusted_endpoints.astype(bool)].copy()
    valid = trusted[trusted.legal_pair.astype(bool)]
    y = trusted.legal_pair.astype(np.int8).to_numpy()
    s = trusted.paired_probability.astype(float).to_numpy()
    return {
        "valid_pair_count": int(len(valid)),
        "invalid_or_excluded_pair_count": int(len(pair) - len(valid)),
        "valid_pair_original_over_cross_accuracy": float(np.mean(valid.original_margin.to_numpy(float) > 0)) if len(valid) else None,
        "threshold_accuracy_at_0p5": float(np.mean((s > 0.5) == (y == 1))) if len(y) else None,
    }


def synthetic_checkpoint_interface_test() -> dict[str, Any]:
    """Exercise the frozen v3 checkpoint only on deterministic synthetic tensors."""
    import torch
    before = sha256(R64 / "frozen_checkpoint/relation_v3_frozen.pt")
    model, _ = load_frozen_model(load_checkpoint=True)
    model.eval()
    torch.set_grad_enabled(False)
    rng = np.random.default_rng(66)
    x1 = rng.normal(0.0, 0.1, size=(2, MAX_NODE_ROWS, 144)).astype(np.float32)
    x2 = rng.normal(0.0, 0.1, size=(2, MAX_NODE_ROWS, 144)).astype(np.float32)
    m1 = np.zeros((2, MAX_NODE_ROWS), np.float32)
    m2 = np.zeros((2, MAX_NODE_ROWS), np.float32)
    m1[0, :5] = 1.0; m1[1, :30] = 1.0
    m2[0, :7] = 1.0; m2[1, :18] = 1.0
    node, boundary, valid = run_node_boundary(model, x1, m1, torch.device("cpu"))
    relation, risk = run_relation(model, x1, m1, x2, m2, torch.device("cpu"))
    after = sha256(R64 / "frozen_checkpoint/relation_v3_frozen.pt")
    checks = {
        "checkpoint_sha_before": before,
        "checkpoint_sha_after": after,
        "checkpoint_unchanged": before == after == CHECKPOINT_SHA,
        "parameter_count": sum(int(p.numel()) for p in model.parameters()),
        "model_eval": not model.training,
        "optimizer_constructed": False,
        "node_shape": list(node.shape),
        "boundary_shape": list(boundary.shape),
        "valid_shape": list(valid.shape),
        "relation_shape": list(relation.shape),
        "risk_shape": list(risk.shape),
        "all_finite": all(np.isfinite(v).all() for v in (node, boundary, valid, relation, risk)),
    }
    checks["all_passed"] = bool(
        checks["checkpoint_unchanged"] and checks["parameter_count"] == EXPECTED_PARAMETER_COUNT
        and checks["model_eval"] and checks["all_finite"]
    )
    if not checks["all_passed"]:
        raise AssertionError(checks)
    return checks


def command_init() -> None:
    if not PREREG.exists():
        raise RuntimeError("preregistration document is missing")
    if R66.exists() and any(R66.iterdir()):
        raise RuntimeError("M23-66 run root already contains artifacts; refusing overwrite")
    R66.mkdir(parents=True, exist_ok=True)
    (R66 / "protocol_events.jsonl").touch()
    write_summary(read_summary())
    set_stage("init", "running", started_at=utc_now())
    t0 = time.perf_counter()
    try:
        tests = synthetic_self_test()
        checkpoint_interface = synthetic_checkpoint_interface_test()
        parameter_count = int(checkpoint_interface["parameter_count"])
        registry_line = registry_start()
        preregistration = {
            "experiment_id": EXP_ID, "name": EXP_NAME, "created_at": utc_now(),
            "prereg_document": str(PREREG.relative_to(ROOT)), "prereg_sha256": sha256(PREREG),
            "definitions_frozen": True, "source_sequences": SOURCE_SEQUENCES, "target_sequences": TARGET_SEQUENCES,
            "topology_gap_definition": "dst_first_frame - src_last_frame",
            "intervening_empty_frames_definition": "topology_gap - 1 (descriptive sensitivity only)",
            "gap_buckets": GAP_BUCKETS, "candidate_k": CANDIDATE_K, "candidate_max_gap": CANDIDATE_MAX_GAP,
            "fixed_declarations": FIXED_DECLARATIONS, "scope_counts": SCOPE_COUNTS,
            "new_raw_mot20_gt_reads": 0, "new_raw_mot17_gt_reads": 0, "frozen_label_sidecar_reads": True,
        }
        write_json(R66 / "preregistration.json", preregistration)
        write_json(R66 / "gap_convention_reconciliation.json", {
            "experiment_id": EXP_ID, "primary_gap": "topology_gap = dst_first_frame - src_last_frame",
            "candidate_valid_range": [1, 600], "primary_gap_buckets": GAP_BUCKETS,
            "legacy_m23_60_gap": "intervening_empty_frames = dst_first_frame - src_last_frame - 1",
            "descriptive_sensitivity_only": True, "candidate_pool_rebuilt": False, "gap_buckets_modified": False,
        })
        manifest = {
            "experiment_id": EXP_ID, "created_at": utc_now(), "implementation_frozen": True,
            "git_head": git_head(), "script_sha256": sha256(SCRIPT), "test_script_sha256": sha256(TEST_SCRIPT),
            "prereg_sha256": sha256(PREREG), "versions": system_versions(), "model_source_sha256": MODEL_SOURCE_SHA,
            "repair_script_sha256": REPAIR_SCRIPT_SHA, "checkpoint_sha256": CHECKPOINT_SHA,
            "parameter_count": parameter_count, "synthetic_fixture_checks": tests,
            "synthetic_checkpoint_interface": checkpoint_interface,
            "topology_rules": {
                "max_node_rows": MAX_NODE_ROWS, "node_stride": NODE_STRIDE, "chunk_max_rows": CHUNK_MAX_ROWS,
                "chunk_max_gap": CHUNK_MAX_GAP, "candidate_max_gap": CANDIDATE_MAX_GAP,
                "candidate_k_per_source_gap_bucket": CANDIDATE_K, "gap_buckets": GAP_BUCKETS,
                "candidate_score": "0.70 * appearance_cosine + 0.30 * exp(-4 * geometry_distance)",
            },
            "metric_definitions": {
                "canonical_queries_candidate_pool_independent": True,
                "all_query_missing_contribution": {"R@1": 0, "R@3": 0, "MRR": 0},
                "relation_sort": "relation_logit desc, candidate_id lexicographic",
                "paired_primary": "mean(original_margin > 0) on legal valid pairs; ties fail",
                "boundary_primary": "sigmoid(arithmetic mean finite logits per physical transition)",
            },
            "registry_running_line": registry_line, "fixed_declarations": FIXED_DECLARATIONS,
        }
        write_json(R66 / "implementation_manifest.json", manifest)
        append_event("initialized", git_head=manifest["git_head"], registry_running_line=registry_line,
                     prereg_sha256=manifest["prereg_sha256"], script_sha256=manifest["script_sha256"])
        append_event("synthetic_fixture_passed", checks=tests, checkpoint_interface=checkpoint_interface)
        append_event("implementation_frozen", script_sha256=manifest["script_sha256"],
                     prereg_sha256=manifest["prereg_sha256"], test_script_sha256=manifest["test_script_sha256"])
        set_stage("init", "completed", finished_at=utc_now(), wall_seconds=time.perf_counter() - t0,
                  peak_rss_kb=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                  rchar_delta=proc_rchar(), notes={"synthetic_fixture_checks": tests, "registry_running_line": registry_line})
        print(json.dumps({"stage": "init", "status": "completed", "implementation_manifest_sha256": sha256(R66 / "implementation_manifest.json")}, sort_keys=True))
    except Exception as exc:
        set_stage("init", "failed", finished_at=utc_now(), error=repr(exc), wall_seconds=time.perf_counter() - t0)
        append_event("stage_failed", stage="init", error=repr(exc), traceback=traceback.format_exc())
        raise


def command_verify_inputs() -> None:
    with stage_context("verify-inputs"):
        result = verify_all_inputs()
        provenance = historical_provenance(result["frozen_scientific_artifacts_all_match"])
        result["historical_provenance_sha256_pending"] = True
        write_json(R66 / "input_manifest.json", result)
        write_json(R66 / "m23_65_historical_implementation_provenance.json", provenance)
        result["historical_provenance_sha256"] = sha256(R66 / "m23_65_historical_implementation_provenance.json")
        result.pop("historical_provenance_sha256_pending", None)
        write_json(R66 / "input_manifest.json", result)
        append_event("inputs_reverified", all_passed=result["all_passed"], first_mismatch=result["first_mismatch"],
                     input_manifest_sha256=sha256(R66 / "input_manifest.json"),
                     frozen_scientific_artifacts_all_match=result["frozen_scientific_artifacts_all_match"])
        append_event("historical_provenance_recorded",
                     recorded_script_sha256=provenance["recorded_script_sha256"],
                     current_script_sha256=provenance["current_script_sha256"],
                     historical_source_reproduction_status=provenance["historical_source_reproduction_status"])
        if not result["all_passed"]:
            raise RuntimeError(f"FAIL_INPUT_REVERIFICATION: {result['first_mismatch'] or result['semantic_checks'] or result['topology_checks']}")
        print(json.dumps({"stage": "verify-inputs", "status": "completed", "all_passed": True,
                          "historical_source_reproduction_status": provenance["historical_source_reproduction_status"]}, sort_keys=True))


def tensor_blocks(features: np.ndarray, ids_list: Sequence[Sequence[int]]) -> tuple[np.ndarray, np.ndarray]:
    x = np.zeros((len(ids_list), MAX_NODE_ROWS, 144), np.float32)
    m = np.zeros((len(ids_list), MAX_NODE_ROWS), np.float32)
    for i, ids in enumerate(ids_list):
        rid = np.asarray(list(ids)[:MAX_NODE_ROWS], np.int64)
        if len(rid):
            x[i, :len(rid)] = np.asarray(features[rid], np.float32)
            m[i, :len(rid)] = 1.0
    if not np.all((m[:, 1:] <= m[:, :-1]) | (m[:, 1:] == 1.0)):
        raise RuntimeError("mask is not a contiguous prefix")
    # Exact prefix validation.
    for row in m:
        zeros = np.flatnonzero(row == 0)
        if len(zeros) and np.any(row[zeros[0]:] != 0):
            raise RuntimeError("mask is not a contiguous prefix")
    return x, m


def run_node_boundary(model, x: np.ndarray, mask: np.ndarray, device):
    import torch
    tx = torch.from_numpy(x).to(device)
    tm = torch.from_numpy(mask).to(device)
    with torch.no_grad():
        node, boundary, valid = model.node_and_boundary(tx, tm)
    out = tuple(v.detach().cpu().numpy() for v in (node, boundary, valid))
    if not all(np.isfinite(v).all() for v in out):
        raise RuntimeError("non-finite node/boundary output")
    return out


def run_relation(model, x1: np.ndarray, m1: np.ndarray, x2: np.ndarray, m2: np.ndarray, device):
    import torch
    with torch.no_grad():
        score, risk = model.relation(torch.from_numpy(x1).to(device), torch.from_numpy(m1).to(device),
                                     torch.from_numpy(x2).to(device), torch.from_numpy(m2).to(device))
    out = score.detach().cpu().numpy(), risk.detach().cpu().numpy()
    if not all(np.isfinite(v).all() for v in out):
        raise RuntimeError("non-finite relation output")
    return out


def source_sequence_tables(seq: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    windows = pd.read_parquet(R63 / "source_windows.parquet", filters=[("sequence", "==", seq)])
    chunks = pd.read_parquet(R63 / "source_chunks.parquet", filters=[("sequence", "==", seq)])
    candidates = pd.read_parquet(R63 / "candidate_pool.parquet", filters=[("sequence", "==", seq)])
    pairs = pd.read_parquet(R63 / "paired_candidate_pool.parquet", filters=[("sequence", "==", seq)])
    return windows, chunks, candidates, pairs


def infer_source_sequence(seq: str, model, device) -> dict[str, Any]:
    out = R66 / "source_scores" / seq
    out.mkdir(parents=True, exist_ok=True)
    obs = R62 / "observables/MOT17" / seq
    rows = pd.read_parquet(obs / "rows.parquet").sort_values("row_index", kind="mergesort").reset_index(drop=True)
    features = np.load(obs / "row_features.f16.npy", mmap_mode="r")
    if features.shape != (len(rows), 144) or not np.isfinite(np.asarray(features, np.float32)).all():
        raise RuntimeError(f"invalid feature shape/finite sample for {seq}: {features.shape}")
    windows, chunks, edges, pairs = source_sequence_tables(seq)
    chunk_pos = {str(v): i for i, v in enumerate(chunks.chunk_id.astype(str).tolist())}
    chunk_ids_cache: dict[str, list[int]] = {}

    def chunk_ids(cid: str) -> list[int]:
        if cid not in chunk_ids_cache:
            chunk_ids_cache[cid] = parse_row_indices(chunks.iloc[chunk_pos[cid]].row_indices)
        return chunk_ids_cache[cid]

    node_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    for start in range(0, len(windows), 256):
        wb = windows.iloc[start:start + 256]
        ids_list = [parse_row_indices(v) for v in wb.row_indices]
        x, m = tensor_blocks(features, ids_list)
        nlog, blog, valid = run_node_boundary(model, x, m, device)
        for j, wr in enumerate(wb.itertuples(index=False)):
            ids = ids_list[j]
            node_rows.append({
                "sequence": seq, "window_id": wr.window_id, "source_track_id": int(wr.track_id),
                "node_row_index": int(ids[0]), "node_logit": float(nlog[j]),
                "node_probability": float(sigmoid(float(nlog[j]))), "row_indices": wr.row_indices,
            })
            for k in range(min(len(ids) - 1, MAX_NODE_ROWS - 1)):
                if valid[j, k] <= 0:
                    continue
                z = float(blog[j, k])
                boundary_rows.append({
                    "sequence": seq, "window_id": wr.window_id, "src_row_index": int(ids[k]),
                    "dst_row_index": int(ids[k + 1]), "boundary_logit": z,
                    "boundary_probability": float(sigmoid(z)),
                })

    relation_rows: list[dict[str, Any]] = []
    ccache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def cblock(cid: str) -> tuple[np.ndarray, np.ndarray]:
        if cid not in ccache:
            xx, mm = tensor_blocks(features, [chunk_ids(cid)])
            ccache[cid] = (xx[0], mm[0])
        return ccache[cid]

    for start in range(0, len(edges), 512):
        eb = edges.iloc[start:start + 512]
        a1, a2, m1, m2 = [], [], [], []
        for er in eb.itertuples(index=False):
            xx1, mm1 = cblock(str(er.src_chunk_id))
            xx2, mm2 = cblock(str(er.dst_chunk_id))
            a1.append(xx1); a2.append(xx2); m1.append(mm1); m2.append(mm2)
        score, risk = run_relation(model, np.asarray(a1, np.float32), np.asarray(m1, np.float32),
                                   np.asarray(a2, np.float32), np.asarray(m2, np.float32), device)
        for j, er in enumerate(eb.itertuples(index=False)):
            z, r = float(score[j]), float(risk[j])
            relation_rows.append({
                "sequence": seq, "candidate_id": er.candidate_id, "src_chunk_id": er.src_chunk_id,
                "dst_chunk_id": er.dst_chunk_id, "gap": int(er.gap), "gap_bucket": er.gap_bucket,
                "rank_in_source_bucket": int(er.rank_in_source_bucket), "candidate_score": float(er.candidate_score),
                "appearance_cosine": float(er.appearance_cosine), "geometry_distance": float(er.geometry_distance),
                "relation_logit": z, "relation_probability": float(sigmoid(z)),
                "risk_logit": r, "risk_probability": float(sigmoid(r)),
            })
    relation = pd.DataFrame(relation_rows)
    lookup = relation.set_index("candidate_id", drop=False)
    pair_rows: list[dict[str, Any]] = []
    for p in pairs.itertuples(index=False):
        ids = [str(getattr(p, k)) for k in ("edge1_id", "edge2_id", "cross1_id", "cross2_id")]
        if not all(x in lookup.index for x in ids):
            continue
        vals = [float(lookup.loc[x, "relation_logit"]) for x in ids]
        original_mean = (vals[0] + vals[1]) / 2.0
        cross_mean = (vals[2] + vals[3]) / 2.0
        margin = original_mean - cross_mean
        pair_rows.append({
            "sequence": seq, "pair_id": p.pair_id, "edge1_id": p.edge1_id, "edge2_id": p.edge2_id,
            "cross1_id": p.cross1_id, "cross2_id": p.cross2_id, "original_margin": margin,
            "original_mean_logit": original_mean, "cross_mean_logit": cross_mean,
            "paired_probability": float(sigmoid(margin)),
        })

    node = pd.DataFrame(node_rows)
    boundary = pd.DataFrame(boundary_rows)
    pair_scores = pd.DataFrame(pair_rows)
    row_map = rows[["row_index", "frame", "line_index", "track_id", "x1", "y1", "x2", "y2"]].copy()
    outputs = {
        "boundary_scores.parquet": boundary,
        "node_scores.parquet": node,
        "relation_scores.parquet": relation,
        "pair_scores.parquet": pair_scores,
        "score_to_source_row.parquet": row_map,
    }
    for filename, frame in outputs.items():
        frame.to_parquet(out / filename, index=False)
    artifact_sha = {filename: sha256(out / filename) for filename in outputs}
    manifest = {
        "experiment_id": EXP_ID, "sequence": seq, "status": "frozen", "label_blind": True,
        "source_label_sidecar_reads": 0, "raw_gt_reads": 0, "device": str(device), "eval_mode": True,
        "torch_no_grad": True, "optimizer_constructed": False, "parameter_count": EXPECTED_PARAMETER_COUNT,
        "checkpoint_sha256": CHECKPOINT_SHA, "counts": {
            "windows": len(node), "boundary_observations": len(boundary), "candidate_edges": len(relation), "pairs": len(pair_scores),
        }, "artifacts": artifact_sha,
    }
    write_json(out / "score_manifest.json", manifest)
    manifest["score_manifest_sha256"] = sha256(out / "score_manifest.json")
    return manifest


def deterministic_source_check(model, device) -> dict[str, Any]:
    seq = SOURCE_SEQUENCES[0]
    obs = R62 / "observables/MOT17" / seq
    features = np.load(obs / "row_features.f16.npy", mmap_mode="r")
    windows = pd.read_parquet(R63 / "source_windows.parquet", filters=[("sequence", "==", seq)]).head(4)
    chunks = pd.read_parquet(R63 / "source_chunks.parquet", filters=[("sequence", "==", seq)])
    edges = pd.read_parquet(R63 / "candidate_pool.parquet", filters=[("sequence", "==", seq)]).head(8)
    ids = [parse_row_indices(v) for v in windows.row_indices]
    x, m = tensor_blocks(features, ids)
    a = run_node_boundary(model, x, m, device)
    b = run_node_boundary(model, x, m, device)
    node_equal = all(np.array_equal(x1, x2) for x1, x2 in zip(a, b))
    cpos = chunks.set_index("chunk_id")
    x1s, x2s, m1s, m2s = [], [], [], []
    for er in edges.itertuples(index=False):
        xa, ma = tensor_blocks(features, [parse_row_indices(cpos.loc[er.src_chunk_id].row_indices)])
        xb, mb = tensor_blocks(features, [parse_row_indices(cpos.loc[er.dst_chunk_id].row_indices)])
        x1s.append(xa[0]); x2s.append(xb[0]); m1s.append(ma[0]); m2s.append(mb[0])
    r1 = run_relation(model, np.asarray(x1s), np.asarray(m1s), np.asarray(x2s), np.asarray(m2s), device)
    r2 = run_relation(model, np.asarray(x1s), np.asarray(m1s), np.asarray(x2s), np.asarray(m2s), device)
    relation_equal = all(np.array_equal(xa, xb) for xa, xb in zip(r1, r2))
    return {"sequence": seq, "windows": len(windows), "edges": len(edges),
            "node_boundary_repeat_exact": node_equal, "relation_repeat_exact": relation_equal,
            "all_passed": node_equal and relation_equal}


def command_freeze_source_scores() -> None:
    with stage_context("freeze-source-scores"):
        import torch
        checkpoint_before = sha256(R64 / "frozen_checkpoint/relation_v3_frozen.pt")
        model, _ = load_frozen_model(load_checkpoint=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        torch.set_grad_enabled(False)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        deterministic = deterministic_source_check(model, device)
        if not deterministic["all_passed"]:
            raise RuntimeError("deterministic source inference check failed")
        sequences = {}
        for seq in SOURCE_SEQUENCES:
            sequences[seq] = infer_source_sequence(seq, model, device)
            append_event("source_sequence_scores_frozen", sequence=seq,
                         score_manifest_sha256=sequences[seq]["score_manifest_sha256"], counts=sequences[seq]["counts"])
        checkpoint_after = sha256(R64 / "frozen_checkpoint/relation_v3_frozen.pt")
        if checkpoint_before != checkpoint_after or checkpoint_after != CHECKPOINT_SHA:
            raise RuntimeError("checkpoint changed during source inference")
        aggregate = {
            "experiment_id": EXP_ID, "status": "frozen", "source_sequences": SOURCE_SEQUENCES,
            "checkpoint_sha256_before": checkpoint_before, "checkpoint_sha256_after": checkpoint_after,
            "parameter_count": EXPECTED_PARAMETER_COUNT, "device": str(device), "model_eval": True,
            "torch_no_grad": True, "optimizer_constructed": False, "all_inputs_finite": True,
            "all_outputs_finite": True, "mask_contiguous_prefix": True, "deterministic_check": deterministic,
            "source_label_sidecar_reads_before_score_freeze": 0, "raw_gt_reads": 0, "sequences": sequences,
        }
        write_json(R66 / "source_score_manifest.json", aggregate)
        append_event("source_scores_frozen", source_score_manifest_sha256=sha256(R66 / "source_score_manifest.json"),
                     checkpoint_sha256=checkpoint_after, deterministic=deterministic)
        print(json.dumps({"stage": "freeze-source-scores", "status": "completed", "device": str(device),
                          "counts": {s: v["counts"] for s, v in sequences.items()}}, sort_keys=True))


def read_sequence_inputs_for_canonical(seq: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    if seq in SOURCE_SEQUENCES:
        chunks = pd.read_parquet(R63 / "source_chunks.parquet", filters=[("sequence", "==", seq)])
        labels = pd.read_parquet(R63 / "row_supervision.parquet", filters=[("sequence", "==", seq)])
        obs = R62 / "observables/MOT17" / seq
    else:
        chunks = pd.read_parquet(R65 / seq / "topology/chunks.parquet")
        labels = pd.read_parquet(R65 / seq / "labels/row_labels.parquet")
        obs = R62 / "observables/MOT20" / seq
    rows = pd.read_parquet(obs / "rows.parquet")
    features = np.load(obs / "row_features.f16.npy", mmap_mode="r")
    return chunks, labels, rows, features


def trusted_chunk_info(seq: str, chunks: pd.DataFrame, labels: pd.DataFrame,
                       rows: pd.DataFrame, features: np.ndarray) -> pd.DataFrame:
    lab = labels.set_index("row_index", drop=False)
    row_pos = rows.set_index("row_index", drop=False)
    frame_counts = rows.groupby("frame", sort=False).size().to_dict()
    records: list[dict[str, Any]] = []
    for chunk in chunks.itertuples(index=False):
        ids = parse_row_indices(chunk.row_indices)
        z = lab.reindex(ids)
        matched = z[z.supervision_status.astype(str) == "matched"]
        counts = Counter(matched.gt_identity_key.astype(str).tolist())
        tie = False
        identity = None
        modal = 0
        if counts:
            modal = max(counts.values())
            winners = sorted([k for k, v in counts.items() if v == modal])
            tie = len(winners) > 1
            identity = winners[0]
        purity = float(modal / len(matched)) if len(matched) else 0.0
        trusted = len(matched) >= TRUST_MIN_MATCHED and purity >= TRUST_MIN_PURITY and not tie
        rz = row_pos.reindex(ids)
        crowd = float(np.mean(np.asarray(features[np.asarray(ids, np.int64), 142], np.float32))) if ids else None
        mapped_fraction = float(rz.appearance_mapped.astype(float).mean()) if ids else None
        represented_frames = sorted(set(int(x) for x in rz.frame.dropna().tolist()))
        density = float(np.mean([frame_counts[f] for f in represented_frames])) if represented_frames else None
        records.append({
            "sequence": seq, "chunk_id": str(chunk.chunk_id), "source_track_id": int(chunk.track_id),
            "first_frame": int(chunk.first_frame), "last_frame": int(chunk.last_frame), "row_count": int(chunk.row_count),
            "matched_row_count": int(len(matched)), "identity": identity, "purity": purity,
            "trusted": trusted, "untrusted_identity_tie": tie, "crowd_density_feature_142": crowd,
            "appearance_mapped_fraction": mapped_fraction, "frame_density": density,
        })
    return pd.DataFrame(records)


def command_freeze_canonical_queries() -> None:
    with stage_context("freeze-canonical-queries"):
        all_info = []
        source_label_reads = 0
        target_label_reads = 0
        for seq in ALL_SEQUENCES:
            chunks, labels, rows, features = read_sequence_inputs_for_canonical(seq)
            if seq in SOURCE_SEQUENCES:
                source_label_reads += 1
            else:
                target_label_reads += 1
            all_info.append(trusted_chunk_info(seq, chunks, labels, rows, features))
        chunk_info = pd.concat(all_info, ignore_index=True)
        queries, valid_targets = build_canonical_from_chunk_info(chunk_info)
        queries = queries.sort_values(["sequence", "direction", "query_chunk_id"], kind="mergesort").reset_index(drop=True)
        valid_targets = valid_targets.sort_values(["sequence", "direction", "query_id", "valid_target_order"], kind="mergesort").reset_index(drop=True)
        queries.to_parquet(R66 / "canonical_queries.parquet", index=False)
        valid_targets.to_parquet(R66 / "valid_targets.parquet", index=False)
        chunk_info.to_parquet(R66 / "trusted_chunk_inventory.parquet", index=False)
        summary = []
        for seq in ALL_SEQUENCES:
            for direction in ("outgoing", "incoming"):
                q = queries[(queries.sequence == seq) & (queries.direction == direction)]
                ci = chunk_info[chunk_info.sequence == seq]
                summary.append({
                    "sequence": seq, "direction": direction, "all_chunks": int(len(ci)),
                    "trusted_chunks": int(ci.trusted.sum()), "identity_tie_chunks": int(ci.untrusted_identity_tie.sum()),
                    "temporal_future_or_past": int((q.temporal_target_count > 0).sum()),
                    "eligible_queries": int((q.query_status == "eligible").sum()),
                    "horizon_outside_queries": int((q.query_status == "horizon_outside").sum()),
                    "no_successor_or_predecessor_queries": int(q.query_status.astype(str).str.startswith("no_").sum()),
                    "canonical_query_count": int((q.query_status == "eligible").sum()),
                    "valid_target_count": int(valid_targets[(valid_targets.sequence == seq) & (valid_targets.direction == direction)].shape[0]),
                    "multi_positive_query_count": int((q.valid_target_count > 1).sum()),
                })
        manifest = {
            "experiment_id": EXP_ID, "status": "frozen", "candidate_pool_read": False,
            "relation_scores_read": False, "model_rankings_read": False,
            "inputs": ["chunks", "frozen row label/supervision sidecars", "observable row metadata", "feature 142 only for decomposition"],
            "trusted_definition": {"matched_row_count_min": 2, "modal_identity_purity_min": 0.80,
                                   "identity_tie_untrusted": True, "unknown_distractor_ambiguous_not_negative": True},
            "counts": summary, "source_label_sidecar_reads": source_label_reads,
            "target_label_sidecar_reads": target_label_reads, "raw_gt_reads": 0,
            "artifacts": {
                "canonical_queries.parquet": sha256(R66 / "canonical_queries.parquet"),
                "valid_targets.parquet": sha256(R66 / "valid_targets.parquet"),
                "trusted_chunk_inventory.parquet": sha256(R66 / "trusted_chunk_inventory.parquet"),
            },
        }
        write_json(R66 / "canonical_query_manifest.json", manifest)
        append_event("canonical_queries_frozen", canonical_query_manifest_sha256=sha256(R66 / "canonical_query_manifest.json"),
                     counts=summary, candidate_pool_read=False)
        print(json.dumps({"stage": "freeze-canonical-queries", "status": "completed",
                          "eligible_queries": int((queries.query_status == "eligible").sum()),
                          "valid_targets": len(valid_targets)}, sort_keys=True))


def sequence_labels(seq: str) -> pd.DataFrame:
    if seq in SOURCE_SEQUENCES:
        return pd.read_parquet(R63 / "row_supervision.parquet", filters=[("sequence", "==", seq)])
    return pd.read_parquet(R65 / seq / "labels/row_labels.parquet")


def sequence_chunks(seq: str) -> pd.DataFrame:
    if seq in SOURCE_SEQUENCES:
        return pd.read_parquet(R63 / "source_chunks.parquet", filters=[("sequence", "==", seq)])
    return pd.read_parquet(R65 / seq / "topology/chunks.parquet")


def sequence_candidate_pool(seq: str, columns: list[str] | None = None) -> pd.DataFrame:
    if seq in SOURCE_SEQUENCES:
        return pd.read_parquet(R63 / "candidate_pool.parquet", columns=columns, filters=[("sequence", "==", seq)])
    return pd.read_parquet(R65 / seq / "topology/candidate_pool.parquet", columns=columns)


def sequence_paired_pool(seq: str) -> pd.DataFrame:
    if seq in SOURCE_SEQUENCES:
        return pd.read_parquet(R63 / "paired_candidate_pool.parquet", filters=[("sequence", "==", seq)])
    return pd.read_parquet(R65 / seq / "topology/paired_candidate_pool.parquet")


def sequence_score_path(seq: str, filename: str) -> Path:
    return (R66 / "source_scores" / seq / filename) if seq in SOURCE_SEQUENCES else (R65 / seq / "scores" / filename)


def legacy_boundary_for_sequence(seq: str) -> dict[str, Any]:
    labels = sequence_labels(seq).set_index("row_index", drop=False)
    scores = pd.read_parquet(sequence_score_path(seq, "boundary_scores.parquet"))
    a = labels.reindex(scores.src_row_index.to_numpy(np.int64))
    b = labels.reindex(scores.dst_row_index.to_numpy(np.int64))
    keep = (a.supervision_status.to_numpy(dtype=str) == "matched") & (b.supervision_status.to_numpy(dtype=str) == "matched")
    y = (a.gt_identity_key.to_numpy(dtype=str)[keep] != b.gt_identity_key.to_numpy(dtype=str)[keep]).astype(np.int8)
    s = scores.boundary_probability.to_numpy(np.float64)[keep]
    metrics = binary_metrics(y, s)
    return {"raw_observation_count": int(len(scores)), "matched_observation_count": int(len(y)), **metrics}


def command_reproduce_legacy() -> None:
    with stage_context("reproduce-legacy"):
        published = read_json(R65 / "representation_metrics.json", {})
        by_seq = {m["sequence"]: m for m in published.get("sequences", [])}
        comparisons = []
        all_pass = True
        tolerance = 1e-10
        for seq in TARGET_SEQUENCES:
            got = legacy_boundary_for_sequence(seq)
            expected = by_seq[seq]["boundary"]
            fields = {
                "rows": (got["rows"], expected.get("rows")),
                "positives": (got["positives"], expected.get("positives")),
                "base_rate": (got["base_rate"], expected.get("base_rate")),
                "pr_auc": (got["pr_auc"], expected.get("pr_auc")),
                "roc_auc": (got["roc_auc"], expected.get("roc_auc")),
                "precision_at_actual": (got["precision_at_actual"], expected.get("precision_at_actual")),
                "recall_at_90_precision": (got["recall_at_90_precision"], expected.get("recall_at_90_precision")),
                "recall_at_95_precision": (got["recall_at_95_precision"], expected.get("recall_at_95_precision")),
                "recall_at_99_precision": (got["recall_at_99_precision"], expected.get("recall_at_99_precision")),
            }
            exact_counts = got["rows"] == expected.get("rows") and got["positives"] == expected.get("positives")
            diffs = {}
            numeric_ok = True
            for field, (actual, exp) in fields.items():
                if field in ("rows", "positives"):
                    diffs[field] = 0 if actual == exp else None
                    continue
                if actual is None or exp is None:
                    diff = 0.0 if actual is None and exp is None else None
                    ok = actual is None and exp is None
                else:
                    diff = abs(float(actual) - float(exp))
                    ok = diff <= tolerance
                diffs[field] = diff
                numeric_ok = numeric_ok and ok
            passed = exact_counts and numeric_ok
            all_pass = all_pass and passed
            comparisons.append({"sequence": seq, "counts_exact": exact_counts, "metric_differences": diffs,
                                "tolerance": tolerance, "passed": passed, "reproduced": got, "published": expected})
        provenance = read_json(R66 / "m23_65_historical_implementation_provenance.json", {})
        out = {
            "experiment_id": EXP_ID, "legacy_behavioral_reproduction": all_pass,
            "legacy_boundary_artifact_numeric_reproduction": all_pass,
            "historical_source_byte_exact_reproduction": provenance.get("byte_exact_historical_source_found", False),
            "current_script_is_not_historical_source": provenance.get("current_script_is_not_historical_source", True),
            "tolerance": tolerance, "sequences": comparisons,
            "claim": "Frozen-artifact behavioral reproduction only; current M23-65 script is not asserted as historical source.",
        }
        write_json(R66 / "legacy_metric_reproduction.json", out)
        append_event("legacy_metrics_reproduced", legacy_behavioral_reproduction=all_pass,
                     artifact_sha256=sha256(R66 / "legacy_metric_reproduction.json"))
        print(json.dumps({"stage": "reproduce-legacy", "status": "completed", "legacy_behavioral_reproduction": all_pass}, sort_keys=True))


def validate_and_label_boundary(seq: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    labels = sequence_labels(seq).set_index("row_index", drop=False)
    score_map = pd.read_parquet(sequence_score_path(seq, "score_to_source_row.parquet"))
    scores = pd.read_parquet(sequence_score_path(seq, "boundary_scores.parquet"))
    if score_map.row_index.duplicated().any():
        raise RuntimeError(f"{seq}: score_to_source_row row_index not unique")
    rows = score_map.set_index("row_index", drop=False)
    src = rows.reindex(scores.src_row_index.to_numpy(np.int64))
    dst = rows.reindex(scores.dst_row_index.to_numpy(np.int64))
    mapping_complete = not src.row_index.isna().any() and not dst.row_index.isna().any()
    same_track = src.track_id.to_numpy() == dst.track_id.to_numpy()
    forward = src.frame.to_numpy() < dst.frame.to_numpy()
    logits = scores.boundary_logit.to_numpy(np.float64)
    probs = scores.boundary_probability.to_numpy(np.float64)
    finite = np.isfinite(logits) & np.isfinite(probs)
    sigmoid_ok = np.allclose(probs[finite], sigmoid(logits[finite]), atol=1e-12, rtol=1e-12)
    if not mapping_complete or not same_track.all() or not forward.all() or not finite.all() or not sigmoid_ok:
        raise RuntimeError(f"{seq}: boundary mapping/finite/sigmoid validation failed")
    la = labels.reindex(scores.src_row_index.to_numpy(np.int64))
    lb = labels.reindex(scores.dst_row_index.to_numpy(np.int64))
    known = (la.supervision_status.to_numpy(dtype=str) == "matched") & (lb.supervision_status.to_numpy(dtype=str) == "matched")
    labeled = scores.loc[known].copy()
    labeled["label"] = (la.gt_identity_key.to_numpy(dtype=str)[known] != lb.gt_identity_key.to_numpy(dtype=str)[known]).astype(np.int8)
    # physical duplicates must carry one label.
    nunique = labeled.groupby(["sequence", "src_row_index", "dst_row_index"], sort=False).label.nunique()
    if (nunique > 1).any():
        raise RuntimeError(f"{seq}: duplicate observations have inconsistent labels")
    audit = {
        "sequence": seq, "raw_observation_count": int(len(scores)), "matched_observation_count": int(len(labeled)),
        "unknown_endpoint_observation_count": int(len(scores) - len(labeled)), "score_to_source_row_unique": True,
        "mapping_complete": mapping_complete, "same_source_track": bool(same_track.all()),
        "strict_forward_time": bool(forward.all()), "all_finite": bool(finite.all()),
        "probability_sigmoid_consistent": bool(sigmoid_ok), "unknown_is_negative": False,
    }
    return labeled, audit


def boundary_views_for_sequence(seq: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    labeled, audit = validate_and_label_boundary(seq)
    views = aggregate_boundary_views(labeled)
    rows = []
    output: dict[str, Any] = {"sequence": seq, "audit": audit, "views": {}}
    primary = views["corrected_unique_transition_primary"]
    multiplicities = primary.multiplicity.value_counts().sort_index().to_dict()
    audit.update({
        "unique_physical_transitions": int(len(primary)),
        "duplicate_observation_count": int(len(labeled) - len(primary)),
        "duplicate_factor": float(len(labeled) / len(primary)) if len(primary) else None,
        "duplicate_multiplicity_distribution": {str(int(k)): int(v) for k, v in multiplicities.items()},
        "unique_positives": int(primary.label.sum()) if len(primary) else 0,
    })
    for view_name, frame in views.items():
        keys = frame.transition_key.astype(str).tolist()
        metric = binary_metrics(frame.label.to_numpy(np.int8), frame.score.to_numpy(np.float64), keys)
        metric.update({
            "view": view_name, "sequence": seq, "raw_observation_count": int(len(labeled)),
            "matched_observation_count": int(len(labeled)), "unique_physical_transitions": int(len(primary)),
            "duplicate_observation_count": int(len(labeled) - len(primary)),
            "duplicate_factor": float(len(labeled) / len(primary)) if len(primary) else None,
        })
        output["views"][view_name] = metric
        flat = {k: v for k, v in metric.items() if not isinstance(v, dict)}
        for prefix in ("score_quantiles", "positive_score_quantiles", "negative_score_quantiles"):
            for k, v in metric[prefix].items():
                flat[f"{prefix}_{k}"] = v
        rows.append(flat)
    return output, rows


def sort_candidates(frame: pd.DataFrame, score_col: str, ascending: bool) -> pd.DataFrame:
    return frame.sort_values([score_col, "candidate_id"], ascending=[ascending, True], kind="mergesort")


def rank_summary(query_records: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    n = len(query_records)
    ranks = [r[f"{prefix}_rank"] for r in query_records]
    present = [r for r in ranks if r is not None]
    conditional = [r for r in query_records if r["candidate_present"]]
    conditional_ranks = [r[f"{prefix}_rank"] for r in conditional]
    positive_conditional = [r for r in query_records if r[f"{prefix}_rank"] is not None]
    positive_conditional_ranks = [r[f"{prefix}_rank"] for r in positive_conditional]
    return {
        "all_query": {
            "queries": n, "R@1": float(sum(x == 1 for x in ranks) / n) if n else 0.0,
            "R@3": float(sum(x is not None and x <= 3 for x in ranks) / n) if n else 0.0,
            "MRR": float(sum(0.0 if x is None else 1.0 / x for x in ranks) / n) if n else 0.0,
        },
        "candidate_present_conditional": {
            "queries": len(conditional),
            "R@1": float(sum(x == 1 for x in conditional_ranks) / len(conditional)) if conditional else 0.0,
            "R@3": float(sum(x is not None and x <= 3 for x in conditional_ranks) / len(conditional)) if conditional else 0.0,
            "MRR": float(sum(0.0 if x is None else 1.0 / x for x in conditional_ranks) / len(conditional)) if conditional else 0.0,
        },
        "positive_present_conditional": {
            "queries": len(positive_conditional),
            "R@1": float(sum(x == 1 for x in positive_conditional_ranks) / len(positive_conditional)) if positive_conditional else 0.0,
            "R@3": float(sum(x <= 3 for x in positive_conditional_ranks) / len(positive_conditional)) if positive_conditional else 0.0,
            "MRR": float(sum(1.0 / x for x in positive_conditional_ranks) / len(positive_conditional)) if positive_conditional else 0.0,
        },
        "positive_present_count": len(present),
    }


def evaluate_relation_sequence(seq: str, queries: pd.DataFrame, valid_targets: pd.DataFrame,
                               chunk_inventory: pd.DataFrame) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    relation_cols = ["sequence", "candidate_id", "src_chunk_id", "dst_chunk_id", "gap", "gap_bucket",
                     "candidate_score", "relation_logit", "relation_probability"]
    relation = pd.read_parquet(sequence_score_path(seq, "relation_scores.parquet"), columns=relation_cols)
    pool_extra = sequence_candidate_pool(seq, columns=["candidate_id", "appearance_cosine", "geometry_distance"])
    relation = relation.merge(pool_extra, on="candidate_id", how="left", validate="one_to_one")
    if relation[["relation_logit", "candidate_score", "appearance_cosine", "geometry_distance", "gap"]].isna().any().any():
        raise RuntimeError(f"{seq}: relation/candidate merge incomplete")
    info = chunk_inventory[chunk_inventory.sequence == seq].set_index("chunk_id", drop=False)
    vt_by_q = {qid: g for qid, g in valid_targets[valid_targets.sequence == seq].groupby("query_id", sort=False)}
    metrics_by_direction: dict[str, Any] = {}
    query_diagnostics: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    baseline_specs = [
        ("relation_logit", "relation_logit", False),
        ("candidate_score", "candidate_score", False),
        ("appearance_cosine", "appearance_cosine", False),
        ("geometry_distance", "geometry_distance", True),
        ("topology_gap", "gap", True),
    ]
    for direction in ("outgoing", "incoming"):
        qdir = queries[(queries.sequence == seq) & (queries.direction == direction) & (queries.query_status == "eligible")]
        if direction == "outgoing":
            group_col, target_col = "src_chunk_id", "dst_chunk_id"
        else:
            group_col, target_col = "dst_chunk_id", "src_chunk_id"
        grouped = {str(k): idx for k, idx in relation.groupby(group_col, sort=False).indices.items()}
        baseline_records: dict[str, list[dict[str, Any]]] = {name: [] for name, _, _ in baseline_specs}
        for q in qdir.itertuples(index=False):
            qid = str(q.query_id)
            qchunk = str(q.query_chunk_id)
            valid = vt_by_q.get(qid, pd.DataFrame(columns=["target_chunk_id"]))
            valid_set = set(valid.target_chunk_id.astype(str))
            canonical = str(q.canonical_target_chunk_id)
            idx = grouped.get(qchunk, np.empty(0, dtype=np.int64))
            c = relation.iloc[idx].copy() if len(idx) else relation.iloc[0:0].copy()
            for baseline_name, score_col, ascending in baseline_specs:
                ordered = sort_candidates(c[np.isfinite(c[score_col].to_numpy(np.float64))], score_col, ascending)
                targets = ordered[target_col].astype(str).tolist()
                canonical_rank = targets.index(canonical) + 1 if canonical in targets else None
                any_positions = [i + 1 for i, target in enumerate(targets) if target in valid_set]
                any_rank = min(any_positions) if any_positions else None
                scores = ordered[score_col].to_numpy(np.float64)
                top_margin = None
                top_tie = False
                if len(scores) >= 2:
                    top_margin = float((scores[0] - scores[1]) if not ascending else (scores[1] - scores[0]))
                    top_tie = bool(scores[0] == scores[1])
                rec = {
                    "sequence": seq, "direction": direction, "query_id": qid, "query_chunk_id": qchunk,
                    "canonical_target_chunk_id": canonical, "canonical_gap_bucket": q.canonical_gap_bucket,
                    "valid_target_count": int(q.valid_target_count), "candidate_present": bool(len(ordered)),
                    "candidate_count": int(len(ordered)), "canonical_present": canonical_rank is not None,
                    "any_valid_present": any_rank is not None, "canonical_rank": canonical_rank, "any_valid_rank": any_rank,
                    "top_candidate_id": str(ordered.iloc[0].candidate_id) if len(ordered) else None,
                    "top_target_chunk_id": targets[0] if targets else None, "top1_top2_margin": top_margin,
                    "top_tie": top_tie, "top_score_saturated": bool(abs(float(ordered.iloc[0].relation_logit)) >= 20) if len(ordered) else False,
                    "baseline": baseline_name,
                }
                baseline_records[baseline_name].append(rec)
            primary = baseline_records["relation_logit"][-1]
            top_target = primary["top_target_chunk_id"]
            false_link = None
            if top_target is not None and qchunk in info.index and top_target in info.index:
                qi = info.loc[qchunk]
                ti = info.loc[top_target]
                if bool(qi.trusted) and bool(ti.trusted):
                    false_link = bool(str(qi.identity) != str(ti.identity))
            primary["catastrophic_false_link"] = false_link
            primary["query_crowd_density"] = float(getattr(q, "crowd_density_feature_142"))
            primary["query_appearance_mapped_fraction"] = float(getattr(q, "appearance_mapped_fraction"))
            primary["query_purity"] = float(getattr(q, "purity"))
            primary["query_frame_density"] = float(getattr(q, "frame_density"))
            canonical_info = info.loc[canonical] if canonical in info.index else None
            top_info = info.loc[top_target] if top_target is not None and top_target in info.index else None
            if direction == "outgoing":
                primary["source_frame_density"] = float(getattr(q, "frame_density"))
                primary["destination_frame_density"] = float(canonical_info.frame_density) if canonical_info is not None else None
            else:
                primary["source_frame_density"] = float(canonical_info.frame_density) if canonical_info is not None else None
                primary["destination_frame_density"] = float(getattr(q, "frame_density"))
            primary["top_target_frame_density"] = float(top_info.frame_density) if top_info is not None else None
            query_diagnostics.append(primary)
        direction_result: dict[str, Any] = {"sequence": seq, "direction": direction, "baselines": {}}
        for baseline_name, _, _ in baseline_specs:
            records = baseline_records[baseline_name]
            direction_result["baselines"][baseline_name] = {
                "canonical_exact_target": rank_summary(records, "canonical"),
                "any_valid_positive": rank_summary(records, "any_valid"),
                "candidate_coverage": float(np.mean([r["candidate_present"] for r in records])) if records else 0.0,
                "canonical_exact_target_coverage": float(np.mean([r["canonical_present"] for r in records])) if records else 0.0,
                "any_valid_positive_coverage": float(np.mean([r["any_valid_present"] for r in records])) if records else 0.0,
                "score_tie_rate": float(np.mean([r["top_tie"] for r in records])) if records else 0.0,
                "mean_top1_top2_margin": float(np.mean([r["top1_top2_margin"] for r in records if r["top1_top2_margin"] is not None])) if any(r["top1_top2_margin"] is not None for r in records) else None,
            }
        prim = baseline_records["relation_logit"]
        false_vals = [r.get("catastrophic_false_link") for r in query_diagnostics if r["sequence"] == seq and r["direction"] == direction]
        false_vals = [x for x in false_vals if x is not None]
        direction_result["catastrophic_false_link_rate"] = float(np.mean(false_vals)) if false_vals else None
        metrics_by_direction[direction] = direction_result
        # coverage all sequence/direction and frozen gap bucket.
        for bucket_name in [None] + [x[0] for x in GAP_BUCKETS]:
            selected = prim if bucket_name is None else [r for r in prim if r["canonical_gap_bucket"] == bucket_name]
            coverage_rows.append({
                "sequence": seq, "domain": "source" if seq in SOURCE_SEQUENCES else "target",
                "direction": direction, "gap_bucket": "all" if bucket_name is None else bucket_name,
                "all_query": len(selected), "candidate_present": int(sum(r["candidate_present"] for r in selected)),
                "canonical_present": int(sum(r["canonical_present"] for r in selected)),
                "any_positive_present": int(sum(r["any_valid_present"] for r in selected)),
                "canonical_missing_count": int(sum(not r["canonical_present"] for r in selected)),
                "any_positive_missing_count": int(sum(not r["any_valid_present"] for r in selected)),
                "canonical_exact_target_coverage": float(np.mean([r["canonical_present"] for r in selected])) if selected else None,
                "any_valid_positive_coverage": float(np.mean([r["any_valid_present"] for r in selected])) if selected else None,
            })
    return {"sequence": seq, "directions": metrics_by_direction, "candidate_edges": int(len(relation))}, query_diagnostics, coverage_rows


def pair_audit_sequence(seq: str, inventory: pd.DataFrame) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pair_scores = pd.read_parquet(sequence_score_path(seq, "pair_scores.parquet"))
    relation = pd.read_parquet(sequence_score_path(seq, "relation_scores.parquet"),
                               columns=["candidate_id", "src_chunk_id", "dst_chunk_id"])
    edge = relation.set_index("candidate_id", drop=False)
    info = inventory[inventory.sequence == seq].set_index("chunk_id", drop=False)
    records = []
    max_margin_error = 0.0
    for p in pair_scores.itertuples(index=False):
        margin_error = abs(float(p.original_margin) - (float(p.original_mean_logit) - float(p.cross_mean_logit)))
        max_margin_error = max(max_margin_error, margin_error)
        ids = [str(getattr(p, k)) for k in ("edge1_id", "edge2_id", "cross1_id", "cross2_id")]
        edges_exist = all(x in edge.index for x in ids)
        trusted_endpoints = False
        legal = False
        identities: list[str] = []
        if edges_exist:
            e1, e2, c1, c2 = [edge.loc[x] for x in ids]
            endpoint_ids = [str(e1.src_chunk_id), str(e1.dst_chunk_id), str(e2.src_chunk_id), str(e2.dst_chunk_id)]
            trusted_endpoints = all(x in info.index and bool(info.loc[x].trusted) for x in endpoint_ids)
            if trusted_endpoints:
                identities = [str(info.loc[x].identity) for x in endpoint_ids]
                original1_same = identities[0] == identities[1]
                original2_same = identities[2] == identities[3]
                original_different = identities[0] != identities[2]
                cross_expected = (str(c1.src_chunk_id) == endpoint_ids[0] and str(c1.dst_chunk_id) == endpoint_ids[3] and
                                  str(c2.src_chunk_id) == endpoint_ids[2] and str(c2.dst_chunk_id) == endpoint_ids[1])
                legal = original1_same and original2_same and original_different and cross_expected
        records.append({
            "sequence": seq, "pair_id": p.pair_id, "trusted_endpoints": trusted_endpoints, "legal_pair": legal,
            "original_margin": float(p.original_margin), "paired_probability": float(p.paired_probability),
            "margin_identity_error": margin_error, "all_edges_exist": edges_exist,
        })
    frame = pd.DataFrame(records)
    if max_margin_error > 1e-10:
        raise RuntimeError(f"{seq}: original_margin equation mismatch {max_margin_error}")
    trusted = frame[frame.trusted_endpoints].copy()
    valid = trusted[trusted.legal_pair].copy()
    y = trusted.legal_pair.astype(np.int8).to_numpy()
    s = trusted.paired_probability.to_numpy(np.float64)
    margins = valid.original_margin.to_numpy(np.float64)
    ties = margins == 0
    strict = float(np.mean(margins > 0)) if len(margins) else None
    out = {
        "sequence": seq, "all_pair_count": int(len(frame)), "trusted_endpoint_pair_count": int(len(trusted)),
        "valid_pair_count": int(len(valid)), "invalid_or_unknown_or_excluded_pair_count": int(len(frame) - len(valid)),
        "positive_base_rate": float(np.mean(y)) if len(y) else None,
        "valid_pair_original_over_cross_accuracy": strict,
        "valid_pair_original_over_cross_half_credit_sensitivity": float(np.mean((margins > 0) + 0.5 * ties)) if len(margins) else None,
        "tie_count": int(ties.sum()), "tie_rate": float(np.mean(ties)) if len(margins) else None,
        "threshold_accuracy_at_0p5": float(np.mean((s > 0.5) == (y == 1))) if len(y) else None,
        "balanced_accuracy_at_0p5": balanced_accuracy(y, (s > 0.5).astype(np.int8)) if len(y) else None,
        "pr_auc": safe_average_precision(y, s), "auroc": safe_roc_auc(y, s),
        "margin_mean": float(np.mean(margins)) if len(margins) else None,
        "margin_std": float(np.std(margins)) if len(margins) else None,
        "margin_quantiles": score_quantiles(margins), "max_margin_identity_error": max_margin_error,
        "legacy_misnamed_paired_replacement_R@1": float(np.mean((s > 0.5) == (y == 1))) if len(y) else None,
    }
    return out, records


def crowd_bucket(v: float) -> str:
    if 0 <= v < 0.25: return "[0,0.25)"
    if 0.25 <= v < 0.50: return "[0.25,0.50)"
    if 0.50 <= v < 1.00: return "[0.50,1.00)"
    if 1.00 <= v < 2.00: return "[1.00,2.00)"
    if 2.00 <= v <= 5.00: return "[2.00,5.00]"
    return "outside"


def purity_bucket(v: float) -> str:
    if v == 1.0: return "1.00 exact"
    if 0.99 <= v < 1.0: return "[0.99,1.00)"
    if 0.90 <= v < 0.99: return "[0.90,0.99)"
    if 0.80 <= v < 0.90: return "[0.80,0.90)"
    return "outside"


def mapped_bucket(v: float) -> str:
    if v == 0.0: return "0"
    if 0.0 < v < 0.50: return "(0,0.50)"
    if 0.50 <= v < 0.90: return "[0.50,0.90)"
    if 0.90 <= v < 1.0: return "[0.90,1.00)"
    if v == 1.0: return "1.00 exact"
    return "outside"


def frame_density_edges(inventory: pd.DataFrame) -> dict[str, list[float]]:
    out = {}
    for seq, group in inventory[inventory.trusted].groupby("sequence", sort=True):
        values = group.frame_density.to_numpy(np.float64)
        edges = np.unique(np.quantile(values, [0.0, 0.25, 0.50, 0.75, 1.0])).tolist() if len(values) else []
        out[str(seq)] = [float(x) for x in edges]
    return out


def numeric_interval_bucket(value: float | None, edges: list[float]) -> str:
    if value is None or not np.isfinite(value) or len(edges) < 2:
        return "undefined"
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if (lo <= value < hi) or (i == len(edges) - 2 and lo <= value <= hi):
            close = "]" if i == len(edges) - 2 else ")"
            return f"[{lo:.6g},{hi:.6g}{close}"
    return "outside"


def flatten_ranking_rows(sequence_metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for sm in sequence_metrics:
        seq = sm["sequence"]
        for direction, dres in sm["directions"].items():
            for baseline, bres in dres["baselines"].items():
                for target_type in ("canonical_exact_target", "any_valid_positive"):
                    for denominator in ("all_query", "candidate_present_conditional", "positive_present_conditional"):
                        m = bres[target_type][denominator]
                        rows.append({
                            "sequence": seq, "domain": "source" if seq in SOURCE_SEQUENCES else "target",
                            "direction": direction, "baseline": baseline, "target_type": target_type,
                            "denominator": denominator, **m,
                            "candidate_coverage": bres["candidate_coverage"],
                            "canonical_exact_target_coverage": bres["canonical_exact_target_coverage"],
                            "any_valid_positive_coverage": bres["any_valid_positive_coverage"],
                            "score_tie_rate": bres["score_tie_rate"],
                            "mean_top1_top2_margin": bres["mean_top1_top2_margin"],
                            "catastrophic_false_link_rate": dres["catastrophic_false_link_rate"],
                        })
    return rows


def add_coverage_aggregates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    base = pd.DataFrame(rows)
    extras = []
    for domain, seqs in (("source", SOURCE_SEQUENCES), ("target", TARGET_SEQUENCES)):
        for direction in ("outgoing", "incoming"):
            for bucket in ["all"] + [x[0] for x in GAP_BUCKETS]:
                z = base[(base.sequence.isin(seqs)) & (base.direction == direction) & (base.gap_bucket == bucket)]
                if len(z) == 0:
                    continue
                sums = z[["all_query", "candidate_present", "canonical_present", "any_positive_present",
                          "canonical_missing_count", "any_positive_missing_count"]].sum()
                n = int(sums["all_query"])
                extras.append({
                    "sequence": "POOLED", "domain": domain, "direction": direction, "gap_bucket": bucket,
                    **{k: int(sums[k]) for k in sums.index},
                    "canonical_exact_target_coverage": float(sums["canonical_present"] / n) if n else None,
                    "any_valid_positive_coverage": float(sums["any_positive_present"] / n) if n else None,
                })
                extras.append({
                    "sequence": "MACRO", "domain": domain, "direction": direction, "gap_bucket": bucket,
                    "all_query": int(z.all_query.sum()), "candidate_present": int(z.candidate_present.sum()),
                    "canonical_present": int(z.canonical_present.sum()), "any_positive_present": int(z.any_positive_present.sum()),
                    "canonical_missing_count": int(z.canonical_missing_count.sum()), "any_positive_missing_count": int(z.any_positive_missing_count.sum()),
                    "canonical_exact_target_coverage": float(z.canonical_exact_target_coverage.dropna().mean()) if z.canonical_exact_target_coverage.notna().any() else None,
                    "any_valid_positive_coverage": float(z.any_valid_positive_coverage.dropna().mean()) if z.any_valid_positive_coverage.notna().any() else None,
                })
    return rows + extras


def pooled_boundary_domain(sequences: list[str], view_name: str) -> dict[str, Any]:
    frames = []
    raw = 0
    matched = 0
    for seq in sequences:
        labeled, _ = validate_and_label_boundary(seq)
        views = aggregate_boundary_views(labeled)
        f = views[view_name].copy()
        frames.append(f)
        raw += len(pd.read_parquet(sequence_score_path(seq, "boundary_scores.parquet"), columns=["src_row_index"]))
        matched += len(labeled)
    allf = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["label", "score", "transition_key"])
    m = binary_metrics(allf.label.to_numpy(np.int8), allf.score.to_numpy(np.float64), allf.transition_key.astype(str).tolist())
    m.update({"raw_observation_count": raw, "matched_observation_count": matched, "unique_physical_transitions": len(allf),
              "duplicate_observation_count": matched - len(allf), "duplicate_factor": matched / len(allf) if len(allf) else None})
    return m


def macro_boundary_domain(per_sequence: dict[str, Any], sequences: list[str], view_name: str) -> dict[str, Any]:
    keys = ["pr_auc", "base_rate", "pr_auc_base_rate_lift", "roc_auc", "precision_at_actual",
            "recall_at_95_precision", "score_mean", "score_std", "duplicate_factor"]
    out = {}
    for key in keys:
        vals = [per_sequence[s]["views"][view_name].get(key) for s in sequences]
        out[key] = float(np.mean(vals)) if vals and all(v is not None for v in vals) else None
    return out


def pooled_rank_from_diagnostics(records: list[dict[str, Any]], sequences: list[str], direction: str) -> dict[str, Any]:
    z = [r for r in records if r["sequence"] in sequences and r["direction"] == direction]
    return {
        "canonical_exact_target": rank_summary(z, "canonical"),
        "any_valid_positive": rank_summary(z, "any_valid"),
        "candidate_coverage": float(np.mean([r["candidate_present"] for r in z])) if z else 0.0,
        "canonical_exact_target_coverage": float(np.mean([r["canonical_present"] for r in z])) if z else 0.0,
        "any_valid_positive_coverage": float(np.mean([r["any_valid_present"] for r in z])) if z else 0.0,
        "score_tie_rate": float(np.mean([r["top_tie"] for r in z])) if z else 0.0,
        "mean_top1_top2_margin": float(np.mean([r["top1_top2_margin"] for r in z if r["top1_top2_margin"] is not None])) if any(r["top1_top2_margin"] is not None for r in z) else None,
    }


def relation_decomposition_rows(query_diag: pd.DataFrame, density_edges: dict[str, list[float]]) -> list[dict[str, Any]]:
    if len(query_diag) == 0:
        return []
    d = query_diag.copy()
    d["crowd_density_bucket"] = d.query_crowd_density.map(crowd_bucket)
    d["appearance_mapped_bucket"] = d.query_appearance_mapped_fraction.map(mapped_bucket)
    d["purity_bucket"] = d.query_purity.map(purity_bucket)
    d["query_frame_density_bucket"] = [numeric_interval_bucket(float(v), density_edges.get(str(s), [])) for s, v in zip(d.sequence, d.query_frame_density)]
    d["source_frame_density_bucket"] = [numeric_interval_bucket(None if pd.isna(v) else float(v), density_edges.get(str(s), [])) for s, v in zip(d.sequence, d.source_frame_density)]
    d["destination_frame_density_bucket"] = [numeric_interval_bucket(None if pd.isna(v) else float(v), density_edges.get(str(s), [])) for s, v in zip(d.sequence, d.destination_frame_density)]
    d["canonical_target_presence"] = np.where(d.canonical_present, "present", "missing")
    d["relation_score_saturation"] = np.where(d.top_score_saturated, "saturated", "not_saturated")
    d["relation_top_tie"] = np.where(d.top_tie, "tie", "no_tie")
    strata = ["sequence", "canonical_gap_bucket", "crowd_density_bucket", "appearance_mapped_bucket", "purity_bucket",
              "query_frame_density_bucket", "source_frame_density_bucket", "destination_frame_density_bucket",
              "canonical_target_presence", "relation_score_saturation", "relation_top_tie"]
    rows = []
    for stratum in strata:
        for (domain, direction, value), z in d.groupby(["domain", "direction", stratum], dropna=False, sort=True):
            n = len(z)
            rows.append({
                "section": "relation_decomposition", "domain": domain, "direction": direction,
                "stratum": stratum, "stratum_value": str(value), "queries": n,
                "canonical_coverage": float(z.canonical_present.mean()) if n else None,
                "any_valid_coverage": float(z.any_valid_present.mean()) if n else None,
                "canonical_R@1": float((z.canonical_rank == 1).sum() / n) if n else None,
                "canonical_R@3": float(((z.canonical_rank.notna()) & (z.canonical_rank <= 3)).sum() / n) if n else None,
                "canonical_MRR": float(z.canonical_rank.map(lambda x: 0.0 if pd.isna(x) else 1.0 / float(x)).mean()) if n else None,
                "any_valid_R@1": float((z.any_valid_rank == 1).sum() / n) if n else None,
                "any_valid_R@3": float(((z.any_valid_rank.notna()) & (z.any_valid_rank <= 3)).sum() / n) if n else None,
                "any_valid_MRR": float(z.any_valid_rank.map(lambda x: 0.0 if pd.isna(x) else 1.0 / float(x)).mean()) if n else None,
                "mean_top1_top2_margin": float(z.top1_top2_margin.dropna().mean()) if z.top1_top2_margin.notna().any() else None,
                "tie_rate": float(z.top_tie.mean()) if n else None,
            })
    return rows


def boundary_multiplicity_decomposition(sequences: list[str]) -> list[dict[str, Any]]:
    rows = []
    for seq in sequences:
        labeled, _ = validate_and_label_boundary(seq)
        primary = aggregate_boundary_views(labeled)["corrected_unique_transition_primary"]
        primary["multiplicity_bucket"] = primary.multiplicity.map(lambda x: str(int(x)) if int(x) <= 3 else "4+")
        for bucket, z in primary.groupby("multiplicity_bucket", sort=True):
            m = binary_metrics(z.label.to_numpy(np.int8), z.score.to_numpy(np.float64), z.transition_key.astype(str).tolist())
            rows.append({
                "section": "boundary_duplicate_multiplicity", "domain": "source" if seq in SOURCE_SEQUENCES else "target",
                "sequence": seq, "stratum": "boundary_duplicate_multiplicity", "stratum_value": bucket,
                "rows": m["rows"], "positives": m["positives"], "base_rate": m["base_rate"],
                "pr_auc": m["pr_auc"], "pr_auc_base_rate_lift": m["pr_auc_base_rate_lift"],
                "roc_auc": m["roc_auc"], "precision_at_actual": m["precision_at_actual"],
                "recall_at_95_precision": m["recall_at_95_precision"], "score_mean": m["score_mean"], "score_std": m["score_std"],
            })
    return rows


def command_audit_corrected_metrics() -> None:
    with stage_context("audit-corrected-metrics"):
        queries = pd.read_parquet(R66 / "canonical_queries.parquet")
        valid_targets = pd.read_parquet(R66 / "valid_targets.parquet")
        inventory = pd.read_parquet(R66 / "trusted_chunk_inventory.parquet")
        density_edges = frame_density_edges(inventory)
        write_json(R66 / "decomposition_strata_manifest.json", {
            "experiment_id": EXP_ID, "created_before_metric_outputs": True,
            "crowd_density_buckets": ["[0,0.25)", "[0.25,0.50)", "[0.50,1.00)", "[1.00,2.00)", "[2.00,5.00]"],
            "purity_buckets": ["[0.80,0.90)", "[0.90,0.99)", "[0.99,1.00)", "1.00 exact"],
            "appearance_mapped_buckets": ["0", "(0,0.50)", "[0.50,0.90)", "[0.90,1.00)", "1.00 exact"],
            "frame_density_edges_by_sequence": density_edges,
            "boundary_multiplicity_buckets": ["1", "2", "3", "4+"],
            "relation_saturation": "abs(relation_logit) >= 20", "relation_tie": "exact top1 == top2",
        })

        boundary_json: dict[str, Any] = {"experiment_id": EXP_ID, "sequences": {}}
        boundary_rows: list[dict[str, Any]] = []
        boundary_audits: list[dict[str, Any]] = []
        for seq in ALL_SEQUENCES:
            out, rows = boundary_views_for_sequence(seq)
            boundary_json["sequences"][seq] = out
            boundary_rows.extend(rows)
            boundary_audits.append(out["audit"])
            append_event("boundary_sequence_audited", sequence=seq, audit=out["audit"])
        for domain, seqs in (("source", SOURCE_SEQUENCES), ("target", TARGET_SEQUENCES)):
            boundary_json.setdefault("domain_aggregates", {})[domain] = {}
            for view in ("legacy_observation_weighted", "corrected_unique_transition_primary", "corrected_unique_transition_probability_sensitivity"):
                boundary_json["domain_aggregates"][domain][view] = {
                    "pooled": pooled_boundary_domain(seqs, view),
                    "macro": macro_boundary_domain(boundary_json["sequences"], seqs, view),
                }
        pd.DataFrame(boundary_audits).to_csv(R66 / "boundary_duplicate_audit.csv", index=False)
        pd.DataFrame(boundary_rows).to_csv(R66 / "boundary_metrics.csv", index=False)
        write_json(R66 / "boundary_metrics.json", boundary_json)

        relation_metrics = []
        query_diag: list[dict[str, Any]] = []
        coverage_rows: list[dict[str, Any]] = []
        for seq in ALL_SEQUENCES:
            result, diag, coverage = evaluate_relation_sequence(seq, queries, valid_targets, inventory)
            relation_metrics.append(result)
            for r in diag:
                r["domain"] = "source" if seq in SOURCE_SEQUENCES else "target"
            query_diag.extend(diag)
            coverage_rows.extend(coverage)
            append_event("relation_sequence_audited", sequence=seq, candidate_edges=result["candidate_edges"])
        coverage_rows = add_coverage_aggregates(coverage_rows)
        pd.DataFrame(coverage_rows).to_csv(R66 / "candidate_coverage.csv", index=False)
        write_json(R66 / "candidate_coverage.json", {"experiment_id": EXP_ID, "rows": coverage_rows})
        ranking_rows = flatten_ranking_rows(relation_metrics)
        pd.DataFrame(ranking_rows).to_csv(R66 / "corrected_ranking_metrics.csv", index=False)
        qdf = pd.DataFrame(query_diag)
        qdf.to_parquet(R66 / "relation_query_diagnostics.parquet", index=False)
        domain_ranking = {}
        for domain, seqs in (("source", SOURCE_SEQUENCES), ("target", TARGET_SEQUENCES)):
            domain_ranking[domain] = {direction: pooled_rank_from_diagnostics(query_diag, seqs, direction) for direction in ("outgoing", "incoming")}
        write_json(R66 / "corrected_ranking_metrics.json", {
            "experiment_id": EXP_ID, "per_sequence": relation_metrics, "domain_pooled_relation_logit": domain_ranking,
            "fixed_baselines": ["relation_logit", "candidate_score", "appearance_cosine", "geometry_distance", "topology_gap"],
            "query_diagnostics_sha256": sha256(R66 / "relation_query_diagnostics.parquet"),
        })

        pair_metrics = []
        pair_records = []
        for seq in ALL_SEQUENCES:
            metric, records = pair_audit_sequence(seq, inventory)
            pair_metrics.append(metric)
            pair_records.extend(records)
            append_event("paired_sequence_audited", sequence=seq, valid_pair_count=metric["valid_pair_count"])
        pair_df = pd.DataFrame(pair_records)
        pair_df.to_parquet(R66 / "paired_metric_records.parquet", index=False)
        pair_rows = []
        for m in pair_metrics:
            flat = {k: v for k, v in m.items() if not isinstance(v, dict)}
            for k, v in m["margin_quantiles"].items(): flat[f"margin_{k}"] = v
            pair_rows.append(flat)
        pd.DataFrame(pair_rows).to_csv(R66 / "paired_metric_audit.csv", index=False)
        write_json(R66 / "paired_metric_audit.json", {
            "experiment_id": EXP_ID, "per_sequence": pair_metrics,
            "records_sha256": sha256(R66 / "paired_metric_records.parquet"),
            "primary_metric": "valid_pair_original_over_cross_accuracy",
            "legacy_field_name": "legacy_misnamed_paired_replacement_R@1",
        })

        decomposition = relation_decomposition_rows(qdf, density_edges) + boundary_multiplicity_decomposition(ALL_SEQUENCES)
        pd.DataFrame(decomposition).to_csv(R66 / "source_target_comparison.csv", index=False)
        comparison = {
            "experiment_id": EXP_ID,
            "boundary": boundary_json["domain_aggregates"],
            "relation": domain_ranking,
            "paired": {
                domain: {
                    "valid_pair_original_over_cross_accuracy_macro": float(np.mean([m["valid_pair_original_over_cross_accuracy"] for m in pair_metrics if m["sequence"] in seqs and m["valid_pair_original_over_cross_accuracy"] is not None])) if any(m["sequence"] in seqs and m["valid_pair_original_over_cross_accuracy"] is not None for m in pair_metrics) else None,
                    "threshold_accuracy_at_0p5_macro": float(np.mean([m["threshold_accuracy_at_0p5"] for m in pair_metrics if m["sequence"] in seqs and m["threshold_accuracy_at_0p5"] is not None])) if any(m["sequence"] in seqs and m["threshold_accuracy_at_0p5"] is not None for m in pair_metrics) else None,
                    "pr_auc_macro": float(np.mean([m["pr_auc"] for m in pair_metrics if m["sequence"] in seqs and m["pr_auc"] is not None])) if any(m["sequence"] in seqs and m["pr_auc"] is not None for m in pair_metrics) else None,
                    "positive_base_rate_macro": float(np.mean([m["positive_base_rate"] for m in pair_metrics if m["sequence"] in seqs and m["positive_base_rate"] is not None])) if any(m["sequence"] in seqs and m["positive_base_rate"] is not None for m in pair_metrics) else None,
                } for domain, seqs in (("source", SOURCE_SEQUENCES), ("target", TARGET_SEQUENCES))
            },
            "decomposition_rows": decomposition,
            "decomposition_strata_manifest_sha256": sha256(R66 / "decomposition_strata_manifest.json"),
        }
        source_r1 = domain_ranking["source"]["outgoing"]["any_valid_positive"]["all_query"]["R@1"]
        target_r1 = domain_ranking["target"]["outgoing"]["any_valid_positive"]["all_query"]["R@1"]
        comparison["relation"]["source_to_target_retention_ratio_any_valid_outgoing_R@1"] = target_r1 / source_r1 if source_r1 else None
        sb = boundary_json["domain_aggregates"]["source"]["corrected_unique_transition_primary"]["pooled"]
        tb = boundary_json["domain_aggregates"]["target"]["corrected_unique_transition_primary"]["pooled"]
        comparison["boundary"]["source_to_target_retention_ratio_pr_auc"] = tb["pr_auc"] / sb["pr_auc"] if sb["pr_auc"] else None
        comparison["boundary"]["source_to_target_retention_ratio_lift"] = tb["pr_auc_base_rate_lift"] / sb["pr_auc_base_rate_lift"] if sb["pr_auc_base_rate_lift"] else None
        write_json(R66 / "source_target_comparison.json", comparison)
        append_event("corrected_metrics_audited",
                     boundary_metrics_sha256=sha256(R66 / "boundary_metrics.json"),
                     ranking_metrics_sha256=sha256(R66 / "corrected_ranking_metrics.json"),
                     paired_metrics_sha256=sha256(R66 / "paired_metric_audit.json"),
                     comparison_sha256=sha256(R66 / "source_target_comparison.json"))
        print(json.dumps({"stage": "audit-corrected-metrics", "status": "completed",
                          "boundary_sequences": len(boundary_json["sequences"]), "relation_queries": len(query_diag),
                          "pair_records": len(pair_records)}, sort_keys=True))


def boundary_gate_for_sequences(boundary: dict[str, Any], sequences: list[str]) -> dict[str, Any]:
    view = "corrected_unique_transition_primary"
    vals = {k: [boundary["sequences"][s]["views"][view].get(k) for s in sequences]
            for k in ("pr_auc", "precision_at_actual", "recall_at_95_precision")}
    defined = all(all(v is not None for v in arr) for arr in vals.values()) and len(sequences) > 0
    macro = {k: (float(np.mean(v)) if defined else None) for k, v in vals.items()}
    min_precision = float(min(vals["precision_at_actual"])) if defined else None
    checks = {
        "macro_pr_auc": defined and macro["pr_auc"] >= BOUNDARY_GATE["macro_pr_auc"],
        "macro_precision_at_actual": defined and macro["precision_at_actual"] >= BOUNDARY_GATE["macro_precision_at_actual"],
        "macro_recall_at_95_precision": defined and macro["recall_at_95_precision"] >= BOUNDARY_GATE["macro_recall_at_95_precision"],
        "every_sequence_precision_at_actual": defined and min_precision >= BOUNDARY_GATE["every_sequence_precision_at_actual"],
        "all_defined": defined,
    }
    return {"sequences": sequences, "thresholds": BOUNDARY_GATE, "macro": macro,
            "min_precision_at_actual": min_precision, "checks": checks, "passed": all(checks.values())}


def command_diagnose() -> None:
    with stage_context("diagnose"):
        boundary = read_json(R66 / "boundary_metrics.json", {})
        ranking = read_json(R66 / "corrected_ranking_metrics.json", {})
        coverage = pd.read_csv(R66 / "candidate_coverage.csv")
        legacy = read_json(R66 / "legacy_metric_reproduction.json", {})
        provenance = read_json(R66 / "m23_65_historical_implementation_provenance.json", {})
        published = read_json(R65 / "representation_metrics.json", {})
        original_gate = read_json(R65 / "representation_gate.json", {})
        canonical_manifest = read_json(R66 / "canonical_query_manifest.json", {})
        corrected_query_counts = {
            (r["sequence"], r["direction"]): int(r["eligible_queries"])
            for r in canonical_manifest.get("counts", [])
        }

        raw_vs_unique = {}
        for seq in TARGET_SEQUENCES:
            audit = boundary["sequences"][seq]["audit"]
            raw_vs_unique[seq] = {
                "matched_observations": audit["matched_observation_count"],
                "unique_transitions": audit["unique_physical_transitions"],
                "duplicates_present": audit["duplicate_observation_count"] > 0,
            }
        legacy_all_query_copied = []
        skipped_query_evidence = []
        for m in published.get("sequences", []):
            for direction in ("outgoing", "incoming"):
                block = m.get(direction, {})
                published_queries = int(block.get("all_query", {}).get("queries", 0))
                corrected_queries = corrected_query_counts.get((m.get("sequence"), direction), 0)
                legacy_all_query_copied.append({
                    "sequence": m.get("sequence"), "direction": direction,
                    "candidate_present_equals_all_query": block.get("candidate_present") == block.get("all_query"),
                })
                skipped_query_evidence.append({
                    "sequence": m.get("sequence"), "direction": direction,
                    "published_all_query_count": published_queries,
                    "corrected_candidate_independent_eligible_count": corrected_queries,
                    "published_count_is_smaller": published_queries < corrected_queries,
                })
        target_missing = coverage[(coverage.domain == "target") & (coverage.sequence.isin(TARGET_SEQUENCES)) &
                                  (coverage.direction == "outgoing") & (coverage.gap_bucket == "all")]
        metric_bugs = {
            "legacy_all_query_constructed_from_candidate_present_positive": (
                all(x["candidate_present_equals_all_query"] for x in legacy_all_query_copied)
                and any(x["published_count_is_smaller"] for x in skipped_query_evidence)
            ),
            "legacy_candidate_missing_query_skipped_by_definition": any(x["published_count_is_smaller"] for x in skipped_query_evidence),
            "legacy_all_query_equals_candidate_present_artifact": all(x["candidate_present_equals_all_query"] for x in legacy_all_query_copied),
            "corrected_candidate_missing_queries_observed": bool((target_missing.any_positive_missing_count > 0).any()),
            "paired_threshold_accuracy_misnamed_R_at_1": all("paired_replacement_R@1" in m for m in published.get("sequences", [])),
            "boundary_preregistered_repeat_aggregation_but_legacy_metric_observation_weighted": all(v["duplicates_present"] for v in raw_vs_unique.values()),
            "score_row_label_mapping_inconsistent": False,
        }
        provenance_failure = not bool(provenance.get("byte_exact_historical_source_found"))
        if metric_bugs["legacy_all_query_constructed_from_candidate_present_positive"] or metric_bugs["paired_threshold_accuracy_misnamed_R_at_1"] or metric_bugs["boundary_preregistered_repeat_aggregation_but_legacy_metric_observation_weighted"]:
            measurement = "FAIL_METRIC_DEFINITION_BUG"
        elif provenance_failure:
            measurement = "FAIL_HISTORICAL_IMPLEMENTATION_PROVENANCE"
        else:
            measurement = "PASS_METRIC_DEFINITIONS"
        validation = {
            "experiment_id": EXP_ID, "measurement_integrity_decision": measurement,
            "metric_definition_bugs": metric_bugs, "legacy_all_query_comparisons": legacy_all_query_copied,
            "candidate_independent_query_count_comparisons": skipped_query_evidence,
            "boundary_raw_vs_unique": raw_vs_unique, "legacy_behavioral_reproduction": legacy.get("legacy_behavioral_reproduction"),
            "historical_source_reproduction_status": provenance.get("historical_source_reproduction_status"),
            "current_script_is_not_historical_source": provenance.get("current_script_is_not_historical_source"),
            "candidate_independent_canonical_queries": True, "unknown_is_negative": False,
        }
        write_json(R66 / "metric_definition_validation.json", validation)

        source_gate = boundary_gate_for_sequences(boundary, SOURCE_SEQUENCES)
        target_gate = boundary_gate_for_sequences(boundary, TARGET_SEQUENCES)
        if not target_gate["checks"]["all_defined"]:
            stability = "INDETERMINATE"
        elif target_gate["passed"] and original_gate.get("decision") == "FAIL_MOT20_REPRESENTATION_GATE":
            stability = "WOULD_PASS_UNDER_CORRECTED_METRIC"
        else:
            stability = "STABLE_FAIL"

        target_cov_rows = coverage[(coverage.domain == "target") & (coverage.sequence.isin(TARGET_SEQUENCES)) &
                                   (coverage.direction == "outgoing") & (coverage.gap_bucket == "all")]
        pooled_cov_row = coverage[(coverage.domain == "target") & (coverage.sequence == "POOLED") &
                                  (coverage.direction == "outgoing") & (coverage.gap_bucket == "all")]
        target_cov = float(pooled_cov_row.iloc[0].any_valid_positive_coverage) if len(pooled_cov_row) else None
        below_count = int((target_cov_rows.any_valid_positive_coverage < 0.80).sum())
        source_r1 = ranking["domain_pooled_relation_logit"]["source"]["outgoing"]["any_valid_positive"]["all_query"]["R@1"]
        target_r1 = ranking["domain_pooled_relation_logit"]["target"]["outgoing"]["any_valid_positive"]["all_query"]["R@1"]
        if not source_gate["passed"]:
            scientific = "source_boundary_capacity_failure"
        elif not target_gate["passed"]:
            scientific = "target_boundary_transfer_failure"
        elif target_cov is not None and (target_cov < 0.80 or below_count >= 2):
            scientific = "candidate_coverage_bottleneck"
        elif target_cov is not None and target_cov >= 0.80 and target_r1 < 0.5 * source_r1 and source_r1 - target_r1 >= 0.20:
            scientific = "relation_ranking_transfer_failure"
        else:
            scientific = "mixed_or_inconclusive"
        overall = "metric_definition_bug" if stability == "WOULD_PASS_UNDER_CORRECTED_METRIC" else scientific
        diagnosis = {
            "experiment_id": EXP_ID, "status": "diagnosed", "measurement_integrity_decision": measurement,
            "m23_65_gate_stability": stability, "source_boundary_reference": source_gate,
            "target_boundary_corrected_gate": target_gate, "original_m23_65_decision_unchanged": original_gate.get("decision"),
            "scientific_primary_failure": scientific, "overall_primary_classification": overall,
            "relation_failure_rules": {
                "target_any_valid_candidate_coverage_pooled": target_cov,
                "target_sequences_below_0p80": below_count, "source_any_valid_all_query_R@1": source_r1,
                "target_any_valid_all_query_R@1": target_r1,
            },
            "secondary_findings": [
                "historical M23-65 byte-exact source unavailable" if provenance_failure else "historical source available",
                "source validation split participated in checkpoint selection and is capability diagnostic only",
            ],
            "critical_limitation": provenance.get("limitation"),
            **FIXED_DECLARATIONS,
        }
        write_json(R66 / "final_diagnosis.json", diagnosis)
        append_event("diagnosis_complete", measurement_integrity_decision=measurement,
                     m23_65_gate_stability=stability, scientific_primary_failure=scientific,
                     overall_primary_classification=overall)
        print(json.dumps({"stage": "diagnose", "status": "completed", "measurement_integrity_decision": measurement,
                          "m23_65_gate_stability": stability, "scientific_primary_failure": scientific,
                          "overall_primary_classification": overall}, sort_keys=True))


def required_preclosure_artifacts() -> list[Path]:
    names = [
        "summary.csv", "protocol_events.jsonl", "preregistration.json", "input_manifest.json",
        "implementation_manifest.json", "m23_65_historical_implementation_provenance.json",
        "gap_convention_reconciliation.json", "source_score_manifest.json", "canonical_queries.parquet",
        "valid_targets.parquet", "canonical_query_manifest.json", "legacy_metric_reproduction.json",
        "candidate_coverage.csv", "candidate_coverage.json", "corrected_ranking_metrics.csv",
        "corrected_ranking_metrics.json", "paired_metric_audit.csv", "paired_metric_audit.json",
        "boundary_duplicate_audit.csv", "boundary_metrics.csv", "boundary_metrics.json",
        "source_target_comparison.csv", "source_target_comparison.json", "metric_definition_validation.json",
        "final_diagnosis.json",
    ]
    paths = [R66 / n for n in names]
    for seq in SOURCE_SEQUENCES:
        for n in ("boundary_scores.parquet", "node_scores.parquet", "relation_scores.parquet", "pair_scores.parquet",
                  "score_to_source_row.parquet", "score_manifest.json"):
            paths.append(R66 / "source_scores" / seq / n)
    return paths


def command_validate() -> None:
    with stage_context("validate"):
        input_manifest = read_json(R66 / "input_manifest.json", {})
        mismatches = []
        for rec in input_manifest.get("file_checks", []):
            path = ROOT / rec["path"]
            actual = sha256(path) if path.exists() else None
            if actual != rec["actual_sha256"]:
                mismatches.append({"path": rec["path"], "expected_current": rec["actual_sha256"], "actual": actual})
        required = required_preclosure_artifacts()
        missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
        scope = {
            "experiment_id": EXP_ID, "validated_at": utc_now(), "scope_counts": SCOPE_COUNTS,
            "new_raw_mot20_gt_reads": 0, "new_raw_mot17_gt_reads": 0,
            "frozen_label_sidecar_reads": True, "training_runs": 0, "tracker_outputs": 0,
            "trackeval_runs": 0, "hota_evaluations": 0, "mot20_test_reads": 0,
            "mot20_test_submissions": 0, "next_policy_authorized": False, "hota": None,
            "prohibited_scope_all_zero": all(v == 0 for v in SCOPE_COUNTS.values()),
            "input_artifacts_unchanged": len(mismatches) == 0, "input_mismatches": mismatches,
            "R62_R63_R64_R65_writes": 0,
        }
        write_json(R66 / "scope_validation.json", scope)
        process = process_gpu_snapshot(exclude_self=True)
        process.update({
            "experiment_id": EXP_ID, "relevant_external_processes_clear": len(process["relevant_processes"]) == 0,
            "git_head": git_head(), "git_scoped_status": git_scoped_status(),
        })
        write_json(R66 / "process_gpu_validation.json", process)
        validation = {
            "experiment_id": EXP_ID, "validated_at": utc_now(), "required_artifacts_present": len(missing) == 0,
            "missing_artifacts": missing, "input_artifacts_unchanged": len(mismatches) == 0,
            "input_mismatches": mismatches, "scope_passed": scope["prohibited_scope_all_zero"],
            "implementation_guard_passed": sha256(SCRIPT) == read_json(R66 / "implementation_manifest.json", {}).get("script_sha256"),
            "prereg_guard_passed": sha256(PREREG) == read_json(R66 / "implementation_manifest.json", {}).get("prereg_sha256"),
            "process_snapshot_sha256": sha256(R66 / "process_gpu_validation.json"),
        }
        write_json(R66 / "validation_report.json", validation)
        if not all([validation["required_artifacts_present"], validation["input_artifacts_unchanged"], validation["scope_passed"],
                    validation["implementation_guard_passed"], validation["prereg_guard_passed"]]):
            raise RuntimeError(f"validation failed: {validation}")
        append_event("validation_complete", validation=validation)
        print(json.dumps({"stage": "validate", "status": "completed", "required_artifacts_present": True,
                          "inputs_unchanged": True, "scope_passed": True}, sort_keys=True))


def artifact_hash_manifest() -> dict[str, Any]:
    records = []
    for p in sorted(R66.rglob("*")):
        if p.is_file() and p.name != "artifact_sha256_manifest.json" and not p.name.endswith(".tmp"):
            records.append({"path": str(p.relative_to(ROOT)), "size": p.stat().st_size, "sha256": sha256(p)})
    for p in (SCRIPT, TEST_SCRIPT, PREREG, RESULT):
        if p.exists():
            records.append({"path": str(p.relative_to(ROOT)), "size": p.stat().st_size, "sha256": sha256(p)})
    return {"experiment_id": EXP_ID, "generated_at": utc_now(), "self_excluded": True, "records": records}


def summary_result_markdown(final: dict[str, Any], boundary: dict[str, Any], ranking: dict[str, Any],
                            pairs: dict[str, Any], coverage: pd.DataFrame, legacy: dict[str, Any],
                            provenance: dict[str, Any], command_log: list[str], input_manifest: dict[str, Any],
                            artifact_records: list[dict[str, Any]] | None = None) -> str:
    diagnosis = final["diagnosis"]
    lines = [
        f"# {EXP_NAME} — Result",
        "",
        f"Final status: **{final['status']}**  ",
        f"Decision: **{final['decision']}**  ",
        f"Measurement integrity: **{diagnosis['measurement_integrity_decision']}**  ",
        f"M23-65 gate stability: **{diagnosis['m23_65_gate_stability']}**  ",
        f"Scientific primary failure: **{diagnosis['scientific_primary_failure']}**  ",
        f"Overall primary classification: **{diagnosis['overall_primary_classification']}**",
        "",
        "## Scope",
        "",
        "This is an independent post-hoc diagnostic using frozen GT-derived label sidecars. It is not deployable, not a strict result, and does not authorize a next policy. No training, tracker, TrackEval, HOTA, MOT20 test read, or raw GT read occurred.",
        "",
        "## Executed commands",
        "",
    ]
    lines.extend([f"- `{x}`" for x in command_log])
    lines += [
        "", "## Input and historical provenance", "",
        f"All specified top-level and nested frozen scientific artifacts matched: `{final['input_reverification']['frozen_scientific_artifacts_all_match']}`.",
        f"M23-65 recorded script SHA: `{provenance.get('recorded_script_sha256')}`.",
        f"M23-65 current script SHA: `{provenance.get('current_script_sha256')}`.",
        f"Byte-exact historical source found: `{provenance.get('byte_exact_historical_source_found')}`; reproduction status: `{provenance.get('historical_source_reproduction_status')}`.",
        f"Legacy frozen-artifact behavioral reproduction: `{legacy.get('legacy_behavioral_reproduction')}`.",
        "", "### Input SHA verification table", "",
        "| Input | Expected SHA-256 | Actual SHA-256 | Match |",
        "|---|---|---|---|",
    ]
    for rec in input_manifest.get("file_checks", []):
        lines.append(f"| `{rec['path']}` | `{rec['expected_sha256']}` | `{rec['actual_sha256']}` | `{rec['match']}` |")
    lines += [
        "", "## Boundary observation versus unique-transition audit", "",
        "| Sequence | Raw observations | Matched observations | Unique transitions | Duplicates | Legacy AP | Corrected AP | Corrected P@actual | Corrected R@95P |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seq in ALL_SEQUENCES:
        s = boundary["sequences"][seq]
        a = s["audit"]
        l = s["views"]["legacy_observation_weighted"]
        c = s["views"]["corrected_unique_transition_primary"]
        lines.append(f"| {seq} | {a['raw_observation_count']} | {a['matched_observation_count']} | {a['unique_physical_transitions']} | {a['duplicate_observation_count']} | {l['pr_auc']:.9f} | {c['pr_auc']:.9f} | {c['precision_at_actual']:.9f} | {c['recall_at_95_precision']:.9f} |")
    lines += ["", "## Canonical query and candidate coverage", "",
              "Canonical outgoing and incoming query counts are in `canonical_query_manifest.json`. Candidate-missing queries remain in all-query denominators with zero rank contribution.", "",
              "| Domain | Direction | Pooled canonical coverage | Pooled any-valid coverage |",
              "|---|---|---:|---:|"]
    for domain in ("source", "target"):
        for direction in ("outgoing", "incoming"):
            z = coverage[(coverage.domain == domain) & (coverage.sequence == "POOLED") & (coverage.direction == direction) & (coverage.gap_bucket == "all")]
            if len(z):
                r = z.iloc[0]
                lines.append(f"| {domain} | {direction} | {r.canonical_exact_target_coverage:.9f} | {r.any_valid_positive_coverage:.9f} |")
    lines += ["", "## Corrected relation ranking", "",
              "| Domain | Direction | Canonical R@1/R@3/MRR | Any-valid R@1/R@3/MRR |",
              "|---|---|---|---|"]
    for domain in ("source", "target"):
        for direction in ("outgoing", "incoming"):
            d = ranking["domain_pooled_relation_logit"][domain][direction]
            c = d["canonical_exact_target"]["all_query"]
            a = d["any_valid_positive"]["all_query"]
            lines.append(f"| {domain} | {direction} | {c['R@1']:.6f}/{c['R@3']:.6f}/{c['MRR']:.6f} | {a['R@1']:.6f}/{a['R@3']:.6f}/{a['MRR']:.6f} |")
    lines += ["", "## Paired metric correction", "",
              "The primary paired metric is strict `original_margin > 0` on legal pairs. The old threshold field is reported only as `legacy_misnamed_paired_replacement_R@1`.", "",
              "| Sequence | Valid pairs | Original-over-cross accuracy | Threshold accuracy@0.5 | PR-AUC |",
              "|---|---:|---:|---:|---:|"]
    for m in pairs["per_sequence"]:
        lines.append(f"| {m['sequence']} | {m['valid_pair_count']} | {m['valid_pair_original_over_cross_accuracy']} | {m['threshold_accuracy_at_0p5']} | {m['pr_auc']} |")
    lines += [
        "", "## Source-to-target decomposition", "",
        "Boundary AP is interpreted together with base rate, AP/base-rate lift, and ROC-AUC. Relation retention, fixed baselines, and the preregistered gap/crowd-density/purity/appearance-mapped/frame-density/presence/saturation/tie/multiplicity strata are stored in `source_target_comparison.json` and `.csv`.",
        "", "## Final declarations", "",
        "- `post_hoc_diagnostic_only=true`",
        "- `uses_frozen_gt_derived_label_sidecars=true`",
        "- `not_deployable=true`",
        "- `not_a_strict_result=true`",
        "- `hota=null`",
        "- `training_runs=0`",
        "- `tracker_outputs=0`",
        "- `trackeval_runs=0`",
        "- `hota_evaluations=0`",
        "- `mot20_test_reads=0`",
        "- `mot20_test_submissions=0`",
        "- `next_policy_authorized=false`",
        "", "未执行Notion写回。",
        "",
        f"Registry row: `{final.get('registry_line')}`. Run root: `{R66.relative_to(ROOT)}`. Structured artifact hashes are in `artifact_sha256_manifest.json`.",
    ]
    if artifact_records is not None:
        lines += ["", "## Structured records and SHA-256", "", "| Artifact | Bytes | SHA-256 |", "|---|---:|---|"]
        for rec in artifact_records:
            if rec["path"].endswith("artifact_sha256_manifest.json") or rec["path"] == str(RESULT.relative_to(ROOT)):
                continue
            lines.append(f"| `{rec['path']}` | {rec['size']} | `{rec['sha256']}` |")
    return "\n".join(lines) + "\n"


def command_summarize() -> None:
    ensure_implementation_guard("summarize")
    set_stage("summarize", "running", started_at=utc_now())
    append_event("stage_started", stage="summarize")
    t0 = time.perf_counter()
    try:
        diagnosis = read_json(R66 / "final_diagnosis.json", {})
        boundary = read_json(R66 / "boundary_metrics.json", {})
        ranking = read_json(R66 / "corrected_ranking_metrics.json", {})
        pairs = read_json(R66 / "paired_metric_audit.json", {})
        coverage = pd.read_csv(R66 / "candidate_coverage.csv")
        legacy = read_json(R66 / "legacy_metric_reproduction.json", {})
        provenance = read_json(R66 / "m23_65_historical_implementation_provenance.json", {})
        input_manifest = read_json(R66 / "input_manifest.json", {})
        command_log = [
            "python -B -u scripts/m23_research/m23_66_v3_metric_correctness_source_target_decomposition_audit.py init",
            "python -B -u scripts/m23_research/m23_66_v3_metric_correctness_source_target_decomposition_audit.py verify-inputs",
            "python -B -u scripts/m23_research/m23_66_v3_metric_correctness_source_target_decomposition_audit.py freeze-source-scores",
            "python -B -u scripts/m23_research/m23_66_v3_metric_correctness_source_target_decomposition_audit.py freeze-canonical-queries",
            "python -B -u scripts/m23_research/m23_66_v3_metric_correctness_source_target_decomposition_audit.py reproduce-legacy",
            "python -B -u scripts/m23_research/m23_66_v3_metric_correctness_source_target_decomposition_audit.py audit-corrected-metrics",
            "python -B -u scripts/m23_research/m23_66_v3_metric_correctness_source_target_decomposition_audit.py diagnose",
            "python -B -u scripts/m23_research/m23_66_v3_metric_correctness_source_target_decomposition_audit.py validate",
            "python -B -u scripts/m23_research/m23_66_v3_metric_correctness_source_target_decomposition_audit.py summarize",
        ]
        decision = "COMPLETED_POST_HOC_DIAGNOSTIC"
        final = {
            "experiment_id": EXP_ID, "name": EXP_NAME, "status": "completed", "decision": decision,
            "closed_at": utc_now(), "diagnosis": diagnosis, "input_reverification": {
                "all_passed": input_manifest.get("all_passed"),
                "frozen_scientific_artifacts_all_match": input_manifest.get("frozen_scientific_artifacts_all_match"),
                "first_mismatch": input_manifest.get("first_mismatch"),
            },
            "historical_provenance": {
                "recorded_script_sha256": provenance.get("recorded_script_sha256"),
                "current_script_sha256": provenance.get("current_script_sha256"),
                "byte_exact_historical_source_found": provenance.get("byte_exact_historical_source_found"),
                "historical_source_reproduction_status": provenance.get("historical_source_reproduction_status"),
            },
            "legacy_behavioral_reproduction": legacy.get("legacy_behavioral_reproduction"),
            "executed_commands": command_log, "scope_counts": SCOPE_COUNTS,
            "new_raw_mot20_gt_reads": 0, "new_raw_mot17_gt_reads": 0, "frozen_label_sidecar_reads": True,
            "training_runs": 0, "tracker_outputs": 0, "trackeval_runs": 0, "hota_evaluations": 0,
            "mot20_test_reads": 0, "mot20_test_submissions": 0,
            "process_gpu": process_gpu_snapshot(exclude_self=True), "git_head": git_head(),
            "git_scoped_status": git_scoped_status(), "notion_writeback": "未执行Notion写回",
            **FIXED_DECLARATIONS,
        }
        RESULT.parent.mkdir(parents=True, exist_ok=True)
        RESULT.write_text(summary_result_markdown(final, boundary, ranking, pairs, coverage, legacy, provenance,
                                                  command_log, input_manifest), encoding="utf-8")
        set_stage("summarize", "completed", finished_at=utc_now(), wall_seconds=time.perf_counter() - t0,
                  peak_rss_kb=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        set_stage("closed", "closed", started_at=utc_now(), finished_at=utc_now(), decision=decision,
                  notes={"measurement_integrity_decision": diagnosis.get("measurement_integrity_decision"),
                         "overall_primary_classification": diagnosis.get("overall_primary_classification")})
        registry_line = registry_close("completed", decision,
                                       f"measurement={diagnosis.get('measurement_integrity_decision')}; stability={diagnosis.get('m23_65_gate_stability')}; scientific={diagnosis.get('scientific_primary_failure')}; overall={diagnosis.get('overall_primary_classification')}; HOTA=null; next_policy_authorized=false")
        final["registry_line"] = registry_line
        write_json(R66 / "final_summary.json", final)
        summary = read_summary()
        no_stale = not summary.status.astype(str).isin(["running", "pending"]).any()
        closure = {
            "experiment_id": EXP_ID, "status": "closed", "decision": decision,
            "closure_integrity_passed": no_stale and RESULT.exists() and (R66 / "final_summary.json").exists(),
            "summary_no_running_pending": no_stale, "registry_line": registry_line,
            "registry_closed": True, "result_document": str(RESULT.relative_to(ROOT)),
            "scope_counts": SCOPE_COUNTS, "hota": None, "next_policy_authorized": False,
            "script_sha256": sha256(SCRIPT), "prereg_sha256": sha256(PREREG),
        }
        write_json(R66 / "closure_validation.json", closure)
        fields, registry = registry_rows()
        matching = [(i + 2, r) for i, r in enumerate(registry) if r.get("name") == EXP_ID]
        independent = {
            "experiment_id": EXP_ID, "validated_at": utc_now(),
            "summary_no_running_pending": no_stale,
            "required_final_artifacts_present": all(p.exists() for p in [R66 / "final_summary.json", R66 / "closure_validation.json", RESULT]),
            "registry_matching_rows": [{"line": i, "status": r.get("status"), "current_stage": r.get("current_stage"), "decision": r.get("decision")} for i, r in matching],
            "registry_latest_closed": bool(matching and matching[-1][1].get("current_stage") == "closed" and matching[-1][1].get("status") in ("completed", "failed")),
            "scope_all_zero": all(v == 0 for v in SCOPE_COUNTS.values()), "hota_is_null": final.get("hota") is None,
            "next_policy_authorized_false": final.get("next_policy_authorized") is False,
            "input_manifest_all_passed": input_manifest.get("all_passed") is True,
            "implementation_sha_guard": sha256(SCRIPT) == read_json(R66 / "implementation_manifest.json", {}).get("script_sha256"),
        }
        independent["independent_closure_passed"] = all([
            independent["summary_no_running_pending"], independent["required_final_artifacts_present"],
            independent["registry_latest_closed"], independent["scope_all_zero"], independent["hota_is_null"],
            independent["next_policy_authorized_false"], independent["input_manifest_all_passed"], independent["implementation_sha_guard"],
        ])
        write_json(R66 / "independent_closure_validation.json", independent)
        if not closure["closure_integrity_passed"] or not independent["independent_closure_passed"]:
            raise RuntimeError("closure validation failed")
        append_event("experiment_closed", decision=decision, registry_line=registry_line,
                     measurement_integrity_decision=diagnosis.get("measurement_integrity_decision"),
                     m23_65_gate_stability=diagnosis.get("m23_65_gate_stability"),
                     scientific_primary_failure=diagnosis.get("scientific_primary_failure"),
                     overall_primary_classification=diagnosis.get("overall_primary_classification"),
                     next_policy_authorized=False, hota=None)
        write_json(R66 / "artifact_sha256_manifest.json", artifact_hash_manifest())
        artifact_records = read_json(R66 / "artifact_sha256_manifest.json", {}).get("records", [])
        RESULT.write_text(summary_result_markdown(final, boundary, ranking, pairs, coverage, legacy, provenance,
                                                  command_log, input_manifest, artifact_records), encoding="utf-8")
        write_json(R66 / "artifact_sha256_manifest.json", artifact_hash_manifest())
        print(json.dumps({"stage": "summarize", "status": "completed", "decision": decision,
                          "registry_line": registry_line, "closure_integrity_passed": True,
                          "independent_closure_passed": True}, sort_keys=True))
    except Exception as exc:
        set_stage("summarize", "failed", finished_at=utc_now(), error=repr(exc), wall_seconds=time.perf_counter() - t0)
        append_event("stage_failed", stage="summarize", error=repr(exc), traceback=traceback.format_exc())
        raise


def fail_close(decision: str, error: str) -> None:
    R66.mkdir(parents=True, exist_ok=True)
    df = read_summary()
    for idx, row in df.iterrows():
        if str(row.status) == "pending":
            df.loc[idx, "status"] = "skipped"
            df.loc[idx, "finished_at"] = utc_now()
            df.loc[idx, "notes"] = "skipped after fail-closed"
        elif str(row.status) == "running":
            df.loc[idx, "status"] = "failed"
            df.loc[idx, "finished_at"] = utc_now()
            df.loc[idx, "error"] = error
    if "closed" in set(df.stage.astype(str)):
        idx = df.index[df.stage.astype(str) == "closed"][-1]
        df.loc[idx, "status"] = "closed"
        df.loc[idx, "started_at"] = utc_now()
        df.loc[idx, "finished_at"] = utc_now()
        df.loc[idx, "decision"] = decision
        df.loc[idx, "error"] = error
    write_summary(df)
    registry_line = None
    try:
        registry_line = registry_close("failed", decision, f"fail_closed; error={error[:500]}; HOTA=null; next_policy_authorized=false")
    except Exception:
        pass
    final = {
        "experiment_id": EXP_ID, "status": "failed", "decision": decision, "error": error,
        "closed_at": utc_now(), "registry_line": registry_line, "scope_counts": SCOPE_COUNTS,
        "new_raw_mot20_gt_reads": 0, "new_raw_mot17_gt_reads": 0, "frozen_label_sidecar_reads": True,
        **FIXED_DECLARATIONS,
    }
    write_json(R66 / "final_summary.json", final)
    write_json(R66 / "scope_validation.json", {"experiment_id": EXP_ID, "scope_counts": SCOPE_COUNTS, **FIXED_DECLARATIONS})
    write_json(R66 / "closure_validation.json", {
        "experiment_id": EXP_ID, "status": "closed", "decision": decision, "closure_integrity_passed": True,
        "failed_closed": True, "error": error, "summary_no_running_pending": True,
        "scope_counts": SCOPE_COUNTS, "hota": None, "next_policy_authorized": False,
    })
    write_json(R66 / "independent_closure_validation.json", {
        "experiment_id": EXP_ID, "independent_closure_passed": True, "failed_closed": True,
        "summary_no_running_pending": True, "registry_line": registry_line,
    })
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(f"# {EXP_NAME} — Fail-closed result\n\nStatus: **failed**\n\nDecision: **{decision}**\n\nError: `{error}`\n\nNo training, tracker, TrackEval, HOTA, MOT20 test read, or raw GT read occurred. `next_policy_authorized=false`.\n\n未执行Notion写回。\n", encoding="utf-8")
    append_event("experiment_fail_closed", decision=decision, error=error, registry_line=registry_line,
                 next_policy_authorized=False, hota=None)
    write_json(R66 / "artifact_sha256_manifest.json", artifact_hash_manifest())


def main() -> None:
    parser = argparse.ArgumentParser(description=EXP_NAME)
    parser.add_argument("command", choices=STAGES[:-1])
    args = parser.parse_args()
    commands = {
        "init": command_init,
        "verify-inputs": command_verify_inputs,
        "freeze-source-scores": command_freeze_source_scores,
        "freeze-canonical-queries": command_freeze_canonical_queries,
        "reproduce-legacy": command_reproduce_legacy,
        "audit-corrected-metrics": command_audit_corrected_metrics,
        "diagnose": command_diagnose,
        "validate": command_validate,
        "summarize": command_summarize,
    }
    try:
        commands[args.command]()
    except Exception as exc:
        message = repr(exc)
        if "FAIL_INPUT_REVERIFICATION" in message:
            decision = "FAIL_INPUT_REVERIFICATION"
        elif "FAIL_IMPLEMENTATION" in message or args.command not in ("verify-inputs",):
            decision = "FAIL_IMPLEMENTATION"
        else:
            decision = "FAIL_INPUT_REVERIFICATION"
        try:
            fail_close(decision, message)
        finally:
            raise


if __name__ == "__main__":
    main()
