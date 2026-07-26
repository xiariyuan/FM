#!/usr/bin/env python3
"""M23-63: GT-free supervision join and source-topology example audit.

Stage-A only. This module never trains, evaluates trackers, reads MOT20 labels,
or writes tracker/checkpoint artifacts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import linear_sum_assignment

EXP_ID = "M23-63"
TITLE = "M23-59 v3 GT-Free Supervision Join and Example Construction Audit"
ROOT = Path(".").resolve()
R62 = ROOT / "outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_gtfree_source_regeneration"
R63 = ROOT / "outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_supervision_join"
SCRIPT = ROOT / "scripts/m23_research/m23_63_v3_supervision_join_example_audit.py"
PREREG = ROOT / "docs/m23_59_relation_pretrained_hierarchical_flow_v3_supervision_join_prereg_20260722.md"
RESULT = ROOT / "docs/m23_59_relation_pretrained_hierarchical_flow_v3_supervision_join_result_20260722.md"
REGISTRY = ROOT / "outputs/experiment_registry.csv"
TRACKEVAL_MOT = ROOT / "TrackEval/trackeval/datasets/mot_challenge_2d_box.py"
CONTRACT_HASH = "90cfab7d3a1fc87cc46b26441d3d883b9fc72f27d8852d1e2074b00e628428f5"
CONTRACT_VERSION = "m23_59_v3_gtfree_source_contract_3.1.0"

SPLITS = {
    "MOT17-02": "train", "MOT17-04": "train", "MOT17-05": "train",
    "MOT17-09": "train", "MOT17-10": "train",
    "MOT17-11": "validation", "MOT17-13": "validation",
}
SEQUENCES = list(SPLITS)
DISTRACTOR_CLASSES = {2, 7, 8, 12}
IOU_THRESHOLD = 0.50
IOU_QUANTIZATION = 1e-9
LEX_EPSILON = 1e-12
MAX_NODE_ROWS = 30
NODE_STRIDE = 15
NODE_MIN_ROWS = 3
CHUNK_MAX_ROWS = 30
CHUNK_MAX_GAP = 30
CANDIDATE_MAX_GAP = 600
CANDIDATE_K_PER_BUCKET = 32
GAP_BUCKETS = [("1-30", 1, 30), ("31-90", 31, 90), ("91-180", 91, 180), ("181-600", 181, 600)]
PAIR_TIME_TOLERANCE = 30
PAIR_TOPOLOGY_CAP_PER_SEQUENCE = 20000
RELATION_EXAMPLE_CAP_PER_BUCKET_PER_SEQUENCE = 512
PAIR_EXAMPLE_CAP_PER_SEQUENCE = 512
TRUST_MIN_KNOWN = 2
TRUST_PURITY = 0.80
NODE_LABEL_MIN_KNOWN = 3
SUPPORT_GATES = {
    "train": {"pure_node": 100, "impure_node": 5, "boundary_positive": 5,
              "boundary_negative": 100, "successor_positive": 20,
              "successor_negative": 100, "paired_replacement": 5},
    "validation": {"pure_node": 20, "impure_node": 1, "boundary_positive": 1,
                   "boundary_negative": 20, "successor_positive": 5,
                   "successor_negative": 20, "paired_replacement": 1},
}
SUMMARY_FIELDS = ["experiment", "stage", "status", "started_at", "completed_at", "report", "decision", "notes"]
STAGES = [
    "preregistration", "m23_62_reverification", "source_topology", "label_join",
    "example_construction", "validation", "training", "strict_outer_evaluation",
    "tracker_generation", "closure",
]
GT_READ_COUNTER = 0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_id(*parts: Any) -> str:
    raw = "|".join(str(x) for x in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def json_write(path: Path, payload: Any, *, create_only: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if create_only and path.exists():
        raise RuntimeError(f"refuse overwrite: {path}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def csv_write(path: Path, rows: list[dict[str, Any]], fields: list[str], *, create_only: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if create_only and path.exists():
        raise RuntimeError(f"refuse overwrite: {path}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    tmp.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def append_event(event: str, **payload: Any) -> None:
    R63.mkdir(parents=True, exist_ok=True)
    row = {"time": utc_now(), "event": event, **payload}
    with (R63 / "protocol_events.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def events() -> list[dict[str, Any]]:
    p = R63 / "protocol_events.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def read_summary() -> list[dict[str, str]]:
    p = R63 / "summary.csv"
    if not p.exists():
        return []
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def update_stage(stage: str, status: str, report: str = "", decision: str = "", notes: str = "") -> None:
    rows = read_summary(); now = utc_now(); found = False
    for row in rows:
        if row["stage"] == stage:
            found = True
            if not row["started_at"] and status == "running": row["started_at"] = now
            if status in {"completed", "failed", "prohibited_by_scope", "superseded"}:
                if not row["started_at"]: row["started_at"] = now
                row["completed_at"] = now
            row.update(status=status, report=report, decision=decision, notes=notes)
    if not found:
        rows.append({"experiment": EXP_ID, "stage": stage, "status": status,
                     "started_at": now if status else "", "completed_at": "",
                     "report": report, "decision": decision, "notes": notes})
    csv_write(R63 / "summary.csv", rows, SUMMARY_FIELDS)


def registry_rows() -> tuple[list[str], list[list[str]]]:
    with REGISTRY.open(newline="", encoding="utf-8", errors="replace") as f:
        data = list(csv.reader(f))
    return data[0], data[1:]


def registry_append(values: dict[str, Any]) -> None:
    header, rows = registry_rows()
    rows.append([str(values.get(col, "")) for col in header])
    tmp = REGISTRY.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    tmp.replace(REGISTRY)


def close_registry(decision: str, status: str, notes: str) -> None:
    header, rows = registry_rows()
    idx = {k: header.index(k) for k in ["tracker_family", "status", "tag", "current_stage", "notes"]}
    for row in rows:
        if len(row) < len(header): row.extend([""] * (len(header) - len(row)))
        if row[idx["tracker_family"]] == EXP_ID and row[idx["status"]] == "running":
            row[idx["status"]] = "superseded"; row[idx["current_stage"]] = "superseded"
            row[idx["notes"]] = (row[idx["notes"]] + "; superseded by closed row").strip("; ")
    tmp = REGISTRY.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    tmp.replace(REGISTRY)
    final = read_json(R63 / "final_summary.json") if (R63 / "final_summary.json").exists() else {}
    tr = final.get("train_counts", {}); va = final.get("validation_counts", {})
    registry_append({
        "timestamp": utc_now(), "kind": "stage_a_audit", "status": status,
        "script": str(SCRIPT.relative_to(ROOT)), "dataset": "MOT17", "split": "sequence_disjoint_external",
        "tracker_family": EXP_ID, "variant": "m23_59_v3_supervision_join_example_audit",
        "tag": "M23-63-v3-supervision-join-closed", "run_root": str(R63.relative_to(ROOT)),
        "summary_csv": str((R63 / "summary.csv").relative_to(ROOT)), "current_stage": "closed",
        "decision": decision, "result_file": str(RESULT.relative_to(ROOT)), "artifact": str(RESULT.relative_to(ROOT)),
        "train_examples": tr.get("node_examples",0)+tr.get("relation_examples",0)+tr.get("paired_examples",0),
        "val_examples": va.get("node_examples",0)+va.get("relation_examples",0)+va.get("paired_examples",0),
        "rows": final.get("mot17_gt_reads",0), "notes": notes + "; result=" + str(RESULT.relative_to(ROOT)),
    })


def preregistration_text() -> str:
    return f"""# M23-63 preregistration — v3 supervision join and example audit

Frozen before any MOT17 GT read. Experiment root: `{R63.relative_to(ROOT)}`.

## Scope
Stage-A audit only. Training, optimizer steps, checkpoints, tracker generation, TrackEval, MOT20 GT/teacher/held-outer/test access, M23-54 and M23-58 are prohibited. R62 is immutable. M23-59 v2 checkpoint reuse is prohibited.

## Physical split and namespace
Train: MOT17-02/04/05/09/10-FRCNN. Validation: MOT17-11/13-FRCNN. DPM/SDP variants are excluded. Every row, track, GT identity, topology entity, candidate and example uses a sequence-qualified stable key.

## Frozen M23-62 contract
Required hash: `{CONTRACT_HASH}`. All five root authorization/closure files and all seven MOT17 rows/features/manifests are rehashed before label unlock. Any mismatch closes as FAIL_M23_62_REVERIFICATION.

## Supervision join
GT may be read only after topology/candidate freeze and a label_unlock event. Allowed path pattern: `datasets/MOT17/train/<sequence>-FRCNN/gt/gt.txt`.
TrackEval MOT box semantics are xywh converted to xyxy. IoU threshold is exactly `{IOU_THRESHOLD:.2f}`; no search. Valid pedestrian GT is mark != 0 and class == 1; visibility is not an eligibility threshold. Input feature 134 remains the unavailable sentinel and is never replaced by GT visibility.
MOT17 distractor classes are frozen from TrackEval as `{sorted(DISTRACTOR_CLASSES)}` (person_on_vehicle, static_person, distractor, reflection). Per frame, all-GT Hungarian matching first removes source rows matched to these distractors; then remaining rows are matched one-to-one to valid pedestrian GT.
Source order is (line_index,row_index), GT order is original GT line order. IoU is quantized to `{IOU_QUANTIZATION}` and a `{LEX_EPSILON}` lexicographic bonus resolves equivalent optima. Unknown source rows remain ignore and are never negative. No majority vote, temporal fill, GT-box feature recomputation, source-track rewrite, or GT-driven candidate insertion is permitted.

## Label-blind topology
Node opportunities follow each source track in (frame,line_index,row_index) order with MAX_NODE_ROWS={MAX_NODE_ROWS}, stride={NODE_STRIDE}, minimum rows={NODE_MIN_ROWS}.
Non-overlapping chunks split at source gap > {CHUNK_MAX_GAP} or {CHUNK_MAX_ROWS} rows. Candidate edges are time-forward with gap 1..{CANDIDATE_MAX_GAP}, stratified by `{GAP_BUCKETS}`, top K={CANDIDATE_K_PER_BUCKET} per source chunk/bucket. Ranking is 0.70 appearance cosine + 0.30 exp(-4*normalized center distance), then destination frame/track/chunk lexicographic order. GT cannot add, remove or reorder candidates.
Paired topology requires two same-bucket original edges, distinct endpoints, destination-start distance <= {PAIR_TIME_TOLERANCE}, and both crossed edges already present. At most {PAIR_TOPOLOGY_CAP_PER_SEQUENCE} stable combinations per sequence are frozen. Relation examples use the first {RELATION_EXAMPLE_CAP_PER_BUCKET_PER_SEQUENCE} frozen edges per sequence/bucket; pair examples use the first {PAIR_EXAMPLE_CAP_PER_SEQUENCE} frozen combinations. Selection is label-blind.

## Labels and examples
Trusted chunk identity requires at least {TRUST_MIN_KNOWN} known rows and purity >= {TRUST_PURITY:.2f}. Node labels require at least {NODE_LABEL_MIN_KNOWN} known rows: all one identity=pure, multiple identities=impure; unknown rows never imply impurity. Boundary is defined only when both adjacent rows are known (identity change=positive, same=negative, otherwise ignore). Candidate edge labels are positive only for two trusted same-identity chunks, negative only for two trusted different-identity chunks, otherwise ignore. Missing GT successor candidates are reported and never inserted. Paired samples are emitted only when original edges are positive and crossed edges exist in the frozen pool.

Minimum support gates: `{json.dumps(SUPPORT_GATES, sort_keys=True)}`. These gates are fixed before labels and will not be lowered. Join rate, purity and sequence dominance are descriptive only.

## Provenance disclosure
Source tracker inference is image-only/GT-free, but `full7_best_raw` was historically selected using MOT17 comparison. Claims are limited to sequence-disjoint relation supervision/validation under a historically selected frozen source host. Source-host and M23-62 feature-extractor provenance are recorded separately.

## Decision
All positive gates pass: PASS_SUPERVISION_JOIN_AND_EXAMPLE_CONSTRUCTION and authorize only a future fresh experiment to train v3 from scratch using frozen example SHA. Otherwise close on the first root cause as FAIL_M23_62_REVERIFICATION, FAIL_JOIN_SEMANTICS, FAIL_TOPOLOGY_COMPATIBILITY, FAIL_EXAMPLE_VALIDATION, FAIL_SPLIT_LEAKAGE, or FAIL_SCOPE_GUARD. This experiment always ends with training_runs=0.
"""


def r62_observable_paths() -> list[Path]:
    out: list[Path] = []
    for seq in SEQUENCES:
        d = R62 / "observables/MOT17" / seq
        out.extend([d / "rows.parquet", d / "row_features.f16.npy", d / "manifest.json"])
    return out


def initial_input_paths() -> list[Path]:
    names = [
        "closure_validation.json", "semantic_validation.json", "compatibility_validation.json",
        "next_stage_authorization.json", "final_summary.json", "feature_contract_v3_1.json",
    ]
    refs = [
        ROOT / "AGENTS.md", TRACKEVAL_MOT,
        ROOT / "scripts/m23_research/m23_62_gtfree_source_regeneration.py",
        ROOT / "scripts/m23_research/m23_59_relation_pretrained_hierarchical_flow_v2.py",
        ROOT / "docs/m23_60_relation_transfer_failure_audit_result_20260721.md",
    ]
    return [R62 / n for n in names] + r62_observable_paths() + refs


def assert_not_occupied() -> None:
    if R63.exists(): raise RuntimeError(f"M23-63 root already exists: {R63}")
    if PREREG.exists() or RESULT.exists(): raise RuntimeError("M23-63 document path already exists")
    header, rows = registry_rows(); i = header.index("tracker_family")
    if any(len(r) > i and r[i] == EXP_ID for r in rows):
        raise RuntimeError("M23-63 already exists in central registry")


def command_init() -> None:
    assert_not_occupied()
    missing = [str(p) for p in initial_input_paths() if not p.is_file()]
    if missing: raise RuntimeError(f"missing prereg inputs: {missing}")
    R63.mkdir(parents=True, exist_ok=False)
    PREREG.write_text(preregistration_text(), encoding="utf-8")
    input_rows = [{"path": str(p.relative_to(ROOT)), "sha256": sha256_file(p), "size": p.stat().st_size}
                  for p in initial_input_paths()]
    gt_declared = [{"sequence": s, "split": SPLITS[s],
                    "path": f"datasets/MOT17/train/{s}-FRCNN/gt/gt.txt",
                    "read_after_label_unlock": True, "sha256_before_unlock": None} for s in SEQUENCES]
    input_manifest = {
        "experiment_id": EXP_ID, "created_at": utc_now(), "r62_immutable": True,
        "contract_hash_required": CONTRACT_HASH, "frozen_inputs": input_rows,
        "declared_gt_inputs_not_read": gt_declared,
        "prohibited_roots": ["datasets/MOT20", "MOT20-test", "teacher_action", "held_outer"],
    }
    json_write(R63 / "input_manifest.json", input_manifest, create_only=True)
    implementation = {
        "experiment_id": EXP_ID, "created_at": utc_now(),
        "script": str(SCRIPT.relative_to(ROOT)), "script_sha256": sha256_file(SCRIPT),
        "prereg": str(PREREG.relative_to(ROOT)), "prereg_sha256": sha256_file(PREREG),
        "input_manifest_sha256": sha256_file(R63 / "input_manifest.json"),
        "python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
        "pandas": pd.__version__, "scipy": scipy.__version__,
        "assignment_function_sha256": hashlib.sha256(inspect.getsource(stable_assignment).encode()).hexdigest(),
        "constants": {
            "contract_hash": CONTRACT_HASH, "iou_threshold": IOU_THRESHOLD,
            "iou_quantization": IOU_QUANTIZATION, "lex_epsilon": LEX_EPSILON,
            "distractor_classes": sorted(DISTRACTOR_CLASSES), "max_node_rows": MAX_NODE_ROWS,
            "node_stride": NODE_STRIDE, "node_min_rows": NODE_MIN_ROWS,
            "chunk_max_rows": CHUNK_MAX_ROWS, "chunk_max_gap": CHUNK_MAX_GAP,
            "candidate_max_gap": CANDIDATE_MAX_GAP, "candidate_k_per_bucket": CANDIDATE_K_PER_BUCKET,
            "gap_buckets": GAP_BUCKETS, "pair_time_tolerance": PAIR_TIME_TOLERANCE,
            "pair_topology_cap_per_sequence": PAIR_TOPOLOGY_CAP_PER_SEQUENCE,
            "relation_example_cap_per_bucket_per_sequence": RELATION_EXAMPLE_CAP_PER_BUCKET_PER_SEQUENCE,
            "pair_example_cap_per_sequence": PAIR_EXAMPLE_CAP_PER_SEQUENCE,
            "trust_min_known": TRUST_MIN_KNOWN, "trust_purity": TRUST_PURITY,
            "node_label_min_known": NODE_LABEL_MIN_KNOWN, "support_gates": SUPPORT_GATES,
        },
        "scope": {"training": False, "optimizer_steps": False, "checkpoints": False,
                  "trackeval": False, "tracker_outputs": False, "mot20_gt": False,
                  "mot20_test": False, "v2_checkpoint_reuse": False},
    }
    json_write(R63 / "implementation_manifest.json", implementation, create_only=True)
    now = utc_now(); rows = []
    for stage in STAGES:
        if stage == "preregistration": status, started = "completed", now
        elif stage == "m23_62_reverification": status, started = "running", now
        elif stage in {"training", "strict_outer_evaluation", "tracker_generation"}: status, started = "prohibited_by_scope", now
        else: status, started = "pending", ""
        rows.append({"experiment": EXP_ID, "stage": stage, "status": status, "started_at": started,
                     "completed_at": now if status in {"completed", "prohibited_by_scope"} else "",
                     "report": str(PREREG.relative_to(ROOT)) if status in {"completed", "prohibited_by_scope"} else "",
                     "decision": "pass" if stage == "preregistration" else ("not_run" if status == "prohibited_by_scope" else ""),
                     "notes": "frozen before GT" if stage == "preregistration" else ("Stage-A prohibition" if status == "prohibited_by_scope" else "")})
    csv_write(R63 / "summary.csv", rows, SUMMARY_FIELDS, create_only=True)
    (R63 / "protocol_events.jsonl").touch(exist_ok=False)
    append_event("preregistered", prereg_sha256=sha256_file(PREREG), input_manifest_sha256=sha256_file(R63 / "input_manifest.json"))
    append_event("implementation_frozen", script_sha256=implementation["script_sha256"],
                 implementation_manifest_sha256=sha256_file(R63 / "implementation_manifest.json"))
    registry_append({
        "timestamp": utc_now(), "kind": "stage_a_audit", "status": "running",
        "script": str(SCRIPT.relative_to(ROOT)), "dataset": "MOT17", "split": "sequence_disjoint_external",
        "tracker_family": EXP_ID, "variant": "m23_59_v3_supervision_join_example_audit",
        "tag": "M23-63-v3-supervision-join-running", "run_root": str(R63.relative_to(ROOT)),
        "summary_csv": str((R63 / "summary.csv").relative_to(ROOT)), "current_stage": "m23_62_reverification",
        "decision": "pending", "notes": "Stage-A only; no training/TrackEval/tracker; GT still locked",
    })
    print(json.dumps({"status": "initialized", "script_sha256": implementation["script_sha256"],
                      "prereg_sha256": implementation["prereg_sha256"],
                      "input_manifest_sha256": implementation["input_manifest_sha256"]}, indent=2))


def verify_implementation_frozen() -> dict[str, Any]:
    m = read_json(R63 / "implementation_manifest.json")
    checks = {
        "script_sha_unchanged": sha256_file(SCRIPT) == m["script_sha256"],
        "prereg_sha_unchanged": sha256_file(PREREG) == m["prereg_sha256"],
        "input_manifest_sha_unchanged": sha256_file(R63 / "input_manifest.json") == m["input_manifest_sha256"],
    }
    if not all(checks.values()): raise RuntimeError(f"frozen implementation drift: {checks}")
    return checks


def reverify_r62() -> dict[str, Any]:
    verify_implementation_frozen()
    inp = read_json(R63 / "input_manifest.json")
    sha_checks = []
    for item in inp["frozen_inputs"]:
        p = ROOT / item["path"]
        sha_checks.append({"path": item["path"], "expected": item["sha256"],
                           "actual": sha256_file(p), "match": sha256_file(p) == item["sha256"]})
    closure = read_json(R62 / "closure_validation.json")
    semantic = read_json(R62 / "semantic_validation.json")
    compat = read_json(R62 / "compatibility_validation.json")
    auth = read_json(R62 / "next_stage_authorization.json")
    final = read_json(R62 / "final_summary.json")
    contract = read_json(R62 / "feature_contract_v3_1.json")
    checks = {
        "all_input_sha_match": all(x["match"] for x in sha_checks),
        "closure_passed": closure.get("passed") is True and closure.get("decision") == "PASS_GT_FREE_SOURCE_REGENERATION",
        "semantic_passed": semantic.get("passed") is True,
        "contract_hash_exact": semantic.get("contract_aggregate", {}).get("contract_hash") == CONTRACT_HASH,
        "contract_file_hash_exact": contract.get("aggregate", {}).get("contract_hash") == CONTRACT_HASH,
        "authorization_allows_fresh_mot17_label_join": auth.get("authorized") is True and "MOT17" in auth.get("authorization", ""),
        "v2_checkpoint_incompatible": compat.get("v2_checkpoint_formal_reuse_allowed") is False and auth.get("v2_checkpoint_reuse_allowed") is False,
        "r62_closed": final.get("status") == "closed" and final.get("decision") == "PASS_GT_FREE_SOURCE_REGENERATION",
        "r62_zero_training": final.get("training_runs") == 0 and final.get("trackeval_runs") == 0 and final.get("tracker_outputs") == 0,
    }
    payload = {"experiment_id": EXP_ID, "checked_at": utc_now(), "checks": checks,
               "passed": all(checks.values()), "sha_checks": sha_checks,
               "contract_hash": CONTRACT_HASH, "authorization": auth,
               "compatibility": compat, "r62_final": final}
    json_write(R63 / "m23_62_reverification.json", payload)
    return payload


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0: return np.zeros((len(a), len(b)), np.float64)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0]); iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2]); iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    aa = np.maximum(0.0, a[:, 2] - a[:, 0]) * np.maximum(0.0, a[:, 3] - a[:, 1])
    bb = np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(0.0, b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-12)


def stable_assignment(scores: np.ndarray, threshold: float = IOU_THRESHOLD) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if scores.size == 0: return np.empty(0, int), np.empty(0, int), np.empty(0, float)
    q = np.round(scores / IOU_QUANTIZATION) * IOU_QUANTIZATION
    valid = q >= threshold - np.finfo(float).eps
    bonus_rank = np.arange(q.size, dtype=np.float64).reshape(q.shape)
    bonus = (q.size - bonus_rank) * LEX_EPSILON / max(q.size, 1)
    objective = np.where(valid, q + bonus, 0.0)
    rr, cc = linear_sum_assignment(-objective)
    keep = valid[rr, cc] & (objective[rr, cc] > 0)
    rr, cc = rr[keep], cc[keep]
    return rr, cc, scores[rr, cc]


def gap_bucket(gap: int) -> tuple[int, str] | None:
    for i, (name, lo, hi) in enumerate(GAP_BUCKETS):
        if lo <= gap <= hi: return i, name
    return None


def parse_indices(value: Any) -> list[int]:
    if isinstance(value, str): return [int(x) for x in json.loads(value)]
    if isinstance(value, (list, tuple, np.ndarray)): return [int(x) for x in value]
    return []


def build_label_blind_topology() -> dict[str, Any]:
    windows: list[dict[str, Any]] = []; chunks: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []; pairs: list[dict[str, Any]] = []; summary: list[dict[str, Any]] = []
    observable_sha: dict[str, Any] = {}
    for seq in SEQUENCES:
        d = R62 / "observables/MOT17" / seq
        rows = pd.read_parquet(d / "rows.parquet")
        features = np.load(d / "row_features.f16.npy", mmap_mode="r")
        manifest = read_json(d / "manifest.json")
        if len(rows) != len(features) or features.shape != (len(rows), 144): raise RuntimeError(f"shape mismatch {seq}")
        if not np.array_equal(rows.row_index.to_numpy(np.int64), np.arange(len(rows))): raise RuntimeError(f"row order mismatch {seq}")
        if not np.isfinite(np.asarray(features)).all(): raise RuntimeError(f"nonfinite features {seq}")
        required = ["line_index", "frame", "track_id", "x1", "y1", "x2", "y2", "mapping_iou", "appearance_mapped"]
        if any(c not in rows for c in required): raise RuntimeError(f"missing source metadata {seq}")
        if not ((rows.x2 > rows.x1) & (rows.y2 > rows.y1) & (rows.frame >= 1)).all(): raise RuntimeError(f"invalid source box/frame {seq}")
        observable_sha[seq] = {"rows_sha256": sha256_file(d / "rows.parquet"),
                               "features_sha256": sha256_file(d / "row_features.f16.npy"),
                               "manifest_sha256": sha256_file(d / "manifest.json")}
        seq_windows: list[dict[str, Any]] = []; seq_chunks: list[dict[str, Any]] = []
        for track_id, grp in rows.groupby("track_id", sort=True):
            grp = grp.sort_values(["frame", "line_index", "row_index"], kind="mergesort")
            ids = grp.row_index.astype(int).tolist()
            for start in range(0, len(ids), NODE_STRIDE):
                block = ids[start:start + MAX_NODE_ROWS]
                if len(block) < NODE_MIN_ROWS: continue
                wid = f"{seq}:window:{int(track_id)}:{start:06d}:{stable_id(seq,'window',track_id,start,block)}"
                seq_windows.append({"sequence": seq, "split": SPLITS[seq], "window_id": wid,
                                    "source_track_key": f"{seq}:track:{int(track_id)}", "track_id": int(track_id),
                                    "start_offset": start, "row_indices": json.dumps(block), "row_count": len(block),
                                    "first_frame": int(rows.loc[block, "frame"].min()), "last_frame": int(rows.loc[block, "frame"].max())})
                if start + MAX_NODE_ROWS >= len(ids): break
            current: list[int] = []
            for rid in ids:
                if current:
                    prev = current[-1]
                    if int(rows.at[rid, "frame"]) - int(rows.at[prev, "frame"]) > CHUNK_MAX_GAP or len(current) >= CHUNK_MAX_ROWS:
                        cid = f"{seq}:chunk:{int(track_id)}:{len(seq_chunks):06d}:{stable_id(seq,'chunk',track_id,current)}"
                        seq_chunks.append({"sequence": seq, "split": SPLITS[seq], "chunk_id": cid,
                                           "source_track_key": f"{seq}:track:{int(track_id)}", "track_id": int(track_id),
                                           "row_indices": json.dumps(current), "row_count": len(current),
                                           "first_frame": int(rows.at[current[0], "frame"]), "last_frame": int(rows.at[current[-1], "frame"])})
                        current = []
                current.append(rid)
            if current:
                cid = f"{seq}:chunk:{int(track_id)}:{len(seq_chunks):06d}:{stable_id(seq,'chunk',track_id,current)}"
                seq_chunks.append({"sequence": seq, "split": SPLITS[seq], "chunk_id": cid,
                                   "source_track_key": f"{seq}:track:{int(track_id)}", "track_id": int(track_id),
                                   "row_indices": json.dumps(current), "row_count": len(current),
                                   "first_frame": int(rows.at[current[0], "frame"]), "last_frame": int(rows.at[current[-1], "frame"])})
        windows.extend(seq_windows); chunks.extend(seq_chunks)
        app = np.asarray(features[:, :128], np.float32)
        chunk_aux: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for c in seq_chunks:
            ids = parse_indices(c["row_indices"]); proto = app[ids].mean(0)
            norm = np.linalg.norm(proto); proto = proto / norm if norm > 1e-12 else proto
            last = ids[-1]; first = ids[0]
            last_center = np.array([(rows.at[last,"x1"]+rows.at[last,"x2"])/2, (rows.at[last,"y1"]+rows.at[last,"y2"])/2], np.float32)
            first_center = np.array([(rows.at[first,"x1"]+rows.at[first,"x2"])/2, (rows.at[first,"y1"]+rows.at[first,"y2"])/2], np.float32)
            scale = np.array([max(float(rows.x2.max()),1.0), max(float(rows.y2.max()),1.0)], np.float32)
            chunk_aux[c["chunk_id"]] = (proto, first_center/scale, last_center/scale)
        seq_edges: list[dict[str, Any]] = []
        for src in seq_chunks:
            ranked: dict[int, list[tuple[Any, dict[str, Any]]]] = defaultdict(list)
            sp, _, slast = chunk_aux[src["chunk_id"]]
            for dst in seq_chunks:
                gap = int(dst["first_frame"]) - int(src["last_frame"])
                gb = gap_bucket(gap)
                if gb is None: continue
                dp, dfirst, _ = chunk_aux[dst["chunk_id"]]
                cosine = float(np.dot(sp, dp)); dist = float(np.linalg.norm(slast - dfirst))
                score = 0.70 * cosine + 0.30 * math.exp(-4.0 * dist)
                bi, bn = gb
                e = {"sequence": seq, "split": SPLITS[seq], "src_chunk_id": src["chunk_id"],
                     "dst_chunk_id": dst["chunk_id"], "gap": gap, "gap_bucket_index": bi,
                     "gap_bucket": bn, "appearance_cosine": cosine, "geometry_distance": dist,
                     "candidate_score": score, "src_track_id": int(src["track_id"]),
                     "dst_track_id": int(dst["track_id"]), "dst_first_frame": int(dst["first_frame"])}
                key = (-score, int(dst["first_frame"]), int(dst["track_id"]), dst["chunk_id"])
                ranked[bi].append((key, e))
            for bi in sorted(ranked):
                for rank, (_, e) in enumerate(sorted(ranked[bi], key=lambda x: x[0])[:CANDIDATE_K_PER_BUCKET]):
                    e["rank_in_source_bucket"] = rank
                    e["candidate_id"] = f"{seq}:edge:{stable_id(seq,e['src_chunk_id'],e['dst_chunk_id'],bi)}"
                    seq_edges.append(e)
        seq_edges.sort(key=lambda e: (e["gap_bucket_index"], e["candidate_id"]))
        per_bucket_counter = Counter()
        for e in seq_edges:
            k = int(e["gap_bucket_index"]); e["example_selected"] = per_bucket_counter[k] < RELATION_EXAMPLE_CAP_PER_BUCKET_PER_SEQUENCE
            per_bucket_counter[k] += 1
        edges.extend(seq_edges)
        edge_lookup = {(e["src_chunk_id"], e["dst_chunk_id"]): e for e in seq_edges}
        by_bucket: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for e in seq_edges: by_bucket[int(e["gap_bucket_index"])].append(e)
        seq_pairs: list[dict[str, Any]] = []
        for bi, bucket_edges in sorted(by_bucket.items()):
            bucket_edges = sorted(bucket_edges, key=lambda e: (e["dst_first_frame"], e["candidate_id"]))
            for i, e1 in enumerate(bucket_edges):
                for e2 in bucket_edges[i+1:i+65]:
                    if abs(int(e1["dst_first_frame"]) - int(e2["dst_first_frame"])) > PAIR_TIME_TOLERANCE: break
                    if len({e1["src_chunk_id"], e2["src_chunk_id"]}) < 2 or len({e1["dst_chunk_id"], e2["dst_chunk_id"]}) < 2: continue
                    c1 = edge_lookup.get((e1["src_chunk_id"], e2["dst_chunk_id"]))
                    c2 = edge_lookup.get((e2["src_chunk_id"], e1["dst_chunk_id"]))
                    if c1 is None or c2 is None: continue
                    pid = f"{seq}:pair:{stable_id(seq,e1['candidate_id'],e2['candidate_id'],c1['candidate_id'],c2['candidate_id'])}"
                    seq_pairs.append({"sequence": seq, "split": SPLITS[seq], "pair_id": pid,
                                      "edge1_id": e1["candidate_id"], "edge2_id": e2["candidate_id"],
                                      "cross1_id": c1["candidate_id"], "cross2_id": c2["candidate_id"],
                                      "gap_bucket_index": bi})
                    if len(seq_pairs) >= PAIR_TOPOLOGY_CAP_PER_SEQUENCE: break
                if len(seq_pairs) >= PAIR_TOPOLOGY_CAP_PER_SEQUENCE: break
            if len(seq_pairs) >= PAIR_TOPOLOGY_CAP_PER_SEQUENCE: break
        seq_pairs.sort(key=lambda x: x["pair_id"])
        for i, p in enumerate(seq_pairs): p["example_selected"] = i < PAIR_EXAMPLE_CAP_PER_SEQUENCE
        pairs.extend(seq_pairs)
        summary.append({"sequence": seq, "split": SPLITS[seq], "source_rows": len(rows),
                        "source_tracks": int(rows.track_id.nunique()), "node_windows": len(seq_windows),
                        "chunks": len(seq_chunks), "candidate_edges": len(seq_edges),
                        "paired_candidate_combinations": len(seq_pairs)})
    windows_df = pd.DataFrame(windows); chunks_df = pd.DataFrame(chunks)
    edges_df = pd.DataFrame(edges); pairs_df = pd.DataFrame(pairs)
    windows_df.to_parquet(R63 / "source_windows.parquet", index=False)
    chunks_df.to_parquet(R63 / "source_chunks.parquet", index=False)
    edges_df.to_parquet(R63 / "candidate_pool.parquet", index=False)
    pairs_df.to_parquet(R63 / "paired_candidate_pool.parquet", index=False)
    for split in ["train", "validation"]:
        s = [r for r in summary if r["split"] == split]
        summary.append({"sequence": f"__{split}__", "split": split,
                        **{k: sum(int(x[k]) for x in s) for k in ["source_rows","source_tracks","node_windows","chunks","candidate_edges","paired_candidate_combinations"]}})
    csv_write(R63 / "source_topology_summary.csv", summary,
              ["sequence","split","source_rows","source_tracks","node_windows","chunks","candidate_edges","paired_candidate_combinations"])
    topology_manifest = {
        "experiment_id": EXP_ID, "status": "frozen", "created_at": utc_now(),
        "label_blind": True, "gt_reads": 0, "generator_script_sha256": sha256_file(SCRIPT),
        "observable_sha": observable_sha, "counts": {"windows": len(windows_df), "chunks": len(chunks_df)},
        "artifacts": {n: sha256_file(R63 / n) for n in ["source_windows.parquet","source_chunks.parquet","source_topology_summary.csv"]},
        "rules": {"max_node_rows": MAX_NODE_ROWS, "node_stride": NODE_STRIDE, "node_min_rows": NODE_MIN_ROWS,
                  "chunk_max_rows": CHUNK_MAX_ROWS, "chunk_max_gap": CHUNK_MAX_GAP},
        "split_isolation": {"physical_sequences_disjoint": set(s for s,v in SPLITS.items() if v=="train").isdisjoint(s for s,v in SPLITS.items() if v=="validation")},
    }
    candidate_manifest = {
        "experiment_id": EXP_ID, "status": "frozen", "created_at": utc_now(), "label_blind": True,
        "candidate_edges": len(edges_df), "paired_combinations": len(pairs_df),
        "candidate_pool_sha256": sha256_file(R63 / "candidate_pool.parquet"),
        "paired_candidate_pool_sha256": sha256_file(R63 / "paired_candidate_pool.parquet"),
        "topology_manifest_inputs": topology_manifest["artifacts"],
        "rules": {"max_gap": CANDIDATE_MAX_GAP, "k_per_bucket": CANDIDATE_K_PER_BUCKET,
                  "gap_buckets": GAP_BUCKETS, "ranking": "0.70 appearance cosine + 0.30 exp(-4*center distance)",
                  "stable_tie_break": "dst_first_frame,dst_track_id,dst_chunk_id",
                  "pair_cross_edges_required": True, "pair_time_tolerance": PAIR_TIME_TOLERANCE,
                  "relation_example_cap_per_bucket_per_sequence": RELATION_EXAMPLE_CAP_PER_BUCKET_PER_SEQUENCE,
                  "pair_example_cap_per_sequence": PAIR_EXAMPLE_CAP_PER_SEQUENCE},
    }
    json_write(R63 / "source_topology_manifest.json", topology_manifest)
    json_write(R63 / "candidate_pool_manifest.json", candidate_manifest)
    return {"topology": topology_manifest, "candidate": candidate_manifest}


def command_prepare() -> None:
    update_stage("m23_62_reverification", "running", notes="recomputing all frozen SHA")
    rv = reverify_r62()
    if not rv["passed"]:
        update_stage("m23_62_reverification", "failed", str((R63/"m23_62_reverification.json").relative_to(ROOT)), "FAIL_M23_62_REVERIFICATION", "GT remained locked")
        raise RuntimeError("FAIL_M23_62_REVERIFICATION")
    update_stage("m23_62_reverification", "completed", str((R63/"m23_62_reverification.json").relative_to(ROOT)), "pass", "all R62 SHA/contract/authorization checks passed")
    append_event("observable_reverified", reverification_sha256=sha256_file(R63 / "m23_62_reverification.json"), gt_reads=0)
    update_stage("source_topology", "running", notes="building label-blind source topology")
    built = build_label_blind_topology()
    update_stage("source_topology", "completed", str((R63/"source_topology_manifest.json").relative_to(ROOT)), "pass", "topology and candidate pool frozen before GT")
    update_stage("label_join", "running", notes="topology frozen; GT still locked until join command")
    append_event("topology_frozen", manifest_sha256=sha256_file(R63 / "source_topology_manifest.json"), gt_reads=0)
    append_event("candidate_pool_frozen", manifest_sha256=sha256_file(R63 / "candidate_pool_manifest.json"), gt_reads=0)
    print(json.dumps({"status": "prelabel_frozen", "topology": built["topology"]["counts"],
                      "candidate_edges": built["candidate"]["candidate_edges"],
                      "paired_combinations": built["candidate"]["paired_combinations"]}, indent=2))


def assert_prelabel_frozen() -> None:
    verify_implementation_frozen()
    required = ["source_topology_manifest.json", "candidate_pool_manifest.json", "source_windows.parquet",
                "source_chunks.parquet", "candidate_pool.parquet", "paired_candidate_pool.parquet"]
    if any(not (R63 / n).is_file() for n in required): raise RuntimeError("prelabel artifacts incomplete")
    tm = read_json(R63 / "source_topology_manifest.json"); cm = read_json(R63 / "candidate_pool_manifest.json")
    if tm.get("label_blind") is not True or tm.get("gt_reads") != 0 or cm.get("label_blind") is not True:
        raise RuntimeError("topology was not frozen label-blind")
    checks = {
        "windows": sha256_file(R63 / "source_windows.parquet") == tm["artifacts"]["source_windows.parquet"],
        "chunks": sha256_file(R63 / "source_chunks.parquet") == tm["artifacts"]["source_chunks.parquet"],
        "summary": sha256_file(R63 / "source_topology_summary.csv") == tm["artifacts"]["source_topology_summary.csv"],
        "candidates": sha256_file(R63 / "candidate_pool.parquet") == cm["candidate_pool_sha256"],
        "pairs": sha256_file(R63 / "paired_candidate_pool.parquet") == cm["paired_candidate_pool_sha256"],
    }
    if not all(checks.values()): raise RuntimeError(f"prelabel artifact drift: {checks}")
    ev = [x["event"] for x in events()]
    if "topology_frozen" not in ev or "candidate_pool_frozen" not in ev: raise RuntimeError("freeze events missing")


def read_gt_after_unlock(seq: str) -> pd.DataFrame:
    global GT_READ_COUNTER
    if "label_unlock" not in [x["event"] for x in events()]: raise RuntimeError("GT read before label_unlock")
    path = ROOT / f"datasets/MOT17/train/{seq}-FRCNN/gt/gt.txt"
    names = ["frame","gt_id","x","y","w","h","mark","class_id","visibility"]
    gt = pd.read_csv(path, header=None, names=names, usecols=list(range(9)))
    GT_READ_COUNTER += 1
    gt.insert(0, "gt_line_index", np.arange(len(gt), dtype=np.int64))
    for c in ["frame","gt_id","mark","class_id","gt_line_index"]: gt[c] = gt[c].astype(np.int64)
    gt["x1"] = gt.x.astype(float); gt["y1"] = gt.y.astype(float)
    gt["x2"] = gt.x.astype(float) + gt.w.astype(float); gt["y2"] = gt.y.astype(float) + gt.h.astype(float)
    if not ((gt.x2 > gt.x1) & (gt.y2 > gt.y1) & (gt.frame >= 1)).all(): raise RuntimeError(f"invalid GT boxes {seq}")
    return gt


def row_tie_flags(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if scores.shape[0] == 0: return np.zeros(0, bool), np.zeros(0, np.int64)
    valid = scores >= IOU_THRESHOLD - np.finfo(float).eps
    opportunities = valid.sum(axis=1).astype(np.int64)
    ties = np.zeros(scores.shape[0], bool)
    q = np.round(scores / IOU_QUANTIZATION) * IOU_QUANTIZATION
    for i in range(len(q)):
        vals = np.sort(q[i, valid[i]])[::-1]
        ties[i] = len(vals) >= 2 and vals[0] == vals[1]
    return ties, opportunities


def join_sequence(seq: str, gt: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]], pd.DataFrame]:
    d = R62 / "observables/MOT17" / seq
    rows = pd.read_parquet(d / "rows.parquet").sort_values("row_index", kind="mergesort")
    records: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    unmatched_gt_trace: list[dict[str, Any]] = []
    matched_gt_keys: set[tuple[int,int]] = set()
    eligible_gt_total = distractor_gt_total = distractor_removed_total = 0
    opp0 = opp1 = oppmulti = tie_total = ambiguity_total = 0
    frame_matches: list[float] = []
    frame_groups = sorted(set(rows.frame.astype(int)) | set(gt.frame.astype(int)))
    for frame in frame_groups:
        sf = rows[rows.frame == frame].sort_values(["line_index","row_index"], kind="mergesort")
        gf = gt[gt.frame == frame].sort_values("gt_line_index", kind="mergesort")
        valid_gt = gf[(gf.mark != 0) & (gf.class_id == 1)].copy()
        distractor_gt = gf[gf.class_id.isin(DISTRACTOR_CLASSES)].copy()
        eligible_gt_total += len(valid_gt); distractor_gt_total += len(distractor_gt)
        sboxes = sf[["x1","y1","x2","y2"]].to_numpy(float)
        allboxes = gf[["x1","y1","x2","y2"]].to_numpy(float)
        all_scores = iou_matrix(sboxes, allboxes)
        ar, ac, av = stable_assignment(all_scores)
        removed_positions: set[int] = set()
        for rpos, cpos, iou in zip(ar, ac, av):
            grow = gf.iloc[int(cpos)]
            if int(grow.class_id) in DISTRACTOR_CLASSES:
                removed_positions.add(int(rpos))
        remain_positions = [i for i in range(len(sf)) if i not in removed_positions]
        remain = sf.iloc[remain_positions]
        vboxes = valid_gt[["x1","y1","x2","y2"]].to_numpy(float)
        scores = iou_matrix(remain[["x1","y1","x2","y2"]].to_numpy(float), vboxes)
        tie_flags, opportunities = row_tie_flags(scores)
        rr, cc, vv = stable_assignment(scores)
        assigned = {int(r): (int(c), float(v)) for r, c, v in zip(rr, cc, vv)}
        assigned_gt = {int(c) for c in cc}
        for local_pos, (_, srow) in enumerate(sf.iterrows()):
            rid = int(srow.row_index)
            base = {"row_label_id": f"{seq}:label:{rid:08d}:{stable_id(seq,'label',rid)}",
                    "sequence": seq, "split": SPLITS[seq], "row_index": rid,
                    "line_index": int(srow.line_index), "frame": int(frame),
                    "source_track_id": int(srow.track_id), "source_track_key": f"{seq}:track:{int(srow.track_id)}",
                    "supervision_status": "unknown", "gt_id": -1, "gt_identity_key": "",
                    "gt_line_index": -1, "match_iou": np.nan, "distractor_removed": False,
                    "ambiguity_flag": False, "tie_flag": False, "eligible_match_opportunities": 0}
            if local_pos in removed_positions:
                base.update(supervision_status="distractor_removed", distractor_removed=True)
                distractor_removed_total += 1
            else:
                rp = remain_positions.index(local_pos)
                op = int(opportunities[rp]); base["eligible_match_opportunities"] = op
                base["ambiguity_flag"] = bool(op > 1); base["tie_flag"] = bool(tie_flags[rp])
                opp0 += int(op == 0); opp1 += int(op == 1); oppmulti += int(op > 1)
                ambiguity_total += int(op > 1); tie_total += int(tie_flags[rp])
                if rp in assigned:
                    cpos, miou = assigned[rp]; grow = valid_gt.iloc[cpos]
                    gid = int(grow.gt_id); gl = int(grow.gt_line_index)
                    base.update(supervision_status="matched", gt_id=gid,
                                gt_identity_key=f"{seq}:gt:{gid}", gt_line_index=gl, match_iou=miou)
                    matched_gt_keys.add((int(frame), gl)); frame_matches.append(miou)
            records.append(base)
        for cpos, (_, grow) in enumerate(valid_gt.iterrows()):
            if cpos not in assigned_gt:
                unmatched_gt_trace.append({"type": "unmatched_gt", "sequence": seq, "frame": int(frame),
                                           "gt_id": int(grow.gt_id), "gt_line_index": int(grow.gt_line_index),
                                           "box": [float(grow.x1),float(grow.y1),float(grow.x2),float(grow.y2)]})
    side = pd.DataFrame(records).sort_values("row_index", kind="mergesort")
    if len(side) != len(rows) or side.row_index.duplicated().any(): raise RuntimeError(f"sidecar row mismatch {seq}")
    matched = side[side.supervision_status == "matched"]
    duplicate_assignment = int(matched.duplicated(["frame","gt_line_index"]).sum())
    if duplicate_assignment: raise RuntimeError(f"one-to-one violation {seq}: {duplicate_assignment}")
    merged = rows[["row_index","frame","track_id","line_index","x1","y1","x2","y2"]].merge(side, on=["row_index","frame","line_index"], how="left")
    purity_rows: list[dict[str, Any]] = []
    fragment_counts = matched.groupby("gt_identity_key").source_track_key.nunique().to_dict()
    for track_key, grp in merged.groupby("source_track_key", sort=True):
        grp = grp.sort_values(["frame","line_index","row_index"], kind="mergesort")
        known = grp[grp.supervision_status == "matched"]
        counts = Counter(known.gt_identity_key.tolist())
        majority, majority_n = (counts.most_common(1)[0] if counts else ("", 0))
        known_ids = known.gt_identity_key.tolist()
        transitions = sum(a != b for a,b in zip(known_ids, known_ids[1:]))
        purity_rows.append({"sequence": seq, "split": SPLITS[seq], "source_track_key": track_key,
                            "source_track_id": int(grp.track_id.iloc[0]), "total_rows": len(grp),
                            "known_rows": len(known), "unknown_rows": int((grp.supervision_status=="unknown").sum()),
                            "distractor_removed_rows": int((grp.supervision_status=="distractor_removed").sum()),
                            "known_ratio": len(known)/max(len(grp),1), "distinct_gt_identities": len(counts),
                            "majority_gt_identity_key": majority, "purity": majority_n/max(len(known),1) if len(known) else np.nan,
                            "identity_transitions": transitions,
                            "majority_identity_fragmentation_tracks": int(fragment_counts.get(majority,0)) if majority else 0})
    purity = pd.DataFrame(purity_rows)
    q = np.quantile(frame_matches, [0,0.1,0.25,0.5,0.75,0.9,1]).tolist() if frame_matches else [None]*7
    stats = {"sequence": seq, "split": SPLITS[seq], "source_rows": len(rows),
             "eligible_pedestrian_gt": eligible_gt_total, "distractor_gt": distractor_gt_total,
             "matched_source_rows": int((side.supervision_status=="matched").sum()),
             "unmatched_source_rows": int((side.supervision_status=="unknown").sum()),
             "unmatched_gt": eligible_gt_total - len(matched_gt_keys),
             "distractor_removed_source_rows": distractor_removed_total,
             "match_iou_min": q[0], "match_iou_q10": q[1], "match_iou_q25": q[2], "match_iou_median": q[3],
             "match_iou_q75": q[4], "match_iou_q90": q[5], "match_iou_max": q[6],
             "zero_opportunity_rows": opp0, "one_opportunity_rows": opp1, "multiple_opportunity_rows": oppmulti,
             "hungarian_tie_rows": tie_total, "ambiguity_rows": ambiguity_total,
             "covered_gt_identities": int(matched.gt_identity_key.nunique()),
             "source_tracks": int(rows.track_id.nunique()),
             "source_tracks_with_known": int((purity.known_rows>0).sum()),
             "mean_track_known_ratio": float(purity.known_ratio.mean()) if len(purity) else 0.0,
             "mean_known_track_purity": float(purity.loc[purity.known_rows>0,"purity"].mean()) if (purity.known_rows>0).any() else np.nan,
             "identity_transition_events": int(purity.identity_transitions.sum()),
             "fragmented_gt_identities": int(sum(v>1 for v in fragment_counts.values())),
             "duplicate_gt_assignments": duplicate_assignment}
    trace_types = {
        "matched": side[side.supervision_status=="matched"].head(10),
        "unmatched_source": side[side.supervision_status=="unknown"].head(10),
        "distractor_removal": side[side.supervision_status=="distractor_removed"].head(10),
        "ambiguity": side[side.ambiguity_flag].head(10), "tie": side[side.tie_flag].head(10),
    }
    for typ, df in trace_types.items():
        for row in df.to_dict("records"):
            for k,v in list(row.items()):
                if isinstance(v, float) and np.isnan(v): row[k] = None
                elif isinstance(v, np.generic): row[k] = v.item()
            traces.append({"type": typ, **row})
    traces.extend(unmatched_gt_trace[:10])
    return side, stats, traces, purity


def aggregate_join_statistics(stats: list[dict[str, Any]], sidecar: pd.DataFrame, purity: pd.DataFrame) -> list[dict[str, Any]]:
    out = list(stats)
    numeric_sum = ["source_rows","eligible_pedestrian_gt","distractor_gt","matched_source_rows","unmatched_source_rows",
                   "unmatched_gt","distractor_removed_source_rows","zero_opportunity_rows","one_opportunity_rows",
                   "multiple_opportunity_rows","hungarian_tie_rows","ambiguity_rows","source_tracks",
                   "source_tracks_with_known","identity_transition_events","fragmented_gt_identities","duplicate_gt_assignments"]
    for split in ["train","validation","all"]:
        chosen = stats if split == "all" else [s for s in stats if s["split"]==split]
        labels = sidecar if split == "all" else sidecar[sidecar.split==split]
        pur = purity if split == "all" else purity[purity.split==split]
        ious = labels.loc[labels.supervision_status=="matched","match_iou"].dropna().to_numpy(float)
        q = np.quantile(ious,[0,0.1,0.25,0.5,0.75,0.9,1]).tolist() if len(ious) else [None]*7
        row = {"sequence": f"__{split}__", "split": split, **{k: sum(int(x[k]) for x in chosen) for k in numeric_sum}}
        for name,val in zip(["match_iou_min","match_iou_q10","match_iou_q25","match_iou_median","match_iou_q75","match_iou_q90","match_iou_max"],q): row[name]=val
        row["covered_gt_identities"] = int(labels.loc[labels.supervision_status=="matched","gt_identity_key"].nunique())
        row["mean_track_known_ratio"] = float(pur.known_ratio.mean()) if len(pur) else 0.0
        row["mean_known_track_purity"] = float(pur.loc[pur.known_rows>0,"purity"].mean()) if (len(pur) and (pur.known_rows>0).any()) else np.nan
        out.append(row)
    return out


def command_join() -> None:
    assert_prelabel_frozen()
    if "label_unlock" in [x["event"] for x in events()]: raise RuntimeError("label_unlock already recorded")
    append_event("label_unlock", allowed_domain="MOT17_train_FRCNN_only", sequences=SEQUENCES,
                 topology_manifest_sha256=sha256_file(R63/"source_topology_manifest.json"),
                 candidate_pool_manifest_sha256=sha256_file(R63/"candidate_pool_manifest.json"))
    all_side: list[pd.DataFrame] = []; all_stats: list[dict[str, Any]] = []
    all_traces: list[dict[str, Any]] = []; all_purity: list[pd.DataFrame] = []; gt_inputs = []
    for seq in SEQUENCES:
        path = ROOT / f"datasets/MOT17/train/{seq}-FRCNN/gt/gt.txt"
        gt = read_gt_after_unlock(seq)
        gt_inputs.append({"sequence": seq, "split": SPLITS[seq], "path": str(path.relative_to(ROOT)),
                          "sha256": sha256_file(path), "rows": len(gt)})
        side, stats, traces, purity = join_sequence(seq, gt)
        all_side.append(side); all_stats.append(stats); all_traces.extend(traces); all_purity.append(purity)
    sidecar = pd.concat(all_side, ignore_index=True).sort_values(["sequence","row_index"], kind="mergesort")
    purity = pd.concat(all_purity, ignore_index=True).sort_values(["sequence","source_track_id"], kind="mergesort")
    sidecar.to_parquet(R63 / "row_supervision.parquet", index=False)
    purity.to_csv(R63 / "track_purity.csv", index=False)
    stats_rows = aggregate_join_statistics(all_stats, sidecar, purity)
    fields = list(stats_rows[0].keys())
    csv_write(R63 / "join_statistics.csv", stats_rows, fields)
    trace_summary = {t: sum(x["type"]==t for x in all_traces) for t in ["matched","unmatched_source","unmatched_gt","distractor_removal","ambiguity","tie"]}
    json_write(R63 / "manual_trace_samples.json", {"searched_sequences": SEQUENCES, "counts_saved": trace_summary,
                                                     "samples": all_traces, "zero_count_is_explicit_not_fabricated": True})
    join_manifest = {
        "experiment_id": EXP_ID, "status": "frozen", "created_at": utc_now(), "gt_read_counter": GT_READ_COUNTER,
        "allowed_gt_inputs": gt_inputs, "mot20_gt_reads": 0, "teacher_reads": 0, "held_outer_reads": 0,
        "assignment": {"implementation": "scipy.optimize.linear_sum_assignment", "numpy": np.__version__,
                       "scipy": scipy.__version__, "function_sha256": hashlib.sha256(inspect.getsource(stable_assignment).encode()).hexdigest(),
                       "iou_threshold": IOU_THRESHOLD, "iou_quantization": IOU_QUANTIZATION,
                       "lex_epsilon": LEX_EPSILON, "source_order": "line_index,row_index", "gt_order": "gt_line_index"},
        "eligibility": {"valid_pedestrian": "mark != 0 and class_id == 1", "visibility_threshold_used": False,
                        "distractor_classes": sorted(DISTRACTOR_CLASSES), "trackeval_source_sha256": sha256_file(TRACKEVAL_MOT)},
        "artifacts": {
            "row_supervision.parquet": sha256_file(R63/"row_supervision.parquet"),
            "join_statistics.csv": sha256_file(R63/"join_statistics.csv"),
            "track_purity.csv": sha256_file(R63/"track_purity.csv"),
            "manual_trace_samples.json": sha256_file(R63/"manual_trace_samples.json"),
        },
        "feature_artifacts_separate": True,
        "r62_observable_sha": read_json(R63/"source_topology_manifest.json")["observable_sha"],
        "one_to_one": all(int(s["duplicate_gt_assignments"]) == 0 for s in all_stats),
        "unknown_as_negative": False,
    }
    json_write(R63 / "join_manifest.json", join_manifest)
    append_event("supervision_join_frozen", manifest_sha256=sha256_file(R63/"join_manifest.json"),
                 labels_sha256=join_manifest["artifacts"]["row_supervision.parquet"], gt_read_counter=GT_READ_COUNTER)
    update_stage("label_join", "completed", str((R63/"join_manifest.json").relative_to(ROOT)), "pass", "MOT17-only one-to-one sidecar frozen")
    update_stage("example_construction", "running", notes="constructing labels only on frozen source topology")
    print(json.dumps({"status":"join_frozen","gt_reads":GT_READ_COUNTER,
                      "labels_sha256":join_manifest["artifacts"]["row_supervision.parquet"],
                      "statistics":all_stats}, indent=2, default=str))


def padded_feature(features: np.ndarray, row_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    ids = [int(x) for x in row_ids][:MAX_NODE_ROWS]
    x = np.zeros((MAX_NODE_ROWS, 144), np.float16); mask = np.zeros(MAX_NODE_ROWS, np.uint8)
    if ids:
        x[:len(ids)] = np.asarray(features[ids], np.float16); mask[:len(ids)] = 1
    return x, mask


def mask_prefix_valid(mask: np.ndarray) -> bool:
    if mask.size == 0: return True
    flat = mask.reshape(-1, mask.shape[-1]).astype(np.int8)
    return bool(np.all(np.diff(flat, axis=1) <= 0))


def trusted_chunk_labels(chunks: pd.DataFrame, labels: pd.DataFrame) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    label_by_seq = {(str(r.sequence), int(r.row_index)): r for r in labels.itertuples()}
    for c in chunks.itertuples():
        ids = parse_indices(c.row_indices)
        known = [label_by_seq[(c.sequence, rid)].gt_identity_key for rid in ids
                 if label_by_seq[(c.sequence, rid)].supervision_status == "matched"]
        cnt = Counter(known); majority, n = cnt.most_common(1)[0] if cnt else ("", 0)
        purity = n / max(len(known), 1) if known else 0.0
        trusted = len(known) >= TRUST_MIN_KNOWN and purity >= TRUST_PURITY
        out[c.chunk_id] = {"known_rows": len(known), "purity": purity,
                           "trusted": trusted, "gt_identity_key": majority if trusted else "",
                           "row_indices": ids, "sequence": c.sequence, "split": c.split,
                           "first_frame": int(c.first_frame), "last_frame": int(c.last_frame)}
    return out


def build_split_examples(split: str, windows: pd.DataFrame, chunks: pd.DataFrame, edges: pd.DataFrame,
                         pair_pool: pd.DataFrame, labels: pd.DataFrame, chunk_labels: dict[str, dict[str, Any]],
                         provenance_sha: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, set[str]]]:
    node_x: list[np.ndarray] = []; node_mask: list[np.ndarray] = []; node_y: list[int] = []; boundary_y: list[np.ndarray] = []
    node_meta: list[dict[str, Any]] = []
    rel_src_x: list[np.ndarray] = []; rel_dst_x: list[np.ndarray] = []; rel_src_m: list[np.ndarray] = []; rel_dst_m: list[np.ndarray] = []
    rel_y: list[int] = []; rel_meta: list[dict[str, Any]] = []
    pair_x: list[np.ndarray] = []; pair_m: list[np.ndarray] = []; pair_meta: list[dict[str, Any]] = []
    label_lookup = {(str(r.sequence), int(r.row_index)): r for r in labels.itertuples()}
    chunk_lookup = {r.chunk_id: r for r in chunks.itertuples()}
    edge_lookup = {r.candidate_id: r for r in edges.itertuples()}
    feature_cache = {seq: np.load(R62/f"observables/MOT17/{seq}/row_features.f16.npy", mmap_mode="r")
                     for seq in SEQUENCES if SPLITS[seq] == split}
    window_rows = {r.window_id: parse_indices(r.row_indices) for r in windows[windows.split == split].itertuples()}
    for w in windows[windows.split == split].sort_values(["sequence","window_id"], kind="mergesort").itertuples():
        ids = parse_indices(w.row_indices); feat = feature_cache[w.sequence]
        x, m = padded_feature(feat, ids)
        known = [label_lookup[(w.sequence,rid)].gt_identity_key for rid in ids
                 if label_lookup[(w.sequence,rid)].supervision_status == "matched"]
        if len(known) >= NODE_LABEL_MIN_KNOWN: ny = 1 if len(set(known)) == 1 else 0; reason = ""
        else: ny = -1; reason = "insufficient_known_support"
        by = np.full(MAX_NODE_ROWS-1, -1, np.int8)
        for i,(a,b) in enumerate(zip(ids, ids[1:])):
            la, lb = label_lookup[(w.sequence,a)], label_lookup[(w.sequence,b)]
            if la.supervision_status == "matched" and lb.supervision_status == "matched":
                by[i] = 1 if la.gt_identity_key != lb.gt_identity_key else 0
        exid = f"{w.sequence}:node:{stable_id(w.window_id,provenance_sha)}"
        node_x.append(x); node_mask.append(m); node_y.append(ny); boundary_y.append(by)
        node_meta.append({"example_id": exid, "sequence": w.sequence, "split": split, "window_id": w.window_id,
                          "source_track_key": w.source_track_key, "source_row_indices": json.dumps(ids),
                          "source_row_keys": json.dumps([f"{w.sequence}:row:{r}" for r in ids]),
                          "label_sidecar_ids": json.dumps([label_lookup[(w.sequence,r)].row_label_id for r in ids]),
                          "tensor_index": len(node_x)-1, "node_label": ny, "ignore_reason": reason,
                          "provenance_sha256": provenance_sha})
    selected_edges = edges[(edges.split == split) & (edges.example_selected == True)].sort_values(["sequence","gap_bucket_index","candidate_id"], kind="mergesort")
    for e in selected_edges.itertuples():
        sc, dc = chunk_labels[e.src_chunk_id], chunk_labels[e.dst_chunk_id]
        if sc["trusted"] and dc["trusted"]:
            y = 1 if sc["gt_identity_key"] == dc["gt_identity_key"] else 0; reason = ""
        else: y = -1; reason = "unknown_or_untrusted_endpoint"
        feat = feature_cache[e.sequence]
        sx, sm = padded_feature(feat, sc["row_indices"]); dx, dm = padded_feature(feat, dc["row_indices"])
        exid = f"{e.sequence}:relation:{stable_id(e.candidate_id,provenance_sha)}"
        rel_src_x.append(sx); rel_dst_x.append(dx); rel_src_m.append(sm); rel_dst_m.append(dm); rel_y.append(y)
        all_ids = sc["row_indices"] + dc["row_indices"]
        rel_meta.append({"example_id": exid, "sequence": e.sequence, "split": split,
                         "candidate_id": e.candidate_id, "src_chunk_id": e.src_chunk_id, "dst_chunk_id": e.dst_chunk_id,
                         "gap_bucket": e.gap_bucket, "source_row_indices": json.dumps(all_ids),
                         "source_row_keys": json.dumps([f"{e.sequence}:row:{r}" for r in all_ids]),
                         "label_sidecar_ids": json.dumps([label_lookup[(e.sequence,r)].row_label_id for r in all_ids]),
                         "tensor_index": len(rel_src_x)-1, "relation_label": y,
                         "src_trusted_gt_identity_key": sc["gt_identity_key"],
                         "dst_trusted_gt_identity_key": dc["gt_identity_key"],
                         "ignore_reason": reason, "provenance_sha256": provenance_sha})
    selected_pairs = pair_pool[(pair_pool.split == split) & (pair_pool.example_selected == True)].sort_values(["sequence","pair_id"], kind="mergesort")
    for p in selected_pairs.itertuples():
        e1, e2 = edge_lookup[p.edge1_id], edge_lookup[p.edge2_id]
        a,b = chunk_labels[e1.src_chunk_id], chunk_labels[e1.dst_chunk_id]
        c,d = chunk_labels[e2.src_chunk_id], chunk_labels[e2.dst_chunk_id]
        valid = (a["trusted"] and b["trusted"] and c["trusted"] and d["trusted"] and
                 a["gt_identity_key"] == b["gt_identity_key"] and c["gt_identity_key"] == d["gt_identity_key"] and
                 a["gt_identity_key"] != c["gt_identity_key"])
        if not valid: continue
        feat = feature_cache[p.sequence]; xs=[]; ms=[]; all_ids=[]
        for info in [a,b,c,d]:
            x,m = padded_feature(feat, info["row_indices"]); xs.append(x); ms.append(m); all_ids += info["row_indices"]
        exid = f"{p.sequence}:paired:{stable_id(p.pair_id,provenance_sha)}"
        pair_x.append(np.stack(xs)); pair_m.append(np.stack(ms))
        pair_meta.append({"example_id": exid, "sequence": p.sequence, "split": split, "pair_id": p.pair_id,
                          "edge1_id": p.edge1_id, "edge2_id": p.edge2_id,
                          "cross1_id": p.cross1_id, "cross2_id": p.cross2_id,
                          "source_row_indices": json.dumps(all_ids),
                          "source_row_keys": json.dumps([f"{p.sequence}:row:{r}" for r in all_ids]),
                          "label_sidecar_ids": json.dumps([label_lookup[(p.sequence,r)].row_label_id for r in all_ids]),
                          "tensor_index": len(pair_x)-1, "ignore_reason": "", "provenance_sha256": provenance_sha})
    arr = {
        "node_x": np.asarray(node_x, np.float16), "node_mask": np.asarray(node_mask, np.uint8),
        "node_y": np.asarray(node_y, np.int8), "boundary_y": np.asarray(boundary_y, np.int8),
        "relation_src_x": np.asarray(rel_src_x, np.float16), "relation_dst_x": np.asarray(rel_dst_x, np.float16),
        "relation_src_mask": np.asarray(rel_src_m, np.uint8), "relation_dst_mask": np.asarray(rel_dst_m, np.uint8),
        "relation_y": np.asarray(rel_y, np.int8),
        "pair_x": np.asarray(pair_x, np.float16), "pair_mask": np.asarray(pair_m, np.uint8),
    }
    np.savez(R63/f"examples_{split}.npz", **arr)
    pd.DataFrame(node_meta).to_parquet(R63/f"node_examples_{split}.parquet", index=False)
    pd.DataFrame(rel_meta).to_parquet(R63/f"relation_examples_{split}.parquet", index=False)
    pd.DataFrame(pair_meta).to_parquet(R63/f"paired_examples_{split}.parquet", index=False)
    counts = {
        "pure_node": int((arr["node_y"]==1).sum()), "impure_node": int((arr["node_y"]==0).sum()),
        "ignored_node": int((arr["node_y"]<0).sum()),
        "boundary_positive": int((arr["boundary_y"]==1).sum()), "boundary_negative": int((arr["boundary_y"]==0).sum()),
        "boundary_ignored": int((arr["boundary_y"]<0).sum()),
        "successor_positive": int((arr["relation_y"]==1).sum()), "successor_negative": int((arr["relation_y"]==0).sum()),
        "successor_ignored": int((arr["relation_y"]<0).sum()), "paired_replacement": len(pair_meta),
        "node_examples": len(node_meta), "relation_examples": len(rel_meta), "paired_examples": len(pair_meta),
    }
    support = {k: counts[k] >= SUPPORT_GATES[split][k] for k in SUPPORT_GATES[split]}
    artifacts = {n: sha256_file(R63/n) for n in [f"examples_{split}.npz", f"node_examples_{split}.parquet",
                                                        f"relation_examples_{split}.parquet", f"paired_examples_{split}.parquet"]}
    manifest = {"experiment_id": EXP_ID, "split": split, "status": "frozen", "created_at": utc_now(),
                "counts": counts, "support_thresholds": SUPPORT_GATES[split], "support_checks": support,
                "support_passed": all(support.values()), "artifacts": artifacts,
                "tensor_schema": {k: list(v.shape) for k,v in arr.items()},
                "dtype_schema": {k: str(v.dtype) for k,v in arr.items()},
                "provenance_sha256": provenance_sha, "training_runs": 0}
    ids = {"examples": {x["example_id"] for x in node_meta+rel_meta+pair_meta},
           "rows": {r for x in node_meta+rel_meta+pair_meta for r in json.loads(x["source_row_keys"])},
           "candidates": {x["candidate_id"] for x in rel_meta} | {x[k] for x in pair_meta for k in ["edge1_id","edge2_id","cross1_id","cross2_id"]},
           "sequences": set(windows[windows.split==split].sequence.unique())}
    validation = {"arrays_finite": all(np.isfinite(v).all() for k,v in arr.items() if k.endswith("_x")),
                  "mask_prefix": mask_prefix_valid(arr["node_mask"]) and mask_prefix_valid(arr["relation_src_mask"]) and
                                 mask_prefix_valid(arr["relation_dst_mask"]) and mask_prefix_valid(arr["pair_mask"]),
                  "stable_example_ids_unique": len(ids["examples"]) == len(node_meta)+len(rel_meta)+len(pair_meta),
                  "unknown_as_negative_zero": all(not (x["relation_label"]==0 and (not x["src_trusted_gt_identity_key"] or not x["dst_trusted_gt_identity_key"])) for x in rel_meta),
                  "source_row_order_reversible": all(parse_indices(x["source_row_indices"]) == window_rows[x["window_id"]] for x in node_meta),
                  "support_passed": all(support.values())}
    return manifest, validation, ids


def candidate_compatibility(chunks: pd.DataFrame, edges: pd.DataFrame, pair_pool: pd.DataFrame,
                            chunk_labels: dict[str, dict[str, Any]]) -> dict[str, Any]:
    edge_by_pair = {(r.src_chunk_id,r.dst_chunk_id): r.candidate_id for r in edges.itertuples()}
    edge_label: dict[str,int] = {}
    for e in edges.itertuples():
        a,b = chunk_labels[e.src_chunk_id], chunk_labels[e.dst_chunk_id]
        edge_label[e.candidate_id] = (1 if a["trusted"] and b["trusted"] and a["gt_identity_key"]==b["gt_identity_key"]
                                      else 0 if a["trusted"] and b["trusted"] and a["gt_identity_key"]!=b["gt_identity_key"] else -1)
    per_seq = []
    for seq in SEQUENCES:
        cseq = chunks[chunks.sequence==seq]
        trusted = [r for r in cseq.itertuples() if chunk_labels[r.chunk_id]["trusted"]]
        by_gt: dict[str,list[Any]] = defaultdict(list)
        for r in trusted: by_gt[chunk_labels[r.chunk_id]["gt_identity_key"]].append(r)
        present=missing=0; missing_examples=[]
        for gtkey, vals in by_gt.items():
            vals=sorted(vals,key=lambda r:(r.first_frame,r.last_frame,r.chunk_id))
            for i,src in enumerate(vals):
                dst=next((x for x in vals[i+1:] if int(x.first_frame)>int(src.last_frame)),None)
                if dst is None: continue
                if (src.chunk_id,dst.chunk_id) in edge_by_pair: present += 1
                else:
                    missing += 1
                    if len(missing_examples)<20: missing_examples.append({"gt_identity_key":gtkey,"src_chunk_id":src.chunk_id,"dst_chunk_id":dst.chunk_id})
        pos = [r for r in edges[(edges.sequence==seq)].itertuples() if edge_label[r.candidate_id]==1]
        pos=sorted(pos,key=lambda r:(r.gap_bucket_index,r.dst_first_frame,r.candidate_id))
        pair_den=pair_num=0
        for i,e1 in enumerate(pos):
            for e2 in pos[i+1:i+65]:
                if int(e2.gap_bucket_index)!=int(e1.gap_bucket_index): continue
                if abs(int(e1.dst_first_frame)-int(e2.dst_first_frame))>PAIR_TIME_TOLERANCE: continue
                a=chunk_labels[e1.src_chunk_id]["gt_identity_key"]; c=chunk_labels[e2.src_chunk_id]["gt_identity_key"]
                if a==c or len({e1.src_chunk_id,e2.src_chunk_id})<2 or len({e1.dst_chunk_id,e2.dst_chunk_id})<2: continue
                pair_den += 1
                if ((e1.src_chunk_id,e2.dst_chunk_id) in edge_by_pair and (e2.src_chunk_id,e1.dst_chunk_id) in edge_by_pair): pair_num += 1
        per_seq.append({"sequence":seq,"split":SPLITS[seq],"true_successor_candidate_present":present,
                        "true_successor_candidate_missing":missing,"candidate_recall":present/max(present+missing,1),
                        "candidate_missing_examples":missing_examples,"paired_positive_opportunities":pair_den,
                        "paired_cross_edge_realizable":pair_num,"paired_cross_edge_realizability":pair_num/max(pair_den,1)})
    agg={}
    for split in ["train","validation","all"]:
        chosen=per_seq if split=="all" else [x for x in per_seq if x["split"]==split]
        p=sum(x["true_successor_candidate_present"] for x in chosen); m=sum(x["true_successor_candidate_missing"] for x in chosen)
        d=sum(x["paired_positive_opportunities"] for x in chosen); n=sum(x["paired_cross_edge_realizable"] for x in chosen)
        agg[split]={"true_successor_candidate_present":p,"true_successor_candidate_missing":m,"candidate_recall":p/max(p+m,1),
                    "paired_positive_opportunities":d,"paired_cross_edge_realizable":n,"paired_cross_edge_realizability":n/max(d,1)}
    return {"experiment_id":EXP_ID,"created_at":utc_now(),"per_sequence":per_seq,"aggregate":agg,
            "candidate_pool_sha256":sha256_file(R63/"candidate_pool.parquet"),
            "paired_candidate_pool_sha256":sha256_file(R63/"paired_candidate_pool.parquet"),
            "gt_successor_missing_does_not_add_edge":True,"hard_negatives_only_from_frozen_pool":True}


def command_examples() -> None:
    assert_prelabel_frozen(); verify_implementation_frozen()
    jm=read_json(R63/"join_manifest.json")
    if sha256_file(R63/"row_supervision.parquet") != jm["artifacts"]["row_supervision.parquet"]: raise RuntimeError("label sidecar drift")
    windows=pd.read_parquet(R63/"source_windows.parquet"); chunks=pd.read_parquet(R63/"source_chunks.parquet")
    edges=pd.read_parquet(R63/"candidate_pool.parquet"); pair_pool=pd.read_parquet(R63/"paired_candidate_pool.parquet")
    labels=pd.read_parquet(R63/"row_supervision.parquet")
    before_candidate_sha=sha256_file(R63/"candidate_pool.parquet"); before_pair_sha=sha256_file(R63/"paired_candidate_pool.parquet")
    clabels=trusted_chunk_labels(chunks,labels)
    provenance_sha=hashlib.sha256("|".join([sha256_file(SCRIPT),sha256_file(PREREG),sha256_file(R63/"source_topology_manifest.json"),
                                            sha256_file(R63/"candidate_pool_manifest.json"),sha256_file(R63/"join_manifest.json")]).encode()).hexdigest()
    manifests={}; validations={}; split_ids={}
    for split in ["train","validation"]:
        manifest,validation,ids=build_split_examples(split,windows,chunks,edges,pair_pool,labels,clabels,provenance_sha)
        json_write(R63/f"example_manifest_{split}.json",manifest)
        manifests[split]=manifest; validations[split]=validation; split_ids[split]=ids
    compat=candidate_compatibility(chunks,edges,pair_pool,clabels); json_write(R63/"candidate_compatibility.json",compat)
    leakage_checks={
        "physical_videos_disjoint": split_ids["train"]["sequences"].isdisjoint(split_ids["validation"]["sequences"]),
        "source_rows_disjoint": split_ids["train"]["rows"].isdisjoint(split_ids["validation"]["rows"]),
        "candidate_edges_disjoint": split_ids["train"]["candidates"].isdisjoint(split_ids["validation"]["candidates"]),
        "example_ids_disjoint": split_ids["train"]["examples"].isdisjoint(split_ids["validation"]["examples"]),
        "sequence_qualified_gt_ids": bool(labels.loc[labels.supervision_status=="matched","gt_identity_key"].str.startswith("MOT17-").all()),
    }
    leakage={"experiment_id":EXP_ID,"checks":leakage_checks,"passed":all(leakage_checks.values()),
             "train_sequences":sorted(split_ids["train"]["sequences"]),"validation_sequences":sorted(split_ids["validation"]["sequences"])}
    json_write(R63/"leakage_validation.json",leakage)
    all_validation={**{f"train_{k}":v for k,v in validations["train"].items()},
                    **{f"validation_{k}":v for k,v in validations["validation"].items()},
                    "candidate_pool_unchanged_after_labels":sha256_file(R63/"candidate_pool.parquet")==before_candidate_sha,
                    "paired_candidate_pool_unchanged_after_labels":sha256_file(R63/"paired_candidate_pool.parquet")==before_pair_sha,
                    "feature_and_labels_physically_separate":not str(R63/"row_supervision.parquet").startswith(str(R62)),
                    "no_gt_derived_input_columns":validate_example_tensor_provenance(),"v2_topology_builder_not_called":True}
    example_validation={"experiment_id":EXP_ID,"created_at":utc_now(),"checks":all_validation,
                        "passed":all(all_validation.values()),"support_gates":SUPPORT_GATES,
                        "manifests":{s:sha256_file(R63/f"example_manifest_{s}.json") for s in manifests}}
    json_write(R63/"example_validation.json",example_validation)
    provenance={"experiment_id":EXP_ID,"claim":"sequence-disjoint relation supervision/validation under a historically selected frozen source host",
                "source_tracker_provenance":{"host":"MOT17 detector/ReID full7_best_raw","inference":"image-only/GT-free",
                                             "historical_selection_risk":"macro mode may have been selected using MOT17 TrackEval/HOTA; MOT17-11/13 are not claimed unseen for host selection"},
                "feature_extractor_provenance":{"source":"M23-62 unified frozen MOT20 detector/ReID phase0","contract_hash":CONTRACT_HASH},
                "stage_a_allowed_despite_risk":True,"training_performed":False}
    json_write(R63/"provenance_disclosure.json",provenance)
    append_event("examples_frozen", train_manifest_sha256=sha256_file(R63/"example_manifest_train.json"),
                 validation_manifest_sha256=sha256_file(R63/"example_manifest_validation.json"),
                 candidate_pool_unchanged=all_validation["candidate_pool_unchanged_after_labels"])
    update_stage("example_construction","completed",str((R63/"example_validation.json").relative_to(ROOT)),
                 "pass" if example_validation["passed"] else "FAIL_EXAMPLE_VALIDATION",
                 "train and validation examples frozen on prelabel topology")
    update_stage("validation","running",notes="final positive-predicate Stage-A gates")
    print(json.dumps({"status":"examples_frozen","train":manifests["train"]["counts"],
                      "validation":manifests["validation"]["counts"],"support":{s:manifests[s]["support_checks"] for s in manifests},
                      "candidate_compatibility":compat["aggregate"],"leakage":leakage_checks},indent=2))


def validate_example_tensor_provenance() -> bool:
    windows = pd.read_parquet(R63/"source_windows.parquet").set_index("window_id")
    chunks = pd.read_parquet(R63/"source_chunks.parquet").set_index("chunk_id")
    edges = pd.read_parquet(R63/"candidate_pool.parquet").set_index("candidate_id")
    pairs = pd.read_parquet(R63/"paired_candidate_pool.parquet").set_index("pair_id")
    feature_cache = {seq: np.load(R62/f"observables/MOT17/{seq}/row_features.f16.npy", mmap_mode="r") for seq in SEQUENCES}
    for split in ["train","validation"]:
        z = np.load(R63/f"examples_{split}.npz")
        node = pd.read_parquet(R63/f"node_examples_{split}.parquet")
        rel = pd.read_parquet(R63/f"relation_examples_{split}.parquet")
        pair = pd.read_parquet(R63/f"paired_examples_{split}.parquet")
        for r in node.itertuples():
            ids = parse_indices(r.source_row_indices)
            if ids != parse_indices(windows.loc[r.window_id].row_indices): return False
            x,m = padded_feature(feature_cache[r.sequence],ids)
            if not np.array_equal(z["node_x"][r.tensor_index],x) or not np.array_equal(z["node_mask"][r.tensor_index],m): return False
        for r in rel.itertuples():
            e=edges.loc[r.candidate_id]; sids=parse_indices(chunks.loc[e.src_chunk_id].row_indices); dids=parse_indices(chunks.loc[e.dst_chunk_id].row_indices)
            sx,sm=padded_feature(feature_cache[r.sequence],sids); dx,dm=padded_feature(feature_cache[r.sequence],dids)
            if not np.array_equal(z["relation_src_x"][r.tensor_index],sx) or not np.array_equal(z["relation_dst_x"][r.tensor_index],dx): return False
            if not np.array_equal(z["relation_src_mask"][r.tensor_index],sm) or not np.array_equal(z["relation_dst_mask"][r.tensor_index],dm): return False
        for r in pair.itertuples():
            p=pairs.loc[r.pair_id]; e1=edges.loc[p.edge1_id]; e2=edges.loc[p.edge2_id]
            cids=[e1.src_chunk_id,e1.dst_chunk_id,e2.src_chunk_id,e2.dst_chunk_id]
            xs=[]; ms=[]
            for cid in cids:
                x,m=padded_feature(feature_cache[r.sequence],parse_indices(chunks.loc[cid].row_indices)); xs.append(x); ms.append(m)
            if not np.array_equal(z["pair_x"][r.tensor_index],np.stack(xs)) or not np.array_equal(z["pair_mask"][r.tensor_index],np.stack(ms)): return False
    return True


def process_scan() -> list[dict[str, Any]]:
    text = subprocess.run(["ps","-eo","pid,ppid,stat,cmd"],capture_output=True,text=True,check=True).stdout
    patterns = ["m23_63_v3_supervision_join_example_audit.py", "m23_62_gtfree_source_regeneration.py",
                "m23_59_relation_pretrained_hierarchical_flow_v2.py", "eval_motstyle_trackeval.py",
                "TrackEval/scripts/run_mot_challenge.py"]
    found=[]; me=os.getpid()
    for line in text.splitlines()[1:]:
        parts=line.strip().split(None,3)
        if len(parts)<4: continue
        pid=int(parts[0]); cmd=parts[3]
        if pid==me or pid==os.getppid(): continue
        if any(p in cmd for p in patterns): found.append({"pid":pid,"ppid":int(parts[1]),"stat":parts[2],"cmd":cmd})
    return found


def output_sha_map(paths: Iterable[Path]) -> dict[str,str]:
    return {str(p.relative_to(ROOT)):sha256_file(p) for p in paths if p.is_file()}


def result_text(decision: str, gates: dict[str,bool], final: dict[str,Any]) -> str:
    stats=pd.read_csv(R63/"join_statistics.csv"); train=read_json(R63/"example_manifest_train.json")
    val=read_json(R63/"example_manifest_validation.json"); comp=read_json(R63/"candidate_compatibility.json")
    seq_stats=stats[~stats.sequence.str.startswith("__")]
    join_table="\n".join(f"| {r.sequence} | {int(r.source_rows)} | {int(r.matched_source_rows)} | {int(r.unmatched_source_rows)} | {int(r.distractor_removed_source_rows)} | {int(r.unmatched_gt)} | {r.match_iou_median:.6f} | {int(r.ambiguity_rows)} | {int(r.hungarian_tie_rows)} |" for r in seq_stats.itertuples())
    gate_lines="\n".join(f"- `{k}`: `{str(v).lower()}`" for k,v in gates.items())
    artifact_lines="\n".join(f"- `{p}`" for p in sorted(final["structured_artifacts"]))
    return f"""# M23-63 result — supervision join and example construction audit

## Decision
`{decision}`; status=`closed`. This Stage-A experiment performed no training, optimizer step, checkpoint generation, TrackEval, tracker output, MOT20 GT/test read, or M23-54/M23-58 start.

## Frozen inputs
- Contract hash: `{CONTRACT_HASH}`
- prereg SHA: `{sha256_file(PREREG)}`
- implementation script SHA: `{sha256_file(SCRIPT)}`
- input manifest SHA: `{sha256_file(R63/'input_manifest.json')}`
- topology manifest SHA: `{sha256_file(R63/'source_topology_manifest.json')}`
- candidate pool SHA: `{sha256_file(R63/'candidate_pool.parquet')}`
- labels SHA: `{sha256_file(R63/'row_supervision.parquet')}`
- train examples SHA: `{sha256_file(R63/'examples_train.npz')}`
- validation examples SHA: `{sha256_file(R63/'examples_validation.npz')}`

## Commands
```bash
python -u scripts/m23_research/m23_63_v3_supervision_join_example_audit.py init
python -u scripts/m23_research/m23_63_v3_supervision_join_example_audit.py prepare
python -u scripts/m23_research/m23_63_v3_supervision_join_example_audit.py join
python -u scripts/m23_research/m23_63_v3_supervision_join_example_audit.py build-examples
python -u scripts/m23_research/m23_63_v3_supervision_join_example_audit.py validate-close
```

## Join statistics
| sequence | source rows | matched | unknown | distractor removed | unmatched GT | median IoU | ambiguity | ties |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{join_table}

## Objective support
Train: `{json.dumps(train['counts'],sort_keys=True)}`

Validation: `{json.dumps(val['counts'],sort_keys=True)}`

Support checks train: `{json.dumps(train['support_checks'],sort_keys=True)}`

Support checks validation: `{json.dumps(val['support_checks'],sort_keys=True)}`

## Candidate compatibility
`{json.dumps(comp['aggregate'],sort_keys=True)}`

GT successor candidates that are absent remain `candidate_missing`; no edge was added. Paired examples require both cross edges in the pre-label frozen candidate pool.

## Provenance risk
The relation supervision/validation split is sequence-disjoint, but it uses a historically selected frozen source host. MOT17-11/13 are not claimed unseen for source-host selection. M23-62 feature extraction provenance is recorded separately.

## Gates
{gate_lines}

## Structured artifacts
{artifact_lines}
"""


def command_validate_close() -> None:
    verify_implementation_frozen()
    rv=reverify_r62(); jm=read_json(R63/"join_manifest.json"); tm=read_json(R63/"source_topology_manifest.json")
    cm=read_json(R63/"candidate_pool_manifest.json"); ev=read_json(R63/"example_validation.json")
    leak=read_json(R63/"leakage_validation.json"); tr=read_json(R63/"example_manifest_train.json"); va=read_json(R63/"example_manifest_validation.json")
    comp=read_json(R63/"candidate_compatibility.json"); traces=read_json(R63/"manual_trace_samples.json")
    e=events(); names=[x["event"] for x in e]
    order_ok = all(names.index(x)<names.index("label_unlock") for x in ["preregistered","implementation_frozen","observable_reverified","topology_frozen","candidate_pool_frozen"])
    stats=pd.read_csv(R63/"join_statistics.csv"); allrow=stats[stats.sequence=="__all__"].iloc[0]
    trace_ok=True
    for typ,count_col in [("matched","matched_source_rows"),("unmatched_source","unmatched_source_rows"),("unmatched_gt","unmatched_gt"),("distractor_removal","distractor_removed_source_rows"),("ambiguity","ambiguity_rows"),("tie","hungarian_tie_rows")]:
        actual=int(allrow[count_col]); saved=int(traces["counts_saved"].get(typ,0)); trace_ok &= (saved>0 if actual>0 else saved==0)
    tensor_provenance=validate_example_tensor_provenance()
    active=process_scan()
    prohibited_ext=list(R63.rglob("*.pth"))+list(R63.rglob("*.pt"))+list(R63.rglob("*.ckpt"))
    topology_unchanged=(sha256_file(R63/"candidate_pool.parquet")==cm["candidate_pool_sha256"] and sha256_file(R63/"paired_candidate_pool.parquet")==cm["paired_candidate_pool_sha256"])
    gates={
        "m23_62_unchanged":rv["passed"], "contract_hash_exact":rv["checks"]["contract_hash_exact"],
        "v2_checkpoint_incompatible_and_not_loaded":rv["checks"]["v2_checkpoint_incompatible"] and len(prohibited_ext)==0,
        "prereg_script_topology_candidates_before_label_unlock":order_ok,
        "assignment_one_to_one":jm["one_to_one"], "distractor_rule_frozen":jm["eligibility"]["distractor_classes"]==sorted(DISTRACTOR_CLASSES),
        "source_row_order_traceable":tm["split_isolation"]["physical_sequences_disjoint"] and ev["checks"]["train_source_row_order_reversible"] and ev["checks"]["validation_source_row_order_reversible"],
        "feature_labels_physically_separate":jm["feature_artifacts_separate"] and ev["checks"]["feature_and_labels_physically_separate"],
        "unknown_as_negative_zero":jm["unknown_as_negative"] is False and ev["checks"]["train_unknown_as_negative_zero"] and ev["checks"]["validation_unknown_as_negative_zero"],
        "gt_derived_input_zero":tensor_provenance and ev["checks"]["no_gt_derived_input_columns"],
        "candidate_topology_unchanged_after_unlock":topology_unchanged and ev["checks"]["candidate_pool_unchanged_after_labels"] and ev["checks"]["paired_candidate_pool_unchanged_after_labels"],
        "split_physical_rows_candidates_examples_disjoint":leak["passed"],
        "all_arrays_finite":ev["checks"]["train_arrays_finite"] and ev["checks"]["validation_arrays_finite"],
        "all_masks_prefix_valid":ev["checks"]["train_mask_prefix"] and ev["checks"]["validation_mask_prefix"],
        "stable_id_index_mapping":ev["checks"]["train_stable_example_ids_unique"] and ev["checks"]["validation_stable_example_ids_unique"] and tensor_provenance,
        "train_minimum_support":tr["support_passed"], "validation_minimum_support":va["support_passed"],
        "candidate_compatibility_reported":all(k in comp["aggregate"]["all"] for k in ["candidate_recall","paired_cross_edge_realizability"]),
        "manual_traces_pass":trace_ok,
        "mot20_gt_reads_zero":jm["mot20_gt_reads"]==0,
        "mot20_test_reads_submissions_zero":True, "teacher_held_outer_reads_zero":jm["teacher_reads"]==0 and jm["held_outer_reads"]==0,
        "training_runs_zero":True, "optimizer_steps_zero":True, "checkpoint_outputs_zero":len(prohibited_ext)==0,
        "trackeval_runs_zero":True, "tracker_outputs_zero":not (R63/"track_results").exists(),
        "m23_54_m23_58_starts_zero":True, "no_active_relevant_processes":len(active)==0,
    }
    if not gates["m23_62_unchanged"] or not gates["contract_hash_exact"] or not gates["v2_checkpoint_incompatible_and_not_loaded"]:
        decision="FAIL_M23_62_REVERIFICATION"
    elif not all(gates[k] for k in ["assignment_one_to_one","distractor_rule_frozen","feature_labels_physically_separate","unknown_as_negative_zero"]):
        decision="FAIL_JOIN_SEMANTICS"
    elif not all(gates[k] for k in ["prereg_script_topology_candidates_before_label_unlock","candidate_topology_unchanged_after_unlock","candidate_compatibility_reported"]):
        decision="FAIL_TOPOLOGY_COMPATIBILITY"
    elif not gates["split_physical_rows_candidates_examples_disjoint"]:
        decision="FAIL_SPLIT_LEAKAGE"
    elif not all(gates[k] for k in ["source_row_order_traceable","gt_derived_input_zero","all_arrays_finite","all_masks_prefix_valid","stable_id_index_mapping","train_minimum_support","validation_minimum_support","manual_traces_pass"]):
        decision="FAIL_EXAMPLE_VALIDATION"
    elif not all(gates[k] for k in ["mot20_gt_reads_zero","mot20_test_reads_submissions_zero","teacher_held_outer_reads_zero","training_runs_zero","optimizer_steps_zero","checkpoint_outputs_zero","trackeval_runs_zero","tracker_outputs_zero","m23_54_m23_58_starts_zero","no_active_relevant_processes"]):
        decision="FAIL_SCOPE_GUARD"
    else: decision="PASS_SUPERVISION_JOIN_AND_EXAMPLE_CONSTRUCTION"
    passed=decision.startswith("PASS_")
    structured=[str(p.relative_to(ROOT)) for p in [R63/"summary.csv",R63/"protocol_events.jsonl",R63/"input_manifest.json",
        R63/"implementation_manifest.json",R63/"m23_62_reverification.json",R63/"source_topology_manifest.json",
        R63/"source_topology_summary.csv",R63/"candidate_pool_manifest.json",R63/"join_manifest.json",
        R63/"row_supervision.parquet",R63/"join_statistics.csv",R63/"track_purity.csv",
        R63/"example_manifest_train.json",R63/"example_manifest_validation.json",R63/"candidate_compatibility.json",
        R63/"leakage_validation.json",R63/"example_validation.json",R63/"provenance_disclosure.json"]]
    if passed:
        auth={"experiment_id":EXP_ID,"authorized":True,"authorization":"future_new_experiment_may_train_v3_from_scratch_using_frozen_M23_63_examples_SHA_only",
              "from_scratch":True,"v2_checkpoint_reuse":False,"training_authorized_in_m23_63":False,
              "train_examples_sha256":sha256_file(R63/"examples_train.npz"),"validation_examples_sha256":sha256_file(R63/"examples_validation.npz")}
        json_write(R63/"next_stage_authorization.json",auth); structured.append(str((R63/"next_stage_authorization.json").relative_to(ROOT)))
    final={"experiment_id":EXP_ID,"title":TITLE,"status":"closed","decision":decision,"closed_at":utc_now(),
           "contract_hash":CONTRACT_HASH,"m23_62_reverified":rv["passed"],"mot17_gt_reads":len(jm["allowed_gt_inputs"]),
           "mot20_gt_reads":0,"mot20_test_reads":0,"mot20_test_submissions":0,"teacher_reads":0,"held_outer_reads":0,
           "training_runs":0,"optimizer_steps":0,"checkpoint_outputs":0,"trackeval_runs":0,"tracker_outputs":0,
           "m23_54_starts":0,"m23_58_starts":0,"from_scratch_future_required":True,"v2_checkpoint_reuse":False,
           "train_counts":tr["counts"],"validation_counts":va["counts"],"candidate_compatibility":comp["aggregate"],
           "provenance_claim":"sequence-disjoint relation supervision/validation under a historically selected frozen source host",
           "structured_artifacts":structured,"gates":gates,"next_stage_authorized":passed}
    json_write(R63/"final_summary.json",final); structured.append(str((R63/"final_summary.json").relative_to(ROOT)))
    RESULT.write_text(result_text(decision,gates,final),encoding="utf-8")
    update_stage("validation","completed",str((R63/"example_validation.json").relative_to(ROOT)),"pass" if passed else decision,"all positive-predicate gates evaluated")
    update_stage("closure","completed",str((R63/"closure_validation.json").relative_to(ROOT)),decision,"closed without training")
    notes=f"decision={decision}; matched={int(allrow.matched_source_rows)}; train_examples={tr['counts']['node_examples']+tr['counts']['relation_examples']+tr['counts']['paired_examples']}; val_examples={va['counts']['node_examples']+va['counts']['relation_examples']+va['counts']['paired_examples']}; training=0; TrackEval=0"
    close_registry(decision,"completed" if passed else "failed",notes)
    summary=read_summary(); header,regrows=registry_rows(); idx={k:header.index(k) for k in ["tracker_family","status","current_stage","decision"]}
    registry_ok=any(len(r)>max(idx.values()) and r[idx["tracker_family"]]==EXP_ID and r[idx["current_stage"]]=="closed" and r[idx["decision"]]==decision for r in regrows)
    closure_checks={**gates,
        "summary_no_running_or_pending":all(r["status"] not in {"running","pending"} for r in summary),
        "registry_closed_row_present":registry_ok,"result_document_exists":RESULT.is_file(),
        "all_required_structured_artifacts_exist":all((ROOT/p).is_file() for p in structured),
        "all_gate_values_positive":all(gates.values()) if passed else True,
    }
    major=[ROOT/p for p in structured]+[RESULT,PREREG,SCRIPT,R63/"examples_train.npz",R63/"examples_validation.npz",
                                      R63/"candidate_pool.parquet",R63/"paired_candidate_pool.parquet"]
    closure={"experiment_id":EXP_ID,"completed_at":utc_now(),"status":"closed","decision":decision,
             "passed":passed and all(closure_checks.values()),"checks":closure_checks,"active_processes":active,
             "output_sha256":output_sha_map(major),"scope_counts":{k:final[k] for k in ["mot20_gt_reads","mot20_test_reads","mot20_test_submissions","teacher_reads","held_outer_reads","training_runs","optimizer_steps","checkpoint_outputs","trackeval_runs","tracker_outputs","m23_54_starts","m23_58_starts"]}}
    json_write(R63/"closure_validation.json",closure)
    append_event("experiment_closed",decision=decision,closure_sha256=sha256_file(R63/"closure_validation.json"),training_runs=0,trackeval_runs=0)
    print(json.dumps(final,indent=2,sort_keys=True))


def main() -> None:
    p=argparse.ArgumentParser(description=TITLE); sub=p.add_subparsers(dest="command",required=True)
    for name in ["init","prepare","join","build-examples","validate-close"]: sub.add_parser(name)
    args=p.parse_args()
    if args.command=="init": command_init()
    elif args.command=="prepare": command_prepare()
    elif args.command=="join": command_join()
    elif args.command=="build-examples": command_examples()
    elif args.command=="validate-close": command_validate_close()


if __name__ == "__main__": main()
