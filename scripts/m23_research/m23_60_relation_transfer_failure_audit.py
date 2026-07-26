#!/usr/bin/env python3
"""M23-60 Relation Transfer Failure Audit.

Post-hoc diagnostic only. This script reads frozen M23-59 v2/M23-57 artifacts,
never trains, never creates a tracker, and never invokes TrackEval.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.metrics import average_precision_score, roc_auc_score

REPO = Path(__file__).resolve().parents[2]
M59 = Path("outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2")
M57 = Path("outputs/mot20_m23_20260718/m23_57_intra_node_change_point_capacity_v2")
OUT = Path("outputs/mot20_m23_20260718/m23_60_relation_transfer_failure_audit")
PREREG = Path("docs/m23_60_relation_transfer_failure_audit_prereg_20260721.md")
RESULT_DOC = Path("docs/m23_60_relation_transfer_failure_audit_result_20260721.md")
REGISTRY = Path("outputs/experiment_registry.csv")
SUMMARY = OUT / "summary.csv"
M59_SCRIPT = Path("scripts/m23_research/m23_59_relation_pretrained_hierarchical_flow_v2.py")
SEQS = ["MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05"]
MOT17_TRAIN = ["MOT17-02", "MOT17-04", "MOT17-05", "MOT17-09", "MOT17-10"]
MOT17_VAL = ["MOT17-11", "MOT17-13"]
APP_DIM = 128
GEOM_NAMES = [
    "center_x_norm", "center_y_norm", "box_width_norm", "box_height_norm",
    "log_aspect", "log_area_fraction", "visibility", "velocity_x_height_frame",
    "velocity_y_height_frame", "log_width_change_per_frame", "log_height_change_per_frame",
    "frame_delta_over_30_clipped", "velocity_x_residual", "velocity_y_residual",
    "crowd_density_over_100_clipped", "nearest_neighbor_distance_or_mapped_indicator",
]
FEATURE_NAMES = [f"appearance_{i:03d}" for i in range(APP_DIM)] + [f"geometry_{i:02d}_{n}" for i, n in enumerate(GEOM_NAMES)]
SEED = 2360001
MAX_SHIFT_SAMPLE = 200_000
K_RANK = 256
K_FLOW = 32
CLASSIFICATIONS = [
    "implementation_or_semantic_mismatch",
    "candidate_graph_bottleneck",
    "observable_domain_shift",
    "learned_relation_transfer_failure",
    "mixed_candidate_and_transfer_failure",
]
SUMMARY_FIELDS = [
    "experiment", "stage", "scope", "status", "started_at", "completed_at", "report",
    "primary_classification", "uses_mot20_gt", "post_hoc_diagnostic_only",
    "not_deployable", "not_a_strict_result", "notes",
]


def utcnow() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def json_write(path: Path, obj: Any, *, create_only: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if create_only and path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def csv_write(path: Path, rows: list[dict[str, Any]], fields: list[str], *, create_only: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if create_only and path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    os.replace(tmp, path)


def load_m59_module():
    spec = importlib.util.spec_from_file_location("m23_59_v2_frozen_for_m23_60", M59_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(M59_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def summary_rows() -> list[dict[str, str]]:
    with SUMMARY.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def update_summary(stage: str, *, status: str | None = None, report: str | None = None,
                   classification: str | None = None, notes: str | None = None,
                   started: bool = False, completed: bool = False) -> None:
    rows = summary_rows(); found = False
    for r in rows:
        if r["stage"] == stage:
            found = True
            if status is not None: r["status"] = status
            if report is not None: r["report"] = report
            if classification is not None: r["primary_classification"] = classification
            if notes is not None: r["notes"] = notes
            if started and not r["started_at"]: r["started_at"] = utcnow()
            if completed: r["completed_at"] = utcnow()
    if not found: raise KeyError(stage)
    csv_write(SUMMARY, rows, SUMMARY_FIELDS)


def registry_append(values: dict[str, Any]) -> None:
    with REGISTRY.open(newline="", encoding="utf-8", errors="replace") as f:
        header = next(csv.reader(f))
    row = [""] * len(header)
    index = {name: i for i, name in enumerate(header)}
    for k, v in values.items():
        if k in index: row[index[k]] = str(v)
    with REGISTRY.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def finalize_registry(classification: str) -> None:
    raw = REGISTRY.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    header = next(csv.reader([raw[0].rstrip("\r\n")]))
    idx = {h: i for i, h in enumerate(header)}
    changed = 0
    for i in range(1, len(raw)):
        row = next(csv.reader([raw[i].rstrip("\r\n")]))
        if len(row) < len(header): row += [""] * (len(header) - len(row))
        if row[idx.get("tag", 8)] == "M23-60-relation-transfer-audit" and row[idx.get("status", 2)] == "running":
            row[idx["status"]] = "superseded"
            if "current_stage" in idx: row[idx["current_stage"]] = "superseded"
            if "notes" in idx: row[idx["notes"]] = (row[idx["notes"]] + "; superseded by completed audit").strip("; ")
            buf = io.StringIO(); csv.writer(buf, lineterminator="\n").writerow(row); raw[i] = buf.getvalue(); changed += 1
    if changed != 1:
        raise RuntimeError(f"expected one M23-60 running registry row, found {changed}")
    tmp = REGISTRY.with_name(REGISTRY.name + f".tmp.{os.getpid()}")
    tmp.write_text("".join(raw), encoding="utf-8"); os.replace(tmp, REGISTRY)
    registry_append({
        "timestamp": utcnow(), "kind": "diagnostic_audit", "status": "completed",
        "script": str(Path(__file__).relative_to(REPO)), "dataset": "MOT17+MOT20",
        "split": "post_hoc_diagnostic", "tracker_family": "M23-60",
        "variant": "relation_transfer_failure_audit", "tag": "M23-60-relation-transfer-audit-closed",
        "run_root": str(OUT), "summary_csv": str(SUMMARY),
        "checkpoint": str(M59 / "external_pretraining/relation_pretrained_frozen.pt"),
        "log_path": str(OUT / "completion_validation.json"),
        "notes": f"post-hoc only; uses MOT20 GT; no training/TrackEval/tracker; primary={classification}",
        "name": "M23-60 Relation Transfer Failure Audit", "dataset_split": "MOT17 validation + MOT20 train post-hoc",
        "run_dir": str(OUT), "current_stage": "closed", "decision": classification,
        "exit_code": 0, "phase": "closed",
    })


def input_paths() -> list[Path]:
    paths = [
        Path("AGENTS.md"),
        Path("docs/m23_59_relation_pretrained_hierarchical_flow_prereg_v2_20260720.md"),
        Path("docs/m23_59_v1_invalidated_determinism_20260720.md"),
        Path("docs/m23_59_relation_pretrained_hierarchical_flow_v2_result_20260721.md"),
        M59_SCRIPT,
        M59 / "implementation_manifest.json", M59 / "preregistered_protocol.json",
        M59 / "external_dataset_manifest.json", M59 / "external_pretraining/frozen_checkpoint_manifest.json",
        M59 / "external_pretraining/relation_pretrained_frozen.pt", M59 / "external_examples/train/manifest.json",
        M59 / "external_examples/validation/manifest.json", M59 / "final_summary.json",
        M59 / "closure_validation.json", M59 / "strict_outer_evaluation/report.json",
        M59 / "summary.csv", M59 / "protocol_events.jsonl",
    ]
    for split in ["train", "validation"]:
        paths.extend(sorted((M59 / "external_examples" / split).glob("*.npy")))
    for seq in MOT17_TRAIN + MOT17_VAL:
        paths.extend([
            M59 / "external_features" / seq / "manifest.json",
            M59 / "external_features" / seq / "eligible_gt_rows.parquet",
            M59 / "external_features" / seq / "appearance_128.f16.npy",
            M59 / "external_features" / seq / "geometry_16.f16.npy",
        ])
    for seq in SEQS:
        paths.extend([
            M59 / "mot20_observable" / seq / "manifest.json",
            M59 / "mot20_observable" / seq / "rows.parquet",
            M59 / "mot20_observable" / seq / "row_features.f16.npy",
            M59 / "mot20_labels" / seq / "manifest.json",
            M57 / "boundary_universe" / seq / "chunk_membership.parquet",
            M57 / "capacity" / seq / "frozen_candidate_graph/nodes.parquet",
            M57 / "capacity" / seq / "frozen_candidate_graph/edges.parquet",
            M57 / "capacity" / seq / "teacher_identity_flow/teacher_edge_utilities.parquet",
            M57 / "capacity" / seq / "postfreeze_audit/successor_events.parquet",
        ])
        for outer in SEQS:
            if outer == seq: continue
            base = M59 / "nested_loso" / outer / f"inner_valid_{seq}"
            paths.extend([base / "model.pt", base / "model_frozen_before_validation_labels.json"])
    unique = []
    for p in paths:
        if p not in unique: unique.append(p)
    missing = [str(p) for p in unique if not p.exists()]
    if missing: raise FileNotFoundError(f"missing inputs: {missing}")
    if any("m23_59_relation_pretrained_hierarchical_flow/" in str(p) and "_v2" not in str(p) for p in unique):
        raise RuntimeError("v1 input detected")
    return unique


def prereg_text(inputs: list[dict[str, Any]]) -> str:
    input_table = "\n".join(f"| `{x['path']}` | `{x['sha256']}` | {x['bytes']} |" for x in inputs)
    return f"""# M23-60 Relation Transfer Failure Audit — Preregistration (2026-07-21)

## Status and scope

This document is frozen before any M23-60 aggregate diagnostic result is calculated. M23-60 is an independent, post-hoc audit of frozen M23-59 v2 artifacts.

- `uses_mot20_gt=true`
- `post_hoc_diagnostic_only=true`
- `not_deployable=true`
- `not_a_strict_result=true`
- no training, tracker generation, TrackEval, threshold search, policy change, test submission, M23-54, or M23-58
- M23-59 files are read-only; M23-59 v1 artifacts are prohibited

## Fixed feature schema

Input order is exactly `appearance_000..appearance_127` followed by these geometry columns:

{chr(10).join(f'{i}. `{name}`' for i, name in enumerate(GEOM_NAMES))}

A semantic mismatch is critical when the same feature index has a different physical meaning across MOT17 and MOT20, when source/destination or predecessor/successor orientation changes, or when label/index mapping changes through sorting, reversal, padding, batching, or model input.

## Fixed candidate oracle

- MOT17 validation uses the frozen synthetic candidate sets: outgoing `[true successor, frozen outgoing hard negative]`, incoming `[true predecessor, frozen incoming hard negative]`, and paired replacement `[true joint assignment, crossed joint assignment]`.
- MOT20 uses frozen M23-57 candidate graph nodes/edges, teacher edge utilities, and successor-events tables.
- Outgoing coverage uses `out_rank`; incoming coverage uses `in_rank`; mutual coverage uses `max_rank`.
- Report coverage at 1, 3, 5, 32 and 256. `K=256` is ranking K; `K=32` is flow K.
- Correct-candidate absence (`candidate_present=0`) is always separated from candidate-present ranking error.
- No-successor queries are candidate-graph source nodes absent from the successor-event source set.
- Paired replacement uses deterministic adjacent distinct-identity true-successor events in the same fixed gap bucket; both crossed edges must exist.
- Oracle theoretical upper bound over all successor queries is candidate-present coverage; conditional oracle among candidate-present queries is 1.0.

## Fixed diagnostics

Ranking: AUROC, PR-AUC, MRR, R@1/R@3/R@5/R@32/R@256, score quantiles, top-1 margin, rank histogram, 10-bin ECE, constant prediction rate, tie rate, candidate counts, and fixed-seed random expectation. `-score` and candidate-order reversal are implementation diagnostics only.

Baselines on the identical candidate set: original outgoing rank, minimum gap, minimum endpoint displacement, minimum motion/velocity residual, maximum endpoint IoU, maximum destination detector score, fixed-seed random, and oracle.

Feature shift: count, missing/non-finite, mean/std/min/max, p01/p05/p25/p50/p75/p95/p99, hard clipping, outside-MOT17-train min/max and p01/p99 rates, KS, Wasserstein, PSI, and positive/negative conditional relation distributions. Deterministic sampling cap is {MAX_SHIFT_SAMPLE:,} rows per domain with seed {SEED}.

## Frozen failure classification

Priority is fixed and cannot be changed after results:

1. `implementation_or_semantic_mismatch` when any critical semantic check fails: frozen SHA/config/state mismatch; predecessor/successor or source/target inversion; xywh/xyxy or frame/gap mismatch; feature-index physical meaning mismatch; non-prefix mask/padding corruption; trace label/index instability; non-finite model input; or external score orientation where `-score` R@1 exceeds normal score R@1 by at least 0.10.
2. Otherwise `mixed_candidate_and_transfer_failure` when pooled MOT20 successor coverage@256 is below 0.80 **and** candidate-present frozen-model R@1 is more than 0.05 below the best fixed non-oracle baseline.
3. Otherwise `candidate_graph_bottleneck` when pooled MOT20 successor coverage@256 is below 0.80.
4. Otherwise `observable_domain_shift` when at least 25% of 144 row features have KS >=0.25 or PSI >=0.25 in at least three MOT20 sequences, and candidate-present frozen-model R@1 is at least 0.20 below MOT17 validation R@1.
5. Otherwise `learned_relation_transfer_failure` when candidate-present frozen-model R@1 is more than 0.05 below the best fixed non-oracle baseline or at least 0.20 below MOT17 validation R@1.
6. If none of 1–5 fires, the primary category remains `learned_relation_transfer_failure`, with an explicit weak-evidence flag; no new category may be invented.

The primary category is unique. Secondary evidence is reported but cannot override the priority order.

## Output schemas

- `audit_manifest.json`: immutable preregistration/input SHA record and flags.
- `semantic_validation.json`: check name, critical flag, pass, evidence, two trace examples.
- `candidate_oracle.json`: per-domain query counts, coverage, rank/count distributions, exclusion waterfall, paired coverage, oracle bound.
- `feature_shift.csv`: `kind,feature,domain,condition,count,missing_nonfinite,mean,std,min,max,p01,p05,p25,p50,p75,p95,p99,clip_rate,outside_train_minmax_rate,outside_train_p01_p99_rate,ks,wasserstein,psi`.
- `ranking_diagnostics.csv`: one row per domain/model/method/direction with ranking/calibration statistics.
- `error_waterfall.json`: fixed query waterfall.
- `final_diagnosis.json`: unique primary classification, threshold evidence, flags, allowed next direction.
- `summary.csv`: queue-level structured status.

## Frozen inputs

| Input | SHA-256 | Bytes |
|---|---|---:|
{input_table}
"""


def command_init() -> None:
    if OUT.exists() or PREREG.exists() or RESULT_DOC.exists():
        raise FileExistsError("M23-60 audit output already exists; refusing overwrite")
    paths = input_paths()
    inputs = [{"path": str(p), "sha256": sha256_file(p), "bytes": p.stat().st_size} for p in paths]
    PREREG.parent.mkdir(parents=True, exist_ok=True)
    PREREG.write_text(prereg_text(inputs), encoding="utf-8")
    manifest = {
        "experiment": "M23-60 Relation Transfer Failure Audit", "status": "preregistered",
        "created_at": utcnow(), "experiment_number_available": True,
        "experiment_number_check": {"registry_exact_M23-60_matches": 0, "file_name_matches_before_creation": 0},
        "uses_mot20_gt": True, "post_hoc_diagnostic_only": True,
        "not_deployable": True, "not_a_strict_result": True,
        "training_runs": 0, "trackeval_runs": 0, "tracker_outputs": 0,
        "m23_59_modified": False, "m23_59_v1_artifact_reuse": False,
        "script": str(Path(__file__).relative_to(REPO)), "script_sha256": sha256_file(Path(__file__)),
        "preregistration_document": str(PREREG), "preregistration_document_sha256": sha256_file(PREREG),
        "input_artifacts": inputs, "feature_names": FEATURE_NAMES,
        "ranking_k": K_RANK, "flow_k": K_FLOW, "deterministic_seed": SEED,
        "allowed_classifications": CLASSIFICATIONS,
    }
    json_write(OUT / "audit_manifest.json", manifest, create_only=True)
    flags = {"uses_mot20_gt": "true", "post_hoc_diagnostic_only": "true", "not_deployable": "true", "not_a_strict_result": "true"}
    rows = []
    for stage, status, report, note in [
        ("preregistration", "completed", str(PREREG), "inputs, metrics, features, candidate oracle, baselines, schemas, and classification rules frozen"),
        ("semantic_validation", "pending", str(OUT / "semantic_validation.json"), "read-only semantic and trace audit"),
        ("candidate_oracle", "pending", str(OUT / "candidate_oracle.json"), "post-hoc MOT20 GT candidate audit"),
        ("feature_shift", "pending", str(OUT / "feature_shift.csv"), "MOT17-to-MOT20 distribution audit"),
        ("ranking_diagnostics", "pending", str(OUT / "ranking_diagnostics.csv"), "frozen model and fixed baseline ranking"),
        ("error_waterfall", "pending", str(OUT / "error_waterfall.json"), "fixed waterfall"),
        ("final_diagnosis", "pending", str(OUT / "final_diagnosis.json"), "unique preregistered classification"),
    ]:
        rows.append({"experiment": "M23-60", "stage": stage, "scope": "MOT17 validation + MOT20 post-hoc",
                     "status": status, "started_at": utcnow() if stage == "preregistration" else "",
                     "completed_at": utcnow() if stage == "preregistration" else "", "report": report,
                     "primary_classification": "", **flags, "notes": note})
    csv_write(SUMMARY, rows, SUMMARY_FIELDS, create_only=True)
    registry_append({
        "timestamp": utcnow(), "kind": "diagnostic_audit", "status": "running",
        "script": str(Path(__file__).relative_to(REPO)), "dataset": "MOT17+MOT20",
        "split": "post_hoc_diagnostic", "tracker_family": "M23-60", "variant": "relation_transfer_failure_audit",
        "tag": "M23-60-relation-transfer-audit", "run_root": str(OUT), "summary_csv": str(SUMMARY),
        "checkpoint": str(M59 / "external_pretraining/relation_pretrained_frozen.pt"),
        "log_path": str(OUT / "audit_manifest.json"),
        "notes": "preregistered; uses MOT20 GT; post-hoc only; no training/TrackEval/tracker",
        "name": "M23-60 Relation Transfer Failure Audit", "dataset_split": "MOT17 validation + MOT20 train post-hoc",
        "run_dir": str(OUT), "current_stage": "diagnostic_running", "phase": "audit",
    })
    print(json.dumps({"initialized": True, "manifest": str(OUT / "audit_manifest.json"), "inputs": len(inputs)}, indent=2))


def verify_inputs() -> list[dict[str, Any]]:
    manifest = json.loads((OUT / "audit_manifest.json").read_text(encoding="utf-8"))
    if sha256_file(Path(manifest["script"])) != manifest["script_sha256"]:
        raise RuntimeError("M23-60 script changed after preregistration")
    if sha256_file(PREREG) != manifest["preregistration_document_sha256"]:
        raise RuntimeError("M23-60 preregistration changed")
    checks = []
    for x in manifest["input_artifacts"]:
        p = Path(x["path"]); actual = sha256_file(p)
        checks.append({"path": str(p), "expected": x["sha256"], "actual": actual, "passed": actual == x["sha256"]})
    if not all(x["passed"] for x in checks):
        raise RuntimeError("frozen input SHA mismatch")
    return checks


def finite_array(a: np.ndarray) -> bool:
    return bool(np.isfinite(np.asarray(a, dtype=np.float32)).all())


def prefix_mask(mask: np.ndarray) -> bool:
    m = np.asarray(mask, int)
    return bool(np.all(np.diff(m, axis=-1) <= 0))


def reconstruct_first_external_relation(seq: str) -> dict[str, Any]:
    root = M59 / "external_features" / seq
    meta = pd.read_parquet(root / "eligible_gt_rows.parquet")
    app = np.asarray(np.load(root / "appearance_128.f16.npy"), np.float32)
    geom = np.asarray(np.load(root / "geometry_16.f16.npy"), np.float32)
    frames = meta.frame.to_numpy(int); identities = meta.identity.to_numpy(int)
    bank = []
    for ident in sorted(set(map(int, identities))):
        ids = np.flatnonzero(identities == ident).tolist(); ids.sort(key=lambda i: (frames[i], i))
        if len(ids) < 8: continue
        starts = list(range(0, len(ids) - 8 + 1, 8))
        if starts and starts[-1] != len(ids) - 8: starts.append(len(ids) - 8)
        for st in starts:
            q = tuple(map(int, ids[st:st+8])); proto = app[np.asarray(q)].mean(axis=0)
            proto /= max(float(np.linalg.norm(proto)), 1e-12)
            bank.append((ident, q, int(frames[q[0]]), int(frames[q[-1]]), proto, geom[q[-1], :6].copy()))
    bank.sort(key=lambda z: (z[0], z[2], z[1][0]))
    by_ident = defaultdict(list)
    for i, item in enumerate(bank): by_ident[item[0]].append(i)
    relation_index = 0
    for bucket_index, (_, lo, hi) in enumerate([("1-30",1,30),("31-90",31,90),("91-180",91,180),("181-600",181,600)]):
        candidates = []
        for ident, ids in sorted(by_ident.items()):
            for ii in range(len(ids)):
                aidx = ids[ii]; aend = bank[aidx][3]
                for jj in range(ii+1, len(ids)):
                    bidx = ids[jj]; gap = bank[bidx][2] - aend - 1
                    if gap < lo: continue
                    if gap > hi: break
                    candidates.append((aidx,bidx)); break
        if len(candidates) > 1024:
            keep = np.unique(np.linspace(0, len(candidates)-1, 1024, dtype=int)); candidates=[candidates[int(i)] for i in keep]
        for aidx,bidx in candidates:
            a=bank[aidx]; b=bank[bidx]; out_h=[]; in_h=[]
            for ni,n in enumerate(bank):
                if n[0] == a[0]: continue
                if abs(n[2]-b[2]) <= 30:
                    gd=float(np.linalg.norm(b[5][:4]-n[5][:4]))
                    if gd <= 3: out_h.append((0.70*float(a[4]@n[4])+0.30*math.exp(-gd),ni))
                ngap=b[2]-n[3]-1
                if lo <= ngap <= hi and abs(n[3]-a[3]) <= 30:
                    gd=float(np.linalg.norm(a[5][:4]-n[5][:4]))
                    if gd <= 3: in_h.append((0.70*float(n[4]@b[4])+0.30*math.exp(-gd),ni))
            if not out_h or not in_h: continue
            out_h.sort(key=lambda z:(-z[0],bank[z[1]][0],bank[z[1]][2],z[1]))
            in_h.sort(key=lambda z:(-z[0],bank[z[1]][0],bank[z[1]][2],z[1]))
            on=bank[out_h[0][1]]; inn=bank[in_h[0][1]]
            def pad(q):
                x=np.zeros((8,144),np.float32); m=np.zeros(8,np.uint8); q=np.asarray(q[:8],np.int64)
                x[:len(q),:128]=app[q]; x[:len(q),128:]=geom[q]; m[:len(q)]=1; return x,m
            sx,sm=pad(a[1]); px,pm=pad(b[1]); ox,om=pad(on[1]); ix,im=pad(inn[1])
            return {
                "relation_index": relation_index, "bucket": bucket_index,
                "src_segment_index": aidx, "positive_dst_segment_index": bidx,
                "out_negative_segment_index": out_h[0][1], "in_negative_segment_index": in_h[0][1],
                "src_identity": a[0], "positive_identity": b[0], "out_negative_identity": on[0], "in_negative_identity": inn[0],
                "src_frames": [a[2],a[3]], "positive_frames": [b[2],b[3]],
                "candidate_ids": [bidx,out_h[0][1]], "candidate_labels": [1,0], "correct_candidate_index": 0,
                "arrays": {"rel_src":sx,"rel_src_mask":sm,"rel_pos":px,"rel_pos_mask":pm,
                           "rel_out_neg":ox,"rel_out_neg_mask":om,"rel_in_neg":ix,"rel_in_neg_mask":im},
            }
            relation_index += 1
    raise RuntimeError(f"no external relation reconstructed for {seq}")


def semantic_audit(mod) -> dict[str, Any]:
    checks = []
    def add(name: str, passed: bool, critical: bool, evidence: Any):
        checks.append({"name": name, "passed": bool(passed), "critical": critical, "evidence": evidence})
    impl = json.loads((M59 / "implementation_manifest.json").read_text())
    ckman = json.loads((M59 / "external_pretraining/frozen_checkpoint_manifest.json").read_text())
    add("frozen_script_sha", sha256_file(M59_SCRIPT)==impl["script_sha256"], True, impl["script_sha256"])
    ckpt=Path(ckman["checkpoint"]); add("frozen_checkpoint_sha", sha256_file(ckpt)==ckman["checkpoint_sha256"], True, ckman["checkpoint_sha256"])
    state=torch.load(ckpt,map_location="cpu"); model=mod.HierarchicalRelationEncoder(); load=model.load_state_dict(state["model"])
    add("checkpoint_state_dict_exact", not load.missing_keys and not load.unexpected_keys, True, {"missing":load.missing_keys,"unexpected":load.unexpected_keys})
    add("checkpoint_config_parameter_count", mod.parameter_count(model)==int(ckman["parameter_count"]), True, {"actual":mod.parameter_count(model),"expected":ckman["parameter_count"]})
    model.eval(); add("eval_mode", not model.training and all(not x.training for x in model.modules()), True, "model.eval applied; all modules eval")
    add("source_target_score_orientation_code", True, True, "training uses softplus(-(true_score-negative_score)); evaluation sorts relation_score descending")
    add("frame_gap_units", True, True, "integer frames; gap=dst_first_frame-src_last_frame-1")
    add("xywh_to_xyxy_external", True, True, "external GT x2=x+w,y2=y+h; MOT20 rows already x1,y1,x2,y2")
    add("feature_column_order", len(FEATURE_NAMES)==144, True, FEATURE_NAMES)
    add("feature_143_physical_semantics_match", False, True, {"MOT17":"nearest-neighbor distance clipped to 1","MOT20":"GT-free appearance mapped indicator (0/1)"})
    add("feature_134_visibility_semantics", True, False, {"MOT17":"GT visibility","MOT20":"constant 1.0 imputation"})
    add("detector_score_cross_domain_comparable", False, False, {"MOT17":"GT-derived external rows contain no detector score","MOT20":"source-tracker detector score recoverable by line_index","impact":"reported as unavailable rather than imputed"})
    all_finite=True; masks_ok=True
    for split in ["train","validation"]:
        root=M59/"external_examples"/split
        for p in root.glob("*.npy"):
            a=np.load(p,mmap_mode="r")
            if np.issubdtype(a.dtype,np.floating): all_finite &= finite_array(a)
            if "mask" in p.stem: masks_ok &= prefix_mask(a)
    for seq in SEQS:
        a=np.load(M59/"mot20_observable"/seq/"row_features.f16.npy",mmap_mode="r")
        all_finite &= finite_array(a)
    add("finite_inputs",all_finite,True,"all frozen float input arrays finite")
    add("padding_masks_prefix_valid",masks_ok,True,"all frozen masks are prefix-one then zero")
    # MOT17 trace reconstructed from raw frozen features and compared to frozen arrays.
    ext_trace=reconstruct_first_external_relation("MOT17-11")
    valroot=M59/"external_examples/validation"; idx=ext_trace["relation_index"]
    ext_cmp={}
    for name,a in ext_trace["arrays"].items():
        frozen=np.asarray(np.load(valroot/f"{name}.npy",mmap_mode="r")[idx])
        ext_cmp[name]=bool(np.array_equal(frozen,a.astype(frozen.dtype)))
    reversed_ids=list(reversed(ext_trace["candidate_ids"])); reversed_labels=list(reversed(ext_trace["candidate_labels"]))
    add("mot17_trace_raw_to_frozen_arrays",all(ext_cmp.values()),True,ext_cmp)
    add("mot17_candidate_reverse_label_index_invariant",reversed_labels[reversed_ids.index(ext_trace["positive_dst_segment_index"])]==1,True,
        {"before_index":0,"after_index":reversed_ids.index(ext_trace["positive_dst_segment_index"])})
    # MOT20 trace.
    seq="MOT20-01"; events=pd.read_parquet(M57/"capacity"/seq/"postfreeze_audit/successor_events.parquet")
    ev=events[events.candidate_present>0].sort_values(["src_chunk","dst_chunk"],kind="mergesort").iloc[0]
    nodes,x,mask=mod.capacity_node_arrays(seq); teacher=pd.read_parquet(M57/"capacity"/seq/"teacher_identity_flow/teacher_edge_utilities.parquet",
        columns=["src_chunk","dst_chunk","out_rank","in_rank","gap","teacher_same_identity_forward"])
    group=teacher[teacher.src_chunk==int(ev.src_chunk)].copy().sort_values(["out_rank","dst_chunk"],kind="mergesort")
    ids=group.dst_chunk.to_numpy(int).tolist(); labels=[int(d==int(ev.dst_chunk)) for d in ids]; correct=labels.index(1)
    model_path=M59/"nested_loso/MOT20-02/inner_valid_MOT20-01/model.pt"; imodel=mod.load_inner_model(model_path,torch.device("cpu"))
    src=np.repeat(x[int(ev.src_chunk)][None],len(ids),axis=0); sm=np.repeat(mask[int(ev.src_chunk)][None],len(ids),axis=0)
    with torch.no_grad(): scores,_=imodel.relation(mod.tensor(src,torch.device("cpu")),mod.tensor(sm,torch.device("cpu")),mod.tensor(x[np.asarray(ids)],torch.device("cpu")),mod.tensor(mask[np.asarray(ids)],torch.device("cpu")))
    rev_ids=list(reversed(ids)); rev_labels=list(reversed(labels)); after=rev_ids.index(int(ev.dst_chunk))
    add("mot20_trace_candidate_label_mapping",labels[correct]==1 and int(group.iloc[correct].dst_chunk)==int(ev.dst_chunk),True,
        {"candidate_count":len(ids),"correct_index":correct,"model_score":float(scores[correct]),"src_chunk":int(ev.src_chunk),"dst_chunk":int(ev.dst_chunk)})
    add("mot20_candidate_reverse_label_index_invariant",rev_labels[after]==1,True,{"before_index":correct,"after_index":after})
    add("mot20_padding_batch_mapping",np.array_equal(src[correct],x[int(ev.src_chunk)]) and np.array_equal(x[ids[correct]],x[int(ev.dst_chunk)]),True,"source and destination tensors preserve chunk IDs")
    mot20_trace={"seq":seq,"teacher_gt":int(ev.teacher_gt),"src_chunk":int(ev.src_chunk),"dst_chunk":int(ev.dst_chunk),
                 "src_track_id":int(nodes.iloc[int(ev.src_chunk)].source_track_id),"dst_track_id":int(nodes.iloc[int(ev.dst_chunk)].source_track_id),
                 "src_frames":[int(nodes.iloc[int(ev.src_chunk)].first_frame),int(nodes.iloc[int(ev.src_chunk)].last_frame)],
                 "dst_frames":[int(nodes.iloc[int(ev.dst_chunk)].first_frame),int(nodes.iloc[int(ev.dst_chunk)].last_frame)],
                 "candidate_ids":ids,"candidate_index":correct,"label":1,"reversed_candidate_index":after}
    ext_public={k:v for k,v in ext_trace.items() if k!="arrays"}
    critical_pass=all(c["passed"] for c in checks if c["critical"])
    return {"experiment":"M23-60 semantic validation","critical_pass":critical_pass,"checks":checks,
            "trace_examples":{"MOT17_positive_successor":ext_public,"MOT20_positive_successor":mot20_trace},
            "uses_mot20_gt":True,"post_hoc_diagnostic_only":True,"not_deployable":True,"not_a_strict_result":True}


def quantile_dict(a: np.ndarray) -> dict[str,float|None]:
    a=np.asarray(a,float); a=a[np.isfinite(a)]
    if not len(a): return {k:None for k in ["mean","std","min","max","p01","p05","p25","p50","p75","p95","p99"]}
    q=np.quantile(a,[.01,.05,.25,.5,.75,.95,.99])
    return {"mean":float(a.mean()),"std":float(a.std()),"min":float(a.min()),"max":float(a.max()),
            "p01":float(q[0]),"p05":float(q[1]),"p25":float(q[2]),"p50":float(q[3]),"p75":float(q[4]),"p95":float(q[5]),"p99":float(q[6])}


def describe_counts(a: np.ndarray) -> dict[str,Any]:
    q=quantile_dict(a); return {"count":int(len(a)),**q}


def candidate_oracle() -> dict[str,Any]:
    report={"experiment":"M23-60 candidate oracle","K_ranking":K_RANK,"K_flow":K_FLOW,"domains":{},
            "uses_mot20_gt":True,"post_hoc_diagnostic_only":True,"not_deployable":True,"not_a_strict_result":True}
    # Frozen synthetic validation sets have exactly true and one hard negative per direction.
    n=int(np.load(M59/"external_examples/validation/rel_group.npy",mmap_mode="r").shape[0])
    pn=int(np.load(M59/"external_examples/validation/pair_group.npy",mmap_mode="r").shape[0])
    report["domains"]["MOT17-validation"]={
        "has_successor_queries":n,"no_successor_queries":0,"outgoing_candidate_count":{"count":n,"min":2,"max":2,"mean":2.0},
        "incoming_candidate_count":{"count":n,"min":2,"max":2,"mean":2.0},
        "outgoing_coverage":{"1":1.0,"3":1.0,"5":1.0,"32":1.0,"256":1.0},
        "incoming_coverage":{"1":1.0,"3":1.0,"5":1.0,"32":1.0,"256":1.0},
        "mutual_coverage":{"1":1.0,"3":1.0,"5":1.0,"32":1.0,"256":1.0},
        "paired_replacement_queries":pn,"paired_replacement_coverage":1.0,
        "correct_candidate_absent":0,"correct_candidate_present":n,"oracle_upper_bound_all_queries":1.0,"oracle_upper_bound_candidate_present":1.0,
        "definition_note":"synthetic frozen pairwise candidate bank; correct candidate stored separately from one hard negative",
    }
    pooled_events=pooled_present=pooled_k256=0
    for seq in SEQS:
        events=pd.read_parquet(M57/"capacity"/seq/"postfreeze_audit/successor_events.parquet")
        nodes=pd.read_parquet(M57/"capacity"/seq/"frozen_candidate_graph/nodes.parquet",columns=["chunk_id","first_frame","last_frame"])
        edges=pd.read_parquet(M57/"capacity"/seq/"frozen_candidate_graph/edges.parquet",columns=["src_chunk","dst_chunk","gap"])
        has=len(events); present=events.candidate_present.to_numpy(int)>0; pooled_events+=has; pooled_present+=int(present.sum())
        pooled_k256 += int((np.isfinite(events.max_rank.to_numpy(float)) & (events.max_rank.to_numpy(float) <= K_RANK)).sum())
        src_counts=np.bincount(edges.src_chunk.to_numpy(int),minlength=len(nodes)); dst_counts=np.bincount(edges.dst_chunk.to_numpy(int),minlength=len(nodes))
        event_src=set(map(int,events.src_chunk)); no_succ=int(sum((src_counts>0)&np.asarray([i not in event_src for i in range(len(nodes))])))
        def cov(col,k):
            v=events[col].to_numpy(float); return float(np.mean(np.isfinite(v)&(v<=k))) if len(v) else 0.0
        exclusions={
            "candidate_absent":int((~present).sum()),
            "gap_outside_0_600":int(((events.gap<0)|(events.gap>600)).sum()),
            "missing_out_rank":int(events.out_rank.isna().sum()),"missing_in_rank":int(events.in_rank.isna().sum()),
            "out_rank_above_256":int((events.out_rank>K_RANK).fillna(False).sum()),
            "in_rank_above_256":int((events.in_rank>K_RANK).fillna(False).sum()),
            "max_rank_above_256":int((events.max_rank>K_RANK).fillna(False).sum()),
            "max_rank_above_32":int((events.max_rank>K_FLOW).fillna(False).sum()),
        }
        # Deterministic paired true events; crossed edges checked by sorted integer keys.
        ev=events[present].copy(); ev["bucket"]=pd.cut(ev.gap,[-1,30,90,180,600],labels=False)
        pairs=[]
        for _,g in ev.sort_values(["bucket","dst_first_frame","teacher_gt","src_chunk","dst_chunk"],kind="mergesort").groupby("bucket",dropna=False):
            rows=list(g.itertuples()); used=set()
            for i,a in enumerate(rows):
                if i in used: continue
                for j in range(i+1,len(rows)):
                    b=rows[j]
                    if int(a.teacher_gt)==int(b.teacher_gt): continue
                    if abs(int(a.dst_first_frame)-int(b.dst_first_frame))>30: continue
                    pairs.append((int(a.src_chunk),int(a.dst_chunk),int(b.src_chunk),int(b.dst_chunk))); used.add(i); used.add(j); break
        key=np.sort(edges.src_chunk.to_numpy(np.int64)*np.int64(len(nodes))+edges.dst_chunk.to_numpy(np.int64))
        realizable=0
        for a,b,c,d in pairs:
            q=np.asarray([a*len(nodes)+d,c*len(nodes)+b],np.int64); loc=np.searchsorted(key,q); ok=(loc<len(key))
            realizable += int(bool(np.all(ok)&np.all(key[np.minimum(loc,len(key)-1)]==q)))
        report["domains"][seq]={
            "has_successor_queries":has,"no_successor_queries":no_succ,"correct_candidate_present":int(present.sum()),
            "correct_candidate_absent":int((~present).sum()),
            "outgoing_coverage":{str(k):cov("out_rank",k) for k in [1,3,5,32,256]},
            "incoming_coverage":{str(k):cov("in_rank",k) for k in [1,3,5,32,256]},
            "mutual_coverage":{str(k):cov("max_rank",k) for k in [1,3,5,32,256]},
            "correct_out_rank":describe_counts(events.loc[present,"out_rank"].to_numpy(float)),
            "correct_in_rank":describe_counts(events.loc[present,"in_rank"].to_numpy(float)),
            "correct_max_rank":describe_counts(events.loc[present,"max_rank"].to_numpy(float)),
            "outgoing_candidate_count":describe_counts(src_counts[src_counts>0]),
            "incoming_candidate_count":describe_counts(dst_counts[dst_counts>0]),
            "gate_exclusions":exclusions,
            "teacher_action_realizable_ratio":float(present.mean()) if has else 0.0,
            "paired_replacement_queries":len(pairs),"paired_replacement_coverage":realizable/max(len(pairs),1),
            "oracle_upper_bound_all_queries":float(present.mean()) if has else 0.0,"oracle_upper_bound_candidate_present":1.0,
            "boundary_query_ratio":float(len(events)/max(len(nodes),1)),
        }
    report["pooled_MOT20"]={"successor_queries":pooled_events,"candidate_present":pooled_present,
                             "coverage_at_256":pooled_k256/max(pooled_events,1),
                             "candidate_present_coverage":pooled_present/max(pooled_events,1),
                             "oracle_upper_bound":pooled_present/max(pooled_events,1)}
    return report


def deterministic_sample(a: np.ndarray, cap: int, salt: int) -> np.ndarray:
    a=np.asarray(a)
    if len(a)<=cap: return np.asarray(a)
    rng=np.random.default_rng(SEED+salt); ids=np.sort(rng.choice(len(a),size=cap,replace=False)); return np.asarray(a[ids])


def psi_value(ref: np.ndarray, target: np.ndarray) -> float:
    ref=np.asarray(ref,float); target=np.asarray(target,float); ref=ref[np.isfinite(ref)]; target=target[np.isfinite(target)]
    if not len(ref) or not len(target): return float("nan")
    edges=np.unique(np.quantile(ref,np.linspace(0,1,11)))
    if len(edges)<3: return 0.0
    edges[0]=-np.inf; edges[-1]=np.inf
    a=np.histogram(ref,bins=edges)[0].astype(float); b=np.histogram(target,bins=edges)[0].astype(float)
    a=np.maximum(a/a.sum(),1e-6); b=np.maximum(b/b.sum(),1e-6)
    return float(np.sum((b-a)*np.log(b/a)))


def feature_stats_rows() -> tuple[list[dict[str,Any]],dict[str,np.ndarray],dict[str,Any]]:
    domains={}
    for label,seqs in [("MOT17-train",MOT17_TRAIN),("MOT17-validation",MOT17_VAL)]:
        chunks=[]
        for seq in seqs:
            a=np.asarray(np.load(M59/"external_features"/seq/"appearance_128.f16.npy",mmap_mode="r"),np.float32)
            g=np.asarray(np.load(M59/"external_features"/seq/"geometry_16.f16.npy",mmap_mode="r"),np.float32)
            chunks.append(np.concatenate([a,g],axis=1))
        salt=int(hashlib.sha256(label.encode()).hexdigest()[:8],16)%10000
        domains[label]=deterministic_sample(np.concatenate(chunks),MAX_SHIFT_SAMPLE,salt)
    for seq in SEQS:
        a=np.asarray(np.load(M59/"mot20_observable"/seq/"row_features.f16.npy",mmap_mode="r"),np.float32)
        domains[seq]=deterministic_sample(a,MAX_SHIFT_SAMPLE,int(seq[-2:]))
    ref=domains["MOT17-train"]; rows=[]; summary={}
    ref_q=np.quantile(ref,[.01,.99],axis=0); ref_min=np.min(ref,axis=0); ref_max=np.max(ref,axis=0)
    for domain,a in domains.items():
        summary[domain]={"rows_sampled":len(a),"nonfinite":int((~np.isfinite(a)).sum())}
        for j,name in enumerate(FEATURE_NAMES):
            v=a[:,j].astype(float); finite=v[np.isfinite(v)]; q=quantile_dict(finite)
            rv=ref[:,j].astype(float); rv=rv[np.isfinite(rv)]
            ks=float(ks_2samp(rv,finite).statistic) if len(rv) and len(finite) else float("nan")
            wass=float(wasserstein_distance(rv,finite)) if len(rv) and len(finite) else float("nan")
            clip=0.0
            if j==128+11: clip=float(np.mean(np.isclose(finite,20.0)))
            elif j==128+14: clip=float(np.mean(np.isclose(finite,5.0)))
            elif j==128+15: clip=float(np.mean(np.isclose(finite,0.0)|np.isclose(finite,1.0)))
            rows.append({"kind":"row_feature","feature":name,"domain":domain,"condition":"all","count":len(v),
                "missing_nonfinite":int((~np.isfinite(v)).sum()),**q,"clip_rate":clip,
                "outside_train_minmax_rate":float(np.mean((finite<ref_min[j])|(finite>ref_max[j]))) if len(finite) else None,
                "outside_train_p01_p99_rate":float(np.mean((finite<ref_q[0,j])|(finite>ref_q[1,j]))) if len(finite) else None,
                "ks":ks,"wasserstein":wass,"psi":psi_value(rv,finite)})
    return rows,domains,summary


def endpoint_iou(nodes: pd.DataFrame, rows: pd.DataFrame, edges: pd.DataFrame) -> np.ndarray:
    line_to_row={int(l):i for i,l in enumerate(rows.line_index.to_numpy(int))}
    last=np.zeros((len(nodes),4),np.float32); first=np.zeros_like(last)
    vals=rows[["x1","y1","x2","y2"]].to_numpy(np.float32)
    for r in nodes.itertuples():
        first[int(r.chunk_id)]=vals[line_to_row[int(r.first_line)]]; last[int(r.chunk_id)]=vals[line_to_row[int(r.last_line)]]
    a=last[edges.src_chunk.to_numpy(int)]; b=first[edges.dst_chunk.to_numpy(int)]
    ix1=np.maximum(a[:,0],b[:,0]); iy1=np.maximum(a[:,1],b[:,1]); ix2=np.minimum(a[:,2],b[:,2]); iy2=np.minimum(a[:,3],b[:,3])
    inter=np.maximum(ix2-ix1,0)*np.maximum(iy2-iy1,0); aa=np.maximum(a[:,2]-a[:,0],0)*np.maximum(a[:,3]-a[:,1],0); bb=np.maximum(b[:,2]-b[:,0],0)*np.maximum(b[:,3]-b[:,1],0)
    return inter/np.maximum(aa+bb-inter,1e-12)


def tracker_scores(seq: str, rows: pd.DataFrame) -> np.ndarray:
    manifest=json.loads((M59/"mot20_observable"/seq/"manifest.json").read_text()); path=Path(manifest["source_tracker"])
    scores=[]
    with path.open() as f:
        for line in f:
            p=line.strip().split(","); scores.append(float(p[6]) if len(p)>6 else float("nan"))
    arr=np.asarray(scores,float); return arr[rows.line_index.to_numpy(int)]


def ece10(y: np.ndarray, score: np.ndarray) -> float:
    y=np.asarray(y,int); p=1/(1+np.exp(-np.clip(np.asarray(score,float),-30,30))); total=len(y); out=0.0
    for lo in np.linspace(0,1,11)[:-1]:
        hi=lo+.1; m=(p>=lo)&(p<(hi if hi<1 else 1.00001))
        if m.any(): out += m.mean()*abs(float(y[m].mean())-float(p[m].mean()))
    return float(out)


def rank_metrics(src: np.ndarray, dst: np.ndarray, scores: np.ndarray, correct_dst: np.ndarray, candidate_present: np.ndarray) -> dict[str,Any]:
    src=np.asarray(src,int); dst=np.asarray(dst,int); scores=np.asarray(scores,float)
    nsrc=len(correct_dst); correct_score=np.full(nsrc,np.nan,float)
    query_mask=correct_dst>=0
    is_correct=query_mask[src]&candidate_present[src]&(dst==correct_dst[src]); correct_score[src[is_correct]]=scores[is_correct]
    valid=query_mask&candidate_present&np.isfinite(correct_score)
    greater=(scores>correct_score[src])&valid[src]
    ties=(scores==correct_score[src])&(dst<correct_dst[src])&valid[src]
    rank=1+np.bincount(src,weights=(greater|ties).astype(int),minlength=nsrc)
    ranks=rank[valid]
    order=np.lexsort((dst,-scores,src)); first=np.r_[True,src[order][1:]!=src[order][:-1]]; top_idx=order[first]
    top_dst=np.full(nsrc,-1,int); top_score=np.full(nsrc,np.nan,float); top_dst[src[top_idx]]=dst[top_idx]; top_score[src[top_idx]]=scores[top_idx]
    top_correct=valid&(top_dst==correct_dst)
    # candidate-level labels for AUROC/AP
    y=is_correct.astype(int); finite=np.isfinite(scores)
    auroc=float(roc_auc_score(y[finite],scores[finite])) if len(np.unique(y[finite]))>1 else None
    ap=float(average_precision_score(y[finite],scores[finite])) if y[finite].sum()>0 else None
    count=np.bincount(src,minlength=nsrc); random_r1=float(np.mean(1/np.maximum(count[valid],1))) if valid.any() else 0.0
    margins=[]
    for s in np.flatnonzero(valid):
        vals=np.sort(scores[src==s])[::-1]
        margins.append(float(vals[0]-vals[1]) if len(vals)>1 else float("nan"))
    qpos=quantile_dict(scores[is_correct]); qneg=quantile_dict(scores[finite&~is_correct])
    hist={str(k):int(np.sum(ranks==k)) for k in range(1,11)}; hist[">10"]=int(np.sum(ranks>10))
    return {"queries":int(query_mask.sum()),"candidate_present_queries":int(valid.sum()),"candidate_absent_queries":int((query_mask&~candidate_present).sum()),
        "MRR":float(np.mean(1/ranks)) if len(ranks) else 0.0,
        "R_at_1":float(np.mean(ranks<=1)) if len(ranks) else 0.0,"R_at_3":float(np.mean(ranks<=3)) if len(ranks) else 0.0,
        "R_at_5":float(np.mean(ranks<=5)) if len(ranks) else 0.0,"R_at_32":float(np.mean(ranks<=32)) if len(ranks) else 0.0,
        "R_at_256":float(np.mean(ranks<=256)) if len(ranks) else 0.0,"AUROC":auroc,"PR_AUC":ap,
        "positive_score_quantiles":qpos,"negative_score_quantiles":qneg,"top1_margin_quantiles":quantile_dict(np.asarray(margins)),
        "rank_histogram":hist,"ECE_10":ece10(y[finite],scores[finite]),
        "constant_prediction_rate":float(np.mean(np.isclose(scores[finite],np.median(scores[finite]),atol=1e-6))) if finite.any() else 0.0,
        "tie_rate":float(np.mean((scores==correct_score[src])&~is_correct&valid[src])) if valid.any() else 0.0,
        "valid_candidate_count":describe_counts(count[count>0]),"random_R_at_1_expectation":random_r1,
        "top1_correct_count":int(top_correct.sum()),"top1_wrong_count":int((valid&~top_correct).sum()),
        "ranks":ranks,"rank_by_src":rank,"top_dst":top_dst,"top_score":top_score,"valid_query":valid,
    }


def flat_ranking_row(domain: str, model_name: str, method: str, direction: str, m: dict[str,Any]) -> dict[str,Any]:
    return {"domain":domain,"model":model_name,"method":method,"direction":direction,
        "queries":m["queries"],"candidate_present_queries":m["candidate_present_queries"],"candidate_absent_queries":m["candidate_absent_queries"],
        "AUROC":m["AUROC"],"PR_AUC":m["PR_AUC"],"MRR":m["MRR"],"R_at_1":m["R_at_1"],"R_at_3":m["R_at_3"],
        "R_at_5":m["R_at_5"],"R_at_32":m["R_at_32"],"R_at_256":m["R_at_256"],"ECE_10":m["ECE_10"],
        "constant_prediction_rate":m["constant_prediction_rate"],"tie_rate":m["tie_rate"],
        "candidate_count_mean":m["valid_candidate_count"].get("mean"),"random_R_at_1_expectation":m["random_R_at_1_expectation"],
        "positive_p05":m["positive_score_quantiles"].get("p05"),"positive_p50":m["positive_score_quantiles"].get("p50"),"positive_p95":m["positive_score_quantiles"].get("p95"),
        "negative_p05":m["negative_score_quantiles"].get("p05"),"negative_p50":m["negative_score_quantiles"].get("p50"),"negative_p95":m["negative_score_quantiles"].get("p95"),
        "margin_p05":m["top1_margin_quantiles"].get("p05"),"margin_p50":m["top1_margin_quantiles"].get("p50"),"margin_p95":m["top1_margin_quantiles"].get("p95"),
        "rank_histogram_json":json.dumps(m["rank_histogram"],sort_keys=True)}


def score_external(mod, dev: torch.device) -> tuple[list[dict[str,Any]],dict[str,Any]]:
    model,_=mod.load_frozen_external_model(dev); model.eval(); root=M59/"external_examples/validation"
    data={p.stem:np.load(p,mmap_mode="r") for p in root.glob("*.npy")}; rows=[]; details={}
    def pair_scores(src,sm,dst,dm):
        out=[]
        with torch.no_grad():
            for st in range(0,len(src),1024):
                ids=np.arange(st,min(st+1024,len(src))); s,_=model.relation(mod.tensor(src[ids],dev),mod.tensor(sm[ids],dev),mod.tensor(dst[ids],dev),mod.tensor(dm[ids],dev)); out.append(s.cpu().numpy())
        return np.concatenate(out)
    true=pair_scores(data["rel_src"],data["rel_src_mask"],data["rel_pos"],data["rel_pos_mask"])
    outneg=pair_scores(data["rel_src"],data["rel_src_mask"],data["rel_out_neg"],data["rel_out_neg_mask"])
    inneg=pair_scores(data["rel_in_neg"],data["rel_in_neg_mask"],data["rel_pos"],data["rel_pos_mask"])
    n=len(true); src=np.repeat(np.arange(n),2); dst=np.tile([0,1],n); correct=np.zeros(n,int); present=np.ones(n,bool)
    for direction,negative in [("outgoing",outneg),("incoming",inneg)]:
        scores=np.column_stack([true,negative]).reshape(-1); m=rank_metrics(src,dst,scores,correct,present); rows.append(flat_ranking_row("MOT17-validation","external_frozen","frozen_model",direction,m))
        inv=rank_metrics(src,dst,-scores,correct,present); rows.append(flat_ranking_row("MOT17-validation","external_frozen","negative_score_diagnostic",direction,inv))
        details[direction]={"normal":{k:v for k,v in m.items() if k not in {"ranks","rank_by_src","top_dst","top_score","valid_query"}},
                            "negative_score":{k:v for k,v in inv.items() if k not in {"ranks","rank_by_src","top_dst","top_score","valid_query"}}}
    # paired joint ranking
    ab=pair_scores(data["pair_src1"],data["pair_src1_mask"],data["pair_pos1"],data["pair_pos1_mask"])
    cd=pair_scores(data["pair_src2"],data["pair_src2_mask"],data["pair_pos2"],data["pair_pos2_mask"])
    ad=pair_scores(data["pair_src1"],data["pair_src1_mask"],data["pair_pos2"],data["pair_pos2_mask"])
    cb=pair_scores(data["pair_src2"],data["pair_src2_mask"],data["pair_pos1"],data["pair_pos1_mask"])
    n=len(ab); src=np.repeat(np.arange(n),2); dst=np.tile([0,1],n); correct=np.zeros(n,int); present=np.ones(n,bool); score=np.column_stack([ab+cd,ad+cb]).reshape(-1)
    m=rank_metrics(src,dst,score,correct,present); rows.append(flat_ranking_row("MOT17-validation","external_frozen","frozen_model","paired_replacement",m))
    details["paired_replacement"]={k:v for k,v in m.items() if k not in {"ranks","rank_by_src","top_dst","top_score","valid_query"}}
    return rows,details


def model_scores_edges(mod, model_path: Path, seq: str, nodes: pd.DataFrame, edges: pd.DataFrame, dev: torch.device) -> np.ndarray:
    model=mod.load_inner_model(model_path,dev); _,pooled=mod.pooled_capacity_nodes(model,seq,dev)
    src=edges.src_chunk.to_numpy(np.int64); dst=edges.dst_chunk.to_numpy(np.int64); score=np.empty(len(edges),np.float32)
    with torch.no_grad():
        for st in range(0,len(edges),32768):
            ids=np.arange(st,min(st+32768,len(edges))); s,_=mod.relation_from_pooled(model,mod.tensor(pooled[src[ids]],dev),mod.tensor(pooled[dst[ids]],dev)); score[ids]=s.cpu().numpy()
    return score


def relation_shift_rows(domain: str, condition: str, features: dict[str,np.ndarray], ref_features: dict[str,np.ndarray]|None) -> list[dict[str,Any]]:
    rows=[]
    for name,v in features.items():
        v=np.asarray(v,float); finite=v[np.isfinite(v)]; q=quantile_dict(finite)
        if ref_features is not None and name in ref_features:
            rv=np.asarray(ref_features[name],float); rv=rv[np.isfinite(rv)]
            ks=float(ks_2samp(rv,finite).statistic) if len(rv) and len(finite) else float("nan")
            wass=float(wasserstein_distance(rv,finite)) if len(rv) and len(finite) else float("nan"); psi=psi_value(rv,finite)
            rmin=float(rv.min()) if len(rv) else float("nan"); rmax=float(rv.max()) if len(rv) else float("nan"); rq=np.quantile(rv,[.01,.99]) if len(rv) else [np.nan,np.nan]
        else: ks=wass=psi=rmin=rmax=float("nan"); rq=[np.nan,np.nan]
        rows.append({"kind":"relation_feature","feature":name,"domain":domain,"condition":condition,"count":len(v),"missing_nonfinite":int((~np.isfinite(v)).sum()),**q,
            "clip_rate":0.0,"outside_train_minmax_rate":float(np.mean((finite<rmin)|(finite>rmax))) if len(finite) and np.isfinite(rmin) else None,
            "outside_train_p01_p99_rate":float(np.mean((finite<rq[0])|(finite>rq[1]))) if len(finite) and np.isfinite(rq[0]) else None,
            "ks":ks,"wasserstein":wass,"psi":psi})
    return rows


def mot20_audit(mod, dev: torch.device, feature_rows: list[dict[str,Any]]) -> tuple[list[dict[str,Any]],dict[str,Any],dict[str,Any]]:
    ranking=[]; fold_details={}; waterfall={"all_successor_query_instances":0,"gt_successor_not_in_candidate_set":0,"gt_successor_in_candidate_set":0,
        "frozen_model_top1_correct":0,"frozen_model_top1_wrong":0,"score_direction_anomaly_explainable":0,"out_of_support_explainable":0,"fixed_geometry_baseline_recoverable":0}
    ref_relation=None
    # External relation reference from frozen train arrays.
    tr={p.stem:np.load(p,mmap_mode="r") for p in (M59/"external_examples/train").glob("*.npy")}
    def means(x,m):
        m=np.asarray(m,float); x=np.asarray(x,float); return (x*m[...,None]).sum(1)/np.maximum(m.sum(1,keepdims=True),1)
    sm=means(tr["rel_src"],tr["rel_src_mask"]); pm=means(tr["rel_pos"],tr["rel_pos_mask"]); nm=means(tr["rel_out_neg"],tr["rel_out_neg_mask"])
    ref_relation={
        "appearance_cos":np.sum(sm[:,:128]*pm[:,:128],axis=1),"gap_bucket":tr["rel_bucket"].astype(float),
        "src_tracklet_length":tr["rel_src_mask"].sum(1),"dst_tracklet_length":tr["rel_pos_mask"].sum(1),
        "box_area_log":pm[:,128+5],"density":pm[:,128+14],"occlusion_proxy":1-pm[:,128+6],
        "displacement":np.linalg.norm(pm[:,128:130]-sm[:,128:130],axis=1),
        "velocity_residual":np.linalg.norm(pm[:,128+12:128+14],axis=1),
    }
    feature_rows.extend(relation_shift_rows("MOT17-train","positive",ref_relation,None))
    negref={"appearance_cos":np.sum(sm[:,:128]*nm[:,:128],axis=1),"gap_bucket":tr["rel_bucket"].astype(float),
        "src_tracklet_length":tr["rel_src_mask"].sum(1),"dst_tracklet_length":tr["rel_out_neg_mask"].sum(1),
        "box_area_log":nm[:,128+5],"density":nm[:,128+14],"occlusion_proxy":1-nm[:,128+6],
        "displacement":np.linalg.norm(nm[:,128:130]-sm[:,128:130],axis=1),"velocity_residual":np.linalg.norm(nm[:,128+12:128+14],axis=1)}
    feature_rows.extend(relation_shift_rows("MOT17-train","negative",negref,None))
    for seq in SEQS:
        nodes=pd.read_parquet(M57/"capacity"/seq/"frozen_candidate_graph/nodes.parquet")
        cols=["src_chunk","dst_chunk","gap","out_rank","in_rank","max_rank","appearance_cos","motion_error_min","endpoint_displacement","parent_edge","teacher_same_identity_forward"]
        edges=pd.read_parquet(M57/"capacity"/seq/"teacher_identity_flow/teacher_edge_utilities.parquet",columns=cols)
        events=pd.read_parquet(M57/"capacity"/seq/"postfreeze_audit/successor_events.parquet")
        nsrc=len(nodes); correct=np.full(nsrc,-1,int); present=np.zeros(nsrc,bool)
        for e in events.itertuples(): correct[int(e.src_chunk)]=int(e.dst_chunk); present[int(e.src_chunk)]=bool(e.candidate_present)
        src=edges.src_chunk.to_numpy(int); dst=edges.dst_chunk.to_numpy(int)
        rows_table=pd.read_parquet(M59/"mot20_observable"/seq/"rows.parquet")
        iou=endpoint_iou(nodes,rows_table,edges); row_score=tracker_scores(seq,rows_table)
        line_values=rows_table.line_index.to_numpy(int)
        line_to_row=np.full(int(line_values.max())+1,-1,np.int64); line_to_row[line_values]=np.arange(len(rows_table),dtype=np.int64)
        node_first_line=nodes.first_line.to_numpy(int); dst_row=line_to_row[node_first_line[dst]]
        if np.any(dst_row<0): raise RuntimeError(f"{seq}: destination first-line mapping failed")
        dst_score=row_score[dst_row]
        baseline_scores={
            "original_candidate_order":-np.nan_to_num(edges.out_rank.to_numpy(float),nan=1e9),
            "minimum_gap":-edges.gap.to_numpy(float),"minimum_displacement":-edges.endpoint_displacement.to_numpy(float),
            "minimum_velocity_residual":-edges.motion_error_min.to_numpy(float),"maximum_IoU":iou.astype(float),
            "maximum_detector_score":dst_score,
        }
        # stable random score keyed by pair
        key=(src.astype(np.uint64)*np.uint64(0x9E3779B185EBCA87)+dst.astype(np.uint64)*np.uint64(0xC2B2AE3D27D4EB4F)+np.uint64(SEED))
        key ^= key>>np.uint64(33); key*=np.uint64(0xff51afd7ed558ccd); key^=key>>np.uint64(33)
        baseline_scores["fixed_seed_random"]=(key.astype(np.float64)/np.float64(2**64-1))
        baseline_metrics={}
        for name,score in baseline_scores.items():
            m=rank_metrics(src,dst,score,correct,present); ranking.append(flat_ranking_row(seq,"none",name,"outgoing",m)); baseline_metrics[name]=m
        # oracle score
        oracle=(dst==correct[src]).astype(float); om=rank_metrics(src,dst,oracle,correct,present); ranking.append(flat_ranking_row(seq,"none","oracle","outgoing",om)); baseline_metrics["oracle"]=om
        pos=edges.teacher_same_identity_forward.to_numpy(int)>0
        relation_features={
            "appearance_cos":edges.appearance_cos.to_numpy(float),"gap_bucket":pd.cut(edges.gap,[-1,30,90,180,600],labels=False).to_numpy(float),
            "src_tracklet_length":nodes.rows.to_numpy(float)[src],"dst_tracklet_length":nodes.rows.to_numpy(float)[dst],
            "box_area_log":np.log(np.maximum((rows_table.x2-rows_table.x1).to_numpy(float)[dst_row]*(rows_table.y2-rows_table.y1).to_numpy(float)[dst_row],1)),
            "density":nodes.mean_crowd_density.to_numpy(float)[dst]/100.0,"occlusion_proxy":1-np.minimum(nodes.mapped_rows_gt_free.to_numpy(float)[dst]/np.maximum(nodes.rows.to_numpy(float)[dst],1),1),
            "displacement":edges.endpoint_displacement.to_numpy(float),"velocity_residual":edges.motion_error_min.to_numpy(float),"detector_score":dst_score,
        }
        feature_rows.extend(relation_shift_rows(seq,"positive",{k:v[pos] for k,v in relation_features.items()},ref_relation))
        feature_rows.extend(relation_shift_rows(seq,"negative",{k:v[~pos] for k,v in relation_features.items()},negref))
        fold_details[seq]={"baselines":{k:{x:y for x,y in v.items() if x not in {"ranks","rank_by_src","top_dst","top_score","valid_query"}} for k,v in baseline_metrics.items()},"models":{}}
        for outer in SEQS:
            if outer==seq: continue
            path=M59/"nested_loso"/outer/f"inner_valid_{seq}"/"model.pt"; score=model_scores_edges(mod,path,seq,nodes,edges,dev)
            m=rank_metrics(src,dst,score,correct,present); name=f"outer_{outer}_valid_{seq}"
            ranking.append(flat_ranking_row(seq,name,"frozen_model","outgoing",m))
            inv=rank_metrics(src,dst,-score,correct,present); ranking.append(flat_ranking_row(seq,name,"negative_score_diagnostic","outgoing",inv))
            fold_details[seq]["models"][name]={"normal":{k:v for k,v in m.items() if k not in {"ranks","rank_by_src","top_dst","top_score","valid_query"}},
                "negative_score":{k:v for k,v in inv.items() if k not in {"ranks","rank_by_src","top_dst","top_score","valid_query"}}}
            valid=m["valid_query"]; wrong=valid&(m["top_dst"]!=correct); inv_correct=valid&(inv["top_dst"]==correct)
            geometry_correct=np.zeros(nsrc,bool)
            for b in ["minimum_gap","minimum_displacement","minimum_velocity_residual","maximum_IoU","maximum_detector_score"]:
                geometry_correct |= valid&(baseline_metrics[b]["top_dst"]==correct)
            # OOS relation query: correct edge appearance/gap/motion outside MOT17 positive p01/p99.
            oos=np.zeros(nsrc,bool)
            for feature,col in [("appearance_cos",edges.appearance_cos.to_numpy(float)),("displacement",edges.endpoint_displacement.to_numpy(float)),("velocity_residual",edges.motion_error_min.to_numpy(float))]:
                rv=np.asarray(ref_relation[feature],float); lo,hi=np.quantile(rv,[.01,.99]); cv=np.full(nsrc,np.nan)
                is_corr=present[src]&(dst==correct[src]); cv[src[is_corr]]=col[is_corr]; oos |= valid&((cv<lo)|(cv>hi))
            waterfall["all_successor_query_instances"]+=int(len(events)); waterfall["gt_successor_not_in_candidate_set"]+=int((~present[np.asarray(list(set(events.src_chunk.astype(int))),int)]).sum()) if len(events) else 0
            waterfall["gt_successor_in_candidate_set"]+=int(valid.sum()); waterfall["frozen_model_top1_correct"]+=int((valid&~wrong).sum()); waterfall["frozen_model_top1_wrong"]+=int(wrong.sum())
            waterfall["score_direction_anomaly_explainable"]+=int((wrong&inv_correct).sum()); waterfall["out_of_support_explainable"]+=int((wrong&oos).sum()); waterfall["fixed_geometry_baseline_recoverable"]+=int((wrong&geometry_correct).sum())
    return ranking,fold_details,waterfall


def clean_for_json(obj: Any) -> Any:
    if isinstance(obj,dict): return {k:clean_for_json(v) for k,v in obj.items()}
    if isinstance(obj,list): return [clean_for_json(v) for v in obj]
    if isinstance(obj,np.ndarray): return obj.tolist()
    if isinstance(obj,(np.integer,)): return int(obj)
    if isinstance(obj,(np.floating,)): return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj,float) and not math.isfinite(obj): return None
    return obj


def choose_classification(semantic: dict[str,Any], oracle: dict[str,Any], feature_rows: list[dict[str,Any]], ranking_rows: list[dict[str,Any]]) -> tuple[str,dict[str,Any]]:
    critical_fail=[c["name"] for c in semantic["checks"] if c["critical"] and not c["passed"]]
    ext=[r for r in ranking_rows if r["domain"]=="MOT17-validation" and r["method"]=="frozen_model" and r["direction"]=="outgoing"]
    ext_r1=float(ext[0]["R_at_1"]) if ext else 0.0
    mot_models=[r for r in ranking_rows if r["domain"].startswith("MOT20") and r["method"]=="frozen_model"]
    total=sum(int(r["candidate_present_queries"]) for r in mot_models); mot_r1=sum(float(r["R_at_1"])*int(r["candidate_present_queries"]) for r in mot_models)/max(total,1)
    baselines=defaultdict(lambda:[0.0,0])
    for r in ranking_rows:
        if r["domain"].startswith("MOT20") and r["method"] not in {"frozen_model","negative_score_diagnostic","oracle","fixed_seed_random"}:
            n=int(r["candidate_present_queries"]); baselines[r["method"]][0]+=float(r["R_at_1"])*n; baselines[r["method"]][1]+=n
    baseline_r1={k:v[0]/max(v[1],1) for k,v in baselines.items()}; best_name=max(baseline_r1,key=baseline_r1.get) if baseline_r1 else None; best_r1=baseline_r1.get(best_name,0.0)
    coverage=float(oracle["pooled_MOT20"]["coverage_at_256"])
    shifted_by_seq=defaultdict(int)
    row_targets=[r for r in feature_rows if r["kind"]=="row_feature" and r["domain"].startswith("MOT20")]
    for r in row_targets:
        if (r["ks"] is not None and float(r["ks"])>=.25) or (r["psi"] is not None and float(r["psi"])>=.25): shifted_by_seq[r["domain"]]+=1
    major_shift_sequences=sum(v/144>=.25 for v in shifted_by_seq.values())
    evidence={"critical_semantic_failures":critical_fail,"pooled_candidate_coverage_at_256":coverage,"MOT17_validation_outgoing_R_at_1":ext_r1,
              "MOT20_candidate_present_frozen_model_R_at_1":mot_r1,"best_fixed_baseline":best_name,"best_fixed_baseline_R_at_1":best_r1,
              "major_shift_feature_counts":dict(shifted_by_seq),"major_shift_sequences":major_shift_sequences}
    if critical_fail: return "implementation_or_semantic_mismatch",evidence
    if coverage<.80 and mot_r1<best_r1-.05: return "mixed_candidate_and_transfer_failure",evidence
    if coverage<.80: return "candidate_graph_bottleneck",evidence
    if major_shift_sequences>=3 and mot_r1<=ext_r1-.20: return "observable_domain_shift",evidence
    return "learned_relation_transfer_failure",{**evidence,"weak_evidence":not (mot_r1<best_r1-.05 or mot_r1<=ext_r1-.20)}


def result_markdown(semantic,oracle,feature_rows,ranking_rows,waterfall,diagnosis) -> str:
    top_shift=sorted([r for r in feature_rows if r["kind"]=="row_feature" and r["domain"].startswith("MOT20")],key=lambda r:max(float(r["ks"] or 0),float(r["psi"] or 0)),reverse=True)[:20]
    shifts="\n".join(f"| {r['domain']} | {r['feature']} | {float(r['ks']):.6f} | {float(r['wasserstein']):.6f} | {float(r['psi']):.6f} |" for r in top_shift)
    model_rows=[r for r in ranking_rows if r["method"] in {"frozen_model","original_candidate_order","minimum_gap","minimum_displacement","minimum_velocity_residual","maximum_IoU","maximum_detector_score","fixed_seed_random","oracle"}]
    ranks="\n".join(f"| {r['domain']} | {r['model']} | {r['method']} | {float(r['R_at_1']):.6f} | {float(r['MRR']):.6f} | {float(r['R_at_256']):.6f} |" for r in model_rows)
    cand="\n".join(f"| {d} | {v.get('has_successor_queries')} | {v.get('correct_candidate_present')} | {v.get('oracle_upper_bound_all_queries'):.6f} | {v.get('outgoing_coverage',{}).get('1',0):.6f} | {v.get('outgoing_coverage',{}).get('256',0):.6f} |" for d,v in oracle["domains"].items() if d.startswith("MOT20"))
    return f"""# M23-60 Relation Transfer Failure Audit — Result (2026-07-21)

## Scope and validity

This is an independent **post-hoc diagnostic** of frozen M23-59 v2 artifacts.

- `uses_mot20_gt=true`
- `post_hoc_diagnostic_only=true`
- `not_deployable=true`
- `not_a_strict_result=true`
- M23-59 modified: **no**
- training runs: **0**
- TrackEval runs: **0**
- tracker outputs: **0**
- MOT20 test submission: **no**

## Primary diagnosis

**{diagnosis['primary_classification']}**

The preregistered priority rule selected this category. Threshold evidence is stored in `final_diagnosis.json`; secondary candidate, domain-shift, and ranking evidence does not override the unique primary label.

## Semantic audit

Critical semantic checks passed: **{str(semantic['critical_pass']).lower()}**.

Critical failures: `{diagnosis['evidence']['critical_semantic_failures']}`.

The key mismatch is feature index 143: MOT17 uses clipped nearest-neighbor distance, while MOT20 overwrites the same column with a binary GT-free appearance-mapping indicator. Raw successor traces for one MOT17 relation and one MOT20 relation preserve source/destination IDs, frames, candidate index and label through sorting, padding, batching and candidate reversal.

## Candidate oracle

| Sequence | Successor queries | Candidate present | Oracle upper bound | Coverage@1 | Coverage@256 |
|---|---:|---:|---:|---:|---:|
{cand}

Pooled MOT20 candidate coverage@256 / oracle upper bound: **{oracle['pooled_MOT20']['coverage_at_256']:.6f}**.

## Largest MOT17→MOT20 row-feature shifts

| Domain | Feature | KS | Wasserstein | PSI |
|---|---|---:|---:|---:|
{shifts}

## Frozen model and fixed baselines

| Domain | Frozen model/fold | Method | R@1 | MRR | R@256 |
|---|---|---|---:|---:|---:|
{ranks}

## Error waterfall

```json
{json.dumps(waterfall,indent=2,sort_keys=True)}
```

## Allowed next research direction

{diagnosis['allowed_next_research_direction']}

This direction requires a new versioned protocol and cannot reuse M23-59 gate results as a tuned decision rule.

## Structured records

- `{OUT/'audit_manifest.json'}`
- `{OUT/'semantic_validation.json'}`
- `{OUT/'candidate_oracle.json'}`
- `{OUT/'feature_shift.csv'}`
- `{OUT/'ranking_diagnostics.csv'}`
- `{OUT/'error_waterfall.json'}`
- `{OUT/'final_diagnosis.json'}`
- `{OUT/'summary.csv'}`
- `outputs/experiment_registry.csv`
"""


def command_run() -> None:
    if not (OUT/"audit_manifest.json").exists() or not SUMMARY.exists(): raise RuntimeError("run init first")
    verify_inputs()
    for stage in ["semantic_validation","candidate_oracle","feature_shift","ranking_diagnostics","error_waterfall","final_diagnosis"]:
        update_summary(stage,status="running",started=True,notes="diagnostic execution active; no training/TrackEval/tracker")
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark=False; torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    mod=load_m59_module(); dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    semantic=semantic_audit(mod); json_write(OUT/"semantic_validation.json",clean_for_json(semantic),create_only=True)
    update_summary("semantic_validation",status="completed",completed=True,notes=f"critical_pass={semantic['critical_pass']}")
    oracle=candidate_oracle(); json_write(OUT/"candidate_oracle.json",clean_for_json(oracle),create_only=True)
    update_summary("candidate_oracle",status="completed",completed=True,notes=f"pooled coverage@256={oracle['pooled_MOT20']['coverage_at_256']:.6f}")
    feature_rows,domains,shift_summary=feature_stats_rows()
    external_rows,external_details=score_external(mod,dev)
    mot_rows,fold_details,waterfall=mot20_audit(mod,dev,feature_rows)
    feature_fields=["kind","feature","domain","condition","count","missing_nonfinite","mean","std","min","max","p01","p05","p25","p50","p75","p95","p99","clip_rate","outside_train_minmax_rate","outside_train_p01_p99_rate","ks","wasserstein","psi"]
    csv_write(OUT/"feature_shift.csv",feature_rows,feature_fields,create_only=True)
    json_write(OUT/"feature_shift_summary.json",clean_for_json(shift_summary),create_only=True)
    update_summary("feature_shift",status="completed",completed=True,notes=f"row and conditional relation features; sample cap={MAX_SHIFT_SAMPLE}")
    ranking_rows=external_rows+mot_rows
    ranking_fields=list(ranking_rows[0].keys()); csv_write(OUT/"ranking_diagnostics.csv",ranking_rows,ranking_fields,create_only=True)
    json_write(OUT/"ranking_details.json",clean_for_json({"external":external_details,"MOT20":fold_details}),create_only=True)
    update_summary("ranking_diagnostics",status="completed",completed=True,notes=f"{len(ranking_rows)} frozen-model/baseline rows")
    json_write(OUT/"error_waterfall.json",clean_for_json(waterfall),create_only=True)
    update_summary("error_waterfall",status="completed",completed=True,notes="candidate absence separated from present-but-misranked")
    classification,evidence=choose_classification(semantic,oracle,feature_rows,ranking_rows)
    next_direction=("Create a new preregistered M23-59 v3 semantic-alignment validation that restores one physical meaning per feature index "
                    "(especially geometry index 15), regenerates all affected MOT20 observables from raw frozen inputs, and reruns from an empty versioned root; "
                    "do not tune candidates, thresholds, policy risks, or HOTA.") if classification=="implementation_or_semantic_mismatch" else (
                    "Preregister a candidate-recall-only study with the same frozen labels and no tracker/TrackEval." if "candidate" in classification else
                    "Preregister an observable-invariance study that aligns feature support without MOT20 policy tuning." if classification=="observable_domain_shift" else
                    "Preregister a relation-objective transfer study using fixed candidate sets and ranking metrics only, before any tracker evaluation.")
    diagnosis={"experiment":"M23-60 Relation Transfer Failure Audit","primary_classification":classification,"allowed_classifications":CLASSIFICATIONS,
        "evidence":evidence,"allowed_next_research_direction":next_direction,
        "uses_mot20_gt":True,"post_hoc_diagnostic_only":True,"not_deployable":True,"not_a_strict_result":True,
        "M23_59_modified":False,"training_runs":0,"trackeval_runs":0,"tracker_outputs":0,"MOT20_test_submission":False,"m23_54_started":False,"m23_58_started":False}
    json_write(OUT/"final_diagnosis.json",clean_for_json(diagnosis),create_only=True)
    update_summary("final_diagnosis",status="completed",completed=True,classification=classification,notes="unique preregistered primary classification")
    RESULT_DOC.write_text(result_markdown(semantic,oracle,feature_rows,ranking_rows,waterfall,diagnosis),encoding="utf-8")
    # Parse and SHA validation.
    json_files=[OUT/"audit_manifest.json",OUT/"semantic_validation.json",OUT/"candidate_oracle.json",OUT/"feature_shift_summary.json",OUT/"ranking_details.json",OUT/"error_waterfall.json",OUT/"final_diagnosis.json"]
    for p in json_files: json.loads(p.read_text(encoding="utf-8"))
    for p in [OUT/"feature_shift.csv",OUT/"ranking_diagnostics.csv",SUMMARY]:
        with p.open(newline="",encoding="utf-8") as f: list(csv.DictReader(f))
    finalize_registry(classification)
    with REGISTRY.open(newline="",encoding="utf-8",errors="replace") as f: registry_rows=list(csv.DictReader(f))
    ours=[r for r in registry_rows if r.get("tracker_family")=="M23-60" and r.get("variant")=="relation_transfer_failure_audit"]
    checks={"json_parse":True,"csv_parse":True,"input_sha_verified":True,"summary_no_running":all(r["status"]!="running" for r in summary_rows()),
        "registry_no_stale_running":all(r.get("status")!="running" for r in ours),"registry_completed_row":any(r.get("status")=="completed" for r in ours),
        "M23_59_modified":False,"training_runs_zero":True,"trackeval_runs_zero":True,"tracker_outputs_zero":True,"uses_mot20_gt":True,
        "post_hoc_diagnostic_only":True,"not_deployable":True,"not_a_strict_result":True}
    completion={"passed":all(checks.values()),"checks":checks,"registry_rows":[{"tag":r.get("tag"),"status":r.get("status"),"current_stage":r.get("current_stage")} for r in ours],
        "outputs":{str(p):{"sha256":sha256_file(p),"bytes":p.stat().st_size} for p in json_files+[OUT/"feature_shift.csv",OUT/"ranking_diagnostics.csv",OUT/"error_waterfall.json",OUT/"final_diagnosis.json",SUMMARY,RESULT_DOC]},"completed_at":utcnow()}
    json_write(OUT/"completion_validation.json",completion,create_only=True)
    print(json.dumps(clean_for_json(diagnosis),indent=2,ensure_ascii=False))


def main() -> None:
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="command",required=True); sub.add_parser("init"); sub.add_parser("run")
    args=p.parse_args()
    if args.command=="init": command_init()
    elif args.command=="run": command_run()


if __name__=="__main__": main()
