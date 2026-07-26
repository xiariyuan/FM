#!/usr/bin/env python3
"""M23-64 frozen-pool pair reconstruction and from-scratch v3 training.

R62/R63 are immutable inputs. No MOT20 labels, trackers, TrackEval or HOTA.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import inspect
import json
import math
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd
import scipy
import torch
import torch.nn as nn

EXP_ID = "M23-64"
TITLE = "M23-59 v3 Frozen-Pool Paired Reconstruction and From-Scratch Relation Training"
ROOT = Path(".").resolve()
R62 = ROOT / "outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_gtfree_source_regeneration"
R63 = ROOT / "outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_supervision_join"
R64 = ROOT / "outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_pair_reconstruction_training"
SCRIPT = ROOT / "scripts/m23_research/m23_64_v3_pair_reconstruction_training.py"
PREREG = ROOT / "docs/m23_59_relation_pretrained_hierarchical_flow_v3_pair_reconstruction_training_prereg_20260723.md"
RESULT = ROOT / "docs/m23_59_relation_pretrained_hierarchical_flow_v3_pair_reconstruction_training_result_20260723.md"
REGISTRY = ROOT / "outputs/experiment_registry.csv"
V2_SCRIPT = ROOT / "scripts/m23_research/m23_59_relation_pretrained_hierarchical_flow_v2.py"
CONTRACT_HASH = "90cfab7d3a1fc87cc46b26441d3d883b9fc72f27d8852d1e2074b00e628428f5"
TRAIN_SEQS = ["MOT17-02", "MOT17-04", "MOT17-05", "MOT17-09", "MOT17-10"]
VAL_SEQS = ["MOT17-11", "MOT17-13"]
SEQS = TRAIN_SEQS + VAL_SEQS
SPLIT = {s: "train" for s in TRAIN_SEQS} | {s: "validation" for s in VAL_SEQS}
SEQ_GROUP = {"train": {s: i for i, s in enumerate(TRAIN_SEQS)}, "validation": {s: i for i, s in enumerate(VAL_SEQS)}}
TRUST_MIN_KNOWN = 2
TRUST_PURITY = 0.80
EXPECTED_VALID = {"MOT17-02": 42, "MOT17-04": 17, "MOT17-05": 17, "MOT17-09": 55,
                  "MOT17-10": 51, "MOT17-11": 63, "MOT17-13": 36}
EXPECTED_SPLIT = {"train": 182, "validation": 99}
SEEDS = [2359001, 2359002, 2359003]
EPOCHS = 30
EXPECTED_PARAMS = 881124
MAX_NODE_ROWS = 30
FEATURE_DIM = 144
GAP_BUCKETS = [("1-30", 1, 30), ("31-90", 31, 90), ("91-180", 91, 180), ("181-600", 181, 600)]
SUMMARY_FIELDS = ["experiment", "stage", "unit", "status", "started_at", "completed_at", "decision", "report", "notes"]
STAGES = ["preregistration", "input_reverification", "pair_diagnostic", "corrected_reconstruction",
          "tensor_validation", "stage_a_gate", "training_config", "seed_2359001", "seed_2359002",
          "seed_2359003", "checkpoint_selection", "stage_b_gate", "closure"]
SCOPE_ZERO = {"mot20_gt_reads": 0, "mot20_test_reads": 0, "mot20_test_submissions": 0,
              "teacher_reads": 0, "held_outer_reads": 0, "trackeval_runs": 0, "tracker_outputs": 0,
              "hota_evaluations": 0, "v2_checkpoint_loads": 0, "warm_starts": 0,
              "m23_54_starts": 0, "m23_58_starts": 0}
_V2 = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_id(*parts: Any) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:24]


def json_write(path: Path, obj: Any, *, create_only: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if create_only and path.exists():
        raise RuntimeError(f"refuse overwrite: {path}")
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def csv_write(path: Path, rows: list[dict[str, Any]], fields: list[str], *, create_only: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if create_only and path.exists():
        raise RuntimeError(f"refuse overwrite: {path}")
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    tmp.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_indices(v: Any) -> list[int]:
    if isinstance(v, str): return [int(x) for x in json.loads(v)]
    if isinstance(v, (list, tuple, np.ndarray)): return [int(x) for x in v]
    return []


def append_event(event: str, **payload: Any) -> None:
    R64.mkdir(parents=True, exist_ok=True)
    with (R64 / "protocol_events.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"time": utc_now(), "event": event, **payload}, sort_keys=True, ensure_ascii=False) + "\n")


def read_summary() -> list[dict[str, str]]:
    p = R64 / "summary.csv"
    if not p.exists(): return []
    with p.open(newline="", encoding="utf-8") as f: return list(csv.DictReader(f))


def update_stage(stage: str, status: str, *, unit: str = "", decision: str = "", report: str = "", notes: str = "") -> None:
    rows = read_summary(); now = utc_now(); found = False
    for r in rows:
        if r["stage"] == stage and (not unit or r["unit"] == unit):
            found = True
            if status == "running" and not r["started_at"]: r["started_at"] = now
            if status in {"completed", "failed", "prohibited_by_scope", "superseded"}:
                if not r["started_at"]: r["started_at"] = now
                r["completed_at"] = now
            r.update(status=status, decision=decision, report=report, notes=notes)
    if not found:
        rows.append({"experiment": EXP_ID, "stage": stage, "unit": unit, "status": status,
                     "started_at": now if status else "", "completed_at": now if status in {"completed","failed"} else "",
                     "decision": decision, "report": report, "notes": notes})
    csv_write(R64 / "summary.csv", rows, SUMMARY_FIELDS)


def registry_data() -> tuple[list[str], list[list[str]]]:
    with REGISTRY.open(newline="", encoding="utf-8", errors="replace") as f: data = list(csv.reader(f))
    return data[0], data[1:]


def registry_append(values: dict[str, Any]) -> None:
    header, rows = registry_data(); rows.append([str(values.get(c, "")) for c in header])
    tmp = REGISTRY.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    tmp.replace(REGISTRY)


def registry_update_running(stage: str, notes: str = "") -> None:
    header, rows = registry_data(); idx = {k: header.index(k) for k in ["tracker_family","status","current_stage","notes"]}
    for r in rows:
        if len(r) < len(header): r.extend([""]*(len(header)-len(r)))
        if r[idx["tracker_family"]] == EXP_ID and r[idx["status"]] == "running":
            r[idx["current_stage"]] = stage
            if notes: r[idx["notes"]] = notes
    tmp = REGISTRY.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(header); w.writerows(rows)
    tmp.replace(REGISTRY)


def registry_close(final: dict[str, Any]) -> None:
    header, rows = registry_data(); idx = {k: header.index(k) for k in ["tracker_family","status","current_stage","notes"]}
    for r in rows:
        if len(r) < len(header): r.extend([""]*(len(header)-len(r)))
        if r[idx["tracker_family"]] == EXP_ID and r[idx["status"]] == "running":
            r[idx["status"]] = "superseded"; r[idx["current_stage"]] = "superseded"
            r[idx["notes"]] = (r[idx["notes"]] + "; superseded by closed row").strip("; ")
    tmp = REGISTRY.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(header); w.writerows(rows)
    tmp.replace(REGISTRY)
    selection = final.get("selection", {})
    registry_append({
        "timestamp": utc_now(), "kind": "relation_training", "status": "completed" if final["decision"].startswith("PASS_") else "failed",
        "script": str(SCRIPT.relative_to(ROOT)), "dataset": "MOT17", "split": "sequence_disjoint_external",
        "tracker_family": EXP_ID, "variant": "v3_pair_reconstruction_from_scratch", "tag": "M23-64-closed",
        "run_root": str(R64.relative_to(ROOT)), "summary_csv": str((R64/"summary.csv").relative_to(ROOT)),
        "current_stage": "closed", "decision": final["decision"], "result_file": str(RESULT.relative_to(ROOT)),
        "artifact": str(RESULT.relative_to(ROOT)), "train_examples": final.get("pair_counts",{}).get("train",0),
        "val_examples": final.get("pair_counts",{}).get("validation",0), "seed": selection.get("seed",""),
        "selected_seed": selection.get("seed",""), "selected_epoch": selection.get("epoch",""),
        "checkpoint_sha256": selection.get("checkpoint_sha256",""), "best_metric": selection.get("composite",""),
        "training_runs": final.get("training_runs",0), "trackeval_runs": 0, "HOTA": "",
        "next_action": "M23-65 representation gate" if final.get("next_stage_authorized") else "none",
        "notes": f"corrected_pairs={final.get('pair_counts')}; TrackEval=0; HOTA=empty; next_stage_authorized={final.get('next_stage_authorized')}; result={RESULT.relative_to(ROOT)}",
    })


def load_v2():
    global _V2
    if _V2 is None:
        spec = importlib.util.spec_from_file_location("m23_64_v2_source", V2_SCRIPT)
        mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
        _V2 = mod
    return _V2


def proc_rchar() -> int:
    try:
        for line in Path(f"/proc/{os.getpid()}/io").read_text().splitlines():
            if line.startswith("rchar:"): return int(line.split(":",1)[1])
    except Exception: pass
    return 0


def peak_rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)


def peak_gpu_mb() -> float:
    return float(torch.cuda.max_memory_allocated() / (1024**2)) if torch.cuda.is_available() else 0.0


def git_head() -> str:
    return subprocess.run(["git","rev-parse","HEAD"], capture_output=True, text=True, check=True).stdout.strip()


def input_paths() -> list[Path]:
    return [
        R62/"feature_contract_v3_1.json", R62/"closure_validation.json", R62/"final_summary.json",
        R63/"closure_validation.json", R63/"final_summary.json", R63/"source_topology_manifest.json",
        R63/"candidate_pool.parquet", R63/"paired_candidate_pool.parquet", R63/"source_chunks.parquet",
        R63/"row_supervision.parquet", R63/"join_manifest.json", R63/"example_manifest_train.json",
        R63/"example_manifest_validation.json", R63/"examples_train.npz", R63/"examples_validation.npz",
        R63/"node_examples_train.parquet", R63/"node_examples_validation.parquet",
        R63/"relation_examples_train.parquet", R63/"relation_examples_validation.parquet",
        V2_SCRIPT, ROOT/"AGENTS.md",
    ]


def prereg_text() -> str:
    v2 = load_v2()
    return f"""# M23-64 preregistration — frozen-pool pair reconstruction and from-scratch relation training

Frozen before any M23-64 corrected example construction or optimizer step. Git HEAD is recorded in the input manifest. R62 and R63 are immutable inputs.

## Scope
Stage A reconstructs paired examples from the complete frozen R63 pool. Stage B starts automatically only if every Stage A gate passes. This experiment never reads MOT20 GT, teacher action, held-outer labels or MOT20 test; never generates a tracker; never runs TrackEval or HOTA; never loads an M23-59 v2 checkpoint; and never starts M23-54/M23-58.

## Frozen inputs
Required contract hash: `{CONTRACT_HASH}`. R63 must be closed as `FAIL_EXAMPLE_VALIDATION` with exactly one failed scientific gate: `train_minimum_support`; train paired support must be 2<5, while join semantics, topology, provenance, split isolation, unknown handling and scope guards pass. Every declared R62/R63 artifact is rehashed before Stage A and again at closure.

## Stage A diagnostic
Trust is fixed at known rows >= {TRUST_MIN_KNOWN}, majority purity >= {TRUST_PURITY:.2f}; identity keys must be sequence-qualified. All 140,000 rows of `paired_candidate_pool.parquet` are labeled. For edge1 a->b and edge2 c->d, `valid_paired_positive` is exactly: all four endpoint chunks trusted; a.identity=b.identity; c.identity=d.identity; a.identity!=c.identity; and edge1, edge2, cross1 and cross2 exist in the pre-label frozen candidate pool. No pair or edge may be created, replaced or searched outside the pool.

The prior diagnostic is frozen only as a replication target, not as an input label: train=182, validation=99; per sequence `{json.dumps(EXPECTED_VALID, sort_keys=True)}`. Any mismatch closes as `FAIL_PAIR_DIAGNOSTIC_REPLICATION` without changing rules.

Corrected examples contain every unique valid pair exactly once in pair_id order, without random choice, oversampling or duplication. M23-63 node and relation NPZ members and metadata must remain byte-exact; only pair_x, pair_mask and paired metadata change. Each NPZ member is loaded once into memory for full provenance validation. Stage A records wall time, rchar delta and peak RSS; TB-scale repeated member reads fail the performance guard.

## Stage A gates
M23-63 input closure and SHA valid; diagnostic exact; 182/99 pair counts and nonzero support in every physical sequence; train pair support >=5 and validation >=1; node/relation byte-exact; all pair IDs and all four edges frozen; no topology or input mutation; tensors finite; prefix masks; full row and pair provenance; physical split isolation; no unknown-as-negative; all scope counts zero; no checkpoint loaded; training not started.

## Stage B frozen protocol
From scratch only: `from_scratch=true`, `v2_checkpoint_reuse=false`, `warm_start=false`. Model is v2 `HierarchicalRelationEncoder`, exact parameter count {EXPECTED_PARAMS}. Seeds `{SEEDS}`, epochs={EPOCHS}, optimizer AdamW, LR={v2.EXTERNAL_LR}, weight decay={v2.WEIGHT_DECAY}, node/relation batch sizes={v2.BATCH_NODES}/{v2.BATCH_RELATIONS}, gradient clipping 5.0, gap buckets `{v2.GAP_BUCKETS}`. Loss weights are `{json.dumps(v2.LOSS_WEIGHTS, sort_keys=True)}`. Group-DRO update, class-balanced focal losses, listwise relation losses, catastrophic-risk terms, edit sparsity, batch permutation and modulo cycling are inherited exactly from v2.

The unchanged frozen relation example set is adapted into v2 triplets deterministically. Each labeled positive edge uses the highest frozen candidate_score labeled negative with the same source as outgoing negative and the highest score labeled negative with the same destination as incoming negative; ties use candidate_id. No candidate is added and no GT search occurs.

Checkpoint selection composite is fixed to: 0.30 node PR-AUC + 0.35 conditional-boundary PR-AUC + 0.25 mean(outgoing R@1,incoming R@1) + 0.10*(1-catastrophic false-link rate). Within a seed, ties select earlier epoch; across seeds, ties select lower seed then earlier epoch. No HOTA/MOT20 metric participates.

The v2 source defines selection/health metrics but no separate external acceptance threshold. Therefore Stage B requires all 90 epoch records complete, finite, reloadable, exact parameter count, stable example/checkpoint SHA, calculable validation metrics and nonconstant score outputs. No post-hoc numerical acceptance threshold may be invented.

## Decisions
Stage A failure closes using the first applicable root cause: `FAIL_M23_63_INPUT_REVERIFICATION`, `FAIL_PAIR_DIAGNOSTIC_REPLICATION`, `FAIL_CORRECTED_PAIR_RECONSTRUCTION`, `FAIL_NONPAIRED_INVARIANCE`, `FAIL_TENSOR_PROVENANCE`, `FAIL_SPLIT_LEAKAGE`, `FAIL_SCOPE_GUARD`. Stage B failure closes as `FAIL_TRAINING_NONFINITE`, `FAIL_TRAINING_INCOMPLETE`, `FAIL_CHECKPOINT_SELECTION`, `FAIL_EXTERNAL_VALIDATION_HEALTH`, or `FAIL_SCOPE_GUARD`. Full pass is `PASS_V3_FROM_SCRATCH_RELATION_TRAINING` and authorizes only a new M23-65 to reverify checkpoint/observable SHA and run the MOT20 representation gate.
"""


def assert_unoccupied() -> None:
    if R64.exists() or PREREG.exists() or RESULT.exists(): raise RuntimeError("M23-64 path occupied")
    header, rows = registry_data(); i = header.index("tracker_family")
    if any(len(r)>i and r[i]==EXP_ID for r in rows): raise RuntimeError("M23-64 registry occupied")


def command_init() -> None:
    assert_unoccupied()
    missing=[str(p) for p in input_paths() if not p.is_file()]
    if missing: raise RuntimeError(f"missing frozen inputs: {missing}")
    R64.mkdir(parents=True, exist_ok=False)
    PREREG.write_text(prereg_text(), encoding="utf-8")
    frozen=[{"path":str(p.relative_to(ROOT)),"sha256":sha256_file(p),"size":p.stat().st_size} for p in input_paths()]
    inp={"experiment_id":EXP_ID,"created_at":utc_now(),"git_head":git_head(),"contract_hash_required":CONTRACT_HASH,
         "r62_immutable":True,"r63_immutable":True,"frozen_inputs":frozen,
         "scope":{**SCOPE_ZERO,"training_runs_before_stage_a":0},
         "allowed_domains":["R62 frozen MOT17 features","R63 frozen topology/candidates/supervision/examples","v2 source definitions"],
         "prohibited_domains":["MOT20 GT","MOT20 teacher","held outer","MOT20 test","tracker","TrackEval","HOTA","v2 checkpoints"]}
    json_write(R64/"input_manifest.json",inp,create_only=True)
    v2=load_v2(); model=v2.HierarchicalRelationEncoder(); params=v2.parameter_count(model)
    impl={"experiment_id":EXP_ID,"created_at":utc_now(),"script":str(SCRIPT.relative_to(ROOT)),"script_sha256":sha256_file(SCRIPT),
          "prereg":str(PREREG.relative_to(ROOT)),"prereg_sha256":sha256_file(PREREG),
          "input_manifest_sha256":sha256_file(R64/"input_manifest.json"),"git_head":inp["git_head"],
          "python":sys.version,"platform":platform.platform(),"numpy":np.__version__,"pandas":pd.__version__,
          "torch":torch.__version__,"scipy":scipy.__version__,"v2_script_sha256":sha256_file(V2_SCRIPT),
          "model_parameter_count":params,"expected_parameter_count":EXPECTED_PARAMS,
          "v2_training_source_functions":{n:sha256_text(inspect.getsource(getattr(v2,n))) for n in
              ["HierarchicalRelationEncoder","group_weighted","training_objective","validation_metrics","sample_indices","set_determinism"]},
          "constants":{"trust_min_known":TRUST_MIN_KNOWN,"trust_purity":TRUST_PURITY,"expected_valid":EXPECTED_VALID,
                       "seeds":SEEDS,"epochs":EPOCHS,"learning_rate":v2.EXTERNAL_LR,"weight_decay":v2.WEIGHT_DECAY,
                       "batch_nodes":v2.BATCH_NODES,"batch_relations":v2.BATCH_RELATIONS,"loss_weights":v2.LOSS_WEIGHTS,
                       "gap_buckets":v2.GAP_BUCKETS,"parameter_count":params},
          "from_scratch":True,"v2_checkpoint_reuse":False,"warm_start":False}
    if params != EXPECTED_PARAMS: raise RuntimeError(f"unexpected model params {params}")
    json_write(R64/"implementation_manifest.json",impl,create_only=True)
    now=utc_now(); rows=[]
    for stage in STAGES:
        status="completed" if stage=="preregistration" else ("running" if stage=="input_reverification" else "pending")
        rows.append({"experiment":EXP_ID,"stage":stage,"unit":"","status":status,
                     "started_at":now if status in {"completed","running"} else "","completed_at":now if status=="completed" else "",
                     "decision":"pass" if status=="completed" else "","report":str(PREREG.relative_to(ROOT)) if status=="completed" else "",
                     "notes":"frozen before corrected examples and training" if stage=="preregistration" else ""})
    csv_write(R64/"summary.csv",rows,SUMMARY_FIELDS,create_only=True)
    (R64/"protocol_events.jsonl").touch(exist_ok=False)
    append_event("preregistered",prereg_sha256=impl["prereg_sha256"],input_manifest_sha256=impl["input_manifest_sha256"])
    append_event("implementation_frozen",script_sha256=impl["script_sha256"],implementation_manifest_sha256=sha256_file(R64/"implementation_manifest.json"))
    registry_append({"timestamp":utc_now(),"kind":"relation_training","status":"running","script":str(SCRIPT.relative_to(ROOT)),
                     "dataset":"MOT17","split":"sequence_disjoint_external","tracker_family":EXP_ID,
                     "variant":"v3_pair_reconstruction_from_scratch","tag":"M23-64-running","run_root":str(R64.relative_to(ROOT)),
                     "summary_csv":str((R64/"summary.csv").relative_to(ROOT)),"current_stage":"input_reverification","decision":"pending",
                     "training_runs":0,"trackeval_runs":0,"HOTA":"","notes":"R62/R63 immutable; corrected examples not yet built; training not started"})
    print(json.dumps({"status":"initialized","script_sha256":impl["script_sha256"],"prereg_sha256":impl["prereg_sha256"],
                      "input_manifest_sha256":impl["input_manifest_sha256"],"parameter_count":params},indent=2))


def verify_frozen_implementation() -> dict[str,bool]:
    m=read_json(R64/"implementation_manifest.json")
    c={"script":sha256_file(SCRIPT)==m["script_sha256"],"prereg":sha256_file(PREREG)==m["prereg_sha256"],
       "input_manifest":sha256_file(R64/"input_manifest.json")==m["input_manifest_sha256"],"git_head":git_head()==m["git_head"]}
    if not all(c.values()): raise RuntimeError(f"frozen implementation drift: {c}")
    return c


def reverify_inputs() -> dict[str,Any]:
    verify_frozen_implementation(); inp=read_json(R64/"input_manifest.json")
    shas=[]
    for x in inp["frozen_inputs"]:
        p=ROOT/x["path"]; actual=sha256_file(p)
        shas.append({"path":x["path"],"expected":x["sha256"],"actual":actual,"match":actual==x["sha256"]})
    r62c=read_json(R62/"closure_validation.json"); r62f=read_json(R62/"final_summary.json")
    r63c=read_json(R63/"closure_validation.json"); r63f=read_json(R63/"final_summary.json")
    failed=[k for k,v in r63f.get("gates",{}).items() if not v]
    checks={"all_sha_match":all(x["match"] for x in shas),
            "contract_exact":read_json(R62/"feature_contract_v3_1.json")["aggregate"]["contract_hash"]==CONTRACT_HASH,
            "r62_closed_valid":r62c.get("passed") is True and r62f.get("status")=="closed",
            "r63_closed_expected_failure":r63f.get("status")=="closed" and r63f.get("decision")=="FAIL_EXAMPLE_VALIDATION",
            "r63_only_train_support_failed":failed==["train_minimum_support"],
            "r63_specific_paired_failure":r63f.get("train_counts",{}).get("paired_replacement")==2 and r63f.get("validation_counts",{}).get("paired_replacement")==3,
            "r63_other_gates_positive":all(v for k,v in r63f.get("gates",{}).items() if k!="train_minimum_support"),
            "r63_scope_zero":all(r63f.get(k)==0 for k in ["mot20_gt_reads","mot20_test_reads","mot20_test_submissions","teacher_reads","held_outer_reads","training_runs","optimizer_steps","checkpoint_outputs","trackeval_runs","tracker_outputs","m23_54_starts","m23_58_starts"]),
            "no_r63_authorization":not (R63/"next_stage_authorization.json").exists()}
    out={"experiment_id":EXP_ID,"checked_at":utc_now(),"checks":checks,"passed":all(checks.values()),"sha_checks":shas,
         "r62_closure_sha256":sha256_file(R62/"closure_validation.json"),"r62_final_sha256":sha256_file(R62/"final_summary.json"),
         "r63_closure_sha256":sha256_file(R63/"closure_validation.json"),"r63_final_sha256":sha256_file(R63/"final_summary.json"),
         "r63_failed_gates":failed}
    json_write(R64/"m23_63_reverification.json",out)
    return out


def trusted_chunks(chunks: pd.DataFrame, labels: pd.DataFrame) -> dict[str,dict[str,Any]]:
    lookup={(str(r.sequence),int(r.row_index)):r for r in labels.itertuples()}; out={}
    for c in chunks.itertuples():
        ids=parse_indices(c.row_indices); known=[]
        for rid in ids:
            lab=lookup[(c.sequence,rid)]
            if lab.supervision_status=="matched": known.append(lab.gt_identity_key)
        counts=Counter(known); majority,n=counts.most_common(1)[0] if counts else ("",0)
        purity=n/max(len(known),1) if known else 0.0
        trusted=len(known)>=TRUST_MIN_KNOWN and purity>=TRUST_PURITY and majority.startswith(c.sequence+":gt:")
        out[c.chunk_id]={"sequence":c.sequence,"split":c.split,"row_indices":ids,"known_rows":len(known),"purity":purity,
                         "trusted":trusted,"identity":majority if trusted else ""}
    return out


def trace_row(row: pd.Series, edge_lookup: dict[str,Any], chunks: dict[str,dict[str,Any]], reason: str) -> dict[str,Any]:
    e1=edge_lookup.get(row.edge1_id); e2=edge_lookup.get(row.edge2_id)
    ids=[]
    if e1 is not None and e2 is not None:
        for cid in [e1.src_chunk_id,e1.dst_chunk_id,e2.src_chunk_id,e2.dst_chunk_id]: ids.append(chunks[cid]["identity"])
    return {"reason":reason,"sequence":row.sequence,"pair_id":row.pair_id,"edge1_id":row.edge1_id,"edge2_id":row.edge2_id,
            "cross1_id":row.cross1_id,"cross2_id":row.cross2_id,"endpoint_identities":ids,
            "m23_63_example_selected":bool(row.example_selected)}


def build_pair_supervision() -> tuple[pd.DataFrame,dict[str,Any]]:
    chunks_df=pd.read_parquet(R63/"source_chunks.parquet"); labels=pd.read_parquet(R63/"row_supervision.parquet")
    edges=pd.read_parquet(R63/"candidate_pool.parquet"); pairs=pd.read_parquet(R63/"paired_candidate_pool.parquet")
    cinfo=trusted_chunks(chunks_df,labels); edge_lookup={r.candidate_id:r for r in edges.itertuples()}; edge_ids=set(edge_lookup)
    rows=[]; traces=defaultdict(list); first_difference=None
    for p in pairs.sort_values(["sequence","pair_id"],kind="mergesort").itertuples(index=False):
        edge_present=[x in edge_ids for x in [p.edge1_id,p.edge2_id,p.cross1_id,p.cross2_id]]
        e1=edge_lookup.get(p.edge1_id); e2=edge_lookup.get(p.edge2_id)
        if e1 is None or e2 is None:
            a=b=c=d={"trusted":False,"identity":"","row_indices":[]}; endpoint_trusted=False
        else:
            a,b,c,d=[cinfo[x] for x in [e1.src_chunk_id,e1.dst_chunk_id,e2.src_chunk_id,e2.dst_chunk_id]]
            endpoint_trusted=all(x["trusted"] for x in [a,b,c,d])
        edge1_consistent=endpoint_trusted and a["identity"]==b["identity"]
        edge2_consistent=endpoint_trusted and c["identity"]==d["identity"]
        separated=endpoint_trusted and a["identity"]!=c["identity"]
        valid=bool(endpoint_trusted and edge1_consistent and edge2_consistent and separated and all(edge_present))
        reason="valid" if valid else ("cross_edge_missing" if not all(edge_present) else "untrusted_endpoint" if not endpoint_trusted else
                                       "original_edge_identity_mismatch" if not (edge1_consistent and edge2_consistent) else "same_identity_collision")
        rec={"sequence":p.sequence,"split":p.split,"pair_id":p.pair_id,"edge1_id":p.edge1_id,"edge2_id":p.edge2_id,
             "cross1_id":p.cross1_id,"cross2_id":p.cross2_id,"gap_bucket_index":int(p.gap_bucket_index),
             "m23_63_example_selected":bool(p.example_selected),"all_four_edges_present":all(edge_present),
             "all_endpoints_trusted":endpoint_trusted,"edge1_original_consistent":edge1_consistent,
             "edge2_original_consistent":edge2_consistent,"original_identity_separated":separated,
             "edge1_identity":a["identity"] if e1 is not None else "","edge2_identity":c["identity"] if e2 is not None else "",
             "valid_paired_positive":valid,"diagnostic_reason":reason}
        rows.append(rec)
        if len(traces[reason])<20: traces[reason].append(rec)
    df=pd.DataFrame(rows)
    df.to_parquet(R64/"pair_supervision.parquet",index=False)
    stats=[]
    for seq in SEQS:
        q=df[df.sequence==seq]; valid=q[q.valid_paired_positive]
        stats.append({"sequence":seq,"split":SPLIT[seq],"pool_total":len(q),"valid_total":len(valid),
                      "m23_63_selected_valid":int(valid.m23_63_example_selected.sum()),
                      "valid_not_selected":int((~valid.m23_63_example_selected).sum()),"pair_id_unique":not q.pair_id.duplicated().any(),
                      "untrusted_endpoint":int((q.diagnostic_reason=="untrusted_endpoint").sum()),
                      "same_identity_collision":int((q.diagnostic_reason=="same_identity_collision").sum()),
                      "original_edge_identity_mismatch":int((q.diagnostic_reason=="original_edge_identity_mismatch").sum()),
                      "cross_edge_missing":int((q.diagnostic_reason=="cross_edge_missing").sum())})
    for split in ["train","validation","all"]:
        q=df if split=="all" else df[df.split==split]; valid=q[q.valid_paired_positive]
        stats.append({"sequence":f"__{split}__","split":split,"pool_total":len(q),"valid_total":len(valid),
                      "m23_63_selected_valid":int(valid.m23_63_example_selected.sum()),"valid_not_selected":int((~valid.m23_63_example_selected).sum()),
                      "pair_id_unique":not q.pair_id.duplicated().any(),"untrusted_endpoint":int((q.diagnostic_reason=="untrusted_endpoint").sum()),
                      "same_identity_collision":int((q.diagnostic_reason=="same_identity_collision").sum()),
                      "original_edge_identity_mismatch":int((q.diagnostic_reason=="original_edge_identity_mismatch").sum()),
                      "cross_edge_missing":int((q.diagnostic_reason=="cross_edge_missing").sum())})
    csv_write(R64/"pair_supervision_statistics.csv",stats,list(stats[0].keys()))
    actual={seq:int(df[(df.sequence==seq)&df.valid_paired_positive].shape[0]) for seq in SEQS}
    exact=actual==EXPECTED_VALID and sum(actual[s] for s in TRAIN_SEQS)==EXPECTED_SPLIT["train"] and sum(actual[s] for s in VAL_SEQS)==EXPECTED_SPLIT["validation"]
    if not exact:
        for seq in SEQS:
            if actual[seq]!=EXPECTED_VALID[seq]:
                first_difference={"sequence":seq,"expected":EXPECTED_VALID[seq],"actual":actual[seq]}; break
    trace_payload={"experiment_id":EXP_ID,"saved_at":utc_now(),"trace_categories":dict(traces),
                   "per_sequence_valid_first":{seq:df[(df.sequence==seq)&df.valid_paired_positive].head(1).to_dict("records") for seq in SEQS},
                   "zero_count_categories_are_explicit":True}
    json_write(R64/"pair_manual_traces.json",trace_payload)
    manifest={"experiment_id":EXP_ID,"created_at":utc_now(),"complete_frozen_pool_rows":len(df),"rules":{"trust_min_known":TRUST_MIN_KNOWN,"trust_purity":TRUST_PURITY,
              "valid_predicate":"a,b,c,d trusted and a.id=b.id and c.id=d.id and a.id!=c.id and all four frozen edges present"},
              "actual_per_sequence":actual,"expected_per_sequence":EXPECTED_VALID,"actual_train":sum(actual[s] for s in TRAIN_SEQS),
              "actual_validation":sum(actual[s] for s in VAL_SEQS),"expected_split":EXPECTED_SPLIT,"diagnostic_exact":exact,
              "first_difference":first_difference,"pair_id_unique":not df.pair_id.duplicated().any(),
              "inputs":{"chunks_sha256":sha256_file(R63/"source_chunks.parquet"),"candidate_pool_sha256":sha256_file(R63/"candidate_pool.parquet"),
                        "paired_pool_sha256":sha256_file(R63/"paired_candidate_pool.parquet"),"supervision_sha256":sha256_file(R63/"row_supervision.parquet")},
              "artifacts":{"pair_supervision_sha256":sha256_file(R64/"pair_supervision.parquet"),
                           "statistics_sha256":sha256_file(R64/"pair_supervision_statistics.csv"),"traces_sha256":sha256_file(R64/"pair_manual_traces.json")}}
    json_write(R64/"pair_reconstruction_manifest.json",manifest)
    return df,manifest


def padded(features: np.ndarray, ids: list[int]) -> tuple[np.ndarray,np.ndarray]:
    q=[int(x) for x in ids][:MAX_NODE_ROWS]
    x=np.zeros((MAX_NODE_ROWS,FEATURE_DIM),np.float16); m=np.zeros(MAX_NODE_ROWS,np.uint8)
    if q: x[:len(q)]=np.asarray(features[q],np.float16); m[:len(q)]=1
    return x,m


def load_npz_once(path: Path) -> dict[str,np.ndarray]:
    with np.load(path) as archive:
        return {name:archive[name] for name in archive.files}


def reconstruct_examples(pair_df: pd.DataFrame) -> dict[str,Any]:
    chunks_df=pd.read_parquet(R63/"source_chunks.parquet"); chunks={r.chunk_id:r for r in chunks_df.itertuples()}
    edges_df=pd.read_parquet(R63/"candidate_pool.parquet"); edges={r.candidate_id:r for r in edges_df.itertuples()}
    labels=pd.read_parquet(R63/"row_supervision.parquet"); label_lookup={(r.sequence,int(r.row_index)):r.row_label_id for r in labels.itertuples()}
    outputs={}
    for split in ["train","validation"]:
        old_npz=load_npz_once(R63/f"examples_{split}.npz")
        valid=pair_df[(pair_df.split==split)&pair_df.valid_paired_positive].sort_values(["sequence","pair_id"],kind="mergesort")
        pair_x=[]; pair_m=[]; meta=[]; feature_cache={s:np.load(R62/f"observables/MOT17/{s}/row_features.f16.npy",mmap_mode="r") for s in SEQS if SPLIT[s]==split}
        for p in valid.itertuples(index=False):
            e1=edges[p.edge1_id]; e2=edges[p.edge2_id]
            cids=[e1.src_chunk_id,e1.dst_chunk_id,e2.src_chunk_id,e2.dst_chunk_id]
            xs=[]; ms=[]; all_rows=[]; all_labels=[]
            for cid in cids:
                ids=parse_indices(chunks[cid].row_indices); x,m=padded(feature_cache[p.sequence],ids)
                xs.append(x); ms.append(m); all_rows.extend(ids); all_labels.extend(label_lookup[(p.sequence,r)] for r in ids)
            idx=len(pair_x); pair_x.append(np.stack(xs)); pair_m.append(np.stack(ms))
            meta.append({"example_id":f"{p.sequence}:paired-v3:{stable_id(p.pair_id,sha256_file(R64/'pair_supervision.parquet'))}",
                         "sequence":p.sequence,"split":split,"pair_id":p.pair_id,"edge1_id":p.edge1_id,"edge2_id":p.edge2_id,
                         "cross1_id":p.cross1_id,"cross2_id":p.cross2_id,"source_row_indices":json.dumps(all_rows),
                         "source_row_keys":json.dumps([f"{p.sequence}:row:{r}" for r in all_rows]),
                         "label_sidecar_ids":json.dumps(all_labels),"tensor_index":idx,"ignore_reason":"",
                         "provenance_sha256":sha256_text("|".join([sha256_file(R64/'pair_supervision.parquet'),sha256_file(R63/'candidate_pool.parquet'),sha256_file(R63/'row_supervision.parquet')]))})
        corrected={k:v for k,v in old_npz.items() if k not in {"pair_x","pair_mask"}}
        corrected["pair_x"]=np.asarray(pair_x,np.float16); corrected["pair_mask"]=np.asarray(pair_m,np.uint8)
        np.savez(R64/f"examples_{split}.npz",**corrected)
        for kind in ["node","relation"]:
            shutil.copyfile(R63/f"{kind}_examples_{split}.parquet",R64/f"{kind}_examples_{split}.parquet")
        pd.DataFrame(meta).to_parquet(R64/f"paired_examples_{split}.parquet",index=False)
        counts=read_json(R63/f"example_manifest_{split}.json")["counts"].copy()
        counts["paired_replacement"]=len(meta); counts["paired_examples"]=len(meta)
        manifest={"experiment_id":EXP_ID,"split":split,"status":"frozen","created_at":utc_now(),"counts":counts,
                  "corrected_pair_count":len(meta),"all_valid_pairs_used":len(meta)==len(valid),"unique_pair_ids":len({x["pair_id"] for x in meta})==len(meta),
                  "nonpaired_source":"byte-exact M23-63 members and metadata","pair_source":"all valid rows of frozen paired pool",
                  "artifacts":{n:sha256_file(R64/n) for n in [f"examples_{split}.npz",f"node_examples_{split}.parquet",f"relation_examples_{split}.parquet",f"paired_examples_{split}.parquet"]},
                  "input_shas":{"m23_63_examples":sha256_file(R63/f"examples_{split}.npz"),"pair_supervision":sha256_file(R64/"pair_supervision.parquet")}}
        json_write(R64/f"example_manifest_{split}.json",manifest)
        outputs[split]=manifest
    return outputs


def mask_prefix(mask: np.ndarray) -> bool:
    if mask.size==0:return True
    flat=mask.reshape(-1,mask.shape[-1]).astype(np.int8)
    return bool(np.all(np.diff(flat,axis=1)<=0))


def validate_corrected_examples() -> tuple[dict[str,Any],dict[str,Any],dict[str,Any],dict[str,Any]]:
    started=time.perf_counter(); r0=proc_rchar()
    chunks_df=pd.read_parquet(R63/"source_chunks.parquet"); chunks={r.chunk_id:r for r in chunks_df.itertuples()}
    edges_df=pd.read_parquet(R63/"candidate_pool.parquet"); edges={r.candidate_id:r for r in edges_df.itertuples()}; edge_ids=set(edges)
    pair_pool=pd.read_parquet(R63/"paired_candidate_pool.parquet"); pair_ids=set(pair_pool.pair_id)
    labels=pd.read_parquet(R63/"row_supervision.parquet"); label_by_id=set(labels.row_label_id); label_by_row={(r.sequence,int(r.row_index)):r for r in labels.itertuples()}
    nonpaired={}; provenance={}; split_ids={}
    total_examples={"node":0,"relation":0,"pair":0}; member_reads={}
    for split in ["train","validation"]:
        old=load_npz_once(R63/f"examples_{split}.npz"); new=load_npz_once(R64/f"examples_{split}.npz")
        member_reads[split]={"m23_63_members":list(old),"m23_64_members":list(new),"each_member_loaded_once":True}
        nonpair_members=[k for k in old if k not in {"pair_x","pair_mask"}]
        array_checks={k:np.array_equal(old[k],new[k]) for k in nonpair_members}
        meta_checks={kind:sha256_file(R63/f"{kind}_examples_{split}.parquet")==sha256_file(R64/f"{kind}_examples_{split}.parquet") for kind in ["node","relation"]}
        nonpaired[split]={"array_member_checks":array_checks,"metadata_sha_checks":meta_checks,
                          "all_nonpaired_exact":all(array_checks.values()) and all(meta_checks.values()),
                          "pair_tensor_changed":not np.array_equal(old["pair_x"],new["pair_x"]) or old["pair_x"].shape!=new["pair_x"].shape,
                          "pair_metadata_changed":sha256_file(R63/f"paired_examples_{split}.parquet")!=sha256_file(R64/f"paired_examples_{split}.parquet")}
        node=pd.read_parquet(R64/f"node_examples_{split}.parquet"); rel=pd.read_parquet(R64/f"relation_examples_{split}.parquet"); pair=pd.read_parquet(R64/f"paired_examples_{split}.parquet")
        total_examples["node"]+=len(node); total_examples["relation"]+=len(rel); total_examples["pair"]+=len(pair)
        feature_cache={s:np.load(R62/f"observables/MOT17/{s}/row_features.f16.npy",mmap_mode="r") for s in SEQS if SPLIT[s]==split}
        node_ok=True; rel_ok=True; pair_ok=True; ids_unique=True; unknown_negative=True; label_ids_ok=True
        for r in node.itertuples():
            ids=parse_indices(r.source_row_indices); x,m=padded(feature_cache[r.sequence],ids); i=int(r.tensor_index)
            node_ok &= i<len(new["node_x"]) and np.array_equal(new["node_x"][i],x) and np.array_equal(new["node_mask"][i],m)
            label_ids_ok &= all(x in label_by_id for x in json.loads(r.label_sidecar_ids))
        for r in rel.itertuples():
            e=edges.get(r.candidate_id)
            if e is None: rel_ok=False; continue
            sids=parse_indices(chunks[e.src_chunk_id].row_indices); dids=parse_indices(chunks[e.dst_chunk_id].row_indices)
            sx,sm=padded(feature_cache[r.sequence],sids); dx,dm=padded(feature_cache[r.sequence],dids); i=int(r.tensor_index)
            rel_ok &= i<len(new["relation_src_x"]) and np.array_equal(new["relation_src_x"][i],sx) and np.array_equal(new["relation_dst_x"][i],dx)
            rel_ok &= np.array_equal(new["relation_src_mask"][i],sm) and np.array_equal(new["relation_dst_mask"][i],dm)
            if int(r.relation_label)==0: unknown_negative &= bool(r.src_trusted_gt_identity_key) and bool(r.dst_trusted_gt_identity_key)
            label_ids_ok &= all(x in label_by_id for x in json.loads(r.label_sidecar_ids))
        seen=set()
        for r in pair.itertuples():
            ids_unique &= r.pair_id not in seen; seen.add(r.pair_id)
            if r.pair_id not in pair_ids or not all(x in edge_ids for x in [r.edge1_id,r.edge2_id,r.cross1_id,r.cross2_id]): pair_ok=False; continue
            e1=edges[r.edge1_id]; e2=edges[r.edge2_id]; xs=[]; ms=[]
            for cid in [e1.src_chunk_id,e1.dst_chunk_id,e2.src_chunk_id,e2.dst_chunk_id]:
                x,m=padded(feature_cache[r.sequence],parse_indices(chunks[cid].row_indices)); xs.append(x); ms.append(m)
            i=int(r.tensor_index); pair_ok &= i<len(new["pair_x"]) and np.array_equal(new["pair_x"][i],np.stack(xs)) and np.array_equal(new["pair_mask"][i],np.stack(ms))
            label_ids_ok &= all(x in label_by_id for x in json.loads(r.label_sidecar_ids))
        finite=all(np.isfinite(v).all() for k,v in new.items() if np.issubdtype(v.dtype,np.floating))
        masks=all(mask_prefix(new[k]) for k in ["node_mask","relation_src_mask","relation_dst_mask","pair_mask"])
        index_stable=(node.tensor_index.tolist()==list(range(len(node))) and rel.tensor_index.tolist()==list(range(len(rel))) and pair.tensor_index.tolist()==list(range(len(pair))))
        provenance[split]={"node_tensor_provenance":node_ok,"relation_tensor_provenance":rel_ok,"pair_tensor_provenance":pair_ok,
                           "pair_ids_unique":ids_unique,"label_sidecar_ids_valid":label_ids_ok,"all_arrays_finite":finite,
                           "mask_prefix_valid":masks,"stable_tensor_index":index_stable,"no_unknown_as_negative":unknown_negative}
        split_ids[split]={"sequences":set(node.sequence)|set(rel.sequence)|set(pair.sequence),
                          "examples":set(node.example_id)|set(rel.example_id)|set(pair.example_id),
                          "rows":{x for frame in [node,rel,pair] for s in frame.source_row_keys for x in json.loads(s)},
                          "candidates":set(rel.candidate_id)|{x for r in pair.itertuples() for x in [r.edge1_id,r.edge2_id,r.cross1_id,r.cross2_id]}}
    nonpaired_report={"experiment_id":EXP_ID,"created_at":utc_now(),"splits":nonpaired,
                      "passed":all(v["all_nonpaired_exact"] and v["pair_tensor_changed"] and v["pair_metadata_changed"] for v in nonpaired.values()),
                      "r63_inputs_unchanged":all(sha256_file(ROOT/x["path"])==x["sha256"] for x in read_json(R64/"input_manifest.json")["frozen_inputs"])}
    json_write(R64/"nonpaired_invariance_validation.json",nonpaired_report)
    prov_checks={f"{split}_{k}":v for split,d in provenance.items() for k,v in d.items()}
    prov_report={"experiment_id":EXP_ID,"created_at":utc_now(),"checks":prov_checks,"passed":all(prov_checks.values()),"splits":provenance,
                 "npz_loading_rule":"each member materialized once per archive before loops","member_reads":member_reads,"example_counts":total_examples}
    json_write(R64/"tensor_provenance_validation.json",prov_report)
    leak_checks={"physical_sequences_disjoint":split_ids["train"]["sequences"].isdisjoint(split_ids["validation"]["sequences"]),
                 "example_ids_disjoint":split_ids["train"]["examples"].isdisjoint(split_ids["validation"]["examples"]),
                 "source_rows_disjoint":split_ids["train"]["rows"].isdisjoint(split_ids["validation"]["rows"]),
                 "candidate_ids_disjoint":split_ids["train"]["candidates"].isdisjoint(split_ids["validation"]["candidates"]),
                 "expected_train_sequences":split_ids["train"]["sequences"]==set(TRAIN_SEQS),
                 "expected_validation_sequences":split_ids["validation"]["sequences"]==set(VAL_SEQS)}
    leakage={"experiment_id":EXP_ID,"created_at":utc_now(),"checks":leak_checks,"passed":all(leak_checks.values()),
             "train_sequences":sorted(split_ids["train"]["sequences"]),"validation_sequences":sorted(split_ids["validation"]["sequences"])}
    json_write(R64/"leakage_validation.json",leakage)
    wall=time.perf_counter()-started; rdelta=proc_rchar()-r0
    perf={"experiment_id":EXP_ID,"created_at":utc_now(),"validation_wall_seconds":wall,"process_rchar_delta":rdelta,
          "process_rchar_delta_gib":rdelta/(1024**3),"peak_rss_mb":peak_rss_mb(),"example_counts":total_examples,
          "npz_member_read_policy":"one full materialization per member per archive","tb_scale_repeated_read":rdelta>=1024**4,
          "performance_passed":rdelta<1024**4}
    json_write(R64/"performance_validation.json",perf)
    corrected={"experiment_id":EXP_ID,"created_at":utc_now(),"checks":{
        "nonpaired_invariance":nonpaired_report["passed"],"tensor_provenance":prov_report["passed"],"split_isolation":leakage["passed"],
        "performance":perf["performance_passed"],"train_pairs_exact":read_json(R64/"example_manifest_train.json")["corrected_pair_count"]==EXPECTED_SPLIT["train"],
        "validation_pairs_exact":read_json(R64/"example_manifest_validation.json")["corrected_pair_count"]==EXPECTED_SPLIT["validation"]},
        "passed":False}
    corrected["passed"]=all(corrected["checks"].values())
    json_write(R64/"corrected_example_validation.json",corrected)
    return nonpaired_report,prov_report,leakage,perf


def build_training_arrays(split: str) -> tuple[dict[str,np.ndarray],dict[str,Any]]:
    arrays=load_npz_once(R64/f"examples_{split}.npz")
    node_meta=pd.read_parquet(R64/f"node_examples_{split}.parquet")
    rel_meta=pd.read_parquet(R64/f"relation_examples_{split}.parquet")
    pair_meta=pd.read_parquet(R64/f"paired_examples_{split}.parquet")
    candidate=pd.read_parquet(R63/"candidate_pool.parquet")
    score_by_id=dict(zip(candidate.candidate_id,candidate.candidate_score)); edge_by_id={r.candidate_id:r for r in candidate.itertuples()}
    # Node adapter: R63 1=pure/0=impure; v2 1=impure/0=pure.
    keep=np.flatnonzero(arrays["node_y"]>=0)
    node_label=(1-arrays["node_y"][keep]).astype(np.int8)
    boundary=arrays["boundary_y"][keep].astype(np.int8)
    node_sequences=node_meta.iloc[keep].sequence.tolist()
    node_group=np.asarray([SEQ_GROUP[split][s] for s in node_sequences],np.int16)
    aba_event=(np.sum(boundary>0,axis=1)>=2).astype(np.uint8)
    # Frozen relation example adapter: positives receive highest-score frozen labeled negative
    # with same source and destination respectively. Candidate IDs break exact score ties.
    rel_meta=rel_meta.copy(); rel_meta["candidate_score"]=[float(score_by_id[x]) for x in rel_meta.candidate_id]
    negatives=rel_meta[rel_meta.relation_label==0]
    out_neg={}
    for src,g in negatives.groupby("src_chunk_id",sort=False):
        g=g.sort_values(["candidate_score","candidate_id"],ascending=[False,True],kind="mergesort"); out_neg[src]=int(g.iloc[0].tensor_index)
    in_neg={}
    for dst,g in negatives.groupby("dst_chunk_id",sort=False):
        g=g.sort_values(["candidate_score","candidate_id"],ascending=[False,True],kind="mergesort"); in_neg[dst]=int(g.iloc[0].tensor_index)
    triplets=[]
    for r in rel_meta[rel_meta.relation_label==1].sort_values(["sequence","candidate_id"],kind="mergesort").itertuples():
        if r.src_chunk_id in out_neg and r.dst_chunk_id in in_neg:
            triplets.append((int(r.tensor_index),out_neg[r.src_chunk_id],in_neg[r.dst_chunk_id],r.sequence,int(edge_by_id[r.candidate_id].gap_bucket_index),r.candidate_id))
    if not triplets: raise RuntimeError(f"{split}: no deterministic frozen relation triplets")
    pidx=np.asarray([x[0] for x in triplets],np.int64); oidx=np.asarray([x[1] for x in triplets],np.int64); iidx=np.asarray([x[2] for x in triplets],np.int64)
    # Paired corrected tensors are [src1,pos1,src2,pos2].
    pair_bucket_map=dict(zip(pd.read_parquet(R63/"paired_candidate_pool.parquet").pair_id,pd.read_parquet(R63/"paired_candidate_pool.parquet").gap_bucket_index))
    data={
        "node_x":arrays["node_x"][keep],"node_mask":arrays["node_mask"][keep],"node_label":node_label,
        "boundary_label":boundary,"node_group":node_group,"aba_event":aba_event,
        "rel_src":arrays["relation_src_x"][pidx],"rel_src_mask":arrays["relation_src_mask"][pidx],
        "rel_pos":arrays["relation_dst_x"][pidx],"rel_pos_mask":arrays["relation_dst_mask"][pidx],
        "rel_out_neg":arrays["relation_dst_x"][oidx],"rel_out_neg_mask":arrays["relation_dst_mask"][oidx],
        "rel_in_neg":arrays["relation_src_x"][iidx],"rel_in_neg_mask":arrays["relation_src_mask"][iidx],
        "rel_group":np.asarray([SEQ_GROUP[split][x[3]] for x in triplets],np.int16),
        "rel_bucket":np.asarray([x[4] for x in triplets],np.int8),
        "pair_src1":arrays["pair_x"][:,0],"pair_src1_mask":arrays["pair_mask"][:,0],
        "pair_pos1":arrays["pair_x"][:,1],"pair_pos1_mask":arrays["pair_mask"][:,1],
        "pair_src2":arrays["pair_x"][:,2],"pair_src2_mask":arrays["pair_mask"][:,2],
        "pair_pos2":arrays["pair_x"][:,3],"pair_pos2_mask":arrays["pair_mask"][:,3],
        "pair_group":np.asarray([SEQ_GROUP[split][s] for s in pair_meta.sequence],np.int16),
        "pair_bucket":np.asarray([int(pair_bucket_map[x]) for x in pair_meta.pair_id],np.int8),
    }
    report={"split":split,"node_examples_input":len(node_meta),"node_examples_used":len(keep),"node_ignored":int((arrays["node_y"]<0).sum()),
            "positive_relation_edges":int((rel_meta.relation_label==1).sum()),"negative_relation_edges":int((rel_meta.relation_label==0).sum()),
            "ignored_relation_edges":int((rel_meta.relation_label<0).sum()),"relation_triplets":len(triplets),
            "positive_edges_without_complete_triplet":int((rel_meta.relation_label==1).sum())-len(triplets),"corrected_pairs":len(pair_meta),
            "triplet_rule":"highest candidate_score frozen labeled negative sharing source/destination; candidate_id tie",
            "triplet_candidate_ids":[{"positive":x[5],"out_negative_tensor_index":int(x[1]),"in_negative_tensor_index":int(x[2])} for x in triplets],
            "array_shapes":{k:list(v.shape) for k,v in data.items()},"array_dtypes":{k:str(v.dtype) for k,v in data.items()}}
    return data,report


def training_config() -> dict[str,Any]:
    v2=load_v2(); model=v2.HierarchicalRelationEncoder()
    cfg={"experiment_id":EXP_ID,"created_at":utc_now(),"from_scratch":True,"v2_checkpoint_reuse":False,"warm_start":False,
         "model":"HierarchicalRelationEncoder","parameter_count":v2.parameter_count(model),"expected_parameter_count":EXPECTED_PARAMS,
         "seeds":SEEDS,"epochs":EPOCHS,"optimizer":"AdamW","learning_rate":v2.EXTERNAL_LR,"weight_decay":v2.WEIGHT_DECAY,
         "batch_nodes":v2.BATCH_NODES,"batch_relations":v2.BATCH_RELATIONS,"gradient_clip_norm":5.0,
         "loss_weights":v2.LOSS_WEIGHTS,"group_dro_eta":0.1,"gap_buckets":v2.GAP_BUCKETS,
         "checkpoint_composite":"0.30 node_pr_auc + 0.35 boundary_pr_auc + 0.25 mean(outgoing_R1,incoming_R1) + 0.10*(1-catastrophic_false_link)",
         "within_seed_tie_break":"earlier epoch","across_seed_tie_break":"lower seed then earlier epoch",
         "health_acceptance":"complete finite noncollapsed metrics only; no independent post-hoc numerical threshold",
         "train_sequences":TRAIN_SEQS,"validation_sequences":VAL_SEQS,
         "source_script":str(V2_SCRIPT.relative_to(ROOT)),"source_script_sha256":sha256_file(V2_SCRIPT),
         "source_function_sha256":{n:sha256_text(inspect.getsource(getattr(v2,n))) for n in ["HierarchicalRelationEncoder","training_objective","validation_metrics","sample_indices","group_weighted"]},
         "corrected_train_examples_sha256":sha256_file(R64/"examples_train.npz"),"corrected_validation_examples_sha256":sha256_file(R64/"examples_validation.npz")}
    json_write(R64/"training_config.json",cfg); return cfg


def flatten_metrics(metrics: dict[str,Any]) -> dict[str,Any]:
    b=metrics["conditional_boundary"]; n=metrics["node"]
    return {"validation_composite":metrics["checkpoint_selection_composite"],"validation_boundary_pr_auc":b.get("pr_auc"),
            "validation_boundary_precision_at_actual":b.get("precision_at_actual_count"),"validation_boundary_recall_at_95_precision":b.get("recall_at_95_precision"),
            "validation_node_pr_auc":n.get("pr_auc"),"validation_outgoing_R1":metrics["outgoing_successor_R_at_1_pairwise"],
            "validation_incoming_R1":metrics["incoming_predecessor_R_at_1_pairwise"],"validation_paired_replacement_R1":metrics["paired_replacement_R_at_1"],
            "validation_catastrophic_false_link_rate":metrics["catastrophic_false_link_rate"],
            "validation_ABA_exact_two_boundary_recall":metrics["A_to_B_to_A_exact_two_boundary_recall"],
            "validation_risk_pr_auc":metrics["risk"].get("pr_auc")}


def validation_score_diagnostics(model: nn.Module,data: dict[str,np.ndarray],dev:torch.device) -> dict[str,Any]:
    v2=load_v2(); model.eval(); node=[]; boundary=[]; pos=[]; out=[]; inc=[]; pair_margin=[]
    with torch.no_grad():
        for start in range(0,len(data["node_label"]),1024):
            ids=np.arange(start,min(start+1024,len(data["node_label"])))
            nl,bl,valid=model.node_and_boundary(v2.tensor(data["node_x"][ids],dev),v2.tensor(data["node_mask"][ids],dev))
            node.append(torch.sigmoid(nl).cpu().numpy()); boundary.append(torch.sigmoid(bl).cpu().numpy()[valid.cpu().numpy()>0])
        for start in range(0,len(data["rel_group"]),1024):
            ids=np.arange(start,min(start+1024,len(data["rel_group"])))
            ps,_=model.relation(v2.tensor(data["rel_src"][ids],dev),v2.tensor(data["rel_src_mask"][ids],dev),v2.tensor(data["rel_pos"][ids],dev),v2.tensor(data["rel_pos_mask"][ids],dev))
            os,_=model.relation(v2.tensor(data["rel_src"][ids],dev),v2.tensor(data["rel_src_mask"][ids],dev),v2.tensor(data["rel_out_neg"][ids],dev),v2.tensor(data["rel_out_neg_mask"][ids],dev))
            ins,_=model.relation(v2.tensor(data["rel_in_neg"][ids],dev),v2.tensor(data["rel_in_neg_mask"][ids],dev),v2.tensor(data["rel_pos"][ids],dev),v2.tensor(data["rel_pos_mask"][ids],dev))
            pos.append(ps.cpu().numpy());out.append(os.cpu().numpy());inc.append(ins.cpu().numpy())
        for start in range(0,len(data["pair_group"]),1024):
            ids=np.arange(start,min(start+1024,len(data["pair_group"])))
            a=v2.tensor(data["pair_src1"][ids],dev);am=v2.tensor(data["pair_src1_mask"][ids],dev);b=v2.tensor(data["pair_pos1"][ids],dev);bm=v2.tensor(data["pair_pos1_mask"][ids],dev)
            c=v2.tensor(data["pair_src2"][ids],dev);cm=v2.tensor(data["pair_src2_mask"][ids],dev);d=v2.tensor(data["pair_pos2"][ids],dev);dm=v2.tensor(data["pair_pos2_mask"][ids],dev)
            ab,_=model.relation(a,am,b,bm);cd,_=model.relation(c,cm,d,dm);ad,_=model.relation(a,am,d,dm);cb,_=model.relation(c,cm,b,bm)
            pair_margin.append(((ab+cd)-(ad+cb)).cpu().numpy())
    cat=lambda xs:np.concatenate(xs) if xs else np.zeros(0,float)
    vals={"node_probability":cat(node),"boundary_probability":cat(boundary),"relation_positive_logit":cat(pos),
          "relation_out_negative_logit":cat(out),"relation_in_negative_logit":cat(inc),"paired_margin":cat(pair_margin)}
    return {k:{"rows":len(v),"mean":float(v.mean()) if len(v) else None,"std":float(v.std()) if len(v) else None,
               "min":float(v.min()) if len(v) else None,"max":float(v.max()) if len(v) else None} for k,v in vals.items()}


def epoch_metrics_row(epoch:int,parts:list[dict[str,float]],metrics:dict[str,Any],q_node:torch.Tensor,q_rel:torch.Tensor,
                      grad_norms:list[float],wall:float,gpu_peak:float,selected:bool,lr:float) -> dict[str,Any]:
    means={f"train_{k}_loss" if k not in {"total","group_dro","sparsity","count"} else f"train_{k}":float(np.mean([p[k] for p in parts])) for k in parts[0]}
    row={"epoch":epoch,**means,**flatten_metrics(metrics),"validation_metrics":metrics,
         "group_dro_node_weights":[float(x) for x in q_node.detach().cpu().numpy()],
         "group_dro_relation_weights":[float(x) for x in q_rel.detach().cpu().numpy()],
         "gradient_norm_mean":float(np.mean(grad_norms)),"gradient_norm_max":float(np.max(grad_norms)),
         "finite":bool(all(np.isfinite(list(means.values()))) and all(np.isfinite(x) for x in flatten_metrics(metrics).values() if x is not None)),
         "learning_rate":lr,"checkpoint_selected":selected,"wall_seconds":wall,"gpu_peak_memory_mb":gpu_peak}
    return row


def train_one_seed(seed:int,train:dict[str,np.ndarray],val:dict[str,np.ndarray],cfg:dict[str,Any]) -> dict[str,Any]:
    v2=load_v2(); v2.set_determinism(seed); dev=v2.device()
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    model=v2.HierarchicalRelationEncoder().to(dev); params=v2.parameter_count(model)
    if params!=EXPECTED_PARAMS: raise RuntimeError(f"parameter mismatch seed={seed}: {params}")
    optimizer=torch.optim.AdamW(model.parameters(),lr=v2.EXTERNAL_LR,weight_decay=v2.WEIGHT_DECAY)
    q_node=torch.ones(len(TRAIN_SEQS),device=dev)/len(TRAIN_SEQS);q_rel=torch.ones(len(TRAIN_SEQS),device=dev)/len(TRAIN_SEQS)
    rng=np.random.default_rng(seed); root=R64/"training"/f"seed_{seed}"; root.mkdir(parents=True,exist_ok=True)
    history=[]; metrics_path=root/"metrics.jsonl"; best=None; total_start=time.perf_counter()
    update_stage(f"seed_{seed}","running",unit=str(seed),notes="from scratch; 0/30 epochs")
    registry_update_running(f"seed_{seed}",f"seed={seed} running from scratch; epochs_completed=0; TrackEval=0")
    for epoch in range(1,EPOCHS+1):
        epoch_start=time.perf_counter(); model.train(); parts=[]; grad_norms=[]
        nb=list(v2.sample_indices(len(train["node_label"]),v2.BATCH_NODES,rng))
        rb=list(v2.sample_indices(len(train["rel_group"]),v2.BATCH_RELATIONS,rng))
        pb=list(v2.sample_indices(len(train["pair_group"]),v2.BATCH_RELATIONS,rng)) if len(train["pair_group"]) else [np.zeros(0,np.int64)]
        for step in range(max(len(nb),len(rb),len(pb))):
            loss,q_node,q_rel,p=v2.training_objective(model,train,nb[step%len(nb)],rb[step%len(rb)],pb[step%len(pb)],dev,q_node,q_rel)
            if not torch.isfinite(loss): raise FloatingPointError(f"nonfinite loss seed={seed} epoch={epoch}")
            optimizer.zero_grad(set_to_none=True);loss.backward(); norm=nn.utils.clip_grad_norm_(model.parameters(),5.0)
            if not torch.isfinite(norm): raise FloatingPointError(f"nonfinite gradient seed={seed} epoch={epoch}")
            optimizer.step();parts.append(p);grad_norms.append(float(norm.detach().cpu()))
        metrics=v2.validation_metrics(model,val,dev); composite=float(metrics["checkpoint_selection_composite"])
        candidate=(composite,-epoch); is_best=best is None or candidate>best[0]
        if is_best:
            ckpt=root/"best_checkpoint.pt"
            torch.save({"model":model.state_dict(),"seed":seed,"epoch":epoch,"parameter_count":params,"validation":metrics,
                        "training_config_sha256":sha256_file(R64/"training_config.json"),"train_examples_sha256":sha256_file(R64/"examples_train.npz"),
                        "validation_examples_sha256":sha256_file(R64/"examples_validation.npz"),"script_sha256":sha256_file(SCRIPT),
                        "from_scratch":True,"v2_checkpoint_reuse":False,"warm_start":False},ckpt)
            best=(candidate,epoch,metrics,sha256_file(ckpt))
        row=epoch_metrics_row(epoch,parts,metrics,q_node,q_rel,grad_norms,time.perf_counter()-epoch_start,peak_gpu_mb(),is_best,v2.EXTERNAL_LR)
        row["checkpoint_selected_this_epoch"]=is_best;history.append(row)
        with metrics_path.open("a",encoding="utf-8") as f:f.write(json.dumps(row,sort_keys=True,ensure_ascii=False)+"\n")
        json_write(root/"training_history.json",history)
        manifest={"experiment_id":EXP_ID,"seed":seed,"status":"running" if epoch<EPOCHS else "completed","epochs_completed":epoch,"expected_epochs":EPOCHS,
                  "current_best_epoch":best[1],"current_best_composite":best[0][0],"checkpoint":"best_checkpoint.pt",
                  "checkpoint_sha256":sha256_file(root/"best_checkpoint.pt"),"parameter_count":params,"from_scratch":True,
                  "training_config_sha256":sha256_file(R64/"training_config.json"),"train_examples_sha256":sha256_file(R64/"examples_train.npz"),
                  "validation_examples_sha256":sha256_file(R64/"examples_validation.npz"),"all_epochs_finite":all(x["finite"] for x in history)}
        json_write(root/"checkpoint_manifest.json",manifest)
        update_stage(f"seed_{seed}","running",unit=str(seed),notes=f"from scratch; {epoch}/{EPOCHS} epochs; best_epoch={best[1]}; best_composite={best[0][0]:.9f}")
        registry_update_running(f"seed_{seed}",f"seed={seed} running from scratch; epochs_completed={epoch}; best_composite={best[0][0]:.9f}; TrackEval=0")
        print(json.dumps({"event":"epoch_completed","seed":seed,"epoch":epoch,"composite":composite,"best_epoch":best[1],"finite":row["finite"]}),flush=True)
        if not row["finite"]: raise FloatingPointError(f"nonfinite metric seed={seed} epoch={epoch}")
    assert best is not None
    for row in history: row["checkpoint_selected_final"]=int(row["epoch"])==int(best[1])
    json_write(root/"training_history.json",history)
    with metrics_path.open("w",encoding="utf-8") as f:
        for row in history:f.write(json.dumps(row,sort_keys=True,ensure_ascii=False)+"\n")
    state=torch.load(root/"best_checkpoint.pt",map_location="cpu",weights_only=False)
    reload_model=v2.HierarchicalRelationEncoder().to(dev);reload_model.load_state_dict(state["model"]);reload_model.eval()
    reload_metrics=v2.validation_metrics(reload_model,val,dev)
    reload_consistent=abs(float(reload_metrics["checkpoint_selection_composite"])-float(best[2]["checkpoint_selection_composite"]))<=1e-12
    diagnostics=validation_score_diagnostics(reload_model,val,dev)
    noncollapsed=all(v["std"] is not None and v["std"]>0.0 for k,v in diagnostics.items() if v["rows"]>1)
    final_manifest={"experiment_id":EXP_ID,"seed":seed,"status":"completed","epochs_completed":EPOCHS,"expected_epochs":EPOCHS,
                    "best_epoch":best[1],"best_composite":best[0][0],"best_validation":best[2],"checkpoint":"best_checkpoint.pt",
                    "checkpoint_sha256":sha256_file(root/"best_checkpoint.pt"),"parameter_count":params,"from_scratch":True,
                    "v2_checkpoint_reuse":False,"warm_start":False,"all_epochs_finite":all(x["finite"] for x in history),
                    "checkpoint_reload_consistent":reload_consistent,"reload_validation":reload_metrics,"score_diagnostics":diagnostics,
                    "noncollapsed":noncollapsed,"wall_seconds":time.perf_counter()-total_start,"peak_gpu_memory_mb":peak_gpu_mb(),
                    "training_history_sha256":sha256_file(root/"training_history.json"),"metrics_jsonl_sha256":sha256_file(metrics_path),
                    "training_config_sha256":sha256_file(R64/"training_config.json"),"train_examples_sha256":sha256_file(R64/"examples_train.npz"),
                    "validation_examples_sha256":sha256_file(R64/"examples_validation.npz")}
    json_write(root/"checkpoint_manifest.json",final_manifest)
    update_stage(f"seed_{seed}","completed",unit=str(seed),decision="pass",report=str((root/"checkpoint_manifest.json").relative_to(ROOT)),
                 notes=f"30/30 finite; best_epoch={best[1]}; composite={best[0][0]:.9f}")
    append_event("training_seed_completed",seed=seed,best_epoch=best[1],best_composite=best[0][0],checkpoint_sha256=final_manifest["checkpoint_sha256"])
    return final_manifest


def subset_validation(data:dict[str,np.ndarray],group:int) -> dict[str,np.ndarray]:
    ni=np.flatnonzero(data["node_group"]==group);ri=np.flatnonzero(data["rel_group"]==group);pi=np.flatnonzero(data["pair_group"]==group)
    node_keys={"node_x","node_mask","node_label","boundary_label","node_group","aba_event"}
    rel_keys={"rel_src","rel_src_mask","rel_pos","rel_pos_mask","rel_out_neg","rel_out_neg_mask","rel_in_neg","rel_in_neg_mask","rel_group","rel_bucket"}
    pair_keys={"pair_src1","pair_src1_mask","pair_pos1","pair_pos1_mask","pair_src2","pair_src2_mask","pair_pos2","pair_pos2_mask","pair_group","pair_bucket"}
    return {k:v[ni] if k in node_keys else v[ri] if k in rel_keys else v[pi] for k,v in data.items() if k in node_keys|rel_keys|pair_keys}


def relation_bucket_diagnostics(model:nn.Module,data:dict[str,np.ndarray],dev:torch.device) -> dict[str,Any]:
    v2=load_v2(); out={};model.eval()
    with torch.no_grad():
        for bi,(name,_,_) in enumerate(GAP_BUCKETS):
            ids=np.flatnonzero(data["rel_bucket"]==bi); pids=np.flatnonzero(data["pair_bucket"]==bi)
            if len(ids):
                ps,_=model.relation(v2.tensor(data["rel_src"][ids],dev),v2.tensor(data["rel_src_mask"][ids],dev),v2.tensor(data["rel_pos"][ids],dev),v2.tensor(data["rel_pos_mask"][ids],dev))
                os,_=model.relation(v2.tensor(data["rel_src"][ids],dev),v2.tensor(data["rel_src_mask"][ids],dev),v2.tensor(data["rel_out_neg"][ids],dev),v2.tensor(data["rel_out_neg_mask"][ids],dev))
                ins,_=model.relation(v2.tensor(data["rel_in_neg"][ids],dev),v2.tensor(data["rel_in_neg_mask"][ids],dev),v2.tensor(data["rel_pos"][ids],dev),v2.tensor(data["rel_pos_mask"][ids],dev))
                or1=float((ps>os).float().mean().cpu());ir1=float((ps>ins).float().mean().cpu())
            else:or1=ir1=None
            pair_r1=None
            if len(pids):
                a=v2.tensor(data["pair_src1"][pids],dev);am=v2.tensor(data["pair_src1_mask"][pids],dev);b=v2.tensor(data["pair_pos1"][pids],dev);bm=v2.tensor(data["pair_pos1_mask"][pids],dev)
                c=v2.tensor(data["pair_src2"][pids],dev);cm=v2.tensor(data["pair_src2_mask"][pids],dev);d=v2.tensor(data["pair_pos2"][pids],dev);dm=v2.tensor(data["pair_pos2_mask"][pids],dev)
                ab,_=model.relation(a,am,b,bm);cd,_=model.relation(c,cm,d,dm);ad,_=model.relation(a,am,d,dm);cb,_=model.relation(c,cm,b,bm)
                pair_r1=float(((ab+cd)>(ad+cb)).float().mean().cpu())
            out[name]={"relation_triplets":len(ids),"paired_examples":len(pids),"outgoing_R1":or1,"incoming_R1":ir1,"paired_R1":pair_r1,"not_used_for_checkpoint_selection":True}
    return out


def pair_accuracy_ci(successes:int,total:int) -> dict[str,Any]:
    if total<=0:return {"successes":0,"total":0,"accuracy":None,"clopper_pearson_95":[None,None]}
    alpha=.05;lo=0.0 if successes==0 else float(scipy.stats.beta.ppf(alpha/2,successes,total-successes+1));hi=1.0 if successes==total else float(scipy.stats.beta.ppf(1-alpha/2,successes+1,total-successes))
    return {"successes":successes,"total":total,"accuracy":successes/total,"clopper_pearson_95":[lo,hi]}


def select_global_checkpoint(seed_reports:list[dict[str,Any]],val:dict[str,np.ndarray],cfg:dict[str,Any]) -> tuple[dict[str,Any],dict[str,Any]]:
    ranked=sorted(seed_reports,key=lambda r:(-float(r["best_composite"]),int(r["seed"]),int(r["best_epoch"])))
    selected=ranked[0];source=R64/"training"/f"seed_{selected['seed']}"/"best_checkpoint.pt"
    frozen_dir=R64/"frozen_checkpoint";frozen_dir.mkdir(parents=True,exist_ok=True);frozen=frozen_dir/"relation_v3_frozen.pt";shutil.copyfile(source,frozen)
    v2=load_v2();dev=v2.device();state=torch.load(frozen,map_location="cpu",weights_only=False);model=v2.HierarchicalRelationEncoder().to(dev);model.load_state_dict(state["model"]);model.eval()
    metrics=v2.validation_metrics(model,val,dev); per_seq={}
    for group,seq in enumerate(VAL_SEQS):per_seq[seq]=v2.validation_metrics(model,subset_validation(val,group),dev)
    score_diag=validation_score_diagnostics(model,val,dev); bucket_diag=relation_bucket_diagnostics(model,val,dev)
    pair_success=int(round(float(metrics["paired_replacement_R_at_1"])*len(val["pair_group"])))
    external={"experiment_id":EXP_ID,"created_at":utc_now(),"selected_seed":selected["seed"],"selected_epoch":selected["best_epoch"],
              "metrics":metrics,"flattened":flatten_metrics(metrics),"validation_pair_count":len(val["pair_group"]),
              "paired_confidence_interval":pair_accuracy_ci(pair_success,len(val["pair_group"])),"per_sequence":per_seq,
              "per_gap_bucket":bucket_diag,"score_diagnostics":score_diag,"additional_diagnostics_not_used_for_checkpoint_selection":True}
    json_write(R64/"external_validation_metrics.json",external)
    selection={"experiment_id":EXP_ID,"created_at":utc_now(),"rule":"max fixed v2 composite; tie lower seed then earlier epoch",
               "candidates":[{"seed":r["seed"],"epoch":r["best_epoch"],"composite":r["best_composite"],"checkpoint_sha256":r["checkpoint_sha256"]} for r in ranked],
               "selected_seed":selected["seed"],"selected_epoch":selected["best_epoch"],"selected_composite":selected["best_composite"],
               "source_checkpoint_sha256":sha256_file(source),"frozen_checkpoint_sha256":sha256_file(frozen),"tie_break_correct":True,
               "mot20_or_hota_used":False}
    json_write(frozen_dir/"checkpoint_selection.json",selection)
    manifest={"experiment_id":EXP_ID,"created_at":utc_now(),"checkpoint":"relation_v3_frozen.pt","checkpoint_sha256":sha256_file(frozen),
              "model_config_sha256":sha256_file(R64/"training_config.json"),"corrected_train_examples_sha256":sha256_file(R64/"examples_train.npz"),
              "corrected_validation_examples_sha256":sha256_file(R64/"examples_validation.npz"),"training_script_sha256":sha256_file(SCRIPT),
              "seed":selected["seed"],"epoch":selected["best_epoch"],"composite":selected["best_composite"],"parameter_count":EXPECTED_PARAMS,
              "contract_hash":CONTRACT_HASH,"validation_metrics":metrics,"from_scratch":True,"v2_checkpoint_reuse":False,"warm_start":False,
              "checkpoint_reload_success":True,"eval_mode":not model.training,"selection_sha256":sha256_file(frozen_dir/"checkpoint_selection.json")}
    json_write(frozen_dir/"checkpoint_manifest.json",manifest)
    return manifest,external


def scope_processes() -> list[dict[str,Any]]:
    text=subprocess.run(["ps","-eo","pid,ppid,stat,cmd"],capture_output=True,text=True,check=True).stdout
    patterns=["m23_64_v3_pair_reconstruction_training.py","m23_63_v3_supervision_join_example_audit.py","eval_motstyle_trackeval.py","TrackEval/scripts/run_mot_challenge.py"]
    out=[]
    for line in text.splitlines()[1:]:
        p=line.strip().split(None,3)
        if len(p)<4:continue
        pid=int(p[0])
        if pid in {os.getpid(),os.getppid()}:continue
        if any(x in p[3] for x in patterns):out.append({"pid":pid,"ppid":int(p[1]),"stat":p[2],"cmd":p[3]})
    return out


def stage_a_gate(rv:dict[str,Any],pair_manifest:dict[str,Any],nonpaired:dict[str,Any],prov:dict[str,Any],leak:dict[str,Any],perf:dict[str,Any]) -> dict[str,Any]:
    train_manifest=read_json(R64/"example_manifest_train.json");val_manifest=read_json(R64/"example_manifest_validation.json")
    frozen=read_json(R64/"input_manifest.json"); sha_match=all(sha256_file(ROOT/x["path"])==x["sha256"] for x in frozen["frozen_inputs"])
    checks={"m23_63_closure_valid":rv["passed"],"m23_63_only_paired_train_support_failed":rv["checks"]["r63_only_train_support_failed"],
            "all_r62_r63_sha_match":sha_match,"contract_hash_match":rv["checks"]["contract_exact"],
            "candidate_pool_unmodified":sha256_file(R63/"candidate_pool.parquet")==pair_manifest["inputs"]["candidate_pool_sha256"],
            "paired_pool_unmodified":sha256_file(R63/"paired_candidate_pool.parquet")==pair_manifest["inputs"]["paired_pool_sha256"],
            "supervision_sidecar_unmodified":sha256_file(R63/"row_supervision.parquet")==pair_manifest["inputs"]["supervision_sha256"],
            "pair_diagnostic_exact":pair_manifest["diagnostic_exact"],"train_valid_pair_182":pair_manifest["actual_train"]==182,
            "validation_valid_pair_99":pair_manifest["actual_validation"]==99,
            "all_train_sequences_nonzero":all(pair_manifest["actual_per_sequence"][s]>0 for s in TRAIN_SEQS),
            "all_validation_sequences_nonzero":all(pair_manifest["actual_per_sequence"][s]>0 for s in VAL_SEQS),
            "train_paired_support_at_least_5":train_manifest["corrected_pair_count"]>=5,
            "validation_paired_support_at_least_1":val_manifest["corrected_pair_count"]>=1,
            "node_relation_byte_exact":nonpaired["passed"],"all_pair_ids_from_frozen_pool":prov["checks"]["train_pair_tensor_provenance"] and prov["checks"]["validation_pair_tensor_provenance"],
            "no_edge_added":pair_manifest["complete_frozen_pool_rows"]==140000 and pair_manifest["pair_id_unique"],
            "no_topology_changed":nonpaired["r63_inputs_unchanged"],"all_arrays_finite":prov["checks"]["train_all_arrays_finite"] and prov["checks"]["validation_all_arrays_finite"],
            "masks_prefix_valid":prov["checks"]["train_mask_prefix_valid"] and prov["checks"]["validation_mask_prefix_valid"],
            "all_tensor_provenance_valid":prov["passed"],"split_isolation":leak["passed"],
            "no_unknown_as_negative":prov["checks"]["train_no_unknown_as_negative"] and prov["checks"]["validation_no_unknown_as_negative"],
            "validator_no_tb_reads":perf["performance_passed"],"scope_guard":all(v==0 for v in SCOPE_ZERO.values()),
            "no_checkpoint_loaded":True,"training_not_started":not (R64/"training").exists()}
    report={"experiment_id":EXP_ID,"created_at":utc_now(),"stage":"A","checks":checks,"passed":all(checks.values()),
            "decision":"PASS_STAGE_A" if all(checks.values()) else "", "pair_counts":{"train":pair_manifest["actual_train"],"validation":pair_manifest["actual_validation"],"per_sequence":pair_manifest["actual_per_sequence"]},
            "performance":perf}
    if not report["passed"]:
        if not rv["passed"] or not sha_match:report["decision"]="FAIL_M23_63_INPUT_REVERIFICATION"
        elif not pair_manifest["diagnostic_exact"]:report["decision"]="FAIL_PAIR_DIAGNOSTIC_REPLICATION"
        elif not (checks["train_valid_pair_182"] and checks["validation_valid_pair_99"] and checks["all_pair_ids_from_frozen_pool"] and checks["no_edge_added"]):report["decision"]="FAIL_CORRECTED_PAIR_RECONSTRUCTION"
        elif not nonpaired["passed"]:report["decision"]="FAIL_NONPAIRED_INVARIANCE"
        elif not prov["passed"] or not perf["performance_passed"]:report["decision"]="FAIL_TENSOR_PROVENANCE"
        elif not leak["passed"]:report["decision"]="FAIL_SPLIT_LEAKAGE"
        else:report["decision"]="FAIL_SCOPE_GUARD"
    json_write(R64/"stage_a_gate.json",report);return report


def stage_b_gate(seed_reports:list[dict[str,Any]],manifest:dict[str,Any],external:dict[str,Any]) -> dict[str,Any]:
    histories={r["seed"]:read_json(R64/"training"/f"seed_{r['seed']}"/"training_history.json") for r in seed_reports}
    cfg=read_json(R64/"training_config.json"); selection=read_json(R64/"frozen_checkpoint/checkpoint_selection.json")
    ranked=sorted(seed_reports,key=lambda r:(-float(r["best_composite"]),int(r["seed"]),int(r["best_epoch"])))
    flat=external["flattened"]
    required=["validation_composite","validation_boundary_pr_auc","validation_boundary_precision_at_actual","validation_boundary_recall_at_95_precision",
              "validation_node_pr_auc","validation_outgoing_R1","validation_incoming_R1","validation_paired_replacement_R1","validation_catastrophic_false_link_rate"]
    checks={"three_seeds_completed":len(seed_reports)==3 and all(r["status"]=="completed" for r in seed_reports),
            "ninety_epoch_records_complete":sum(len(x) for x in histories.values())==90 and all(len(x)==30 for x in histories.values()),
            "all_losses_finite":all(r["all_epochs_finite"] and all(x["finite"] for x in histories[r["seed"]]) for r in seed_reports),
            "checkpoints_reload":all(r["checkpoint_reload_consistent"] for r in seed_reports),"eval_mode_consistent":manifest["eval_mode"],
            "parameter_count_exact":manifest["parameter_count"]==EXPECTED_PARAMS and all(r["parameter_count"]==EXPECTED_PARAMS for r in seed_reports),
            "selected_checkpoint_sha_stable":sha256_file(R64/"frozen_checkpoint/relation_v3_frozen.pt")==manifest["checkpoint_sha256"],
            "example_sha_stable":sha256_file(R64/"examples_train.npz")==cfg["corrected_train_examples_sha256"] and sha256_file(R64/"examples_validation.npz")==cfg["corrected_validation_examples_sha256"],
            "selection_composite_exact":selection["selected_seed"]==ranked[0]["seed"] and selection["selected_epoch"]==ranked[0]["best_epoch"] and abs(selection["selected_composite"]-ranked[0]["best_composite"])<=1e-12,
            "tie_break_correct":selection["tie_break_correct"],"validation_metrics_complete":all(flat.get(k) is not None and np.isfinite(flat[k]) for k in required),
            "paired_metric_uses_99":external["validation_pair_count"]==99,"external_validation_noncollapsed":all(r["noncollapsed"] for r in seed_reports),
            "from_scratch_only":all(r["from_scratch"] and not r["v2_checkpoint_reuse"] and not r["warm_start"] for r in seed_reports),
            "scope_guard":all(v==0 for v in SCOPE_ZERO.values()),"training_runs_three":len(seed_reports)==3}
    report={"experiment_id":EXP_ID,"created_at":utc_now(),"stage":"B","checks":checks,"passed":all(checks.values()),
            "decision":"PASS_STAGE_B" if all(checks.values()) else "","training_runs":len(seed_reports),"selected":selection,"validation":external}
    if not report["passed"]:
        if not checks["all_losses_finite"]:report["decision"]="FAIL_TRAINING_NONFINITE"
        elif not (checks["three_seeds_completed"] and checks["ninety_epoch_records_complete"]):report["decision"]="FAIL_TRAINING_INCOMPLETE"
        elif not (checks["selection_composite_exact"] and checks["tie_break_correct"] and checks["selected_checkpoint_sha_stable"]):report["decision"]="FAIL_CHECKPOINT_SELECTION"
        elif not (checks["validation_metrics_complete"] and checks["external_validation_noncollapsed"] and checks["paired_metric_uses_99"]):report["decision"]="FAIL_EXTERNAL_VALIDATION_HEALTH"
        else:report["decision"]="FAIL_SCOPE_GUARD"
    json_write(R64/"stage_b_gate.json",report);return report


def mark_remaining(decision:str) -> None:
    for r in read_summary():
        if r["status"] in {"pending","running"}:
            update_stage(r["stage"],"failed" if r["stage"]=="closure" else "prohibited_by_scope",unit=r["unit"],decision=decision,
                         notes="closed by fail-closed decision" if r["stage"]!="closure" else "experiment closed")


def result_document(final:dict[str,Any]) -> str:
    stats=pd.read_csv(R64/"pair_supervision_statistics.csv") if (R64/"pair_supervision_statistics.csv").exists() else pd.DataFrame()
    pair_rows="\n".join(f"| {r.sequence} | {int(r.pool_total)} | {int(r.valid_total)} | {int(r.m23_63_selected_valid)} | {int(r.valid_not_selected)} |" for r in stats.itertuples() if not str(r.sequence).startswith("__"))
    stage_a=read_json(R64/"stage_a_gate.json") if (R64/"stage_a_gate.json").exists() else {}
    seed_lines=""
    if (R64/"training").exists():
        vals=[]
        for seed in SEEDS:
            p=R64/"training"/f"seed_{seed}"/"checkpoint_manifest.json"
            if p.exists():
                m=read_json(p);vals.append(f"| {seed} | {m.get('epochs_completed')} | {m.get('best_epoch','')} | {m.get('best_composite','')} | {m.get('checkpoint_sha256','')} |")
        seed_lines="\n".join(vals)
    return f"""# M23-64 result — frozen-pool pair reconstruction and from-scratch relation training

## Final decision
`{final['decision']}`; status=`closed`; training_runs={final['training_runs']}; TrackEval=0; tracker_outputs=0; HOTA is intentionally empty.

## Commands
```bash
python -u scripts/m23_research/m23_64_v3_pair_reconstruction_training.py init
python -u scripts/m23_research/m23_64_v3_pair_reconstruction_training.py run-all
```

## Frozen SHA
- script: `{sha256_file(SCRIPT)}`
- prereg: `{sha256_file(PREREG)}`
- input manifest: `{sha256_file(R64/'input_manifest.json')}`
- R63 candidate pool: `{sha256_file(R63/'candidate_pool.parquet')}`
- R63 paired pool: `{sha256_file(R63/'paired_candidate_pool.parquet')}`
- R63 supervision: `{sha256_file(R63/'row_supervision.parquet')}`

## Pair diagnostic
| sequence | frozen pool | valid corrected | M23-63 selected valid | valid not selected |
|---|---:|---:|---:|---:|
{pair_rows}

Prior diagnostic exact replication: `{stage_a.get('checks',{}).get('pair_diagnostic_exact')}`. Candidate topology was not modified.

## Stage A gates
```json
{json.dumps(stage_a.get('checks',{}),indent=2,sort_keys=True)}
```

## Training
| seed | epochs | best epoch | best composite | checkpoint SHA |
|---:|---:|---:|---:|---|
{seed_lines}

Selected checkpoint: `{json.dumps(final.get('selection',{}),sort_keys=True)}`

MOT17 validation: `{json.dumps(final.get('external_validation',{}),sort_keys=True)}`

## Scope
`{json.dumps(final['scope_counts'],sort_keys=True)}`

No MOT20 GT, teacher, held-outer or test input was read. No tracker, TrackEval, HOTA, v2 checkpoint load, warm start, M23-54 or M23-58 was performed.
"""


def close_experiment(decision:str,stage_a:dict[str,Any]|None=None,stage_b:dict[str,Any]|None=None,training_runs:int=0,error:str="") -> dict[str,Any]:
    pair_counts={"train":0,"validation":0,"per_sequence":{}}
    if (R64/"pair_reconstruction_manifest.json").exists():
        p=read_json(R64/"pair_reconstruction_manifest.json");pair_counts={"train":p["actual_train"],"validation":p["actual_validation"],"per_sequence":p["actual_per_sequence"]}
    selection={};external={}
    if (R64/"frozen_checkpoint/checkpoint_manifest.json").exists():
        m=read_json(R64/"frozen_checkpoint/checkpoint_manifest.json");selection={"seed":m["seed"],"epoch":m["epoch"],"composite":m["composite"],"checkpoint_sha256":m["checkpoint_sha256"]}
    if (R64/"external_validation_metrics.json").exists():external=read_json(R64/"external_validation_metrics.json")
    passed=decision=="PASS_V3_FROM_SCRATCH_RELATION_TRAINING"
    final={"experiment_id":EXP_ID,"title":TITLE,"status":"closed","decision":decision,"closed_at":utc_now(),"pair_counts":pair_counts,
           "stage_a_passed":bool(stage_a and stage_a.get("passed")),"stage_b_passed":bool(stage_b and stage_b.get("passed")),
           "training_runs":training_runs,"selection":selection,"external_validation":external.get("flattened",{}),"scope_counts":{**SCOPE_ZERO,"training_runs":training_runs},
           "next_stage_authorized":passed,"hota":None,"error":error}
    if passed:
        auth={"experiment_id":EXP_ID,"authorized":True,"only_authorized_experiment":"M23-65","authorization":["reverify M23-64 checkpoint SHA","reverify M23-62 MOT20 observable SHA","run MOT20 v3 representation gate"],
              "tracker_or_trackeval_authorized_in_m23_64":False,"checkpoint_sha256":selection["checkpoint_sha256"]}
        json_write(R64/"next_stage_authorization.json",auth)
    json_write(R64/"final_summary.json",final)
    RESULT.write_text(result_document(final),encoding="utf-8")
    mark_remaining(decision);update_stage("closure","completed",decision=decision,report=str(RESULT.relative_to(ROOT)),notes="closed with structured validation")
    registry_close(final)
    active=scope_processes(); summary=read_summary(); header,rows=registry_data(); idx={k:header.index(k) for k in ["tracker_family","status","current_stage","decision"]}
    registry_ok=any(len(r)>max(idx.values()) and r[idx["tracker_family"]]==EXP_ID and r[idx["current_stage"]]=="closed" and r[idx["decision"]]==decision for r in rows)
    inp=read_json(R64/"input_manifest.json");input_unchanged=all(sha256_file(ROOT/x["path"])==x["sha256"] for x in inp["frozen_inputs"])
    checks={"final_summary_exists":(R64/"final_summary.json").is_file(),"result_exists":RESULT.is_file(),"summary_no_running_pending":all(r["status"] not in {"running","pending"} for r in summary),
            "registry_closed":registry_ok,"frozen_inputs_unchanged":input_unchanged,"no_active_relevant_processes":not active,
            "scope_counts_valid":all(final["scope_counts"][k]==0 for k in SCOPE_ZERO),"authorization_matches_decision":((R64/"next_stage_authorization.json").exists()==passed),
            "hota_empty":final["hota"] is None}
    artifacts=[p for p in R64.rglob("*") if p.is_file()]+[SCRIPT,PREREG,RESULT]
    closure={"experiment_id":EXP_ID,"created_at":utc_now(),"status":"closed","decision":decision,"scientific_passed":passed,
             "closure_integrity_passed":all(checks.values()),"checks":checks,"active_processes":active,"scope_counts":final["scope_counts"],
             "output_sha256":{str(p.relative_to(ROOT)):sha256_file(p) for p in artifacts if p!=R64/"closure_validation.json"}}
    json_write(R64/"closure_validation.json",closure);append_event("experiment_closed",decision=decision,closure_sha256=sha256_file(R64/"closure_validation.json"),training_runs=training_runs,trackeval_runs=0)
    return final


def command_run_all() -> None:
    update_stage("input_reverification","running",notes="recomputing all frozen R62/R63 SHA");registry_update_running("input_reverification")
    rv=reverify_inputs()
    if not rv["passed"]:
        update_stage("input_reverification","failed",decision="FAIL_M23_63_INPUT_REVERIFICATION",report=str((R64/"m23_63_reverification.json").relative_to(ROOT)))
        close_experiment("FAIL_M23_63_INPUT_REVERIFICATION",training_runs=0);return
    update_stage("input_reverification","completed",decision="pass",report=str((R64/"m23_63_reverification.json").relative_to(ROOT)));append_event("m23_63_inputs_reverified",report_sha256=sha256_file(R64/"m23_63_reverification.json"))
    update_stage("pair_diagnostic","running",notes="labeling all 140000 frozen pair combinations");registry_update_running("pair_diagnostic")
    pair_df,pair_manifest=build_pair_supervision()
    if not pair_manifest["diagnostic_exact"]:
        update_stage("pair_diagnostic","failed",decision="FAIL_PAIR_DIAGNOSTIC_REPLICATION",report=str((R64/"pair_reconstruction_manifest.json").relative_to(ROOT)))
        close_experiment("FAIL_PAIR_DIAGNOSTIC_REPLICATION",training_runs=0);return
    update_stage("pair_diagnostic","completed",decision="pass",report=str((R64/"pair_reconstruction_manifest.json").relative_to(ROOT)),notes="prior 182/99 exactly replicated")
    append_event("pair_diagnostic_replicated",train=pair_manifest["actual_train"],validation=pair_manifest["actual_validation"],manifest_sha256=sha256_file(R64/"pair_reconstruction_manifest.json"))
    update_stage("corrected_reconstruction","running",notes="replacing only paired tensors/metadata");registry_update_running("corrected_reconstruction")
    reconstruct_examples(pair_df);update_stage("corrected_reconstruction","completed",decision="pass",report=str((R64/"example_manifest_train.json").relative_to(ROOT)))
    update_stage("tensor_validation","running",notes="one-time NPZ member materialization; full tensor provenance");registry_update_running("tensor_validation")
    nonpaired,prov,leak,perf=validate_corrected_examples();update_stage("tensor_validation","completed" if prov["passed"] else "failed",decision="pass" if prov["passed"] else "FAIL_TENSOR_PROVENANCE",report=str((R64/"tensor_provenance_validation.json").relative_to(ROOT)))
    stage_a=stage_a_gate(rv,pair_manifest,nonpaired,prov,leak,perf)
    if not stage_a["passed"]:
        update_stage("stage_a_gate","failed",decision=stage_a["decision"],report=str((R64/"stage_a_gate.json").relative_to(ROOT)))
        close_experiment(stage_a["decision"],stage_a=stage_a,training_runs=0);return
    update_stage("stage_a_gate","completed",decision="pass",report=str((R64/"stage_a_gate.json").relative_to(ROOT)))
    append_event("corrected_examples_frozen",train_examples_sha256=sha256_file(R64/"examples_train.npz"),validation_examples_sha256=sha256_file(R64/"examples_validation.npz"),
                 train_manifest_sha256=sha256_file(R64/"example_manifest_train.json"),validation_manifest_sha256=sha256_file(R64/"example_manifest_validation.json"),
                 tensor_validator_sha256=sha256_text(inspect.getsource(validate_corrected_examples)),stage_a_gate_sha256=sha256_file(R64/"stage_a_gate.json"))
    update_stage("training_config","running",notes="building deterministic v2 adapter; training still not started");registry_update_running("training_config")
    train,train_report=build_training_arrays("train");val,val_report=build_training_arrays("validation");cfg=training_config()
    json_write(R64/"training_data_adapter.json",{"train":train_report,"validation":val_report,"created_at":utc_now()})
    update_stage("training_config","completed",decision="pass",report=str((R64/"training_config.json").relative_to(ROOT)),notes=f"train triplets={len(train['rel_group'])}; validation triplets={len(val['rel_group'])}")
    seed_reports=[]
    try:
        for seed in SEEDS:seed_reports.append(train_one_seed(seed,train,val,cfg))
    except FloatingPointError as exc:
        close_experiment("FAIL_TRAINING_NONFINITE",stage_a=stage_a,training_runs=len(seed_reports),error=repr(exc));return
    except Exception as exc:
        close_experiment("FAIL_TRAINING_INCOMPLETE",stage_a=stage_a,training_runs=len(seed_reports),error=repr(exc));return
    update_stage("checkpoint_selection","running",notes="fixed v2 composite and tie-break");registry_update_running("checkpoint_selection")
    try:
        manifest,external=select_global_checkpoint(seed_reports,val,cfg)
    except Exception as exc:
        close_experiment("FAIL_CHECKPOINT_SELECTION",stage_a=stage_a,training_runs=3,error=repr(exc));return
    update_stage("checkpoint_selection","completed",decision="pass",report=str((R64/"frozen_checkpoint/checkpoint_manifest.json").relative_to(ROOT)),notes=f"selected seed={manifest['seed']} epoch={manifest['epoch']}")
    append_event("frozen_checkpoint_selected",seed=manifest["seed"],epoch=manifest["epoch"],checkpoint_sha256=manifest["checkpoint_sha256"])
    stage_b=stage_b_gate(seed_reports,manifest,external)
    update_stage("stage_b_gate","completed" if stage_b["passed"] else "failed",decision="pass" if stage_b["passed"] else stage_b["decision"],report=str((R64/"stage_b_gate.json").relative_to(ROOT)))
    decision="PASS_V3_FROM_SCRATCH_RELATION_TRAINING" if stage_b["passed"] else stage_b["decision"]
    final=close_experiment(decision,stage_a=stage_a,stage_b=stage_b,training_runs=3)
    print(json.dumps(final,indent=2,sort_keys=True))


def main() -> None:
    p=argparse.ArgumentParser(description=TITLE);sub=p.add_subparsers(dest="command",required=True);sub.add_parser("init");sub.add_parser("run-all")
    a=p.parse_args()
    if a.command=="init":command_init()
    else:command_run_all()


if __name__=="__main__":main()
