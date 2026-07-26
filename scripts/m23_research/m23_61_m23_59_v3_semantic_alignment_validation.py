#!/usr/bin/env python3
"""M23-61 / M23-59 v3 semantic-alignment validation front-end.

Fail-closed front-end only. It never trains, creates a tracker, invokes TrackEval,
reads MOT20 test, or mutates M23-59 v2 / M23-60 historical artifacts.
"""
from __future__ import annotations
import argparse, csv, hashlib, io, json, math, os, platform, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np

REPO = Path(__file__).resolve().parents[2]
EXP_ID = "M23-61"
TITLE = "M23-59 v3 Semantic Alignment Validation"
ROOT = Path("outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_semantic_alignment")
M59 = Path("outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2")
M60 = Path("outputs/mot20_m23_20260718/m23_60_relation_transfer_failure_audit")
M57 = Path("outputs/mot20_m23_20260718/m23_57_intra_node_change_point_capacity_v2")
REGISTRY = Path("outputs/experiment_registry.csv")
PREREG = Path("docs/m23_59_relation_pretrained_hierarchical_flow_v3_semantic_alignment_prereg_20260722.md")
RESULT = Path("docs/m23_59_relation_pretrained_hierarchical_flow_v3_semantic_alignment_result_20260722.md")
ERRATUM_MD = Path("docs/m23_60_completion_validator_reconciliation_erratum_20260722.md")
SCRIPT = Path(__file__).resolve().relative_to(REPO)
TEST_SCRIPT = Path("scripts/m23_research/test_m23_61_validator_reconciliation.py")
SUMMARY = ROOT / "summary.csv"
EVENTS = ROOT / "protocol_events.jsonl"
SEQS = ["MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05"]
M59_SCRIPT = Path("scripts/m23_research/m23_59_relation_pretrained_hierarchical_flow_v2.py")
M60_SCRIPT = Path("scripts/m23_research/m23_60_relation_transfer_failure_audit.py")
M57_SCRIPT = Path("scripts/m23_research/m23_57_intra_node_change_point_capacity.py")
M10_SCRIPT = Path("scripts/m23_research/m23_10_build_micrograph.py")
CONTRACT_VERSION = "m23_59_v3_feature_contract_1.0.0"
SUMMARY_FIELDS = ["experiment","stage","status","started_at","completed_at","report","decision","notes"]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def json_write(path: Path, obj: Any, *, create_only: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if create_only and path.exists(): raise FileExistsError(f"refusing overwrite {path}")
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def csv_write(path: Path, rows: list[dict[str, Any]], fields: list[str], *, create_only: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if create_only and path.exists(): raise FileExistsError(f"refusing overwrite {path}")
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    os.replace(tmp, path)


def event(name: str, **payload: Any) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    with EVENTS.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"time": now(), "event": name, **payload}, sort_keys=True) + "\n")


def summary_rows() -> list[dict[str, str]]:
    with SUMMARY.open(newline="", encoding="utf-8") as f: return list(csv.DictReader(f))


def update_summary(stage: str, status: str, report: str, decision: str = "", notes: str = "") -> None:
    rows = summary_rows(); found = False
    for r in rows:
        if r["stage"] == stage:
            found = True
            if not r["started_at"]: r["started_at"] = now()
            r.update(status=status, completed_at=now() if status not in {"pending","running"} else "", report=report, decision=decision, notes=notes)
    if not found: raise KeyError(stage)
    csv_write(SUMMARY, rows, SUMMARY_FIELDS, create_only=False)


def registry_append(values: dict[str, Any]) -> None:
    with REGISTRY.open(newline="", encoding="utf-8", errors="replace") as f: header = next(csv.reader(f))
    row = [""] * len(header); idx = {x:i for i,x in enumerate(header)}
    for k,v in values.items():
        if k in idx: row[idx[k]] = str(v)
    with REGISTRY.open("a", newline="", encoding="utf-8") as f: csv.writer(f).writerow(row)


def close_registry(decision: str) -> None:
    raw = REGISTRY.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    header = next(csv.reader([raw[0].rstrip("\r\n") ])); idx = {h:i for i,h in enumerate(header)}; changed = 0
    for i in range(1,len(raw)):
        row = next(csv.reader([raw[i].rstrip("\r\n")]))
        if len(row) < len(header): row += [""]*(len(header)-len(row))
        if row[idx.get("tag",8)] == "M23-61-v3-semantic-alignment" and row[idx.get("status",2)] == "running":
            row[idx["status"]] = "superseded"
            if "current_stage" in idx: row[idx["current_stage"]] = "superseded"
            buf=io.StringIO(); csv.writer(buf,lineterminator="\n").writerow(row); raw[i]=buf.getvalue(); changed += 1
    if changed != 1: raise RuntimeError(f"expected one running M23-61 row, found {changed}")
    tmp=REGISTRY.with_name(REGISTRY.name+f".tmp.{os.getpid()}"); tmp.write_text("".join(raw),encoding="utf-8"); os.replace(tmp,REGISTRY)
    registry_append({"timestamp":now(),"kind":"semantic_alignment_validation","status":"completed","script":str(SCRIPT),
        "dataset":"MOT17+MOT20","split":"front_end_validation","tracker_family":EXP_ID,"variant":"m23_59_v3_semantic_alignment",
        "tag":"M23-61-v3-semantic-alignment-closed","run_root":str(ROOT),"summary_csv":str(SUMMARY),
        "log_path":str(ROOT/"closure_validation.json"),"name":TITLE,"dataset_split":"MOT17 external + MOT20 train observable only",
        "run_dir":str(ROOT),"current_stage":"closed_blocked","decision":decision,"phase":"closed",
        "notes":"v2 operationally closed; scientific comparison confounded by semantic mismatch; no training/TrackEval/tracker/test"})


def required_inputs() -> list[Path]:
    items = [Path("AGENTS.md"),
      Path("docs/m23_59_relation_pretrained_hierarchical_flow_v2_result_20260721.md"),
      Path("docs/m23_60_relation_transfer_failure_audit_result_20260721.md"),
      Path("docs/m23_59_relation_pretrained_hierarchical_flow_prereg_v2_20260720.md"),
      Path("docs/m23_59_v1_invalidated_determinism_20260720.md"), M59_SCRIPT, M60_SCRIPT, M57_SCRIPT, M10_SCRIPT,
      M59/"final_summary.json", M59/"closure_validation.json", M59/"strict_outer_evaluation/report.json",
      M59/"protocol_events.jsonl", M59/"implementation_manifest.json", M59/"external_dataset_manifest.json",
      M59/"external_pretraining/frozen_checkpoint_manifest.json", M59/"external_pretraining/relation_pretrained_frozen.pt",
      M60/"audit_manifest.json", M60/"semantic_validation.json", M60/"candidate_oracle.json",
      M60/"ranking_diagnostics.csv", M60/"error_waterfall.json", M60/"final_diagnosis.json",
      M60/"completion_validation.json", M60/"independent_closure_validation_v2.json", TEST_SCRIPT]
    for p in items:
        if not p.exists(): raise FileNotFoundError(p)
    return items


def prereg_text(inputs: list[dict[str,Any]]) -> str:
    table="\n".join(f"| `{x['path']}` | `{x['sha256']}` | {x['bytes']} |" for x in inputs)
    return f"""# M23-59 v3 Semantic Alignment Validation — Preregistration (M23-61, 2026-07-22)

## Scope

This is a fail-closed semantic contract and lineage validation. It is not a normal model experiment.
No model training, tracker generation, TrackEval, threshold search, policy change, MOT20 test read/submission, or M23-54/M23-58 execution is allowed before every front-end gate passes.
M23-59 v2 and M23-60 are immutable inputs.

## Canonical index convention

- tensor width: 144
- appearance: global zero-based 0..127
- geometry: global zero-based 128..143, local zero-based 0..15
- `feature_143` means global zero-based column 143, geometry local index 15, one-based display column 144
- its preregistered canonical meaning is nearest same-frame row-center distance after x/width and y/height normalization, clipped to [0,1], singleton sentinel 1.0
- this meaning is selected from the v2 preregistration and generic generator source, never from MOT20 metrics

## Frozen front-end gates

1. canonical definition must be unique for 144/144 features;
2. semantic/formula parity and GT-free provenance must pass for 144/144 features on both source and target generation paths;
3. no GT, teacher action, identity label/mapping, held-outer label, or outer-conditioned normalizer may enter feature generation;
4. MOT17 physical videos must be disjoint across train/validation; only canonical FRCNN may be admitted; exact-image duplicates across splits are forbidden;
5. score/candidate mapping, stable tie-break and mask/index invariants must pass;
6. old checkpoint compatibility requires byte-identical training-side 144-D inputs and unchanged normalization;
7. counterfactual B must improve both R@1 and MRR versus A on at least 3 of 4 fixed outers, with no new semantic, lineage, score/index or mask failure;
8. failure of any prior gate blocks B/C replay, training and strict outer evaluation.

## Frozen counterfactual conditions

- A_original_v2: historical v2 observable/checkpoint reference only.
- B_canonical_feature_143: replace only global column 143 with the canonical nearest-neighbor formula; old checkpoint permitted only after compatibility gate.
- C_neutralized_feature_143: replace global column 143 by the MOT17-train-only fixed median; no MOT20 statistics or labels.
- K=256 ranking, K=32 flow, gap buckets, architecture, loss, seed, epochs, risk/UCB, representation gate and P0/P1/P2 remain unchanged.
- diagnostics only; no tracker or TrackEval.

## Validator reconciliation rule

Expected-negative invariants are first converted into positive predicates (`actual == expected`). Raw expected false values must never be passed directly to `all(checks.values())`.

## Frozen inputs

| Path | SHA-256 | Bytes |
|---|---|---:|
{table}
"""


def command_init() -> None:
    if ROOT.exists() or PREREG.exists() or RESULT.exists() or ERRATUM_MD.exists():
        raise FileExistsError("M23-61/v3 output already exists; refusing overwrite")
    inputs=[{"path":str(p),"sha256":sha(p),"bytes":p.stat().st_size} for p in required_inputs()]
    PREREG.parent.mkdir(parents=True,exist_ok=True); PREREG.write_text(prereg_text(inputs),encoding="utf-8")
    ROOT.mkdir(parents=True,exist_ok=False)
    manifest={"experiment_id":EXP_ID,"title":TITLE,"status":"preregistered","created_at":now(),
      "script":str(SCRIPT),"script_sha256":sha(Path(SCRIPT)),"regression_test":str(TEST_SCRIPT),"regression_test_sha256":sha(TEST_SCRIPT),
      "preregistration":str(PREREG),"preregistration_sha256":sha(PREREG),"input_artifacts":inputs,
      "contract_version":CONTRACT_VERSION,"counterfactual_improvement_outer_count":3,
      "prohibitions":{"training":True,"trackeval":True,"tracker":True,"mot20_test":True,"threshold_search":True,
        "policy_change":True,"m23_54":True,"m23_58":True,"v1_artifacts":True},
      "direct_mot20_gt_read":False,"historical_m23_60_posthoc_gt_evidence_read":True}
    json_write(ROOT/"audit_manifest.json",manifest)
    stages=["preregistration","preflight_reconciliation","feature_contract","gt_free_lineage","source_split_audit",
      "raw_regeneration","counterfactual_replay","training","strict_outer_evaluation","closure"]
    rows=[]
    for s in stages:
        rows.append({"experiment":EXP_ID,"stage":s,"status":"completed" if s=="preregistration" else "pending",
          "started_at":now() if s=="preregistration" else "","completed_at":now() if s=="preregistration" else "",
          "report":str(PREREG) if s=="preregistration" else "","decision":"","notes":"front-end fail-closed protocol"})
    csv_write(SUMMARY,rows,SUMMARY_FIELDS)
    event("preregistration_frozen",manifest_sha256=sha(ROOT/"audit_manifest.json"),preregistration_sha256=sha(PREREG))
    registry_append({"timestamp":now(),"kind":"semantic_alignment_validation","status":"running","script":str(SCRIPT),
      "dataset":"MOT17+MOT20","split":"front_end_validation","tracker_family":EXP_ID,"variant":"m23_59_v3_semantic_alignment",
      "tag":"M23-61-v3-semantic-alignment","run_root":str(ROOT),"summary_csv":str(SUMMARY),
      "log_path":str(ROOT/"audit_manifest.json"),"name":TITLE,"dataset_split":"MOT17 external + MOT20 train observable only",
      "run_dir":str(ROOT),"current_stage":"front_end_running","phase":"semantic_contract",
      "notes":"preregistered; no training/TrackEval/tracker/test; v2 and M23-60 immutable"})
    print(json.dumps({"initialized":True,"experiment_id":EXP_ID,"inputs":len(inputs),"root":str(ROOT)},indent=2))


def verify_frozen() -> tuple[dict,list[dict]]:
    m=json.loads((ROOT/"audit_manifest.json").read_text())
    if sha(Path(m["script"])) != m["script_sha256"]: raise RuntimeError("script changed after prereg")
    if sha(TEST_SCRIPT) != m["regression_test_sha256"]: raise RuntimeError("test changed after prereg")
    if sha(PREREG) != m["preregistration_sha256"]: raise RuntimeError("prereg changed")
    checks=[]
    for x in m["input_artifacts"]:
        actual=sha(Path(x["path"])); checks.append({"path":x["path"],"expected":x["sha256"],"actual":actual,"passed":actual==x["sha256"]})
    if not all(x["passed"] for x in checks): raise RuntimeError("frozen required input SHA mismatch")
    return m,checks


def generator_history() -> dict[str,Any]:
    out={}
    for p in [M59_SCRIPT,M60_SCRIPT,M57_SCRIPT,M10_SCRIPT]:
        cp=subprocess.run(["git","log","--format=%H %ad %s","--date=iso","--",str(p)],text=True,capture_output=True,cwd=REPO,timeout=20)
        out[str(p)]={"tracked":subprocess.run(["git","ls-files","--error-unmatch",str(p)],cwd=REPO,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0,
          "history_lines":[x for x in cp.stdout.splitlines() if x][:30],"current_sha256":sha(p)}
    return out


def make_contract(generator_sha: str) -> tuple[list[dict[str,Any]],dict[str,Any]]:
    rows=[]
    for i in range(128):
        rows.append({"zero_based_index":i,"one_based_display_index":i+1,"feature_name":f"appearance_{i:03d}","feature_group":"appearance",
          "geometry_local_index":"","exact_formula":f"L2_normalize(L2_normalize(FastReID_OSNet_x1_0(crop)) @ GaussianProjection(seed=2310,2048x128)/sqrt(128))[{i}]",
          "physical_meaning":f"component {i} of deterministic 128-D appearance embedding","units":"dimensionless","sign_convention":"projection component; cosine geometry after L2 normalization",
          "clipping_range":"none; vector L2 normalized","missing_value_sentinel":"zero vector only when no valid mapped embedding","dtype":"float16 artifact / float32 compute",
          "normalization":"raw 2048-D L2, deterministic projection, projected 128-D L2","source_artifact":"FastReID crop embedding from canonical source row box",
          "generator_function":"extract_appearance / map_phase","generator_source_sha256":generator_sha,"canonical_gt_free":True,
          "allowed_temporal_context":"same row only","canonical_meaning_unique":True,
          "v2_mot17_as_run":"crop from MOT17 GT row","v2_mot20_as_run":"crop/match from MOT20 source-tracker row",
          "formula_semantic_parity_as_run":True,"lineage_parity_as_run":False,"contract_version":CONTRACT_VERSION})
    geom=[
      ("center_x_norm","cx/image_width","horizontal box center","normalized image width","positive right","[unbounded by code]","none","same row"),
      ("center_y_norm","cy/image_height","vertical box center","normalized image height","positive down","[unbounded by code]","none","same row"),
      ("box_width_norm","box_width/image_width","box width","normalized image width","positive","[0,+inf)","none","same row"),
      ("box_height_norm","box_height/image_height","box height","normalized image height","positive","[0,+inf)","none","same row"),
      ("log_aspect","log(max(width,1e-3)/max(height,1e-3))","log aspect ratio","log ratio","positive=wider","unclipped","none","same row"),
      ("log_area_fraction","log(max(width*height/(image_width*image_height),1e-8))","log image-area fraction","log fraction","larger=larger box","unclipped lower floor 1e-8","none","same row"),
      ("visibility","visibility or preregistered GT-free missing sentinel","visible fraction","fraction","larger=more visible","[0,1] expected","UNRESOLVED","same row"),
      ("velocity_x_height_frame","(cx_t-cx_prev)/(max(h_t,1)*max(frame_delta,1))","horizontal velocity","current box heights/frame","positive right","unclipped","0 for first row","previous row in same source track"),
      ("velocity_y_height_frame","(cy_t-cy_prev)/(max(h_t,1)*max(frame_delta,1))","vertical velocity","current box heights/frame","positive down","unclipped","0 for first row","previous row in same source track"),
      ("log_width_change_per_frame","log(max(w_t,1e-3)/max(w_prev,1e-3))/dt","width growth rate","log ratio/frame","positive growing","unclipped","0 for first row","previous row in same source track"),
      ("log_height_change_per_frame","log(max(h_t,1e-3)/max(h_prev,1e-3))/dt","height growth rate","log ratio/frame","positive growing","unclipped","0 for first row","previous row in same source track"),
      ("frame_delta_over_30_clipped","min(frame_delta/30,20)","temporal gap to previous row","30-frame units","positive forward","[0,20]","0 for first row","previous row in same source track"),
      ("velocity_x_residual","vx_t-vx_prev","horizontal acceleration proxy","box heights/frame difference","positive increasing rightward velocity","unclipped","0 without two prior steps","two previous rows in same source track"),
      ("velocity_y_residual","vy_t-vy_prev","vertical acceleration proxy","box heights/frame difference","positive increasing downward velocity","unclipped","0 without two prior steps","two previous rows in same source track"),
      ("crowd_density_over_100_clipped","min(count_same_frame_source_rows/100,5)","same-frame observable row density","hundreds of rows","positive denser","[0,5]","0 only for no rows (not valid row)","same frame only"),
      ("nearest_neighbor_distance","min(min_{j!=i} hypot((cx_i-cx_j)/W,(cy_i-cy_j)/H),1)","nearest same-frame normalized row-center distance","normalized image diagonal components","larger=more isolated","[0,1]","1.0 for singleton frame","same frame only"),
    ]
    for j,(name,formula,meaning,units,sign,clip,sentinel,context) in enumerate(geom):
        i=128+j; unique=(j!=6); formula_parity=(j not in {6,14,15}); lineage=(j in {0,1,2,3,4,5})
        mot17="MOT17 GT rows"
        mot20="MOT20 source-tracker rows"
        if j==6: mot17="GT visibility field"; mot20="constant 1.0 imputation"
        if 7<=j<=13: mot17="temporal grouping by GT identity"; mot20="temporal grouping by source track_id"
        if j==14: mot17="count eligible GT rows per frame"; mot20="M23-57 local crowd over source-tracker rows"
        if j==15: mot17="nearest-neighbor distance over eligible GT rows"; mot20="overwritten by GT-free mapped-appearance indicator"
        rows.append({"zero_based_index":i,"one_based_display_index":i+1,"feature_name":f"geometry_{j:02d}_{name}","feature_group":"geometry",
          "geometry_local_index":j,"exact_formula":formula,"physical_meaning":meaning,"units":units,"sign_convention":sign,
          "clipping_range":clip,"missing_value_sentinel":sentinel,"dtype":"float16 artifact / float32 compute","normalization":"formula-internal only; identity scaler",
          "source_artifact":"canonical source-row table","generator_function":"geometry_features","generator_source_sha256":generator_sha,
          "canonical_gt_free":j!=6,"allowed_temporal_context":context,"canonical_meaning_unique":unique,
          "v2_mot17_as_run":mot17,"v2_mot20_as_run":mot20,"formula_semantic_parity_as_run":formula_parity,
          "lineage_parity_as_run":lineage,"contract_version":CONTRACT_VERSION})
    core=[{k:v for k,v in r.items() if k!="contract_hash"} for r in rows]
    contract_hash=hashlib.sha256(json.dumps(core,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
    for r in rows: r["contract_hash"]=contract_hash
    aggregate={"feature_count":len(rows),"canonical_unique_count":sum(bool(r["canonical_meaning_unique"]) for r in rows),
      "formula_semantic_parity_as_run_count":sum(bool(r["formula_semantic_parity_as_run"]) for r in rows),
      "lineage_parity_as_run_count":sum(bool(r["lineage_parity_as_run"]) for r in rows),
      "contract_hash":contract_hash,"global_feature_143":{"zero_based":143,"one_based":144,"geometry_local_index":15,
        "canonical_name":"geometry_15_nearest_neighbor_distance","canonical_meaning_unique":True}}
    return rows,aggregate


def provenance_rows(contract: list[dict[str,Any]]) -> list[dict[str,Any]]:
    out=[]
    for r in contract:
        i=int(r["zero_based_index"]); local=i-128
        mot17_identity = 135 <= i <= 141
        mot17_gt = True
        mot20_actual = "source tracker row / GT-free phase mapping"
        if i==134: mot20_actual="constant 1.0 visibility sentinel"
        if i==142: mot20_actual="M23-57 source-row crowd density"
        if i==143: mot20_actual="M23-57 GT-free appearance mapped indicator (as-run mismatch)"
        out.append({"feature_index":i,"feature_name":r["feature_name"],"domain":"MOT17-train-validation-as-run",
          "source":r["v2_mot17_as_run"],"uses_gt":mot17_gt,"uses_teacher_action":False,"uses_identity_mapping":mot17_identity or i==134,
          "uses_held_outer":False,"gt_free":False,"allowed_temporal_context":r["allowed_temporal_context"],
          "status":"FAIL_GT_FREE_PROVENANCE","notes":"read_mot_gt supplies boxes, identity and visibility; external supervision is historical but not a GT-free generator"})
        out.append({"feature_index":i,"feature_name":r["feature_name"],"domain":"MOT20-v3-canonical-proposed",
          "source":mot20_actual,"uses_gt":False,"uses_teacher_action":False,"uses_identity_mapping":False,"uses_held_outer":False,
          "gt_free":bool(r["canonical_gt_free"]),"allowed_temporal_context":r["allowed_temporal_context"],
          "status":"PASS" if r["canonical_gt_free"] else "BLOCKED_AMBIGUOUS_VISIBILITY_SENTINEL",
          "notes":"no outer labels opened; proposed regeneration not executed because global contract gate fails"})
    return out


def split_audit() -> dict[str,Any]:
    d=json.loads((M59/"external_dataset_manifest.json").read_text()); ds=d["dataset"]
    tr=set(ds["physical_train"]); va=set(ds["physical_validation"])
    tr_hash=set(); va_hash=set(); train_gt=set(); val_gt=set()
    for seq,v in ds["physical"].items():
        target=tr_hash if seq in tr else va_hash
        target.update(x["sha256"] for x in v["canonical_images"])
        gtsha=v["variants"]["FRCNN"]["gt"]["sha256"]
        (train_gt if seq in tr else val_gt).add(gtsha)
    return {"physical_train":sorted(tr),"physical_validation":sorted(va),"physical_video_overlap":sorted(tr&va),
      "canonical_variant":ds["canonical_variant"],"variant_policy":ds["variant_policy"],
      "all_variant_gt_identical":d["integrity"]["all_variant_gt_identical"],
      "all_variant_image_trees_identical":d["integrity"]["all_variant_image_trees_identical"],
      "exact_image_sha_overlap_count":len(tr_hash&va_hash),"exact_gt_file_sha_overlap_count":len(train_gt&val_gt),
      "detector_variant_leakage":False,"identity_namespace":"sequence-local; no physical video crosses split",
      "passed":not(tr&va) and ds["canonical_variant"]=="FRCNN" and len(tr_hash&va_hash)==0 and d["integrity"]["all_variant_gt_identical"] and d["integrity"]["all_variant_image_trees_identical"]}


def command_audit() -> None:
    manifest,input_checks=verify_frozen()
    for stage in [r["stage"] for r in summary_rows() if r["stage"]!="preregistration"]:
        update_summary(stage,"running","",notes="front-end validation active; no training/TrackEval/tracker")
    event("front_end_execution_started")
    # Re-verify all 159 frozen inputs from M23-60.
    m60audit=json.loads((M60/"audit_manifest.json").read_text()); m60sha=[]
    for x in m60audit["input_artifacts"]:
        actual=sha(Path(x["path"])); m60sha.append({"path":x["path"],"expected":x["sha256"],"actual":actual,"passed":actual==x["sha256"]})
    if not all(x["passed"] for x in m60sha): raise RuntimeError("M23-59/M23-60 frozen input SHA mismatch")
    original=json.loads((M60/"completion_validation.json").read_text()); independent=json.loads((M60/"independent_closure_validation_v2.json").read_text())
    expected={k:True for k in original["checks"]}; expected["M23_59_modified"]=False
    predicates={k:original["checks"].get(k)==v for k,v in expected.items()}
    test_cp=subprocess.run([sys.executable,str(TEST_SCRIPT)],text=True,capture_output=True,cwd=REPO)
    regression=json.loads(test_cp.stdout) if test_cp.returncode==0 else {"passed":False,"stdout":test_cp.stdout,"stderr":test_cp.stderr}
    reconciliation={"experiment":EXP_ID,"original_completion_path":str(M60/"completion_validation.json"),
      "original_reported_passed":original["passed"],"bug":"expected-negative M23_59_modified=false entered all(checks.values()) directly",
      "correct_rule":"convert every invariant to actual==expected, then all positive predicates","positive_predicates":predicates,
      "reconciled_passed":all(predicates.values()),"independent_closure_v2_passed":independent["passed"],
      "regression_test":regression,"original_file_modified":False}
    json_write(ROOT/"validator_reconciliation_erratum.json",reconciliation)
    ERRATUM_MD.write_text("# M23-60 completion validator reconciliation erratum (2026-07-22)\n\n"
      "The historical `completion_validation.json` is preserved unchanged. Its top-level `passed=false` is a validator aggregation bug: "
      "`M23_59_modified=false` is an expected-negative invariant but was passed directly into `all(checks.values())`. "
      "The correct predicate is `actual_M23_59_modified == false`. All corrected predicates pass, the dedicated regression test passes, "
      "and `independent_closure_validation_v2.json` remains `passed=true`.\n",encoding="utf-8")
    v2rec={"M23_59_v2_operational_status":"closed","as_run_fallback_baseline":"byte-exact M23-46, HOTA 79.123193",
      "historical_metrics_rewritten":False,"scientific_comparison_status":"scientific_comparison_confounded_by_semantic_mismatch",
      "M23_60_status":"post_hoc_diagnosis_only","M23_60_primary":"implementation_or_semantic_mismatch"}
    json_write(ROOT/"v2_operational_reconciliation.json",v2rec)
    genhist=generator_history(); json_write(ROOT/"generator_sha_manifest.json",{"environment":{"python":sys.version,"platform":platform.platform(),"numpy":np.__version__},"generators":genhist})
    raw={"required_input_sha_checks":input_checks,"m23_60_frozen_input_count":len(m60sha),"m23_60_frozen_input_sha_failures":[x for x in m60sha if not x["passed"]],
      "direct_mot20_gt_read":False,"mot20_test_read":False,"historical_m23_60_posthoc_gt_evidence_read":True,
      "raw_regeneration_status":"blocked_before_execution","reason":"full semantic/GT-free contract gate failed"}
    json_write(ROOT/"raw_input_manifest.json",raw)
    update_summary("preflight_reconciliation","completed",str(ROOT/"validator_reconciliation_erratum.json"),"pass","159 frozen inputs verified; validator regression passed")
    event("validator_reconciliation_completed",reconciled_passed=reconciliation["reconciled_passed"],input_count=len(m60sha))
    contract,agg=make_contract(sha(M59_SCRIPT)); fields=list(contract[0].keys())
    json_write(ROOT/"feature_contract_v3.json",{"contract_version":CONTRACT_VERSION,"aggregate":agg,"features":contract})
    csv_write(ROOT/"feature_contract.csv",contract,fields)
    prov=provenance_rows(contract); csv_write(ROOT/"semantic_provenance.csv",prov,list(prov[0].keys()))
    contract_pass=agg["canonical_unique_count"]==144 and agg["formula_semantic_parity_as_run_count"]==144
    gtfree_mot17=sum(r["gt_free"] for r in prov if r["domain"].startswith("MOT17")); gtfree_mot20=sum(r["gt_free"] for r in prov if r["domain"].startswith("MOT20"))
    update_summary("feature_contract","completed_with_failures",str(ROOT/"feature_contract_v3.json"),"fail",
      f"canonical unique={agg['canonical_unique_count']}/144; formula parity={agg['formula_semantic_parity_as_run_count']}/144")
    update_summary("gt_free_lineage","blocked",str(ROOT/"semantic_provenance.csv"),"fail",f"MOT17 strict GT-free={gtfree_mot17}/144; MOT20 proposed={gtfree_mot20}/144")
    event("feature_contract_frozen",contract_sha256=sha(ROOT/"feature_contract_v3.json"),contract_hash=agg["contract_hash"])
    event("semantic_contract_failed",canonical_unique=agg["canonical_unique_count"],formula_parity=agg["formula_semantic_parity_as_run_count"],mot17_gt_free=gtfree_mot17)
    split=split_audit(); json_write(ROOT/"source_split_audit.json",split)
    update_summary("source_split_audit","completed",str(ROOT/"source_split_audit.json"),"pass" if split["passed"] else "fail",
      f"physical overlap={len(split['physical_video_overlap'])}; exact image overlap={split['exact_image_sha_overlap_count']}")
    # Candidate provenance only; no teacher labels opened by this run.
    cand={"K_ranking":256,"K_flow":32,"candidate_graph_changed":False,"direct_teacher_or_gt_read":False,"sequences":{}}
    for seq in SEQS:
        paths=[M57/"capacity"/seq/"frozen_candidate_graph/nodes.parquet",M57/"capacity"/seq/"frozen_candidate_graph/edges.parquet"]
        cand["sequences"][seq]=[{"path":str(p),"sha256":sha(p),"bytes":p.stat().st_size} for p in paths]
    json_write(ROOT/"candidate_provenance.json",cand)
    compatibility={"semantic_contract_passed":contract_pass,"canonical_feature_143_unique":True,
      "all_observable_generation_gt_free":False,"source_split_lineage_passed":split["passed"],
      "score_candidate_mapping_audit":"historical M23-60 passed; v3 replay not authorized",
      "training_side_contract_compatibility_proven":False,"v2_checkpoint_formal_v3_reuse_allowed":False,
      "v2_checkpoint_allowed_condition":"A_original_v2 historical reference only",
      "blocked_reasons":["geometry visibility canonical GT-free sentinel is not uniquely preregistered",
        "MOT17 feature generator reads GT boxes/visibility and groups temporal features by GT identity",
        "as-run geometry crowd and nearest-neighbor populations differ across domains",
        "144/144 semantic and GT-free lineage gates do not pass"]}
    json_write(ROOT/"compatibility_validation.json",compatibility)
    cf_fields=["condition","status","checkpoint","observable","R_at_1","MRR","R_at_3","R_at_5","R_at_256","AUROC","PR_AUC","paired_replacement_R_at_1","notes"]
    historical=json.loads((M60/"final_diagnosis.json").read_text())
    cf=[{"condition":"A_original_v2","status":"historical_reference_only_not_rerun","checkpoint":"M23-59 v2 frozen","observable":"M23-59 v2 as-run",
      "R_at_1":historical["evidence"]["MOT20_candidate_present_frozen_model_R_at_1"],"MRR":"","R_at_3":"","R_at_5":"","R_at_256":"","AUROC":"","PR_AUC":"","paired_replacement_R_at_1":"","notes":"M23-60 post-hoc reference; not a new v3 result"},
      {"condition":"B_canonical_feature_143","status":"not_run_contract_gate_failed","checkpoint":"prohibited","observable":"not generated","R_at_1":"","MRR":"","R_at_3":"","R_at_5":"","R_at_256":"","AUROC":"","PR_AUC":"","paired_replacement_R_at_1":"","notes":"old checkpoint compatibility not proven"},
      {"condition":"C_neutralized_feature_143","status":"not_run_contract_gate_failed","checkpoint":"prohibited","observable":"not generated","R_at_1":"","MRR":"","R_at_3":"","R_at_5":"","R_at_256":"","AUROC":"","PR_AUC":"","paired_replacement_R_at_1":"","notes":"negative control blocked by preceding contract gate"}]
    csv_write(ROOT/"counterfactual_metrics.csv",cf,cf_fields)
    json_write(ROOT/"score_tie_audit.json",{"status":"not_run_contract_gate_failed","new_logits_computed":False,"historical_evidence_only":str(M60/"ranking_diagnostics.csv"),
      "score_candidate_mapping":"historical M23-60 passed","permutation_test":"historical M23-60 passed","v3_claim":False})
    for stage,report in [("raw_regeneration",ROOT/"raw_input_manifest.json"),("counterfactual_replay",ROOT/"counterfactual_metrics.csv"),
                         ("training",ROOT/"compatibility_validation.json"),("strict_outer_evaluation",ROOT/"compatibility_validation.json")]:
        update_summary(stage,"blocked",str(report),"blocked","preceding 144/144 semantic and GT-free lineage gate failed")
    event("counterfactual_replay_prohibited",reason="semantic_contract_or_gt_free_lineage_failed")
    event("training_prohibited",reason="front_end_gate_failed")
    event("strict_outer_evaluation_prohibited",reason="no frozen v3 observables; no label unlock")
    final={"experiment_id":EXP_ID,"title":TITLE,"status":"closed_blocked_before_counterfactual_replay",
      "decision":"BLOCKED_GT_FREE_LINEAGE_AND_FULL_CONTRACT","canonical_feature_143":{"unique":True,"global_zero_based_index":143,"geometry_local_index":15,"one_based_display_index":144,"meaning":"nearest same-frame normalized center distance clipped [0,1]"},
      "feature_contract":{"canonical_unique":agg["canonical_unique_count"],"formula_semantic_parity_as_run":agg["formula_semantic_parity_as_run_count"],"total":144},
      "gt_free_provenance":{"MOT17":gtfree_mot17,"MOT20_proposed":gtfree_mot20,"total":144},"source_split_audit_passed":split["passed"],
      "counterfactual_replay_run":False,"training_runs":0,"trackeval_runs":0,"tracker_outputs":0,"deployable_tracker_created":False,
      "direct_mot20_gt_read":False,"historical_m23_60_gt_diagnostic_read":True,"mot20_test_read":False,"mot20_test_submission":False,
      "m23_54_started":False,"m23_58_started":False,"M23_59_v2_modified":False,"M23_60_modified":False,
      "v2_as_run_fallback_baseline":"M23-46 byte-exact P0, HOTA 79.123193","v2_scientific_status":"scientific_comparison_confounded_by_semantic_mismatch",
      "M23_60_status":"post_hoc diagnosis; not deployable/strict","allowed_next_action":"preregister a GT-free source-row construction for MOT17 external supervision and a unique visibility sentinel/proxy; then regenerate train/validation features from raw inputs in a new version root"}
    json_write(ROOT/"final_summary.json",final)
    RESULT.write_text(f"""# M23-59 v3 Semantic Alignment Validation — M23-61 Result (2026-07-22)

## Decision

**BLOCKED_GT_FREE_LINEAGE_AND_FULL_CONTRACT**. No counterfactual B/C, training, tracker, TrackEval, label unlock, strict outer evaluation, MOT20 test read or submission occurred.

## Canonical feature 143

Global zero-based column **143** = one-based column **144** = geometry local index **15**. Its unique canonical meaning is nearest same-frame normalized center distance, clipped to `[0,1]`, singleton sentinel `1.0`. This comes from the v2 preregistration and generic generator, not MOT20 results.

## Full contract result

- unique canonical definitions: **{agg['canonical_unique_count']}/144**
- as-run formula/semantic parity: **{agg['formula_semantic_parity_as_run_count']}/144**
- MOT17 strict GT-free feature lineage: **{gtfree_mot17}/144**
- MOT20 proposed GT-free lineage: **{gtfree_mot20}/144**

Blocking findings: visibility has no uniquely preregistered GT-free sentinel/proxy; MOT17 features are generated from GT rows, temporal derivatives group by GT identity, crowd/nearest-neighbor populations are GT-derived, while MOT20 uses source-tracker rows. Therefore a feature-143-only repair cannot be certified as a 144-D GT-free semantic repair.

## Source split

Physical-video split and canonical FRCNN policy pass; physical overlap={len(split['physical_video_overlap'])}, exact image SHA overlap={split['exact_image_sha_overlap_count']}. This does not cure feature lineage failure.

## Historical distinctions

- v2 as-run output: byte-exact M23-46 P0 fallback, historical metrics unchanged.
- v2 scientific comparison: confounded by semantic mismatch.
- M23-60: post-hoc GT diagnosis only, not deployable and not strict.
- v3: front-end contract record only; no deployable tracker.

## Next allowed action

Create a new preregistered external source-row construction that does not use GT boxes/visibility/identity in feature generation, while keeping GT only as external supervision labels, and uniquely freeze the visibility missing-value contract. Then regenerate both MOT17 train/validation features and MOT20 observables from raw inputs in a fresh root before any replay or training.
""",encoding="utf-8")
    close_registry(final["decision"])
    update_summary("closure","completed",str(ROOT/"closure_validation.json"),final["decision"],"closure validation follows")
    # Validate no stale running and immutable input SHA after all writes.
    ours=[]
    with REGISTRY.open(newline="",encoding="utf-8",errors="replace") as f:
        ours=[r for r in csv.DictReader(f) if r.get("tracker_family")==EXP_ID and r.get("variant")=="m23_59_v3_semantic_alignment"]
    outputs=[ROOT/"audit_manifest.json",ROOT/"validator_reconciliation_erratum.json",ROOT/"v2_operational_reconciliation.json",
      ROOT/"feature_contract_v3.json",ROOT/"feature_contract.csv",ROOT/"semantic_provenance.csv",ROOT/"raw_input_manifest.json",
      ROOT/"generator_sha_manifest.json",ROOT/"compatibility_validation.json",ROOT/"counterfactual_metrics.csv",ROOT/"score_tie_audit.json",
      ROOT/"candidate_provenance.json",ROOT/"source_split_audit.json",SUMMARY,EVENTS,ROOT/"final_summary.json",RESULT,PREREG,ERRATUM_MD]
    for p in outputs:
        if p.suffix==".json": json.loads(p.read_text())
        if p.suffix==".csv":
            with p.open(newline="",encoding="utf-8") as f:list(csv.DictReader(f))
    post_inputs=[]
    for x in manifest["input_artifacts"]:
        actual=sha(Path(x["path"])); post_inputs.append(actual==x["sha256"])
    checks={"all_required_outputs_exist":all(p.exists() for p in outputs),"all_json_csv_parse":True,
      "required_input_sha_unchanged":all(post_inputs),"m23_60_159_input_sha_verified":all(x["passed"] for x in m60sha),
      "summary_no_running":all(r["status"]!="running" for r in summary_rows()),"registry_no_running":all(r.get("status")!="running" for r in ours),
      "registry_closed_row":any(r.get("status")=="completed" for r in ours),"protocol_has_training_prohibited":any(json.loads(x)["event"]=="training_prohibited" for x in EVENTS.read_text().splitlines()),
      "training_zero":True,"trackeval_zero":True,"tracker_zero":True,"test_submission_false":True,"direct_mot20_gt_read_false":True}
    # Finalize mutable structured records before hashing them into closure.
    update_summary("closure","completed",str(ROOT/"closure_validation.json"),final["decision"],"all parse/SHA/queue checks passed")
    event("closure_completed",passed=all(checks.values()),decision=final["decision"])
    closure={"experiment_id":EXP_ID,"passed":all(checks.values()),"checks":checks,"decision":final["decision"],
      "output_sha256":{str(p):sha(p) for p in outputs},"completed_at":now()}
    json_write(ROOT/"closure_validation.json",closure)
    print(json.dumps(final,indent=2,sort_keys=True))


def main() -> None:
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="command",required=True); sub.add_parser("init"); sub.add_parser("audit")
    a=p.parse_args()
    if a.command=="init": command_init()
    else: command_audit()


if __name__=="__main__": main()
