#!/usr/bin/env python3
"""Run OC-SORT observation-centric re-update experiments on MOT20-01 + MOT20-02.

Uses best params from Phase 4 sweep: buffer=70, match=0.5.
Tests OC-SORT alone and combined with other improvements.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

REPO = Path("/gemini/code/FMtrack-main/FM-Track")
TRACKER = REPO / "scripts" / "dmm_base_tracker.py"
EVAL = REPO / "scripts" / "eval_motstyle_trackeval.py"
GT_ROOT = Path("/gemini/code/datasets/MOT20/train")
DUMP_01 = REPO / "outputs" / "dmm_phase0_mot20_01" / "MOT20-01" / "dump_yolox_reid.npz"
DUMP_02 = REPO / "outputs" / "dmm_phase0_mot20_02" / "MOT20-02" / "dump_yolox_reid.npz"
OUT_ROOT = REPO / "outputs" / "dmm_phase5_ocsport"
REGISTRY = REPO / "outputs" / "experiment_registry.csv"

# Best params from Phase 4 sweep
BEST_BUFFER = 70
BEST_MATCH = 0.5

CONFIGS: List[Dict] = [
    # OC-SORT with best params from sweep
    {"name": "O0_ocsport_buf70_m05", "buffer": BEST_BUFFER, "match": BEST_MATCH, "oc_sort": True},
    # OC-SORT with even more aggressive params
    {"name": "O1_ocsport_buf70_m04", "buffer": BEST_BUFFER, "match": 0.4, "oc_sort": True},
    {"name": "O2_ocsport_buf90_m05", "buffer": 90, "match": BEST_MATCH, "oc_sort": True},
    # OC-SORT with max_gap limits (only correct short gaps, avoid bad long-gap re-activations)
    {"name": "O3_ocsport_buf70_m05_gmax20", "buffer": BEST_BUFFER, "match": BEST_MATCH, "oc_sort": True, "oc_max_gap": 20},
    {"name": "O4_ocsport_buf70_m05_gmax10", "buffer": BEST_BUFFER, "match": BEST_MATCH, "oc_sort": True, "oc_max_gap": 10},
    # Reference: best params without OC-SORT (same as P3, for comparison)
    {"name": "O5_base_buf70_m05", "buffer": BEST_BUFFER, "match": BEST_MATCH, "oc_sort": False},
]


def run(cmd: List[str], log_path: Path) -> int:
    print(f"[run] {cmd[3].split('/')[-1]}...", flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return proc.returncode


def parse_summary(path: Path) -> Dict:
    if not path.exists():
        return {}
    lines = path.read_text().strip().splitlines()
    if len(lines) < 2:
        return {}
    h, v = lines[0].split(), lines[1].split()
    out = {}
    for k, x in zip(h, v):
        try: out[k] = float(x)
        except ValueError: out[k] = x
    return out


def run_config(cfg: Dict, seq: str, dump: Path) -> Dict:
    name = cfg["name"]
    run_dir = OUT_ROOT / f"{name}_{seq}"
    out_txt = run_dir / "track_results" / f"{seq}.txt"
    summary_json = run_dir / f"{seq}_summary.json"
    log_path = run_dir / "tracker.log"

    eval_summary = run_dir / "trackeval_work" / "eval" / name / "pedestrian_summary.txt"
    if eval_summary.exists():
        m = parse_summary(eval_summary)
        if m.get("HOTA") is not None:
            print(f"  [skip] {name} on {seq}", flush=True)
            return _result(name, seq, "completed", cfg, m)

    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(TRACKER),
        "--dump-npz", str(dump), "--seq", seq,
        "--out", str(out_txt), "--summary-json", str(summary_json),
        "--assoc-mode", "botsort_reid",
        "--track-buffer", str(cfg["buffer"]),
        "--match-thresh", str(cfg["match"]),
        "--second-match-thresh", str(min(cfg["match"], 0.5)),
    ]
    if cfg.get("oc_sort"):
        cmd.append("--oc-sort-enable")
        if "oc_max_gap" in cfg:
            cmd += ["--oc-sort-max-gap", str(cfg["oc_max_gap"])]
    rc = run(cmd, log_path)
    if rc != 0:
        return _result(name, seq, "tracker_failed", cfg, {})

    eval_cmd = [
        sys.executable, str(EVAL),
        "--benchmark-name", "MOT20", "--split-to-eval", "train",
        "--gt-root", str(GT_ROOT), "--results-dir", str(run_dir / "track_results"),
        "--tracker-name", name, "--work-dir", str(run_dir / "trackeval_work"),
        "--seqs", seq,
    ]
    rc = run(eval_cmd, run_dir / "eval.log")
    if rc != 0:
        return _result(name, seq, "eval_failed", cfg, {})
    m = parse_summary(eval_summary)
    return _result(name, seq, "completed", cfg, m)


def _result(name, seq, status, cfg, m):
    return {
        "name": name, "seq": seq, "status": status,
        "buffer": cfg["buffer"], "match": cfg["match"],
        "oc_sort": cfg.get("oc_sort", False), "oc_max_gap": cfg.get("oc_max_gap", ""),
        "HOTA": m.get("HOTA"), "DetA": m.get("DetA"), "AssA": m.get("AssA"),
        "IDF1": m.get("IDF1"), "MOTA": m.get("MOTA"), "IDSW": m.get("IDSW"),
        "Frag": m.get("Frag"), "CLR_FN": m.get("CLR_FN"),
    }


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["name", "seq", "status", "buffer", "match", "oc_sort", "oc_max_gap",
              "HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW", "Frag", "CLR_FN"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"[summary] wrote {path}", flush=True)


def append_registry(rows):
    ts = datetime.now().astimezone().isoformat()
    with REGISTRY.open("a", newline="") as f:
        w = csv.writer(f)
        for r in rows:
            notes = f"OCSort buf={r['buffer']} match={r['match']} oc={r['oc_sort']} gap={r.get('oc_max_gap','')} HOTA={r.get('HOTA','')} IDSW={r.get('IDSW','')}"
            w.writerow([ts, "tracker", r["status"], str(TRACKER), "MOT20", f"train/{r['seq']}",
                        "DMMPhase5", r["name"], f"{r['name']}_{r['seq']}",
                        str(OUT_ROOT / f"{r['name']}_{r['seq']}"), "", "", "", notes, "",
                        r.get("HOTA", ""), r.get("DetA", ""), r.get("AssA", ""),
                        r.get("IDF1", ""), r.get("MOTA", ""), r.get("IDSW", ""), r.get("Frag", ""),
                        r["seq"]])
    print(f"[registry] appended", flush=True)


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for cfg in CONFIGS:
        for seq, dump in [("MOT20-01", DUMP_01), ("MOT20-02", DUMP_02)]:
            print(f"\n=== {cfg['name']} on {seq} ===", flush=True)
            r = run_config(cfg, seq, dump)
            all_rows.append(r)
            write_csv(all_rows, OUT_ROOT / "summary.csv")
            print(f"  HOTA={r.get('HOTA')} IDSW={r.get('IDSW')} Frag={r.get('Frag')}", flush=True)
    write_csv(all_rows, OUT_ROOT / "summary.csv")
    append_registry(all_rows)
    print("\n=== DONE ===", flush=True)
    for r in all_rows:
        print(f"  {r['name']:35s} {r['seq']:10s} HOTA={r.get('HOTA'):>7} IDSW={r.get('IDSW'):>5} Frag={r.get('Frag'):>5}", flush=True)


if __name__ == "__main__":
    main()
