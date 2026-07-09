#!/usr/bin/env python3
"""Phase 10: GMC (Global Motion Compensation) experiments.

Tests GMC with best params (buf=70, match=0.5, new_track=0.5).
Compares with and without GMC on MOT20-01 and MOT20-02.
"""
from __future__ import annotations
import csv, subprocess, sys
from pathlib import Path
from typing import Dict, List

REPO = Path("/gemini/code/FMtrack-main/FM-Track")
TRACKER = REPO / "scripts" / "dmm_base_tracker.py"
EVAL = REPO / "scripts" / "eval_motstyle_trackeval.py"
GT_ROOT = Path("/gemini/code/datasets/MOT20/train")
DUMP_01 = REPO / "outputs" / "dmm_phase0_mot20_01" / "MOT20-01" / "dump_yolox_reid.npz"
DUMP_02 = REPO / "outputs" / "dmm_phase0_mot20_02" / "MOT20-02" / "dump_yolox_reid.npz"
WARP_01 = REPO / "outputs" / "dmm_phase0_mot20_01" / "gmc_warps.npy"
WARP_02 = REPO / "outputs" / "dmm_phase0_mot20_02" / "gmc_warps.npy"
OUT_ROOT = REPO / "outputs" / "dmm_phase10_gmc"
BASE = ["--track-buffer", "70", "--match-thresh", "0.5", "--new-track-thresh", "0.5"]

CONFIGS = [
    {"name": "G0_ref", "gmc": False},
    {"name": "G1_gmc", "gmc": True},
    {"name": "G2_gmc_m04", "gmc": True, "match": "0.4"},
    {"name": "G3_gmc_m06", "gmc": True, "match": "0.6"},
]


def run(cmd, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return proc.returncode


def parse_summary(path):
    if not path.exists(): return {}
    lines = path.read_text().strip().splitlines()
    if len(lines) < 2: return {}
    h, v = lines[0].split(), lines[1].split()
    return {k: (float(x) if _is_float(x) else x) for k, x in zip(h, v)}


def _is_float(s):
    try: float(s); return True
    except ValueError: return False


def run_config(cfg, seq, dump, warp):
    name = cfg["name"]
    run_dir = OUT_ROOT / f"{name}_{seq}"
    eval_summary = run_dir / "trackeval_work" / "eval" / name / "pedestrian_summary.txt"
    if eval_summary.exists():
        m = parse_summary(eval_summary)
        if m.get("HOTA") is not None:
            return {"name": name, "seq": seq, "status": "completed", "HOTA": m["HOTA"], "DetA": m.get("DetA"), "AssA": m.get("AssA"), "IDSW": m.get("IDSW"), "Frag": m.get("Frag"), "CLR_FN": m.get("CLR_FN")}
    run_dir.mkdir(parents=True, exist_ok=True)
    match = cfg.get("match", "0.5")
    cmd = [sys.executable, str(TRACKER),
        "--dump-npz", str(dump), "--seq", seq,
        "--out", str(run_dir / "track_results" / f"{seq}.txt"),
        "--summary-json", str(run_dir / f"{seq}_summary.json"),
        "--assoc-mode", "botsort_reid",
        "--track-buffer", "70", "--match-thresh", match,
        "--second-match-thresh", str(min(float(match), 0.5)),
        "--new-track-thresh", "0.5"]
    if cfg["gmc"]:
        cmd += ["--gmc-enable", "--gmc-warp-path", str(warp)]
    rc = run(cmd, run_dir / "tracker.log")
    if rc != 0: return {"name": name, "seq": seq, "status": "tracker_failed"}
    eval_cmd = [sys.executable, str(EVAL),
        "--benchmark-name", "MOT20", "--split-to-eval", "train",
        "--gt-root", str(GT_ROOT), "--results-dir", str(run_dir / "track_results"),
        "--tracker-name", name, "--work-dir", str(run_dir / "trackeval_work"), "--seqs", seq]
    rc = run(eval_cmd, run_dir / "eval.log")
    if rc != 0: return {"name": name, "seq": seq, "status": "eval_failed"}
    m = parse_summary(eval_summary)
    return {"name": name, "seq": seq, "status": "completed", "HOTA": m.get("HOTA"), "DetA": m.get("DetA"), "AssA": m.get("AssA"), "IDSW": m.get("IDSW"), "Frag": m.get("Frag"), "CLR_FN": m.get("CLR_FN")}


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for cfg in CONFIGS:
        for seq, dump, warp in [("MOT20-01", DUMP_01, WARP_01), ("MOT20-02", DUMP_02, WARP_02)]:
            print(f"=== {cfg['name']} on {seq} ===", flush=True)
            r = run_config(cfg, seq, dump, warp)
            all_rows.append(r)
            print(f"  HOTA={r.get('HOTA')} IDSW={r.get('IDSW')} Frag={r.get('Frag')}", flush=True)
    fields = ["name", "seq", "status", "HOTA", "DetA", "AssA", "IDSW", "Frag", "CLR_FN"]
    with (OUT_ROOT / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in all_rows: w.writerow(r)
    print("\n=== DONE ===", flush=True)
    for r in all_rows:
        print(f"  {r['name']:20s} {r['seq']:10s} HOTA={r.get('HOTA'):>7} IDSW={r.get('IDSW'):>5} Frag={r.get('Frag'):>5}", flush=True)


if __name__ == "__main__":
    main()
