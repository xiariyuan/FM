#!/usr/bin/env python3
"""Phase 9: NSA Kalman (Noise Scale Adaptive).

Base: buf=70, match=0.5, new_track=0.5 (Phase 6/7 best).
"""
from __future__ import annotations
import csv, subprocess, sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

REPO = Path("/gemini/code/FMtrack-main/FM-Track")
TRACKER = REPO / "scripts" / "dmm_base_tracker.py"
EVAL = REPO / "scripts" / "eval_motstyle_trackeval.py"
GT_ROOT = Path("/gemini/code/datasets/MOT20/train")
DUMP_01 = REPO / "outputs" / "dmm_phase0_mot20_01" / "MOT20-01" / "dump_yolox_reid.npz"
DUMP_02 = REPO / "outputs" / "dmm_phase0_mot20_02" / "MOT20-02" / "dump_yolox_reid.npz"
OUT_ROOT = REPO / "outputs" / "dmm_phase9_nsa"
REGISTRY = REPO / "outputs" / "experiment_registry.csv"

BASE = ["--track-buffer", "70", "--match-thresh", "0.5", "--new-track-thresh", "0.5"]

CONFIGS: List[Dict] = [
    {"name": "N0_ref", "flags": []},
    {"name": "N1_nsa05", "flags": ["--nsa-k", "0.5"]},
    {"name": "N2_nsa10", "flags": ["--nsa-k", "1.0"]},
    {"name": "N3_nsa20", "flags": ["--nsa-k", "2.0"]},
    {"name": "N4_nsa30", "flags": ["--nsa-k", "3.0"]},
    {"name": "N5_nsa05_m04", "flags": ["--nsa-k", "0.5", "--match-thresh", "0.4"]},
    {"name": "N6_nsa10_m04", "flags": ["--nsa-k", "1.0", "--match-thresh", "0.4"]},
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


def run_config(cfg, seq, dump):
    name = cfg["name"]
    run_dir = OUT_ROOT / f"{name}_{seq}"
    out_txt = run_dir / "track_results" / f"{seq}.txt"
    log_path = run_dir / "tracker.log"
    eval_summary = run_dir / "trackeval_work" / "eval" / name / "pedestrian_summary.txt"
    if eval_summary.exists():
        m = parse_summary(eval_summary)
        if m.get("HOTA") is not None:
            print(f"  [skip] {name} on {seq}", flush=True)
            return _result(name, seq, "completed", m)
    run_dir.mkdir(parents=True, exist_ok=True)
    rc = run([sys.executable, str(TRACKER),
        "--dump-npz", str(dump), "--seq", seq,
        "--out", str(out_txt), "--summary-json", str(run_dir / f"{seq}_summary.json"),
        "--assoc-mode", "botsort_reid"] + BASE + cfg["flags"], log_path)
    if rc != 0: return _result(name, seq, "tracker_failed", {})
    rc = run([sys.executable, str(EVAL),
        "--benchmark-name", "MOT20", "--split-to-eval", "train",
        "--gt-root", str(GT_ROOT), "--results-dir", str(run_dir / "track_results"),
        "--tracker-name", name, "--work-dir", str(run_dir / "trackeval_work"), "--seqs", seq],
        run_dir / "eval.log")
    if rc != 0: return _result(name, seq, "eval_failed", {})
    return _result(name, seq, "completed", parse_summary(eval_summary))


def _result(name, seq, status, m):
    return {"name": name, "seq": seq, "status": status,
            "HOTA": m.get("HOTA"), "DetA": m.get("DetA"), "AssA": m.get("AssA"),
            "IDF1": m.get("IDF1"), "MOTA": m.get("MOTA"), "IDSW": m.get("IDSW"),
            "Frag": m.get("Frag"), "CLR_FN": m.get("CLR_FN"), "CLR_FP": m.get("CLR_FP")}


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["name", "seq", "status", "HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW", "Frag", "CLR_FN", "CLR_FP"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows: w.writerow(r)


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for cfg in CONFIGS:
        for seq, dump in [("MOT20-01", DUMP_01), ("MOT20-02", DUMP_02)]:
            print(f"=== {cfg['name']} on {seq} ===", flush=True)
            r = run_config(cfg, seq, dump)
            all_rows.append(r)
            write_csv(all_rows, OUT_ROOT / "summary.csv")
            print(f"  HOTA={r.get('HOTA')} IDSW={r.get('IDSW')} Frag={r.get('Frag')}", flush=True)
    print("\n=== DONE ===", flush=True)
    for r in all_rows:
        print(f"  {r['name']:20s} {r['seq']:10s} HOTA={r.get('HOTA'):>7} IDSW={r.get('IDSW'):>5} Frag={r.get('Frag'):>5}", flush=True)


if __name__ == "__main__":
    main()
