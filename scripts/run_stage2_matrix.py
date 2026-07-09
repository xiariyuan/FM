#!/usr/bin/env python3
"""Phase 8: Stage 2 matching with ReID + Lost track recovery.

Base: buf=70, match=0.5, new_track=0.5 (Phase 6/7 best).
"""
from __future__ import annotations
import csv, json, subprocess, sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

REPO = Path("/gemini/code/FMtrack-main/FM-Track")
TRACKER = REPO / "scripts" / "dmm_base_tracker.py"
EVAL = REPO / "scripts" / "eval_motstyle_trackeval.py"
GT_ROOT = Path("/gemini/code/datasets/MOT20/train")
DUMP_01 = REPO / "outputs" / "dmm_phase0_mot20_01" / "MOT20-01" / "dump_yolox_reid.npz"
DUMP_02 = REPO / "outputs" / "dmm_phase0_mot20_02" / "MOT20-02" / "dump_yolox_reid.npz"
OUT_ROOT = REPO / "outputs" / "dmm_phase8_stage2"
REGISTRY = REPO / "outputs" / "experiment_registry.csv"

BASE_FLAGS = ["--track-buffer", "70", "--match-thresh", "0.5", "--new-track-thresh", "0.5"]

CONFIGS: List[Dict] = [
    {"name": "T0_ref", "flags": []},
    {"name": "T1_s2reid", "flags": ["--stage2-reid-enable"]},
    {"name": "T2_s2lost", "flags": ["--stage2-lost-enable"]},
    {"name": "T3_s2both", "flags": ["--stage2-reid-enable", "--stage2-lost-enable"]},
    {"name": "T4_s2both_2nd07", "flags": ["--stage2-reid-enable", "--stage2-lost-enable", "--second-match-thresh", "0.7"]},
    {"name": "T5_s2both_2nd03", "flags": ["--stage2-reid-enable", "--stage2-lost-enable", "--second-match-thresh", "0.3"]},
]


def run(cmd, log_path):
    print(f"[run] ...", flush=True)
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
    summary_json = run_dir / f"{seq}_summary.json"
    log_path = run_dir / "tracker.log"
    eval_summary = run_dir / "trackeval_work" / "eval" / name / "pedestrian_summary.txt"
    if eval_summary.exists():
        m = parse_summary(eval_summary)
        if m.get("HOTA") is not None:
            print(f"  [skip] {name} on {seq}", flush=True)
            return _result(name, seq, "completed", cfg, m)
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(TRACKER),
        "--dump-npz", str(dump), "--seq", seq,
        "--out", str(out_txt), "--summary-json", str(summary_json),
        "--assoc-mode", "botsort_reid"] + BASE_FLAGS + cfg["flags"]
    rc = run(cmd, log_path)
    if rc != 0: return _result(name, seq, "tracker_failed", cfg, {})
    eval_cmd = [sys.executable, str(EVAL),
        "--benchmark-name", "MOT20", "--split-to-eval", "train",
        "--gt-root", str(GT_ROOT), "--results-dir", str(run_dir / "track_results"),
        "--tracker-name", name, "--work-dir", str(run_dir / "trackeval_work"), "--seqs", seq]
    rc = run(eval_cmd, run_dir / "eval.log")
    if rc != 0: return _result(name, seq, "eval_failed", cfg, {})
    return _result(name, seq, "completed", cfg, parse_summary(eval_summary))


def _result(name, seq, status, cfg, m):
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
    print(f"[summary] wrote {path}", flush=True)


def append_registry(rows):
    ts = datetime.now().astimezone().isoformat()
    with REGISTRY.open("a", newline="") as f:
        w = csv.writer(f)
        for r in rows:
            if r["name"] == "T0_ref": continue
            notes = f"Stage2 {r['name']} HOTA={r.get('HOTA','')} IDSW={r.get('IDSW','')}"
            w.writerow([ts, "tracker", r["status"], str(TRACKER), "MOT20", f"train/{r['seq']}",
                        "DMMPhase8", r["name"], f"{r['name']}_{r['seq']}",
                        str(OUT_ROOT / f"{r['name']}_{r['seq']}"), "", "", "", notes, "",
                        r.get("HOTA", ""), r.get("DetA", ""), r.get("AssA", ""),
                        r.get("IDF1", ""), r.get("MOTA", ""), r.get("IDSW", ""), r.get("Frag", ""),
                        r["seq"]])
    print("[registry] appended", flush=True)


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
        print(f"  {r['name']:25s} {r['seq']:10s} HOTA={r.get('HOTA'):>7} IDSW={r.get('IDSW'):>5} Frag={r.get('Frag'):>5}", flush=True)


if __name__ == "__main__":
    main()
