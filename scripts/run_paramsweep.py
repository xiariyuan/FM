#!/usr/bin/env python3
"""Parameter sweep: track_buffer × match_thresh on MOT20-01 + MOT20-02.

No DMM, just base tracker with different params. Goal: find best params
that reduce IDSW/Frag on MOT20-02 without hurting MOT20-01.
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
OUT_ROOT = REPO / "outputs" / "dmm_phase4_paramsweep"
REGISTRY = REPO / "outputs" / "experiment_registry.csv"

CONFIGS: List[Dict] = [
    {"name": "P0_base",          "buffer": 30, "match": 0.7},
    {"name": "P1_buf50_m07",     "buffer": 50, "match": 0.7},
    {"name": "P2_buf50_m05",     "buffer": 50, "match": 0.5},
    {"name": "P3_buf70_m05",     "buffer": 70, "match": 0.5},
    {"name": "P4_buf50_m06",     "buffer": 50, "match": 0.6},
    {"name": "P5_buf70_m06",     "buffer": 70, "match": 0.6},
]


def run(cmd: List[str], log_path: Path) -> int:
    print(f"[run] {' '.join(cmd[:6])}...", flush=True)
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
    return {k: (float(x) if _is_float(x) else x) for k, x in zip(h, v)}


def _is_float(s: str) -> bool:
    try:
        float(s); return True
    except ValueError:
        return False


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
            return {"name": name, "seq": seq, "status": "completed", **_metrics(m, cfg)}

    run_dir.mkdir(parents=True, exist_ok=True)
    tracker_cmd = [
        sys.executable, str(TRACKER),
        "--dump-npz", str(dump), "--seq", seq,
        "--out", str(out_txt), "--summary-json", str(summary_json),
        "--assoc-mode", "botsort_reid",
        "--track-buffer", str(cfg["buffer"]),
        "--match-thresh", str(cfg["match"]),
        "--second-match-thresh", str(min(cfg["match"], 0.5)),
    ]
    rc = run(tracker_cmd, log_path)
    if rc != 0:
        return {"name": name, "seq": seq, "status": "tracker_failed", "rc": rc, **_cfg_fields(cfg)}

    eval_cmd = [
        sys.executable, str(EVAL),
        "--benchmark-name", "MOT20", "--split-to-eval", "train",
        "--gt-root", str(GT_ROOT), "--results-dir", str(run_dir / "track_results"),
        "--tracker-name", name, "--work-dir", str(run_dir / "trackeval_work"),
        "--seqs", seq,
    ]
    rc = run(eval_cmd, run_dir / "eval.log")
    if rc != 0:
        return {"name": name, "seq": seq, "status": "eval_failed", "rc": rc, **_cfg_fields(cfg)}

    m = parse_summary(eval_summary)
    return {"name": name, "seq": seq, "status": "completed", **_metrics(m, cfg)}


def _cfg_fields(cfg): return {"track_buffer": cfg["buffer"], "match_thresh": cfg["match"]}
def _metrics(m, cfg): return {
    "track_buffer": cfg["buffer"], "match_thresh": cfg["match"],
    "HOTA": m.get("HOTA"), "DetA": m.get("DetA"), "AssA": m.get("AssA"),
    "IDF1": m.get("IDF1"), "MOTA": m.get("MOTA"), "IDSW": m.get("IDSW"),
    "Frag": m.get("Frag"), "CLR_FN": m.get("CLR_FN"),
}


def write_csv(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["name", "seq", "status", "track_buffer", "match_thresh",
              "HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW", "Frag", "CLR_FN"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[summary] wrote {path}", flush=True)


def append_registry(rows: List[Dict]) -> None:
    ts = datetime.now().astimezone().isoformat()
    with REGISTRY.open("a", newline="") as f:
        w = csv.writer(f)
        for r in rows:
            if r["name"] == "P0_base":
                continue
            notes = f"ParamSweep buf={r['track_buffer']} match={r['match_thresh']} HOTA={r.get('HOTA','')} IDSW={r.get('IDSW','')}"
            w.writerow([ts, "tracker", r["status"], str(TRACKER), "MOT20", f"train/{r['seq']}",
                        "DMMPhase4", r["name"], f"{r['name']}_{r['seq']}",
                        str(OUT_ROOT / f"{r['name']}_{r['seq']}"), "", "", "", notes, "",
                        r.get("HOTA", ""), r.get("DetA", ""), r.get("AssA", ""),
                        r.get("IDF1", ""), r.get("MOTA", ""), r.get("IDSW", ""), r.get("Frag", ""),
                        r["seq"]])
    print(f"[registry] appended {len(rows)} rows", flush=True)


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_rows: List[Dict] = []
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
        print(f"  {r['name']:20s} {r['seq']:10s} HOTA={r.get('HOTA'):>7} IDSW={r.get('IDSW'):>5} Frag={r.get('Frag'):>5}", flush=True)


if __name__ == "__main__":
    main()
