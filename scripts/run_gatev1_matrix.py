#!/usr/bin/env python3
"""Run the DMM GateV1 experiment matrix (G2-G5) on MOT20-01 + MOT20-02.

G0 (base, no DMM) and G1 (v2best) are reused from existing runs.
GateV1 = density gate (observe-only in dense frames) + advantage gate
(same-metric v2 cost comparison before overriding the base primary match).

Outputs:
  outputs/dmm_phase3_gatev1/<variant>_<seq>/  tracker + eval outputs
  outputs/dmm_phase3_gatev1/summary.csv       queue-level summary
"""
from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

REPO = Path("/gemini/code/FMtrack-main/FM-Track")
TRACKER = REPO / "scripts" / "dmm_base_tracker.py"
EVAL = REPO / "scripts" / "eval_motstyle_trackeval.py"
GT_ROOT = Path("/gemini/code/datasets/MOT20/train")
DUMP_01 = REPO / "outputs" / "dmm_phase0_mot20_01" / "MOT20-01" / "dump_yolox_reid.npz"
DUMP_02 = REPO / "outputs" / "dmm_phase0_mot20_02" / "MOT20-02" / "dump_yolox_reid.npz"
OUT_ROOT = REPO / "outputs" / "dmm_phase3_gatev1"
REGISTRY = REPO / "outputs" / "experiment_registry.csv"

# v2best frozen params (shared by all GateV1 variants)
V2BEST_FLAGS = [
    "--dmm-enable",
    "--dmm-margin-thresh", "0.05",
    "--dmm-reid-dist-thresh", "0.20",
    "--dmm-v2-recovery-thresh", "0.40",
    "--dmm-v2-crowd-resolve-iou", "-1.0",
    "--dmm-v2-recovery-min-iou", "0.0",
    "--dmm-v2-dir-gate", "0.5",
]

# G0 = base (no DMM) [reuse existing]
# G1 = v2best (no gate) [reuse existing]
# G2-G5 = GateV1 variants
GATE_VARIANTS: List[Dict] = [
    {"name": "G2_gatev1_d50_a000", "density": 50, "advantage": 0.00},
    {"name": "G3_gatev1_d50_a030", "density": 50, "advantage": 0.03},
    {"name": "G4_gatev1_d45_a030", "density": 45, "advantage": 0.03},
    {"name": "G5_gatev1_d60_a030", "density": 60, "advantage": 0.03},
]

# Existing G0/G1 results to include in the summary
EXISTING = [
    {"name": "G0_base", "seq": "MOT20-01",
     "run_dir": REPO / "outputs" / "dmm_phase1_base_mot20_01_reid",
     "tracker_name": "dmm_phase1_base_mot20_01_reid"},
    {"name": "G0_base", "seq": "MOT20-02",
     "run_dir": REPO / "outputs" / "dmm_phase1_base_mot20_02_reid",
     "tracker_name": "dmm_phase1_base_mot20_02_reid"},
    {"name": "G1_v2best", "seq": "MOT20-01",
     "run_dir": REPO / "outputs" / "dmm_phase2_v2best_mot20_01",
     "tracker_name": "dmm_phase2_v2best"},
    {"name": "G1_v2best", "seq": "MOT20-02",
     "run_dir": REPO / "outputs" / "dmm_phase2_v2best_mot20_02",
     "tracker_name": "dmm_phase2_v2best_mot20_02"},
]


def run(cmd: List[str], log_path: Path | None = None) -> int:
    print(f"[run] {' '.join(cmd)}", flush=True)
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as f:
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
        return proc.returncode
    proc = subprocess.run(cmd)
    return proc.returncode


def parse_summary(summary_path: Path) -> Dict[str, float]:
    """Parse a TrackEval pedestrian_summary.txt into a metrics dict."""
    if not summary_path.exists():
        return {}
    lines = summary_path.read_text().strip().splitlines()
    if len(lines) < 2:
        return {}
    header = lines[0].split()
    vals = lines[1].split()
    out = {}
    for k, v in zip(header, vals):
        try:
            out[k] = float(v)
        except ValueError:
            out[k] = v
    return out


def parse_dmm_stats(summary_json: Path) -> Dict:
    if not summary_json.exists():
        return {}
    with summary_json.open() as f:
        d = json.load(f)
    return {"dmm_stats": d.get("dmm_stats", {}), "config": d.get("config", {})}


def run_variant(variant: Dict, seq: str, dump_npz: Path) -> Dict:
    name = variant["name"]
    run_dir = OUT_ROOT / f"{name}_{seq}"
    out_txt = run_dir / "track_results" / f"{seq}.txt"
    summary_json = run_dir / f"{seq}_summary.json"
    dmm_csv = run_dir / "dmm_events.csv"
    log_path = run_dir / "tracker.log"
    run_dir.mkdir(parents=True, exist_ok=True)

    tracker_cmd = [
        sys.executable, str(TRACKER),
        "--dump-npz", str(dump_npz),
        "--seq", seq,
        "--out", str(out_txt),
        "--summary-json", str(summary_json),
        "--assoc-mode", "botsort_reid",
        "--dmm-csv", str(dmm_csv),
    ] + V2BEST_FLAGS + [
        "--dmm-gate-v1-enable",
        "--dmm-gate-v1-density-thresh", str(variant["density"]),
        "--dmm-gate-v1-advantage-margin", str(variant["advantage"]),
    ]

    # Resume: skip if eval already completed
    eval_summary = run_dir / "trackeval_work" / "eval" / name / "pedestrian_summary.txt"
    if eval_summary.exists():
        metrics = parse_summary(eval_summary)
        extras = parse_dmm_stats(summary_json)
        if metrics and metrics.get("HOTA") is not None:
            print(f"  [skip] {name} on {seq} already evaluated", flush=True)
            return {
                "name": name, "seq": seq, "status": "completed",
                "density_thresh": variant["density"], "advantage_margin": variant["advantage"],
                "run_dir": str(run_dir),
                "HOTA": metrics.get("HOTA"), "DetA": metrics.get("DetA"),
                "AssA": metrics.get("AssA"), "IDF1": metrics.get("IDF1"),
                "MOTA": metrics.get("MOTA"), "IDSW": metrics.get("IDSW"),
                "Frag": metrics.get("Frag"), "CLR_FN": metrics.get("CLR_FN"),
                "dmm_stats": extras.get("dmm_stats", {}),
            }

    run_dir.mkdir(parents=True, exist_ok=True)
    rc = run(tracker_cmd, log_path=log_path)
    if rc != 0:
        return {"name": name, "seq": seq, "status": "tracker_failed", "rc": rc}

    # Eval
    eval_work = run_dir / "trackeval_work"
    tracker_name = name
    eval_cmd = [
        sys.executable, str(EVAL),
        "--benchmark-name", "MOT20",
        "--split-to-eval", "train",
        "--gt-root", str(GT_ROOT),
        "--results-dir", str(run_dir / "track_results"),
        "--tracker-name", tracker_name,
        "--work-dir", str(eval_work),
        "--seqs", seq,
    ]
    eval_log = run_dir / "eval.log"
    rc = run(eval_cmd, log_path=eval_log)
    if rc != 0:
        return {"name": name, "seq": seq, "status": "eval_failed", "rc": rc}

    summary_path = eval_work / "eval" / tracker_name / "pedestrian_summary.txt"
    metrics = parse_summary(summary_path)
    extras = parse_dmm_stats(summary_json)
    return {
        "name": name,
        "seq": seq,
        "status": "completed",
        "density_thresh": variant["density"],
        "advantage_margin": variant["advantage"],
        "run_dir": str(run_dir),
        "HOTA": metrics.get("HOTA"),
        "DetA": metrics.get("DetA"),
        "AssA": metrics.get("AssA"),
        "IDF1": metrics.get("IDF1"),
        "MOTA": metrics.get("MOTA"),
        "IDSW": metrics.get("IDSW"),
        "Frag": metrics.get("Frag"),
        "CLR_FN": metrics.get("CLR_FN"),
        "dmm_stats": extras.get("dmm_stats", {}),
    }


def collect_existing() -> List[Dict]:
    rows = []
    for ex in EXISTING:
        summary_path = ex["run_dir"] / "trackeval_work" / "eval" / ex["tracker_name"] / "pedestrian_summary.txt"
        metrics = parse_summary(summary_path)
        rows.append({
            "name": ex["name"],
            "seq": ex["seq"],
            "status": "completed",
            "density_thresh": "",
            "advantage_margin": "",
            "run_dir": str(ex["run_dir"]),
            "HOTA": metrics.get("HOTA"),
            "DetA": metrics.get("DetA"),
            "AssA": metrics.get("AssA"),
            "IDF1": metrics.get("IDF1"),
            "MOTA": metrics.get("MOTA"),
            "IDSW": metrics.get("IDSW"),
            "Frag": metrics.get("Frag"),
            "CLR_FN": metrics.get("CLR_FN"),
            "dmm_stats": {},
        })
    return rows


def write_summary_csv(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["name", "seq", "status", "density_thresh", "advantage_margin",
              "HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW", "Frag", "CLR_FN",
              "run_dir", "dmm_stats"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            row = dict(r)
            row["dmm_stats"] = json.dumps(r.get("dmm_stats", {}))
            w.writerow(row)
    print(f"[summary] wrote {path}", flush=True)


def append_registry(rows: List[Dict]) -> None:
    """Append GateV1 tracker rows to the central registry."""
    ts = datetime.now().astimezone().isoformat()
    with REGISTRY.open("a", newline="") as f:
        w = csv.writer(f)
        for r in rows:
            if r["name"].startswith("G0") or r["name"].startswith("G1"):
                continue
            notes = f"GateV1 density={r.get('density_thresh','')} adv={r.get('advantage_margin','')} HOTA={r.get('HOTA','')} IDSW={r.get('IDSW','')}"
            w.writerow([
                ts, "tracker", r["status"], str(TRACKER), "MOT20", f"train/{r['seq']}",
                "DMMPhase3", r["name"], f"{r['name']}_{r['seq']}", r["run_dir"], "",
                "", "", notes, "", r.get("HOTA", ""), r.get("DetA", ""), r.get("AssA", ""),
                r.get("IDF1", ""), r.get("MOTA", ""), r.get("IDSW", ""), r.get("Frag", ""),
                r["seq"],
            ])
    print(f"[registry] appended {len(rows)} rows", flush=True)


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_rows: List[Dict] = []

    # G0/G1 from existing
    all_rows.extend(collect_existing())

    # G2-G5
    for variant in GATE_VARIANTS:
        for seq, dump in [("MOT20-01", DUMP_01), ("MOT20-02", DUMP_02)]:
            print(f"\n=== {variant['name']} on {seq} ===", flush=True)
            row = run_variant(variant, seq, dump)
            all_rows.append(row)
            write_summary_csv(all_rows, OUT_ROOT / "summary.csv")
            print(f"  HOTA={row.get('HOTA')} IDSW={row.get('IDSW')} status={row.get('status')}", flush=True)

    write_summary_csv(all_rows, OUT_ROOT / "summary.csv")
    append_registry(all_rows)
    print("\n=== DONE ===", flush=True)
    for r in all_rows:
        print(f"  {r['name']:30s} {r['seq']:10s} HOTA={r.get('HOTA'):>7} IDSW={r.get('IDSW'):>5} Frag={r.get('Frag'):>5}", flush=True)


if __name__ == "__main__":
    main()
