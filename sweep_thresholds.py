#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sweep FM-Track inference thresholds locally (no Codabench uploads needed).

Usage (run from repo root where submit_and_evaluate.py lives):
  python sweep_thresholds.py \
    --config-path configs/r50_dino_fa_mot_v2_mot17.yaml \
    --checkpoint ./outputs/fa_mot_v2_resume/checkpoint_28.pth \
    --data-root /gemini/code/datasets \
    --split train \
    --out-root ./outputs/sweep_epoch28

Optional:
  --dataset MOT17
  --assignment-protocol hungarian
  --dry-run
  --runs-json '[{"det":0.2,"newborn":0.2,"id":0.1,"area":0,"minlen":0}, ...]'
"""

from __future__ import annotations
import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, List

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    pd = None
    _HAS_PANDAS = False

import csv


def parse_pedestrian_summary(path: Path) -> Dict[str, float]:
    """
    TrackEval pedestrian_summary.txt format:
      line1: metric names separated by spaces
      line2: metric values separated by spaces
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    with path.open("r", encoding="utf-8") as f:
        names = f.readline().strip().split()
        vals = f.readline().strip().split()
    out = {}
    for n, v in zip(names, vals):
        try:
            out[n] = float(v)
        except ValueError:
            # keep non-float metrics out
            pass
    return out


def build_run_name(p: Dict[str, Any]) -> str:
    return f"det{p['det']}_new{p['newborn']}_id{p['id']}_area{p['area']}_min{p['minlen']}"


def default_runs() -> List[Dict[str, Any]]:
    # A small, high-value set (recall-friendly first), plus your current config as baseline.
    return [
        {"det": 0.20, "newborn": 0.20, "id": 0.10, "area": 0,  "minlen": 0},
        {"det": 0.20, "newborn": 0.20, "id": 0.05, "area": 0,  "minlen": 0},
        {"det": 0.30, "newborn": 0.30, "id": 0.10, "area": 0,  "minlen": 0},
        {"det": 0.30, "newborn": 0.20, "id": 0.10, "area": 0,  "minlen": 0},
        {"det": 0.10, "newborn": 0.10, "id": 0.05, "area": 0,  "minlen": 0},
        # your current settings (as a reference)
        {"det": 0.40, "newborn": 0.60, "id": 0.30, "area": 50, "minlen": 3},
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-path", required=True, help="Path to yaml config")
    ap.add_argument("--checkpoint", required=True, help="Path to model checkpoint .pth")
    ap.add_argument("--data-root", required=True, help="Dataset root (contains MOT17 etc.)")
    ap.add_argument("--dataset", default="MOT17", help="Inference dataset name (default: MOT17)")
    ap.add_argument("--split", default="train", choices=["train", "test"], help="Inference split (default: train)")
    ap.add_argument("--out-root", required=True, help="Root dir for sweep outputs")
    ap.add_argument("--assignment-protocol", default=None, help="Optional, e.g. hungarian or object-max")
    ap.add_argument("--miss-tolerance", default=None, type=int, help="Optional miss tolerance override")
    ap.add_argument("--runs-json", default=None, help="JSON list of runs with keys det,newborn,id,area,minlen")
    ap.add_argument("--dry-run", action="store_true", help="Print commands only, do not execute")
    ap.add_argument("--keep-going", action="store_true", help="Continue even if one run fails")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.runs_json:
        runs = json.loads(args.runs_json)
        assert isinstance(runs, list) and len(runs) > 0
    else:
        runs = default_runs()

    records = []

    def write_csv_records(rows: List[Dict[str, Any]], path: Path) -> None:
        if _HAS_PANDAS:
            df = pd.DataFrame(rows)
            df.to_csv(path, index=False)
            return
        if not rows:
            return
        fieldnames = sorted({k for row in rows for k in row.keys()})
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    for idx, p in enumerate(runs, 1):
        run_name = build_run_name(p)
        out_dir = out_root / run_name
        cmd = [
            "python", "submit_and_evaluate.py",
            "--config-path", args.config_path,
            "--inference-model", args.checkpoint,
            "--inference-mode", "evaluate",
            "--inference-dataset", args.dataset,
            "--inference-split", args.split,
            "--det-thresh", str(p["det"]),
            "--newborn-thresh", str(p["newborn"]),
            "--id-thresh", str(p["id"]),
            "--area-thresh", str(p["area"]),
            "--min-track-len", str(p["minlen"]),
            "--outputs-dir", str(out_dir),
            "--data-root", args.data_root,
        ]
        if args.assignment_protocol:
            cmd += ["--assignment-protocol", args.assignment_protocol]
        if args.miss_tolerance is not None:
            cmd += ["--miss-tolerance", str(args.miss_tolerance)]

        print(f"\n[{idx}/{len(runs)}] {run_name}")
        print("CMD:", " ".join(cmd))

        t0 = time.time()
        ok = True
        err = ""
        if not args.dry_run:
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                ok = False
                err = f"CalledProcessError: {e}"
            except Exception as e:
                ok = False
                err = f"{type(e).__name__}: {e}"
        dt = time.time() - t0

        metrics_path = out_dir / "tracker" / "tracker_default" / "pedestrian_summary.txt"
        metrics = {}
        if ok and (not args.dry_run):
            try:
                metrics = parse_pedestrian_summary(metrics_path)
            except Exception as e:
                ok = False
                err = f"MetricsParseError: {e}"

        rec = {
            "run": run_name,
            "ok": ok,
            "seconds": round(dt, 2),
            "det_thresh": p["det"],
            "newborn_thresh": p["newborn"],
            "id_thresh": p["id"],
            "area_thresh": p["area"],
            "min_track_len": p["minlen"],
            "metrics_path": str(metrics_path),
            "error": err,
        }
        # pull key metrics if available
        for k in ["HOTA", "MOTA", "IDF1", "DetRe", "DetPr", "AssA", "DetA", "AssRe", "AssPr", "LocA", "CLR_Re", "CLR_Pr"]:
            if k in metrics:
                rec[k] = metrics[k]
        records.append(rec)

        # save incremental results
        write_csv_records(records, out_root / "sweep_results.csv")

        if (not ok) and (not args.keep_going) and (not args.dry_run):
            print(f"Run failed, stopping. Error: {err}")
            break

    write_csv_records(records, out_root / "sweep_results.csv")

    # Pretty markdown summary
    def topk(metric: str, k: int = 5):
        if _HAS_PANDAS:
            df = pd.DataFrame(records)
            if metric not in df.columns:
                return pd.DataFrame()
            d2 = df[df["ok"] == True].copy()
            if len(d2) == 0:
                return pd.DataFrame()
            return d2.sort_values(metric, ascending=False).head(k)[
                ["run", metric, "DetRe", "DetPr", "AssA", "MOTA", "IDF1", "seconds",
                 "det_thresh", "newborn_thresh", "id_thresh", "area_thresh", "min_track_len"]
            ]

        valid = [r for r in records if r.get("ok")]
        valid = [r for r in valid if isinstance(r.get(metric), (int, float))]
        valid = sorted(valid, key=lambda r: r[metric], reverse=True)[:k]
        return valid

    def to_markdown_table(rows, metric: str):
        if _HAS_PANDAS:
            return rows.to_markdown(index=False) if len(rows) else "(no results)\n"
        if not rows:
            return "(no results)\n"
        cols = [
            "run", metric, "DetRe", "DetPr", "AssA", "MOTA", "IDF1", "seconds",
            "det_thresh", "newborn_thresh", "id_thresh", "area_thresh", "min_track_len"
        ]
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        lines = [header, sep]
        for r in rows:
            line = "| " + " | ".join(str(r.get(c, "")) for c in cols) + " |"
            lines.append(line)
        return "\n".join(lines) + "\n"

    md = []
    md.append("# Sweep summary\n")
    md.append(f"- Dataset: **{args.dataset}**  Split: **{args.split}**\n")
    md.append(f"- Out root: `{out_root}`\n")
    md.append("\n## Top by HOTA\n")
    t = topk("HOTA")
    md.append(to_markdown_table(t, "HOTA"))
    md.append("\n## Top by DetRe (recall)\n")
    t = topk("DetRe")
    md.append(to_markdown_table(t, "DetRe"))
    md.append("\n## Top by IDF1\n")
    t = topk("IDF1")
    md.append(to_markdown_table(t, "IDF1"))

    (out_root / "sweep_summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nSaved: {out_root / 'sweep_results.csv'}")
    print(f"Saved: {out_root / 'sweep_summary.md'}")


if __name__ == "__main__":
    main()
