#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Re-evaluate a series of checkpoints with a consistent config.

Usage:
  python scripts/reval_checkpoints.py \
    --config-path configs/bytetrack_fa_mot_mot17_v5_seqid.yaml \
    --checkpoints-dir outputs/bytetrack_fa_mot_mot17_v5_seqid \
    --out-root outputs/bytetrack_fa_mot_mot17_v5_seqid/reval_consistent \
    --data-root /gemini/code/datasets \
    --dataset MOT17 --split train \
    --val-sequences MOT17-04,MOT17-05
"""

from __future__ import annotations
import argparse
import csv
import glob
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, Any, List


def _parse_pedestrian_summary(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        names = f.readline().strip().split()
        vals = f.readline().strip().split()
    out: Dict[str, float] = {}
    for k, v in zip(names, vals):
        try:
            out[k] = float(v)
        except ValueError:
            continue
    return out


def _find_summary_path(out_dir: Path, dataset: str, split: str) -> Path:
    # Prefer tracker_min* if present
    candidates = list(out_dir.glob(f"tracker_min*/{dataset}-{split}/pedestrian_summary.txt"))
    if candidates:
        return sorted(candidates)[0]
    return out_dir / "tracker" / f"{dataset}-{split}" / "pedestrian_summary.txt"


def _write_csv(records: List[Dict[str, Any]], path: Path) -> None:
    if not records:
        return
    fieldnames = sorted({k for r in records for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r)


def _extract_epoch(path: str) -> int:
    m = re.search(r"checkpoint_epoch_(\d+)\.pth", path)
    if m:
        return int(m.group(1))
    m = re.search(r"checkpoint_(\d+)\.pth", path)
    if m:
        return int(m.group(1))
    return -1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-path", required=True)
    ap.add_argument("--checkpoints-dir", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--dataset", default="MOT17")
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--val-sequences", default=None)
    ap.add_argument("--start-epoch", type=int, default=None)
    ap.add_argument("--end-epoch", type=int, default=None)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep-going", action="store_true")
    args = ap.parse_args()

    ckpt_glob = os.path.join(args.checkpoints_dir, "checkpoint_epoch_*.pth")
    ckpts = sorted(glob.glob(ckpt_glob), key=_extract_epoch)
    if not ckpts:
        raise SystemExit(f"No checkpoints found with {ckpt_glob}")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    records_by_epoch: Dict[int, Dict[str, Any]] = {}
    existing_csv = out_root / "reval_results.csv"
    if existing_csv.is_file():
        try:
            with existing_csv.open("r", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    try:
                        ep = int(r.get("epoch", -1))
                    except ValueError:
                        continue
                    if ep >= 0:
                        records_by_epoch[ep] = r
        except Exception:
            pass

    for idx, ckpt in enumerate(ckpts, 1):
        epoch = _extract_epoch(ckpt)
        if args.start_epoch is not None and epoch < args.start_epoch:
            continue
        if args.end_epoch is not None and epoch > args.end_epoch:
            continue
        if args.skip_existing and epoch in records_by_epoch:
            continue
        run_name = f"epoch_{epoch}"
        out_dir = out_root / run_name
        cmd = [
            "/root/miniconda3/bin/python", "-u", "submit_bytetrack.py",
            "--config-path", args.config_path,
            "--inference-model", ckpt,
            "--inference-dataset", args.dataset,
            "--inference-split", args.split,
            "--data-root", args.data_root,
            "--output-dir", str(out_dir),
            "--eval-only-val",
        ]
        if args.val_sequences:
            cmd += ["--val-sequences", args.val_sequences]

        print(f"\n[{idx}/{len(ckpts)}] {run_name}")
        print("CMD:", " ".join(cmd))

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

        metrics = {}
        metrics_path = _find_summary_path(out_dir, args.dataset, args.split)
        if ok and (not args.dry_run):
            try:
                metrics = _parse_pedestrian_summary(metrics_path)
            except Exception as e:
                ok = False
                err = f"MetricsParseError: {e}"

        rec: Dict[str, Any] = {
            "epoch": epoch,
            "checkpoint": ckpt,
            "ok": ok,
            "metrics_path": str(metrics_path),
            "error": err,
        }
        for k in ["HOTA", "MOTA", "IDF1", "AssA", "DetA", "DetRe", "DetPr", "AssRe", "AssPr", "IDSW", "Frag"]:
            if k in metrics:
                rec[k] = metrics[k]
        records_by_epoch[epoch] = rec
        records_sorted = [records_by_epoch[k] for k in sorted(records_by_epoch.keys())]
        _write_csv(records_sorted, out_root / "reval_results.csv")

        if (not ok) and (not args.keep_going) and (not args.dry_run):
            print(f"Run failed, stopping. Error: {err}")
            break

    records_sorted = [records_by_epoch[k] for k in sorted(records_by_epoch.keys())]
    _write_csv(records_sorted, out_root / "reval_results.csv")


if __name__ == "__main__":
    main()
