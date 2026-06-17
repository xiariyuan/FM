#!/usr/bin/env python3
"""Run pseudotrack ablation: measure wall time, NPZ size, groups/candidates for different configs."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = "/root/miniconda3/bin/python"
BOT_ROOT = REPO_ROOT / "external/BoT-SORT-main"
BUILD_SCRIPT = REPO_ROOT / "scripts/build_gt_pseudotrack_groups.py"
DATA_ROOT = "/gemini/code/datasets"

# Ablation configs: (label, candidate_topk, max_hard_neg, batch_size, max_history)
CONFIGS = [
    ("A2a_baseline", 16, 6, 4, 8),
    ("A2b_topk8",     8, 6, 4, 8),
    ("A2c_topk8_hn3", 8, 3, 4, 8),
    ("A2d_topk8_hn3_b16", 8, 3, 16, 8),
]

# Single shard for fair comparison
SHARD_SPEC = [
    ("MOT20-01", "train_half", 1, 120),
    ("MOT20-05", "val_half",   1, 120),
]


def run_one(label: str, topk: int, hn: int, batch: int, hist: int, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for seq, split_part, start, end in SHARD_SPEC:
        suffix = f"f{start:04d}_{end:04d}"
        npz_path = out_dir / f"{seq}_{suffix}_groups.npz"
        csv_path = out_dir / f"{seq}_{suffix}_pairs.csv"

        cmd = [
            PYTHON_BIN, str(BUILD_SCRIPT),
            "--dataset", "MOT20",
            "--data-root", DATA_ROOT,
            "--seqs", seq,
            "--split-part", split_part,
            "--fast-reid-config", str(BOT_ROOT / "fast_reid/configs/MOT20/sbs_S50.yml"),
            "--fast-reid-weights", str(BOT_ROOT / "pretrained/mot20_sbs_S50.pth"),
            "--device", "cuda",
            "--batch-size", str(batch),
            "--max-history", str(hist),
            "--min-history", "3",
            "--feature-dtype", "float16",
            "--seed", "123",
            "--smooth-alpha", "0.9",
            "--iou-pos", "0.7",
            "--iou-ignore", "0.5",
            "--max-gap", "30",
            "--candidate-topk", str(topk),
            "--max-hard-negatives", str(hn),
            "--max-random-negatives", "2",
            "--positive-keep-prob", "0.7",
            "--frame-start", str(start),
            "--frame-end", str(end),
            "--out-npz", str(npz_path),
            "--out-csv", str(csv_path),
        ]

        print(f"[{label}] {seq} {split_part} f{start:04d}_{end:04d} topk={topk} hn={hn} batch={batch} hist={hist}", flush=True)
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        wall = time.time() - t0

        status = "success" if proc.returncode == 0 and npz_path.exists() else "failed"
        npz_size_mb = npz_path.stat().st_size / 1024 / 1024 if npz_path.exists() else 0.0

        # Count groups/candidates from npz
        groups = 0
        candidates = 0
        if status == "success":
            try:
                import numpy as np
                d = np.load(str(npz_path), allow_pickle=True)
                groups = len(d.files) - 1  # minus 'n_groups' key if present
                # try to get from data
                if 'n_groups' in d.files:
                    groups = int(d['n_groups'])
                # count candidate rows from the first file
                for k in d.files:
                    if k != 'n_groups':
                        arr = d[k].item() if hasattr(d[k], 'item') else d[k]
                        if hasattr(arr, '__len__'):
                            candidates = len(arr) if not isinstance(arr, dict) else 0
                        break
                d.close()
            except Exception:
                pass

        results.append({
            "exp_id": label,
            "seq": seq,
            "frame_range": f"{start}-{end}",
            "candidate_topk": topk,
            "max_hard_negatives": hn,
            "batch_size": batch,
            "max_history": hist,
            "wall_seconds": round(wall, 2),
            "npz_size_mb": round(npz_size_mb, 2),
            "groups": groups,
            "candidates": candidates,
            "status": status,
        })

        if status != "success":
            print(f"  FAILED: {proc.stderr[-300:] if proc.stderr else 'no stderr'}", flush=True)

    return results


def main():
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_root = REPO_ROOT / f"outputs/mot20_pseudotrack_ablation_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)

    all_results = []
    for label, topk, hn, batch, hist in CONFIGS:
        shard_dir = out_root / label
        results = run_one(label, topk, hn, batch, hist, shard_dir)
        all_results.extend(results)

    # Write summary.csv
    import csv
    summary_path = out_root / "summary.csv"
    fieldnames = list(all_results[0].keys())
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)

    print(f"\n[done] Ablation results: {summary_path}")
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
