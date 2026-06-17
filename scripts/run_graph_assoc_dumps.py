#!/usr/bin/env python3
"""Generate graph-association candidate row dumps for DanceTrack sequences.

Runs tracking with --graph-assoc-enable --graph-assoc-dump-candidate-rows
to produce per-sequence candidate JSONL files for learned commit training.
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path("/gemini/code/FMtrack-main/FM-Track")
RUN_SCRIPT = REPO_ROOT / "scripts" / "run_botsort_dancetrack_val.sh"
DATA_ROOT = Path("/gemini/code/datasets/DanceTrack/extracted")
REGISTRY_CSV = REPO_ROOT / "outputs" / "experiment_registry.csv"


def iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate graph-assoc candidate dumps.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--seq-ids", nargs="+", type=int, required=True)
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    parser.add_argument("--split", default="val")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    summary_csv = out_dir / "summary.csv"
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    SUMMARY_FIELDS = ["seq_id", "seq_name", "status", "started_at", "finished_at",
                      "result_file", "candidate_jsonl", "notes"]
    rows = []
    for seq_id in args.seq_ids:
        seq_name = f"dancetrack{seq_id:04d}"
        result_dir = REPO_ROOT / "external" / "BoT-SORT-main" / "YOLOX_outputs" / args.exp_name
        rows.append({
            "seq_id": str(seq_id), "seq_name": seq_name, "status": "pending",
            "started_at": "", "finished_at": "",
            "result_file": str(result_dir / "track_results" / f"{seq_name}.txt"),
            "candidate_jsonl": str(result_dir / "graph_assoc_analysis" / f"{seq_name}_candidates.jsonl"),
            "notes": "",
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(rows)

    for row in rows:
        row["status"] = "running"
        row["started_at"] = iso_now()
        with summary_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
            w.writeheader()
            w.writerows(rows)

        seq_id = row["seq_id"]
        seq_name = row["seq_name"]
        log_path = log_dir / f"{seq_name}.log"
        cmd = [
            "env",
            f"DATA_ROOT={Path(args.data_root).resolve()}",
            f"SPLIT={args.split}",
            "PYTHONUNBUFFERED=1",
            "bash", str(RUN_SCRIPT), "base",
            "--experiment-name", args.exp_name,
            "--seq-ids", str(seq_id),
            "--graph-assoc-enable",
            "--graph-assoc-dump-candidate-rows",
            "--graph-assoc-candidate-rerank-top-k", "3",
            "--graph-assoc-top-k", "2",
            "--graph-assoc-max-rows", "3",
            "--graph-assoc-max-cols", "3",
        ]
        with log_path.open("w") as handle:
            proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=handle, stderr=subprocess.STDOUT)

        row["finished_at"] = iso_now()
        if proc.returncode == 0:
            row["status"] = "success"
            row["notes"] = str(log_path)
        else:
            row["status"] = "failed"
            row["notes"] = f"{log_path} rc={proc.returncode}"
        with summary_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
            w.writeheader()
            w.writerows(rows)

    # Write source_manifest.csv
    manifest_path = out_dir / "source_manifest.csv"
    MANIFEST_FIELDS = ["rows_jsonl", "source_tag", "host_variant", "split_tag",
                       "dataset", "data_root", "split", "split_part", "seq_name",
                       "dataset_tag", "feature_version"]
    split_tag = args.split  # "train" or "val"
    # Path construction: data_root / dataset / split / seq
    # DanceTrack path: /gemini/code/datasets/DanceTrack/extracted/train/dancetrack0001
    # We set data_root so that data_root + "/" + dataset + "/" + split + "/" + seq = correct path
    # Simplest: data_root = parent of extracted, dataset = "DanceTrack/extracted"
    resolved = Path(args.data_root).resolve()
    data_root_str = str(resolved.parent)
    dataset_str = resolved.name  # "extracted" — but this loses "DanceTrack"
    # Better: keep full path structure by embedding in data_root
    # data_root = /gemini/code/datasets, dataset = DanceTrack/extracted
    # But dataset arg is constrained. Use manifest dataset field directly.
    dataset_str = "DanceTrack/extracted"
    with manifest_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow({
                "rows_jsonl": row["candidate_jsonl"],
                "source_tag": row["seq_name"],
                "host_variant": "bc_v2_gap15_graph_assoc",
                "split_tag": split_tag,
                "dataset": dataset_str,
                "data_root": data_root_str,
                "split": split_tag,
                "split_part": "full",
                "seq_name": row["seq_name"],
                "dataset_tag": "graph_assoc_commit",
                "feature_version": "graph_assoc_v1",
            })
    print(f"[done] manifest written to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
