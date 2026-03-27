#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path("/gemini/code/FMtrack-main/FM-Track")
TRACKEVAL_SCRIPT = REPO_ROOT / "TrackEval" / "scripts" / "run_mot_challenge.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MOT-style tracking outputs with TrackEval.")
    parser.add_argument("--benchmark-name", required=True, help="Benchmark label passed to TrackEval, e.g. DanceTrack")
    parser.add_argument("--split-to-eval", default="val", help="Split label for TrackEval metadata, e.g. val")
    parser.add_argument("--gt-root", required=True, help="GT root containing seq/gt/gt.txt and seqinfo.ini")
    parser.add_argument("--results-dir", required=True, help="Directory containing tracker *.txt outputs")
    parser.add_argument("--tracker-name", required=True, help="Tracker name used in TrackEval temp folder")
    parser.add_argument("--work-dir", required=True, help="Work directory for copied tracker outputs and eval logs")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--keep-workdir", action="store_true")
    return parser.parse_args()


def build_seqmap(seqs: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name"])
        for seq in seqs:
            writer.writerow([seq])


def main() -> None:
    args = parse_args()
    gt_root = Path(args.gt_root)
    results_dir = Path(args.results_dir)
    work_dir = Path(args.work_dir)
    tracker_root = work_dir / "trackers"
    tracker_data = tracker_root / args.tracker_name / "data"

    if not gt_root.exists():
        raise FileNotFoundError(f"GT root not found: {gt_root}")
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    if work_dir.exists() and not args.keep_workdir:
        shutil.rmtree(work_dir)
    tracker_data.mkdir(parents=True, exist_ok=True)

    seq_dirs = sorted(path for path in gt_root.iterdir() if path.is_dir())
    if not seq_dirs:
        raise FileNotFoundError(f"No sequences found under: {gt_root}")

    seqs = [path.name for path in seq_dirs]
    for seq in seqs:
        gt_file = gt_root / seq / "gt" / "gt.txt"
        seqinfo = gt_root / seq / "seqinfo.ini"
        if not gt_file.exists():
            raise FileNotFoundError(f"Missing gt file: {gt_file}")
        if not seqinfo.exists():
            raise FileNotFoundError(f"Missing seqinfo.ini: {seqinfo}")

    result_files = sorted(results_dir.glob("*.txt"))
    if not result_files:
        raise FileNotFoundError(f"No tracker txt files found in: {results_dir}")

    missing = [seq for seq in seqs if not (results_dir / f"{seq}.txt").exists()]
    if missing:
        raise FileNotFoundError(f"Missing tracker outputs for sequences: {', '.join(missing[:10])}")

    for path in result_files:
        shutil.copy2(path, tracker_data / path.name)

    seqmap_path = work_dir / "seqmaps" / f"{args.benchmark_name}_{args.split_to_eval}.txt"
    build_seqmap(seqs, seqmap_path)

    output_root = work_dir / "eval"
    cmd = [
        args.python_bin,
        str(TRACKEVAL_SCRIPT),
        "--GT_FOLDER", str(gt_root),
        "--TRACKERS_FOLDER", str(tracker_root),
        "--OUTPUT_FOLDER", str(output_root),
        "--TRACKERS_TO_EVAL", args.tracker_name,
        "--BENCHMARK", args.benchmark_name,
        "--SPLIT_TO_EVAL", args.split_to_eval,
        "--SEQMAP_FILE", str(seqmap_path),
        "--SKIP_SPLIT_FOL", "True",
        "--DO_PREPROC", "True",
        "--TRACKER_SUB_FOLDER", "data",
        "--OUTPUT_SUB_FOLDER", "",
        "--PRINT_ONLY_COMBINED", "True",
        "--METRICS", "HOTA", "CLEAR", "Identity",
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    print(f"[done] TrackEval results: {output_root / args.tracker_name}")


if __name__ == "__main__":
    main()
