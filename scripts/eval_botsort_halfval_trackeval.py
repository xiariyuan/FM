#!/usr/bin/env python3
from __future__ import annotations

import argparse
import configparser
import csv
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate BoT-SORT validation-half outputs with TrackEval.")
    parser.add_argument("--dataset", required=True, choices=["MOT17", "MOT20"])
    parser.add_argument("--data-root", default="/gemini/code/datasets")
    parser.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT), help="Repository root; defaults to this script's parent repo")
    parser.add_argument("--results-dir", required=True, help="Directory containing BoT-SORT *.txt tracking outputs")
    parser.add_argument("--tracker-name", required=True, help="Tracker name used inside TrackEval temp folder")
    parser.add_argument("--work-dir", required=True, help="Output working directory for GT/tracker/eval artifacts")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--keep-workdir", action="store_true")
    parser.add_argument(
        "--remap-results-from-fullval",
        action="store_true",
        help="Remap tracking results from original full-sequence frame ids to half-val frame ids.",
    )
    return parser.parse_args()


def start_frame_for_half_val(seq_length: int) -> int:
    return seq_length // 2 + 2


def prepare_seqinfo(src_ini: Path, dst_ini: Path, seq_name: str, new_length: int) -> None:
    parser = configparser.ConfigParser()
    parser.read(src_ini)
    if "Sequence" not in parser:
        raise ValueError(f"Invalid seqinfo.ini: {src_ini}")
    parser["Sequence"]["name"] = seq_name
    parser["Sequence"]["seqLength"] = str(new_length)
    dst_ini.parent.mkdir(parents=True, exist_ok=True)
    with dst_ini.open("w") as f:
        parser.write(f)


def prepare_gt_for_seq(data_root: Path, dataset: str, seq: str, dst_root: Path) -> None:
    src_seq_dir = data_root / dataset / "train" / seq
    src_ini = src_seq_dir / "seqinfo.ini"
    src_gt = src_seq_dir / "gt" / "gt.txt"
    if not src_ini.exists():
        raise FileNotFoundError(f"Missing seqinfo.ini: {src_ini}")
    if not src_gt.exists():
        raise FileNotFoundError(f"Missing gt: {src_gt}")

    parser = configparser.ConfigParser()
    parser.read(src_ini)
    seq_length = int(parser["Sequence"]["seqLength"])
    start_frame = start_frame_for_half_val(seq_length)
    new_length = seq_length - start_frame + 1
    if new_length <= 0:
        raise ValueError(f"Invalid half-val length for {seq}: {seq_length}")

    dst_seq_dir = dst_root / seq
    prepare_seqinfo(src_ini, dst_seq_dir / "seqinfo.ini", seq, new_length)
    (dst_seq_dir / "gt").mkdir(parents=True, exist_ok=True)

    with src_gt.open("r") as f:
        rows = [row.strip().split(",") for row in f if row.strip()]

    kept_rows: list[list[str]] = []
    for row in rows:
        frame_id = int(float(row[0]))
        if frame_id < start_frame:
            continue
        row[0] = str(frame_id - start_frame + 1)
        kept_rows.append(row)

    with (dst_seq_dir / "gt" / "gt.txt").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(kept_rows)


def prepare_tracker_results_for_seq(
    data_root: Path,
    dataset: str,
    src_result: Path,
    dst_result: Path,
    remap_from_fullval: bool,
) -> None:
    if not remap_from_fullval:
        shutil.copy2(src_result, dst_result)
        return

    seq = src_result.stem
    src_seq_dir = data_root / dataset / "train" / seq
    src_ini = src_seq_dir / "seqinfo.ini"
    parser = configparser.ConfigParser()
    parser.read(src_ini)
    seq_length = int(parser["Sequence"]["seqLength"])
    start_frame = start_frame_for_half_val(seq_length)
    new_length = seq_length - start_frame + 1

    with src_result.open("r") as f:
        rows = [row.strip().split(",") for row in f if row.strip()]

    if rows:
        frame_ids = [int(float(row[0])) for row in rows]
        # Some reused baselines are already indexed in val-half coordinates.
        # In that case, a second remap would incorrectly drop every row.
        if min(frame_ids) >= 1 and max(frame_ids) <= new_length:
            with dst_result.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(rows)
            return

    kept_rows: list[list[str]] = []
    for row in rows:
        frame_id = int(float(row[0]))
        if frame_id < start_frame:
            continue
        row[0] = str(frame_id - start_frame + 1)
        kept_rows.append(row)

    with dst_result.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(kept_rows)


def build_seqmap(seqs: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name"])
        for seq in seqs:
            writer.writerow([seq])


def run_trackeval(args: argparse.Namespace, seqs: list[str], tracker_root: Path, gt_root: Path, work_dir: Path) -> Path:
    seqmap_path = work_dir / "seqmaps" / f"{args.dataset}_val_half.txt"
    build_seqmap(seqs, seqmap_path)
    output_root = work_dir / "eval"
    cmd = [
        args.python_bin,
        str(Path(args.repo_root) / "TrackEval" / "scripts" / "run_mot_challenge.py"),
        "--GT_FOLDER", str(gt_root),
        "--TRACKERS_FOLDER", str(tracker_root),
        "--OUTPUT_FOLDER", str(output_root),
        "--TRACKERS_TO_EVAL", args.tracker_name,
        "--BENCHMARK", args.dataset,
        "--SPLIT_TO_EVAL", "train",
        "--SEQMAP_FILE", str(seqmap_path),
        "--SKIP_SPLIT_FOL", "True",
        "--DO_PREPROC", "True",
        "--TRACKER_SUB_FOLDER", "data",
        "--OUTPUT_SUB_FOLDER", "",
        "--PRINT_ONLY_COMBINED", "True",
        "--METRICS", "HOTA", "CLEAR", "Identity",
    ]
    subprocess.run(cmd, check=True, cwd=Path(args.repo_root))
    return output_root / args.tracker_name


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    results_dir = Path(args.results_dir)
    work_dir = Path(args.work_dir)
    gt_root = work_dir / "gt"
    tracker_root = work_dir / "trackers"
    tracker_data = tracker_root / args.tracker_name / "data"

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    if work_dir.exists() and not args.keep_workdir:
        shutil.rmtree(work_dir)
    tracker_data.mkdir(parents=True, exist_ok=True)

    result_files = sorted(results_dir.glob("*.txt"))
    if not result_files:
        raise FileNotFoundError(f"No tracking txt files found in: {results_dir}")

    seqs = [path.stem for path in result_files]
    for seq in seqs:
        prepare_gt_for_seq(data_root, args.dataset, seq, gt_root)
    for path in result_files:
        prepare_tracker_results_for_seq(
            data_root=data_root,
            dataset=args.dataset,
            src_result=path,
            dst_result=tracker_data / path.name,
            remap_from_fullval=args.remap_results_from_fullval,
        )

    eval_dir = run_trackeval(args, seqs, tracker_root, gt_root, work_dir)
    print(f"[done] TrackEval results: {eval_dir}")


if __name__ == "__main__":
    main()
