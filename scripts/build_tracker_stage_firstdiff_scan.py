#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence


REPO_ROOT = Path("/gemini/code/FMtrack-main/FM-Track")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


REGISTRY_CSV = REPO_ROOT / "outputs" / "experiment_registry.csv"
META_FIELDS = [
    "run_root",
    "seq_name",
    "base_stage",
    "alt_stage",
    "status",
    "error",
]
SUMMARY_FIELDS = [
    "run_name",
    "seq_name",
    "base_stage",
    "alt_stage",
    "first_diff_frame",
    "delta_HOTA",
    "delta_AssA",
    "delta_IDF1",
    "delta_MOTA",
    "delta_IDs",
    "delta_Frag",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two tracker stages within one run and report the first output-diff frame on one sequence."
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--seq-name", required=True)
    parser.add_argument("--base-stage", required=True, help="tracker stage name, e.g. raw, raw_post, fgas, fgas_post")
    parser.add_argument("--alt-stage", required=True, help="tracker stage name, e.g. raw, raw_post, fgas, fgas_post")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--registry-csv", default=str(REGISTRY_CSV))
    return parser.parse_args()


def write_single_row_csv(path: Path, fieldnames: Iterable[str], row: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_rows_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def append_registry(args: argparse.Namespace, summary_csv: Path, status: str, notes: str) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "append_experiment_record.py"),
        "--csv",
        str(args.registry_csv),
        "--kind",
        "analysis",
        "--status",
        status,
        "--script",
        "scripts/build_tracker_stage_firstdiff_scan.py",
        "--dataset",
        "MOT17",
        "--split",
        "tracker_stage_firstdiff_scan",
        "--tracker-family",
        "deep_ocsort_fgas",
        "--variant",
        Path(args.out_dir).name,
        "--tag",
        Path(args.out_dir).name,
        "--run-root",
        str(Path(args.out_dir)),
        "--summary-csv",
        str(summary_csv),
        "--notes",
        notes,
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def _resolve_tracker_seq_file(run_root: Path, seq_name: str, tracker_stage: str) -> Path:
    pattern = f"results/trackers/*/*_{tracker_stage}/data/{seq_name}.txt"
    matches = sorted(run_root.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly 1 tracker file for {run_root} seq={seq_name} stage={tracker_stage}, found {len(matches)}"
        )
    return matches[0]


def _load_tracker_frames(path: Path) -> Dict[int, tuple[tuple[object, ...], ...]]:
    frame_rows: dict[int, list[tuple[object, ...]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            frame_id = int(float(parts[0]))
            track_id = int(float(parts[1]))
            bbox = tuple(round(float(value), 3) for value in parts[2:6])
            score = round(float(parts[6]), 6) if len(parts) > 6 and parts[6] else 1.0
            frame_rows[frame_id].append((track_id, bbox, score))
    return {frame_id: tuple(sorted(rows)) for frame_id, rows in frame_rows.items()}


def _find_first_diff_frame(
    base_frames: Mapping[int, tuple[tuple[object, ...], ...]],
    alt_frames: Mapping[int, tuple[tuple[object, ...], ...]],
) -> int:
    for frame_id in sorted(set(base_frames) | set(alt_frames)):
        if base_frames.get(frame_id, ()) != alt_frames.get(frame_id, ()):
            return frame_id
    return -1


def _read_seq_metrics(run_root: Path, seq_name: str, stage_name: str) -> Dict[str, float]:
    metrics_csv = run_root / "per_sequence_metrics.csv"
    with metrics_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("name", "")) != str(stage_name):
                continue
            if str(row.get("seq", "")) != str(seq_name):
                continue
            return {
                "HOTA": float(row.get("HOTA", 0.0)),
                "AssA": float(row.get("AssA", 0.0)),
                "IDF1": float(row.get("IDF1", 0.0)),
                "MOTA": float(row.get("MOTA", 0.0)),
                "IDs": float(row.get("IDs", 0.0)),
                "Frag": float(row.get("Frag", 0.0)),
            }
    raise ValueError(f"Missing {stage_name} seq metrics for {run_root} seq={seq_name}")


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_csv = out_dir / "meta_summary.csv"
    summary_csv = out_dir / "summary.csv"

    meta_row: Dict[str, object] = {
        "run_root": str(Path(args.run_root)),
        "seq_name": str(args.seq_name),
        "base_stage": str(args.base_stage),
        "alt_stage": str(args.alt_stage),
        "status": "running",
        "error": "",
    }
    write_single_row_csv(meta_csv, META_FIELDS, meta_row)
    append_registry(args, summary_csv, "running", "scanning first output-diff frame between tracker stages within one run")

    try:
        run_root = Path(args.run_root)
        base_seq_file = _resolve_tracker_seq_file(run_root, str(args.seq_name), str(args.base_stage))
        alt_seq_file = _resolve_tracker_seq_file(run_root, str(args.seq_name), str(args.alt_stage))
        base_frames = _load_tracker_frames(base_seq_file)
        alt_frames = _load_tracker_frames(alt_seq_file)
        base_metrics = _read_seq_metrics(run_root, str(args.seq_name), str(args.base_stage))
        alt_metrics = _read_seq_metrics(run_root, str(args.seq_name), str(args.alt_stage))

        summary_row = {
            "run_name": run_root.name,
            "seq_name": str(args.seq_name),
            "base_stage": str(args.base_stage),
            "alt_stage": str(args.alt_stage),
            "first_diff_frame": int(_find_first_diff_frame(base_frames, alt_frames)),
            "delta_HOTA": float(alt_metrics["HOTA"] - base_metrics["HOTA"]),
            "delta_AssA": float(alt_metrics["AssA"] - base_metrics["AssA"]),
            "delta_IDF1": float(alt_metrics["IDF1"] - base_metrics["IDF1"]),
            "delta_MOTA": float(alt_metrics["MOTA"] - base_metrics["MOTA"]),
            "delta_IDs": float(alt_metrics["IDs"] - base_metrics["IDs"]),
            "delta_Frag": float(alt_metrics["Frag"] - base_metrics["Frag"]),
        }

        write_rows_csv(summary_csv, SUMMARY_FIELDS, [summary_row])
        meta_row["status"] = "success"
        write_single_row_csv(meta_csv, META_FIELDS, meta_row)
        append_registry(args, summary_csv, "success", "tracker stage first-diff scan built")
        return 0
    except Exception as exc:
        meta_row["status"] = "failed"
        meta_row["error"] = str(exc)
        write_single_row_csv(meta_csv, META_FIELDS, meta_row)
        append_registry(args, summary_csv, "failed", f"tracker stage first-diff scan failed: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
