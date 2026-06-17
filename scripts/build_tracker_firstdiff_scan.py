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
    "base_run_root",
    "seq_name",
    "tracker_stage",
    "source_glob",
    "source_runs",
    "matched_runs",
    "identical_runs",
    "status",
    "error",
]
SUMMARY_FIELDS = [
    "run_name",
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
        description="Scan tracker runs for the first output-diff frame against a base run on one sequence."
    )
    parser.add_argument("--base-run-root", required=True)
    parser.add_argument("--seq-name", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--glob",
        required=True,
        help="glob pattern for candidate alt run roots, resolved relative to repo root",
    )
    parser.add_argument(
        "--tracker-stage",
        default="fgas",
        help="tracker stage suffix to compare, e.g. fgas, fgas_post, raw, or raw_post",
    )
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
        "scripts/build_tracker_firstdiff_scan.py",
        "--dataset",
        "MOT17",
        "--split",
        "tracker_firstdiff_scan",
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


def _read_seq_metrics(run_root: Path, seq_name: str) -> Dict[str, float]:
    metrics_csv = run_root / "per_sequence_metrics.csv"
    with metrics_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("name", "")) != "fgas":
                continue
            if str(row.get("seq", "")) != seq_name:
                continue
            return {
                "HOTA": float(row.get("HOTA", 0.0)),
                "AssA": float(row.get("AssA", 0.0)),
                "IDF1": float(row.get("IDF1", 0.0)),
                "MOTA": float(row.get("MOTA", 0.0)),
                "IDs": float(row.get("IDs", 0.0)),
                "Frag": float(row.get("Frag", 0.0)),
            }
    raise ValueError(f"Missing fgas seq metrics for {run_root} seq={seq_name}")


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_csv = out_dir / "meta_summary.csv"
    summary_csv = out_dir / "summary.csv"

    meta_row: Dict[str, object] = {
        "base_run_root": str(Path(args.base_run_root)),
        "seq_name": str(args.seq_name),
        "tracker_stage": str(args.tracker_stage),
        "source_glob": str(args.glob),
        "source_runs": 0,
        "matched_runs": 0,
        "identical_runs": 0,
        "status": "running",
        "error": "",
    }
    write_single_row_csv(meta_csv, META_FIELDS, meta_row)
    append_registry(args, summary_csv, "running", "scanning first output-diff frames against base tracker run")

    try:
        base_run_root = Path(args.base_run_root)
        base_seq_file = _resolve_tracker_seq_file(base_run_root, str(args.seq_name), str(args.tracker_stage))
        base_frames = _load_tracker_frames(base_seq_file)
        base_metrics = _read_seq_metrics(base_run_root, str(args.seq_name))

        rows: list[dict[str, object]] = []
        alt_roots = sorted(path for path in REPO_ROOT.glob(str(args.glob)) if path.is_dir())
        meta_row["source_runs"] = int(len(alt_roots))

        for alt_root in alt_roots:
            if alt_root.resolve() == base_run_root.resolve():
                continue
            try:
                alt_seq_file = _resolve_tracker_seq_file(alt_root, str(args.seq_name), str(args.tracker_stage))
                alt_frames = _load_tracker_frames(alt_seq_file)
                alt_metrics = _read_seq_metrics(alt_root, str(args.seq_name))
            except (FileNotFoundError, ValueError):
                continue

            first_diff = _find_first_diff_frame(base_frames, alt_frames)
            meta_row["matched_runs"] = int(meta_row["matched_runs"]) + 1
            if first_diff < 0:
                meta_row["identical_runs"] = int(meta_row["identical_runs"]) + 1

            rows.append(
                {
                    "run_name": alt_root.name,
                    "first_diff_frame": int(first_diff),
                    "delta_HOTA": float(alt_metrics["HOTA"] - base_metrics["HOTA"]),
                    "delta_AssA": float(alt_metrics["AssA"] - base_metrics["AssA"]),
                    "delta_IDF1": float(alt_metrics["IDF1"] - base_metrics["IDF1"]),
                    "delta_MOTA": float(alt_metrics["MOTA"] - base_metrics["MOTA"]),
                    "delta_IDs": float(alt_metrics["IDs"] - base_metrics["IDs"]),
                    "delta_Frag": float(alt_metrics["Frag"] - base_metrics["Frag"]),
                }
            )

        rows.sort(key=lambda row: (int(row["first_diff_frame"]), str(row["run_name"])))
        write_rows_csv(summary_csv, SUMMARY_FIELDS, rows)
        meta_row["status"] = "success"
        write_single_row_csv(meta_csv, META_FIELDS, meta_row)
        append_registry(
            args,
            summary_csv,
            "success",
            f"tracker first-diff scan built: matched_runs={meta_row['matched_runs']}",
        )
        return 0
    except Exception as exc:
        meta_row["status"] = "failed"
        meta_row["error"] = str(exc)
        write_single_row_csv(meta_csv, META_FIELDS, meta_row)
        append_registry(args, summary_csv, "failed", f"tracker first-diff scan failed: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
