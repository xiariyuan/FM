#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


REPO_ROOT = Path("/gemini/code/FMtrack-main/FM-Track")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


REGISTRY_CSV = REPO_ROOT / "outputs" / "experiment_registry.csv"
META_FIELDS = [
    "run_root",
    "seq_name",
    "base_stage",
    "alt_stage",
    "frame_spec",
    "matcher_frame_spec",
    "status",
    "error",
]
SUMMARY_FIELDS = [
    "run_name",
    "seq_name",
    "base_stage",
    "alt_stage",
    "frame_spec",
    "matcher_frame_spec",
    "first_diff_frame",
    "delta_HOTA",
    "delta_AssA",
    "delta_IDF1",
    "delta_MOTA",
    "delta_IDs",
    "delta_Frag",
    "base_only_track_ids_at_first_diff",
    "alt_only_track_ids_at_first_diff",
    "shared_track_ids_with_bbox_change_at_first_diff",
]
TRACKER_FIELDS = [
    "stage",
    "frame_id",
    "track_id",
    "x",
    "y",
    "w",
    "h",
    "score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export side-by-side tracker rows and matcher-case rows for specific probe frames."
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--seq-name", required=True)
    parser.add_argument("--base-stage", required=True)
    parser.add_argument("--alt-stage", required=True)
    parser.add_argument(
        "--frames",
        required=True,
        help="Comma-separated frame ids or closed ranges, for example: 134-145,217,280-284",
    )
    parser.add_argument(
        "--matcher-frames",
        default="",
        help="Comma-separated frame ids or closed ranges for matcher_case_rows.jsonl filtering. Defaults to --frames.",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--matcher-jsonl", default="")
    parser.add_argument("--registry-csv", default=str(REGISTRY_CSV))
    return parser.parse_args()


def parse_frame_spec(spec: str) -> list[int]:
    values: set[int] = set()
    for chunk in str(spec).split(","):
        token = chunk.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"Invalid frame range: {token}")
            values.update(range(start, end + 1))
        else:
            values.add(int(token))
    return sorted(values)


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
        "scripts/build_fgas_case_probe.py",
        "--dataset",
        "MOT17",
        "--split",
        "fgas_case_probe",
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


def _flatten_tracker_rows(stage: str, frame_rows: Mapping[int, tuple[tuple[object, ...], ...]], frames: Sequence[int]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for frame_id in frames:
        for track_id, bbox, score in frame_rows.get(frame_id, ()):
            rows.append(
                {
                    "stage": str(stage),
                    "frame_id": int(frame_id),
                    "track_id": int(track_id),
                    "x": float(bbox[0]),
                    "y": float(bbox[1]),
                    "w": float(bbox[2]),
                    "h": float(bbox[3]),
                    "score": float(score),
                }
            )
    return rows


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


def _track_ids_with_bbox_change(
    base_rows: Sequence[tuple[object, ...]],
    alt_rows: Sequence[tuple[object, ...]],
) -> tuple[list[int], list[int], list[int]]:
    base_map = {int(track_id): (bbox, score) for track_id, bbox, score in base_rows}
    alt_map = {int(track_id): (bbox, score) for track_id, bbox, score in alt_rows}
    base_ids = set(base_map)
    alt_ids = set(alt_map)
    changed_shared_ids = sorted(
        track_id for track_id in (base_ids & alt_ids) if tuple(base_map[track_id]) != tuple(alt_map[track_id])
    )
    return sorted(base_ids - alt_ids), sorted(alt_ids - base_ids), changed_shared_ids


def _load_filtered_matcher_rows(path: Path, seq_name: str, frames: Sequence[int]) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    selected: list[dict[str, object]] = []
    frame_set = set(int(frame_id) for frame_id in frames)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if str(row.get("seq_name", "")) != str(seq_name):
                continue
            frame_id = int(row.get("frame_id", -1))
            if frame_id not in frame_set:
                continue
            selected.append(dict(row))
    selected.sort(key=lambda row: (int(row.get("frame_id", -1)), int(row.get("track_index", -1))))
    return selected


def _matcher_fieldnames(rows: Sequence[Mapping[str, object]]) -> list[str]:
    preferred = [
        "seq_name",
        "frame_id",
        "track_index",
        "track_internal_id",
        "track_gt_id",
        "target_track_time_since_update",
        "target_track_hit_streak",
        "target_track_hits",
        "raw_det_index",
        "fgas_det_index",
        "base_best_det_index",
        "component_row_count",
        "component_col_count",
        "changed_match",
        "takeover_applied",
        "solver_changed_row",
        "row_no_match",
        "flip_type",
        "label",
    ]
    fieldnames: list[str] = []
    seen: set[str] = set()
    for key in preferred:
        for row in rows:
            if key in row and key not in seen:
                fieldnames.append(key)
                seen.add(key)
                break
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    return fieldnames


def main() -> int:
    args = parse_args()
    run_root = Path(args.run_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_csv = out_dir / "meta_summary.csv"
    summary_csv = out_dir / "summary.csv"
    tracker_rows_csv = out_dir / "tracker_rows.csv"
    matcher_rows_csv = out_dir / "matcher_case_rows.csv"

    matcher_frame_spec = args.matcher_frames or args.frames
    meta_row: Dict[str, object] = {
        "run_root": str(run_root),
        "seq_name": str(args.seq_name),
        "base_stage": str(args.base_stage),
        "alt_stage": str(args.alt_stage),
        "frame_spec": str(args.frames),
        "matcher_frame_spec": str(matcher_frame_spec),
        "status": "running",
        "error": "",
    }
    write_single_row_csv(meta_csv, META_FIELDS, meta_row)
    append_registry(args, summary_csv, "running", "building focused FGAS case probe")

    try:
        frame_ids = parse_frame_spec(args.frames)
        matcher_frame_ids = parse_frame_spec(matcher_frame_spec)
        matcher_jsonl = Path(args.matcher_jsonl) if args.matcher_jsonl else run_root / "matcher_case_rows.jsonl"

        base_seq_file = _resolve_tracker_seq_file(run_root, str(args.seq_name), str(args.base_stage))
        alt_seq_file = _resolve_tracker_seq_file(run_root, str(args.seq_name), str(args.alt_stage))
        base_frames = _load_tracker_frames(base_seq_file)
        alt_frames = _load_tracker_frames(alt_seq_file)
        base_metrics = _read_seq_metrics(run_root, str(args.seq_name), str(args.base_stage))
        alt_metrics = _read_seq_metrics(run_root, str(args.seq_name), str(args.alt_stage))
        first_diff_frame = _find_first_diff_frame(base_frames, alt_frames)
        base_only_ids, alt_only_ids, changed_shared_ids = _track_ids_with_bbox_change(
            base_frames.get(first_diff_frame, ()),
            alt_frames.get(first_diff_frame, ()),
        )

        summary_row = {
            "run_name": run_root.name,
            "seq_name": str(args.seq_name),
            "base_stage": str(args.base_stage),
            "alt_stage": str(args.alt_stage),
            "frame_spec": str(args.frames),
            "matcher_frame_spec": str(matcher_frame_spec),
            "first_diff_frame": int(first_diff_frame),
            "delta_HOTA": float(alt_metrics["HOTA"] - base_metrics["HOTA"]),
            "delta_AssA": float(alt_metrics["AssA"] - base_metrics["AssA"]),
            "delta_IDF1": float(alt_metrics["IDF1"] - base_metrics["IDF1"]),
            "delta_MOTA": float(alt_metrics["MOTA"] - base_metrics["MOTA"]),
            "delta_IDs": float(alt_metrics["IDs"] - base_metrics["IDs"]),
            "delta_Frag": float(alt_metrics["Frag"] - base_metrics["Frag"]),
            "base_only_track_ids_at_first_diff": " ".join(str(track_id) for track_id in base_only_ids),
            "alt_only_track_ids_at_first_diff": " ".join(str(track_id) for track_id in alt_only_ids),
            "shared_track_ids_with_bbox_change_at_first_diff": " ".join(str(track_id) for track_id in changed_shared_ids),
        }
        tracker_rows = _flatten_tracker_rows(str(args.base_stage), base_frames, frame_ids) + _flatten_tracker_rows(
            str(args.alt_stage), alt_frames, frame_ids
        )
        tracker_rows.sort(key=lambda row: (int(row["frame_id"]), str(row["stage"]), int(row["track_id"])))
        matcher_rows = _load_filtered_matcher_rows(matcher_jsonl, str(args.seq_name), matcher_frame_ids)

        write_rows_csv(summary_csv, SUMMARY_FIELDS, [summary_row])
        write_rows_csv(tracker_rows_csv, TRACKER_FIELDS, tracker_rows)
        if matcher_rows:
            write_rows_csv(matcher_rows_csv, _matcher_fieldnames(matcher_rows), matcher_rows)
        else:
            write_rows_csv(matcher_rows_csv, ["seq_name", "frame_id"], [])
        meta_row["status"] = "success"
        write_single_row_csv(meta_csv, META_FIELDS, meta_row)
        append_registry(args, summary_csv, "success", "focused FGAS case probe built")
        return 0
    except Exception as exc:
        meta_row["status"] = "failed"
        meta_row["error"] = str(exc)
        write_single_row_csv(meta_csv, META_FIELDS, meta_row)
        append_registry(args, summary_csv, "failed", f"focused FGAS case probe failed: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
