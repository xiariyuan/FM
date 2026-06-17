#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple


REPO_ROOT = Path("/gemini/code/FMtrack-main/FM-Track")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


REGISTRY_CSV = REPO_ROOT / "outputs" / "experiment_registry.csv"
EVAL_SCRIPT = REPO_ROOT / "scripts" / "eval_motstyle_trackeval.py"
SUMMARY_FIELDS = [
    "base_run_root",
    "alt_run_root",
    "case_jsonl",
    "gt_root",
    "filter_desc",
    "source_rows",
    "exported_rows",
    "target_evaluable_rows",
    "seqs_with_output_diff",
    "first_diff_aligned_rows",
    "first_diff_misaligned_rows",
    "rank_better_rows",
    "rank_worse_rows",
    "rank_tie_rows",
    "status",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build suffix TrackEval metrics for matcher-case rows by comparing two tracker runs from each case frame to sequence end."
    )
    parser.add_argument("--base-run-root", required=True)
    parser.add_argument("--alt-run-root", required=True)
    parser.add_argument("--case-jsonl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--gt-root", default="external/Deep-OC-SORT-main/results/gt/MOT17-val")
    parser.add_argument(
        "--solver-changed-row",
        type=int,
        choices=[0, 1],
        default=0,
        help="only keep matcher_case rows with the given solver_changed_row flag",
    )
    parser.add_argument(
        "--require-owner-other-track",
        action="store_true",
        help="keep only rows whose baseline-best detection is owned by another raw track",
    )
    parser.add_argument("--registry-csv", default=str(REGISTRY_CSV))
    return parser.parse_args()


def write_single_row_csv(path: Path, fieldnames: Sequence[str], row: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
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
        "scripts/build_matcher_case_suffix_trackeval.py",
        "--dataset",
        "MOT17",
        "--split",
        "matcher_case_suffix_trackeval",
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
    subprocess.run(cmd, cwd=REPO_ROOT, check=False)


def _resolve_single_tracker_dir(run_root: Path) -> Path:
    mot_val_root = run_root / "results" / "trackers" / "MOT17-val"
    if not mot_val_root.is_dir():
        raise FileNotFoundError(f"Missing tracker root: {mot_val_root}")
    data_dirs = sorted(path for path in mot_val_root.glob("*_fgas/data") if path.is_dir())
    if len(data_dirs) != 1:
        raise ValueError(f"Expected exactly one *_fgas/data dir under {mot_val_root}, got {len(data_dirs)}")
    return data_dirs[0]


def _keep_row(
    row: Mapping[str, object],
    *,
    solver_changed_row: int,
    require_owner_other_track: bool,
) -> bool:
    keep = bool(
        int(row.get("takeover_applied", 0)) == 1
        and int(row.get("solver_changed_row", 0)) == int(solver_changed_row)
        and int(row.get("changed_match", 0)) == 1
        and int(row.get("raw_det_index", -1)) >= 0
    )
    if keep and require_owner_other_track:
        keep = bool(int(row.get("base_best_det_raw_owned_by_other_track", 0)) == 1)
    return keep


def _case_key(row: Mapping[str, object]) -> Tuple[object, ...]:
    return (
        str(row.get("seq_name", "")),
        int(row.get("frame_id", -1)),
        int(row.get("track_index", -1)),
        int(row.get("track_gt_id", -1)),
        int(row.get("raw_det_index", -1)),
        int(row.get("fgas_det_index", -1)),
        int(row.get("base_best_det_index", -1)),
        int(row.get("solver_changed_row", 0)),
        int(row.get("label", 0)),
    )


def _load_gt_boxes(path: Path) -> Dict[int, list[Dict[str, object]]]:
    frames: Dict[int, list[Dict[str, object]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split(",")
            if len(parts) < 9:
                continue
            frame_id = int(parts[0])
            gt_id = int(parts[1])
            mark = int(float(parts[6]))
            label = int(float(parts[7]))
            if mark <= 0 or label != 1:
                continue
            frames.setdefault(frame_id, []).append({"gt_id": gt_id})
    return frames


def _load_tracker_boxes(path: Path) -> Dict[int, list[Dict[str, object]]]:
    frames: Dict[int, list[Dict[str, object]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            frame_id = int(float(parts[0]))
            track_id = int(float(parts[1]))
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            frames.setdefault(frame_id, []).append(
                {
                    "track_id": int(track_id),
                    "bbox": (x, y, w, h),
                }
            )
    return frames


def _frame_signature(rows: Sequence[Mapping[str, object]]) -> Tuple[Tuple[int, Tuple[float, float, float, float]], ...]:
    items = []
    for row in rows:
        bbox = row["bbox"]
        items.append(
            (
                int(row["track_id"]),
                (
                    round(float(bbox[0]), 3),
                    round(float(bbox[1]), 3),
                    round(float(bbox[2]), 3),
                    round(float(bbox[3]), 3),
                ),
            )
        )
    return tuple(sorted(items))


def _find_first_output_diff_frame(
    base_frames: Mapping[int, Sequence[Mapping[str, object]]],
    alt_frames: Mapping[int, Sequence[Mapping[str, object]]],
) -> int:
    all_frames = sorted(set(base_frames.keys()) | set(alt_frames.keys()))
    for frame_id in all_frames:
        if _frame_signature(base_frames.get(int(frame_id), [])) != _frame_signature(alt_frames.get(int(frame_id), [])):
            return int(frame_id)
    return -1


def _target_is_evaluable(gt_frames: Mapping[int, Sequence[Mapping[str, object]]], *, frame_id: int, target_gt_id: int) -> bool:
    if int(target_gt_id) <= 0:
        return False
    for gt_row in gt_frames.get(int(frame_id), []):
        if int(gt_row["gt_id"]) == int(target_gt_id):
            return True
    return False


def _read_seq_length(seqinfo_path: Path) -> int:
    for line in seqinfo_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("seqLength="):
            return int(line.split("=", 1)[1].strip())
    raise ValueError(f"seqLength not found in {seqinfo_path}")


def _shift_csv_lines(src_path: Path, dst_path: Path, start_frame: int) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shifted_lines: list[str] = []
    with src_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split(",")
            frame_id = int(float(parts[0]))
            if frame_id < int(start_frame):
                continue
            parts[0] = str(frame_id - int(start_frame) + 1)
            shifted_lines.append(",".join(parts))
    with dst_path.open("w", encoding="utf-8") as handle:
        if shifted_lines:
            handle.write("\n".join(shifted_lines))
            handle.write("\n")


def _write_shifted_seqinfo(src_path: Path, dst_path: Path, start_frame: int) -> int:
    lines = src_path.read_text(encoding="utf-8").splitlines()
    orig_seq_length = _read_seq_length(src_path)
    suffix_length = max(1, orig_seq_length - int(start_frame) + 1)
    out_lines: list[str] = []
    for line in lines:
        if line.startswith("seqLength="):
            out_lines.append(f"seqLength={suffix_length}")
        else:
            out_lines.append(line)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return suffix_length


def _find_summary_file(work_dir: Path) -> Path:
    candidates = sorted((work_dir / "eval").glob("**/pedestrian_summary.txt"))
    if not candidates:
        raise FileNotFoundError(f"TrackEval summary not found under {work_dir}")
    return candidates[0]


def _parse_summary(path: Path) -> Dict[str, float]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError(f"Invalid TrackEval summary: {path}")
    keys = lines[0].split()
    values = lines[1].split()
    if len(values) < len(keys):
        raise ValueError(f"TrackEval summary shorter than header: {path}")
    row: Dict[str, float] = {}
    for key, value in zip(keys, values):
        try:
            row[key] = float(value)
        except ValueError:
            continue
    return row


def _metric_tuple(metrics: Mapping[str, float]) -> Tuple[float, float, float, float, float, float]:
    return (
        float(metrics.get("HOTA", 0.0)),
        float(metrics.get("AssA", 0.0)),
        float(metrics.get("IDF1", 0.0)),
        float(metrics.get("MOTA", 0.0)),
        -float(metrics.get("IDSW", 0.0)),
        -float(metrics.get("Frag", 0.0)),
    )


def _compare_metric(base_value: float, alt_value: float, tol: float = 1e-9) -> str:
    if float(alt_value) > float(base_value) + float(tol):
        return "beneficial"
    if float(alt_value) < float(base_value) - float(tol):
        return "harmful"
    return "tie"


def _compare_rank(base_metrics: Mapping[str, float], alt_metrics: Mapping[str, float]) -> str:
    base_key = _metric_tuple(base_metrics)
    alt_key = _metric_tuple(alt_metrics)
    if alt_key > base_key:
        return "beneficial"
    if alt_key < base_key:
        return "harmful"
    return "tie"


def _prepare_suffix_eval_assets(
    *,
    gt_root: Path,
    tracker_dir: Path,
    seq_name: str,
    start_frame: int,
    dst_root: Path,
) -> Tuple[Path, Path, int]:
    src_seq_root = gt_root / seq_name
    src_gt_path = src_seq_root / "gt" / "gt.txt"
    src_seqinfo_path = src_seq_root / "seqinfo.ini"
    src_tracker_path = tracker_dir / f"{seq_name}.txt"
    if not src_gt_path.is_file():
        raise FileNotFoundError(f"Missing GT file: {src_gt_path}")
    if not src_seqinfo_path.is_file():
        raise FileNotFoundError(f"Missing seqinfo.ini: {src_seqinfo_path}")
    if not src_tracker_path.is_file():
        raise FileNotFoundError(f"Missing tracker output: {src_tracker_path}")

    gt_out_root = dst_root / "gt"
    tracker_out_root = dst_root / "tracker"
    suffix_length = _write_shifted_seqinfo(src_seqinfo_path, gt_out_root / seq_name / "seqinfo.ini", start_frame=int(start_frame))
    _shift_csv_lines(src_gt_path, gt_out_root / seq_name / "gt" / "gt.txt", start_frame=int(start_frame))
    _shift_csv_lines(src_tracker_path, tracker_out_root / f"{seq_name}.txt", start_frame=int(start_frame))
    return gt_out_root, tracker_out_root, suffix_length


def _run_trackeval(
    *,
    gt_root: Path,
    tracker_results_dir: Path,
    tracker_name: str,
    work_dir: Path,
) -> Dict[str, float]:
    if work_dir.exists():
        shutil.rmtree(work_dir)
    cmd = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--benchmark-name",
        "MOT17",
        "--split-to-eval",
        "suffix",
        "--gt-root",
        str(gt_root),
        "--results-dir",
        str(tracker_results_dir),
        "--tracker-name",
        tracker_name,
        "--work-dir",
        str(work_dir),
        "--keep-workdir",
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    summary_path = _find_summary_file(work_dir)
    metrics = _parse_summary(summary_path)
    metrics["summary_path"] = str(summary_path)
    return metrics


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "summary.csv"
    rows_jsonl = out_dir / "suffix_trackeval_rows.jsonl"

    summary_row: Dict[str, object] = {
        "base_run_root": str(Path(args.base_run_root)),
        "alt_run_root": str(Path(args.alt_run_root)),
        "case_jsonl": str(Path(args.case_jsonl)),
        "gt_root": str(Path(args.gt_root)),
        "filter_desc": f"takeover_applied=1|solver_changed_row={int(args.solver_changed_row)}|changed_match=1|raw_det_index>=0",
        "source_rows": 0,
        "exported_rows": 0,
        "target_evaluable_rows": 0,
        "seqs_with_output_diff": 0,
        "first_diff_aligned_rows": 0,
        "first_diff_misaligned_rows": 0,
        "rank_better_rows": 0,
        "rank_worse_rows": 0,
        "rank_tie_rows": 0,
        "status": "running",
        "error": "",
    }
    if bool(args.require_owner_other_track):
        summary_row["filter_desc"] = str(summary_row["filter_desc"]) + "|base_best_det_raw_owned_by_other_track=1"

    write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)
    append_registry(args, summary_csv, "running", "building matcher-case suffix TrackEval labels")

    try:
        base_run_root = Path(args.base_run_root)
        alt_run_root = Path(args.alt_run_root)
        gt_root = Path(args.gt_root)
        base_tracker_dir = _resolve_single_tracker_dir(base_run_root)
        alt_tracker_dir = _resolve_single_tracker_dir(alt_run_root)

        base_seq_cache: Dict[str, Dict[int, list[Dict[str, object]]]] = {}
        alt_seq_cache: Dict[str, Dict[int, list[Dict[str, object]]]] = {}
        gt_seq_cache: Dict[str, Dict[int, list[Dict[str, object]]]] = {}
        seq_first_diff_cache: Dict[str, int] = {}
        seen_case_keys: set[Tuple[object, ...]] = set()
        eval_cache: Dict[Tuple[str, int], Dict[str, object]] = {}
        diff_seq_names: set[str] = set()

        with Path(args.case_jsonl).open("r", encoding="utf-8") as handle, rows_jsonl.open("w", encoding="utf-8") as out_handle:
            for line in handle:
                row = json.loads(line)
                summary_row["source_rows"] = int(summary_row["source_rows"]) + 1
                if not _keep_row(
                    row,
                    solver_changed_row=int(args.solver_changed_row),
                    require_owner_other_track=bool(args.require_owner_other_track),
                ):
                    continue
                case_key = _case_key(row)
                if case_key in seen_case_keys:
                    continue
                seen_case_keys.add(case_key)

                seq_name = str(row.get("seq_name", ""))
                frame_id = int(row.get("frame_id", -1))
                target_gt_id = int(row.get("track_gt_id", -1))
                if not seq_name or frame_id < 0:
                    continue

                if seq_name not in base_seq_cache:
                    base_seq_cache[seq_name] = _load_tracker_boxes(base_tracker_dir / f"{seq_name}.txt")
                if seq_name not in alt_seq_cache:
                    alt_seq_cache[seq_name] = _load_tracker_boxes(alt_tracker_dir / f"{seq_name}.txt")
                if seq_name not in gt_seq_cache:
                    gt_seq_cache[seq_name] = _load_gt_boxes(gt_root / seq_name / "gt" / "gt.txt")
                if seq_name not in seq_first_diff_cache:
                    seq_first_diff_cache[seq_name] = _find_first_output_diff_frame(
                        base_seq_cache[seq_name],
                        alt_seq_cache[seq_name],
                    )
                target_evaluable = int(
                    _target_is_evaluable(
                        gt_seq_cache[seq_name],
                        frame_id=frame_id,
                        target_gt_id=target_gt_id,
                    )
                )
                first_output_diff_frame = int(seq_first_diff_cache[seq_name])
                seq_has_output_diff = int(first_output_diff_frame >= 0)
                if seq_has_output_diff:
                    diff_seq_names.add(seq_name)
                output_diff_at_frame = int(
                    _frame_signature(base_seq_cache[seq_name].get(frame_id, []))
                    != _frame_signature(alt_seq_cache[seq_name].get(frame_id, []))
                )
                is_seq_first_output_diff_frame = int(seq_has_output_diff and frame_id == first_output_diff_frame)
                first_output_diff_frame_offset = int(frame_id - first_output_diff_frame) if seq_has_output_diff else -1

                cache_key = (seq_name, frame_id)
                if cache_key not in eval_cache:
                    case_root = out_dir / "work" / f"{seq_name}_frame{frame_id}"
                    base_assets_root = case_root / "base_assets"
                    alt_assets_root = case_root / "alt_assets"
                    base_gt_root, base_tracker_results_dir, suffix_length = _prepare_suffix_eval_assets(
                        gt_root=gt_root,
                        tracker_dir=base_tracker_dir,
                        seq_name=seq_name,
                        start_frame=frame_id,
                        dst_root=base_assets_root,
                    )
                    alt_gt_root, alt_tracker_results_dir, _ = _prepare_suffix_eval_assets(
                        gt_root=gt_root,
                        tracker_dir=alt_tracker_dir,
                        seq_name=seq_name,
                        start_frame=frame_id,
                        dst_root=alt_assets_root,
                    )
                    base_metrics = _run_trackeval(
                        gt_root=base_gt_root,
                        tracker_results_dir=base_tracker_results_dir,
                        tracker_name="base",
                        work_dir=case_root / "trackeval_base",
                    )
                    alt_metrics = _run_trackeval(
                        gt_root=alt_gt_root,
                        tracker_results_dir=alt_tracker_results_dir,
                        tracker_name="alt",
                        work_dir=case_root / "trackeval_alt",
                    )
                    eval_cache[cache_key] = {
                        "suffix_length": int(suffix_length),
                        "base_metrics": base_metrics,
                        "alt_metrics": alt_metrics,
                        "rank_status": _compare_rank(base_metrics, alt_metrics),
                    }

                cached = eval_cache[cache_key]
                base_metrics = cached["base_metrics"]
                alt_metrics = cached["alt_metrics"]
                rank_status = str(cached["rank_status"])

                export_row = dict(row)
                export_row.update(
                    {
                        "target_gt_evaluable": int(target_evaluable),
                        "seq_has_output_diff": int(seq_has_output_diff),
                        "seq_first_output_diff_frame": int(first_output_diff_frame),
                        "case_output_diff_at_frame": int(output_diff_at_frame),
                        "is_seq_first_output_diff_frame": int(is_seq_first_output_diff_frame),
                        "case_vs_seq_first_output_diff_offset": int(first_output_diff_frame_offset),
                        "suffix_length": int(cached["suffix_length"]),
                        "suffix_rank_status": rank_status,
                        "suffix_hota_status": _compare_metric(float(base_metrics.get("HOTA", 0.0)), float(alt_metrics.get("HOTA", 0.0))),
                        "suffix_assa_status": _compare_metric(float(base_metrics.get("AssA", 0.0)), float(alt_metrics.get("AssA", 0.0))),
                        "suffix_idf1_status": _compare_metric(float(base_metrics.get("IDF1", 0.0)), float(alt_metrics.get("IDF1", 0.0))),
                        "base_suffix_hota": float(base_metrics.get("HOTA", 0.0)),
                        "alt_suffix_hota": float(alt_metrics.get("HOTA", 0.0)),
                        "delta_suffix_hota": float(alt_metrics.get("HOTA", 0.0) - base_metrics.get("HOTA", 0.0)),
                        "base_suffix_assa": float(base_metrics.get("AssA", 0.0)),
                        "alt_suffix_assa": float(alt_metrics.get("AssA", 0.0)),
                        "delta_suffix_assa": float(alt_metrics.get("AssA", 0.0) - base_metrics.get("AssA", 0.0)),
                        "base_suffix_idf1": float(base_metrics.get("IDF1", 0.0)),
                        "alt_suffix_idf1": float(alt_metrics.get("IDF1", 0.0)),
                        "delta_suffix_idf1": float(alt_metrics.get("IDF1", 0.0) - base_metrics.get("IDF1", 0.0)),
                        "base_suffix_mota": float(base_metrics.get("MOTA", 0.0)),
                        "alt_suffix_mota": float(alt_metrics.get("MOTA", 0.0)),
                        "delta_suffix_mota": float(alt_metrics.get("MOTA", 0.0) - base_metrics.get("MOTA", 0.0)),
                        "base_suffix_idsw": float(base_metrics.get("IDSW", 0.0)),
                        "alt_suffix_idsw": float(alt_metrics.get("IDSW", 0.0)),
                        "delta_suffix_idsw": float(alt_metrics.get("IDSW", 0.0) - base_metrics.get("IDSW", 0.0)),
                        "base_suffix_frag": float(base_metrics.get("Frag", 0.0)),
                        "alt_suffix_frag": float(alt_metrics.get("Frag", 0.0)),
                        "delta_suffix_frag": float(alt_metrics.get("Frag", 0.0) - base_metrics.get("Frag", 0.0)),
                        "base_suffix_summary_path": str(base_metrics.get("summary_path", "")),
                        "alt_suffix_summary_path": str(alt_metrics.get("summary_path", "")),
                    }
                )
                out_handle.write(json.dumps(export_row))
                out_handle.write("\n")

                summary_row["exported_rows"] = int(summary_row["exported_rows"]) + 1
                summary_row["target_evaluable_rows"] = int(summary_row["target_evaluable_rows"]) + int(target_evaluable)
                summary_row["seqs_with_output_diff"] = int(len(diff_seq_names))
                if seq_has_output_diff:
                    if is_seq_first_output_diff_frame:
                        summary_row["first_diff_aligned_rows"] = int(summary_row["first_diff_aligned_rows"]) + 1
                    else:
                        summary_row["first_diff_misaligned_rows"] = int(summary_row["first_diff_misaligned_rows"]) + 1
                if rank_status == "beneficial":
                    summary_row["rank_better_rows"] = int(summary_row["rank_better_rows"]) + 1
                elif rank_status == "harmful":
                    summary_row["rank_worse_rows"] = int(summary_row["rank_worse_rows"]) + 1
                else:
                    summary_row["rank_tie_rows"] = int(summary_row["rank_tie_rows"]) + 1

        summary_row["status"] = "success"
        write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)
        append_registry(
            args,
            summary_csv,
            "success",
            f"matcher-case suffix TrackEval built: rows={summary_row['exported_rows']}",
        )
        return 0
    except Exception as exc:
        summary_row["status"] = "failed"
        summary_row["error"] = repr(exc)
        write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)
        append_registry(args, summary_csv, "failed", f"matcher-case suffix TrackEval failed: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
