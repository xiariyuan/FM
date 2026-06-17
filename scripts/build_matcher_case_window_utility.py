#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


REPO_ROOT = Path("/gemini/code/FMtrack-main/FM-Track")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


REGISTRY_CSV = REPO_ROOT / "outputs" / "experiment_registry.csv"
SUMMARY_FIELDS = [
    "base_run_root",
    "alt_run_root",
    "case_jsonl",
    "gt_root",
    "window",
    "iou_thresh",
    "filter_desc",
    "source_rows",
    "exported_rows",
    "target_evaluable_rows",
    "window_diff_rows",
    "utility_better_rows",
    "utility_worse_rows",
    "utility_tie_rows",
    "status",
    "error",
]


@dataclass
class LocalMetrics:
    gt_instances: int = 0
    matched_frames: int = 0
    fp_total: int = 0
    fn_total: int = 0
    id_switches: int = 0
    fragments: int = 0
    matched_gt_count: int = 0
    score: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build sequence-window utility labels for matcher-case rows by comparing two tracker runs against MOT GT."
    )
    parser.add_argument("--base-run-root", required=True)
    parser.add_argument("--alt-run-root", required=True)
    parser.add_argument("--case-jsonl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--gt-root", default="external/Deep-OC-SORT-main/results/gt/MOT17-val")
    parser.add_argument("--window", type=int, default=30, help="analyze frames [frame_id, frame_id + window]")
    parser.add_argument("--iou-thresh", type=float, default=0.5)
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
        "scripts/build_matcher_case_window_utility.py",
        "--dataset",
        "MOT17",
        "--split",
        "matcher_case_window_utility",
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


def _load_tracker_boxes(path: Path) -> Dict[int, List[Dict[str, object]]]:
    frames: Dict[int, List[Dict[str, object]]] = {}
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


def _load_gt_boxes(path: Path) -> Dict[int, List[Dict[str, object]]]:
    frames: Dict[int, List[Dict[str, object]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split(",")
            if len(parts) < 9:
                continue
            frame_id = int(parts[0])
            gt_id = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            mark = int(float(parts[6]))
            label = int(float(parts[7]))
            visibility = float(parts[8])
            # TrackEval-style MOT person filtering for local utility comparison.
            if mark <= 0 or label != 1:
                continue
            frames.setdefault(frame_id, []).append(
                {
                    "gt_id": int(gt_id),
                    "bbox": (x, y, w, h),
                    "visibility": float(visibility),
                }
            )
    return frames


def _iou_tlwh(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, aw, ah = [float(v) for v in box_a]
    bx1, by1, bw, bh = [float(v) for v in box_b]
    ax2 = ax1 + max(aw, 0.0)
    ay2 = ay1 + max(ah, 0.0)
    bx2 = bx1 + max(bw, 0.0)
    by2 = by1 + max(bh, 0.0)
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, aw) * max(0.0, ah)
    area_b = max(0.0, bw) * max(0.0, bh)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def _greedy_match(gt_rows: Sequence[Mapping[str, object]], track_rows: Sequence[Mapping[str, object]], iou_thresh: float) -> Dict[int, int]:
    triples: List[Tuple[float, int, int]] = []
    for gt_idx, gt_row in enumerate(gt_rows):
        for trk_idx, trk_row in enumerate(track_rows):
            iou = _iou_tlwh(gt_row["bbox"], trk_row["bbox"])
            if iou >= float(iou_thresh):
                triples.append((float(iou), int(gt_idx), int(trk_idx)))
    triples.sort(reverse=True)
    used_gt: set[int] = set()
    used_trk: set[int] = set()
    matched: Dict[int, int] = {}
    for iou, gt_idx, trk_idx in triples:
        if gt_idx in used_gt or trk_idx in used_trk:
            continue
        used_gt.add(gt_idx)
        used_trk.add(trk_idx)
        matched[int(gt_rows[gt_idx]["gt_id"])] = int(track_rows[trk_idx]["track_id"])
    return matched


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


def _target_is_evaluable(gt_frames: Mapping[int, Sequence[Mapping[str, object]]], *, frame_id: int, target_gt_id: int) -> bool:
    if int(target_gt_id) <= 0:
        return False
    for gt_row in gt_frames.get(int(frame_id), []):
        if int(gt_row["gt_id"]) == int(target_gt_id):
            return True
    return False


def _evaluate_window(
    *,
    gt_frames: Mapping[int, Sequence[Mapping[str, object]]],
    tracker_frames: Mapping[int, Sequence[Mapping[str, object]]],
    start_frame: int,
    end_frame: int,
    iou_thresh: float,
) -> Tuple[LocalMetrics, Dict[int, Dict[int, int]]]:
    metrics = LocalMetrics()
    frame_matches: Dict[int, Dict[int, int]] = {}
    last_frame_match_by_gt: Dict[int, int] = {}
    had_match_before: Dict[int, bool] = {}
    matched_gt_ids: set[int] = set()

    for frame_id in range(int(start_frame), int(end_frame) + 1):
        gt_rows = list(gt_frames.get(frame_id, []))
        tracker_rows = list(tracker_frames.get(frame_id, []))
        matches = _greedy_match(gt_rows, tracker_rows, iou_thresh=float(iou_thresh))
        frame_matches[int(frame_id)] = dict(matches)

        metrics.gt_instances += int(len(gt_rows))
        metrics.matched_frames += int(len(matches))
        metrics.fn_total += int(max(0, len(gt_rows) - len(matches)))
        metrics.fp_total += int(max(0, len(tracker_rows) - len(matches)))

        gt_ids_this_frame = {int(row["gt_id"]) for row in gt_rows}
        for gt_id in gt_ids_this_frame:
            current_track_id = matches.get(int(gt_id))
            previous_track_id = last_frame_match_by_gt.get(int(gt_id))
            if current_track_id is not None:
                matched_gt_ids.add(int(gt_id))
                if previous_track_id is not None and int(previous_track_id) != int(current_track_id):
                    metrics.id_switches += 1
                elif previous_track_id is None and had_match_before.get(int(gt_id), False):
                    metrics.fragments += 1
                last_frame_match_by_gt[int(gt_id)] = int(current_track_id)
                had_match_before[int(gt_id)] = True
            else:
                last_frame_match_by_gt[int(gt_id)] = None

    metrics.matched_gt_count = int(len(matched_gt_ids))
    metrics.score = float(
        metrics.matched_frames
        - metrics.fn_total
        - metrics.fp_total
        - 2 * metrics.id_switches
        - metrics.fragments
    )
    return metrics, frame_matches


def _compare_metrics(base_metrics: LocalMetrics, alt_metrics: LocalMetrics) -> str:
    base_key = (
        int(base_metrics.id_switches),
        int(base_metrics.fragments),
        int(base_metrics.fn_total),
        int(base_metrics.fp_total),
        -int(base_metrics.matched_frames),
    )
    alt_key = (
        int(alt_metrics.id_switches),
        int(alt_metrics.fragments),
        int(alt_metrics.fn_total),
        int(alt_metrics.fp_total),
        -int(alt_metrics.matched_frames),
    )
    if alt_key < base_key:
        return "beneficial"
    if alt_key > base_key:
        return "harmful"
    return "tie"


def _first_assignment_diff_offset(
    *,
    base_matches: Mapping[int, Mapping[int, int]],
    alt_matches: Mapping[int, Mapping[int, int]],
    start_frame: int,
    end_frame: int,
) -> int:
    for frame_id in range(int(start_frame), int(end_frame) + 1):
        base_map = dict(base_matches.get(int(frame_id), {}))
        alt_map = dict(alt_matches.get(int(frame_id), {}))
        if base_map != alt_map:
            return int(frame_id - start_frame)
    return -1


def _first_output_diff_offset(
    *,
    base_frames: Mapping[int, Sequence[Mapping[str, object]]],
    alt_frames: Mapping[int, Sequence[Mapping[str, object]]],
    start_frame: int,
    end_frame: int,
) -> int:
    for frame_id in range(int(start_frame), int(end_frame) + 1):
        base_sig = _frame_signature(base_frames.get(int(frame_id), []))
        alt_sig = _frame_signature(alt_frames.get(int(frame_id), []))
        if base_sig != alt_sig:
            return int(frame_id - start_frame)
    return -1


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "summary.csv"
    utility_jsonl = out_dir / "utility_rows.jsonl"

    summary_row: Dict[str, object] = {
        "base_run_root": str(Path(args.base_run_root)),
        "alt_run_root": str(Path(args.alt_run_root)),
        "case_jsonl": str(Path(args.case_jsonl)),
        "gt_root": str(Path(args.gt_root)),
        "window": int(args.window),
        "iou_thresh": float(args.iou_thresh),
        "filter_desc": f"takeover_applied=1|solver_changed_row={int(args.solver_changed_row)}|changed_match=1|raw_det_index>=0",
        "source_rows": 0,
        "exported_rows": 0,
        "target_evaluable_rows": 0,
        "window_diff_rows": 0,
        "utility_better_rows": 0,
        "utility_worse_rows": 0,
        "utility_tie_rows": 0,
        "status": "running",
        "error": "",
    }
    if bool(args.require_owner_other_track):
        summary_row["filter_desc"] = str(summary_row["filter_desc"]) + "|base_best_det_raw_owned_by_other_track=1"

    write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)
    append_registry(args, summary_csv, "running", "building matcher-case window utility labels")

    try:
        base_run_root = Path(args.base_run_root)
        alt_run_root = Path(args.alt_run_root)
        base_tracker_dir = _resolve_single_tracker_dir(base_run_root)
        alt_tracker_dir = _resolve_single_tracker_dir(alt_run_root)
        gt_root = Path(args.gt_root)

        base_seq_cache: Dict[str, Dict[int, List[Dict[str, object]]]] = {}
        alt_seq_cache: Dict[str, Dict[int, List[Dict[str, object]]]] = {}
        gt_seq_cache: Dict[str, Dict[int, List[Dict[str, object]]]] = {}
        window_cache: Dict[Tuple[str, int], Dict[str, object]] = {}
        seen_case_keys: set[Tuple[object, ...]] = set()

        with Path(args.case_jsonl).open("r", encoding="utf-8") as handle, utility_jsonl.open("w", encoding="utf-8") as out_handle:
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

                cache_key = (seq_name, frame_id)
                if cache_key not in window_cache:
                    start_frame = int(frame_id)
                    end_frame = int(frame_id + int(args.window))
                    base_metrics, base_matches = _evaluate_window(
                        gt_frames=gt_seq_cache[seq_name],
                        tracker_frames=base_seq_cache[seq_name],
                        start_frame=start_frame,
                        end_frame=end_frame,
                        iou_thresh=float(args.iou_thresh),
                    )
                    alt_metrics, alt_matches = _evaluate_window(
                        gt_frames=gt_seq_cache[seq_name],
                        tracker_frames=alt_seq_cache[seq_name],
                        start_frame=start_frame,
                        end_frame=end_frame,
                        iou_thresh=float(args.iou_thresh),
                    )
                    compare_status = _compare_metrics(base_metrics, alt_metrics)
                    first_assign_diff_offset = _first_assignment_diff_offset(
                        base_matches=base_matches,
                        alt_matches=alt_matches,
                        start_frame=start_frame,
                        end_frame=end_frame,
                    )
                    first_output_diff_offset = _first_output_diff_offset(
                        base_frames=base_seq_cache[seq_name],
                        alt_frames=alt_seq_cache[seq_name],
                        start_frame=start_frame,
                        end_frame=end_frame,
                    )
                    window_cache[cache_key] = {
                        "base_metrics": base_metrics,
                        "alt_metrics": alt_metrics,
                        "compare_status": compare_status,
                        "first_assign_diff_offset": int(first_assign_diff_offset),
                        "first_output_diff_offset": int(first_output_diff_offset),
                    }

                cached = window_cache[cache_key]
                base_metrics = cached["base_metrics"]
                alt_metrics = cached["alt_metrics"]
                compare_status = str(cached["compare_status"])
                first_assign_diff_offset = int(cached["first_assign_diff_offset"])
                first_output_diff_offset = int(cached["first_output_diff_offset"])
                target_evaluable = int(
                    _target_is_evaluable(
                        gt_seq_cache[seq_name],
                        frame_id=frame_id,
                        target_gt_id=target_gt_id,
                    )
                )

                export_row = dict(row)
                export_row.update(
                    {
                        "window": int(args.window),
                        "iou_thresh": float(args.iou_thresh),
                        "target_gt_evaluable": int(target_evaluable),
                        "window_diff": int(first_output_diff_offset >= 0),
                        "first_output_diff_offset": int(first_output_diff_offset),
                        "first_eval_assignment_diff_offset": int(first_assign_diff_offset),
                        "window_utility_status": str(compare_status),
                        "window_utility_label": 1 if compare_status == "beneficial" else 0,
                        "window_utility_tie": 1 if compare_status == "tie" else 0,
                        "base_window_gt_instances": int(base_metrics.gt_instances),
                        "base_window_matched_frames": int(base_metrics.matched_frames),
                        "base_window_fp_total": int(base_metrics.fp_total),
                        "base_window_fn_total": int(base_metrics.fn_total),
                        "base_window_id_switches": int(base_metrics.id_switches),
                        "base_window_fragments": int(base_metrics.fragments),
                        "base_window_matched_gt_count": int(base_metrics.matched_gt_count),
                        "base_window_score": float(base_metrics.score),
                        "alt_window_gt_instances": int(alt_metrics.gt_instances),
                        "alt_window_matched_frames": int(alt_metrics.matched_frames),
                        "alt_window_fp_total": int(alt_metrics.fp_total),
                        "alt_window_fn_total": int(alt_metrics.fn_total),
                        "alt_window_id_switches": int(alt_metrics.id_switches),
                        "alt_window_fragments": int(alt_metrics.fragments),
                        "alt_window_matched_gt_count": int(alt_metrics.matched_gt_count),
                        "alt_window_score": float(alt_metrics.score),
                        "window_score_delta": float(alt_metrics.score - base_metrics.score),
                    }
                )
                out_handle.write(json.dumps(export_row))
                out_handle.write("\n")

                summary_row["exported_rows"] = int(summary_row["exported_rows"]) + 1
                summary_row["target_evaluable_rows"] = int(summary_row["target_evaluable_rows"]) + int(target_evaluable)
                summary_row["window_diff_rows"] = int(summary_row["window_diff_rows"]) + int(first_output_diff_offset >= 0)
                if compare_status == "beneficial":
                    summary_row["utility_better_rows"] = int(summary_row["utility_better_rows"]) + 1
                elif compare_status == "harmful":
                    summary_row["utility_worse_rows"] = int(summary_row["utility_worse_rows"]) + 1
                else:
                    summary_row["utility_tie_rows"] = int(summary_row["utility_tie_rows"]) + 1

        summary_row["status"] = "success"
        write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)
        append_registry(
            args,
            summary_csv,
            "success",
            f"matcher-case window utility built: rows={summary_row['exported_rows']}",
        )
        return 0
    except Exception as exc:
        summary_row["status"] = "failed"
        summary_row["error"] = repr(exc)
        write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)
        append_registry(args, summary_csv, "failed", f"matcher-case window utility failed: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
