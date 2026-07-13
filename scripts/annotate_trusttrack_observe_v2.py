#!/usr/bin/env python3
"""Offline GT annotation for TrustTrack observe-v2 association logs.

The online observe wrapper never reads GT. This script joins its chosen pairs to:

1. TrackEval-style fixed-IoU detection-to-GT labels for the chosen detections.
2. TrackEval-style fixed-IoU output-to-GT labels for the tracker result file.
3. Track-centric prior-output identity context.
4. GT-centric tracker-ID transitions and future transition flags.

The two identity views are intentionally kept separate. A track-history mismatch
is not automatically an evaluation ID switch; GT-centric output transitions are
the evaluation-oriented label.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

import current_output_id_oracle_v2 as oracle

Box = Tuple[float, float, float, float]
OutputRow = Tuple[int, Box]
DetectionRow = Tuple[int, Box, float]


def load_chosen_observations(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row.get("chosen", "0") or 0) != 1:
                continue
            rows.append(row)
    return rows


def load_output_rows(path: Path) -> Dict[int, List[OutputRow]]:
    by_frame: Dict[int, List[OutputRow]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 6:
                raise ValueError(f"{path}:{line_no}: expected at least 6 columns")
            frame = int(float(parts[0]))
            track_id = int(float(parts[1]))
            x, y, w, h = map(float, parts[2:6])
            by_frame[frame].append((track_id, oracle.xywh_to_xyxy(x, y, w, h)))
    return by_frame


def load_needed_detections(
    dump_path: Path,
    needed_global_ids: set[int],
) -> Tuple[Dict[int, DetectionRow], dict]:
    dump = np.load(dump_path, allow_pickle=True)
    detections = np.asarray(dump["detections"])
    columns = {str(name): index for index, name in enumerate(dump["columns"].tolist())}
    required = {"frame", "global_det_idx", "x1", "y1", "x2", "y2", "score"}
    missing = sorted(required - set(columns))
    if missing:
        raise KeyError(f"dump is missing columns: {missing}")

    found: Dict[int, DetectionRow] = {}
    duplicate_global_ids = 0
    for row in detections:
        global_id = int(row[columns["global_det_idx"]])
        if global_id not in needed_global_ids:
            continue
        if global_id in found:
            duplicate_global_ids += 1
            continue
        found[global_id] = (
            int(row[columns["frame"]]),
            (
                float(row[columns["x1"]]),
                float(row[columns["y1"]]),
                float(row[columns["x2"]]),
                float(row[columns["y2"]]),
            ),
            float(row[columns["score"]]),
        )
    stats = {
        "needed_global_ids": len(needed_global_ids),
        "found_global_ids": len(found),
        "missing_global_ids": len(needed_global_ids - set(found)),
        "duplicate_global_ids": duplicate_global_ids,
    }
    if stats["missing_global_ids"]:
        missing_examples = sorted(needed_global_ids - set(found))[:20]
        raise KeyError(f"missing chosen detection global IDs: {missing_examples}")
    if duplicate_global_ids:
        raise RuntimeError(f"duplicate global_det_idx values found: {duplicate_global_ids}")
    return found, stats


def fixed_iou_frame_mapping(
    items: Sequence[Tuple[int, Box]],
    gt_rows: Sequence[oracle.GTRow],
    *,
    benchmark: str,
    iou_threshold: float,
    preproc_iou_threshold: float,
) -> Tuple[Dict[int, Tuple[int, float]], Dict[int, Tuple[int, float]], set[int]]:
    """Map item ID <-> valid pedestrian GT ID with TrackEval preprocessing."""
    if not items:
        return {}, {}, set()
    item_boxes = [box for _, box in items]
    all_gt_boxes = [row[1] for row in gt_rows]
    distractors = oracle.distractor_class_ids(benchmark)
    preproc_matches = oracle.thresholded_hungarian(
        oracle.iou_matrix(all_gt_boxes, item_boxes),
        preproc_iou_threshold,
    )
    removed_item_indices = {
        item_index
        for gt_index, item_index, _ in preproc_matches
        if gt_rows[gt_index][3] in distractors
    }
    valid_gt = [row for row in gt_rows if row[2] != 0 and row[3] == 1]
    kept_indices = [index for index in range(len(items)) if index not in removed_item_indices]
    matches = oracle.thresholded_hungarian(
        oracle.iou_matrix(
            [item_boxes[index] for index in kept_indices],
            [row[1] for row in valid_gt],
        ),
        iou_threshold,
    )
    item_to_gt: Dict[int, Tuple[int, float]] = {}
    gt_to_item: Dict[int, Tuple[int, float]] = {}
    for local_item_index, gt_index, overlap in matches:
        item_id = int(items[kept_indices[local_item_index]][0])
        gt_id = int(valid_gt[gt_index][0])
        item_to_gt[item_id] = (gt_id, float(overlap))
        gt_to_item[gt_id] = (item_id, float(overlap))
    removed_item_ids = {int(items[index][0]) for index in removed_item_indices}
    return item_to_gt, gt_to_item, removed_item_ids


def build_output_identity_maps(
    output_by_frame: Dict[int, List[OutputRow]],
    gt_by_frame: Dict[int, List[oracle.GTRow]],
    *,
    benchmark: str,
    iou_threshold: float,
    preproc_iou_threshold: float,
) -> Tuple[
    Dict[int, Dict[int, Tuple[int, float]]],
    Dict[int, Dict[int, Tuple[int, float]]],
    Dict[int, List[Tuple[int, int, float]]],
    dict,
]:
    track_to_gt_by_frame: Dict[int, Dict[int, Tuple[int, float]]] = {}
    gt_to_track_by_frame: Dict[int, Dict[int, Tuple[int, float]]] = {}
    gt_history: Dict[int, List[Tuple[int, int, float]]] = defaultdict(list)
    removed_rows = 0
    matched_rows = 0

    for frame in sorted(output_by_frame):
        track_to_gt, gt_to_track, removed = fixed_iou_frame_mapping(
            output_by_frame[frame],
            gt_by_frame.get(frame, []),
            benchmark=benchmark,
            iou_threshold=iou_threshold,
            preproc_iou_threshold=preproc_iou_threshold,
        )
        track_to_gt_by_frame[frame] = track_to_gt
        gt_to_track_by_frame[frame] = gt_to_track
        removed_rows += len(removed)
        matched_rows += len(track_to_gt)
        for gt_id, (track_id, overlap) in gt_to_track.items():
            gt_history[gt_id].append((frame, track_id, overlap))

    stats = {
        "output_frames": len(output_by_frame),
        "output_rows": sum(len(rows) for rows in output_by_frame.values()),
        "output_preproc_removed_rows": removed_rows,
        "output_matched_rows": matched_rows,
    }
    return track_to_gt_by_frame, gt_to_track_by_frame, gt_history, stats


def previous_track_gt(
    history: Sequence[Tuple[int, int, float]],
    frame: int,
) -> Tuple[int | None, int | None, float | None]:
    previous = None
    for event in history:
        if event[0] >= frame:
            break
        previous = event
    if previous is None:
        return None, None, None
    return int(previous[1]), int(previous[0]), float(previous[2])


def previous_gt_tracker(
    history: Sequence[Tuple[int, int, float]],
    frame: int,
) -> Tuple[int | None, int | None, float | None]:
    return previous_track_gt(history, frame)


def future_gt_transition(
    history: Sequence[Tuple[int, int, float]],
    frame: int,
    current_tracker_id: int | None,
    horizon: int,
) -> Tuple[int, int | None, int | None]:
    if current_tracker_id is None:
        return 0, None, None
    for future_frame, tracker_id, _ in history:
        if future_frame <= frame:
            continue
        if future_frame - frame > horizon:
            break
        if int(tracker_id) != int(current_tracker_id):
            return 1, int(future_frame), int(tracker_id)
    return 0, None, None


def _parse_float(row: dict, key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return math.nan


def annotate(args: argparse.Namespace) -> dict:
    observe_rows = load_chosen_observations(Path(args.observe_csv))
    if not observe_rows:
        raise RuntimeError("observe CSV contains no chosen rows")
    needed_global_ids = {int(row["det_global_idx"]) for row in observe_rows}
    detections, detection_load_stats = load_needed_detections(
        Path(args.dump_npz), needed_global_ids
    )
    gt_by_frame = oracle.load_all_gt(Path(args.gt_txt))
    output_by_frame = load_output_rows(Path(args.result_txt))
    (
        track_to_gt_by_frame,
        gt_to_track_by_frame,
        gt_history,
        output_stats,
    ) = build_output_identity_maps(
        output_by_frame,
        gt_by_frame,
        benchmark=args.benchmark,
        iou_threshold=args.iou_threshold,
        preproc_iou_threshold=args.preproc_iou_threshold,
    )

    track_output_history: Dict[int, List[Tuple[int, int, float]]] = defaultdict(list)
    for frame in sorted(track_to_gt_by_frame):
        for track_id, (gt_id, overlap) in track_to_gt_by_frame[frame].items():
            track_output_history[int(track_id)].append((frame, int(gt_id), float(overlap)))

    chosen_detections_by_frame: Dict[int, Dict[int, DetectionRow]] = defaultdict(dict)
    for row in observe_rows:
        global_id = int(row["det_global_idx"])
        frame, box, score = detections[global_id]
        observed_frame = int(row["frame"])
        if frame != observed_frame:
            raise RuntimeError(
                f"det frame mismatch for {global_id}: dump={frame}, observe={observed_frame}"
            )
        chosen_detections_by_frame[frame][global_id] = detections[global_id]

    det_to_gt_by_frame: Dict[int, Dict[int, Tuple[int, float]]] = {}
    removed_dets_by_frame: Dict[int, set[int]] = {}
    for frame, detection_map in chosen_detections_by_frame.items():
        items = [(global_id, value[1]) for global_id, value in detection_map.items()]
        det_to_gt, _, removed = fixed_iou_frame_mapping(
            items,
            gt_by_frame.get(frame, []),
            benchmark=args.benchmark,
            iou_threshold=args.iou_threshold,
            preproc_iou_threshold=args.preproc_iou_threshold,
        )
        det_to_gt_by_frame[frame] = det_to_gt
        removed_dets_by_frame[frame] = removed

    annotated: List[dict] = []
    counts = Counter()
    stage_counts = Counter()
    transition_counts = Counter()

    for row in observe_rows:
        frame = int(row["frame"])
        track_id = int(row["track_id"])
        global_id = int(row["det_global_idx"])
        det_match = det_to_gt_by_frame.get(frame, {}).get(global_id)
        det_gt = int(det_match[0]) if det_match else None
        det_gt_iou = float(det_match[1]) if det_match else None
        det_preproc_removed = global_id in removed_dets_by_frame.get(frame, set())

        prior_track_gt, prior_track_gt_frame, prior_track_gt_iou = previous_track_gt(
            track_output_history.get(track_id, []), frame
        )
        current_output_match = track_to_gt_by_frame.get(frame, {}).get(track_id)
        current_output_gt = int(current_output_match[0]) if current_output_match else None
        current_output_gt_iou = float(current_output_match[1]) if current_output_match else None

        if det_gt is None or prior_track_gt is None:
            track_history_label = "unknown"
        elif det_gt == prior_track_gt:
            track_history_label = "same_identity"
        else:
            track_history_label = "cross_identity"

        gt_current_match = (
            gt_to_track_by_frame.get(frame, {}).get(det_gt) if det_gt is not None else None
        )
        gt_current_tracker = int(gt_current_match[0]) if gt_current_match else None
        gt_current_iou = float(gt_current_match[1]) if gt_current_match else None
        gt_prev_tracker, gt_prev_frame, gt_prev_iou = (
            previous_gt_tracker(gt_history.get(det_gt, []), frame)
            if det_gt is not None
            else (None, None, None)
        )
        if gt_current_tracker is None or gt_prev_tracker is None:
            gt_transition_label = "unknown"
        elif gt_current_tracker == gt_prev_tracker:
            gt_transition_label = "stable"
        else:
            gt_transition_label = "changed"

        chosen_track_is_current_output = int(
            gt_current_tracker is not None and gt_current_tracker == track_id
        )
        future3, future3_frame, future3_tracker = (
            future_gt_transition(
                gt_history.get(det_gt, []),
                frame,
                gt_current_tracker,
                3,
            )
            if det_gt is not None
            else (0, None, None)
        )
        future10, future10_frame, future10_tracker = (
            future_gt_transition(
                gt_history.get(det_gt, []),
                frame,
                gt_current_tracker,
                10,
            )
            if det_gt is not None
            else (0, None, None)
        )

        enriched = dict(row)
        enriched.update(
            {
                "det_gt": det_gt,
                "det_gt_iou": det_gt_iou,
                "det_preproc_removed": int(det_preproc_removed),
                "prior_track_gt": prior_track_gt,
                "prior_track_gt_frame": prior_track_gt_frame,
                "prior_track_gt_iou": prior_track_gt_iou,
                "track_history_label": track_history_label,
                "current_output_gt": current_output_gt,
                "current_output_gt_iou": current_output_gt_iou,
                "gt_current_tracker": gt_current_tracker,
                "gt_current_iou": gt_current_iou,
                "gt_prev_tracker": gt_prev_tracker,
                "gt_prev_frame": gt_prev_frame,
                "gt_prev_iou": gt_prev_iou,
                "gt_transition_label": gt_transition_label,
                "chosen_track_is_current_output": chosen_track_is_current_output,
                "gt_future_transition_3": int(future3),
                "gt_future_transition_3_frame": future3_frame,
                "gt_future_transition_3_tracker": future3_tracker,
                "gt_future_transition_10": int(future10),
                "gt_future_transition_10_frame": future10_frame,
                "gt_future_transition_10_tracker": future10_tracker,
            }
        )
        annotated.append(enriched)
        counts[track_history_label] += 1
        transition_counts[gt_transition_label] += 1
        stage_counts[str(row["stage"])] += 1

    output_path = Path(args.out_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in annotated:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(annotated)

    summary = {
        "seq": args.seq,
        "observe_csv": args.observe_csv,
        "result_txt": args.result_txt,
        "dump_npz": args.dump_npz,
        "gt_txt": args.gt_txt,
        "out_csv": args.out_csv,
        "benchmark": args.benchmark,
        "iou_threshold": float(args.iou_threshold),
        "preproc_iou_threshold": float(args.preproc_iou_threshold),
        "chosen_rows": len(annotated),
        "detection_load": detection_load_stats,
        "output_mapping": output_stats,
        "stage_counts": dict(stage_counts),
        "track_history_labels": dict(counts),
        "gt_transition_labels": dict(transition_counts),
        "det_gt_assigned": sum(row["det_gt"] is not None for row in annotated),
        "det_preproc_removed": sum(int(row["det_preproc_removed"]) for row in annotated),
        "chosen_track_is_current_output": sum(
            int(row["chosen_track_is_current_output"]) for row in annotated
        ),
        "future_gt_transition_3": sum(
            int(row["gt_future_transition_3"]) for row in annotated
        ),
        "future_gt_transition_10": sum(
            int(row["gt_future_transition_10"]) for row in annotated
        ),
    }
    summary["det_gt_assignment_rate"] = summary["det_gt_assigned"] / max(1, len(annotated))
    summary["chosen_output_alignment_rate"] = summary[
        "chosen_track_is_current_output"
    ] / max(1, summary["det_gt_assigned"])
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq", required=True)
    parser.add_argument("--observe-csv", required=True)
    parser.add_argument("--result-txt", required=True)
    parser.add_argument("--dump-npz", required=True)
    parser.add_argument("--gt-txt", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--benchmark", default="MOT20")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--preproc-iou-threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = annotate(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
