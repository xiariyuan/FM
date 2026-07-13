#!/usr/bin/env python3
"""TrackEval-aligned fixed-IoU current-output ID oracle.

This is an offline diagnostic only. It preserves every tracker output row,
frame, box, score and optional tail column. It changes IDs only:

1. Reproduce MOTChallenge distractor preprocessing at IoU >= 0.5.
2. Match the remaining tracker rows to valid pedestrian GT with Hungarian
   assignment after applying the requested IoU threshold.
3. Give matched rows the GT identity and unmatched/ignored rows a stable offset
   identity derived from the original tracker ID.

The result is a fixed-IoU ID-reassignment evaluation upper bound. It is not a
mathematical HOTA ceiling and must never be used by an online tracker or for a
submission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

Box = Tuple[float, float, float, float]
TrackerRow = Tuple[int, int, float, float, float, float, float, List[str]]
GTRow = Tuple[int, Box, int, int]

_CLASS_NAME_TO_ID = {
    "pedestrian": 1,
    "person_on_vehicle": 2,
    "car": 3,
    "bicycle": 4,
    "motorbike": 5,
    "non_mot_vehicle": 6,
    "static_person": 7,
    "distractor": 8,
    "occluder": 9,
    "occluder_on_ground": 10,
    "occluder_full": 11,
    "reflection": 12,
    "crowd": 13,
}
_DISTRACTOR_NAMES = {
    "person_on_vehicle",
    "static_person",
    "distractor",
    "reflection",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def xywh_to_xyxy(x: float, y: float, w: float, h: float) -> Box:
    return (x, y, x + w, y + h)


def iou_matrix(a: Sequence[Box], b: Sequence[Box]) -> np.ndarray:
    if not a or not b:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    xx1 = np.maximum(aa[:, None, 0], bb[None, :, 0])
    yy1 = np.maximum(aa[:, None, 1], bb[None, :, 1])
    xx2 = np.minimum(aa[:, None, 2], bb[None, :, 2])
    yy2 = np.minimum(aa[:, None, 3], bb[None, :, 3])
    inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
    area_a = np.maximum(0.0, aa[:, 2] - aa[:, 0]) * np.maximum(0.0, aa[:, 3] - aa[:, 1])
    area_b = np.maximum(0.0, bb[:, 2] - bb[:, 0]) * np.maximum(0.0, bb[:, 3] - bb[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, np.maximum(union, 1e-12))


def thresholded_hungarian(
    similarities: np.ndarray,
    threshold: float,
) -> List[Tuple[int, int, float]]:
    """Apply TrackEval's threshold-before-Hungarian rule."""
    if similarities.size == 0:
        return []
    eps = np.finfo(float).eps
    valid = similarities + eps >= float(threshold)
    scores = np.where(valid, similarities, 0.0)
    rows, cols = linear_sum_assignment(-scores)
    return [
        (int(row), int(col), float(similarities[row, col]))
        for row, col in zip(rows.tolist(), cols.tolist())
        if bool(valid[row, col])
    ]


def load_tracker(path: Path) -> Dict[int, List[TrackerRow]]:
    by_frame: Dict[int, List[TrackerRow]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 6:
                raise ValueError(f"{path}:{line_no}: expected at least 6 columns")
            frame = int(float(parts[0]))
            tracker_id = int(float(parts[1]))
            x, y, w, h = map(float, parts[2:6])
            score = float(parts[6]) if len(parts) > 6 else 1.0
            tail = parts[7:] if len(parts) > 7 else []
            by_frame[frame].append((frame, tracker_id, x, y, w, h, score, tail))
    return by_frame


def load_all_gt(path: Path) -> Dict[int, List[GTRow]]:
    by_frame: Dict[int, List[GTRow]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 6:
                raise ValueError(f"{path}:{line_no}: expected at least 6 columns")
            frame = int(float(parts[0]))
            gt_id = int(float(parts[1]))
            x, y, w, h = map(float, parts[2:6])
            marked = int(float(parts[6])) if len(parts) > 6 else 1
            cls = int(float(parts[7])) if len(parts) > 7 else 1
            by_frame[frame].append((gt_id, xywh_to_xyxy(x, y, w, h), marked, cls))
    return by_frame


def distractor_class_ids(benchmark: str) -> set[int]:
    names = set(_DISTRACTOR_NAMES)
    if benchmark.upper() == "MOT20":
        names.add("non_mot_vehicle")
    return {_CLASS_NAME_TO_ID[name] for name in names}


def build_oracle(
    tracker_by_frame: Dict[int, List[TrackerRow]],
    gt_by_frame: Dict[int, List[GTRow]],
    *,
    benchmark: str,
    iou_threshold: float,
    preproc_iou_threshold: float,
    unmatched_offset: int,
) -> Tuple[List[str], dict]:
    distractors = distractor_class_ids(benchmark)
    output_lines: List[str] = []
    match_ious: List[float] = []
    original_ids: set[int] = set()
    oracle_ids: set[int] = set()
    duplicate_id_frames = 0
    stats = {
        "frames_with_tracker_outputs": 0,
        "tracker_rows": 0,
        "preproc_removed_tracker_rows": 0,
        "evaluation_tracker_rows": 0,
        "matched_to_valid_pedestrian_gt": 0,
        "unmatched_evaluation_rows": 0,
    }

    for frame in sorted(tracker_by_frame):
        stats["frames_with_tracker_outputs"] += 1
        tracker_rows = tracker_by_frame[frame]
        gt_rows = gt_by_frame.get(frame, [])
        tracker_boxes = [xywh_to_xyxy(row[2], row[3], row[4], row[5]) for row in tracker_rows]
        all_gt_boxes = [row[1] for row in gt_rows]

        preproc_matches = thresholded_hungarian(
            iou_matrix(all_gt_boxes, tracker_boxes),
            preproc_iou_threshold,
        )
        removed_tracker_indices = {
            tracker_index
            for gt_index, tracker_index, _ in preproc_matches
            if gt_rows[gt_index][3] in distractors
        }

        valid_gt = [
            row
            for row in gt_rows
            if row[2] != 0 and row[3] == _CLASS_NAME_TO_ID["pedestrian"]
        ]
        kept_tracker_indices = [
            index for index in range(len(tracker_rows)) if index not in removed_tracker_indices
        ]
        kept_tracker_boxes = [tracker_boxes[index] for index in kept_tracker_indices]
        eval_matches = thresholded_hungarian(
            iou_matrix(kept_tracker_boxes, [row[1] for row in valid_gt]),
            iou_threshold,
        )
        gt_id_by_tracker_index = {
            kept_tracker_indices[tracker_local_index]: int(valid_gt[gt_index][0])
            for tracker_local_index, gt_index, _ in eval_matches
        }
        match_ious.extend(overlap for _, _, overlap in eval_matches)

        stats["tracker_rows"] += len(tracker_rows)
        stats["preproc_removed_tracker_rows"] += len(removed_tracker_indices)
        stats["evaluation_tracker_rows"] += len(kept_tracker_indices)
        stats["matched_to_valid_pedestrian_gt"] += len(eval_matches)
        stats["unmatched_evaluation_rows"] += len(kept_tracker_indices) - len(eval_matches)

        frame_oracle_ids: List[int] = []
        for index, row in enumerate(tracker_rows):
            frame_id, original_id, x, y, w, h, score, tail = row
            original_ids.add(original_id)
            oracle_id = int(gt_id_by_tracker_index.get(index, unmatched_offset + original_id))
            oracle_ids.add(oracle_id)
            frame_oracle_ids.append(oracle_id)
            fields = [
                str(frame_id),
                str(oracle_id),
                f"{x:.6f}",
                f"{y:.6f}",
                f"{w:.6f}",
                f"{h:.6f}",
                f"{score:.6f}",
            ]
            fields.extend(tail if tail else ["-1", "-1", "-1"])
            output_lines.append(",".join(fields))

        if len(frame_oracle_ids) != len(set(frame_oracle_ids)):
            duplicate_id_frames += 1

    if duplicate_id_frames:
        raise RuntimeError(f"oracle output has duplicate IDs in {duplicate_id_frames} frame(s)")

    ious = np.asarray(match_ious, dtype=np.float64)
    evaluation_rows = int(stats["evaluation_tracker_rows"])
    matched_rows = int(stats["matched_to_valid_pedestrian_gt"])
    stats.update(
        {
            "benchmark": benchmark.upper(),
            "iou_threshold": float(iou_threshold),
            "preproc_iou_threshold": float(preproc_iou_threshold),
            "assignment_mode": "trackeval_preproc_then_zero_invalid_hungarian",
            "match_rate_evaluation_rows": matched_rows / max(1, evaluation_rows),
            "match_iou_mean": float(np.mean(ious)) if ious.size else None,
            "match_iou_min": float(np.min(ious)) if ious.size else None,
            "match_iou_p10": float(np.quantile(ious, 0.10)) if ious.size else None,
            "match_iou_median": float(np.median(ious)) if ious.size else None,
            "duplicate_id_frames": duplicate_id_frames,
            "original_unique_ids": len(original_ids),
            "oracle_unique_ids": len(oracle_ids),
            "unmatched_id_offset": int(unmatched_offset),
        }
    )
    return output_lines, stats


def verify_non_id_consistency(source_path: Path, oracle_path: Path) -> dict:
    source_lines = [line for line in source_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    oracle_lines = [line for line in oracle_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = {
        "source_rows": len(source_lines),
        "oracle_rows": len(oracle_lines),
        "row_count_equal": len(source_lines) == len(oracle_lines),
        "frame_mismatch": 0,
        "box_mismatch": 0,
        "score_mismatch": 0,
        "tail_mismatch": 0,
        "same_id_rows": 0,
        "changed_id_rows": 0,
    }
    if len(source_lines) != len(oracle_lines):
        raise RuntimeError(f"row count mismatch: {len(source_lines)} vs {len(oracle_lines)}")

    for source_line, oracle_line in zip(source_lines, oracle_lines):
        source = source_line.split(",")
        oracle = oracle_line.split(",")
        if int(float(source[0])) != int(float(oracle[0])):
            result["frame_mismatch"] += 1
        if any(float(source[index]) != float(oracle[index]) for index in range(2, 6)):
            result["box_mismatch"] += 1
        source_score = float(source[6]) if len(source) > 6 else 1.0
        oracle_score = float(oracle[6]) if len(oracle) > 6 else 1.0
        if source_score != oracle_score:
            result["score_mismatch"] += 1
        source_tail = source[7:] if len(source) > 7 else ["-1", "-1", "-1"]
        oracle_tail = oracle[7:] if len(oracle) > 7 else ["-1", "-1", "-1"]
        if source_tail != oracle_tail:
            result["tail_mismatch"] += 1
        if int(float(source[1])) == int(float(oracle[1])):
            result["same_id_rows"] += 1
        else:
            result["changed_id_rows"] += 1

    result["non_id_mismatch_total"] = sum(
        int(result[key])
        for key in ("frame_mismatch", "box_mismatch", "score_mismatch", "tail_mismatch")
    )
    if result["non_id_mismatch_total"]:
        raise RuntimeError(f"non-ID content changed: {result}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-txt", required=True)
    parser.add_argument("--gt-txt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--consistency-json", default="")
    parser.add_argument("--benchmark", default="MOT20")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--preproc-iou-threshold", type=float, default=0.5)
    parser.add_argument("--unmatched-id-offset", type=int, default=1_000_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = Path(args.result_txt)
    gt_path = Path(args.gt_txt)
    output_path = Path(args.out)
    summary_path = Path(args.summary_json)
    consistency_path = (
        Path(args.consistency_json)
        if args.consistency_json
        else output_path.parent.parent / "geometry_consistency.json"
    )

    if not 0.0 <= float(args.iou_threshold) <= 1.0:
        raise ValueError("--iou-threshold must be in [0, 1]")
    if not 0.0 <= float(args.preproc_iou_threshold) <= 1.0:
        raise ValueError("--preproc-iou-threshold must be in [0, 1]")

    lines, stats = build_oracle(
        load_tracker(source_path),
        load_all_gt(gt_path),
        benchmark=str(args.benchmark),
        iou_threshold=float(args.iou_threshold),
        preproc_iou_threshold=float(args.preproc_iou_threshold),
        unmatched_offset=int(args.unmatched_id_offset),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    consistency = verify_non_id_consistency(source_path, output_path)
    consistency_path.parent.mkdir(parents=True, exist_ok=True)
    consistency_path.write_text(
        json.dumps(consistency, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    stats.update(
        {
            "result_txt": str(source_path),
            "gt_txt": str(gt_path),
            "out": str(output_path),
            "source_sha256": _sha256(source_path),
            "oracle_sha256": _sha256(output_path),
            "consistency_json": str(consistency_path),
            "consistency": consistency,
            "diagnostic_only": True,
            "is_mathematical_hota_ceiling": False,
        }
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
