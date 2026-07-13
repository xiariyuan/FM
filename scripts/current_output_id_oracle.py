#!/usr/bin/env python3
"""Current-output ID oracle for MOTChallenge results.

Keeps every tracker output box, score and frame unchanged. For each frame, it
Hungarian-matches tracker boxes to valid pedestrian GT boxes by IoU and replaces
the IDs of matched outputs with their GT IDs. Unmatched outputs receive stable,
offset IDs derived from their original tracker IDs.

This is an offline diagnostic upper bound only. It must never be used by the
online tracker or for test-set submission.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except Exception as exc:  # pragma: no cover
    linear_sum_assignment = None
    _SCIPY_IMPORT_ERROR = exc
else:
    _SCIPY_IMPORT_ERROR = None


Box = Tuple[float, float, float, float]
TrackerRow = Tuple[int, int, float, float, float, float, float, List[str]]


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


def load_tracker(path: Path) -> Dict[int, List[TrackerRow]]:
    by_frame: Dict[int, List[TrackerRow]] = defaultdict(list)
    with path.open('r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 6:
                raise ValueError(f'{path}:{line_no}: expected at least 6 columns')
            frame = int(float(parts[0]))
            tid = int(float(parts[1]))
            x, y, w, h = map(float, parts[2:6])
            score = float(parts[6]) if len(parts) > 6 else 1.0
            tail = parts[7:] if len(parts) > 7 else []
            by_frame[frame].append((frame, tid, x, y, w, h, score, tail))
    return by_frame


def load_valid_pedestrian_gt(path: Path) -> Dict[int, List[Tuple[int, Box]]]:
    """Use the final evaluation target GT: zero_marked != 0 and class == 1."""
    by_frame: Dict[int, List[Tuple[int, Box]]] = defaultdict(list)
    with path.open('r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 6:
                raise ValueError(f'{path}:{line_no}: expected at least 6 columns')
            frame = int(float(parts[0]))
            gid = int(float(parts[1]))
            x, y, w, h = map(float, parts[2:6])
            mark = int(float(parts[6])) if len(parts) > 6 else 1
            cls = int(float(parts[7])) if len(parts) > 7 else 1
            if mark != 0 and cls == 1:
                by_frame[frame].append((gid, xywh_to_xyxy(x, y, w, h)))
    return by_frame


def hungarian_matches(ious: np.ndarray, threshold: float) -> List[Tuple[int, int, float]]:
    if ious.size == 0:
        return []
    if linear_sum_assignment is None:
        raise RuntimeError(f'scipy.optimize.linear_sum_assignment unavailable: {_SCIPY_IMPORT_ERROR}')
    # The assignment is global. Invalid pairs are allowed in the raw assignment
    # but removed afterward, exactly as TrackEval thresholds matched pairs.
    rows, cols = linear_sum_assignment(-ious)
    out: List[Tuple[int, int, float]] = []
    for r, c in zip(rows.tolist(), cols.tolist()):
        ov = float(ious[r, c])
        if ov + np.finfo(float).eps >= threshold:
            out.append((r, c, ov))
    return out


def make_oracle(
    tracker_by_frame: Dict[int, List[TrackerRow]],
    gt_by_frame: Dict[int, List[Tuple[int, Box]]],
    iou_threshold: float,
    unmatched_offset: int,
) -> Tuple[List[str], dict]:
    output_lines: List[str] = []
    matched = 0
    unmatched = 0
    frames = 0
    match_ious: List[float] = []
    duplicate_id_frames = 0
    original_ids = set()
    oracle_ids = set()

    for frame in sorted(tracker_by_frame):
        frames += 1
        tracker_rows = tracker_by_frame[frame]
        gt_rows = gt_by_frame.get(frame, [])
        tracker_boxes = [xywh_to_xyxy(r[2], r[3], r[4], r[5]) for r in tracker_rows]
        gt_boxes = [r[1] for r in gt_rows]
        assignments = hungarian_matches(iou_matrix(tracker_boxes, gt_boxes), iou_threshold)
        assigned_gt_by_tracker = {tr: gt_rows[gc][0] for tr, gc, _ in assignments}
        matched += len(assignments)
        unmatched += len(tracker_rows) - len(assignments)
        match_ious.extend(ov for _, _, ov in assignments)

        frame_ids: List[int] = []
        for idx, row in enumerate(tracker_rows):
            frame_id, original_id, x, y, w, h, score, tail = row
            original_ids.add(original_id)
            if idx in assigned_gt_by_tracker:
                oracle_id = int(assigned_gt_by_tracker[idx])
            else:
                oracle_id = int(unmatched_offset + original_id)
            oracle_ids.add(oracle_id)
            frame_ids.append(oracle_id)
            # Standard MOT output. Preserve optional tail columns if present.
            fields = [
                str(frame_id), str(oracle_id),
                f'{x:.6f}', f'{y:.6f}', f'{w:.6f}', f'{h:.6f}', f'{score:.6f}'
            ]
            fields.extend(tail if tail else ['-1', '-1', '-1'])
            output_lines.append(','.join(fields))
        if len(frame_ids) != len(set(frame_ids)):
            duplicate_id_frames += 1

    stats = {
        'frames_with_tracker_outputs': frames,
        'tracker_rows': matched + unmatched,
        'matched_to_valid_pedestrian_gt': matched,
        'unmatched_tracker_rows': unmatched,
        'match_rate': matched / max(1, matched + unmatched),
        'iou_threshold': iou_threshold,
        'match_iou_mean': float(np.mean(match_ious)) if match_ious else None,
        'match_iou_min': float(np.min(match_ious)) if match_ious else None,
        'match_iou_p10': float(np.quantile(match_ious, 0.10)) if match_ious else None,
        'match_iou_median': float(np.median(match_ious)) if match_ious else None,
        'duplicate_id_frames': duplicate_id_frames,
        'original_unique_ids': len(original_ids),
        'oracle_unique_ids': len(oracle_ids),
        'unmatched_id_offset': unmatched_offset,
    }
    if duplicate_id_frames:
        raise RuntimeError(f'oracle output has duplicate IDs in {duplicate_id_frames} frame(s)')
    return output_lines, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--result-txt', required=True)
    ap.add_argument('--gt-txt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--summary-json', default='')
    ap.add_argument('--iou-threshold', type=float, default=0.5)
    ap.add_argument('--unmatched-id-offset', type=int, default=1_000_000)
    args = ap.parse_args()

    result_path = Path(args.result_txt)
    gt_path = Path(args.gt_txt)
    out_path = Path(args.out)
    tracker = load_tracker(result_path)
    gt = load_valid_pedestrian_gt(gt_path)
    lines, stats = make_oracle(
        tracker,
        gt,
        iou_threshold=float(args.iou_threshold),
        unmatched_offset=int(args.unmatched_id_offset),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')
    stats.update({
        'result_txt': str(result_path),
        'gt_txt': str(gt_path),
        'out': str(out_path),
    })
    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
