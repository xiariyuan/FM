#!/usr/bin/env python3
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - scipy is optional for lightweight smoke environments
    linear_sum_assignment = None


def iou_tlwh(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = max(aw * ah + bw * bh - inter, 1e-12)
    return inter / union


def tlwh_from_row(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (float(row["x"]), float(row["y"]), float(row["w"]), float(row["h"]))


def read_mot_txt(path: str | Path, *, treat_second_col_as_gt: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for raw in reader:
            if len(raw) < 6:
                continue
            frame = int(float(raw[0]))
            entity_id = int(float(raw[1]))
            row = {
                "frame": frame,
                "gt_id" if treat_second_col_as_gt else "track_id": entity_id,
                "x": float(raw[2]),
                "y": float(raw[3]),
                "w": float(raw[4]),
                "h": float(raw[5]),
                "score": float(raw[6]) if len(raw) > 6 else 1.0,
                "raw": raw,
            }
            rows.append(row)
    return rows


def group_by_frame(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["frame"])].append(row)
    return dict(grouped)


def best_gt_match(
    tracker_row: dict[str, Any],
    gt_rows: list[dict[str, Any]],
    *,
    iou_thresh: float,
) -> tuple[int, float, dict[str, Any] | None]:
    """Legacy independent best-GT lookup retained for callers that need a local probe."""
    best_iou = 0.0
    best_gt: dict[str, Any] | None = None
    tracker_box = tlwh_from_row(tracker_row)
    for gt_row in gt_rows:
        score = iou_tlwh(tracker_box, tlwh_from_row(gt_row))
        if score > best_iou:
            best_iou = score
            best_gt = gt_row
    if best_gt is None or best_iou < iou_thresh:
        return -1, best_iou, None
    return int(best_gt["gt_id"]), best_iou, best_gt


def _frame_matches_one_to_one(
    tracker_frame_rows: list[dict[str, Any]],
    gt_frame_rows: list[dict[str, Any]],
    *,
    iou_thresh: float,
) -> dict[int, tuple[dict[str, Any], float]]:
    if not tracker_frame_rows or not gt_frame_rows:
        return {}

    iou_matrix: list[list[float]] = []
    for tracker_row in tracker_frame_rows:
        tracker_box = tlwh_from_row(tracker_row)
        iou_matrix.append([iou_tlwh(tracker_box, tlwh_from_row(gt_row)) for gt_row in gt_frame_rows])

    matches: dict[int, tuple[dict[str, Any], float]] = {}
    if linear_sum_assignment is not None:
        cost_matrix = [[1.0 - score for score in row] for row in iou_matrix]
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        for row_idx, col_idx in zip(row_indices, col_indices):
            score = float(iou_matrix[int(row_idx)][int(col_idx)])
            if score >= float(iou_thresh):
                matches[int(row_idx)] = (gt_frame_rows[int(col_idx)], score)
        return matches

    candidates: list[tuple[float, int, int]] = []
    for row_idx, row in enumerate(iou_matrix):
        for col_idx, score in enumerate(row):
            if float(score) >= float(iou_thresh):
                candidates.append((float(score), row_idx, col_idx))
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    for score, row_idx, col_idx in sorted(candidates, reverse=True):
        if row_idx in used_rows or col_idx in used_cols:
            continue
        matches[row_idx] = (gt_frame_rows[col_idx], score)
        used_rows.add(row_idx)
        used_cols.add(col_idx)
    return matches


def build_alignment(
    tracker_rows: list[dict[str, Any]],
    gt_rows: list[dict[str, Any]],
    *,
    iou_thresh: float,
    dataset: str,
    split: str,
    seq_name: str,
) -> list[dict[str, Any]]:
    gt_by_frame = group_by_frame(gt_rows)
    tracker_by_frame = group_by_frame(tracker_rows)
    aligned: list[dict[str, Any]] = []
    for frame in sorted(tracker_by_frame):
        frame_tracker_rows = tracker_by_frame[frame]
        frame_gt_rows = gt_by_frame.get(frame, [])
        frame_matches = _frame_matches_one_to_one(frame_tracker_rows, frame_gt_rows, iou_thresh=iou_thresh)
        for row_idx, row in enumerate(frame_tracker_rows):
            match = frame_matches.get(row_idx)
            if match is None:
                gt_id, gt_iou, gt_row = -1, 0.0, None
            else:
                gt_row, gt_iou = match
                gt_id = int(gt_row["gt_id"])
            entry = {
                "dataset": dataset,
                "split": split,
                "seq_name": seq_name,
                "frame": frame,
                "track_id": int(row["track_id"]),
                "score": float(row["score"]),
                "x": float(row["x"]),
                "y": float(row["y"]),
                "w": float(row["w"]),
                "h": float(row["h"]),
                "gt_id": int(gt_id),
                "gt_iou": float(gt_iou),
                "is_match": int(gt_id > 0),
                "gt_x": float(gt_row["x"]) if gt_row is not None else "",
                "gt_y": float(gt_row["y"]) if gt_row is not None else "",
                "gt_w": float(gt_row["w"]) if gt_row is not None else "",
                "gt_h": float(gt_row["h"]) if gt_row is not None else "",
            }
            aligned.append(entry)
    return aligned


def alignment_diagnostics(aligned_rows: list[dict[str, Any]], gt_rows: list[dict[str, Any]] | None = None) -> dict[str, int]:
    matched_rows = [row for row in aligned_rows if int(row.get("is_match", 0)) == 1]
    frame_gt_counts: dict[tuple[int, int], int] = defaultdict(int)
    for row in matched_rows:
        frame_gt_counts[(int(row["frame"]), int(row["gt_id"]))] += 1
    duplicate_gt_assignments = sum(max(0, count - 1) for count in frame_gt_counts.values())
    gt_total = len(gt_rows or [])
    return {
        "num_tracker_rows": len(aligned_rows),
        "num_gt_rows": gt_total,
        "num_matched_pairs": len(matched_rows),
        "unmatched_tracker_rows": len(aligned_rows) - len(matched_rows),
        "unmatched_gt_rows": max(0, gt_total - len(matched_rows)) if gt_rows is not None else 0,
        "duplicate_gt_assignments": duplicate_gt_assignments,
    }
