#!/usr/bin/env python3
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any


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
    aligned: list[dict[str, Any]] = []
    for row in tracker_rows:
        frame = int(row["frame"])
        gt_id, gt_iou, gt_row = best_gt_match(row, gt_by_frame.get(frame, []), iou_thresh=iou_thresh)
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
