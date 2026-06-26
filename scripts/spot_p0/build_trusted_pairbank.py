#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.spot_common.io_utils import ensure_dir, write_json, write_manifest, write_single_row_csv
from scripts.spot_common.mot_format import iou_tlwh, read_mot_txt

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover
    linear_sum_assignment = None

SUMMARY_FIELDS = [
    "status",
    "error",
    "dataset",
    "split",
    "seq_name",
    "input_rows",
    "output_rows",
    "frames",
    "det_gt_labeled_rows",
    "track_gt_labeled_rows",
    "positive_rows",
    "label_source",
    "trusted",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate a SPOT/RGSA pairbank with one-to-one GT ids.")
    parser.add_argument("--pair-csv", required=True)
    parser.add_argument("--gt-txt", required=True)
    parser.add_argument("--dataset", default="unknown")
    parser.add_argument("--split", default="unknown")
    parser.add_argument("--seq-name", default="unknown_seq")
    parser.add_argument("--out-dir", default="outputs/spot_trusted_pairbank")
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument("--max-frame", type=int, default=0, help="Optional cap on frame_id; 0 means no frame cap.")
    return parser.parse_args()


def _float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _parse_tlwh(raw: str) -> tuple[float, float, float, float] | None:
    if not raw:
        return None
    parts = [part.strip() for part in str(raw).split(",")]
    if len(parts) < 4:
        return None
    try:
        return (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]))
    except (TypeError, ValueError):
        return None


def _gt_box(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (float(row["x"]), float(row["y"]), float(row["w"]), float(row["h"]))


def _one_to_one_box_gt(
    boxes: list[tuple[str, tuple[float, float, float, float]]],
    gt_rows: list[dict[str, Any]],
    *,
    iou_thresh: float,
) -> dict[str, tuple[int, float]]:
    if not boxes or not gt_rows:
        return {}
    ious = [[iou_tlwh(box, _gt_box(gt)) for gt in gt_rows] for _, box in boxes]
    out: dict[str, tuple[int, float]] = {}
    if linear_sum_assignment is not None:
        costs = [[1.0 - score for score in row] for row in ious]
        row_idx, col_idx = linear_sum_assignment(costs)
        for r, c in zip(row_idx, col_idx):
            score = float(ious[int(r)][int(c)])
            if score >= iou_thresh:
                out[boxes[int(r)][0]] = (int(gt_rows[int(c)]["gt_id"]), score)
        return out
    candidates: list[tuple[float, int, int]] = []
    for r, row in enumerate(ious):
        for c, score in enumerate(row):
            if score >= iou_thresh:
                candidates.append((float(score), r, c))
    used_r: set[int] = set()
    used_c: set[int] = set()
    for score, r, c in sorted(candidates, reverse=True):
        if r in used_r or c in used_c:
            continue
        out[boxes[r][0]] = (int(gt_rows[c]["gt_id"]), score)
        used_r.add(r)
        used_c.add(c)
    return out


def _iter_frames(pair_csv: Path):
    with pair_csv.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        current_frame: int | None = None
        rows: list[dict[str, Any]] = []
        for row in reader:
            frame = int(float(row.get("frame_id") or row.get("frame") or 0))
            if rows and frame != current_frame:
                yield current_frame, fieldnames, rows
                rows = []
            current_frame = frame
            rows.append(row)
        if rows:
            yield current_frame, fieldnames, rows


def main() -> int:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    summary_csv = out_dir / "summary.csv"
    out_csv = out_dir / "trusted_pairbank.csv"
    summary = {
        "status": "running",
        "error": "",
        "dataset": args.dataset,
        "split": args.split,
        "seq_name": args.seq_name,
        "input_rows": 0,
        "output_rows": 0,
        "frames": 0,
        "det_gt_labeled_rows": 0,
        "track_gt_labeled_rows": 0,
        "positive_rows": 0,
        "label_source": "gt_one_to_one_iou",
        "trusted": 0,
    }
    write_single_row_csv(summary_csv, summary, SUMMARY_FIELDS)

    try:
        gt_rows = read_mot_txt(args.gt_txt, treat_second_col_as_gt=True)
        gt_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in gt_rows:
            gt_by_frame[int(row["frame"])].append(row)

        writer = None
        out_handle = None
        input_rows = output_rows = frame_count = 0
        det_labeled = track_labeled = positive_rows = 0
        extra_fields = ["det_gt_id", "track_gt_id", "det_gt_iou", "track_gt_iou", "is_positive"]

        try:
            for frame_id, fieldnames, frame_rows in _iter_frames(Path(args.pair_csv).expanduser()):
                if args.max_frame and int(frame_id) > int(args.max_frame):
                    break
                frame_count += 1
                gt_frame = gt_by_frame.get(int(frame_id), [])
                det_boxes: dict[str, tuple[float, float, float, float]] = {}
                track_boxes: dict[str, tuple[float, float, float, float]] = {}
                for row in frame_rows:
                    det_id = str(row.get("det_id") or row.get("det_index") or "")
                    det_box = _parse_tlwh(str(row.get("det_tlwh", "")))
                    if det_id and det_box is not None:
                        det_boxes.setdefault(det_id, det_box)
                    track_id = str(row.get("track_id") or "")
                    track_box = _parse_tlwh(str(row.get("track_tlwh", "")))
                    if track_id and track_box is not None:
                        track_boxes.setdefault(track_id, track_box)
                det_map = _one_to_one_box_gt(list(det_boxes.items()), gt_frame, iou_thresh=float(args.iou_thresh))
                track_map = _one_to_one_box_gt(list(track_boxes.items()), gt_frame, iou_thresh=float(args.iou_thresh))
                if writer is None:
                    ordered = list(fieldnames) + [f for f in extra_fields if f not in fieldnames]
                    out_handle = out_csv.open("w", encoding="utf-8", newline="")
                    writer = csv.DictWriter(out_handle, fieldnames=ordered)
                    writer.writeheader()
                for row in frame_rows:
                    input_rows += 1
                    det_id = str(row.get("det_id") or row.get("det_index") or "")
                    track_id = str(row.get("track_id") or "")
                    det_gt, det_iou = det_map.get(det_id, (-1, 0.0))
                    track_gt, track_iou = track_map.get(track_id, (-1, 0.0))
                    is_positive = int(det_gt > 0 and track_gt > 0 and det_gt == track_gt)
                    row.update(
                        {
                            "det_gt_id": det_gt,
                            "track_gt_id": track_gt,
                            "det_gt_iou": round(float(det_iou), 6),
                            "track_gt_iou": round(float(track_iou), 6),
                            "is_positive": is_positive,
                        }
                    )
                    if det_gt > 0:
                        det_labeled += 1
                    if track_gt > 0:
                        track_labeled += 1
                    if is_positive:
                        positive_rows += 1
                    writer.writerow(row)
                    output_rows += 1
        finally:
            if out_handle is not None:
                out_handle.close()

        trusted = int(output_rows > 0 and det_labeled > 0 and track_labeled > 0)
        summary.update(
            {
                "status": "completed",
                "input_rows": input_rows,
                "output_rows": output_rows,
                "frames": frame_count,
                "det_gt_labeled_rows": det_labeled,
                "track_gt_labeled_rows": track_labeled,
                "positive_rows": positive_rows,
                "trusted": trusted,
            }
        )
        write_single_row_csv(summary_csv, summary, SUMMARY_FIELDS)
        write_json(summary, out_dir / "trusted_pairbank_metrics.json")
        write_manifest(
            out_dir,
            phase="spot_trusted_pairbank",
            script=str(Path(__file__).resolve().relative_to(REPO_ROOT)),
            args=vars(args),
            status="ok",
            metrics=summary,
            artifacts={"summary_csv": str(summary_csv), "trusted_pairbank": str(out_csv)},
            notes="one-to-one GT-id trusted pairbank",
        )
        return 0
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = str(exc)
        write_single_row_csv(summary_csv, summary, SUMMARY_FIELDS)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
