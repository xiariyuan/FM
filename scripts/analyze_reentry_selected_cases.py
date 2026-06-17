#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze selected re-entry cases with frame-level tracker context.")
    parser.add_argument("--cases-csv", required=True, help="CSV containing at least seq_name, gt_id, exit_frame, reentry_frame, gap, bucket.")
    parser.add_argument("--gt-annotations-json", required=True)
    parser.add_argument("--baseline-results-dir", required=True)
    parser.add_argument("--compare-results-dir", required=True)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--compare-label", default="compare")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--context-before", type=int, default=15)
    parser.add_argument("--context-after", type=int, default=15)
    return parser.parse_args()


def parse_mot_rows(path: Path) -> Dict[int, List[dict]]:
    rows_by_frame: Dict[int, List[dict]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 6:
                continue
            frame = int(float(row[0]))
            ident = int(float(row[1]))
            rows_by_frame[frame].append(
                {
                    "id": ident,
                    "tlx": float(row[2]),
                    "tly": float(row[3]),
                    "w": float(row[4]),
                    "h": float(row[5]),
                    "score": float(row[6]) if len(row) > 6 else 1.0,
                }
            )
    return rows_by_frame


def parse_coco_track_rows(path: Path) -> Dict[str, Dict[int, List[dict]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    images_by_id = {int(img["id"]): img for img in data.get("images", [])}
    seq_rows: Dict[str, Dict[int, List[dict]]] = defaultdict(lambda: defaultdict(list))
    for ann in data.get("annotations", []):
        image = images_by_id.get(int(ann["image_id"]))
        if image is None:
            continue
        file_name = str(image.get("file_name", ""))
        seq_name = file_name.split("/", 1)[0]
        frame = int(image.get("frame_id", 0))
        bbox = ann.get("bbox", [0, 0, 0, 0])
        seq_rows[seq_name][frame].append(
            {
                "id": int(ann["track_id"]),
                "tlx": float(bbox[0]),
                "tly": float(bbox[1]),
                "w": float(bbox[2]),
                "h": float(bbox[3]),
            }
        )
    return seq_rows


def tlwh_to_tlbr(row: dict) -> Tuple[float, float, float, float]:
    return (
        float(row["tlx"]),
        float(row["tly"]),
        float(row["tlx"]) + float(row["w"]),
        float(row["tly"]) + float(row["h"]),
    )


def iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def get_gt_row(gt_rows: Dict[int, List[dict]], frame: int, gt_id: int) -> dict | None:
    for row in gt_rows.get(frame, []):
        if int(row["id"]) == int(gt_id):
            return row
    return None


def best_tracker_match(gt_row: dict | None, tracker_rows: List[dict]) -> Tuple[str, float]:
    if gt_row is None:
        return "", 0.0
    gt_box = tlwh_to_tlbr(gt_row)
    best_tid = ""
    best_iou = 0.0
    for row in tracker_rows:
        score = iou(gt_box, tlwh_to_tlbr(row))
        if score > best_iou:
            best_iou = score
            best_tid = str(int(row["id"]))
    return best_tid, best_iou


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = list(csv.DictReader(Path(args.cases_csv).open("r", encoding="utf-8")))
    gt_by_seq = parse_coco_track_rows(Path(args.gt_annotations_json).expanduser().resolve())
    baseline_dir = Path(args.baseline_results_dir).expanduser().resolve()
    compare_dir = Path(args.compare_results_dir).expanduser().resolve()

    tracker_cache: Dict[Tuple[str, str], Dict[int, List[dict]]] = {}
    event_rows: List[dict] = []
    frame_rows: List[dict] = []

    for case_idx, case in enumerate(cases, start=1):
        seq_name = case["seq_name"]
        gt_id = int(case["gt_id"])
        exit_frame = int(case["exit_frame"])
        reentry_frame = int(case["reentry_frame"])
        window_start = max(1, exit_frame - int(args.context_before))
        window_end = reentry_frame + int(args.context_after)

        for tag, root in [(args.baseline_label, baseline_dir), (args.compare_label, compare_dir)]:
            key = (tag, seq_name)
            if key not in tracker_cache:
                tracker_cache[key] = parse_mot_rows(root / f"{seq_name}.txt")

        gt_rows = gt_by_seq[seq_name]
        base_rows = tracker_cache[(args.baseline_label, seq_name)]
        comp_rows = tracker_cache[(args.compare_label, seq_name)]

        event_rows.append(
            {
                "case_id": case_idx,
                "seq_name": seq_name,
                "gt_id": gt_id,
                "exit_frame": exit_frame,
                "reentry_frame": reentry_frame,
                "gap": case["gap"],
                "bucket": case["bucket"],
                "window_start": window_start,
                "window_end": window_end,
            }
        )

        for frame in range(window_start, window_end + 1):
            gt_row = get_gt_row(gt_rows, frame, gt_id)
            base_tid, base_iou = best_tracker_match(gt_row, base_rows.get(frame, []))
            comp_tid, comp_iou = best_tracker_match(gt_row, comp_rows.get(frame, []))
            frame_rows.append(
                {
                    "case_id": case_idx,
                    "seq_name": seq_name,
                    "gt_id": gt_id,
                    "frame": frame,
                    "is_exit_frame": int(frame == exit_frame),
                    "is_reentry_frame": int(frame == reentry_frame),
                    "gt_present": int(gt_row is not None),
                    f"{args.baseline_label}_best_tid": base_tid,
                    f"{args.baseline_label}_best_iou": round(base_iou, 6),
                    f"{args.compare_label}_best_tid": comp_tid,
                    f"{args.compare_label}_best_iou": round(comp_iou, 6),
                    f"{args.baseline_label}_track_count": len(base_rows.get(frame, [])),
                    f"{args.compare_label}_track_count": len(comp_rows.get(frame, [])),
                }
            )

    with (out_dir / "selected_cases.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(event_rows[0].keys()) if event_rows else ["case_id"])
        writer.writeheader()
        writer.writerows(event_rows)

    with (out_dir / "selected_case_frame_context.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(frame_rows[0].keys()) if frame_rows else ["case_id"])
        writer.writeheader()
        writer.writerows(frame_rows)

    with (out_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_count", "context_before", "context_after", "baseline_label", "compare_label"])
        writer.writeheader()
        writer.writerow(
            {
                "case_count": len(event_rows),
                "context_before": int(args.context_before),
                "context_after": int(args.context_after),
                "baseline_label": args.baseline_label,
                "compare_label": args.compare_label,
            }
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
