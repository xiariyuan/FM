#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


REPO_ROOT = Path("/gemini/code/FMtrack-main/FM-Track")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze trajectory-level context for re-entry failure cases.")
    parser.add_argument("--cases-csv", required=True)
    parser.add_argument("--gt-annotations-json", required=True)
    parser.add_argument("--baseline-results-dir", required=True)
    parser.add_argument("--engine-results-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--context-before", type=int, default=10)
    parser.add_argument("--context-after", type=int, default=10)
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
            tlx = float(row[2])
            tly = float(row[3])
            w = float(row[4])
            h = float(row[5])
            score = float(row[6]) if len(row) > 6 else 1.0
            rows_by_frame[frame].append(
                {
                    "id": ident,
                    "tlx": tlx,
                    "tly": tly,
                    "w": w,
                    "h": h,
                    "score": score,
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


def get_gt_row(gt_rows: Dict[int, List[dict]], frame: int, gt_id: str) -> dict | None:
    for row in gt_rows.get(frame, []):
        if str(int(row["id"])) == str(gt_id):
            return row
    return None


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = [row for row in csv.DictReader(Path(args.cases_csv).open("r", encoding="utf-8")) if row.get("baseline_correct_engine_not") == "1"]
    gt_by_seq = parse_coco_track_rows(Path(args.gt_annotations_json).expanduser().resolve())

    event_rows: List[dict] = []
    frame_rows: List[dict] = []
    sequence_summary: Dict[str, int] = defaultdict(int)

    baseline_dir = Path(args.baseline_results_dir).expanduser().resolve()
    engine_dir = Path(args.engine_results_dir).expanduser().resolve()

    tracker_cache: Dict[Tuple[str, str], Dict[int, List[dict]]] = {}

    for case_idx, case in enumerate(cases, start=1):
        seq_name = case["seq_name"]
        gt_id = case["gt_id"]
        exit_frame = int(case["exit_frame"])
        reentry_frame = int(case["reentry_frame"])
        window_start = max(1, exit_frame - int(args.context_before))
        window_end = reentry_frame + int(args.context_after)
        sequence_summary[seq_name] += 1

        for tag, root in [("baseline", baseline_dir), ("engine", engine_dir)]:
            key = (tag, seq_name)
            if key not in tracker_cache:
                tracker_cache[key] = parse_mot_rows(root / f"{seq_name}.txt")

        gt_rows = gt_by_seq[seq_name]
        base_rows = tracker_cache[("baseline", seq_name)]
        engine_rows = tracker_cache[("engine", seq_name)]

        event_rows.append(
            {
                "case_id": case_idx,
                "seq_name": seq_name,
                "gt_id": gt_id,
                "exit_frame": exit_frame,
                "reentry_frame": reentry_frame,
                "gap": case["gap"],
                "bucket": case["bucket"],
                "baseline_tid": case["baseline_tid"],
                "baseline_oracle": case["baseline_oracle"],
                "engine_tid": case["engine_tid"],
                "engine_oracle": case["engine_oracle"],
                "window_start": window_start,
                "window_end": window_end,
            }
        )

        for frame in range(window_start, window_end + 1):
            gt_row = get_gt_row(gt_rows, frame, gt_id)
            base_tid, base_iou = best_tracker_match(gt_row, base_rows.get(frame, []))
            engine_tid, engine_iou = best_tracker_match(gt_row, engine_rows.get(frame, []))
            frame_rows.append(
                {
                    "case_id": case_idx,
                    "seq_name": seq_name,
                    "gt_id": gt_id,
                    "frame": frame,
                    "is_exit_frame": int(frame == exit_frame),
                    "is_reentry_frame": int(frame == reentry_frame),
                    "gt_present": int(gt_row is not None),
                    "baseline_best_tid": base_tid,
                    "baseline_best_iou": round(base_iou, 6),
                    "engine_best_tid": engine_tid,
                    "engine_best_iou": round(engine_iou, 6),
                    "baseline_track_count": len(base_rows.get(frame, [])),
                    "engine_track_count": len(engine_rows.get(frame, [])),
                }
            )

    with (out_dir / "failure_cases.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(event_rows[0].keys()) if event_rows else ["case_id"])
        writer.writeheader()
        writer.writerows(event_rows)

    with (out_dir / "failure_case_frame_context.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(frame_rows[0].keys()) if frame_rows else ["case_id"])
        writer.writeheader()
        writer.writerows(frame_rows)

    with (out_dir / "failure_case_sequence_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seq_name", "case_count"])
        writer.writeheader()
        writer.writerows([{"seq_name": seq, "case_count": count} for seq, count in sorted(sequence_summary.items())])

    with (out_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_count", "sequence_count", "context_before", "context_after"])
        writer.writeheader()
        writer.writerow(
            {
                "case_count": len(event_rows),
                "sequence_count": len(sequence_summary),
                "context_before": int(args.context_before),
                "context_after": int(args.context_after),
            }
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
