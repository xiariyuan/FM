#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


REPO_ROOT = Path("/gemini/code/FMtrack-main/FM-Track")
REGISTRY_CSV = REPO_ROOT / "outputs" / "experiment_registry.csv"

SUMMARY_FIELDS = [
    "status",
    "error",
    "case_count",
    "flip_type_count",
    "context_before",
    "iou_thresh",
    "exit_same_count",
    "exit_different_count",
    "exit_baseline_missing_count",
    "exit_compare_missing_count",
    "exit_both_missing_count",
    "pre_exit_any_diverge_count",
    "pre_exit_all_same_count",
    "median_same_ratio",
    "median_frames_to_first_diverge",
]

CASE_FIELDS = [
    "case_id",
    "flip_type",
    "seq_name",
    "gt_id",
    "exit_frame",
    "reentry_frame",
    "gap",
    "bucket",
    "window_start",
    "window_end",
    "baseline_exit_tid",
    "compare_exit_tid",
    "exit_relation",
    "baseline_reentry_tid",
    "compare_reentry_tid",
    "reentry_relation",
    "gt_present_frames",
    "both_present_frames",
    "same_tid_frames",
    "different_tid_frames",
    "baseline_only_frames",
    "compare_only_frames",
    "both_missing_frames",
    "same_ratio",
    "pre_exit_any_diverge",
    "pre_exit_all_same",
    "first_diverge_frame",
    "frames_to_first_diverge",
    "last_same_frame",
    "exit_iou_baseline",
    "exit_iou_compare",
    "reentry_iou_baseline",
    "reentry_iou_compare",
]

FRAME_FIELDS = [
    "case_id",
    "flip_type",
    "seq_name",
    "gt_id",
    "frame",
    "is_exit_frame",
    "is_reentry_frame",
    "gt_present",
    "baseline_tid",
    "baseline_iou",
    "compare_tid",
    "compare_iou",
    "relation",
]

GROUP_FIELDS = [
    "flip_type",
    "case_count",
    "exit_same_count",
    "exit_different_count",
    "exit_baseline_missing_count",
    "exit_compare_missing_count",
    "exit_both_missing_count",
    "pre_exit_any_diverge_count",
    "pre_exit_all_same_count",
    "median_same_ratio",
    "median_frames_to_first_diverge",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze pre-exit anchor drift between two tracker runs.")
    parser.add_argument("--cases-csv", required=True, help="Usually critical_flips.csv from re-entry error decomposition.")
    parser.add_argument("--gt-annotations-json", required=True, help="COCO-style GT annotations json, e.g. DanceTrack val.json.")
    parser.add_argument("--baseline-results-dir", required=True, help="Directory containing baseline MOT result txt files.")
    parser.add_argument("--compare-results-dir", required=True, help="Directory containing comparison MOT result txt files.")
    parser.add_argument("--compare-name", default="compare", help="Label used in output fields and registry notes.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--context-before", type=int, default=15)
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument("--registry-csv", default=str(REGISTRY_CSV))
    return parser.parse_args()


def write_rows(path: Path, fieldnames: Iterable[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_single_row_csv(path: Path, fieldnames: Iterable[str], row: Dict[str, object]) -> None:
    write_rows(path, fieldnames, [row])


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
        "scripts/analyze_reentry_pre_exit_drift.py",
        "--dataset",
        "DanceTrack",
        "--split",
        "val",
        "--tracker-family",
        "reentry_pre_exit_drift",
        "--variant",
        Path(args.out_dir).name,
        "--tag",
        Path(args.out_dir).name,
        "--run-root",
        str(Path(args.out_dir).resolve()),
        "--summary-csv",
        str(summary_csv.resolve()),
        "--notes",
        notes,
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=False)


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


def match_tracker_id(gt_row: dict | None, tracker_rows: List[dict], iou_thresh: float) -> Tuple[str, float]:
    if gt_row is None:
        return "", 0.0
    gt_box = tlwh_to_tlbr(gt_row)
    best_tid = ""
    best_iou = 0.0
    for row in tracker_rows:
        score = iou(gt_box, tlwh_to_tlbr(row))
        if score >= float(iou_thresh) and score > best_iou:
            best_iou = score
            best_tid = str(int(row["id"]))
    return best_tid, best_iou


def relation_from_ids(baseline_tid: str, compare_tid: str, gt_present: bool) -> str:
    if not gt_present:
        return "gt_absent"
    if baseline_tid and compare_tid:
        return "same" if baseline_tid == compare_tid else "different"
    if baseline_tid and not compare_tid:
        return "compare_missing"
    if compare_tid and not baseline_tid:
        return "baseline_missing"
    return "both_missing"


def infer_flip_type(row: dict) -> str:
    if row.get("baseline_correct_engine_not") == "1":
        return "baseline_correct_compare_not"
    if row.get("engine_correct_baseline_not") == "1":
        return "compare_correct_baseline_not"
    return "changed_noncritical"


def safe_median(values: List[float | int]) -> float:
    if not values:
        return 0.0
    return float(statistics.median(values))


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "summary.csv"

    write_single_row_csv(
        summary_csv,
        SUMMARY_FIELDS,
        {
            "status": "running",
            "error": "",
            "case_count": 0,
            "flip_type_count": 0,
            "context_before": int(args.context_before),
            "iou_thresh": float(args.iou_thresh),
            "exit_same_count": 0,
            "exit_different_count": 0,
            "exit_baseline_missing_count": 0,
            "exit_compare_missing_count": 0,
            "exit_both_missing_count": 0,
            "pre_exit_any_diverge_count": 0,
            "pre_exit_all_same_count": 0,
            "median_same_ratio": 0.0,
            "median_frames_to_first_diverge": 0.0,
        },
    )
    append_registry(summary_csv=summary_csv, args=args, status="running", notes=f"starting compare={args.compare_name}")

    cases_csv = Path(args.cases_csv).expanduser().resolve()
    gt_json = Path(args.gt_annotations_json).expanduser().resolve()
    baseline_dir = Path(args.baseline_results_dir).expanduser().resolve()
    compare_dir = Path(args.compare_results_dir).expanduser().resolve()

    case_inputs = [row for row in csv.DictReader(cases_csv.open("r", encoding="utf-8"))]
    gt_by_seq = parse_coco_track_rows(gt_json)

    baseline_cache: Dict[str, Dict[int, List[dict]]] = {}
    compare_cache: Dict[str, Dict[int, List[dict]]] = {}

    case_rows: List[Dict[str, object]] = []
    frame_rows: List[Dict[str, object]] = []

    exit_relation_counter: Counter = Counter()
    group_counter: Dict[str, Counter] = defaultdict(Counter)
    same_ratios: List[float] = []
    frames_to_first_diverge_values: List[int] = []

    try:
        for case_id, case in enumerate(case_inputs, start=1):
            seq_name = str(case["seq_name"])
            gt_id = int(case["gt_id"])
            exit_frame = int(case["exit_frame"])
            reentry_frame = int(case["reentry_frame"])
            flip_type = infer_flip_type(case)
            window_start = max(1, exit_frame - int(args.context_before))
            window_end = exit_frame

            if seq_name not in baseline_cache:
                baseline_cache[seq_name] = parse_mot_rows(baseline_dir / f"{seq_name}.txt")
            if seq_name not in compare_cache:
                compare_cache[seq_name] = parse_mot_rows(compare_dir / f"{seq_name}.txt")

            gt_rows = gt_by_seq[seq_name]
            base_rows = baseline_cache[seq_name]
            cmp_rows = compare_cache[seq_name]

            stats = Counter()
            relations: List[Tuple[int, str]] = []

            baseline_exit_tid = ""
            compare_exit_tid = ""
            baseline_reentry_tid = ""
            compare_reentry_tid = ""
            exit_iou_baseline = 0.0
            exit_iou_compare = 0.0
            reentry_iou_baseline = 0.0
            reentry_iou_compare = 0.0

            for frame in range(window_start, window_end + 1):
                gt_row = get_gt_row(gt_rows, frame, gt_id)
                gt_present = gt_row is not None
                base_tid, base_iou = match_tracker_id(gt_row, base_rows.get(frame, []), float(args.iou_thresh))
                cmp_tid, cmp_iou = match_tracker_id(gt_row, cmp_rows.get(frame, []), float(args.iou_thresh))
                relation = relation_from_ids(base_tid, cmp_tid, gt_present)
                relations.append((frame, relation))

                if gt_present:
                    stats["gt_present_frames"] += 1
                    if relation == "same":
                        stats["both_present_frames"] += 1
                        stats["same_tid_frames"] += 1
                    elif relation == "different":
                        stats["both_present_frames"] += 1
                        stats["different_tid_frames"] += 1
                    elif relation == "baseline_missing":
                        stats["compare_only_frames"] += 1
                    elif relation == "compare_missing":
                        stats["baseline_only_frames"] += 1
                    elif relation == "both_missing":
                        stats["both_missing_frames"] += 1

                frame_rows.append(
                    {
                        "case_id": case_id,
                        "flip_type": flip_type,
                        "seq_name": seq_name,
                        "gt_id": gt_id,
                        "frame": frame,
                        "is_exit_frame": int(frame == exit_frame),
                        "is_reentry_frame": 0,
                        "gt_present": int(gt_present),
                        "baseline_tid": base_tid,
                        "baseline_iou": round(base_iou, 6),
                        "compare_tid": cmp_tid,
                        "compare_iou": round(cmp_iou, 6),
                        "relation": relation,
                    }
                )

                if frame == exit_frame:
                    baseline_exit_tid = base_tid
                    compare_exit_tid = cmp_tid
                    exit_iou_baseline = base_iou
                    exit_iou_compare = cmp_iou

            gt_row = get_gt_row(gt_rows, reentry_frame, gt_id)
            baseline_reentry_tid, reentry_iou_baseline = match_tracker_id(gt_row, base_rows.get(reentry_frame, []), float(args.iou_thresh))
            compare_reentry_tid, reentry_iou_compare = match_tracker_id(gt_row, cmp_rows.get(reentry_frame, []), float(args.iou_thresh))
            reentry_relation = relation_from_ids(baseline_reentry_tid, compare_reentry_tid, gt_row is not None)

            frame_rows.append(
                {
                    "case_id": case_id,
                    "flip_type": flip_type,
                    "seq_name": seq_name,
                    "gt_id": gt_id,
                    "frame": reentry_frame,
                    "is_exit_frame": 0,
                    "is_reentry_frame": 1,
                    "gt_present": int(gt_row is not None),
                    "baseline_tid": baseline_reentry_tid,
                    "baseline_iou": round(reentry_iou_baseline, 6),
                    "compare_tid": compare_reentry_tid,
                    "compare_iou": round(reentry_iou_compare, 6),
                    "relation": reentry_relation,
                }
            )

            exit_relation = relations[-1][1] if relations else "both_missing"
            first_diverge_frame = ""
            for frame, relation in relations[:-1]:
                if relation != "same":
                    first_diverge_frame = frame
                    break
            last_same_frame = ""
            for frame, relation in reversed(relations):
                if relation == "same":
                    last_same_frame = frame
                    break

            same_ratio = (
                float(stats["same_tid_frames"]) / float(stats["gt_present_frames"])
                if stats["gt_present_frames"] > 0
                else 0.0
            )
            pre_exit_any_diverge = int(any(relation != "same" for _, relation in relations[:-1]))
            pre_exit_all_same = int(all(relation == "same" for _, relation in relations if relation != "gt_absent") and len(relations) > 0)
            frames_to_first_diverge = (
                int(exit_frame) - int(first_diverge_frame)
                if first_diverge_frame != ""
                else ""
            )

            exit_relation_counter[exit_relation] += 1
            group_counter[flip_type]["case_count"] += 1
            group_counter[flip_type][f"exit_{exit_relation}_count"] += 1
            group_counter[flip_type]["pre_exit_any_diverge_count"] += int(pre_exit_any_diverge)
            group_counter[flip_type]["pre_exit_all_same_count"] += int(pre_exit_all_same)
            same_ratios.append(same_ratio)
            group_counter[flip_type]["same_ratio_sum"] += float(same_ratio)
            if frames_to_first_diverge != "":
                frames_to_first_diverge_values.append(int(frames_to_first_diverge))
                group_counter[flip_type]["frames_to_first_diverge_sum"] += int(frames_to_first_diverge)
                group_counter[flip_type]["frames_to_first_diverge_count"] += 1

            case_rows.append(
                {
                    "case_id": case_id,
                    "flip_type": flip_type,
                    "seq_name": seq_name,
                    "gt_id": gt_id,
                    "exit_frame": exit_frame,
                    "reentry_frame": reentry_frame,
                    "gap": int(case["gap"]),
                    "bucket": case["bucket"],
                    "window_start": window_start,
                    "window_end": window_end,
                    "baseline_exit_tid": baseline_exit_tid,
                    "compare_exit_tid": compare_exit_tid,
                    "exit_relation": exit_relation,
                    "baseline_reentry_tid": baseline_reentry_tid,
                    "compare_reentry_tid": compare_reentry_tid,
                    "reentry_relation": reentry_relation,
                    "gt_present_frames": int(stats["gt_present_frames"]),
                    "both_present_frames": int(stats["both_present_frames"]),
                    "same_tid_frames": int(stats["same_tid_frames"]),
                    "different_tid_frames": int(stats["different_tid_frames"]),
                    "baseline_only_frames": int(stats["baseline_only_frames"]),
                    "compare_only_frames": int(stats["compare_only_frames"]),
                    "both_missing_frames": int(stats["both_missing_frames"]),
                    "same_ratio": round(same_ratio, 6),
                    "pre_exit_any_diverge": int(pre_exit_any_diverge),
                    "pre_exit_all_same": int(pre_exit_all_same),
                    "first_diverge_frame": first_diverge_frame,
                    "frames_to_first_diverge": frames_to_first_diverge,
                    "last_same_frame": last_same_frame,
                    "exit_iou_baseline": round(exit_iou_baseline, 6),
                    "exit_iou_compare": round(exit_iou_compare, 6),
                    "reentry_iou_baseline": round(reentry_iou_baseline, 6),
                    "reentry_iou_compare": round(reentry_iou_compare, 6),
                }
            )

        group_rows: List[Dict[str, object]] = []
        for flip_type in sorted(group_counter):
            group_stats = group_counter[flip_type]
            group_rows.append(
                {
                    "flip_type": flip_type,
                    "case_count": int(group_stats.get("case_count", 0)),
                    "exit_same_count": int(group_stats.get("exit_same_count", 0)),
                    "exit_different_count": int(group_stats.get("exit_different_count", 0)),
                    "exit_baseline_missing_count": int(group_stats.get("exit_baseline_missing_count", 0)),
                    "exit_compare_missing_count": int(group_stats.get("exit_compare_missing_count", 0)),
                    "exit_both_missing_count": int(group_stats.get("exit_both_missing_count", 0)),
                    "pre_exit_any_diverge_count": int(group_stats.get("pre_exit_any_diverge_count", 0)),
                    "pre_exit_all_same_count": int(group_stats.get("pre_exit_all_same_count", 0)),
                    "median_same_ratio": round(
                        float(group_stats.get("same_ratio_sum", 0.0)) / float(max(1, group_stats.get("case_count", 0))),
                        6,
                    ),
                    "median_frames_to_first_diverge": round(
                        float(group_stats.get("frames_to_first_diverge_sum", 0.0))
                        / float(max(1, group_stats.get("frames_to_first_diverge_count", 0))),
                        6,
                    ),
                }
            )

        write_rows(out_dir / "pre_exit_anchor_drift_cases.csv", CASE_FIELDS, case_rows)
        write_rows(out_dir / "pre_exit_anchor_drift_frames.csv", FRAME_FIELDS, frame_rows)
        write_rows(out_dir / "flip_type_summary.csv", GROUP_FIELDS, group_rows)
        write_single_row_csv(
            summary_csv,
            SUMMARY_FIELDS,
            {
                "status": "success",
                "error": "",
                "case_count": len(case_rows),
                "flip_type_count": len(group_rows),
                "context_before": int(args.context_before),
                "iou_thresh": float(args.iou_thresh),
                "exit_same_count": int(exit_relation_counter.get("same", 0)),
                "exit_different_count": int(exit_relation_counter.get("different", 0)),
                "exit_baseline_missing_count": int(exit_relation_counter.get("baseline_missing", 0)),
                "exit_compare_missing_count": int(exit_relation_counter.get("compare_missing", 0)),
                "exit_both_missing_count": int(exit_relation_counter.get("both_missing", 0)),
                "pre_exit_any_diverge_count": int(sum(int(row["pre_exit_any_diverge"]) for row in case_rows)),
                "pre_exit_all_same_count": int(sum(int(row["pre_exit_all_same"]) for row in case_rows)),
                "median_same_ratio": round(safe_median(same_ratios), 6),
                "median_frames_to_first_diverge": round(safe_median(frames_to_first_diverge_values), 6),
            },
        )
        append_registry(summary_csv=summary_csv, args=args, status="success", notes=f"compare={args.compare_name} case_count={len(case_rows)}")
        return 0
    except Exception as exc:
        write_single_row_csv(
            summary_csv,
            SUMMARY_FIELDS,
            {
                "status": "failed",
                "error": str(exc),
                "case_count": 0,
                "flip_type_count": 0,
                "context_before": int(args.context_before),
                "iou_thresh": float(args.iou_thresh),
                "exit_same_count": 0,
                "exit_different_count": 0,
                "exit_baseline_missing_count": 0,
                "exit_compare_missing_count": 0,
                "exit_both_missing_count": 0,
                "pre_exit_any_diverge_count": 0,
                "pre_exit_all_same_count": 0,
                "median_same_ratio": 0.0,
                "median_frames_to_first_diverge": 0.0,
            },
        )
        append_registry(summary_csv=summary_csv, args=args, status="failed", notes=f"compare={args.compare_name} error={exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
