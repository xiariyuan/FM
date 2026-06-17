#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


REPO_ROOT = Path("/gemini/code/FMtrack-main/FM-Track")
REGISTRY_CSV = REPO_ROOT / "outputs" / "experiment_registry.csv"

SUMMARY_FIELDS = [
    "dataset",
    "split",
    "seq_count",
    "tracker_count",
    "gap_bins",
    "min_gap",
    "gt_event_count",
    "status",
    "error",
]

TRACKER_SUMMARY_FIELDS = [
    "tracker_name",
    "event_count",
    "recoverable_events",
    "unrecoverable_events",
    "correct_recoveries",
    "wrong_recoveries",
    "missed_recoveries",
    "wrong_recovery_recoverable",
    "wrong_recovery_unrecoverable",
    "missed_recovery_recoverable",
    "missed_recovery_unrecoverable",
    "recovery_rate",
    "false_recovery_rate",
    "anchor_available_rate",
    "recoverable_recovery_rate",
    "recoverable_miss_rate",
    "reentry_precision",
    "reentry_recall",
    "reentry_f1",
    "oracle_recovery_rate",
]

BUCKET_FIELDS = [
    "tracker_name",
    "bucket",
    "event_count",
    "recoverable_events",
    "unrecoverable_events",
    "correct_recoveries",
    "wrong_recoveries",
    "missed_recoveries",
    "wrong_recovery_recoverable",
    "wrong_recovery_unrecoverable",
    "missed_recovery_recoverable",
    "missed_recovery_unrecoverable",
    "recovery_rate",
    "false_recovery_rate",
    "anchor_available_rate",
    "recoverable_recovery_rate",
    "recoverable_miss_rate",
    "reentry_precision",
    "reentry_recall",
    "reentry_f1",
    "oracle_recovery_rate",
]

EVENT_FIELDS = [
    "tracker_name",
    "seq_name",
    "gt_id",
    "exit_frame",
    "reentry_frame",
    "gap",
    "bucket",
    "recoverable",
    "oracle_track_id",
    "tracker_track_id",
    "status",
    "status_group",
]

GT_SEQUENCE_FIELDS = [
    "seq_name",
    "gt_event_count",
    "max_gap",
]


@dataclass(frozen=True)
class GTEvent:
    seq_name: str
    gt_id: int
    exit_frame: int
    reentry_frame: int
    gap: int
    bucket: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate long-gap re-entry recovery by gap buckets.")
    parser.add_argument("--dataset", default="MOT20")
    parser.add_argument("--split", default="val_half")
    parser.add_argument("--gt-root", help="Directory containing per-sequence GT folders")
    parser.add_argument("--gt-annotations-json", help="COCO-style annotations JSON for datasets such as DanceTrack")
    parser.add_argument(
        "--tracker",
        action="append",
        nargs=2,
        metavar=("NAME", "RESULTS_DIR"),
        help="Tracker name and directory containing per-sequence MOT results (*.txt). Repeat for multiple trackers.",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--gap-bins",
        default="0:30,30:100,100:500,500:",
        help="Comma-separated bins as start:end, end optional for open upper bound.",
    )
    parser.add_argument("--min-gap", type=int, default=1)
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument(
        "--match-mode",
        choices=["iou_thresh", "best_overlap"],
        default="iou_thresh",
        help="How to assign a tracker ID to the GT box at a frame. "
             "'iou_thresh' requires IoU >= --iou-thresh; "
             "'best_overlap' always returns the best-overlap tracker if any overlap exists.",
    )
    parser.add_argument("--registry-csv", default=str(REGISTRY_CSV))
    parser.add_argument(
        "--seq-list",
        nargs="+",
        default=None,
        help="Optional sequence-name filter, e.g. dancetrack0014 dancetrack0019. Limits GT events and tracker loading to these sequences.",
    )
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
        "scripts/eval_reentry_gap_buckets.py",
        "--dataset",
        str(args.dataset),
        "--split",
        str(args.split),
        "--tracker-family",
        "reentry_gap_eval",
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


def parse_gap_bins(spec: str) -> List[tuple[int, int | None, str]]:
    bins: List[tuple[int, int | None, str]] = []
    for item in [part.strip() for part in spec.split(",") if part.strip()]:
        start_s, end_s = item.split(":", 1)
        start = int(start_s)
        end = int(end_s) if end_s.strip() else None
        label = f"{start}-{end if end is not None else 'inf'}"
        bins.append((start, end, label))
    return bins


def bucket_for_gap(gap: int, bins: Sequence[tuple[int, int | None, str]]) -> str:
    for start, end, label in bins:
        if gap < start:
            continue
        if end is None or gap < end:
            return label
    return bins[-1][2]


def parse_mot_rows(path: Path, require_vis: bool = False) -> Dict[int, List[dict[str, float | int]]]:
    rows_by_frame: Dict[int, List[dict[str, float | int]]] = defaultdict(list)
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
            cls = int(float(row[7])) if len(row) > 7 else 1
            vis = float(row[8]) if len(row) > 8 else 1.0
            if require_vis and score <= 0:
                continue
            rows_by_frame[frame].append(
                {
                    "id": ident,
                    "tlx": tlx,
                    "tly": tly,
                    "w": w,
                    "h": h,
                    "score": score,
                    "cls": cls,
                    "vis": vis,
                }
            )
    return rows_by_frame


def parse_coco_track_rows(path: Path) -> Dict[str, Dict[int, List[dict[str, float | int]]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    images_by_id: Dict[int, dict[str, object]] = {}
    seq_rows: Dict[str, Dict[int, List[dict[str, float | int]]]] = defaultdict(lambda: defaultdict(list))

    for image in data.get("images", []):
        image_id = int(image["id"])
        images_by_id[image_id] = image

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
                "score": float(ann.get("conf", 1.0)),
                "cls": int(ann.get("category_id", 1)),
                "vis": 1.0,
            }
        )
    return seq_rows


def extract_gt_events(gt_rows: Dict[int, List[dict[str, float | int]]], seq_name: str, min_gap: int, bins: Sequence[tuple[int, int | None, str]]) -> List[GTEvent]:
    frames_by_id: Dict[int, List[int]] = defaultdict(list)
    for frame, rows in gt_rows.items():
        for row in rows:
            if int(row["score"]) <= 0:
                continue
            if int(row["cls"]) != 1:
                continue
            frames_by_id[int(row["id"])].append(int(frame))

    events: List[GTEvent] = []
    for gt_id, frames in frames_by_id.items():
        frames = sorted(set(frames))
        for prev_frame, next_frame in zip(frames, frames[1:]):
            gap = int(next_frame) - int(prev_frame) - 1
            if gap < int(min_gap):
                continue
            events.append(
                GTEvent(
                    seq_name=seq_name,
                    gt_id=int(gt_id),
                    exit_frame=int(prev_frame),
                    reentry_frame=int(next_frame),
                    gap=int(gap),
                    bucket=bucket_for_gap(int(gap), bins),
                )
            )
    return events


def tlwh_to_tlbr(row: dict[str, float | int]) -> tuple[float, float, float, float]:
    x1 = float(row["tlx"])
    y1 = float(row["tly"])
    x2 = x1 + float(row["w"])
    y2 = y1 + float(row["h"])
    return x1, y1, x2, y2


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
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
    if denom <= 0.0:
        return 0.0
    return inter / denom


def match_tracker_id_at_frame(
    gt_rows: Dict[int, List[dict[str, float | int]]],
    tracker_rows: Dict[int, List[dict[str, float | int]]],
    frame: int,
    gt_id: int,
    iou_thresh: float,
    match_mode: str = "iou_thresh",
) -> int | None:
    gt_frame_rows = gt_rows.get(frame, [])
    tracker_frame_rows = tracker_rows.get(frame, [])
    gt_row = None
    for row in gt_frame_rows:
        if int(row["id"]) == int(gt_id):
            gt_row = row
            break
    if gt_row is None:
        return None

    gt_box = tlwh_to_tlbr(gt_row)
    best_id = None
    best_iou = -1.0
    for row in tracker_frame_rows:
        score = iou(gt_box, tlwh_to_tlbr(row))
        if score > best_iou:
            best_iou = score
            best_id = int(row["id"])
    if best_id is None:
        return None
    if str(match_mode) == "best_overlap":
        return best_id if best_iou > 0.0 else None
    return best_id if best_iou >= float(iou_thresh) else None


def evaluate_tracker_events(
    events: Sequence[GTEvent],
    gt_by_seq: Dict[str, Dict[int, List[dict[str, float | int]]]],
    tracker_by_seq: Dict[str, Dict[int, List[dict[str, float | int]]]],
    iou_thresh: float,
    match_mode: str = "iou_thresh",
) -> tuple[List[Dict[str, object]], Counter]:
    event_rows: List[Dict[str, object]] = []
    stats: Counter = Counter()
    bucket_stats: Counter = Counter()

    for event in events:
        gt_rows = gt_by_seq[event.seq_name]
        tracker_rows = tracker_by_seq.get(event.seq_name, {})
        oracle_track_id = match_tracker_id_at_frame(
            gt_rows,
            tracker_rows,
            event.exit_frame,
            event.gt_id,
            iou_thresh,
            match_mode=match_mode,
        )
        tracker_track_id = match_tracker_id_at_frame(
            gt_rows,
            tracker_rows,
            event.reentry_frame,
            event.gt_id,
            iou_thresh,
            match_mode=match_mode,
        )
        recoverable = oracle_track_id is not None
        same_identity = recoverable and tracker_track_id is not None and int(tracker_track_id) == int(oracle_track_id)
        wrong_identity = tracker_track_id is not None and (oracle_track_id is None or int(tracker_track_id) != int(oracle_track_id))

        if same_identity:
            status = "correct_recovery"
            status_group = "correct"
        elif wrong_identity and recoverable:
            status = "wrong_recovery_recoverable"
            status_group = "wrong_recovery"
        elif wrong_identity:
            status = "wrong_recovery_unrecoverable"
            status_group = "wrong_recovery"
        elif recoverable:
            status = "missed_recovery_recoverable"
            status_group = "missed_recovery"
        else:
            status = "missed_recovery_unrecoverable"
            status_group = "missed_recovery"

        stats["events"] += 1
        bucket_stats[(event.bucket, "events")] += 1
        if recoverable:
            stats["recoverable"] += 1
            bucket_stats[(event.bucket, "recoverable")] += 1
        else:
            stats["unrecoverable"] += 1
            bucket_stats[(event.bucket, "unrecoverable")] += 1
        if same_identity:
            stats["correct"] += 1
            bucket_stats[(event.bucket, "correct")] += 1
        elif wrong_identity:
            stats["wrong"] += 1
            bucket_stats[(event.bucket, "wrong")] += 1
            if recoverable:
                stats["wrong_recovery_recoverable"] += 1
                bucket_stats[(event.bucket, "wrong_recovery_recoverable")] += 1
            else:
                stats["wrong_recovery_unrecoverable"] += 1
                bucket_stats[(event.bucket, "wrong_recovery_unrecoverable")] += 1
        else:
            stats["missed"] += 1
            bucket_stats[(event.bucket, "missed")] += 1
            if recoverable:
                stats["missed_recovery_recoverable"] += 1
                bucket_stats[(event.bucket, "missed_recovery_recoverable")] += 1
            else:
                stats["missed_recovery_unrecoverable"] += 1
                bucket_stats[(event.bucket, "missed_recovery_unrecoverable")] += 1

        event_rows.append(
            {
                "seq_name": event.seq_name,
                "gt_id": int(event.gt_id),
                "exit_frame": int(event.exit_frame),
                "reentry_frame": int(event.reentry_frame),
                "gap": int(event.gap),
                "bucket": event.bucket,
                "recoverable": int(recoverable),
                "oracle_track_id": "" if oracle_track_id is None else int(oracle_track_id),
                "tracker_track_id": "" if tracker_track_id is None else int(tracker_track_id),
                "status": status,
                "status_group": status_group,
            }
        )

    stats["bucket_stats"] = bucket_stats
    return event_rows, stats


def safe_ratio(num: int, den: int) -> float:
    return float(num) / float(den) if den else 0.0


def tracker_summary_row(name: str, stats: Counter) -> Dict[str, object]:
    correct = int(stats.get("correct", 0))
    wrong = int(stats.get("wrong", 0))
    missed = int(stats.get("missed", 0))
    events = int(stats.get("events", 0))
    recoverable = int(stats.get("recoverable", 0))
    unrecoverable = int(stats.get("unrecoverable", 0))
    wrong_recovery_recoverable = int(stats.get("wrong_recovery_recoverable", 0))
    wrong_recovery_unrecoverable = int(stats.get("wrong_recovery_unrecoverable", 0))
    missed_recovery_recoverable = int(stats.get("missed_recovery_recoverable", 0))
    missed_recovery_unrecoverable = int(stats.get("missed_recovery_unrecoverable", 0))
    precision = safe_ratio(correct, correct + wrong)
    recall = safe_ratio(correct, events)
    f1 = safe_ratio(2.0 * precision * recall, precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "tracker_name": name,
        "event_count": events,
        "recoverable_events": recoverable,
        "unrecoverable_events": unrecoverable,
        "correct_recoveries": correct,
        "wrong_recoveries": wrong,
        "missed_recoveries": missed,
        "wrong_recovery_recoverable": wrong_recovery_recoverable,
        "wrong_recovery_unrecoverable": wrong_recovery_unrecoverable,
        "missed_recovery_recoverable": missed_recovery_recoverable,
        "missed_recovery_unrecoverable": missed_recovery_unrecoverable,
        "recovery_rate": round(safe_ratio(correct, events), 6),
        "false_recovery_rate": round(safe_ratio(wrong, events), 6),
        "anchor_available_rate": round(safe_ratio(recoverable, events), 6),
        "recoverable_recovery_rate": round(safe_ratio(correct, recoverable), 6),
        "recoverable_miss_rate": round(safe_ratio(missed_recovery_recoverable, recoverable), 6),
        "reentry_precision": round(precision, 6),
        "reentry_recall": round(recall, 6),
        "reentry_f1": round(f1, 6),
        "oracle_recovery_rate": round(safe_ratio(recoverable, events), 6),
    }


def bucket_rows(name: str, bins: Sequence[tuple[int, int | None, str]], stats: Counter) -> List[Dict[str, object]]:
    bucket_stats: Counter = stats["bucket_stats"]
    rows: List[Dict[str, object]] = []
    for _, _, label in bins:
        events = int(bucket_stats.get((label, "events"), 0))
        recoverable = int(bucket_stats.get((label, "recoverable"), 0))
        unrecoverable = int(bucket_stats.get((label, "unrecoverable"), 0))
        correct = int(bucket_stats.get((label, "correct"), 0))
        wrong = int(bucket_stats.get((label, "wrong"), 0))
        missed = int(bucket_stats.get((label, "missed"), 0))
        wrong_recovery_recoverable = int(bucket_stats.get((label, "wrong_recovery_recoverable"), 0))
        wrong_recovery_unrecoverable = int(bucket_stats.get((label, "wrong_recovery_unrecoverable"), 0))
        missed_recovery_recoverable = int(bucket_stats.get((label, "missed_recovery_recoverable"), 0))
        missed_recovery_unrecoverable = int(bucket_stats.get((label, "missed_recovery_unrecoverable"), 0))
        precision = safe_ratio(correct, correct + wrong)
        recall = safe_ratio(correct, events)
        f1 = safe_ratio(2.0 * precision * recall, precision + recall) if (precision + recall) > 0 else 0.0
        rows.append(
            {
                "tracker_name": name,
                "bucket": label,
                "event_count": events,
                "recoverable_events": recoverable,
                "unrecoverable_events": unrecoverable,
                "correct_recoveries": correct,
                "wrong_recoveries": wrong,
                "missed_recoveries": missed,
                "wrong_recovery_recoverable": wrong_recovery_recoverable,
                "wrong_recovery_unrecoverable": wrong_recovery_unrecoverable,
                "missed_recovery_recoverable": missed_recovery_recoverable,
                "missed_recovery_unrecoverable": missed_recovery_unrecoverable,
                "recovery_rate": round(safe_ratio(correct, events), 6),
                "false_recovery_rate": round(safe_ratio(wrong, events), 6),
                "anchor_available_rate": round(safe_ratio(recoverable, events), 6),
                "recoverable_recovery_rate": round(safe_ratio(correct, recoverable), 6),
                "recoverable_miss_rate": round(safe_ratio(missed_recovery_recoverable, recoverable), 6),
                "reentry_precision": round(precision, 6),
                "reentry_recall": round(recall, 6),
                "reentry_f1": round(f1, 6),
                "oracle_recovery_rate": round(safe_ratio(recoverable, events), 6),
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    if not args.gt_root and not args.gt_annotations_json:
        raise ValueError("One of --gt-root or --gt-annotations-json is required.")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = out_dir / "summary.csv"
    tracker_summary_csv = out_dir / "tracker_summary.csv"
    bucket_summary_csv = out_dir / "bucket_summary.csv"
    event_details_csv = out_dir / "event_details.csv"
    gt_sequence_summary_csv = out_dir / "gt_sequence_summary.csv"

    summary_row: Dict[str, object] = {
        "dataset": str(args.dataset),
        "split": str(args.split),
        "seq_count": 0,
        "tracker_count": int(len(args.tracker or [])),
        "gap_bins": str(args.gap_bins),
        "min_gap": int(args.min_gap),
        "gt_event_count": 0,
        "status": "running",
        "error": "",
    }
    write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)
    write_rows(tracker_summary_csv, TRACKER_SUMMARY_FIELDS, [])
    write_rows(bucket_summary_csv, BUCKET_FIELDS, [])
    write_rows(event_details_csv, EVENT_FIELDS, [])
    write_rows(gt_sequence_summary_csv, GT_SEQUENCE_FIELDS, [])
    append_registry(args, summary_csv, "running", "evaluating re-entry gap buckets")

    try:
        bins = parse_gap_bins(str(args.gap_bins))
        seq_filter = set(str(name) for name in (args.seq_list or []))

        gt_by_seq: Dict[str, Dict[int, List[dict[str, float | int]]]] = {}
        events: List[GTEvent] = []
        gt_seq_rows: List[Dict[str, object]] = []
        if args.gt_root:
            gt_root = Path(args.gt_root).expanduser().resolve()
            for seq_dir in sorted(path for path in gt_root.iterdir() if path.is_dir()):
                gt_path = seq_dir / "gt" / "gt.txt"
                if not gt_path.is_file():
                    continue
                seq_name = seq_dir.name
                if seq_filter and seq_name not in seq_filter:
                    continue
                gt_rows = parse_mot_rows(gt_path, require_vis=True)
                gt_by_seq[seq_name] = gt_rows
        else:
            gt_by_seq = parse_coco_track_rows(Path(args.gt_annotations_json).expanduser().resolve())
            if seq_filter:
                gt_by_seq = {seq_name: rows for seq_name, rows in gt_by_seq.items() if seq_name in seq_filter}

        for seq_name in sorted(gt_by_seq):
            gt_rows = gt_by_seq[seq_name]
            seq_events = extract_gt_events(gt_rows, seq_name, int(args.min_gap), bins)
            events.extend(seq_events)
            gt_seq_rows.append(
                {
                    "seq_name": seq_name,
                    "gt_event_count": int(len(seq_events)),
                    "max_gap": int(max((event.gap for event in seq_events), default=0)),
                }
            )

        summary_row["seq_count"] = int(len(gt_by_seq))
        summary_row["gt_event_count"] = int(len(events))

        tracker_rows_out: List[Dict[str, object]] = []
        bucket_rows_out: List[Dict[str, object]] = []
        event_rows_out: List[Dict[str, object]] = []

        for tracker_name, tracker_dir_raw in args.tracker or []:
            tracker_dir = Path(tracker_dir_raw).expanduser().resolve()
            tracker_by_seq: Dict[str, Dict[int, List[dict[str, float | int]]]] = {}
            for seq_name in gt_by_seq:
                tracker_path = tracker_dir / f"{seq_name}.txt"
                if tracker_path.is_file():
                    tracker_by_seq[seq_name] = parse_mot_rows(tracker_path, require_vis=False)
                else:
                    tracker_by_seq[seq_name] = {}

            tracker_event_rows, stats = evaluate_tracker_events(
                events=events,
                gt_by_seq=gt_by_seq,
                tracker_by_seq=tracker_by_seq,
                iou_thresh=float(args.iou_thresh),
                match_mode=str(args.match_mode),
            )
            tracker_rows_out.append(tracker_summary_row(tracker_name, stats))
            bucket_rows_out.extend(bucket_rows(tracker_name, bins, stats))
            for row in tracker_event_rows:
                row["tracker_name"] = tracker_name
                event_rows_out.append(row)

        write_rows(tracker_summary_csv, TRACKER_SUMMARY_FIELDS, tracker_rows_out)
        write_rows(bucket_summary_csv, BUCKET_FIELDS, bucket_rows_out)
        write_rows(event_details_csv, EVENT_FIELDS, event_rows_out)
        write_rows(gt_sequence_summary_csv, GT_SEQUENCE_FIELDS, gt_seq_rows)

        summary_row["status"] = "success"
        write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)
        append_registry(args, summary_csv, "success", "re-entry gap bucket evaluation complete")
        return 0
    except Exception as exc:
        summary_row["status"] = "failed"
        summary_row["error"] = str(exc)
        write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)
        append_registry(args, summary_csv, "failed", f"re-entry gap bucket evaluation failed: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
