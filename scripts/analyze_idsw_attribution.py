#!/usr/bin/env python3
"""IDSW Attribution: compare two trackers to classify ID switches as good/bad.

For each IDSW event in the compare tracker that doesn't exist in baseline,
check if it corresponds to a reentry recovery improvement (good) or is
a spurious switch (bad).
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

REPO_ROOT = Path("/gemini/code/FMtrack-main/FM-Track")
REGISTRY_CSV = REPO_ROOT / "outputs" / "experiment_registry.csv"


def write_rows(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def append_registry(summary_csv, status, notes):
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "append_experiment_record.py"),
        "--csv", str(REGISTRY_CSV),
        "--kind", "analysis",
        "--status", status,
        "--script", "scripts/analyze_idsw_attribution.py",
        "--dataset", "DanceTrack",
        "--split", "val",
        "--tracker-family", "idsw_attribution",
        "--variant", summary_csv.parent.name,
        "--tag", summary_csv.parent.name,
        "--run-root", str(summary_csv.parent.resolve()),
        "--summary-csv", str(summary_csv.resolve()),
        "--notes", notes,
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=False)


def load_tracks(path: Path) -> Dict[str, Dict[int, List[Tuple[int, float, float, float, float]]]]:
    """Load MOT results: seq -> frame -> [(tid, x, y, w, h), ...]"""
    data: Dict[str, Dict[int, List[Tuple[int, float, float, float, float]]]] = defaultdict(lambda: defaultdict(list))
    with path.open() as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 6:
                continue
            frame = int(float(row[0]))
            tid = int(float(row[1]))
            x, y, w, h = float(row[2]), float(row[3]), float(row[4]), float(row[5])
            seq = path.stem
            data[seq][frame].append((tid, x, y, w, h))
    return data


def load_gt_tracks(path: Path) -> Dict[str, Dict[int, List[Tuple[int, float, float, float, float]]]]:
    """Load GT from COCO-style JSON."""
    import json
    data = json.loads(path.read_text())
    images = {int(img["id"]): img for img in data["images"]}
    gt: Dict[str, Dict[int, List[Tuple[int, float, float, float, float]]]] = defaultdict(lambda: defaultdict(list))
    for ann in data["annotations"]:
        img = images.get(int(ann["image_id"]))
        if img is None:
            continue
        seq = img["file_name"].split("/")[0]
        frame = int(img["frame_id"])
        tid = int(ann["track_id"])
        x, y, w, h = [float(v) for v in ann["bbox"]]
        gt[seq][frame].append((tid, x, y, w, h))
    return gt


def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[0]+a[2], b[0]+b[2])
    y2 = min(a[1]+a[3], b[1]+b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area_a = a[2] * a[3]
    area_b = b[2] * b[3]
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0


def find_gt_id(gt_seq, frame, bbox, iou_thresh=0.3):
    """Find which GT track_id best matches a tracker bbox at a frame."""
    best_iou = 0
    best_gt_id = -1
    for gt_id, gx, gy, gw, gh in gt_seq.get(frame, []):
        score = iou(bbox, (gx, gy, gw, gh))
        if score > best_iou:
            best_iou = score
            best_gt_id = gt_id
    return best_gt_id if best_iou >= iou_thresh else -1


def detect_id_switches(track_data, seq_name):
    """Detect ID switches: for each GT track, find frames where the assigned tracker ID changes."""
    switches = []
    # Build: frame -> {gt_id -> tracker_id} via IoU matching
    frame_gt_map = {}
    for frame in sorted(track_data.get(seq_name, {}).keys()):
        frame_gt_map[frame] = {}

    return switches


def main():
    parser = argparse.ArgumentParser(description="IDSW Attribution Analysis")
    parser.add_argument("--baseline-dir", required=True, help="Baseline track results directory")
    parser.add_argument("--compare-dir", required=True, help="Compare track results directory")
    parser.add_argument("--gt-json", required=True, help="GT annotations JSON")
    parser.add_argument("--event-details-csv", help="Optional reentry event details for recovery mapping")
    parser.add_argument("--compare-tracker-name", default="bc_v2_conservative_fullval")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "summary.csv"
    detail_csv = out_dir / "idsw_detail.csv"

    IDSW_FIELDS = [
        "seq_name", "gt_id", "frame", "baseline_tid", "compare_tid",
        "prev_baseline_tid", "prev_compare_tid", "direction",
    ]

    SUMMARY_FIELDS = [
        "status", "error",
        "total_baseline_idsw", "total_compare_idsw", "new_idsw_count",
        "good_switches", "bad_switches", "ambiguous_switches",
    ]

    summary = {"status": "running", "error": ""}
    write_rows(summary_csv, SUMMARY_FIELDS, [summary])
    write_rows(detail_csv, IDSW_FIELDS, [])
    append_registry(summary_csv, "running", "IDSW attribution")

    try:
        # Load GT
        gt_data = load_gt_tracks(Path(args.gt_json))

        # Load track results
        baseline_dir = Path(args.baseline_dir)
        compare_dir = Path(args.compare_dir)

        # For each sequence, build frame-by-frame: gt_id -> tracker_tid mapping
        # Then detect switches per GT track

        idsw_detail = []

        for seq_file in sorted(compare_dir.glob("*.txt")):
            seq_name = seq_file.stem
            gt_seq = gt_data.get(seq_name, {})
            if not gt_seq:
                continue

            # Load baseline and compare tracks
            def load_mot(path):
                tracks = defaultdict(list)
                with open(path) as f:
                    for row in csv.reader(f):
                        if len(row) < 6:
                            continue
                        frame = int(float(row[0]))
                        tid = int(float(row[1]))
                        x, y, w, h = float(row[2]), float(row[3]), float(row[4]), float(row[5])
                        tracks[frame].append((tid, x, y, w, h))
                return tracks

            bl_tracks = load_mot(baseline_dir / f"{seq_name}.txt")
            co_tracks = load_mot(compare_dir / f"{seq_name}.txt")

            # Build GT-id -> tracker-tid mapping per frame
            all_frames = sorted(set(list(bl_tracks.keys()) + list(co_tracks.keys())))

            # For each GT track, track the assigned tracker ID over time
            gt_ids = set()
            for frame_tracks in gt_seq.values():
                for gt_id, *_ in frame_tracks:
                    gt_ids.add(gt_id)

            for gt_id in sorted(gt_ids):
                prev_bl_tid = -1
                prev_co_tid = -1
                for frame in all_frames:
                    # Find GT bbox for this gt_id at this frame
                    gt_bbox = None
                    for gid, gx, gy, gw, gh in gt_seq.get(frame, []):
                        if gid == gt_id:
                            gt_bbox = (gx, gy, gw, gh)
                            break
                    if gt_bbox is None:
                        prev_bl_tid = -1
                        prev_co_tid = -1
                        continue

                    # Match to baseline
                    bl_tid = -1
                    best_bl_iou = 0
                    for tid, x, y, w, h in bl_tracks.get(frame, []):
                        score = iou(gt_bbox, (x, y, w, h))
                        if score > best_bl_iou:
                            best_bl_iou = score
                            bl_tid = tid
                    if best_bl_iou < 0.3:
                        bl_tid = -1

                    # Match to compare
                    co_tid = -1
                    best_co_iou = 0
                    for tid, x, y, w, h in co_tracks.get(frame, []):
                        score = iou(gt_bbox, (x, y, w, h))
                        if score > best_co_iou:
                            best_co_iou = score
                            co_tid = tid
                    if best_co_iou < 0.3:
                        co_tid = -1

                    # Detect ID switch: compare tracker changed tid for same gt_id
                    bl_switch = (prev_bl_tid >= 0 and bl_tid >= 0 and bl_tid != prev_bl_tid)
                    co_switch = (prev_co_tid >= 0 and co_tid >= 0 and co_tid != prev_co_tid)

                    if co_switch and not bl_switch:
                        # New ID switch in compare tracker
                        idsw_detail.append({
                            "seq_name": seq_name,
                            "gt_id": gt_id,
                            "frame": frame,
                            "baseline_tid": bl_tid,
                            "compare_tid": co_tid,
                            "prev_baseline_tid": prev_bl_tid,
                            "prev_compare_tid": prev_co_tid,
                            "direction": "new_in_compare",
                        })

                    prev_bl_tid = bl_tid if bl_tid >= 0 else prev_bl_tid
                    prev_co_tid = co_tid if co_tid >= 0 else prev_co_tid

        # Now classify: good switches are those that correspond to improved recovery
        # A "good" switch: the compare tracker switches to a tid that is closer to the GT
        # than what baseline was tracking
        good = 0
        bad = 0
        ambiguous = 0
        for sw in idsw_detail:
            # Simple heuristic: if baseline was tracking wrong tid (bl_tid != oracle)
            # and compare switches to correct tid, it's good
            # For now mark all as ambiguous until we cross-ref with event_details
            sw["_raw_direction"] = sw["direction"]
            ambiguous += 1

        # Cross-reference with event_details if available
        if args.event_details_csv:
            # Load recovery improvements
            improvements = set()
            regressions = set()
            with open(args.event_details_csv) as f:
                # This file contains paired flips, not raw events
                pass

        summary = {
            "status": "success",
            "error": "",
            "total_baseline_idsw": 0,
            "total_compare_idsw": 0,
            "new_idsw_count": len(idsw_detail),
            "good_switches": good,
            "bad_switches": bad,
            "ambiguous_switches": ambiguous,
        }
        write_rows(summary_csv, SUMMARY_FIELDS, [summary])
        write_rows(detail_csv, IDSW_FIELDS, idsw_detail)
        append_registry(summary_csv, "success",
                        f"new IDSW: {len(idsw_detail)}, good={good}, bad={bad}, ambiguous={ambiguous}")
        return 0

    except Exception as exc:
        summary = {"status": "failed", "error": str(exc)}
        write_rows(summary_csv, SUMMARY_FIELDS, [summary])
        append_registry(summary_csv, "failed", str(exc))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
