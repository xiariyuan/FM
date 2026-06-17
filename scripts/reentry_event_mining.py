#!/usr/bin/env python3
"""Re-entry event mining + gap-bucket evaluation for MOT20.

Reads GT annotations and tracking outputs, identifies re-entry events
(same GT ID appearing after a gap), and computes:
1. Event density: how many re-entry events exist per sequence
2. Gap distribution: histogram of gap lengths
3. Baseline recovery rate: how many re-entries the baseline tracker recovers
4. Oracle ceiling: how much IDF1/HOTA could improve if all re-entries were recovered
5. Gap-bucket stratified recovery rates
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


def parse_mot_txt(path: str) -> dict[int, list[tuple[int, float, float, float, float]]]:
    """Parse MOT format txt → {gt_id: [(frame, x, y, w, h), ...]}"""
    tracks: dict[int, list[tuple[int, float, float, float, float]]] = defaultdict(list)
    with open(path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            frame = int(parts[0])
            tid = int(parts[1])
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            tracks[tid].append((frame, x, y, w, h))
    for tid in tracks:
        tracks[tid].sort(key=lambda r: r[0])
    return dict(tracks)


def find_reentry_events(
    gt_tracks: dict[int, list[tuple[int, float, float, float, float]]],
    min_gap: int = 10,
) -> list[dict]:
    """Find re-entry events in GT: same ID with frame gaps > min_gap.

    Returns list of event dicts with: gt_id, exit_frame, reentry_frame, gap.
    """
    events = []
    for tid, frames in gt_tracks.items():
        frame_nums = [f for f, _, _, _, _ in frames]
        for i in range(1, len(frame_nums)):
            gap = frame_nums[i] - frame_nums[i - 1]
            if gap > min_gap:
                events.append({
                    "gt_id": tid,
                    "exit_frame": frame_nums[i - 1],
                    "reentry_frame": frame_nums[i],
                    "gap": gap,
                })
    events.sort(key=lambda e: e["gap"], reverse=True)
    return events


def gap_bucket(gap: int) -> str:
    if gap <= 30:
        return "10-30"
    elif gap <= 100:
        return "31-100"
    elif gap <= 300:
        return "101-300"
    else:
        return "300+"


def check_recovery(
    tracker_tracks: dict[int, list[tuple[int, float, float, float, float]]],
    event: dict,
    iou_thresh: float = 0.3,
) -> dict:
    """Check if a tracker recovered the re-entry detection.

    Returns dict with recovery info: recovered (bool), tracker_id, iou.
    """
    reentry_frame = event["reentry_frame"]
    gt_id = event["gt_id"]

    # Find GT bbox at reentry frame
    gt_bboxes = {f: (x, y, w, h) for f, x, y, w, h in []}
    # Need to look up from original tracks - this is done externally
    return {"recovered": False, "tracker_id": -1, "iou": 0.0}


def compute_iou(box1, box2) -> float:
    """Compute IoU between two (x, y, w, h) boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)
    inter = max(0, xb - xa) * max(0, yb - ya)
    union = w1 * h1 + w2 * h2 - inter
    return inter / max(union, 1e-6)


def analyze_sequence(
    seq_name: str,
    gt_path: str,
    tracker_path: str,
    reference_path: str | None = None,
    min_gap: int = 10,
    iou_thresh: float = 0.3,
) -> dict:
    """Full analysis for one sequence."""
    gt_tracks = parse_mot_txt(gt_path)
    tracker_tracks = parse_mot_txt(tracker_path)
    ref_tracks = parse_mot_txt(reference_path) if reference_path else None

    # Find re-entry events
    events = find_reentry_events(gt_tracks, min_gap=min_gap)

    # Check recovery for tracker and reference
    gt_frame_map: dict[int, dict[int, tuple[float, float, float, float]]] = defaultdict(dict)
    for tid, frames in gt_tracks.items():
        for f, x, y, w, h in frames:
            gt_frame_map[f][tid] = (x, y, w, h)

    tracker_frame_map: dict[int, dict[int, tuple[float, float, float, float]]] = defaultdict(dict)
    for tid, frames in tracker_tracks.items():
        for f, x, y, w, h in frames:
            tracker_frame_map[f][tid] = (x, y, w, h)

    ref_frame_map: dict[int, dict[int, tuple[float, float, float, float]]] = defaultdict(dict)
    if ref_tracks:
        for tid, frames in ref_tracks.items():
            for f, x, y, w, h in frames:
                ref_frame_map[f][tid] = (x, y, w, h)

    def find_best_match(frame_map, frame, gt_bbox):
        """Find the best IoU match for a GT bbox in a tracker's detections at a frame."""
        best_iou = 0.0
        best_tid = -1
        detections = frame_map.get(frame, {})
        for tid, det_bbox in detections.items():
            iou = compute_iou(gt_bbox, det_bbox)
            if iou > best_iou:
                best_iou = iou
                best_tid = tid
        return best_tid, best_iou

    # Evaluate each event
    results = []
    for event in events:
        reentry_frame = event["reentry_frame"]
        gt_id = event["gt_id"]
        gt_bbox = gt_frame_map.get(reentry_frame, {}).get(gt_id)
        if gt_bbox is None:
            continue

        bucket = gap_bucket(event["gap"])

        # Tracker recovery
        trk_tid, trk_iou = find_best_match(tracker_frame_map, reentry_frame, gt_bbox)
        tracker_recovered = trk_iou >= iou_thresh

        # Reference recovery
        ref_recovered = False
        ref_tid, ref_iou = -1, 0.0
        if ref_frame_map:
            ref_tid, ref_iou = find_best_match(ref_frame_map, reentry_frame, gt_bbox)
            ref_recovered = ref_iou >= iou_thresh

        results.append({
            **event,
            "bucket": bucket,
            "tracker_recovered": tracker_recovered,
            "tracker_tid": trk_tid,
            "tracker_iou": round(trk_iou, 3),
            "ref_recovered": ref_recovered,
            "ref_tid": ref_tid,
            "ref_iou": round(ref_iou, 3),
        })

    # Aggregate by bucket
    bucket_stats = defaultdict(lambda: {"total": 0, "tracker_recovered": 0, "ref_recovered": 0})
    for r in results:
        b = r["bucket"]
        bucket_stats[b]["total"] += 1
        if r["tracker_recovered"]:
            bucket_stats[b]["tracker_recovered"] += 1
        if r["ref_recovered"]:
            bucket_stats[b]["ref_recovered"] += 1

    total = len(results)
    tracker_total = sum(1 for r in results if r["tracker_recovered"])
    ref_total = sum(1 for r in results if r["ref_recovered"])

    return {
        "seq": seq_name,
        "total_reentry_events": total,
        "tracker_recovery_count": tracker_total,
        "tracker_recovery_rate": round(tracker_total / max(total, 1), 4),
        "ref_recovery_count": ref_total,
        "ref_recovery_rate": round(ref_total / max(total, 1), 4),
        "recovery_gain": round((tracker_total - ref_total) / max(total, 1), 4),
        "bucket_stats": dict(bucket_stats),
        "events": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Re-entry event mining + gap-bucket eval")
    parser.add_argument("--gt-root", default="/gemini/code/datasets/MOT20/train")
    parser.add_argument("--tracker-dir", required=True, help="Path to tracker results dir")
    parser.add_argument("--reference-dir", default=None, help="Path to reference results dir")
    parser.add_argument("--seq-ids", nargs="+", type=int, default=[1, 2, 3, 5])
    parser.add_argument("--min-gap", type=int, default=10, help="Minimum gap to count as re-entry")
    parser.add_argument("--iou-thresh", type=float, default=0.3)
    parser.add_argument("--output-json", default=None, help="Output JSON path")
    args = parser.parse_args()

    all_results = []
    for seq_id in args.seq_ids:
        seq_name = f"MOT20-{seq_id:02d}"
        gt_path = Path(args.gt_root) / seq_name / "gt" / "gt.txt"
        if not gt_path.exists():
            # Try the standard MOT format
            gt_path = Path(args.gt_root) / seq_name / "gt" / "gt_val_half.txt"
        if not gt_path.exists():
            gt_path = Path(args.gt_root) / seq_name / "gt" / "gt.txt"
        if not gt_path.exists():
            print(f"WARNING: GT not found for {seq_name}, skipping")
            continue

        tracker_path = Path(args.tracker_dir) / f"{seq_name}.txt"
        if not tracker_path.exists():
            print(f"WARNING: Tracker results not found for {seq_name}, skipping")
            continue

        ref_path = None
        if args.reference_dir:
            ref_path = Path(args.reference_dir) / f"{seq_name}.txt"
            if not ref_path.exists():
                ref_path = None

        result = analyze_sequence(
            seq_name, str(gt_path), str(tracker_path),
            str(ref_path) if ref_path else None,
            min_gap=args.min_gap, iou_thresh=args.iou_thresh,
        )
        all_results.append(result)

    # Print summary table
    print("\n" + "=" * 80)
    print("RE-ENTRY EVENT MINING + GAP-BUCKET RECOVERY ANALYSIS")
    print("=" * 80)
    print(f"Config: min_gap={args.min_gap}, iou_thresh={args.iou_thresh}")
    print()

    # Header
    print(f"{'Seq':<12} {'Events':>8} {'Trk Rec':>8} {'Trk Rate':>10} {'Ref Rec':>8} {'Ref Rate':>10} {'Gain':>8}")
    print("-" * 74)
    for r in all_results:
        print(f"{r['seq']:<12} {r['total_reentry_events']:>8} "
              f"{r['tracker_recovery_count']:>8} {r['tracker_recovery_rate']:>10.4f} "
              f"{r['ref_recovery_count']:>8} {r['ref_recovery_rate']:>10.4f} "
              f"{r['recovery_gain']:>+8.4f}")

    # Aggregate
    total_events = sum(r["total_reentry_events"] for r in all_results)
    total_trk = sum(r["tracker_recovery_count"] for r in all_results)
    total_ref = sum(r["ref_recovery_count"] for r in all_results)
    print("-" * 74)
    print(f"{'TOTAL':<12} {total_events:>8} {total_trk:>8} "
          f"{total_trk / max(total_events, 1):>10.4f} "
          f"{total_ref:>8} {total_ref / max(total_events, 1):>10.4f} "
          f"{(total_trk - total_ref) / max(total_events, 1):>+8.4f}")

    # Bucket breakdown
    print("\n" + "=" * 80)
    print("GAP-BUCKET BREAKDOWN")
    print("=" * 80)
    all_buckets = defaultdict(lambda: {"total": 0, "tracker_recovered": 0, "ref_recovered": 0})
    for r in all_results:
        for b, stats in r["bucket_stats"].items():
            all_buckets[b]["total"] += stats["total"]
            all_buckets[b]["tracker_recovered"] += stats["tracker_recovered"]
            all_buckets[b]["ref_recovered"] += stats["ref_recovered"]

    print(f"{'Bucket':<12} {'Events':>8} {'Trk Rec':>8} {'Trk Rate':>10} {'Ref Rec':>8} {'Ref Rate':>10} {'Gain':>8}")
    print("-" * 74)
    for bucket_name in sorted(all_buckets.keys()):
        s = all_buckets[bucket_name]
        trk_rate = s["tracker_recovered"] / max(s["total"], 1)
        ref_rate = s["ref_recovered"] / max(s["total"], 1)
        print(f"{bucket_name:<12} {s['total']:>8} {s['tracker_recovered']:>8} "
              f"{trk_rate:>10.4f} {s['ref_recovered']:>8} {ref_rate:>10.4f} "
              f"{trk_rate - ref_rate:>+8.4f}")

    # Save JSON
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nDetailed results saved to {args.output_json}")

    return all_results


if __name__ == "__main__":
    main()
