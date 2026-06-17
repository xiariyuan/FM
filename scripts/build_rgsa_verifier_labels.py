#!/usr/bin/env python3
"""Build verifier labels for RGSA Stage 2.

From existing oracle dump pairbank, generates verifier labels:
  confirm_local = 1 when correct_candidate_rank == 0 (host's top-1 is correct)
  veto_local = 1 when correct_candidate_rank != 0

Does NOT re-dump data. Uses existing pairbank + GT matching.
"""

import argparse
import csv
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from models.rgsa_contract import VERIFIER_FEATURE_NAMES


def safe_float(value, default=0.0):
    if value in ("", None):
        return float(default)
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return out


def iou_tlwh(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / max(union, 1e-8)


def parse_tlwh(s):
    parts = s.split(",")
    return tuple(float(p) for p in parts[:4])


def load_gt_train_half(gt_path):
    gt = defaultdict(list)
    with open(gt_path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            fid = int(parts[0])
            gid = int(parts[1])
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            gt[fid].append((gid, (x, y, w, h)))
    if not gt:
        return gt
    max_fid = max(gt.keys())
    half_fid = max_fid // 2
    return {fid: entries for fid, entries in gt.items() if fid <= half_fid}


def match_box_to_gt(box_tlwh, gt_entries, iou_thresh=0.7):
    best_iou = 0.0
    best_gid = -1
    for gid, gt_tlwh in gt_entries:
        iou = iou_tlwh(box_tlwh, gt_tlwh)
        if iou > best_iou:
            best_iou = iou
            best_gid = gid
    if best_iou >= iou_thresh:
        return best_gid, best_iou
    return -1, best_iou


def main():
    parser = argparse.ArgumentParser(description="Build RGSA verifier labels")
    parser.add_argument("--oracle-dir", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--dataset", default="MOT17")
    parser.add_argument("--seqs", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--iou-threshold", type=float, default=0.7)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    global_counts = Counter()
    seq_summaries = []

    for seq in args.seqs:
        pairbank_path = os.path.join(args.oracle_dir, seq, "pairbank.csv")
        if not os.path.exists(pairbank_path):
            print(f"[warn] missing {pairbank_path}, skipping")
            continue

        gt_path = os.path.join(args.data_root, args.dataset, "train", seq, "gt", "gt_train_half.txt")
        if not os.path.exists(gt_path):
            gt_path = os.path.join(args.data_root, args.dataset, "train", seq, "gt", "gt.txt")
        gt = load_gt_train_half(gt_path)

        rows = []
        with open(pairbank_path) as f:
            for row in csv.DictReader(f):
                rows.append(row)
        if not rows:
            continue

        # Group by frame, then by det
        frames = defaultdict(list)
        for row in rows:
            frames[int(row.get("frame_id", 0))].append(row)

        seq_out = os.path.join(args.out_dir, seq)
        os.makedirs(seq_out, exist_ok=True)

        verifier_rows = []
        confirm_count = 0
        veto_count = 0

        for fid in sorted(frames.keys()):
            frame_rows = frames[fid]
            gt_entries = gt.get(fid, [])
            det_groups = defaultdict(list)
            for row in frame_rows:
                det_groups[int(row.get("det_id", 0))].append(row)

            for det_id, pairs in det_groups.items():
                # Sort by anchor_sim descending
                pairs.sort(key=lambda r: -float(r.get("anchor_sim", 0)))

                # Match detection to GT
                det_tlwh_str = pairs[0].get("det_tlwh", "") if pairs else ""
                det_gt_id = -1
                if det_tlwh_str and gt_entries:
                    try:
                        det_gt_id, _ = match_box_to_gt(parse_tlwh(det_tlwh_str), gt_entries, args.iou_threshold)
                    except (ValueError, IndexError):
                        det_gt_id = -1

                if det_gt_id < 0:
                    continue  # no GT match, skip

                # Check if host's top-1 track matches GT
                top1 = pairs[0]
                top1_track_tlwh = top1.get("track_tlwh", "")
                top1_track_gt_id = -1
                if top1_track_tlwh:
                    try:
                        top1_track_gt_id, _ = match_box_to_gt(parse_tlwh(top1_track_tlwh), gt_entries, args.iou_threshold)
                    except (ValueError, IndexError):
                        top1_track_gt_id = -1

                # Verifier label: confirm if top-1 is correct
                label = 0 if (top1_track_gt_id == det_gt_id) else 1
                if label == 0:
                    confirm_count += 1
                else:
                    veto_count += 1
                global_counts[label] += 1

                # Build verifier feature row
                entry = {fn: top1.get(fn, "") for fn in VERIFIER_FEATURE_NAMES}
                entry.update({
                    "seq_name": seq,
                    "frame_id": fid,
                    "det_id": det_id,
                    "track_id": top1.get("track_id", ""),
                    "det_gt_id": det_gt_id,
                    "top1_track_gt_id": top1_track_gt_id,
                    "correct_rank": 0 if label == 0 else -1,
                    "label": label,
                })
                verifier_rows.append(entry)

        if verifier_rows:
            out_path = os.path.join(seq_out, "verifier_labels.csv")
            with open(out_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(verifier_rows[0].keys()))
                writer.writeheader()
                writer.writerows(verifier_rows)

        seq_summaries.append({
            "seq_name": seq,
            "confirm": confirm_count,
            "veto": veto_count,
            "confirm_rate": round(confirm_count / max(confirm_count + veto_count, 1), 4),
        })
        print(f"[{seq}] confirm={confirm_count} veto={veto_count} "
              f"confirm_rate={confirm_count / max(confirm_count + veto_count, 1):.4f}")

    # Write summary
    summary_path = os.path.join(args.out_dir, "summary.csv")
    if seq_summaries:
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(seq_summaries[0].keys()))
            writer.writeheader()
            writer.writerows(seq_summaries)

    total = global_counts[0] + global_counts[1]
    print(f"\n[global] confirm={global_counts[0]} veto={global_counts[1]} "
          f"confirm_rate={global_counts[0] / max(total, 1):.4f}")
    print(f"[output] {args.out_dir}")


if __name__ == "__main__":
    main()
