#!/usr/bin/env python3
"""Build DCRC diagnostic dataset from existing HACA oracle dump.

Extends pairbank with density/ambiguity features per host edge.
Does NOT re-run BoT-SORT — reads existing pairbank CSVs.

Output: one CSV per seq with per-frame per-det diagnostic features.
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


def safe_float(v, default=0.0):
    if v in ("", None):
        return float(default)
    try:
        out = float(v)
    except (TypeError, ValueError):
        return float(default)
    return float(default) if not math.isfinite(out) else out


def iou_tlwh(a, b):
    ax, ay, aw, ah = a[:4]
    bx, by, bw, bh = b[:4]
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / max(union, 1e-8)


def parse_tlwh(s):
    parts = s.split(",")
    return tuple(float(p) for p in parts[:4])


def load_gt_half(gt_path, half="train"):
    """Load GT, keep first half (train) or second half (val) of frames."""
    gt = defaultdict(list)
    with open(gt_path) as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 6:
                continue
            fid = int(p[0])
            gid = int(p[1])
            x, y, w, h = float(p[2]), float(p[3]), float(p[4]), float(p[5])
            gt[fid].append((gid, np.array([x, y, w, h])))
    if not gt:
        return gt
    max_fid = max(gt.keys())
    half_fid = max_fid // 2
    if half == "train":
        return {f: v for f, v in gt.items() if f <= half_fid}
    else:
        return {f: v for f, v in gt.items() if f > half_fid}


def compute_local_density(det_tlwh, all_det_tlwhs, radius=100):
    """Count nearby detections within radius pixels."""
    if det_tlwh is None:
        return 0
    cx, cy = det_tlwh[0] + det_tlwh[2] / 2, det_tlwh[1] + det_tlwh[3] / 2
    count = 0
    for other in all_det_tlwhs:
        if other is None:
            continue
        ox, oy = other[0] + other[2] / 2, other[1] + other[3] / 2
        dist = math.sqrt((cx - ox) ** 2 + (cy - oy) ** 2)
        if dist < radius:
            count += 1
    return count


def compute_track_density(track_tlwh, all_track_tlwhs, radius=100):
    """Count nearby tracks within radius pixels."""
    if track_tlwh is None:
        return 0
    cx, cy = track_tlwh[0] + track_tlwh[2] / 2, track_tlwh[1] + track_tlwh[3] / 2
    count = 0
    for other in all_track_tlwhs:
        if other is None:
            continue
        ox, oy = other[0] + other[2] / 2, other[1] + other[3] / 2
        dist = math.sqrt((cx - ox) ** 2 + (cy - oy) ** 2)
        if dist < radius:
            count += 1
    return count


def density_bucket_quantile(nearby_count, quantile_edges):
    """Discretize local density using data-driven quantile edges."""
    if nearby_count <= quantile_edges[0]:
        return "sparse"
    elif nearby_count <= quantile_edges[1]:
        return "moderate"
    elif nearby_count <= quantile_edges[2]:
        return "dense"
    else:
        return "crowded"


def main():
    parser = argparse.ArgumentParser(description="Build DCRC diagnostic dataset")
    parser.add_argument("--oracle-dir", required=True, help="RGSA oracle dump dir")
    parser.add_argument("--data-root", required=True, help="Dataset root")
    parser.add_argument("--dataset", default="MOT17")
    parser.add_argument("--seqs", nargs="+", required=True)
    parser.add_argument("--split", required=True, choices=["train", "val", "test", "transfer"])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.7)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--density-radius", type=float, default=100.0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    all_rows = []

    for seq in args.seqs:
        pairbank_path = os.path.join(args.oracle_dir, seq, "pairbank.csv")
        if not os.path.exists(pairbank_path):
            print(f"[warn] missing {pairbank_path}")
            continue

        # Load GT — strict split isolation
        gt_path = os.path.join(args.data_root, args.dataset, "train", seq, "gt", "gt_train_half.txt")
        if not os.path.exists(gt_path):
            gt_path = os.path.join(args.data_root, args.dataset, "train", seq, "gt", "gt.txt")

        # train split uses first half; val/test/transfer use second half
        if args.split == "train":
            gt_half = "train"
        else:
            gt_half = "val"
        gt = load_gt_half(gt_path, gt_half)

        # Read pairbank
        frames = defaultdict(list)
        with open(pairbank_path) as f:
            for row in csv.DictReader(f):
                frames[int(row.get("frame_id", 0))].append(row)

        seq_rows = 0
        for fid in sorted(frames.keys()):
            frame_rows = frames[fid]
            gt_entries = gt.get(fid, [])

            # Collect all det and track tlwh for density computation
            all_det_tlwhs = []
            all_track_tlwhs = []
            det_map = defaultdict(list)
            for row in frame_rows:
                det_id = int(row.get("det_id", 0))
                det_map[det_id].append(row)
                det_tlwh_str = row.get("det_tlwh", "")
                if det_tlwh_str:
                    try:
                        all_det_tlwhs.append(parse_tlwh(det_tlwh_str))
                    except (ValueError, IndexError):
                        all_det_tlwhs.append(None)

            for row in frame_rows:
                track_tlwh_str = row.get("track_tlwh", "")
                if track_tlwh_str:
                    try:
                        all_track_tlwhs.append(parse_tlwh(track_tlwh_str))
                    except (ValueError, IndexError):
                        all_track_tlwhs.append(None)

            # Process each detection (take top1 host edge by s_final, not anchor_sim)
            for det_id, pairs in det_map.items():
                # Sort by s_final descending — this is what the runtime actually commits on
                pairs.sort(key=lambda r: -safe_float(r.get("s_final", 0)))
                top1 = pairs[0] if pairs else None
                if top1 is None:
                    continue

                # Also record rank by anchor_sim for diagnostic cross-reference
                pairs_by_anchor = sorted(pairs, key=lambda r: -safe_float(r.get("anchor_sim", 0)))
                host_rank_by_anchor = 0
                for i, p in enumerate(pairs_by_anchor):
                    if p is top1:
                        host_rank_by_anchor = i
                        break

                # Match detection to GT
                det_tlwh_str = top1.get("det_tlwh", "")
                det_tlwh = None
                det_gt_id = -1
                if det_tlwh_str:
                    try:
                        det_tlwh = parse_tlwh(det_tlwh_str)
                        for gid, g_tlwh in gt_entries:
                            if iou_tlwh(det_tlwh, g_tlwh) >= args.iou_threshold:
                                det_gt_id = gid
                                break
                    except (ValueError, IndexError):
                        pass

                # Match top1 track to GT
                track_tlwh_str = top1.get("track_tlwh", "")
                track_tlwh = None
                track_gt_id = -1
                if track_tlwh_str:
                    try:
                        track_tlwh = parse_tlwh(track_tlwh_str)
                        for gid, g_tlwh in gt_entries:
                            if iou_tlwh(track_tlwh, g_tlwh) >= args.iou_threshold:
                                track_gt_id = gid
                                break
                    except (ValueError, IndexError):
                        pass

                label_commit_ok = 1 if (det_gt_id >= 0 and track_gt_id == det_gt_id) else 0

                # Density features
                nearby_det_count = compute_local_density(det_tlwh, all_det_tlwhs, args.density_radius)
                nearby_track_count = compute_track_density(track_tlwh, all_track_tlwhs, args.density_radius)

                # Competition features (based on s_final ordering)
                candidate_count = len(pairs)
                top1_s_final = safe_float(pairs[0].get("s_final", 0)) if pairs else 0
                top2_s_final = safe_float(pairs[1].get("s_final", 0)) if len(pairs) > 1 else 0
                top1_top2_gap = top1_s_final - top2_s_final

                # IoU features among candidates
                local_ious = []
                for p in pairs[1:5]:
                    p_tlwh_str = p.get("track_tlwh", "")
                    if p_tlwh_str and track_tlwh:
                        try:
                            local_ious.append(iou_tlwh(track_tlwh, parse_tlwh(p_tlwh_str)))
                        except (ValueError, IndexError):
                            pass
                local_iou_mean = float(np.mean(local_ious)) if local_ious else 0.0
                local_iou_max = float(np.max(local_ious)) if local_ious else 0.0

                # Ambiguity score: combination of margin, entropy, density
                margin = safe_float(top1.get("margin", 0))
                entropy = safe_float(top1.get("entropy", 0))
                activation = safe_float(top1.get("activation", 0))
                ambiguity_score = (1 - margin) * (1 + entropy) * (1 + nearby_det_count / 10.0)

                row_out = {
                    "seq_name": seq,
                    "frame_id": fid,
                    "det_id": det_id,
                    "track_id": top1.get("track_id", ""),
                    # Runtime signals
                    "s_final": safe_float(top1.get("s_final", 0)),
                    "margin": margin,
                    "entropy": entropy,
                    "activation": activation,
                    "bg_prob": safe_float(top1.get("bg_prob", 0)),
                    "beta_hist": safe_float(top1.get("beta_hist", 0)),
                    "beta_ood": safe_float(top1.get("beta_ood", 0)),
                    "ood_score": safe_float(top1.get("ood_score", 0)),
                    "track_gap": safe_float(top1.get("track_gap", 0)),
                    "track_age": safe_float(top1.get("track_age", 0)),
                    "history_len": safe_float(top1.get("history_len", 0)),
                    "det_score": safe_float(top1.get("det_score", 0)),
                    # Density features
                    "nearby_det_count": nearby_det_count,
                    "nearby_track_count": nearby_track_count,
                    "local_iou_mean": local_iou_mean,
                    "local_iou_max": local_iou_max,
                    "candidate_count": candidate_count,
                    "top1_top2_gap": top1_top2_gap,
                    "det_competition_degree": nearby_det_count,
                    "track_competition_degree": nearby_track_count,
                    "scene_density_bucket": "",  # filled in post-hoc with quantiles
                    # Ambiguity features
                    "ambiguity_score": ambiguity_score,
                    "score_temperature_proxy": 1.0 / max(margin, 0.01),
                    "host_rank_by_s_final": 0,
                    "host_rank_by_anchor": host_rank_by_anchor,
                    # Label
                    "label_commit_ok": label_commit_ok,
                    "det_gt_id": det_gt_id,
                    "track_gt_id": track_gt_id,
                }
                all_rows.append(row_out)
                seq_rows += 1

        print(f"[{seq}] {seq_rows} diagnostic edges")

    # Post-hoc: compute quantile edges from all collected density values and assign buckets
    if all_rows:
        all_densities = np.array([r["nearby_det_count"] for r in all_rows])
        q25, q50, q75 = np.percentile(all_densities, [25, 50, 75])
        quantile_edges = [q25, q50, q75]
        print(f"[density quantiles] Q25={q25:.1f} Q50={q50:.1f} Q75={q75:.1f}")
        print(f"[density distribution] min={all_densities.min():.0f} max={all_densities.max():.0f} mean={all_densities.mean():.1f} median={np.median(all_densities):.0f}")
        for row in all_rows:
            row["scene_density_bucket"] = density_bucket_quantile(row["nearby_det_count"], quantile_edges)

    # Write output
    if all_rows:
        out_path = os.path.join(args.out_dir, f"{args.split}_diagnostic_edges.csv")
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"[saved] {out_path} ({len(all_rows)} rows)")

        # Summary by density bucket
        bucket_counts = Counter()
        bucket_correct = Counter()
        for row in all_rows:
            b = row["scene_density_bucket"]
            bucket_counts[b] += 1
            if row["label_commit_ok"] == 1:
                bucket_correct[b] += 1

        summary_path = os.path.join(args.out_dir, f"{args.split}_density_summary.csv")
        with open(summary_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["bucket", "count", "correct", "accuracy"])
            for b in ["sparse", "moderate", "dense", "crowded"]:
                cnt = bucket_counts.get(b, 0)
                cor = bucket_correct.get(b, 0)
                acc = cor / max(cnt, 1)
                w.writerow([b, cnt, cor, f"{acc:.4f}"])
                print(f"  {b}: {cnt} edges, accuracy={acc:.4f}")

    print(f"[output] {args.out_dir}")


if __name__ == "__main__":
    main()
