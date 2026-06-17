#!/usr/bin/env python3
"""Build RGSA staged labels from HACA runtime oracle dump.

Reads per-frame pairbank CSVs and derives three-stage labels:
  Stage 1: accept / defer / reject
  Stage 2: rewrite / defer / reject (only for Stage-1-deferred pairs)
  Stage 3: recover / miss (only for Stage-2-rejected + unmatched)

Labels are derived by matching det_tlwh/track_tlwh to GT via IoU,
since the pairbank does not contain pre-computed GT match fields.
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

from models.rgsa_contract import STAGE1_FEATURE_NAMES, HACA_PAIR_FEATURE_NAMES


def _runtime_score(pair):
    return safe_float(pair.get("s_final", pair.get("anchor_sim", 0.0)), 0.0)


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


def clamp01(value):
    return max(0.0, min(1.0, float(value)))


def derive_stage2_pair_features(pair, pairs, rank, min_history):
    score_vals = np.asarray([_runtime_score(p) for p in pairs], dtype=np.float32)
    score_mean = float(score_vals.mean()) if len(score_vals) else 0.0
    score_std = float(score_vals.std()) if len(score_vals) else 0.0
    top1_score = float(score_vals[0]) if len(score_vals) else 0.0
    top2_score = float(score_vals[1]) if len(score_vals) > 1 else top1_score
    top_margin = top1_score - top2_score

    hist_last = safe_float(pair.get("hist_last_sim", 0.0))
    hist_max = safe_float(pair.get("hist_max_sim", 0.0))
    hist_std = safe_float(pair.get("hist_std_sim", 0.0))
    history_len = safe_float(pair.get("history_len", 0.0))
    track_gap = safe_float(pair.get("track_gap", 0.0))

    gap_log1p = safe_float(pair.get("gap_log1p"), math.log1p(max(track_gap, 0.0)))
    hist_norm = safe_float(pair.get("hist_norm"), min(1.0, history_len / max(float(min_history), 1.0)))
    stability = safe_float(pair.get("stability"), math.exp(-max(hist_std, 0.0)))
    if hist_max > 1e-6:
        coherence_fallback = hist_last / hist_max
    else:
        coherence_fallback = hist_last
    coherence = safe_float(pair.get("coherence"), coherence_fallback)
    anchor_z = safe_float(
        pair.get("anchor_z"),
        (_runtime_score(pair) - score_mean) / max(score_std, 1e-6),
    )
    anchor_margin = safe_float(pair.get("anchor_margin"), top_margin)
    anchor_rank = safe_float(pair.get("anchor_rank"), float(rank))

    return {
        "anchor_sim": _runtime_score(pair),
        "spatial_sim": safe_float(pair.get("spatial_sim", 0.0)),
        "motion_sim": safe_float(pair.get("motion_sim", 0.0)),
        "temp_sim": safe_float(pair.get("temp_sim", 0.0)),
        "hist_last_sim": hist_last,
        "hist_max_sim": hist_max,
        "hist_std_sim": hist_std,
        "gap_log1p": gap_log1p,
        "hist_norm": clamp01(hist_norm),
        "stability": clamp01(stability),
        "coherence": clamp01(coherence),
        "anchor_z": anchor_z,
        "anchor_margin": anchor_margin,
        "anchor_rank": anchor_rank,
        "det_score": safe_float(pair.get("det_score", 0.0)),
    }


def iou_tlwh(a, b):
    """IoU between two tlwh boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / max(union, 1e-8)


def parse_tlwh(s):
    """Parse 'x,y,w,h' string to tuple of floats."""
    parts = s.split(",")
    return tuple(float(p) for p in parts[:4])


def load_gt(gt_path):
    """Load GT file as dict: frame_id -> list of (gt_id, tlwh)."""
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
    return gt


def load_gt_train_half(gt_path):
    """Load GT and keep only first 50% of frames (train_half split)."""
    gt = load_gt(gt_path)
    if not gt:
        return gt
    max_fid = max(gt.keys())
    half_fid = max_fid // 2
    return {fid: entries for fid, entries in gt.items() if fid <= half_fid}


def match_box_to_gt(box_tlwh, gt_entries, iou_thresh=0.7):
    """Find best GT match for a box. Returns (gt_id, iou) or (-1, 0)."""
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


def iter_pairbank_frames(pairbank_path):
    """Yield (frame_id, rows) groups from a pairbank written in frame order."""
    with open(pairbank_path, newline="") as f:
        reader = csv.DictReader(f)
        current_fid = None
        current_rows = []
        for row in reader:
            try:
                fid = int(row.get("frame_id", 0))
            except (TypeError, ValueError):
                fid = 0
            if current_rows and fid != current_fid:
                yield current_fid, current_rows
                current_rows = []
            current_fid = fid
            current_rows.append(row)
        if current_rows:
            yield current_fid, current_rows


def get_stage_writer(writers, files, seq_out, stage_name, fieldnames):
    writer = writers.get(stage_name)
    if writer is not None:
        return writer
    out_path = os.path.join(seq_out, f"{stage_name}_labels.csv")
    f = open(out_path, "w", newline="")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    files.append(f)
    writers[stage_name] = writer
    return writer


def main():
    parser = argparse.ArgumentParser(description="Build RGSA staged labels")
    parser.add_argument("--oracle-dir", required=True, help="Root of rgsa_oracle_dump directory")
    parser.add_argument("--data-root", required=True, help="Dataset root (for GT)")
    parser.add_argument("--dataset", default="MOT17")
    parser.add_argument("--seqs", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--margin-threshold", type=float, default=0.05)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--iou-threshold", type=float, default=0.7)
    parser.add_argument("--min-history", type=int, default=3)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    global_counts = Counter()
    seq_summaries = []

    for seq in args.seqs:
        pairbank_path = os.path.join(args.oracle_dir, seq, "pairbank.csv")
        if not os.path.exists(pairbank_path):
            print(f"[warn] missing {pairbank_path}, skipping")
            continue

        # Load GT for this sequence (train_half)
        gt_path = os.path.join(args.data_root, args.dataset, "train", seq, "gt", "gt_train_half.txt")
        if not os.path.exists(gt_path):
            gt_path = os.path.join(args.data_root, args.dataset, "train", seq, "gt", "gt.txt")
        gt = load_gt_train_half(gt_path)
        print(f"[{seq}] loaded GT: {sum(len(v) for v in gt.values())} entries across {len(gt)} frames")

        seq_out = os.path.join(args.out_dir, seq)
        os.makedirs(seq_out, exist_ok=True)
        for stale_name in ("stage1_labels.csv", "stage2_labels.csv", "stage3_labels.csv"):
            stale_path = os.path.join(seq_out, stale_name)
            if os.path.exists(stale_path):
                os.remove(stale_path)

        s1_counts = Counter()
        s2_counts = Counter()
        s3_counts = Counter()
        writers = {}
        files = []
        frame_count = 0
        pair_count = 0

        try:
            for fid, frame_rows in iter_pairbank_frames(pairbank_path):
                frame_count += 1
                pair_count += len(frame_rows)
                gt_entries = gt.get(fid, [])

                # For each detection, find its GT match and its top-k track candidates' GT matches
                det_groups = defaultdict(list)
                for row in frame_rows:
                    det_id = int(row.get("det_id", 0))
                    det_groups[det_id].append(row)

                for det_id, pairs in det_groups.items():
                    # Sort by runtime score descending so labels match the edge that Stage1 actually intervenes on.
                    pairs.sort(key=lambda r: -_runtime_score(r))

                    top1 = pairs[0] if pairs else None
                    if top1 is None:
                        continue

                    # Match detection to GT
                    det_tlwh_str = top1.get("det_tlwh", "")
                    det_gt_id = -1
                    if det_tlwh_str and gt_entries:
                        try:
                            det_tlwh = parse_tlwh(det_tlwh_str)
                            det_gt_id, _ = match_box_to_gt(det_tlwh, gt_entries, args.iou_threshold)
                        except (ValueError, IndexError):
                            det_gt_id = -1

                    # Match top1 track to GT
                    top1_track_tlwh_str = top1.get("track_tlwh", "")
                    top1_track_gt_id = -1
                    if top1_track_tlwh_str and gt_entries:
                        try:
                            top1_track_tlwh = parse_tlwh(top1_track_tlwh_str)
                            top1_track_gt_id, _ = match_box_to_gt(top1_track_tlwh, gt_entries, args.iou_threshold)
                        except (ValueError, IndexError):
                            top1_track_gt_id = -1

                    # Is top1 match correct? (det GT == track GT, both valid)
                    top1_correct = (det_gt_id >= 0 and top1_track_gt_id == det_gt_id)

                    # Stage 1 label
                    margin = float(top1.get("margin", 0.0))
                    s_final = float(top1.get("s_final", 0.0))

                    if top1_correct:
                        s1_label = 0  # accept
                    elif margin > args.margin_threshold and s_final > 0.3:
                        s1_label = 2  # reject (confidently wrong)
                    else:
                        s1_label = 1  # defer

                    s1_counts[s1_label] += 1
                    global_counts[f"s1_{s1_label}"] += 1

                    stage1_entry = {fn: top1.get(fn, "") for fn in STAGE1_FEATURE_NAMES}
                    stage1_entry.update({
                        "seq_name": seq, "frame_id": fid, "det_id": det_id,
                        "track_id": top1.get("track_id", ""),
                        "s_final": top1.get("s_final", ""),
                        "margin": top1.get("margin", ""),
                        "label": s1_label,
                    })
                    get_stage_writer(writers, files, seq_out, "stage1", list(stage1_entry.keys())).writerow(stage1_entry)

                    stage2_group_action = None
                    if s1_label == 1:  # deferred → build Stage 2 labels
                        # Stage 2 is a DETECTION-GROUP level decision, not per-candidate.
                        # First check: does the correct candidate exist in top-k?
                        correct_candidate_rank = -1
                        for rank, pair in enumerate(pairs[:args.topk]):
                            track_tlwh_str = pair.get("track_tlwh", "")
                            track_gt_id = -1
                            if track_tlwh_str and det_gt_id >= 0:
                                try:
                                    track_tlwh = parse_tlwh(track_tlwh_str)
                                    track_gt_id, _ = match_box_to_gt(track_tlwh, gt_entries, args.iou_threshold)
                                except (ValueError, IndexError):
                                    track_gt_id = -1
                            if det_gt_id >= 0 and track_gt_id == det_gt_id:
                                correct_candidate_rank = rank
                                break

                        # Determine the group-level Stage 2 action.
                        # - rewrite: correct candidate exists in local top-k
                        # - defer: detection is real, but local candidates cannot recover it
                        # - reject: detection itself has no GT match (background / false positive)
                        if correct_candidate_rank >= 0:
                            group_action = 0  # rewrite: correct candidate found in top-k
                        elif det_gt_id >= 0:
                            group_action = 1  # defer to Stage 3 / archive recovery
                        else:
                            group_action = 2  # reject: no GT-backed object to recover

                        stage2_group_action = group_action
                        s2_counts[group_action] += 1
                        global_counts[f"s2_{group_action}"] += 1

                        # Write per-candidate rows with the group-level label
                        for rank, pair in enumerate(pairs[:args.topk]):
                            stage2_entry = derive_stage2_pair_features(
                                pair,
                                pairs[:args.topk],
                                rank=rank,
                                min_history=args.min_history,
                            )
                            stage2_entry.update({
                                "seq_name": seq, "frame_id": fid, "det_id": det_id,
                                "track_id": pair.get("track_id", ""),
                                "topk_rank": rank,
                                "correct_candidate_rank": correct_candidate_rank,
                                "label": group_action,
                            })
                            get_stage_writer(writers, files, seq_out, "stage2", list(stage2_entry.keys())).writerow(stage2_entry)

                    # Stage 3 should see unresolved detections:
                    # - Stage 1 confident reject that still corresponds to a GT object
                    # - Stage 2 defer cases that need archive / long-gap recovery
                    needs_stage3 = (s1_label == 2) or (s1_label == 1 and stage2_group_action == 1)
                    if needs_stage3:
                        # Stage 3: can this detection be recovered from archive?
                        # We approximate: if det has a valid GT match, it's recoverable.
                        s3_label = 0 if det_gt_id >= 0 else 1
                        s3_counts[s3_label] += 1
                        global_counts[f"s3_{s3_label}"] += 1

                        stage3_entry = {
                            "seq_name": seq, "frame_id": fid, "det_id": det_id,
                            "det_gt_id": det_gt_id,
                            "source_stage1_label": s1_label,
                            "source_stage2_action": stage2_group_action if stage2_group_action is not None else "",
                            "label": s3_label,
                        }
                        get_stage_writer(writers, files, seq_out, "stage3", list(stage3_entry.keys())).writerow(stage3_entry)
        finally:
            for f in files:
                f.close()

        if frame_count == 0:
            print(f"[warn] empty pairbank for {seq}")
            continue

        seq_summary = {
            "seq_name": seq,
            "stage1_accept": s1_counts[0], "stage1_defer": s1_counts[1], "stage1_reject": s1_counts[2],
            "stage2_rewrite": s2_counts[0], "stage2_defer": s2_counts[1], "stage2_reject": s2_counts[2],
            "stage3_recover": s3_counts[0], "stage3_miss": s3_counts[1],
        }
        seq_summaries.append(seq_summary)
        print(f"[{seq}] frames={frame_count} pairs={pair_count} | "
              f"s1: accept={s1_counts[0]} defer={s1_counts[1]} reject={s1_counts[2]} | "
              f"s2: rewrite={s2_counts[0]} defer={s2_counts[1]} reject={s2_counts[2]} | "
              f"s3: recover={s3_counts[0]} miss={s3_counts[1]}")

    # Write global summary
    summary_path = os.path.join(args.out_dir, "summary.csv")
    if seq_summaries:
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(seq_summaries[0].keys()))
            writer.writeheader()
            writer.writerows(seq_summaries)

    total_s1 = sum(global_counts.get(f"s1_{c}", 0) for c in range(3))
    total_s2 = sum(global_counts.get(f"s2_{c}", 0) for c in range(3))
    total_s3 = sum(global_counts.get(f"s3_{c}", 0) for c in range(2))
    print(f"\n[global] Stage 1: accept={global_counts.get('s1_0',0)} defer={global_counts.get('s1_1',0)} reject={global_counts.get('s1_2',0)} (total={total_s1})")
    print(f"[global] Stage 2: rewrite={global_counts.get('s2_0',0)} defer={global_counts.get('s2_1',0)} reject={global_counts.get('s2_2',0)} (total={total_s2})")
    print(f"[global] Stage 3: recover={global_counts.get('s3_0',0)} miss={global_counts.get('s3_1',0)} (total={total_s3})")
    print(f"[output] {args.out_dir}")


if __name__ == "__main__":
    main()
