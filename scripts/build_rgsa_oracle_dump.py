#!/usr/bin/env python3
"""Build RGSA oracle dump from BoT-SORT + HACA v3 online inference.

Runs HACA v3 inference on MOT17 train_half sequences and dumps per-frame
per-pair runtime features for training data construction.

This script wraps the existing BoT-SORT track.py entry point and adds
a dump hook into haca_fuse_distance to capture all HACA runtime signals.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
BOT_ROOT = REPO_ROOT / "external" / "BoT-SORT-main"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(BOT_ROOT))


def _iou_tlwh(a, b):
    ax, ay, aw, ah = a[:4]
    bx, by, bw, bh = b[:4]
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / max(union, 1e-8)


def load_gt(gt_path):
    """Load GT as {frame_id: [(gt_id, tlwh), ...]}"""
    gt = {}
    with open(gt_path) as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 7:
                continue
            fid = int(p[0])
            gid = int(p[1])
            x, y, w, h = float(p[2]), float(p[3]), float(p[4]), float(p[5])
            gt.setdefault(fid, []).append((gid, np.array([x, y, w, h])))
    return gt


def match_to_gt(det_tlwh, gt_entries, iou_thresh=0.7):
    """Find GT ID that matches a detection via IoU."""
    best_iou = 0.0
    best_gid = -1
    for gid, gt_tlwh in gt_entries:
        iou = _iou_tlwh(det_tlwh, gt_tlwh)
        if iou > best_iou:
            best_iou = iou
            best_gid = gid
    return (best_gid if best_iou >= iou_thresh else -1, best_iou)


class OracleDumper:
    """Collects per-pair HACA runtime features during inference."""

    def __init__(self, seq_name, gt_data, out_dir):
        self.seq_name = seq_name
        self.gt_data = gt_data
        self.out_dir = out_dir
        self.rows = []
        self.frame_counter = 0

    def dump_frame(self, frame_id, haca_debug, detections, tracks, matches,
                   u_track, u_detection, removed_stracks=None):
        """Dump one frame's HACA debug data.

        Args:
            frame_id: int
            haca_debug: dict from haca_fuse_distance with return_debug=True
            detections: list of detection objects
            tracks: list of active track objects
            matches: list of (track_idx, det_idx) from Hungarian
            u_track: unmatched track indices
            u_detection: unmatched detection indices
            removed_stracks: list of removed/archive tracks
        """
        if haca_debug is None:
            return

        matched_set = set(matches) if matches else set()
        matched_dets = {m[1] for m in matched_set}
        gt_entries = self.gt_data.get(frame_id, [])

        # Build track-id map (track pool index -> track_id)
        track_id_map = {}
        for i, t in enumerate(tracks):
            track_id_map[i] = int(getattr(t, "track_id", i))

        # Per detection, iterate over all valid tracks
        anchor_sim = haca_debug.get("anchor_sim", None)
        if anchor_sim is None:
            return

        n_tracks, n_dets = anchor_sim.shape

        for det_idx in range(n_dets):
            det = detections[det_idx] if det_idx < len(detections) else None
            if det is None:
                continue

            det_tlwh = getattr(det, "tlwh", None)
            det_score = float(getattr(det, "score", 0.0))

            # GT match for this detection
            gt_match_tid = -1
            if det_tlwh is not None and gt_entries:
                gt_match_tid, _ = match_to_gt(det_tlwh, gt_entries)

            # Find which track was matched to this det
            matched_track_idx = None
            for t_idx, d_idx in matched_set:
                if d_idx == det_idx:
                    matched_track_idx = t_idx
                    break
            is_matched = matched_track_idx is not None

            # Find host top-1 track (best anchor_sim)
            valid_tracks = []
            for t_idx in range(n_tracks):
                sim = float(anchor_sim[t_idx, det_idx])
                valid_tracks.append((t_idx, sim))
            valid_tracks.sort(key=lambda x: -x[1])

            top1_correct = False
            if valid_tracks and gt_match_tid >= 0:
                top1_tid = track_id_map.get(valid_tracks[0][0], -1)
                # Check if top1 track's GT matches detection's GT
                top1_track = tracks[valid_tracks[0][0]] if valid_tracks[0][0] < len(tracks) else None
                if top1_track is not None:
                    top1_gt_id = getattr(top1_track, "analysis_gt_id", -1)
                    if top1_gt_id < 0:
                        top1_gt_id = getattr(top1_track, "gt_id", -1)
                    top1_correct = (top1_gt_id == gt_match_tid)

            # Check recoverability in top-k (Stage 2)
            topk = 5
            gt_in_topk = False
            for rank, (t_idx, _) in enumerate(valid_tracks[:topk]):
                track = tracks[t_idx] if t_idx < len(tracks) else None
                if track is None:
                    continue
                track_gt_id = getattr(track, "analysis_gt_id", -1)
                if track_gt_id < 0:
                    track_gt_id = getattr(track, "gt_id", -1)
                if track_gt_id == gt_match_tid and gt_match_tid >= 0:
                    gt_in_topk = True
                    break

            # Check recoverability in archive (Stage 3)
            gt_in_archive = False
            if removed_stracks and gt_match_tid >= 0:
                for rt in removed_stracks:
                    rt_gt_id = getattr(rt, "analysis_gt_id", -1)
                    if rt_gt_id < 0:
                        rt_gt_id = getattr(rt, "gt_id", -1)
                    if rt_gt_id == gt_match_tid:
                        gt_in_archive = True
                        break

            # Case type
            if top1_correct:
                case_type = "correct_top1"
            elif gt_in_topk:
                case_type = "correct_in_topk"
            elif not is_matched and gt_in_archive:
                case_type = "recoverable_late"
            elif not is_matched:
                case_type = "unmatched"
            else:
                case_type = "wrong_match"

            # Write rows for top-k candidates
            for rank, (t_idx, sim_val) in enumerate(valid_tracks[:topk]):
                track = tracks[t_idx] if t_idx < len(tracks) else None
                if track is None:
                    continue

                track_id = track_id_map.get(t_idx, -1)
                track_gap = int(frame_id) - int(getattr(track, "end_frame", frame_id)) if hasattr(track, "end_frame") else 0
                track_age = int(getattr(track, "frame_id", frame_id)) - int(getattr(track, "start_frame", frame_id)) if hasattr(track, "start_frame") else 0
                history_len = len(getattr(track, "smooth_feat_history", [])) if hasattr(track, "smooth_feat_history") else 0

                row = {
                    "seq_name": self.seq_name,
                    "frame_id": frame_id,
                    "det_id": det_idx,
                    "track_id": track_id,
                    "topk_rank": rank,
                    "anchor_sim": float(anchor_sim[t_idx, det_idx]) if anchor_sim is not None else 0.0,
                    "spatial_sim": float(haca_debug.get("spatial_sim", np.zeros_like(anchor_sim))[t_idx, det_idx]),
                    "motion_sim": float(haca_debug.get("motion_sim", np.zeros_like(anchor_sim))[t_idx, det_idx]),
                    "temp_sim": float(haca_debug.get("haca_temp_sim", np.zeros_like(anchor_sim))[t_idx, det_idx]),
                    "hist_last_sim": float(haca_debug.get("haca_hist_last", np.zeros_like(anchor_sim))[t_idx, det_idx]),
                    "hist_max_sim": float(haca_debug.get("haca_hist_max", np.zeros_like(anchor_sim))[t_idx, det_idx]),
                    "hist_std_sim": float(haca_debug.get("haca_hist_std", np.zeros_like(anchor_sim))[t_idx, det_idx]),
                    "s_prebg": float(haca_debug.get("final_sim", anchor_sim)[t_idx, det_idx]),
                    "s_final": float(haca_debug.get("final_sim", anchor_sim)[t_idx, det_idx]),
                    "activation": float(haca_debug.get("haca_comp_active", np.zeros_like(anchor_sim))[t_idx, det_idx]) if "haca_comp_active" in haca_debug else 0.0,
                    "margin": float(haca_debug.get("haca_comp_margin", np.zeros(n_dets))[det_idx]) if "haca_comp_margin" in haca_debug else 0.0,
                    "entropy": float(haca_debug.get("haca_comp_entropy", np.zeros(n_dets))[det_idx]) if "haca_comp_entropy" in haca_debug else 0.0,
                    "bg_prob": float(haca_debug.get("haca_background", np.zeros(n_dets))[det_idx]) if "haca_background" in haca_debug else 0.0,
                    "beta_pred": float(haca_debug.get("haca_beta_pred", np.zeros_like(anchor_sim))[t_idx, det_idx]),
                    "beta_hist": float(haca_debug.get("haca_beta_hist", np.zeros_like(anchor_sim))[t_idx, det_idx]),
                    "beta_ood": float(haca_debug.get("haca_beta_ood", np.zeros_like(anchor_sim))[t_idx, det_idx]),
                    "ood_score": float(haca_debug.get("haca_ood_score", np.zeros_like(anchor_sim))[t_idx, det_idx]),
                    "track_gap": track_gap,
                    "track_age": track_age,
                    "history_len": history_len,
                    "det_score": det_score,
                    "matched_by_host": 1 if (is_matched and t_idx == matched_track_idx) else 0,
                    "gt_match_tid": gt_match_tid,
                    "gt_is_correct_top1": 1 if top1_correct else 0,
                    "gt_is_correct_topk": 1 if gt_in_topk else 0,
                    "gt_recoverable_in_stage2": 1 if gt_in_topk else 0,
                    "gt_recoverable_in_stage3": 1 if gt_in_archive else 0,
                    "case_type": case_type,
                }
                self.rows.append(row)

    def save(self):
        os.makedirs(self.out_dir, exist_ok=True)
        if not self.rows:
            print(f"[warn] no data for {self.seq_name}")
            return

        fieldnames = list(self.rows[0].keys())
        csv_path = os.path.join(self.out_dir, "pairbank.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)

        # Recoverability stats
        case_counts = {}
        for row in self.rows:
            ct = row["case_type"]
            case_counts[ct] = case_counts.get(ct, 0) + 1

        stats_path = os.path.join(self.out_dir, "recoverability_stats.csv")
        with open(stats_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["case_type", "count"])
            for ct, cnt in sorted(case_counts.items()):
                writer.writerow([ct, cnt])

        print(f"[saved] {csv_path} ({len(self.rows)} rows)")
        print(f"[saved] {stats_path}")


def main():
    parser = argparse.ArgumentParser(description="Build RGSA oracle dump")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--dataset", default="MOT17")
    parser.add_argument("--seqs", nargs="+", required=True)
    parser.add_argument("--split-part", default="train_half")
    parser.add_argument("--haca-checkpoint", required=True)
    parser.add_argument("--calibrator-npz", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print("[NOTE] This script requires integration with BoT-SORT runtime.")
    print("[NOTE] For now, use it as a reference — the actual dump is done by")
    print("[NOTE] modifying haca_assoc.py to log debug data during inference.")
    print(f"[config] haca_checkpoint={args.haca_checkpoint}")
    print(f"[config] out_dir={args.out_dir}")

    # The actual dump requires modifying bot_sort.py to pass haca_debug
    # to the OracleDumper after each frame. This is implemented as a
    # hook in the haca_fuse_distance return_debug=True path.
    #
    # Usage pattern:
    # 1. Run BoT-SORT with --laplace-assoc --laplace-assoc-mode haca_v3
    #    --laplace-haca-checkpoint <npz>
    # 2. In bot_sort.py update(), capture haca_debug dict per frame
    # 3. Pass to OracleDumper.dump_frame()
    # 4. At end of sequence, call dumper.save()

    print("\n[integration] The dump is built by running BoT-SORT with return_debug=True")
    print("[integration] and capturing haca_debug per frame. See the hook in bot_sort.py.")
    print(f"[output] Will write to {args.out_dir}/{{seq_name}}/pairbank.csv")


if __name__ == "__main__":
    main()
