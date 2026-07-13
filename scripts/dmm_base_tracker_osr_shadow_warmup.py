#!/usr/bin/env python3
"""Shadow Reliability Warm-up for OSRTrack.

Runs two online trackers on the same frames:
  1) main tracker: starts as baseline; OSR is enabled only after warm-up passes.
  2) shadow tracker: runs pending-high OSR, but its outputs are ignored. It is
     used only to estimate online release reliability:
       pending_high_released / proposal_confirmed.

If shadow reliability is high, main tracker switches on pending-high recovery.
If reliability is low after enough shadow proposals, main tracker keeps OSR off.
This avoids letting unreliable low-score recovery damage the main tracker.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple

import numpy as np

import dmm_base_tracker as base
import dmm_base_tracker_osr as osr_v1
import dmm_base_tracker_osr_confirm as confirm_mod


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Shadow warm-up OSRTrack runner.')
    p.add_argument('--dump-npz', required=True)
    p.add_argument('--seq', default='MOT20-01')
    p.add_argument('--out', required=True)
    p.add_argument('--summary-json', default='')
    p.add_argument('--assoc-mode', default='botsort_reid', choices=['iou', 'botsort_reid'])
    p.add_argument('--track-high-thresh', type=float, default=0.6)
    p.add_argument('--track-low-thresh', type=float, default=0.1)
    p.add_argument('--new-track-thresh', type=float, default=0.5)
    p.add_argument('--track-buffer', type=int, default=70)
    p.add_argument('--match-thresh', type=float, default=0.5)
    p.add_argument('--second-match-thresh', type=float, default=0.5)
    p.add_argument('--unconfirmed-match-thresh', type=float, default=0.7)
    p.add_argument('--proximity-thresh', type=float, default=0.5)
    p.add_argument('--appearance-thresh', type=float, default=0.25)
    p.add_argument('--frame-rate', type=int, default=25)
    p.add_argument('--min-box-area', type=float, default=10.0)
    p.add_argument('--aspect-ratio-thresh', type=float, default=1.6)
    p.add_argument('--limit-frames', type=int, default=0)
    p.add_argument('--activate-new-after-first', action='store_true')
    p.add_argument('--debug-assoc', action='store_true')
    p.add_argument('--stage2-reid-enable', action='store_true')
    p.add_argument('--stage2-lost-enable', action='store_true')
    p.add_argument('--nsa-k', type=float, default=0.0)
    p.add_argument('--gmc-enable', action='store_true')
    p.add_argument('--gmc-warp-path', default='')

    # OSR pair gate inherited from v1/v2.
    p.add_argument('--osr-min-det-score', type=float, default=0.20)
    p.add_argument('--osr-min-memory-sim', type=float, default=0.75)
    p.add_argument('--osr-max-lost-age', type=int, default=15)
    p.add_argument('--osr-min-track-len', type=int, default=15)
    p.add_argument('--osr-max-center-step', type=float, default=35.0)
    p.add_argument('--osr-min-margin', type=float, default=0.05)
    p.add_argument('--osr-max-ambiguity', type=int, default=1)
    p.add_argument('--osr-min-score', type=float, default=0.64)

    # Confirm / pending-high settings.
    p.add_argument('--confirm-min-hits', type=int, default=2)
    p.add_argument('--confirm-min-avg-score', type=float, default=0.66)
    p.add_argument('--confirm-min-avg-memory', type=float, default=0.76)
    p.add_argument('--confirm-max-avg-ambiguity', type=float, default=0.15)
    p.add_argument('--confirm-min-last-score', type=float, default=0.62)
    p.add_argument('--confirm-require-consecutive', action='store_true')

    # Shadow warm-up decision.
    p.add_argument('--shadow-enable', action='store_true')
    p.add_argument('--shadow-min-confirmed', type=int, default=20)
    p.add_argument('--shadow-min-release-ratio', type=float, default=0.55)
    p.add_argument('--shadow-decision-max-frame', type=int, default=0,
                   help='Optional max frame to wait for a decision. 0 means no max.')
    p.add_argument('--shadow-disable-on-fail', action='store_true')
    p.add_argument('--shadow-keep-observing-on-fail', action='store_true')
    p.add_argument('--shadow-max-observe-frame', type=int, default=0,
                   help='Keep observing until this frame before disabling on failed ratio. 0 means no late cutoff.')
    return p.parse_args()


def build_cfg(args: argparse.Namespace) -> base.TrackerConfig:
    return base.TrackerConfig(
        track_high_thresh=args.track_high_thresh,
        track_low_thresh=args.track_low_thresh,
        new_track_thresh=args.new_track_thresh,
        track_buffer=args.track_buffer,
        match_thresh=args.match_thresh,
        second_match_thresh=args.second_match_thresh,
        unconfirmed_match_thresh=args.unconfirmed_match_thresh,
        proximity_thresh=args.proximity_thresh,
        appearance_thresh=args.appearance_thresh,
        frame_rate=args.frame_rate,
        min_box_area=args.min_box_area,
        aspect_ratio_thresh=args.aspect_ratio_thresh,
        assoc_mode=args.assoc_mode,
        activate_new_after_first=bool(args.activate_new_after_first),
        debug_assoc=bool(args.debug_assoc),
        stage2_reid_enable=bool(args.stage2_reid_enable),
        stage2_lost_enable=bool(args.stage2_lost_enable),
        nsa_k=float(args.nsa_k),
        gmc_enable=bool(args.gmc_enable),
        gmc_warp_path=str(args.gmc_warp_path),
    )


def build_osr(args: argparse.Namespace, enable: bool) -> osr_v1.OSRConfig:
    return osr_v1.OSRConfig(
        enable=bool(enable),
        max_lost_age=int(args.osr_max_lost_age),
        min_track_len=int(args.osr_min_track_len),
        min_det_score=float(args.osr_min_det_score),
        min_memory_sim=float(args.osr_min_memory_sim),
        max_center_step=float(args.osr_max_center_step),
        min_margin=float(args.osr_min_margin),
        max_ambiguity=int(args.osr_max_ambiguity),
        min_score=float(args.osr_min_score),
    )


def build_confirm(args: argparse.Namespace, enable: bool) -> confirm_mod.ConfirmConfig:
    return confirm_mod.ConfirmConfig(
        enable=bool(enable),
        min_hits=int(args.confirm_min_hits),
        max_miss=1,
        max_age=5,
        min_avg_score=float(args.confirm_min_avg_score),
        min_avg_memory=float(args.confirm_min_avg_memory),
        max_avg_ambiguity=float(args.confirm_max_avg_ambiguity),
        min_last_score=float(args.confirm_min_last_score),
        require_consecutive=bool(args.confirm_require_consecutive),
        active_iou=2.0,
        active_ioa=2.0,
        suppress_output=False,
        suppress_until_high=True,
        adaptive_release_gate=False,
        adaptive_min_confirmed=20,
        adaptive_min_release_ratio=0.55,
    )


def make_tracker(cfg: base.TrackerConfig, osr: osr_v1.OSRConfig, confirm: confirm_mod.ConfirmConfig) -> confirm_mod.ConfirmedOSRTracker:
    confirm_mod._CONFIRM = confirm
    return confirm_mod.ConfirmedOSRTracker(cfg, osr)


def update_with_counter(tracker, counter: int, boxes, scores, feats, det_ids):
    base.DMMTrack._count = int(counter)
    out = tracker.update(boxes, scores, feats, det_ids)
    return out, int(base.DMMTrack._count)


def main() -> None:
    args = parse_args()
    dump = base.load_dump(Path(args.dump_npz))
    det = np.asarray(dump['detections'], dtype=np.float32)
    feat = np.asarray(dump['features'], dtype=np.float32)
    offsets = np.asarray(dump['frame_offsets'], dtype=np.int64)
    columns = [str(x) for x in dump['columns'].tolist()]
    col = {name: i for i, name in enumerate(columns)}
    n_frames = len(offsets) - 1
    if args.limit_frames and args.limit_frames > 0:
        n_frames = min(n_frames, int(args.limit_frames))

    cfg = build_cfg(args)
    main_confirm = build_confirm(args, enable=True)
    shadow_confirm = build_confirm(args, enable=True)

    # Main starts with OSR off. Shadow always runs OSR for reliability probing.
    main_tracker = make_tracker(cfg, build_osr(args, enable=False), main_confirm)
    shadow_tracker = make_tracker(cfg, build_osr(args, enable=bool(args.shadow_enable)), shadow_confirm)
    main_count = 0
    shadow_count = 0

    state = 'warmup' if bool(args.shadow_enable) else 'enabled'
    decision_frame = -1
    decision_ratio = 0.0
    last_check_frame = -1
    last_check_ratio = 0.0
    last_check_confirmed = 0
    last_check_released = 0
    if not bool(args.shadow_enable):
        main_tracker.osr.enable = True

    mot_rows: List[Tuple[int, int, float, float, float, float, float]] = []
    frames_with_outputs = 0

    for frame in range(1, n_frames + 1):
        start = int(offsets[frame - 1]); end = int(offsets[frame])
        rows = det[start:end]; feats = feat[start:end]
        if rows.size:
            boxes = rows[:, [col['x1'], col['y1'], col['x2'], col['y2']]].astype(np.float32)
            frame_scores = rows[:, col['score']].astype(np.float32)
            det_ids = rows[:, col['global_det_idx']].astype(np.int64)
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            frame_scores = np.zeros((0,), dtype=np.float32)
            det_ids = np.zeros((0,), dtype=np.int64)
            feats = np.zeros((0, feat.shape[1] if feat.ndim == 2 else 0), dtype=np.float32)

        # Main output is what we evaluate.
        online_targets, main_count = update_with_counter(main_tracker, main_count, boxes, frame_scores, feats, det_ids)

        # Shadow runs in an isolated ID counter space; its output is ignored.
        if bool(args.shadow_enable) and state == 'warmup':
            _shadow_out, shadow_count = update_with_counter(shadow_tracker, shadow_count, boxes, frame_scores, feats, det_ids)
            confirmed = int(shadow_tracker.osr_stats.get('proposal_confirmed', 0))
            released = int(shadow_tracker.osr_stats.get('pending_high_released', 0))
            ratio = float(released) / max(1.0, float(confirmed))
            max_observe_frame = int(args.shadow_max_observe_frame or args.shadow_decision_max_frame or 0)
            if confirmed >= int(args.shadow_min_confirmed):
                last_check_frame = int(frame)
                last_check_ratio = float(ratio)
                last_check_confirmed = int(confirmed)
                last_check_released = int(released)
                if ratio >= float(args.shadow_min_release_ratio):
                    decision_frame = int(frame)
                    decision_ratio = float(ratio)
                    main_tracker.osr.enable = True
                    state = 'enabled'
                else:
                    if bool(args.shadow_keep_observing_on_fail) and (max_observe_frame <= 0 or frame < max_observe_frame):
                        main_tracker.osr.enable = False
                        state = 'warmup'
                    else:
                        decision_frame = int(frame)
                        decision_ratio = float(ratio)
                        main_tracker.osr.enable = False
                        state = 'disabled'
            elif max_observe_frame > 0 and frame >= max_observe_frame:
                last_check_frame = int(frame)
                last_check_ratio = float(ratio)
                last_check_confirmed = int(confirmed)
                last_check_released = int(released)
                decision_frame = int(frame)
                decision_ratio = float(ratio)
                if confirmed >= int(args.shadow_min_confirmed) and ratio >= float(args.shadow_min_release_ratio):
                    main_tracker.osr.enable = True
                    state = 'enabled'
                else:
                    main_tracker.osr.enable = False
                    state = 'disabled' if bool(args.shadow_disable_on_fail) or bool(args.shadow_keep_observing_on_fail) else 'warmup'

        frame_output_count = 0
        for track in online_targets:
            tlwh = track.tlwh
            x, y, w, h = [float(v) for v in tlwh]
            if w * h < float(args.min_box_area):
                continue
            if w <= 0 or h <= 0:
                continue
            if w / max(h, 1e-12) > float(args.aspect_ratio_thresh):
                continue
            mot_rows.append((frame, int(track.track_id), x, y, w, h, float(track.score)))
            frame_output_count += 1
        if frame_output_count > 0:
            frames_with_outputs += 1

    base.write_mot_results(Path(args.out), mot_rows)
    summary = {
        'seq': args.seq,
        'frames': int(n_frames),
        'rows': len(mot_rows),
        'frames_with_outputs': frames_with_outputs,
        'unique_tracks': len({r[1] for r in mot_rows}),
        'config': vars(args),
        'warmup_state': state,
        'decision_frame': decision_frame,
        'decision_ratio': decision_ratio,
        'last_check_frame': last_check_frame,
        'last_check_ratio': last_check_ratio,
        'last_check_confirmed': last_check_confirmed,
        'last_check_released': last_check_released,
        'main_osr_stats': main_tracker.osr_stats,
        'shadow_osr_stats': shadow_tracker.osr_stats,
        'main_osr_enabled_final': bool(main_tracker.osr.enable),
        'main_confirm': asdict(main_confirm),
        'shadow_confirm': asdict(shadow_confirm),
    }
    if args.summary_json:
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
