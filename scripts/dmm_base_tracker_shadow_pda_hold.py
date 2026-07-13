#!/usr/bin/env python3
"""ShadowPDA-Hold experiment.

Medium-quality low evidence is not forced into public recovery. Instead, when it
matches the same contact+identity conditions used by pending evidence, it can
hold the shadow state alive for a few frames and slightly reinforce existence.

This file intentionally imports the recovered contact+identity wrapper and the
pending extension. It does not patch either file.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

import dmm_base_tracker_shadow_pda_pending as pending_mod

spda = pending_mod.spda


@dataclass
class HoldConfig:
    enable: bool = False
    bonus_logit: float = 0.25
    max_logit: float = 2.0
    keep_frames: int = 3
    reset_miss: bool = True


class HoldShadowPDATracker(pending_mod.PendingShadowPDATracker):
    def __init__(self, cfg, osr, shadow, pending, hold: HoldConfig):
        super().__init__(cfg, osr, shadow, pending)
        self.hold = hold
        self.shadow_stats.update({
            'hold_candidates': 0,
            'hold_started': 0,
            'hold_updated': 0,
            'hold_expired': 0,
            'hold_preserved': 0,
        })

    def _clear_hold(self, st) -> None:
        if bool(getattr(st, 'hold_active', False)):
            self.shadow_stats['hold_expired'] += 1
        setattr(st, 'hold_active', False)
        setattr(st, 'hold_until_frame', -1)
        setattr(st, 'hold_hits', 0)

    def _start_or_update_hold(self, st, det, identity_margin: float) -> None:
        was_active = bool(getattr(st, 'hold_active', False))
        if was_active:
            self.shadow_stats['hold_updated'] += 1
        else:
            self.shadow_stats['hold_started'] += 1
        setattr(st, 'hold_active', True)
        setattr(st, 'hold_until_frame', int(self.frame_id) + int(self.hold.keep_frames))
        setattr(st, 'hold_hits', int(getattr(st, 'hold_hits', 0)) + 1)
        setattr(st, 'hold_tlbr', np.asarray(det.tlbr, dtype=np.float32).copy())
        setattr(st, 'hold_det_score', float(det.score))
        setattr(st, 'hold_identity_margin', float(identity_margin))
        st.existence_logit = min(float(self.hold.max_logit), float(st.existence_logit) + float(self.hold.bonus_logit))
        st.reliability = spda._sigmoid(float(st.existence_logit))
        if self.hold.reset_miss:
            st.miss_count = 0

    def _decay_shadow_miss(self, track) -> None:
        tid = int(track.track_id)
        st = self.shadows.get(tid)
        if st is not None and bool(getattr(st, 'hold_active', False)):
            if int(self.frame_id) <= int(getattr(st, 'hold_until_frame', -1)):
                st.age = int(self.frame_id - st.start_frame)
                st.last_frame = int(self.frame_id)
                st.miss_count = 0
                st.reliability = spda._sigmoid(float(st.existence_logit))
                self.shadow_stats['hold_preserved'] += 1
                return
            self._clear_hold(st)
        return super()._decay_shadow_miss(track)

    def _expire_shadows(self, alive_lost_ids: set[int]) -> None:
        stale: List[int] = []
        for tid, st in self.shadows.items():
            if tid not in alive_lost_ids:
                stale.append(int(tid))
                continue
            if bool(getattr(st, 'hold_active', False)) and int(self.frame_id) <= int(getattr(st, 'hold_until_frame', -1)):
                continue
            if bool(getattr(st, 'hold_active', False)):
                self._clear_hold(st)
            if st.age > int(self.shadow.max_shadow_age) or st.miss_count > int(self.shadow.max_shadow_miss) or st.existence_logit < float(self.shadow.delete_logit):
                stale.append(int(tid))
        for tid in stale:
            self.shadows.pop(tid, None)
            self.shadow_stats['shadow_deleted'] += 1

    def _try_ultra_low_recover(self, lost_tracks, low_dets, active_tracks=None) -> List[Tuple[int, object]]:
        before = len(self.shadow_event_rows)
        out = super()._try_ultra_low_recover(lost_tracks, low_dets, active_tracks)
        if not self.hold.enable or not self.pending.enable:
            return out
        low_by_gid = {int(getattr(d, 'det_global_idx', -1)): d for d in low_dets}
        for row in self.shadow_event_rows[before:]:
            if int(row.get('recovered', 0)) == 1:
                continue
            if int(row.get('pending_candidate', 0)) != 1:
                continue
            tid = int(row.get('track_id', -1))
            gid = int(row.get('det_global_idx', -1))
            st = self.shadows.get(tid)
            det = low_by_gid.get(gid)
            if st is None or det is None:
                continue
            self.shadow_stats['hold_candidates'] += 1
            self._start_or_update_hold(st, det, float(row.get('identity_margin', 0.0)))
            row['hold_candidate'] = 1
            row['hold_until_frame'] = int(getattr(st, 'hold_until_frame', -1))
            row['hold_hits'] = int(getattr(st, 'hold_hits', 0))
        return out


def parse_args():
    hold_parser = argparse.ArgumentParser(add_help=False)
    hold_parser.add_argument('--shadow-hold-enable', action='store_true')
    hold_parser.add_argument('--shadow-hold-bonus-logit', type=float, default=0.25)
    hold_parser.add_argument('--shadow-hold-max-logit', type=float, default=2.0)
    hold_parser.add_argument('--shadow-hold-keep-frames', type=int, default=3)
    hold_parser.add_argument('--shadow-hold-no-reset-miss', action='store_true')
    h_args, remaining = hold_parser.parse_known_args()
    old = sys.argv
    try:
        sys.argv = [sys.argv[0]] + remaining
        args, pending = pending_mod.parse_args()
    finally:
        sys.argv = old
    hold = HoldConfig(
        enable=bool(h_args.shadow_hold_enable),
        bonus_logit=float(h_args.shadow_hold_bonus_logit),
        max_logit=float(h_args.shadow_hold_max_logit),
        keep_frames=int(h_args.shadow_hold_keep_frames),
        reset_miss=not bool(h_args.shadow_hold_no_reset_miss),
    )
    return args, pending, hold


def make_tracker(args, pending, hold):
    temp = pending_mod.make_tracker(args, pending)
    return HoldShadowPDATracker(temp.cfg, temp.osr, temp.shadow, pending, hold)


def main() -> None:
    args, pending, hold = parse_args()
    dump = spda.base.load_dump(Path(args.dump_npz))
    det = np.asarray(dump['detections'], dtype=np.float32)
    feat = np.asarray(dump['features'], dtype=np.float32)
    offsets = np.asarray(dump['frame_offsets'], dtype=np.int64)
    columns = [str(x) for x in dump['columns'].tolist()]
    col = {name: i for i, name in enumerate(columns)}
    n_frames = len(offsets) - 1
    if args.limit_frames and args.limit_frames > 0:
        n_frames = min(n_frames, int(args.limit_frames))
    tracker = make_tracker(args, pending, hold)
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
        online_targets = tracker.update(boxes, frame_scores, feats, det_ids)
        frame_count = 0
        for track in online_targets:
            x, y, w, h = [float(v) for v in track.tlwh]
            if w * h < float(args.min_box_area) or w <= 0 or h <= 0:
                continue
            if w / max(h, 1e-12) > float(args.aspect_ratio_thresh):
                continue
            mot_rows.append((frame, int(track.track_id), x, y, w, h, float(track.score)))
            frame_count += 1
        if frame_count > 0:
            frames_with_outputs += 1
    spda.base.write_mot_results(Path(args.out), mot_rows)
    summary = {
        'seq': args.seq,
        'frames': int(n_frames),
        'rows': len(mot_rows),
        'frames_with_outputs': frames_with_outputs,
        'unique_tracks': len({r[1] for r in mot_rows}),
        'config': vars(args),
        'pending': asdict(pending),
        'hold': asdict(hold),
        'osr': asdict(tracker.osr),
        'shadow': asdict(tracker.shadow),
        'osr_stats': tracker.osr_stats,
        'shadow_stats': tracker.shadow_stats,
        'active_shadow_states': len(tracker.shadows),
    }
    if args.summary_json:
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    if args.shadow_events_csv:
        out_csv = Path(args.shadow_events_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        fields = sorted({k for row in tracker.shadow_event_rows for k in row.keys()})
        with out_csv.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader(); writer.writerows(tracker.shadow_event_rows)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
