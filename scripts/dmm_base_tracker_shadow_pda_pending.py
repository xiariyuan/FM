#!/usr/bin/env python3
"""Pending extension for recovered ShadowPDA.

This file does not patch dmm_base_tracker_shadow_pda.py. It imports the recovered
contact+identity implementation and overrides only the ultra-low recovery stage.
Medium-quality low evidence is stored as pending shadow evidence and promoted
only if the next frame confirms it.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

import dmm_base_tracker_shadow_pda as spda


@dataclass
class PendingConfig:
    enable: bool = False
    min_det_score: float = 0.35
    max_det_score: float = 0.50
    min_identity_margin: float = 0.03
    min_contact_overlap: float = 0.20
    min_p_best: float = 0.60
    min_reliability: float = 0.65
    min_low_iou: float = 0.20
    max_low_center_step: float = 80.0
    max_lost_age: int = 6
    confirm_min_low_iou: float = 0.70
    confirm_max_center_step: float = 20.0
    confirm_min_det_score: float = 0.35
    confirm_max_gap: int = 1


class PendingShadowPDATracker(spda.ShadowPDATracker):
    def __init__(self, cfg: spda.base.TrackerConfig, osr: spda.osr_v1.OSRConfig, shadow: spda.ShadowPDAConfig, pending: PendingConfig):
        super().__init__(cfg, osr, shadow)
        self.pending = pending
        self.shadow_stats.update({
            'pending_candidates': 0,
            'pending_started': 0,
            'pending_promoted': 0,
            'pending_expired': 0,
        })

    def _pending_active(self, st) -> bool:
        return bool(getattr(st, 'pending_active', False))

    def _clear_pending(self, st) -> None:
        if self._pending_active(st):
            self.shadow_stats['pending_expired'] += 1
        setattr(st, 'pending_active', False)
        setattr(st, 'pending_hits', 0)
        setattr(st, 'pending_frame', -1)
        setattr(st, 'pending_det_idx', -1)

    def _try_ultra_low_recover(self, lost_tracks: Sequence[spda.base.DMMTrack], low_dets: Sequence[spda.base.DMMTrack], active_tracks: Sequence[spda.base.DMMTrack] | None = None) -> List[Tuple[int, spda.base.DMMTrack]]:
        if not self.shadow.enable or not self.shadow.ultra_low_enable or len(lost_tracks) == 0 or len(low_dets) == 0:
            return []
        active_tracks = active_tracks or []
        low_by_gid = {int(getattr(d, 'det_global_idx', -1)): d for d in low_dets}
        out: List[Tuple[int, spda.base.DMMTrack]] = []
        used_low: set[int] = set()
        for idx, track in enumerate(lost_tracks):
            tid = int(track.track_id)
            st = self.shadows.get(tid)
            if st is None or st.state != 'confirmed':
                continue
            gid = int(st.last_low_det_idx)
            det = low_by_gid.get(gid)
            if det is None or gid in used_low:
                continue
            self.shadow_stats['ultra_low_candidates'] += 1
            lost_age = int(self.frame_id - track.frame_id)
            mem, _smooth, _bank = self._memory_similarity(track, det)
            active_best_sim, _unused, active_best_tid, active_overlap_now = self._active_identity_competition(det, active_tracks, tid)
            identity_margin = float(mem - active_best_sim)

            p_active = self._pending_active(st)
            p_frame = int(getattr(st, 'pending_frame', -1))
            p_tlbr = np.asarray(getattr(st, 'pending_tlbr', det.tlbr), dtype=np.float32)
            p_gap = int(self.frame_id) - p_frame if p_active else 10**9
            p_iou = self._box_iou(p_tlbr, det.tlbr) if p_active else 0.0
            p_center = float(np.linalg.norm(self._tlbr_center(det.tlbr) - self._tlbr_center(p_tlbr))) if p_active else 1e6
            pending_confirm_ok = (
                bool(self.pending.enable)
                and p_active
                and p_gap <= int(self.pending.confirm_max_gap)
                and (p_iou >= float(self.pending.confirm_min_low_iou) or p_center <= float(self.pending.confirm_max_center_step))
                and float(det.score) >= float(self.pending.confirm_min_det_score)
                and identity_margin >= float(self.pending.min_identity_margin)
                and st.active_overlap >= float(self.pending.min_contact_overlap)
                and lost_age <= int(self.pending.max_lost_age)
            )

            checks = {
                'reliability': st.reliability >= float(self.shadow.ultra_reliability),
                'support': st.support_count >= int(self.shadow.ultra_min_support),
                'consecutive_support': st.consecutive_support >= int(self.shadow.ultra_min_consecutive_support),
                'p_best': st.p_best >= float(self.shadow.ultra_min_p_best),
                'identity_margin': identity_margin >= float(self.shadow.ultra_min_identity_margin),
                'contact_overlap': st.active_overlap >= float(self.shadow.ultra_min_contact_overlap),
                'entropy': st.entropy <= float(self.shadow.ultra_max_entropy),
                'margin': st.margin >= float(self.shadow.ultra_min_margin),
                'risk': st.risk <= float(self.shadow.ultra_max_risk),
                'active_overlap': st.active_overlap <= float(self.shadow.ultra_max_active_overlap),
                'low_consistency': (st.low_iou >= float(self.shadow.ultra_min_low_iou) or st.low_center_step <= float(self.shadow.ultra_max_low_center_step)),
                'avg_memory': st.avg_memory >= float(self.shadow.ultra_min_avg_memory),
                'lost_app': mem >= float(self.shadow.ultra_min_app),
                'det_score': float(det.score) >= float(self.shadow.ultra_min_det_score),
                'lost_age': lost_age <= int(self.shadow.ultra_max_lost_age),
            }
            direct_ok = all(bool(v) for v in checks.values())
            ok = bool(direct_ok or pending_confirm_ok)

            pending_candidate = (
                bool(self.pending.enable)
                and not ok
                and float(det.score) >= float(self.pending.min_det_score)
                and float(det.score) < float(self.pending.max_det_score)
                and identity_margin >= float(self.pending.min_identity_margin)
                and st.active_overlap >= float(self.pending.min_contact_overlap)
                and st.p_best >= float(self.pending.min_p_best)
                and st.reliability >= float(self.pending.min_reliability)
                and (st.low_iou >= float(self.pending.min_low_iou) or st.low_center_step <= float(self.pending.max_low_center_step))
                and lost_age <= int(self.pending.max_lost_age)
            )
            blocked = [k for k, v in checks.items() if not bool(v)]
            recovery_type = 'direct' if direct_ok else ('pending_promote' if pending_confirm_ok else '')

            self._record_shadow_event({
                'frame': int(self.frame_id),
                'track_id': int(tid),
                'det_global_idx': int(gid),
                'reliability': float(st.reliability),
                'existence_logit': float(st.existence_logit),
                'support_count': int(st.support_count),
                'consecutive_support': int(st.consecutive_support),
                'miss_count': int(st.miss_count),
                'last_support_frame': int(st.last_support_frame),
                'p_best': float(st.p_best),
                'p_second': float(st.p_second),
                'p_clutter': float(st.p_clutter),
                'entropy': float(st.entropy),
                'margin': float(st.margin),
                'risk': float(st.risk),
                'active_overlap': float(st.active_overlap),
                'ghost_iou': float(st.ghost_iou),
                'ghost_center_step': float(st.ghost_center_step),
                'low_iou': float(st.low_iou),
                'low_center_step': float(st.low_center_step),
                'active_overlap_current': float(active_overlap_now),
                'lost_app_sim': float(mem),
                'active_best_app_sim': float(active_best_sim),
                'identity_margin': float(identity_margin),
                'active_best_track_id': int(active_best_tid),
                'pending_active_before': int(p_active),
                'pending_gap': int(p_gap) if p_gap < 10**8 else -1,
                'pending_low_iou': float(p_iou),
                'pending_center_step': float(p_center),
                'pending_candidate': int(bool(pending_candidate)),
                'pending_confirm_ok': int(bool(pending_confirm_ok)),
                'recovery_type': recovery_type,
                'avg_memory': float(st.avg_memory),
                'avg_det_score': float(st.avg_det_score),
                'det_score': float(det.score),
                'lost_age': int(lost_age),
                'blocked_reason': '|'.join(blocked),
                'recovered': int(ok),
            })

            if ok:
                out.append((idx, det))
                used_low.add(gid)
                if pending_confirm_ok:
                    self.shadow_stats['pending_promoted'] += 1
                setattr(st, 'pending_active', False)
                setattr(st, 'pending_hits', 0)
                self.shadow_stats['ultra_low_recovered'] += 1
                self.shadow_stats['low_public_recovered'] += 1
            else:
                if pending_candidate:
                    self.shadow_stats['pending_candidates'] += 1
                    if not p_active:
                        self.shadow_stats['pending_started'] += 1
                        setattr(st, 'pending_hits', 1)
                    else:
                        setattr(st, 'pending_hits', int(getattr(st, 'pending_hits', 0)) + 1)
                    setattr(st, 'pending_active', True)
                    setattr(st, 'pending_frame', int(self.frame_id))
                    setattr(st, 'pending_det_idx', int(gid))
                    setattr(st, 'pending_tlbr', np.asarray(det.tlbr, dtype=np.float32).copy())
                    setattr(st, 'pending_score', float(det.score))
                    setattr(st, 'pending_identity_margin', float(identity_margin))
                    setattr(st, 'pending_contact_overlap', float(st.active_overlap))
                elif p_active and p_gap > int(self.pending.confirm_max_gap):
                    self._clear_pending(st)
                self.shadow_stats['ultra_low_blocked'] += 1
        return out


def parse_args() -> tuple[argparse.Namespace, PendingConfig]:
    pending_parser = argparse.ArgumentParser(add_help=False)
    pending_parser.add_argument('--shadow-pending-enable', action='store_true')
    pending_parser.add_argument('--shadow-pending-min-det-score', type=float, default=0.35)
    pending_parser.add_argument('--shadow-pending-max-det-score', type=float, default=0.50)
    pending_parser.add_argument('--shadow-pending-min-identity-margin', type=float, default=0.03)
    pending_parser.add_argument('--shadow-pending-min-contact-overlap', type=float, default=0.20)
    pending_parser.add_argument('--shadow-pending-min-p-best', type=float, default=0.60)
    pending_parser.add_argument('--shadow-pending-min-reliability', type=float, default=0.65)
    pending_parser.add_argument('--shadow-pending-min-low-iou', type=float, default=0.20)
    pending_parser.add_argument('--shadow-pending-max-low-center-step', type=float, default=80.0)
    pending_parser.add_argument('--shadow-pending-max-lost-age', type=int, default=6)
    pending_parser.add_argument('--shadow-pending-confirm-min-low-iou', type=float, default=0.70)
    pending_parser.add_argument('--shadow-pending-confirm-max-center-step', type=float, default=20.0)
    pending_parser.add_argument('--shadow-pending-confirm-min-det-score', type=float, default=0.35)
    pending_parser.add_argument('--shadow-pending-confirm-max-gap', type=int, default=1)
    p_args, remaining = pending_parser.parse_known_args()
    old_argv = sys.argv
    try:
        sys.argv = [sys.argv[0]] + remaining
        args = spda.parse_args()
    finally:
        sys.argv = old_argv
    pending = PendingConfig(
        enable=bool(p_args.shadow_pending_enable),
        min_det_score=float(p_args.shadow_pending_min_det_score),
        max_det_score=float(p_args.shadow_pending_max_det_score),
        min_identity_margin=float(p_args.shadow_pending_min_identity_margin),
        min_contact_overlap=float(p_args.shadow_pending_min_contact_overlap),
        min_p_best=float(p_args.shadow_pending_min_p_best),
        min_reliability=float(p_args.shadow_pending_min_reliability),
        min_low_iou=float(p_args.shadow_pending_min_low_iou),
        max_low_center_step=float(p_args.shadow_pending_max_low_center_step),
        max_lost_age=int(p_args.shadow_pending_max_lost_age),
        confirm_min_low_iou=float(p_args.shadow_pending_confirm_min_low_iou),
        confirm_max_center_step=float(p_args.shadow_pending_confirm_max_center_step),
        confirm_min_det_score=float(p_args.shadow_pending_confirm_min_det_score),
        confirm_max_gap=int(p_args.shadow_pending_confirm_max_gap),
    )
    return args, pending


def make_tracker(args: argparse.Namespace, pending: PendingConfig) -> PendingShadowPDATracker:
    cfg = spda.base.TrackerConfig(
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
    osr = spda.osr_v1.OSRConfig(
        enable=bool(args.osr_enable),
        max_lost_age=int(args.osr_max_lost_age),
        min_track_len=int(args.osr_min_track_len),
        min_det_score=float(args.osr_min_det_score),
        min_memory_sim=float(args.osr_min_memory_sim),
        max_center_step=float(args.osr_max_center_step),
        max_area_ratio=float(args.osr_max_area_ratio),
        max_height_ratio=float(args.osr_max_height_ratio),
        min_margin=float(args.osr_min_margin),
        max_ambiguity=int(args.osr_max_ambiguity),
        min_score=float(args.osr_min_score),
        rank_slack=float(args.osr_rank_slack),
    )
    shadow = spda.ShadowPDAConfig(
        enable=bool(args.shadow_enable),
        rho=float(args.shadow_rho),
        tau=float(args.shadow_tau),
        evidence_center=float(args.shadow_evidence_center),
        evidence_scale=float(args.shadow_evidence_scale),
        clutter_logit=float(args.shadow_clutter_logit),
        clutter_penalty=float(args.shadow_clutter_penalty),
        miss_penalty=float(args.shadow_miss_penalty),
        entropy_penalty=float(args.shadow_entropy_penalty),
        margin_penalty=float(args.shadow_margin_penalty),
        margin_min=float(args.shadow_margin_min),
        active_overlap_penalty=float(args.shadow_active_overlap_penalty),
        tentative_reliability=float(args.shadow_tentative_reliability),
        confirm_reliability=float(args.shadow_confirm_reliability),
        delete_logit=float(args.shadow_delete_logit),
        min_support_prob=float(args.shadow_min_support_prob),
        min_support_confirm=int(args.shadow_min_support_confirm),
        max_entropy_confirm=float(args.shadow_max_entropy_confirm),
        min_margin_confirm=float(args.shadow_min_margin_confirm),
        max_shadow_age=int(args.shadow_max_age),
        max_shadow_miss=int(args.shadow_max_miss),
        high_score_thresh=float(args.shadow_high_score_thresh),
        high_min_app=float(args.shadow_high_min_app),
        high_min_score=float(args.shadow_high_min_score),
        high_max_center_step=float(args.shadow_high_max_center_step),
        high_fusion_thresh=float(args.shadow_high_fusion_thresh),
        primary_guidance_enable=bool(args.shadow_primary_guidance_enable),
        active_guard_cost=float(args.shadow_active_guard_cost),
        guidance_margin=float(args.shadow_guidance_margin),
        ultra_low_enable=bool(args.shadow_ultra_low_enable),
        ultra_reliability=float(args.shadow_ultra_reliability),
        ultra_min_det_score=float(args.shadow_ultra_min_det_score),
        ultra_min_app=float(args.shadow_ultra_min_app),
        ultra_min_avg_memory=float(args.shadow_ultra_min_avg_memory),
        ultra_min_p_best=float(args.shadow_ultra_min_p_best),
        ultra_min_identity_margin=float(args.shadow_ultra_min_identity_margin),
        ultra_min_contact_overlap=float(args.shadow_ultra_min_contact_overlap),
        ultra_max_entropy=float(args.shadow_ultra_max_entropy),
        ultra_min_margin=float(args.shadow_ultra_min_margin),
        ultra_max_risk=float(args.shadow_ultra_max_risk),
        ultra_max_active_overlap=float(args.shadow_ultra_max_active_overlap),
        ultra_min_support=int(args.shadow_ultra_min_support),
        ultra_min_consecutive_support=int(args.shadow_ultra_min_consecutive_support),
        ultra_min_ghost_iou=float(args.shadow_ultra_min_ghost_iou),
        ultra_max_ghost_center_step=float(args.shadow_ultra_max_ghost_center_step),
        ultra_min_low_iou=float(args.shadow_ultra_min_low_iou),
        ultra_max_low_center_step=float(args.shadow_ultra_max_low_center_step),
        ultra_max_lost_age=int(args.shadow_ultra_max_lost_age),
    )
    return PendingShadowPDATracker(cfg, osr, shadow, pending)


def main() -> None:
    args, pending = parse_args()
    dump = spda.base.load_dump(Path(args.dump_npz))
    det = np.asarray(dump['detections'], dtype=np.float32)
    feat = np.asarray(dump['features'], dtype=np.float32)
    offsets = np.asarray(dump['frame_offsets'], dtype=np.int64)
    columns = [str(x) for x in dump['columns'].tolist()]
    col = {name: i for i, name in enumerate(columns)}
    n_frames = len(offsets) - 1
    if args.limit_frames and args.limit_frames > 0:
        n_frames = min(n_frames, int(args.limit_frames))
    tracker = make_tracker(args, pending)
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
    spda.base.write_mot_results(Path(args.out), mot_rows)
    summary = {
        'seq': args.seq,
        'frames': int(n_frames),
        'rows': len(mot_rows),
        'frames_with_outputs': frames_with_outputs,
        'unique_tracks': len({r[1] for r in mot_rows}),
        'config': vars(args),
        'pending': asdict(pending),
        'osr': asdict(tracker.osr),
        'shadow': asdict(tracker.shadow),
        'osr_stats': tracker.osr_stats,
        'shadow_stats': tracker.shadow_stats,
        'active_shadow_states': len(tracker.shadows),
    }
    if args.summary_json:
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    if args.debug_csv:
        spda.base.write_debug(Path(args.debug_csv), tracker.debug_rows)
    if args.shadow_events_csv:
        out_csv = Path(args.shadow_events_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({k for row in tracker.shadow_event_rows for k in row.keys()})
        with out_csv.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(tracker.shadow_event_rows)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
