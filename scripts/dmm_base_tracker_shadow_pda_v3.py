#!/usr/bin/env python3
"""ShadowPDA-V3: decoupled state and output posteriors.

V3 separates low-confidence evidence into two online decisions:
1) state posterior: whether the lost identity likely still exists;
2) output posterior: whether the current low box is valid enough to publish.
Only when both are high does ShadowPDA perform public recovery.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

import dmm_base_tracker_shadow_pda as spda


@dataclass
class V3Config:
    theta_public: float = 1.85
    theta_state: float = 2.05
    theta_output: float = 1.25
    w_reliability: float = 1.0
    w_identity: float = 1.2
    w_app: float = 0.8
    w_det: float = 0.7
    w_contact_identity: float = 0.8
    w_temporal: float = 0.8
    w_loc: float = 0.6
    w_loc_risk: float = 0.6
    w_state_loc: float = 0.25
    w_output_det: float = 1.0
    w_output_loc: float = 0.8
    w_output_temporal: float = 0.6
    w_output_contact: float = 0.4
    w_output_det_weak: float = 1.0
    w_output_clutter: float = 0.8
    w_output_loc_risk: float = 0.6
    w_output_no_contact: float = 0.4
    w_active: float = 0.9
    w_clutter: float = 1.0
    w_entropy: float = 0.6
    w_det_weak: float = 0.8
    w_temporal_weak: float = 0.5
    w_age: float = 0.4
    id_scale: float = 0.30
    det_ref: float = 0.50
    max_motion_center_step: float = 120.0
    loc_aspect_scale: float = 0.7
    max_lost_age: int = 6
    min_support: int = 2
    min_consecutive_support: int = 2
    min_temporal_iou: float = 0.20
    max_temporal_center_step: float = 80.0
    min_lost_app: float = 0.75
    min_avg_memory: float = 0.75
    min_identity_margin: float = 0.03
    min_contact_context: float = 0.20


def clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


class ShadowPDAV3Tracker(spda.ShadowPDATracker):
    def __init__(self, cfg, osr, shadow, v2: V3Config):
        super().__init__(cfg, osr, shadow)
        self.v2 = v2
        self.shadow_stats.update({
            'v3_public_candidates': 0,
            'v3_public_recovered': 0,
            'v3_public_blocked': 0,
        })

    def _temporal_score(self, st) -> float:
        iou_score = clip01(float(st.low_iou))
        center_score = 1.0 - clip01(float(st.low_center_step) / max(1e-6, float(self.v2.max_temporal_center_step)))
        return float(max(iou_score, center_score))

    def _localization_score(self, track, st, det) -> tuple[float, dict[str, float]]:
        tbox = np.asarray(track.tlbr, dtype=np.float32)
        dbox = np.asarray(det.tlbr, dtype=np.float32)
        tc = self._tlbr_center(tbox)
        dc = self._tlbr_center(dbox)
        center_dist = float(np.linalg.norm(dc - tc))
        z_motion = 1.0 - clip01(center_dist / max(1e-6, float(self.v2.max_motion_center_step)))

        th = max(1e-6, float(tbox[3] - tbox[1]))
        dh = max(1e-6, float(dbox[3] - dbox[1]))
        tw = max(1e-6, float(tbox[2] - tbox[0]))
        dw = max(1e-6, float(dbox[2] - dbox[0]))
        tarea = max(1e-6, tw * th)
        darea = max(1e-6, dw * dh)
        z_height = float(np.exp(-abs(np.log(dh / th))))
        z_area = float(np.exp(-abs(np.log(darea / tarea))))
        z_size = 0.5 * z_height + 0.5 * z_area

        tar = tw / th
        dar = dw / dh
        z_aspect = float(np.exp(-abs(np.log(dar / max(1e-6, tar))) / max(1e-6, float(self.v2.loc_aspect_scale))))
        z_low_to_low = self._temporal_score(st)
        z_localization = 0.4 * z_low_to_low + 0.3 * z_motion + 0.2 * z_size + 0.1 * z_aspect
        z_localization = clip01(z_localization)
        return z_localization, {
            'z_localization': float(z_localization),
            'z_loc_risk': float(1.0 - z_localization),
            'z_loc_low_to_low': float(z_low_to_low),
            'z_loc_motion': float(z_motion),
            'z_loc_size': float(z_size),
            'z_loc_aspect': float(z_aspect),
            'loc_center_dist': float(center_dist),
        }

    def _public_score(
        self,
        st,
        det,
        lost_age: int,
        lost_app: float,
        identity_margin: float,
        active_best_sim: float,
        active_overlap_now: float,
        track,
    ) -> tuple[float, dict[str, float]]:
        temporal = self._temporal_score(st)
        loc_score, loc_terms = self._localization_score(track, st, det)
        identity_dominance = clip01(float(identity_margin) / max(1e-6, float(self.v2.id_scale)))
        det_score = clip01(float(det.score))
        det_weakness = clip01((float(self.v2.det_ref) - det_score) / max(1e-6, float(self.v2.det_ref)))
        temporal_weakness = 1.0 - temporal
        competition_uncertainty = (1.0 - identity_dominance) ** 2
        active_risk = clip01(float(active_overlap_now)) * clip01(float(active_best_sim)) * competition_uncertainty
        contact_identity = clip01(float(st.active_overlap)) * identity_dominance
        clutter_risk = clip01(float(st.p_clutter)) + clip01(float(st.entropy)) + det_weakness + temporal_weakness
        z = {
            'z_reliability': clip01(float(st.reliability)),
            'z_identity_dominance': float(identity_dominance),
            'z_app': clip01(float(lost_app)),
            'z_det': float(det_score),
            'z_contact': clip01(float(st.active_overlap)),
            'z_contact_identity': float(contact_identity),
            'z_temporal': float(temporal),
            'z_clutter': clip01(float(st.p_clutter)),
            'z_entropy': clip01(float(st.entropy)),
            'z_det_weakness': float(det_weakness),
            'z_temporal_weakness': float(temporal_weakness),
            'z_competition_uncertainty': float(competition_uncertainty),
            'z_active_risk': float(active_risk),
            'z_clutter_risk': float(clutter_risk),
            'z_age': clip01(float(lost_age) / max(1, int(self.v2.max_lost_age))),
        }
        z.update(loc_terms)
        state_score = (
            float(self.v2.w_reliability) * z['z_reliability']
            + float(self.v2.w_identity) * z['z_identity_dominance']
            + float(self.v2.w_app) * z['z_app']
            + float(self.v2.w_contact_identity) * z['z_contact_identity']
            + float(self.v2.w_temporal) * z['z_temporal']
            + float(self.v2.w_state_loc) * z['z_localization']
            - float(self.v2.w_active) * z['z_active_risk']
            - float(self.v2.w_entropy) * z['z_entropy']
            - float(self.v2.w_age) * z['z_age']
        )
        output_score = (
            float(self.v2.w_output_det) * z['z_det']
            + float(self.v2.w_output_loc) * z['z_localization']
            + float(self.v2.w_output_temporal) * z['z_temporal']
            + float(self.v2.w_output_contact) * z['z_contact']
            - float(self.v2.w_output_det_weak) * z['z_det_weakness']
            - float(self.v2.w_output_clutter) * z['z_clutter']
            - float(self.v2.w_output_loc_risk) * z['z_loc_risk']
            - float(self.v2.w_output_no_contact) * (1.0 - z['z_contact'])
        )
        # Keep public_score as the limiting posterior for ranking/logging.
        score = min(float(state_score - float(self.v2.theta_state)), float(output_score - float(self.v2.theta_output))) + float(self.v2.theta_public)
        z['state_score'] = float(state_score)
        z['output_score'] = float(output_score)
        z['state_margin'] = float(state_score - float(self.v2.theta_state))
        z['output_margin'] = float(output_score - float(self.v2.theta_output))
        return float(score), z

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
            self.shadow_stats['v3_public_candidates'] += 1

            lost_age = int(self.frame_id - track.frame_id)
            lost_app, _smooth, _bank = self._memory_similarity(track, det)
            active_best_sim, _unused, active_best_tid, active_overlap_now = self._active_identity_competition(det, active_tracks, tid)
            identity_margin = float(lost_app - active_best_sim)
            public_score, z = self._public_score(st, det, lost_age, lost_app, identity_margin, active_best_sim, active_overlap_now, track)

            safety = {
                'lost_age': lost_age <= int(self.v2.max_lost_age),
                'support': st.support_count >= int(self.v2.min_support),
                'consecutive_support': st.consecutive_support >= int(self.v2.min_consecutive_support),
                'temporal': (st.low_iou >= float(self.v2.min_temporal_iou) or st.low_center_step <= float(self.v2.max_temporal_center_step)),
                'lost_app': float(lost_app) >= float(self.v2.min_lost_app),
                'avg_memory': float(st.avg_memory) >= float(self.v2.min_avg_memory),
                'identity_margin': identity_margin >= float(self.v2.min_identity_margin),
                'contact_context': float(st.active_overlap) >= float(self.v2.min_contact_context),
            }
            ok = bool(public_score >= float(self.v2.theta_public) and z.get('state_score', 0.0) >= float(self.v2.theta_state) and z.get('output_score', 0.0) >= float(self.v2.theta_output) and all(bool(v) for v in safety.values()))
            blocked = [k for k, v in safety.items() if not bool(v)]
            if public_score < float(self.v2.theta_public):
                blocked.append('public_score')
            if z.get('state_score', 0.0) < float(self.v2.theta_state):
                blocked.append('state_score')
            if z.get('output_score', 0.0) < float(self.v2.theta_output):
                blocked.append('output_score')

            event = {
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
                'active_overlap_current': float(active_overlap_now),
                'lost_app_sim': float(lost_app),
                'active_best_app_sim': float(active_best_sim),
                'identity_margin': float(identity_margin),
                'active_best_track_id': int(active_best_tid),
                'avg_memory': float(st.avg_memory),
                'avg_det_score': float(st.avg_det_score),
                'det_score': float(det.score),
                'lost_age': int(lost_age),
                'ghost_iou': float(st.ghost_iou),
                'ghost_center_step': float(st.ghost_center_step),
                'low_iou': float(st.low_iou),
                'low_center_step': float(st.low_center_step),
                'public_score': float(public_score),
                'theta_public': float(self.v2.theta_public),
                'blocked_reason': '|'.join(blocked),
                'recovered': int(ok),
            }
            event.update(z)
            self._record_shadow_event(event)

            if ok:
                out.append((idx, det))
                used_low.add(gid)
                self.shadow_stats['ultra_low_recovered'] += 1
                self.shadow_stats['low_public_recovered'] += 1
                self.shadow_stats['v3_public_recovered'] += 1
            else:
                self.shadow_stats['ultra_low_blocked'] += 1
                self.shadow_stats['v3_public_blocked'] += 1
        return out


def parse_args() -> tuple[argparse.Namespace, V3Config]:
    v2_parser = argparse.ArgumentParser(add_help=False)
    v2_parser.add_argument('--v2-theta-public', type=float, default=1.85)
    v2_parser.add_argument('--v3-theta-state', type=float, default=2.05)
    v2_parser.add_argument('--v3-theta-output', type=float, default=1.25)
    v2_parser.add_argument('--v2-w-reliability', type=float, default=1.0)
    v2_parser.add_argument('--v2-w-identity', type=float, default=1.2)
    v2_parser.add_argument('--v2-w-app', type=float, default=0.8)
    v2_parser.add_argument('--v2-w-det', type=float, default=0.7)
    v2_parser.add_argument('--v2-w-contact-identity', type=float, default=0.8)
    v2_parser.add_argument('--v2-w-temporal', type=float, default=0.8)
    v2_parser.add_argument('--v2-w-loc', type=float, default=0.6)
    v2_parser.add_argument('--v2-w-loc-risk', type=float, default=0.6)
    v2_parser.add_argument('--v3-w-state-loc', type=float, default=0.25)
    v2_parser.add_argument('--v3-w-output-det', type=float, default=1.0)
    v2_parser.add_argument('--v3-w-output-loc', type=float, default=0.8)
    v2_parser.add_argument('--v3-w-output-temporal', type=float, default=0.6)
    v2_parser.add_argument('--v3-w-output-contact', type=float, default=0.4)
    v2_parser.add_argument('--v3-w-output-det-weak', type=float, default=1.0)
    v2_parser.add_argument('--v3-w-output-clutter', type=float, default=0.8)
    v2_parser.add_argument('--v3-w-output-loc-risk', type=float, default=0.6)
    v2_parser.add_argument('--v3-w-output-no-contact', type=float, default=0.4)
    v2_parser.add_argument('--v2-w-active', type=float, default=0.9)
    v2_parser.add_argument('--v2-w-clutter', type=float, default=1.0)
    v2_parser.add_argument('--v2-w-entropy', type=float, default=0.6)
    v2_parser.add_argument('--v2-w-det-weak', type=float, default=0.8)
    v2_parser.add_argument('--v2-w-temporal-weak', type=float, default=0.5)
    v2_parser.add_argument('--v2-w-age', type=float, default=0.4)
    v2_parser.add_argument('--v2-id-scale', type=float, default=0.30)
    v2_parser.add_argument('--v2-det-ref', type=float, default=0.50)
    v2_parser.add_argument('--v2-max-motion-center-step', type=float, default=120.0)
    v2_parser.add_argument('--v2-loc-aspect-scale', type=float, default=0.7)
    v2_parser.add_argument('--v2-max-lost-age', type=int, default=6)
    v2_parser.add_argument('--v2-min-support', type=int, default=2)
    v2_parser.add_argument('--v2-min-consecutive-support', type=int, default=2)
    v2_parser.add_argument('--v2-min-temporal-iou', type=float, default=0.20)
    v2_parser.add_argument('--v2-max-temporal-center-step', type=float, default=80.0)
    v2_parser.add_argument('--v2-min-lost-app', type=float, default=0.75)
    v2_parser.add_argument('--v2-min-avg-memory', type=float, default=0.75)
    v2_parser.add_argument('--v2-min-identity-margin', type=float, default=0.03)
    v2_parser.add_argument('--v2-min-contact-context', type=float, default=0.20)
    v2_args, remaining = v2_parser.parse_known_args()
    old_argv = sys.argv
    try:
        sys.argv = [sys.argv[0]] + remaining
        args = spda.parse_args()
    finally:
        sys.argv = old_argv
    v2 = V3Config(
        theta_public=float(v2_args.v2_theta_public),
        theta_state=float(v2_args.v3_theta_state),
        theta_output=float(v2_args.v3_theta_output),
        w_reliability=float(v2_args.v2_w_reliability),
        w_identity=float(v2_args.v2_w_identity),
        w_app=float(v2_args.v2_w_app),
        w_det=float(v2_args.v2_w_det),
        w_contact_identity=float(v2_args.v2_w_contact_identity),
        w_temporal=float(v2_args.v2_w_temporal),
        w_loc=float(v2_args.v2_w_loc),
        w_loc_risk=float(v2_args.v2_w_loc_risk),
        w_state_loc=float(v2_args.v3_w_state_loc),
        w_output_det=float(v2_args.v3_w_output_det),
        w_output_loc=float(v2_args.v3_w_output_loc),
        w_output_temporal=float(v2_args.v3_w_output_temporal),
        w_output_contact=float(v2_args.v3_w_output_contact),
        w_output_det_weak=float(v2_args.v3_w_output_det_weak),
        w_output_clutter=float(v2_args.v3_w_output_clutter),
        w_output_loc_risk=float(v2_args.v3_w_output_loc_risk),
        w_output_no_contact=float(v2_args.v3_w_output_no_contact),
        w_active=float(v2_args.v2_w_active),
        w_clutter=float(v2_args.v2_w_clutter),
        w_entropy=float(v2_args.v2_w_entropy),
        w_det_weak=float(v2_args.v2_w_det_weak),
        w_temporal_weak=float(v2_args.v2_w_temporal_weak),
        w_age=float(v2_args.v2_w_age),
        id_scale=float(v2_args.v2_id_scale),
        det_ref=float(v2_args.v2_det_ref),
        max_motion_center_step=float(v2_args.v2_max_motion_center_step),
        loc_aspect_scale=float(v2_args.v2_loc_aspect_scale),
        max_lost_age=int(v2_args.v2_max_lost_age),
        min_support=int(v2_args.v2_min_support),
        min_consecutive_support=int(v2_args.v2_min_consecutive_support),
        min_temporal_iou=float(v2_args.v2_min_temporal_iou),
        max_temporal_center_step=float(v2_args.v2_max_temporal_center_step),
        min_lost_app=float(v2_args.v2_min_lost_app),
        min_avg_memory=float(v2_args.v2_min_avg_memory),
        min_identity_margin=float(v2_args.v2_min_identity_margin),
        min_contact_context=float(v2_args.v2_min_contact_context),
    )
    return args, v2


def make_tracker(args: argparse.Namespace, v2: V3Config) -> ShadowPDAV3Tracker:
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
    return ShadowPDAV3Tracker(cfg, osr, shadow, v2)


def main() -> None:
    args, v2 = parse_args()
    dump = spda.base.load_dump(Path(args.dump_npz))
    det = np.asarray(dump['detections'], dtype=np.float32)
    feat = np.asarray(dump['features'], dtype=np.float32)
    offsets = np.asarray(dump['frame_offsets'], dtype=np.int64)
    columns = [str(x) for x in dump['columns'].tolist()]
    col = {name: i for i, name in enumerate(columns)}
    n_frames = len(offsets) - 1
    if args.limit_frames and args.limit_frames > 0:
        n_frames = min(n_frames, int(args.limit_frames))
    tracker = make_tracker(args, v2)
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
        'v3': asdict(v2),
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
            writer.writeheader(); writer.writerows(tracker.shadow_event_rows)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
