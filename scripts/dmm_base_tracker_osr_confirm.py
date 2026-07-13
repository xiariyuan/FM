#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import dmm_base_tracker as base
import dmm_base_tracker_osr as osr_v1


@dataclass
class ConfirmConfig:
    enable: bool = True
    min_hits: int = 2
    max_miss: int = 1
    max_age: int = 5
    min_avg_score: float = 0.62
    min_avg_memory: float = 0.72
    max_avg_ambiguity: float = 0.28
    min_last_score: float = 0.58
    require_consecutive: bool = False
    active_iou: float = 2.0
    active_ioa: float = 2.0
    suppress_output: bool = False
    suppress_until_high: bool = False
    adaptive_release_gate: bool = False
    adaptive_min_confirmed: int = 20
    adaptive_min_release_ratio: float = 0.55
    motion_only: bool = False


@dataclass
class Proposal:
    track_id: int
    start_frame: int
    last_frame: int
    hits: int
    misses: int
    score_sum: float
    memory_sum: float
    ambiguity_sum: float
    last_det_global_idx: int
    last_low_index: int

    def avg_score(self) -> float:
        return self.score_sum / max(1, self.hits)

    def avg_memory(self) -> float:
        return self.memory_sum / max(1, self.hits)

    def avg_ambiguity(self) -> float:
        return self.ambiguity_sum / max(1, self.hits)


_CONFIRM = ConfirmConfig(enable=False)
_EVENTS_CSV = ""
_LAST_TRACKER = None


class ConfirmedOSRTracker(osr_v1.OSRTracker):
    def __init__(self, cfg: base.TrackerConfig, osr: osr_v1.OSRConfig):
        global _LAST_TRACKER
        super().__init__(cfg, osr)
        self.confirm = _CONFIRM
        self.osr_proposals: Dict[int, Proposal] = {}
        self.osr_event_rows: List[dict] = []
        _LAST_TRACKER = self
        self.osr_stats.update({
            'proposal_created': 0,
            'proposal_updated': 0,
            'proposal_confirmed': 0,
            'proposal_expired': 0,
            'proposal_blocked_avg_score': 0,
            'proposal_blocked_avg_memory': 0,
            'proposal_blocked_avg_ambiguity': 0,
            'proposal_blocked_last_score': 0,
            'proposal_blocked_active': 0,
            'pending_high_released': 0,
            'pending_high_hidden_frames': 0,
            'adaptive_release_blocked': 0,
        })

    def update(self, bboxes_xyxy, scores, features, det_global_ids):
        out = super().update(bboxes_xyxy, scores, features, det_global_ids)
        filtered = []
        for t in out:
            if bool(getattr(t, 'osr_pending_high_confirm', False)):
                if float(getattr(t, 'score', 0.0)) >= float(self.cfg.track_high_thresh):
                    setattr(t, 'osr_pending_high_confirm', False)
                    self.osr_stats['pending_high_released'] += 1
                    filtered.append(t)
                else:
                    self.osr_stats['pending_high_hidden_frames'] += 1
                    continue
            elif bool(self.confirm.suppress_output) and int(getattr(t, 'osr_suppress_output_frame', -1)) == int(self.frame_id):
                continue
            else:
                filtered.append(t)
        return filtered

    def _expire_stale(self) -> None:
        stale = []
        for tid, prop in self.osr_proposals.items():
            gap = int(self.frame_id - prop.last_frame)
            age = int(self.frame_id - prop.start_frame)
            if gap > int(self.confirm.max_miss) or age > int(self.confirm.max_age):
                stale.append(int(tid))
        for tid in stale:
            self.osr_proposals.pop(tid, None)
            self.osr_stats['proposal_expired'] += 1

    def _confirmed(self, prop: Proposal, pair: osr_v1.OSRPair) -> bool:
        if prop.hits < int(self.confirm.min_hits):
            return False
        if prop.avg_score() < float(self.confirm.min_avg_score):
            self.osr_stats['proposal_blocked_avg_score'] += 1
            return False
        if prop.avg_memory() < float(self.confirm.min_avg_memory):
            self.osr_stats['proposal_blocked_avg_memory'] += 1
            return False
        if prop.avg_ambiguity() > float(self.confirm.max_avg_ambiguity):
            self.osr_stats['proposal_blocked_avg_ambiguity'] += 1
            return False
        if float(pair.score) < float(self.confirm.min_last_score):
            self.osr_stats['proposal_blocked_last_score'] += 1
            return False
        return True


    def _active_conflict(self, det: base.DMMTrack, recover_tid: int) -> bool:
        # Safety gate: a low-score box should not recover a lost ID if it is
        # already strongly occupied by a currently active trajectory.
        for other in list(self.tracked_stracks):
            if int(getattr(other, 'track_id', -1)) == int(recover_tid):
                continue
            if getattr(other, 'state', None) != base.TrackState.Tracked:
                continue
            if not bool(getattr(other, 'is_activated', False)):
                continue
            iou = self._iou_one(other.tlbr, det.tlbr)
            ioa = self._ioa_min_one(other.tlbr, det.tlbr)
            if iou >= float(self.confirm.active_iou) or ioa >= float(self.confirm.active_ioa):
                return True
        return False

    def _osr_recover(
        self,
        lost_tracks: Sequence[base.DMMTrack],
        low_dets: Sequence[base.DMMTrack],
    ) -> Tuple[List[Tuple[int, int]], List[int], object, List[List[osr_v1.OSRPair]]]:
        cand_matches, u_low_v1, cost, pair_grid = super()._osr_recover(lost_tracks, low_dets)
        if not bool(self.confirm.enable):
            return cand_matches, u_low_v1, cost, pair_grid

        self._expire_stale()
        if bool(self.confirm.adaptive_release_gate):
            nconf = int(self.osr_stats.get('proposal_confirmed', 0))
            nrel = int(self.osr_stats.get('pending_high_released', 0))
            ratio = float(nrel) / max(1.0, float(nconf))
            if nconf >= int(self.confirm.adaptive_min_confirmed) and ratio < float(self.confirm.adaptive_min_release_ratio):
                self.osr_stats['adaptive_release_blocked'] += len(cand_matches)
                return [], list(range(len(low_dets))), cost, pair_grid
        confirmed: List[Tuple[int, int]] = []
        seen_tids: set[int] = set()

        for ilost, ilow in cand_matches:
            track = lost_tracks[int(ilost)]
            det = low_dets[int(ilow)]
            pair = pair_grid[int(ilost)][int(ilow)]
            tid = int(track.track_id)
            if self._active_conflict(det, tid):
                self.osr_stats['proposal_blocked_active'] += 1
                continue
            seen_tids.add(tid)
            prev = self.osr_proposals.get(tid)
            reset = prev is None
            if prev is not None:
                gap = int(self.frame_id - prev.last_frame)
                if bool(self.confirm.require_consecutive) and gap != 1:
                    reset = True
                elif gap > int(self.confirm.max_miss) + 1:
                    reset = True
            if reset:
                prop = Proposal(
                    tid, int(self.frame_id), int(self.frame_id), 1, 0,
                    float(pair.score), float(pair.memory_sim), float(pair.ambiguity_penalty),
                    int(getattr(det, 'det_global_idx', -1)), int(ilow),
                )
                self.osr_proposals[tid] = prop
                self.osr_stats['proposal_created'] += 1
            else:
                prop = prev
                prop.last_frame = int(self.frame_id)
                prop.hits += 1
                prop.misses = 0
                prop.score_sum += float(pair.score)
                prop.memory_sum += float(pair.memory_sim)
                prop.ambiguity_sum += float(pair.ambiguity_penalty)
                prop.last_det_global_idx = int(getattr(det, 'det_global_idx', -1))
                prop.last_low_index = int(ilow)
                self.osr_stats['proposal_updated'] += 1

            if self._confirmed(prop, pair):
                self.osr_event_rows.append({
                    'frame': int(self.frame_id),
                    'track_id': int(tid),
                    'det_global_idx': int(getattr(det, 'det_global_idx', -1)),
                    'low_index': int(ilow),
                    'score': float(pair.score),
                    'memory_sim': float(pair.memory_sim),
                    'ambiguity_penalty': float(pair.ambiguity_penalty),
                    'lost_age': int(pair.lost_age),
                    'track_age': int(pair.track_age),
                    'proposal_hits': int(prop.hits),
                    'proposal_avg_score': float(prop.avg_score()),
                    'proposal_avg_memory': float(prop.avg_memory()),
                    'proposal_avg_ambiguity': float(prop.avg_ambiguity()),
                })
                if bool(self.confirm.motion_only):
                    det.curr_feat = None
                if bool(self.confirm.suppress_until_high):
                    setattr(track, 'osr_pending_high_confirm', True)
                if bool(self.confirm.suppress_output):
                    setattr(track, 'osr_suppress_output_frame', int(self.frame_id))
                confirmed.append((int(ilost), int(ilow)))
                self.osr_proposals.pop(tid, None)
                self.osr_stats['proposal_confirmed'] += 1

        for tid, prop in list(self.osr_proposals.items()):
            if tid not in seen_tids and int(self.frame_id - prop.last_frame) > 0:
                prop.misses += 1
                if prop.misses > int(self.confirm.max_miss):
                    self.osr_proposals.pop(tid, None)
                    self.osr_stats['proposal_expired'] += 1

        used_low = {int(j) for _, j in confirmed}
        return confirmed, sorted(set(range(len(low_dets))) - used_low), cost, pair_grid


def parse_confirm_and_strip() -> ConfirmConfig:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--osr-confirm-enable', action='store_true')
    p.add_argument('--osr-confirm-min-hits', type=int, default=2)
    p.add_argument('--osr-proposal-max-miss', type=int, default=1)
    p.add_argument('--osr-proposal-max-age', type=int, default=5)
    p.add_argument('--osr-confirm-min-avg-score', type=float, default=0.62)
    p.add_argument('--osr-confirm-min-avg-memory', type=float, default=0.72)
    p.add_argument('--osr-confirm-max-avg-ambiguity', type=float, default=0.28)
    p.add_argument('--osr-confirm-min-last-score', type=float, default=0.58)
    p.add_argument('--osr-confirm-require-consecutive', action='store_true')
    p.add_argument('--osr-confirm-active-iou', type=float, default=2.0)
    p.add_argument('--osr-confirm-active-ioa', type=float, default=2.0)
    p.add_argument('--osr-confirm-suppress-output', action='store_true')
    p.add_argument('--osr-confirm-suppress-until-high', action='store_true')
    p.add_argument('--osr-adaptive-release-gate', action='store_true')
    p.add_argument('--osr-adaptive-min-confirmed', type=int, default=20)
    p.add_argument('--osr-adaptive-min-release-ratio', type=float, default=0.55)
    p.add_argument('--osr-confirm-motion-only', action='store_true')
    p.add_argument('--osr-events-csv', default='')
    args, rest = p.parse_known_args()
    global _EVENTS_CSV
    _EVENTS_CSV = str(args.osr_events_csv)
    sys.argv = [sys.argv[0]] + rest
    return ConfirmConfig(
        enable=bool(args.osr_confirm_enable),
        min_hits=int(args.osr_confirm_min_hits),
        max_miss=int(args.osr_proposal_max_miss),
        max_age=int(args.osr_proposal_max_age),
        min_avg_score=float(args.osr_confirm_min_avg_score),
        min_avg_memory=float(args.osr_confirm_min_avg_memory),
        max_avg_ambiguity=float(args.osr_confirm_max_avg_ambiguity),
        min_last_score=float(args.osr_confirm_min_last_score),
        require_consecutive=bool(args.osr_confirm_require_consecutive),
        active_iou=float(args.osr_confirm_active_iou),
        active_ioa=float(args.osr_confirm_active_ioa),
        suppress_output=bool(args.osr_confirm_suppress_output),
        suppress_until_high=bool(args.osr_confirm_suppress_until_high),
        adaptive_release_gate=bool(args.osr_adaptive_release_gate),
        adaptive_min_confirmed=int(args.osr_adaptive_min_confirmed),
        adaptive_min_release_ratio=float(args.osr_adaptive_min_release_ratio),
        motion_only=bool(args.osr_confirm_motion_only),
    )


def main() -> None:
    global _CONFIRM
    _CONFIRM = parse_confirm_and_strip()
    osr_v1.OSRTracker = ConfirmedOSRTracker
    osr_v1.main()
    if _EVENTS_CSV and _LAST_TRACKER is not None:
        rows = getattr(_LAST_TRACKER, 'osr_event_rows', [])
        from pathlib import Path as _Path
        path = _Path(_EVENTS_CSV)
        path.parent.mkdir(parents=True, exist_ok=True)
        if rows:
            fields = list(rows[0].keys())
            with path.open('w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader(); w.writerows(rows)
        else:
            path.write_text('\n', encoding='utf-8')


if __name__ == '__main__':
    main()
