#!/usr/bin/env python3
"""Ghost Proposal OSRTrack.

Low-score detections are used only as shadow evidence. They update short-lived
GhostProposal objects attached to lost track IDs, but they never re-activate the
main track. A lost ID is recovered only when a later high-score detection matches
an already confirmed ghost proposal.

This keeps the method online while avoiding low-score boxes contaminating the
main Kalman/ReID state.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

import dmm_base_tracker as base
import dmm_base_tracker_osr as osr_v1


@dataclass
class GhostConfig:
    enable: bool = True
    min_hits: int = 2
    max_age: int = 15
    max_miss: int = 2
    require_consecutive: bool = True
    min_avg_score: float = 0.66
    min_avg_memory: float = 0.76
    max_avg_ambiguity: float = 0.15
    high_min_memory: float = 0.70
    high_min_score: float = 0.60
    high_max_gap: int = 10
    high_max_center_step: float = 80.0
    high_min_iou: float = 0.0
    high_score_thresh: float = 0.60


@dataclass
class GhostProposal:
    track_id: int
    start_frame: int
    last_frame: int
    hits: int
    misses: int
    score_sum: float
    memory_sum: float
    ambiguity_sum: float
    ghost_tlbr: np.ndarray
    last_det_global_idx: int

    def avg_score(self) -> float:
        return float(self.score_sum / max(1, self.hits))

    def avg_memory(self) -> float:
        return float(self.memory_sum / max(1, self.hits))

    def avg_ambiguity(self) -> float:
        return float(self.ambiguity_sum / max(1, self.hits))

    def confirmed(self, cfg: GhostConfig) -> bool:
        return (
            self.hits >= int(cfg.min_hits)
            and self.avg_score() >= float(cfg.min_avg_score)
            and self.avg_memory() >= float(cfg.min_avg_memory)
            and self.avg_ambiguity() <= float(cfg.max_avg_ambiguity)
        )


_GHOST = GhostConfig(enable=False)


class GhostOSRTracker(osr_v1.OSRTracker):
    def __init__(self, cfg: base.TrackerConfig, osr: osr_v1.OSRConfig):
        super().__init__(cfg, osr)
        self.ghost = _GHOST
        self.ghost_proposals: Dict[int, GhostProposal] = {}
        self.osr_stats.update({
            'ghost_created': 0,
            'ghost_updated': 0,
            'ghost_expired': 0,
            'ghost_low_supports': 0,
            'ghost_low_candidates': 0,
            'ghost_confirmed_active': 0,
            'ghost_high_candidates': 0,
            'ghost_high_recovered': 0,
            'ghost_primary_recovered': 0,
            'ghost_high_blocked': 0,
            'ghost_unconfirmed_low': 0,
        })

    @staticmethod
    def _tlbr_center(box: np.ndarray) -> np.ndarray:
        return np.asarray([(float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0], dtype=np.float32)

    def _expire_ghosts(self) -> None:
        stale: List[int] = []
        for tid, prop in self.ghost_proposals.items():
            age = int(self.frame_id - prop.start_frame)
            gap = int(self.frame_id - prop.last_frame)
            if age > int(self.ghost.max_age) or gap > int(self.ghost.max_miss):
                stale.append(int(tid))
        for tid in stale:
            self.ghost_proposals.pop(tid, None)
            self.osr_stats['ghost_expired'] += 1

    def _update_ghosts_from_low(
        self,
        lost_tracks: Sequence[base.DMMTrack],
        low_dets: Sequence[base.DMMTrack],
    ) -> Tuple[object, List[List[osr_v1.OSRPair]]]:
        cand_matches, _u_low, cost, pair_grid = super()._osr_recover(lost_tracks, low_dets)
        self.osr_stats['ghost_low_candidates'] += len(cand_matches)
        seen: set[int] = set()
        for ilost, ilow in cand_matches:
            track = lost_tracks[int(ilost)]
            det = low_dets[int(ilow)]
            pair = pair_grid[int(ilost)][int(ilow)]
            tid = int(track.track_id)
            prev = self.ghost_proposals.get(tid)
            reset = prev is None
            if prev is not None:
                gap = int(self.frame_id - prev.last_frame)
                if bool(self.ghost.require_consecutive) and gap != 1:
                    reset = True
                elif gap > int(self.ghost.max_miss) + 1:
                    reset = True
            if reset:
                prop = GhostProposal(
                    track_id=tid,
                    start_frame=int(self.frame_id),
                    last_frame=int(self.frame_id),
                    hits=1,
                    misses=0,
                    score_sum=float(pair.score),
                    memory_sum=float(pair.memory_sim),
                    ambiguity_sum=float(pair.ambiguity_penalty),
                    ghost_tlbr=np.asarray(det.tlbr, dtype=np.float32).copy(),
                    last_det_global_idx=int(getattr(det, 'det_global_idx', -1)),
                )
                self.ghost_proposals[tid] = prop
                self.osr_stats['ghost_created'] += 1
            else:
                prop = prev
                prop.last_frame = int(self.frame_id)
                prop.hits += 1
                prop.misses = 0
                prop.score_sum += float(pair.score)
                prop.memory_sum += float(pair.memory_sim)
                prop.ambiguity_sum += float(pair.ambiguity_penalty)
                prop.ghost_tlbr = np.asarray(det.tlbr, dtype=np.float32).copy()
                prop.last_det_global_idx = int(getattr(det, 'det_global_idx', -1))
                self.osr_stats['ghost_updated'] += 1
            self.osr_stats['ghost_low_supports'] += 1
            if prop.confirmed(self.ghost):
                self.osr_stats['ghost_confirmed_active'] += 1
            seen.add(tid)

        for tid, prop in list(self.ghost_proposals.items()):
            if tid not in seen and int(self.frame_id - prop.last_frame) > 0:
                prop.misses += 1
                if prop.misses > int(self.ghost.max_miss):
                    self.ghost_proposals.pop(tid, None)
                    self.osr_stats['ghost_expired'] += 1
        return cost, pair_grid

    def _match_ghosts_to_high(
        self,
        lost_tracks: Sequence[base.DMMTrack],
        high_dets: Sequence[base.DMMTrack],
    ) -> Tuple[List[Tuple[int, int]], set[int]]:
        if not self.ghost.enable or len(lost_tracks) == 0 or len(high_dets) == 0 or not self.ghost_proposals:
            return [], set()
        lost_by_id = {int(t.track_id): i for i, t in enumerate(lost_tracks)}
        rows: List[int] = []
        props: List[GhostProposal] = []
        for tid, prop in self.ghost_proposals.items():
            if tid not in lost_by_id:
                continue
            if not prop.confirmed(self.ghost):
                continue
            if int(self.frame_id - prop.last_frame) > int(self.ghost.high_max_gap):
                continue
            rows.append(lost_by_id[int(tid)])
            props.append(prop)
        if not props:
            return [], set()

        score = np.full((len(props), len(high_dets)), -1e9, dtype=np.float32)
        for i, prop in enumerate(props):
            track = lost_tracks[rows[i]]
            gc = self._tlbr_center(prop.ghost_tlbr)
            for j, det in enumerate(high_dets):
                if float(det.score) < float(self.ghost.high_score_thresh):
                    continue
                mem, _smooth, _bank = self._memory_similarity(track, det)
                if mem < float(self.ghost.high_min_memory):
                    continue
                dc = self._center(det)
                gap = max(1, int(self.frame_id - prop.last_frame))
                center_step = float(np.linalg.norm(dc - gc)) / float(gap)
                if center_step > float(self.ghost.high_max_center_step):
                    continue
                iou = self._iou_one(prop.ghost_tlbr, det.tlbr)
                if iou < float(self.ghost.high_min_iou):
                    continue
                motion_score = float(np.exp(-center_step / max(float(self.ghost.high_max_center_step), 1e-6)))
                s = 0.65 * float(mem) + 0.25 * motion_score + 0.10 * min(1.0, float(det.score))
                score[i, j] = s
        valid = score >= float(self.ghost.high_min_score)
        if not np.any(valid):
            return [], set()
        cost = np.full_like(score, 1e6, dtype=np.float32)
        cost[valid] = 1.0 - score[valid]
        matches, _u_p, _u_d = base.matching.linear_assignment(cost, thresh=float(1.0 - self.ghost.high_min_score))
        out: List[Tuple[int, int]] = []
        used_high: set[int] = set()
        for iprop, jdet in matches:
            iprop = int(iprop); jdet = int(jdet)
            if not valid[iprop, jdet]:
                continue
            out.append((rows[iprop], jdet))
            used_high.add(jdet)
            self.ghost_proposals.pop(int(props[iprop].track_id), None)
            self.osr_stats['ghost_high_recovered'] += 1
        self.osr_stats['ghost_high_candidates'] += int(np.sum(valid))
        return out, used_high


    def _apply_ghost_guidance(self, tracks: Sequence[base.DMMTrack], high_dets: Sequence[base.DMMTrack], dists: np.ndarray) -> np.ndarray:
        """Use confirmed ghost proposals to improve high-score association costs.

        This does not update the main track state. It only lets a confirmed ghost
        act as a temporary observation-centric proxy when assigning current
        high-score detections to lost tracks.
        """
        if not self.ghost.enable or dists.size == 0 or not self.ghost_proposals:
            return dists
        out = dists.copy()
        for i, track in enumerate(tracks):
            if getattr(track, 'state', None) != base.TrackState.Lost:
                continue
            prop = self.ghost_proposals.get(int(track.track_id))
            if prop is None or not prop.confirmed(self.ghost):
                continue
            if int(self.frame_id - prop.last_frame) > int(self.ghost.high_max_gap):
                continue
            gc = self._tlbr_center(prop.ghost_tlbr)
            for j, det in enumerate(high_dets):
                if float(det.score) < float(self.ghost.high_score_thresh):
                    continue
                mem, _smooth, _bank = self._memory_similarity(track, det)
                if mem < float(self.ghost.high_min_memory):
                    continue
                dc = self._center(det)
                gap = max(1, int(self.frame_id - prop.last_frame))
                center_step = float(np.linalg.norm(dc - gc)) / float(gap)
                if center_step > float(self.ghost.high_max_center_step):
                    continue
                iou = self._iou_one(prop.ghost_tlbr, det.tlbr)
                if iou < float(self.ghost.high_min_iou):
                    continue
                motion_score = float(np.exp(-center_step / max(float(self.ghost.high_max_center_step), 1e-6)))
                score = 0.65 * float(mem) + 0.25 * motion_score + 0.10 * min(1.0, float(det.score))
                if score < float(self.ghost.high_min_score):
                    continue
                out[i, j] = min(float(out[i, j]), float(1.0 - score))
                self.osr_stats['ghost_high_candidates'] += 1
        return out

    def update(self, bboxes_xyxy: np.ndarray, scores: np.ndarray, features: np.ndarray, det_global_ids: np.ndarray) -> List[base.DMMTrack]:
        self.frame_id += 1
        activated: List[base.DMMTrack] = []
        refind: List[base.DMMTrack] = []
        lost: List[base.DMMTrack] = []
        removed: List[base.DMMTrack] = []

        if bboxes_xyxy.size == 0:
            bboxes_xyxy = np.zeros((0, 4), dtype=np.float32)
            scores = np.zeros((0,), dtype=np.float32)
            features = np.zeros((0, 0), dtype=np.float32)
            det_global_ids = np.zeros((0,), dtype=np.int64)

        valid = scores > float(self.cfg.track_low_thresh)
        bboxes_xyxy = bboxes_xyxy[valid]
        scores = scores[valid]
        features = features[valid] if features.shape[0] == valid.shape[0] else np.zeros((int(valid.sum()), 0), dtype=np.float32)
        det_global_ids = det_global_ids[valid]

        high = scores > float(self.cfg.track_high_thresh)
        low = np.logical_and(scores > float(self.cfg.track_low_thresh), scores < float(self.cfg.track_high_thresh))
        high_dets = self._make_detections(bboxes_xyxy[high], scores[high], features[high], det_global_ids[high], with_feat=True)
        low_dets = self._make_detections(bboxes_xyxy[low], scores[low], features[low], det_global_ids[low], with_feat=bool(self.cfg.stage2_reid_enable or self.osr.enable))

        unconfirmed: List[base.DMMTrack] = []
        tracked: List[base.DMMTrack] = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked.append(track)

        strack_pool = base.joint_stracks(tracked, self.lost_stracks)
        if self.gmc_warps is not None and self.frame_id < len(self.gmc_warps):
            H = self.gmc_warps[self.frame_id - 1]
            base.DMMTrack.multi_gmc(strack_pool, H)
            if unconfirmed:
                base.DMMTrack.multi_gmc(unconfirmed, H)
        base.DMMTrack.multi_predict(strack_pool)

        dists, debug = base.assoc_cost(strack_pool, high_dets, self.cfg)
        dists = self._apply_ghost_guidance(strack_pool, high_dets, dists)
        debug['final'] = dists.copy()
        row_m, col_m = base.row_col_margins(dists)
        matches, u_track, u_detection = base.matching.linear_assignment(dists, thresh=float(self.cfg.match_thresh))
        if self.cfg.debug_assoc:
            self._record_assoc_debug('primary', strack_pool, high_dets, dists, matches, row_m, col_m, debug)
        for itracked, idet in matches:
            track = strack_pool[int(itracked)]
            det = high_dets[int(idet)]
            if track.state == base.TrackState.Tracked:
                track.update(det, self.frame_id)
                activated.append(track)
            else:
                prop = self.ghost_proposals.get(int(track.track_id))
                if prop is not None and prop.confirmed(self.ghost):
                    self.osr_stats['ghost_primary_recovered'] += 1
                track.re_activate(det, self.frame_id, new_id=False)
                self.ghost_proposals.pop(int(track.track_id), None)
                refind.append(track)

        r_tracked = [strack_pool[int(i)] for i in u_track if strack_pool[int(i)].state == base.TrackState.Tracked]
        r_lost = [strack_pool[int(i)] for i in u_track if strack_pool[int(i)].state == base.TrackState.Lost]

        low_remaining = list(range(len(low_dets)))
        if len(r_tracked) > 0 and len(low_dets) > 0:
            if bool(self.cfg.stage2_reid_enable):
                dists2, debug2 = base.assoc_cost(r_tracked, low_dets, self.cfg)
            else:
                dists2 = base.matching.iou_distance(r_tracked, low_dets).astype(np.float32)
                debug2 = {'raw_iou': dists2, 'final': dists2}
            matches2, u_track2, u_low = base.matching.linear_assignment(dists2, thresh=float(self.cfg.second_match_thresh))
            if self.cfg.debug_assoc:
                r2, c2 = base.row_col_margins(dists2)
                self._record_assoc_debug('secondary_tracked', r_tracked, low_dets, dists2, matches2, r2, c2, debug2)
            for itracked, idet in matches2:
                track = r_tracked[int(itracked)]
                det = low_dets[int(idet)]
                track.update(det, self.frame_id)
                activated.append(track)
            for it in u_track2:
                track = r_tracked[int(it)]
                if track.state != base.TrackState.Lost:
                    track.mark_lost()
                    lost.append(track)
            low_remaining = [int(x) for x in u_low]
        else:
            for track in r_tracked:
                if track.state != base.TrackState.Lost:
                    track.mark_lost()
                    lost.append(track)

        remain_high0 = [high_dets[int(i)] for i in u_detection]
        ghost_matches, used_high = self._match_ghosts_to_high(r_lost, remain_high0)
        for ilost, ihigh in ghost_matches:
            track = r_lost[int(ilost)]
            det = remain_high0[int(ihigh)]
            track.re_activate(det, self.frame_id, new_id=False)
            refind.append(track)

        rem_low_dets = [low_dets[i] for i in low_remaining]
        if self.ghost.enable and len(r_lost) > 0 and len(rem_low_dets) > 0:
            _cost, _pairs = self._update_ghosts_from_low(r_lost, rem_low_dets)
        self._expire_ghosts()

        remain_high = [d for i, d in enumerate(remain_high0) if i not in used_high]
        dists3 = base.matching.iou_distance(unconfirmed, remain_high).astype(np.float32)
        if not self.cfg.mot20:
            dists3 = base.matching.fuse_score(dists3, remain_high).astype(np.float32)
        matches3, u_unconfirmed, u_detection3 = base.matching.linear_assignment(dists3, thresh=float(self.cfg.unconfirmed_match_thresh))
        if self.cfg.debug_assoc:
            r3, c3 = base.row_col_margins(dists3)
            self._record_assoc_debug('unconfirmed', unconfirmed, remain_high, dists3, matches3, r3, c3, {'raw_iou': dists3, 'final': dists3})
        for itracked, idet in matches3:
            unconfirmed[int(itracked)].update(remain_high[int(idet)], self.frame_id)
            activated.append(unconfirmed[int(itracked)])
        for it in u_unconfirmed:
            track = unconfirmed[int(it)]
            track.mark_removed()
            removed.append(track)

        for inew in u_detection3:
            track = remain_high[int(inew)]
            if track.score < float(self.cfg.new_track_thresh):
                continue
            track.activate(self.kalman_filter, self.frame_id, activate_new_after_first=self.cfg.activate_new_after_first)
            activated.append(track)

        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == base.TrackState.Tracked]
        self.tracked_stracks = base.joint_stracks(self.tracked_stracks, activated)
        self.tracked_stracks = base.joint_stracks(self.tracked_stracks, refind)
        self.lost_stracks = base.sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost)
        self.lost_stracks = base.sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed)
        self.tracked_stracks, self.lost_stracks = base.remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        return [track for track in self.tracked_stracks if track.is_activated]


def parse_ghost_and_strip() -> GhostConfig:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--ghost-enable', action='store_true')
    p.add_argument('--ghost-min-hits', type=int, default=2)
    p.add_argument('--ghost-max-age', type=int, default=15)
    p.add_argument('--ghost-max-miss', type=int, default=2)
    p.add_argument('--ghost-require-consecutive', action='store_true')
    p.add_argument('--ghost-min-avg-score', type=float, default=0.66)
    p.add_argument('--ghost-min-avg-memory', type=float, default=0.76)
    p.add_argument('--ghost-max-avg-ambiguity', type=float, default=0.15)
    p.add_argument('--ghost-high-min-memory', type=float, default=0.70)
    p.add_argument('--ghost-high-min-score', type=float, default=0.60)
    p.add_argument('--ghost-high-max-gap', type=int, default=10)
    p.add_argument('--ghost-high-max-center-step', type=float, default=80.0)
    p.add_argument('--ghost-high-min-iou', type=float, default=0.0)
    p.add_argument('--ghost-high-score-thresh', type=float, default=0.60)
    args, rest = p.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    return GhostConfig(
        enable=bool(args.ghost_enable),
        min_hits=int(args.ghost_min_hits),
        max_age=int(args.ghost_max_age),
        max_miss=int(args.ghost_max_miss),
        require_consecutive=bool(args.ghost_require_consecutive),
        min_avg_score=float(args.ghost_min_avg_score),
        min_avg_memory=float(args.ghost_min_avg_memory),
        max_avg_ambiguity=float(args.ghost_max_avg_ambiguity),
        high_min_memory=float(args.ghost_high_min_memory),
        high_min_score=float(args.ghost_high_min_score),
        high_max_gap=int(args.ghost_high_max_gap),
        high_max_center_step=float(args.ghost_high_max_center_step),
        high_min_iou=float(args.ghost_high_min_iou),
        high_score_thresh=float(args.ghost_high_score_thresh),
    )


def main() -> None:
    global _GHOST
    _GHOST = parse_ghost_and_strip()
    osr_v1.OSRTracker = GhostOSRTracker
    osr_v1.main()


if __name__ == '__main__':
    main()
