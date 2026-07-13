#!/usr/bin/env python3
"""Online Safe Recovery (OSRTrack) experimental tracker.

This script imports the stable original dmm_base_tracker implementation and
adds an online safety-aware lost-track recovery stage. It is intentionally kept
separate from scripts/dmm_base_tracker.py so the verified baseline remains safe.

OSR idea:
  - normal high-score association remains unchanged;
  - unmatched Tracked tracks still use the original low-score second stage;
  - Lost tracks are recovered from remaining low-score detections only if the
    pair has high memory appearance, motion/size consistency and low ambiguity;
  - low-score detections are never allowed to start new tracks.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

import dmm_base_tracker as base


@dataclass
class OSRConfig:
    enable: bool = False
    max_lost_age: int = 30
    min_track_len: int = 8
    min_det_score: float = 0.08
    min_memory_sim: float = 0.68
    max_center_step: float = 60.0
    max_area_ratio: float = 4.0
    max_height_ratio: float = 2.5
    min_margin: float = 0.03
    max_ambiguity: int = 2
    min_score: float = 0.58
    rank_slack: float = 0.02
    w_app: float = 0.42
    w_motion: float = 0.20
    w_size: float = 0.12
    w_height: float = 0.08
    w_det: float = 0.06
    w_quality: float = 0.08
    w_ambiguity: float = 0.22


@dataclass
class OSRPair:
    score: float
    memory_sim: float
    motion_score: float
    size_score: float
    height_score: float
    det_score: float
    track_quality: float
    ambiguity_penalty: float
    center_step: float
    area_ratio: float
    height_ratio: float
    lost_age: int
    track_age: int
    valid_pre: bool


class OSRTracker(base.DMMBaseTracker):
    def __init__(self, cfg: base.TrackerConfig, osr: OSRConfig):
        if bool(cfg.dmm_enable):
            raise ValueError("OSRTracker first version requires dmm_enable=False; keep DMM experiments separate.")
        super().__init__(cfg)
        self.osr = osr
        self.osr_rows: List[Dict[str, object]] = []
        self.osr_stats: Dict[str, int] = {
            'frames': 0,
            'frames_with_candidates': 0,
            'candidate_pairs': 0,
            'valid_pairs': 0,
            'recovered': 0,
            'blocked_low_score': 0,
            'blocked_track_quality': 0,
            'blocked_memory': 0,
            'blocked_motion': 0,
            'blocked_geometry': 0,
            'blocked_ambiguity': 0,
        }

    @staticmethod
    def _center(t: base.DMMTrack) -> np.ndarray:
        tlwh = t.tlwh
        return np.asarray([tlwh[0] + tlwh[2] / 2.0, tlwh[1] + tlwh[3] / 2.0], dtype=np.float32)

    @staticmethod
    def _area(t: base.DMMTrack) -> float:
        tlwh = t.tlwh
        return max(1.0, float(tlwh[2] * tlwh[3]))

    @staticmethod
    def _height(t: base.DMMTrack) -> float:
        return max(1.0, float(t.tlwh[3]))

    @staticmethod
    def _unit(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float32)
        n = float(np.linalg.norm(v))
        if n < 1e-12:
            return np.zeros_like(v, dtype=np.float32)
        return v / n

    def _memory_similarity(self, track: base.DMMTrack, det: base.DMMTrack) -> Tuple[float, float, float]:
        if det.curr_feat is None:
            return 0.0, 0.0, 0.0
        df = self._unit(det.curr_feat)
        smooth_sim = 0.0
        if track.smooth_feat is not None:
            smooth_sim = float(np.dot(self._unit(track.smooth_feat), df))
        bank_sims = []
        for f in list(getattr(track, 'features', [])):
            if f is not None and np.asarray(f).size:
                bank_sims.append(float(np.dot(self._unit(f), df)))
        if bank_sims:
            bank_max = float(max(bank_sims))
            topk = sorted(bank_sims, reverse=True)[:min(5, len(bank_sims))]
            bank_topk = float(np.mean(topk))
        else:
            bank_max = 0.0
            bank_topk = 0.0
        # Safety-oriented memory: use both long-term smooth and robust best/top-k bank.
        memory_sim = max(0.45 * smooth_sim + 0.55 * bank_topk, bank_max, smooth_sim)
        return float(memory_sim), float(smooth_sim), float(bank_topk)

    def _track_quality(self, track: base.DMMTrack) -> float:
        age = max(1, int(track.frame_id - track.start_frame + 1))
        q_age = min(1.0, math.log1p(age) / math.log1p(40.0))
        q_len = min(1.0, math.log1p(max(1, int(getattr(track, 'tracklet_len', 0)) + 1)) / math.log1p(20.0))
        q_score = max(0.0, min(1.0, float(getattr(track, 'score', 0.0))))
        q_feat = min(1.0, len(list(getattr(track, 'features', []))) / 10.0)
        return float(0.35 * q_age + 0.25 * q_len + 0.25 * q_score + 0.15 * q_feat)

    def _osr_pair(self, track: base.DMMTrack, det: base.DMMTrack) -> OSRPair:
        lost_age = max(1, int(self.frame_id - track.frame_id))
        track_age = max(1, int(track.frame_id - track.start_frame + 1))
        if det.score < self.osr.min_det_score:
            self.osr_stats['blocked_low_score'] += 1
            return OSRPair(0, 0, 0, 0, 0, float(det.score), 0, 1, 1e6, 1e6, 1e6, lost_age, track_age, False)
        if lost_age > self.osr.max_lost_age or track_age < self.osr.min_track_len:
            self.osr_stats['blocked_track_quality'] += 1
            return OSRPair(0, 0, 0, 0, 0, float(det.score), 0, 1, 1e6, 1e6, 1e6, lost_age, track_age, False)

        memory_sim, _smooth_sim, _bank_topk = self._memory_similarity(track, det)
        if memory_sim < self.osr.min_memory_sim:
            self.osr_stats['blocked_memory'] += 1
            return OSRPair(0, memory_sim, 0, 0, 0, float(det.score), 0, 1, 1e6, 1e6, 1e6, lost_age, track_age, False)

        tc = self._center(track)
        dc = self._center(det)
        dist = float(np.linalg.norm(tc - dc))
        center_step = dist / float(max(1, lost_age))
        if center_step > self.osr.max_center_step:
            self.osr_stats['blocked_motion'] += 1
            return OSRPair(0, memory_sim, 0, 0, 0, float(det.score), 0, 1, center_step, 1e6, 1e6, lost_age, track_age, False)

        area_ratio = max(self._area(track), self._area(det)) / max(1.0, min(self._area(track), self._area(det)))
        height_ratio = max(self._height(track), self._height(det)) / max(1.0, min(self._height(track), self._height(det)))
        if area_ratio > self.osr.max_area_ratio or height_ratio > self.osr.max_height_ratio:
            self.osr_stats['blocked_geometry'] += 1
            return OSRPair(0, memory_sim, 0, 0, 0, float(det.score), 0, 1, center_step, area_ratio, height_ratio, lost_age, track_age, False)

        scale = math.sqrt(max(self._area(track), self._area(det), 1.0))
        motion_score = math.exp(-dist / max(scale * 3.0, 1e-6))
        motion_score *= math.exp(-max(0.0, lost_age - 1) / max(float(self.osr.max_lost_age), 1.0))
        size_score = math.exp(-abs(math.log(max(area_ratio, 1e-6))))
        height_score = math.exp(-abs(math.log(max(height_ratio, 1e-6))))
        track_quality = self._track_quality(track)
        score = (
            self.osr.w_app * memory_sim
            + self.osr.w_motion * motion_score
            + self.osr.w_size * size_score
            + self.osr.w_height * height_score
            + self.osr.w_det * float(det.score)
            + self.osr.w_quality * track_quality
        )
        return OSRPair(float(score), memory_sim, float(motion_score), float(size_score), float(height_score), float(det.score), float(track_quality), 0.0, float(center_step), float(area_ratio), float(height_ratio), lost_age, track_age, True)

    def _osr_recover(self, lost_tracks: Sequence[base.DMMTrack], low_dets: Sequence[base.DMMTrack]) -> Tuple[List[Tuple[int, int]], List[int], np.ndarray, List[List[OSRPair]]]:
        if not self.osr.enable or len(lost_tracks) == 0 or len(low_dets) == 0:
            return [], list(range(len(low_dets))), np.zeros((len(lost_tracks), len(low_dets)), dtype=np.float32), []
        self.osr_stats['frames'] += 1
        pair_grid: List[List[OSRPair]] = []
        score = np.full((len(lost_tracks), len(low_dets)), -1e9, dtype=np.float32)
        for i, t in enumerate(lost_tracks):
            row_pairs = []
            for j, d in enumerate(low_dets):
                p = self._osr_pair(t, d)
                row_pairs.append(p)
                self.osr_stats['candidate_pairs'] += 1
                if p.valid_pre:
                    score[i, j] = float(p.score)
                    self.osr_stats['valid_pairs'] += 1
            pair_grid.append(row_pairs)
        if np.any(score > -1e8):
            self.osr_stats['frames_with_candidates'] += 1

        # Compute ambiguity and margins on preliminary scores.
        row_best = np.max(score, axis=1) if score.size else np.zeros((len(lost_tracks),), dtype=np.float32)
        col_best = np.max(score, axis=0) if score.size else np.zeros((len(low_dets),), dtype=np.float32)
        valid_mask = np.zeros_like(score, dtype=bool)
        for i in range(score.shape[0]):
            row_vals = np.sort(score[i][score[i] > -1e8])[::-1]
            row_second = float(row_vals[1]) if row_vals.size >= 2 else -1e9
            row_margin = float(row_vals[0] - row_second) if row_vals.size else 1.0
            for j in range(score.shape[1]):
                if score[i, j] <= -1e8:
                    continue
                col_vals = np.sort(score[:, j][score[:, j] > -1e8])[::-1]
                col_second = float(col_vals[1]) if col_vals.size >= 2 else -1e9
                col_margin = float(col_vals[0] - col_second) if col_vals.size else 1.0
                near_row = int(np.sum(score[i] >= score[i, j] - self.osr.rank_slack)) - 1
                near_col = int(np.sum(score[:, j] >= score[i, j] - self.osr.rank_slack)) - 1
                ambiguity = max(0, near_row + near_col)
                pair_grid[i][j].ambiguity_penalty = min(1.0, float(ambiguity) / max(1.0, float(self.osr.max_ambiguity + 1)))
                final_score = float(score[i, j]) - self.osr.w_ambiguity * pair_grid[i][j].ambiguity_penalty
                score[i, j] = final_score
                is_near_best = (row_best[i] - final_score <= self.osr.rank_slack + self.osr.w_ambiguity) and (col_best[j] - final_score <= self.osr.rank_slack + self.osr.w_ambiguity)
                margin_ok = min(row_margin, col_margin) >= self.osr.min_margin
                ambiguity_ok = ambiguity <= self.osr.max_ambiguity
                if final_score >= self.osr.min_score and is_near_best and margin_ok and ambiguity_ok:
                    valid_mask[i, j] = True
                else:
                    self.osr_stats['blocked_ambiguity'] += 1

        cost = np.full_like(score, 1e6, dtype=np.float32)
        cost[valid_mask] = 1.0 - score[valid_mask]
        matches, _u_lost, u_low = base.matching.linear_assignment(cost, thresh=float(1.0 - self.osr.min_score))
        out_matches: List[Tuple[int, int]] = []
        for i, j in matches:
            if valid_mask[int(i), int(j)]:
                out_matches.append((int(i), int(j)))
        return out_matches, [int(x) for x in u_low], cost, pair_grid

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
        row_m, col_m = base.row_col_margins(dists)
        matches, u_track, u_detection = base.matching.linear_assignment(dists, thresh=float(self.cfg.match_thresh))
        if self.cfg.debug_assoc:
            self._record_assoc_debug("primary", strack_pool, high_dets, dists, matches, row_m, col_m, debug)
        for itracked, idet in matches:
            track = strack_pool[int(itracked)]
            det = high_dets[int(idet)]
            if track.state == base.TrackState.Tracked:
                track.update(det, self.frame_id)
                activated.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind.append(track)

        r_tracked = [strack_pool[int(i)] for i in u_track if strack_pool[int(i)].state == base.TrackState.Tracked]
        r_lost = [strack_pool[int(i)] for i in u_track if strack_pool[int(i)].state == base.TrackState.Lost]

        if self.osr.enable:
            # Stage 2a: keep original low-score second matching for still-tracked unmatched tracks only.
            low_remaining = list(range(len(low_dets)))
            if len(r_tracked) > 0 and len(low_dets) > 0:
                if bool(self.cfg.stage2_reid_enable):
                    dists2, debug2 = base.assoc_cost(r_tracked, low_dets, self.cfg)
                else:
                    dists2 = base.matching.iou_distance(r_tracked, low_dets).astype(np.float32)
                    debug2 = {"raw_iou": dists2, "final": dists2}
                matches2, u_track2, u_low = base.matching.linear_assignment(dists2, thresh=float(self.cfg.second_match_thresh))
                if self.cfg.debug_assoc:
                    r2, c2 = base.row_col_margins(dists2)
                    self._record_assoc_debug("secondary_tracked", r_tracked, low_dets, dists2, matches2, r2, c2, debug2)
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

            # Stage 2b: OSR lost-track recovery using only remaining low-score detections.
            rem_low_dets = [low_dets[i] for i in low_remaining]
            osr_matches, osr_u_low, osr_cost, osr_pairs = self._osr_recover(r_lost, rem_low_dets)
            for ilost, ilow in osr_matches:
                track = r_lost[int(ilost)]
                det = rem_low_dets[int(ilow)]
                track.re_activate(det, self.frame_id, new_id=False)
                refind.append(track)
                self.osr_stats['recovered'] += 1
            if self.cfg.debug_assoc and len(r_lost) > 0 and len(rem_low_dets) > 0:
                self._record_osr_debug(r_lost, rem_low_dets, osr_matches, osr_pairs)
        else:
            r_all = r_tracked + r_lost if bool(self.cfg.stage2_lost_enable) else r_tracked
            if bool(self.cfg.stage2_reid_enable) and len(r_all) > 0 and len(low_dets) > 0:
                dists2, debug2 = base.assoc_cost(r_all, low_dets, self.cfg)
            else:
                dists2 = base.matching.iou_distance(r_all, low_dets).astype(np.float32)
                debug2 = {"raw_iou": dists2, "final": dists2}
            matches2, u_track2, _u_low = base.matching.linear_assignment(dists2, thresh=float(self.cfg.second_match_thresh))
            if self.cfg.debug_assoc:
                r2, c2 = base.row_col_margins(dists2)
                self._record_assoc_debug("secondary", r_all, low_dets, dists2, matches2, r2, c2, debug2)
            for itracked, idet in matches2:
                track = r_all[int(itracked)]
                det = low_dets[int(idet)]
                if track.state == base.TrackState.Lost:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind.append(track)
                else:
                    track.update(det, self.frame_id)
                    activated.append(track)
            for it in u_track2:
                track = r_all[int(it)]
                if track.state != base.TrackState.Lost:
                    track.mark_lost()
                    lost.append(track)

        remain_high = [high_dets[int(i)] for i in u_detection]
        dists3 = base.matching.iou_distance(unconfirmed, remain_high).astype(np.float32)
        if not self.cfg.mot20:
            dists3 = base.matching.fuse_score(dists3, remain_high).astype(np.float32)
        matches3, u_unconfirmed, u_detection3 = base.matching.linear_assignment(dists3, thresh=float(self.cfg.unconfirmed_match_thresh))
        if self.cfg.debug_assoc:
            r3, c3 = base.row_col_margins(dists3)
            self._record_assoc_debug("unconfirmed", unconfirmed, remain_high, dists3, matches3, r3, c3, {"raw_iou": dists3, "final": dists3})
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

    def _record_osr_debug(self, tracks: Sequence[base.DMMTrack], dets: Sequence[base.DMMTrack], matches: Sequence[Tuple[int, int]], pairs: List[List[OSRPair]]) -> None:
        chosen = {(int(i), int(j)) for i, j in matches}
        for i, track in enumerate(tracks):
            for j, det in enumerate(dets):
                if i >= len(pairs) or j >= len(pairs[i]):
                    continue
                p = pairs[i][j]
                self.debug_rows.append({
                    'frame': self.frame_id,
                    'stage': 'osr_recovery',
                    'track_row': i,
                    'det_col': j,
                    'track_id': track.track_id,
                    'det_global_idx': det.det_global_idx,
                    'chosen': int((i, j) in chosen),
                    'score': p.score,
                    'memory_sim': p.memory_sim,
                    'motion_score': p.motion_score,
                    'size_score': p.size_score,
                    'height_score': p.height_score,
                    'det_score': p.det_score,
                    'track_quality': p.track_quality,
                    'ambiguity_penalty': p.ambiguity_penalty,
                    'center_step': p.center_step,
                    'area_ratio': p.area_ratio,
                    'height_ratio': p.height_ratio,
                    'lost_age': p.lost_age,
                    'track_age': p.track_age,
                    'valid_pre': int(p.valid_pre),
                })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Replay Phase0 dump with Online Safe Recovery tracker.')
    parser.add_argument('--dump-npz', required=True)
    parser.add_argument('--seq', default='MOT20-01')
    parser.add_argument('--out', required=True)
    parser.add_argument('--summary-json', default='')
    parser.add_argument('--debug-csv', default='')
    parser.add_argument('--assoc-mode', default='botsort_reid', choices=['iou', 'botsort_reid'])
    parser.add_argument('--track-high-thresh', type=float, default=0.6)
    parser.add_argument('--track-low-thresh', type=float, default=0.1)
    parser.add_argument('--new-track-thresh', type=float, default=0.7)
    parser.add_argument('--track-buffer', type=int, default=30)
    parser.add_argument('--match-thresh', type=float, default=0.7)
    parser.add_argument('--second-match-thresh', type=float, default=0.5)
    parser.add_argument('--unconfirmed-match-thresh', type=float, default=0.7)
    parser.add_argument('--proximity-thresh', type=float, default=0.5)
    parser.add_argument('--appearance-thresh', type=float, default=0.25)
    parser.add_argument('--frame-rate', type=int, default=25)
    parser.add_argument('--min-box-area', type=float, default=10.0)
    parser.add_argument('--aspect-ratio-thresh', type=float, default=1.6)
    parser.add_argument('--limit-frames', type=int, default=0)
    parser.add_argument('--activate-new-after-first', action='store_true')
    parser.add_argument('--debug-assoc', action='store_true')
    parser.add_argument('--stage2-reid-enable', action='store_true')
    parser.add_argument('--stage2-lost-enable', action='store_true')
    parser.add_argument('--nsa-k', type=float, default=0.0)
    parser.add_argument('--gmc-enable', action='store_true')
    parser.add_argument('--gmc-warp-path', default='')
    parser.add_argument('--osr-enable', action='store_true')
    parser.add_argument('--osr-max-lost-age', type=int, default=30)
    parser.add_argument('--osr-min-track-len', type=int, default=8)
    parser.add_argument('--osr-min-det-score', type=float, default=0.08)
    parser.add_argument('--osr-min-memory-sim', type=float, default=0.68)
    parser.add_argument('--osr-max-center-step', type=float, default=60.0)
    parser.add_argument('--osr-max-area-ratio', type=float, default=4.0)
    parser.add_argument('--osr-max-height-ratio', type=float, default=2.5)
    parser.add_argument('--osr-min-margin', type=float, default=0.03)
    parser.add_argument('--osr-max-ambiguity', type=int, default=2)
    parser.add_argument('--osr-min-score', type=float, default=0.58)
    parser.add_argument('--osr-rank-slack', type=float, default=0.02)
    return parser.parse_args()


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

    cfg = base.TrackerConfig(
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
    osr = OSRConfig(
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
    tracker = OSRTracker(cfg, osr)
    mot_rows: List[Tuple[int, int, float, float, float, float, float]] = []
    frames_with_outputs = 0
    for frame in range(1, n_frames + 1):
        start = int(offsets[frame - 1]); end = int(offsets[frame])
        rows = det[start:end]; feats = feat[start:end]
        if rows.size:
            boxes = rows[:, [col['x1'], col['y1'], col['x2'], col['y2']]].astype(np.float32)
            scores = rows[:, col['score']].astype(np.float32)
            det_ids = rows[:, col['global_det_idx']].astype(np.int64)
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            scores = np.zeros((0,), dtype=np.float32)
            det_ids = np.zeros((0,), dtype=np.int64)
            feats = np.zeros((0, feat.shape[1] if feat.ndim == 2 else 0), dtype=np.float32)
        online_targets = tracker.update(boxes, scores, feats, det_ids)
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
        'osr': asdict(osr),
        'osr_stats': tracker.osr_stats,
    }
    if args.summary_json:
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    if args.debug_csv:
        base.write_debug(Path(args.debug_csv), tracker.debug_rows)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
