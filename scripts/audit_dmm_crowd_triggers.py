#!/usr/bin/env python3
"""Audit DMM crowd/ambiguity trigger candidates.

This is an analysis-only script. It does not modify tracker outputs. It checks
whether the intended DMM trigger family (crowd overlap + low association margin)
lands near identity switches / identity instability in the independent Phase 1
base tracker.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


def tlwh_to_tlbr(x: float, y: float, w: float, h: float) -> np.ndarray:
    return np.asarray([x, y, x + w, y + h], dtype=np.float32)


def iou_one(a: np.ndarray, b: np.ndarray) -> float:
    xx1 = max(float(a[0]), float(b[0])); yy1 = max(float(a[1]), float(b[1]))
    xx2 = min(float(a[2]), float(b[2])); yy2 = min(float(a[3]), float(b[3]))
    iw = max(0.0, xx2 - xx1); ih = max(0.0, yy2 - yy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    return float(inter / max(area_a + area_b - inter, 1e-12))


def ioa_min_one(a: np.ndarray, b: np.ndarray) -> float:
    xx1 = max(float(a[0]), float(b[0])); yy1 = max(float(a[1]), float(b[1]))
    xx2 = min(float(a[2]), float(b[2])); yy2 = min(float(a[3]), float(b[3]))
    iw = max(0.0, xx2 - xx1); ih = max(0.0, yy2 - yy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    return float(inter / max(min(area_a, area_b), 1e-12))


def load_dump(path: Path) -> tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    z = np.load(path, allow_pickle=True)
    det = np.asarray(z['detections'], dtype=np.float32)
    columns = [str(x) for x in z['columns'].tolist()]
    return det, np.asarray(z['frame_offsets'], dtype=np.int64), {c: i for i, c in enumerate(columns)}


def load_tracker_rows(path: Path) -> Dict[int, List[dict]]:
    by_frame: Dict[int, List[dict]] = defaultdict(list)
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = line.split(',')
            if len(p) < 7:
                continue
            fr = int(float(p[0])); tid = int(float(p[1]))
            x, y, w, h, score = map(float, p[2:7])
            by_frame[fr].append({'frame': fr, 'track_id': tid, 'box': tlwh_to_tlbr(x, y, w, h), 'score': score})
    return by_frame


def load_gt_rows(path: Path) -> Dict[int, List[dict]]:
    by_frame: Dict[int, List[dict]] = defaultdict(list)
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = line.split(',')
            if len(p) < 7:
                continue
            fr = int(float(p[0])); gid = int(float(p[1]))
            x, y, w, h = map(float, p[2:6])
            mark = int(float(p[6])) if len(p) > 6 else 1
            cls = int(float(p[7])) if len(p) > 7 else 1
            vis = float(p[8]) if len(p) > 8 else 1.0
            # MOTChallenge pedestrian class = 1, mark=1 means valid target.
            if mark != 1 or cls != 1:
                continue
            by_frame[fr].append({'frame': fr, 'gt_id': gid, 'box': tlwh_to_tlbr(x, y, w, h), 'vis': vis})
    return by_frame


def greedy_assign_tracks_to_gt(tracks_by_frame: Dict[int, List[dict]], gt_by_frame: Dict[int, List[dict]], min_iou: float) -> Dict[tuple[int, int], tuple[int, float]]:
    assigned: Dict[tuple[int, int], tuple[int, float]] = {}
    for frame, tracks in tracks_by_frame.items():
        gts = gt_by_frame.get(frame, [])
        pairs = []
        for ti, tr in enumerate(tracks):
            for gi, gt in enumerate(gts):
                iou = iou_one(tr['box'], gt['box'])
                if iou >= min_iou:
                    pairs.append((iou, ti, gi))
        pairs.sort(reverse=True)
        used_t, used_g = set(), set()
        for iou, ti, gi in pairs:
            if ti in used_t or gi in used_g:
                continue
            used_t.add(ti); used_g.add(gi)
            assigned[(frame, int(tracks[ti]['track_id']))] = (int(gts[gi]['gt_id']), float(iou))
    return assigned


def find_track_id_switch_frames(assignments: Dict[tuple[int, int], tuple[int, float]], max_gap: int) -> tuple[set[int], List[dict]]:
    by_track: Dict[int, List[tuple[int, int, float]]] = defaultdict(list)
    for (frame, tid), (gid, iou) in assignments.items():
        by_track[tid].append((frame, gid, iou))
    switch_frames: set[int] = set()
    events: List[dict] = []
    for tid, rows in by_track.items():
        rows.sort()
        prev_frame = None; prev_gid = None; prev_iou = None
        for frame, gid, iou in rows:
            if prev_gid is not None and gid != prev_gid and prev_frame is not None and frame - prev_frame <= max_gap:
                switch_frames.add(frame)
                events.append({
                    'frame': frame,
                    'track_id': tid,
                    'prev_frame': prev_frame,
                    'prev_gt_id': prev_gid,
                    'new_gt_id': gid,
                    'prev_iou': prev_iou,
                    'new_iou': iou,
                    'gap': frame - prev_frame,
                })
            prev_frame = frame; prev_gid = gid; prev_iou = iou
    return switch_frames, events


def build_detection_overlap_index(det: np.ndarray, offsets: np.ndarray, col: Dict[str, int], ioa_thresh: float, iou_thresh: float) -> Dict[int, dict]:
    info: Dict[int, dict] = {}
    for frame in range(1, len(offsets)):
        s = int(offsets[frame - 1]); e = int(offsets[frame])
        rows = det[s:e]
        if rows.size == 0:
            continue
        boxes = rows[:, [col['x1'], col['y1'], col['x2'], col['y2']]].astype(np.float32)
        gids = rows[:, col['global_det_idx']].astype(np.int64)
        n = len(rows)
        peers = {int(g): [] for g in gids}
        max_iou = {int(g): 0.0 for g in gids}
        max_ioa = {int(g): 0.0 for g in gids}
        for i in range(n):
            for j in range(i + 1, n):
                iou = iou_one(boxes[i], boxes[j])
                ioa = ioa_min_one(boxes[i], boxes[j])
                if iou >= iou_thresh or ioa >= ioa_thresh:
                    gi = int(gids[i]); gj = int(gids[j])
                    peers[gi].append(gj); peers[gj].append(gi)
                gi = int(gids[i]); gj = int(gids[j])
                max_iou[gi] = max(max_iou[gi], iou); max_iou[gj] = max(max_iou[gj], iou)
                max_ioa[gi] = max(max_ioa[gi], ioa); max_ioa[gj] = max(max_ioa[gj], ioa)
        for k, g in enumerate(gids):
            gid = int(g)
            info[gid] = {
                'frame': frame,
                'box': boxes[k].tolist(),
                'score': float(rows[k, col['score']]),
                'crowd_peer_count': len(peers[gid]),
                'crowd_cluster_size': 1 + len(peers[gid]),
                'max_iou_with_peer': float(max_iou[gid]),
                'max_ioa_with_peer': float(max_ioa[gid]),
                'is_crowd': int(len(peers[gid]) > 0),
            }
    return info


def min_distance_to_set(frame: int, frames: set[int]) -> int:
    if not frames:
        return 10**9
    # small set for MOT20-01; direct min is fine.
    return min(abs(frame - f) for f in frames)


def parse_debug(path: Path) -> List[dict]:
    with path.open('r', encoding='utf-8', newline='') as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    fields = list(rows[0].keys())
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Audit DMM crowd + association ambiguity trigger candidates.')
    ap.add_argument('--dump-npz', required=True)
    ap.add_argument('--track-results', required=True)
    ap.add_argument('--assoc-debug-csv', required=True)
    ap.add_argument('--gt', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--margin-thresh', type=float, default=0.05)
    ap.add_argument('--ioa-thresh', type=float, default=0.60)
    ap.add_argument('--iou-thresh', type=float, default=0.30)
    ap.add_argument('--assign-iou-thresh', type=float, default=0.50)
    ap.add_argument('--idsw-gap', type=int, default=5)
    ap.add_argument('--near-window', type=int, default=5)
    ap.add_argument('--top-rank-only', action='store_true', help='Only rank-0 rows from top3 debug')
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    det, offsets, col = load_dump(Path(args.dump_npz))
    tracks_by_frame = load_tracker_rows(Path(args.track_results))
    gt_by_frame = load_gt_rows(Path(args.gt))
    assignments = greedy_assign_tracks_to_gt(tracks_by_frame, gt_by_frame, min_iou=float(args.assign_iou_thresh))
    idsw_frames, idsw_events = find_track_id_switch_frames(assignments, max_gap=int(args.idsw_gap))
    det_overlap = build_detection_overlap_index(det, offsets, col, ioa_thresh=float(args.ioa_thresh), iou_thresh=float(args.iou_thresh))
    debug_rows = parse_debug(Path(args.assoc_debug_csv))

    candidates: List[dict] = []
    total_primary = 0; total_margin = 0; total_crowd = 0; total_both = 0; total_near = 0
    for row in debug_rows:
        if row.get('stage') != 'primary':
            continue
        if args.top_rank_only and int(float(row.get('rank', 0) or 0)) != 0:
            continue
        total_primary += 1
        try:
            frame = int(float(row['frame']))
            det_gid = int(float(row['det_global_idx']))
            row_margin = float(row['row_margin'])
            col_margin = float(row['col_margin'])
            rank = int(float(row.get('rank', 0) or 0))
            chosen = int(float(row.get('chosen', 0) or 0))
        except Exception:
            continue
        margin_hit = row_margin < float(args.margin_thresh) and col_margin < float(args.margin_thresh)
        if margin_hit:
            total_margin += 1
        dinfo = det_overlap.get(det_gid, {'is_crowd': 0, 'crowd_cluster_size': 1, 'crowd_peer_count': 0, 'max_iou_with_peer': 0.0, 'max_ioa_with_peer': 0.0, 'score': math.nan})
        crowd_hit = bool(dinfo.get('is_crowd', 0))
        if crowd_hit:
            total_crowd += 1
        both = bool(margin_hit and crowd_hit)
        if both:
            total_both += 1
        dist = min_distance_to_set(frame, idsw_frames)
        near = dist <= int(args.near_window)
        if both and near:
            total_near += 1
        if both:
            out = dict(row)
            out.update({
                'margin_hit': int(margin_hit),
                'crowd_hit': int(crowd_hit),
                'near_id_switch_window': int(near),
                'distance_to_idsw_frame': int(dist if dist < 10**9 else -1),
                'crowd_cluster_size': int(dinfo.get('crowd_cluster_size', 1)),
                'crowd_peer_count': int(dinfo.get('crowd_peer_count', 0)),
                'max_iou_with_peer': float(dinfo.get('max_iou_with_peer', 0.0)),
                'max_ioa_with_peer': float(dinfo.get('max_ioa_with_peer', 0.0)),
                'det_dump_score': float(dinfo.get('score', math.nan)),
                'rank': rank,
                'chosen': chosen,
            })
            candidates.append(out)

    by_frame = defaultdict(int)
    near_by_frame = defaultdict(int)
    for c in candidates:
        by_frame[int(float(c['frame']))] += 1
        if int(c['near_id_switch_window']):
            near_by_frame[int(float(c['frame']))] += 1
    top_frames = sorted(by_frame.items(), key=lambda x: (-x[1], x[0]))[:20]
    top_near_frames = sorted(near_by_frame.items(), key=lambda x: (-x[1], x[0]))[:20]
    summary = {
        'status': 'completed',
        'dump_npz': str(args.dump_npz),
        'track_results': str(args.track_results),
        'assoc_debug_csv': str(args.assoc_debug_csv),
        'gt': str(args.gt),
        'frames': int(len(offsets) - 1),
        'tracker_rows': int(sum(len(v) for v in tracks_by_frame.values())),
        'gt_rows': int(sum(len(v) for v in gt_by_frame.values())),
        'assigned_track_gt_pairs': int(len(assignments)),
        'estimated_track_id_switch_events': int(len(idsw_events)),
        'estimated_track_id_switch_frames': int(len(idsw_frames)),
        'total_primary_debug_rows': int(total_primary),
        'margin_thresh': float(args.margin_thresh),
        'ioa_thresh': float(args.ioa_thresh),
        'iou_thresh': float(args.iou_thresh),
        'near_window': int(args.near_window),
        'primary_margin_hit_rows': int(total_margin),
        'primary_crowd_hit_rows': int(total_crowd),
        'dmm_candidate_rows_margin_and_crowd': int(total_both),
        'dmm_candidate_rows_near_estimated_idsw': int(total_near),
        'candidate_unique_frames': int(len(by_frame)),
        'candidate_near_idsw_unique_frames': int(len(near_by_frame)),
        'top_candidate_frames': [{'frame': int(f), 'rows': int(n)} for f, n in top_frames],
        'top_near_idsw_candidate_frames': [{'frame': int(f), 'rows': int(n)} for f, n in top_near_frames],
        'notes': 'Estimated ID switches are from greedy track-to-GT assignment; use as trigger audit, not official metric.',
    }
    with (out_dir / 'summary.json').open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write('\n')
    write_csv(out_dir / 'dmm_candidate_rows.csv', candidates)
    write_csv(out_dir / 'estimated_id_switch_events.csv', idsw_events)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
