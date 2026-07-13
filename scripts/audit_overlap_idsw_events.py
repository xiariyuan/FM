#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
from scipy.optimize import linear_sum_assignment


def read_mot(path: Path, is_gt: bool = False) -> Dict[int, List[dict]]:
    by_frame: Dict[int, List[dict]] = defaultdict(list)
    if not path.exists():
        return by_frame
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 6:
                continue
            try:
                fr = int(float(parts[0])); tid = int(float(parts[1]))
                x = float(parts[2]); y = float(parts[3]); w = float(parts[4]); h = float(parts[5])
            except Exception:
                continue
            score = float(parts[6]) if len(parts) > 6 and parts[6] != '' else 1.0
            if is_gt:
                # MOTChallenge GT: frame,id,x,y,w,h,mark,class,visibility
                mark = int(float(parts[6])) if len(parts) > 6 and parts[6] != '' else 1
                cls = int(float(parts[7])) if len(parts) > 7 and parts[7] != '' else 1
                if mark <= 0 or cls != 1:
                    continue
            if w <= 0 or h <= 0:
                continue
            by_frame[fr].append({'frame': fr, 'id': tid, 'x': x, 'y': y, 'w': w, 'h': h, 'score': score})
    return by_frame


def box_arr(rows: List[dict]) -> np.ndarray:
    if not rows:
        return np.zeros((0, 4), dtype=np.float32)
    a = np.zeros((len(rows), 4), dtype=np.float32)
    for i, r in enumerate(rows):
        x, y, w, h = r['x'], r['y'], r['w'], r['h']
        a[i] = [x, y, x + w, y + h]
    return a


def pair_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.clip((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), 1e-6, None)
    area_b = np.clip((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]), 1e-6, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-6, None)


def pair_ioa_minarea(a: np.ndarray) -> np.ndarray:
    n = len(a)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], a[None, :, :2])
    rb = np.minimum(a[:, None, 2:], a[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area = np.clip((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), 1e-6, None)
    min_area = np.minimum(area[:, None], area[None, :])
    out = inter / np.clip(min_area, 1e-6, None)
    np.fill_diagonal(out, 0.0)
    return out


def frame_match(tracks: List[dict], gts: List[dict], iou_thr: float) -> List[Tuple[int, int, float]]:
    if not tracks or not gts:
        return []
    ious = pair_iou(box_arr(tracks), box_arr(gts))
    cost = 1.0 - ious
    ri, ci = linear_sum_assignment(cost)
    matches = []
    for r, c in zip(ri, ci):
        val = float(ious[r, c])
        if val >= iou_thr:
            matches.append((r, c, val))
    return matches


def update_recent_overlap(recent: Dict[int, deque], frame: int, track_i: int, track_j: int, ioa: float, max_window: int) -> None:
    for a, b in [(track_i, track_j), (track_j, track_i)]:
        dq = recent[a]
        dq.append((frame, b, ioa))
        while dq and frame - dq[0][0] > max_window:
            dq.popleft()


def summarize_recent(recent: Dict[int, deque], tid: int, frame: int, windows: List[int]) -> dict:
    dq = recent.get(tid, deque())
    out = {}
    for w in windows:
        vals = [(fr, p, ioa) for fr, p, ioa in dq if 0 <= frame - fr <= w]
        out[f'overlap_within_{w}'] = int(bool(vals))
        out[f'overlap_max_ioa_{w}'] = max((v[2] for v in vals), default=0.0)
        partners = sorted({str(v[1]) for v in vals})
        out[f'overlap_partners_{w}'] = '|'.join(partners[:20])
    return out


def read_selected_links(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rows.append({
                'seq': r['seq'],
                'track_a': int(float(r['track_a'])),
                'track_b': int(float(r['track_b'])),
                'score': float(r.get('score', 0) or 0),
                'same_gt': int(float(r.get('same_gt', 0) or 0)),
                'gap': int(float(r.get('gap', 0) or 0)),
            })
    return rows


def write_csv(path: Path, rows: List[dict], fieldnames: List[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys = []
        seen = set()
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    seen.add(k); keys.append(k)
        fieldnames = keys
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--tracker-dir', required=True)
    ap.add_argument('--gt-root', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--seqs', nargs='+', default=['MOT20-01','MOT20-02','MOT20-03','MOT20-05'])
    ap.add_argument('--iou-thr', type=float, default=0.5)
    ap.add_argument('--overlap-thrs', type=float, nargs='+', default=[0.3, 0.5, 0.8])
    ap.add_argument('--event-overlap-thr', type=float, default=0.3)
    ap.add_argument('--write-overlap-thr', type=float, default=0.8)
    ap.add_argument('--windows', type=int, nargs='+', default=[1, 3, 5, 10])
    ap.add_argument('--selected-links', default='')
    args = ap.parse_args()

    tracker_dir = Path(args.tracker_dir)
    gt_root = Path(args.gt_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    max_window = max(args.windows)

    all_matched = []
    all_idsw = []
    all_track_switch = []
    all_overlap_events = []
    by_seq_summary = []
    track_stats_by_seq: Dict[str, Dict[int, dict]] = {}
    track_overlap_summary_by_seq: Dict[str, Dict[int, dict]] = {}

    for seq in args.seqs:
        tr_by_frame = read_mot(tracker_dir / f'{seq}.txt', is_gt=False)
        gt_by_frame = read_mot(gt_root / seq / 'gt' / 'gt.txt', is_gt=True)
        frames = sorted(set(tr_by_frame) | set(gt_by_frame))
        recent_overlap: Dict[int, deque] = defaultdict(deque)
        last_gt_for_track: Dict[int, Tuple[int, int, float]] = {}
        last_track_for_gt: Dict[int, Tuple[int, int, float]] = {}
        track_stats: Dict[int, dict] = defaultdict(lambda: {'start': 10**18, 'end': -1, 'rows': 0, 'max_overlap_ioa': 0.0, 'overlap_frames_030': 0, 'overlap_frames_050': 0, 'overlap_frames_080': 0})
        overlap_frame_seen: Dict[Tuple[int, float], int] = {}
        seq_overlap_counts = {str(t): 0 for t in args.overlap_thrs}
        matched_count = 0
        seq_idsw = []
        seq_track_switch = []
        seq_overlap_events = []

        for fr in frames:
            tracks = tr_by_frame.get(fr, [])
            gts = gt_by_frame.get(fr, [])
            tboxes = box_arr(tracks)
            # Track stats and overlap.
            for r in tracks:
                tid = int(r['id'])
                st = track_stats[tid]
                st['start'] = min(st['start'], fr); st['end'] = max(st['end'], fr); st['rows'] += 1
            ioa_mat = pair_ioa_minarea(tboxes)
            n = len(tracks)
            # per-frame per-track threshold flags.
            per_track_frame_thr = defaultdict(set)
            for i in range(n):
                for j in range(i + 1, n):
                    ioa = float(ioa_mat[i, j])
                    if ioa <= 0:
                        continue
                    tid_i = int(tracks[i]['id']); tid_j = int(tracks[j]['id'])
                    if ioa >= args.event_overlap_thr:
                        update_recent_overlap(recent_overlap, fr, tid_i, tid_j, ioa, max_window)
                    for thr in args.overlap_thrs:
                        if ioa >= thr:
                            seq_overlap_counts[str(thr)] += 1
                            per_track_frame_thr[tid_i].add(thr); per_track_frame_thr[tid_j].add(thr)
                    if ioa >= args.write_overlap_thr:
                        seq_overlap_events.append({'seq': seq, 'frame': fr, 'track_i': tid_i, 'track_j': tid_j, 'ioa_min_area': ioa})
                    track_stats[tid_i]['max_overlap_ioa'] = max(track_stats[tid_i]['max_overlap_ioa'], ioa)
                    track_stats[tid_j]['max_overlap_ioa'] = max(track_stats[tid_j]['max_overlap_ioa'], ioa)
            for tid, thrs in per_track_frame_thr.items():
                if any(float(t) >= 0.3 for t in thrs): track_stats[tid]['overlap_frames_030'] += 1
                if any(float(t) >= 0.5 for t in thrs): track_stats[tid]['overlap_frames_050'] += 1
                if any(float(t) >= 0.8 for t in thrs): track_stats[tid]['overlap_frames_080'] += 1

            # Track-GT matching.
            matches = frame_match(tracks, gts, args.iou_thr)
            for ti, gi, iou in matches:
                tid = int(tracks[ti]['id']); gid = int(gts[gi]['id'])
                matched_count += 1
                mrow = {'seq': seq, 'frame': fr, 'track_id': tid, 'gt_id': gid, 'iou': iou, 'track_score': tracks[ti].get('score', 1.0)}
                all_matched.append(mrow)
                prev_gt = last_gt_for_track.get(tid)
                if prev_gt is not None and prev_gt[0] != gid:
                    rec = {
                        'seq': seq, 'frame': fr, 'track_id': tid,
                        'prev_gt_id': prev_gt[0], 'new_gt_id': gid,
                        'prev_frame': prev_gt[1], 'prev_iou': prev_gt[2], 'new_iou': iou,
                        'gap_since_prev_match': fr - prev_gt[1],
                    }
                    rec.update(summarize_recent(recent_overlap, tid, fr, args.windows))
                    seq_track_switch.append(rec); all_track_switch.append(rec)
                last_gt_for_track[tid] = (gid, fr, iou)

                prev_track = last_track_for_gt.get(gid)
                if prev_track is not None and prev_track[0] != tid:
                    rec = {
                        'seq': seq, 'frame': fr, 'gt_id': gid,
                        'prev_track_id': prev_track[0], 'new_track_id': tid,
                        'prev_frame': prev_track[1], 'prev_iou': prev_track[2], 'new_iou': iou,
                        'gap_since_prev_match': fr - prev_track[1],
                    }
                    # IDSW may be caused by overlap of either old or new track.
                    new_recent = summarize_recent(recent_overlap, tid, fr, args.windows)
                    old_recent = summarize_recent(recent_overlap, prev_track[0], fr, args.windows)
                    for w in args.windows:
                        rec[f'new_track_overlap_within_{w}'] = new_recent[f'overlap_within_{w}']
                        rec[f'new_track_overlap_max_ioa_{w}'] = new_recent[f'overlap_max_ioa_{w}']
                        rec[f'new_track_overlap_partners_{w}'] = new_recent[f'overlap_partners_{w}']
                        rec[f'old_track_overlap_within_{w}'] = old_recent[f'overlap_within_{w}']
                        rec[f'old_track_overlap_max_ioa_{w}'] = old_recent[f'overlap_max_ioa_{w}']
                        rec[f'old_track_overlap_partners_{w}'] = old_recent[f'overlap_partners_{w}']
                        rec[f'either_overlap_within_{w}'] = int(new_recent[f'overlap_within_{w}'] or old_recent[f'overlap_within_{w}'])
                        rec[f'either_overlap_max_ioa_{w}'] = max(new_recent[f'overlap_max_ioa_{w}'], old_recent[f'overlap_max_ioa_{w}'])
                    # crude oracle: if old and new track overlapped recently, local pair correction is plausible.
                    rec['old_new_pair_recent_overlap_10'] = int(any((fr0 <= fr and fr - fr0 <= 10 and partner == prev_track[0]) for fr0, partner, ioa in recent_overlap.get(tid, deque())))
                    seq_idsw.append(rec); all_idsw.append(rec)
                last_track_for_gt[gid] = (tid, fr, iou)

        all_overlap_events.extend(seq_overlap_events)
        # Save per-seq track stats for selected-link endpoint audit.
        clean_stats = {}
        for tid, st in track_stats.items():
            if st['start'] == 10**18:
                continue
            clean_stats[tid] = dict(st)
        track_stats_by_seq[seq] = clean_stats
        track_overlap_summary_by_seq[seq] = clean_stats
        seq_summary = {
            'seq': seq,
            'frames': len(frames),
            'tracker_rows': sum(len(v) for v in tr_by_frame.values()),
            'gt_rows': sum(len(v) for v in gt_by_frame.values()),
            'matched_rows': matched_count,
            'idsw_events_gt_centric': len(seq_idsw),
            'track_gt_switch_events': len(seq_track_switch),
            'overlap_pair_events_written': len(seq_overlap_events),
        }
        for thr, cnt in seq_overlap_counts.items():
            seq_summary[f'overlap_pair_count_ioa_ge_{thr}'] = cnt
        for w in args.windows:
            denom = max(len(seq_idsw), 1)
            seq_summary[f'idsw_either_overlap_within_{w}'] = sum(int(r.get(f'either_overlap_within_{w}', 0)) for r in seq_idsw)
            seq_summary[f'idsw_either_overlap_rate_{w}'] = seq_summary[f'idsw_either_overlap_within_{w}'] / denom
            denom2 = max(len(seq_track_switch), 1)
            seq_summary[f'track_switch_overlap_within_{w}'] = sum(int(r.get(f'overlap_within_{w}', 0)) for r in seq_track_switch)
            seq_summary[f'track_switch_overlap_rate_{w}'] = seq_summary[f'track_switch_overlap_within_{w}'] / denom2
        by_seq_summary.append(seq_summary)
        print(json.dumps(seq_summary, sort_keys=True))

    # Optional selected-link endpoint overlap audit.
    a23_link_rows = []
    if args.selected_links:
        for r in read_selected_links(Path(args.selected_links)):
            seq = r['seq']; a = r['track_a']; b = r['track_b']
            st_a = track_stats_by_seq.get(seq, {}).get(a, {})
            st_b = track_stats_by_seq.get(seq, {}).get(b, {})
            rec = dict(r)
            for prefix, st in [('a', st_a), ('b', st_b)]:
                rec[f'{prefix}_start'] = st.get('start', '')
                rec[f'{prefix}_end'] = st.get('end', '')
                rec[f'{prefix}_rows'] = st.get('rows', '')
                rec[f'{prefix}_max_overlap_ioa'] = st.get('max_overlap_ioa', 0.0)
                rec[f'{prefix}_overlap_frames_030'] = st.get('overlap_frames_030', 0)
                rec[f'{prefix}_overlap_frames_050'] = st.get('overlap_frames_050', 0)
                rec[f'{prefix}_overlap_frames_080'] = st.get('overlap_frames_080', 0)
            rec['either_track_has_overlap_030'] = int(float(rec.get('a_max_overlap_ioa', 0) or 0) >= 0.3 or float(rec.get('b_max_overlap_ioa', 0) or 0) >= 0.3)
            rec['either_track_has_overlap_080'] = int(float(rec.get('a_max_overlap_ioa', 0) or 0) >= 0.8 or float(rec.get('b_max_overlap_ioa', 0) or 0) >= 0.8)
            a23_link_rows.append(rec)

    # Write outputs.
    write_csv(out_dir / 'matched_tracker_gt.csv', all_matched)
    write_csv(out_dir / 'idsw_events.csv', all_idsw)
    write_csv(out_dir / 'track_gt_switch_events.csv', all_track_switch)
    write_csv(out_dir / 'overlap_pair_events.csv', all_overlap_events)
    write_csv(out_dir / 'by_seq_overlap_report.csv', by_seq_summary)
    if a23_link_rows:
        write_csv(out_dir / 'a23_selected_link_overlap_report.csv', a23_link_rows)

    summary = {
        'tracker_dir': str(tracker_dir),
        'gt_root': str(gt_root),
        'seqs': args.seqs,
        'iou_thr': args.iou_thr,
        'event_overlap_thr': args.event_overlap_thr,
        'write_overlap_thr': args.write_overlap_thr,
        'windows': args.windows,
        'total_matched_rows': len(all_matched),
        'total_idsw_events_gt_centric': len(all_idsw),
        'total_track_gt_switch_events': len(all_track_switch),
        'total_overlap_pair_events_written': len(all_overlap_events),
        'by_seq': by_seq_summary,
    }
    for w in args.windows:
        summary[f'idsw_either_overlap_within_{w}'] = sum(int(r.get(f'either_overlap_within_{w}', 0)) for r in all_idsw)
        summary[f'idsw_either_overlap_rate_{w}'] = summary[f'idsw_either_overlap_within_{w}'] / max(len(all_idsw), 1)
        summary[f'track_switch_overlap_within_{w}'] = sum(int(r.get(f'overlap_within_{w}', 0)) for r in all_track_switch)
        summary[f'track_switch_overlap_rate_{w}'] = summary[f'track_switch_overlap_within_{w}'] / max(len(all_track_switch), 1)
    if a23_link_rows:
        false_links = [r for r in a23_link_rows if int(r.get('same_gt', 0)) == 0]
        true_links = [r for r in a23_link_rows if int(r.get('same_gt', 0)) == 1]
        summary['a23_selected_links'] = len(a23_link_rows)
        summary['a23_false_links'] = len(false_links)
        summary['a23_true_links'] = len(true_links)
        summary['a23_false_either_overlap_030'] = sum(int(r['either_track_has_overlap_030']) for r in false_links)
        summary['a23_false_either_overlap_030_rate'] = summary['a23_false_either_overlap_030'] / max(len(false_links), 1)
        summary['a23_true_either_overlap_030'] = sum(int(r['either_track_has_overlap_030']) for r in true_links)
        summary['a23_true_either_overlap_030_rate'] = summary['a23_true_either_overlap_030'] / max(len(true_links), 1)

    (out_dir / 'overlap_vs_idsw_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    md = ['# A31 Overlap Event Audit', '', f"tracker_dir: `{tracker_dir}`", f"gt_root: `{gt_root}`", '', '## Summary', '```json', json.dumps(summary, indent=2, sort_keys=True), '```']
    (out_dir / 'overlap_vs_idsw_summary.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    print('DONE', out_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
