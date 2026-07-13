#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
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
            p = line.split(',')
            if len(p) < 6:
                continue
            try:
                fr = int(float(p[0])); tid = int(float(p[1]))
                x = float(p[2]); y = float(p[3]); w = float(p[4]); h = float(p[5])
            except Exception:
                continue
            if w <= 0 or h <= 0:
                continue
            score = float(p[6]) if len(p) > 6 and p[6] != '' else 1.0
            if is_gt:
                mark = int(float(p[6])) if len(p) > 6 and p[6] != '' else 1
                cls = int(float(p[7])) if len(p) > 7 and p[7] != '' else 1
                if mark <= 0 or cls != 1:
                    continue
            by_frame[fr].append({'frame': fr, 'id': tid, 'x': x, 'y': y, 'w': w, 'h': h, 'score': score})
    return by_frame


def box_arr(rows: List[dict]) -> np.ndarray:
    a = np.zeros((len(rows), 4), dtype=np.float32)
    for i, r in enumerate(rows):
        a[i] = [r['x'], r['y'], r['x'] + r['w'], r['y'] + r['h']]
    return a


def pair_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    aa = np.clip((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), 1e-6, None)
    bb = np.clip((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]), 1e-6, None)
    return inter / np.clip(aa[:, None] + bb[None, :] - inter, 1e-6, None)


def pair_ioa_minarea(a: np.ndarray) -> np.ndarray:
    n = len(a)
    out = np.zeros((n, n), dtype=np.float32)
    if n == 0:
        return out
    lt = np.maximum(a[:, None, :2], a[None, :, :2])
    rb = np.minimum(a[:, None, 2:], a[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area = np.clip((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), 1e-6, None)
    out = inter / np.clip(np.minimum(area[:, None], area[None, :]), 1e-6, None)
    np.fill_diagonal(out, 0.0)
    return out


def match_frame(tracks: List[dict], gts: List[dict], thr: float) -> List[Tuple[int, int, float]]:
    if not tracks or not gts:
        return []
    ious = pair_iou(box_arr(tracks), box_arr(gts))
    ri, ci = linear_sum_assignment(1.0 - ious)
    out = []
    for r, c in zip(ri, ci):
        val = float(ious[r, c])
        if val >= thr:
            out.append((r, c, val))
    return out


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or x == '':
            return default
        return float(x)
    except Exception:
        return default


def build_track_gt_maps(track_dir: Path, gt_root: Path, seqs: List[str], iou_thr: float):
    match_gt = {}  # (seq, frame, track_id) -> {gt_id,iou}
    track_gt_switch_keys = set()
    gt_idsw_keys = set()
    overlap_max = {}  # (seq, frame, track_id) -> max_ioa
    seq_reports = []
    all_switch_rows = []
    all_idsw_rows = []
    for seq in seqs:
        tr = read_mot(track_dir / f'{seq}.txt', is_gt=False)
        gt = read_mot(gt_root / seq / 'gt' / 'gt.txt', is_gt=True)
        frames = sorted(set(tr) | set(gt))
        last_gt_for_track = {}
        last_track_for_gt = {}
        seq_switch = 0
        seq_idsw = 0
        for fr in frames:
            tracks = tr.get(fr, [])
            gts = gt.get(fr, [])
            # Overlap max per track from tracker output rows.
            boxes = box_arr(tracks)
            ioa = pair_ioa_minarea(boxes)
            for i, row in enumerate(tracks):
                mx = float(np.max(ioa[i])) if len(tracks) > 1 else 0.0
                overlap_max[(seq, fr, int(row['id']))] = mx
            matches = match_frame(tracks, gts, iou_thr)
            for ti, gi, iou in matches:
                tid = int(tracks[ti]['id']); gid = int(gts[gi]['id'])
                match_gt[(seq, fr, tid)] = {'gt_id': gid, 'iou': iou}
                prev_gt = last_gt_for_track.get(tid)
                if prev_gt is not None and prev_gt[0] != gid:
                    track_gt_switch_keys.add((seq, fr, tid))
                    seq_switch += 1
                    all_switch_rows.append({'seq': seq, 'frame': fr, 'track_id': tid, 'prev_gt': prev_gt[0], 'new_gt': gid, 'prev_frame': prev_gt[1], 'gap': fr - prev_gt[1]})
                last_gt_for_track[tid] = (gid, fr, iou)
                prev_track = last_track_for_gt.get(gid)
                if prev_track is not None and prev_track[0] != tid:
                    gt_idsw_keys.add((seq, fr, tid))
                    seq_idsw += 1
                    all_idsw_rows.append({'seq': seq, 'frame': fr, 'gt_id': gid, 'prev_track': prev_track[0], 'new_track': tid, 'prev_frame': prev_track[1], 'gap': fr - prev_track[1]})
                last_track_for_gt[gid] = (tid, fr, iou)
        seq_reports.append({'seq': seq, 'frames': len(frames), 'track_gt_switch_events': seq_switch, 'gt_idsw_events': seq_idsw, 'matched_track_gt_rows': sum(1 for k in match_gt if k[0] == seq)})
    return match_gt, track_gt_switch_keys, gt_idsw_keys, overlap_max, seq_reports, all_switch_rows, all_idsw_rows


def write_csv(path: Path, rows: List[dict], fields: List[str] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        seen = set()
        for r in rows:
            for k in r:
                if k not in seen:
                    seen.add(k); fields.append(k)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--track-dir', required=True)
    ap.add_argument('--spot-dir', required=True)
    ap.add_argument('--gt-root', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--seqs', nargs='+', default=['MOT20-01','MOT20-02','MOT20-03','MOT20-05'])
    ap.add_argument('--iou-thr', type=float, default=0.5)
    ap.add_argument('--margin-thrs', type=float, nargs='+', default=[0.02,0.05,0.10])
    ap.add_argument('--overlap-thrs', type=float, nargs='+', default=[0.3,0.5,0.8])
    ap.add_argument('--windows', type=int, nargs='+', default=[1,3,5,10])
    ap.add_argument('--sample-limit', type=int, default=5000)
    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    match_gt, switch_keys, idsw_keys, overlap_max, seq_reports, switch_rows, idsw_rows = build_track_gt_maps(Path(args.track_dir), Path(args.gt_root), args.seqs, args.iou_thr)
    max_w = max(args.windows)
    overall = defaultdict(int)
    by_seq = {s: defaultdict(int) for s in args.seqs}
    lookback = defaultdict(deque)  # (seq, track_id) -> event feature deque
    switch_event_samples = []
    idsw_event_samples = []
    bad_commit_samples = []
    all_event_sample = []
    lookback_reports = []

    for seq in args.seqs:
        pair_file = Path(args.spot_dir) / f'{seq}_pairs.csv'
        if not pair_file.exists():
            continue
        with pair_file.open(newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                fr = int(float(row.get('frame', row.get('frame_id', 0)) or 0))
                tid = int(float(row.get('track_id', -1) or -1))
                if fr <= 0 or tid < 0:
                    continue
                key = (seq, fr, tid)
                gtinfo = match_gt.get(key)
                gt_id = int(gtinfo['gt_id']) if gtinfo else -1
                iou = float(gtinfo['iou']) if gtinfo else 0.0
                margin = safe_float(row.get('spot_margin', row.get('cost_margin')), default=999.0)
                row_margin = safe_float(row.get('row_margin'), default=999.0)
                col_margin = safe_float(row.get('col_margin'), default=999.0)
                cost = safe_float(row.get('cost'), default=999.0)
                ov = float(overlap_max.get(key, 0.0))
                matched = int(gt_id >= 0)
                track_switch = int(key in switch_keys)
                gt_idsw = int(key in idsw_keys)
                low_flags = {f'low_margin_le_{thr}': int(margin <= thr) for thr in args.margin_thrs}
                ov_flags = {f'overlap_ge_{thr}': int(ov >= thr) for thr in args.overlap_thrs}
                ev = {
                    'seq': seq, 'frame': fr, 'track_id': tid, 'gt_id': gt_id,
                    'matched_gt': matched, 'track_gt_switch': track_switch, 'gt_idsw': gt_idsw,
                    'spot_margin': margin, 'row_margin': row_margin, 'col_margin': col_margin, 'cost': cost,
                    'det_score': safe_float(row.get('det_score'), 0.0),
                    'track_age': int(safe_float(row.get('track_age'), 0)),
                    'lost_age': int(safe_float(row.get('lost_age'), 0)),
                    'overlap_max_ioa': ov,
                    'iou_to_gt': iou,
                }
                ev.update(low_flags); ev.update(ov_flags)
                overall['primary_match_events'] += 1; by_seq[seq]['primary_match_events'] += 1
                overall['matched_gt_events'] += matched; by_seq[seq]['matched_gt_events'] += matched
                overall['unmatched_fp_events'] += int(not matched); by_seq[seq]['unmatched_fp_events'] += int(not matched)
                overall['track_gt_switch_events_seen_in_primary'] += track_switch; by_seq[seq]['track_gt_switch_events_seen_in_primary'] += track_switch
                overall['gt_idsw_events_seen_in_primary'] += gt_idsw; by_seq[seq]['gt_idsw_events_seen_in_primary'] += gt_idsw
                for thr in args.margin_thrs:
                    if margin <= thr:
                        overall[f'events_low_margin_le_{thr}'] += 1; by_seq[seq][f'events_low_margin_le_{thr}'] += 1
                        if track_switch:
                            overall[f'track_switch_low_margin_le_{thr}'] += 1; by_seq[seq][f'track_switch_low_margin_le_{thr}'] += 1
                        if gt_idsw:
                            overall[f'gt_idsw_low_margin_le_{thr}'] += 1; by_seq[seq][f'gt_idsw_low_margin_le_{thr}'] += 1
                for thr in args.overlap_thrs:
                    if ov >= thr:
                        overall[f'events_overlap_ge_{thr}'] += 1; by_seq[seq][f'events_overlap_ge_{thr}'] += 1
                        if track_switch:
                            overall[f'track_switch_overlap_ge_{thr}'] += 1; by_seq[seq][f'track_switch_overlap_ge_{thr}'] += 1
                        if gt_idsw:
                            overall[f'gt_idsw_overlap_ge_{thr}'] += 1; by_seq[seq][f'gt_idsw_overlap_ge_{thr}'] += 1
                # update lookback then compute if current event is an error event
                dq = lookback[(seq, tid)]
                dq.append({'frame': fr, 'margin': margin, 'overlap': ov})
                while dq and fr - dq[0]['frame'] > max_w:
                    dq.popleft()
                if track_switch or gt_idsw:
                    lb = {'seq': seq, 'frame': fr, 'track_id': tid, 'gt_id': gt_id, 'track_gt_switch': track_switch, 'gt_idsw': gt_idsw, 'spot_margin': margin, 'overlap_max_ioa': ov}
                    for w in args.windows:
                        vals = [e for e in dq if 0 <= fr - e['frame'] <= w]
                        for thr in args.margin_thrs:
                            hit = int(any(e['margin'] <= thr for e in vals))
                            lb[f'lookback_{w}_low_margin_le_{thr}'] = hit
                            if track_switch:
                                overall[f'track_switch_lookback_{w}_low_margin_le_{thr}'] += hit
                            if gt_idsw:
                                overall[f'gt_idsw_lookback_{w}_low_margin_le_{thr}'] += hit
                        for thr in args.overlap_thrs:
                            hit = int(any(e['overlap'] >= thr for e in vals))
                            lb[f'lookback_{w}_overlap_ge_{thr}'] = hit
                            if track_switch:
                                overall[f'track_switch_lookback_{w}_overlap_ge_{thr}'] += hit
                            if gt_idsw:
                                overall[f'gt_idsw_lookback_{w}_overlap_ge_{thr}'] += hit
                    if len(lookback_reports) < args.sample_limit:
                        lookback_reports.append(lb)
                    if track_switch and len(switch_event_samples) < args.sample_limit:
                        switch_event_samples.append(ev)
                    if gt_idsw and len(idsw_event_samples) < args.sample_limit:
                        idsw_event_samples.append(ev)
                    if len(bad_commit_samples) < args.sample_limit:
                        bad_commit_samples.append(ev)
                if len(all_event_sample) < args.sample_limit:
                    all_event_sample.append(ev)

    # derived rates
    summary = {
        'inputs': {'track_dir': args.track_dir, 'spot_dir': args.spot_dir, 'gt_root': args.gt_root},
        'seqs': args.seqs,
        'iou_thr': args.iou_thr,
        'margin_thrs': args.margin_thrs,
        'overlap_thrs': args.overlap_thrs,
        'windows': args.windows,
        'seq_gt_track_reports': seq_reports,
        'overall_counts': dict(overall),
        'by_seq_counts': {k: dict(v) for k,v in by_seq.items()},
        'limitations': [
            'Current SPOT pair logs contain selected primary matches and margins, but not full top-k candidate identities.',
            'candidate_contains_true_identity requires a deeper online candidate logger in A37_00b.'
        ],
    }
    def rate(num, den): return float(num) / float(den) if den else 0.0
    derived = {}
    den_sw = overall.get('track_gt_switch_events_seen_in_primary', 0)
    den_id = overall.get('gt_idsw_events_seen_in_primary', 0)
    den_ev = overall.get('primary_match_events', 0)
    for thr in args.margin_thrs:
        derived[f'event_low_margin_rate_le_{thr}'] = rate(overall.get(f'events_low_margin_le_{thr}',0), den_ev)
        derived[f'track_switch_low_margin_rate_le_{thr}'] = rate(overall.get(f'track_switch_low_margin_le_{thr}',0), den_sw)
        derived[f'gt_idsw_low_margin_rate_le_{thr}'] = rate(overall.get(f'gt_idsw_low_margin_le_{thr}',0), den_id)
    for thr in args.overlap_thrs:
        derived[f'event_overlap_rate_ge_{thr}'] = rate(overall.get(f'events_overlap_ge_{thr}',0), den_ev)
        derived[f'track_switch_overlap_rate_ge_{thr}'] = rate(overall.get(f'track_switch_overlap_ge_{thr}',0), den_sw)
        derived[f'gt_idsw_overlap_rate_ge_{thr}'] = rate(overall.get(f'gt_idsw_overlap_ge_{thr}',0), den_id)
    for w in args.windows:
        for thr in args.margin_thrs:
            derived[f'track_switch_lookback_{w}_low_margin_rate_le_{thr}'] = rate(overall.get(f'track_switch_lookback_{w}_low_margin_le_{thr}',0), den_sw)
            derived[f'gt_idsw_lookback_{w}_low_margin_rate_le_{thr}'] = rate(overall.get(f'gt_idsw_lookback_{w}_low_margin_le_{thr}',0), den_id)
        for thr in args.overlap_thrs:
            derived[f'track_switch_lookback_{w}_overlap_rate_ge_{thr}'] = rate(overall.get(f'track_switch_lookback_{w}_overlap_ge_{thr}',0), den_sw)
            derived[f'gt_idsw_lookback_{w}_overlap_rate_ge_{thr}'] = rate(overall.get(f'gt_idsw_lookback_{w}_overlap_ge_{thr}',0), den_id)
    summary['derived_rates'] = derived
    (out_dir / 'a37_event_audit_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    write_csv(out_dir / 'event_sample.csv', all_event_sample)
    write_csv(out_dir / 'bad_commit_event_sample.csv', bad_commit_samples)
    write_csv(out_dir / 'track_switch_event_sample.csv', switch_event_samples)
    write_csv(out_dir / 'gt_idsw_event_sample.csv', idsw_event_samples)
    write_csv(out_dir / 'error_event_lookback_sample.csv', lookback_reports)
    write_csv(out_dir / 'offline_track_gt_switch_events.csv', switch_rows)
    write_csv(out_dir / 'offline_gt_idsw_events.csv', idsw_rows)
    md = []
    md.append('# A37 Identity Escrow Event Audit')
    md.append('')
    md.append('## Key rates')
    for k in sorted(derived):
        if any(x in k for x in ['low_margin_rate_le_0.05','overlap_rate_ge_0.5','overlap_rate_ge_0.8']):
            md.append(f'- {k}: {derived[k]:.4f}')
    md.append('')
    md.append('## Counts')
    md.append('```json')
    md.append(json.dumps(dict(overall), indent=2, sort_keys=True))
    md.append('```')
    md.append('')
    md.append('## Limitations')
    for x in summary['limitations']:
        md.append(f'- {x}')
    (out_dir / 'a37_event_audit_summary.md').write_text('\n'.join(md)+'\n', encoding='utf-8')
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
