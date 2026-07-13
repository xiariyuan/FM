#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def ff(x, d=0.0):
    try:
        return float(x)
    except Exception:
        return d


def ii(x, d=0):
    try:
        return int(float(x))
    except Exception:
        return d


def center(r):
    return r['x'] + r['w'] / 2.0, r['y'] + r['h'] / 2.0


def area(r):
    return max(1e-9, r['w'] * r['h'])


def read_tracks(path: Path):
    tracks = defaultdict(list)
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            p = line.strip().split(',')
            if len(p) < 6:
                continue
            r = {
                'parts': p,
                'line_no': line_no,
                'frame': ii(p[0]),
                'tid': ii(p[1]),
                'x': ff(p[2]),
                'y': ff(p[3]),
                'w': ff(p[4]),
                'h': ff(p[5]),
                'score': ff(p[6], 1.0) if len(p) > 6 else 1.0,
            }
            tracks[r['tid']].append(r)
    for rs in tracks.values():
        rs.sort(key=lambda r: (r['frame'], r['line_no']))
    return tracks


def transition_features(a, b):
    dt = max(1, b['frame'] - a['frame'])
    ca, cb = center(a), center(b)
    center_step = math.hypot(cb[0] - ca[0], cb[1] - ca[1]) / dt
    height_ratio = max(a['h'], b['h']) / max(1e-9, min(a['h'], b['h']))
    area_ratio = max(area(a), area(b)) / max(1e-9, min(area(a), area(b)))
    score_min = min(a['score'], b['score'])
    score_drop = abs(a['score'] - b['score'])
    missing_gap = max(0, b['frame'] - a['frame'] - 1)
    risk = score_drop * center_step * max(0.0, height_ratio - 1.0)
    return {
        'score_min': score_min,
        'score_drop': score_drop,
        'center_step': center_step,
        'height_ratio': height_ratio,
        'area_ratio': area_ratio,
        'missing_gap': missing_gap,
        'risk': risk,
    }


def is_candidate(feat, args):
    return (
        feat['score_min'] <= args.score_min_le and
        feat['score_drop'] >= args.score_drop_ge and
        feat['center_step'] >= args.center_step_ge and
        feat['height_ratio'] >= args.height_ratio_ge and
        feat['area_ratio'] >= args.area_ratio_ge and
        feat['missing_gap'] >= args.missing_gap_ge
    )


def select_cuts(tracks, args, seq_budget):
    candidates = []
    track_lengths = {tid: len(rs) for tid, rs in tracks.items()}
    for tid, rs in tracks.items():
        n = len(rs)
        for i in range(1, n):
            if i < args.min_seg_len or n - i < args.min_seg_len:
                continue
            feat = transition_features(rs[i - 1], rs[i])
            if is_candidate(feat, args):
                candidates.append({'tid': tid, 'idx': i, **feat})
    candidates.sort(key=lambda x: (-x['risk'], -x['score_drop'], -x['center_step']))
    selected = defaultdict(list)
    selected_flat = []
    for c in candidates:
        if len(selected_flat) >= seq_budget:
            break
        tid = c['tid']
        n = track_lengths[tid]
        idx = c['idx']
        existing = sorted(selected[tid])
        cuts = [0] + existing + [n]
        ok = True
        for a, b in zip(cuts[:-1], cuts[1:]):
            if a < idx < b:
                if idx - a < args.min_seg_len or b - idx < args.min_seg_len:
                    ok = False
                break
        if not ok:
            continue
        selected[tid].append(idx)
        selected_flat.append(c)
    return selected, candidates, selected_flat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--budget-pct', type=float, default=0.04)
    ap.add_argument('--budget-min', type=int, default=0)
    ap.add_argument('--budget-max', type=int, default=999999)
    ap.add_argument('--score-min-le', type=float, default=0.8)
    ap.add_argument('--score-drop-ge', type=float, default=0.05)
    ap.add_argument('--center-step-ge', type=float, default=12.0)
    ap.add_argument('--height-ratio-ge', type=float, default=1.05)
    ap.add_argument('--area-ratio-ge', type=float, default=1.0)
    ap.add_argument('--missing-gap-ge', type=int, default=0)
    ap.add_argument('--min-seg-len', type=int, default=5)
    ap.add_argument('--summary-json', default='')
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        'budget_pct': args.budget_pct,
        'params': vars(args),
        'tracks': 0,
        'rows': 0,
        'candidate_points': 0,
        'split_points': 0,
        'tracks_split': 0,
        'extra_segments': 0,
        'by_seq': [],
    }
    for txt in sorted(Path(args.input_dir).glob('MOT20-*.txt')):
        seq = txt.stem
        tracks = read_tracks(txt)
        seq_budget = int(round(len(tracks) * args.budget_pct))
        seq_budget = max(args.budget_min, min(args.budget_max, seq_budget))
        selected, candidates, selected_flat = select_cuts(tracks, args, seq_budget)
        max_tid = max(tracks) if tracks else 0
        next_tid = max_tid + 1
        out_rows = []
        seq_sum = {
            'seq': seq,
            'tracks': len(tracks),
            'budget': seq_budget,
            'candidate_points': len(candidates),
            'split_points': len(selected_flat),
            'tracks_split': len(selected),
            'extra_segments': len(selected_flat),
            'rows': sum(len(v) for v in tracks.values()),
        }
        for tid, rs in sorted(tracks.items()):
            cut_idxs = sorted(selected.get(tid, []))
            cuts = [0] + cut_idxs + [len(rs)]
            for seg_i, (s, e) in enumerate(zip(cuts[:-1], cuts[1:])):
                new_tid = tid if seg_i == 0 else next_tid
                if seg_i > 0:
                    next_tid += 1
                for r in rs[s:e]:
                    p = list(r['parts'])
                    p[1] = str(new_tid)
                    out_rows.append((r['frame'], new_tid, r['line_no'], p))
        out_rows.sort(key=lambda x: (x[0], x[1], x[2]))
        with (out / txt.name).open('w') as f:
            for _, _, _, p in out_rows:
                f.write(','.join(p) + '\n')
        for k in ['tracks', 'rows', 'candidate_points', 'split_points', 'tracks_split', 'extra_segments']:
            summary[k] += seq_sum[k]
        summary['by_seq'].append(seq_sum)
    if args.summary_json:
        Path(args.summary_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
