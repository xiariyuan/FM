#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def safe_mean(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def safe_std(xs: List[float]) -> float:
    return float(statistics.pstdev(xs)) if len(xs) > 1 else 0.0


def safe_median(xs: List[float]) -> float:
    return float(statistics.median(xs)) if xs else 0.0


def iou_xywh(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix = max(0.0, min(ax2, bx2) - max(ax, bx))
    iy = max(0.0, min(ay2, by2) - max(ay, by))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    union = max(0.0, aw) * max(0.0, ah) + max(0.0, bw) * max(0.0, bh) - inter
    return inter / union if union > 0 else 0.0


def center(r: dict) -> Tuple[float, float]:
    return (float(r['x']) + float(r['w']) / 2.0, float(r['y']) + float(r['h']) / 2.0)


def bottom_y(r: dict) -> float:
    return float(r['y']) + float(r['h'])


def area(r: dict) -> float:
    return max(0.0, float(r['w'])) * max(0.0, float(r['h']))


def load_mot(path: Path) -> Dict[int, List[dict]]:
    tracks: Dict[int, List[dict]] = defaultdict(list)
    if not path.exists():
        return tracks
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 6:
                continue
            try:
                row = {
                    'frame': int(float(parts[0])),
                    'tid': int(float(parts[1])),
                    'x': float(parts[2]),
                    'y': float(parts[3]),
                    'w': float(parts[4]),
                    'h': float(parts[5]),
                    'score': float(parts[6]) if len(parts) > 6 else 1.0,
                }
            except Exception:
                continue
            tracks[row['tid']].append(row)
    for tid in list(tracks):
        tracks[tid].sort(key=lambda r: r['frame'])
    return tracks


def load_gt(gt_path: Path) -> Dict[int, List[Tuple[int, float, float, float, float]]]:
    by_frame: Dict[int, List[Tuple[int, float, float, float, float]]] = defaultdict(list)
    with gt_path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 6:
                continue
            try:
                frame = int(float(parts[0]))
                gid = int(float(parts[1]))
                x, y, w, h = map(float, parts[2:6])
                mark = int(float(parts[6])) if len(parts) > 6 else 1
                cls = int(float(parts[7])) if len(parts) > 7 else 1
            except Exception:
                continue
            if mark != 1 or cls != 1:
                continue
            by_frame[frame].append((gid, x, y, w, h))
    return by_frame


def best_gt_for_row(row: dict, gt_by_frame: Dict[int, List[Tuple[int, float, float, float, float]]]) -> Tuple[int, float]:
    box = (row['x'], row['y'], row['w'], row['h'])
    best_gid, best_iou = -1, 0.0
    for gid, x, y, w, h in gt_by_frame.get(int(row['frame']), []):
        v = iou_xywh(box, (x, y, w, h))
        if v > best_iou:
            best_gid, best_iou = gid, v
    return best_gid, best_iou


def endpoint_velocity(rows: List[dict], side: str, k: int = 5) -> Tuple[float, float, float]:
    if len(rows) < 2:
        return 0.0, 0.0, 0.0
    sub = rows[-k:] if side == 'end' else rows[:k]
    if len(sub) < 2:
        sub = rows[-2:] if side == 'end' else rows[:2]
    first, last = sub[0], sub[-1]
    dt = max(1, int(last['frame']) - int(first['frame']))
    c1, c2 = center(first), center(last)
    vx = (c2[0] - c1[0]) / dt
    vy = (c2[1] - c1[1]) / dt
    return vx, vy, math.hypot(vx, vy)


def enrich_rows(rows: List[dict], gt_by_frame: dict, iou_thr: float, original_keys: set | None = None) -> List[dict]:
    out = []
    for r in rows:
        rr = dict(r)
        gid, biou = best_gt_for_row(rr, gt_by_frame)
        rr['best_gt'] = gid if biou >= iou_thr else -1
        rr['best_iou'] = biou
        rr['matched'] = int(biou >= iou_thr)
        if original_keys is not None:
            rr['is_interpolated'] = 0 if (rr['tid'], rr['frame']) in original_keys else 1
        else:
            rr['is_interpolated'] = 0
        out.append(rr)
    return out


def track_stats(seq: str, condition: str, tid: int, rows: List[dict]) -> dict:
    n = len(rows)
    frames = [int(r['frame']) for r in rows]
    scores = [float(r['score']) for r in rows]
    areas = [area(r) for r in rows]
    heights = [float(r['h']) for r in rows]
    widths = [float(r['w']) for r in rows]
    bottoms = [bottom_y(r) for r in rows]
    matched = [int(r.get('matched', 0)) for r in rows]
    matched_count = sum(matched)
    matched_gts = [int(r.get('best_gt', -1)) for r in rows if int(r.get('best_gt', -1)) >= 0]
    gt_counter = Counter(matched_gts)
    dominant_gt, dominant_gt_count = (-1, 0) if not gt_counter else gt_counter.most_common(1)[0]
    frame_gaps = [frames[i + 1] - frames[i] for i in range(len(frames) - 1)]
    missing_gap_frames = sum(max(0, g - 1) for g in frame_gaps)
    num_gaps = sum(1 for g in frame_gaps if g > 1)
    max_gap = max([0] + [g - 1 for g in frame_gaps])
    speeds = []
    for a, b in zip(rows, rows[1:]):
        dt = max(1, int(b['frame']) - int(a['frame']))
        ca, cb = center(a), center(b)
        speeds.append(math.hypot(cb[0] - ca[0], cb[1] - ca[1]) / dt)
    vx_end, vy_end, speed_end = endpoint_velocity(rows, 'end')
    vx_start, vy_start, speed_start = endpoint_velocity(rows, 'start')
    gt_switches = 0
    prev_gt = None
    for g in matched_gts:
        if prev_gt is not None and g != prev_gt:
            gt_switches += 1
        prev_gt = g
    interp_count = sum(int(r.get('is_interpolated', 0)) for r in rows)
    matched_ratio = matched_count / n if n else 0.0
    dominant_gt_ratio_total = dominant_gt_count / n if n else 0.0
    dominant_gt_ratio_matched = dominant_gt_count / matched_count if matched_count else 0.0
    label = 'ambiguous_track'
    if n >= 2 and matched_ratio >= 0.6 and dominant_gt_ratio_total >= 0.6:
        label = 'good_track'
    elif matched_ratio <= 0.2:
        label = 'bad_track'
    return {
        'condition': condition,
        'seq': seq,
        'track_id': tid,
        'row_count': n,
        'start_frame': min(frames) if frames else 0,
        'end_frame': max(frames) if frames else 0,
        'duration': (max(frames) - min(frames) + 1) if frames else 0,
        'num_gaps': num_gaps,
        'missing_gap_frames': missing_gap_frames,
        'max_gap': max_gap,
        'interpolated_count': interp_count,
        'interpolated_fraction': interp_count / n if n else 0.0,
        'avg_score': safe_mean(scores),
        'min_score': min(scores) if scores else 0.0,
        'max_score': max(scores) if scores else 0.0,
        'median_score': safe_median(scores),
        'score_std': safe_std(scores),
        'avg_area': safe_mean(areas),
        'median_area': safe_median(areas),
        'area_std': safe_std(areas),
        'avg_height': safe_mean(heights),
        'median_height': safe_median(heights),
        'height_std': safe_std(heights),
        'avg_width': safe_mean(widths),
        'avg_bottom_y': safe_mean(bottoms),
        'avg_center_speed': safe_mean(speeds),
        'max_center_speed': max(speeds) if speeds else 0.0,
        'center_speed_std': safe_std(speeds),
        'vx_start': vx_start,
        'vy_start': vy_start,
        'speed_start': speed_start,
        'vx_end': vx_end,
        'vy_end': vy_end,
        'speed_end': speed_end,
        'matched_count': matched_count,
        'matched_ratio': matched_ratio,
        'dominant_gt': dominant_gt,
        'dominant_gt_count': dominant_gt_count,
        'dominant_gt_ratio_total': dominant_gt_ratio_total,
        'dominant_gt_ratio_matched': dominant_gt_ratio_matched,
        'gt_switches_inside_track': gt_switches,
        'quality_label': label,
        'first_x': rows[0]['x'] if rows else 0.0,
        'first_y': rows[0]['y'] if rows else 0.0,
        'first_w': rows[0]['w'] if rows else 0.0,
        'first_h': rows[0]['h'] if rows else 0.0,
        'first_score': rows[0]['score'] if rows else 0.0,
        'last_x': rows[-1]['x'] if rows else 0.0,
        'last_y': rows[-1]['y'] if rows else 0.0,
        'last_w': rows[-1]['w'] if rows else 0.0,
        'last_h': rows[-1]['h'] if rows else 0.0,
        'last_score': rows[-1]['score'] if rows else 0.0,
    }


def pair_features(a: dict, b: dict) -> dict:
    gap = int(b['start_frame']) - int(a['end_frame'])
    ax = float(a['last_x']) + float(a['last_w']) / 2.0
    ay = float(a['last_y']) + float(a['last_h']) / 2.0
    bx = float(b['first_x']) + float(b['first_w']) / 2.0
    by = float(b['first_y']) + float(b['first_h']) / 2.0
    dist = math.hypot(bx - ax, by - ay)
    pred_x = ax + float(a['vx_end']) * gap
    pred_y = ay + float(a['vy_end']) * gap
    pred_dist = math.hypot(bx - pred_x, by - pred_y)
    va = (float(a['vx_end']), float(a['vy_end']))
    vb = (float(b['vx_start']), float(b['vy_start']))
    na = math.hypot(*va)
    nb = math.hypot(*vb)
    vel_cos = (va[0] * vb[0] + va[1] * vb[1]) / (na * nb) if na > 1e-6 and nb > 1e-6 else 0.0
    h_ratio = max(float(a['last_h']), float(b['first_h'])) / max(1e-6, min(float(a['last_h']), float(b['first_h'])))
    area_a = max(1e-6, float(a['last_w']) * float(a['last_h']))
    area_b = max(1e-6, float(b['first_w']) * float(b['first_h']))
    area_ratio = max(area_a, area_b) / max(1e-6, min(area_a, area_b))
    bottom_gap = abs((float(a['last_y']) + float(a['last_h'])) - (float(b['first_y']) + float(b['first_h'])))
    same_gt = int(int(a['dominant_gt']) >= 0 and int(a['dominant_gt']) == int(b['dominant_gt']))
    return {
        'seq': a['seq'],
        'track_a': a['track_id'],
        'track_b': b['track_id'],
        'gap': gap,
        'center_distance': dist,
        'center_distance_per_frame': dist / max(1, gap),
        'predicted_distance': pred_dist,
        'predicted_distance_per_frame': pred_dist / max(1, gap),
        'velocity_cosine': vel_cos,
        'height_ratio': h_ratio,
        'area_ratio': area_ratio,
        'bottom_y_gap': bottom_gap,
        'len_a': a['row_count'],
        'len_b': b['row_count'],
        'duration_a': a['duration'],
        'duration_b': b['duration'],
        'avg_score_a': a['avg_score'],
        'avg_score_b': b['avg_score'],
        'last_score_a': a['last_score'],
        'first_score_b': b['first_score'],
        'matched_ratio_a': a['matched_ratio'],
        'matched_ratio_b': b['matched_ratio'],
        'dominant_gt_a': a['dominant_gt'],
        'dominant_gt_b': b['dominant_gt'],
        'same_gt': same_gt,
        'quality_label_a': a['quality_label'],
        'quality_label_b': b['quality_label'],
    }


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    fields = list(rows[0].keys())
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def summarize_quality(rows: List[dict]) -> dict:
    by_cond = defaultdict(list)
    for r in rows:
        by_cond[r['condition']].append(r)
    out = {}
    for cond, rs in by_cond.items():
        c = Counter(r['quality_label'] for r in rs)
        out[cond] = {
            'tracks': len(rs),
            'good': c.get('good_track', 0),
            'bad': c.get('bad_track', 0),
            'ambiguous': c.get('ambiguous_track', 0),
            'mean_matched_ratio': safe_mean([float(r['matched_ratio']) for r in rs]),
            'mean_interpolated_fraction': safe_mean([float(r['interpolated_fraction']) for r in rs]),
            'short_tracks_len_le_3': sum(1 for r in rs if int(r['row_count']) <= 3),
            'low_match_tracks': sum(1 for r in rs if float(r['matched_ratio']) <= 0.2),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--online-dir', required=True)
    ap.add_argument('--interp-dir', required=True)
    ap.add_argument('--gt-root', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--iou-thresh', type=float, default=0.5)
    ap.add_argument('--max-link-gap', type=int, default=60)
    ap.add_argument('--max-center-step', type=float, default=80.0)
    ap.add_argument('--max-area-ratio', type=float, default=4.0)
    ap.add_argument('--min-track-len-for-link', type=int, default=5)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    online_dir = Path(args.online_dir)
    interp_dir = Path(args.interp_dir)
    gt_root = Path(args.gt_root)

    all_quality: List[dict] = []
    online_tracklets: List[dict] = []
    pairs: List[dict] = []

    seqs = sorted([p.stem for p in online_dir.glob('MOT20-*.txt')])
    for seq in seqs:
        gt_by_frame = load_gt(gt_root / seq / 'gt' / 'gt.txt')
        online_tracks = load_mot(online_dir / f'{seq}.txt')
        interp_tracks = load_mot(interp_dir / f'{seq}.txt')
        online_keys = {(tid, r['frame']) for tid, rows in online_tracks.items() for r in rows}

        seq_online_stats = []
        for tid, rows in sorted(online_tracks.items()):
            enriched = enrich_rows(rows, gt_by_frame, args.iou_thresh, None)
            st = track_stats(seq, 'online', tid, enriched)
            all_quality.append(st)
            online_tracklets.append(st)
            seq_online_stats.append(st)

        for tid, rows in sorted(interp_tracks.items()):
            enriched = enrich_rows(rows, gt_by_frame, args.iou_thresh, online_keys)
            st = track_stats(seq, 'interp_gap30', tid, enriched)
            all_quality.append(st)

        # AFLink-lite candidate pairs from online tracklets.
        valid = [r for r in seq_online_stats if int(r['row_count']) >= args.min_track_len_for_link and float(r['matched_ratio']) > 0.0]
        valid.sort(key=lambda r: (int(r['start_frame']), int(r['end_frame'])))
        for i, a in enumerate(valid):
            for b in valid:
                gap = int(b['start_frame']) - int(a['end_frame'])
                if gap <= 0 or gap > args.max_link_gap:
                    continue
                pf = pair_features(a, b)
                if float(pf['center_distance_per_frame']) > args.max_center_step:
                    continue
                if float(pf['area_ratio']) > args.max_area_ratio:
                    continue
                pairs.append(pf)

    write_csv(out_dir / 'track_quality_rows.csv', all_quality)
    write_csv(out_dir / 'tracklet_rows.csv', online_tracklets)
    write_csv(out_dir / 'aflink_pair_candidates.csv', pairs)

    quality_summary = summarize_quality(all_quality)
    pair_counter = Counter((p['seq'], p['same_gt']) for p in pairs)
    pair_summary = {
        'candidate_pairs': len(pairs),
        'positive_pairs': sum(int(p['same_gt']) for p in pairs),
        'negative_pairs': len(pairs) - sum(int(p['same_gt']) for p in pairs),
        'positive_rate': (sum(int(p['same_gt']) for p in pairs) / len(pairs)) if pairs else 0.0,
        'by_seq': {},
    }
    for seq in seqs:
        pos = pair_counter.get((seq, 1), 0)
        neg = pair_counter.get((seq, 0), 0)
        pair_summary['by_seq'][seq] = {'pairs': pos + neg, 'positive': pos, 'negative': neg, 'positive_rate': pos / (pos + neg) if pos + neg else 0.0}

    (out_dir / 'track_quality_summary.json').write_text(json.dumps(quality_summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    (out_dir / 'aflink_candidate_summary.json').write_text(json.dumps(pair_summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')

    md = ['# A22_00 Track Quality Summary', '']
    for cond, s in quality_summary.items():
        md.append(f'## {cond}')
        md.append('| metric | value |')
        md.append('|---|---:|')
        for k, v in s.items():
            md.append(f'| {k} | {v} |')
        md.append('')
    (out_dir / 'track_quality_summary.md').write_text('\n'.join(md) + '\n', encoding='utf-8')

    md = ['# A22_00 AFLink Candidate Summary', '', '| seq | pairs | positive | negative | positive_rate |', '|---|---:|---:|---:|---:|']
    for seq, s in pair_summary['by_seq'].items():
        md.append(f"| {seq} | {s['pairs']} | {s['positive']} | {s['negative']} | {s['positive_rate']:.4f} |")
    md += ['', f"total_pairs: {pair_summary['candidate_pairs']}", f"positive_pairs: {pair_summary['positive_pairs']}", f"positive_rate: {pair_summary['positive_rate']:.4f}"]
    (out_dir / 'aflink_candidate_summary.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    print(json.dumps({'quality_summary': quality_summary, 'pair_summary': pair_summary}, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
