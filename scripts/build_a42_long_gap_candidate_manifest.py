#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


def af(v, d=0.0):
    try:
        if v is None or v == '':
            return d
        return float(v)
    except Exception:
        return d


def ai(v, d=0):
    try:
        if v is None or v == '':
            return d
        return int(float(v))
    except Exception:
        return d


def read_csv(path: Path) -> List[dict]:
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    fields, seen = [], set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fields.append(k)
    if not fields:
        fields = ['seq']
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def load_reid(npz_path: Path) -> dict:
    z = np.load(npz_path)
    idx = {(str(seq), str(int(tid))): i for i, (seq, tid) in enumerate(zip(z['seq'], z['track_id']))}
    return {
        'idx': idx,
        'start': z['start'].astype(np.float32),
        'end': z['end'].astype(np.float32),
        'global_mean': z['global_mean'].astype(np.float32),
        'high_score': z['high_score'].astype(np.float32),
    }


def debt_tags(t: dict, side: str) -> List[str]:
    tags = []
    row_count = af(t.get('row_count'))
    duration = af(t.get('duration'), row_count)
    avg_score = af(t.get('avg_score'), 1.0)
    first_score = af(t.get('first_score'), 1.0)
    last_score = af(t.get('last_score'), 1.0)
    if row_count < 30:
        tags.append('short_tracklet')
    if row_count < 10:
        tags.append('very_short_tracklet')
    if avg_score < 0.75:
        tags.append('low_avg_score')
    if side == 'source' and last_score < 0.35:
        tags.append('weak_end_boundary')
    if side == 'target' and first_score < 0.35:
        tags.append('weak_start_boundary')
    if duration > 0 and row_count / duration < 0.85:
        tags.append('internal_gaps')
    if abs(af(t.get('vx_start')) - af(t.get('vx_end'))) + abs(af(t.get('vy_start')) - af(t.get('vy_end'))) > 20:
        tags.append('motion_instability')
    return tags


def center_from(t: dict, prefix: str) -> Tuple[float, float]:
    x, y, w, h = af(t.get(f'{prefix}_x')), af(t.get(f'{prefix}_y')), af(t.get(f'{prefix}_w')), af(t.get(f'{prefix}_h'))
    return x + w / 2.0, y + h / 2.0


def bottom_from(t: dict, prefix: str) -> float:
    return af(t.get(f'{prefix}_y')) + af(t.get(f'{prefix}_h'))


def zone_bucket(v: float, width: float) -> int:
    if width <= 0:
        return 0
    return int(max(0, min(9, v // width)))


def gap_bucket(gap: int) -> str:
    if gap <= 60:
        return '1_60'
    if gap <= 150:
        return '61_150'
    return '151_300'


def build_base_candidates(tracklets: List[dict], max_gap: int, split: str) -> List[dict]:
    by_seq = defaultdict(list)
    for t in tracklets:
        by_seq[t.get('seq', '')].append(t)
    rows = []
    for seq, ts in sorted(by_seq.items()):
        ts = sorted(ts, key=lambda r: (ai(r.get('start_frame')), ai(r.get('end_frame')), ai(r.get('track_id'))))
        n = len(ts)
        for i, a in enumerate(ts):
            ea = ai(a.get('end_frame'))
            ca = center_from(a, 'last')
            for b in ts:
                sb = ai(b.get('start_frame'))
                if sb <= ea:
                    continue
                gap = sb - ea
                if gap > max_gap:
                    break
                cb = center_from(b, 'first')
                center_distance = ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5
                pred_x = ca[0] + af(a.get('vx_end')) * gap
                pred_y = ca[1] + af(a.get('vy_end')) * gap
                predicted_distance = ((pred_x - cb[0]) ** 2 + (pred_y - cb[1]) ** 2) ** 0.5
                area_a = max(1e-6, af(a.get('last_w')) * af(a.get('last_h')))
                area_b = max(1e-6, af(b.get('first_w')) * af(b.get('first_h')))
                height_ratio = af(b.get('first_h')) / max(1e-6, af(a.get('last_h')))
                area_ratio = area_b / area_a
                # velocity cosine between source exit and target entry trend.
                va = np.array([af(a.get('vx_end')), af(a.get('vy_end'))], dtype=np.float32)
                vb = np.array([af(b.get('vx_start')), af(b.get('vy_start'))], dtype=np.float32)
                denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
                velocity_cosine = float(np.dot(va, vb) / denom) if denom > 1e-6 else 0.0
                gt_a, gt_b = ai(a.get('dominant_gt'), -1), ai(b.get('dominant_gt'), -2)
                same_gt = int(gt_a > 0 and gt_a == gt_b and af(a.get('dominant_gt_ratio_total'), 0) >= 0.5 and af(b.get('dominant_gt_ratio_total'), 0) >= 0.5)
                stags, ttags = debt_tags(a, 'source'), debt_tags(b, 'target')
                x_end_bucket = zone_bucket(ca[0], 200.0)
                x_start_bucket = zone_bucket(cb[0], 200.0)
                bottom_end_bucket = zone_bucket(bottom_from(a, 'last'), 100.0)
                bottom_start_bucket = zone_bucket(bottom_from(b, 'first'), 100.0)
                row = {
                    'split': split,
                    'seq': seq,
                    'track_a': str(ai(a.get('track_id'))),
                    'track_b': str(ai(b.get('track_id'))),
                    'gap': gap,
                    'gap_bucket': gap_bucket(gap),
                    'len_a': a.get('row_count', ''),
                    'len_b': b.get('row_count', ''),
                    'duration_a': a.get('duration', ''),
                    'duration_b': b.get('duration', ''),
                    'avg_score_a': a.get('avg_score', ''),
                    'avg_score_b': b.get('avg_score', ''),
                    'last_score_a': a.get('last_score', ''),
                    'first_score_b': b.get('first_score', ''),
                    'start_frame_a': a.get('start_frame', ''),
                    'end_frame_a': a.get('end_frame', ''),
                    'start_frame_b': b.get('start_frame', ''),
                    'end_frame_b': b.get('end_frame', ''),
                    'center_distance': center_distance,
                    'center_distance_per_frame': center_distance / max(1, gap),
                    'predicted_distance': predicted_distance,
                    'predicted_distance_per_frame': predicted_distance / max(1, gap),
                    'velocity_cosine': velocity_cosine,
                    'height_ratio': height_ratio,
                    'area_ratio': area_ratio,
                    'bottom_y_gap': bottom_from(b, 'first') - bottom_from(a, 'last'),
                    'x_end_bucket': x_end_bucket,
                    'x_start_bucket': x_start_bucket,
                    'x_bucket_delta': abs(x_end_bucket - x_start_bucket),
                    'bottom_end_bucket': bottom_end_bucket,
                    'bottom_start_bucket': bottom_start_bucket,
                    'bottom_bucket_delta': abs(bottom_end_bucket - bottom_start_bucket),
                    'zone_exact_match': int(x_end_bucket == x_start_bucket and bottom_end_bucket == bottom_start_bucket),
                    'zone_near_match': int(abs(x_end_bucket - x_start_bucket) <= 1 and abs(bottom_end_bucket - bottom_start_bucket) <= 1),
                    'source_debt_tags': '|'.join(stags),
                    'target_debt_tags': '|'.join(ttags),
                    'source_debt_count': len(stags),
                    'target_debt_count': len(ttags),
                    'edge_debt_score': len(stags) + len(ttags),
                    'geometry_risk': int(not (0.55 <= height_ratio <= 1.8)) + int(not (0.25 <= area_ratio <= 4.0)) + int(abs(bottom_from(b, 'first') - bottom_from(a, 'last')) > 320),
                    'motion_risk': int(predicted_distance / max(1, gap) > 15) + int(center_distance / max(1, gap) > 20) + int(velocity_cosine < -0.35),
                    'dominant_gt_a': gt_a if split == 'train' else -1,
                    'dominant_gt_b': gt_b if split == 'train' else -1,
                    'same_gt': same_gt if split == 'train' else 0,
                }
                rows.append(row)
    return rows


def add_reid_features(rows: List[dict], reid: dict, chunk_size: int = 20000) -> None:
    idx = reid['idx']
    missing = 0
    for start in range(0, len(rows), chunk_size):
        chunk = rows[start:start + chunk_size]
        ia, ib, valid = [], [], []
        for j, r in enumerate(chunk):
            ka, kb = (r['seq'], str(ai(r['track_a']))), (r['seq'], str(ai(r['track_b'])))
            if ka in idx and kb in idx:
                ia.append(idx[ka]); ib.append(idx[kb]); valid.append(j)
            else:
                missing += 1
                r.update({'has_reid': 0, 'cos_end_start': 0.0, 'cos_end_global': 0.0, 'cos_global_start': 0.0, 'cos_global_global': 0.0, 'cos_high_high': 0.0, 'cos_start_start': 0.0, 'cos_end_end': 0.0, 'appearance_mean': 0.0, 'appearance_max': 0.0, 'appearance_min': 0.0, 'appearance_std': 0.0, 'appearance_gap_consistency': 0.0})
        if not valid:
            continue
        ia = np.asarray(ia, dtype=np.int64)
        ib = np.asarray(ib, dtype=np.int64)
        es = np.sum(reid['end'][ia] * reid['start'][ib], axis=1)
        eg = np.sum(reid['end'][ia] * reid['global_mean'][ib], axis=1)
        gs = np.sum(reid['global_mean'][ia] * reid['start'][ib], axis=1)
        gg = np.sum(reid['global_mean'][ia] * reid['global_mean'][ib], axis=1)
        hh = np.sum(reid['high_score'][ia] * reid['high_score'][ib], axis=1)
        ss = np.sum(reid['start'][ia] * reid['start'][ib], axis=1)
        ee = np.sum(reid['end'][ia] * reid['end'][ib], axis=1)
        feats = np.stack([es, eg, gs, gg, hh, ss, ee], axis=1)
        means = feats.mean(axis=1)
        maxs = feats.max(axis=1)
        mins = feats.min(axis=1)
        stds = feats.std(axis=1)
        consistency = maxs - mins
        for n, j in enumerate(valid):
            r = chunk[j]
            r.update({
                'has_reid': 1,
                'cos_end_start': float(es[n]),
                'cos_end_global': float(eg[n]),
                'cos_global_start': float(gs[n]),
                'cos_global_global': float(gg[n]),
                'cos_high_high': float(hh[n]),
                'cos_start_start': float(ss[n]),
                'cos_end_end': float(ee[n]),
                'appearance_mean': float(means[n]),
                'appearance_max': float(maxs[n]),
                'appearance_min': float(mins[n]),
                'appearance_std': float(stds[n]),
                'appearance_gap_consistency': float(consistency[n]),
            })


def add_rank_margin(rows: List[dict], score_col: str) -> None:
    # rank successors per source and predecessors per target.
    for prefix, key_col in [('out', 'track_a'), ('in', 'track_b')]:
        groups = defaultdict(list)
        for i, r in enumerate(rows):
            groups[(r['seq'], r[key_col])].append((i, af(r.get(score_col))))
        for _key, items in groups.items():
            items = sorted(items, key=lambda x: x[1], reverse=True)
            best = items[0][1] if items else 0.0
            second = items[1][1] if len(items) > 1 else 0.0
            for rank, (idx_i, score) in enumerate(items, start=1):
                rows[idx_i][f'{prefix}_rank_by_{score_col}'] = rank
                rows[idx_i][f'{prefix}_best_{score_col}'] = best
                rows[idx_i][f'{prefix}_second_{score_col}'] = second
                rows[idx_i][f'{prefix}_margin_to_second_{score_col}'] = score - (second if rank == 1 else best)
                rows[idx_i][f'{prefix}_group_size'] = len(items)


def add_score(rows: List[dict]) -> None:
    for r in rows:
        app = af(r.get('appearance_max'))
        mean = af(r.get('appearance_mean'))
        gg = af(r.get('cos_global_global'))
        es = af(r.get('cos_end_start'))
        gap = ai(r.get('gap'))
        bucket_penalty = 0.0 if gap <= 60 else (0.04 if gap <= 150 else 0.08)
        geom_penalty = 0.04 * ai(r.get('geometry_risk')) + 0.04 * ai(r.get('motion_risk'))
        zone_bonus = 0.03 * ai(r.get('zone_near_match')) + 0.02 * ai(r.get('zone_exact_match'))
        debt_bonus = min(0.08, 0.02 * ai(r.get('edge_debt_score')))
        r['a42_appearance_score'] = 0.45 * app + 0.25 * mean + 0.20 * gg + 0.10 * es
        r['a42_rule_score_v1'] = af(r.get('a42_appearance_score')) + zone_bonus + debt_bonus - geom_penalty - bucket_penalty


def summarize(rows: List[dict], split: str) -> dict:
    by_bucket = defaultdict(list)
    for r in rows:
        by_bucket[r['gap_bucket']].append(r)
    out = {'split': split, 'rows': len(rows), 'by_bucket': {} }
    for b, rs in sorted(by_bucket.items()):
        d = {
            'rows': len(rs),
            'has_reid': sum(ai(r.get('has_reid')) for r in rs),
            'appearance_max_mean': sum(af(r.get('appearance_max')) for r in rs) / len(rs) if rs else 0.0,
            'rule_score_mean': sum(af(r.get('a42_rule_score_v1')) for r in rs) / len(rs) if rs else 0.0,
        }
        if split == 'train':
            d['same_gt'] = sum(ai(r.get('same_gt')) for r in rs)
            d['same_gt_rate'] = d['same_gt'] / len(rs) if rs else 0.0
            for thr in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
                sel = [r for r in rs if af(r.get('appearance_max')) >= thr]
                tp = sum(ai(r.get('same_gt')) for r in sel)
                d[f'app_ge_{thr:.2f}_selected'] = len(sel)
                d[f'app_ge_{thr:.2f}_precision'] = tp / len(sel) if sel else 0.0
                d[f'app_ge_{thr:.2f}_tp'] = tp
        out['by_bucket'][b] = d
    return out


def build_manifest(split: str, tracklet_rows: Path, reid_npz: Path, out_csv: Path, max_gap: int) -> dict:
    tracklets = read_csv(tracklet_rows)
    rows = build_base_candidates(tracklets, max_gap=max_gap, split=split)
    reid = load_reid(reid_npz)
    add_reid_features(rows, reid)
    add_score(rows)
    for col in ['appearance_max', 'a42_rule_score_v1', 'cos_global_global', 'cos_high_high']:
        add_rank_margin(rows, col)
    # Useful combined rank/margin fields for scorers.
    for r in rows:
        r['max_rank_by_a42_rule_score_v1'] = max(ai(r.get('out_rank_by_a42_rule_score_v1'), 999), ai(r.get('in_rank_by_a42_rule_score_v1'), 999))
        r['min_margin_by_a42_rule_score_v1'] = min(af(r.get('out_margin_to_second_a42_rule_score_v1')), af(r.get('in_margin_to_second_a42_rule_score_v1')))
        r['max_rank_by_appearance_max'] = max(ai(r.get('out_rank_by_appearance_max'), 999), ai(r.get('in_rank_by_appearance_max'), 999))
        r['min_margin_by_appearance_max'] = min(af(r.get('out_margin_to_second_appearance_max')), af(r.get('in_margin_to_second_appearance_max')))
    write_csv(out_csv, rows)
    return summarize(rows, split)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-tracklets', required=True)
    ap.add_argument('--train-reid-npz', required=True)
    ap.add_argument('--test-tracklets', required=True)
    ap.add_argument('--test-reid-npz', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--max-gap', type=int, default=300)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_summary = build_manifest('train', Path(args.train_tracklets), Path(args.train_reid_npz), out / 'a42_long_gap_candidates_train.csv', args.max_gap)
    test_summary = build_manifest('test', Path(args.test_tracklets), Path(args.test_reid_npz), out / 'a42_long_gap_candidates_test.csv', args.max_gap)
    summary = {
        'train': train_summary,
        'test': test_summary,
        'decision': 'A42_01_LONG_GAP_MANIFEST_BUILT_READY_FOR_SCORER',
        'leakage_note': 'same_gt/dominant_gt are present only in train manifest for diagnostics; all deployable features are no-GT fields.',
    }
    (out / 'a42_01_manifest_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    md = ['# A42_01 Long-Gap Candidate Manifest', '', '```json', json.dumps(summary, indent=2, sort_keys=True)[:12000], '```', '']
    (out / 'decision.md').write_text('\n'.join(md), encoding='utf-8')
    print(json.dumps({'out_dir': str(out), 'train_rows': train_summary['rows'], 'test_rows': test_summary['rows'], 'decision': summary['decision']}, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
