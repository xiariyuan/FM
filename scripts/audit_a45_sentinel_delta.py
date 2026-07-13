#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


def fnum(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def inum(x, default=0):
    try:
        return int(float(x))
    except Exception:
        return default


def read_mot(path: Path, is_gt=False) -> Dict[int, List[dict]]:
    by = defaultdict(list)
    with path.open('r', encoding='utf-8', errors='ignore') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            p = line.split(',')
            if len(p) < 6:
                continue
            fr = inum(p[0]); tid = inum(p[1])
            x, y, w, h = map(fnum, p[2:6])
            if w <= 0 or h <= 0:
                continue
            score = fnum(p[6], 1.0) if len(p) > 6 else 1.0
            vis = 1.0
            if is_gt:
                mark = inum(p[6], 1) if len(p) > 6 else 1
                cls = inum(p[7], 1) if len(p) > 7 else 1
                if mark == 0 or cls != 1:
                    continue
                vis = fnum(p[8], 1.0) if len(p) > 8 else 1.0
                score = 1.0
            by[fr].append({'frame': fr, 'id': tid, 'x': x, 'y': y, 'w': w, 'h': h, 'score': score, 'vis': vis})
    return by


def boxes_array(rows: List[dict]) -> np.ndarray:
    arr = np.zeros((len(rows), 4), dtype=np.float32)
    for i, r in enumerate(rows):
        arr[i, 0] = r['x']; arr[i, 1] = r['y']; arr[i, 2] = r['x'] + r['w']; arr[i, 3] = r['y'] + r['h']
    return arr


def iou_matrix(gt: List[dict], det: List[dict]) -> np.ndarray:
    g = boxes_array(gt); d = boxes_array(det)
    gx1, gy1, gx2, gy2 = g[:, 0:1], g[:, 1:2], g[:, 2:3], g[:, 3:4]
    dx1, dy1, dx2, dy2 = d[:, 0][None, :], d[:, 1][None, :], d[:, 2][None, :], d[:, 3][None, :]
    ix1 = np.maximum(gx1, dx1); iy1 = np.maximum(gy1, dy1)
    ix2 = np.minimum(gx2, dx2); iy2 = np.minimum(gy2, dy2)
    inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    ga = np.maximum(0, gx2 - gx1) * np.maximum(0, gy2 - gy1)
    da = np.maximum(0, dx2 - dx1) * np.maximum(0, dy2 - dy1)
    union = ga + da - inter
    return np.where(union > 0, inter / union, 0.0)


def match_frame(gt: List[dict], det: List[dict], thr=0.5):
    if not gt:
        return [], [], list(range(len(det)))
    if not det:
        return [], list(range(len(gt))), []
    mat = iou_matrix(gt, det)
    gi, dj = linear_sum_assignment(-mat)
    matches = []
    mg, md = set(), set()
    for gidx, didx in zip(gi, dj):
        val = float(mat[gidx, didx])
        if val >= thr:
            matches.append((gidx, didx, val)); mg.add(gidx); md.add(didx)
    return matches, [i for i in range(len(gt)) if i not in mg], [j for j in range(len(det)) if j not in md]


def score_bucket(s):
    if s < 0.03: return '<0.03'
    if s < 0.05: return '0.03-0.05'
    if s < 0.10: return '0.05-0.10'
    if s < 0.20: return '0.10-0.20'
    if s < 0.50: return '0.20-0.50'
    return '>=0.50'


def area_bucket(r):
    a = r['w'] * r['h']
    if a < 32*32: return 'small'
    if a < 96*96: return 'medium'
    return 'large'


def track_stats(det_by: Dict[int, List[dict]]) -> List[dict]:
    by = defaultdict(list)
    for fr, rows in det_by.items():
        for r in rows:
            by[r['id']].append(r)
    out = []
    for tid, rows in by.items():
        frames = [r['frame'] for r in rows]
        scores = [r.get('score', 1.0) for r in rows]
        out.append({'id': tid, 'len': len(rows), 'start': min(frames), 'end': max(frames), 'span': max(frames)-min(frames)+1, 'mean_score': sum(scores)/len(scores), 'min_score': min(scores), 'max_score': max(scores)})
    return out


def summarize(gt_by, det_by, name):
    frames = sorted(set(gt_by) | set(det_by))
    out = {'name': name, 'gt': 0, 'det': 0, 'tp': 0, 'fn': 0, 'fp': 0,
           'matched_gt': set(), 'fp_rows': [], 'tp_rows': [], 'fn_rows': [],
           'tp_score': Counter(), 'fp_score': Counter(), 'tp_area': Counter(), 'fp_area': Counter(), 'recovered_area': Counter(), 'lost_area': Counter()}
    for fr in frames:
        gt = gt_by.get(fr, []); det = det_by.get(fr, [])
        out['gt'] += len(gt); out['det'] += len(det)
        matches, fn_idx, fp_idx = match_frame(gt, det, 0.5)
        out['tp'] += len(matches); out['fn'] += len(fn_idx); out['fp'] += len(fp_idx)
        for gi, di, val in matches:
            g, d = gt[gi], det[di]
            out['matched_gt'].add((fr, g['id']))
            out['tp_rows'].append({'frame': fr, 'det_id': d['id'], 'gt_id': g['id'], 'iou': val, 'score': d['score'], 'area_bucket': area_bucket(g), 'vis': g.get('vis', 1.0)})
            out['tp_score'][score_bucket(d['score'])] += 1
            out['tp_area'][area_bucket(g)] += 1
        for gi in fn_idx:
            g = gt[gi]
            out['fn_rows'].append({'frame': fr, 'gt_id': g['id'], 'area_bucket': area_bucket(g), 'vis': g.get('vis', 1.0), 'w': g['w'], 'h': g['h']})
        for di in fp_idx:
            d = det[di]
            out['fp_rows'].append({'frame': fr, 'det_id': d['id'], 'score': d['score'], 'area_bucket': area_bucket(d), 'w': d['w'], 'h': d['h']})
            out['fp_score'][score_bucket(d['score'])] += 1
            out['fp_area'][area_bucket(d)] += 1
    return out


def write_csv(path: Path, rows: List[dict], max_rows=None):
    rows = rows if max_rows is None else rows[:max_rows]
    fields = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); fields.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=fields or ['x'], extrasaction='ignore')
        w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt', required=True)
    ap.add_argument('--baseline', required=True)
    ap.add_argument('--sentinel', required=True)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    gt = read_mot(Path(args.gt), is_gt=True)
    base = read_mot(Path(args.baseline))
    sent = read_mot(Path(args.sentinel))
    B = summarize(gt, base, 'baseline')
    S = summarize(gt, sent, 'sentinel')
    base_tracks = track_stats(base); sent_tracks = track_stats(sent)
    recovered = S['matched_gt'] - B['matched_gt']
    lost = B['matched_gt'] - S['matched_gt']
    for fr, rows in gt.items():
        for g in rows:
            key = (fr, g['id'])
            if key in recovered: S['recovered_area'][area_bucket(g)] += 1
            if key in lost: S['lost_area'][area_bucket(g)] += 1
    summary = {
        'baseline': {'gt': B['gt'], 'det': B['det'], 'tp': B['tp'], 'fn': B['fn'], 'fp': B['fp'], 'precision': B['tp']/B['det'], 'recall': B['tp']/B['gt']},
        'sentinel': {'gt': S['gt'], 'det': S['det'], 'tp': S['tp'], 'fn': S['fn'], 'fp': S['fp'], 'precision': S['tp']/S['det'], 'recall': S['tp']/S['gt']},
        'delta': {'det': S['det']-B['det'], 'tp': S['tp']-B['tp'], 'fn': S['fn']-B['fn'], 'fp': S['fp']-B['fp'], 'recovered_gt_frame_ids': len(recovered), 'lost_gt_frame_ids': len(lost)},
        'baseline_tracks': {'n_tracks': len(base_tracks), 'len_mean': sum(x['len'] for x in base_tracks)/len(base_tracks), 'short_le5': sum(x['len']<=5 for x in base_tracks), 'short_le10': sum(x['len']<=10 for x in base_tracks)},
        'sentinel_tracks': {'n_tracks': len(sent_tracks), 'len_mean': sum(x['len'] for x in sent_tracks)/len(sent_tracks), 'short_le5': sum(x['len']<=5 for x in sent_tracks), 'short_le10': sum(x['len']<=10 for x in sent_tracks)},
        'baseline_fp_score': dict(B['fp_score']), 'sentinel_fp_score': dict(S['fp_score']),
        'baseline_tp_score': dict(B['tp_score']), 'sentinel_tp_score': dict(S['tp_score']),
        'baseline_fp_area': dict(B['fp_area']), 'sentinel_fp_area': dict(S['fp_area']),
        'baseline_tp_area': dict(B['tp_area']), 'sentinel_tp_area': dict(S['tp_area']),
        'recovered_area': dict(S['recovered_area']), 'lost_area': dict(S['lost_area']),
    }
    # Decision based on signal/noise.
    rec, lostn, fpd = len(recovered), len(lost), S['fp']-B['fp']
    if rec > lostn and fpd < rec * 1.5:
        decision = 'A45_02_HAS_FILTERABLE_SIGNAL__MINE_PRECISION_GATES'
    else:
        decision = 'A45_02_LOW_SCORE_SENTINEL_TOO_NOISY__STOP_LOW_SCORE_TRACKER_PATH'
    summary['decision'] = decision
    (out/'a45_02_match_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n')
    write_csv(out/'baseline_track_stats.csv', sorted(base_tracks, key=lambda x: x['len']))
    write_csv(out/'sentinel_track_stats.csv', sorted(sent_tracks, key=lambda x: x['len']))
    write_csv(out/'sentinel_fp_sample.csv', S['fp_rows'], max_rows=20000)
    write_csv(out/'sentinel_tp_sample.csv', S['tp_rows'], max_rows=20000)
    md = ['# A45_02 Sentinel Delta Audit', '', '## Decision', '', '```text', decision, '```', '', '## Key numbers',
          f"- baseline TP/FN/FP: {B['tp']} / {B['fn']} / {B['fp']}",
          f"- sentinel TP/FN/FP: {S['tp']} / {S['fn']} / {S['fp']}",
          f"- delta TP/FN/FP: {S['tp']-B['tp']} / {S['fn']-B['fn']} / {S['fp']-B['fp']}",
          f"- recovered GT frame-ids: {rec}", f"- lost GT frame-ids: {lostn}",
          f"- baseline tracks / short<=10: {len(base_tracks)} / {summary['baseline_tracks']['short_le10']}",
          f"- sentinel tracks / short<=10: {len(sent_tracks)} / {summary['sentinel_tracks']['short_le10']}", '', '## Interpretation']
    if decision.endswith('STOP_LOW_SCORE_TRACKER_PATH'):
        md.append('The low-score sentinel added recall but the FP/track-fragmentation cost is too high. Do not expand this config or run a larger low-score threshold matrix.')
        md.append('Next should move to detector retraining / better detector source, or use the strong historical graphassoc halfval baseline as a sequence-specific reference rather than this sentinel.')
    else:
        md.append('There may be enough signal to mine precision gates before any merge.')
    (out/'decision.md').write_text('\n'.join(md)+'\n')
    print(json.dumps({'decision': decision, **summary['delta'], 'baseline_tracks': len(base_tracks), 'sentinel_tracks': len(sent_tracks)}, indent=2, sort_keys=True))

if __name__ == '__main__':
    main()
