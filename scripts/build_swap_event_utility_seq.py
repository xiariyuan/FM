from __future__ import annotations

import argparse
import gc
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def read_filtered_csv(path: str, seq: str, usecols: list[str], chunksize: int = 200000) -> pd.DataFrame:
    parts = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        x = chunk[chunk['seq'] == seq]
        if len(x):
            parts.append(x.copy())
    if not parts:
        return pd.DataFrame(columns=usecols)
    out = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()
    return out


def load_tracks(path: Path):
    by = defaultdict(dict)
    with path.open() as f:
        for line in f:
            p = line.strip().split(',')
            if len(p) < 7:
                continue
            fr = int(float(p[0])); tid = int(float(p[1]))
            by[tid][fr] = tuple(map(float, p[2:7]))
    return dict(by)


def load_reid(root: Path, seq: str):
    p = root / seq / 'tracklet_reid_features.npz'
    if not p.exists():
        return {}
    with np.load(p) as z:
        tids = z['track_id'].astype(int)
        start = z['start'].astype(np.float32)
        end = z['end'].astype(np.float32)
        glob = z['global_mean'].astype(np.float32)
        high = z['high_score'].astype(np.float32)
    return {int(t): {'start': start[i], 'end': end[i], 'global': glob[i], 'high': high[i]} for i, t in enumerate(tids)}


def cosine(a, b):
    if a is None or b is None:
        return 0.0
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def centers(row):
    x, y, w, h, _ = row
    return np.array([x + .5 * w, y + .5 * h], dtype=np.float64)


def nearest_before(track, frame, window=5):
    for f in range(frame - 1, frame - window - 1, -1):
        if f in track:
            return f, track[f]
    return None


def nearest_after(track, frame, window=5):
    for f in range(frame, frame + window + 1):
        if f in track:
            return f, track[f]
    return None


def previous_match(match_map, tid, frame, lookback):
    m = match_map.get(tid, {})
    for f in range(frame - 1, max(0, frame - lookback) - 1, -1):
        if f in m:
            return m[f][0], f, m[f][1]
    return None


def swap_utility(match_map, ta, tb, frame, horizon, lookback):
    pa = previous_match(match_map, ta, frame, lookback)
    pb = previous_match(match_map, tb, frame, lookback)
    if pa is None or pb is None or pa[0] == pb[0]:
        return None
    ga, gb = pa[0], pb[0]
    keep = 0; swap = 0; observed = 0
    ma = match_map.get(ta, {}); mb = match_map.get(tb, {})
    for f in range(frame, frame + horizon + 1):
        if f in ma:
            g = ma[f][0]; observed += 1
            keep += int(g == ga); swap += int(g == gb)
        if f in mb:
            g = mb[f][0]; observed += 1
            keep += int(g == gb); swap += int(g == ga)
    return {
        f'keep_support_{horizon}': keep,
        f'swap_support_{horizon}': swap,
        f'observed_support_{horizon}': observed,
        f'swap_utility_{horizon}': swap - keep,
        'prev_gt_a': ga,
        'prev_gt_b': gb,
        'prev_match_frame_a': pa[1],
        'prev_match_frame_b': pb[1],
        'prev_match_iou_a': pa[2],
        'prev_match_iou_b': pb[2],
    }


def episode_candidates(overlap: pd.DataFrame, min_peak: float):
    rows = []
    overlap = overlap.sort_values(['track_i', 'track_j', 'frame'])
    for (a, b), g in overlap.groupby(['track_i', 'track_j'], sort=False):
        frames = g['frame'].to_numpy(np.int32)
        ioas = g['ioa_min_area'].to_numpy(np.float32)
        start = 0
        for i in range(1, len(g) + 1):
            if i == len(g) or frames[i] - frames[i - 1] > 1:
                rf = frames[start:i]; ri = ioas[start:i]
                if len(rf) and float(ri.max()) >= min_peak:
                    j = int(np.argmax(ri))
                    rows.append({
                        'track_a': int(a), 'track_b': int(b), 'frame': int(rf[j]),
                        'candidate_ioa': float(ri[j]), 'episode_start': int(rf[0]),
                        'episode_end': int(rf[-1]), 'episode_duration': int(rf[-1] - rf[0] + 1),
                        'episode_observed_frames': int(len(rf)), 'episode_ioa_mean': float(ri.mean()),
                        'episode_ioa_std': float(ri.std()), 'episode_ioa_max': float(ri.max()),
                        'episode_peak_frac': float((rf[j] - rf[0]) / max(1, rf[-1] - rf[0])),
                    })
                start = i
    return rows


def add_pair_features(base, tracks, reid, obs, debt):
    a = base['track_a']; b = base['track_b']; fr = base['frame']
    ta = tracks.get(a, {}); tb = tracks.get(b, {})
    ap = nearest_before(ta, fr, 6); bp = nearest_before(tb, fr, 6)
    an = nearest_after(ta, fr, 6); bn = nearest_after(tb, fr, 6)
    if ap is None or bp is None or an is None or bn is None:
        return None
    af0, ar0 = ap; bf0, br0 = bp; af1, ar1 = an; bf1, br1 = bn
    ca0, cb0, ca1, cb1 = centers(ar0), centers(br0), centers(ar1), centers(br1)
    hscale = max(1.0, .25 * (ar0[3] + br0[3] + ar1[3] + br1[3]))
    va = (ca1 - ca0) / max(1, af1 - af0)
    vb = (cb1 - cb0) / max(1, bf1 - bf0)
    keep_cost = (np.linalg.norm(ca1 - (ca0 + va * max(1, af1 - af0))) + np.linalg.norm(cb1 - (cb0 + vb * max(1, bf1 - bf0)))) / hscale
    swap_cost = (np.linalg.norm(cb1 - (ca0 + va * max(1, bf1 - af0))) + np.linalg.norm(ca1 - (cb0 + vb * max(1, af1 - bf0)))) / hscale
    pre = ca0 - cb0; post = ca1 - cb1
    base.update({
        'pre_frame_a': af0, 'pre_frame_b': bf0, 'post_frame_a': af1, 'post_frame_b': bf1,
        'pre_center_distance': float(np.linalg.norm(pre) / hscale),
        'post_center_distance': float(np.linalg.norm(post) / hscale),
        'pre_dx': float(pre[0] / hscale), 'pre_dy': float(pre[1] / hscale),
        'post_dx': float(post[0] / hscale), 'post_dy': float(post[1] / hscale),
        'x_order_flip': int(pre[0] * post[0] < 0), 'y_order_flip': int(pre[1] * post[1] < 0),
        'velocity_cos': float(np.dot(va, vb) / max(1e-8, np.linalg.norm(va) * np.linalg.norm(vb))),
        'velocity_a': float(np.linalg.norm(va) / hscale), 'velocity_b': float(np.linalg.norm(vb) / hscale),
        'keep_motion_cost': float(keep_cost), 'swap_motion_cost': float(swap_cost),
        'swap_motion_advantage': float(keep_cost - swap_cost),
        'pre_height_ratio': float(max(ar0[3], br0[3]) / max(1.0, min(ar0[3], br0[3]))),
        'post_height_ratio': float(max(ar1[3], br1[3]) / max(1.0, min(ar1[3], br1[3]))),
        'score_a_pre': float(ar0[4]), 'score_b_pre': float(br0[4]),
        'score_a_post': float(ar1[4]), 'score_b_post': float(br1[4]),
        'score_pair_min': float(min(ar0[4], br0[4], ar1[4], br1[4])),
    })
    ra = reid.get(a); rb = reid.get(b)
    base.update({
        'reid_global_cos': cosine(ra['global'] if ra else None, rb['global'] if rb else None),
        'reid_start_cos': cosine(ra['start'] if ra else None, rb['start'] if rb else None),
        'reid_end_cos': cosine(ra['end'] if ra else None, rb['end'] if rb else None),
        'reid_high_cos': cosine(ra['high'] if ra else None, rb['high'] if rb else None),
        'reid_a_end_b_start': cosine(ra['end'] if ra else None, rb['start'] if rb else None),
        'reid_b_end_a_start': cosine(rb['end'] if rb else None, ra['start'] if ra else None),
    })
    observable_cols = [
        'track_len','track_span','track_density','score_mean','score_std','score_min','score_p10',
        'log_area_mean','log_area_std','aspect_mean','aspect_std','height_mean','height_cv',
        'speed_mean','speed_std','speed_max','accel_mean','accel_max','appearance_drift',
        'overlap_event_count','overlap_frame_count','overlap_partner_count','overlap_ioa_mean',
        'overlap_ioa_max','overlap_frame_fraction','overlap_events_per_frame','overlap_partners_per_100f'
    ]
    for prefix, tid in [('a', a), ('b', b)]:
        r = obs.get(tid)
        for c in observable_cols:
            v = getattr(r, c, 0.0) if r is not None else 0.0
            base[f'{prefix}_{c}'] = float(v) if pd.notna(v) else 0.0
        dr = debt.get(tid)
        base[f'{prefix}_debt_score'] = float(getattr(dr, 'score', 0.0)) if dr is not None else 0.0
        base[f'{prefix}_debt_rank'] = float(getattr(dr, 'rank', 0.0)) if dr is not None else 0.0
    for c in ['debt_score','appearance_drift','overlap_event_count','overlap_partner_count','overlap_frame_fraction','speed_max','score_std','track_len']:
        av = base[f'a_{c}']; bv = base[f'b_{c}']
        base[f'pair_max_{c}'] = max(av, bv)
        base[f'pair_min_{c}'] = min(av, bv)
        base[f'pair_absdiff_{c}'] = abs(av - bv)
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq', required=True)
    ap.add_argument('--overlap-csv', required=True)
    ap.add_argument('--matches-csv', required=True)
    ap.add_argument('--track-file', required=True)
    ap.add_argument('--reid-root', required=True)
    ap.add_argument('--obs-csv', required=True)
    ap.add_argument('--debt-csv', required=True)
    ap.add_argument('--out-file', required=True)
    ap.add_argument('--min-peak-ioa', type=float, default=.65)
    ap.add_argument('--lookback', type=int, default=6)
    args = ap.parse_args()
    seq = args.seq
    print('[swap-seq] loading overlap', seq, flush=True)
    overlap = read_filtered_csv(args.overlap_csv, seq, ['seq','frame','track_i','track_j','ioa_min_area'])
    candidates = episode_candidates(overlap, args.min_peak_ioa)
    del overlap; gc.collect()
    print('[swap-seq] candidates', len(candidates), flush=True)
    matches_df = read_filtered_csv(args.matches_csv, seq, ['seq','frame','track_id','gt_id','iou'])
    matches = defaultdict(dict)
    for r in matches_df.itertuples(index=False):
        matches[int(r.track_id)][int(r.frame)] = (int(r.gt_id), float(r.iou))
    del matches_df; gc.collect()
    tracks = load_tracks(Path(args.track_file))
    obs_df = read_filtered_csv(args.obs_csv, seq, list(pd.read_csv(args.obs_csv, nrows=0).columns))
    obs = {int(r.track_id): r for r in obs_df.itertuples(index=False)}
    debt_df = read_filtered_csv(args.debt_csv, seq, list(pd.read_csv(args.debt_csv, nrows=0).columns))
    debt = {int(r.track_id): r for r in debt_df.itertuples(index=False)}
    del obs_df, debt_df; gc.collect()
    reid = load_reid(Path(args.reid_root), seq)
    rows = []
    for i, c in enumerate(candidates, 1):
        feat = add_pair_features(dict(c), tracks, reid, obs, debt)
        if feat is None:
            continue
        labels = {}
        ok = True
        for h in [10,30,60]:
            u = swap_utility(matches, c['track_a'], c['track_b'], c['frame'], h, args.lookback)
            if u is None:
                ok = False; break
            labels.update(u)
        if not ok:
            continue
        feat.update(labels)
        for h in [10,30,60]:
            feat[f'swap_positive_{h}'] = int(feat[f'swap_utility_{h}'] > 0)
        feat['swap_strong_30'] = int(feat['swap_utility_30'] >= 4)
        feat['swap_strong_60'] = int(feat['swap_utility_60'] >= 6)
        feat['seq'] = seq
        rows.append(feat)
        if i % 1000 == 0:
            print('[swap-seq]', seq, i, '/', len(candidates), 'labeled', len(rows), flush=True)
    df = pd.DataFrame(rows)
    label_prefixes = ('prev_gt_','prev_match_','keep_support_','swap_support_','observed_support_','swap_utility_','swap_positive_','swap_strong_')
    pct = {}
    for c in df.select_dtypes(include='number').columns:
        if c in {'track_a','track_b','frame'} or c.startswith(label_prefixes):
            continue
        pct[c+'_seqpct'] = df[c].rank(pct=True, method='average')
    if pct:
        df = pd.concat([df, pd.DataFrame(pct)], axis=1)
    out = Path(args.out_file); out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    summary = {
        'seq': seq, 'candidate_episodes': len(candidates), 'labeled_rows': len(df),
        'positive10': int((df.swap_utility_10 > 0).sum()) if len(df) else 0,
        'positive30': int((df.swap_utility_30 > 0).sum()) if len(df) else 0,
        'positive60': int((df.swap_utility_60 > 0).sum()) if len(df) else 0,
        'strong30': int((df.swap_utility_30 >= 4).sum()) if len(df) else 0,
        'strong60': int((df.swap_utility_60 >= 6).sum()) if len(df) else 0,
        'columns': len(df.columns),
    }
    out.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)

if __name__ == '__main__':
    main()
