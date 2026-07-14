from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEQS = ["MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05"]


def cosine(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return float("nan")
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def load_track_rows(path: Path) -> Dict[int, List[dict]]:
    tracks: Dict[int, List[dict]] = defaultdict(list)
    with path.open() as f:
        for line in f:
            p = line.strip().split(',')
            if len(p) < 6:
                continue
            fr = int(float(p[0])); tid = int(float(p[1]))
            x, y, w, h = map(float, p[2:6])
            score = float(p[6]) if len(p) > 6 else 1.0
            tracks[tid].append({
                'frame': fr, 'x': x, 'y': y, 'w': w, 'h': h, 'score': score,
                'cx': x + 0.5 * w, 'cy': y + 0.5 * h,
            })
    for tid in tracks:
        tracks[tid].sort(key=lambda r: r['frame'])
    return tracks


def track_geometry_features(rows: List[dict]) -> dict:
    frames = np.asarray([r['frame'] for r in rows], dtype=np.int64)
    scores = np.asarray([r['score'] for r in rows], dtype=np.float64)
    widths = np.asarray([r['w'] for r in rows], dtype=np.float64)
    heights = np.asarray([r['h'] for r in rows], dtype=np.float64)
    areas = np.maximum(widths * heights, 1e-6)
    aspects = widths / np.maximum(heights, 1e-6)
    diffs = np.diff(frames)
    span = int(frames[-1] - frames[0] + 1)
    gap_vals = diffs[diffs > 1] - 1
    speeds = []
    accel = []
    prev_v = None
    for a, b in zip(rows[:-1], rows[1:]):
        dt = max(1, b['frame'] - a['frame'])
        norm = max(1.0, 0.5 * (a['h'] + b['h']))
        vx = (b['cx'] - a['cx']) / dt / norm
        vy = (b['cy'] - a['cy']) / dt / norm
        v = math.hypot(vx, vy)
        speeds.append(v)
        if prev_v is not None:
            accel.append(abs(v - prev_v))
        prev_v = v
    return {
        'track_len': len(rows),
        'start_frame': int(frames[0]),
        'end_frame': int(frames[-1]),
        'track_span': span,
        'track_density': len(rows) / max(1, span),
        'internal_gap_count': int(len(gap_vals)),
        'internal_gap_frames': int(gap_vals.sum()) if len(gap_vals) else 0,
        'max_internal_gap': int(gap_vals.max()) if len(gap_vals) else 0,
        'score_mean': float(scores.mean()),
        'score_std': float(scores.std()),
        'score_min': float(scores.min()),
        'score_p10': float(np.quantile(scores, 0.1)),
        'log_area_mean': float(np.log(areas).mean()),
        'log_area_std': float(np.log(areas).std()),
        'aspect_mean': float(aspects.mean()),
        'aspect_std': float(aspects.std()),
        'height_mean': float(heights.mean()),
        'height_cv': float(heights.std() / max(1e-6, heights.mean())),
        'speed_mean': float(np.mean(speeds)) if speeds else 0.0,
        'speed_std': float(np.std(speeds)) if speeds else 0.0,
        'speed_max': float(np.max(speeds)) if speeds else 0.0,
        'accel_mean': float(np.mean(accel)) if accel else 0.0,
        'accel_max': float(np.max(accel)) if accel else 0.0,
        'first_cx': float(rows[0]['cx']), 'first_cy': float(rows[0]['cy']),
        'last_cx': float(rows[-1]['cx']), 'last_cy': float(rows[-1]['cy']),
        'first_h': float(rows[0]['h']), 'last_h': float(rows[-1]['h']),
    }


def load_feature_maps(feature_root: Path, seq: str) -> tuple[Dict[int, dict], dict]:
    d = feature_root / seq
    z = np.load(d / 'tracklet_reid_features.npz')
    idx_rows = list(csv.DictReader((d / 'tracklet_reid_index.csv').open()))
    ids = [int(x) for x in z['track_id']]
    if len(ids) != len(idx_rows):
        raise RuntimeError(f'feature/index size mismatch for {seq}')
    out = {}
    for i, tid in enumerate(ids):
        meta = idx_rows[i]
        out[tid] = {
            'start_feat': z['start'][i].astype(np.float32),
            'end_feat': z['end'][i].astype(np.float32),
            'global_feat': z['global_mean'][i].astype(np.float32),
            'high_feat': z['high_score'][i].astype(np.float32),
            'index_track_len': int(meta['track_len']),
            'index_start_frame': int(meta['start_frame']),
            'index_end_frame': int(meta['end_frame']),
            'index_avg_score': float(meta['avg_score']),
            'feature_samples': int(meta['total_available_features']),
        }
    return out, {'tracks': len(ids)}


def build_similarity_cache(feats: Dict[int, dict], key_a: str, key_b: str) -> tuple[list[int], dict[int, int], np.ndarray]:
    tids = sorted(feats)
    index = {tid: i for i, tid in enumerate(tids)}
    if not tids:
        return tids, index, np.zeros((0, 0), dtype=np.float32)
    a = np.stack([feats[tid][key_a] for tid in tids], axis=0).astype(np.float32)
    b = np.stack([feats[tid][key_b] for tid in tids], axis=0).astype(np.float32)
    an = np.linalg.norm(a, axis=1, keepdims=True)
    bn = np.linalg.norm(b, axis=1, keepdims=True)
    a = a / np.maximum(an, 1e-12)
    b = b / np.maximum(bn, 1e-12)
    return tids, index, a @ b.T


def add_temporal_ambiguity(rows: Dict[int, dict], feats: Dict[int, dict], max_gap: int = 300) -> None:
    tids = sorted(rows)
    feat_tids, feat_index, end_start_sim = build_similarity_cache(feats, 'end_feat', 'start_feat')
    start_frames = np.asarray([rows[tid]['start_frame'] for tid in tids], dtype=np.int64)
    end_frames = np.asarray([rows[tid]['end_frame'] for tid in tids], dtype=np.int64)
    first_cx = np.asarray([rows[tid]['first_cx'] for tid in tids], dtype=np.float64)
    first_cy = np.asarray([rows[tid]['first_cy'] for tid in tids], dtype=np.float64)
    first_h = np.asarray([rows[tid]['first_h'] for tid in tids], dtype=np.float64)
    last_cx = np.asarray([rows[tid]['last_cx'] for tid in tids], dtype=np.float64)
    last_cy = np.asarray([rows[tid]['last_cy'] for tid in tids], dtype=np.float64)
    last_h = np.asarray([rows[tid]['last_h'] for tid in tids], dtype=np.float64)
    tid_to_local = {tid: i for i, tid in enumerate(tids)}

    for tid in tids:
        i = tid_to_local[tid]
        fi = feat_index.get(tid)
        for prefix in ['pred', 'succ']:
            if prefix == 'pred':
                gaps = start_frames[i] - end_frames
                mask = (gaps >= 1) & (gaps <= max_gap)
                cand_local = np.flatnonzero(mask)
                sims = np.asarray([
                    end_start_sim[feat_index[tids[j]], fi]
                    if fi is not None and tids[j] in feat_index else np.nan
                    for j in cand_local
                ], dtype=np.float64)
                dx = first_cx[i] - last_cx[cand_local]
                dy = first_cy[i] - last_cy[cand_local]
                norm = np.maximum(1.0, 0.5 * (first_h[i] + last_h[cand_local]))
                motions = np.hypot(dx, dy) / norm / np.maximum(1, gaps[cand_local])
            else:
                gaps = start_frames - end_frames[i]
                mask = (gaps >= 1) & (gaps <= max_gap)
                cand_local = np.flatnonzero(mask)
                sims = np.asarray([
                    end_start_sim[fi, feat_index[tids[j]]]
                    if fi is not None and tids[j] in feat_index else np.nan
                    for j in cand_local
                ], dtype=np.float64)
                dx = first_cx[cand_local] - last_cx[i]
                dy = first_cy[cand_local] - last_cy[i]
                norm = np.maximum(1.0, 0.5 * (first_h[cand_local] + last_h[i]))
                motions = np.hypot(dx, dy) / norm / np.maximum(1, gaps[cand_local])

            finite = np.isfinite(sims)
            cand_local = cand_local[finite]
            sims = sims[finite]
            motions = motions[finite]
            cand_gaps = gaps[cand_local]
            if len(sims):
                order = np.argsort(-sims, kind='mergesort')
                best_idx = int(order[0])
                second_idx = int(order[1]) if len(order) > 1 else None
                best_sim = float(sims[best_idx])
                second_sim = float(sims[second_idx]) if second_idx is not None else float('nan')
                best_gap = int(cand_gaps[best_idx])
                best_motion = float(motions[best_idx])
            else:
                best_sim = float('nan'); second_sim = float('nan')
                best_gap = 0; best_motion = float('nan')
            r = rows[tid]
            r[f'{prefix}_best_sim'] = best_sim
            r[f'{prefix}_second_sim'] = second_sim
            r[f'{prefix}_sim_margin'] = best_sim - second_sim if math.isfinite(best_sim) and math.isfinite(second_sim) else float('nan')
            r[f'{prefix}_best_gap'] = best_gap
            r[f'{prefix}_best_motion'] = best_motion
            r[f'{prefix}_candidate_count'] = int(len(sims))
            for thr in [0.7, 0.8, 0.9]:
                r[f'{prefix}_count_sim_{str(thr).replace(".", "")}'] = int(np.sum(sims >= thr))

def build_labels(matches_csv: Path) -> Dict[str, Dict[int, dict]]:
    counts: Dict[str, Dict[int, Counter]] = defaultdict(lambda: defaultdict(Counter))
    with matches_csv.open(newline='') as f:
        for r in csv.DictReader(f):
            seq = r['seq']
            if seq not in SEQS:
                continue
            counts[seq][int(r['track_id'])][int(r['gt_id'])] += 1
    labels: Dict[str, Dict[int, dict]] = {}
    for seq, by_tid in counts.items():
        by_gt: Dict[int, Counter] = defaultdict(Counter)
        for tid, c in by_tid.items():
            for gt, n in c.items():
                by_gt[gt][tid] += n
        dom_gt = {tid: c.most_common(1)[0][0] for tid, c in by_tid.items() if c}
        dom_tid = {gt: c.most_common(1)[0][0] for gt, c in by_gt.items() if c}
        gt_total = {gt: sum(c.values()) for gt, c in by_gt.items()}
        gt_dom_rows = {gt: c[dom_tid[gt]] for gt, c in by_gt.items()}
        seq_labels = {}
        for tid, c in by_tid.items():
            total = sum(c.values())
            dgt = dom_gt[tid]
            dominant_rows = c[dgt]
            contamination = total - dominant_rows
            fragment_rows = sum(n for gt, n in c.items() if dom_tid[gt] != tid)
            repair_debt = 0
            core_rows = 0
            for gt, n in c.items():
                if dgt == gt and dom_tid[gt] == tid:
                    core_rows += n
                else:
                    repair_debt += n
            family_debt = gt_total[dgt] - gt_dom_rows[dgt]
            seq_labels[tid] = {
                'matched_rows': total,
                'dominant_gt': dgt,
                'dominant_gt_rows': dominant_rows,
                'track_gt_purity': dominant_rows / max(1, total),
                'track_unique_gt': len(c),
                'contamination_rows': contamination,
                'fragment_rows': fragment_rows,
                'repair_debt_rows': repair_debt,
                'repair_debt_ratio_matched': repair_debt / max(1, total),
                'core_rows': core_rows,
                'family_total_rows': gt_total[dgt],
                'family_debt_rows': family_debt,
                'family_debt_ratio': family_debt / max(1, gt_total[dgt]),
                'family_unique_tracker_ids': len(by_gt[dgt]),
                'is_dominant_tid_for_gt': int(dom_tid[dgt] == tid),
            }
        labels[seq] = seq_labels
    return labels


def add_overlap_features(dataset: Dict[Tuple[str, int], dict], overlap_csv: Path, feat_maps: Dict[str, Dict[int, dict]]) -> None:
    sim_cache = {}
    for seq, feats in feat_maps.items():
        tids, index, matrix = build_similarity_cache(feats, 'global_feat', 'global_feat')
        sim_cache[seq] = (index, matrix)
    agg = defaultdict(lambda: {
        'events': 0, 'sum_ioa': 0.0, 'max_ioa': 0.0, 'frames': set(), 'partners': set(),
        'sim_sum': 0.0, 'sim_count': 0, 'sim_max': -1.0, 'sim_high08': 0, 'sim_high09': 0,
    })
    with overlap_csv.open(newline='') as f:
        for r in csv.DictReader(f):
            seq = r['seq']
            if seq not in sim_cache:
                continue
            fr = int(r['frame']); a = int(r['track_i']); b = int(r['track_j']); ioa = float(r['ioa_min_area'])
            index, matrix = sim_cache[seq]
            sim = float(matrix[index[a], index[b]]) if a in index and b in index else float('nan')
            for tid, other in [(a, b), (b, a)]:
                key = (seq, tid)
                if key not in dataset:
                    continue
                x = agg[key]
                x['events'] += 1; x['sum_ioa'] += ioa; x['max_ioa'] = max(x['max_ioa'], ioa)
                x['frames'].add(fr); x['partners'].add(other)
                if math.isfinite(sim):
                    x['sim_sum'] += sim; x['sim_count'] += 1; x['sim_max'] = max(x['sim_max'], sim)
                    x['sim_high08'] += int(sim >= 0.8); x['sim_high09'] += int(sim >= 0.9)
    for key, row in dataset.items():
        x = agg[key]
        row['overlap_event_count'] = x['events']
        row['overlap_frame_count'] = len(x['frames'])
        row['overlap_partner_count'] = len(x['partners'])
        row['overlap_ioa_mean'] = x['sum_ioa'] / max(1, x['events'])
        row['overlap_ioa_max'] = x['max_ioa']
        row['overlap_frame_fraction'] = len(x['frames']) / max(1, row['track_len'])
        row['overlap_comp_sim_mean'] = x['sim_sum'] / max(1, x['sim_count']) if x['sim_count'] else float('nan')
        row['overlap_comp_sim_max'] = x['sim_max'] if x['sim_count'] else float('nan')
        row['overlap_comp_sim_high08_frac'] = x['sim_high08'] / max(1, x['sim_count'])
        row['overlap_comp_sim_high09_frac'] = x['sim_high09'] / max(1, x['sim_count'])

def family_capture_metrics(df: pd.DataFrame, order: np.ndarray, frac: float) -> dict:
    n = max(1, int(round(len(df) * frac)))
    sel = df.iloc[order[:n]]
    total_debt = float(df['repair_debt_rows'].sum())
    high = df['repair_debt_rows'] >= 20
    selected_high = sel['repair_debt_rows'] >= 20
    family_table = df.groupby('dominant_gt', as_index=False)['family_debt_rows'].max()
    total_family_debt = float(family_table['family_debt_rows'].sum())
    selected_families = set(int(x) for x in sel['dominant_gt'] if int(x) >= 0)
    captured_family = float(family_table[family_table['dominant_gt'].isin(selected_families)]['family_debt_rows'].sum())
    return {
        'fraction': frac,
        'selected_tracks': n,
        'debt_mass_recall': float(sel['repair_debt_rows'].sum() / max(1.0, total_debt)),
        'high_debt_precision': float(selected_high.mean()),
        'high_debt_recall': float(selected_high.sum() / max(1, high.sum())),
        'family_debt_mass_recall': captured_family / max(1.0, total_family_debt),
        'selected_unique_families': len(selected_families),
    }


def evaluate_models(df: pd.DataFrame, feature_cols: List[str], out_dir: Path) -> tuple[list, pd.DataFrame, pd.DataFrame]:
    results = []
    pred_frames = []
    importance = np.zeros(len(feature_cols), dtype=np.float64)
    models = {
        'ridge': lambda: make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), Ridge(alpha=10.0)),
        'rf': lambda: make_pipeline(SimpleImputer(strategy='median'), RandomForestRegressor(
            n_estimators=400, min_samples_leaf=5, max_features=0.65, n_jobs=-1, random_state=42)),
        'hgb': lambda: make_pipeline(SimpleImputer(strategy='median'), HistGradientBoostingRegressor(
            max_iter=250, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=12,
            l2_regularization=2.0, random_state=42)),
    }
    for holdout in SEQS:
        train = df[df['seq'] != holdout].copy()
        test = df[df['seq'] == holdout].copy().reset_index(drop=True)
        if len(train) == 0 or len(test) == 0:
            continue
        Xtr = train[feature_cols]; Xte = test[feature_cols]
        ytr = np.log1p(train['repair_debt_rows'].to_numpy(dtype=float))
        ytest = test['repair_debt_rows'].to_numpy(dtype=float)
        scores = {
            'track_length': test['track_len'].to_numpy(dtype=float),
            'observable_handcrafted': (
                test['track_len_pct'].to_numpy(dtype=float)
                + test['overlap_frame_fraction_pct'].to_numpy(dtype=float)
                + (1.0 - test['start_end_cos'].fillna(test['start_end_cos'].median()).to_numpy(dtype=float))
                + test['temporal_ambiguity_count_pct'].to_numpy(dtype=float)
            ),
        }
        for name, factory in models.items():
            model = factory()
            model.fit(Xtr, ytr)
            pred = np.expm1(model.predict(Xte))
            scores[name] = np.maximum(0.0, pred)
            if name == 'rf':
                rf = model[-1]
                importance += rf.feature_importances_
        for name, score in scores.items():
            order = np.argsort(-score, kind='mergesort')
            rho = spearmanr(score, ytest).statistic if len(np.unique(score)) > 1 else float('nan')
            positive = ytest >= 20
            ap = average_precision_score(positive.astype(int), score) if positive.any() else float('nan')
            auc = roc_auc_score(positive.astype(int), score) if positive.any() and (~positive).any() else float('nan')
            rec = {
                'holdout': holdout,
                'model': name,
                'tracks': len(test),
                'positive_tracks_debt_ge20': int(positive.sum()),
                'total_repair_debt_rows': float(ytest.sum()),
                'spearman_debt_rows': float(rho) if rho is not None else float('nan'),
                'average_precision_debt_ge20': float(ap),
                'roc_auc_debt_ge20': float(auc),
                'topk': [family_capture_metrics(test, order, f) for f in [0.1, 0.2, 0.3, 0.5]],
            }
            results.append(rec)
            pf = test[['seq', 'track_id', 'dominant_gt', 'repair_debt_rows', 'family_debt_rows']].copy()
            pf['model'] = name; pf['score'] = score; pf['rank'] = pd.Series(score).rank(ascending=False, method='first').astype(int)
            pred_frames.append(pf)
    importance /= max(1, len(SEQS))
    imp_df = pd.DataFrame({'feature': feature_cols, 'rf_importance': importance}).sort_values('rf_importance', ascending=False)
    preds = pd.concat(pred_frames, ignore_index=True)
    return results, preds, imp_df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--track-root', required=True)
    ap.add_argument('--feature-root', required=True)
    ap.add_argument('--matches-csv', required=True)
    ap.add_argument('--overlap-csv', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--seqs', nargs='+', default=SEQS)
    args = ap.parse_args()

    track_root = Path(args.track_root)
    feature_root = Path(args.feature_root)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    labels = build_labels(Path(args.matches_csv))
    dataset: Dict[Tuple[str, int], dict] = {}
    feat_maps: Dict[str, Dict[int, dict]] = {}

    for seq in args.seqs:
        tracks = load_track_rows(track_root / f'{seq}.txt')
        fmap, _ = load_feature_maps(feature_root, seq)
        feat_maps[seq] = fmap
        seq_rows: Dict[int, dict] = {}
        for tid, rows in tracks.items():
            row = {'seq': seq, 'track_id': tid}
            row.update(track_geometry_features(rows))
            f = fmap.get(tid)
            if f:
                row['has_reid_feature'] = 1
                row['feature_samples'] = f['feature_samples']
                row['feature_index_len_ratio'] = f['index_track_len'] / max(1, row['track_len'])
                row['feature_start_match'] = int(f['index_start_frame'] == row['start_frame'])
                row['feature_end_match'] = int(f['index_end_frame'] == row['end_frame'])
                row['feature_exact_alignment'] = int(
                    f['index_track_len'] == row['track_len'] and
                    f['index_start_frame'] == row['start_frame'] and
                    f['index_end_frame'] == row['end_frame'])
                row['start_end_cos'] = cosine(f['start_feat'], f['end_feat'])
                row['start_global_cos'] = cosine(f['start_feat'], f['global_feat'])
                row['end_global_cos'] = cosine(f['end_feat'], f['global_feat'])
                row['high_global_cos'] = cosine(f['high_feat'], f['global_feat'])
                row['start_high_cos'] = cosine(f['start_feat'], f['high_feat'])
                row['end_high_cos'] = cosine(f['end_feat'], f['high_feat'])
            else:
                row.update({
                    'has_reid_feature': 0, 'feature_samples': 0, 'feature_index_len_ratio': float('nan'),
                    'feature_start_match': 0, 'feature_end_match': 0, 'feature_exact_alignment': 0,
                    'start_end_cos': float('nan'), 'start_global_cos': float('nan'),
                    'end_global_cos': float('nan'), 'high_global_cos': float('nan'),
                    'start_high_cos': float('nan'), 'end_high_cos': float('nan'),
                })
            label = labels.get(seq, {}).get(tid)
            if label:
                row.update(label)
            else:
                row.update({
                    'matched_rows': 0, 'dominant_gt': -1, 'dominant_gt_rows': 0,
                    'track_gt_purity': 0.0, 'track_unique_gt': 0, 'contamination_rows': 0,
                    'fragment_rows': 0, 'repair_debt_rows': 0, 'repair_debt_ratio_matched': 0.0,
                    'core_rows': 0, 'family_total_rows': 0, 'family_debt_rows': 0,
                    'family_debt_ratio': 0.0, 'family_unique_tracker_ids': 0,
                    'is_dominant_tid_for_gt': 0,
                })
            row['repair_debt_ratio_track'] = row['repair_debt_rows'] / max(1, row['track_len'])
            seq_rows[tid] = row
            dataset[(seq, tid)] = row
        add_temporal_ambiguity(seq_rows, fmap, max_gap=300)
        for tid, row in seq_rows.items():
            row['temporal_ambiguity_count'] = (
                row.get('pred_count_sim_08', 0) + row.get('succ_count_sim_08', 0))
            row['temporal_best_sim'] = np.nanmax([
                row.get('pred_best_sim', float('nan')), row.get('succ_best_sim', float('nan'))
            ]) if any(math.isfinite(x) for x in [row.get('pred_best_sim', float('nan')), row.get('succ_best_sim', float('nan'))]) else float('nan')

    add_overlap_features(dataset, Path(args.overlap_csv), feat_maps)
    df = pd.DataFrame(list(dataset.values())).sort_values(['seq', 'track_id']).reset_index(drop=True)

    # Observable within-sequence percentile features improve cross-sequence calibration.
    pct_base = [
        'track_len', 'track_span', 'internal_gap_frames', 'speed_max',
        'overlap_event_count', 'overlap_frame_fraction', 'overlap_partner_count',
        'overlap_comp_sim_max', 'temporal_ambiguity_count', 'temporal_best_sim',
    ]
    for col in pct_base:
        df[col + '_pct'] = df.groupby('seq')[col].rank(pct=True, method='average').fillna(0.0)

    label_cols = {
        'matched_rows', 'dominant_gt', 'dominant_gt_rows', 'track_gt_purity', 'track_unique_gt',
        'contamination_rows', 'fragment_rows', 'repair_debt_rows', 'repair_debt_ratio_matched',
        'core_rows', 'family_total_rows', 'family_debt_rows', 'family_debt_ratio',
        'family_unique_tracker_ids', 'is_dominant_tid_for_gt', 'repair_debt_ratio_track',
    }
    id_cols = {'seq', 'track_id', 'start_frame', 'end_frame', 'first_cx', 'first_cy', 'last_cx', 'last_cy', 'first_h', 'last_h'}
    feature_cols = [c for c in df.columns if c not in label_cols and c not in id_cols]
    # Remove accidental non-numeric columns.
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]

    df.to_csv(out_dir / 'tracker_identity_debt_dataset.csv', index=False)
    results, preds, imp = evaluate_models(df, feature_cols, out_dir)
    preds.to_csv(out_dir / 'loo_predictions.csv', index=False)
    imp.to_csv(out_dir / 'rf_feature_importance.csv', index=False)

    # Aggregate across folds for compact comparison.
    aggregate = {}
    for model in sorted({r['model'] for r in results}):
        rr = [r for r in results if r['model'] == model]
        aggregate[model] = {
            'mean_spearman': float(np.nanmean([r['spearman_debt_rows'] for r in rr])),
            'mean_average_precision': float(np.nanmean([r['average_precision_debt_ge20'] for r in rr])),
            'mean_roc_auc': float(np.nanmean([r['roc_auc_debt_ge20'] for r in rr])),
            'mean_topk': {},
        }
        for frac in [0.1, 0.2, 0.3, 0.5]:
            vals = [next(x for x in r['topk'] if x['fraction'] == frac) for r in rr]
            aggregate[model]['mean_topk'][str(frac)] = {
                k: float(np.mean([v[k] for v in vals]))
                for k in ['debt_mass_recall', 'high_debt_precision', 'high_debt_recall', 'family_debt_mass_recall']
            }

    report = {
        'protocol': {
            'feature_policy': 'tracker-observable only; GT excluded from all feature columns',
            'label_policy': 'GT used only after feature construction for repair-debt labels',
            'validation': 'leave-one-sequence-out across selected sequences',
            'target': 'log1p(repair_debt_rows)',
            'high_debt_definition': 'repair_debt_rows >= 20',
        },
        'rows': len(df),
        'rows_by_seq': df.groupby('seq').size().to_dict(),
        'positive_debt_rows_by_seq': df.groupby('seq')['repair_debt_rows'].sum().to_dict(),
        'feature_count': len(feature_cols),
        'feature_columns': feature_cols,
        'fold_results': results,
        'aggregate': aggregate,
        'top20_rf_features': imp.head(20).to_dict(orient='records'),
    }
    (out_dir / 'loo_report.json').write_text(json.dumps(report, indent=2) + '\n')

    lines = [
        '# Identity Debt Proxy Benchmark', '',
        '- Features: tracker-observable only.',
        '- Labels: GT applied only after feature construction.',
        '- Validation: leave-one-sequence-out.', '',
        '## Aggregate', '',
        '| model | Spearman | AP(debt>=20) | ROC-AUC | Top10 debt recall | Top30 debt recall | Top50 family debt recall |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for model, r in aggregate.items():
        lines.append(
            f"| {model} | {r['mean_spearman']:.4f} | {r['mean_average_precision']:.4f} | {r['mean_roc_auc']:.4f} | "
            f"{r['mean_topk']['0.1']['debt_mass_recall']:.4f} | {r['mean_topk']['0.3']['debt_mass_recall']:.4f} | "
            f"{r['mean_topk']['0.5']['family_debt_mass_recall']:.4f} |")
    lines += ['', '## RF feature importance', '', '| rank | feature | importance |', '|---:|---|---:|']
    for i, r in enumerate(imp.head(20).to_dict(orient='records'), 1):
        lines.append(f"| {i} | {r['feature']} | {r['rf_importance']:.6f} |")
    (out_dir / 'README.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({
        'rows': len(df), 'feature_count': len(feature_cols), 'aggregate': aggregate,
        'm02': [r for r in results if r['holdout'] == 'MOT20-02'],
        'top10_features': imp.head(10).to_dict(orient='records'),
    }, indent=2))


if __name__ == '__main__':
    main()
