#!/usr/bin/env python3
"""Safe-link validator experiment for A-Link.

This is stricter than the same-GT validator. It treats same-GT but impure/weak
tracklet pairs as *risky* and excludes them from training, so the model learns
"safe to merge" rather than merely "same majority GT".

Inputs:
  - A-Link v2 tiered_edges/selected_links details.
  - Oracle tracklet_labels with majority_gt, purity, match_frac, etc.

Outputs:
  - safe_edge_dataset.csv
  - loso_summary.csv
  - loso_predictions.csv with validator_score, compatible with apply_validator_links.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Tuple

FEATURES = [
    'edge_score', 'app_sim', 'bank_max', 'bank_topk', 'end_start_sim', 'global_sim', 'high_sim',
    'motion_score', 'size_score', 'direction_score', 'gap_score', 'quality_score',
    'center_step', 'pred_dist', 'gap', 'area_ratio', 'height_ratio',
    'source_rank', 'target_rank', 'source_margin', 'target_margin',
    'source_purity', 'target_purity', 'min_purity',
    'source_match_frac', 'target_match_frac', 'min_match_frac',
    'source_majority_count', 'target_majority_count', 'min_majority_count',
    'source_length', 'target_length', 'min_length',
]
TIER_MAP = {'tier1': 3.0, 'tier2': 2.0, 'tier3': 1.0, 'reject': 0.0, '': 0.0}


def ff(x, default=0.0) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def ii(x, default=0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default


def read_csv(path: Path) -> List[dict]:
    if not path.is_file():
        return []
    with path.open('r', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(set().union(*(r.keys() for r in rows))) if rows else ['seq']
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader(); w.writerows(rows)


def load_tracklet_labels(oracle_dir: Path, seq: str) -> Dict[int, dict]:
    out = {}
    for r in read_csv(oracle_dir / f'{seq}_tracklet_labels.csv'):
        tid = ii(r.get('tid'))
        out[tid] = {
            'eligible': ii(r.get('eligible')),
            'majority_gt': ii(r.get('majority_gt'), -1),
            'purity': ff(r.get('purity')),
            'match_frac': ff(r.get('match_frac')),
            'majority_count': ii(r.get('majority_count')),
            'matched_count': ii(r.get('matched_count')),
            'length': ii(r.get('length')),
        }
    return out


def build_rows(detail_dir: Path, oracle_dir: Path, seqs: Iterable[str], args) -> List[dict]:
    all_rows = []
    for seq in seqs:
        labels = load_tracklet_labels(oracle_dir, seq)
        selected_pairs = {(ii(r.get('track_a')), ii(r.get('track_b'))) for r in read_csv(detail_dir / f'{seq}_selected_links.csv')}
        for r in read_csv(detail_dir / f'{seq}_tiered_edges.csv'):
            a = ii(r.get('track_a')); b = ii(r.get('track_b'))
            la = labels.get(a, {}); lb = labels.get(b, {})
            gt_a = ii(la.get('majority_gt'), -1); gt_b = ii(lb.get('majority_gt'), -1)
            eligible = int(ii(la.get('eligible')) and ii(lb.get('eligible')))
            same_gt = int(eligible and gt_a > 0 and gt_a == gt_b)
            sp = ff(la.get('purity')); tp = ff(lb.get('purity'))
            sm = ff(la.get('match_frac')); tm = ff(lb.get('match_frac'))
            sc = ii(la.get('majority_count')); tc = ii(lb.get('majority_count'))
            sl = ii(la.get('length')); tl = ii(lb.get('length'))
            min_p = min(sp, tp); min_m = min(sm, tm); min_c = min(sc, tc); min_l = min(sl, tl)
            gap = ii(r.get('gap'))
            safe = int(
                same_gt and
                min_p >= args.safe_min_purity and
                min_m >= args.safe_min_match_frac and
                min_c >= args.safe_min_majority_count and
                min_l >= args.safe_min_length and
                gap <= args.safe_max_gap
            )
            risky = int(same_gt and not safe)
            unsafe = int(not same_gt)
            # safe_train_label is 1 safe, 0 unsafe, blank for risky excluded from supervised training.
            train_label = 1 if safe else (0 if unsafe else '')
            out = {
                'seq': seq,
                'track_a': a,
                'track_b': b,
                'label_same_gt': same_gt,
                'label_safe': safe,
                'label_risky': risky,
                'label_unsafe': unsafe,
                'train_label': train_label,
                'selected_original': int((a, b) in selected_pairs),
                'tier': r.get('tier', ''),
                'tier_num': TIER_MAP.get(r.get('tier', ''), 0.0),
                'gt_a': gt_a,
                'gt_b': gt_b,
                'source_purity': sp,
                'target_purity': tp,
                'min_purity': min_p,
                'source_match_frac': sm,
                'target_match_frac': tm,
                'min_match_frac': min_m,
                'source_majority_count': sc,
                'target_majority_count': tc,
                'min_majority_count': min_c,
                'source_length': sl,
                'target_length': tl,
                'min_length': min_l,
            }
            for k in FEATURES:
                if k not in out:
                    out[k] = ff(r.get(k))
            all_rows.append(out)
    return all_rows


def standardize_fit(rows: List[dict], feats: List[str]) -> Tuple[Dict[str, float], Dict[str, float]]:
    mu, sig = {}, {}
    for f in feats:
        vals = [ff(r.get(f)) for r in rows]
        m = mean(vals) if vals else 0.0
        var = mean([(v - m) ** 2 for v in vals]) if vals else 0.0
        mu[f] = m
        sig[f] = math.sqrt(max(var, 1e-12))
    return mu, sig


def transform(rows: List[dict], feats: List[str], mu: Dict[str, float], sig: Dict[str, float]):
    return [[(ff(r.get(f)) - mu[f]) / sig[f] for f in feats] for r in rows]


def train_predict(train_rows: List[dict], test_rows: List[dict], feats: List[str]) -> Tuple[List[float], str]:
    # Only safe vs unsafe for training. Risky same-GT rows are excluded from training.
    train = [r for r in train_rows if str(r.get('train_label')) in ('0', '1')]
    y = [ii(r.get('train_label')) for r in train]
    if not train or len(set(y)) < 2:
        return [ff(r.get('edge_score')) for r in test_rows], 'edge_score_fallback'
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        mu, sig = standardize_fit(train, feats)
        X = transform(train, feats, mu, sig)
        Xt = transform(test_rows, feats, mu, sig)
        models = [
            ('logreg_safe', LogisticRegression(max_iter=2000, class_weight='balanced', C=0.3, solver='liblinear')),
            ('rf_safe', RandomForestClassifier(n_estimators=240, max_depth=4, min_samples_leaf=3, class_weight='balanced_subsample', random_state=11)),
            ('hgb_safe', HistGradientBoostingClassifier(max_iter=90, learning_rate=0.045, max_leaf_nodes=8, l2_regularization=0.15, random_state=11)),
        ]
        best = None
        for name, model in models:
            try:
                model.fit(X, y)
                p = model.predict_proba(X)[:, 1].tolist()
                auc = roc_auc_score(y, p)
                if best is None or auc > best[0]:
                    best = (auc, name, model, mu, sig)
            except Exception:
                continue
        if best is None:
            raise RuntimeError('all sklearn models failed')
        auc, name, model, mu, sig = best
        pred = model.predict_proba(Xt)[:, 1].tolist()
        return [float(x) for x in pred], f'{name}_trainauc_{auc:.3f}'
    except Exception as e:
        # fallback safety heuristic: prioritize purity/match plus app/motion; penalize step/gap.
        feats2 = feats
        mu, sig = standardize_fit(train, feats2)
        weights = {
            'min_purity': 0.9, 'min_match_frac': 0.9, 'min_majority_count': 0.25, 'min_length': 0.2,
            'app_sim': 0.55, 'bank_max': 0.45, 'motion_score': 0.35, 'size_score': 0.25,
            'source_margin': 0.25, 'target_margin': 0.25, 'center_step': -0.35, 'gap': -0.12,
            'area_ratio': -0.18, 'height_ratio': -0.18, 'tier_num': 0.15,
        }
        out = []
        for r in test_rows:
            s = 0.0
            for k, w in weights.items():
                if k == 'tier_num':
                    s += w * ff(r.get(k))
                elif k in mu:
                    s += w * ((ff(r.get(k)) - mu[k]) / sig[k])
            out.append(float(s))
        return out, f'safety_heuristic ({e})'


def greedy_select(rows: List[dict], scores: List[float], max_links: int | None = None) -> List[dict]:
    order = sorted(range(len(rows)), key=lambda i: (-scores[i], ff(rows[i].get('gap')), ii(rows[i].get('track_a')), ii(rows[i].get('track_b'))))
    used_a, used_b = set(), set()
    out = []
    for i in order:
        r = rows[i]
        a = ii(r.get('track_a')); b = ii(r.get('track_b'))
        if a in used_a or b in used_b:
            continue
        used_a.add(a); used_b.add(b)
        rr = dict(r); rr['validator_score'] = scores[i]
        out.append(rr)
        if max_links is not None and len(out) >= max_links:
            break
    return out


def metric(rows: List[dict], key: str = 'label_safe') -> dict:
    n = len(rows)
    tp = sum(ii(r.get(key)) for r in rows)
    return {'n': n, 'tp': tp, 'precision': tp / n if n else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--detail-dir', required=True)
    ap.add_argument('--oracle-detail-dir', required=True)
    ap.add_argument('--seqs', nargs='+', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--safe-min-purity', type=float, default=0.85)
    ap.add_argument('--safe-min-match-frac', type=float, default=0.50)
    ap.add_argument('--safe-min-majority-count', type=int, default=5)
    ap.add_argument('--safe-min-length', type=int, default=5)
    ap.add_argument('--safe-max-gap', type=int, default=120)
    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rows = build_rows(Path(args.detail_dir), Path(args.oracle_detail_dir), args.seqs, args)
    write_csv(out_dir / 'safe_edge_dataset.csv', rows)
    feats = FEATURES + ['tier_num']
    summaries = []
    preds = []
    for test_seq in args.seqs:
        train_rows = [r for r in rows if r['seq'] != test_seq]
        test_rows = [r for r in rows if r['seq'] == test_seq]
        if not test_rows:
            continue
        scores, model_name = train_predict(train_rows, test_rows, feats)
        orig = [r for r in test_rows if ii(r.get('selected_original'))]
        k = len(orig)
        sel = greedy_select(test_rows, scores, k if k else None)
        raw_order = sorted(range(len(test_rows)), key=lambda i: -scores[i])[:k]
        raw = [test_rows[i] for i in raw_order]
        row = {
            'test_seq': test_seq,
            'model': model_name,
            'tiered_edges': len(test_rows),
            'safe_pos': sum(ii(r.get('label_safe')) for r in test_rows),
            'risky_pos': sum(ii(r.get('label_risky')) for r in test_rows),
            'unsafe_neg': sum(ii(r.get('label_unsafe')) for r in test_rows),
            'orig_k': k,
            'orig_safe_tp': metric(orig, 'label_safe')['tp'],
            'orig_safe_precision': metric(orig, 'label_safe')['precision'],
            'orig_samegt_tp': metric(orig, 'label_same_gt')['tp'],
            'orig_samegt_precision': metric(orig, 'label_same_gt')['precision'],
            'validator_greedy_k_safe_tp': metric(sel, 'label_safe')['tp'],
            'validator_greedy_k_safe_precision': metric(sel, 'label_safe')['precision'],
            'validator_greedy_k_samegt_tp': metric(sel, 'label_same_gt')['tp'],
            'validator_greedy_k_samegt_precision': metric(sel, 'label_same_gt')['precision'],
            'raw_topk_safe_tp': metric(raw, 'label_safe')['tp'],
            'raw_topk_safe_precision': metric(raw, 'label_safe')['precision'],
        }
        summaries.append(row)
        for r, sc in zip(test_rows, scores):
            rr = dict(r)
            rr['validator_score'] = float(sc)
            rr['test_seq'] = test_seq
            rr['model'] = model_name
            preds.append(rr)
        write_csv(out_dir / f'{test_seq}_safe_validator_greedy_selected.csv', sel)
    write_csv(out_dir / 'safe_loso_summary.csv', summaries)
    write_csv(out_dir / 'loso_predictions.csv', preds)
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
