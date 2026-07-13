#!/usr/bin/env python3
"""Tiered Edge Validator experiment for A-Link v2.

This is a diagnostic / calibration script, not a test submission generator.
It builds a labelled dataset from v2 tiered_edges and oracle tracklet labels,
then performs leave-one-sequence-out validation to answer:
  Can existing edge features separate true links from false links better than the
  hand-written edge_score / tier sorting?

True label definition:
  track_a and track_b are both eligible and have the same majority_gt.
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


def load_labels(oracle_dir: Path, seq: str) -> Tuple[Dict[int, int], Dict[int, int]]:
    labels = read_csv(oracle_dir / f'{seq}_tracklet_labels.csv')
    eligible, gt = {}, {}
    for r in labels:
        tid = ii(r.get('tid'))
        eligible[tid] = ii(r.get('eligible'))
        gt[tid] = ii(r.get('majority_gt'), -1)
    return eligible, gt


def build_rows(detail_dir: Path, oracle_dir: Path, seqs: Iterable[str]) -> List[dict]:
    all_rows = []
    for seq in seqs:
        eligible, gt = load_labels(oracle_dir, seq)
        tiered = read_csv(detail_dir / f'{seq}_tiered_edges.csv')
        selected_pairs = {(ii(r.get('track_a')), ii(r.get('track_b'))) for r in read_csv(detail_dir / f'{seq}_selected_links.csv')}
        for r in tiered:
            a = ii(r.get('track_a')); b = ii(r.get('track_b'))
            label = int(eligible.get(a, 0) and eligible.get(b, 0) and gt.get(a, -1) > 0 and gt.get(a) == gt.get(b))
            out = {
                'seq': seq,
                'track_a': a,
                'track_b': b,
                'label': label,
                'selected_original': int((a, b) in selected_pairs),
                'tier': r.get('tier', ''),
                'tier_num': TIER_MAP.get(r.get('tier', ''), 0.0),
                'gt_a': gt.get(a, -1),
                'gt_b': gt.get(b, -1),
            }
            for k in FEATURES:
                out[k] = ff(r.get(k))
            all_rows.append(out)
    return all_rows


def standardize_fit(rows: List[dict], feats: List[str]) -> Tuple[Dict[str, float], Dict[str, float]]:
    mu, sig = {}, {}
    for f in feats:
        vals = [ff(r.get(f)) for r in rows]
        m = mean(vals) if vals else 0.0
        var = mean([(v - m) ** 2 for v in vals]) if vals else 0.0
        s = math.sqrt(max(var, 1e-12))
        mu[f] = m; sig[f] = s
    return mu, sig


def transform(rows: List[dict], feats: List[str], mu: Dict[str, float], sig: Dict[str, float]):
    return [[(ff(r.get(f)) - mu[f]) / sig[f] for f in feats] for r in rows]


def labels(rows: List[dict]) -> List[int]:
    return [ii(r.get('label')) for r in rows]


def train_predict_sklearn(train_rows: List[dict], test_rows: List[dict], feats: List[str]) -> Tuple[List[float], str]:
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
    except Exception as e:
        raise RuntimeError(f'sklearn_unavailable: {e}')
    y = labels(train_rows)
    if len(set(y)) < 2:
        return [0.0 for _ in test_rows], 'constant'
    mu, sig = standardize_fit(train_rows, feats)
    X_train = transform(train_rows, feats, mu, sig)
    X_test = transform(test_rows, feats, mu, sig)
    models = []
    # Small dataset: use stable, low-complexity models first.
    models.append(('logreg', LogisticRegression(max_iter=2000, class_weight='balanced', C=0.5, solver='liblinear')))
    models.append(('rf', RandomForestClassifier(n_estimators=200, max_depth=4, min_samples_leaf=3, class_weight='balanced_subsample', random_state=7)))
    models.append(('hgb', HistGradientBoostingClassifier(max_iter=80, learning_rate=0.05, max_leaf_nodes=8, l2_regularization=0.1, random_state=7)))
    best_name, best_model, best_auc = None, None, -1.0
    # pick model by in-train AUC only as a weak sanity preference; LOSO metrics are still reported on held-out seq.
    for name, model in models:
        try:
            model.fit(X_train, y)
            p = model.predict_proba(X_train)[:, 1].tolist()
            auc = roc_auc_score(y, p) if len(set(y)) > 1 else 0.5
            if auc > best_auc:
                best_name, best_model, best_auc = name, model, auc
        except Exception:
            continue
    if best_model is None:
        return [ff(r.get('edge_score')) for r in test_rows], 'edge_score_fallback'
    pred = best_model.predict_proba(X_test)[:, 1].tolist()
    return [float(x) for x in pred], f'{best_name}_trainauc_{best_auc:.3f}'


def train_predict_heuristic(train_rows: List[dict], test_rows: List[dict], feats: List[str]) -> Tuple[List[float], str]:
    # Fallback: standardized hand score emphasizing app+motion+size+margin, penalizing step/gap.
    mu, sig = standardize_fit(train_rows, feats)
    weights = {
        'app_sim': 0.8, 'bank_max': 0.7, 'bank_topk': 0.5, 'end_start_sim': 0.4,
        'motion_score': 0.5, 'size_score': 0.35, 'source_margin': 0.3, 'target_margin': 0.3,
        'center_step': -0.35, 'gap': -0.15, 'area_ratio': -0.2, 'height_ratio': -0.2,
        'tier_num': 0.2,
    }
    out = []
    for r in test_rows:
        s = 0.0
        for k, w in weights.items():
            if k == 'tier_num':
                v = ff(r.get(k))
                s += w * v
            elif k in mu:
                s += w * ((ff(r.get(k)) - mu[k]) / sig[k])
        out.append(float(s))
    return out, 'heuristic'


def greedy_select(rows: List[dict], scores: List[float], max_links: int | None = None) -> List[dict]:
    order = sorted(range(len(rows)), key=lambda i: (-scores[i], ff(rows[i].get('gap')), ii(rows[i].get('track_a')), ii(rows[i].get('track_b'))))
    used_a, used_b = set(), set()
    selected = []
    for i in order:
        r = rows[i]
        a = ii(r.get('track_a')); b = ii(r.get('track_b'))
        if a in used_a or b in used_b:
            continue
        used_a.add(a); used_b.add(b)
        rr = dict(r); rr['validator_score'] = scores[i]
        selected.append(rr)
        if max_links is not None and len(selected) >= max_links:
            break
    return selected


def metrics_for(rows: List[dict]) -> dict:
    n = len(rows); tp = sum(ii(r.get('label')) for r in rows)
    return {'n': n, 'tp': tp, 'precision': tp / n if n else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--detail-dir', required=True)
    ap.add_argument('--oracle-detail-dir', required=True)
    ap.add_argument('--seqs', nargs='+', required=True)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = build_rows(Path(args.detail_dir), Path(args.oracle_detail_dir), args.seqs)
    write_csv(out_dir / 'tiered_edge_dataset.csv', all_rows)
    feat_cols = FEATURES + ['tier_num']
    summaries = []
    all_pred_rows = []
    for test_seq in args.seqs:
        train_rows = [r for r in all_rows if r['seq'] != test_seq]
        test_rows = [r for r in all_rows if r['seq'] == test_seq]
        if not test_rows or not train_rows:
            continue
        try:
            scores, model_name = train_predict_sklearn(train_rows, test_rows, feat_cols)
        except Exception as e:
            scores, model_name = train_predict_heuristic(train_rows, test_rows, feat_cols)
            model_name += f' ({e})'
        original_selected = [r for r in test_rows if ii(r.get('selected_original'))]
        k_orig = len(original_selected)
        greedy_k = greedy_select(test_rows, scores, k_orig if k_orig else None)
        greedy_all_pos_threshold = greedy_select(test_rows, scores, None)
        # Top-K without conflict, plus raw topK for ranking diagnostic.
        raw_order = sorted(range(len(test_rows)), key=lambda i: -scores[i])[:k_orig]
        raw_topk = [test_rows[i] for i in raw_order]
        s_orig = metrics_for(original_selected)
        s_greedy = metrics_for(greedy_k)
        s_raw = metrics_for(raw_topk)
        row = {
            'test_seq': test_seq,
            'model': model_name,
            'tiered_edges': len(test_rows),
            'positives': sum(ii(r.get('label')) for r in test_rows),
            'orig_k': k_orig,
            'orig_tp': s_orig['tp'], 'orig_precision': s_orig['precision'],
            'validator_greedy_k_tp': s_greedy['tp'], 'validator_greedy_k_precision': s_greedy['precision'],
            'raw_topk_tp': s_raw['tp'], 'raw_topk_precision': s_raw['precision'],
        }
        summaries.append(row)
        for r, sc in zip(test_rows, scores):
            rr = dict(r); rr['validator_score'] = sc; rr['test_seq'] = test_seq; rr['model'] = model_name
            all_pred_rows.append(rr)
        write_csv(out_dir / f'{test_seq}_validator_greedy_selected.csv', greedy_k)
    write_csv(out_dir / 'loso_summary.csv', summaries)
    write_csv(out_dir / 'loso_predictions.csv', all_pred_rows)
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
