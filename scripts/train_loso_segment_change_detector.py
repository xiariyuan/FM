from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score

ID_COLS = {'seq', 'track_id', 'boundary_frame', 'prev_frame'}
LABEL_PREFIXES = ('label_switch_', 'distance_to_switch_')


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float('nan')


def balanced_weights(frame: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    counts = frame['seq'].value_counts()
    seq_weight = frame['seq'].map({seq: len(frame) / (len(counts) * n) for seq, n in counts.items()}).to_numpy(float)
    positives = int(y.sum())
    class_weight = max(1.0, (len(y) - positives) / max(1, positives))
    weights = seq_weight * np.where(y == 1, class_weight, 1.0)
    return weights / max(float(weights.mean()), 1e-12)


def nms_events(df: pd.DataFrame, score_col: str, radius: int) -> pd.DataFrame:
    selected = []
    for tid, group in df.sort_values(score_col, ascending=False).groupby('track_id', sort=False):
        kept = []
        for row in group.itertuples(index=False):
            frame = int(row.boundary_frame)
            if any(abs(frame - old) <= radius for old in kept):
                continue
            kept.append(frame)
            selected.append(row._asdict())
    return pd.DataFrame(selected).sort_values(score_col, ascending=False).reset_index(drop=True)


def event_metrics(pool: pd.DataFrame, selected: pd.DataFrame, exact_events: pd.DataFrame,
                  score_col: str, tolerance: int, budgets=(50,100,200,500,1000,2000,5000)):
    truth = {(int(r.track_id), int(r.boundary_frame)) for r in exact_events.itertuples(index=False)}
    rows = []
    for budget in budgets:
        pred = [(int(r.track_id), int(r.boundary_frame)) for r in selected.head(budget).itertuples(index=False)]
        hit_truth = set(); hit_pred = 0
        for tid, frame in pred:
            matches = [(t, f) for t, f in truth if t == tid and abs(f - frame) <= tolerance]
            if matches:
                nearest = min(matches, key=lambda x: abs(x[1] - frame))
                hit_truth.add(nearest); hit_pred += 1
        rows.append({
            'score': score_col,
            'budget': budget,
            'selected': len(pred),
            'true_events': len(truth),
            'matched_true_events': len(hit_truth),
            'event_recall': len(hit_truth) / max(1, len(truth)),
            'event_precision_upper': hit_pred / max(1, len(pred)),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features-csv', action='append', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--persistence', type=int, default=5)
    ap.add_argument('--label-window', type=int, default=3)
    ap.add_argument('--candidate-ioa', type=float, default=.1)
    ap.add_argument('--nms-radius', type=int, default=3)
    args = ap.parse_args()

    frames = [pd.read_csv(path) for path in args.features_csv]
    seqs = [str(frame['seq'].iloc[0]) for frame in frames]
    if len(seqs) != len(set(seqs)):
        raise RuntimeError('duplicate sequence feature files')

    common = set(frames[0].columns)
    for frame in frames[1:]:
        common &= set(frame.columns)
    numeric_common = [
        c for c in sorted(common)
        if all(pd.api.types.is_numeric_dtype(frame[c]) for frame in frames)
    ]
    raw_features = [
        c for c in numeric_common
        if c not in ID_COLS
        and not c.startswith(LABEL_PREFIXES)
    ]
    data = pd.concat(frames, ignore_index=True, sort=False)
    exact_col = f'label_switch_p{args.persistence}'
    dist_col = f'distance_to_switch_p{args.persistence}'
    pool = data[data['overlap_max_ioa'] >= args.candidate_ioa].copy().reset_index(drop=True)
    pool['target'] = (pool[dist_col] <= args.label_window).astype(int)

    percentile_features = []
    for feature in raw_features:
        name = f'pct__{feature}'
        pool[name] = pool.groupby('seq')[feature].rank(pct=True, method='average')
        percentile_features.append(name)

    variants = {
        'raw': raw_features,
        'raw_plus_percentile': raw_features + percentile_features,
    }
    fold_rows = []
    budget_rows = []

    for variant, features in variants.items():
        for fold_index, held_out in enumerate(seqs, 1):
            train = pool[pool['seq'] != held_out].copy()
            test = pool[pool['seq'] == held_out].copy()
            ytr = train['target'].to_numpy(int)
            yte = test['target'].to_numpy(int)

            imputer = SimpleImputer(strategy='median')
            xtr = imputer.fit_transform(train[features])
            xte = imputer.transform(test[features])
            weights = balanced_weights(train, ytr)
            model = HistGradientBoostingClassifier(
                max_iter=250,
                learning_rate=.05,
                max_leaf_nodes=15,
                min_samples_leaf=20,
                l2_regularization=4.0,
                random_state=3000 + fold_index,
            )
            model.fit(xtr, ytr, sample_weight=weights)
            pred = model.predict_proba(xte)[:, 1]
            score_col = f'loso_unary_{variant}'
            pool.loc[test.index, score_col] = pred

            exact_events = data[(data['seq'] == held_out) & (data[exact_col] == 1)][['track_id','boundary_frame']].drop_duplicates()
            test_scored = pool.loc[test.index].copy()
            selected = nms_events(test_scored, score_col, args.nms_radius)
            fold_rows.append({
                'variant': variant,
                'held_out_seq': held_out,
                'train_rows': len(train),
                'test_rows': len(test),
                'train_positive': int(ytr.sum()),
                'test_positive': int(yte.sum()),
                'prevalence': float(yte.mean()),
                'ap': float(average_precision_score(yte, pred)),
                'auc': safe_auc(yte, pred),
                'spearman_to_distance': float(spearmanr(pred, -np.minimum(test_scored[dist_col], 1000)).statistic),
                'exact_events': len(exact_events),
                'candidate_exact_event_recall': float(
                    len(test_scored.merge(exact_events, on=['track_id','boundary_frame'])) / max(1, len(exact_events))
                ),
            })
            for row in event_metrics(test_scored, selected, exact_events, score_col, args.label_window):
                budget_rows.append({'variant': variant, 'held_out_seq': held_out, **row})

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    score_cols = [c for c in pool.columns if c.startswith('loso_unary_')]
    output_cols = ['seq','track_id','boundary_frame','prev_frame', *raw_features, exact_col, dist_col, 'target', *score_cols]
    pool[output_cols].to_csv(out / 'loso_unary_change_scores.csv', index=False)
    pd.DataFrame(fold_rows).to_csv(out / 'fold_metrics.csv', index=False)
    pd.DataFrame(budget_rows).to_csv(out / 'event_budget_metrics.csv', index=False)

    fold_df = pd.DataFrame(fold_rows)
    aggregate = []
    for variant, group in fold_df.groupby('variant'):
        aggregate.append({
            'variant': variant,
            'mean_seq_ap': float(group['ap'].mean()),
            'min_seq_ap': float(group['ap'].min()),
            'mean_seq_auc': float(group['auc'].mean()),
            'mean_seq_spearman_to_distance': float(group['spearman_to_distance'].mean()),
        })
    report = {
        'protocol': {
            'scope': 'End-to-end unary leave-one-sequence-out scoring on observable change-point features',
            'held_out_sequences': seqs,
            'labels': f'persistent GT switch p{args.persistence} within +/-{args.label_window} frames',
            'candidate_rule': f'overlap_max_ioa >= {args.candidate_ioa}',
            'validation': 'Train on three sequences, score the fourth; held-out GT is never used for model fitting.',
            'sequence_balancing': 'Equal total training weight per source sequence, then positive/negative class balancing.',
            'raw_feature_count': len(raw_features),
            'raw_features': raw_features,
            'variants': {name: len(features) for name, features in variants.items()},
        },
        'folds': fold_rows,
        'aggregate': aggregate,
    }
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
