from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score

ID_COLS = {
    'seq', 'track_id', 'track_a', 'track_b', 'boundary_frame', 'prev_frame',
    'proposal_rank', 'partner_rank', 'partner_frame_delta', 'label_pair_class',
}
FORBIDDEN_PREFIXES = ('label_', 'a_gt_', 'b_gt_', 'distance_to_switch_', 'oof_pair_', 'loso_unary_')
FORBIDDEN_EXACT = {
    'target', 'oof_et', 'oof_hgb', 'oof_ensemble',
    'debt_score', 'debt_rank', 'debt_pct',
}
TARGETS = {
    'reciprocal': 'label_reciprocal_swap',
    'related': 'label_pair_related',
}


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float('nan')


def balanced_weights(frame: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    counts = frame['seq'].value_counts()
    seq_weight = frame['seq'].map({seq: len(frame) / (len(counts) * n) for seq, n in counts.items()}).to_numpy(float)
    positives = int(y.sum())
    class_weight = max(1.0, (len(y) - positives) / max(1, positives))
    weights = seq_weight * np.where(y == 1, class_weight, 1.0)
    return weights / max(float(weights.mean()), 1e-12)


def event_budget_rows(frame: pd.DataFrame, score: str, target: str, budgets=(25, 50, 100, 200, 500)):
    best = frame.sort_values(score, ascending=False).groupby(
        ['track_a', 'boundary_frame'], as_index=False, sort=False
    ).head(1).sort_values(score, ascending=False)
    true_events = int(frame[frame[target] == 1][['track_a', 'boundary_frame']].drop_duplicates().shape[0])
    rows = []
    for budget in budgets:
        selected = best.head(budget)
        hits = int(selected[target].sum())
        rows.append({
            'budget': budget,
            'selected': len(selected),
            'true_events': true_events,
            'hits': hits,
            'precision': hits / max(1, len(selected)),
            'recall_in_bank': hits / max(1, true_events),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pair-bank', action='append', required=True, help='Repeat for each sequence bank')
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    frames = [pd.read_csv(path) for path in args.pair_bank]
    sequence_names = [str(frame['seq'].iloc[0]) for frame in frames]
    if len(sequence_names) != len(set(sequence_names)):
        raise RuntimeError('duplicate sequence banks')

    common = set(frames[0].columns)
    for frame in frames[1:]:
        common &= set(frame.columns)
    numeric_common = [
        c for c in sorted(common)
        if all(pd.api.types.is_numeric_dtype(frame[c]) for frame in frames)
    ]
    raw_features = [
        c for c in numeric_common
        if c not in ID_COLS | FORBIDDEN_EXACT
        and not c.startswith(FORBIDDEN_PREFIXES)
    ]
    if not raw_features:
        raise RuntimeError('no common observable features')

    data = pd.concat(frames, ignore_index=True, sort=False)
    data['pair_swap_margin_filled'] = data['pair_swap_margin'].fillna(-10.0)
    data['pair_related_similarity'] = data[['old_handoff_score', 'new_source_score']].max(axis=1).fillna(-10.0)
    if 'loso_unary_raw_plus_percentile' in data.columns:
        unary_baseline_col = 'loso_unary_raw_plus_percentile'
    elif 'oof_hgb' in data.columns:
        unary_baseline_col = 'oof_hgb'
    else:
        unary_baseline_col = None
    data['diagnostic_unary_score'] = data[unary_baseline_col] if unary_baseline_col else -data['proposal_rank']

    percentile_features = []
    for feature in raw_features:
        name = f'pct__{feature}'
        data[name] = data.groupby('seq')[feature].rank(pct=True, method='average')
        percentile_features.append(name)

    variants = {
        'raw': raw_features,
        'raw_plus_percentile': raw_features + percentile_features,
    }
    fold_rows = []
    budget_rows = []
    all_importance = []

    for target_name, target_col in TARGETS.items():
        for variant_name, features in variants.items():
            for fold_index, held_out in enumerate(sequence_names, 1):
                train = data[data['seq'] != held_out].copy()
                test = data[data['seq'] == held_out].copy()
                ytr = train[target_col].to_numpy(int)
                yte = test[target_col].to_numpy(int)
                if ytr.sum() == 0 or ytr.sum() == len(ytr):
                    raise RuntimeError(f'one-class training data for {target_name}, held out {held_out}')

                imputer = SimpleImputer(strategy='median')
                xtr = imputer.fit_transform(train[features])
                xte = imputer.transform(test[features])
                weights = balanced_weights(train, ytr)

                et = ExtraTreesClassifier(
                    n_estimators=500,
                    min_samples_leaf=3,
                    max_features=0.75,
                    n_jobs=-1,
                    random_state=1100 + fold_index,
                )
                et.fit(xtr, ytr, sample_weight=weights)
                pet = et.predict_proba(xte)[:, 1]

                hgb = HistGradientBoostingClassifier(
                    max_iter=300,
                    learning_rate=0.04,
                    max_leaf_nodes=15,
                    min_samples_leaf=12,
                    l2_regularization=4.0,
                    random_state=2100 + fold_index,
                )
                hgb.fit(xtr, ytr, sample_weight=weights)
                phgb = hgb.predict_proba(xte)[:, 1]
                ensemble = 0.5 * pet + 0.5 * phgb

                prefix = f'loso_{target_name}_{variant_name}'
                data.loc[test.index, f'{prefix}_et'] = pet
                data.loc[test.index, f'{prefix}_hgb'] = phgb
                data.loc[test.index, f'{prefix}_ensemble'] = ensemble

                baseline_col = 'pair_swap_margin_filled' if target_name == 'reciprocal' else 'pair_related_similarity'
                metrics = {
                    'target': target_name,
                    'variant': variant_name,
                    'held_out_seq': held_out,
                    'train_rows': len(train),
                    'test_rows': len(test),
                    'train_positive': int(ytr.sum()),
                    'test_positive': int(yte.sum()),
                    'prevalence': float(yte.mean()),
                    'diagnostic_unary_ap': float(average_precision_score(yte, test['diagnostic_unary_score'])),
                    'pair_heuristic_ap': float(average_precision_score(yte, test[baseline_col])),
                    'et_ap': float(average_precision_score(yte, pet)),
                    'et_auc': safe_auc(yte, pet),
                    'hgb_ap': float(average_precision_score(yte, phgb)),
                    'hgb_auc': safe_auc(yte, phgb),
                    'ensemble_ap': float(average_precision_score(yte, ensemble)),
                    'ensemble_auc': safe_auc(yte, ensemble),
                }
                fold_rows.append(metrics)
                scored = data.loc[test.index].copy()
                score_col = f'{prefix}_ensemble'
                for row in event_budget_rows(scored, score_col, target_col):
                    budget_rows.append({
                        'target': target_name,
                        'variant': variant_name,
                        'held_out_seq': held_out,
                        'score': score_col,
                        **row,
                    })
                all_importance.extend({
                    'target': target_name,
                    'variant': variant_name,
                    'held_out_seq': held_out,
                    'feature': feature,
                    'importance': float(value),
                } for feature, value in zip(features, et.feature_importances_))

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    score_cols = [c for c in data.columns if c.startswith('loso_')]
    keep_cols = [
        'seq', 'track_a', 'track_b', 'boundary_frame', 'proposal_rank', 'partner_rank',
        'label_reciprocal_swap', 'label_pair_related', 'pair_swap_margin_filled',
        'pair_related_similarity', 'diagnostic_unary_score', *score_cols,
    ]
    data[keep_cols].to_csv(out / 'loso_pair_state_scores.csv', index=False)
    pd.DataFrame(fold_rows).to_csv(out / 'fold_metrics.csv', index=False)
    pd.DataFrame(budget_rows).to_csv(out / 'event_budget_metrics.csv', index=False)
    importance = pd.DataFrame(all_importance)
    importance.to_csv(out / 'feature_importance.csv', index=False)

    fold_df = pd.DataFrame(fold_rows)
    aggregate = []
    for (target, variant), group in fold_df.groupby(['target', 'variant']):
        aggregate.append({
            'target': target,
            'variant': variant,
            'mean_seq_ensemble_ap': float(group['ensemble_ap'].mean()),
            'min_seq_ensemble_ap': float(group['ensemble_ap'].min()),
            'mean_seq_ensemble_auc': float(group['ensemble_auc'].mean()),
            'mean_seq_pair_heuristic_ap': float(group['pair_heuristic_ap'].mean()),
            'mean_seq_diagnostic_unary_ap': float(group['diagnostic_unary_ap'].mean()),
        })
    report = {
        'protocol': {
            'scope': 'Pair-stage outer LOSO evaluated on held-out candidate banks produced by unary models that exclude that held-out sequence',
            'held_out_sequences': sequence_names,
            'candidate_warning': ('Held-out candidate scoring is outer-test-clean. However, each source-sequence training bank was generated by its own leave-one-sequence-out unary model, which may include the current outer-held-out sequence when constructing other training banks. Fully nested candidate generation is still required for a leakage-free end-to-end estimate.'),
            'feature_policy': 'GT-derived columns, IDs, exact ranks, all unary learned-score columns, and debt columns are excluded from pair-model features.',
            'reported_unary_baseline': unary_baseline_col,
            'sequence_balancing': 'Equal total training weight per source sequence, then positive/negative class balancing.',
            'raw_feature_count': len(raw_features),
            'raw_features': raw_features,
            'variants': {name: len(features) for name, features in variants.items()},
        },
        'folds': fold_rows,
        'aggregate': aggregate,
        'top_features': importance.sort_values('importance', ascending=False).groupby(
            ['target', 'variant', 'held_out_seq']
        ).head(15).to_dict(orient='records'),
    }
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'protocol': report['protocol'], 'aggregate': aggregate}, indent=2))


if __name__ == '__main__':
    main()
