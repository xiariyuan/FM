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
FORBIDDEN_PREFIXES = (
    'label_', 'a_gt_', 'b_gt_', 'distance_to_switch_',
    'oof_pair_', 'loso_unary_',
)
FORBIDDEN_EXACT = {
    'target', 'oof_et', 'oof_hgb', 'oof_ensemble',
    'debt_score', 'debt_rank', 'debt_pct',
}
TARGETS = {
    'reciprocal': 'label_reciprocal_swap',
    'related': 'label_pair_related',
}


def parse_mapping(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if '=' not in value:
            raise ValueError(f'expected SEQ=PATH, got {value}')
        seq, path = value.split('=', 1)
        if seq in result:
            raise ValueError(f'duplicate sequence {seq}')
        result[seq] = Path(path)
    return result


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


def add_percentiles(frame: pd.DataFrame, raw_features: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    for feature in raw_features:
        frame[f'pct__{feature}'] = frame.groupby('seq')[feature].rank(pct=True, method='average')
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-bank', action='append', required=True, help='Repeat SEQ=PATH for same-sequence OOF training banks')
    ap.add_argument('--test-bank', action='append', required=True, help='Repeat SEQ=PATH for unary-LOSO held-out banks')
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    train_paths = parse_mapping(args.train_bank)
    test_paths = parse_mapping(args.test_bank)
    if set(train_paths) != set(test_paths):
        raise RuntimeError('train/test sequence sets differ')
    seqs = sorted(train_paths)

    train_frames = {seq: pd.read_csv(path) for seq, path in train_paths.items()}
    test_frames = {seq: pd.read_csv(path) for seq, path in test_paths.items()}
    for seq in seqs:
        if set(train_frames[seq]['seq'].astype(str).unique()) != {seq}:
            raise RuntimeError(f'train bank seq mismatch for {seq}')
        if set(test_frames[seq]['seq'].astype(str).unique()) != {seq}:
            raise RuntimeError(f'test bank seq mismatch for {seq}')

    common = set.intersection(*[set(frame.columns) for frame in list(train_frames.values()) + list(test_frames.values())])
    numeric_common = [
        c for c in sorted(common)
        if all(pd.api.types.is_numeric_dtype(frame[c]) for frame in list(train_frames.values()) + list(test_frames.values()))
    ]
    raw_features = [
        c for c in numeric_common
        if c not in ID_COLS | FORBIDDEN_EXACT
        and not c.startswith(FORBIDDEN_PREFIXES)
    ]
    if not raw_features:
        raise RuntimeError('no common observable features')

    train_frames = {seq: add_percentiles(frame, raw_features) for seq, frame in train_frames.items()}
    test_frames = {seq: add_percentiles(frame, raw_features) for seq, frame in test_frames.items()}
    percentile_features = [f'pct__{f}' for f in raw_features]
    variants = {
        'raw': raw_features,
        'raw_plus_percentile': raw_features + percentile_features,
    }

    fold_rows = []
    budget_rows = []
    score_outputs = []
    importance_rows = []

    for held_out in seqs:
        train = pd.concat([train_frames[seq] for seq in seqs if seq != held_out], ignore_index=True, sort=False)
        test = test_frames[held_out].copy().reset_index(drop=True)
        test['pair_swap_margin_filled'] = test['pair_swap_margin'].fillna(-10.0)
        test['pair_related_similarity'] = test[['old_handoff_score', 'new_source_score']].max(axis=1).fillna(-10.0)
        if 'loso_unary_raw_plus_percentile' in test.columns:
            test['heldout_unary_score'] = test['loso_unary_raw_plus_percentile']
        else:
            test['heldout_unary_score'] = -test['proposal_rank']

        for target_name, target_col in TARGETS.items():
            for variant_name, features in variants.items():
                ytr = train[target_col].to_numpy(int)
                yte = test[target_col].to_numpy(int)
                imputer = SimpleImputer(strategy='median')
                xtr = imputer.fit_transform(train[features])
                xte = imputer.transform(test[features])
                weights = balanced_weights(train, ytr)

                et = ExtraTreesClassifier(
                    n_estimators=500,
                    min_samples_leaf=3,
                    max_features=0.75,
                    n_jobs=-1,
                    random_state=4100 + seqs.index(held_out),
                )
                et.fit(xtr, ytr, sample_weight=weights)
                pet = et.predict_proba(xte)[:, 1]

                hgb = HistGradientBoostingClassifier(
                    max_iter=300,
                    learning_rate=0.04,
                    max_leaf_nodes=15,
                    min_samples_leaf=12,
                    l2_regularization=4.0,
                    random_state=5100 + seqs.index(held_out),
                )
                hgb.fit(xtr, ytr, sample_weight=weights)
                phgb = hgb.predict_proba(xte)[:, 1]
                ensemble = 0.5 * pet + 0.5 * phgb

                score_col = f'outer_clean_{target_name}_{variant_name}_ensemble'
                test[score_col] = ensemble
                baseline_col = 'pair_swap_margin_filled' if target_name == 'reciprocal' else 'pair_related_similarity'
                fold_rows.append({
                    'target': target_name,
                    'variant': variant_name,
                    'held_out_seq': held_out,
                    'train_rows': len(train),
                    'test_rows': len(test),
                    'train_positive': int(ytr.sum()),
                    'test_positive': int(yte.sum()),
                    'prevalence': float(yte.mean()),
                    'heldout_unary_ap': float(average_precision_score(yte, test['heldout_unary_score'])),
                    'pair_heuristic_ap': float(average_precision_score(yte, test[baseline_col])),
                    'et_ap': float(average_precision_score(yte, pet)),
                    'et_auc': safe_auc(yte, pet),
                    'hgb_ap': float(average_precision_score(yte, phgb)),
                    'hgb_auc': safe_auc(yte, phgb),
                    'ensemble_ap': float(average_precision_score(yte, ensemble)),
                    'ensemble_auc': safe_auc(yte, ensemble),
                })
                for row in event_budget_rows(test, score_col, target_col):
                    budget_rows.append({
                        'target': target_name,
                        'variant': variant_name,
                        'held_out_seq': held_out,
                        'score': score_col,
                        **row,
                    })
                importance_rows.extend({
                    'target': target_name,
                    'variant': variant_name,
                    'held_out_seq': held_out,
                    'feature': feature,
                    'importance': float(value),
                } for feature, value in zip(features, et.feature_importances_))

        keep_cols = [
            'seq', 'track_a', 'track_b', 'boundary_frame', 'proposal_rank', 'partner_rank',
            'label_reciprocal_swap', 'label_pair_related', 'heldout_unary_score',
            'pair_swap_margin_filled', 'pair_related_similarity',
        ] + [c for c in test.columns if c.startswith('outer_clean_')]
        score_outputs.append(test[keep_cols])

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    scores = pd.concat(score_outputs, ignore_index=True, sort=False)
    scores.to_csv(out / 'outer_clean_pair_scores.csv', index=False)
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(out / 'fold_metrics.csv', index=False)
    pd.DataFrame(budget_rows).to_csv(out / 'event_budget_metrics.csv', index=False)
    importance = pd.DataFrame(importance_rows)
    importance.to_csv(out / 'feature_importance.csv', index=False)

    aggregate = []
    for (target, variant), group in fold_df.groupby(['target', 'variant']):
        aggregate.append({
            'target': target,
            'variant': variant,
            'mean_seq_ensemble_ap': float(group['ensemble_ap'].mean()),
            'min_seq_ensemble_ap': float(group['ensemble_ap'].min()),
            'mean_seq_ensemble_auc': float(group['ensemble_auc'].mean()),
            'mean_seq_pair_heuristic_ap': float(group['pair_heuristic_ap'].mean()),
            'mean_seq_heldout_unary_ap': float(group['heldout_unary_ap'].mean()),
        })
    report = {
        'protocol': {
            'scope': 'Outer-clean pair-stage LOSO',
            'held_out_sequences': seqs,
            'training_candidate_banks': 'Each training sequence uses its own same-sequence grouped-OOF unary candidate bank; these banks do not use held-out sequence labels.',
            'heldout_candidate_banks': 'Held-out sequence uses unary scores trained only on the other three sequences.',
            'feature_policy': 'GT-derived columns, IDs, exact ranks, all unary learned-score columns, and debt columns are excluded from pair-model features.',
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
