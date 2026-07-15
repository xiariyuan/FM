from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score

KEYS = ['seq', 'canonical_rank', 'u', 'v', 'boundary_frame']

BASE_FEATURES = [
    # Outer-clean proposal and pair-state confidence.
    'outer_clean_transaction_score',
    'outer_clean_reciprocal_raw_plus_percentile_ensemble',
    'outer_clean_related_raw_plus_percentile_ensemble',
    'heldout_unary_score',
    'cluster_size',
    'appearance_change',
    'appearance_margin',
    'boundary_position_ratio',
    'overlap_max_ioa',
    'overlap_partner_hits',
    'prediction_error_norm',
    'proto_cos',
    'velocity_cos',
    'partner_ioa',
    'pair_swap_margin',
    'old_handoff_score',
    'new_source_score',
    'center_dist_norm',
    'bottom_dist_norm',
    'sim_a_left_a_right',
    'sim_b_left_b_right',
    'sim_a_left_b_right',
    'sim_b_left_a_right',
    # Track survival and future pair interaction.
    'age_min',
    'age_max',
    'pair_min_future_life',
    'pair_max_future_life',
    'pair_end_gap_abs',
    'pre_rows_min',
    'pre_match_iou_min',
    'future_overlap_total_count',
    'future_overlap_last_offset',
    'future_overlap_max_ioa',
    'future_overlap_mean_ioa',
    'copresent_frac_h30',
    'copresent_frac_h60',
    'copresent_frac_h120',
    'copresent_frac_h300',
    'reid_coverage_min_h30',
    'reid_coverage_min_h60',
    'reid_coverage_min_h120',
    'reid_coverage_min_h300',
    'match_iou_min_h30',
    'match_iou_min_h60',
    'match_iou_min_h120',
    'match_iou_min_h300',
    'swap_margin_h30',
    'swap_margin_h60',
    'swap_margin_h120',
    'swap_margin_h300',
    'overlap_count_h30',
    'overlap_count_h60',
    'overlap_count_h120',
    'overlap_count_h300',
    'overlap_max_h30',
    'overlap_max_h60',
    'overlap_max_h120',
    'overlap_max_h300',
    'chunk0_swap_margin',
    'chunk1_swap_margin',
    'chunk2_swap_margin',
    'chunk3_swap_margin',
    'chunk0_valid_both',
    'chunk1_valid_both',
    'chunk2_valid_both',
    'chunk3_valid_both',
    'swap_margin_future_mean',
    'swap_margin_future_min',
    'swap_margin_future_max',
    'swap_margin_positive_horizons',
    'swap_margin_non_decreasing',
]


def safe_auc(y: np.ndarray, p: np.ndarray) -> float | None:
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, p))


def safe_spearman(y: np.ndarray, p: np.ndarray) -> float | None:
    value = spearmanr(y, p, nan_policy='omit').statistic
    return float(value) if np.isfinite(value) else None


def rank_pct(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method='average', pct=True).to_numpy(float)


def add_symmetric_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame['age_min'] = frame[['u_age_at_event', 'v_age_at_event']].min(axis=1)
    frame['age_max'] = frame[['u_age_at_event', 'v_age_at_event']].max(axis=1)
    frame['pre_rows_min'] = frame[['u_pre_rows', 'v_pre_rows']].min(axis=1)
    frame['pre_match_iou_min'] = frame[['u_pre_match_iou', 'v_pre_match_iou']].min(axis=1)
    for horizon in [30, 60, 120, 300]:
        frame[f'reid_coverage_min_h{horizon}'] = frame[[
            f'u_reid_coverage_h{horizon}', f'v_reid_coverage_h{horizon}'
        ]].min(axis=1)
        frame[f'match_iou_min_h{horizon}'] = frame[[
            f'u_match_iou_h{horizon}', f'v_match_iou_h{horizon}'
        ]].min(axis=1)
    return frame


def sequence_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame['seq'].value_counts()
    weights = frame['seq'].map({seq: len(frame) / (len(counts) * count) for seq, count in counts.items()})
    return weights.to_numpy(float)


def normalized_target(frame: pd.DataFrame) -> np.ndarray:
    values = np.zeros(len(frame), dtype=float)
    for _, indices in frame.groupby('seq').groups.items():
        y = frame.loc[indices, 'delta_HOTA'].to_numpy(float)
        scale = max(float(np.mean(np.abs(y))), 0.02)
        values[frame.index.get_indexer(indices)] = y / scale
    return values


def topk_metrics(frame: pd.DataFrame, score_col: str, prefix: str) -> dict:
    ranked = frame.sort_values(score_col, ascending=False)
    result: dict[str, float | int] = {}
    for k in [3, 5, 10]:
        selected = ranked.head(k)
        result[f'{prefix}_top{k}_positive_precision'] = float((selected.delta_HOTA > 0).mean())
        result[f'{prefix}_top{k}_utility_sum'] = float(selected.delta_HOTA.sum())
        result[f'{prefix}_top{k}_utility_mean'] = float(selected.delta_HOTA.mean())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', action='append', required=True)
    parser.add_argument('--utility-csv', required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()

    features = pd.concat([pd.read_csv(path) for path in args.features], ignore_index=True, sort=False)
    features = add_symmetric_features(features)
    utility = pd.read_csv(args.utility_csv)
    label_columns = KEYS + [
        'accepted', 'rejected', 'delta_HOTA', 'delta_DetA', 'delta_AssA',
        'delta_IDF1', 'delta_IDSW',
    ]
    data = features.merge(utility[label_columns], on=KEYS, how='inner', validate='one_to_one')
    if len(data) != len(features) or len(data) != len(utility):
        raise RuntimeError(f'expected exact 1:1 join, got features={len(features)}, utility={len(utility)}, joined={len(data)}')

    missing = [feature for feature in BASE_FEATURES if feature not in data.columns]
    if missing:
        raise RuntimeError(f'missing compact features: {missing}')

    for feature in BASE_FEATURES:
        data[f'pct__{feature}'] = data.groupby('seq')[feature].rank(pct=True, method='average')
    variants = {
        'compact': BASE_FEATURES,
        'compact_plus_percentile': BASE_FEATURES + [f'pct__{feature}' for feature in BASE_FEATURES],
    }

    sequences = sorted(data.seq.unique())
    fold_rows = []
    all_predictions = []
    importance_rows = []

    for held_out in sequences:
        train = data[data.seq != held_out].copy().reset_index(drop=True)
        test = data[data.seq == held_out].copy().reset_index(drop=True)
        y_train_raw = train.delta_HOTA.to_numpy(float)
        y_train = normalized_target(train)
        y_test = test.delta_HOTA.to_numpy(float)
        positive_train = (y_train_raw > 0).astype(int)
        positive_test = (y_test > 0).astype(int)
        weights = sequence_weights(train)

        for variant, feature_names in variants.items():
            imputer = SimpleImputer(strategy='median')
            x_train = imputer.fit_transform(train[feature_names])
            x_test = imputer.transform(test[feature_names])

            et = ExtraTreesRegressor(
                n_estimators=1200,
                max_depth=6,
                min_samples_leaf=3,
                max_features=0.65,
                random_state=6100 + sequences.index(held_out),
                n_jobs=-1,
            )
            hgb = HistGradientBoostingRegressor(
                max_iter=250,
                learning_rate=0.035,
                max_leaf_nodes=7,
                min_samples_leaf=8,
                l2_regularization=5.0,
                random_state=7100 + sequences.index(held_out),
            )
            classifier = ExtraTreesClassifier(
                n_estimators=1200,
                max_depth=6,
                min_samples_leaf=3,
                max_features=0.65,
                class_weight='balanced',
                random_state=8100 + sequences.index(held_out),
                n_jobs=-1,
            )
            et.fit(x_train, y_train, sample_weight=weights)
            hgb.fit(x_train, y_train, sample_weight=weights)
            classifier.fit(x_train, positive_train, sample_weight=weights)

            pred_et = et.predict(x_test)
            pred_hgb = hgb.predict(x_test)
            prob_positive = classifier.predict_proba(x_test)[:, 1]
            utility_score = 0.5 * pred_et + 0.5 * pred_hgb
            rank_score = (
                0.4 * rank_pct(pred_et)
                + 0.4 * rank_pct(pred_hgb)
                + 0.2 * rank_pct(prob_positive)
            )

            scored = test[KEYS + ['delta_HOTA', 'accepted', 'outer_clean_transaction_score']].copy()
            scored['variant'] = variant
            scored['held_out_seq'] = held_out
            scored['pred_utility_et'] = pred_et
            scored['pred_utility_hgb'] = pred_hgb
            scored['pred_positive_prob'] = prob_positive
            scored['pred_utility_ensemble'] = utility_score
            scored['pred_transaction_rank'] = rank_score
            all_predictions.append(scored)

            metrics = {
                'variant': variant,
                'held_out_seq': held_out,
                'train_rows': len(train),
                'test_rows': len(test),
                'train_positive': int(positive_train.sum()),
                'test_positive': int(positive_test.sum()),
                'positive_prevalence': float(positive_test.mean()),
                'et_spearman': safe_spearman(y_test, pred_et),
                'hgb_spearman': safe_spearman(y_test, pred_hgb),
                'ensemble_utility_spearman': safe_spearman(y_test, utility_score),
                'ensemble_rank_spearman': safe_spearman(y_test, rank_score),
                'positive_ap': float(average_precision_score(positive_test, prob_positive)),
                'positive_auc': safe_auc(positive_test, prob_positive),
                'baseline_pair_score_spearman': safe_spearman(y_test, test.outer_clean_transaction_score.to_numpy(float)),
            }
            metrics.update(topk_metrics(scored, 'pred_transaction_rank', 'model'))
            metrics.update(topk_metrics(scored, 'outer_clean_transaction_score', 'pair_baseline'))
            fold_rows.append(metrics)
            importance_rows.extend({
                'variant': variant,
                'held_out_seq': held_out,
                'feature': feature,
                'importance': float(value),
            } for feature, value in zip(feature_names, et.feature_importances_))

    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions = pd.concat(all_predictions, ignore_index=True, sort=False)
    predictions.to_csv(output / 'loso_utility_scores.csv', index=False)
    fold_frame = pd.DataFrame(fold_rows)
    fold_frame.to_csv(output / 'fold_metrics.csv', index=False)
    importance = pd.DataFrame(importance_rows)
    importance.to_csv(output / 'feature_importance.csv', index=False)

    aggregate = []
    for variant, group in fold_frame.groupby('variant'):
        row = {
            'variant': variant,
            'mean_seq_ensemble_rank_spearman': float(group.ensemble_rank_spearman.mean()),
            'min_seq_ensemble_rank_spearman': float(group.ensemble_rank_spearman.min()),
            'mean_seq_positive_ap': float(group.positive_ap.mean()),
            'mean_seq_positive_auc': float(group.positive_auc.dropna().mean()),
            'mean_seq_pair_baseline_spearman': float(group.baseline_pair_score_spearman.mean()),
        }
        for k in [3, 5, 10]:
            row[f'mean_seq_model_top{k}_positive_precision'] = float(group[f'model_top{k}_positive_precision'].mean())
            row[f'total_model_top{k}_utility_sum'] = float(group[f'model_top{k}_utility_sum'].sum())
            row[f'mean_seq_pair_top{k}_positive_precision'] = float(group[f'pair_baseline_top{k}_positive_precision'].mean())
            row[f'total_pair_top{k}_utility_sum'] = float(group[f'pair_baseline_top{k}_utility_sum'].sum())
        aggregate.append(row)

    report = {
        'protocol': {
            'scope': 'Four-sequence future-utility leave-one-sequence-out diagnostic on pre-registered Top-20 outer-clean candidates',
            'validation': 'For each held-out sequence, utility labels from that sequence are excluded from all model fitting and feature preprocessing.',
            'candidate_policy': 'Candidate lists and order were fixed before utility labeling using outer-clean unary/pair models.',
            'future_policy': 'Future tracker/ReID/overlap features use offline future video only; no GT-derived feature is included.',
            'target': 'TrackEval single-event permanent-transaction delta_HOTA, sequence-scale normalized on training sequences only.',
            'primary_rank': '0.4 rank(ET utility) + 0.4 rank(HGB utility) + 0.2 rank(ET positive probability), ranked within held-out sequence.',
            'sample_size_warning': 'Only 20 labeled candidates per sequence; this is a minimal strict utility pilot, not a final high-powered estimate.',
            'feature_variants': {name: len(values) for name, values in variants.items()},
            'compact_features': BASE_FEATURES,
        },
        'dataset': {
            'events': len(data),
            'sequences': sequences,
            'positive_events': int((data.delta_HOTA > 0).sum()),
            'negative_events': int((data.delta_HOTA < 0).sum()),
            'zero_events': int((data.delta_HOTA == 0).sum()),
        },
        'folds': fold_rows,
        'aggregate': aggregate,
        'top_features': importance.sort_values('importance', ascending=False).groupby(
            ['variant', 'held_out_seq']
        ).head(20).to_dict(orient='records'),
    }
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'dataset': report['dataset'], 'aggregate': aggregate}, indent=2), flush=True)


if __name__ == '__main__':
    main()
