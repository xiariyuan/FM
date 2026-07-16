from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score


KEYS = ['seq', 'canonical_rank', 'u', 'v', 'boundary_frame', 'transaction_type']
TARGET = 'full_idtp_delta_norm'
P15_WINDOWS = [(21, 40), (41, 60), (61, 80), (81, 100)]
TOP_K = [5, 10, 20, 40, 80]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def rounded(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.select_dtypes(include=[np.number]).columns:
        result[column] = result[column].round(12)
    return result


def load_frame(
    feature_path: str,
    label_path: str,
    features: list[str],
    include_hota: bool,
) -> pd.DataFrame:
    feature_frame = pd.read_csv(feature_path)
    label_columns = KEYS + [TARGET, 'effective_start_frame']
    if include_hota:
        label_columns.append('delta_HOTA')
    labels = pd.read_csv(label_path, usecols=label_columns)
    frame = feature_frame.merge(labels, on=KEYS, how='inner', validate='one_to_one')
    missing = [feature for feature in features if feature not in frame.columns]
    if missing:
        raise RuntimeError(f'missing compact features: {missing}')
    if include_hota:
        frame['domain'] = 'P15_MOT20'
    else:
        frame['delta_HOTA'] = np.nan
        frame['domain'] = 'P17_MOT17'
    return frame


def sequence_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.seq.value_counts()
    mapping = {
        sequence: len(frame) / (len(counts) * count)
        for sequence, count in counts.items()
    }
    return frame.seq.map(mapping).to_numpy(float)


def fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    imputer = SimpleImputer(strategy='median')
    x_train = imputer.fit_transform(train[features])
    x_test = imputer.transform(test[features])
    weights = sequence_weights(train)
    regressor = ExtraTreesRegressor(
        n_estimators=500,
        max_depth=7,
        min_samples_leaf=3,
        max_features=0.65,
        random_state=seed,
        n_jobs=-1,
    )
    classifier = ExtraTreesClassifier(
        n_estimators=500,
        max_depth=7,
        min_samples_leaf=3,
        max_features=0.65,
        class_weight='balanced',
        random_state=seed + 36,
        n_jobs=-1,
    )
    regressor.fit(x_train, train[TARGET], sample_weight=weights)
    classifier.fit(
        x_train,
        (train[TARGET] > 0.0).astype(int),
        sample_weight=weights,
    )
    return regressor.predict(x_test), classifier.predict_proba(x_test)[:, 1]


def prediction_metrics(frame: pd.DataFrame, prefix: str) -> dict[str, object]:
    labels = (frame[TARGET] > 0.0).astype(int)
    return {
        'evaluation': prefix,
        'events': int(len(frame)),
        'positive_events': int(labels.sum()),
        'positive_fraction': float(labels.mean()),
        'spearman_regression': float(
            spearmanr(frame[TARGET], frame.pred_reg).statistic
        ),
        'auc_regression': float(roc_auc_score(labels, frame.pred_reg)),
        'auc_classifier': float(roc_auc_score(labels, frame.pred_clf)),
        'average_precision_regression': float(
            average_precision_score(labels, frame.pred_reg)
        ),
        'average_precision_classifier': float(
            average_precision_score(labels, frame.pred_clf)
        ),
    }


def topk_summary(
    frame: pd.DataFrame,
    evaluation: str,
) -> pd.DataFrame:
    rows = []
    for score in ['pred_reg', 'pred_clf']:
        for top_k in TOP_K:
            selected = frame.nlargest(min(top_k, len(frame)), score)
            rows.append(
                {
                    'evaluation': evaluation,
                    'score': score,
                    'top_k': top_k,
                    'selected': len(selected),
                    'positive': int((selected[TARGET] > 0.0).sum()),
                    'negative': int((selected[TARGET] < 0.0).sum()),
                    'zero': int((selected[TARGET] == 0.0).sum()),
                    'positive_precision': float((selected[TARGET] > 0.0).mean()),
                    'utility_sum': float(selected[TARGET].sum()),
                    'utility_minimum': float(selected[TARGET].min()),
                }
            )
    return pd.DataFrame(rows)


def temporal_block_oof(
    frame: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    ordered = frame.sort_values(
        ['effective_start_frame', 'canonical_rank']
    ).reset_index(drop=True)
    ordered['temporal_block'] = np.minimum(
        3, (np.arange(len(ordered)) * 4) // len(ordered)
    )
    outputs = []
    for block in range(4):
        train = ordered[ordered.temporal_block != block].copy()
        test = ordered[ordered.temporal_block == block].copy()
        pred_reg, pred_clf = fit_predict(train, test, features, 500 + block)
        test['pred_reg'] = pred_reg
        test['pred_clf'] = pred_clf
        outputs.append(test)
    return pd.concat(outputs, ignore_index=True).sort_values(
        ['temporal_block', 'effective_start_frame', 'canonical_rank']
    )


def temporal_topone_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for score in ['pred_reg', 'pred_clf']:
        for block, group in frame.groupby('temporal_block', sort=True):
            selected = group.nlargest(5, score)
            rows.append(
                {
                    'score': score,
                    'temporal_block': int(block),
                    'selected': len(selected),
                    'positive': int((selected[TARGET] > 0.0).sum()),
                    'negative': int((selected[TARGET] < 0.0).sum()),
                    'zero': int((selected[TARGET] == 0.0).sum()),
                    'positive_precision': float((selected[TARGET] > 0.0).mean()),
                    'utility_sum': float(selected[TARGET].sum()),
                    'utility_minimum': float(selected[TARGET].min()),
                }
            )
    return pd.DataFrame(rows)


def p15_loso(
    p15: pd.DataFrame,
    p17: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    outputs = []
    sequences = sorted(p15.seq.unique())
    for mode in ['p15_only', 'augmented_p17']:
        for fold, held_out in enumerate(sequences):
            train = p15[p15.seq != held_out].copy()
            if mode == 'augmented_p17':
                train = pd.concat([train, p17], ignore_index=True)
            test = p15[p15.seq == held_out].copy()
            pred_reg, pred_clf = fit_predict(
                train,
                test,
                features,
                1000 * fold + (100 if mode == 'augmented_p17' else 0),
            )
            test['pred_reg'] = pred_reg
            test['pred_clf'] = pred_clf
            test['training_mode'] = mode
            test['held_out_seq'] = held_out
            outputs.append(test)
    return pd.concat(outputs, ignore_index=True)


def p15_window_topone(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (mode, sequence), sequence_frame in predictions.groupby(
        ['training_mode', 'seq'], sort=True
    ):
        for start, end in P15_WINDOWS:
            window = sequence_frame[
                sequence_frame.canonical_rank.between(start, end)
            ]
            for score in ['pred_reg', 'pred_clf']:
                selected = window.sort_values(
                    [score, 'canonical_rank', 'transaction_type'],
                    ascending=[False, True, True],
                ).iloc[0]
                rows.append(
                    {
                        **{column: selected[column] for column in KEYS},
                        'training_mode': mode,
                        'score': score,
                        'rank_start': start,
                        'rank_end': end,
                        'predicted_score': float(selected[score]),
                        TARGET: float(selected[TARGET]),
                        'delta_HOTA': float(selected.delta_HOTA),
                    }
                )
    return pd.DataFrame(rows)


def p15_window_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (mode, score), group in frame.groupby(
        ['training_mode', 'score'], sort=True
    ):
        local_by_sequence = group.groupby('seq')[TARGET].sum()
        hota_by_sequence = group.groupby('seq').delta_HOTA.sum()
        rows.append(
            {
                'training_mode': mode,
                'score': score,
                'selected_windows': len(group),
                'local_positive': int((group[TARGET] > 0.0).sum()),
                'local_negative': int((group[TARGET] < 0.0).sum()),
                'local_zero': int((group[TARGET] == 0.0).sum()),
                'local_sum': float(group[TARGET].sum()),
                'local_worst_sequence': float(local_by_sequence.min()),
                'hota_positive': int((group.delta_HOTA > 0.0).sum()),
                'hota_negative': int((group.delta_HOTA < 0.0).sum()),
                'hota_zero': int((group.delta_HOTA == 0.0).sum()),
                'hota_sum': float(group.delta_HOTA.sum()),
                'hota_worst_sequence': float(hota_by_sequence.min()),
                'hota_catastrophic': int((group.delta_HOTA <= -0.05).sum()),
                **{
                    f'{sequence}_local': float(value)
                    for sequence, value in local_by_sequence.items()
                },
                **{
                    f'{sequence}_hota': float(value)
                    for sequence, value in hota_by_sequence.items()
                },
            }
        )
    return pd.DataFrame(rows)


def feature_shift(
    p15: pd.DataFrame,
    p17: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    rows = []
    for feature in features:
        left = p15[feature].dropna().to_numpy(float)
        right = p17[feature].dropna().to_numpy(float)
        pooled = np.concatenate([left, right])
        q25, q75 = np.quantile(pooled, [0.25, 0.75])
        scale = max((q75 - q25) / 1.349, 1e-6)
        standardized_shift = (np.median(right) - np.median(left)) / scale
        ks = ks_2samp(left, right)
        rows.append(
            {
                'feature': feature,
                'p15_median': float(np.median(left)),
                'p17_median': float(np.median(right)),
                'standardized_median_shift': float(standardized_shift),
                'absolute_standardized_shift': float(abs(standardized_shift)),
                'ks_statistic': float(ks.statistic),
                'ks_pvalue': float(ks.pvalue),
                'p15_missing_fraction': float(p15[feature].isna().mean()),
                'p17_missing_fraction': float(p17[feature].isna().mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ['absolute_standardized_shift', 'ks_statistic'], ascending=False
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--p15-motion', required=True)
    parser.add_argument('--p15-labels', required=True)
    parser.add_argument('--p17-motion', required=True)
    parser.add_argument('--p17-labels', required=True)
    parser.add_argument('--compact-features', required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()

    features = pd.read_csv(args.compact_features).feature.astype(str).tolist()
    p15 = load_frame(args.p15_motion, args.p15_labels, features, True)
    p17 = load_frame(args.p17_motion, args.p17_labels, features, False)

    p15_to_p17 = p17.copy()
    p15_to_p17['pred_reg'], p15_to_p17['pred_clf'] = fit_predict(
        p15, p17, features, 17
    )
    p17_to_p15 = p15.copy()
    p17_to_p15['pred_reg'], p17_to_p15['pred_clf'] = fit_predict(
        p17, p15, features, 1700
    )
    temporal_oof = temporal_block_oof(p17, features)
    p15_loso_predictions = p15_loso(p15, p17, features)
    p15_topone = p15_window_topone(p15_loso_predictions)
    p15_summary = p15_window_summary(p15_topone)
    shifts = feature_shift(p15, p17, features)

    metric_rows = [
        prediction_metrics(p15_to_p17, 'P15_to_P17_zero_shot'),
        prediction_metrics(p17_to_p15, 'P17_to_P15_zero_shot'),
        prediction_metrics(temporal_oof, 'P17_temporal_4block_OOF'),
    ]
    for sequence, group in p17_to_p15.groupby('seq', sort=True):
        metric_rows.append(
            prediction_metrics(group, f'P17_to_{sequence}_zero_shot')
        )
    metrics = pd.DataFrame(metric_rows)
    zero_shot_topk = topk_summary(p15_to_p17, 'P15_to_P17_zero_shot')
    temporal_top5 = temporal_topone_summary(temporal_oof)

    baseline_reg = p15_summary[
        (p15_summary.training_mode == 'p15_only')
        & (p15_summary.score == 'pred_reg')
    ].iloc[0]
    augmented_reg = p15_summary[
        (p15_summary.training_mode == 'augmented_p17')
        & (p15_summary.score == 'pred_reg')
    ].iloc[0]
    naive_pooling_allowed = bool(
        augmented_reg.local_sum > baseline_reg.local_sum
        and augmented_reg.local_worst_sequence >= baseline_reg.local_worst_sequence
        and augmented_reg.hota_worst_sequence >= baseline_reg.hota_worst_sequence
        and augmented_reg.hota_catastrophic <= baseline_reg.hota_catastrophic
    )
    p15_to_p17_metric = metrics[
        metrics.evaluation == 'P15_to_P17_zero_shot'
    ].iloc[0]
    temporal_metric = metrics[
        metrics.evaluation == 'P17_temporal_4block_OOF'
    ].iloc[0]
    p15_top20 = zero_shot_topk[
        (zero_shot_topk.score == 'pred_reg')
        & (zero_shot_topk.top_k == 20)
    ].iloc[0]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    rounded(metrics).to_csv(out_dir / 'cross_domain_metrics.csv', index=False)
    rounded(zero_shot_topk).to_csv(out_dir / 'p15_to_p17_topk.csv', index=False)
    rounded(temporal_top5).to_csv(out_dir / 'p17_temporal_top5.csv', index=False)
    rounded(shifts).to_csv(out_dir / 'compact_feature_shift.csv', index=False)
    rounded(p15_to_p17).to_csv(out_dir / 'p15_to_p17_predictions.csv', index=False)
    rounded(p17_to_p15).to_csv(out_dir / 'p17_to_p15_predictions.csv', index=False)
    rounded(temporal_oof).to_csv(out_dir / 'p17_temporal_oof_predictions.csv', index=False)
    rounded(p15_loso_predictions).to_csv(
        out_dir / 'p15_loso_predictions.csv', index=False
    )
    rounded(p15_topone).to_csv(out_dir / 'p15_window_topone.csv', index=False)
    rounded(p15_summary).to_csv(out_dir / 'p15_window_summary.csv', index=False)

    report = {
        'protocol': {
            'scope': 'Fixed-model cross-domain audit for the independent MOT17-09 directional local teacher bank.',
            'features': 'The 36 preregistered P16 compact actual-anchor motion features.',
            'model': 'ExtraTreesRegressor/Classifier with 500 trees, max_depth=7, min_samples_leaf=3, max_features=0.65. No hyperparameter or threshold sweep.',
            'weighting': 'Equal total sample weight per training sequence/domain label.',
            'p17_temporal_evaluation': 'Four deterministic equal-count temporal blocks ordered by effective handoff frame; each block is held out once.',
            'p15_incremental_evaluation': 'P15 sequence LOSO, comparing P15-only training against P15 plus all MOT17-09 events.',
            'p15_locked_labels_read': 0,
            'p15_locked_trackeval_calls': 0,
            'p15_remaining_locked_rows_untouched': 156,
        },
        'dataset': {
            'p15_events': len(p15),
            'p15_sequences': sorted(p15.seq.unique().tolist()),
            'p17_events': len(p17),
            'p17_sequences': sorted(p17.seq.unique().tolist()),
            'combined_events': len(p15) + len(p17),
            'compact_features': len(features),
        },
        'zero_shot_P15_to_P17': {
            'spearman': float(p15_to_p17_metric.spearman_regression),
            'auc_regression': float(p15_to_p17_metric.auc_regression),
            'average_precision_regression': float(
                p15_to_p17_metric.average_precision_regression
            ),
            'top20_positive': int(p15_top20.positive),
            'top20_utility_sum': float(p15_top20.utility_sum),
        },
        'p17_temporal_oof': {
            'spearman': float(temporal_metric.spearman_regression),
            'auc_regression': float(temporal_metric.auc_regression),
            'average_precision_regression': float(
                temporal_metric.average_precision_regression
            ),
            'regression_top5_each_block_utility': {
                str(int(row.temporal_block)): float(row.utility_sum)
                for _, row in temporal_top5[
                    temporal_top5.score == 'pred_reg'
                ].iterrows()
            },
        },
        'p15_incremental_loso': {
            'p15_only_regression': baseline_reg.to_dict(),
            'augmented_p17_regression': augmented_reg.to_dict(),
            'naive_pooling_allowed': naive_pooling_allowed,
        },
        'domain_shift': {
            'top_shifted_features': shifts.head(10)[
                [
                    'feature',
                    'standardized_median_shift',
                    'ks_statistic',
                ]
            ].to_dict('records'),
        },
        'decision': {
            'p17_bank_retained': True,
            'naive_pooling_into_p15_model': naive_pooling_allowed,
            'p15_policy_changed': False,
            'locked_manifest_created': False,
            'next_model_family': 'Domain-conditioned or hierarchical utility model with domain-specific calibration and shared representation; no pooled threshold tuning on frozen P15.',
        },
    }
    report_path = out_dir / 'report.json'
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    manifest = {
        path.name: sha256(path)
        for path in sorted(out_dir.iterdir())
        if path.is_file() and path.name != 'prediction_manifest.json'
    }
    (out_dir / 'prediction_manifest.json').write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
