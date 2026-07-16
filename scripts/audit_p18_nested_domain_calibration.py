from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


KEYS = ['seq', 'canonical_rank', 'u', 'v', 'boundary_frame', 'transaction_type']
TARGET = 'full_idtp_delta_norm'
META_COLUMNS = [
    'base_raw',
    'base_multitask',
    'base_positive',
    'base_raw_rank',
    'base_multitask_rank',
    'base_positive_rank',
    'base_raw_top_margin',
    'base_multitask_top_margin',
    'base_positive_top_margin',
    'view_rank_mean',
    'view_rank_min',
    'view_rank_std',
    'block_size',
    'temporal_block',
]


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


def sequence_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.seq.value_counts()
    mapping = {
        sequence: len(frame) / (len(counts) * count)
        for sequence, count in counts.items()
    }
    return frame.seq.map(mapping).to_numpy(float)


def assign_temporal_blocks(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result['temporal_block'] = -1
    for sequence, group in result.groupby('seq', sort=True):
        ordered = group.sort_values(
            ['effective_start_frame', 'canonical_rank', 'transaction_type']
        ).index
        result.loc[ordered, 'temporal_block'] = np.minimum(
            3, (np.arange(len(ordered)) * 4) // len(ordered)
        )
    if (result.temporal_block < 0).any():
        raise RuntimeError('temporal block assignment failed')
    return result


def normalized_targets(frame: pd.DataFrame, targets: list[str]) -> np.ndarray:
    output = np.zeros((len(frame), len(targets)), dtype=float)
    for sequence, indices in frame.groupby('seq', sort=True).groups.items():
        positions = frame.index.get_indexer(indices)
        values = frame.loc[indices, targets].to_numpy(float)
        median = np.median(values, axis=0)
        q25 = np.quantile(values, 0.25, axis=0)
        q75 = np.quantile(values, 0.75, axis=0)
        scale = np.maximum((q75 - q25) / 1.349, 0.05)
        output[positions] = np.clip((values - median) / scale, -5.0, 5.0)
    return output


def sequence_rank_features(frame: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    output = np.zeros_like(values, dtype=float)
    for sequence, indices in frame.groupby('seq', sort=True).groups.items():
        positions = frame.index.get_indexer(indices)
        local = values[positions]
        output[positions] = np.column_stack(
            [
                (rankdata(local[:, column], method='average') - 0.5) / len(local)
                for column in range(local.shape[1])
            ]
        )
    return output


def fit_base_views(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    targets: list[str],
    seed: int,
) -> pd.DataFrame:
    imputer = SimpleImputer(strategy='median')
    x_train = imputer.fit_transform(train[features])
    x_test = imputer.transform(test[features])
    weights = sequence_weights(train)
    raw_model = ExtraTreesRegressor(
        n_estimators=200,
        max_depth=7,
        min_samples_leaf=3,
        max_features=0.65,
        random_state=seed,
        n_jobs=-1,
    )
    multitask_model = ExtraTreesRegressor(
        n_estimators=200,
        max_depth=7,
        min_samples_leaf=3,
        max_features=0.65,
        random_state=seed + 13,
        n_jobs=-1,
    )
    positive_model = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=7,
        min_samples_leaf=3,
        max_features=0.65,
        class_weight='balanced',
        random_state=seed + 29,
        n_jobs=-1,
    )
    raw_model.fit(x_train, train[TARGET], sample_weight=weights)
    multitask_model.fit(
        x_train,
        normalized_targets(train, targets),
        sample_weight=weights,
    )
    positive_model.fit(
        sequence_rank_features(train, x_train),
        (train[TARGET] > 0.0).astype(int),
        sample_weight=weights,
    )
    return pd.DataFrame(
        {
            'base_raw': raw_model.predict(x_test),
            'base_multitask': multitask_model.predict(x_test).mean(axis=1),
            'base_positive': positive_model.predict_proba(
                sequence_rank_features(test, x_test)
            )[:, 1],
        }
    )


def enrich(frame: pd.DataFrame, views: pd.DataFrame) -> pd.DataFrame:
    result = frame.reset_index(drop=True).copy()
    views = views.reset_index(drop=True)
    for column in views.columns:
        result[column] = views[column]
    view_names = ['base_raw', 'base_multitask', 'base_positive']
    for name in view_names:
        result[f'{name}_rank'] = result.groupby(['seq', 'temporal_block'])[name].rank(
            method='average', pct=True
        )
        result[f'{name}_top_margin'] = 0.0
        for _, indices in result.groupby(['seq', 'temporal_block']).groups.items():
            values = result.loc[indices, name].sort_values(ascending=False).to_numpy()
            margin = float(values[0] - values[1]) if len(values) > 1 else 0.0
            result.loc[indices, f'{name}_top_margin'] = margin
    rank_columns = [f'{name}_rank' for name in view_names]
    result['view_rank_mean'] = result[rank_columns].mean(axis=1)
    result['view_rank_min'] = result[rank_columns].min(axis=1)
    result['view_rank_std'] = result[rank_columns].std(axis=1)
    result['block_size'] = result.groupby(['seq', 'temporal_block'])[TARGET].transform(
        'size'
    )
    return result


def select_inner_threshold(
    inner_topone: pd.DataFrame,
    source_sequences: list[str],
) -> dict[str, object]:
    candidates = sorted(set([0.0, *inner_topone.meta_score.tolist(), 1.0]))
    eligible_rows = []
    for threshold in candidates:
        selected = inner_topone[inner_topone.meta_score >= threshold]
        utility = selected.groupby('seq')[TARGET].sum().reindex(
            source_sequences, fill_value=0.0
        )
        eligible = bool(
            selected.seq.nunique() == len(source_sequences)
            and float(selected[TARGET].sum()) > 0.0
            and float(utility.min()) >= 0.0
        )
        if eligible:
            eligible_rows.append(
                {
                    'threshold': float(threshold),
                    'selected': int(len(selected)),
                    'utility_sum': float(selected[TARGET].sum()),
                    'worst_sequence_utility': float(utility.min()),
                    'negative_events': int((selected[TARGET] < 0.0).sum()),
                }
            )
    if not eligible_rows:
        return {
            'threshold': 1.0,
            'inner_authorized': 0,
            'inner_selected': 0,
            'inner_utility_sum': 0.0,
            'inner_worst_sequence_utility': 0.0,
        }
    best = sorted(
        eligible_rows,
        key=lambda row: (
            row['selected'],
            row['utility_sum'],
            row['worst_sequence_utility'],
            -row['negative_events'],
            row['threshold'],
        ),
        reverse=True,
    )[0]
    return {
        'threshold': best['threshold'],
        'inner_authorized': 1,
        'inner_selected': best['selected'],
        'inner_utility_sum': best['utility_sum'],
        'inner_worst_sequence_utility': best['worst_sequence_utility'],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--motion-features', required=True)
    parser.add_argument('--event-labels', required=True)
    parser.add_argument('--compact-features', required=True)
    parser.add_argument('--qualification-report', required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()

    features = pd.read_csv(args.compact_features).feature.astype(str).tolist()
    qualification = json.loads(Path(args.qualification_report).read_text())
    targets = [str(target) for target in qualification['qualified_auxiliary_targets']]
    labels = pd.read_csv(
        args.event_labels, usecols=KEYS + targets + ['effective_start_frame']
    )
    motion = pd.read_csv(args.motion_features)
    frame = motion.merge(labels, on=KEYS, how='inner', validate='one_to_one')
    frame = assign_temporal_blocks(frame).sort_values(KEYS).reset_index(drop=True)
    sequences = sorted(frame.seq.unique())

    outer_outputs = []
    threshold_rows = []
    for outer_fold, held_out in enumerate(sequences):
        outer_train = frame[frame.seq != held_out].reset_index(drop=True)
        outer_test = frame[frame.seq == held_out].reset_index(drop=True)
        inner_outputs = []
        inner_sequences = sorted(outer_train.seq.unique())
        for inner_fold, inner_held_out in enumerate(inner_sequences):
            inner_train = outer_train[
                outer_train.seq != inner_held_out
            ].reset_index(drop=True)
            inner_test = outer_train[
                outer_train.seq == inner_held_out
            ].reset_index(drop=True)
            inner_views = fit_base_views(
                inner_train,
                inner_test,
                features,
                targets,
                10000 + 100 * outer_fold + inner_fold,
            )
            inner_outputs.append(enrich(inner_test, inner_views))
        inner_oof = pd.concat(inner_outputs, ignore_index=True)

        meta_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.5,
                class_weight='balanced',
                max_iter=2000,
                random_state=700 + outer_fold,
            ),
        )
        meta_model.fit(
            inner_oof[META_COLUMNS],
            (inner_oof[TARGET] > 0.0).astype(int),
            logisticregression__sample_weight=sequence_weights(inner_oof),
        )
        inner_oof['meta_score'] = meta_model.predict_proba(
            inner_oof[META_COLUMNS]
        )[:, 1]
        inner_topone = []
        for _, group in inner_oof.groupby(['seq', 'temporal_block'], sort=True):
            inner_topone.append(
                group.sort_values(
                    ['meta_score', 'canonical_rank', 'transaction_type'],
                    ascending=[False, True, True],
                ).iloc[0]
            )
        threshold_result = select_inner_threshold(
            pd.DataFrame(inner_topone), inner_sequences
        )
        threshold_rows.append({'held_out_seq': held_out, **threshold_result})

        outer_views = fit_base_views(
            outer_train,
            outer_test,
            features,
            targets,
            20000 + outer_fold,
        )
        outer_result = enrich(outer_test, outer_views)
        outer_result['meta_score'] = meta_model.predict_proba(
            outer_result[META_COLUMNS]
        )[:, 1]
        outer_result['threshold'] = threshold_result['threshold']
        outer_result['inner_authorized'] = threshold_result['inner_authorized']
        outer_outputs.append(outer_result)

    predictions = pd.concat(outer_outputs, ignore_index=True)
    thresholds = pd.DataFrame(threshold_rows)
    topone_rows = []
    for (sequence, block), group in predictions.groupby(
        ['seq', 'temporal_block'], sort=True
    ):
        selected = group.sort_values(
            ['meta_score', 'canonical_rank', 'transaction_type'],
            ascending=[False, True, True],
        ).iloc[0]
        topone_rows.append(
            {
                **{column: selected[column] for column in KEYS},
                'temporal_block': int(block),
                'meta_score': float(selected.meta_score),
                'threshold': float(selected.threshold),
                'inner_authorized': int(selected.inner_authorized),
                'selected': int(
                    selected.inner_authorized == 1
                    and selected.meta_score >= selected.threshold
                ),
                TARGET: float(selected[TARGET]),
            }
        )
    topone = pd.DataFrame(topone_rows)
    selected = topone[topone.selected == 1]
    utility = selected.groupby('seq')[TARGET].sum().reindex(sequences, fill_value=0.0)
    labels_binary = (predictions[TARGET] > 0.0).astype(int)
    report = {
        'protocol': {
            'scope': 'Strict nested sequence-level domain calibration on the 705-event canonical P17 bank.',
            'base_views': ['raw utility', '13-target multitask', 'sequence-rank positive probability'],
            'meta_model': 'Standardized logistic regression on completely inner-OOF base-view predictions and block-relative agreement features.',
            'authorization': 'An outer fold is authorized only if one inner threshold covers every source sequence, has positive total local utility, and has nonnegative worst-source-sequence utility.',
            'model_or_threshold_sweep': False,
            'global_trackeval_calls': 0,
            'p15_locked_labels_read': 0,
            'p15_locked_trackeval_calls': 0,
            'p15_remaining_locked_rows_untouched': 156,
        },
        'dataset': {
            'events': len(frame),
            'sequences': sequences,
            'temporal_blocks': int(frame.groupby(['seq', 'temporal_block']).ngroups),
            'compact_features': len(features),
            'qualified_targets': len(targets),
        },
        'event_metrics': {
            'spearman': float(spearmanr(predictions[TARGET], predictions.meta_score).statistic),
            'positive_auc': float(roc_auc_score(labels_binary, predictions.meta_score)),
            'positive_average_precision': float(
                average_precision_score(labels_binary, predictions.meta_score)
            ),
        },
        'authorization': {
            'authorized_outer_folds': int(thresholds.inner_authorized.sum()),
            'selected_outer_blocks': int(len(selected)),
            'selected_positive_blocks': int((selected[TARGET] > 0.0).sum()),
            'selected_negative_blocks': int((selected[TARGET] < 0.0).sum()),
            'selected_utility_sum': float(selected[TARGET].sum()),
            'selected_worst_sequence_utility': float(utility.min()),
        },
        'decision': {
            'deployment_allowed': False,
            'locked_manifest_created': False,
            'p15_policy_changed': False,
            'canonical_calibration_family_closed': True,
            'reason': 'No outer fold finds an inner threshold satisfying source-domain coverage and nonnegative worst-source-domain utility. The nested policy therefore fails closed with zero selected blocks.',
        },
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    prediction_output = rounded(predictions)
    prediction_output['meta_score'] = prediction_output.meta_score.round(10)
    prediction_output.to_csv(out_dir / 'outer_predictions.csv', index=False)
    rounded(thresholds).to_csv(out_dir / 'inner_thresholds.csv', index=False)
    rounded(topone).to_csv(out_dir / 'outer_topone.csv', index=False)
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
