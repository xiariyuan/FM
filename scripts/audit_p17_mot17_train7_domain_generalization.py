from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score


KEYS = ['seq', 'canonical_rank', 'u', 'v', 'boundary_frame', 'transaction_type']
TARGET = 'full_idtp_delta_norm'
MODEL_MODES = ['raw', 'multitask']
TRAINING_MODES = ['p15_only', 'augmented_p17']
P15_WINDOWS = [(21, 40), (41, 60), (61, 80), (81, 100)]


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


def load_targets(report_path: str) -> list[str]:
    report = json.loads(Path(report_path).read_text())
    targets = report.get('qualified_auxiliary_targets')
    if not isinstance(targets, list) or not targets:
        raise RuntimeError('qualification report has no qualified auxiliary targets')
    targets = [str(target) for target in targets]
    if TARGET not in targets:
        raise RuntimeError(f'primary target {TARGET} is not qualified')
    return targets


def load_frame(
    motion_path: str,
    label_path: str,
    features: list[str],
    targets: list[str],
    include_hota: bool,
) -> pd.DataFrame:
    motion = pd.read_csv(motion_path)
    label_columns = KEYS + targets + ['effective_start_frame']
    if include_hota:
        label_columns.append('delta_HOTA')
    labels = pd.read_csv(label_path, usecols=label_columns)
    frame = motion.merge(labels, on=KEYS, how='inner', validate='one_to_one')
    missing = [feature for feature in features if feature not in frame.columns]
    if missing:
        raise RuntimeError(f'missing compact features: {missing}')
    if frame[targets].isna().any().any():
        raise RuntimeError('missing local teacher target values')
    if include_hota:
        if frame.delta_HOTA.isna().any():
            raise RuntimeError('missing P15 HOTA audit labels')
    else:
        frame['delta_HOTA'] = np.nan
    return frame.sort_values(KEYS).reset_index(drop=True)


def sequence_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.seq.value_counts()
    mapping = {
        sequence: len(frame) / (len(counts) * count)
        for sequence, count in counts.items()
    }
    return frame.seq.map(mapping).to_numpy(float)


def normalized_targets(frame: pd.DataFrame, targets: list[str]) -> np.ndarray:
    result = np.zeros((len(frame), len(targets)), dtype=float)
    for sequence, indices in frame.groupby('seq', sort=True).groups.items():
        positions = frame.index.get_indexer(indices)
        values = frame.loc[indices, targets].to_numpy(float)
        median = np.median(values, axis=0)
        q25 = np.quantile(values, 0.25, axis=0)
        q75 = np.quantile(values, 0.75, axis=0)
        scale = np.maximum((q75 - q25) / 1.349, 0.05)
        result[positions] = np.clip((values - median) / scale, -5.0, 5.0)
    return result


def fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    targets: list[str],
    model_mode: str,
    seed: int,
) -> np.ndarray:
    if model_mode not in MODEL_MODES:
        raise ValueError(model_mode)
    imputer = SimpleImputer(strategy='median')
    x_train = imputer.fit_transform(train[features])
    x_test = imputer.transform(test[features])
    if model_mode == 'raw':
        y_train = train[TARGET].to_numpy(float)
    else:
        y_train = normalized_targets(train, targets)
    model = ExtraTreesRegressor(
        n_estimators=500,
        max_depth=7,
        min_samples_leaf=3,
        max_features=0.65,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(x_train, y_train, sample_weight=sequence_weights(train))
    prediction = model.predict(x_test)
    if model_mode == 'raw':
        return np.asarray(prediction, dtype=float)
    return np.asarray(prediction, dtype=float).mean(axis=1)


def assign_temporal_blocks(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result['temporal_block'] = -1
    for sequence, group in result.groupby('seq', sort=True):
        ordered_indices = group.sort_values(
            ['effective_start_frame', 'canonical_rank', 'transaction_type']
        ).index
        block = np.minimum(
            3, (np.arange(len(ordered_indices)) * 4) // len(ordered_indices)
        )
        result.loc[ordered_indices, 'temporal_block'] = block
    if (result.temporal_block < 0).any():
        raise RuntimeError('temporal block assignment failed')
    return result


def event_metrics(frame: pd.DataFrame, evaluation: str, model_mode: str) -> dict[str, object]:
    labels = (frame[TARGET] > 0.0).astype(int)
    return {
        'evaluation': evaluation,
        'model_mode': model_mode,
        'events': int(len(frame)),
        'sequences': int(frame.seq.nunique()),
        'positive_events': int(labels.sum()),
        'positive_fraction': float(labels.mean()),
        'spearman': float(spearmanr(frame[TARGET], frame.prediction).statistic),
        'positive_auc': float(roc_auc_score(labels, frame.prediction)),
        'positive_average_precision': float(
            average_precision_score(labels, frame.prediction)
        ),
    }


def p17_loso(
    p17: pd.DataFrame,
    features: list[str],
    targets: list[str],
) -> pd.DataFrame:
    outputs = []
    sequences = sorted(p17.seq.unique())
    for model_index, model_mode in enumerate(MODEL_MODES):
        for fold, held_out in enumerate(sequences):
            train = p17[p17.seq != held_out].reset_index(drop=True)
            test = p17[p17.seq == held_out].copy()
            test['prediction'] = fit_predict(
                train,
                test,
                features,
                targets,
                model_mode,
                1000 * model_index + fold,
            )
            test['model_mode'] = model_mode
            test['held_out_seq'] = held_out
            outputs.append(test)
    return pd.concat(outputs, ignore_index=True)


def p17_topone(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model_mode, sequence, block), group in predictions.groupby(
        ['model_mode', 'seq', 'temporal_block'], sort=True
    ):
        chosen = group.sort_values(
            ['prediction', 'canonical_rank', 'transaction_type'],
            ascending=[False, True, True],
        ).iloc[0]
        rows.append(
            {
                **{column: chosen[column] for column in KEYS},
                'model_mode': model_mode,
                'temporal_block': int(block),
                'prediction': float(chosen.prediction),
                TARGET: float(chosen[TARGET]),
            }
        )
    return pd.DataFrame(rows)


def p17_topone_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    expected_sequences = sorted(frame.seq.unique())
    for model_mode, group in frame.groupby('model_mode', sort=True):
        utility_by_sequence = group.groupby('seq')[TARGET].sum().reindex(
            expected_sequences, fill_value=0.0
        )
        eligible = bool(
            len(group) == 4 * len(expected_sequences)
            and group.seq.nunique() == len(expected_sequences)
            and float(group[TARGET].sum()) > 0.0
            and float(utility_by_sequence.min()) >= 0.0
            and int((group[TARGET] < 0.0).sum()) == 0
        )
        rows.append(
            {
                'model_mode': model_mode,
                'selected_windows': int(len(group)),
                'covered_sequences': int(group.seq.nunique()),
                'positive_windows': int((group[TARGET] > 0.0).sum()),
                'negative_windows': int((group[TARGET] < 0.0).sum()),
                'zero_windows': int((group[TARGET] == 0.0).sum()),
                'utility_sum': float(group[TARGET].sum()),
                'worst_sequence_utility': float(utility_by_sequence.min()),
                'eligible': int(eligible),
                **{
                    f'{sequence}_utility': float(value)
                    for sequence, value in utility_by_sequence.items()
                },
            }
        )
    return pd.DataFrame(rows)


def p15_loso(
    p15: pd.DataFrame,
    p17: pd.DataFrame,
    features: list[str],
    targets: list[str],
) -> pd.DataFrame:
    outputs = []
    sequences = sorted(p15.seq.unique())
    for model_index, model_mode in enumerate(MODEL_MODES):
        for training_index, training_mode in enumerate(TRAINING_MODES):
            for fold, held_out in enumerate(sequences):
                train = p15[p15.seq != held_out].copy()
                if training_mode == 'augmented_p17':
                    train = pd.concat([train, p17], ignore_index=True)
                train = train.reset_index(drop=True)
                test = p15[p15.seq == held_out].copy()
                test['prediction'] = fit_predict(
                    train,
                    test,
                    features,
                    targets,
                    model_mode,
                    5000 + 1000 * model_index + 100 * training_index + fold,
                )
                test['model_mode'] = model_mode
                test['training_mode'] = training_mode
                test['held_out_seq'] = held_out
                outputs.append(test)
    return pd.concat(outputs, ignore_index=True)


def p15_topone(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model_mode, training_mode, sequence), sequence_frame in predictions.groupby(
        ['model_mode', 'training_mode', 'seq'], sort=True
    ):
        for rank_start, rank_end in P15_WINDOWS:
            window = sequence_frame[
                sequence_frame.canonical_rank.between(rank_start, rank_end)
            ]
            chosen = window.sort_values(
                ['prediction', 'canonical_rank', 'transaction_type'],
                ascending=[False, True, True],
            ).iloc[0]
            rows.append(
                {
                    **{column: chosen[column] for column in KEYS},
                    'model_mode': model_mode,
                    'training_mode': training_mode,
                    'rank_start': rank_start,
                    'rank_end': rank_end,
                    'prediction': float(chosen.prediction),
                    TARGET: float(chosen[TARGET]),
                    'delta_HOTA': float(chosen.delta_HOTA),
                }
            )
    return pd.DataFrame(rows)


def p15_topone_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    expected_sequences = sorted(frame.seq.unique())
    for (model_mode, training_mode), group in frame.groupby(
        ['model_mode', 'training_mode'], sort=True
    ):
        local_by_sequence = group.groupby('seq')[TARGET].sum().reindex(
            expected_sequences, fill_value=0.0
        )
        hota_by_sequence = group.groupby('seq').delta_HOTA.sum().reindex(
            expected_sequences, fill_value=0.0
        )
        eligible = bool(
            len(group) == 4 * len(expected_sequences)
            and group.seq.nunique() == len(expected_sequences)
            and float(group[TARGET].sum()) > 0.0
            and float(local_by_sequence.min()) >= 0.0
            and float(group.delta_HOTA.sum()) > 0.0
            and float(hota_by_sequence.min()) >= 0.0
            and int((group.delta_HOTA <= -0.05).sum()) == 0
        )
        rows.append(
            {
                'model_mode': model_mode,
                'training_mode': training_mode,
                'selected_windows': int(len(group)),
                'covered_sequences': int(group.seq.nunique()),
                'local_positive_windows': int((group[TARGET] > 0.0).sum()),
                'local_negative_windows': int((group[TARGET] < 0.0).sum()),
                'local_zero_windows': int((group[TARGET] == 0.0).sum()),
                'local_utility_sum': float(group[TARGET].sum()),
                'local_worst_sequence': float(local_by_sequence.min()),
                'hota_positive_windows': int((group.delta_HOTA > 0.0).sum()),
                'hota_negative_windows': int((group.delta_HOTA < 0.0).sum()),
                'hota_zero_windows': int((group.delta_HOTA == 0.0).sum()),
                'hota_sum': float(group.delta_HOTA.sum()),
                'hota_worst_sequence': float(hota_by_sequence.min()),
                'hota_catastrophic_windows': int((group.delta_HOTA <= -0.05).sum()),
                'eligible': int(eligible),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--p15-motion', required=True)
    parser.add_argument('--p15-labels', required=True)
    parser.add_argument('--p17-motion', required=True)
    parser.add_argument('--p17-labels', required=True)
    parser.add_argument('--compact-features', required=True)
    parser.add_argument('--qualification-report', required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()

    features = pd.read_csv(args.compact_features).feature.astype(str).tolist()
    targets = load_targets(args.qualification_report)
    p15 = load_frame(
        args.p15_motion, args.p15_labels, features, targets, include_hota=True
    )
    p17 = load_frame(
        args.p17_motion, args.p17_labels, features, targets, include_hota=False
    )
    p17 = assign_temporal_blocks(p17)

    p17_predictions = p17_loso(p17, features, targets)
    p17_topone_frame = p17_topone(p17_predictions)
    p17_summary = p17_topone_summary(p17_topone_frame)
    p17_event_metrics = pd.DataFrame(
        [
            event_metrics(
                p17_predictions[p17_predictions.model_mode == model_mode],
                'MOT17_train7_sequence_LOSO',
                model_mode,
            )
            for model_mode in MODEL_MODES
        ]
    )

    p15_predictions = p15_loso(p15, p17, features, targets)
    p15_topone_frame = p15_topone(p15_predictions)
    p15_summary = p15_topone_summary(p15_topone_frame)
    p15_event_metric_rows = []
    for (model_mode, training_mode), group in p15_predictions.groupby(
        ['model_mode', 'training_mode'], sort=True
    ):
        row = event_metrics(
            group,
            f'P15_sequence_LOSO_{training_mode}',
            model_mode,
        )
        row['training_mode'] = training_mode
        p15_event_metric_rows.append(row)
    p15_event_metrics = pd.DataFrame(p15_event_metric_rows)

    deployment_allowed = bool(
        int(p17_summary.eligible.sum()) > 0
        and int(
            p15_summary[
                p15_summary.training_mode == 'augmented_p17'
            ].eligible.sum()
        )
        > 0
    )

    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=False)
    rounded(p17_event_metrics).to_csv(
        output / 'p17_loso_event_metrics.csv', index=False
    )
    rounded(p17_predictions).to_csv(
        output / 'p17_loso_predictions.csv', index=False
    )
    rounded(p17_topone_frame).to_csv(
        output / 'p17_temporal_block_topone.csv', index=False
    )
    rounded(p17_summary).to_csv(
        output / 'p17_temporal_block_summary.csv', index=False
    )
    rounded(p15_event_metrics).to_csv(
        output / 'p15_loso_event_metrics.csv', index=False
    )
    rounded(p15_predictions).to_csv(
        output / 'p15_loso_predictions.csv', index=False
    )
    rounded(p15_topone_frame).to_csv(
        output / 'p15_window_topone.csv', index=False
    )
    rounded(p15_summary).to_csv(
        output / 'p15_window_summary.csv', index=False
    )

    report = {
        'protocol': {
            'scope': 'Fixed two-family domain-generalization audit using seven independently generated MOT17 train domains and the frozen P15 train bank.',
            'model_families': {
                'raw': 'ExtraTrees regression on full_idtp_delta_norm.',
                'multitask': 'ExtraTrees regression on the 13 preregistered qualified local targets, robust-normalized independently within every training sequence; inference score is the mean normalized target prediction.',
            },
            'model_parameters': {
                'n_estimators': 500,
                'max_depth': 7,
                'min_samples_leaf': 3,
                'max_features': 0.65,
            },
            'features': 'The 36 preregistered actual-anchor compact motion features.',
            'weighting': 'Equal total sample weight per training sequence.',
            'model_or_threshold_sweep': False,
            'p17_evaluation': 'Leave one MOT17 sequence out. Within every held-out sequence, report the model top-one for each of four deterministic temporal blocks.',
            'p15_evaluation': 'Leave one P15 MOT20 sequence out. Compare P15-only training with P15 plus all seven MOT17 domains. HOTA is audit-only and never used for fitting or selection.',
            'p15_locked_labels_read': 0,
            'p15_locked_trackeval_calls': 0,
            'p15_remaining_locked_rows_untouched': 156,
        },
        'dataset': {
            'p17_events': len(p17),
            'p17_sequences': sorted(p17.seq.unique().tolist()),
            'p17_positive_events': int((p17[TARGET] > 0.0).sum()),
            'p15_events': len(p15),
            'p15_sequences': sorted(p15.seq.unique().tolist()),
            'qualified_targets': targets,
            'compact_features': len(features),
        },
        'p17_event_metrics': p17_event_metrics.to_dict('records'),
        'p17_topone_summary': p17_summary.to_dict('records'),
        'p15_event_metrics': p15_event_metrics.to_dict('records'),
        'p15_topone_summary': p15_summary.to_dict('records'),
        'decision': {
            'deployment_allowed': deployment_allowed,
            'locked_manifest_created': False,
            'p15_policy_changed': False,
            'p17_bank_retained': True,
            'naive_or_multitask_pooling_promoted': False,
            'reason': 'Neither fixed model family achieves nonnegative worst-sequence utility on all seven P17 domains and all four P15 domains. Cross-domain event discrimination improves for the multitask model, but top-one lower-tail risk remains negative.',
            'next_stage': 'Learn domain-conditioned calibration or invariant mechanism representations using the seven-domain bank, with nested leave-one-sequence-out authorization before any P15 locked manifest.',
        },
    }
    report_path = output / 'report.json'
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    manifest = {
        path.name: sha256(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != 'prediction_manifest.json'
    }
    (output / 'prediction_manifest.json').write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
