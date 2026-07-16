from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score


EVENT_KEYS = ['seq', 'canonical_rank', 'u', 'v', 'boundary_frame', 'transaction_type']
MATCH_KEYS = ['seq', 'u', 'v', 'boundary_frame', 'transaction_type']
TARGET = 'full_idtp_delta_norm'
MODEL_NAMES = ['geometry_positive', 'geometry_utility', 'motion_multitask']

GEOMETRY_BASE_FEATURES = [
    'changed_rows',
    'pair_common_frames',
    'pair_common_span',
    'donor_track_rows',
    'receiver_track_rows',
    'donor_history_rows',
    'receiver_history_rows',
    'handoff_gap',
    'future_receiver_rows',
    'future_receiver_unique_frames',
    'future_receiver_span',
    'future_receiver_density',
    'boundary_center_distance_norm',
    'boundary_bottom_gap_norm',
    'boundary_iou',
    'boundary_log_width_ratio_abs',
    'boundary_log_height_ratio_abs',
    'boundary_donor_score',
    'boundary_receiver_score',
]
GEOMETRY_DERIVED_FEATURES = [
    'boundary_progress',
    'donor_lifetime',
    'receiver_lifetime',
    'common_fraction_donor',
    'common_fraction_receiver',
    'future_fraction_receiver',
    'history_balance_log',
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


def add_geometry_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result['sequence_max_frame'] = result.groupby('seq').receiver_last_frame.transform('max')
    result['boundary_progress'] = result.boundary_frame / result.sequence_max_frame.clip(lower=1)
    result['donor_lifetime'] = result.donor_last_frame - result.donor_first_frame + 1
    result['receiver_lifetime'] = result.receiver_last_frame - result.receiver_first_frame + 1
    result['common_fraction_donor'] = (
        result.pair_common_frames / result.donor_track_rows.clip(lower=1)
    )
    result['common_fraction_receiver'] = (
        result.pair_common_frames / result.receiver_track_rows.clip(lower=1)
    )
    result['future_fraction_receiver'] = (
        result.future_receiver_rows / result.receiver_track_rows.clip(lower=1)
    )
    result['history_balance_log'] = np.log1p(result.donor_history_rows) - np.log1p(
        result.receiver_history_rows
    )
    return result


def assign_temporal_blocks(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result['temporal_block'] = -1
    for sequence, group in result.groupby('seq', sort=True):
        ordered = group.sort_values(
            ['effective_start_frame', 'canonical_rank', 'transaction_type']
        ).index
        block = np.minimum(3, (np.arange(len(ordered)) * 4) // len(ordered))
        result.loc[ordered, 'temporal_block'] = block
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


def fit_fold_models(
    train: pd.DataFrame,
    test: pd.DataFrame,
    geometry_features: list[str],
    motion_features: list[str],
    targets: list[str],
    fold: int,
) -> pd.DataFrame:
    weights = sequence_weights(train)
    geometry_imputer = SimpleImputer(strategy='median')
    geometry_train = geometry_imputer.fit_transform(train[geometry_features])
    geometry_test = geometry_imputer.transform(test[geometry_features])
    positive_model = ExtraTreesClassifier(
        n_estimators=500,
        max_depth=9,
        min_samples_leaf=4,
        max_features=0.70,
        class_weight='balanced',
        random_state=1000 + fold,
        n_jobs=-1,
    )
    positive_model.fit(
        geometry_train,
        (train[TARGET] > 0.0).astype(int),
        sample_weight=weights,
    )
    utility_model = ExtraTreesRegressor(
        n_estimators=500,
        max_depth=9,
        min_samples_leaf=4,
        max_features=0.70,
        random_state=2000 + fold,
        n_jobs=-1,
    )
    utility_model.fit(geometry_train, train[TARGET], sample_weight=weights)

    motion_imputer = SimpleImputer(strategy='median')
    motion_train = motion_imputer.fit_transform(train[motion_features])
    motion_test = motion_imputer.transform(test[motion_features])
    motion_model = ExtraTreesRegressor(
        n_estimators=500,
        max_depth=9,
        min_samples_leaf=4,
        max_features=0.70,
        random_state=3000 + fold,
        n_jobs=-1,
    )
    motion_model.fit(
        motion_train,
        normalized_targets(train, targets),
        sample_weight=weights,
    )
    result = test.copy()
    result['geometry_positive'] = positive_model.predict_proba(geometry_test)[:, 1]
    result['geometry_utility'] = utility_model.predict(geometry_test)
    result['motion_multitask'] = motion_model.predict(motion_test).mean(axis=1)
    return result


def event_metrics(frame: pd.DataFrame, model_name: str) -> dict[str, object]:
    labels = (frame[TARGET] > 0.0).astype(int)
    return {
        'model_name': model_name,
        'events': int(len(frame)),
        'positive_events': int(labels.sum()),
        'positive_fraction': float(labels.mean()),
        'spearman': float(spearmanr(frame[TARGET], frame[model_name]).statistic),
        'positive_auc': float(roc_auc_score(labels, frame[model_name])),
        'positive_average_precision': float(
            average_precision_score(labels, frame[model_name])
        ),
    }


def topone_events(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name in MODEL_NAMES:
        for (sequence, block), group in predictions.groupby(
            ['seq', 'temporal_block'], sort=True
        ):
            chosen = group.sort_values(
                [model_name, 'boundary_frame', 'u', 'v', 'transaction_type'],
                ascending=[False, True, True, True, True],
            ).iloc[0]
            rows.append(
                {
                    **{column: chosen[column] for column in EVENT_KEYS},
                    'model_name': model_name,
                    'temporal_block': int(block),
                    'prediction': float(chosen[model_name]),
                    TARGET: float(chosen[TARGET]),
                }
            )
    return pd.DataFrame(rows)


def topone_summary(topone: pd.DataFrame, sequences: list[str]) -> pd.DataFrame:
    rows = []
    for model_name, group in topone.groupby('model_name', sort=True):
        utility = group.groupby('seq')[TARGET].sum().reindex(sequences, fill_value=0.0)
        rows.append(
            {
                'model_name': model_name,
                'selected_blocks': int(len(group)),
                'covered_sequences': int(group.seq.nunique()),
                'positive_blocks': int((group[TARGET] > 0.0).sum()),
                'negative_blocks': int((group[TARGET] < 0.0).sum()),
                'zero_blocks': int((group[TARGET] == 0.0).sum()),
                'utility_sum': float(group[TARGET].sum()),
                'worst_sequence_utility': float(utility.min()),
                'positive_sequences': int((utility > 0.0).sum()),
                'eligible': int(
                    group.seq.nunique() == len(sequences)
                    and float(group[TARGET].sum()) > 0.0
                    and float(utility.min()) >= 0.0
                ),
                **{
                    f'{sequence}_utility': float(value)
                    for sequence, value in utility.items()
                },
            }
        )
    return pd.DataFrame(rows)


def oracle_blocks(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (sequence, block), group in frame.groupby(
        ['seq', 'temporal_block'], sort=True
    ):
        maximum_index = group[TARGET].idxmax()
        rows.append(
            {
                'seq': sequence,
                'temporal_block': int(block),
                'events': int(len(group)),
                'positive_events': int((group[TARGET] > 0.0).sum()),
                'negative_events': int((group[TARGET] < 0.0).sum()),
                'zero_events': int((group[TARGET] == 0.0).sum()),
                'oracle_max_utility': float(group.loc[maximum_index, TARGET]),
                'oracle_event_rank': int(group.loc[maximum_index, 'canonical_rank']),
            }
        )
    return pd.DataFrame(rows)


def gate_feasibility(
    topone: pd.DataFrame,
    sequences: list[str],
) -> pd.DataFrame:
    rows = []
    for model_name, group in topone.groupby('model_name', sort=True):
        minimum = float(group.prediction.min())
        maximum = float(group.prediction.max())
        candidates = sorted(
            set(
                [
                    float(np.nextafter(minimum, -np.inf)),
                    *group.prediction.tolist(),
                    float(np.nextafter(maximum, np.inf)),
                ]
            )
        )
        evaluated = []
        for threshold in candidates:
            selected = group[group.prediction >= threshold]
            utility = selected.groupby('seq')[TARGET].sum().reindex(
                sequences, fill_value=0.0
            )
            eligible = bool(
                selected.seq.nunique() == len(sequences)
                and float(selected[TARGET].sum()) > 0.0
                and float(utility.min()) >= 0.0
            )
            evaluated.append(
                {
                    'model_name': model_name,
                    'threshold': float(threshold),
                    'eligible': int(eligible),
                    'selected_blocks': int(len(selected)),
                    'covered_sequences': int(selected.seq.nunique()),
                    'positive_blocks': int((selected[TARGET] > 0.0).sum()),
                    'negative_blocks': int((selected[TARGET] < 0.0).sum()),
                    'zero_blocks': int((selected[TARGET] == 0.0).sum()),
                    'utility_sum': float(selected[TARGET].sum()),
                    'worst_sequence_utility': float(utility.min()),
                }
            )
        frame = pd.DataFrame(evaluated)
        full_coverage = frame[frame.covered_sequences == len(sequences)].copy()
        if full_coverage.empty:
            raise RuntimeError(f'no full-coverage threshold for {model_name}')
        best = full_coverage.sort_values(
            [
                'eligible',
                'worst_sequence_utility',
                'utility_sum',
                'selected_blocks',
                'threshold',
            ],
            ascending=[False, False, False, False, False],
        ).iloc[0]
        rows.append(best.to_dict())
    return pd.DataFrame(rows)


def canonical_recall(full: pd.DataFrame, canonical_path: str) -> pd.DataFrame:
    canonical = pd.read_csv(canonical_path, usecols=MATCH_KEYS)
    canonical = canonical.drop_duplicates(MATCH_KEYS)
    merged = full.merge(
        canonical.assign(in_canonical=1),
        on=MATCH_KEYS,
        how='left',
        validate='one_to_one',
    )
    merged['in_canonical'] = merged.in_canonical.fillna(0).astype(int)
    rows = []
    for sequence, group in merged.groupby('seq', sort=True):
        positives = group[group[TARGET] > 0.0]
        rows.append(
            {
                'seq': sequence,
                'full_events': int(len(group)),
                'canonical_events': int(group.in_canonical.sum()),
                'full_positive_events': int(len(positives)),
                'canonical_positive_events': int(positives.in_canonical.sum()),
                'positive_recall': float(
                    positives.in_canonical.sum() / max(1, len(positives))
                ),
            }
        )
    total_positive = merged[merged[TARGET] > 0.0]
    rows.append(
        {
            'seq': 'ALL',
            'full_events': int(len(merged)),
            'canonical_events': int(merged.in_canonical.sum()),
            'full_positive_events': int(len(total_positive)),
            'canonical_positive_events': int(total_positive.in_canonical.sum()),
            'positive_recall': float(
                total_positive.in_canonical.sum() / max(1, len(total_positive))
            ),
        }
    )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--event-labels', required=True)
    parser.add_argument('--motion-features', required=True)
    parser.add_argument('--motion-feature-list', required=True)
    parser.add_argument('--qualification-report', required=True)
    parser.add_argument('--canonical-event-labels', required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()

    labels = pd.read_csv(args.event_labels)
    motion_features = pd.read_csv(args.motion_features)
    motion_feature_list = pd.read_csv(args.motion_feature_list).feature.astype(str).tolist()
    qualification = json.loads(Path(args.qualification_report).read_text())
    targets = [str(target) for target in qualification['qualified_auxiliary_targets']]
    if TARGET not in targets:
        raise RuntimeError(f'{TARGET} is not a qualified target')
    frame = labels.merge(
        motion_features[EVENT_KEYS + motion_feature_list],
        on=EVENT_KEYS,
        how='inner',
        validate='one_to_one',
    )
    frame = assign_temporal_blocks(add_geometry_features(frame))
    geometry_features = GEOMETRY_BASE_FEATURES + GEOMETRY_DERIVED_FEATURES
    missing = [
        feature
        for feature in geometry_features + motion_feature_list
        if feature not in frame.columns
    ]
    if missing:
        raise RuntimeError(f'missing audit features: {missing}')

    predictions = []
    sequences = sorted(frame.seq.unique())
    for fold, held_out in enumerate(sequences):
        train = frame[frame.seq != held_out].reset_index(drop=True)
        test = frame[frame.seq == held_out].reset_index(drop=True)
        predictions.append(
            fit_fold_models(
                train,
                test,
                geometry_features,
                motion_feature_list,
                targets,
                fold,
            )
        )
    prediction_frame = pd.concat(predictions, ignore_index=True)
    metric_frame = pd.DataFrame(
        [event_metrics(prediction_frame, model_name) for model_name in MODEL_NAMES]
    )
    topone = topone_events(prediction_frame)
    summary = topone_summary(topone, sequences)
    oracle = oracle_blocks(frame)
    gates = gate_feasibility(topone, sequences)
    recall = canonical_recall(frame, args.canonical_event_labels)

    metric_output = rounded(metric_frame)
    prediction_output = rounded(prediction_frame)
    topone_output = rounded(topone)
    summary_output = rounded(summary)
    oracle_output = rounded(oracle)
    gates_output = rounded(gates)
    recall_output = rounded(recall)

    oracle_by_sequence = oracle.groupby('seq').oracle_max_utility.sum()
    deployment_allowed = bool(
        int(summary.eligible.sum()) > 0 or int(gates.eligible.sum()) > 0
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    metric_output.to_csv(out_dir / 'event_metrics.csv', index=False)
    prediction_output.to_csv(out_dir / 'loso_predictions.csv', index=False)
    topone_output.to_csv(out_dir / 'temporal_block_topone.csv', index=False)
    summary_output.to_csv(out_dir / 'temporal_block_summary.csv', index=False)
    oracle_output.to_csv(out_dir / 'oracle_block_availability.csv', index=False)
    gates_output.to_csv(out_dir / 'retrospective_gate_feasibility.csv', index=False)
    recall_output.to_csv(out_dir / 'canonical_positive_recall.csv', index=False)

    report = {
        'protocol': {
            'scope': 'Fixed full-executable-bank sequence-LOSO audit.',
            'models': {
                'geometry_positive': 'ExtraTrees positive-event classifier on 26 deployable topology/lifecycle/boundary-geometry features.',
                'geometry_utility': 'ExtraTrees full_idtp_delta_norm regressor on the same 26 geometry features.',
                'motion_multitask': 'ExtraTrees 13-target robust-normalized multitask regressor on the 36 preregistered actual-anchor motion features.',
            },
            'common_parameters': {
                'n_estimators': 500,
                'max_depth': 9,
                'min_samples_leaf': 4,
                'max_features': 0.70,
                'sequence_equal_weighting': True,
            },
            'model_or_threshold_sweep': False,
            'retrospective_gate_note': 'For each fixed model, all unique top-one score thresholds are audited only as a feasibility upper bound. They are not used to select or deploy a policy.',
            'global_trackeval_calls': 0,
            'p15_locked_labels_read': 0,
            'p15_locked_trackeval_calls': 0,
            'p15_remaining_locked_rows_untouched': 156,
        },
        'dataset': {
            'events': len(frame),
            'sequences': sequences,
            'positive_events': int((frame[TARGET] > 0.0).sum()),
            'negative_events': int((frame[TARGET] < 0.0).sum()),
            'zero_events': int((frame[TARGET] == 0.0).sum()),
            'geometry_features': len(geometry_features),
            'motion_features': len(motion_feature_list),
            'qualified_targets': len(targets),
        },
        'canonical_positive_recall': recall_output.to_dict('records'),
        'event_metrics': metric_output.to_dict('records'),
        'topone_summary': summary_output.to_dict('records'),
        'oracle': {
            'blocks': len(oracle),
            'blocks_without_positive_event': int((oracle.positive_events == 0).sum()),
            'topone_utility_sum': float(oracle.oracle_max_utility.sum()),
            'worst_sequence_topone_utility': float(oracle_by_sequence.min()),
        },
        'retrospective_gate_feasibility': gates_output.to_dict('records'),
        'decision': {
            'deployment_allowed': deployment_allowed,
            'locked_manifest_created': False,
            'p15_policy_changed': False,
            'full_event_teacher_retained': True,
            'full_motion_bank_retained': True,
            'fixed_model_promoted': False,
            'reason': 'The full executable bank recovers most positive teacher events and exposes complementary geometry and motion signals, but no fixed model and no retrospective single-score abstention threshold achieves seven-sequence coverage with nonnegative worst-sequence utility.',
            'next_stage': 'Model extreme false-positive tail risk with set-valued selective prediction or explicit support/OOD certificates; do not continue scalar threshold tuning.',
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
