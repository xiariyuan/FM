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
from sklearn.metrics import roc_auc_score

from train_loso_local_counterfactual_ensemble import (
    MODEL_FEATURES,
    SEQUENCES,
    TARGET,
    TRAIN_WINDOWS,
    load_training_frame,
)


KEYS = ['seq', 'canonical_rank', 'u', 'v', 'boundary_frame', 'transaction_type']
SEEDS = [17, 53, 89, 131, 173]
MOTION_COMPONENTS = [
    ('source_stability', 'source_center_resid_20_h120_std', False),
    ('source_residual', 'source_center_resid_20_full_mean', False),
    ('transfer_margin', 'motion_margin_20_full_q25', True),
    ('velocity_margin', 'velocity_margin_20_h60', True),
    ('source_win_fraction', 'motion_source_win_fraction_20_h120', True),
]
FALLBACK_THRESHOLDS = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
FALLBACK_TOPK = [2, 3, 5]
FALLBACK_RULES = ['max_motion_min', 'max_motion_q25', 'lexicographic_min_q25']


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def rounded(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.select_dtypes(include=[np.number]).columns:
        result[column] = result[column].round(12)
    return result


def load_targets(report_path: Path) -> list[str]:
    report = json.loads(report_path.read_text(encoding='utf-8'))
    targets = report['qualified_auxiliary_targets']
    if TARGET not in targets:
        raise RuntimeError(f'{TARGET} is not a qualified target')
    return [str(value) for value in targets]


def attach_targets(frame: pd.DataFrame, labels_path: Path, targets: list[str]) -> pd.DataFrame:
    labels = pd.read_csv(labels_path)
    missing_targets = [target for target in targets if target not in frame.columns]
    if missing_targets:
        frame = frame.merge(
            labels[KEYS + missing_targets].drop_duplicates(KEYS),
            on=KEYS,
            how='left',
            validate='one_to_one',
        )
    if frame[targets].isna().any().any():
        raise RuntimeError('missing local teacher targets')
    return frame


def normalized_targets(frame: pd.DataFrame, targets: list[str]) -> np.ndarray:
    output = np.zeros((len(frame), len(targets)), dtype=float)
    for _, indices in frame.groupby('seq', sort=True).groups.items():
        positions = frame.index.get_indexer(indices)
        values = frame.loc[indices, targets].to_numpy(float)
        median = np.median(values, axis=0)
        q25 = np.quantile(values, 0.25, axis=0)
        q75 = np.quantile(values, 0.75, axis=0)
        scale = np.maximum((q75 - q25) / 1.349, 0.05)
        output[positions] = np.clip((values - median) / scale, -5.0, 5.0)
    return output


def sequence_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.seq.value_counts()
    mapping = {seq: len(frame) / (len(counts) * count) for seq, count in counts.items()}
    return frame.seq.map(mapping).to_numpy(float)


def run_loso_increment(frame: pd.DataFrame, compact_motion: list[str]) -> pd.DataFrame:
    rows = []
    for mode, features in [
        ('existing', MODEL_FEATURES),
        ('motion', compact_motion),
        ('expanded', MODEL_FEATURES + compact_motion),
    ]:
        folds = []
        for held_out in SEQUENCES:
            train = frame[frame.seq != held_out].reset_index(drop=True)
            test = frame[frame.seq == held_out].reset_index(drop=True)
            imputer = SimpleImputer(strategy='median')
            x_train = imputer.fit_transform(train[features])
            x_test = imputer.transform(test[features])
            weights = sequence_weights(train)
            regressor = ExtraTreesRegressor(
                n_estimators=300,
                max_depth=7,
                min_samples_leaf=3,
                max_features=0.65,
                random_state=17,
                n_jobs=1,
            )
            regressor.fit(x_train, train[TARGET].to_numpy(float), sample_weight=weights)
            predicted_local = regressor.predict(x_test)
            classifier = ExtraTreesClassifier(
                n_estimators=300,
                max_depth=7,
                min_samples_leaf=3,
                max_features=0.65,
                class_weight='balanced',
                random_state=53,
                n_jobs=1,
            )
            classifier.fit(
                x_train,
                (train.delta_HOTA > 0).astype(int),
                sample_weight=weights,
            )
            predicted_hota_probability = classifier.predict_proba(x_test)[:, 1]
            fold = test[KEYS + [TARGET, 'delta_HOTA']].copy()
            fold['predicted_local'] = predicted_local
            fold['predicted_hota_probability'] = predicted_hota_probability
            folds.append(fold)
        output = pd.concat(folds, ignore_index=True)
        rows.append({
            'mode': mode,
            'features': len(features),
            'spearman_local': float(spearmanr(output[TARGET], output.predicted_local).statistic),
            'spearman_hota': float(spearmanr(output.delta_HOTA, output.predicted_local).statistic),
            'auc_hota_from_local': float(
                roc_auc_score((output.delta_HOTA > 0).astype(int), output.predicted_local)
            ),
            'auc_hota_classifier': float(
                roc_auc_score(
                    (output.delta_HOTA > 0).astype(int),
                    output.predicted_hota_probability,
                )
            ),
        })
    return pd.DataFrame(rows)


def run_multitask_predictions(
    frame: pd.DataFrame,
    targets: list[str],
    compact_motion: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions = []
    summaries = []
    topone_outputs = []
    for mode, features in [
        ('existing', MODEL_FEATURES),
        ('expanded_motion', MODEL_FEATURES + compact_motion),
    ]:
        fold_outputs = []
        for fold_index, held_out in enumerate(SEQUENCES):
            train = frame[frame.seq != held_out].reset_index(drop=True)
            test = frame[frame.seq == held_out].reset_index(drop=True)
            imputer = SimpleImputer(strategy='median')
            x_train = imputer.fit_transform(train[features])
            x_test = imputer.transform(test[features])
            y_train = normalized_targets(train, targets)
            weights = sequence_weights(train)
            seed_predictions = []
            for seed in SEEDS:
                model = ExtraTreesRegressor(
                    n_estimators=300,
                    max_depth=7,
                    min_samples_leaf=3,
                    max_features=0.65,
                    random_state=seed + 1000 * fold_index,
                    n_jobs=1,
                )
                model.fit(x_train, y_train, sample_weight=weights)
                seed_predictions.append(model.predict(x_test))
            array = np.stack(seed_predictions, axis=0)
            mean = array.mean(axis=0)
            dispersion = array.std(axis=0)
            output = test[KEYS + targets + ['delta_HOTA', 'delta_AssA']].copy()
            output['mean_positive_fraction'] = np.mean(mean > 0, axis=1)
            output['mean_target_q25'] = np.quantile(mean, 0.25, axis=1)
            output['mean_target_mean'] = mean.mean(axis=1)
            output['mean_target_min'] = mean.min(axis=1)
            output['prediction_dispersion'] = dispersion.mean(axis=1)
            output['risk_lcb'] = output.mean_target_q25 - output.prediction_dispersion
            output['mode'] = mode
            output['held_out'] = held_out
            fold_outputs.append(output)
        prediction = pd.concat(fold_outputs, ignore_index=True)
        predictions.append(prediction)
        for score in ['mean_positive_fraction', 'mean_target_q25', 'mean_target_mean', 'risk_lcb']:
            selected = []
            for seq in SEQUENCES:
                for start, end in TRAIN_WINDOWS:
                    block = prediction[
                        (prediction.seq == seq)
                        & prediction.canonical_rank.between(start, end)
                    ]
                    chosen = block.sort_values(
                        [score, 'mean_target_q25', 'canonical_rank', 'transaction_type'],
                        ascending=[False, False, True, True],
                    ).iloc[0].copy()
                    chosen['rank_start'] = start
                    chosen['rank_end'] = end
                    chosen['score_name'] = score
                    selected.append(chosen)
            topone = pd.DataFrame(selected)
            sequence_hota = topone.groupby('seq').delta_HOTA.sum()
            sequence_local = topone.groupby('seq')[TARGET].sum()
            summaries.append({
                'mode': mode,
                'score': score,
                'hota_sum': float(topone.delta_HOTA.sum()),
                'hota_positive_windows': int((topone.delta_HOTA > 0).sum()),
                'hota_negative_windows': int((topone.delta_HOTA < 0).sum()),
                'hota_zero_windows': int((topone.delta_HOTA == 0).sum()),
                'hota_worst_sequence': float(sequence_hota.min()),
                'hota_catastrophic_windows': int((topone.delta_HOTA <= -0.05).sum()),
                'local_sum': float(topone[TARGET].sum()),
                'local_positive_windows': int((topone[TARGET] > 0).sum()),
                'local_negative_windows': int((topone[TARGET] < 0).sum()),
                'local_worst_sequence': float(sequence_local.min()),
                **{seq.lower().replace('-', '_') + '_hota': float(sequence_hota[seq]) for seq in SEQUENCES},
            })
            topone['mode'] = mode
            topone['score_name'] = score
            topone_outputs.append(topone)
    return (
        pd.concat(predictions, ignore_index=True),
        pd.DataFrame(summaries),
        pd.concat(topone_outputs, ignore_index=True),
    )


def add_motion_certificate(prediction: pd.DataFrame, motion: pd.DataFrame) -> pd.DataFrame:
    output = prediction.merge(
        motion[KEYS + [column for _, column, _ in MOTION_COMPONENTS]],
        on=KEYS,
        how='inner',
        validate='one_to_one',
    )
    for name, _, _ in MOTION_COMPONENTS:
        output[f'{name}_rank'] = np.nan
    output['base_rank'] = np.nan
    for seq, seq_frame in output.groupby('seq'):
        for start, end in TRAIN_WINDOWS:
            indices = seq_frame.index[seq_frame.canonical_rank.between(start, end)]
            count = len(indices)
            output.loc[indices, 'base_rank'] = output.loc[indices, 'mean_target_mean'].rank(
                method='average', pct=True
            )
            rank_columns = []
            for name, column, higher_is_better in MOTION_COMPONENTS:
                raw = output.loc[indices, column].rank(method='average', pct=True)
                rank = raw if higher_is_better else 1.0 - raw + 1.0 / count
                rank_column = f'{name}_rank'
                output.loc[indices, rank_column] = rank
                rank_columns.append(rank_column)
            output.loc[indices, 'motion_median'] = output.loc[indices, rank_columns].median(axis=1)
            output.loc[indices, 'motion_q25'] = output.loc[indices, rank_columns].quantile(0.25, axis=1)
            output.loc[indices, 'motion_min'] = output.loc[indices, rank_columns].min(axis=1)
            output.loc[indices, 'motion_mean'] = output.loc[indices, rank_columns].mean(axis=1)
    return output


def run_fallback_diagnostic(event_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    topone_outputs = []
    for threshold in FALLBACK_THRESHOLDS:
        for topk in FALLBACK_TOPK:
            for fallback_rule in FALLBACK_RULES:
                rows = []
                for seq in SEQUENCES:
                    for start, end in TRAIN_WINDOWS:
                        block = event_frame[
                            (event_frame.seq == seq)
                            & event_frame.canonical_rank.between(start, end)
                        ].sort_values(
                            ['mean_target_mean', 'mean_target_q25', 'canonical_rank', 'transaction_type'],
                            ascending=[False, False, True, True],
                        )
                        chosen = block.iloc[0].copy()
                        reason = 'keep_top1'
                        if chosen.motion_min < threshold:
                            feasible = block.head(topk)
                            feasible = feasible[feasible.motion_min >= threshold].copy()
                            if len(feasible):
                                if fallback_rule == 'max_motion_min':
                                    chosen = feasible.sort_values(
                                        ['motion_min', 'mean_target_mean', 'canonical_rank'],
                                        ascending=[False, False, True],
                                    ).iloc[0].copy()
                                elif fallback_rule == 'max_motion_q25':
                                    chosen = feasible.sort_values(
                                        ['motion_q25', 'mean_target_mean', 'canonical_rank'],
                                        ascending=[False, False, True],
                                    ).iloc[0].copy()
                                else:
                                    chosen = feasible.sort_values(
                                        ['motion_min', 'motion_q25', 'mean_target_mean', 'canonical_rank'],
                                        ascending=[False, False, False, True],
                                    ).iloc[0].copy()
                                reason = 'fallback'
                            else:
                                chosen['delta_HOTA'] = 0.0
                                chosen[TARGET] = 0.0
                                reason = 'abstain'
                        chosen['rank_start'] = start
                        chosen['rank_end'] = end
                        chosen['decision_reason'] = reason
                        rows.append(chosen)
                topone = pd.DataFrame(rows)
                sequence_hota = topone.groupby('seq').delta_HOTA.sum()
                sequence_local = topone.groupby('seq')[TARGET].sum()
                summaries.append({
                    'threshold': threshold,
                    'topk': topk,
                    'fallback_rule': fallback_rule,
                    'hota_sum': float(topone.delta_HOTA.sum()),
                    'hota_positive_windows': int((topone.delta_HOTA > 0).sum()),
                    'hota_negative_windows': int((topone.delta_HOTA < 0).sum()),
                    'hota_worst_sequence': float(sequence_hota.min()),
                    'hota_catastrophic_windows': int((topone.delta_HOTA <= -0.05).sum()),
                    'local_sum': float(topone[TARGET].sum()),
                    'local_positive_windows': int((topone[TARGET] > 0).sum()),
                    'local_negative_windows': int((topone[TARGET] < 0).sum()),
                    'local_worst_sequence': float(sequence_local.min()),
                    'fallbacks': int((topone.decision_reason == 'fallback').sum()),
                    'abstains': int((topone.decision_reason == 'abstain').sum()),
                    **{seq.lower().replace('-', '_') + '_hota': float(sequence_hota[seq]) for seq in SEQUENCES},
                })
                topone['threshold'] = threshold
                topone['topk'] = topk
                topone['fallback_rule'] = fallback_rule
                topone_outputs.append(topone)
    return pd.DataFrame(summaries), pd.concat(topone_outputs, ignore_index=True)


def appearance_diagnostic(
    frame: pd.DataFrame,
    appearance_path: Path,
    changed_rows_path: Path,
) -> dict[str, object]:
    appearance = pd.read_csv(appearance_path)
    changed = pd.read_csv(changed_rows_path)
    composition = changed.groupby(KEYS + ['row_class']).size().unstack(fill_value=0).reset_index()
    for column in ['benefit', 'harm', 'other_gt', 'shared_history_identity', 'unmatched']:
        if column not in composition:
            composition[column] = 0
    composition['labeled_rows'] = composition[
        ['benefit', 'harm', 'other_gt', 'shared_history_identity', 'unmatched']
    ].sum(axis=1)
    composition['foreign_event'] = (
        (composition.other_gt + composition.unmatched) > 0
    ).astype(int)
    merged = frame.merge(
        appearance[KEYS + ['pair_vs_third_h120_q25']],
        on=KEYS,
        how='inner',
        validate='one_to_one',
    ).merge(
        composition[KEYS + ['foreign_event']],
        on=KEYS,
        how='inner',
        validate='one_to_one',
    )
    auc = float(roc_auc_score(merged.foreign_event, merged.pair_vs_third_h120_q25))
    oriented_auc = max(auc, 1.0 - auc)
    problem = merged[(merged.seq == 'MOT20-02') & (merged.canonical_rank == 72)].iloc[0]
    return {
        'feature': 'pair_vs_third_h120_q25',
        'foreign_event_oriented_auc': oriented_auc,
        'problem_event': {
            'seq': 'MOT20-02',
            'canonical_rank': 72,
            'foreign_event': int(problem.foreign_event),
            'pair_vs_third_h120_q25': float(problem.pair_vs_third_h120_q25),
            'full_idtp_delta_norm': float(problem[TARGET]),
            'delta_HOTA': float(problem.delta_HOTA),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-features', action='append', required=True)
    parser.add_argument('--train-executability', required=True)
    parser.add_argument('--event-labels', required=True)
    parser.add_argument('--qualification-report', required=True)
    parser.add_argument('--motion-features', required=True)
    parser.add_argument('--motion-compact-list', required=True)
    parser.add_argument('--appearance-features', required=True)
    parser.add_argument('--changed-rows', required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()

    targets = load_targets(Path(args.qualification_report))
    frame = load_training_frame(
        args.train_features,
        args.train_executability,
        args.event_labels,
    )
    frame = attach_targets(frame, Path(args.event_labels), targets)
    motion = pd.read_csv(args.motion_features)
    compact_motion = pd.read_csv(args.motion_compact_list).feature.astype(str).tolist()
    frame = frame.merge(
        motion[KEYS + compact_motion],
        on=KEYS,
        how='inner',
        validate='one_to_one',
    ).sort_values(KEYS).reset_index(drop=True)

    loso = run_loso_increment(frame, compact_motion)
    predictions, multitask_summary, multitask_topone = run_multitask_predictions(
        frame, targets, compact_motion
    )
    expanded_predictions = predictions[predictions['mode'] == 'expanded_motion'].copy()
    certificate_events = add_motion_certificate(expanded_predictions, motion)
    fallback_summary, fallback_topone = run_fallback_diagnostic(certificate_events)
    appearance = appearance_diagnostic(
        frame,
        Path(args.appearance_features),
        Path(args.changed_rows),
    )

    best_multitask = multitask_summary.sort_values(
        ['hota_worst_sequence', 'hota_catastrophic_windows', 'hota_sum'],
        ascending=[False, True, False],
    ).iloc[0]
    best_fallback = fallback_summary.sort_values(
        ['local_worst_sequence', 'local_sum', 'hota_worst_sequence', 'hota_sum'],
        ascending=[False, False, False, False],
    ).iloc[0]
    simultaneous_nonnegative = fallback_summary[
        (fallback_summary.local_worst_sequence >= 0)
        & (fallback_summary.hota_worst_sequence >= 0)
        & (fallback_summary.hota_catastrophic_windows == 0)
    ]

    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=False)
    rounded(loso).to_csv(output / 'motion_loso_increment.csv', index=False)
    rounded(multitask_summary).to_csv(output / 'multitask_topone_summary.csv', index=False)
    rounded(multitask_topone).to_csv(output / 'multitask_topone.csv', index=False)
    rounded(fallback_summary).to_csv(output / 'fallback_summary.csv', index=False)
    rounded(fallback_topone).to_csv(output / 'fallback_topone.csv', index=False)
    (output / 'appearance_foreign_diagnostic.json').write_text(
        json.dumps(appearance, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    report = {
        'protocol': {
            'scope': 'Train-only closure diagnostics. This script replays only the already registered fixed LOSO, multitask top-one, and constrained fallback configurations.',
            'locked_labels_read': 0,
            'locked_trackeval_calls': 0,
            'remaining_locked_rows_untouched': 156,
            'fallback_thresholds': FALLBACK_THRESHOLDS,
            'fallback_topk': FALLBACK_TOPK,
            'fallback_rules': FALLBACK_RULES,
        },
        'motion_loso': {
            row['mode']: {
                key: row[key]
                for key in ['features', 'spearman_local', 'spearman_hota', 'auc_hota_from_local', 'auc_hota_classifier']
            }
            for row in loso.to_dict(orient='records')
        },
        'appearance_foreign_diagnostic': appearance,
        'best_non_nested_multitask': best_multitask.to_dict(),
        'best_constrained_fallback': best_fallback.to_dict(),
        'simultaneous_nonnegative_fallback_configs': int(len(simultaneous_nonnegative)),
        'closure_decision': 'no_op',
        'locked_manifest_created': False,
    }
    report_path = output / 'report.json'
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    manifest = {
        path.name: sha256_file(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != 'prediction_manifest.json'
    }
    (output / 'prediction_manifest.json').write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
