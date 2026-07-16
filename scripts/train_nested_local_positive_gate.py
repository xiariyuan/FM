from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

from train_loso_local_counterfactual_ensemble import (
    MODEL_FEATURES,
    SEQUENCES,
    TARGET,
    TRAIN_WINDOWS,
    attach_sequence_robust_target,
    build_pairwise_examples,
    load_training_frame,
    rounded,
    score_pairwise_group,
    sequence_weights,
    sha256,
    window_mask,
)


THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
INNER_ELIGIBILITY = {
    'selected_windows_min': 6,
    'covered_sequences': 3,
    'local_positive_precision_min': 0.85,
    'local_utility_strictly_positive': True,
    'worst_sequence_local_utility_nonnegative': True,
    'negative_mass_max_fraction_of_positive_sum': 0.10,
    'catastrophic_local_target_threshold': -0.10,
    'catastrophic_selected_max': 0,
}
OUTER_DEPLOYMENT_THRESHOLDS = {
    'selected_windows_min': 6,
    'covered_sequences_min': 2,
    'positive_HOTA_precision_min': 0.75,
    'utility_sum_strictly_positive': True,
    'worst_selected_sequence_utility_nonnegative': True,
    'negative_mass_max_fraction_of_positive_sum': 0.20,
    'catastrophic_delta_HOTA_threshold': -0.05,
    'catastrophic_windows_max': 0,
    'event_positive_HOTA_auc_min': 0.70,
    'event_local_positive_auc_min': 0.75,
}


def fit_models(train: pd.DataFrame) -> tuple[SimpleImputer, dict[str, Any], int]:
    work = attach_sequence_robust_target(train)
    imputer = SimpleImputer(strategy='median')
    transformed = imputer.fit_transform(work[MODEL_FEATURES])
    weights = sequence_weights(work)
    local_positive = (work[TARGET] > 0).astype(int).to_numpy()

    models: dict[str, Any] = {}
    for seed in [17, 53]:
        classifier = ExtraTreesClassifier(
            n_estimators=900,
            max_depth=8,
            min_samples_leaf=3,
            max_features=0.70,
            class_weight='balanced',
            random_state=seed,
            n_jobs=-1,
        )
        classifier.fit(transformed, local_positive, sample_weight=weights)
        models[f'positive_s{seed}'] = classifier

    pair_x, pair_y, pair_weights = build_pairwise_examples(work, transformed)
    for seed in [17, 53]:
        ranker = ExtraTreesClassifier(
            n_estimators=800,
            max_depth=8,
            min_samples_leaf=3,
            max_features=0.70,
            class_weight='balanced',
            random_state=seed,
            n_jobs=-1,
        )
        ranker.fit(pair_x, pair_y, sample_weight=pair_weights)
        models[f'pair_s{seed}'] = ranker
    return imputer, models, len(pair_x)


def score_sequence(
    frame: pd.DataFrame,
    imputer: SimpleImputer,
    models: dict[str, Any],
) -> pd.DataFrame:
    parts = []
    for start, end in TRAIN_WINDOWS:
        group = frame[window_mask(frame, start, end)].copy().reset_index(drop=True)
        if not len(group):
            continue
        transformed = imputer.transform(group[MODEL_FEATURES])
        p17 = models['positive_s17'].predict_proba(transformed)[:, 1]
        p53 = models['positive_s53'].predict_proba(transformed)[:, 1]
        pair17 = score_pairwise_group(models['pair_s17'], transformed)
        pair53 = score_pairwise_group(models['pair_s53'], transformed)
        group['positive_prob_s17'] = p17
        group['positive_prob_s53'] = p53
        group['positive_prob_min'] = np.minimum(p17, p53)
        group['positive_prob_mean'] = (p17 + p53) / 2.0
        # This diagnostic is not used by the gate or ranker. Quantize it
        # explicitly so parallel tree summation cannot alter CSV bytes at
        # the 1e-12 boundary across otherwise identical runs.
        group['positive_prob_disagreement'] = np.round(np.abs(p17 - p53), 10)
        group['pairwise_score_s17'] = pair17
        group['pairwise_score_s53'] = pair53
        group['pairwise_score_mean'] = (pair17 + pair53) / 2.0
        group['pairwise_score_disagreement'] = np.round(
            np.abs(pair17 - pair53), 10
        )
        group['rank_start'] = start
        group['rank_end'] = end
        group['window_candidates'] = len(group)
        parts.append(group)
    return pd.concat(parts, ignore_index=True, sort=False)


def select_with_threshold(predictions: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    for _, group in predictions.groupby(['seq', 'rank_start', 'rank_end'], sort=True):
        eligible = group[group.positive_prob_min >= threshold]
        if not len(eligible):
            continue
        chosen = eligible.sort_values(
            [
                'pairwise_score_mean',
                'positive_prob_min',
                'positive_prob_mean',
                'canonical_rank',
                'transaction_type',
            ],
            ascending=[False, False, False, True, True],
        ).iloc[0].copy()
        chosen['gate_threshold'] = threshold
        rows.append(chosen)
    return pd.DataFrame(rows).reset_index(drop=True)


def summarize_local(selection: pd.DataFrame, expected_sequences: list[str]) -> dict[str, Any]:
    if not len(selection):
        return {
            'selected_windows': 0,
            'covered_sequences': 0,
            'local_positive_windows': 0,
            'local_nonpositive_windows': 0,
            'local_positive_precision': 0.0,
            'local_positive_sum': 0.0,
            'local_negative_mass': 0.0,
            'local_negative_mass_fraction': 0.0,
            'local_utility_sum': 0.0,
            'worst_sequence_local_utility': 0.0,
            'catastrophic_local_windows': 0,
            'sequence_local_utility': {sequence: 0.0 for sequence in expected_sequences},
        }
    positive = selection[TARGET][selection[TARGET] > 0]
    negative = selection[TARGET][selection[TARGET] < 0]
    positive_sum = float(positive.sum())
    negative_mass = float(-negative.sum())
    sequence_utility = {
        sequence: float(selection.loc[selection.seq == sequence, TARGET].sum())
        for sequence in expected_sequences
    }
    covered = sorted(selection.seq.unique())
    return {
        'selected_windows': int(len(selection)),
        'covered_sequences': len(covered),
        'local_positive_windows': int((selection[TARGET] > 0).sum()),
        'local_nonpositive_windows': int((selection[TARGET] <= 0).sum()),
        'local_positive_precision': float((selection[TARGET] > 0).mean()),
        'local_positive_sum': positive_sum,
        'local_negative_mass': negative_mass,
        'local_negative_mass_fraction': negative_mass / max(positive_sum, 1e-12),
        'local_utility_sum': float(selection[TARGET].sum()),
        'worst_sequence_local_utility': min(
            sequence_utility[sequence] for sequence in covered
        ),
        'catastrophic_local_windows': int(
            (selection[TARGET] <= INNER_ELIGIBILITY['catastrophic_local_target_threshold']).sum()
        ),
        'sequence_local_utility': sequence_utility,
    }


def inner_eligible(summary: dict[str, Any]) -> bool:
    rule = INNER_ELIGIBILITY
    return bool(
        summary['selected_windows'] >= rule['selected_windows_min']
        and summary['covered_sequences'] == rule['covered_sequences']
        and summary['local_positive_precision'] >= rule['local_positive_precision_min']
        and summary['local_utility_sum'] > 0
        and summary['worst_sequence_local_utility'] >= 0
        and summary['local_negative_mass_fraction']
        <= rule['negative_mass_max_fraction_of_positive_sum']
        and summary['catastrophic_local_windows'] <= rule['catastrophic_selected_max']
    )


def calibrate_threshold(outer_train: pd.DataFrame) -> tuple[float | None, pd.DataFrame, pd.DataFrame]:
    inner_prediction_parts = []
    inner_sequences = sorted(outer_train.seq.unique())
    for inner_index, inner_held_out in enumerate(inner_sequences):
        inner_train = outer_train[outer_train.seq != inner_held_out].reset_index(drop=True)
        inner_test = outer_train[outer_train.seq == inner_held_out].reset_index(drop=True)
        imputer, models, pair_examples = fit_models(inner_train)
        predictions = score_sequence(inner_test, imputer, models)
        predictions['inner_held_out_seq'] = inner_held_out
        predictions['inner_fold_index'] = inner_index
        predictions['inner_pairwise_train_examples'] = pair_examples
        inner_prediction_parts.append(predictions)
    predictions = pd.concat(inner_prediction_parts, ignore_index=True, sort=False)

    rows = []
    for threshold in THRESHOLDS:
        selection = select_with_threshold(predictions, threshold)
        summary = summarize_local(selection, inner_sequences)
        rows.append({
            'threshold': threshold,
            **{key: value for key, value in summary.items() if key != 'sequence_local_utility'},
            'sequence_local_utility': json.dumps(
                summary['sequence_local_utility'], sort_keys=True
            ),
            'eligible': int(inner_eligible(summary)),
        })
    calibration = pd.DataFrame(rows)
    eligible = calibration[calibration.eligible == 1].sort_values(
        [
            'selected_windows',
            'worst_sequence_local_utility',
            'local_utility_sum',
            'threshold',
        ],
        ascending=[False, False, False, True],
    )
    chosen = float(eligible.iloc[0].threshold) if len(eligible) else None
    return chosen, predictions, calibration


def summarize_global(selection: pd.DataFrame) -> dict[str, Any]:
    if not len(selection):
        return {
            'selected_windows': 0,
            'covered_sequences': 0,
            'positive_windows': 0,
            'negative_windows': 0,
            'zero_windows': 0,
            'positive_precision': 0.0,
            'positive_sum': 0.0,
            'negative_mass': 0.0,
            'negative_mass_fraction': 0.0,
            'utility_sum': 0.0,
            'worst_selected_sequence_utility': 0.0,
            'catastrophic_windows': 0,
            'sequence_utility': {sequence: 0.0 for sequence in SEQUENCES},
        }
    positive = selection.delta_HOTA[selection.delta_HOTA > 0]
    negative = selection.delta_HOTA[selection.delta_HOTA < 0]
    positive_sum = float(positive.sum())
    negative_mass = float(-negative.sum())
    covered = sorted(selection.seq.unique())
    sequence_utility = {
        sequence: float(selection.loc[selection.seq == sequence, 'delta_HOTA'].sum())
        for sequence in SEQUENCES
    }
    return {
        'selected_windows': int(len(selection)),
        'covered_sequences': len(covered),
        'positive_windows': int((selection.delta_HOTA > 0).sum()),
        'negative_windows': int((selection.delta_HOTA < 0).sum()),
        'zero_windows': int((selection.delta_HOTA == 0).sum()),
        'positive_precision': float((selection.delta_HOTA > 0).mean()),
        'positive_sum': positive_sum,
        'negative_mass': negative_mass,
        'negative_mass_fraction': negative_mass / max(positive_sum, 1e-12),
        'utility_sum': float(selection.delta_HOTA.sum()),
        'worst_selected_sequence_utility': min(
            sequence_utility[sequence] for sequence in covered
        ),
        'catastrophic_windows': int(
            (selection.delta_HOTA <= OUTER_DEPLOYMENT_THRESHOLDS['catastrophic_delta_HOTA_threshold']).sum()
        ),
        'sequence_utility': sequence_utility,
    }


def deployment_allowed(
    event_metrics: dict[str, float],
    global_summary: dict[str, Any],
) -> bool:
    rule = OUTER_DEPLOYMENT_THRESHOLDS
    return bool(
        global_summary['selected_windows'] >= rule['selected_windows_min']
        and global_summary['covered_sequences'] >= rule['covered_sequences_min']
        and global_summary['positive_precision'] >= rule['positive_HOTA_precision_min']
        and global_summary['utility_sum'] > 0
        and global_summary['worst_selected_sequence_utility'] >= 0
        and global_summary['negative_mass_fraction']
        <= rule['negative_mass_max_fraction_of_positive_sum']
        and global_summary['catastrophic_windows'] <= rule['catastrophic_windows_max']
        and event_metrics['positive_HOTA_auc'] >= rule['event_positive_HOTA_auc_min']
        and event_metrics['local_positive_auc'] >= rule['event_local_positive_auc_min']
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-features', action='append', required=True)
    parser.add_argument('--train-executability', required=True)
    parser.add_argument('--local-labels', required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()

    frame = load_training_frame(
        args.train_features,
        args.train_executability,
        args.local_labels,
    )
    outer_prediction_parts = []
    outer_selection_parts = []
    inner_prediction_parts = []
    calibration_parts = []
    outer_rows = []

    for outer_index, held_out in enumerate(SEQUENCES):
        outer_train = frame[frame.seq != held_out].reset_index(drop=True)
        outer_test = frame[frame.seq == held_out].reset_index(drop=True)
        threshold, inner_predictions, calibration = calibrate_threshold(outer_train)
        inner_predictions['outer_held_out_seq'] = held_out
        calibration['outer_held_out_seq'] = held_out
        inner_prediction_parts.append(inner_predictions)
        calibration_parts.append(calibration)

        imputer, models, pair_examples = fit_models(outer_train)
        predictions = score_sequence(outer_test, imputer, models)
        predictions['outer_held_out_seq'] = held_out
        predictions['outer_fold_index'] = outer_index
        predictions['chosen_threshold'] = (
            threshold if threshold is not None else math.nan
        )
        outer_prediction_parts.append(predictions)
        if threshold is None:
            selection = pd.DataFrame()
        else:
            selection = select_with_threshold(predictions, threshold)
            selection['outer_held_out_seq'] = held_out
            if len(selection):
                outer_selection_parts.append(selection)
        outer_rows.append({
            'held_out_seq': held_out,
            'chosen_threshold': threshold,
            'fallback_no_op': int(threshold is None),
            'outer_test_events': len(outer_test),
            'outer_selected_windows': len(selection),
            'pairwise_train_examples': pair_examples,
        })

    outer_predictions = (
        pd.concat(outer_prediction_parts, ignore_index=True, sort=False)
        if outer_prediction_parts else pd.DataFrame()
    )
    outer_selection = (
        pd.concat(outer_selection_parts, ignore_index=True, sort=False)
        if outer_selection_parts else pd.DataFrame()
    )
    inner_predictions = pd.concat(inner_prediction_parts, ignore_index=True, sort=False)
    calibrations = pd.concat(calibration_parts, ignore_index=True, sort=False)

    if len(outer_predictions):
        event_metrics = {
            'positive_HOTA_auc': float(
                roc_auc_score(
                    (outer_predictions.delta_HOTA > 0).astype(int),
                    outer_predictions.positive_prob_min,
                )
            ),
            'local_positive_auc': float(
                roc_auc_score(
                    (outer_predictions[TARGET] > 0).astype(int),
                    outer_predictions.positive_prob_min,
                )
            ),
        }
    else:
        event_metrics = {'positive_HOTA_auc': 0.0, 'local_positive_auc': 0.0}
    global_summary = summarize_global(outer_selection)
    allowed = deployment_allowed(event_metrics, global_summary)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    rounded(outer_predictions).to_csv(out_dir / 'outer_event_predictions.csv', index=False)
    rounded(outer_selection).to_csv(out_dir / 'outer_selected_windows.csv', index=False)
    rounded(inner_predictions).to_csv(out_dir / 'inner_event_predictions.csv', index=False)
    rounded(calibrations).to_csv(out_dir / 'inner_threshold_calibration.csv', index=False)
    rounded(pd.DataFrame(outer_rows)).to_csv(out_dir / 'outer_fold_choices.csv', index=False)
    report = {
        'protocol': {
            'scope': 'Nested sequence-disjoint local-positive gate plus pairwise ranker on train ranks21-100 only. No locked artifact is read.',
            'target': TARGET,
            'gate': 'Minimum probability from two ExtraTrees local-positive classifiers.',
            'ranker': 'Mean win score from two ExtraTrees local-target pairwise rankers.',
            'threshold_grid': THRESHOLDS,
            'inner_threshold_selection': 'Choose the highest-coverage eligible threshold using local counterfactual labels only; ties by worst-sequence local utility, total local utility, then lower threshold.',
            'inner_eligibility': INNER_ELIGIBILITY,
            'outer_deployment_thresholds_preregistered': OUTER_DEPLOYMENT_THRESHOLDS,
            'locked_artifacts_read': False,
        },
        'dataset': {
            'events': len(frame),
            'sequences': SEQUENCES,
            'windows': 16,
        },
        'outer_folds': outer_rows,
        'event_metrics': event_metrics,
        'outer_selection': global_summary,
        'deployment_allowed': allowed,
        'next_step': (
            'freeze locked predictions without reading locked utility labels'
            if allowed else
            'keep locked pool untouched and stop this gate family'
        ),
    }
    (out_dir / 'report.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    manifest = {
        path.name: sha256(path)
        for path in sorted(out_dir.iterdir())
        if path.is_file() and path.name != 'manifest.json'
    }
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
