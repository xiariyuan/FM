from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer

from train_loso_future_transaction_utility_ranker import BASE_FEATURES, KEYS, add_symmetric_features


DIRECTION_KEYS = KEYS + ['transaction_type']

DIRECTIONAL_FEATURES = [
    'direction_u_to_v',
    'executor_changed_rows',
    'executor_changed_rows_log1p',
    'executor_handoff_delay',
    'executor_impact_receiver_future',
    'executor_impact_pair_future',
    'donor_age_at_event',
    'receiver_age_at_event',
    'donor_future_life',
    'receiver_future_life',
    'donor_pre_rows',
    'receiver_pre_rows',
    'donor_pre_match_iou',
    'receiver_pre_match_iou',
    'donor_receiver_age_gap',
    'donor_receiver_future_gap',
    'donor_receiver_pre_rows_gap',
    'donor_receiver_pre_iou_gap',
    'donor_ends_before_receiver',
]

COMPACT_BASE_FEATURES = [
    'outer_clean_transaction_score',
    'outer_clean_reciprocal_raw_plus_percentile_ensemble',
    'outer_clean_related_raw_plus_percentile_ensemble',
    'heldout_unary_score',
    'cluster_size',
    'boundary_position_ratio',
    'overlap_max_ioa',
    'overlap_partner_hits',
    'prediction_error_norm',
    'partner_ioa',
    'pair_swap_margin',
    'old_handoff_score',
    'new_source_score',
    'center_dist_norm',
    'bottom_dist_norm',
    'pair_min_future_life',
    'pair_max_future_life',
    'pair_end_gap_abs',
    'future_overlap_total_count',
    'future_overlap_last_offset',
    'future_overlap_max_ioa',
    'future_overlap_mean_ioa',
    'copresent_frac_h30',
    'copresent_frac_h60',
    'copresent_frac_h120',
    'copresent_frac_h300',
    'swap_margin_h30',
    'swap_margin_h60',
    'swap_margin_h120',
    'swap_margin_h300',
    'swap_margin_future_mean',
    'swap_margin_future_min',
    'swap_margin_future_max',
    'swap_margin_positive_horizons',
    'swap_margin_non_decreasing',
]

POLICIES = [
    'q25_positive',
    'q25_p55',
    'median_p60_cat10',
    'consensus_safe',
    'prob_margin_25',
    'prob_margin_40',
]
CAPS = [1, 2, 3]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_directional_features(feature_paths: list[str], executability_path: str) -> pd.DataFrame:
    features = pd.concat([pd.read_csv(path) for path in feature_paths], ignore_index=True, sort=False)
    features = add_symmetric_features(features)
    executable = pd.read_csv(executability_path)
    required_exec = DIRECTION_KEYS + [
        'accepted', 'changed_rows', 'effective_start_frame', 'reject_reason',
    ]
    missing_exec = [column for column in required_exec if column not in executable.columns]
    if missing_exec:
        raise RuntimeError(f'executability missing columns: {missing_exec}')
    executable = executable[required_exec].copy()
    frame = features.merge(executable, on=KEYS, how='inner', validate='one_to_many')
    if len(frame) != len(executable):
        raise RuntimeError(
            f'feature/executability join mismatch: features={len(features)}, '
            f'executability={len(executable)}, joined={len(frame)}'
        )
    if frame[DIRECTION_KEYS].duplicated().any():
        raise RuntimeError('duplicate directional keys after feature join')

    frame['executor_accepted'] = frame['accepted'].astype(int)
    frame['executor_changed_rows'] = frame['changed_rows'].astype(float)
    frame['executor_changed_rows_log1p'] = np.log1p(frame.executor_changed_rows.clip(lower=0))
    frame['direction_u_to_v'] = (frame.transaction_type == 'u_to_v').astype(int)
    valid_types = {'u_to_v', 'v_to_u'}
    if set(frame.transaction_type.unique()) - valid_types:
        raise RuntimeError(f'unknown transaction types: {sorted(set(frame.transaction_type.unique()) - valid_types)}')

    u_to_v = frame.direction_u_to_v.astype(bool)
    donor_prefix = np.where(u_to_v, 'u', 'v')
    receiver_prefix = np.where(u_to_v, 'v', 'u')
    source_columns = {
        'age_at_event': ('u_age_at_event', 'v_age_at_event'),
        'future_life': ('u_future_life', 'v_future_life'),
        'track_start': ('u_track_start', 'v_track_start'),
        'track_end': ('u_track_end', 'v_track_end'),
        'pre_rows': ('u_pre_rows', 'v_pre_rows'),
        'pre_match_iou': ('u_pre_match_iou', 'v_pre_match_iou'),
    }
    for suffix, (u_column, v_column) in source_columns.items():
        frame[f'donor_{suffix}'] = np.where(donor_prefix == 'u', frame[u_column], frame[v_column])
        frame[f'receiver_{suffix}'] = np.where(receiver_prefix == 'u', frame[u_column], frame[v_column])

    frame['executor_handoff_delay'] = frame.effective_start_frame - frame.boundary_frame
    frame['executor_impact_receiver_future'] = (
        frame.executor_changed_rows / (frame.receiver_future_life.fillna(0) + 1.0)
    )
    frame['executor_impact_pair_future'] = (
        frame.executor_changed_rows
        / (frame.donor_future_life.fillna(0) + frame.receiver_future_life.fillna(0) + 1.0)
    )
    frame['donor_receiver_age_gap'] = frame.donor_age_at_event - frame.receiver_age_at_event
    frame['donor_receiver_future_gap'] = frame.donor_future_life - frame.receiver_future_life
    frame['donor_receiver_pre_rows_gap'] = frame.donor_pre_rows - frame.receiver_pre_rows
    frame['donor_receiver_pre_iou_gap'] = frame.donor_pre_match_iou - frame.receiver_pre_match_iou
    frame['donor_ends_before_receiver'] = (frame.donor_track_end < frame.receiver_track_end).astype(int)
    return frame


def attach_train_labels(frame: pd.DataFrame, utility_path: str) -> pd.DataFrame:
    utility = pd.read_csv(utility_path)
    required = DIRECTION_KEYS + ['delta_HOTA']
    missing = [column for column in required if column not in utility.columns]
    if missing:
        raise RuntimeError(f'utility labels missing columns: {missing}')
    labels = utility[required].copy()
    if labels[DIRECTION_KEYS].duplicated().any():
        raise RuntimeError('duplicate directional utility labels')
    joined = frame.merge(labels, on=DIRECTION_KEYS, how='inner', validate='one_to_one')
    if len(joined) != len(frame) or len(joined) != len(labels):
        raise RuntimeError(
            f'train label join mismatch: features={len(frame)}, labels={len(labels)}, joined={len(joined)}'
        )
    return joined


def sequence_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.seq.value_counts()
    mapping = {seq: len(frame) / (len(counts) * count) for seq, count in counts.items()}
    return frame.seq.map(mapping).to_numpy(float)


def normalized_target(frame: pd.DataFrame) -> np.ndarray:
    result = np.zeros(len(frame), dtype=float)
    for _, indices in frame.groupby('seq').groups.items():
        values = frame.loc[indices, 'delta_HOTA'].to_numpy(float)
        scale = max(float(np.mean(np.abs(values))), 0.02)
        positions = frame.index.get_indexer(indices)
        result[positions] = values / scale
    return result


def fit_classifier(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    seed: int,
) -> tuple[ExtraTreesClassifier | None, float | None]:
    unique = np.unique(y)
    if len(unique) == 1:
        return None, float(unique[0])
    model = ExtraTreesClassifier(
        n_estimators=800,
        max_depth=7,
        min_samples_leaf=4,
        max_features=0.65,
        class_weight='balanced',
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(x, y, sample_weight=weights)
    return model, None


def classifier_probability(
    model: ExtraTreesClassifier | None,
    constant: float | None,
    x: np.ndarray,
) -> np.ndarray:
    if model is None:
        return np.full(len(x), float(constant), dtype=float)
    return model.predict_proba(x)[:, 1]


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str], seed: int) -> pd.DataFrame:
    accepted_train = train[train.executor_accepted == 1].copy().reset_index(drop=True)
    if len(accepted_train) < 20:
        raise RuntimeError(f'too few accepted training events: {len(accepted_train)}')
    imputer = SimpleImputer(strategy='median')
    x_train = imputer.fit_transform(accepted_train[features])
    x_test = imputer.transform(test[features])
    y_raw = accepted_train.delta_HOTA.to_numpy(float)
    y_norm = normalized_target(accepted_train)
    weights = sequence_weights(accepted_train)

    expected = ExtraTreesRegressor(
        n_estimators=1000,
        max_depth=7,
        min_samples_leaf=4,
        max_features=0.65,
        random_state=seed,
        n_jobs=-1,
    )
    q25 = HistGradientBoostingRegressor(
        loss='quantile',
        quantile=0.25,
        max_iter=300,
        learning_rate=0.035,
        max_leaf_nodes=7,
        min_samples_leaf=10,
        l2_regularization=6.0,
        random_state=seed + 1000,
    )
    median = HistGradientBoostingRegressor(
        loss='quantile',
        quantile=0.50,
        max_iter=300,
        learning_rate=0.035,
        max_leaf_nodes=7,
        min_samples_leaf=10,
        l2_regularization=6.0,
        random_state=seed + 2000,
    )
    expected.fit(x_train, np.clip(y_norm, -3.0, 3.0), sample_weight=weights)
    q25.fit(x_train, y_norm, sample_weight=weights)
    median.fit(x_train, y_norm, sample_weight=weights)

    positive_model, positive_constant = fit_classifier(
        x_train, (y_raw > 0).astype(int), weights, seed + 3000
    )
    negative_model, negative_constant = fit_classifier(
        x_train, (y_raw < 0).astype(int), weights, seed + 4000
    )
    catastrophic_model, catastrophic_constant = fit_classifier(
        x_train, (y_raw <= -0.10).astype(int), weights, seed + 5000
    )

    result = test[DIRECTION_KEYS + ['executor_accepted', 'outer_clean_transaction_score']].copy()
    result['pred_expected'] = expected.predict(x_test)
    result['pred_q25'] = q25.predict(x_test)
    result['pred_median'] = median.predict(x_test)
    result['prob_positive'] = classifier_probability(positive_model, positive_constant, x_test)
    result['prob_negative'] = classifier_probability(negative_model, negative_constant, x_test)
    result['prob_catastrophic'] = classifier_probability(catastrophic_model, catastrophic_constant, x_test)
    result['prob_margin'] = (
        result.prob_positive - result.prob_negative - 2.0 * result.prob_catastrophic
    )
    return result


def policy_mask(frame: pd.DataFrame, policy: str) -> pd.Series:
    accepted = frame.executor_accepted == 1
    if policy == 'q25_positive':
        return accepted & (frame.pred_q25 > 0)
    if policy == 'q25_p55':
        return accepted & (frame.pred_q25 > 0) & (frame.prob_positive >= 0.55)
    if policy == 'median_p60_cat10':
        return (
            accepted
            & (frame.pred_median > 0)
            & (frame.prob_positive >= 0.60)
            & (frame.prob_catastrophic <= 0.10)
        )
    if policy == 'consensus_safe':
        return (
            accepted
            & (frame.pred_expected > 0)
            & (frame.pred_median > 0)
            & (frame.prob_positive >= 0.60)
            & (frame.prob_negative <= 0.40)
            & (frame.prob_catastrophic <= 0.10)
        )
    if policy == 'prob_margin_25':
        return accepted & (frame.prob_margin >= 0.25)
    if policy == 'prob_margin_40':
        return accepted & (frame.prob_margin >= 0.40)
    raise ValueError(policy)


def policy_score(frame: pd.DataFrame, policy: str) -> pd.Series:
    if policy.startswith('q25'):
        return frame.pred_q25 + 0.25 * frame.prob_margin
    if policy.startswith('median') or policy == 'consensus_safe':
        return frame.pred_median + 0.25 * frame.prob_margin
    return frame.prob_margin + 0.10 * frame.pred_expected


def conflict_free(frame: pd.DataFrame, policy: str, cap: int) -> pd.DataFrame:
    candidates = frame[policy_mask(frame, policy)].copy()
    if not len(candidates):
        return candidates
    candidates['selection_score'] = policy_score(candidates, policy)
    candidates = candidates.sort_values('selection_score', ascending=False)
    selected = []
    used_tracks: set[int] = set()
    for row in candidates.to_dict('records'):
        u, v = int(row['u']), int(row['v'])
        if u in used_tracks or v in used_tracks:
            continue
        selected.append(row)
        used_tracks.update([u, v])
        if len(selected) >= cap:
            break
    return pd.DataFrame(selected)


def evaluate_inner_predictions(predictions: pd.DataFrame, variant: str) -> list[dict]:
    rows = []
    sequences = sorted(predictions.seq.unique())
    for policy in POLICIES:
        for cap in CAPS:
            selected_parts = []
            for _, group in predictions.groupby('seq'):
                selected = conflict_free(group, policy, cap)
                if len(selected):
                    selected_parts.append(selected)
            selected = (
                pd.concat(selected_parts, ignore_index=True, sort=False)
                if selected_parts
                else pd.DataFrame()
            )
            sequence_utility = {seq: 0.0 for seq in sequences}
            if len(selected):
                for seq, group in selected.groupby('seq'):
                    sequence_utility[seq] = float(group.delta_HOTA.sum())
            positive_sum = (
                float(selected.loc[selected.delta_HOTA > 0, 'delta_HOTA'].sum())
                if len(selected)
                else 0.0
            )
            negative_mass = (
                float(-selected.loc[selected.delta_HOTA < 0, 'delta_HOTA'].sum())
                if len(selected)
                else 0.0
            )
            worst_sequence = min(sequence_utility.values()) if sequence_utility else 0.0
            positive_precision = float((selected.delta_HOTA > 0).mean()) if len(selected) else 0.0
            robust_score = positive_sum - 4.0 * negative_mass + 2.0 * min(0.0, worst_sequence)
            eligible = bool(
                len(selected) > 0
                and positive_precision >= 0.60
                and negative_mass <= max(0.02, 0.50 * positive_sum)
                and worst_sequence >= -0.02
            )
            rows.append({
                'variant': variant,
                'policy': policy,
                'cap': cap,
                'selected': len(selected),
                'positive': int((selected.delta_HOTA > 0).sum()) if len(selected) else 0,
                'negative': int((selected.delta_HOTA < 0).sum()) if len(selected) else 0,
                'positive_precision': positive_precision,
                'positive_sum': positive_sum,
                'negative_mass': negative_mass,
                'raw_utility_sum': float(selected.delta_HOTA.sum()) if len(selected) else 0.0,
                'worst_sequence_utility': worst_sequence,
                'robust_score': robust_score,
                'eligible': int(eligible),
                'sequence_utility': json.dumps(sequence_utility, sort_keys=True),
            })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-features', action='append', required=True)
    ap.add_argument('--test-features', action='append', required=True)
    ap.add_argument('--train-executability', required=True)
    ap.add_argument('--test-executability', required=True)
    ap.add_argument('--train-utility', required=True)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    train = attach_train_labels(
        load_directional_features(args.train_features, args.train_executability),
        args.train_utility,
    )
    test = load_directional_features(args.test_features, args.test_executability)
    if set(train.seq.unique()) != set(test.seq.unique()):
        raise RuntimeError('train/test sequence sets differ')

    variants = {
        'compact_directional': list(dict.fromkeys(COMPACT_BASE_FEATURES + DIRECTIONAL_FEATURES)),
        'full_directional': list(dict.fromkeys(BASE_FEATURES + DIRECTIONAL_FEATURES)),
    }
    for name, features in variants.items():
        missing = [feature for feature in features if feature not in train.columns or feature not in test.columns]
        if missing:
            raise RuntimeError(f'{name} missing features: {missing}')

    sequences = sorted(train.seq.unique())
    calibration_rows = []
    outer_prediction_parts = []
    selected_parts = []
    fold_rows = []

    for outer_index, held_out in enumerate(sequences):
        outer_train = train[train.seq != held_out].copy().reset_index(drop=True)
        outer_test = test[test.seq == held_out].copy().reset_index(drop=True)
        candidates = []
        for variant_index, (variant, feature_names) in enumerate(variants.items()):
            inner_parts = []
            for inner_index, inner_held_out in enumerate(sorted(outer_train.seq.unique())):
                inner_train = outer_train[outer_train.seq != inner_held_out].copy().reset_index(drop=True)
                inner_test = outer_train[outer_train.seq == inner_held_out].copy().reset_index(drop=True)
                prediction = fit_predict(
                    inner_train,
                    inner_test,
                    feature_names,
                    seed=23000 + 1000 * outer_index + 100 * variant_index + inner_index,
                )
                prediction = prediction.merge(
                    inner_test[DIRECTION_KEYS + ['delta_HOTA']],
                    on=DIRECTION_KEYS,
                    how='left',
                    validate='one_to_one',
                )
                inner_parts.append(prediction)
            inner_predictions = pd.concat(inner_parts, ignore_index=True, sort=False)
            inner_results = evaluate_inner_predictions(inner_predictions, variant)
            for row in inner_results:
                row['held_out_seq'] = held_out
            calibration_rows.extend(inner_results)
            candidates.extend(inner_results)

        candidate_frame = pd.DataFrame(candidates)
        eligible = candidate_frame[candidate_frame.eligible == 1].copy()
        if len(eligible):
            chosen = eligible.sort_values(
                ['robust_score', 'raw_utility_sum', 'positive_precision', 'selected'],
                ascending=[False, False, False, True],
            ).iloc[0].to_dict()
            chosen_variant = str(chosen['variant'])
            chosen_policy = str(chosen['policy'])
            chosen_cap = int(chosen['cap'])
            no_op = False
        else:
            chosen_variant = 'compact_directional'
            chosen_policy = 'no_op'
            chosen_cap = 0
            no_op = True

        outer_prediction = fit_predict(
            outer_train,
            outer_test,
            variants[chosen_variant],
            seed=29000 + outer_index,
        )
        outer_prediction['chosen_variant'] = chosen_variant
        outer_prediction['chosen_policy'] = chosen_policy
        outer_prediction['chosen_cap'] = chosen_cap
        outer_prediction['selected_by_gate'] = 0
        selected = pd.DataFrame()
        if not no_op:
            selected = conflict_free(outer_prediction, chosen_policy, chosen_cap)
        if len(selected):
            selected_keys = set(selected[DIRECTION_KEYS].itertuples(index=False, name=None))
            outer_prediction['selected_by_gate'] = [
                int(key in selected_keys)
                for key in outer_prediction[DIRECTION_KEYS].itertuples(index=False, name=None)
            ]
            selected = selected.copy()
            selected['selected_by_gate'] = 1
            selected['chosen_variant'] = chosen_variant
            selected['chosen_policy'] = chosen_policy
            selected['chosen_cap'] = chosen_cap
            selected_parts.append(selected)
        outer_prediction_parts.append(outer_prediction)
        fold_rows.append({
            'held_out_seq': held_out,
            'chosen_variant': chosen_variant,
            'chosen_policy': chosen_policy,
            'chosen_cap': chosen_cap,
            'inner_eligible_policies': int(len(eligible)),
            'outer_selected': int(len(selected)),
            'fallback_no_op': int(no_op),
        })

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    predictions = pd.concat(outer_prediction_parts, ignore_index=True, sort=False)
    selected = (
        pd.concat(selected_parts, ignore_index=True, sort=False)
        if selected_parts
        else pd.DataFrame(columns=predictions.columns)
    )
    prediction_path = out / 'locked_test_predictions.csv'
    selected_path = out / 'selected_transactions.csv'
    predictions.to_csv(prediction_path, index=False)
    selected.to_csv(selected_path, index=False)
    pd.DataFrame(calibration_rows).to_csv(out / 'inner_calibration.csv', index=False)
    pd.DataFrame(fold_rows).to_csv(out / 'fold_selection.csv', index=False)

    report = {
        'protocol': {
            'scope': (
                'Directional handoff locked LOSO: train utility on frozen ranks 21-100 from '
                'the three non-held-out sequences; select only from frozen ranks 1-20 of the held-out sequence.'
            ),
            'locked_label_policy': (
                'This script never accepts or reads a locked rank1-20 utility/TrackEval label file.'
            ),
            'inner_selection': (
                'Leave-one-training-sequence-out chooses a whitelisted feature variant, fixed gate, and cap. '
                'If no policy satisfies conservative eligibility constraints, the fold is a no-op.'
            ),
            'eligibility': (
                'Positive precision >=0.60, negative mass <= max(0.02, 0.5*positive sum), '
                'worst training-sequence utility >= -0.02.'
            ),
            'executor_features': (
                'Only deterministic directional replay metadata available before TrackEval is used; '
                'GT/evaluation columns are excluded by an explicit feature whitelist.'
            ),
            'conflict_policy': (
                'At most the calibrated cap per sequence; selected transactions may not share either track ID, '
                'which also prevents choosing both directions of the same pair.'
            ),
            'feature_variants': {name: len(features) for name, features in variants.items()},
            'policies': POLICIES,
            'caps': CAPS,
        },
        'dataset': {
            'training_directional_events': len(train),
            'training_executable': int(train.executor_accepted.sum()),
            'training_positive': int((train.delta_HOTA > 0).sum()),
            'training_negative': int((train.delta_HOTA < 0).sum()),
            'locked_directional_events': len(test),
            'locked_executable': int(test.executor_accepted.sum()),
        },
        'folds': fold_rows,
        'selected_transactions': int(len(selected)),
    }
    report_path = out / 'report.json'
    report_path.write_text(json.dumps(report, indent=2) + '\n')

    input_paths = [
        *[Path(path) for path in args.train_features],
        *[Path(path) for path in args.test_features],
        Path(args.train_executability),
        Path(args.test_executability),
        Path(args.train_utility),
    ]
    manifest = {
        'predictions_frozen_before_locked_evaluation': True,
        'locked_utility_or_trackeval_labels_read': False,
        'directional_keys': DIRECTION_KEYS,
        'output_sha256': {
            prediction_path.name: sha256(prediction_path),
            selected_path.name: sha256(selected_path),
            report_path.name: sha256(report_path),
        },
        'input_sha256': {str(path): sha256(path) for path in input_paths},
    }
    (out / 'prediction_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({**report, 'prediction_manifest': manifest['output_sha256']}, indent=2))


if __name__ == '__main__':
    main()
