from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer

from train_loso_future_transaction_utility_ranker import (
    BASE_FEATURES,
    KEYS,
    add_symmetric_features,
    normalized_target,
    sequence_weights,
)

POLICIES = [
    'lower_q10_positive',
    'lower_q20_positive_p50',
    'lower_q30_consensus',
    'consensus_p55_n45',
    'uncertainty_positive_p50',
]
CAPS = [1, 2, 3, 5]


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str], seed: int) -> pd.DataFrame:
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    imputer = SimpleImputer(strategy='median')
    x_train = imputer.fit_transform(train[features])
    x_test = imputer.transform(test[features])
    y_train_raw = train.delta_HOTA.to_numpy(float)
    y_train = normalized_target(train)
    positive = (y_train_raw > 0).astype(int)
    negative = (y_train_raw < 0).astype(int)
    weights = sequence_weights(train)

    et = ExtraTreesRegressor(
        n_estimators=1000,
        max_depth=6,
        min_samples_leaf=3,
        max_features=0.65,
        random_state=seed,
        n_jobs=-1,
    )
    hgb = HistGradientBoostingRegressor(
        max_iter=250,
        learning_rate=0.035,
        max_leaf_nodes=7,
        min_samples_leaf=8,
        l2_regularization=5.0,
        random_state=seed + 1000,
    )
    pos = ExtraTreesClassifier(
        n_estimators=1000,
        max_depth=6,
        min_samples_leaf=3,
        max_features=0.65,
        class_weight='balanced',
        random_state=seed + 2000,
        n_jobs=-1,
    )
    neg = ExtraTreesClassifier(
        n_estimators=1000,
        max_depth=6,
        min_samples_leaf=3,
        max_features=0.65,
        class_weight='balanced',
        random_state=seed + 3000,
        n_jobs=-1,
    )
    et.fit(x_train, y_train, sample_weight=weights)
    hgb.fit(x_train, y_train, sample_weight=weights)
    pos.fit(x_train, positive, sample_weight=weights)
    neg.fit(x_train, negative, sample_weight=weights)

    pred_et = et.predict(x_test)
    pred_hgb = hgb.predict(x_test)
    tree_predictions = np.stack([tree.predict(x_test) for tree in et.estimators_], axis=0)
    et_std = tree_predictions.std(axis=0)
    result = test[KEYS + ['delta_HOTA', 'accepted', 'outer_clean_transaction_score']].copy()
    result['pred_et'] = pred_et
    result['pred_hgb'] = pred_hgb
    result['pred_mean'] = 0.5 * pred_et + 0.5 * pred_hgb
    result['pred_et_std'] = et_std
    result['prob_positive'] = pos.predict_proba(x_test)[:, 1]
    result['prob_negative'] = neg.predict_proba(x_test)[:, 1]
    return result


def normalized_actual(frame: pd.DataFrame) -> np.ndarray:
    values = np.zeros(len(frame), dtype=float)
    for _, indices in frame.groupby('seq').groups.items():
        y = frame.loc[indices, 'delta_HOTA'].to_numpy(float)
        scale = max(float(np.mean(np.abs(y))), 0.02)
        positions = frame.index.get_indexer(indices)
        values[positions] = y / scale
    return values


def add_bounds(frame: pd.DataFrame, residual_quantiles: dict[str, float]) -> pd.DataFrame:
    frame = frame.copy()
    for name, residual in residual_quantiles.items():
        frame[f'lower_{name}'] = frame.pred_mean + residual
    frame['uncertainty_score'] = frame.pred_mean - frame.pred_et_std
    frame['consensus_score'] = frame[['pred_et', 'pred_hgb']].min(axis=1)
    return frame


def policy_mask(frame: pd.DataFrame, policy: str) -> pd.Series:
    if policy == 'lower_q10_positive':
        return frame.lower_q10 > 0
    if policy == 'lower_q20_positive_p50':
        return (frame.lower_q20 > 0) & (frame.prob_positive >= 0.50)
    if policy == 'lower_q30_consensus':
        return (frame.lower_q30 > 0) & (frame.pred_et > 0) & (frame.pred_hgb > 0)
    if policy == 'consensus_p55_n45':
        return (
            (frame.pred_et > 0)
            & (frame.pred_hgb > 0)
            & (frame.prob_positive >= 0.55)
            & (frame.prob_negative <= 0.45)
        )
    if policy == 'uncertainty_positive_p50':
        return (frame.uncertainty_score > 0) & (frame.prob_positive >= 0.50)
    raise ValueError(policy)


def policy_score(frame: pd.DataFrame, policy: str) -> pd.Series:
    if policy == 'lower_q10_positive':
        return frame.lower_q10
    if policy == 'lower_q20_positive_p50':
        return frame.lower_q20 + 0.10 * frame.prob_positive
    if policy == 'lower_q30_consensus':
        return frame.consensus_score + 0.10 * frame.prob_positive
    if policy == 'consensus_p55_n45':
        return frame.consensus_score + 0.20 * (frame.prob_positive - frame.prob_negative)
    if policy == 'uncertainty_positive_p50':
        return frame.uncertainty_score + 0.10 * frame.prob_positive
    raise ValueError(policy)


def select_conflict_free(frame: pd.DataFrame, policy: str, cap: int) -> pd.DataFrame:
    candidates = frame[policy_mask(frame, policy)].copy()
    candidates['policy_score'] = policy_score(candidates, policy)
    candidates = candidates.sort_values('policy_score', ascending=False)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', action='append', required=True)
    parser.add_argument('--utility-csv', required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()

    feature_frame = pd.concat([pd.read_csv(path) for path in args.features], ignore_index=True, sort=False)
    feature_frame = add_symmetric_features(feature_frame)
    utility = pd.read_csv(args.utility_csv)
    labels = utility[KEYS + ['accepted', 'delta_HOTA']]
    data = feature_frame.merge(labels, on=KEYS, how='inner', validate='one_to_one')
    if len(data) != len(feature_frame) or len(data) != len(utility):
        raise RuntimeError('feature/utility join is not exact')
    missing = [feature for feature in BASE_FEATURES if feature not in data.columns]
    if missing:
        raise RuntimeError(f'missing features: {missing}')

    features = list(BASE_FEATURES)
    sequences = sorted(data.seq.unique())
    outer_rows = []
    selected_rows = []
    calibration_rows = []

    for outer_index, held_out in enumerate(sequences):
        outer_train = data[data.seq != held_out].copy().reset_index(drop=True)
        outer_test = data[data.seq == held_out].copy().reset_index(drop=True)

        inner_parts = []
        inner_sequences = sorted(outer_train.seq.unique())
        for inner_index, inner_held_out in enumerate(inner_sequences):
            inner_train = outer_train[outer_train.seq != inner_held_out].copy().reset_index(drop=True)
            inner_test = outer_train[outer_train.seq == inner_held_out].copy().reset_index(drop=True)
            inner_pred = fit_predict(
                inner_train,
                inner_test,
                features,
                seed=9000 + 100 * outer_index + inner_index,
            )
            inner_parts.append(inner_pred)
        inner = pd.concat(inner_parts, ignore_index=True, sort=False)
        inner['actual_normalized'] = normalized_actual(inner)
        inner['residual'] = inner.actual_normalized - inner.pred_mean
        residual_quantiles = {
            'q10': float(inner.residual.quantile(0.10)),
            'q20': float(inner.residual.quantile(0.20)),
            'q30': float(inner.residual.quantile(0.30)),
        }
        inner = add_bounds(inner, residual_quantiles)

        policy_results = []
        for policy in POLICIES:
            for cap in CAPS:
                selected_parts = []
                for seq, group in inner.groupby('seq'):
                    selected = select_conflict_free(group, policy, cap)
                    if len(selected):
                        selected_parts.append(selected)
                selected = pd.concat(selected_parts, ignore_index=True, sort=False) if selected_parts else pd.DataFrame()
                seq_sums = {seq: 0.0 for seq in inner_sequences}
                seq_raw_sums = {seq: 0.0 for seq in inner_sequences}
                if len(selected):
                    for seq, group in selected.groupby('seq'):
                        scale = max(float(np.mean(np.abs(inner[inner.seq == seq].delta_HOTA))), 0.02)
                        seq_sums[seq] = float((group.delta_HOTA / scale).sum())
                        seq_raw_sums[seq] = float(group.delta_HOTA.sum())
                values = list(seq_sums.values())
                robust_score = float(sum(values) + 2.0 * min(0.0, min(values)))
                positive_precision = float((selected.delta_HOTA > 0).mean()) if len(selected) else 0.0
                policy_results.append({
                    'held_out_seq': held_out,
                    'policy': policy,
                    'cap': cap,
                    'selected': len(selected),
                    'positive_precision': positive_precision,
                    'raw_utility_sum': float(selected.delta_HOTA.sum()) if len(selected) else 0.0,
                    'normalized_utility_sum': float(sum(values)),
                    'min_sequence_normalized_utility': float(min(values)),
                    'robust_score': robust_score,
                    'residual_q10': residual_quantiles['q10'],
                    'residual_q20': residual_quantiles['q20'],
                    'residual_q30': residual_quantiles['q30'],
                })
        policy_frame = pd.DataFrame(policy_results)
        chosen = policy_frame.sort_values(
            ['robust_score', 'raw_utility_sum', 'positive_precision', 'selected'],
            ascending=[False, False, False, True],
        ).iloc[0].to_dict()
        calibration_rows.extend(policy_results)

        outer_prediction = fit_predict(
            outer_train,
            outer_test,
            features,
            seed=12000 + outer_index,
        )
        outer_prediction = add_bounds(outer_prediction, residual_quantiles)
        selected = select_conflict_free(
            outer_prediction,
            str(chosen['policy']),
            int(chosen['cap']),
        )
        outer_prediction['chosen_policy'] = str(chosen['policy'])
        outer_prediction['chosen_cap'] = int(chosen['cap'])
        outer_prediction['selected_by_gate'] = 0
        if len(selected):
            selected_keys = set(
                tuple(row) for row in selected[KEYS].itertuples(index=False, name=None)
            )
            outer_prediction['selected_by_gate'] = [
                int(tuple(row) in selected_keys)
                for row in outer_prediction[KEYS].itertuples(index=False, name=None)
            ]
            selected = selected.copy()
            selected['chosen_policy'] = str(chosen['policy'])
            selected['chosen_cap'] = int(chosen['cap'])
            selected_rows.append(selected)
        outer_rows.append(outer_prediction)

    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    outer = pd.concat(outer_rows, ignore_index=True, sort=False)
    outer.to_csv(output / 'outer_predictions.csv', index=False)
    calibration = pd.DataFrame(calibration_rows)
    calibration.to_csv(output / 'inner_policy_calibration.csv', index=False)
    selected = pd.concat(selected_rows, ignore_index=True, sort=False) if selected_rows else pd.DataFrame()
    selected.to_csv(output / 'selected_transactions.csv', index=False)

    fold_summary = []
    for seq, group in outer.groupby('seq'):
        chosen = group[group.selected_by_gate == 1]
        fold_summary.append({
            'held_out_seq': seq,
            'chosen_policy': str(group.chosen_policy.iloc[0]),
            'chosen_cap': int(group.chosen_cap.iloc[0]),
            'selected': len(chosen),
            'positive': int((chosen.delta_HOTA > 0).sum()) if len(chosen) else 0,
            'negative': int((chosen.delta_HOTA < 0).sum()) if len(chosen) else 0,
            'positive_precision': float((chosen.delta_HOTA > 0).mean()) if len(chosen) else None,
            'isolated_utility_sum': float(chosen.delta_HOTA.sum()) if len(chosen) else 0.0,
            'best_selected_utility': float(chosen.delta_HOTA.max()) if len(chosen) else None,
            'worst_selected_utility': float(chosen.delta_HOTA.min()) if len(chosen) else None,
        })
    report = {
        'protocol': {
            'scope': 'Nested sequence-level risk-gated future transaction utility pilot',
            'outer_validation': 'Held-out sequence utility labels are excluded from feature preprocessing, model fitting, residual calibration, policy selection, and cap selection.',
            'inner_validation': 'The other three sequences are internally leave-one-sequence-out to estimate utility residual quantiles and choose among five pre-defined gates and four caps.',
            'selection': 'Greedy conflict-free by raw track ID; no track may participate in more than one permanent transaction.',
            'objective': 'Inner normalized utility sum with an additional 2x penalty for the worst negative sequence utility.',
            'sample_size_warning': 'Top-20 per sequence only; policy estimates have high variance.',
            'features': features,
            'policies': POLICIES,
            'caps': CAPS,
        },
        'folds': fold_summary,
        'aggregate': {
            'selected': int(len(selected)),
            'positive': int((selected.delta_HOTA > 0).sum()) if len(selected) else 0,
            'negative': int((selected.delta_HOTA < 0).sum()) if len(selected) else 0,
            'positive_precision': float((selected.delta_HOTA > 0).mean()) if len(selected) else None,
            'isolated_utility_sum': float(selected.delta_HOTA.sum()) if len(selected) else 0.0,
        },
    }
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
