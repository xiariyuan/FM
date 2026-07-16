from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

from train_locked_loso_directional_handoff_utility import (
    DIRECTIONAL_FEATURES,
    DIRECTION_KEYS,
    load_directional_features,
)
from train_loso_future_transaction_utility_ranker import BASE_FEATURES


SEQUENCES = ['MOT20-01', 'MOT20-02', 'MOT20-03', 'MOT20-05']
TRAIN_WINDOWS = [(21, 40), (41, 60), (61, 80), (81, 100)]
TARGET = 'full_idtp_delta_norm'
MODEL_FEATURES = list(dict.fromkeys(BASE_FEATURES + DIRECTIONAL_FEATURES))
COMPONENTS = [
    'extra_reg_s17',
    'extra_reg_s53',
    'hist_reg',
    'extra_pair_s17',
    'extra_pair_s53',
    'hist_pair',
]
DEPLOYMENT_THRESHOLDS = {
    'selected_windows': 16,
    'positive_windows_min': 12,
    'utility_sum_strictly_positive': True,
    'worst_sequence_utility_nonnegative': True,
    'negative_mass_max_fraction_of_positive_sum': 0.20,
    'catastrophic_windows_max': 0,
    'catastrophic_delta_HOTA_threshold': -0.05,
    'event_target_spearman_min': 0.35,
    'positive_HOTA_auc_min': 0.70,
}


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


def assert_feature_whitelist() -> None:
    forbidden_tokens = ['delta_', 'trackeval', 'ground_truth', 'gt_', 'matched_gt', 'row_class']
    forbidden = [
        feature
        for feature in MODEL_FEATURES
        if any(token in feature.lower() for token in forbidden_tokens)
    ]
    if forbidden:
        raise RuntimeError(f'forbidden label-derived model features: {forbidden}')


def load_training_frame(
    feature_paths: list[str],
    executability_path: str,
    local_labels_path: str,
) -> pd.DataFrame:
    features = load_directional_features(feature_paths, executability_path)
    labels = pd.read_csv(local_labels_path)
    required = DIRECTION_KEYS + [TARGET, 'delta_HOTA', 'delta_AssA', 'delta_IDF1']
    missing = [column for column in required if column not in labels.columns]
    if missing:
        raise RuntimeError(f'local labels missing columns: {missing}')
    if labels[DIRECTION_KEYS].duplicated().any():
        raise RuntimeError('duplicate local-label directional keys')
    labels = labels[required].copy()
    frame = features.merge(labels, on=DIRECTION_KEYS, how='inner', validate='one_to_one')
    if len(frame) != len(labels):
        raise RuntimeError(
            f'feature/local-label join mismatch: labels={len(labels)}, joined={len(frame)}'
        )
    if len(frame) != 225:
        raise RuntimeError(f'expected 225 executable labeled events, found {len(frame)}')
    if not (frame.executor_accepted == 1).all():
        raise RuntimeError('local labels joined to non-executable events')
    missing_features = [feature for feature in MODEL_FEATURES if feature not in frame.columns]
    if missing_features:
        raise RuntimeError(f'missing model features: {missing_features}')
    return frame.reset_index(drop=True)


def sequence_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.seq.value_counts()
    return frame.seq.map(
        {sequence: len(frame) / (len(counts) * count) for sequence, count in counts.items()}
    ).to_numpy(float)


def attach_sequence_robust_target(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy().reset_index(drop=True)
    normalized = np.zeros(len(result), dtype=float)
    for _, indices in result.groupby('seq').groups.items():
        values = result.loc[indices, TARGET].to_numpy(float)
        median = float(np.median(values))
        q25, q75 = np.quantile(values, [0.25, 0.75])
        scale = max(float((q75 - q25) / 1.349), 0.05)
        normalized[result.index.get_indexer(indices)] = (values - median) / scale
    result[f'{TARGET}_seqnorm'] = normalized
    return result


def window_mask(frame: pd.DataFrame, start: int, end: int) -> pd.Series:
    return frame.canonical_rank.between(start, end)


def build_pairwise_examples(
    frame: pd.DataFrame,
    transformed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    work = frame.reset_index(drop=True)
    x_parts: list[np.ndarray] = []
    y_parts: list[int] = []
    weight_parts: list[float] = []
    norm_column = f'{TARGET}_seqnorm'
    for _, sequence_frame in work.groupby('seq', sort=True):
        for start, end in TRAIN_WINDOWS:
            indices = sequence_frame.index[
                sequence_frame.canonical_rank.between(start, end)
            ].to_numpy()
            if len(indices) < 2:
                continue
            raw_values = work.loc[indices, TARGET].to_numpy(float)
            norm_values = work.loc[indices, norm_column].to_numpy(float)
            window_x: list[np.ndarray] = []
            window_y: list[int] = []
            window_weights: list[float] = []
            for left in range(len(indices)):
                for right in range(left + 1, len(indices)):
                    raw_left = float(raw_values[left])
                    raw_right = float(raw_values[right])
                    if abs(raw_left - raw_right) < 1e-12:
                        continue
                    difference = transformed[indices[left]] - transformed[indices[right]]
                    label = int(raw_left > raw_right)
                    normalized_gap = abs(float(norm_values[left] - norm_values[right]))
                    weight = min(normalized_gap, 4.0) + 0.05
                    if (raw_left > 0) != (raw_right > 0):
                        weight *= 2.0
                    window_x.extend([difference, -difference])
                    window_y.extend([label, 1 - label])
                    window_weights.extend([weight, weight])
            if not window_x:
                continue
            weights = np.asarray(window_weights, dtype=float)
            weights *= len(weights) / max(float(weights.sum()), 1e-12)
            x_parts.extend(window_x)
            y_parts.extend(window_y)
            weight_parts.extend(weights.tolist())
    if not x_parts:
        raise RuntimeError('no pairwise examples generated')
    return np.asarray(x_parts), np.asarray(y_parts), np.asarray(weight_parts)


def fit_components(train: pd.DataFrame) -> tuple[SimpleImputer, dict[str, Any], int]:
    work = attach_sequence_robust_target(train)
    imputer = SimpleImputer(strategy='median')
    x_train = imputer.fit_transform(work[MODEL_FEATURES])
    weights = sequence_weights(work)
    y_reg = work[f'{TARGET}_seqnorm'].to_numpy(float)

    models: dict[str, Any] = {}
    for seed in [17, 53]:
        model = ExtraTreesRegressor(
            n_estimators=800,
            max_depth=8,
            min_samples_leaf=3,
            max_features=0.70,
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(x_train, y_reg, sample_weight=weights)
        models[f'extra_reg_s{seed}'] = model

    hist_reg = HistGradientBoostingRegressor(
        max_iter=350,
        learning_rate=0.035,
        max_leaf_nodes=15,
        min_samples_leaf=10,
        l2_regularization=6.0,
        random_state=17,
    )
    hist_reg.fit(x_train, y_reg, sample_weight=weights)
    models['hist_reg'] = hist_reg

    pair_x, pair_y, pair_weights = build_pairwise_examples(work, x_train)
    for seed in [17, 53]:
        model = ExtraTreesClassifier(
            n_estimators=700,
            max_depth=8,
            min_samples_leaf=3,
            max_features=0.70,
            class_weight='balanced',
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(pair_x, pair_y, sample_weight=pair_weights)
        models[f'extra_pair_s{seed}'] = model

    hist_pair = HistGradientBoostingClassifier(
        max_iter=350,
        learning_rate=0.035,
        max_leaf_nodes=15,
        min_samples_leaf=10,
        l2_regularization=6.0,
        random_state=17,
    )
    hist_pair.fit(pair_x, pair_y, sample_weight=pair_weights)
    models['hist_pair'] = hist_pair
    return imputer, models, len(pair_x)


def score_pairwise_group(model: Any, transformed: np.ndarray) -> np.ndarray:
    count = len(transformed)
    if count == 1:
        return np.asarray([0.5], dtype=float)
    scores = np.zeros(count, dtype=float)
    for index in range(count):
        opponents = [other for other in range(count) if other != index]
        differences = transformed[index] - transformed[opponents]
        probabilities = model.predict_proba(differences)[:, 1]
        scores[index] = float(np.mean(probabilities))
    return scores


def percentile_score(values: pd.Series) -> pd.Series:
    if len(values) <= 1:
        return pd.Series([0.5] * len(values), index=values.index, dtype=float)
    ranks = values.rank(method='average', ascending=True)
    return (ranks - 1.0) / (len(values) - 1.0)


def score_sequence(
    test: pd.DataFrame,
    imputer: SimpleImputer,
    models: dict[str, Any],
) -> pd.DataFrame:
    result_parts: list[pd.DataFrame] = []
    for start, end in TRAIN_WINDOWS:
        group = test[window_mask(test, start, end)].copy().reset_index(drop=True)
        if not len(group):
            raise RuntimeError(f'empty held-out window {start}-{end} for {test.seq.iloc[0]}')
        transformed = imputer.transform(group[MODEL_FEATURES])
        group['rank_start'] = start
        group['rank_end'] = end
        group['window_candidates'] = len(group)
        for component in COMPONENTS:
            model = models[component]
            if '_reg' in component:
                raw_scores = model.predict(transformed)
            else:
                raw_scores = score_pairwise_group(model, transformed)
            group[f'{component}_raw'] = raw_scores
            group[f'{component}_rank'] = percentile_score(
                pd.Series(raw_scores, index=group.index)
            )
        rank_matrix = group[[f'{component}_rank' for component in COMPONENTS]].to_numpy(float)
        group['ensemble_q25'] = np.quantile(rank_matrix, 0.25, axis=1)
        group['ensemble_median'] = np.median(rank_matrix, axis=1)
        group['ensemble_mean'] = np.mean(rank_matrix, axis=1)
        group['ensemble_std'] = np.std(rank_matrix, axis=1)
        top_votes = np.zeros(len(group), dtype=int)
        for component in COMPONENTS:
            values = group[f'{component}_rank'].to_numpy(float)
            top_votes += values == np.max(values)
        group['component_top_votes'] = top_votes
        group['component_top_vote_fraction'] = top_votes / len(COMPONENTS)
        result_parts.append(group)
    return pd.concat(result_parts, ignore_index=True, sort=False)


def safe_spearman(left: pd.Series, right: pd.Series) -> float:
    frame = pd.DataFrame({'left': left, 'right': right}).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if len(frame) < 3 or frame.left.nunique() < 2 or frame.right.nunique() < 2:
        return math.nan
    return float(spearmanr(frame.left, frame.right).statistic)


def select_window_topone(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, group in predictions.groupby(['seq', 'rank_start', 'rank_end'], sort=True):
        chosen = group.sort_values(
            [
                'ensemble_q25',
                'ensemble_median',
                'component_top_votes',
                'canonical_rank',
                'transaction_type',
            ],
            ascending=[False, False, False, True, True],
        ).iloc[0].copy()
        rows.append(chosen)
    return pd.DataFrame(rows).reset_index(drop=True)


def summarize_topone(topone: pd.DataFrame) -> dict[str, Any]:
    positive = topone.delta_HOTA[topone.delta_HOTA > 0]
    negative = topone.delta_HOTA[topone.delta_HOTA < 0]
    sequence_utility = {
        sequence: float(topone.loc[topone.seq == sequence, 'delta_HOTA'].sum())
        for sequence in SEQUENCES
    }
    positive_sum = float(positive.sum())
    negative_mass = float(-negative.sum())
    return {
        'selected_windows': int(len(topone)),
        'positive_windows': int((topone.delta_HOTA > 0).sum()),
        'negative_windows': int((topone.delta_HOTA < 0).sum()),
        'zero_windows': int((topone.delta_HOTA == 0).sum()),
        'positive_sum': positive_sum,
        'negative_mass': negative_mass,
        'negative_mass_fraction': negative_mass / max(positive_sum, 1e-12),
        'utility_sum': float(topone.delta_HOTA.sum()),
        'worst_sequence_utility': min(sequence_utility.values()),
        'sequence_utility': sequence_utility,
        'catastrophic_windows': int(
            (topone.delta_HOTA <= DEPLOYMENT_THRESHOLDS['catastrophic_delta_HOTA_threshold']).sum()
        ),
    }


def oracle_topone(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    rows = []
    for sequence in SEQUENCES:
        sequence_frame = work[work.seq == sequence]
        for start, end in TRAIN_WINDOWS:
            group = sequence_frame[window_mask(sequence_frame, start, end)]
            chosen = group.sort_values(
                [TARGET, 'canonical_rank', 'transaction_type'],
                ascending=[False, True, True],
            ).iloc[0].copy()
            chosen['rank_start'] = start
            chosen['rank_end'] = end
            rows.append(chosen)
    return pd.DataFrame(rows).reset_index(drop=True)


def deployment_eligibility(
    event_metrics: dict[str, float],
    topone_metrics: dict[str, Any],
) -> bool:
    thresholds = DEPLOYMENT_THRESHOLDS
    return bool(
        topone_metrics['selected_windows'] == thresholds['selected_windows']
        and topone_metrics['positive_windows'] >= thresholds['positive_windows_min']
        and topone_metrics['utility_sum'] > 0
        and topone_metrics['worst_sequence_utility'] >= 0
        and topone_metrics['negative_mass_fraction']
        <= thresholds['negative_mass_max_fraction_of_positive_sum']
        and topone_metrics['catastrophic_windows'] <= thresholds['catastrophic_windows_max']
        and event_metrics['target_spearman'] >= thresholds['event_target_spearman_min']
        and event_metrics['positive_HOTA_auc'] >= thresholds['positive_HOTA_auc_min']
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-features', action='append', required=True)
    parser.add_argument('--train-executability', required=True)
    parser.add_argument('--local-labels', required=True)
    parser.add_argument('--qualification-report', required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()

    assert_feature_whitelist()
    qualification = json.loads(Path(args.qualification_report).read_text())
    qualified_targets = qualification.get('qualified_auxiliary_targets', [])
    if TARGET not in qualified_targets:
        raise RuntimeError(f'primary target {TARGET} was not qualified')

    frame = load_training_frame(
        args.train_features,
        args.train_executability,
        args.local_labels,
    )
    prediction_parts = []
    fold_rows = []
    for fold_index, held_out in enumerate(SEQUENCES):
        train = frame[frame.seq != held_out].reset_index(drop=True)
        test = frame[frame.seq == held_out].reset_index(drop=True)
        imputer, models, pair_examples = fit_components(train)
        predictions = score_sequence(test, imputer, models)
        predictions['held_out_seq'] = held_out
        predictions['outer_fold_index'] = fold_index
        prediction_parts.append(predictions)
        fold_topone = select_window_topone(predictions)
        fold_summary = summarize_topone(fold_topone)
        fold_rows.append({
            'held_out_seq': held_out,
            'train_rows': len(train),
            'test_rows': len(test),
            'pairwise_train_examples': pair_examples,
            **{
                key: value
                for key, value in fold_summary.items()
                if key != 'sequence_utility'
            },
            'sequence_utility': json.dumps(
                fold_summary['sequence_utility'], sort_keys=True
            ),
        })

    predictions = pd.concat(prediction_parts, ignore_index=True, sort=False)
    topone = select_window_topone(predictions)
    event_metrics = {
        'target_spearman': safe_spearman(predictions.ensemble_q25, predictions[TARGET]),
        'delta_AssA_spearman': safe_spearman(
            predictions.ensemble_q25, predictions.delta_AssA
        ),
        'delta_HOTA_spearman': safe_spearman(
            predictions.ensemble_q25, predictions.delta_HOTA
        ),
        'positive_HOTA_auc': float(
            roc_auc_score((predictions.delta_HOTA > 0).astype(int), predictions.ensemble_q25)
        ),
    }
    topone_metrics = summarize_topone(topone)
    oracle = oracle_topone(frame)
    oracle_metrics = summarize_topone(oracle)
    eligible = deployment_eligibility(event_metrics, topone_metrics)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    rounded(predictions).to_csv(out_dir / 'outer_event_predictions.csv', index=False)
    rounded(topone).to_csv(out_dir / 'outer_window_topone.csv', index=False)
    rounded(pd.DataFrame(fold_rows)).to_csv(out_dir / 'outer_fold_summary.csv', index=False)
    rounded(oracle).to_csv(out_dir / 'oracle_window_topone.csv', index=False)
    (out_dir / 'model_feature_whitelist.json').write_text(
        json.dumps(MODEL_FEATURES, indent=2) + '\n'
    )
    report = {
        'protocol': {
            'scope': 'Sequence-disjoint outer LOSO on train ranks21-100 only. No locked artifact is read.',
            'primary_auxiliary_target': TARGET,
            'target_selection_rule': 'Highest preregistered pooled Spearman with delta_AssA among qualified local targets; ties by positive-HOTA AUC.',
            'components': COMPONENTS,
            'ensemble': 'Within each held-out sequence-window, convert every component score to percentile rank and use the component 25th percentile as the robust score.',
            'feature_count': len(MODEL_FEATURES),
            'feature_whitelist_sha256': hashlib.sha256(
                json.dumps(MODEL_FEATURES, separators=(',', ':')).encode()
            ).hexdigest(),
            'deployment_thresholds_preregistered': DEPLOYMENT_THRESHOLDS,
            'locked_artifacts_read': False,
        },
        'dataset': {
            'events': len(frame),
            'sequences': SEQUENCES,
            'windows': 16,
        },
        'event_metrics': event_metrics,
        'outer_topone': topone_metrics,
        'local_target_oracle_topone': oracle_metrics,
        'deployment_allowed': eligible,
        'next_step': (
            'freeze and hash a locked prediction manifest before reading any locked utility label'
            if eligible else
            'do not access locked labels; diagnose observable-feature predictability'
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
