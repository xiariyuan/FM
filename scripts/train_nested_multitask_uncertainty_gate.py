from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer

from train_loso_local_counterfactual_ensemble import (
    MODEL_FEATURES,
    SEQUENCES,
    TARGET,
    TRAIN_WINDOWS,
    load_training_frame,
)

KEYS = ['seq', 'canonical_rank', 'u', 'v', 'boundary_frame', 'transaction_type']
SEEDS = [17, 53, 89, 131, 173]


@dataclass(frozen=True)
class ModelConfig:
    name: str
    max_depth: int | None
    min_samples_leaf: int
    max_features: float


@dataclass(frozen=True)
class ScoreConfig:
    name: str
    primary: str
    tie_break: str


@dataclass(frozen=True)
class GateConfig:
    name: str
    dispersion_cap: float | None
    require_q25_positive: bool
    stability_floor: float | None


MODEL_CONFIGS = [
    ModelConfig('extra_d7_l3', 7, 3, 0.70),
    ModelConfig('extra_d10_l5', 10, 5, 0.70),
    ModelConfig('extra_full_l5', None, 5, 0.70),
]

SCORE_CONFIGS = [
    ScoreConfig('mean_positive_fraction', 'mean_positive_fraction', 'mean_target_q25'),
    ScoreConfig('seedq25_positive_fraction', 'seedq25_positive_fraction', 'seedq25_target_q25'),
    ScoreConfig('mean_target_q25', 'mean_target_q25', 'mean_positive_fraction'),
]

GATE_CONFIGS = [
    GateConfig('all', None, False, None),
    GateConfig('disp_le_015', 0.015, False, None),
    GateConfig('disp_le_020', 0.020, False, None),
    GateConfig('disp_le_025', 0.025, False, None),
    GateConfig('disp_le_030', 0.030, False, None),
    GateConfig('disp_le_040', 0.040, False, None),
    GateConfig('stable_080', None, False, 0.80),
    GateConfig('disp_le_020_stable_080', 0.020, False, 0.80),
    GateConfig('disp_le_025_stable_080', 0.025, False, 0.80),
    GateConfig('disp_le_030_stable_080', 0.030, False, 0.80),
    GateConfig('q25_pos_disp_le_025_stable_080', 0.025, True, 0.80),
    GateConfig('q25_pos_disp_le_030_stable_080', 0.030, True, 0.80),
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


def load_qualified_targets(report_path: str) -> list[str]:
    report = json.loads(Path(report_path).read_text())
    targets = report.get('qualified_auxiliary_targets')
    if not isinstance(targets, list) or not targets:
        raise RuntimeError('qualification report has no qualified_auxiliary_targets')
    if TARGET not in targets:
        raise RuntimeError(f'primary target {TARGET} is not qualified')
    return [str(target) for target in targets]


def attach_targets(frame: pd.DataFrame, label_path: str, targets: list[str]) -> pd.DataFrame:
    labels = pd.read_csv(label_path)
    required = KEYS + targets
    missing = [column for column in required if column not in labels.columns]
    if missing:
        raise RuntimeError(f'event labels missing columns: {missing}')
    target_frame = labels[required].drop_duplicates(KEYS)
    if len(target_frame) != len(labels):
        raise RuntimeError('event labels contain duplicate directional keys')
    missing_targets = [target for target in targets if target not in frame.columns]
    if missing_targets:
        frame = frame.merge(
            target_frame[KEYS + missing_targets],
            on=KEYS,
            how='left',
            validate='one_to_one',
        )
    if frame[targets].isna().any().any():
        raise RuntimeError('missing qualified target values after merge')
    return frame


def sequence_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.seq.value_counts()
    mapping = {seq: len(frame) / (len(counts) * count) for seq, count in counts.items()}
    return frame.seq.map(mapping).to_numpy(float)


def normalized_training_targets(frame: pd.DataFrame, targets: list[str]) -> np.ndarray:
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


def fit_predict_multitask(
    train: pd.DataFrame,
    test: pd.DataFrame,
    targets: list[str],
    config: ModelConfig,
    seed_offset: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    imputer = SimpleImputer(strategy='median')
    x_train = imputer.fit_transform(train[MODEL_FEATURES])
    x_test = imputer.transform(test[MODEL_FEATURES])
    y_train = normalized_training_targets(train, targets)
    weights = sequence_weights(train)

    predictions = []
    for seed in SEEDS:
        model = ExtraTreesRegressor(
            n_estimators=500,
            max_depth=config.max_depth,
            min_samples_leaf=config.min_samples_leaf,
            max_features=config.max_features,
            random_state=seed + seed_offset,
            n_jobs=-1,
        )
        model.fit(x_train, y_train, sample_weight=weights)
        predictions.append(model.predict(x_test))
    array = np.stack(predictions, axis=0)

    seed_positive_fraction = np.mean(array > 0.0, axis=2)
    seed_target_q25 = np.quantile(array, 0.25, axis=2)
    seed_target_mean = np.mean(array, axis=2)
    seed_target_min = np.min(array, axis=2)

    result = test[KEYS + targets + ['delta_HOTA', 'delta_AssA']].copy()
    result['mean_positive_fraction'] = seed_positive_fraction.mean(axis=0)
    result['seedq25_positive_fraction'] = np.quantile(seed_positive_fraction, 0.25, axis=0)
    result['seedmin_positive_fraction'] = seed_positive_fraction.min(axis=0)
    result['mean_target_q25'] = seed_target_q25.mean(axis=0)
    result['seedq25_target_q25'] = np.quantile(seed_target_q25, 0.25, axis=0)
    result['seedmin_target_q25'] = seed_target_q25.min(axis=0)
    result['mean_target_mean'] = seed_target_mean.mean(axis=0)
    result['mean_target_min'] = seed_target_min.mean(axis=0)
    result['prediction_dispersion'] = array.std(axis=0).mean(axis=1)
    return result, array


def seed_window_choice(
    frame: pd.DataFrame,
    seed_array: np.ndarray,
    score: ScoreConfig,
    start: int,
    end: int,
) -> list[int]:
    mask = frame.canonical_rank.between(start, end).to_numpy()
    positions = np.flatnonzero(mask)
    selected: list[int] = []
    for seed_index in range(seed_array.shape[0]):
        values = seed_array[seed_index, positions]
        positive_fraction = np.mean(values > 0.0, axis=1)
        target_q25 = np.quantile(values, 0.25, axis=1)
        if score.primary.endswith('positive_fraction'):
            primary = positive_fraction
            tie = target_q25
        else:
            primary = target_q25
            tie = positive_fraction
        block = frame.iloc[positions].copy()
        block['_primary'] = primary
        block['_tie'] = tie
        chosen = block.sort_values(
            ['_primary', '_tie', 'canonical_rank', 'transaction_type'],
            ascending=[False, False, True, True],
        ).index[0]
        selected.append(int(chosen))
    return selected


def select_window_topone(
    prediction: pd.DataFrame,
    seed_array: np.ndarray,
    score: ScoreConfig,
    windows: list[tuple[int, int]],
) -> pd.DataFrame:
    rows = []
    for start, end in windows:
        block = prediction[prediction.canonical_rank.between(start, end)].copy()
        if not len(block):
            continue
        chosen = block.sort_values(
            [score.primary, score.tie_break, 'canonical_rank', 'transaction_type'],
            ascending=[False, False, True, True],
        ).iloc[0].copy()
        seed_choices = seed_window_choice(prediction, seed_array, score, start, end)
        chosen_index = int(chosen.name)
        chosen['candidate_stability'] = np.mean(np.asarray(seed_choices) == chosen_index)
        chosen['seed_choice_count'] = int(np.sum(np.asarray(seed_choices) == chosen_index))
        chosen['rank_start'] = start
        chosen['rank_end'] = end
        chosen['score_name'] = score.name
        rows.append(chosen)
    return pd.DataFrame(rows).reset_index(drop=True)


def gate_mask(frame: pd.DataFrame, gate: GateConfig) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    if gate.dispersion_cap is not None:
        mask &= frame.prediction_dispersion <= gate.dispersion_cap
    if gate.require_q25_positive:
        mask &= frame.mean_target_q25 > 0.0
    if gate.stability_floor is not None:
        mask &= frame.candidate_stability >= gate.stability_floor
    return mask


def summarize(frame: pd.DataFrame, expected_sequences: list[str]) -> dict[str, object]:
    positive = frame[TARGET] > 0.0
    negative = frame[TARGET] < 0.0
    positive_sum = float(frame.loc[positive, TARGET].sum())
    negative_mass = float(-frame.loc[negative, TARGET].sum())
    denominator = int(positive.sum() + negative.sum())
    precision = float(positive.sum() / denominator) if denominator else 0.0
    sequence_utility = {
        sequence: float(frame.loc[frame.seq == sequence, TARGET].sum())
        for sequence in expected_sequences
    }
    selected_sequences = sorted(frame.seq.unique().tolist()) if len(frame) else []
    stability_min = float(frame.candidate_stability.min()) if len(frame) else 0.0
    stability_mean = float(frame.candidate_stability.mean()) if len(frame) else 0.0
    return {
        'selected_windows': int(len(frame)),
        'covered_sequences': int(len(selected_sequences)),
        'selected_sequences': selected_sequences,
        'positive_windows': int(positive.sum()),
        'negative_windows': int(negative.sum()),
        'zero_windows': int((frame[TARGET] == 0.0).sum()),
        'positive_precision': precision,
        'positive_sum': positive_sum,
        'negative_mass': negative_mass,
        'utility_sum': float(frame[TARGET].sum()),
        'worst_sequence_utility': min(sequence_utility.values()) if selected_sequences else 0.0,
        'sequence_utility': sequence_utility,
        'candidate_stability_min': stability_min,
        'candidate_stability_mean': stability_mean,
    }


def inner_eligible(summary: dict[str, object], expected_sequences: int) -> bool:
    positive_sum = float(summary['positive_sum'])
    negative_mass = float(summary['negative_mass'])
    return bool(
        int(summary['selected_windows']) >= 6
        and int(summary['covered_sequences']) == expected_sequences
        and float(summary['positive_precision']) >= 0.70
        and positive_sum > 0.0
        and float(summary['utility_sum']) > 0.0
        and float(summary['worst_sequence_utility']) >= 0.0
        and negative_mass <= 0.20 * positive_sum
        and float(summary['candidate_stability_min']) >= 0.80
    )


def global_hota_summary(frame: pd.DataFrame) -> dict[str, object]:
    positive = frame.delta_HOTA > 0.0
    negative = frame.delta_HOTA < 0.0
    positive_sum = float(frame.loc[positive, 'delta_HOTA'].sum())
    negative_mass = float(-frame.loc[negative, 'delta_HOTA'].sum())
    denominator = int(positive.sum() + negative.sum())
    sequence_utility = {
        sequence: float(frame.loc[frame.seq == sequence, 'delta_HOTA'].sum())
        for sequence in SEQUENCES
    }
    return {
        'selected_windows': int(len(frame)),
        'covered_sequences': int(frame.seq.nunique()) if len(frame) else 0,
        'positive_windows': int(positive.sum()),
        'negative_windows': int(negative.sum()),
        'zero_windows': int((frame.delta_HOTA == 0.0).sum()),
        'positive_precision': float(positive.sum() / denominator) if denominator else 0.0,
        'positive_sum': positive_sum,
        'negative_mass': negative_mass,
        'utility_sum': float(frame.delta_HOTA.sum()),
        'worst_sequence_utility': min(sequence_utility.values()),
        'sequence_utility': sequence_utility,
        'catastrophic_windows': int((frame.delta_HOTA <= -0.05).sum()),
        'candidate_stability_min': float(frame.candidate_stability.min()) if len(frame) else 0.0,
        'candidate_stability_mean': float(frame.candidate_stability.mean()) if len(frame) else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-features', action='append', required=True)
    parser.add_argument('--train-executability', required=True)
    parser.add_argument('--event-labels', required=True)
    parser.add_argument('--qualification-report', required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()

    targets = load_qualified_targets(args.qualification_report)
    frame = load_training_frame(
        args.train_features,
        args.train_executability,
        args.event_labels,
    )
    frame = attach_targets(frame, args.event_labels, targets)
    frame = frame.sort_values(KEYS).reset_index(drop=True)

    inner_summaries = []
    inner_predictions = []
    outer_choices = []
    outer_all_predictions = []
    outer_topones = []
    outer_selected = []

    for outer_index, held_out in enumerate(SEQUENCES):
        outer_train = frame[frame.seq != held_out].reset_index(drop=True)
        outer_test = frame[frame.seq == held_out].reset_index(drop=True)
        inner_sequences = [sequence for sequence in SEQUENCES if sequence != held_out]
        candidate_rows = []

        for model_index, model_config in enumerate(MODEL_CONFIGS):
            inner_fold_predictions: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}
            for inner_index, inner_held in enumerate(inner_sequences):
                inner_train = outer_train[outer_train.seq != inner_held].reset_index(drop=True)
                inner_test = outer_train[outer_train.seq == inner_held].reset_index(drop=True)
                prediction, seed_array = fit_predict_multitask(
                    inner_train,
                    inner_test,
                    targets,
                    model_config,
                    seed_offset=10000 * outer_index + 1000 * model_index + 100 * inner_index,
                )
                inner_fold_predictions[inner_held] = (prediction, seed_array)

            for score_index, score_config in enumerate(SCORE_CONFIGS):
                topone_parts = []
                for inner_held in inner_sequences:
                    prediction, seed_array = inner_fold_predictions[inner_held]
                    topone = select_window_topone(
                        prediction,
                        seed_array,
                        score_config,
                        TRAIN_WINDOWS,
                    )
                    topone['outer_held_out_seq'] = held_out
                    topone['inner_held_out_seq'] = inner_held
                    topone['model_config'] = model_config.name
                    topone_parts.append(topone)
                all_topone = pd.concat(topone_parts, ignore_index=True)

                for gate_index, gate_config in enumerate(GATE_CONFIGS):
                    selected = all_topone[gate_mask(all_topone, gate_config)].copy()
                    summary = summarize(selected, inner_sequences)
                    eligible = inner_eligible(summary, len(inner_sequences))
                    row = {
                        **summary,
                        'eligible': int(eligible),
                        'outer_held_out_seq': held_out,
                        'model_config': model_config.name,
                        'score_config': score_config.name,
                        'gate_config': gate_config.name,
                        'model_order': model_index,
                        'score_order': score_index,
                        'gate_order': gate_index,
                    }
                    candidate_rows.append(row)
                    inner_summaries.append(row)
                    selected['gate_config'] = gate_config.name
                    selected['eligible_config_gate'] = int(eligible)
                    inner_predictions.append(selected)

        eligible_rows = [row for row in candidate_rows if row['eligible'] == 1]
        if not eligible_rows:
            outer_choices.append({
                'held_out_seq': held_out,
                'chosen_model_config': 'no_op',
                'chosen_score_config': 'no_op',
                'chosen_gate_config': 'no_op',
                'eligible_config_gates': 0,
                'outer_topone_windows': 0,
                'outer_selected_windows': 0,
            })
            continue

        chosen = sorted(
            eligible_rows,
            key=lambda row: (
                -float(row['worst_sequence_utility']),
                -float(row['positive_precision']),
                -float(row['utility_sum']),
                -int(row['selected_windows']),
                -float(row['candidate_stability_mean']),
                int(row['model_order']),
                int(row['score_order']),
                int(row['gate_order']),
            ),
        )[0]
        model_config = next(config for config in MODEL_CONFIGS if config.name == chosen['model_config'])
        score_config = next(config for config in SCORE_CONFIGS if config.name == chosen['score_config'])
        gate_config = next(config for config in GATE_CONFIGS if config.name == chosen['gate_config'])

        prediction, seed_array = fit_predict_multitask(
            outer_train,
            outer_test,
            targets,
            model_config,
            seed_offset=50000 + 10000 * outer_index,
        )
        prediction['held_out_seq'] = held_out
        prediction['chosen_model_config'] = model_config.name
        prediction['chosen_score_config'] = score_config.name
        prediction['chosen_gate_config'] = gate_config.name
        outer_all_predictions.append(prediction)

        topone = select_window_topone(prediction, seed_array, score_config, TRAIN_WINDOWS)
        topone['held_out_seq'] = held_out
        topone['chosen_model_config'] = model_config.name
        topone['chosen_score_config'] = score_config.name
        topone['chosen_gate_config'] = gate_config.name
        topone['selected_by_gate'] = gate_mask(topone, gate_config).astype(int)
        outer_topones.append(topone)
        selected = topone[topone.selected_by_gate == 1].copy()
        outer_selected.append(selected)
        outer_choices.append({
            'held_out_seq': held_out,
            'chosen_model_config': model_config.name,
            'chosen_score_config': score_config.name,
            'chosen_gate_config': gate_config.name,
            'eligible_config_gates': len(eligible_rows),
            'outer_topone_windows': int(len(topone)),
            'outer_selected_windows': int(len(selected)),
        })

    outer_selected_frame = pd.concat(outer_selected, ignore_index=True) if outer_selected else pd.DataFrame()
    outer_topone_frame = pd.concat(outer_topones, ignore_index=True) if outer_topones else pd.DataFrame()
    outer_prediction_frame = pd.concat(outer_all_predictions, ignore_index=True) if outer_all_predictions else pd.DataFrame()
    inner_prediction_frame = pd.concat(inner_predictions, ignore_index=True) if inner_predictions else pd.DataFrame()
    inner_summary_frame = pd.DataFrame(inner_summaries)
    outer_choice_frame = pd.DataFrame(outer_choices)

    local_summary = summarize(outer_selected_frame, SEQUENCES) if len(outer_selected_frame) else summarize(pd.DataFrame(columns=['seq', TARGET, 'candidate_stability']), SEQUENCES)
    hota_summary = global_hota_summary(outer_selected_frame) if len(outer_selected_frame) else {
        'selected_windows': 0,
        'covered_sequences': 0,
        'positive_windows': 0,
        'negative_windows': 0,
        'zero_windows': 0,
        'positive_precision': 0.0,
        'positive_sum': 0.0,
        'negative_mass': 0.0,
        'utility_sum': 0.0,
        'worst_sequence_utility': 0.0,
        'sequence_utility': {sequence: 0.0 for sequence in SEQUENCES},
        'catastrophic_windows': 0,
        'candidate_stability_min': 0.0,
        'candidate_stability_mean': 0.0,
    }
    deployment_allowed = bool(
        hota_summary['selected_windows'] >= 8
        and hota_summary['covered_sequences'] == 4
        and hota_summary['positive_precision'] >= 0.60
        and hota_summary['positive_sum'] > 0.0
        and hota_summary['utility_sum'] > 0.0
        and hota_summary['worst_sequence_utility'] >= 0.0
        and hota_summary['negative_mass'] <= 0.25 * hota_summary['positive_sum']
        and hota_summary['catastrophic_windows'] == 0
        and hota_summary['candidate_stability_min'] >= 0.80
        and local_summary['utility_sum'] > 0.0
        and local_summary['worst_sequence_utility'] >= 0.0
    )

    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=False)
    rounded(inner_summary_frame).to_csv(output / 'inner_config_gate_summary.csv', index=False)
    rounded(inner_prediction_frame).to_csv(output / 'inner_selected_predictions.csv', index=False)
    rounded(outer_choice_frame).to_csv(output / 'outer_fold_choices.csv', index=False)
    rounded(outer_prediction_frame).to_csv(output / 'outer_event_predictions.csv', index=False)
    rounded(outer_topone_frame).to_csv(output / 'outer_window_topone.csv', index=False)
    rounded(outer_selected_frame).to_csv(output / 'outer_selected_windows.csv', index=False)

    report = {
        'protocol': {
            'scope': 'Train-only nested sequence-disjoint multitask local utility student. No locked feature, utility, or TrackEval artifact is read.',
            'qualified_targets': targets,
            'model_configs': [asdict(config) for config in MODEL_CONFIGS],
            'score_configs': [asdict(config) for config in SCORE_CONFIGS],
            'gate_configs': [asdict(config) for config in GATE_CONFIGS],
            'seeds': SEEDS,
            'target_normalization': 'Per training sequence and target: median/IQR robust normalization, clipped to [-5,5]. Held-out labels are never used for normalization or prediction.',
            'inner_selection': 'Three-way inner sequence LOSO inside every outer sequence fold. Eligibility and selection use full_idtp_delta_norm only, never delta_HOTA.',
            'inner_eligibility': 'selected>=6, all 3 inner sequences covered, local positive precision>=0.70, positive and total local utility>0, worst inner sequence>=0, negative mass<=20% positive sum, minimum seed candidate stability>=0.80.',
            'global_eligibility': 'selected>=8, all 4 sequences covered, HOTA precision>=0.60, total and positive HOTA>0, worst sequence>=0, negative mass<=25% positive sum, no HOTA<=-0.05, minimum seed stability>=0.80, and local utility total/worst>=(0,0).',
            'locked_labels_read': 0,
            'locked_trackeval_calls': 0,
            'remaining_locked_rows_untouched': 156,
        },
        'dataset': {
            'events': int(len(frame)),
            'sequences': SEQUENCES,
            'qualified_targets': len(targets),
            'observable_features': len(MODEL_FEATURES),
            'windows': len(SEQUENCES) * len(TRAIN_WINDOWS),
        },
        'outer_fold_choices': outer_choices,
        'outer_local_summary': local_summary,
        'outer_hota_summary': hota_summary,
        'deployment_allowed': deployment_allowed,
        'locked_manifest_created': False,
        'remaining_locked_labels_unread': 156,
    }
    (output / 'report.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    manifest = {
        path.name: sha256(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != 'prediction_manifest.json'
    }
    (output / 'prediction_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
