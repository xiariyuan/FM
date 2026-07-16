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
MOTION_COMPONENTS = [
    ('source_stability', 'source_center_resid_20_h120_std', False),
    ('source_residual', 'source_center_resid_20_full_mean', False),
    ('transfer_margin', 'motion_margin_20_full_q25', True),
    ('velocity_margin', 'velocity_margin_20_h60', True),
    ('source_win_fraction', 'motion_source_win_fraction_20_h120', True),
]


@dataclass(frozen=True)
class GateConfig:
    name: str
    minimum_motion_min: float | None
    minimum_motion_q25: float | None


GATE_CONFIGS = [
    GateConfig('stable080_no_motion_veto', None, None),
    GateConfig('stable080_motion_min_ge_010', 0.10, None),
    GateConfig('stable080_motion_min_ge_015', 0.15, None),
    GateConfig('stable080_motion_min_ge_020', 0.20, None),
    GateConfig('stable080_motion_min_ge_025', 0.25, None),
    GateConfig('stable080_motion_min_ge_030', 0.30, None),
    GateConfig('stable080_motion_q25_ge_020', None, 0.20),
    GateConfig('stable080_motion_q25_ge_025', None, 0.25),
    GateConfig('stable080_motion_q25_ge_030', None, 0.30),
    GateConfig('stable080_motion_q25_ge_035', None, 0.35),
    GateConfig('stable080_min015_q25_025', 0.15, 0.25),
    GateConfig('stable080_min020_q25_030', 0.20, 0.30),
]


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


def load_qualified_targets(path: Path) -> list[str]:
    report = json.loads(path.read_text(encoding='utf-8'))
    targets = report.get('qualified_auxiliary_targets')
    if not isinstance(targets, list) or not targets:
        raise RuntimeError('qualification report has no qualified_auxiliary_targets')
    if TARGET not in targets:
        raise RuntimeError(f'primary target {TARGET} is not qualified')
    return [str(value) for value in targets]


def attach_targets(frame: pd.DataFrame, labels_path: Path, targets: list[str]) -> pd.DataFrame:
    labels = pd.read_csv(labels_path)
    missing = [column for column in KEYS + targets if column not in labels.columns]
    if missing:
        raise RuntimeError(f'event labels missing columns: {missing}')
    target_frame = labels[KEYS + targets].drop_duplicates(KEYS)
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
        raise RuntimeError('missing qualified target values')
    return frame


def attach_motion(
    frame: pd.DataFrame,
    motion_path: Path,
    compact_list_path: Path,
) -> tuple[pd.DataFrame, list[str]]:
    motion = pd.read_csv(motion_path)
    compact = pd.read_csv(compact_list_path).feature.astype(str).tolist()
    compact = list(dict.fromkeys(compact))
    required = KEYS + compact + [column for _, column, _ in MOTION_COMPONENTS]
    missing = [column for column in required if column not in motion.columns]
    if missing:
        raise RuntimeError(f'motion features missing columns: {missing}')
    motion_columns = list(dict.fromkeys(required))
    motion_frame = motion[motion_columns].drop_duplicates(KEYS)
    if len(motion_frame) != len(motion):
        raise RuntimeError('motion features contain duplicate event keys')
    frame = frame.merge(motion_frame, on=KEYS, how='inner', validate='one_to_one')
    return frame, compact


def sequence_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.seq.value_counts()
    mapping = {seq: len(frame) / (len(counts) * count) for seq, count in counts.items()}
    return frame.seq.map(mapping).to_numpy(float)


def normalized_training_targets(frame: pd.DataFrame, targets: list[str]) -> np.ndarray:
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


def fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    targets: list[str],
    features: list[str],
    seed_offset: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    imputer = SimpleImputer(strategy='median')
    x_train = imputer.fit_transform(train[features])
    x_test = imputer.transform(test[features])
    y_train = normalized_training_targets(train, targets)
    weights = sequence_weights(train)
    predictions = []
    for seed in SEEDS:
        model = ExtraTreesRegressor(
            n_estimators=300,
            max_depth=7,
            min_samples_leaf=3,
            max_features=0.65,
            random_state=seed + seed_offset,
            n_jobs=1,
        )
        model.fit(x_train, y_train, sample_weight=weights)
        predictions.append(model.predict(x_test))
    array = np.stack(predictions, axis=0)
    seed_target_mean = array.mean(axis=2)
    seed_target_q25 = np.quantile(array, 0.25, axis=2)
    seed_positive_fraction = np.mean(array > 0.0, axis=2)
    result = test[KEYS + targets + ['delta_HOTA', 'delta_AssA']].copy()
    for _, column, _ in MOTION_COMPONENTS:
        result[column] = test[column].to_numpy()
    result['mean_target_mean'] = seed_target_mean.mean(axis=0)
    result['mean_target_q25'] = seed_target_q25.mean(axis=0)
    result['mean_positive_fraction'] = seed_positive_fraction.mean(axis=0)
    result['prediction_dispersion'] = array.std(axis=0).mean(axis=1)
    return result, array


def seed_choices(
    prediction: pd.DataFrame,
    seed_array: np.ndarray,
    start: int,
    end: int,
) -> list[int]:
    mask = prediction.canonical_rank.between(start, end).to_numpy()
    positions = np.flatnonzero(mask)
    choices: list[int] = []
    for seed_index in range(seed_array.shape[0]):
        values = seed_array[seed_index, positions]
        target_mean = values.mean(axis=1)
        target_q25 = np.quantile(values, 0.25, axis=1)
        block = prediction.iloc[positions].copy()
        block['_primary'] = target_mean
        block['_tie'] = target_q25
        chosen_index = block.sort_values(
            ['_primary', '_tie', 'canonical_rank', 'transaction_type'],
            ascending=[False, False, True, True],
        ).index[0]
        choices.append(int(chosen_index))
    return choices


def add_motion_certificate(prediction: pd.DataFrame) -> pd.DataFrame:
    result = prediction.copy()
    for name, _, _ in MOTION_COMPONENTS:
        result[f'{name}_rank'] = np.nan
    result['motion_median'] = np.nan
    result['motion_q25'] = np.nan
    result['motion_min'] = np.nan
    result['motion_mean'] = np.nan
    for start, end in TRAIN_WINDOWS:
        indices = result.index[result.canonical_rank.between(start, end)]
        count = len(indices)
        if not count:
            continue
        rank_columns = []
        for name, column, higher_is_better in MOTION_COMPONENTS:
            raw = result.loc[indices, column].rank(method='average', pct=True)
            rank = raw if higher_is_better else 1.0 - raw + 1.0 / count
            rank_column = f'{name}_rank'
            result.loc[indices, rank_column] = rank
            rank_columns.append(rank_column)
        result.loc[indices, 'motion_median'] = result.loc[indices, rank_columns].median(axis=1)
        result.loc[indices, 'motion_q25'] = result.loc[indices, rank_columns].quantile(0.25, axis=1)
        result.loc[indices, 'motion_min'] = result.loc[indices, rank_columns].min(axis=1)
        result.loc[indices, 'motion_mean'] = result.loc[indices, rank_columns].mean(axis=1)
    return result


def select_topone(prediction: pd.DataFrame, seed_array: np.ndarray) -> pd.DataFrame:
    prediction = add_motion_certificate(prediction)
    rows = []
    for start, end in TRAIN_WINDOWS:
        block = prediction[prediction.canonical_rank.between(start, end)].copy()
        if block.empty:
            continue
        chosen = block.sort_values(
            ['mean_target_mean', 'mean_target_q25', 'canonical_rank', 'transaction_type'],
            ascending=[False, False, True, True],
        ).iloc[0].copy()
        choices = seed_choices(prediction, seed_array, start, end)
        chosen_index = int(chosen.name)
        chosen['candidate_stability'] = float(np.mean(np.asarray(choices) == chosen_index))
        chosen['seed_choice_count'] = int(np.sum(np.asarray(choices) == chosen_index))
        chosen['rank_start'] = start
        chosen['rank_end'] = end
        rows.append(chosen)
    return pd.DataFrame(rows).reset_index(drop=True)


def gate_mask(frame: pd.DataFrame, gate: GateConfig) -> pd.Series:
    mask = frame.candidate_stability >= 0.80
    if gate.minimum_motion_min is not None:
        mask &= frame.motion_min >= gate.minimum_motion_min
    if gate.minimum_motion_q25 is not None:
        mask &= frame.motion_q25 >= gate.minimum_motion_q25
    return mask


def summarize_local(frame: pd.DataFrame, expected_sequences: list[str]) -> dict[str, object]:
    positive = frame[TARGET] > 0.0
    negative = frame[TARGET] < 0.0
    positive_sum = float(frame.loc[positive, TARGET].sum())
    negative_mass = float(-frame.loc[negative, TARGET].sum())
    denominator = int(positive.sum() + negative.sum())
    sequence_utility = {
        seq: float(frame.loc[frame.seq == seq, TARGET].sum())
        for seq in expected_sequences
    }
    selected_sequences = sorted(frame.seq.unique().tolist()) if len(frame) else []
    return {
        'selected_windows': int(len(frame)),
        'covered_sequences': int(len(selected_sequences)),
        'selected_sequences': selected_sequences,
        'positive_windows': int(positive.sum()),
        'negative_windows': int(negative.sum()),
        'zero_windows': int((frame[TARGET] == 0.0).sum()),
        'positive_precision': float(positive.sum() / denominator) if denominator else 0.0,
        'positive_sum': positive_sum,
        'negative_mass': negative_mass,
        'utility_sum': float(frame[TARGET].sum()),
        'worst_sequence_utility': min(sequence_utility.values()),
        'sequence_utility': sequence_utility,
        'catastrophic_local_windows': int((frame[TARGET] <= -0.05).sum()),
        'candidate_stability_min': float(frame.candidate_stability.min()) if len(frame) else 0.0,
        'candidate_stability_mean': float(frame.candidate_stability.mean()) if len(frame) else 0.0,
        'motion_min_min': float(frame.motion_min.min()) if len(frame) else 0.0,
        'motion_q25_min': float(frame.motion_q25.min()) if len(frame) else 0.0,
    }


def inner_eligible(summary: dict[str, object]) -> bool:
    positive_sum = float(summary['positive_sum'])
    return bool(
        int(summary['selected_windows']) >= 6
        and int(summary['covered_sequences']) == 3
        and float(summary['positive_precision']) >= 0.70
        and positive_sum > 0.0
        and float(summary['utility_sum']) > 0.0
        and float(summary['worst_sequence_utility']) >= 0.0
        and float(summary['negative_mass']) <= 0.20 * positive_sum
        and int(summary['catastrophic_local_windows']) == 0
        and float(summary['candidate_stability_min']) >= 0.80
    )


def summarize_hota(frame: pd.DataFrame) -> dict[str, object]:
    positive = frame.delta_HOTA > 0.0
    negative = frame.delta_HOTA < 0.0
    positive_sum = float(frame.loc[positive, 'delta_HOTA'].sum())
    negative_mass = float(-frame.loc[negative, 'delta_HOTA'].sum())
    denominator = int(positive.sum() + negative.sum())
    sequence_utility = {
        seq: float(frame.loc[frame.seq == seq, 'delta_HOTA'].sum())
        for seq in SEQUENCES
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
    }


def empty_local_summary() -> dict[str, object]:
    return {
        'selected_windows': 0,
        'covered_sequences': 0,
        'selected_sequences': [],
        'positive_windows': 0,
        'negative_windows': 0,
        'zero_windows': 0,
        'positive_precision': 0.0,
        'positive_sum': 0.0,
        'negative_mass': 0.0,
        'utility_sum': 0.0,
        'worst_sequence_utility': 0.0,
        'sequence_utility': {seq: 0.0 for seq in SEQUENCES},
        'catastrophic_local_windows': 0,
        'candidate_stability_min': 0.0,
        'candidate_stability_mean': 0.0,
        'motion_min_min': 0.0,
        'motion_q25_min': 0.0,
    }


def empty_hota_summary() -> dict[str, object]:
    return {
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
        'sequence_utility': {seq: 0.0 for seq in SEQUENCES},
        'catastrophic_windows': 0,
        'candidate_stability_min': 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-features', action='append', required=True)
    parser.add_argument('--train-executability', required=True)
    parser.add_argument('--event-labels', required=True)
    parser.add_argument('--qualification-report', required=True)
    parser.add_argument('--motion-features', required=True)
    parser.add_argument('--motion-compact-list', required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()

    event_labels_path = Path(args.event_labels)
    targets = load_qualified_targets(Path(args.qualification_report))
    frame = load_training_frame(
        args.train_features,
        args.train_executability,
        args.event_labels,
    )
    frame = attach_targets(frame, event_labels_path, targets)
    frame, compact_motion = attach_motion(
        frame,
        Path(args.motion_features),
        Path(args.motion_compact_list),
    )
    frame = frame.sort_values(KEYS).reset_index(drop=True)
    model_features = MODEL_FEATURES + compact_motion

    inner_summaries: list[dict[str, object]] = []
    inner_topones: list[pd.DataFrame] = []
    outer_choices: list[dict[str, object]] = []
    outer_predictions: list[pd.DataFrame] = []
    outer_topones: list[pd.DataFrame] = []
    outer_selected: list[pd.DataFrame] = []

    for outer_index, held_out in enumerate(SEQUENCES):
        outer_train = frame[frame.seq != held_out].reset_index(drop=True)
        outer_test = frame[frame.seq == held_out].reset_index(drop=True)
        inner_sequences = [seq for seq in SEQUENCES if seq != held_out]
        inner_parts = []
        for inner_index, inner_held in enumerate(inner_sequences):
            inner_train = outer_train[outer_train.seq != inner_held].reset_index(drop=True)
            inner_test = outer_train[outer_train.seq == inner_held].reset_index(drop=True)
            prediction, seed_array = fit_predict(
                inner_train,
                inner_test,
                targets,
                model_features,
                seed_offset=10000 * outer_index + 1000 * inner_index,
            )
            topone = select_topone(prediction, seed_array)
            topone['outer_held_out_seq'] = held_out
            topone['inner_held_out_seq'] = inner_held
            inner_parts.append(topone)
        all_inner_topone = pd.concat(inner_parts, ignore_index=True)

        eligible_rows = []
        for gate_order, gate in enumerate(GATE_CONFIGS):
            selected = all_inner_topone[gate_mask(all_inner_topone, gate)].copy()
            summary = summarize_local(selected, inner_sequences)
            eligible = inner_eligible(summary)
            row = {
                **summary,
                'outer_held_out_seq': held_out,
                'gate_config': gate.name,
                'gate_order': gate_order,
                'eligible': int(eligible),
            }
            inner_summaries.append(row)
            selected['gate_config'] = gate.name
            selected['eligible_gate'] = int(eligible)
            inner_topones.append(selected)
            if eligible:
                eligible_rows.append(row)

        if not eligible_rows:
            outer_choices.append({
                'held_out_seq': held_out,
                'chosen_gate_config': 'no_op',
                'eligible_gates': 0,
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
                -float(row['motion_min_min']),
                int(row['gate_order']),
            ),
        )[0]
        gate = next(config for config in GATE_CONFIGS if config.name == chosen['gate_config'])
        prediction, seed_array = fit_predict(
            outer_train,
            outer_test,
            targets,
            model_features,
            seed_offset=50000 + 10000 * outer_index,
        )
        prediction = add_motion_certificate(prediction)
        prediction['held_out_seq'] = held_out
        prediction['chosen_gate_config'] = gate.name
        outer_predictions.append(prediction)
        topone = select_topone(prediction, seed_array)
        topone['held_out_seq'] = held_out
        topone['chosen_gate_config'] = gate.name
        topone['selected_by_gate'] = gate_mask(topone, gate).astype(int)
        outer_topones.append(topone)
        selected = topone[topone.selected_by_gate == 1].copy()
        outer_selected.append(selected)
        outer_choices.append({
            'held_out_seq': held_out,
            'chosen_gate_config': gate.name,
            'eligible_gates': len(eligible_rows),
            'outer_topone_windows': int(len(topone)),
            'outer_selected_windows': int(len(selected)),
        })

    inner_summary_frame = pd.DataFrame(inner_summaries)
    inner_topone_frame = pd.concat(inner_topones, ignore_index=True) if inner_topones else pd.DataFrame()
    outer_choice_frame = pd.DataFrame(outer_choices)
    outer_prediction_frame = pd.concat(outer_predictions, ignore_index=True) if outer_predictions else pd.DataFrame()
    outer_topone_frame = pd.concat(outer_topones, ignore_index=True) if outer_topones else pd.DataFrame()
    outer_selected_frame = pd.concat(outer_selected, ignore_index=True) if outer_selected else pd.DataFrame()

    local_summary = summarize_local(outer_selected_frame, SEQUENCES) if len(outer_selected_frame) else empty_local_summary()
    hota_summary = summarize_hota(outer_selected_frame) if len(outer_selected_frame) else empty_hota_summary()
    deployment_allowed = bool(
        int(hota_summary['selected_windows']) >= 8
        and int(hota_summary['covered_sequences']) == 4
        and float(hota_summary['positive_precision']) >= 0.60
        and float(hota_summary['positive_sum']) > 0.0
        and float(hota_summary['utility_sum']) > 0.0
        and float(hota_summary['worst_sequence_utility']) >= 0.0
        and float(hota_summary['negative_mass']) <= 0.25 * float(hota_summary['positive_sum'])
        and int(hota_summary['catastrophic_windows']) == 0
        and float(hota_summary['candidate_stability_min']) >= 0.80
        and float(local_summary['utility_sum']) > 0.0
        and float(local_summary['worst_sequence_utility']) >= 0.0
        and int(local_summary['catastrophic_local_windows']) == 0
    )

    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=False)
    rounded(inner_summary_frame).to_csv(output / 'inner_gate_summary.csv', index=False)
    rounded(inner_topone_frame).to_csv(output / 'inner_selected_topone.csv', index=False)
    rounded(outer_choice_frame).to_csv(output / 'outer_fold_choices.csv', index=False)
    rounded(outer_prediction_frame).to_csv(output / 'outer_event_predictions.csv', index=False)
    rounded(outer_topone_frame).to_csv(output / 'outer_window_topone.csv', index=False)
    rounded(outer_selected_frame).to_csv(output / 'outer_selected_windows.csv', index=False)

    report = {
        'protocol': {
            'scope': 'Train-only nested sequence-disjoint multitask ranker with post-ranking actual-anchor motion hazard veto.',
            'ranker': {
                'features': '90 existing observable features plus 36 preregistered actual-anchor motion features.',
                'model': 'ExtraTreesRegressor(n_estimators=300,max_depth=7,min_samples_leaf=3,max_features=0.65).',
                'score': 'Mean normalized prediction across 13 qualified local teacher targets.',
                'seeds': SEEDS,
            },
            'motion_certificate_components': [
                {'name': name, 'feature': column, 'higher_is_better': higher}
                for name, column, higher in MOTION_COMPONENTS
            ],
            'motion_policy': 'Motion never reranks candidates. The multitask top-one is either retained or abstained.',
            'gate_configs': [asdict(config) for config in GATE_CONFIGS],
            'inner_selection': 'For each outer sequence, three inner sequence-disjoint folds choose the veto using full_idtp_delta_norm only. delta_HOTA is outer audit only.',
            'inner_eligibility': 'selected>=6, all 3 inner sequences covered, local precision>=0.70, positive/total utility>0, worst sequence>=0, negative mass<=20% positive sum, no local utility<=-0.05, stability>=0.80.',
            'global_eligibility': 'selected>=8, all 4 sequences covered, HOTA precision>=0.60, positive/total HOTA>0, worst sequence>=0, negative mass<=25% positive sum, no HOTA<=-0.05, stability>=0.80, and local total/worst>=0 with no local catastrophe.',
            'locked_labels_read': 0,
            'locked_trackeval_calls': 0,
            'remaining_locked_rows_untouched': 156,
        },
        'dataset': {
            'events': int(len(frame)),
            'sequences': SEQUENCES,
            'qualified_targets': int(len(targets)),
            'existing_features': int(len(MODEL_FEATURES)),
            'motion_features': int(len(compact_motion)),
            'total_model_features': int(len(model_features)),
            'windows': int(len(SEQUENCES) * len(TRAIN_WINDOWS)),
        },
        'outer_fold_choices': outer_choices,
        'outer_local_summary': local_summary,
        'outer_hota_summary': hota_summary,
        'deployment_allowed': deployment_allowed,
        'locked_manifest_created': False,
        'remaining_locked_labels_unread': 156,
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
