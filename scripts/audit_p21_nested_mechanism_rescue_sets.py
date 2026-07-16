from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer


KEYS = ['seq', 'canonical_rank', 'u', 'v', 'boundary_frame', 'transaction_type']
TARGET = 'full_idtp_delta_norm'
MECHANISMS = ['receiver_split', 'same_identity_merge']
GEOMETRY_FEATURES = [
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
ALPHA = 0.10
MAX_MEAN_ADDED = 12.0
MAX_COMBINED_SET_SIZE = 100


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


def conformal_quantile(values: np.ndarray, alpha: float) -> int:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 0
    level = min(1.0, math.ceil((len(values) + 1) * (1.0 - alpha)) / len(values))
    return int(math.ceil(float(np.quantile(values, level, method='higher'))))


def minimum_sample_size_for_nonmaximum_quantile(alpha: float) -> int:
    for size in range(1, 100000):
        order = math.ceil((size + 1) * (1.0 - alpha))
        if order < size:
            return size
    raise RuntimeError('failed to resolve finite-sample quantile threshold')


def assign_temporal_blocks(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result['temporal_block'] = -1
    for sequence, group in result.groupby('seq', sort=True):
        ordered = group.sort_values(
            ['effective_start_frame', 'canonical_rank', 'transaction_type']
        ).index
        blocks = np.minimum(3, (np.arange(len(ordered)) * 4) // len(ordered))
        result.loc[ordered, 'temporal_block'] = blocks
    if (result.temporal_block < 0).any():
        raise RuntimeError('temporal block assignment failed')
    return result


def add_mechanism_labels(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    positive = result[TARGET] > 0.0
    result['same_identity_merge_label'] = (
        positive
        & (result.full_history_dominant_same_gt == 1)
        & (result.full_future_dominant_gt == result.full_donor_dominant_gt)
    ).astype(int)
    result['receiver_split_label'] = (
        positive
        & (result.full_history_donor_matched == 0)
        & (result.full_history_receiver_matched > 0)
        & (result.full_future_matched > 0)
        & (result.full_receiver_dominant_gt != result.full_future_dominant_gt)
    ).astype(int)
    if (
        (result.same_identity_merge_label == 1)
        & (result.receiver_split_label == 1)
    ).any():
        raise RuntimeError('mechanism labels overlap')
    return result


def sequence_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.seq.value_counts()
    mapping = {
        sequence: len(frame) / (len(counts) * count)
        for sequence, count in counts.items()
    }
    return frame.seq.map(mapping).to_numpy(float)


def fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    seed: int,
) -> pd.DataFrame:
    imputer = SimpleImputer(strategy='median', add_indicator=True)
    x_train = imputer.fit_transform(train[features])
    x_test = imputer.transform(test[features])
    weights = sequence_weights(train)
    result = test[KEYS + ['temporal_block', TARGET] + [f'{m}_label' for m in MECHANISMS]].copy()
    for mechanism_index, mechanism in enumerate(MECHANISMS):
        label = f'{mechanism}_label'
        if train[label].nunique() != 2:
            raise RuntimeError(f'mechanism training fold lacks both classes: {mechanism}')
        model = ExtraTreesClassifier(
            n_estimators=256,
            max_depth=10,
            min_samples_leaf=3,
            max_features=0.75,
            class_weight='balanced',
            random_state=seed + 100 * mechanism_index,
            n_jobs=-1,
        )
        model.fit(x_train, train[label].astype(int), sample_weight=weights)
        # Quantize before ranking so parallel tree-reduction tail noise cannot
        # change deterministic tie ordering or serialized artifacts.
        result[f'{mechanism}_score'] = np.round(
            model.predict_proba(x_test)[:, 1], 12
        )
    return result


def add_block_ranks(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for mechanism in MECHANISMS:
        result[f'{mechanism}_rank'] = -1
    for (_, _), group in result.groupby(['seq', 'temporal_block'], sort=True):
        for mechanism in MECHANISMS:
            order = group.sort_values(
                [f'{mechanism}_score', 'boundary_frame', 'u', 'v', 'transaction_type'],
                ascending=[False, True, True, True, True],
            ).index
            result.loc[order, f'{mechanism}_rank'] = np.arange(1, len(order) + 1)
    return result


def calibration_rows(inner_predictions: pd.DataFrame, outer_sequence: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    ranked = add_block_ranks(inner_predictions)
    for mechanism in MECHANISMS:
        label = f'{mechanism}_label'
        rank = f'{mechanism}_rank'
        for (sequence, block), group in ranked.groupby(['seq', 'temporal_block'], sort=True):
            positives = group[group[label] == 1]
            if positives.empty:
                continue
            rows.append(
                {
                    'outer_sequence': outer_sequence,
                    'mechanism': mechanism,
                    'calibration_sequence': sequence,
                    'temporal_block': int(block),
                    'positive_events': int(len(positives)),
                    'minimum_positive_rank': int(positives[rank].min()),
                }
            )
    return pd.DataFrame(rows)


def build_rescue_members(
    predictions: pd.DataFrame,
    base_keys: pd.DataFrame,
    radii: dict[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranked = add_block_ranks(predictions)
    ranked = ranked.merge(
        base_keys.assign(base_member=1), on=KEYS, how='left', validate='one_to_one'
    )
    ranked['base_member'] = ranked.base_member.fillna(0).astype(int)
    reasons: dict[tuple[object, ...], list[str]] = {}
    for mechanism in MECHANISMS:
        radius = int(radii[mechanism])
        if radius <= 0:
            continue
        mask = (ranked[f'{mechanism}_rank'] <= radius) & (ranked.base_member == 0)
        for key in ranked.loc[mask, KEYS].itertuples(index=False, name=None):
            reasons.setdefault(tuple(key), []).append(mechanism)
    if reasons:
        rescue = ranked.merge(
            pd.DataFrame(
                [dict(zip(KEYS, key), rescue_reason='+'.join(value)) for key, value in reasons.items()]
            ),
            on=KEYS,
            how='inner',
            validate='one_to_one',
        )
    else:
        rescue = ranked.iloc[0:0].copy()
        rescue['rescue_reason'] = pd.Series(dtype=str)
    return ranked, rescue


def block_summary(
    full: pd.DataFrame,
    base_keys: pd.DataFrame,
    rescue_keys: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    membership = base_keys.assign(base_member=1).merge(
        rescue_keys.assign(rescue_member=1), on=KEYS, how='outer'
    )
    membership['base_member'] = membership.base_member.fillna(0).astype(int)
    membership['rescue_member'] = membership.rescue_member.fillna(0).astype(int)
    members = full.merge(membership, on=KEYS, how='inner', validate='one_to_one')
    rows = []
    for (sequence, block), group in full.groupby(['seq', 'temporal_block'], sort=True):
        selected = members[(members.seq == sequence) & (members.temporal_block == block)]
        base = selected[selected.base_member == 1]
        rows.append(
            {
                'seq': sequence,
                'temporal_block': int(block),
                'events': int(len(group)),
                'block_has_positive': int((group[TARGET] > 0.0).any()),
                'base_set_size': int(len(base)),
                'rescue_added': int((selected.rescue_member == 1).sum()),
                'combined_set_size': int(len(selected)),
                'base_has_positive': int((base[TARGET] > 0.0).any()),
                'combined_has_positive': int((selected[TARGET] > 0.0).any()),
                'combined_positive_events': int((selected[TARGET] > 0.0).sum()),
                'combined_negative_events': int((selected[TARGET] < 0.0).sum()),
                'combined_zero_events': int((selected[TARGET] == 0.0).sum()),
                'combined_oracle_utility': float(selected[TARGET].max()) if len(selected) else 0.0,
            }
        )
    return pd.DataFrame(rows), members


def summarize_mechanisms(full: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    rows = []
    member_keys = members[KEYS].drop_duplicates()
    tagged = full.merge(member_keys.assign(in_combined_set=1), on=KEYS, how='left')
    tagged['in_combined_set'] = tagged.in_combined_set.fillna(0).astype(int)
    for mechanism in MECHANISMS:
        label = f'{mechanism}_label'
        positives = tagged[tagged[label] == 1]
        block_total = positives[['seq', 'temporal_block']].drop_duplicates()
        covered = positives[positives.in_combined_set == 1][['seq', 'temporal_block']].drop_duplicates()
        rows.append(
            {
                'mechanism': mechanism,
                'positive_events': int(len(positives)),
                'covered_positive_events': int(positives.in_combined_set.sum()),
                'event_recall': float(positives.in_combined_set.mean()) if len(positives) else 0.0,
                'positive_blocks': int(len(block_total)),
                'covered_positive_blocks': int(len(covered)),
                'block_recall': float(len(covered) / len(block_total)) if len(block_total) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--event-labels', required=True)
    parser.add_argument('--executability', required=True)
    parser.add_argument('--change-features', required=True)
    parser.add_argument('--change-feature-list', required=True)
    parser.add_argument('--motion-features', required=True)
    parser.add_argument('--motion-feature-list', required=True)
    parser.add_argument('--base-set-members', required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()

    labels = pd.read_csv(args.event_labels)
    executable = pd.read_csv(args.executability)
    change = pd.read_csv(args.change_features)
    change_features = pd.read_csv(args.change_feature_list).feature.astype(str).tolist()
    motion = pd.read_csv(args.motion_features)
    motion_features = pd.read_csv(args.motion_feature_list).feature.astype(str).tolist()
    base_keys = pd.read_csv(args.base_set_members, usecols=KEYS).drop_duplicates()

    executable_start = executable[KEYS + ['effective_start_frame']].copy()
    if 'effective_start_frame' in labels.columns:
        check = labels[KEYS + ['effective_start_frame']].merge(
            executable_start,
            on=KEYS,
            how='inner',
            validate='one_to_one',
            suffixes=('_labels', '_executable'),
        )
        if len(check) != len(labels) or not np.array_equal(
            check.effective_start_frame_labels.to_numpy(int),
            check.effective_start_frame_executable.to_numpy(int),
        ):
            raise RuntimeError('effective_start_frame mismatch between labels and executability')
        frame = labels.copy()
    else:
        frame = labels.merge(
            executable_start, on=KEYS, how='inner', validate='one_to_one'
        )
    frame = frame.merge(change[KEYS + change_features], on=KEYS, how='inner', validate='one_to_one')
    frame = frame.merge(motion[KEYS + motion_features], on=KEYS, how='inner', validate='one_to_one')
    frame = assign_temporal_blocks(add_mechanism_labels(frame))
    features = list(dict.fromkeys(GEOMETRY_FEATURES + change_features + motion_features))
    missing = [feature for feature in features if feature not in frame.columns]
    if missing:
        raise RuntimeError(f'missing model features: {missing}')
    forbidden_tokens = ['gt', 'idtp', 'utility', 'trackeval', 'matched', 'label']
    forbidden = [
        feature for feature in features if any(token in feature.lower() for token in forbidden_tokens)
    ]
    if forbidden:
        raise RuntimeError(f'forbidden model features: {forbidden}')
    if len(frame) != 11705 or frame.duplicated(KEYS).any():
        raise RuntimeError('full-bank key integrity failure')
    if len(base_keys) == 0 or base_keys.duplicated(KEYS).any():
        raise RuntimeError('base-set key integrity failure')

    sequences = sorted(frame.seq.unique())
    outer_predictions: list[pd.DataFrame] = []
    rescue_members: list[pd.DataFrame] = []
    calibration_parts: list[pd.DataFrame] = []
    radius_rows: list[dict[str, object]] = []
    for outer_index, held_out in enumerate(sequences):
        outer_train = frame[frame.seq != held_out].reset_index(drop=True)
        outer_test = frame[frame.seq == held_out].reset_index(drop=True)
        inner_parts = []
        for inner_index, inner_held_out in enumerate(sorted(outer_train.seq.unique())):
            inner_train = outer_train[outer_train.seq != inner_held_out].reset_index(drop=True)
            inner_test = outer_train[outer_train.seq == inner_held_out].reset_index(drop=True)
            inner_parts.append(
                fit_predict(
                    inner_train,
                    inner_test,
                    features,
                    seed=41000 + 1000 * outer_index + 10 * inner_index,
                )
            )
        inner_predictions = pd.concat(inner_parts, ignore_index=True)
        calibration = calibration_rows(inner_predictions, held_out)
        calibration_parts.append(calibration)
        radii: dict[str, int] = {}
        for mechanism in MECHANISMS:
            values = calibration.loc[
                calibration.mechanism == mechanism, 'minimum_positive_rank'
            ].to_numpy(float)
            radii[mechanism] = conformal_quantile(values, ALPHA)
            radius_rows.append(
                {
                    'outer_sequence': held_out,
                    'mechanism': mechanism,
                    'calibration_positive_blocks': int(len(values)),
                    'conformal_rank_radius': int(radii[mechanism]),
                    'maximum_calibration_rank': int(np.max(values)) if len(values) else 0,
                    'median_calibration_rank': float(np.median(values)) if len(values) else np.nan,
                    'quantile_uses_maximum': int(
                        len(values) > 0 and radii[mechanism] == int(np.max(values))
                    ),
                    'radius_to_median_ratio': float(
                        radii[mechanism] / max(float(np.median(values)), 1.0)
                    ) if len(values) else np.nan,
                }
            )
        prediction = fit_predict(
            outer_train,
            outer_test,
            features,
            seed=51000 + 1000 * outer_index,
        )
        outer_base = base_keys[base_keys.seq == held_out].copy()
        ranked, rescue = build_rescue_members(prediction, outer_base, radii)
        ranked['outer_sequence'] = held_out
        rescue['outer_sequence'] = held_out
        outer_predictions.append(ranked)
        rescue_members.append(rescue)

    prediction_frame = pd.concat(outer_predictions, ignore_index=True)
    rescue_frame = pd.concat(rescue_members, ignore_index=True)
    calibration_frame = pd.concat(calibration_parts, ignore_index=True)
    radius_frame = pd.DataFrame(radius_rows)
    rescue_keys = rescue_frame[KEYS].drop_duplicates()
    blocks, combined_members = block_summary(frame, base_keys, rescue_keys)
    mechanism_summary = summarize_mechanisms(frame, combined_members)

    positive_blocks = blocks[blocks.block_has_positive == 1]
    sequence_oracle = blocks.groupby('seq').combined_oracle_utility.sum().reindex(sequences, fill_value=0.0)
    newly_covered = int(
        ((blocks.base_has_positive == 0) & (blocks.combined_has_positive == 1)).sum()
    )
    coverage = float(positive_blocks.combined_has_positive.mean())
    mean_added = float(blocks.rescue_added.mean())
    maximum_size = int(blocks.combined_set_size.max())
    retained = bool(
        coverage == 1.0
        and newly_covered >= 1
        and mean_added <= MAX_MEAN_ADDED
        and maximum_size <= MAX_COMBINED_SET_SIZE
        and float(sequence_oracle.min()) >= 0.0
    )
    minimum_nonmaximum_size = minimum_sample_size_for_nonmaximum_quantile(ALPHA)
    maximum_quantile_folds = int(radius_frame.quantile_uses_maximum.sum())

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    rounded(prediction_frame).to_csv(out_dir / 'outer_mechanism_predictions.csv', index=False)
    rounded(rescue_frame).to_csv(out_dir / 'rescue_set_members.csv', index=False)
    rounded(combined_members).to_csv(out_dir / 'combined_set_members.csv', index=False)
    rounded(blocks).to_csv(out_dir / 'block_summary.csv', index=False)
    rounded(mechanism_summary).to_csv(out_dir / 'mechanism_summary.csv', index=False)
    rounded(calibration_frame).to_csv(out_dir / 'inner_positive_rank_calibration.csv', index=False)
    rounded(radius_frame).to_csv(out_dir / 'outer_rank_radii.csv', index=False)
    pd.DataFrame({'feature': features}).to_csv(out_dir / 'model_feature_list.csv', index=False)

    report = {
        'protocol': {
            'scope': 'Strict nested sequence-LOSO mechanism-aware conformal rescue set.',
            'base_set': 'Frozen P20 rank-conformal candidate set; its generic radius is not changed.',
            'mechanisms': {
                'receiver_split': 'Positive intervention with unmatched donor history and a receiver-history to future dominant-identity change.',
                'same_identity_merge': 'Positive intervention whose donor, receiver, and future teacher identities agree.',
            },
            'model': {
                'family': 'ExtraTreesClassifier',
                'n_estimators': 256,
                'max_depth': 10,
                'min_samples_leaf': 3,
                'max_features': 0.75,
                'class_weight': 'balanced',
                'sequence_equal_weighting': True,
            },
            'rank_calibration': 'Finite-sample higher conformal quantile of the minimum mechanism-positive rank in completely inner-OOF source blocks.',
            'alpha': ALPHA,
            'fixed_acceptance': {
                'positive_block_coverage': 1.0,
                'minimum_newly_covered_blocks': 1,
                'maximum_mean_added_candidates': MAX_MEAN_ADDED,
                'maximum_combined_set_size': MAX_COMBINED_SET_SIZE,
                'minimum_worst_sequence_set_oracle_utility': 0.0,
            },
            'model_or_threshold_sweep': False,
            'global_trackeval_calls': 0,
            'p15_locked_labels_read': 0,
            'p15_locked_trackeval_calls': 0,
            'p15_remaining_locked_rows_untouched': 156,
        },
        'dataset': {
            'events': int(len(frame)),
            'sequences': sequences,
            'temporal_blocks': int(len(blocks)),
            'positive_available_blocks': int(len(positive_blocks)),
            'receiver_split_positive_events': int(frame.receiver_split_label.sum()),
            'same_identity_merge_positive_events': int(frame.same_identity_merge_label.sum()),
            'model_features': int(len(features)),
            'forbidden_model_features': forbidden,
        },
        'rank_radii': rounded(radius_frame).to_dict('records'),
        'result': {
            'base_covered_positive_blocks': int(positive_blocks.base_has_positive.sum()),
            'combined_covered_positive_blocks': int(positive_blocks.combined_has_positive.sum()),
            'conditional_positive_coverage': coverage,
            'newly_covered_blocks': newly_covered,
            'mean_base_set_size': float(blocks.base_set_size.mean()),
            'mean_rescue_added': mean_added,
            'mean_combined_set_size': float(blocks.combined_set_size.mean()),
            'maximum_combined_set_size': maximum_size,
            'rescue_unique_events': int(len(rescue_keys)),
            'combined_positive_events': int((combined_members[TARGET] > 0.0).sum()),
            'combined_negative_events': int((combined_members[TARGET] < 0.0).sum()),
            'combined_zero_events': int((combined_members[TARGET] == 0.0).sum()),
            'combined_set_oracle_utility_sum': float(blocks.combined_oracle_utility.sum()),
            'combined_set_oracle_worst_sequence': float(sequence_oracle.min()),
            'mechanism_summary': rounded(mechanism_summary).to_dict('records'),
        },
        'calibration_diagnostics': {
            'minimum_positive_blocks_for_nonmaximum_quantile': minimum_nonmaximum_size,
            'outer_mechanism_calibrations': int(len(radius_frame)),
            'calibrations_using_maximum_rank': maximum_quantile_folds,
            'all_calibrations_use_maximum_rank': bool(
                maximum_quantile_folds == len(radius_frame)
            ),
            'median_receiver_split_radius_to_median_ratio': float(
                radius_frame.loc[
                    radius_frame.mechanism == 'receiver_split', 'radius_to_median_ratio'
                ].median()
            ),
            'median_same_identity_merge_radius_to_median_ratio': float(
                radius_frame.loc[
                    radius_frame.mechanism == 'same_identity_merge', 'radius_to_median_ratio'
                ].median()
            ),
            'interpretation': 'At alpha=0.10, fewer than 19 positive calibration blocks force the finite-sample higher conformal quantile to equal the maximum observed positive rank.',
        },
        'decision': {
            'mechanism_rescue_set_retained': retained,
            'deployment_allowed': False,
            'locked_manifest_created': False,
            'p15_policy_changed': False,
            'reason': (
                'The mechanism-aware rescue set closes the remaining positive-block coverage gap within the preregistered efficiency limits.'
                if retained
                else 'The mechanism models close the remaining positive-block coverage gap, but block-level finite-sample conformal calibration is forced to use the maximum source-block rank and violates the preregistered efficiency limits.'
            ),
            'next_stage': (
                'Develop a mechanism-conditioned within-set pairwise risk certificate and fail-closed single-action selector; keep P20 and P21 retrieval radii frozen.'
                if retained
                else 'Replace block-rank conformal calibration with cluster-aware event-level risk control or structured pairwise mechanism certificates; keep the P20 generic radius frozen and do not tune scalar utility gates.'
            ),
        },
    }
    (out_dir / 'report.json').write_text(
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
