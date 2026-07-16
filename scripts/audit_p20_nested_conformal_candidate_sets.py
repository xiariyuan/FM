from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler


EVENT_KEYS = ['seq', 'canonical_rank', 'u', 'v', 'boundary_frame', 'transaction_type']
TARGET = 'full_idtp_delta_norm'
MODELS = ['geometry_positive', 'geometry_utility', 'motion_multitask']
ALPHA = 0.10
OOD_P_MIN = 0.10
PCA_COMPONENTS = 15


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


def load_p19_module(path: str):
    spec = importlib.util.spec_from_file_location('p19_audit', path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot import P19 module: {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def conformal_quantile(values: np.ndarray, alpha: float) -> int:
    values = np.asarray(values, dtype=int)
    if not len(values):
        return 0
    level = min(1.0, math.ceil((len(values) + 1) * (1.0 - alpha)) / len(values))
    return int(np.quantile(values, level, method='higher'))


def add_block_ranks(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for model in MODELS:
        result[f'{model}_rank'] = result.groupby(
            ['seq', 'temporal_block'], sort=False
        )[model].rank(method='first', ascending=False).astype(int)
    result['union_rank'] = result[
        [f'{model}_rank' for model in MODELS]
    ].min(axis=1).astype(int)
    return result


def block_descriptor(group: pd.DataFrame, radius: int) -> dict[str, object]:
    record: dict[str, object] = {
        'seq': str(group.seq.iloc[0]),
        'temporal_block': int(group.temporal_block.iloc[0]),
        'events': int(len(group)),
        'log_events': float(np.log1p(len(group))),
        'set_radius': int(radius),
    }
    top_sets: dict[str, set[int]] = {}
    for model in MODELS:
        values = np.sort(group[model].to_numpy(float))[::-1]
        for quantile in [0.50, 0.75, 0.90, 0.95, 0.99]:
            record[f'{model}_q{int(100 * quantile)}'] = float(
                np.quantile(values, quantile)
            )
        record[f'{model}_std'] = float(np.std(values))
        record[f'{model}_range'] = float(values[0] - values[-1])
        record[f'{model}_gap12'] = float(values[0] - values[min(1, len(values) - 1)])
        record[f'{model}_gap1k'] = float(
            values[0] - values[min(max(radius - 1, 0), len(values) - 1)]
        )
        top_sets[model] = set(
            group.nsmallest(min(radius, len(group)), f'{model}_rank').index.tolist()
        )
    pairs = [
        ('geometry_positive', 'geometry_utility'),
        ('geometry_positive', 'motion_multitask'),
        ('geometry_utility', 'motion_multitask'),
    ]
    for left, right in pairs:
        intersection = len(top_sets[left] & top_sets[right])
        union = len(top_sets[left] | top_sets[right])
        record[f'overlap_{left}_{right}'] = float(
            intersection / max(1, min(len(top_sets[left]), len(top_sets[right])))
        )
        record[f'jaccard_{left}_{right}'] = float(intersection / max(1, union))
    record['union_size'] = int(len(set().union(*top_sets.values())))
    return record


def robust_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    median = np.nanmedian(values, axis=0)
    q25 = np.nanquantile(values, 0.25, axis=0)
    q75 = np.nanquantile(values, 0.75, axis=0)
    scale = np.where(q75 - q25 > 1e-9, q75 - q25, 1.0)
    return median, scale


def block_ood(
    inner_predictions: pd.DataFrame,
    outer_predictions: pd.DataFrame,
    radius: int,
) -> pd.DataFrame:
    inner = pd.DataFrame(
        [
            block_descriptor(group, radius)
            for _, group in inner_predictions.groupby(
                ['seq', 'temporal_block'], sort=True
            )
        ]
    )
    outer = pd.DataFrame(
        [
            block_descriptor(group, radius)
            for _, group in outer_predictions.groupby(
                ['seq', 'temporal_block'], sort=True
            )
        ]
    )
    feature_columns = [
        column for column in inner.columns if column not in ['seq', 'temporal_block']
    ]
    x_inner = inner[feature_columns].to_numpy(float)
    x_outer = outer[feature_columns].to_numpy(float)
    median, scale = robust_scale(x_inner)
    z_inner = np.nan_to_num((x_inner - median) / scale)
    z_outer = np.nan_to_num((x_outer - median) / scale)
    calibration = []
    for index, row in inner.iterrows():
        cross_sequence = inner.seq.to_numpy() != row.seq
        distances = np.sqrt(
            np.mean((z_inner[cross_sequence] - z_inner[index]) ** 2, axis=1)
        )
        calibration.append(float(distances.min()))
    calibration_array = np.asarray(calibration, dtype=float)
    records = []
    for index, row in outer.iterrows():
        distance = float(
            np.sqrt(np.mean((z_inner - z_outer[index]) ** 2, axis=1)).min()
        )
        p_value = float(
            (1 + np.sum(calibration_array >= distance))
            / (len(calibration_array) + 1)
        )
        records.append(
            {
                'seq': str(row.seq),
                'temporal_block': int(row.temporal_block),
                'ood_distance': distance,
                'ood_p_value': p_value,
                'ood_certificate_pass': int(p_value >= OOD_P_MIN),
            }
        )
    return pd.DataFrame(records)


def positive_support_sets(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    imputer = SimpleImputer(strategy='median')
    scaler = RobustScaler(quantile_range=(25, 75))
    x_train = scaler.fit_transform(imputer.fit_transform(train[feature_columns]))
    x_test = scaler.transform(imputer.transform(test[feature_columns]))
    components = min(PCA_COMPONENTS, x_train.shape[1], max(1, len(x_train) - 1))
    pca = PCA(n_components=components, random_state=0)
    z_train = pca.fit_transform(x_train)
    z_test = pca.transform(x_test)
    positive = train[TARGET] > 0.0
    negative = train[TARGET] < 0.0
    calibration_scores: list[float] = []
    for sequence in sorted(train.seq.unique()):
        query_indices = np.flatnonzero(
            (train.seq.to_numpy() == sequence) & positive.to_numpy()
        )
        positive_reference = np.flatnonzero(
            (train.seq.to_numpy() != sequence) & positive.to_numpy()
        )
        negative_reference = np.flatnonzero(
            (train.seq.to_numpy() != sequence) & negative.to_numpy()
        )
        if not len(query_indices):
            continue
        if not len(positive_reference) or not len(negative_reference):
            raise RuntimeError('positive-support calibration lacks cross-sequence references')
        positive_distance = NearestNeighbors(n_neighbors=1).fit(
            z_train[positive_reference]
        ).kneighbors(z_train[query_indices], return_distance=True)[0][:, 0]
        negative_distance = NearestNeighbors(n_neighbors=1).fit(
            z_train[negative_reference]
        ).kneighbors(z_train[query_indices], return_distance=True)[0][:, 0]
        calibration_scores.extend(
            np.log((positive_distance + 1e-6) / (negative_distance + 1e-6)).tolist()
        )
    calibration_array = np.asarray(calibration_scores, dtype=float)
    threshold_level = min(
        1.0,
        math.ceil((len(calibration_array) + 1) * (1.0 - ALPHA))
        / len(calibration_array),
    )
    threshold = float(
        np.quantile(calibration_array, threshold_level, method='higher')
    )
    positive_indices = np.flatnonzero(positive.to_numpy())
    negative_indices = np.flatnonzero(negative.to_numpy())
    positive_distance = NearestNeighbors(n_neighbors=1).fit(
        z_train[positive_indices]
    ).kneighbors(z_test, return_distance=True)[0][:, 0]
    negative_distance = NearestNeighbors(n_neighbors=1).fit(
        z_train[negative_indices]
    ).kneighbors(z_test, return_distance=True)[0][:, 0]
    nonconformity = np.log(
        (positive_distance + 1e-6) / (negative_distance + 1e-6)
    )
    p_values = np.asarray(
        [
            (1 + np.sum(calibration_array >= value))
            / (len(calibration_array) + 1)
            for value in nonconformity
        ],
        dtype=float,
    )
    scored = test.copy()
    scored['positive_support_nonconformity'] = nonconformity
    scored['positive_support_p_value'] = p_values
    scored['positive_support_pass'] = (nonconformity <= threshold).astype(int)
    summaries = []
    members = []
    for (sequence, block), group in scored.groupby(
        ['seq', 'temporal_block'], sort=True
    ):
        candidate_set = group[group.positive_support_pass == 1].copy()
        positive_group = group[group[TARGET] > 0.0]
        summaries.append(
            {
                'seq': sequence,
                'temporal_block': int(block),
                'events': int(len(group)),
                'set_size': int(len(candidate_set)),
                'block_has_positive': int(len(positive_group) > 0),
                'set_has_positive': int((candidate_set[TARGET] > 0.0).any()),
                'set_positive_count': int((candidate_set[TARGET] > 0.0).sum()),
                'set_negative_count': int((candidate_set[TARGET] < 0.0).sum()),
                'set_max_utility': float(candidate_set[TARGET].max()),
                'set_min_utility': float(candidate_set[TARGET].min()),
            }
        )
        if len(candidate_set):
            members.append(
                candidate_set[
                    EVENT_KEYS
                    + [
                        'temporal_block',
                        TARGET,
                        'positive_support_nonconformity',
                        'positive_support_p_value',
                    ]
                ]
            )
    member_frame = (
        pd.concat(members, ignore_index=True)
        if members
        else pd.DataFrame()
    )
    metadata = {
        'calibration_positive_events': int(len(calibration_array)),
        'nonconformity_threshold': threshold,
        'pca_components': int(components),
        'pca_explained_variance': float(pca.explained_variance_ratio_.sum()),
    }
    return pd.DataFrame(summaries), member_frame, metadata


def mechanism_label(row: pd.Series) -> str:
    donor_gt = int(row.full_donor_dominant_gt)
    receiver_gt = int(row.full_receiver_dominant_gt)
    future_gt = int(row.full_future_dominant_gt)
    if int(row.full_history_dominant_same_gt) == 1:
        return 'same_identity_merge'
    if donor_gt < 0 and receiver_gt >= 0 and future_gt >= 0 and receiver_gt != future_gt:
        return 'ephemeral_anchor_receiver_split'
    return 'other_uncovered_positive'


def set_summary(frame: pd.DataFrame, sequences: list[str]) -> dict[str, object]:
    positive_blocks = frame[frame.block_has_positive == 1]
    oracle_by_sequence = frame.groupby('seq').set_max_utility.sum().reindex(
        sequences, fill_value=0.0
    )
    return {
        'blocks': int(len(frame)),
        'positive_available_blocks': int(len(positive_blocks)),
        'covered_positive_blocks': int(positive_blocks.set_has_positive.sum()),
        'conditional_positive_coverage': float(
            positive_blocks.set_has_positive.mean()
        ),
        'mean_set_size': float(frame.set_size.mean()),
        'median_set_size': float(frame.set_size.median()),
        'maximum_set_size': int(frame.set_size.max()),
        'positive_members': int(frame.set_positive_count.sum()),
        'negative_members': int(frame.set_negative_count.sum()),
        'oracle_set_utility_sum': float(frame.set_max_utility.sum()),
        'oracle_set_worst_sequence': float(oracle_by_sequence.min()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--p19-script', required=True)
    parser.add_argument('--event-labels', required=True)
    parser.add_argument('--motion-features', required=True)
    parser.add_argument('--motion-feature-list', required=True)
    parser.add_argument('--qualification-report', required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()

    p19 = load_p19_module(args.p19_script)
    labels = pd.read_csv(args.event_labels)
    motion = pd.read_csv(args.motion_features)
    motion_features = pd.read_csv(args.motion_feature_list).feature.astype(str).tolist()
    qualification = json.loads(Path(args.qualification_report).read_text())
    targets = [str(target) for target in qualification['qualified_auxiliary_targets']]
    frame = labels.merge(
        motion[EVENT_KEYS + motion_features],
        on=EVENT_KEYS,
        how='inner',
        validate='one_to_one',
    )
    frame = p19.assign_temporal_blocks(p19.add_geometry_features(frame))
    geometry_features = p19.GEOMETRY_BASE_FEATURES + p19.GEOMETRY_DERIVED_FEATURES
    support_features = geometry_features + motion_features
    sequences = sorted(frame.seq.unique())

    calibration_records = []
    rank_block_records = []
    rank_members = []
    ood_records = []
    support_block_records = []
    support_members = []
    support_fold_records = []
    forensic_records = []

    for outer_index, held_out in enumerate(sequences):
        outer_train = frame[frame.seq != held_out].reset_index(drop=True)
        outer_test = frame[frame.seq == held_out].reset_index(drop=True)
        inner_parts = []
        for inner_index, inner_held_out in enumerate(sorted(outer_train.seq.unique())):
            inner_train = outer_train[
                outer_train.seq != inner_held_out
            ].reset_index(drop=True)
            inner_test = outer_train[
                outer_train.seq == inner_held_out
            ].reset_index(drop=True)
            inner_parts.append(
                p19.fit_fold_models(
                    inner_train,
                    inner_test,
                    geometry_features,
                    motion_features,
                    targets,
                    fold=100 + 10 * outer_index + inner_index,
                )
            )
        inner_predictions = add_block_ranks(
            pd.concat(inner_parts, ignore_index=True)
        )
        positive_ranks = []
        for (sequence, block), group in inner_predictions.groupby(
            ['seq', 'temporal_block'], sort=True
        ):
            positive = group[group[TARGET] > 0.0]
            minimum_rank = (
                int(positive.union_rank.min()) if len(positive) else np.nan
            )
            calibration_records.append(
                {
                    'outer_held_out': held_out,
                    'source_seq': sequence,
                    'temporal_block': int(block),
                    'events': int(len(group)),
                    'positive_events': int(len(positive)),
                    'minimum_positive_union_rank': minimum_rank,
                }
            )
            if len(positive):
                positive_ranks.append(minimum_rank)
        radius = conformal_quantile(np.asarray(positive_ranks), ALPHA)
        outer_predictions = add_block_ranks(
            p19.fit_fold_models(
                outer_train,
                outer_test,
                geometry_features,
                motion_features,
                targets,
                fold=1000 + outer_index,
            )
        )
        ood = block_ood(inner_predictions, outer_predictions, radius)
        ood['outer_held_out'] = held_out
        ood_records.append(ood)

        for (sequence, block), group in outer_predictions.groupby(
            ['seq', 'temporal_block'], sort=True
        ):
            candidate_set = group[group.union_rank <= radius].copy()
            positive = group[group[TARGET] > 0.0]
            rank_block_records.append(
                {
                    'seq': sequence,
                    'temporal_block': int(block),
                    'set_radius': int(radius),
                    'events': int(len(group)),
                    'set_size': int(len(candidate_set)),
                    'block_has_positive': int(len(positive) > 0),
                    'set_has_positive': int((candidate_set[TARGET] > 0.0).any()),
                    'set_positive_count': int((candidate_set[TARGET] > 0.0).sum()),
                    'set_negative_count': int((candidate_set[TARGET] < 0.0).sum()),
                    'set_max_utility': float(candidate_set[TARGET].max()),
                    'set_min_utility': float(candidate_set[TARGET].min()),
                }
            )
            member_columns = EVENT_KEYS + [
                'temporal_block',
                TARGET,
                *MODELS,
                *[f'{model}_rank' for model in MODELS],
                'union_rank',
            ]
            rank_members.append(candidate_set[member_columns])
            if len(positive) and not (candidate_set[TARGET] > 0.0).any():
                missed = positive.copy()
                missed['mechanism'] = missed.apply(mechanism_label, axis=1)
                forensic_columns = EVENT_KEYS + [
                    'temporal_block',
                    TARGET,
                    'mechanism',
                    *MODELS,
                    *[f'{model}_rank' for model in MODELS],
                    'union_rank',
                    'donor_anchor',
                    'receiver_anchor',
                    'changed_rows',
                    'pair_common_frames',
                    'donor_history_rows',
                    'receiver_history_rows',
                    'future_receiver_rows',
                    'full_donor_dominant_gt',
                    'full_receiver_dominant_gt',
                    'full_future_dominant_gt',
                    'full_history_dominant_same_gt',
                    'full_idtp_delta',
                ]
                forensic_records.append(missed[forensic_columns])

        support_blocks, support_member_frame, support_metadata = positive_support_sets(
            outer_train,
            outer_test,
            support_features,
        )
        support_blocks['outer_held_out'] = held_out
        support_block_records.append(support_blocks)
        if len(support_member_frame):
            support_members.append(support_member_frame)
        support_fold_records.append({'outer_held_out': held_out, **support_metadata})

    calibration_frame = rounded(pd.DataFrame(calibration_records))
    rank_block_frame = rounded(pd.DataFrame(rank_block_records))
    rank_member_frame = rounded(pd.concat(rank_members, ignore_index=True))
    ood_frame = rounded(pd.concat(ood_records, ignore_index=True))
    support_block_frame = rounded(pd.concat(support_block_records, ignore_index=True))
    support_member_frame = rounded(pd.concat(support_members, ignore_index=True))
    support_fold_frame = rounded(pd.DataFrame(support_fold_records))
    forensic_frame = rounded(
        pd.concat(forensic_records, ignore_index=True)
        if forensic_records
        else pd.DataFrame()
    )

    rank_report = set_summary(rank_block_frame, sequences)
    support_report = set_summary(support_block_frame, sequences)
    ood_joined = rank_block_frame.merge(
        ood_frame[
            ['seq', 'temporal_block', 'ood_p_value', 'ood_certificate_pass']
        ],
        on=['seq', 'temporal_block'],
        how='left',
        validate='one_to_one',
    )
    ood_certified = ood_joined[ood_joined.ood_certificate_pass == 1]
    ood_positive = ood_certified[ood_certified.block_has_positive == 1]
    mechanism_counts = (
        forensic_frame.mechanism.value_counts().sort_index().to_dict()
        if len(forensic_frame)
        else {}
    )
    report = {
        'protocol': {
            'scope': 'Strict nested sequence-LOSO set-valued candidate retrieval audit.',
            'rank_set': 'Union of the top-k events from three fixed P19 views, represented by minimum within-block rank.',
            'rank_radius_calibration': 'Finite-sample higher conformal quantile of the minimum positive union rank in completely inner-OOF source blocks.',
            'alpha': ALPHA,
            'block_ood': 'Nearest cross-sequence block descriptor distance with a pooled conformal p-value.',
            'block_ood_p_min': OOD_P_MIN,
            'positive_support_set': 'Cross-sequence positive-vs-negative nearest-neighbor nonconformity in source-only robust-scaled PCA space.',
            'pca_components': PCA_COMPONENTS,
            'model_or_threshold_sweep': False,
            'global_trackeval_calls': 0,
            'p15_locked_labels_read': 0,
            'p15_locked_trackeval_calls': 0,
            'p15_remaining_locked_rows_untouched': 156,
        },
        'dataset': {
            'events': int(len(frame)),
            'sequences': sequences,
            'temporal_blocks': int(
                frame[['seq', 'temporal_block']].drop_duplicates().shape[0]
            ),
            'positive_events': int((frame[TARGET] > 0.0).sum()),
            'negative_events': int((frame[TARGET] < 0.0).sum()),
            'zero_events': int((frame[TARGET] == 0.0).sum()),
        },
        'rank_conformal_set': rank_report,
        'positive_support_set': support_report,
        'block_ood_certificate': {
            'certified_blocks': int(len(ood_certified)),
            'certified_sequences': int(ood_certified.seq.nunique()),
            'certified_positive_blocks': int(len(ood_positive)),
            'covered_certified_positive_blocks': int(
                ood_positive.set_has_positive.sum()
            ),
            'conditional_positive_coverage': float(
                ood_positive.set_has_positive.mean()
            ),
            'missed_certified_positive_blocks': int(
                (ood_positive.set_has_positive == 0).sum()
            ),
        },
        'uncovered_positive_forensic': {
            'events': int(len(forensic_frame)),
            'mechanism_counts': mechanism_counts,
        },
        'decision': {
            'deployment_allowed': False,
            'locked_manifest_created': False,
            'p15_policy_changed': False,
            'rank_conformal_set_retained': True,
            'positive_support_set_retained': False,
            'current_block_ood_certificate_retained': False,
            'reason': 'The nested rank set exceeds the fixed 90% conditional coverage target with substantially smaller sets, but it is not an executable single-action policy. The positive-support set closes the one remaining coverage miss only by expanding to hundreds of candidates, and the block-level OOD certificate does not identify the miss.',
            'next_stage': 'Build receiver change-point and ephemeral-anchor split features, then use them as a mechanism-specific rescue head inside the conformal candidate set. Do not enlarge the generic rank radius or tune scalar gates.',
        },
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    outputs = {
        'inner_positive_rank_calibration.csv': calibration_frame,
        'rank_conformal_block_summary.csv': rank_block_frame,
        'rank_conformal_set_members.csv': rank_member_frame,
        'block_ood_certificate.csv': ood_frame,
        'positive_support_block_summary.csv': support_block_frame,
        'positive_support_set_members.csv': support_member_frame,
        'positive_support_fold_summary.csv': support_fold_frame,
        'uncovered_positive_forensic.csv': forensic_frame,
    }
    for filename, output in outputs.items():
        output.to_csv(out_dir / filename, index=False)
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
