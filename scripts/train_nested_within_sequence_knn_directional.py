from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler

from train_locked_loso_directional_handoff_utility import (
    COMPACT_BASE_FEATURES,
    DIRECTIONAL_FEATURES,
    DIRECTION_KEYS,
    attach_train_labels,
    load_directional_features,
)


TRAIN_WINDOWS = [(21, 40), (41, 60), (61, 80), (81, 100)]
LOCKED_WINDOWS = [(1, 20)]

MECHANISM_FEATURES = [
    'direction_u_to_v',
    'executor_changed_rows_log1p',
    'executor_handoff_delay',
    'executor_impact_receiver_future',
    'executor_impact_pair_future',
    'donor_age_at_event',
    'receiver_age_at_event',
    'donor_future_life',
    'receiver_future_life',
    'donor_pre_match_iou',
    'receiver_pre_match_iou',
    'cluster_size',
    'boundary_position_ratio',
    'pair_swap_margin',
    'future_overlap_total_count',
    'future_overlap_max_ioa',
    'copresent_frac_h120',
    'swap_margin_future_mean',
]
COMPACT_FEATURES = list(dict.fromkeys(COMPACT_BASE_FEATURES + DIRECTIONAL_FEATURES))
LOCKED_RETRIEVAL_OUTPUT_COLUMNS = list(dict.fromkeys(
    DIRECTION_KEYS
    + COMPACT_FEATURES
    + MECHANISM_FEATURES
    + [
        'retrieval_mean',
        'retrieval_std',
        'retrieval_mean_minus_std',
        'retrieval_median',
        'retrieval_q25',
        'retrieval_positive_fraction',
        'retrieval_p_mean',
        'retrieval_min_distance',
        'retrieval_mean_distance',
        'retrieval_neighbors_used',
        'retrieval_spec',
        'retrieval_score',
        'certificate_pass',
        'rank_start',
        'rank_end',
    ]
))


@dataclass(frozen=True)
class RetrievalSpec:
    name: str
    feature_set: str
    scaler: str
    neighbors: int
    score: str
    certificate: str


SPECS = [
    RetrievalSpec('compact_standard_k1_nearest_positive', 'compact', 'standard', 1, 'mean', 'q25_positive'),
    RetrievalSpec('compact_standard_k3_vote67', 'compact', 'standard', 3, 'positive_fraction', 'p67'),
    RetrievalSpec('compact_standard_k3_q25', 'compact', 'standard', 3, 'q25', 'q25_positive'),
    RetrievalSpec('compact_robust_k3_q25', 'compact', 'robust', 3, 'q25', 'q25_positive'),
    RetrievalSpec('compact_robust_k7_vote', 'compact', 'robust', 7, 'p_mean', 'score_positive'),
    RetrievalSpec('mechanism_robust_k5_pmean', 'mechanism', 'robust', 5, 'p_mean', 'score_positive'),
    RetrievalSpec('mechanism_robust_k5_vote', 'mechanism', 'robust', 5, 'positive_fraction', 'score_positive'),
    RetrievalSpec('mechanism_robust_k7_pmean', 'mechanism', 'robust', 7, 'p_mean', 'score_positive'),
    RetrievalSpec('mechanism_standard_k5_mean', 'mechanism', 'standard', 5, 'mean', 'mean_positive'),
    RetrievalSpec('mechanism_standard_k7_pmean', 'mechanism', 'standard', 7, 'p_mean', 'score_positive'),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def rounded(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    numeric = result.select_dtypes(include=[np.number]).columns
    result[numeric] = result[numeric].round(12)
    return result


def clean_locked_prediction(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop all non-whitelisted diagnostics from locked prediction artifacts."""
    return frame[
        [column for column in LOCKED_RETRIEVAL_OUTPUT_COLUMNS if column in frame.columns]
    ].copy()


def feature_names(spec: RetrievalSpec) -> list[str]:
    if spec.feature_set == 'compact':
        return COMPACT_FEATURES
    if spec.feature_set == 'mechanism':
        return MECHANISM_FEATURES
    raise ValueError(spec.feature_set)


def fit_transformer(train: pd.DataFrame, spec: RetrievalSpec):
    features = feature_names(spec)
    imputer = SimpleImputer(strategy='median')
    x_train = imputer.fit_transform(train[features])
    scaler = RobustScaler() if spec.scaler == 'robust' else StandardScaler()
    x_train = scaler.fit_transform(x_train)
    return features, imputer, scaler, x_train


def candidate_statistics(
    train: pd.DataFrame,
    test: pd.DataFrame,
    spec: RetrievalSpec,
) -> pd.DataFrame:
    train_exec = train[train.executor_accepted == 1].reset_index(drop=True)
    test_exec = test[test.executor_accepted == 1].reset_index(drop=True)
    if len(train_exec) < max(spec.neighbors, 3):
        return pd.DataFrame()
    features, imputer, scaler, x_train = fit_transformer(train_exec, spec)
    x_test = scaler.transform(imputer.transform(test_exec[features]))
    y = train_exec.delta_HOTA.to_numpy(float)
    rows = []
    for index in range(len(test_exec)):
        distances = np.sqrt(np.mean((x_train - x_test[index]) ** 2, axis=1))
        neighbor_indices = np.argsort(distances)[: min(spec.neighbors, len(distances))]
        neighbor_distances = distances[neighbor_indices]
        neighbor_values = y[neighbor_indices]
        weights = 1.0 / (neighbor_distances + 0.10)
        mean = float(np.sum(weights * neighbor_values) / np.sum(weights))
        std = float(np.sqrt(np.sum(weights * (neighbor_values - mean) ** 2) / np.sum(weights)))
        q25 = float(np.quantile(neighbor_values, 0.25))
        median = float(np.median(neighbor_values))
        positive_fraction = float(np.mean(neighbor_values > 0))
        p_mean = float(positive_fraction + 0.10 * np.tanh(mean))
        row = test_exec.iloc[index].to_dict()
        row.update({
            'retrieval_mean': mean,
            'retrieval_std': std,
            'retrieval_mean_minus_std': mean - std,
            'retrieval_median': median,
            'retrieval_q25': q25,
            'retrieval_positive_fraction': positive_fraction,
            'retrieval_p_mean': p_mean,
            'retrieval_min_distance': float(neighbor_distances.min()),
            'retrieval_mean_distance': float(neighbor_distances.mean()),
            'retrieval_neighbors_used': int(len(neighbor_indices)),
            'retrieval_spec': spec.name,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def retrieval_score(frame: pd.DataFrame, spec: RetrievalSpec) -> pd.Series:
    mapping = {
        'mean': 'retrieval_mean',
        'mean_std': 'retrieval_mean_minus_std',
        'q25': 'retrieval_q25',
        'positive_fraction': 'retrieval_positive_fraction',
        'p_mean': 'retrieval_p_mean',
    }
    return frame[mapping[spec.score]]


def certificate_mask(frame: pd.DataFrame, spec: RetrievalSpec) -> pd.Series:
    if spec.certificate == 'q25_positive':
        return frame.retrieval_q25 > 0
    if spec.certificate == 'p67':
        return frame.retrieval_positive_fraction >= (2.0 / 3.0)
    if spec.certificate == 'mean_positive':
        return frame.retrieval_mean > 0
    if spec.certificate == 'score_positive':
        return retrieval_score(frame, spec) > 0
    raise ValueError(spec.certificate)


def choose_window_candidate(
    train: pd.DataFrame,
    test: pd.DataFrame,
    spec: RetrievalSpec,
    windows: list[tuple[int, int]],
) -> pd.DataFrame:
    stats = candidate_statistics(train, test, spec)
    if not len(stats):
        return pd.DataFrame()
    stats['retrieval_score'] = retrieval_score(stats, spec)
    stats['certificate_pass'] = certificate_mask(stats, spec).astype(int)
    rows = []
    for seq, seq_frame in stats.groupby('seq'):
        for rank_start, rank_end in windows:
            window = seq_frame[
                (seq_frame.canonical_rank >= rank_start)
                & (seq_frame.canonical_rank <= rank_end)
                & (seq_frame.certificate_pass == 1)
            ].copy()
            if not len(window):
                continue
            best = window.sort_values(
                ['retrieval_score', 'retrieval_q25', 'retrieval_min_distance', 'canonical_rank'],
                ascending=[False, False, True, True],
            ).iloc[0].to_dict()
            best['rank_start'] = rank_start
            best['rank_end'] = rank_end
            rows.append(best)
    return pd.DataFrame(rows)


def window_subset(frame: pd.DataFrame, excluded_window: tuple[int, int] | None) -> pd.DataFrame:
    if excluded_window is None:
        return frame.copy()
    rank_start, rank_end = excluded_window
    return frame[
        ~((frame.canonical_rank >= rank_start) & (frame.canonical_rank <= rank_end))
    ].reset_index(drop=True)


def evaluate_rows(rows: pd.DataFrame, expected_windows: int, minimum_selected: int) -> dict:
    if not len(rows):
        return {
            'selected_windows': 0,
            'positive_windows': 0,
            'negative_windows': 0,
            'positive_precision': 0.0,
            'utility_sum': 0.0,
            'minimum_selected_utility': 0.0,
            'coverage': 0.0,
            'eligible': 0,
        }
    selected = len(rows)
    positive = int((rows.delta_HOTA > 0).sum())
    negative = int((rows.delta_HOTA <= 0).sum())
    precision = positive / selected
    utility_sum = float(rows.delta_HOTA.sum())
    minimum_utility = float(rows.delta_HOTA.min())
    eligible = bool(
        selected >= minimum_selected
        and positive == selected
        and utility_sum > 0
        and minimum_utility > 0
    )
    return {
        'selected_windows': selected,
        'positive_windows': positive,
        'negative_windows': negative,
        'positive_precision': precision,
        'utility_sum': utility_sum,
        'minimum_selected_utility': minimum_utility,
        'coverage': selected / expected_windows,
        'eligible': int(eligible),
    }


def inner_select_spec(
    sequence_frame: pd.DataFrame,
    available_windows: list[tuple[int, int]],
    minimum_selected: int,
) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    summaries = []
    predictions_by_spec: dict[str, pd.DataFrame] = {}
    for spec in SPECS:
        parts = []
        for held_out_window in available_windows:
            inner_train = window_subset(sequence_frame, held_out_window)
            a, b = held_out_window
            inner_test = sequence_frame[
                (sequence_frame.canonical_rank >= a)
                & (sequence_frame.canonical_rank <= b)
            ].reset_index(drop=True)
            prediction = choose_window_candidate(inner_train, inner_test, spec, [held_out_window])
            if len(prediction):
                prediction['inner_held_out_rank_start'] = a
                prediction['inner_held_out_rank_end'] = b
                parts.append(prediction)
        predictions = pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()
        predictions_by_spec[spec.name] = predictions
        summary = evaluate_rows(predictions, len(available_windows), minimum_selected)
        summary['retrieval_spec'] = spec.name
        summaries.append(summary)
    summary_frame = pd.DataFrame(summaries)
    eligible = summary_frame[summary_frame.eligible == 1].copy()
    if not len(eligible):
        return 'no_op', summary_frame, pd.DataFrame()
    chosen = eligible.sort_values(
        ['minimum_selected_utility', 'utility_sum', 'coverage'],
        ascending=[False, False, False],
    ).iloc[0]
    chosen_name = str(chosen.retrieval_spec)
    return chosen_name, summary_frame, predictions_by_spec[chosen_name]


def load_exclusion(path: str | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame(columns=DIRECTION_KEYS)
    frame = pd.read_csv(path)
    return frame[DIRECTION_KEYS].drop_duplicates().copy()


def exclude_rows(frame: pd.DataFrame, exclusion: pd.DataFrame) -> pd.DataFrame:
    if not len(exclusion):
        return frame.copy()
    merged = frame.merge(
        exclusion.assign(_excluded=1),
        on=DIRECTION_KEYS,
        how='left',
        validate='one_to_one',
    )
    matched = int(merged._excluded.fillna(0).sum())
    if matched != len(exclusion):
        raise RuntimeError(f'exclusion mismatch: expected={len(exclusion)}, matched={matched}')
    return merged[merged._excluded.isna()].drop(columns=['_excluded']).reset_index(drop=True)


def global_summary(rows: pd.DataFrame, sequences: list[str]) -> dict:
    sequence_utility = {seq: 0.0 for seq in sequences}
    if len(rows):
        for seq, group in rows.groupby('seq'):
            sequence_utility[seq] = float(group.delta_HOTA.sum())
    selected_sequences = [seq for seq, value in sequence_utility.items() if value != 0]
    selected = len(rows)
    positive = int((rows.delta_HOTA > 0).sum()) if len(rows) else 0
    negative = int((rows.delta_HOTA <= 0).sum()) if len(rows) else 0
    positive_sum = float(rows.loc[rows.delta_HOTA > 0, 'delta_HOTA'].sum()) if len(rows) else 0.0
    negative_mass = float(-rows.loc[rows.delta_HOTA < 0, 'delta_HOTA'].sum()) if len(rows) else 0.0
    utility_sum = float(rows.delta_HOTA.sum()) if len(rows) else 0.0
    worst_selected_sequence = min(
        [sequence_utility[seq] for seq in selected_sequences], default=0.0
    )
    eligible = bool(
        selected >= 6
        and len(selected_sequences) >= 2
        and positive / selected >= 0.90
        and utility_sum > 0
        and worst_selected_sequence > 0
        and negative_mass <= 0.10 * positive_sum
    )
    return {
        'selected_windows': selected,
        'selected_sequences': selected_sequences,
        'positive_windows': positive,
        'negative_windows': negative,
        'positive_precision': positive / selected if selected else 0.0,
        'positive_sum': positive_sum,
        'negative_mass': negative_mass,
        'utility_sum': utility_sum,
        'worst_selected_sequence_utility': worst_selected_sequence,
        'sequence_utility': sequence_utility,
        'eligible': int(eligible),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-features', action='append', required=True)
    ap.add_argument('--test-features', action='append', required=True)
    ap.add_argument('--train-executability', required=True)
    ap.add_argument('--test-executability', required=True)
    ap.add_argument('--train-utility', required=True)
    ap.add_argument('--locked-exclusion-keys')
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    train = attach_train_labels(
        load_directional_features(args.train_features, args.train_executability),
        args.train_utility,
    )
    locked_all = load_directional_features(args.test_features, args.test_executability)
    exclusion = load_exclusion(args.locked_exclusion_keys)
    locked = exclude_rows(locked_all, exclusion)
    sequences = sorted(train.seq.unique())

    all_features = sorted(set(COMPACT_FEATURES + MECHANISM_FEATURES))
    missing = [column for column in all_features if column not in train.columns or column not in locked.columns]
    if missing:
        raise RuntimeError(f'missing retrieval features: {missing}')

    outer_rows = []
    outer_predictions = []
    outer_inner_summaries = []
    outer_chosen_inner = []

    for seq in sequences:
        sequence_frame = train[train.seq == seq].reset_index(drop=True)
        for held_out_window in TRAIN_WINDOWS:
            available_windows = [window for window in TRAIN_WINDOWS if window != held_out_window]
            outer_train = window_subset(sequence_frame, held_out_window)
            a, b = held_out_window
            outer_test = sequence_frame[
                (sequence_frame.canonical_rank >= a)
                & (sequence_frame.canonical_rank <= b)
            ].reset_index(drop=True)
            chosen_name, summaries, chosen_inner = inner_select_spec(
                outer_train,
                available_windows,
                minimum_selected=2,
            )
            summaries['seq'] = seq
            summaries['outer_rank_start'] = a
            summaries['outer_rank_end'] = b
            outer_inner_summaries.append(summaries)
            if len(chosen_inner):
                chosen_inner['seq_context'] = seq
                chosen_inner['outer_rank_start'] = a
                chosen_inner['outer_rank_end'] = b
                outer_chosen_inner.append(chosen_inner)
            prediction = pd.DataFrame()
            if chosen_name != 'no_op':
                spec = next(item for item in SPECS if item.name == chosen_name)
                prediction = choose_window_candidate(outer_train, outer_test, spec, [held_out_window])
                if len(prediction):
                    prediction['outer_retrieval_spec'] = chosen_name
                    prediction['outer_rank_start'] = a
                    prediction['outer_rank_end'] = b
                    outer_predictions.append(prediction)
            outer_rows.append({
                'seq': seq,
                'rank_start': a,
                'rank_end': b,
                'chosen_retrieval_spec': chosen_name,
                'eligible_inner_specs': int((summaries.eligible == 1).sum()),
                'outer_selected': int(len(prediction)),
            })

    outer_oof = pd.concat(outer_predictions, ignore_index=True, sort=False) if outer_predictions else pd.DataFrame()
    deployment_summary = global_summary(outer_oof, sequences)
    deployment_allowed = bool(deployment_summary['eligible'])

    final_sequence_rows = []
    final_inner_summaries = []
    locked_predictions = []
    for seq in sequences:
        sequence_frame = train[train.seq == seq].reset_index(drop=True)
        chosen_name, summaries, _ = inner_select_spec(
            sequence_frame,
            TRAIN_WINDOWS,
            minimum_selected=3,
        )
        summaries['seq'] = seq
        final_inner_summaries.append(summaries)
        locked_prediction = pd.DataFrame()
        if chosen_name != 'no_op':
            spec = next(item for item in SPECS if item.name == chosen_name)
            locked_test = locked[locked.seq == seq].reset_index(drop=True)
            locked_prediction = choose_window_candidate(
                sequence_frame,
                locked_test,
                spec,
                LOCKED_WINDOWS,
            )
            if len(locked_prediction):
                locked_prediction = clean_locked_prediction(locked_prediction)
                locked_prediction['final_retrieval_spec'] = chosen_name
                locked_prediction['selected_by_global_gate'] = int(deployment_allowed)
                locked_predictions.append(locked_prediction)
        final_sequence_rows.append({
            'seq': seq,
            'chosen_retrieval_spec': chosen_name,
            'eligible_train_specs': int((summaries.eligible == 1).sum()),
            'locked_candidate_count': int(len(locked_prediction)),
        })

    locked_prediction_frame = pd.concat(locked_predictions, ignore_index=True, sort=False) if locked_predictions else pd.DataFrame()
    if len(locked_prediction_frame):
        selected_locked = locked_prediction_frame[
            locked_prediction_frame.selected_by_global_gate == 1
        ].copy()
    else:
        selected_locked = pd.DataFrame()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    outputs = {
        'outer_fold_choices.csv': pd.DataFrame(outer_rows),
        'outer_inner_spec_summary.csv': pd.concat(outer_inner_summaries, ignore_index=True, sort=False),
        'outer_chosen_inner_predictions.csv': pd.concat(outer_chosen_inner, ignore_index=True, sort=False) if outer_chosen_inner else pd.DataFrame(),
        'outer_window_oof.csv': outer_oof,
        'final_sequence_choices.csv': pd.DataFrame(final_sequence_rows),
        'final_train_spec_summary.csv': pd.concat(final_inner_summaries, ignore_index=True, sort=False),
        'locked_candidate_predictions.csv': locked_prediction_frame,
        'selected_transactions.csv': selected_locked,
    }
    for filename, frame in outputs.items():
        rounded(frame).to_csv(out / filename, index=False)

    report = {
        'protocol': {
            'scope': 'Sequence-conditioned nearest-neighbor utility retrieval using train ranks21-100 only; previously revealed locked rows are excluded before prediction.',
            'locked_utility_or_trackeval_labels_read': False,
            'retrieval_specs_preregistered': [spec.__dict__ for spec in SPECS],
            'outer_evaluation': 'For each sequence and each 20-rank train window, the other three windows form outer training. A nested leave-one-window-out over those three selects the retrieval spec.',
            'inner_eligibility': 'At least two of three windows selected; every selected utility strictly positive.',
            'final_sequence_eligibility': 'At least three of four train windows selected; every selected utility strictly positive.',
            'global_eligibility': 'At least six outer selections across at least two sequences, precision>=0.90, positive total and every selected-sequence utility, negative mass<=10% positive sum.',
            'selective_deployment': 'Sequences without local positive evidence remain no-op; coverage of all four sequences is not required.',
        },
        'dataset': {
            'training_directional_rows': int(len(train)),
            'training_executable_rows': int((train.executor_accepted == 1).sum()),
            'locked_rows_before_exclusion': int(len(locked_all)),
            'previously_revealed_locked_rows_excluded': int(len(exclusion)),
            'locked_rows_after_exclusion': int(len(locked)),
            'locked_executable_rows_after_exclusion': int((locked.executor_accepted == 1).sum()),
            'sequences': sequences,
        },
        'outer_oof_summary': deployment_summary,
        'deployment_allowed': deployment_allowed,
        'final_sequence_choices': final_sequence_rows,
        'locked_candidate_count': int(len(locked_prediction_frame)),
        'locked_selected_after_global_gate': int(len(selected_locked)),
        'remaining_locked_labels_unread_before_new_reveal': int(len(locked)),
    }
    (out / 'report.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    manifest = {
        path.name: sha256(path)
        for path in sorted(out.iterdir())
        if path.is_file() and path.name != 'prediction_manifest.json'
    }
    (out / 'prediction_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
