from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer

from train_locked_loso_directional_handoff_utility import (
    DIRECTIONAL_FEATURES,
    DIRECTION_KEYS,
    attach_train_labels,
    load_directional_features,
    normalized_target,
)
from train_loso_future_transaction_utility_ranker import BASE_FEATURES


TRAIN_WINDOWS = [(21, 40), (41, 60), (61, 80), (81, 100)]
LOCKED_WINDOWS = [(1, 20)]
PAIRWISE_FEATURES = list(dict.fromkeys(BASE_FEATURES + DIRECTIONAL_FEATURES))
LOCKED_OUTPUT_COLUMNS = list(dict.fromkeys(
    DIRECTION_KEYS
    + PAIRWISE_FEATURES
    + [
        'pairwise_win_score',
        'pairwise_margin',
        'rank_start',
        'rank_end',
        'window_executable_candidates',
    ]
))


@dataclass(frozen=True)
class RankerSpec:
    name: str
    family: str
    target_scale: str
    weight_scheme: str


RANKERS = [
    RankerSpec('extra_raw_capped', 'extra', 'raw', 'capped'),
    RankerSpec('extra_raw_tail', 'extra', 'raw', 'tail'),
    RankerSpec('extra_norm_capped', 'extra', 'normalized', 'capped'),
    RankerSpec('extra_norm_tail', 'extra', 'normalized', 'tail'),
    RankerSpec('hist_raw_capped', 'hist', 'raw', 'capped'),
    RankerSpec('hist_raw_tail', 'hist', 'raw', 'tail'),
    RankerSpec('hist_norm_capped', 'hist', 'normalized', 'capped'),
    RankerSpec('hist_norm_tail', 'hist', 'normalized', 'tail'),
]

GATES = [
    'all',
    'changed_ge_2',
    'changed_ge_4',
    'changed_ge_8',
    'changed_ge_12',
    'impact_receiver_le_080',
    'changed_ge_4_impact_receiver_le_080',
    'changed_ge_4_impact_pair_le_050',
    'changed_ge_4_margin_ge_002',
    'changed_ge_4_score_ge_070',
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def round_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    numeric = result.select_dtypes(include=[np.number]).columns
    result[numeric] = result[numeric].round(12)
    return result


def clean_locked_prediction(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep only pre-evaluation keys, whitelisted model features, and predictions."""
    return frame[[column for column in LOCKED_OUTPUT_COLUMNS if column in frame.columns]].copy()


def attach_normalized_target(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.reset_index(drop=True).copy()
    result['delta_HOTA_norm'] = normalized_target(result)
    return result


def load_exclusion_keys(path: str | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame(columns=DIRECTION_KEYS)
    frame = pd.read_csv(path)
    missing = [column for column in DIRECTION_KEYS if column not in frame.columns]
    if missing:
        raise RuntimeError(f'locked exclusion file missing columns: {missing}')
    result = frame[DIRECTION_KEYS].drop_duplicates().copy()
    if len(result) != len(frame):
        raise RuntimeError('locked exclusion file contains duplicate directional keys')
    return result


def exclude_locked_rows(locked: pd.DataFrame, exclusion: pd.DataFrame) -> pd.DataFrame:
    if not len(exclusion):
        return locked.copy()
    merged = locked.merge(
        exclusion.assign(_excluded=1),
        on=DIRECTION_KEYS,
        how='left',
        validate='one_to_one',
    )
    matched = int(merged._excluded.fillna(0).sum())
    if matched != len(exclusion):
        raise RuntimeError(f'locked exclusion key mismatch: expected={len(exclusion)}, matched={matched}')
    return merged[merged._excluded.isna()].drop(columns=['_excluded']).reset_index(drop=True)


def pair_weight(ui: float, uj: float, scheme: str) -> float:
    gap = abs(ui - uj)
    base = min(gap, 0.20) + 0.005
    if scheme == 'capped':
        return base
    if scheme == 'tail':
        risk = 8.0 if min(ui, uj) <= -0.10 else (4.0 if min(ui, uj) < 0 else 1.0)
        return base * risk
    raise ValueError(scheme)


def build_pairs(
    frame: pd.DataFrame,
    imputer: SimpleImputer,
    spec: RankerSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    transformed = imputer.transform(frame[PAIRWISE_FEATURES])
    work = frame.reset_index(drop=True).copy()
    target_column = 'delta_HOTA' if spec.target_scale == 'raw' else 'delta_HOTA_norm'
    x_parts: list[np.ndarray] = []
    y_parts: list[int] = []
    w_parts: list[float] = []

    for _, seq_frame in work.groupby('seq'):
        for rank_start, rank_end in TRAIN_WINDOWS:
            indices = seq_frame.index[
                (seq_frame.canonical_rank >= rank_start)
                & (seq_frame.canonical_rank <= rank_end)
            ].to_numpy()
            if len(indices) < 2:
                continue
            utilities = work.loc[indices, target_column].to_numpy(float)
            window_x: list[np.ndarray] = []
            window_y: list[int] = []
            window_w: list[float] = []
            for left in range(len(indices)):
                for right in range(left + 1, len(indices)):
                    ui, uj = float(utilities[left]), float(utilities[right])
                    if abs(ui - uj) < 1e-12:
                        continue
                    difference = transformed[indices[left]] - transformed[indices[right]]
                    label = int(ui > uj)
                    weight = pair_weight(ui, uj, spec.weight_scheme)
                    window_x.extend([difference, -difference])
                    window_y.extend([label, 1 - label])
                    window_w.extend([weight, weight])
            if not window_x:
                continue
            weights = np.asarray(window_w, dtype=float)
            weights *= len(weights) / max(weights.sum(), 1e-12)
            x_parts.extend(window_x)
            y_parts.extend(window_y)
            w_parts.extend(weights.tolist())

    if not x_parts:
        raise RuntimeError('no pairwise training examples generated')
    return np.asarray(x_parts), np.asarray(y_parts), np.asarray(w_parts)


def fit_ranker(frame: pd.DataFrame, spec: RankerSpec, seed: int):
    train = frame[frame.executor_accepted == 1].reset_index(drop=True)
    if len(train) < 20:
        raise RuntimeError(f'too few executable pairwise training rows: {len(train)}')
    imputer = SimpleImputer(strategy='median')
    imputer.fit(train[PAIRWISE_FEATURES])
    x_train, y_train, weights = build_pairs(train, imputer, spec)
    if spec.family == 'extra':
        model = ExtraTreesClassifier(
            n_estimators=600,
            max_depth=7,
            min_samples_leaf=3,
            max_features=0.70,
            class_weight='balanced',
            random_state=seed,
            n_jobs=-1,
        )
    elif spec.family == 'hist':
        model = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.035,
            max_leaf_nodes=15,
            min_samples_leaf=10,
            l2_regularization=6.0,
            random_state=seed,
        )
    else:
        raise ValueError(spec.family)
    model.fit(x_train, y_train, sample_weight=weights)
    return imputer, model, len(x_train)


def score_windows(
    frame: pd.DataFrame,
    imputer: SimpleImputer,
    model,
    windows: list[tuple[int, int]],
    include_labels: bool,
) -> pd.DataFrame:
    executable = frame[frame.executor_accepted == 1].copy()
    rows = []
    for seq, seq_frame in executable.groupby('seq'):
        for rank_start, rank_end in windows:
            window = seq_frame[
                (seq_frame.canonical_rank >= rank_start)
                & (seq_frame.canonical_rank <= rank_end)
            ].copy()
            if not len(window):
                continue
            transformed = imputer.transform(window[PAIRWISE_FEATURES])
            scores = []
            for index in range(len(window)):
                probabilities = model.predict_proba(transformed[index][None, :] - transformed)[:, 1]
                scores.append(float((probabilities.sum() - 0.5) / max(len(window) - 1, 1)))
            window['pairwise_win_score'] = scores
            window = window.sort_values(
                ['pairwise_win_score', 'canonical_rank', 'transaction_type'],
                ascending=[False, True, True],
            )
            best = window.iloc[0].to_dict()
            second_score = float(window.iloc[1].pairwise_win_score) if len(window) > 1 else 0.5
            best['pairwise_margin'] = float(best['pairwise_win_score'] - second_score)
            best['rank_start'] = rank_start
            best['rank_end'] = rank_end
            best['window_executable_candidates'] = int(len(window))
            if not include_labels:
                best.pop('delta_HOTA', None)
                best.pop('delta_HOTA_norm', None)
            rows.append(best)
    return pd.DataFrame(rows)


def gate_mask(frame: pd.DataFrame, gate: str) -> pd.Series:
    if gate == 'all':
        return pd.Series(True, index=frame.index)
    if gate == 'changed_ge_2':
        return frame.executor_changed_rows >= 2
    if gate == 'changed_ge_4':
        return frame.executor_changed_rows >= 4
    if gate == 'changed_ge_8':
        return frame.executor_changed_rows >= 8
    if gate == 'changed_ge_12':
        return frame.executor_changed_rows >= 12
    if gate == 'impact_receiver_le_080':
        return frame.executor_impact_receiver_future <= 0.80
    if gate == 'changed_ge_4_impact_receiver_le_080':
        return (frame.executor_changed_rows >= 4) & (frame.executor_impact_receiver_future <= 0.80)
    if gate == 'changed_ge_4_impact_pair_le_050':
        return (frame.executor_changed_rows >= 4) & (frame.executor_impact_pair_future <= 0.50)
    if gate == 'changed_ge_4_margin_ge_002':
        return (frame.executor_changed_rows >= 4) & (frame.pairwise_margin >= 0.02)
    if gate == 'changed_ge_4_score_ge_070':
        return (frame.executor_changed_rows >= 4) & (frame.pairwise_win_score >= 0.70)
    raise ValueError(gate)


def summarize_selection(
    frame: pd.DataFrame,
    selected_column: str,
    expected_sequences: list[str],
    minimum_selected: int,
    minimum_precision: float,
) -> dict:
    selected = frame[frame[selected_column] == 1].copy()
    sequence_utility = {seq: 0.0 for seq in expected_sequences}
    if len(selected):
        for seq, group in selected.groupby('seq'):
            sequence_utility[seq] = float(group.delta_HOTA.sum())
    covered_sequences = int(selected.seq.nunique()) if len(selected) else 0
    positive_sum = float(selected.loc[selected.delta_HOTA > 0, 'delta_HOTA'].sum()) if len(selected) else 0.0
    negative_mass = float(-selected.loc[selected.delta_HOTA < 0, 'delta_HOTA'].sum()) if len(selected) else 0.0
    utility_sum = float(selected.delta_HOTA.sum()) if len(selected) else 0.0
    precision = float((selected.delta_HOTA > 0).mean()) if len(selected) else 0.0
    worst_sequence = min(sequence_utility.values()) if sequence_utility else 0.0
    eligible = bool(
        len(selected) >= minimum_selected
        and covered_sequences == len(expected_sequences)
        and precision >= minimum_precision
        and positive_sum > 0
        and utility_sum > 0
        and worst_sequence >= 0
        and negative_mass <= 0.25 * positive_sum
    )
    return {
        'selected_windows': int(len(selected)),
        'covered_sequences': covered_sequences,
        'positive_windows': int((selected.delta_HOTA > 0).sum()) if len(selected) else 0,
        'negative_windows': int((selected.delta_HOTA < 0).sum()) if len(selected) else 0,
        'zero_windows': int((selected.delta_HOTA == 0).sum()) if len(selected) else 0,
        'positive_precision': precision,
        'positive_sum': positive_sum,
        'negative_mass': negative_mass,
        'utility_sum': utility_sum,
        'worst_sequence_utility': worst_sequence,
        'sequence_utility': json.dumps(sequence_utility, sort_keys=True),
        'robust_score': positive_sum - 6.0 * negative_mass + 3.0 * min(0.0, worst_sequence),
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

    train = attach_normalized_target(
        attach_train_labels(
            load_directional_features(args.train_features, args.train_executability),
            args.train_utility,
        )
    )
    locked_all = load_directional_features(args.test_features, args.test_executability)
    exclusion = load_exclusion_keys(args.locked_exclusion_keys)
    locked = exclude_locked_rows(locked_all, exclusion)
    sequences = sorted(train.seq.unique())
    if set(sequences) != set(locked.seq.unique()):
        raise RuntimeError('train/locked sequence sets differ after exclusion')
    missing = [feature for feature in PAIRWISE_FEATURES if feature not in train.columns or feature not in locked.columns]
    if missing:
        raise RuntimeError(f'missing pairwise features: {missing}')

    outer_fold_rows = []
    inner_summary_rows = []
    inner_oof_parts = []
    outer_oof_parts = []
    locked_parts = []

    for outer_index, held_out in enumerate(sequences):
        outer_train = train[train.seq != held_out].reset_index(drop=True)
        outer_test = train[train.seq == held_out].reset_index(drop=True)
        locked_test = locked[locked.seq == held_out].reset_index(drop=True)
        train_sequences = sorted(outer_train.seq.unique())
        candidates = []
        config_inner_frames: dict[str, pd.DataFrame] = {}

        for config_index, spec in enumerate(RANKERS):
            parts = []
            for inner_index, inner_held_out in enumerate(train_sequences):
                inner_train = outer_train[outer_train.seq != inner_held_out].reset_index(drop=True)
                inner_test = outer_train[outer_train.seq == inner_held_out].reset_index(drop=True)
                imputer, model, pair_count = fit_ranker(
                    inner_train,
                    spec,
                    seed=61000 + 1000 * outer_index + 100 * config_index + inner_index,
                )
                prediction = score_windows(inner_test, imputer, model, TRAIN_WINDOWS, include_labels=True)
                prediction['ranker_name'] = spec.name
                prediction['pairwise_train_examples'] = pair_count
                prediction['outer_held_out_seq'] = held_out
                prediction['inner_held_out_seq'] = inner_held_out
                parts.append(prediction)
            inner_oof = pd.concat(parts, ignore_index=True, sort=False)
            config_inner_frames[spec.name] = inner_oof
            for gate in GATES:
                evaluated = inner_oof.copy()
                evaluated['gate_name'] = gate
                evaluated['selected_by_gate'] = gate_mask(evaluated, gate).astype(int)
                summary = summarize_selection(
                    evaluated,
                    'selected_by_gate',
                    train_sequences,
                    minimum_selected=6,
                    minimum_precision=0.65,
                )
                summary.update({
                    'outer_held_out_seq': held_out,
                    'ranker_name': spec.name,
                    'family': spec.family,
                    'target_scale': spec.target_scale,
                    'weight_scheme': spec.weight_scheme,
                    'gate_name': gate,
                })
                candidates.append(summary)
                inner_summary_rows.append(summary)

        candidate_frame = pd.DataFrame(candidates)
        eligible = candidate_frame[candidate_frame.eligible == 1].copy()
        if len(eligible):
            chosen = eligible.sort_values(
                ['robust_score', 'utility_sum', 'positive_precision', 'selected_windows'],
                ascending=[False, False, False, True],
            ).iloc[0].to_dict()
            chosen_ranker = str(chosen['ranker_name'])
            chosen_gate = str(chosen['gate_name'])
            fallback_no_op = 0
            chosen_inner = config_inner_frames[chosen_ranker].copy()
            chosen_inner['gate_name'] = chosen_gate
            chosen_inner['selected_by_gate'] = gate_mask(chosen_inner, chosen_gate).astype(int)
            inner_oof_parts.append(chosen_inner)
        else:
            chosen_ranker = 'no_op'
            chosen_gate = 'no_op'
            fallback_no_op = 1

        outer_selected = pd.DataFrame()
        locked_selected = pd.DataFrame()
        outer_topone = pd.DataFrame()
        locked_topone = pd.DataFrame()
        pair_count = 0
        if not fallback_no_op:
            spec = next(item for item in RANKERS if item.name == chosen_ranker)
            imputer, model, pair_count = fit_ranker(
                outer_train,
                spec,
                seed=71000 + outer_index,
            )
            outer_topone = score_windows(outer_test, imputer, model, TRAIN_WINDOWS, include_labels=True)
            outer_topone['ranker_name'] = chosen_ranker
            outer_topone['gate_name'] = chosen_gate
            outer_topone['selected_by_fold_gate'] = gate_mask(outer_topone, chosen_gate).astype(int)
            outer_topone['outer_held_out_seq'] = held_out
            outer_oof_parts.append(outer_topone)
            outer_selected = outer_topone[outer_topone.selected_by_fold_gate == 1]

            locked_topone = score_windows(locked_test, imputer, model, LOCKED_WINDOWS, include_labels=False)
            locked_topone = clean_locked_prediction(locked_topone)
            locked_topone['ranker_name'] = chosen_ranker
            locked_topone['gate_name'] = chosen_gate
            locked_topone['selected_by_fold_gate'] = gate_mask(locked_topone, chosen_gate).astype(int)
            locked_topone['outer_held_out_seq'] = held_out
            locked_parts.append(locked_topone)
            locked_selected = locked_topone[locked_topone.selected_by_fold_gate == 1]

        outer_fold_rows.append({
            'held_out_seq': held_out,
            'chosen_ranker': chosen_ranker,
            'chosen_gate': chosen_gate,
            'eligible_config_gates': int(len(eligible)),
            'fallback_no_op': fallback_no_op,
            'pairwise_train_examples': pair_count,
            'outer_topone_windows': int(len(outer_topone)),
            'outer_selected_windows': int(len(outer_selected)),
            'locked_topone_candidates': int(len(locked_topone)),
            'locked_selected_by_fold_gate': int(len(locked_selected)),
        })

    outer_oof = pd.concat(outer_oof_parts, ignore_index=True, sort=False) if outer_oof_parts else pd.DataFrame()
    locked_predictions = pd.concat(locked_parts, ignore_index=True, sort=False) if locked_parts else pd.DataFrame()
    global_summary = summarize_selection(
        outer_oof,
        'selected_by_fold_gate',
        sequences,
        minimum_selected=8,
        minimum_precision=0.65,
    ) if len(outer_oof) else {
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
        'sequence_utility': json.dumps({seq: 0.0 for seq in sequences}, sort_keys=True),
        'robust_score': 0.0,
        'eligible': 0,
    }
    deployment_allowed = bool(global_summary['eligible'])
    if len(locked_predictions):
        locked_predictions['selected_by_global_gate'] = (
            locked_predictions.selected_by_fold_gate.astype(int) if deployment_allowed else 0
        )
        selected_locked = locked_predictions[locked_predictions.selected_by_global_gate == 1].copy()
    else:
        selected_locked = pd.DataFrame()

    oracle_rows = []
    executable_train = train[train.executor_accepted == 1]
    for seq, seq_frame in executable_train.groupby('seq'):
        for rank_start, rank_end in TRAIN_WINDOWS:
            window = seq_frame[
                (seq_frame.canonical_rank >= rank_start)
                & (seq_frame.canonical_rank <= rank_end)
            ]
            if not len(window):
                continue
            best = window.loc[window.delta_HOTA.idxmax()]
            oracle_rows.append({
                'seq': seq,
                'rank_start': rank_start,
                'rank_end': rank_end,
                'window_executable_candidates': int(len(window)),
                'window_positive_candidates': int((window.delta_HOTA > 0).sum()),
                'oracle_rank': int(best.canonical_rank),
                'oracle_transaction_type': best.transaction_type,
                'oracle_delta_HOTA': float(best.delta_HOTA),
            })

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    outputs = {
        'outer_fold_choices.csv': pd.DataFrame(outer_fold_rows),
        'inner_config_gate_summary.csv': pd.DataFrame(inner_summary_rows),
        'chosen_inner_oof.csv': pd.concat(inner_oof_parts, ignore_index=True, sort=False) if inner_oof_parts else pd.DataFrame(),
        'outer_window_oof.csv': outer_oof,
        'locked_candidate_predictions.csv': locked_predictions,
        'selected_transactions.csv': selected_locked,
        'train_window_oracle_diagnostic.csv': pd.DataFrame(oracle_rows),
    }
    for filename, frame in outputs.items():
        round_numeric(frame).to_csv(out / filename, index=False)

    report = {
        'protocol': {
            'scope': 'Nested LOSO pairwise/listwise directional window ranking on train ranks21-100, followed by frozen prediction on previously unread locked rows only.',
            'locked_utility_or_trackeval_labels_read': False,
            'previously_revealed_locked_candidates_excluded': int(len(exclusion)),
            'remaining_locked_directional_rows': int(len(locked)),
            'ranker_family_preregistered': [spec.__dict__ for spec in RANKERS],
            'gate_family_preregistered': GATES,
            'inner_selection': 'Within each outer fold, three-way inner LOSO selects ranker and gate. Each 20-rank window contributes one pairwise top-one candidate before abstention.',
            'inner_eligibility': 'selected>=6, all three inner sequences covered, precision>=0.65, positive and total utility>0, worst sequence>=0, negative mass<=25% positive sum.',
            'global_eligibility': 'outer OOF selected>=8, all four sequences covered, precision>=0.65, positive and total utility>0, worst sequence>=0, negative mass<=25% positive sum.',
            'pair_weighting': 'Pair weights are normalized within each sequence-window. Tail variants upweight comparisons containing negative and catastrophic utility.',
        },
        'dataset': {
            'training_directional_rows': int(len(train)),
            'training_executable_rows': int((train.executor_accepted == 1).sum()),
            'locked_directional_rows_before_exclusion': int(len(locked_all)),
            'locked_excluded_previously_revealed': int(len(exclusion)),
            'locked_directional_rows_after_exclusion': int(len(locked)),
            'locked_executable_rows_after_exclusion': int((locked.executor_accepted == 1).sum()),
            'sequences': sequences,
            'train_windows': 16,
        },
        'outer_folds': outer_fold_rows,
        'outer_oof_summary': global_summary,
        'deployment_allowed': deployment_allowed,
        'locked_fold_gate_passes': int(locked_predictions.selected_by_fold_gate.sum()) if len(locked_predictions) else 0,
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
