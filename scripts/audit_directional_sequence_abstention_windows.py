from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_loso_future_transaction_utility_ranker import BASE_FEATURES
from train_locked_loso_directional_handoff_utility import (
    COMPACT_BASE_FEATURES,
    DIRECTIONAL_FEATURES,
    DIRECTION_KEYS,
    conflict_free,
    evaluate_inner_predictions,
    fit_predict,
    load_directional_features,
    attach_train_labels,
    policy_mask,
    policy_score,
)


WINDOWS = [(21, 40), (41, 60), (61, 80), (81, 100)]

# Fixed before reading this audit's OOF outcomes. These are intentionally few,
# interpretable, and candidate/window-local; no locked rank1-20 labels are used.
ABSTENTION_GATES = [
    'base',
    'regression_any_positive',
    'median_positive',
    'expected_positive',
    'regression_consensus',
    'q25_positive',
    'classifier_strict',
    'hybrid_safe',
    'unique_signal',
    'sparse_signal',
    'score_gap_010',
    'sparse_regression_any',
]


def build_variants() -> dict[str, list[str]]:
    return {
        'compact_directional': list(dict.fromkeys(COMPACT_BASE_FEATURES + DIRECTIONAL_FEATURES)),
        'full_directional': list(dict.fromkeys(BASE_FEATURES + DIRECTIONAL_FEATURES)),
    }


def choose_inner_policy(
    outer_train: pd.DataFrame,
    variants: dict[str, list[str]],
    outer_index: int,
) -> tuple[dict, pd.DataFrame]:
    rows = []
    for variant_index, (variant, feature_names) in enumerate(variants.items()):
        inner_parts = []
        for inner_index, inner_held_out in enumerate(sorted(outer_train.seq.unique())):
            inner_train = outer_train[outer_train.seq != inner_held_out].copy().reset_index(drop=True)
            inner_test = outer_train[outer_train.seq == inner_held_out].copy().reset_index(drop=True)
            prediction = fit_predict(
                inner_train,
                inner_test,
                feature_names,
                # Match the exact nested-LOSO seed schedule used by the formal
                # locked directional utility v2 training script.
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
        results = evaluate_inner_predictions(inner_predictions, variant)
        rows.extend(results)

    frame = pd.DataFrame(rows)
    eligible = frame[frame.eligible == 1].copy()
    if not len(eligible):
        return {
            'variant': 'compact_directional',
            'policy': 'no_op',
            'cap': 0,
            'fallback_no_op': 1,
            'inner_eligible_policies': 0,
        }, frame
    chosen = eligible.sort_values(
        ['robust_score', 'raw_utility_sum', 'positive_precision', 'selected'],
        ascending=[False, False, False, True],
    ).iloc[0].to_dict()
    return {
        'variant': str(chosen['variant']),
        'policy': str(chosen['policy']),
        'cap': int(chosen['cap']),
        'fallback_no_op': 0,
        'inner_eligible_policies': int(len(eligible)),
    }, frame


def window_descriptor(prediction: pd.DataFrame, policy: str) -> dict:
    if policy == 'no_op':
        return {
            'active_candidates': 0,
            'top_score': np.nan,
            'second_score': np.nan,
            'score_gap': np.nan,
            'executable_rate': float(prediction.executor_accepted.mean()) if len(prediction) else 0.0,
        }
    active = prediction[policy_mask(prediction, policy)].copy()
    if len(active):
        active['selection_score'] = policy_score(active, policy)
        scores = active.selection_score.sort_values(ascending=False).to_numpy(float)
        top = float(scores[0])
        second = float(scores[1]) if len(scores) > 1 else np.nan
        gap = float(top - second) if len(scores) > 1 else np.inf
    else:
        top = second = gap = np.nan
    return {
        'active_candidates': int(len(active)),
        'top_score': top,
        'second_score': second,
        'score_gap': gap,
        'executable_rate': float(prediction.executor_accepted.mean()) if len(prediction) else 0.0,
    }


def gate_passes(gate: str, selected: pd.DataFrame, descriptor: dict) -> bool:
    if not len(selected):
        return False
    row = selected.iloc[0]
    if gate == 'base':
        return True
    if gate == 'regression_any_positive':
        return bool(row.pred_expected > 0 or row.pred_median > 0)
    if gate == 'median_positive':
        return bool(row.pred_median > 0)
    if gate == 'expected_positive':
        return bool(row.pred_expected > 0)
    if gate == 'regression_consensus':
        return bool(row.pred_expected > 0 and row.pred_median > 0)
    if gate == 'q25_positive':
        return bool(row.pred_q25 > 0)
    if gate == 'classifier_strict':
        return bool(
            row.prob_positive >= 0.70
            and row.prob_negative <= 0.25
            and row.prob_catastrophic <= 0.05
        )
    if gate == 'hybrid_safe':
        return bool(
            row.pred_median > 0
            and row.prob_positive >= 0.65
            and row.prob_negative <= 0.30
            and row.prob_catastrophic <= 0.05
        )
    if gate == 'unique_signal':
        return descriptor['active_candidates'] == 1
    if gate == 'sparse_signal':
        return descriptor['active_candidates'] <= 2
    if gate == 'score_gap_010':
        return bool(descriptor['score_gap'] >= 0.10)
    if gate == 'sparse_regression_any':
        return bool(
            descriptor['active_candidates'] <= 2
            and (row.pred_expected > 0 or row.pred_median > 0)
        )
    raise ValueError(gate)


def summarize_gate(units: pd.DataFrame, gate: str) -> dict:
    accepted = units[units[f'pass__{gate}'] == 1].copy()
    sequence_utility = {seq: 0.0 for seq in sorted(units.seq.unique())}
    if len(accepted):
        for seq, group in accepted.groupby('seq'):
            sequence_utility[seq] = round(float(group.delta_HOTA.sum()), 12)
    positive_sum = float(accepted.loc[accepted.delta_HOTA > 0, 'delta_HOTA'].sum()) if len(accepted) else 0.0
    negative_mass = float(-accepted.loc[accepted.delta_HOTA < 0, 'delta_HOTA'].sum()) if len(accepted) else 0.0
    worst_sequence = min(sequence_utility.values()) if sequence_utility else 0.0
    worst_window = float(accepted.delta_HOTA.min()) if len(accepted) else 0.0
    precision = float((accepted.delta_HOTA > 0).mean()) if len(accepted) else 0.0
    utility_sum = float(accepted.delta_HOTA.sum()) if len(accepted) else 0.0
    robust_score = positive_sum - 5.0 * negative_mass + 2.0 * min(0.0, worst_sequence)
    eligible = bool(
        len(accepted) >= 4
        and precision >= 0.75
        and utility_sum > 0
        and worst_sequence >= 0
        and negative_mass <= 0.25 * max(positive_sum, 1e-12)
    )
    return {
        'gate': gate,
        'selected_windows': int(len(accepted)),
        'positive_windows': int((accepted.delta_HOTA > 0).sum()) if len(accepted) else 0,
        'negative_windows': int((accepted.delta_HOTA < 0).sum()) if len(accepted) else 0,
        'positive_precision': precision,
        'positive_sum': positive_sum,
        'negative_mass': negative_mass,
        'utility_sum': utility_sum,
        'utility_mean_per_selected': utility_sum / len(accepted) if len(accepted) else 0.0,
        'worst_window_utility': worst_window,
        'worst_sequence_utility': worst_sequence,
        'sequence_utility': json.dumps(sequence_utility, sort_keys=True),
        'robust_score': robust_score,
        'eligible': int(eligible),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-features', action='append', required=True)
    ap.add_argument('--train-executability', required=True)
    ap.add_argument('--train-utility', required=True)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    data = attach_train_labels(
        load_directional_features(args.train_features, args.train_executability),
        args.train_utility,
    )
    variants = build_variants()
    missing = {
        name: [feature for feature in features if feature not in data.columns]
        for name, features in variants.items()
    }
    missing = {name: columns for name, columns in missing.items() if columns}
    if missing:
        raise RuntimeError(f'missing feature columns: {missing}')

    unit_rows = []
    fold_rows = []
    calibration_parts = []
    for outer_index, held_out in enumerate(sorted(data.seq.unique())):
        outer_train = data[data.seq != held_out].copy().reset_index(drop=True)
        outer_test = data[data.seq == held_out].copy().reset_index(drop=True)
        chosen, calibration = choose_inner_policy(outer_train, variants, outer_index)
        calibration['held_out_seq'] = held_out
        calibration_parts.append(calibration)
        fold_rows.append({'held_out_seq': held_out, **chosen})

        prediction = fit_predict(
            outer_train,
            outer_test,
            variants[chosen['variant']],
            # Match the exact outer-model seed used for the frozen locked v2
            # prediction of this held-out sequence.
            seed=29000 + outer_index,
        )
        prediction = prediction.merge(
            outer_test[DIRECTION_KEYS + ['delta_HOTA']],
            on=DIRECTION_KEYS,
            how='left',
            validate='one_to_one',
        )
        for start, end in WINDOWS:
            window = prediction[
                (prediction.canonical_rank >= start) & (prediction.canonical_rank <= end)
            ].copy()
            if len(window) != 40:
                raise RuntimeError(
                    f'{held_out} ranks {start}-{end}: expected 40 directional rows, got {len(window)}'
                )
            descriptor = window_descriptor(window, chosen['policy'])
            selected = pd.DataFrame()
            if chosen['policy'] != 'no_op':
                selected = conflict_free(window, chosen['policy'], chosen['cap'])
            if len(selected) > 1:
                selected = selected.head(1).copy()
            row = {
                'seq': held_out,
                'rank_start': start,
                'rank_end': end,
                'chosen_variant': chosen['variant'],
                'chosen_policy': chosen['policy'],
                'chosen_cap': chosen['cap'],
                **descriptor,
                'selected': int(len(selected)),
                'delta_HOTA': float(selected.delta_HOTA.sum()) if len(selected) else 0.0,
            }
            if len(selected):
                candidate = selected.iloc[0]
                for column in DIRECTION_KEYS + [
                    'pred_expected', 'pred_q25', 'pred_median', 'prob_positive',
                    'prob_negative', 'prob_catastrophic', 'prob_margin', 'selection_score',
                ]:
                    row[column] = candidate[column]
            for gate in ABSTENTION_GATES:
                row[f'pass__{gate}'] = int(gate_passes(gate, selected, descriptor))
            unit_rows.append(row)

    units = pd.DataFrame(unit_rows)
    numeric_unit_columns = units.select_dtypes(include=[np.number]).columns
    units[numeric_unit_columns] = units[numeric_unit_columns].round(12)
    summaries = pd.DataFrame([summarize_gate(units, gate) for gate in ABSTENTION_GATES])
    numeric_summary_columns = summaries.select_dtypes(include=[np.number]).columns
    summaries[numeric_summary_columns] = summaries[numeric_summary_columns].round(12)
    eligible = summaries[summaries.eligible == 1].copy()
    if len(eligible):
        chosen_gate = str(
            eligible.sort_values(
                ['robust_score', 'utility_sum', 'positive_precision', 'selected_windows'],
                ascending=[False, False, False, True],
            ).iloc[0].gate
        )
    else:
        chosen_gate = 'no_op'

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    units.to_csv(out / 'window_oof_units.csv', index=False)
    summaries.to_csv(out / 'abstention_gate_summary.csv', index=False)
    pd.DataFrame(fold_rows).to_csv(out / 'fold_policy_selection.csv', index=False)
    pd.concat(calibration_parts, ignore_index=True, sort=False).to_csv(
        out / 'inner_calibration.csv', index=False
    )
    report = {
        'protocol': {
            'scope': (
                'Train-only sequence-aware abstention audit. Each held-out sequence ranks 21-100 '
                'is partitioned into four rank windows of 20 base events (40 directional rows), '
                'mimicking a locked top20 deployment unit.'
            ),
            'locked_rank1_20_labels_read': False,
            'nested_loso': (
                'Outer sequence is never used for model/policy selection. Inner LOSO on the other '
                'three sequences chooses feature variant, candidate policy, and cap.'
            ),
            'gate_family_preregistered': ABSTENTION_GATES,
            'gate_eligibility': (
                'At least four selected windows; positive precision >=0.75; total utility >0; '
                'worst sequence utility >=0; negative mass <=25% of positive sum.'
            ),
            'window_definition': WINDOWS,
        },
        'dataset': {
            'directional_rows': len(data),
            'sequences': sorted(data.seq.unique()),
            'deployment_units': len(units),
        },
        'chosen_abstention_gate': chosen_gate,
        'eligible_gates': eligible.gate.tolist(),
        'gate_summary': summaries.to_dict('records'),
        'folds': fold_rows,
    }
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
