from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from train_locked_loso_directional_handoff_utility import (
    COMPACT_BASE_FEATURES,
    DIRECTIONAL_FEATURES,
    DIRECTION_KEYS,
    attach_train_labels,
    evaluate_inner_predictions,
    fit_predict,
    load_directional_features,
    normalized_target,
    policy_score,
)
from train_loso_future_transaction_utility_ranker import BASE_FEATURES


TRAIN_WINDOWS = [(21, 40), (41, 60), (61, 80), (81, 100)]
LOCKED_WINDOWS = [(1, 20)]


@dataclass(frozen=True)
class GateSpec:
    name: str
    calibration: str
    predictor: str
    alpha: float | None = None
    null_p_max: float | None = None


GATES = [
    GateSpec('pooled_expected_lcb_a25', 'pooled', 'expected', alpha=0.25),
    GateSpec('pooled_expected_lcb_a10', 'pooled', 'expected', alpha=0.10),
    GateSpec('group_expected_lcb_a40', 'group', 'expected', alpha=0.40),
    GateSpec('group_expected_lcb_a25', 'group', 'expected', alpha=0.25),
    GateSpec('pooled_median_lcb_a25', 'pooled', 'median', alpha=0.25),
    GateSpec('group_median_lcb_a40', 'group', 'median', alpha=0.40),
    GateSpec('pooled_dual_lcb_a25', 'pooled', 'dual', alpha=0.25),
    GateSpec('group_dual_lcb_a40', 'group', 'dual', alpha=0.40),
    GateSpec('pooled_expected_nullp20', 'pooled', 'expected_null', null_p_max=0.20),
    GateSpec('group_expected_nullp25', 'group', 'expected_null', null_p_max=0.25),
    GateSpec('pooled_median_nullp20', 'pooled', 'median_null', null_p_max=0.20),
    GateSpec('group_dual_nullp25', 'group', 'dual_null', null_p_max=0.25),
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


def add_normalized_labels(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.reset_index(drop=True).copy()
    result['delta_HOTA_norm'] = normalized_target(result)
    return result


def feature_variants() -> dict[str, list[str]]:
    return {
        'compact_directional': list(dict.fromkeys(COMPACT_BASE_FEATURES + DIRECTIONAL_FEATURES)),
        'full_directional': list(dict.fromkeys(BASE_FEATURES + DIRECTIONAL_FEATURES)),
    }


def choose_ranker(
    outer_train: pd.DataFrame,
    variants: dict[str, list[str]],
    outer_index: int,
) -> tuple[dict, dict[str, pd.DataFrame], pd.DataFrame]:
    candidates: list[dict] = []
    predictions_by_variant: dict[str, pd.DataFrame] = {}
    for variant_index, (variant, feature_names) in enumerate(variants.items()):
        inner_parts = []
        for inner_index, inner_held_out in enumerate(sorted(outer_train.seq.unique())):
            inner_train = outer_train[outer_train.seq != inner_held_out].reset_index(drop=True)
            inner_test = outer_train[outer_train.seq == inner_held_out].reset_index(drop=True)
            prediction = fit_predict(
                inner_train,
                inner_test,
                feature_names,
                seed=23000 + 1000 * outer_index + 100 * variant_index + inner_index,
            )
            prediction = prediction.merge(
                inner_test[DIRECTION_KEYS + ['delta_HOTA', 'delta_HOTA_norm']],
                on=DIRECTION_KEYS,
                how='left',
                validate='one_to_one',
            )
            inner_parts.append(prediction)
        inner_predictions = pd.concat(inner_parts, ignore_index=True, sort=False)
        predictions_by_variant[variant] = inner_predictions
        rows = evaluate_inner_predictions(inner_predictions, variant)
        candidates.extend(rows)

    candidate_frame = pd.DataFrame(candidates)
    eligible = candidate_frame[candidate_frame.eligible == 1].copy()
    if len(eligible):
        chosen = eligible.sort_values(
            ['robust_score', 'raw_utility_sum', 'positive_precision', 'selected'],
            ascending=[False, False, False, True],
        ).iloc[0].to_dict()
        choice = {
            'variant': str(chosen['variant']),
            'policy': str(chosen['policy']),
            'cap': int(chosen['cap']),
            'fallback_no_op': 0,
            'inner_eligible_policies': int(len(eligible)),
        }
    else:
        choice = {
            'variant': 'compact_directional',
            'policy': 'no_op',
            'cap': 0,
            'fallback_no_op': 1,
            'inner_eligible_policies': 0,
        }
    return choice, predictions_by_variant, candidate_frame


def window_top_one(
    predictions: pd.DataFrame,
    labels: pd.DataFrame | None,
    policy: str,
    windows: list[tuple[int, int]],
) -> pd.DataFrame:
    frame = predictions.copy()
    if labels is not None:
        label_columns = DIRECTION_KEYS + ['delta_HOTA', 'delta_HOTA_norm']
        missing = [column for column in label_columns if column not in frame.columns]
        if missing:
            frame = frame.merge(
                labels[label_columns],
                on=DIRECTION_KEYS,
                how='left',
                validate='one_to_one',
            )
    if policy == 'no_op':
        return pd.DataFrame()
    frame = frame[frame.executor_accepted == 1].copy()
    if not len(frame):
        return frame
    frame['ranking_score'] = policy_score(frame, policy)
    rows = []
    for seq, seq_frame in frame.groupby('seq'):
        for rank_start, rank_end in windows:
            window = seq_frame[
                (seq_frame.canonical_rank >= rank_start)
                & (seq_frame.canonical_rank <= rank_end)
            ].copy()
            if not len(window):
                continue
            best = window.sort_values(
                ['ranking_score', 'prob_margin', 'pred_expected', 'canonical_rank'],
                ascending=[False, False, False, True],
            ).iloc[0].to_dict()
            best['rank_start'] = rank_start
            best['rank_end'] = rank_end
            rows.append(best)
    return pd.DataFrame(rows)


def conformal_quantile(values: np.ndarray, alpha: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float('inf')
    level = min(1.0, math.ceil((len(values) + 1) * (1.0 - alpha)) / len(values))
    return float(np.quantile(values, level, method='higher'))


def calibrated_quantile(
    calibration: pd.DataFrame,
    column: str,
    alpha: float,
    mode: str,
) -> float:
    if mode == 'pooled':
        return conformal_quantile(calibration[column].to_numpy(float), alpha)
    per_sequence = [
        conformal_quantile(group[column].to_numpy(float), alpha)
        for _, group in calibration.groupby('seq')
    ]
    return max(per_sequence) if per_sequence else float('inf')


def null_p_value(
    calibration: pd.DataFrame,
    score: float,
    mode: str,
) -> float:
    null = calibration[calibration.delta_HOTA_norm <= 0]
    if mode == 'pooled':
        scores = null.prob_margin.to_numpy(float)
        return float((1 + np.sum(scores >= score)) / (len(scores) + 1)) if len(scores) else 1.0
    values = []
    for seq in sorted(calibration.seq.unique()):
        scores = null.loc[null.seq == seq, 'prob_margin'].to_numpy(float)
        values.append(
            float((1 + np.sum(scores >= score)) / (len(scores) + 1)) if len(scores) else 1.0
        )
    return max(values) if values else 1.0


def certify(
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    gate: GateSpec,
) -> pd.DataFrame:
    result = test.copy()
    calibration = calibration.copy()
    calibration['nc_expected'] = calibration.pred_expected - calibration.delta_HOTA_norm
    calibration['nc_median'] = calibration.pred_median - calibration.delta_HOTA_norm
    result['certificate_gate'] = gate.name
    result['certificate_pass'] = 0
    result['lcb_expected'] = np.nan
    result['lcb_median'] = np.nan
    result['null_p_value'] = np.nan

    if gate.alpha is not None:
        q_expected = calibrated_quantile(calibration, 'nc_expected', gate.alpha, gate.calibration)
        q_median = calibrated_quantile(calibration, 'nc_median', gate.alpha, gate.calibration)
        result['lcb_expected'] = result.pred_expected - q_expected
        result['lcb_median'] = result.pred_median - q_median

    if gate.null_p_max is not None:
        result['null_p_value'] = [
            null_p_value(calibration, float(score), gate.calibration)
            for score in result.prob_margin
        ]

    if gate.predictor == 'expected':
        mask = result.lcb_expected > 0
    elif gate.predictor == 'median':
        mask = result.lcb_median > 0
    elif gate.predictor == 'dual':
        mask = (result.lcb_expected > 0) & (result.lcb_median > 0)
    elif gate.predictor == 'expected_null':
        mask = (result.pred_expected > 0) & (result.null_p_value <= float(gate.null_p_max))
    elif gate.predictor == 'median_null':
        mask = (result.pred_median > 0) & (result.null_p_value <= float(gate.null_p_max))
    elif gate.predictor == 'dual_null':
        mask = (
            (result.pred_expected > 0)
            & (result.pred_median > 0)
            & (result.null_p_value <= float(gate.null_p_max))
        )
    else:
        raise ValueError(gate.predictor)
    result['certificate_pass'] = mask.astype(int)
    return result


def summarize_gate(frame: pd.DataFrame, gate_name: str, minimum_selected: int) -> dict:
    selected = frame[frame.certificate_pass == 1].copy()
    sequences = sorted(frame.seq.unique())
    sequence_utility = {seq: 0.0 for seq in sequences}
    if len(selected):
        for seq, group in selected.groupby('seq'):
            sequence_utility[seq] = float(group.delta_HOTA.sum())
    positive_sum = float(selected.loc[selected.delta_HOTA > 0, 'delta_HOTA'].sum()) if len(selected) else 0.0
    negative_mass = float(-selected.loc[selected.delta_HOTA < 0, 'delta_HOTA'].sum()) if len(selected) else 0.0
    precision = float((selected.delta_HOTA > 0).mean()) if len(selected) else 0.0
    worst_sequence = min(sequence_utility.values()) if sequence_utility else 0.0
    utility_sum = float(selected.delta_HOTA.sum()) if len(selected) else 0.0
    eligible = bool(
        len(selected) >= minimum_selected
        and precision >= 0.75
        and positive_sum > 0
        and utility_sum > 0
        and worst_sequence >= 0
        and negative_mass <= 0.25 * positive_sum
    )
    return {
        'gate': gate_name,
        'selected_windows': int(len(selected)),
        'positive_windows': int((selected.delta_HOTA > 0).sum()) if len(selected) else 0,
        'negative_windows': int((selected.delta_HOTA < 0).sum()) if len(selected) else 0,
        'positive_precision': precision,
        'positive_sum': positive_sum,
        'negative_mass': negative_mass,
        'utility_sum': utility_sum,
        'worst_sequence_utility': worst_sequence,
        'sequence_utility': json.dumps(sequence_utility, sort_keys=True),
        'robust_score': positive_sum - 6.0 * negative_mass + 3.0 * min(0.0, worst_sequence),
        'eligible': int(eligible),
    }


def pairwise_calibration_topones(
    outer_train: pd.DataFrame,
    calibration_sequences: list[str],
    variant_features: list[str],
    policy: str,
    seed_base: int,
) -> pd.DataFrame:
    if len(calibration_sequences) != 2:
        raise RuntimeError('pairwise calibration requires exactly two sequences')
    parts = []
    for index, target_seq in enumerate(calibration_sequences):
        source_seq = calibration_sequences[1 - index]
        model_train = outer_train[outer_train.seq == source_seq].reset_index(drop=True)
        model_test = outer_train[outer_train.seq == target_seq].reset_index(drop=True)
        prediction = fit_predict(
            model_train,
            model_test,
            variant_features,
            seed=seed_base + index,
        )
        prediction = prediction.merge(
            model_test[DIRECTION_KEYS + ['delta_HOTA', 'delta_HOTA_norm']],
            on=DIRECTION_KEYS,
            how='left',
            validate='one_to_one',
        )
        top = window_top_one(prediction, None, policy, TRAIN_WINDOWS)
        if len(top):
            parts.append(top)
    return pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()


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

    train = add_normalized_labels(
        attach_train_labels(
            load_directional_features(args.train_features, args.train_executability),
            args.train_utility,
        )
    )
    locked_all = load_directional_features(args.test_features, args.test_executability)
    exclusion = load_exclusion_keys(args.locked_exclusion_keys)
    locked = exclude_locked_rows(locked_all, exclusion)
    if set(train.seq.unique()) != set(locked.seq.unique()):
        raise RuntimeError('train/locked sequence sets differ')

    variants = feature_variants()
    for name, features in variants.items():
        missing = [feature for feature in features if feature not in train.columns or feature not in locked.columns]
        if missing:
            raise RuntimeError(f'{name} missing features: {missing}')

    sequences = sorted(train.seq.unique())
    outer_rows = []
    inner_gate_rows = []
    outer_oof_parts = []
    locked_parts = []
    ranker_calibration_parts = []

    for outer_index, held_out in enumerate(sequences):
        outer_train = train[train.seq != held_out].reset_index(drop=True)
        outer_test = train[train.seq == held_out].reset_index(drop=True)
        locked_test = locked[locked.seq == held_out].reset_index(drop=True)

        choice, predictions_by_variant, ranker_calibration = choose_ranker(
            outer_train,
            variants,
            outer_index,
        )
        ranker_calibration['held_out_seq'] = held_out
        ranker_calibration_parts.append(ranker_calibration)
        variant = choice['variant']
        policy = choice['policy']
        inner_predictions = predictions_by_variant[variant]
        inner_topones = window_top_one(inner_predictions, None, policy, TRAIN_WINDOWS)

        gate_eval_parts = {gate.name: [] for gate in GATES}
        if policy != 'no_op':
            for gate_test_index, gate_test_seq in enumerate(sorted(outer_train.seq.unique())):
                calibration_sequences = [
                    seq for seq in sorted(outer_train.seq.unique()) if seq != gate_test_seq
                ]
                calibration_topones = pairwise_calibration_topones(
                    outer_train,
                    calibration_sequences,
                    variants[variant],
                    policy,
                    seed_base=51000 + 1000 * outer_index + 100 * gate_test_index,
                )
                gate_test_topones = inner_topones[inner_topones.seq == gate_test_seq].copy()
                if not len(calibration_topones) or not len(gate_test_topones):
                    continue
                for gate in GATES:
                    certified = certify(calibration_topones, gate_test_topones, gate)
                    certified['outer_held_out_seq'] = held_out
                    certified['gate_test_seq'] = gate_test_seq
                    gate_eval_parts[gate.name].append(certified)

        gate_summaries = []
        gate_eval_frames = {}
        for gate in GATES:
            parts = gate_eval_parts[gate.name]
            frame = pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()
            gate_eval_frames[gate.name] = frame
            if len(frame):
                summary = summarize_gate(frame, gate.name, minimum_selected=3)
            else:
                summary = {
                    'gate': gate.name,
                    'selected_windows': 0,
                    'positive_windows': 0,
                    'negative_windows': 0,
                    'positive_precision': 0.0,
                    'positive_sum': 0.0,
                    'negative_mass': 0.0,
                    'utility_sum': 0.0,
                    'worst_sequence_utility': 0.0,
                    'sequence_utility': json.dumps({}, sort_keys=True),
                    'robust_score': 0.0,
                    'eligible': 0,
                }
            summary['outer_held_out_seq'] = held_out
            gate_summaries.append(summary)
            if len(frame):
                inner_gate_rows.append(frame)

        gate_summary_frame = pd.DataFrame(gate_summaries)
        eligible_gates = gate_summary_frame[gate_summary_frame.eligible == 1].copy()
        if len(eligible_gates):
            chosen_gate = str(
                eligible_gates.sort_values(
                    ['robust_score', 'utility_sum', 'positive_precision', 'selected_windows'],
                    ascending=[False, False, False, True],
                ).iloc[0]['gate']
            )
        else:
            chosen_gate = 'no_op'

        combined_test = pd.concat(
            [
                outer_test.assign(_split='outer_oof'),
                locked_test.assign(_split='locked'),
            ],
            ignore_index=True,
            sort=False,
        )
        prediction = fit_predict(
            outer_train,
            combined_test,
            variants[variant],
            seed=29000 + outer_index,
        )
        split_columns = DIRECTION_KEYS + ['_split']
        prediction = prediction.merge(
            combined_test[split_columns],
            on=DIRECTION_KEYS,
            how='left',
            validate='one_to_one',
        )
        outer_prediction = prediction[prediction._split == 'outer_oof'].drop(columns=['_split'])
        outer_prediction = outer_prediction.merge(
            outer_test[DIRECTION_KEYS + ['delta_HOTA', 'delta_HOTA_norm']],
            on=DIRECTION_KEYS,
            how='left',
            validate='one_to_one',
        )
        locked_prediction = prediction[prediction._split == 'locked'].drop(columns=['_split'])
        outer_topones = window_top_one(outer_prediction, None, policy, TRAIN_WINDOWS)
        locked_topones = window_top_one(locked_prediction, None, policy, LOCKED_WINDOWS)

        if chosen_gate != 'no_op' and len(inner_topones):
            gate = next(spec for spec in GATES if spec.name == chosen_gate)
            outer_certified = certify(inner_topones, outer_topones, gate) if len(outer_topones) else outer_topones
            locked_certified = certify(inner_topones, locked_topones, gate) if len(locked_topones) else locked_topones
        else:
            outer_certified = outer_topones.copy()
            locked_certified = locked_topones.copy()
            for frame in [outer_certified, locked_certified]:
                frame['certificate_gate'] = chosen_gate
                frame['certificate_pass'] = 0
                frame['lcb_expected'] = np.nan
                frame['lcb_median'] = np.nan
                frame['null_p_value'] = np.nan

        if len(outer_certified):
            outer_certified['outer_held_out_seq'] = held_out
            outer_oof_parts.append(outer_certified)
        if len(locked_certified):
            locked_certified['outer_held_out_seq'] = held_out
            locked_parts.append(locked_certified)

        outer_rows.append({
            'held_out_seq': held_out,
            **choice,
            'chosen_conformal_gate': chosen_gate,
            'eligible_conformal_gates': int(len(eligible_gates)),
            'inner_topone_windows': int(len(inner_topones)),
            'outer_topone_windows': int(len(outer_topones)),
            'outer_certificate_passes': int(outer_certified.certificate_pass.sum()) if len(outer_certified) else 0,
            'locked_topone_candidates': int(len(locked_topones)),
            'locked_certificate_passes': int(locked_certified.certificate_pass.sum()) if len(locked_certified) else 0,
        })

    outer_oof = pd.concat(outer_oof_parts, ignore_index=True, sort=False) if outer_oof_parts else pd.DataFrame()
    locked_candidates = pd.concat(locked_parts, ignore_index=True, sort=False) if locked_parts else pd.DataFrame()
    inner_gate = pd.concat(inner_gate_rows, ignore_index=True, sort=False) if inner_gate_rows else pd.DataFrame()
    ranker_calibration = pd.concat(ranker_calibration_parts, ignore_index=True, sort=False)

    procedure_summary = summarize_gate(outer_oof, 'nested_conformal_procedure', minimum_selected=4) if len(outer_oof) else {
        'gate': 'nested_conformal_procedure',
        'selected_windows': 0,
        'positive_windows': 0,
        'negative_windows': 0,
        'positive_precision': 0.0,
        'positive_sum': 0.0,
        'negative_mass': 0.0,
        'utility_sum': 0.0,
        'worst_sequence_utility': 0.0,
        'sequence_utility': json.dumps({}, sort_keys=True),
        'robust_score': 0.0,
        'eligible': 0,
    }
    deployment_allowed = bool(procedure_summary['eligible'])
    if len(locked_candidates):
        locked_candidates['selected_by_fold_certificate'] = locked_candidates.certificate_pass.astype(int)
        locked_candidates['selected_by_final_gate'] = (
            locked_candidates.certificate_pass.astype(int) if deployment_allowed else 0
        )
    selected_locked = (
        locked_candidates[locked_candidates.selected_by_final_gate == 1].copy()
        if len(locked_candidates)
        else pd.DataFrame()
    )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    files = {
        'outer_fold_choices.csv': rounded(pd.DataFrame(outer_rows)),
        'inner_gate_crossfit.csv': rounded(inner_gate),
        'ranker_inner_calibration.csv': rounded(ranker_calibration),
        'outer_window_oof.csv': rounded(outer_oof),
        'locked_candidate_certificates.csv': rounded(locked_candidates),
        'selected_transactions.csv': rounded(selected_locked),
    }
    for name, frame in files.items():
        frame.to_csv(out / name, index=False)

    report = {
        'protocol': {
            'scope': 'Train-only nested window-conditioned top-one conformal risk audit, followed by prediction on previously unread locked rows only.',
            'locked_utility_or_trackeval_labels_read': False,
            'previously_revealed_locked_candidates_excluded': int(len(exclusion)),
            'outer_loso': 'Each sequence ranks21-100 is an unbiased outer test fold.',
            'ranker_selection': 'Original exact-seed nested LOSO directional variant/policy selection is preserved.',
            'window_top_one': 'Within each 20-rank window, rank all executable directions by the chosen policy score and keep one candidate before risk certification.',
            'conformal_calibration': 'Gate selection uses pairwise cross-fitted top-one residuals from the three outer-train sequences; outer held-out sequence labels are excluded.',
            'gate_family_preregistered': [gate.__dict__ for gate in GATES],
            'inner_gate_eligibility': 'selected>=3, precision>=0.75, positive and total utility>0, worst sequence>=0, negative mass<=25% positive sum.',
            'global_deployment_eligibility': 'outer OOF selected>=4 with the same precision, utility, worst-sequence, and negative-mass constraints.',
        },
        'dataset': {
            'training_directional_rows': int(len(train)),
            'locked_directional_rows_before_exclusion': int(len(locked_all)),
            'locked_excluded_previously_revealed': int(len(exclusion)),
            'locked_directional_rows_after_exclusion': int(len(locked)),
            'locked_executable_rows_after_exclusion': int((locked.executor_accepted == 1).sum()),
            'sequences': sequences,
            'outer_windows': 16,
        },
        'outer_fold_choices': outer_rows,
        'outer_oof_summary': procedure_summary,
        'deployment_allowed': deployment_allowed,
        'locked_fold_certificate_passes': int(locked_candidates.certificate_pass.sum()) if len(locked_candidates) else 0,
        'locked_selected_after_global_gate': int(len(selected_locked)),
        'remaining_locked_directional_labels_unread': 156,
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
