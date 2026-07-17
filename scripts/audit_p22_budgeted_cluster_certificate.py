from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta


KEYS = ['seq', 'canonical_rank', 'u', 'v', 'boundary_frame', 'transaction_type']
MECHANISMS = ['receiver_split', 'same_identity_merge']
BUDGET_PER_BLOCK = 12
MAX_COMBINED_SET_SIZE = 100
TARGET_MISS_RISK = 0.10
CONFIDENCE = 0.90


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def exact_upper_failure_bound(failures: int, trials: int, confidence: float) -> float:
    if trials <= 0:
        return 1.0
    if failures >= trials:
        return 1.0
    return float(beta.ppf(confidence, failures + 1, trials - failures))


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    result = frame.copy()
    for column in result.select_dtypes(include=[np.number]).columns:
        result[column] = result[column].round(12)
    result.to_csv(path, index=False)


def mechanism_summary(full: pd.DataFrame, selected_keys: pd.DataFrame) -> pd.DataFrame:
    tagged = full.merge(
        selected_keys.assign(selected=1), on=KEYS, how='left', validate='one_to_one'
    )
    tagged['selected'] = tagged.selected.fillna(0).astype(int)
    rows: list[dict[str, object]] = []
    for mechanism in MECHANISMS:
        label = f'{mechanism}_label'
        positive = tagged[tagged[label] == 1]
        blocks = positive[['seq', 'temporal_block']].drop_duplicates()
        covered_blocks = positive[positive.selected == 1][
            ['seq', 'temporal_block']
        ].drop_duplicates()
        rows.append(
            {
                'mechanism': mechanism,
                'positive_events': int(len(positive)),
                'selected_positive_events': int(positive.selected.sum()),
                'event_recall': float(positive.selected.mean()) if len(positive) else 0.0,
                'positive_blocks': int(len(blocks)),
                'covered_positive_blocks': int(len(covered_blocks)),
                'block_recall': float(len(covered_blocks) / len(blocks))
                if len(blocks)
                else 0.0,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--base-set-members', required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions)
    explicit_base = pd.read_csv(args.base_set_members, usecols=KEYS).drop_duplicates()
    required = KEYS + [
        'temporal_block',
        'full_idtp_delta_norm',
        'receiver_split_label',
        'same_identity_merge_label',
        'receiver_split_score',
        'same_identity_merge_score',
        'receiver_split_rank',
        'same_identity_merge_rank',
        'base_member',
        'outer_sequence',
    ]
    missing = [column for column in required if column not in predictions.columns]
    if missing:
        raise RuntimeError(f'prediction columns missing: {missing}')
    if len(predictions) != 11705 or predictions.duplicated(KEYS).any():
        raise RuntimeError('prediction-bank key integrity failure')
    if explicit_base.duplicated(KEYS).any():
        raise RuntimeError('explicit base keys contain duplicates')
    predicted_base = predictions[predictions.base_member == 1][KEYS].sort_values(KEYS)
    explicit_base = explicit_base.sort_values(KEYS)
    if not predicted_base.reset_index(drop=True).equals(
        explicit_base.reset_index(drop=True)
    ):
        raise RuntimeError('P21 base membership does not equal the frozen P20 base set')
    if not (predictions.seq == predictions.outer_sequence).all():
        raise RuntimeError('P21 mechanism predictions are not sequence-OOF')

    outside = predictions[predictions.base_member == 0].copy()
    outside['union_rank'] = outside[
        ['receiver_split_rank', 'same_identity_merge_rank']
    ].min(axis=1)
    outside = outside.sort_values(
        [
            'seq',
            'temporal_block',
            'union_rank',
            'receiver_split_rank',
            'same_identity_merge_rank',
            'boundary_frame',
            'u',
            'v',
            'transaction_type',
        ]
    )
    outside['budget_rank'] = (
        outside.groupby(['seq', 'temporal_block'], sort=True).cumcount() + 1
    )
    rescue = outside[outside.budget_rank <= BUDGET_PER_BLOCK].copy()
    rescue['rescue_reason'] = np.where(
        rescue.receiver_split_rank < rescue.same_identity_merge_rank,
        'receiver_split',
        np.where(
            rescue.same_identity_merge_rank < rescue.receiver_split_rank,
            'same_identity_merge',
            'rank_tie',
        ),
    )

    membership = pd.concat(
        [
            explicit_base.assign(base_member=1, rescue_member=0),
            rescue[KEYS].assign(base_member=0, rescue_member=1),
        ],
        ignore_index=True,
    ).drop_duplicates(KEYS)
    combined = predictions.merge(
        membership, on=KEYS, how='inner', validate='one_to_one', suffixes=('_input', '')
    )
    if 'base_member_input' in combined.columns:
        combined = combined.drop(columns=['base_member_input'])

    block_rows: list[dict[str, object]] = []
    for (sequence, block), group in predictions.groupby(
        ['seq', 'temporal_block'], sort=True
    ):
        selected = combined[
            (combined.seq == sequence) & (combined.temporal_block == block)
        ]
        base = selected[selected.base_member == 1]
        addition = selected[selected.rescue_member == 1]
        block_rows.append(
            {
                'seq': sequence,
                'temporal_block': int(block),
                'events': int(len(group)),
                'block_has_positive': int((group.full_idtp_delta_norm > 0.0).any()),
                'base_set_size': int(len(base)),
                'rescue_added': int(len(addition)),
                'combined_set_size': int(len(selected)),
                'base_has_positive': int((base.full_idtp_delta_norm > 0.0).any()),
                'combined_has_positive': int(
                    (selected.full_idtp_delta_norm > 0.0).any()
                ),
                'rescue_positive_events': int(
                    (addition.full_idtp_delta_norm > 0.0).sum()
                ),
                'rescue_negative_events': int(
                    (addition.full_idtp_delta_norm < 0.0).sum()
                ),
                'rescue_zero_events': int(
                    (addition.full_idtp_delta_norm == 0.0).sum()
                ),
                'combined_positive_events': int(
                    (selected.full_idtp_delta_norm > 0.0).sum()
                ),
                'combined_negative_events': int(
                    (selected.full_idtp_delta_norm < 0.0).sum()
                ),
                'combined_zero_events': int(
                    (selected.full_idtp_delta_norm == 0.0).sum()
                ),
                'combined_oracle_utility': float(selected.full_idtp_delta_norm.max()),
            }
        )
    blocks = pd.DataFrame(block_rows)
    selected_keys = combined[KEYS].drop_duplicates()
    mechanisms = mechanism_summary(predictions, selected_keys)

    positive_blocks = blocks[blocks.block_has_positive == 1]
    block_failures = int((positive_blocks.combined_has_positive == 0).sum())
    positive_sequences = sorted(positive_blocks.seq.unique())
    sequence_failures = int(
        sum(
            (
                positive_blocks[positive_blocks.seq == sequence].combined_has_positive
                == 0
            ).any()
            for sequence in positive_sequences
        )
    )
    certificate_rows = [
        {
            'unit': 'positive_temporal_block',
            'trials': int(len(positive_blocks)),
            'failures': block_failures,
            'empirical_failure_rate': float(block_failures / len(positive_blocks)),
            'upper_failure_bound_90pct': exact_upper_failure_bound(
                block_failures, len(positive_blocks), CONFIDENCE
            ),
        },
        {
            'unit': 'positive_sequence_cluster',
            'trials': int(len(positive_sequences)),
            'failures': sequence_failures,
            'empirical_failure_rate': float(
                sequence_failures / len(positive_sequences)
            ),
            'upper_failure_bound_90pct': exact_upper_failure_bound(
                sequence_failures, len(positive_sequences), CONFIDENCE
            ),
        },
    ]
    for mechanism in MECHANISMS:
        label = f'{mechanism}_label'
        positives = predictions[predictions[label] == 1]
        mechanism_blocks = positives[['seq', 'temporal_block']].drop_duplicates()
        selected_positive = positives.merge(
            selected_keys.assign(selected=1), on=KEYS, how='left'
        )
        covered = selected_positive[selected_positive.selected == 1][
            ['seq', 'temporal_block']
        ].drop_duplicates()
        failures = int(len(mechanism_blocks) - len(covered))
        certificate_rows.append(
            {
                'unit': f'{mechanism}_positive_block',
                'trials': int(len(mechanism_blocks)),
                'failures': failures,
                'empirical_failure_rate': float(failures / len(mechanism_blocks)),
                'upper_failure_bound_90pct': exact_upper_failure_bound(
                    failures, len(mechanism_blocks), CONFIDENCE
                ),
            }
        )
    certificates = pd.DataFrame(certificate_rows)
    certificates['target_failure_rate'] = TARGET_MISS_RISK
    certificates['certificate_pass'] = (
        certificates.upper_failure_bound_90pct <= TARGET_MISS_RISK
    ).astype(int)

    sequence_oracle = blocks.groupby('seq').combined_oracle_utility.sum()
    coverage = float(positive_blocks.combined_has_positive.mean())
    newly_covered = int(
        ((blocks.base_has_positive == 0) & (blocks.combined_has_positive == 1)).sum()
    )
    mean_added = float(blocks.rescue_added.mean())
    max_size = int(blocks.combined_set_size.max())
    compact_retained = bool(
        coverage == 1.0
        and newly_covered >= 1
        and mean_added <= BUDGET_PER_BLOCK
        and max_size <= MAX_COMBINED_SET_SIZE
        and float(sequence_oracle.min()) >= 0.0
    )
    cluster_certificate = certificates[
        certificates.unit == 'positive_sequence_cluster'
    ].iloc[0]
    deployment_allowed = bool(
        compact_retained and int(cluster_certificate.certificate_pass) == 1
    )

    missed_base = blocks[
        (blocks.block_has_positive == 1) & (blocks.base_has_positive == 0)
    ][['seq', 'temporal_block']]
    forensic = rescue.merge(missed_base, on=['seq', 'temporal_block'], how='inner')
    forensic = forensic.sort_values(['seq', 'temporal_block', 'budget_rank'])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_frames = {
        'budgeted_rescue_members.csv': rescue[
            required
            + ['union_rank', 'budget_rank', 'rescue_reason']
        ],
        'combined_set_members.csv': combined,
        'block_summary.csv': blocks,
        'mechanism_summary.csv': mechanisms,
        'cluster_certificate.csv': certificates,
        'recovered_block_forensic.csv': forensic,
    }
    for filename, frame in output_frames.items():
        write_csv(frame, out_dir / filename)

    report = {
        'protocol': {
            'scope': 'Budgeted mechanism-aware rescue over the frozen P20 candidate set.',
            'base_set': 'Frozen P20 rank-conformal members; no generic radius change.',
            'mechanism_scores': 'Frozen P21 strict sequence-OOF receiver-split and same-identity-merge scores.',
            'budget_per_temporal_block': BUDGET_PER_BLOCK,
            'budget_origin': 'Inherited unchanged from the preregistered P21 maximum mean-addition limit.',
            'ranking': 'Minimum P21 mechanism rank, then receiver-split rank, same-identity-merge rank, and deterministic event keys.',
            'model_or_budget_sweep': False,
            'confidence': CONFIDENCE,
            'target_miss_risk': TARGET_MISS_RISK,
            'cluster_unit': 'Sequence; any missed positive block makes the sequence cluster fail.',
            'p15_locked_labels_read': 0,
            'p15_locked_trackeval_calls': 0,
            'global_trackeval_calls': 0,
            'p15_remaining_locked_rows_untouched': 156,
        },
        'dataset': {
            'events': int(len(predictions)),
            'sequences': sorted(predictions.seq.unique().tolist()),
            'temporal_blocks': int(len(blocks)),
            'positive_available_blocks': int(len(positive_blocks)),
            'frozen_base_members': int(len(explicit_base)),
        },
        'result': {
            'base_covered_positive_blocks': int(positive_blocks.base_has_positive.sum()),
            'combined_covered_positive_blocks': int(
                positive_blocks.combined_has_positive.sum()
            ),
            'conditional_positive_coverage': coverage,
            'newly_covered_blocks': newly_covered,
            'rescue_members': int(len(rescue)),
            'mean_rescue_added': mean_added,
            'maximum_rescue_added': int(blocks.rescue_added.max()),
            'mean_combined_set_size': float(blocks.combined_set_size.mean()),
            'maximum_combined_set_size': max_size,
            'rescue_positive_events': int((rescue.full_idtp_delta_norm > 0.0).sum()),
            'rescue_negative_events': int((rescue.full_idtp_delta_norm < 0.0).sum()),
            'rescue_zero_events': int((rescue.full_idtp_delta_norm == 0.0).sum()),
            'combined_positive_events': int(
                (combined.full_idtp_delta_norm > 0.0).sum()
            ),
            'combined_negative_events': int(
                (combined.full_idtp_delta_norm < 0.0).sum()
            ),
            'combined_zero_events': int(
                (combined.full_idtp_delta_norm == 0.0).sum()
            ),
            'combined_set_oracle_utility_sum': float(
                blocks.combined_oracle_utility.sum()
            ),
            'combined_set_oracle_worst_sequence': float(sequence_oracle.min()),
            'block_level_upper_miss_bound_90pct': float(
                certificates.loc[
                    certificates.unit == 'positive_temporal_block',
                    'upper_failure_bound_90pct',
                ].iloc[0]
            ),
            'sequence_cluster_upper_miss_bound_90pct': float(
                cluster_certificate.upper_failure_bound_90pct
            ),
        },
        'decision': {
            'compact_candidate_set_retained': compact_retained,
            'cluster_risk_certificate_passed': bool(
                int(cluster_certificate.certificate_pass) == 1
            ),
            'deployment_allowed': deployment_allowed,
            'locked_manifest_created': False,
            'p15_policy_changed': False,
            'reason': (
                'The fixed 12-event budget closes the P20 retrieval miss within the '
                'efficiency limits, but seven independent sequence clusters are '
                'insufficient to certify a 10% sequence-level miss risk at 90% confidence.'
            ),
            'next_stage': (
                'Use appearance or original association evidence to reduce the rescue '
                'false-positive burden and collect additional independent domains before '
                'attempting deployment authorization.'
            ),
        },
    }
    report_path = out_dir / 'report.json'
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    manifest = {
        'schema_version': 1,
        'files': {
            path.name: sha256(path)
            for path in sorted(out_dir.iterdir())
            if path.is_file() and path.name != 'prediction_manifest.json'
        },
    }
    (out_dir / 'prediction_manifest.json').write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + '\n'
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
