from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import numpy as np

from train_locked_loso_directional_handoff_utility import (
    DIRECTION_KEYS,
    attach_train_labels,
    load_directional_features,
)
from train_nested_within_sequence_knn_directional import (
    LOCKED_WINDOWS,
    SPECS,
    TRAIN_WINDOWS,
    choose_window_candidate,
    clean_locked_prediction,
    exclude_rows,
    load_exclusion,
    rounded,
    window_subset,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def python_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-features', action='append', required=True)
    ap.add_argument('--test-features', action='append', required=True)
    ap.add_argument('--train-executability', required=True)
    ap.add_argument('--test-executability', required=True)
    ap.add_argument('--train-utility', required=True)
    ap.add_argument('--locked-exclusion-keys', required=True)
    ap.add_argument('--final-train-spec-summary', required=True)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    train = attach_train_labels(
        load_directional_features(args.train_features, args.train_executability),
        args.train_utility,
    )
    locked = exclude_rows(
        load_directional_features(args.test_features, args.test_executability),
        load_exclusion(args.locked_exclusion_keys),
    )
    spec_summary = pd.read_csv(args.final_train_spec_summary)
    eligible = spec_summary[spec_summary.eligible == 1].copy()
    spec_map = {spec.name: spec for spec in SPECS}

    prediction_rows = []
    for seq in sorted(train.seq.unique()):
        sequence_train = train[train.seq == seq].reset_index(drop=True)
        sequence_locked = locked[locked.seq == seq].reset_index(drop=True)
        sequence_specs = sorted(eligible.loc[eligible.seq == seq, 'retrieval_spec'].unique())
        for spec_name in sequence_specs:
            spec = spec_map[spec_name]
            training_variants = [('all_windows', sequence_train)]
            training_variants.extend(
                (f'exclude_{start}_{end}', window_subset(sequence_train, (start, end)))
                for start, end in TRAIN_WINDOWS
            )
            for variant_name, variant_train in training_variants:
                prediction = choose_window_candidate(
                    variant_train,
                    sequence_locked,
                    spec,
                    LOCKED_WINDOWS,
                )
                if not len(prediction):
                    continue
                prediction = clean_locked_prediction(prediction)
                row = prediction.iloc[0].to_dict()
                row.update({
                    'seq_context': seq,
                    'retrieval_spec_context': spec_name,
                    'training_variant': variant_name,
                })
                prediction_rows.append(row)

    predictions = pd.DataFrame(prediction_rows)
    summaries = []
    for seq in sorted(train.seq.unique()):
        sequence_predictions = predictions[predictions.seq_context == seq].copy()
        sequence_specs = sorted(eligible.loc[eligible.seq == seq, 'retrieval_spec'].unique())
        if not len(sequence_predictions):
            summaries.append({
                'seq': seq,
                'eligible_specs': len(sequence_specs),
                'predictions': 0,
                'unique_candidates': 0,
                'mode_count': 0,
                'mode_fraction': 0.0,
                'full_train_predictions': 0,
                'full_train_mode_fraction': 0.0,
                'leave_window_predictions': 0,
                'leave_window_mode_fraction': 0.0,
                'stable_candidate': 0,
            })
            continue

        key_counts = sequence_predictions.groupby(DIRECTION_KEYS).size().sort_values(ascending=False)
        mode_key = key_counts.index[0]
        mode_count = int(key_counts.iloc[0])
        full = sequence_predictions[sequence_predictions.training_variant == 'all_windows']
        leave = sequence_predictions[sequence_predictions.training_variant != 'all_windows']

        def fraction_for(frame: pd.DataFrame) -> float:
            if not len(frame):
                return 0.0
            mask = pd.Series(True, index=frame.index)
            for column, value in zip(DIRECTION_KEYS, mode_key):
                mask &= frame[column] == value
            return float(mask.mean())

        full_fraction = fraction_for(full)
        leave_fraction = fraction_for(leave)
        stable = bool(
            len(sequence_specs) >= 3
            and full_fraction >= 0.80
            and leave_fraction >= 0.70
            and mode_count / len(sequence_predictions) >= 0.70
        )
        summary = {
            'seq': seq,
            'eligible_specs': len(sequence_specs),
            'predictions': int(len(sequence_predictions)),
            'unique_candidates': int(len(key_counts)),
            'mode_count': mode_count,
            'mode_fraction': mode_count / len(sequence_predictions),
            'full_train_predictions': int(len(full)),
            'full_train_mode_fraction': full_fraction,
            'leave_window_predictions': int(len(leave)),
            'leave_window_mode_fraction': leave_fraction,
            'stable_candidate': int(stable),
        }
        for column, value in zip(DIRECTION_KEYS, mode_key):
            summary[f'mode_{column}'] = python_scalar(value)
        summaries.append(summary)

    summary_frame = pd.DataFrame(summaries)
    stable_sequences = summary_frame.loc[summary_frame.stable_candidate == 1, 'seq'].tolist()
    selected = pd.DataFrame()
    if stable_sequences:
        rows = []
        for seq in stable_sequences:
            summary = summary_frame[summary_frame.seq == seq].iloc[0]
            mask = predictions.seq_context == seq
            for column in DIRECTION_KEYS:
                mask &= predictions[column] == summary[f'mode_{column}']
            rows.append(predictions[mask].iloc[0])
        selected = pd.DataFrame(rows)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    rounded(predictions).to_csv(out / 'locked_stability_predictions.csv', index=False)
    rounded(summary_frame).to_csv(out / 'sequence_stability_summary.csv', index=False)
    rounded(selected).to_csv(out / 'selected_transactions.csv', index=False)
    report = {
        'protocol': {
            'scope': 'Locked prediction stability audit only; no locked utility or TrackEval labels are read.',
            'previously_revealed_locked_rows_excluded': 4,
            'remaining_locked_rows_audited_without_labels': int(len(locked)),
            'stability_rule': 'At least 3 eligible specs; >=80% full-train consensus; >=70% leave-one-window consensus; >=70% overall consensus on one directional key.',
        },
        'sequence_summaries': summaries,
        'stable_sequences': stable_sequences,
        'selected_transactions': int(len(selected)),
        'remaining_locked_labels_unread': int(len(locked)),
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
