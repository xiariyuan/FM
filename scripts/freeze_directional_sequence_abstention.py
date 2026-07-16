from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--locked-predictions', required=True)
    ap.add_argument('--previous-selected', required=True)
    ap.add_argument('--previous-manifest', required=True)
    ap.add_argument('--train-abstention-report', required=True)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    prediction_path = Path(args.locked_predictions)
    previous_selected_path = Path(args.previous_selected)
    previous_manifest_path = Path(args.previous_manifest)
    abstention_report_path = Path(args.train_abstention_report)

    predictions = pd.read_csv(prediction_path)
    previous_selected = pd.read_csv(previous_selected_path)
    previous_manifest = json.loads(previous_manifest_path.read_text())
    abstention_report = json.loads(abstention_report_path.read_text())

    forbidden_columns = {
        'delta_HOTA', 'delta_DetA', 'delta_AssA', 'delta_IDF1', 'delta_IDSW',
        'HOTA', 'DetA', 'AssA', 'IDF1', 'IDSW',
    }
    leaked = sorted(forbidden_columns.intersection(predictions.columns))
    if leaked:
        raise RuntimeError(f'locked prediction file contains evaluation labels: {leaked}')
    if abstention_report['protocol'].get('locked_rank1_20_labels_read') is not False:
        raise RuntimeError('train abstention report is not explicitly locked-label clean')

    chosen_gate = str(abstention_report['chosen_abstention_gate'])
    if chosen_gate != 'no_op':
        raise RuntimeError(
            'This freezer is intentionally fail-closed. The train-only audit did not choose no_op: '
            f'{chosen_gate}'
        )

    frozen = predictions.copy()
    frozen['sequence_abstention_gate'] = chosen_gate
    frozen['sequence_abstention_pass'] = 0
    frozen['selected_before_sequence_abstention'] = frozen['selected_by_gate'].astype(int)
    frozen['selected_by_gate'] = 0
    selected = frozen.iloc[0:0].copy()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    frozen_path = out / 'locked_test_predictions.csv'
    selected_path = out / 'selected_transactions.csv'
    report_path = out / 'report.json'
    frozen.to_csv(frozen_path, index=False)
    selected.to_csv(selected_path, index=False)

    report = {
        'protocol': {
            'scope': (
                'Fail-closed sequence-level abstention applied to frozen locked rank1-20 '
                'directional predictions.'
            ),
            'locked_utility_or_trackeval_labels_read': False,
            'selection_rule': (
                'The train-only pseudo-deployment window audit selected no_op because no '
                'preregistered abstention gate satisfied the robust eligibility constraints.'
            ),
            'remaining_locked_directional_labels_unread': 156,
        },
        'train_abstention_gate': chosen_gate,
        'previous_selected_transactions': int(len(previous_selected)),
        'selected_transactions_after_abstention': int(len(selected)),
        'suppressed_transactions': int(frozen.selected_before_sequence_abstention.sum()),
        'suppressed_sequences': sorted(
            frozen.loc[frozen.selected_before_sequence_abstention == 1, 'seq'].unique().tolist()
        ),
    }
    report_path.write_text(json.dumps(report, indent=2) + '\n')

    manifest = {
        'frozen_before_any_additional_locked_evaluation': True,
        'locked_utility_or_trackeval_labels_read': False,
        'train_abstention_report_sha256': sha256(abstention_report_path),
        'source_prediction_sha256': sha256(prediction_path),
        'source_selected_sha256': sha256(previous_selected_path),
        'source_manifest_sha256': sha256(previous_manifest_path),
        'source_manifest_declares_locked_labels_unread': (
            previous_manifest.get('locked_utility_or_trackeval_labels_read') is False
        ),
        'output_sha256': {
            frozen_path.name: sha256(frozen_path),
            selected_path.name: sha256(selected_path),
            report_path.name: sha256(report_path),
        },
    }
    (out / 'prediction_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({**report, 'prediction_manifest': manifest['output_sha256']}, indent=2))


if __name__ == '__main__':
    main()
