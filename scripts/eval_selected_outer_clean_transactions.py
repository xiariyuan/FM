from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from eval_canonical_segment_transaction_replay import (
    apply_local_transactions,
    reconstruct_best_with_provenance,
    records_to_rows,
)
from eval_assa_swap_merge_fusion import write_rows

SEQS = ['MOT20-01', 'MOT20-02', 'MOT20-03', 'MOT20-05']


def parse_eval(detail: Path) -> dict:
    rows = list(csv.DictReader(detail.open()))
    result = {}
    for row in rows:
        seq = row['seq']
        if seq not in SEQS + ['COMBINED']:
            continue
        result[seq] = {
            'HOTA': float(row['HOTA___AUC']) * 100,
            'DetA': float(row['DetA___AUC']) * 100,
            'AssA': float(row['AssA___AUC']) * 100,
            'IDF1': float(row['IDF1']) * 100 if float(row['IDF1']) < 2 else float(row['IDF1']),
            'IDSW': int(float(row['IDSW'])),
        }
    result['simple_avg_HOTA'] = sum(result[seq]['HOTA'] for seq in SEQS) / len(SEQS)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--merge-links', required=True)
    parser.add_argument('--aggressive-events', required=True)
    parser.add_argument('--selected-transactions', required=True)
    parser.add_argument('--baseline-summary', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--tracker-name', default='nested_risk_gated_fitr')
    args = parser.parse_args()

    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    track_output = output / 'track_results'
    track_output.mkdir(parents=True, exist_ok=True)

    links = pd.read_csv(args.merge_links)
    aggressive = pd.read_csv(args.aggressive_events)
    selected_path = Path(args.selected_transactions)
    selected = pd.read_csv(selected_path) if selected_path.exists() and selected_path.stat().st_size > 0 else pd.DataFrame()
    baseline = json.loads(Path(args.baseline_summary).read_text())['eval']

    execution = []
    for seq in SEQS:
        records, selected_merge_links = reconstruct_best_with_provenance(
            Path(args.source_root) / f'{seq}.txt',
            links[links.seq == seq],
            aggressive[aggressive.seq == seq],
        )
        events = selected[selected.seq == seq].to_dict('records') if len(selected) else []
        modified, accepted, rejected, changed = apply_local_transactions(records, events, 'perm')
        write_rows(track_output / f'{seq}.txt', records_to_rows(modified))
        execution.append({
            'seq': seq,
            'requested': len(events),
            'accepted': len(accepted),
            'rejected': len(rejected),
            'changed_rows': int(changed),
            'selected_merge_links': int(selected_merge_links),
            'accepted_events': accepted,
            'rejected_events': rejected,
        })

    command = [
        sys.executable,
        'scripts/eval_motstyle_trackeval.py',
        '--benchmark-name', 'MOT20',
        '--split-to-eval', 'train',
        '--gt-root', 'datasets/MOT20/train',
        '--results-dir', str(track_output),
        '--tracker-name', args.tracker_name,
        '--work-dir', str(output / 'eval_work'),
        '--seqs', *SEQS,
    ]
    process = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (output / 'eval.log').write_text(process.stdout)
    detail = output / 'eval_work' / 'eval' / args.tracker_name / 'pedestrian_detailed.csv'
    if not detail.exists():
        raise RuntimeError(f'TrackEval failed with return code {process.returncode}')
    evaluation = parse_eval(detail)

    deltas = {}
    for seq in SEQS + ['COMBINED']:
        deltas[seq] = {
            'delta_HOTA': evaluation[seq]['HOTA'] - baseline[seq]['HOTA'],
            'delta_DetA': evaluation[seq]['DetA'] - baseline[seq]['DetA'],
            'delta_AssA': evaluation[seq]['AssA'] - baseline[seq]['AssA'],
            'delta_IDF1': evaluation[seq]['IDF1'] - baseline[seq]['IDF1'],
            'delta_IDSW': evaluation[seq]['IDSW'] - baseline[seq]['IDSW'],
        }
    summary = {
        'protocol': {
            'scope': 'Actual four-sequence TrackEval of nested risk-gated strict FITR Top-20 pilot',
            'mode': 'perm',
            'selection_source': str(args.selected_transactions),
            'baseline': str(args.baseline_summary),
        },
        'execution': execution,
        'eval_returncode': process.returncode,
        'eval': evaluation,
        'delta_vs_baseline': deltas,
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
