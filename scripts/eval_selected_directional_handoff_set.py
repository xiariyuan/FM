from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def parse_metrics(path: Path) -> dict[str, dict]:
    rows = list(csv.DictReader(path.open()))
    result = {}
    for row in rows:
        seq = row['seq']
        result[seq] = {
            'HOTA': float(row['HOTA___AUC']) * 100,
            'DetA': float(row['DetA___AUC']) * 100,
            'AssA': float(row['AssA___AUC']) * 100,
            'IDF1': float(row['IDF1']) * 100 if float(row['IDF1']) < 2 else float(row['IDF1']),
            'IDSW': int(float(row['IDSW'])),
        }
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--individual-utility', required=True)
    ap.add_argument('--individual-audit-root', required=True)
    ap.add_argument('--baseline-summary', required=True)
    ap.add_argument('--selection-manifest', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--tracker-name', default='locked_loso_directional_selected_set')
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results_dir = out / 'track_results'
    results_dir.mkdir(parents=True, exist_ok=True)

    individual = pd.read_csv(args.individual_utility)
    if individual.seq.duplicated().any():
        raise RuntimeError('combined replay expects at most one selected directional handoff per sequence')
    if not len(individual):
        raise RuntimeError('no selected directional handoffs to combine')

    copied = []
    audit_root = Path(args.individual_audit_root)
    for row in individual.to_dict('records'):
        seq = str(row['seq'])
        name = str(row['name'])
        source = audit_root / seq / name / 'track_results' / f'{seq}.txt'
        destination = results_dir / f'{seq}.txt'
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source, destination)
        copied.append({
            'seq': seq,
            'name': name,
            'source': str(source),
            'track_sha256': sha256(destination),
        })

    sequences = sorted(individual.seq.unique())
    work_dir = out / 'eval_work'
    command = [
        sys.executable,
        'scripts/eval_motstyle_trackeval.py',
        '--benchmark-name', 'MOT20',
        '--split-to-eval', 'train',
        '--gt-root', 'datasets/MOT20/train',
        '--results-dir', str(results_dir),
        '--tracker-name', args.tracker_name,
        '--work-dir', str(work_dir),
        '--seqs', *sequences,
    ]
    process = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out / 'eval.log').write_text(process.stdout)
    if process.returncode != 0:
        raise RuntimeError(f'TrackEval failed with return code {process.returncode}')

    detail = work_dir / 'eval' / args.tracker_name / 'pedestrian_detailed.csv'
    if not detail.exists():
        candidates = list((work_dir / 'eval').glob('**/pedestrian_detailed.csv'))
        if len(candidates) != 1:
            raise FileNotFoundError(f'expected one pedestrian_detailed.csv, found {len(candidates)}')
        detail = candidates[0]
    metrics = parse_metrics(detail)
    baseline = json.loads(Path(args.baseline_summary).read_text())['eval']
    deltas = {}
    for seq, current in metrics.items():
        if seq not in baseline:
            continue
        deltas[seq] = {
            metric: current[metric] - baseline[seq][metric]
            for metric in ['HOTA', 'DetA', 'AssA', 'IDF1', 'IDSW']
        }
    simple_avg_hota = sum(metrics[seq]['HOTA'] for seq in sequences) / len(sequences)
    baseline_simple_avg = sum(baseline[seq]['HOTA'] for seq in sequences) / len(sequences)
    selection_manifest = json.loads(Path(args.selection_manifest).read_text())
    summary = {
        'protocol': {
            'scope': 'Combined TrackEval replay of the frozen LOSO-selected directional handoff set.',
            'selection_sha256': selection_manifest['selection_sha256'],
            'locked_rows_consumed': selection_manifest['locked_rows_consumed'],
            'remaining_locked_directional_rows_unread': selection_manifest[
                'remaining_locked_directional_rows_unread'
            ],
        },
        'copied_tracks': copied,
        'eval': metrics,
        'delta_vs_aggressive15_merge_then_swap': deltas,
        'simple_avg_HOTA': simple_avg_hota,
        'baseline_simple_avg_HOTA': baseline_simple_avg,
        'delta_simple_avg_HOTA': simple_avg_hota - baseline_simple_avg,
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
