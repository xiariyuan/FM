from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from eval_canonical_segment_transaction_replay import reconstruct_best_with_provenance, records_to_rows
from eval_directional_identity_handoff import (
    apply_directional_handoff,
    build_directional_sequence_index,
)
from eval_assa_swap_merge_fusion import write_rows


def parse_mapping(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        seq, path = value.split('=', 1)
        result[seq] = Path(path)
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def parse_result(job_dir: Path, name: str, seq: str) -> dict | None:
    detail = job_dir / 'eval_work' / 'eval' / name / 'pedestrian_detailed.csv'
    for _ in range(20):
        if detail.exists():
            break
        candidates = list((job_dir / 'eval_work' / 'eval').glob('**/pedestrian_detailed.csv'))
        if candidates:
            detail = candidates[0]
            break
        time.sleep(0.25)
    if not detail.exists():
        return None
    rows = list(csv.DictReader(detail.open()))
    row = next((item for item in rows if item['seq'] == seq), None)
    if row is None:
        return None
    return {
        'returncode': 0,
        'HOTA': float(row['HOTA___AUC']) * 100,
        'DetA': float(row['DetA___AUC']) * 100,
        'AssA': float(row['AssA___AUC']) * 100,
        'IDF1': float(row['IDF1']) * 100 if float(row['IDF1']) < 2 else float(row['IDF1']),
        'IDSW': int(float(row['IDSW'])),
        'resumed': True,
    }


def evaluate(job: tuple[Path, str, str]) -> tuple[str, dict]:
    job_dir, name, seq = job
    existing = parse_result(job_dir, name, seq)
    if existing is not None:
        return name, existing
    command = [
        sys.executable,
        'scripts/eval_motstyle_trackeval.py',
        '--benchmark-name', 'MOT20',
        '--split-to-eval', 'train',
        '--gt-root', 'datasets/MOT20/train',
        '--results-dir', str(job_dir / 'track_results'),
        '--tracker-name', name,
        '--work-dir', str(job_dir / 'eval_work'),
        '--seqs', seq,
    ]
    process = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (job_dir / 'eval.log').write_text(process.stdout)
    result = parse_result(job_dir, name, seq) or {'returncode': process.returncode}
    result['returncode'] = process.returncode
    result['resumed'] = False
    return name, result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--events', action='append', required=True, help='Repeat SEQ=PATH')
    ap.add_argument('--executability', required=True)
    ap.add_argument('--source-root', required=True)
    ap.add_argument('--merge-links', required=True)
    ap.add_argument('--aggressive-events', required=True)
    ap.add_argument('--baseline-summary', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument(
        '--scope',
        default='Directional transaction utility labels on frozen ranks 21-100 training split only.',
    )
    ap.add_argument('--locked-test-events-or-labels-read', action='store_true')
    args = ap.parse_args()

    event_paths = parse_mapping(args.events)
    executable = pd.read_csv(args.executability)
    links = pd.read_csv(args.merge_links)
    aggressive = pd.read_csv(args.aggressive_events)
    baseline_summary = json.loads(Path(args.baseline_summary).read_text())['eval']
    output = Path(args.out_dir); output.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, dict] = {}
    jobs: list[tuple[Path, str, str]] = []
    digest_to_name: dict[str, str] = {}
    duplicate_of: dict[str, str] = {}

    for seq, event_path in sorted(event_paths.items()):
        events = pd.read_csv(event_path)
        baseline, selected_merge_links = reconstruct_best_with_provenance(
            Path(args.source_root) / f'{seq}.txt',
            links[links.seq == seq],
            aggressive[aggressive.seq == seq],
        )
        sequence_index = build_directional_sequence_index(baseline)
        event_lookup = {
            (int(row.canonical_rank), int(row.u), int(row.v), int(row.boundary_frame)): row._asdict()
            for row in events.itertuples(index=False)
        }
        seq_exec = executable[executable.seq == seq]
        for exec_row in seq_exec.to_dict('records'):
            key = (
                int(exec_row['canonical_rank']), int(exec_row['u']),
                int(exec_row['v']), int(exec_row['boundary_frame']),
            )
            event = event_lookup[key]
            transaction_type = str(exec_row['transaction_type'])
            name = (
                f'{seq.lower().replace("-", "_")}_r{int(event["canonical_rank"]):04d}_'
                f'{int(event["u"])}_{int(event["v"])}_f{int(event["boundary_frame"])}_{transaction_type}'
            )
            base = baseline_summary[seq]
            row = {
                'name': name,
                'seq': seq,
                'canonical_rank': int(event['canonical_rank']),
                'u': int(event['u']),
                'v': int(event['v']),
                'boundary_frame': int(event['boundary_frame']),
                'transaction_type': transaction_type,
                'outer_clean_transaction_score': float(event['outer_clean_transaction_score']),
                'outer_clean_reciprocal_score': float(event['outer_clean_reciprocal_raw_plus_percentile_ensemble']),
                'outer_clean_related_score': float(event['outer_clean_related_raw_plus_percentile_ensemble']),
                'heldout_unary_score': float(event['heldout_unary_score']),
                'cluster_size': int(event.get('cluster_size', 1)),
                'selected_merge_links': int(selected_merge_links),
                'accepted': int(exec_row['accepted']),
                'rejected': int(not int(exec_row['accepted'])),
                'changed_rows': int(exec_row['changed_rows']),
                'effective_start_frame': exec_row.get('effective_start_frame'),
                'reject_reason': str(exec_row.get('reject_reason', '')),
            }
            if not row['accepted']:
                row.update({
                    'HOTA': base['HOTA'], 'DetA': base['DetA'], 'AssA': base['AssA'],
                    'IDF1': base['IDF1'], 'IDSW': base['IDSW'], 'eval_returncode': None,
                })
                metadata[name] = row
                continue

            job_dir = output / seq / name
            track_path = job_dir / 'track_results' / f'{seq}.txt'
            if track_path.exists():
                digest = sha256(track_path)
            else:
                modified, accepted, rejected, changed = apply_directional_handoff(
                    baseline, event, transaction_type, sequence_index=sequence_index,
                )
                if not accepted:
                    raise RuntimeError(f'executability mismatch for {name}: {rejected}')
                if changed != int(exec_row['changed_rows']):
                    raise RuntimeError(f'changed-row mismatch for {name}: {changed} vs {exec_row["changed_rows"]}')
                write_rows(track_path, records_to_rows(modified))
                digest = sha256(track_path)
            row['track_sha256'] = digest
            metadata[name] = row
            if digest in digest_to_name:
                duplicate_of[name] = digest_to_name[digest]
            else:
                digest_to_name[digest] = name
                jobs.append((job_dir, name, seq))

    print(json.dumps({
        'rows': len(metadata),
        'accepted_unique_jobs': len(jobs),
        'duplicates': len(duplicate_of),
        'rejected_without_eval': sum(row['rejected'] for row in metadata.values()),
        'workers': args.workers,
    }, indent=2), flush=True)

    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(evaluate, job): job[1] for job in jobs}
        done = 0
        for future in as_completed(futures):
            name, result = future.result()
            results[name] = result
            done += 1
            print(json.dumps({
                'done': done, 'total': len(jobs), 'name': name,
                'HOTA': result.get('HOTA'), 'AssA': result.get('AssA'),
                'IDSW': result.get('IDSW'), 'returncode': result.get('returncode'),
            }), flush=True)
    for name, original in duplicate_of.items():
        results[name] = dict(results[original])
        results[name]['duplicate_of'] = original

    rows = []
    for name, row in metadata.items():
        if row['accepted']:
            result = results.get(name, {'returncode': -999})
            row.update({
                'HOTA': result.get('HOTA'), 'DetA': result.get('DetA'),
                'AssA': result.get('AssA'), 'IDF1': result.get('IDF1'),
                'IDSW': result.get('IDSW'), 'eval_returncode': result.get('returncode'),
                'duplicate_of': result.get('duplicate_of', ''), 'resumed': result.get('resumed', False),
            })
        base = baseline_summary[row['seq']]
        for metric in ['HOTA', 'DetA', 'AssA', 'IDF1']:
            row[f'delta_{metric}'] = row[metric] - base[metric] if row.get(metric) is not None else None
        row['delta_IDSW'] = row['IDSW'] - base['IDSW'] if row.get('IDSW') is not None else None
        rows.append(row)

    frame = pd.DataFrame(rows).sort_values(['seq', 'canonical_rank', 'transaction_type'])
    frame.to_csv(output / 'directional_utility.csv', index=False)
    by_sequence_type = frame.groupby(['seq', 'transaction_type']).agg(
        events=('canonical_rank', 'size'),
        accepted=('accepted', 'sum'),
        positive=('delta_HOTA', lambda values: int((values > 0).sum())),
        negative=('delta_HOTA', lambda values: int((values < 0).sum())),
        zero=('delta_HOTA', lambda values: int((values == 0).sum())),
        mean_delta_hota=('delta_HOTA', 'mean'),
        best_delta_hota=('delta_HOTA', 'max'),
        worst_delta_hota=('delta_HOTA', 'min'),
    ).reset_index()
    summary = {
        'protocol': {
            'scope': args.scope,
            'locked_test_events_or_labels_read': bool(args.locked_test_events_or_labels_read),
            'types': ['u_to_v', 'v_to_u'],
            'handoff_semantics': 'Receiver inherits donor identity only after donor first disappears; donor reappearance or duplicate ID rejects the transaction.',
        },
        'counts': {
            'rows': len(frame), 'accepted': int(frame.accepted.sum()),
            'positive': int((frame.delta_HOTA > 0).sum()),
            'negative': int((frame.delta_HOTA < 0).sum()),
            'zero': int((frame.delta_HOTA == 0).sum()),
        },
        'by_sequence_type': by_sequence_type.to_dict('records'),
        'top_positive': frame.sort_values('delta_HOTA', ascending=False).head(25).to_dict('records'),
        'worst': frame.sort_values('delta_HOTA').head(25).to_dict('records'),
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
