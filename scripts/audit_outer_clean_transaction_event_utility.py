from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from eval_canonical_segment_transaction_replay import (
    apply_local_transactions,
    reconstruct_best_with_provenance,
    records_to_rows,
)
from eval_assa_swap_merge_fusion import write_rows


def parse_mapping(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if '=' not in value:
            raise ValueError(f'expected SEQ=PATH, got {value}')
        seq, path = value.split('=', 1)
        if seq in result:
            raise ValueError(f'duplicate sequence {seq}')
        result[seq] = Path(path)
    return result


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def parse_existing_result(job_dir: Path, name: str, seq: str) -> dict | None:
    detail = job_dir / 'eval_work' / 'eval' / name / 'pedestrian_detailed.csv'
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
    existing = parse_existing_result(job_dir, name, seq)
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
    result = parse_existing_result(job_dir, name, seq) or {'returncode': process.returncode}
    result['returncode'] = process.returncode
    result['resumed'] = False
    return name, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--ranked-events', action='append', required=True, help='Repeat SEQ=PATH')
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--merge-links', required=True)
    parser.add_argument('--aggressive-events', required=True)
    parser.add_argument('--baseline-summary', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--top-k', type=int, default=20)
    parser.add_argument('--rank-start', type=int, default=1)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()

    ranked_paths = parse_mapping(args.ranked_events)
    baseline_summary = json.loads(Path(args.baseline_summary).read_text())['eval']
    links = pd.read_csv(args.merge_links)
    aggressive = pd.read_csv(args.aggressive_events)
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[Path, str, str]] = []
    metadata: dict[str, dict] = {}
    duplicate_of: dict[str, str] = {}
    sha_to_name: dict[str, str] = {}

    for seq, ranked_path in sorted(ranked_paths.items()):
        events = pd.read_csv(ranked_path)
        events = events.iloc[args.rank_start - 1:args.top_k].copy()
        if seq not in baseline_summary:
            raise RuntimeError(f'baseline summary missing {seq}')
        baseline_records, selected_merge_links = reconstruct_best_with_provenance(
            Path(args.source_root) / f'{seq}.txt',
            links[links.seq == seq],
            aggressive[aggressive.seq == seq],
        )
        for event in events.to_dict('records'):
            rank = int(event['canonical_rank'])
            name = f'{seq.lower().replace("-", "_")}_r{rank:04d}_{int(event["u"])}_{int(event["v"])}_f{int(event["boundary_frame"])}_perm'
            job_dir = output / seq / name
            modified, accepted, rejected, changed = apply_local_transactions(
                baseline_records, [event], 'perm'
            )
            row = {
                'name': name,
                'seq': seq,
                'canonical_rank': rank,
                'u': int(event['u']),
                'v': int(event['v']),
                'boundary_frame': int(event['boundary_frame']),
                'outer_clean_transaction_score': float(event['outer_clean_transaction_score']),
                'outer_clean_reciprocal_score': float(event['outer_clean_reciprocal_raw_plus_percentile_ensemble']),
                'outer_clean_related_score': float(event['outer_clean_related_raw_plus_percentile_ensemble']),
                'heldout_unary_score': float(event['heldout_unary_score']),
                'cluster_size': int(event.get('cluster_size', 1)),
                'selected_merge_links': int(selected_merge_links),
                'mode': 'perm',
                'accepted': int(bool(accepted)),
                'rejected': int(bool(rejected)),
                'changed_rows': int(changed),
                'reject_reason': str(rejected[0]['reason']) if rejected else '',
                'start_frame': int(accepted[0]['start_frame']) if accepted else int(event['boundary_frame']),
                'end_frame': int(accepted[0]['end_frame']) if accepted else None,
                'label_reciprocal_swap_diagnostic': int(event.get('label_reciprocal_swap', 0)),
                'label_pair_related_diagnostic': int(event.get('label_pair_related', 0)),
            }
            if not accepted:
                base = baseline_summary[seq]
                row.update({
                    'HOTA': base['HOTA'],
                    'DetA': base['DetA'],
                    'AssA': base['AssA'],
                    'IDF1': base['IDF1'],
                    'IDSW': base['IDSW'],
                    'eval_returncode': None,
                })
                metadata[name] = row
                continue

            track_path = job_dir / 'track_results' / f'{seq}.txt'
            write_rows(track_path, records_to_rows(modified))
            digest = file_sha(track_path)
            row['track_sha256'] = digest
            metadata[name] = row
            if digest in sha_to_name:
                duplicate_of[name] = sha_to_name[digest]
            else:
                sha_to_name[digest] = name
                jobs.append((job_dir, name, seq))

    print(json.dumps({
        'events_total': len(metadata),
        'accepted_unique_evals': len(jobs),
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
                'done': done,
                'total': len(jobs),
                'name': name,
                'HOTA': result.get('HOTA'),
                'AssA': result.get('AssA'),
                'IDSW': result.get('IDSW'),
                'returncode': result.get('returncode'),
            }), flush=True)

    for name, original in duplicate_of.items():
        results[name] = dict(results[original])
        results[name]['duplicate_of'] = original

    rows = []
    for name, row in metadata.items():
        if row['accepted']:
            result = results.get(name, {'returncode': -999})
            row.update({
                'HOTA': result.get('HOTA'),
                'DetA': result.get('DetA'),
                'AssA': result.get('AssA'),
                'IDF1': result.get('IDF1'),
                'IDSW': result.get('IDSW'),
                'eval_returncode': result.get('returncode'),
                'duplicate_of': result.get('duplicate_of', ''),
                'resumed': result.get('resumed', False),
            })
        base = baseline_summary[row['seq']]
        row['delta_HOTA'] = row['HOTA'] - base['HOTA'] if row.get('HOTA') is not None else None
        row['delta_DetA'] = row['DetA'] - base['DetA'] if row.get('DetA') is not None else None
        row['delta_AssA'] = row['AssA'] - base['AssA'] if row.get('AssA') is not None else None
        row['delta_IDF1'] = row['IDF1'] - base['IDF1'] if row.get('IDF1') is not None else None
        row['delta_IDSW'] = row['IDSW'] - base['IDSW'] if row.get('IDSW') is not None else None
        rows.append(row)

    frame = pd.DataFrame(rows).sort_values(['seq', 'canonical_rank'])
    frame.to_csv(output / 'event_utility.csv', index=False)
    by_sequence = []
    for seq, group in frame.groupby('seq'):
        by_sequence.append({
            'seq': seq,
            'events': len(group),
            'accepted': int(group.accepted.sum()),
            'rejected': int(group.rejected.sum()),
            'positive_hota': int((group.delta_HOTA > 0).sum()),
            'negative_hota': int((group.delta_HOTA < 0).sum()),
            'zero_hota': int((group.delta_HOTA == 0).sum()),
            'mean_delta_hota': float(group.delta_HOTA.mean()),
            'positive_utility_sum': float(group.loc[group.delta_HOTA > 0, 'delta_HOTA'].sum()),
            'best_delta_hota': float(group.delta_HOTA.max()),
            'worst_delta_hota': float(group.delta_HOTA.min()),
        })
    summary = {
        'protocol': {
            'scope': 'Four-sequence outer-clean candidate utility labeling; labels are used only after fixed candidate ranking',
            'selection': 'Pre-registered outer-clean transaction ranking and canonicalization from outer_clean_transaction_candidates/protocol.json',
            'mode': 'perm',
            'rank_start': args.rank_start,
            'top_k': args.top_k,
            'baseline': 'identity_transaction_fusion_eval_v2/aggressive15_merge_then_swap',
            'leakage_policy': 'TrackEval GT creates counterfactual utility labels. These labels must only train utility models for other held-out sequences and never alter candidate construction.',
        },
        'counts': {
            'events': len(frame),
            'accepted': int(frame.accepted.sum()),
            'rejected': int(frame.rejected.sum()),
            'positive_hota': int((frame.delta_HOTA > 0).sum()),
            'negative_hota': int((frame.delta_HOTA < 0).sum()),
        },
        'by_sequence': by_sequence,
        'top_positive': frame.sort_values('delta_HOTA', ascending=False).head(20).to_dict('records'),
        'worst': frame.sort_values('delta_HOTA', ascending=True).head(20).to_dict('records'),
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
