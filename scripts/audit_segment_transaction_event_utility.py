from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
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


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def eval_one(job):
    job_dir, name = job
    cmd = [
        sys.executable, 'scripts/eval_motstyle_trackeval.py',
        '--benchmark-name', 'MOT20', '--split-to-eval', 'train',
        '--gt-root', 'datasets/MOT20/train',
        '--results-dir', str(job_dir / 'track_results'),
        '--tracker-name', name,
        '--work-dir', str(job_dir / 'eval_work'),
        '--seqs', 'MOT20-02',
    ]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (job_dir / 'eval.log').write_text(proc.stdout)
    detail = job_dir / 'eval_work' / 'eval' / name / 'pedestrian_detailed.csv'
    result = {'returncode': proc.returncode}
    if detail.exists():
        rows = list(csv.DictReader(detail.open()))
        r = next((x for x in rows if x['seq'] == 'MOT20-02'), None)
        if r:
            result.update({
                'HOTA': float(r['HOTA___AUC']) * 100,
                'DetA': float(r['DetA___AUC']) * 100,
                'AssA': float(r['AssA___AUC']) * 100,
                'IDF1': float(r['IDF1']) * 100 if float(r['IDF1']) < 2 else float(r['IDF1']),
                'IDSW': int(float(r['IDSW'])),
            })
    return name, result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source-root', required=True)
    ap.add_argument('--merge-links', required=True)
    ap.add_argument('--aggressive-events', required=True)
    ap.add_argument('--ranked-events', required=True)
    ap.add_argument('--best-summary', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--top-k', type=int, default=25)
    ap.add_argument('--rank-start', type=int, default=1)
    ap.add_argument('--modes', nargs='*', default=['h30','h60','perm'])
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    links = pd.read_csv(args.merge_links)
    aggressive = pd.read_csv(args.aggressive_events)
    all_ranked = pd.read_csv(args.ranked_events).head(args.top_k).reset_index(drop=True)
    ranked = all_ranked.iloc[args.rank_start - 1:args.top_k].to_dict('records')
    baseline_summary = json.loads(Path(args.best_summary).read_text())['eval']['MOT20-02']
    baseline, _ = reconstruct_best_with_provenance(
        Path(args.source_root) / 'MOT20-02.txt',
        links[links.seq == 'MOT20-02'], aggressive[aggressive.seq == 'MOT20-02'],
    )

    jobs = []; meta = {}; sha_to_name = {}; duplicate_of = {}
    for rank, event in enumerate(ranked, args.rank_start):
        for mode in args.modes:
            name = f'e{rank:02d}_{int(event["u"])}_{int(event["v"])}_f{int(event["boundary_frame"])}_{mode}'
            job_dir = out / name
            modified, accepted, rejected, changed = apply_local_transactions(baseline, [event], mode)
            row = {
                'name': name, 'rank': rank, 'mode': mode,
                'u': int(event['u']), 'v': int(event['v']),
                'boundary_frame': int(event['boundary_frame']),
                'score_reciprocal': float(event.get('oof_pair_reciprocal_ensemble', float('nan'))),
                'score_related': float(event.get('oof_pair_related_hgb', float('nan'))),
                'pair_swap_margin': float(event.get('pair_swap_margin', float('nan'))),
                'label_pair_class': str(event.get('label_pair_class', '')),
                'cluster_size': int(event.get('cluster_size', 1)),
                'accepted': int(bool(accepted)), 'rejected': int(bool(rejected)),
                'changed_rows': int(changed),
                'reject_reason': str(rejected[0]['reason']) if rejected else '',
                'start_frame': int(accepted[0]['start_frame']) if accepted else int(event['boundary_frame']),
                'end_frame': int(accepted[0]['end_frame']) if accepted else None,
            }
            if not accepted:
                row.update({
                    'HOTA': baseline_summary['HOTA'], 'DetA': baseline_summary['DetA'],
                    'AssA': baseline_summary['AssA'], 'IDF1': baseline_summary['IDF1'],
                    'IDSW': baseline_summary['IDSW'], 'eval_returncode': None,
                })
                meta[name] = row
                continue
            track = job_dir / 'track_results' / 'MOT20-02.txt'
            write_rows(track, records_to_rows(modified))
            h = file_sha(track); row['track_sha256'] = h
            meta[name] = row
            if h in sha_to_name:
                duplicate_of[name] = sha_to_name[h]
            else:
                sha_to_name[h] = name; jobs.append((job_dir, name))

    print(json.dumps({'jobs_total': len(meta), 'accepted_unique_eval': len(jobs),
                      'duplicates': len(duplicate_of), 'rejected_without_eval': sum(x['rejected'] for x in meta.values())}, indent=2), flush=True)
    results = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(eval_one, job): job[1] for job in jobs}
        done = 0
        for fut in as_completed(futures):
            name, result = fut.result(); results[name] = result; done += 1
            print(json.dumps({'done': done, 'total': len(jobs), 'name': name,
                              'HOTA': result.get('HOTA'), 'AssA': result.get('AssA'),
                              'IDSW': result.get('IDSW')}, indent=2), flush=True)

    for name, original in duplicate_of.items():
        results[name] = dict(results[original]); results[name]['duplicate_of'] = original
    rows = []
    for name, row in meta.items():
        if row['accepted']:
            r = results.get(name, {'returncode': -999})
            row.update({
                'HOTA': r.get('HOTA'), 'DetA': r.get('DetA'), 'AssA': r.get('AssA'),
                'IDF1': r.get('IDF1'), 'IDSW': r.get('IDSW'),
                'eval_returncode': r.get('returncode'), 'duplicate_of': r.get('duplicate_of',''),
            })
        row['delta_HOTA'] = row['HOTA'] - baseline_summary['HOTA'] if row.get('HOTA') is not None else None
        row['delta_AssA'] = row['AssA'] - baseline_summary['AssA'] if row.get('AssA') is not None else None
        row['delta_IDF1'] = row['IDF1'] - baseline_summary['IDF1'] if row.get('IDF1') is not None else None
        row['delta_IDSW'] = row['IDSW'] - baseline_summary['IDSW'] if row.get('IDSW') is not None else None
        rows.append(row)
    df = pd.DataFrame(rows).sort_values(['delta_HOTA','rank'], ascending=[False,True])
    df.to_csv(out / 'event_utility.csv', index=False)
    summary = {
        'protocol': {
            'scope': 'MOT20-02 diagnostic counterfactual oracle audit',
            'rank_start': args.rank_start, 'top_k': args.top_k, 'modes': args.modes,
            'baseline': baseline_summary,
            'selection': 'canonical reciprocal pair rank; each event evaluated independently',
        },
        'counts': {
            'rows': len(df), 'accepted': int(df.accepted.sum()), 'rejected': int(df.rejected.sum()),
            'positive_hota': int((df.delta_HOTA > 0).sum()),
            'positive_assa': int((df.delta_AssA > 0).sum()),
            'reduced_idsw': int((df.delta_IDSW < 0).sum()),
        },
        'by_mode': df.groupby('mode').agg(
            evaluated=('accepted','sum'), positive_hota=('delta_HOTA',lambda x:int((x>0).sum())),
            mean_delta_hota=('delta_HOTA','mean'), max_delta_hota=('delta_HOTA','max'),
            mean_delta_assa=('delta_AssA','mean'), min_delta_idsw=('delta_IDSW','min'),
        ).reset_index().to_dict('records'),
        'top_positive': df[df.delta_HOTA > 0].head(20).to_dict('records'),
        'worst': df.tail(20).to_dict('records'),
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)

if __name__ == '__main__':
    main()
