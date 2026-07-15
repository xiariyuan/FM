from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from eval_canonical_segment_transaction_replay import apply_local_transactions, reconstruct_best_with_provenance, records_to_rows
from eval_assa_swap_merge_fusion import write_rows


def evaluate(pdir: Path, name: str):
    cmd = [
        sys.executable, 'scripts/eval_motstyle_trackeval.py',
        '--benchmark-name', 'MOT20', '--split-to-eval', 'train',
        '--gt-root', 'datasets/MOT20/train',
        '--results-dir', str(pdir / 'track_results'),
        '--tracker-name', name, '--work-dir', str(pdir / 'eval_work'),
        '--seqs', 'MOT20-02',
    ]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (pdir / 'eval.log').write_text(proc.stdout)
    detail = pdir / 'eval_work' / 'eval' / name / 'pedestrian_detailed.csv'
    if not detail.exists():
        return {'returncode': proc.returncode}
    row = next(r for r in csv.DictReader(detail.open()) if r['seq'] == 'MOT20-02')
    return {
        'returncode': proc.returncode,
        'HOTA': float(row['HOTA___AUC']) * 100,
        'DetA': float(row['DetA___AUC']) * 100,
        'AssA': float(row['AssA___AUC']) * 100,
        'IDF1': float(row['IDF1']) * 100 if float(row['IDF1']) < 2 else float(row['IDF1']),
        'IDSW': int(float(row['IDSW'])),
    }


def policy_mask(df: pd.DataFrame, name: str):
    consensus = (df['oof_utility_et'] > 0) & (df['oof_utility_hgb'] > 0)
    if name == 'consensus_pos':
        return consensus
    if name == 'consensus_p60':
        return consensus & (df['oof_positive_prob_et'] >= 0.60)
    if name == 'consensus_p65':
        return consensus & (df['oof_positive_prob_et'] >= 0.65)
    if name == 'survival_p60':
        return consensus & (df['oof_positive_prob_et'] >= 0.60) & (df['swap_margin_positive_horizons'] >= 3)
    raise ValueError(name)


def conflict_free(events: pd.DataFrame):
    selected = []
    used_tracks = set()
    rejected = []
    for row in events.to_dict('records'):
        tracks = {int(row['u']), int(row['v'])}
        if tracks & used_tracks:
            rejected.append({**row, 'selector_reject_reason': 'shared_track_conflict'})
            continue
        selected.append(row)
        used_tracks.update(tracks)
    return pd.DataFrame(selected), pd.DataFrame(rejected)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source-root', required=True)
    ap.add_argument('--merge-links', required=True)
    ap.add_argument('--aggressive-events', required=True)
    ap.add_argument('--scores', required=True)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(args.scores).sort_values('oof_transaction_rank', ascending=False)
    links = pd.read_csv(args.merge_links)
    aggressive = pd.read_csv(args.aggressive_events)
    baseline, provenance = reconstruct_best_with_provenance(
        Path(args.source_root) / 'MOT20-02.txt',
        links[links.seq == 'MOT20-02'], aggressive[aggressive.seq == 'MOT20-02'],
    )
    policies = ['consensus_pos', 'consensus_p60', 'consensus_p65', 'survival_p60']
    summary = []
    for policy in policies:
        candidate = scores[policy_mask(scores, policy)].copy()
        for mode in ['raw', 'conflict_free']:
            if mode == 'raw':
                selected = candidate.copy(); selector_rejected = pd.DataFrame()
            else:
                selected, selector_rejected = conflict_free(candidate)
            name = f'{policy}_{mode}'
            pdir = out / name
            modified, accepted, rejected, changed = apply_local_transactions(
                baseline, selected.to_dict('records'), 'perm'
            )
            write_rows(pdir / 'track_results' / 'MOT20-02.txt', records_to_rows(modified))
            pd.DataFrame(selected).to_csv(pdir / 'selected.csv', index=False)
            if not selector_rejected.empty:
                selector_rejected.to_csv(pdir / 'selector_rejected.csv', index=False)
            pd.DataFrame(accepted).to_csv(pdir / 'accepted.csv', index=False)
            if rejected:
                pd.DataFrame(rejected).to_csv(pdir / 'executor_rejected.csv', index=False)
            metrics = evaluate(pdir, name)
            row = {
                'policy': policy, 'set_mode': mode,
                'candidates': len(candidate), 'selected': len(selected),
                'selector_rejected': len(selector_rejected),
                'executor_accepted': len(accepted), 'executor_rejected': len(rejected),
                'changed_rows': changed, **metrics,
            }
            summary.append(row)
            print(json.dumps(row, indent=2), flush=True)
    pd.DataFrame(summary).to_csv(out / 'summary.csv', index=False)
    (out / 'protocol.json').write_text(json.dumps({
        'policies': {
            'consensus_pos': 'ET utility > 0 and HGB utility > 0',
            'consensus_p60': 'consensus_pos and ET positive probability >= 0.60',
            'consensus_p65': 'consensus_pos and ET positive probability >= 0.65',
            'survival_p60': 'consensus_p60 and >=3 positive future swap-margin horizons',
        },
        'set_modes': {
            'raw': 'all policy events in fixed OOF rank order',
            'conflict_free': 'greedy fixed OOF rank order, rejecting any event sharing either raw track with a prior selected event',
        },
        'selection_uses_counterfactual_labels': False,
        'scope': 'MOT20-02 diagnostic; all configurations retained',
        'baseline_provenance': provenance,
    }, indent=2) + '\n')

if __name__ == '__main__':
    main()
