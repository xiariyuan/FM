from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq', required=True)
    ap.add_argument('--shard-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    shard_dir = Path(args.shard_dir)
    summaries = [json.loads(p.read_text()) for p in sorted(shard_dir.glob('shard_*_summary.json'))]
    if not summaries:
        raise RuntimeError('no pair-bank shard summaries found')
    summaries.sort(key=lambda x: int(x['proposal_index_start']))

    expected = 1
    for s in summaries:
        start = int(s['proposal_index_start'])
        end = int(s['proposal_index_end'])
        if start != expected:
            raise RuntimeError(f'non-contiguous proposal shards: expected {expected}, got {start}')
        expected = end + 1
    total = int(summaries[0]['proposals_total'])
    if expected - 1 != total:
        raise RuntimeError(f'incomplete proposal shards: covered {expected - 1}/{total}')

    frames = []
    for s in summaries:
        stem = f"shard_{int(s['proposal_index_start']):04d}_{int(s['proposal_index_end']):04d}"
        csv_path = shard_dir / f'{stem}_segment_pair_state_bank.csv'
        if not csv_path.exists():
            raise RuntimeError(f'missing {csv_path}')
        frames.append(pd.read_csv(csv_path))
    bank = pd.concat(frames, ignore_index=True)
    bank = bank.sort_values(['proposal_rank', 'partner_rank'], kind='mergesort').reset_index(drop=True)
    if bank.duplicated(['proposal_rank', 'track_a', 'track_b', 'partner_rank']).any():
        raise RuntimeError('duplicate pair rows after merge')

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    bank.to_csv(out / f'{args.seq}_segment_pair_state_bank.csv', index=False)
    proposals_with_partner = sum(int(s['proposals_with_partner']) for s in summaries)
    selected = sum(int(s['proposals_selected']) for s in summaries)
    summary = {
        'seq': args.seq,
        'proposal_score_col': summaries[0]['proposal_score_col'],
        'proposal_limit': int(summaries[0]['proposal_limit']),
        'proposals_selected': selected,
        'proposals_with_partner': proposals_with_partner,
        'proposal_partner_coverage': proposals_with_partner / max(1, selected),
        'top_k_partners': int(summaries[0]['top_k_partners']),
        'pair_rows': len(bank),
        'valid_pair_reid_rows': int(bank.valid_pair_reid.sum()) if len(bank) else 0,
        'pair_reid_coverage': float(bank.valid_pair_reid.mean()) if len(bank) else 0.0,
        'class_counts': {str(k): int(v) for k, v in bank.label_pair_class.value_counts().items()} if len(bank) else {},
        'reciprocal_pairs': int(bank.label_reciprocal_swap.sum()) if len(bank) else 0,
        'related_pairs': int(bank.label_pair_related.sum()) if len(bank) else 0,
        'unique_reciprocal_events': int(bank[bank.label_reciprocal_swap == 1][['track_a','boundary_frame']].drop_duplicates().shape[0]) if len(bank) else 0,
        'unique_related_events': int(bank[bank.label_pair_related == 1][['track_a','boundary_frame']].drop_duplicates().shape[0]) if len(bank) else 0,
        'shards': len(summaries),
        'leakage_policy': (
            f'Candidate proposals for {args.seq} use unary scores trained on the other sequences only. '
            'Held-out GT-derived columns are emitted only for diagnostic labels and must be excluded from models.'
            if str(summaries[0]['proposal_score_col']).startswith('loso_unary_') else
            f'Candidate proposals for {args.seq} use diagnostic same-sequence grouped-OOF unary scores. '
            'GT-derived columns are emitted only for diagnostic labels and must be excluded from models.'
        ),
    }
    (out / f'{args.seq}_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
