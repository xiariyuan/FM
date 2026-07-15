from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from eval_canonical_segment_transaction_replay import (
    apply_local_transactions,
    canonical_events,
    reconstruct_best_with_provenance,
)


def parse_mapping(values: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for value in values:
        if '=' not in value:
            raise ValueError(f'expected SEQ=PATH, got {value}')
        seq, path = value.split('=', 1)
        if seq in out:
            raise ValueError(f'duplicate sequence {seq}')
        out[seq] = Path(path)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pair-bank', action='append', required=True, help='Repeat SEQ=PATH')
    ap.add_argument('--outer-clean-scores', required=True)
    ap.add_argument('--source-root', required=True)
    ap.add_argument('--merge-links', required=True)
    ap.add_argument('--aggressive-events', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--cluster-radius', type=int, default=3)
    ap.add_argument('--spacing', type=int, default=30)
    ap.add_argument('--audit-top-k', type=int, default=30)
    args = ap.parse_args()

    bank_paths = parse_mapping(args.pair_bank)
    scores = pd.read_csv(args.outer_clean_scores)
    links = pd.read_csv(args.merge_links)
    aggressive = pd.read_csv(args.aggressive_events)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_selected = []
    summaries = []
    score_keys = ['seq', 'track_a', 'track_b', 'boundary_frame', 'proposal_rank', 'partner_rank']
    score_cols = [
        'outer_clean_reciprocal_raw_plus_percentile_ensemble',
        'outer_clean_related_raw_plus_percentile_ensemble',
        'outer_clean_reciprocal_raw_ensemble',
        'outer_clean_related_raw_ensemble',
        'heldout_unary_score',
    ]

    for seq, bank_path in sorted(bank_paths.items()):
        bank = pd.read_csv(bank_path)
        score_part = scores[scores['seq'] == seq][score_keys + score_cols].copy()
        merged = bank.merge(score_part, on=score_keys, how='inner', validate='one_to_one')
        if len(merged) != len(bank):
            raise RuntimeError(f'{seq}: joined {len(merged)}/{len(bank)} rows')
        merged['outer_clean_transaction_score'] = merged[[
            'outer_clean_reciprocal_raw_plus_percentile_ensemble',
            'outer_clean_related_raw_plus_percentile_ensemble',
        ]].max(axis=1)
        clustered, selected = canonical_events(
            merged,
            'outer_clean_transaction_score',
            args.cluster_radius,
            args.spacing,
        )
        selected_df = pd.DataFrame(selected)
        selected_df.insert(0, 'canonical_rank', range(1, len(selected_df) + 1))
        selected_df['seq'] = seq

        baseline, selected_merge_links = reconstruct_best_with_provenance(
            Path(args.source_root) / f'{seq}.txt',
            links[links.seq == seq],
            aggressive[aggressive.seq == seq],
        )
        audit_rows = []
        for event in selected_df.head(args.audit_top_k).to_dict('records'):
            _, accepted, rejected, changed = apply_local_transactions(baseline, [event], 'perm')
            audit_rows.append({
                'seq': seq,
                'canonical_rank': int(event['canonical_rank']),
                'u': int(event['u']),
                'v': int(event['v']),
                'boundary_frame': int(event['boundary_frame']),
                'outer_clean_transaction_score': float(event['outer_clean_transaction_score']),
                'reciprocal_score': float(event['outer_clean_reciprocal_raw_plus_percentile_ensemble']),
                'related_score': float(event['outer_clean_related_raw_plus_percentile_ensemble']),
                'accepted': int(bool(accepted)),
                'changed_rows': int(changed),
                'reject_reason': str(rejected[0]['reason']) if rejected else '',
            })
        audit = pd.DataFrame(audit_rows)
        selected_df.to_csv(out / f'{seq}_ranked_transactions.csv', index=False)
        pd.DataFrame(clustered).to_csv(out / f'{seq}_clustered_transactions.csv', index=False)
        audit.to_csv(out / f'{seq}_top{args.audit_top_k}_executability.csv', index=False)
        all_selected.append(selected_df)
        summary = {
            'seq': seq,
            'pair_rows': len(bank),
            'clustered_events': len(clustered),
            'canonical_events': len(selected_df),
            'audit_top_k': min(args.audit_top_k, len(selected_df)),
            'audit_accepted': int(audit.accepted.sum()) if len(audit) else 0,
            'audit_rejected': int((audit.accepted == 0).sum()) if len(audit) else 0,
            'selected_merge_links': int(selected_merge_links),
            'top20_reciprocal_labels_diagnostic': int(selected_df.head(20)['label_reciprocal_swap'].sum()) if len(selected_df) else 0,
            'top20_related_labels_diagnostic': int(selected_df.head(20)['label_pair_related'].sum()) if len(selected_df) else 0,
        }
        summaries.append(summary)
        print(json.dumps(summary, indent=2), flush=True)

    pd.concat(all_selected, ignore_index=True).to_csv(out / 'all_ranked_transactions.csv', index=False)
    protocol = {
        'scope': 'Outer-clean transaction candidate construction before utility labeling',
        'ranking': 'max(outer-clean reciprocal raw+percentile probability, outer-clean related raw+percentile probability)',
        'canonicalization': {
            'unordered_pair_cluster_radius': args.cluster_radius,
            'per_track_spacing': args.spacing,
        },
        'utility_labels_used_for_selection': False,
        'heldout_GT_policy': 'GT labels remain in source banks for diagnostic accounting only and are never used by ranking or canonicalization.',
        'summaries': summaries,
    }
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')


if __name__ == '__main__':
    main()
