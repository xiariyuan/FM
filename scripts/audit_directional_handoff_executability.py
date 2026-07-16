from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from eval_canonical_segment_transaction_replay import apply_local_transactions, reconstruct_best_with_provenance
from eval_directional_identity_handoff import build_directional_sequence_index, plan_directional_handoff


def parse_mapping(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        seq, path = value.split('=', 1)
        result[seq] = Path(path)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--events', action='append', required=True)
    ap.add_argument('--source-root', required=True)
    ap.add_argument('--merge-links', required=True)
    ap.add_argument('--aggressive-events', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument(
        '--scope',
        default='Directional handoff executability audit; no TrackEval metrics or GT labels are read.',
    )
    args = ap.parse_args()

    paths = parse_mapping(args.events)
    links = pd.read_csv(args.merge_links)
    aggressive = pd.read_csv(args.aggressive_events)
    rows = []
    for seq, path in sorted(paths.items()):
        baseline, _ = reconstruct_best_with_provenance(
            Path(args.source_root) / f'{seq}.txt',
            links[links.seq == seq],
            aggressive[aggressive.seq == seq],
        )
        events = pd.read_csv(path)
        sequence_index = build_directional_sequence_index(baseline)
        for event in events.to_dict('records'):
            for transaction_type in ['u_to_v', 'v_to_u']:
                plan = plan_directional_handoff(sequence_index, event, transaction_type)
                accepted = [plan['metadata']] if plan['accepted'] else []
                rejected = [{**plan['metadata'], 'reason': plan['reason']}] if not plan['accepted'] else []
                changed = int(plan['metadata'].get('changed_rows', 0))
                rows.append({
                    'seq': seq,
                    'canonical_rank': int(event['canonical_rank']),
                    'u': int(event['u']),
                    'v': int(event['v']),
                    'boundary_frame': int(event['boundary_frame']),
                    'transaction_type': transaction_type,
                    'accepted': int(bool(accepted)),
                    'changed_rows': int(changed),
                    'effective_start_frame': int(accepted[0].get('handoff_frame', accepted[0].get('start_frame'))) if accepted else None,
                    'reject_reason': str(rejected[0]['reason']) if rejected else '',
                })
    frame = pd.DataFrame(rows)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / 'executability.csv', index=False)
    summary = frame.groupby(['seq', 'transaction_type']).agg(
        events=('canonical_rank', 'size'),
        accepted=('accepted', 'sum'),
        mean_changed_rows=('changed_rows', 'mean'),
    ).reset_index()
    reject = frame[frame.accepted == 0].groupby(['transaction_type', 'reject_reason']).size().reset_index(name='count')
    report = {
        'scope': args.scope,
        'summary': summary.to_dict('records'),
        'reject_reasons': reject.to_dict('records'),
    }
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
