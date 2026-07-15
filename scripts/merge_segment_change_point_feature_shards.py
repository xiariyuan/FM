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
        raise RuntimeError('no shard summaries found')
    summaries.sort(key=lambda x: int(x['track_index_start']))

    expected = 1
    seen_ids = set()
    for summary in summaries:
        start = int(summary['track_index_start'])
        end = int(summary['track_index_end'])
        if start != expected:
            raise RuntimeError(f'non-contiguous shard start: expected {expected}, got {start}')
        ids = set(map(int, summary['selected_track_ids']))
        if seen_ids & ids:
            raise RuntimeError('duplicate tracker IDs across shards')
        seen_ids.update(ids)
        expected = end + 1
    tracks_total = int(summaries[0]['tracks_total'])
    if expected - 1 != tracks_total:
        raise RuntimeError(f'incomplete shards: covered {expected - 1}/{tracks_total}')

    frames = []
    for summary in summaries:
        stem = f"shard_{int(summary['track_index_start']):04d}_{int(summary['track_index_end']):04d}"
        csv_path = shard_dir / f'{stem}_segment_change_features.csv'
        if not csv_path.exists():
            raise RuntimeError(f'missing {csv_path}')
        frames.append(pd.read_csv(csv_path))
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(['track_id', 'boundary_frame'], kind='mergesort').reset_index(drop=True)
    if df.duplicated(['track_id', 'boundary_frame']).any():
        raise RuntimeError('duplicate boundary rows after merge')

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f'{args.seq}_segment_change_features.csv', index=False)
    track_rows = sum(int(s['track_rows']) for s in summaries)
    matched_rows = sum(int(s['matched_reid_rows']) for s in summaries)
    merged = {
        'seq': args.seq,
        'track_rows': track_rows,
        'matched_reid_rows': matched_rows,
        'reid_match_coverage': matched_rows / max(1, track_rows),
        'low_iou_rows': sum(int(s['low_iou_rows']) for s in summaries),
        'zero_feature_rows': sum(int(s['zero_feature_rows']) for s in summaries),
        'feature_boundaries': len(df),
        'window': int(summaries[0]['window']),
        'min_iou': float(summaries[0]['min_iou']),
        'tracks_total': tracks_total,
        'shards': len(summaries),
        'persistent_switches': {
            str(p): sum(int(s['persistent_switches'][str(p)]) for s in summaries)
            for p in [1, 3, 5, 10]
        },
        'positive_boundaries': {
            str(p): int(df[f'label_switch_p{p}'].sum()) for p in [1, 3, 5, 10]
        },
    }
    (out / f'{args.seq}_summary.json').write_text(json.dumps(merged, indent=2) + '\n')
    print(json.dumps(merged, indent=2))


if __name__ == '__main__':
    main()
