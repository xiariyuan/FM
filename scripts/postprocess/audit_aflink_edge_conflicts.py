#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict, Counter
from pathlib import Path


def ff(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def ii(x, default=0):
    try:
        return int(float(x))
    except Exception:
        return default


def read_edges(path: Path):
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            row = dict(r)
            row['seq'] = row['seq']
            row['track_a'] = ii(row['track_a'])
            row['track_b'] = ii(row['track_b'])
            row['same_gt'] = ii(row.get('same_gt', 0))
            row['score'] = ff(row.get('aflink_score', row.get('score', 0)))
            row['gap'] = ff(row.get('gap', 0))
            row['dominant_gt_a'] = ii(row.get('dominant_gt_a', -1), -1)
            row['dominant_gt_b'] = ii(row.get('dominant_gt_b', -1), -1)
            rows.append(row)
    return rows


def read_selected(path: Path):
    out = set()
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            seq = r['seq']
            a = ii(r['track_a'])
            b = ii(r['track_b'])
            out.add((seq, a, b))
            rows.append({'seq': seq, 'track_a': a, 'track_b': b, 'score': ff(r.get('score', 0)), 'same_gt': ii(r.get('same_gt', 0)), 'gap': ff(r.get('gap', 0))})
    return out, rows


def add_ranks(edges):
    by_src = defaultdict(list)
    by_dst = defaultdict(list)
    for e in edges:
        by_src[(e['seq'], e['track_a'])].append(e)
        by_dst[(e['seq'], e['track_b'])].append(e)
    for group in by_src.values():
        group.sort(key=lambda x: -x['score'])
        for i, e in enumerate(group, start=1):
            e['rank_as_successor'] = i
            e['src_top_score'] = group[0]['score']
            e['src_second_score'] = group[1]['score'] if len(group) > 1 else 0.0
            e['src_margin_top2'] = group[0]['score'] - (group[1]['score'] if len(group) > 1 else 0.0)
            e['src_num_candidates'] = len(group)
    for group in by_dst.values():
        group.sort(key=lambda x: -x['score'])
        for i, e in enumerate(group, start=1):
            e['rank_as_predecessor'] = i
            e['dst_top_score'] = group[0]['score']
            e['dst_second_score'] = group[1]['score'] if len(group) > 1 else 0.0
            e['dst_margin_top2'] = group[0]['score'] - (group[1]['score'] if len(group) > 1 else 0.0)
            e['dst_num_candidates'] = len(group)
    return by_src, by_dst


def write_csv(path: Path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys = []
        seen = set()
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
        fieldnames = keys
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def percentile(vals, p):
    if not vals:
        return None
    vals = sorted(vals)
    idx = int(round((len(vals) - 1) * p / 100.0))
    return float(vals[idx])


def stats(vals):
    if not vals:
        return {'n': 0}
    return {
        'n': len(vals),
        'mean': float(sum(vals) / len(vals)),
        'p50': percentile(vals, 50),
        'p75': percentile(vals, 75),
        'p90': percentile(vals, 90),
        'p95': percentile(vals, 95),
        'max': float(max(vals)),
        'min': float(min(vals)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--edges', required=True)
    ap.add_argument('--selected', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--near-score-gap', type=float, default=0.10)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    edges = read_edges(Path(args.edges))
    selected_set, selected_rows = read_selected(Path(args.selected))
    by_src, by_dst = add_ranks(edges)
    edge_map = {(e['seq'], e['track_a'], e['track_b']): e for e in edges}

    selected_edges = []
    for s in selected_rows:
        e = edge_map.get((s['seq'], s['track_a'], s['track_b']))
        if e is not None:
            e = dict(e)
            e['selected'] = 1
            selected_edges.append(e)

    false_selected = [e for e in selected_edges if e['same_gt'] == 0]
    true_selected = [e for e in selected_edges if e['same_gt'] == 1]
    true_edges = [e for e in edges if e['same_gt'] == 1]
    false_edges = [e for e in edges if e['same_gt'] == 0]

    false_alt_rows = []
    for e in false_selected:
        src_true = [x for x in by_src[(e['seq'], e['track_a'])] if x['same_gt'] == 1]
        dst_true = [x for x in by_dst[(e['seq'], e['track_b'])] if x['same_gt'] == 1]
        best_src = max(src_true, key=lambda x: x['score'], default=None)
        best_dst = max(dst_true, key=lambda x: x['score'], default=None)
        candidates = []
        if best_src is not None:
            candidates.append(('same_src', best_src))
        if best_dst is not None:
            candidates.append(('same_dst', best_dst))
        if candidates:
            alt_type, alt = max(candidates, key=lambda t: t[1]['score'])
            score_gap = e['score'] - alt['score']
            false_alt_rows.append({
                'seq': e['seq'],
                'false_a': e['track_a'],
                'false_b': e['track_b'],
                'false_score': e['score'],
                'false_gap': e.get('gap', 0),
                'false_src_rank': e.get('rank_as_successor'),
                'false_dst_rank': e.get('rank_as_predecessor'),
                'alt_type': alt_type,
                'true_a': alt['track_a'],
                'true_b': alt['track_b'],
                'true_score': alt['score'],
                'true_gap': alt.get('gap', 0),
                'true_src_rank': alt.get('rank_as_successor'),
                'true_dst_rank': alt.get('rank_as_predecessor'),
                'score_gap_false_minus_true': score_gap,
                'true_within_0_05': int(score_gap <= 0.05),
                'true_within_0_10': int(score_gap <= 0.10),
                'false_gt_a': e.get('dominant_gt_a', -1),
                'false_gt_b': e.get('dominant_gt_b', -1),
                'true_gt_a': alt.get('dominant_gt_a', -1),
                'true_gt_b': alt.get('dominant_gt_b', -1),
            })
        else:
            false_alt_rows.append({
                'seq': e['seq'],
                'false_a': e['track_a'],
                'false_b': e['track_b'],
                'false_score': e['score'],
                'false_gap': e.get('gap', 0),
                'false_src_rank': e.get('rank_as_successor'),
                'false_dst_rank': e.get('rank_as_predecessor'),
                'alt_type': 'none',
                'true_a': '',
                'true_b': '',
                'true_score': '',
                'true_gap': '',
                'true_src_rank': '',
                'true_dst_rank': '',
                'score_gap_false_minus_true': '',
                'true_within_0_05': 0,
                'true_within_0_10': 0,
                'false_gt_a': e.get('dominant_gt_a', -1),
                'false_gt_b': e.get('dominant_gt_b', -1),
                'true_gt_a': '',
                'true_gt_b': '',
            })

    true_rank_rows = []
    for e in true_edges:
        selected = int((e['seq'], e['track_a'], e['track_b']) in selected_set)
        src_top = by_src[(e['seq'], e['track_a'])][0]
        dst_top = by_dst[(e['seq'], e['track_b'])][0]
        true_rank_rows.append({
            'seq': e['seq'],
            'track_a': e['track_a'],
            'track_b': e['track_b'],
            'score': e['score'],
            'gap': e.get('gap', 0),
            'selected': selected,
            'rank_as_successor': e.get('rank_as_successor'),
            'rank_as_predecessor': e.get('rank_as_predecessor'),
            'src_top_same_gt': src_top['same_gt'],
            'src_top_score': src_top['score'],
            'src_top_b': src_top['track_b'],
            'src_score_gap_top_minus_true': src_top['score'] - e['score'],
            'dst_top_same_gt': dst_top['same_gt'],
            'dst_top_score': dst_top['score'],
            'dst_top_a': dst_top['track_a'],
            'dst_score_gap_top_minus_true': dst_top['score'] - e['score'],
        })

    blocked_true_rows = []
    selected_src = {(e['seq'], e['track_a']): e for e in selected_edges}
    selected_dst = {(e['seq'], e['track_b']): e for e in selected_edges}
    for e in true_edges:
        if (e['seq'], e['track_a'], e['track_b']) in selected_set:
            continue
        src_sel = selected_src.get((e['seq'], e['track_a']))
        dst_sel = selected_dst.get((e['seq'], e['track_b']))
        reason = []
        if src_sel is not None:
            reason.append('src_used_by_' + ('true' if src_sel['same_gt'] else 'false'))
        if dst_sel is not None:
            reason.append('dst_used_by_' + ('true' if dst_sel['same_gt'] else 'false'))
        if not reason:
            reason.append('below_or_not_selected_no_conflict')
        blocked_true_rows.append({
            'seq': e['seq'],
            'track_a': e['track_a'],
            'track_b': e['track_b'],
            'score': e['score'],
            'gap': e.get('gap', 0),
            'rank_as_successor': e.get('rank_as_successor'),
            'rank_as_predecessor': e.get('rank_as_predecessor'),
            'reason': '+'.join(reason),
            'src_selected_b': src_sel['track_b'] if src_sel else '',
            'src_selected_score': src_sel['score'] if src_sel else '',
            'src_selected_same_gt': src_sel['same_gt'] if src_sel else '',
            'dst_selected_a': dst_sel['track_a'] if dst_sel else '',
            'dst_selected_score': dst_sel['score'] if dst_sel else '',
            'dst_selected_same_gt': dst_sel['same_gt'] if dst_sel else '',
        })

    by_seq = {}
    for seq in sorted({e['seq'] for e in edges}):
        seq_edges = [e for e in edges if e['seq'] == seq]
        seq_sel = [e for e in selected_edges if e['seq'] == seq]
        seq_false_sel = [e for e in false_selected if e['seq'] == seq]
        seq_true = [e for e in true_edges if e['seq'] == seq]
        seq_true_sel = [e for e in true_selected if e['seq'] == seq]
        seq_alt = [r for r in false_alt_rows if r['seq'] == seq and r['alt_type'] != 'none']
        by_seq[seq] = {
            'edges': len(seq_edges),
            'true_edges': len(seq_true),
            'selected_links': len(seq_sel),
            'selected_true': len(seq_true_sel),
            'selected_false': len(seq_false_sel),
            'selected_precision': len(seq_true_sel) / len(seq_sel) if seq_sel else 0,
            'false_selected_with_true_alt': len(seq_alt),
            'false_selected_with_true_alt_rate': len(seq_alt) / len(seq_false_sel) if seq_false_sel else 0,
            'false_selected_true_alt_within_0_05': sum(1 for r in seq_alt if r.get('true_within_0_05') == 1),
            'false_selected_true_alt_within_0_10': sum(1 for r in seq_alt if r.get('true_within_0_10') == 1),
        }

    false_with_alt = [r for r in false_alt_rows if r.get('alt_type') != 'none']
    score_gaps = [float(r['score_gap_false_minus_true']) for r in false_with_alt if r.get('score_gap_false_minus_true') != '']
    true_rank_src = [int(r['rank_as_successor']) for r in true_rank_rows]
    true_rank_dst = [int(r['rank_as_predecessor']) for r in true_rank_rows]
    blocked_reason_counts = Counter(r['reason'] for r in blocked_true_rows)

    summary = {
        'edges': len(edges),
        'true_edges': len(true_edges),
        'false_edges': len(false_edges),
        'selected_links': len(selected_edges),
        'selected_true': len(true_selected),
        'selected_false': len(false_selected),
        'selected_precision': len(true_selected) / len(selected_edges) if selected_edges else 0,
        'false_selected_with_true_alternative': len(false_with_alt),
        'false_selected_with_true_alternative_rate': len(false_with_alt) / len(false_selected) if false_selected else 0,
        'false_selected_true_alt_within_0_05': sum(1 for r in false_with_alt if r.get('true_within_0_05') == 1),
        'false_selected_true_alt_within_0_10': sum(1 for r in false_with_alt if r.get('true_within_0_10') == 1),
        'false_minus_true_alt_score_gap_stats': stats(score_gaps),
        'true_edge_src_rank_stats': stats(true_rank_src),
        'true_edge_dst_rank_stats': stats(true_rank_dst),
        'blocked_true_reason_counts': dict(blocked_reason_counts),
        'by_seq': by_seq,
    }

    write_csv(out_dir / 'false_selected_nearest_true_alternative.csv', false_alt_rows)
    write_csv(out_dir / 'true_link_score_rank_distribution.csv', true_rank_rows)
    write_csv(out_dir / 'blocked_true_links.csv', blocked_true_rows)
    write_csv(out_dir / 'selected_links_enriched.csv', selected_edges)
    (out_dir / 'edge_conflict_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')

    md = [
        '# A26 Edge Conflict Audit',
        '',
        f"edges: {summary['edges']}",
        f"true_edges: {summary['true_edges']}",
        f"selected_links: {summary['selected_links']}",
        f"selected_true: {summary['selected_true']}",
        f"selected_false: {summary['selected_false']}",
        f"selected_precision: {summary['selected_precision']:.4f}",
        '',
        '## Greedy false-link alternatives',
        f"false_selected_with_true_alternative: {summary['false_selected_with_true_alternative']}",
        f"false_selected_with_true_alternative_rate: {summary['false_selected_with_true_alternative_rate']:.4f}",
        f"true_alt_within_0.05: {summary['false_selected_true_alt_within_0_05']}",
        f"true_alt_within_0.10: {summary['false_selected_true_alt_within_0_10']}",
        f"score_gap_stats: `{json.dumps(summary['false_minus_true_alt_score_gap_stats'], sort_keys=True)}`",
        '',
        '## By sequence',
        '| seq | selected | true | false | precision | false with true alt | alt rate | alt within .10 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for seq, s in by_seq.items():
        md.append(f"| {seq} | {s['selected_links']} | {s['selected_true']} | {s['selected_false']} | {s['selected_precision']:.3f} | {s['false_selected_with_true_alt']} | {s['false_selected_with_true_alt_rate']:.3f} | {s['false_selected_true_alt_within_0_10']} |")
    md += [
        '',
        '## Blocked true link reasons',
        '```json',
        json.dumps(summary['blocked_true_reason_counts'], indent=2, sort_keys=True),
        '```',
    ]
    (out_dir / 'edge_conflict_summary.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
