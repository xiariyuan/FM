#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import networkx as nx


def read_mot(path: Path):
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 6:
                continue
            rows.append((int(float(parts[0])), int(float(parts[1])), parts))
    return rows


def find(parent, x):
    parent.setdefault(x, x)
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def edge_weight(score: float, thr: float, mode: str):
    eps = 1e-6
    s = max(eps, min(1.0 - eps, score))
    if mode == 'score':
        return score
    if mode == 'score_minus_thr':
        return score - thr
    if mode == 'logit':
        return math.log(s / (1.0 - s))
    if mode == 'logit_shift_thr':
        t = max(eps, min(1.0 - eps, thr))
        return math.log(s / (1.0 - s)) - math.log(t / (1.0 - t))
    return score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', required=True)
    ap.add_argument('--scores', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--thr', type=float, default=0.2)
    ap.add_argument('--max-gap', type=int, default=60)
    ap.add_argument('--weight-mode', choices=['score', 'score_minus_thr', 'logit', 'logit_shift_thr'], default='score_minus_thr')
    ap.add_argument('--max-links-per-seq', type=int, default=999999)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_dir = Path(args.input_dir)

    candidates_by_seq = defaultdict(list)
    with open(args.scores, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            score = float(row.get('aflink_score', 0.0) or 0.0)
            gap = int(float(row.get('gap', 0) or 0))
            if score < args.thr or gap <= 0 or gap > args.max_gap:
                continue
            candidates_by_seq[row['seq']].append({
                'seq': row['seq'],
                'track_a': int(float(row['track_a'])),
                'track_b': int(float(row['track_b'])),
                'score': score,
                'same_gt': int(float(row.get('same_gt', 0) or 0)),
                'gap': gap,
            })

    selected_all = []
    by_seq_report = []

    for seq, candidates in sorted(candidates_by_seq.items()):
        G = nx.Graph()
        edge_payload = {}
        for i, cand in enumerate(candidates):
            a_node = ('src', cand['track_a'])
            b_node = ('dst', cand['track_b'])
            w = edge_weight(cand['score'], args.thr, args.weight_mode)
            if w <= 0:
                continue
            # tiny deterministic tie-breaker to avoid random equivalent choices.
            w = float(w) + 1e-9 * cand['score'] - 1e-12 * i
            G.add_edge(a_node, b_node, weight=w)
            edge_payload[(a_node, b_node)] = cand
            edge_payload[(b_node, a_node)] = cand

        matching = nx.algorithms.matching.max_weight_matching(G, maxcardinality=False, weight='weight')
        selected = []
        for u, v in matching:
            cand = edge_payload.get((u, v))
            if cand is not None:
                selected.append(cand)
        selected.sort(key=lambda r: -r['score'])
        if args.max_links_per_seq < len(selected):
            selected = selected[:args.max_links_per_seq]

        parent = {}
        # Time-directed candidate graph should be acyclic; keep a union guard anyway.
        final_selected = []
        used_successor = set()
        used_predecessor = set()
        for cand in selected:
            a = cand['track_a']
            b = cand['track_b']
            if a in used_successor or b in used_predecessor:
                continue
            ra = find(parent, a)
            rb = find(parent, b)
            if ra == rb:
                continue
            parent[rb] = ra
            used_successor.add(a)
            used_predecessor.add(b)
            final_selected.append(cand)

        involved = set()
        for cand in final_selected:
            involved.add(cand['track_a'])
            involved.add(cand['track_b'])
        id_map = {tid: find(parent, tid) for tid in involved}

        input_file = input_dir / f'{seq}.txt'
        output_file = out_dir / f'{seq}.txt'
        if input_file.exists():
            new_rows = []
            for _, tid, parts in read_mot(input_file):
                parts = list(parts)
                parts[1] = str(id_map.get(tid, tid))
                new_rows.append(parts)
            new_rows.sort(key=lambda p: (int(float(p[0])), int(float(p[1])), float(p[2]), float(p[3])))
            with output_file.open('w', encoding='utf-8') as f:
                for parts in new_rows:
                    f.write(','.join(parts) + '\n')

        tp = sum(c['same_gt'] for c in final_selected)
        selected_all.extend(final_selected)
        by_seq_report.append({
            'seq': seq,
            'candidate_after_threshold': len(candidates),
            'graph_edges_positive_weight': G.number_of_edges(),
            'selected_links': len(final_selected),
            'tp_train_label': tp,
            'precision_train_label': tp / len(final_selected) if final_selected else 0.0,
        })

    for input_file in sorted(input_dir.glob('MOT20-*.txt')):
        output_file = out_dir / input_file.name
        if not output_file.exists():
            output_file.write_text(input_file.read_text(encoding='utf-8'), encoding='utf-8')

    total_tp = sum(c['same_gt'] for c in selected_all)
    summary = {
        'solver': 'networkx.max_weight_matching',
        'thr': args.thr,
        'max_gap': args.max_gap,
        'weight_mode': args.weight_mode,
        'max_links_per_seq': args.max_links_per_seq,
        'selected_links': len(selected_all),
        'tp_train_label': total_tp,
        'precision_train_label': total_tp / len(selected_all) if selected_all else 0.0,
        'by_seq': by_seq_report,
    }
    parent_dir = out_dir.parent
    parent_dir.mkdir(parents=True, exist_ok=True)
    (parent_dir / 'link_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    with (parent_dir / 'selected_links.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['seq', 'track_a', 'track_b', 'score', 'same_gt', 'gap'])
        writer.writeheader()
        writer.writerows(selected_all)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
