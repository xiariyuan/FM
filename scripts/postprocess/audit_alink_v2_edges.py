#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import median


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open('r', encoding='utf-8') as f:
        return list(csv.DictReader(f))


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


def summarize(rows: list[dict], labels: dict[tuple[int, int], int], name: str) -> dict:
    n = len(rows)
    tp = 0
    by_tier = {}
    vals_true = {k: [] for k in ['edge_score','app_sim','bank_max','bank_topk','end_start_sim','motion_score','size_score','center_step','gap','source_margin','target_margin']}
    vals_false = {k: [] for k in vals_true}
    for r in rows:
        a = ii(r.get('track_a', r.get('source_tid', 0)))
        b = ii(r.get('track_b', r.get('target_tid', 0)))
        y = int(labels.get((a, b), 0))
        tp += y
        tier = r.get('tier', 'na') or 'na'
        by_tier.setdefault(tier, {'n':0,'tp':0})
        by_tier[tier]['n'] += 1
        by_tier[tier]['tp'] += y
        bucket = vals_true if y else vals_false
        for k in bucket:
            if k in r:
                bucket[k].append(ff(r.get(k)))
    out = {'name': name, 'n': n, 'tp': tp, 'precision': tp / n if n else 0.0}
    for tier, d in sorted(by_tier.items()):
        out[f'{tier}_n'] = d['n']
        out[f'{tier}_tp'] = d['tp']
        out[f'{tier}_precision'] = d['tp'] / d['n'] if d['n'] else 0.0
    for k in vals_true:
        if vals_true[k]: out[f'true_{k}_median'] = median(vals_true[k])
        if vals_false[k]: out[f'false_{k}_median'] = median(vals_false[k])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--alink-detail-dir', required=True)
    ap.add_argument('--oracle-detail-dir', required=True)
    ap.add_argument('--seqs', nargs='+', required=True)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    missed_rows = []
    selected_false = []
    selected_true = []
    for seq in args.seqs:
        labels_rows = read_csv(Path(args.oracle_detail_dir) / f'{seq}_tracklet_labels.csv')
        lab = {}
        eligible_by_tid = {}
        gt_by_tid = {}
        for r in labels_rows:
            tid = ii(r.get('tid'))
            eligible = ii(r.get('eligible'))
            gt = ii(r.get('majority_gt'))
            eligible_by_tid[tid] = eligible
            gt_by_tid[tid] = gt
        # True edge label: same eligible majority GT.
        candidate = read_csv(Path(args.alink_detail_dir) / f'{seq}_candidate_edges.csv')
        tiered = read_csv(Path(args.alink_detail_dir) / f'{seq}_tiered_edges.csv')
        selected = read_csv(Path(args.alink_detail_dir) / f'{seq}_selected_links.csv')
        labels = {}
        all_pairs = set()
        for rows in [candidate, tiered, selected]:
            for r in rows:
                a = ii(r.get('track_a', r.get('source_tid', 0)))
                b = ii(r.get('track_b', r.get('target_tid', 0)))
                all_pairs.add((a, b))
        for a, b in all_pairs:
            labels[(a, b)] = int(eligible_by_tid.get(a, 0) and eligible_by_tid.get(b, 0) and gt_by_tid.get(a, -1) > 0 and gt_by_tid.get(a) == gt_by_tid.get(b))

        # Oracle selected recall.
        oracle = read_csv(Path(args.oracle_detail_dir) / f'{seq}_selected_links.csv')
        cand_pairs = {(ii(r.get('track_a')), ii(r.get('track_b'))) for r in candidate}
        tier_pairs = {(ii(r.get('track_a')), ii(r.get('track_b'))) for r in tiered}
        sel_pairs = {(ii(r.get('track_a')), ii(r.get('track_b'))) for r in selected}
        oracle_pairs = {(ii(r.get('source_tid')), ii(r.get('target_tid'))) for r in oracle}
        oracle_in_cand = len(oracle_pairs & cand_pairs)
        oracle_in_tier = len(oracle_pairs & tier_pairs)
        oracle_in_sel = len(oracle_pairs & sel_pairs)
        for r in oracle:
            a = ii(r.get('source_tid')); b = ii(r.get('target_tid'))
            if (a,b) not in sel_pairs:
                missed_rows.append({'seq': seq, **r, 'in_candidate': int((a,b) in cand_pairs), 'in_tiered': int((a,b) in tier_pairs), 'in_selected': 0})

        for r in selected:
            a = ii(r.get('track_a')); b = ii(r.get('track_b'))
            rec = {'seq': seq, 'is_true': labels.get((a,b),0), **r, 'gt_a': gt_by_tid.get(a, -1), 'gt_b': gt_by_tid.get(b, -1)}
            (selected_true if rec['is_true'] else selected_false).append(rec)

        for name, rows in [('candidate', candidate), ('tiered', tiered), ('selected', selected)]:
            s = summarize(rows, labels, f'{seq}_{name}')
            s.update({'seq': seq, 'kind': name, 'oracle_links': len(oracle_pairs), 'oracle_in_candidate': oracle_in_cand, 'oracle_in_tiered': oracle_in_tier, 'oracle_in_selected': oracle_in_sel})
            summaries.append(s)

    # Write summaries with dynamic fields.
    fields = sorted(set().union(*(r.keys() for r in summaries))) if summaries else ['name']
    with (out_dir / 'edge_audit_summary.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); w.writeheader(); w.writerows(summaries)
    (out_dir / 'edge_audit_summary.json').write_text(json.dumps(summaries, indent=2, sort_keys=True) + '\n')
    for fname, rows in [('missed_oracle_links.csv', missed_rows), ('selected_false_links.csv', selected_false), ('selected_true_links.csv', selected_true)]:
        if rows:
            fields = sorted(set().union(*(r.keys() for r in rows)))
            with (out_dir / fname).open('w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); w.writeheader(); w.writerows(rows)
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
