#!/usr/bin/env python3
"""Extract tiered and selected edges from existing A-Link v2 detail files.

This helper exists because full candidate-edge details for MOT20-05 are slow and
large. For validator training/application, we only need tiered_edges and
selected_links, not rejected candidate_edges.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def copy_rows(src: Path, dst: Path, keep_selected_only: bool = False) -> int:
    if not src.is_file():
        return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open('r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
        fields = rows[0].keys() if rows else []
    if keep_selected_only:
        rows = [r for r in rows if str(r.get('selected', '0')) in ('1','1.0','True','true')]
    with dst.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(fields), extrasaction='ignore')
        w.writeheader(); w.writerows(rows)
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--detail-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--seqs', nargs='+', required=True)
    args = ap.parse_args()
    for seq in args.seqs:
        n1 = copy_rows(Path(args.detail_dir)/f'{seq}_tiered_edges.csv', Path(args.out_dir)/f'{seq}_tiered_edges.csv')
        n2 = copy_rows(Path(args.detail_dir)/f'{seq}_selected_links.csv', Path(args.out_dir)/f'{seq}_selected_links.csv')
        print(seq, 'tiered', n1, 'selected', n2)

if __name__ == '__main__':
    main()
