#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path('/gemini/code/FMtrack-main/FM-Track')
INPUT = REPO / 'outputs' / 'alink_train_inputs' / 'parambest_track_results'
GT = Path('/gemini/code/datasets/MOT20/train')
OUT = REPO / 'outputs' / 'oracle_link_bound_train'
ORACLE = REPO / 'scripts' / 'postprocess' / 'oracle_tracklet_link_bound.py'
EVAL = REPO / 'scripts' / 'eval_motstyle_trackeval.py'
SEQS = ['MOT20-01', 'MOT20-02', 'MOT20-03', 'MOT20-05']
CONFIGS = [
    {'name': 'gap60', 'max_gap': 60},
    {'name': 'gap300', 'max_gap': 300},
    {'name': 'global', 'max_gap': 999999},
]


def run(cmd: list[str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    print('[run]', ' '.join(cmd[:9]), '...', flush=True)
    with log.open('w', encoding='utf-8') as f:
        f.write('[started_at] ' + datetime.now().astimezone().isoformat(timespec='seconds') + '\n')
        f.write('[cmd] ' + ' '.join(cmd) + '\n\n')
        f.flush()
        p = subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT)
        f.write('\n[finished_at] ' + datetime.now().astimezone().isoformat(timespec='seconds') + '\n')
        f.write('[return_code] ' + str(p.returncode) + '\n')
    return p.returncode


def parse_summary(path: Path) -> dict:
    lines = path.read_text().strip().splitlines()
    h = lines[0].split(); v = lines[1].split()
    d = {}
    for k, val in zip(h, v):
        try: d[k] = float(val)
        except Exception: d[k] = val
    return d


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for cfg in CONFIGS:
        name = cfg['name']
        run_dir = OUT / name
        linked = run_dir / 'linked_results'
        summary_json = run_dir / 'oracle_summary.json'
        summary_csv = run_dir / 'oracle_summary.csv'
        if not summary_json.is_file():
            cmd = [
                sys.executable, str(ORACLE),
                '--input-dir', str(INPUT),
                '--gt-root', str(GT),
                '--output-dir', str(linked),
                '--seqs', *SEQS,
                '--iou-thr', '0.5',
                '--max-gap', str(cfg['max_gap']),
                '--min-purity', '0.60',
                '--min-match-frac', '0.20',
                '--min-majority-count', '2',
                '--summary-json', str(summary_json),
                '--summary-csv', str(summary_csv),
            ]
            rc = run(cmd, run_dir / 'oracle.log')
            if rc != 0:
                raise RuntimeError(f'oracle failed {name}')
        eval_summary = run_dir / 'trackeval_work' / 'eval' / name / 'pedestrian_summary.txt'
        if not eval_summary.is_file():
            cmd = [
                sys.executable, str(EVAL),
                '--benchmark-name', 'MOT20',
                '--split-to-eval', 'train',
                '--gt-root', str(GT),
                '--results-dir', str(linked),
                '--tracker-name', name,
                '--work-dir', str(run_dir / 'trackeval_work'),
                '--seqs', *SEQS,
            ]
            rc = run(cmd, run_dir / 'eval.log')
            if rc != 0:
                raise RuntimeError(f'eval failed {name}')
        m = parse_summary(eval_summary)
        link_rows = list(csv.DictReader(summary_csv.open())) if summary_csv.is_file() else []
        selected = sum(int(float(r.get('selected_links', 0))) for r in link_rows)
        remapped = sum(int(float(r.get('remapped_rows', 0))) for r in link_rows)
        row = {
            'name': name,
            'max_gap': cfg['max_gap'],
            'selected_links': selected,
            'remapped_rows': remapped,
            'HOTA': m.get('HOTA'), 'DetA': m.get('DetA'), 'AssA': m.get('AssA'),
            'IDF1': m.get('IDF1'), 'MOTA': m.get('MOTA'), 'IDSW': m.get('IDSW'), 'Frag': m.get('Frag'),
            'CLR_FN': m.get('CLR_FN'), 'CLR_FP': m.get('CLR_FP'),
        }
        rows.append(row)
        print(json.dumps(row, indent=2, sort_keys=True), flush=True)
    fields = list(rows[0].keys()) if rows else ['name']
    with (OUT / 'summary.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    (OUT / 'summary.json').write_text(json.dumps(rows, indent=2, sort_keys=True) + '\n')
    print(json.dumps(rows, indent=2, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
