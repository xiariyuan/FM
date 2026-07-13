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
PHASE0 = REPO / 'outputs' / 'alink_train_inputs' / 'phase0_root'
OUT_ROOT = REPO / 'outputs' / 'alink_train_sweep'
LINKER = REPO / 'scripts' / 'postprocess' / 'safe_tracklet_linker.py'
EVAL = REPO / 'scripts' / 'eval_motstyle_trackeval.py'
GT_ROOT = Path('/gemini/code/datasets/MOT20/train')
SEQS = ['MOT20-01', 'MOT20-02', 'MOT20-03', 'MOT20-05']

CONFIGS = [
    {
        'name': 'strict',
        'max_gap': 60, 'min_source_len': 10, 'min_target_len': 5,
        'min_app': 0.72, 'min_score': 0.72, 'min_margin': 0.03,
        'max_center_step': 80, 'max_area_ratio': 2.5, 'max_height_ratio': 1.8,
    },
    {
        'name': 'medium',
        'max_gap': 80, 'min_source_len': 8, 'min_target_len': 4,
        'min_app': 0.68, 'min_score': 0.68, 'min_margin': 0.02,
        'max_center_step': 100, 'max_area_ratio': 3.0, 'max_height_ratio': 2.0,
    },
    {
        'name': 'recall',
        'max_gap': 100, 'min_source_len': 5, 'min_target_len': 3,
        'min_app': 0.64, 'min_score': 0.64, 'min_margin': 0.01,
        'max_center_step': 120, 'max_area_ratio': 4.0, 'max_height_ratio': 2.5,
    },
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
    out = {}
    for k, val in zip(h, v):
        try: out[k] = float(val)
        except ValueError: out[k] = val
    return out


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for cfg in CONFIGS:
        name = cfg['name']
        run_dir = OUT_ROOT / name
        linked = run_dir / 'linked_results'
        summary_json = run_dir / 'link_summary.json'
        summary_csv = run_dir / 'link_summary.csv'
        if not summary_json.is_file():
            cmd = [
                sys.executable, str(LINKER),
                '--input-dir', str(INPUT),
                '--phase0-root', str(PHASE0),
                '--output-dir', str(linked),
                '--seqs', *SEQS,
                '--max-gap', str(cfg['max_gap']),
                '--min-source-len', str(cfg['min_source_len']),
                '--min-target-len', str(cfg['min_target_len']),
                '--min-app', str(cfg['min_app']),
                '--min-score', str(cfg['min_score']),
                '--min-margin', str(cfg['min_margin']),
                '--max-center-step', str(cfg['max_center_step']),
                '--max-area-ratio', str(cfg['max_area_ratio']),
                '--max-height-ratio', str(cfg['max_height_ratio']),
                '--summary-json', str(summary_json),
                '--summary-csv', str(summary_csv),
            ]
            rc = run(cmd, run_dir / 'link.log')
            if rc != 0:
                raise RuntimeError(f'link failed {name}, see {run_dir / "link.log"}')
        eval_summary = run_dir / 'trackeval_work' / 'eval' / name / 'pedestrian_summary.txt'
        if not eval_summary.is_file():
            cmd = [
                sys.executable, str(EVAL),
                '--benchmark-name', 'MOT20',
                '--split-to-eval', 'train',
                '--gt-root', str(GT_ROOT),
                '--results-dir', str(linked),
                '--tracker-name', name,
                '--work-dir', str(run_dir / 'trackeval_work'),
                '--seqs', *SEQS,
            ]
            rc = run(cmd, run_dir / 'eval.log')
            if rc != 0:
                raise RuntimeError(f'eval failed {name}, see {run_dir / "eval.log"}')
        m = parse_summary(eval_summary)
        link_rows = list(csv.DictReader(summary_csv.open())) if summary_csv.is_file() else []
        selected = sum(int(float(r.get('selected_links', 0))) for r in link_rows)
        remapped = sum(int(float(r.get('remapped_rows', 0))) for r in link_rows)
        row = {
            'name': name,
            'selected_links': selected,
            'remapped_rows': remapped,
            'HOTA': m.get('HOTA'), 'DetA': m.get('DetA'), 'AssA': m.get('AssA'),
            'IDF1': m.get('IDF1'), 'MOTA': m.get('MOTA'), 'IDSW': m.get('IDSW'), 'Frag': m.get('Frag'), 'CLR_FN': m.get('CLR_FN'), 'CLR_FP': m.get('CLR_FP'),
        }
        rows.append(row)
        print(json.dumps(row, indent=2, sort_keys=True), flush=True)
    fields = list(rows[0].keys()) if rows else ['name']
    with (OUT_ROOT / 'summary.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    (OUT_ROOT / 'summary.json').write_text(json.dumps(rows, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps(rows, indent=2, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
