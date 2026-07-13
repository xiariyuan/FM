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
OUT_ROOT = REPO / 'outputs' / 'alink_v2_fast_train_sweep'
LINKER = REPO / 'scripts' / 'postprocess' / 'safe_tracklet_linker_v2.py'
EVAL = REPO / 'scripts' / 'eval_motstyle_trackeval.py'
GT_ROOT = Path('/gemini/code/datasets/MOT20/train')
SEQS = ['MOT20-01', 'MOT20-02', 'MOT20-03', 'MOT20-05']
CONFIGS = [
    ('fast_tight', 0.72, 0.72, 0.62, 0.66, 0.54, 0.62),
    ('fast_default', 0.70, 0.70, 0.58, 0.62, 0.48, 0.58),
    ('fast_recall', 0.68, 0.68, 0.54, 0.58, 0.42, 0.54),
]


def run(cmd: list[str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    print('[run]', ' '.join(cmd[:10]), '...', flush=True)
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
        except Exception: out[k] = val
    return out


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, t1a, t1s, t2a, t2s, t3a, t3s in CONFIGS:
        run_dir = OUT_ROOT / name
        linked = run_dir / 'linked_results'
        sj = run_dir / 'link_summary.json'
        sc = run_dir / 'link_summary.csv'
        if not sj.is_file():
            cmd = [
                sys.executable, str(LINKER),
                '--input-dir', str(INPUT), '--phase0-root', str(PHASE0), '--output-dir', str(linked), '--seqs', *SEQS,
                '--max-gap', '120', '--min-source-len', '3', '--min-target-len', '2',
                '--candidate-max-center-step', '180', '--candidate-max-area-ratio', '8', '--candidate-max-height-ratio', '3.5',
                '--max-bank-feats', '8', '--bank-topk', '3', '--global-k', '12', '--high-k', '6', '--start-k', '4', '--end-k', '4',
                '--tier1-app', str(t1a), '--tier1-score', str(t1s), '--tier1-margin', '0.0',
                '--tier2-app', str(t2a), '--tier2-score', str(t2s), '--tier2-max-rank', '3',
                '--tier3-app', str(t3a), '--tier3-score', str(t3s), '--tier3-max-rank', '4',
                '--summary-json', str(sj), '--summary-csv', str(sc),
            ]
            rc = run(cmd, run_dir / 'link.log')
            if rc != 0: raise RuntimeError(f'link failed {name}')
        es = run_dir / 'trackeval_work' / 'eval' / name / 'pedestrian_summary.txt'
        if not es.is_file():
            cmd = [sys.executable, str(EVAL), '--benchmark-name', 'MOT20', '--split-to-eval', 'train', '--gt-root', str(GT_ROOT), '--results-dir', str(linked), '--tracker-name', name, '--work-dir', str(run_dir / 'trackeval_work'), '--seqs', *SEQS]
            rc = run(cmd, run_dir / 'eval.log')
            if rc != 0: raise RuntimeError(f'eval failed {name}')
        m = parse_summary(es)
        link_rows = list(csv.DictReader(sc.open())) if sc.is_file() else []
        row = {
            'name': name,
            'selected_links': sum(int(float(r.get('selected_links', 0))) for r in link_rows),
            'tiered_edges': sum(int(float(r.get('tiered_edges', 0))) for r in link_rows),
            'candidate_edges': sum(int(float(r.get('candidate_edges', 0))) for r in link_rows),
            'remapped_rows': sum(int(float(r.get('remapped_rows', 0))) for r in link_rows),
            'HOTA': m.get('HOTA'), 'DetA': m.get('DetA'), 'AssA': m.get('AssA'), 'IDF1': m.get('IDF1'), 'MOTA': m.get('MOTA'), 'IDSW': m.get('IDSW'), 'Frag': m.get('Frag'), 'CLR_FN': m.get('CLR_FN'), 'CLR_FP': m.get('CLR_FP'),
        }
        rows.append(row); print(json.dumps(row, indent=2, sort_keys=True), flush=True)
    fields=list(rows[0].keys()) if rows else ['name']
    with (OUT_ROOT/'summary.csv').open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    (OUT_ROOT/'summary.json').write_text(json.dumps(rows, indent=2, sort_keys=True)+'\n')
    print(json.dumps(rows, indent=2, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
