#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path('/gemini/code/FMtrack-main/FM-Track')
OUT = REPO / 'outputs' / 'alink_v3_sequence_aware_train'
EVAL = REPO / 'scripts' / 'eval_motstyle_trackeval.py'
GT = Path('/gemini/code/datasets/MOT20/train')
SEQS = ['MOT20-01', 'MOT20-02', 'MOT20-03', 'MOT20-05']

SRC = {
    'base': REPO / 'outputs' / 'alink_train_inputs' / 'parambest_track_results',
    'old_strict': REPO / 'outputs' / 'alink_train_sweep' / 'strict' / 'linked_results',
    'old_recall': REPO / 'outputs' / 'alink_train_sweep' / 'recall' / 'linked_results',
    'tier1': REPO / 'outputs' / 'alink_v2_select_train_sweep' / 'tier1_only' / 'linked_results',
    'no_tier3': REPO / 'outputs' / 'alink_v2_select_train_sweep' / 'no_tier3' / 'linked_results',
    'cap40': REPO / 'outputs' / 'alink_v2_select_train_sweep' / 'cap40' / 'linked_results',
    'cap60': REPO / 'outputs' / 'alink_v2_select_train_sweep' / 'cap60' / 'linked_results',
    'margin_strict': REPO / 'outputs' / 'alink_v2_select_train_sweep' / 'margin_strict' / 'linked_results',
}

COMBOS = {
    # 02 is the main beneficiary of v2 cap60; 03 is fragile, so start conservative there.
    'mix_a_02cap60_03old_05old': {
        'MOT20-01': 'tier1', 'MOT20-02': 'cap60', 'MOT20-03': 'old_recall', 'MOT20-05': 'old_recall'
    },
    # 03 no_tier3 may reduce wrong weak links while keeping v2 tier1/2.
    'mix_b_02cap60_03not3_05old': {
        'MOT20-01': 'tier1', 'MOT20-02': 'cap60', 'MOT20-03': 'no_tier3', 'MOT20-05': 'old_recall'
    },
    # Try v2 cap60 on 05 too; cap60 global was best v2 but may overfit 03/05.
    'mix_c_02cap60_03old_05cap60': {
        'MOT20-01': 'tier1', 'MOT20-02': 'cap60', 'MOT20-03': 'old_recall', 'MOT20-05': 'cap60'
    },
    # Conservative 01/03/05, aggressive 02 only.
    'mix_d_only02cap60': {
        'MOT20-01': 'old_strict', 'MOT20-02': 'cap60', 'MOT20-03': 'old_recall', 'MOT20-05': 'old_recall'
    },
    # Current best single global selector as reference copied exactly.
    'ref_cap60_global': {
        'MOT20-01': 'cap60', 'MOT20-02': 'cap60', 'MOT20-03': 'cap60', 'MOT20-05': 'cap60'
    },
    # Old recall global reference.
    'ref_old_recall_global': {
        'MOT20-01': 'old_recall', 'MOT20-02': 'old_recall', 'MOT20-03': 'old_recall', 'MOT20-05': 'old_recall'
    },
}


def parse_summary(path: Path) -> dict:
    lines = path.read_text().strip().splitlines()
    h = lines[0].split(); v = lines[1].split()
    out = {}
    for k, val in zip(h, v):
        try:
            out[k] = float(val)
        except Exception:
            out[k] = val
    return out


def run_eval(name: str, results_dir: Path, run_dir: Path) -> None:
    summary = run_dir / 'trackeval_work' / 'eval' / name / 'pedestrian_summary.txt'
    if summary.is_file():
        return
    cmd = [
        sys.executable, str(EVAL), '--benchmark-name', 'MOT20', '--split-to-eval', 'train',
        '--gt-root', str(GT), '--results-dir', str(results_dir), '--tracker-name', name,
        '--work-dir', str(run_dir / 'trackeval_work'), '--seqs', *SEQS,
    ]
    log = run_dir / 'eval.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w', encoding='utf-8') as f:
        p = subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        raise RuntimeError(f'eval failed for {name}: {log}')


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, mapping in COMBOS.items():
        run_dir = OUT / name
        results = run_dir / 'linked_results'
        results.mkdir(parents=True, exist_ok=True)
        manifest = {'name': name, 'mapping': mapping, 'files': {}}
        for seq in SEQS:
            src_name = mapping[seq]
            src = SRC[src_name] / f'{seq}.txt'
            dst = results / f'{seq}.txt'
            if not src.is_file():
                raise FileNotFoundError(f'{seq} source {src_name} missing: {src}')
            if not dst.is_file():
                shutil.copy2(src, dst)
            manifest['files'][seq] = str(src)
        (run_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
        run_eval(name, results, run_dir)
        summary = run_dir / 'trackeval_work' / 'eval' / name / 'pedestrian_summary.txt'
        m = parse_summary(summary)
        row = {
            'name': name,
            'HOTA': m.get('HOTA'), 'DetA': m.get('DetA'), 'AssA': m.get('AssA'),
            'IDF1': m.get('IDF1'), 'MOTA': m.get('MOTA'), 'IDSW': m.get('IDSW'),
            'Frag': m.get('Frag'), 'CLR_FN': m.get('CLR_FN'), 'CLR_FP': m.get('CLR_FP'),
            'MOT20-01': mapping['MOT20-01'], 'MOT20-02': mapping['MOT20-02'],
            'MOT20-03': mapping['MOT20-03'], 'MOT20-05': mapping['MOT20-05'],
        }
        rows.append(row)
        print(json.dumps(row, indent=2, sort_keys=True), flush=True)
    rows.sort(key=lambda r: (-(r['HOTA'] or 0), r['IDSW'] or 999999))
    fields = list(rows[0].keys())
    with (OUT / 'summary.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    (OUT / 'summary.json').write_text(json.dumps(rows, indent=2, sort_keys=True) + '\n')
    print(json.dumps(rows, indent=2, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
