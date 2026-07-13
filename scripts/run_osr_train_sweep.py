#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path('/gemini/code/FMtrack-main/FM-Track')
TRACKER = REPO / 'scripts' / 'dmm_base_tracker_osr.py'
EVAL = REPO / 'scripts' / 'eval_motstyle_trackeval.py'
PHASE0 = REPO / 'outputs' / 'alink_train_inputs' / 'phase0_root'
GT = Path('/gemini/code/datasets/MOT20/train')
OUT = REPO / 'outputs' / 'osr_train_sweep'
SEQS = ['MOT20-01', 'MOT20-02', 'MOT20-03', 'MOT20-05']

CONFIGS = [
    {
        'name': 'osr_safe',
        'osr_max_lost_age': 20,
        'osr_min_track_len': 10,
        'osr_min_det_score': 0.10,
        'osr_min_memory_sim': 0.72,
        'osr_max_center_step': 40,
        'osr_min_margin': 0.05,
        'osr_max_ambiguity': 1,
        'osr_min_score': 0.62,
    },
    {
        'name': 'osr_medium',
        'osr_max_lost_age': 30,
        'osr_min_track_len': 8,
        'osr_min_det_score': 0.08,
        'osr_min_memory_sim': 0.68,
        'osr_max_center_step': 60,
        'osr_min_margin': 0.03,
        'osr_max_ambiguity': 2,
        'osr_min_score': 0.58,
    },
    {
        'name': 'osr_recall',
        'osr_max_lost_age': 45,
        'osr_min_track_len': 5,
        'osr_min_det_score': 0.05,
        'osr_min_memory_sim': 0.64,
        'osr_max_center_step': 80,
        'osr_min_margin': 0.02,
        'osr_max_ambiguity': 3,
        'osr_min_score': 0.54,
    },
]


def run(cmd: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    print('[run]', ' '.join(cmd[:10]), '...', flush=True)
    with log.open('w', encoding='utf-8') as f:
        f.write('[started_at] ' + datetime.now().astimezone().isoformat(timespec='seconds') + '\n')
        f.write('[cmd] ' + ' '.join(cmd) + '\n\n')
        f.flush()
        p = subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT)
        f.write('\n[finished_at] ' + datetime.now().astimezone().isoformat(timespec='seconds') + '\n')
        f.write('[return_code] ' + str(p.returncode) + '\n')
    if p.returncode != 0:
        raise RuntimeError(f'command failed rc={p.returncode}; see {log}')


def parse_summary(path: Path) -> dict:
    lines = path.read_text().strip().splitlines()
    h = lines[0].split(); v = lines[1].split()
    d = {}
    for k, val in zip(h, v):
        try: d[k] = float(val)
        except Exception: d[k] = val
    return d


def run_tracker(seq: str, cfg: dict, run_dir: Path) -> dict:
    out_txt = run_dir / 'track_results' / f'{seq}.txt'
    summary_json = run_dir / f'{seq}_summary.json'
    if out_txt.is_file() and summary_json.is_file():
        return json.loads(summary_json.read_text())
    dump = PHASE0 / seq / 'dump_yolox_reid.npz'
    cmd = [
        sys.executable, str(TRACKER),
        '--dump-npz', str(dump),
        '--seq', seq,
        '--assoc-mode', 'botsort_reid',
        '--track-high-thresh', '0.6',
        '--track-low-thresh', '0.1',
        '--track-buffer', '70',
        '--match-thresh', '0.5',
        '--new-track-thresh', '0.5',
        '--out', str(out_txt),
        '--summary-json', str(summary_json),
        '--osr-enable',
        '--osr-max-lost-age', str(cfg['osr_max_lost_age']),
        '--osr-min-track-len', str(cfg['osr_min_track_len']),
        '--osr-min-det-score', str(cfg['osr_min_det_score']),
        '--osr-min-memory-sim', str(cfg['osr_min_memory_sim']),
        '--osr-max-center-step', str(cfg['osr_max_center_step']),
        '--osr-min-margin', str(cfg['osr_min_margin']),
        '--osr-max-ambiguity', str(cfg['osr_max_ambiguity']),
        '--osr-min-score', str(cfg['osr_min_score']),
    ]
    run(cmd, run_dir / 'logs' / f'{seq}_tracker.log')
    return json.loads(summary_json.read_text())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for cfg in CONFIGS:
        name = cfg['name']
        run_dir = OUT / name
        osr_stats = {}
        for seq in SEQS:
            summ = run_tracker(seq, cfg, run_dir)
            osr_stats[seq] = summ.get('osr_stats', {})
        eval_summary = run_dir / 'trackeval_work' / 'eval' / name / 'pedestrian_summary.txt'
        if not eval_summary.is_file():
            cmd = [
                sys.executable, str(EVAL),
                '--benchmark-name', 'MOT20',
                '--split-to-eval', 'train',
                '--gt-root', str(GT),
                '--results-dir', str(run_dir / 'track_results'),
                '--tracker-name', name,
                '--work-dir', str(run_dir / 'trackeval_work'),
                '--seqs', *SEQS,
            ]
            run(cmd, run_dir / 'eval.log')
        m = parse_summary(eval_summary)
        recovered_total = sum(int(v.get('recovered', 0)) for v in osr_stats.values())
        valid_pairs = sum(int(v.get('valid_pairs', 0)) for v in osr_stats.values())
        candidate_pairs = sum(int(v.get('candidate_pairs', 0)) for v in osr_stats.values())
        row = {
            'name': name,
            'HOTA': m.get('HOTA'), 'DetA': m.get('DetA'), 'AssA': m.get('AssA'),
            'IDF1': m.get('IDF1'), 'MOTA': m.get('MOTA'), 'IDSW': m.get('IDSW'),
            'Frag': m.get('Frag'), 'CLR_FN': m.get('CLR_FN'), 'CLR_FP': m.get('CLR_FP'),
            'osr_recovered': recovered_total,
            'osr_valid_pairs': valid_pairs,
            'osr_candidate_pairs': candidate_pairs,
            **{k: v for k, v in cfg.items() if k != 'name'},
        }
        rows.append(row)
        (run_dir / 'osr_stats_by_seq.json').write_text(json.dumps(osr_stats, indent=2, sort_keys=True) + '\n')
        print(json.dumps(row, indent=2, sort_keys=True), flush=True)
    rows.sort(key=lambda r: (-(r['HOTA'] or 0), r['IDSW'] or 999999))
    fields = list(rows[0].keys())
    with (OUT / 'summary.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    (OUT / 'summary.json').write_text(json.dumps(rows, indent=2, sort_keys=True) + '\n')
    print(json.dumps(rows, indent=2, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
