#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path('/gemini/code/FMtrack-main/FM-Track')
OUT_ROOT = REPO / 'outputs' / 'dmm_parambest_mot20_test'
PHASE0 = REPO / 'outputs' / 'dmm_phase0_mot20_test_parambest'
TRACKER = REPO / 'scripts' / 'dmm_base_tracker.py'
CHECK = REPO / 'scripts' / 'check_mot20_submission.py'
SEQS = ['MOT20-04', 'MOT20-06', 'MOT20-07', 'MOT20-08']


def run(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print('[run]', ' '.join(cmd[:8]), '...', flush=True)
    with log_path.open('w', encoding='utf-8') as f:
        f.write('[started_at] ' + datetime.now().astimezone().isoformat(timespec='seconds') + '\n')
        f.write('[cmd] ' + ' '.join(cmd) + '\n\n')
        f.flush()
        proc = subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT)
        f.write('\n[finished_at] ' + datetime.now().astimezone().isoformat(timespec='seconds') + '\n')
        f.write('[return_code] ' + str(proc.returncode) + '\n')
    if proc.returncode != 0:
        raise RuntimeError(f'Command failed rc={proc.returncode}, see {log_path}')


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results_dir = OUT_ROOT / 'track_results'
    logs_dir = OUT_ROOT / 'logs'
    for seq in SEQS:
        dump = PHASE0 / seq / 'dump_yolox_reid.npz'
        if not dump.is_file():
            raise FileNotFoundError(dump)
        out_txt = results_dir / f'{seq}.txt'
        summary_json = OUT_ROOT / f'{seq}_summary.json'
        cmd = [
            sys.executable, str(TRACKER),
            '--dump-npz', str(dump),
            '--seq', seq,
            '--assoc-mode', 'botsort_reid',
            '--track-buffer', '70',
            '--match-thresh', '0.5',
            '--new-track-thresh', '0.5',
            '--out', str(out_txt),
            '--summary-json', str(summary_json),
        ]
        run(cmd, logs_dir / f'{seq}_tracker.log')
    check_cmd = [
        sys.executable, str(CHECK),
        '--results-dir', str(results_dir),
        '--profile', 'mot20_test_4',
    ]
    run(check_cmd, logs_dir / 'precheck_results_dir.log')
    zip_path = OUT_ROOT / 'MOT20_parambest_buf70_match05_newtrk05_submission.zip'
    if zip_path.exists():
        zip_path.unlink()
    zip_cmd = ['zip', '-j', str(zip_path)] + [str(results_dir / f'{seq}.txt') for seq in SEQS]
    run(zip_cmd, logs_dir / 'zip.log')
    check_zip_cmd = [
        sys.executable, str(CHECK),
        '--zip-path', str(zip_path),
        '--profile', 'mot20_test_4',
    ]
    run(check_zip_cmd, logs_dir / 'precheck_zip.log')
    manifest = {
        'status': 'completed',
        'phase': 'PARAMBEST_MOT20_TEST_SUBMISSION',
        'config': {
            'assoc_mode': 'botsort_reid',
            'track_buffer': 70,
            'match_thresh': 0.5,
            'new_track_thresh': 0.5,
            'disabled_modules': ['DMM', 'GateV1', 'OC-SORT', 'ExtendedProximity', 'Stage2', 'NSA', 'GMC'],
        },
        'phase0_root': str(PHASE0),
        'results_dir': str(results_dir),
        'zip_path': str(zip_path),
        'sequences': SEQS,
        'finished_at': datetime.now().astimezone().isoformat(timespec='seconds'),
    }
    (OUT_ROOT / 'run_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
