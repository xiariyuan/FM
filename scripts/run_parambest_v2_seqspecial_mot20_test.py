#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path('/gemini/code/FMtrack-main/FM-Track')
PHASE0 = REPO / 'outputs' / 'dmm_phase0_mot20_test_parambest'
OUT_ROOT = REPO / 'outputs' / 'dmm_parambest_v2_seqspecial_mot20_test'
TRACKER = REPO / 'scripts' / 'dmm_base_tracker.py'
CHECK = REPO / 'scripts' / 'check_mot20_submission.py'
INTERP = REPO / 'scripts' / 'postprocess' / 'linear_interpolate_mot.py'
SANITIZE = REPO / 'scripts' / 'sanitize_mot_submission.py'
SEQS = ['MOT20-04', 'MOT20-06', 'MOT20-07', 'MOT20-08']
SPECIAL = {'MOT20-06', 'MOT20-08'}


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec='seconds')


def run(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print('[run]', ' '.join(cmd[:10]), '...', flush=True)
    with log_path.open('w', encoding='utf-8') as f:
        f.write('[started_at] ' + now() + '\n')
        f.write('[cmd] ' + ' '.join(cmd) + '\n\n')
        f.flush()
        proc = subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT)
        f.write('\n[finished_at] ' + now() + '\n')
        f.write('[return_code] ' + str(proc.returncode) + '\n')
    if proc.returncode != 0:
        raise RuntimeError(f'Command failed rc={proc.returncode}, see {log_path}')


def run_tracker(seq: str) -> None:
    dump = PHASE0 / seq / 'dump_yolox_reid.npz'
    if not dump.is_file():
        raise FileNotFoundError(dump)
    high = 0.3 if seq in SPECIAL else 0.6
    new = 0.4 if seq in SPECIAL else 0.5
    out_txt = OUT_ROOT / 'track_results' / f'{seq}.txt'
    summary_json = OUT_ROOT / f'{seq}_summary.json'
    if out_txt.is_file() and summary_json.is_file():
        print(f'[skip] tracker {seq} already exists', flush=True)
        return
    cmd = [
        sys.executable, str(TRACKER),
        '--dump-npz', str(dump),
        '--seq', seq,
        '--assoc-mode', 'botsort_reid',
        '--track-high-thresh', str(high),
        '--track-low-thresh', '0.1',
        '--track-buffer', '70',
        '--match-thresh', '0.5',
        '--new-track-thresh', str(new),
        '--out', str(out_txt),
        '--summary-json', str(summary_json),
    ]
    run(cmd, OUT_ROOT / 'logs' / f'{seq}_tracker.log')


def check_results_dir(results_dir: Path, log_name: str) -> None:
    run([
        sys.executable, str(CHECK),
        '--results-dir', str(results_dir),
        '--profile', 'mot20_test_4',
    ], OUT_ROOT / 'logs' / log_name)


def zip_results(results_dir: Path, zip_path: Path, log_name: str) -> None:
    if zip_path.exists():
        zip_path.unlink()
    cmd = ['zip', '-j', str(zip_path)] + [str(results_dir / f'{seq}.txt') for seq in SEQS]
    run(cmd, OUT_ROOT / 'logs' / log_name)
    run([
        sys.executable, str(CHECK),
        '--zip-path', str(zip_path),
        '--profile', 'mot20_test_4',
    ], OUT_ROOT / 'logs' / (log_name.replace('.log', '_precheck.log')))


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for seq in SEQS:
        run_tracker(seq)

    raw_dir = OUT_ROOT / 'track_results'
    check_results_dir(raw_dir, 'raw_precheck_results_dir.log')
    raw_zip = OUT_ROOT / 'MOT20_parambest_v2_seqspecial_high03_new04_0608_raw.zip'
    zip_results(raw_dir, raw_zip, 'raw_zip.log')

    interp_dir = OUT_ROOT / 'interp_results'
    run([
        sys.executable, str(INTERP),
        '--input-dir', str(raw_dir),
        '--output-dir', str(interp_dir),
        '--max-gap', '20',
        '--min-track-len', '2',
        '--min-endpoint-score', '0.0',
        '--max-area-ratio', '3.0',
        '--max-center-step', '80.0',
        '--min-box-side', '2.0',
        '--summary-json', str(OUT_ROOT / 'interp_summary.json'),
        '--summary-csv', str(OUT_ROOT / 'interp_summary.csv'),
    ], OUT_ROOT / 'logs' / 'interp.log')

    sanitized_dir = OUT_ROOT / 'sanitized_interp_results'
    run([
        sys.executable, str(SANITIZE),
        '--input-dir', str(interp_dir),
        '--output-dir', str(sanitized_dir),
        '--data-root', '/gemini/code/datasets',
        '--benchmark', 'MOT20',
    ], OUT_ROOT / 'logs' / 'sanitize_interp.log')
    check_results_dir(sanitized_dir, 'interp_precheck_results_dir.log')
    interp_zip = OUT_ROOT / 'MOT20_parambest_v2_seqspecial_high03_new04_0608_interp.zip'
    zip_results(sanitized_dir, interp_zip, 'interp_zip.log')

    manifest = {
        'status': 'completed',
        'phase': 'PARAMBEST_V2_SEQSPECIAL_MOT20_TEST',
        'config': {
            'MOT20-04': {'track_high_thresh': 0.6, 'new_track_thresh': 0.5},
            'MOT20-06': {'track_high_thresh': 0.3, 'new_track_thresh': 0.4},
            'MOT20-07': {'track_high_thresh': 0.6, 'new_track_thresh': 0.5},
            'MOT20-08': {'track_high_thresh': 0.3, 'new_track_thresh': 0.4},
            'common': {'assoc_mode': 'botsort_reid', 'track_low_thresh': 0.1, 'track_buffer': 70, 'match_thresh': 0.5},
            'disabled_modules': ['DMM', 'GateV1', 'OC-SORT', 'ExtendedProximity', 'Stage2', 'NSA', 'GMC'],
        },
        'phase0_root': str(PHASE0),
        'raw_results_dir': str(raw_dir),
        'raw_zip': str(raw_zip),
        'interp_results_dir': str(sanitized_dir),
        'interp_zip': str(interp_zip),
        'finished_at': now(),
    }
    (OUT_ROOT / 'run_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
