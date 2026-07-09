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
OUT = REPO / 'outputs' / 'dmm_parambest_v4_ultrarecall_mot20_test'
SRC_V3 = REPO / 'outputs' / 'dmm_parambest_v3_recall_mot20_test'
TRACKER = REPO / 'scripts' / 'dmm_base_tracker.py'
CHECK = REPO / 'scripts' / 'check_mot20_submission.py'
INTERP = REPO / 'scripts' / 'postprocess' / 'linear_interpolate_mot.py'
SANITIZE = REPO / 'scripts' / 'sanitize_mot_submission.py'
SEQS = ['MOT20-04', 'MOT20-06', 'MOT20-07', 'MOT20-08']


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec='seconds')


def run(cmd: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    print('[run]', ' '.join(cmd[:10]), '...', flush=True)
    with log.open('w', encoding='utf-8') as f:
        f.write('[started_at] ' + now() + '\n')
        f.write('[cmd] ' + ' '.join(cmd) + '\n\n')
        f.flush()
        p = subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT)
        f.write('\n[finished_at] ' + now() + '\n')
        f.write('[return_code] ' + str(p.returncode) + '\n')
    if p.returncode != 0:
        raise RuntimeError(f'Command failed rc={p.returncode}; see {log}')


def ensure_copied(seq: str) -> None:
    dst = OUT / 'track_results' / f'{seq}.txt'
    if dst.is_file():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    src = SRC_V3 / 'track_results' / f'{seq}.txt'
    if not src.is_file():
        raise FileNotFoundError(src)
    shutil.copy2(src, dst)


def run_tracker(seq: str) -> None:
    out_txt = OUT / 'track_results' / f'{seq}.txt'
    summary = OUT / f'{seq}_summary.json'
    if out_txt.is_file() and summary.is_file():
        print(f'[skip] {seq}', flush=True)
        return
    dump = PHASE0 / seq / 'dump_yolox_reid.npz'
    cmd = [
        sys.executable, str(TRACKER),
        '--dump-npz', str(dump),
        '--seq', seq,
        '--assoc-mode', 'botsort_reid',
        '--track-high-thresh', '0.2',
        '--track-low-thresh', '0.1',
        '--track-buffer', '70',
        '--match-thresh', '0.5',
        '--new-track-thresh', '0.3',
        '--out', str(out_txt),
        '--summary-json', str(summary),
    ]
    run(cmd, OUT / 'logs' / f'{seq}_tracker.log')


def precheck_dir(path: Path, name: str) -> None:
    run([sys.executable, str(CHECK), '--results-dir', str(path), '--profile', 'mot20_test_4'], OUT / 'logs' / name)


def zip_dir(path: Path, zip_path: Path, name: str) -> None:
    if zip_path.exists():
        zip_path.unlink()
    cmd = ['zip', '-j', str(zip_path)] + [str(path / f'{seq}.txt') for seq in SEQS]
    run(cmd, OUT / 'logs' / name)
    run([sys.executable, str(CHECK), '--zip-path', str(zip_path), '--profile', 'mot20_test_4'], OUT / 'logs' / name.replace('.log', '_precheck.log'))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ensure_copied('MOT20-04')
    ensure_copied('MOT20-07')
    run_tracker('MOT20-06')
    run_tracker('MOT20-08')

    raw_dir = OUT / 'track_results'
    precheck_dir(raw_dir, 'raw_precheck_results_dir.log')
    raw_zip = OUT / 'MOT20_parambest_v4_ultrarecall_high02_new03_0608_raw.zip'
    zip_dir(raw_dir, raw_zip, 'raw_zip.log')

    interp40 = OUT / 'interp40_results'
    run([
        sys.executable, str(INTERP),
        '--input-dir', str(raw_dir),
        '--output-dir', str(interp40),
        '--max-gap', '40',
        '--min-track-len', '2',
        '--min-endpoint-score', '0.0',
        '--max-area-ratio', '4.0',
        '--max-center-step', '100.0',
        '--min-box-side', '2.0',
        '--summary-json', str(OUT / 'interp40_summary.json'),
        '--summary-csv', str(OUT / 'interp40_summary.csv'),
    ], OUT / 'logs' / 'interp40.log')

    sanitized = OUT / 'sanitized_interp40_results'
    run([sys.executable, str(SANITIZE), '--input-dir', str(interp40), '--output-dir', str(sanitized), '--data-root', '/gemini/code/datasets', '--benchmark', 'MOT20'], OUT / 'logs' / 'sanitize_interp40.log')
    precheck_dir(sanitized, 'interp40_precheck_results_dir.log')
    interp_zip = OUT / 'MOT20_parambest_v4_ultrarecall_high02_new03_0608_interp40.zip'
    zip_dir(sanitized, interp_zip, 'interp40_zip.log')

    manifest = {
        'status': 'completed',
        'phase': 'PARAMBEST_V4_ULTRARECALL_MOT20_TEST',
        'config': {
            'MOT20-04': 'copied_v3',
            'MOT20-07': 'copied_v3',
            'MOT20-06': {'track_high_thresh': 0.2, 'new_track_thresh': 0.3},
            'MOT20-08': {'track_high_thresh': 0.2, 'new_track_thresh': 0.3},
            'common': {'assoc_mode': 'botsort_reid', 'track_low_thresh': 0.1, 'track_buffer': 70, 'match_thresh': 0.5},
        },
        'raw_zip': str(raw_zip),
        'interp40_zip': str(interp_zip),
        'finished_at': now(),
    }
    (OUT / 'run_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
