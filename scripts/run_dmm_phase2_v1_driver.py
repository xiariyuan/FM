#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path('/gemini/code/FMtrack-main/FM-Track')

tracker_cmd = [
    sys.executable,
    'scripts/dmm_base_tracker.py',
    '--dump-npz', 'outputs/dmm_phase0_mot20_01/MOT20-01/dump_yolox_reid.npz',
    '--seq', 'MOT20-01',
    '--assoc-mode', 'botsort_reid',
    '--dmm-enable',
    '--dmm-margin-thresh', '0.03',
    '--dmm-min-track-age', '20',
    '--dmm-max-triggers-per-frame', '2',
    '--dmm-recovery-thresh', '0.35',
    '--dmm-csv', 'outputs/dmm_phase2_v1_mot20_01_reid_m003/dmm_events.csv',
    '--out', 'outputs/dmm_phase2_v1_mot20_01_reid_m003/track_results/MOT20-01.txt',
]

eval_cmd = [
    sys.executable,
    'scripts/eval_motstyle_trackeval.py',
    '--benchmark-name', 'MOT20',
    '--split-to-eval', 'train',
    '--gt-root', '/gemini/code/datasets/MOT20/train',
    '--results-dir', 'outputs/dmm_phase2_v1_mot20_01_reid_m003/track_results',
    '--tracker-name', 'dmm_phase2_v1_m003',
    '--work-dir', 'outputs/dmm_phase2_v1_mot20_01_reid_m003/trackeval_work',
    '--seqs', 'MOT20-01',
]

print('[driver] running tracker')
subprocess.run(tracker_cmd, cwd=ROOT, check=True)
print('[driver] running TrackEval')
subprocess.run(eval_cmd, cwd=ROOT, check=True)
