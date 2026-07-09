#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path('/gemini/code/FMtrack-main/FM-Track')
cmd = [
    sys.executable,
    'scripts/dump_yolox_reid_phase0.py',
    '--benchmark', 'MOT20',
    '--split', 'test',
    '--seq-ids', '6',
    '--limit-frames', '1',
    '--out-root', 'outputs/dmm_phase0_mot20_test_parambest_smoke06',
    '--overwrite',
    '--no-csv',
]
subprocess.run(cmd, cwd=ROOT, check=True)
