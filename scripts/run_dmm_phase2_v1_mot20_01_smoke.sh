#!/usr/bin/env bash
set -euo pipefail
cd /gemini/code/FMtrack-main/FM-Track
python3 scripts/dmm_base_tracker.py \
  --dump-npz outputs/dmm_phase0_mot20_01/MOT20-01/dump_yolox_reid.npz \
  --seq MOT20-01 \
  --assoc-mode botsort_reid \
  --limit-frames 80 \
  --dmm-enable \
  --dmm-margin-thresh 0.03 \
  --dmm-min-track-age 20 \
  --dmm-max-triggers-per-frame 2 \
  --dmm-recovery-thresh 0.35 \
  --dmm-csv outputs/dmm_phase2_v1_mot20_01_reid_smoke/dmm_events.csv \
  --out outputs/dmm_phase2_v1_mot20_01_reid_smoke/track_results/MOT20-01.txt
