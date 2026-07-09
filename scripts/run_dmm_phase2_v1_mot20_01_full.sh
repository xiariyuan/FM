#!/usr/bin/env bash
set -euo pipefail
cd /gemini/code/FMtrack-main/FM-Track
python3 scripts/dmm_base_tracker.py \
  --dump-npz outputs/dmm_phase0_mot20_01/MOT20-01/dump_yolox_reid.npz \
  --seq MOT20-01 \
  --assoc-mode botsort_reid \
  --dmm-enable \
  --dmm-margin-thresh 0.03 \
  --dmm-min-track-age 20 \
  --dmm-max-triggers-per-frame 2 \
  --dmm-recovery-thresh 0.35 \
  --dmm-csv outputs/dmm_phase2_v1_mot20_01_reid_m003/dmm_events.csv \
  --out outputs/dmm_phase2_v1_mot20_01_reid_m003/track_results/MOT20-01.txt
python3 scripts/eval_motstyle_trackeval.py \
  --benchmark-name MOT20 \
  --split-to-eval train \
  --gt-root /gemini/code/datasets/MOT20/train \
  --results-dir outputs/dmm_phase2_v1_mot20_01_reid_m003/track_results \
  --tracker-name dmm_phase2_v1_m003 \
  --work-dir outputs/dmm_phase2_v1_mot20_01_reid_m003/trackeval_work \
  --seqs MOT20-01
