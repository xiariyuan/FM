#!/usr/bin/env bash
set -euo pipefail

cd /gemini/code/FMtrack-main/FM-Track

DUMP="outputs/dmm_phase0_mot20_01/MOT20-01/dump_yolox_reid.npz"

python scripts/dmm_base_tracker.py \
  --dump-npz "$DUMP" \
  --seq MOT20-01 \
  --assoc-mode botsort_reid \
  --debug-assoc \
  --debug-csv outputs/dmm_phase1_base_mot20_01_reid/assoc_debug/MOT20-01_primary_top3.csv \
  --out outputs/dmm_phase1_base_mot20_01_reid/track_results/MOT20-01.txt

python scripts/eval_motstyle_trackeval.py \
  --benchmark-name MOT20 \
  --split-to-eval train \
  --gt-root /gemini/code/datasets/MOT20/train \
  --results-dir outputs/dmm_phase1_base_mot20_01_reid/track_results \
  --tracker-name dmm_phase1_base_mot20_01_reid \
  --work-dir outputs/dmm_phase1_base_mot20_01_reid/trackeval_work \
  --seqs MOT20-01
