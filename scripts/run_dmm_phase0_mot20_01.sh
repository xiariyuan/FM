#!/usr/bin/env bash
set -euo pipefail

cd /gemini/code/FMtrack-main/FM-Track

python scripts/dump_yolox_reid_phase0.py \
  --seq-ids 1 \
  --split train \
  --out-root outputs/dmm_phase0_mot20_01 \
  --overwrite "$@"
