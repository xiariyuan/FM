#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BOT_ROOT="${REPO_ROOT}/external/BoT-SORT-main"
OUT_ROOT="${REPO_ROOT}/outputs/detector_rebuild"
mkdir -p "${OUT_ROOT}"
cd "${BOT_ROOT}"
export PYTHONPATH="${BOT_ROOT}:${PYTHONPATH:-}"
export MOT20_DET_MAX_EPOCH="${MOT20_DET_MAX_EPOCH:-1}"
export MOT20_DET_EVAL_INTERVAL="${MOT20_DET_EVAL_INTERVAL:-1}"
export MOT20_DET_WORKERS="${MOT20_DET_WORKERS:-0}"
export MOT20_DET_SMOKE_TRAIN_IMAGES="${MOT20_DET_SMOKE_TRAIN_IMAGES:-16}"
export MOT20_DET_SMOKE_VAL_IMAGES="${MOT20_DET_SMOKE_VAL_IMAGES:-8}"
export MOT20_DET_INPUT_H="${MOT20_DET_INPUT_H:-384}"
export MOT20_DET_INPUT_W="${MOT20_DET_INPUT_W:-672}"
export MOT20_DET_TEST_H="${MOT20_DET_TEST_H:-384}"
export MOT20_DET_TEST_W="${MOT20_DET_TEST_W:-672}"
python -u yolox/train.py \
  -f yolox/exps/example/mot/yolox_x_mot20_rebuild_trainhalf.py \
  -d 1 \
  -b "${MOT20_DET_BATCH:-2}" \
  -c pretrained/bytetrack_x_mot20.pth.tar \
  --experiment-name MOT20_DET_REBUILD_SMOKE
