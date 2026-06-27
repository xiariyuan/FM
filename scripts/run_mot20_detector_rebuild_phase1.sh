#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BOT_ROOT="${REPO_ROOT}/external/BoT-SORT-main"
OUT_ROOT="${REPO_ROOT}/outputs/detector_rebuild"
mkdir -p "${OUT_ROOT}"
cd "${BOT_ROOT}"
export PYTHONPATH="${BOT_ROOT}:${PYTHONPATH:-}"
export MOT20_DET_MAX_EPOCH="${MOT20_DET_MAX_EPOCH:-20}"
export MOT20_DET_EVAL_INTERVAL="${MOT20_DET_EVAL_INTERVAL:-2}"
export MOT20_DET_WORKERS="${MOT20_DET_WORKERS:-4}"
unset MOT20_DET_SMOKE_TRAIN_IMAGES MOT20_DET_SMOKE_VAL_IMAGES || true
python -u yolox/train.py \
  -f yolox/exps/example/mot/yolox_x_mot20_rebuild_trainhalf.py \
  -d "${MOT20_DET_DEVICES:-1}" \
  -b "${MOT20_DET_BATCH:-1}" \
  -c pretrained/bytetrack_x_mot20.pth.tar \
  --experiment-name "${MOT20_DET_EXPN:-MOT20_DET_REBUILD_PHASE1}"
