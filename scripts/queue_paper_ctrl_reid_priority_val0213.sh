#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"
cd "${REPO_ROOT}"

CONFIGS_OVERRIDE="${CONFIGS_OVERRIDE:-configs/experiments/bytetrack_fa_mot_mot17_v14_ctrl_base_reid_da_val0213.yaml,configs/experiments/bytetrack_fa_mot_mot17_v14_ctrl_full_reid_da_val0213.yaml}"
BATCH_SIZE="${BATCH_SIZE:-2}"
ACCUMULATE_STEPS="${ACCUMULATE_STEPS:-3}"

export CONFIGS_OVERRIDE
export BATCH_SIZE
export ACCUMULATE_STEPS

bash "${REPO_ROOT}/scripts/queue_paper_ctrl_mot17_val0213.sh"
