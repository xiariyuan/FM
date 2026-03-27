#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/gemini/code/FMtrack-main/FM-Track"
cd "${REPO_ROOT}"

export LD_LIBRARY_PATH="/root/miniconda3/lib:/root/miniconda3/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
DATA_ROOT="${DATA_ROOT:-/gemini/code/datasets}"
CFG="${CFG:-configs/experiments/bytetrack_fa_mot_mot17_v15_laplace_reid_da_val0213.yaml}"
EPOCHS="${EPOCHS:-6}"
BATCH_SIZE="${BATCH_SIZE:-1}"
ACCUMULATE_STEPS="${ACCUMULATE_STEPS:-6}"

TS="$(date +%Y%m%d_%H%M%S)"
EXP_NAME="${EXP_NAME:-bytetrack_fa_mot_mot17_v15_laplace_proxy0213_${TS}}"
OUT_DIR="${OUT_DIR:-outputs/${EXP_NAME}}"

mkdir -p "${OUT_DIR}"
echo "${OUT_DIR}" > outputs/latest_v15_laplace_proxy_dir.txt

echo "[train] cfg=${CFG}"
echo "[train] out_dir=${OUT_DIR}"
echo "[train] epochs=${EPOCHS} batch_size=${BATCH_SIZE} accumulate_steps=${ACCUMULATE_STEPS}"

"${PYTHON_BIN}" -u train_bytetrack.py \
  --config-path "${CFG}" \
  --data-root "${DATA_ROOT}" \
  --outputs-dir "${OUT_DIR}" \
  --exp-name "${EXP_NAME}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --accumulate-steps "${ACCUMULATE_STEPS}" \
  2>&1 | tee "${OUT_DIR}/train.log"

BEST_TXT="$("${PYTHON_BIN}" -u scripts/select_best_bytetrack_ckpt.py --exp-dir "${OUT_DIR}" --metric HOTA --dataset MOT17 --split train)"
echo "${BEST_TXT}" | tee "${OUT_DIR}/best_ckpt.txt"

echo "[done] ${OUT_DIR}"
