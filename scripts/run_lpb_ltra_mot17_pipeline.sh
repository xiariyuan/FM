#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/outputs/lpb_ltra_pipeline_${TS}}"
RUN_ROOT="$(realpath -m "${RUN_ROOT}")"
LOG_PATH="${RUN_ROOT}/pipeline.log"

mkdir -p "${RUN_ROOT}"
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "[start] $(date '+%F %T %z')"
echo "[run_root] ${RUN_ROOT}"

GT_ROOT="${RUN_ROOT}/gt_builder"
TRAIN_ROOT="${RUN_ROOT}/train"
EVAL_ROOT="${RUN_ROOT}/eval"
SKIP_EVAL="${SKIP_EVAL:-0}"
SKIP_BUILD="${SKIP_BUILD:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

if [[ "${SKIP_BUILD}" != "1" ]]; then
  echo "[step] build GT pseudo-track train/val shards"
  OUT_ROOT="${GT_ROOT}" bash "${REPO_ROOT}/scripts/build_mot17_gt_pseudotrack_train_val.sh"
else
  echo "[skip] GT build disabled via SKIP_BUILD=1"
fi

TRAIN_SHARDS_DIR="${GT_ROOT}/train_shards"
VAL_SHARDS_DIR="${GT_ROOT}/val_shards"
TRAIN_LIST="${GT_ROOT}/train_npz_list.txt"
VAL_LIST="${GT_ROOT}/val_npz_list.txt"
CALIB_NPZ="${TRAIN_ROOT}/mot17_lpb_ltra_${TS}.npz"

if [[ "${SKIP_TRAIN}" != "1" ]]; then
  echo "[step] train learnable pole-bank LTRA"
  DATA_ROOT="${GT_ROOT}" \
  TRAIN_SHARDS_DIR="${TRAIN_SHARDS_DIR}" \
  VAL_SHARDS_DIR="${VAL_SHARDS_DIR}" \
  TRAIN_LIST="${TRAIN_LIST}" \
  VAL_LIST="${VAL_LIST}" \
  OUT_ROOT="${TRAIN_ROOT}" \
  OUT_NPZ="${CALIB_NPZ}" \
  bash "${REPO_ROOT}/scripts/train_lpb_ltra_mot17.sh"
else
  echo "[skip] train disabled via SKIP_TRAIN=1"
fi

if [[ "${SKIP_EVAL}" != "1" ]]; then
  echo "[step] run same-base BoT-SORT eval"
  RUN_ROOT="${EVAL_ROOT}" \
  CALIBRATOR_NPZ="${CALIB_NPZ}" \
  bash "${REPO_ROOT}/scripts/run_botsort_lpb_ltra_eval.sh"
else
  echo "[skip] eval disabled via SKIP_EVAL=1"
fi

echo "[done] $(date '+%F %T %z')"
echo "[artifacts] gt_root=${GT_ROOT}"
echo "[artifacts] calibrator=${CALIB_NPZ}"
if [[ "${SKIP_EVAL}" != "1" ]]; then
  echo "[artifacts] eval_summary=${EVAL_ROOT}/eval/mot17_summary.csv"
fi
