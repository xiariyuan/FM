#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/gemini/code/FMtrack-main/FM-Track"
SS_ROOT="${REPO_ROOT}/external/StrongSORT-master"
DATA_ROOT="${DATA_ROOT:-/gemini/code/datasets}"
DET_ROOT="${DET_ROOT:-${SS_ROOT}/MOT17_val_YOLOX+BoT}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs/strongsort_ltra/MOT17_val}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"

cd "${REPO_ROOT}"

if [[ ! -d "${DET_ROOT}" ]]; then
  echo "Missing DET_ROOT: ${DET_ROOT}" >&2
  exit 2
fi

if [[ ! -f "${SS_ROOT}/MOT17_ECC_val.json" ]]; then
  echo "Missing ECC file: ${SS_ROOT}/MOT17_ECC_val.json" >&2
  exit 2
fi

mkdir -p "${OUT_ROOT}"

RESULT_ROOT="${OUT_ROOT}/results"
BASE_DIR="${RESULT_ROOT}/base"
LTRA_DIR="${RESULT_ROOT}/laplace"
BASE_EVAL="${OUT_ROOT}/base_eval"
LTRA_EVAL="${OUT_ROOT}/laplace_eval"

env \
  DATA_ROOT="${DATA_ROOT}" \
  DET_ROOT="${DET_ROOT}" \
  OUT_ROOT="${RESULT_ROOT}" \
  bash scripts/run_strongsort_laplace_matrix.sh MOT17 val

"${PYTHON_BIN}" scripts/eval_botsort_halfval_trackeval.py \
  --dataset MOT17 \
  --data-root "${DATA_ROOT}" \
  --results-dir "${BASE_DIR}" \
  --tracker-name strongsort_mot17_val_base \
  --work-dir "${BASE_EVAL}"

"${PYTHON_BIN}" scripts/eval_botsort_halfval_trackeval.py \
  --dataset MOT17 \
  --data-root "${DATA_ROOT}" \
  --results-dir "${LTRA_DIR}" \
  --tracker-name strongsort_mot17_val_laplace \
  --work-dir "${LTRA_EVAL}"

"${PYTHON_BIN}" scripts/collect_trackeval_metrics.py \
  "${BASE_EVAL}/eval/strongsort_mot17_val_base" \
  "${LTRA_EVAL}/eval/strongsort_mot17_val_laplace" \
  --csv "${OUT_ROOT}/summary.csv" | tee "${OUT_ROOT}/summary.txt"

echo "[done] StrongSORT MOT17 val summary: ${OUT_ROOT}/summary.csv"
