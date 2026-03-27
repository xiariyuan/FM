#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/gemini/code/FMtrack-main/FM-Track"
SS_ROOT="${REPO_ROOT}/external/StrongSORT-master"
DATA_ROOT="${DATA_ROOT:-/gemini/code/datasets}"
DET_ROOT="${DET_ROOT:-}"
OUT_ROOT="${OUT_ROOT:-${SS_ROOT}/results_laplace}"
DATASET="${1:-MOT17}"
MODE="${2:-val}"

if [[ -z "${DET_ROOT}" ]]; then
  echo "Please set DET_ROOT to StrongSORT prepared detection-feature directory, e.g. MOT17_val_YOLOX+BoT" >&2
  exit 2
fi

cd "${SS_ROOT}"
mkdir -p "${OUT_ROOT}/base" "${OUT_ROOT}/laplace"

COMMON_ARGS=(
  "${DATASET}"
  "${MODE}"
  --BoT
  --NSA
  --EMA
  --MC
  --woC
  --root_dataset "${DATA_ROOT}"
  --dir_dets "${DET_ROOT}"
)

ECC_PATH=""
if [[ "${DATASET}" == "MOT17" && "${MODE}" == "val" ]]; then
  ECC_PATH="${REPO_ROOT}/external/StrongSORT-master/MOT17_ECC_val.json"
elif [[ "${DATASET}" == "MOT17" && "${MODE}" == "test" ]]; then
  ECC_PATH="${REPO_ROOT}/external/StrongSORT-master/MOT17_ECC_test.json"
elif [[ "${DATASET}" == "MOT20" && "${MODE}" == "test" ]]; then
  ECC_PATH="${REPO_ROOT}/external/StrongSORT-master/MOT20_ECC_test.json"
fi

if [[ -n "${ECC_PATH}" && -f "${ECC_PATH}" ]]; then
  COMMON_ARGS+=(--ECC --path_ECC "${ECC_PATH}")
fi

python strong_sort.py "${COMMON_ARGS[@]}" --dir_save "${OUT_ROOT}/base"

python strong_sort.py "${COMMON_ARGS[@]}" \
  --dir_save "${OUT_ROOT}/laplace" \
  --LAPLACE \
  --laplace_weight 0.35 \
  --laplace_decay_scales 1 2 4 \
  --laplace_min_history 3

echo "[done] results in ${OUT_ROOT}"
