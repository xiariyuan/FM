#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${1:-${REPO_ROOT}/outputs/mtcr_recovery_${STAMP}}"
RUN_SW_YOLOX="${RUN_SW_YOLOX:-1}"
RUN_SGT="${RUN_SGT:-1}"
RUN_PROXY="${RUN_PROXY:-1}"
RUN_FULL7="${RUN_FULL7:-1}"

mkdir -p "${RUN_ROOT}"

TRAIN_DIR="${RUN_ROOT}/train_mtcr"
bash "${REPO_ROOT}/scripts/train_mtcr_mot17.sh" "${TRAIN_DIR}"
MTCR_CHECKPOINT="$(cat "${TRAIN_DIR}/latest_checkpoint.txt")"

echo "[pipeline] checkpoint=${MTCR_CHECKPOINT}"

run_detector_scope() {
  local detector="$1"
  local scope="$2"
  local out_dir="${RUN_ROOT}/${detector}_${scope}"
  bash "${REPO_ROOT}/scripts/run_bytetrack_mtcr_external.sh" "${detector}" "${scope}" "${MTCR_CHECKPOINT}" "${out_dir}"
}

if [[ "${RUN_SW_YOLOX}" == "1" ]]; then
  if [[ "${RUN_PROXY}" == "1" ]]; then
    run_detector_scope "sw_yolox" "proxy0213"
  fi
  if [[ "${RUN_FULL7}" == "1" ]]; then
    run_detector_scope "sw_yolox" "full7"
  fi
fi

if [[ "${RUN_SGT}" == "1" ]]; then
  if [[ "${RUN_PROXY}" == "1" ]]; then
    run_detector_scope "sgt" "proxy0213"
  fi
  if [[ "${RUN_FULL7}" == "1" ]]; then
    run_detector_scope "sgt" "full7"
  fi
fi

echo "[pipeline] complete run_root=${RUN_ROOT}"
