#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"

TARGET="${1:-all}"
SCOPE="${2:-full7}"
BASE_OUT_ROOT="${3:-${REPO_ROOT}/outputs}"

if [[ "${SCOPE}" != "proxy0213" && "${SCOPE}" != "full7" ]]; then
  echo "Unsupported scope=${SCOPE}" >&2
  exit 2
fi

declare -a DETECTORS=()
if [[ "${TARGET}" == "all" ]]; then
  DETECTORS=(sw_yolox sgt)
elif [[ "${TARGET}" == "sw_yolox" || "${TARGET}" == "sgt" ]]; then
  DETECTORS=("${TARGET}")
else
  echo "Unsupported target=${TARGET}; expected sw_yolox|sgt|all" >&2
  exit 2
fi

for detector in "${DETECTORS[@]}"; do
  for mode in base heuristic; do
    out_dir="${BASE_OUT_ROOT}/mot17_external_${detector}_${mode}_${SCOPE}_$(date +%Y%m%d_%H%M%S)"
    echo "[batch] detector=${detector} mode=${mode} scope=${SCOPE}"
    bash "${REPO_ROOT}/scripts/run_bytetrack_external_ctrl.sh" "${detector}" "${mode}" "${SCOPE}" "${out_dir}"
  done
done
