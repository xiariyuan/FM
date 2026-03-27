#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"

DETECTOR="${1:?detector is required: sw_yolox|sgt}"
MODE="${2:?mode is required: base|heuristic}"
SCOPE="${3:-full7}"
OUT_DIR="${4:-${REPO_ROOT}/outputs/mot17_external_${DETECTOR}_${MODE}_${SCOPE}_$(date +%Y%m%d_%H%M%S)}"

if [[ "${DETECTOR}" != "sw_yolox" && "${DETECTOR}" != "sgt" ]]; then
  echo "Unsupported detector=${DETECTOR}" >&2
  exit 2
fi
if [[ "${MODE}" != "base" && "${MODE}" != "heuristic" ]]; then
  echo "Unsupported mode=${MODE}" >&2
  exit 2
fi
if [[ "${SCOPE}" != "proxy0213" && "${SCOPE}" != "full7" ]]; then
  echo "Unsupported scope=${SCOPE}" >&2
  exit 2
fi

PROFILE="mot17_external_${DETECTOR}_${MODE}_${SCOPE}"
mkdir -p "${OUT_DIR}"

echo "[run] profile=${PROFILE}"
echo "[run] out_dir=${OUT_DIR}"

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/run_bytetrack_profile.py" \
  --exp-profile "${PROFILE}" \
  --out-dir "${OUT_DIR}"
