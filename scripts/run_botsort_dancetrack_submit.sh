#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/gemini/code/FMtrack-main/FM-Track"
BOT_ROOT="${REPO_ROOT}/external/BoT-SORT-main"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"

VARIANT="${1:-base}"         # base | meanrel | ltra | ...
SPLIT="${SPLIT:-test}"       # DanceTrack test split has no GT; used for submission
DATA_ROOT="${DATA_ROOT:-/gemini/code/datasets/DanceTrack/extracted}"

OUT_DIR="${2:-${REPO_ROOT}/outputs/dancetrack_submit/${SPLIT}_${VARIANT}_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="$(realpath -m "${OUT_DIR}")"
mkdir -p "${OUT_DIR}"

echo "[run] DanceTrack tracking: variant=${VARIANT} split=${SPLIT}"
echo "[run] data_root=${DATA_ROOT}" | tee "${OUT_DIR}/meta.txt"

cd "${REPO_ROOT}"

# Run BoT-SORT tracking (writes per-sequence txt files).
env DATA_ROOT="${DATA_ROOT}" SPLIT="${SPLIT}" PYTHONUNBUFFERED=1 \
  bash scripts/run_botsort_dancetrack_val.sh "${VARIANT}" --fp16 | tee "${OUT_DIR}/track.log"

# Infer results dir from naming convention in scripts/run_botsort_dancetrack_val.sh
EXP_NAME="botsort_dancetrack_${SPLIT}_${VARIANT}"
RESULTS_DIR="${BOT_ROOT}/YOLOX_outputs/${EXP_NAME}/track_results"

if [[ ! -d "${RESULTS_DIR}" ]]; then
  echo "[ERR] missing results dir: ${RESULTS_DIR}" >&2
  exit 2
fi

ZIP_PATH="${OUT_DIR}/dancetrack_${EXP_NAME}_submission.zip"

echo "[run] pack submission zip: ${ZIP_PATH}"
"${PYTHON_BIN}" scripts/make_dancetrack_submission_zip.py \
  --results-dir "${RESULTS_DIR}" \
  --data-root "${DATA_ROOT}" \
  --split "${SPLIT}" \
  --out-zip "${ZIP_PATH}" \
  --overwrite | tee "${OUT_DIR}/pack.log"

echo "[OK] results_dir=${RESULTS_DIR}"
echo "[OK] zip=${ZIP_PATH}"

