#!/usr/bin/env bash
# Run the SPOT v0 MOT20-05 three-condition audit with BoT-SORT ReID enabled.
# Conditions:
#   00 baseline:      no SPOT flags
#   01 observe-only: --spot-enable, must be byte-identical to baseline
#   02 freeze-app:   --spot-enable --spot-freeze-app, may change outputs
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BOT_ROOT="${REPO_ROOT}/external/BoT-SORT-main"
DATA_ROOT="${DATA_ROOT:-/gemini/code/datasets/MOT20}"
DEFAULT_RUN_ID="$(git -C "${REPO_ROOT}" rev-parse --short HEAD)"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs/spot_runtime/mot20_05_reid_${DEFAULT_RUN_ID}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MARGIN_THRESH="${SPOT_MARGIN_THRESH:-0.05}"

REID_CFG="${REID_CFG:-fast_reid/configs/MOT20/sbs_S50.yml}"
REID_WTS="${REID_WTS:-pretrained/mot20_sbs_S50.pth}"

BASE_EXP="SPOT_REID_BASELINE_MOT20_05"
OBS_EXP="SPOT_REID_OBSERVE_MOT20_05"
FRZ_EXP="SPOT_REID_FREEZE_MOT20_05"

mkdir -p "${OUT_ROOT}"
cd "${REPO_ROOT}"

{
  echo "START $(date -Iseconds)"
  echo "REPO_ROOT=${REPO_ROOT}"
  echo "BOT_ROOT=${BOT_ROOT}"
  echo "DATA_ROOT=${DATA_ROOT}"
  echo "OUT_ROOT=${OUT_ROOT}"
  echo "GIT_SHA=$(git rev-parse HEAD)"
  echo "REID_CFG=${REID_CFG}"
  echo "REID_WTS=${REID_WTS}"
  echo "SPOT_MARGIN_THRESH=${MARGIN_THRESH}"
} | tee "${OUT_ROOT}/status.txt"

cd "${BOT_ROOT}"
export PYTHONPATH="${BOT_ROOT}:${PYTHONPATH:-}"

COMMON_ARGS=(
  "${DATA_ROOT}"
  --benchmark MOT20
  --eval train
  --seq-ids 5
  --default-parameters
  --with-reid
  --fast-reid-config "${REID_CFG}"
  --fast-reid-weights "${REID_WTS}"
  --device gpu
)

"${PYTHON_BIN}" -u tools/track.py \
  "${COMMON_ARGS[@]}" \
  --experiment-name "${BASE_EXP}" \
  > "${OUT_ROOT}/00_baseline.log" 2>&1

echo "BASELINE_DONE $(date -Iseconds)" | tee -a "${OUT_ROOT}/status.txt"

"${PYTHON_BIN}" -u tools/track.py \
  "${COMMON_ARGS[@]}" \
  --experiment-name "${OBS_EXP}" \
  --spot-enable \
  --spot-debug-dir "${OUT_ROOT}/01_spot_observe/spot_debug" \
  > "${OUT_ROOT}/01_observe.log" 2>&1

echo "OBSERVE_DONE $(date -Iseconds)" | tee -a "${OUT_ROOT}/status.txt"

"${PYTHON_BIN}" -u tools/track.py \
  "${COMMON_ARGS[@]}" \
  --experiment-name "${FRZ_EXP}" \
  --spot-enable \
  --spot-freeze-app \
  --spot-margin-thresh "${MARGIN_THRESH}" \
  --spot-debug-dir "${OUT_ROOT}/02_spot_freeze_app/spot_debug" \
  > "${OUT_ROOT}/02_freeze.log" 2>&1

echo "FREEZE_DONE $(date -Iseconds)" | tee -a "${OUT_ROOT}/status.txt"

cd "${REPO_ROOT}"
"${PYTHON_BIN}" -u scripts/run_spot_parity_audit.py \
  --out-root "${OUT_ROOT}/audit" \
  --baseline-results-dir "${BOT_ROOT}/YOLOX_outputs/${BASE_EXP}/track_results" \
  --observe-results-dir "${BOT_ROOT}/YOLOX_outputs/${OBS_EXP}/track_results" \
  --freeze-results-dir "${BOT_ROOT}/YOLOX_outputs/${FRZ_EXP}/track_results" \
  --observe-spot-debug-dir "${OUT_ROOT}/01_spot_observe/spot_debug" \
  --freeze-spot-debug-dir "${OUT_ROOT}/02_spot_freeze_app/spot_debug" \
  > "${OUT_ROOT}/audit.log" 2>&1

echo "AUDIT_DONE $(date -Iseconds)" | tee -a "${OUT_ROOT}/status.txt"
echo "DONE $(date -Iseconds)" > "${OUT_ROOT}/DONE"
