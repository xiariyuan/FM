#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/gemini/code/FMtrack-main/FM-Track"
cd "${REPO_ROOT}"

DATASET="${1:-MOT20}"
LOG_DIR="outputs/botsort_ltra_stage2/bg_logs"
mkdir -p "${LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/${DATASET}_stage2_${STAMP}.log"

nohup bash scripts/run_botsort_ltra_stage2.sh "${DATASET}" > "${LOG_FILE}" 2>&1 < /dev/null &
PID=$!

echo "[launched] dataset=${DATASET}"
echo "[launched] pid=${PID}"
echo "[launched] log=${LOG_FILE}"
