#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/gemini/code/FMtrack-main/FM-Track"
cd "${REPO_ROOT}"

TARGET="${1:-all}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="outputs/botsort_ltra_stage1/bg_logs"
mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/${TARGET}_${STAMP}.log"

setsid bash -lc "cd '${REPO_ROOT}' && bash scripts/run_botsort_ltra_stage1.sh '${TARGET}'" \
  > "${LOG_PATH}" 2>&1 < /dev/null &

PID=$!
echo "PID=${PID}"
echo "LOG=${LOG_PATH}"
