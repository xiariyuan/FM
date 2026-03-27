#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/gemini/code/FMtrack-main/FM-Track"
cd "${REPO_ROOT}"

WATCH_PID="${1:?usage: watch_resume_stage2.sh <watch_pid> [dataset] [poll_seconds]}"
DATASET="${2:-MOT20}"
POLL_SECONDS="${3:-60}"
LOG_DIR="outputs/botsort_ltra_stage2/watchdog_logs"
mkdir -p "${LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/${DATASET}_resume_${STAMP}.log"

{
  echo "[watchdog] watching pid=${WATCH_PID} dataset=${DATASET}"
  while kill -0 "${WATCH_PID}" 2>/dev/null; do
    sleep "${POLL_SECONDS}"
  done
  echo "[watchdog] pid=${WATCH_PID} exited; resuming stage2"
  bash scripts/run_botsort_ltra_stage2.sh "${DATASET}"
  echo "[watchdog] stage2 resume finished"
} >> "${LOG_FILE}" 2>&1

echo "[watchdog] log=${LOG_FILE}"
