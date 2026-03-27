#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/gemini/code/FMtrack-main/FM-Track"
LOG_DIR="${REPO_ROOT}/outputs/dancetrack_submit"
LOG_FILE="${LOG_DIR}/queue_test_submit.log"

EXTRACT_SESSION="${EXTRACT_SESSION:-extract_dancetrack_full}"
# Prefer waiting for an orchestrator session that may schedule GPU jobs after analysis.
WAIT_QUEUE_SESSION="${WAIT_QUEUE_SESSION:-queue_after_analysis}"
# Fallback: a single GPU-heavy session name to wait for.
WAIT_GPU_SESSION="${WAIT_GPU_SESSION:-analysis_mot20_val_seq05_meanrel}"

mkdir -p "${LOG_DIR}"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${LOG_FILE}"
}

log "queue_dancetrack_test_submit start"
log "waiting for extract session: ${EXTRACT_SESSION}"
while tmux has-session -t "${EXTRACT_SESSION}" 2>/dev/null; do
  sleep 20
done
log "extract session finished"

if tmux has-session -t "${WAIT_QUEUE_SESSION}" 2>/dev/null; then
  log "waiting for queue session to finish: ${WAIT_QUEUE_SESSION}"
  while tmux has-session -t "${WAIT_QUEUE_SESSION}" 2>/dev/null; do
    sleep 20
  done
  log "queue session finished"
else
  log "queue session not found; waiting for gpu session to finish: ${WAIT_GPU_SESSION}"
  while tmux has-session -t "${WAIT_GPU_SESSION}" 2>/dev/null; do
    sleep 20
  done
  log "gpu session finished"
fi

cd "${REPO_ROOT}"

log "start DanceTrack test submission runs"

# Two daily submissions: base vs best LTRA variant (mean+reliability).
bash scripts/run_botsort_dancetrack_submit.sh base "${LOG_DIR}/test_base_$(date +%Y%m%d_%H%M%S)" >> "${LOG_FILE}" 2>&1
log "base submission packaged"

bash scripts/run_botsort_dancetrack_submit.sh meanrel "${LOG_DIR}/test_meanrel_$(date +%Y%m%d_%H%M%S)" >> "${LOG_FILE}" 2>&1
log "meanrel submission packaged"

log "queue_dancetrack_test_submit finished"
