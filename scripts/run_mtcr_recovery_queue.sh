#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
QUEUE_ROOT="${1:-${REPO_ROOT}/outputs/mtcr_recovery_queue_${STAMP}}"
AFTER_PID="${AFTER_PID:-}"
POLL_SECONDS="${POLL_SECONDS:-60}"

mkdir -p "${QUEUE_ROOT}"
QUEUE_LOG="${QUEUE_ROOT}/queue.log"
QUEUE_STATUS="${QUEUE_ROOT}/queue_status.tsv"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${QUEUE_LOG}"
}

run_job() {
  local job_name="$1"
  shift
  local run_root="${QUEUE_ROOT}/${job_name}"
  local status="ok"

  mkdir -p "${run_root}"
  log "job=${job_name} start run_root=${run_root}"
  if env "$@" bash "${REPO_ROOT}/scripts/run_mtcr_recovery_pipeline.sh" "${run_root}" >> "${run_root}/launcher.log" 2>&1; then
    log "job=${job_name} complete"
  else
    status="failed"
    log "job=${job_name} failed; continuing to next queued job"
  fi
  printf "%s\t%s\t%s\n" "$(date '+%F %T')" "${job_name}" "${status}" | tee -a "${QUEUE_STATUS}"
}

wait_for_after_pid() {
  if [[ -z "${AFTER_PID}" ]]; then
    return 0
  fi
  if ! kill -0 "${AFTER_PID}" 2>/dev/null; then
    log "after_pid=${AFTER_PID} is not running; queue starts immediately"
    return 0
  fi
  log "waiting for after_pid=${AFTER_PID} to finish before starting queued jobs"
  while kill -0 "${AFTER_PID}" 2>/dev/null; do
    sleep "${POLL_SECONDS}"
  done
  log "after_pid=${AFTER_PID} finished; starting queued jobs"
}

log "queue_root=${QUEUE_ROOT}"
printf "timestamp\tjob\tstatus\n" > "${QUEUE_STATUS}"

wait_for_after_pid

run_job \
  "bgsafe_v1" \
  EPOCHS=15 \
  PATIENCE=5 \
  RUN_SW_YOLOX=1 \
  RUN_SGT=0 \
  RUN_PROXY=1 \
  RUN_FULL7=1 \
  AMBIGUOUS_WEIGHT=2.0 \
  EASY_WEIGHT=0.35 \
  BACKGROUND_WEIGHT=1.25 \
  LOSS_SAFE_WEIGHT=0.30 \
  LOSS_DUEL_WEIGHT=0.50 \
  LOSS_BG_WEIGHT=1.00 \
  LOSS_GATE_WEIGHT=0.20 \
  DELTA_SCALE=0.80 \
  GATE_CAP=0.15 \
  BG_SCALE=0.85 \
  SELECT_BG_WEIGHT=0.15 \
  SELECT_EASY_WEIGHT=0.08

run_job \
  "bgsafe_v2" \
  EPOCHS=15 \
  PATIENCE=5 \
  RUN_SW_YOLOX=1 \
  RUN_SGT=0 \
  RUN_PROXY=1 \
  RUN_FULL7=1 \
  AMBIGUOUS_WEIGHT=2.5 \
  EASY_WEIGHT=0.30 \
  BACKGROUND_WEIGHT=1.50 \
  LOSS_SAFE_WEIGHT=0.35 \
  LOSS_DUEL_WEIGHT=0.60 \
  LOSS_BG_WEIGHT=1.25 \
  LOSS_GATE_WEIGHT=0.25 \
  DELTA_SCALE=0.70 \
  GATE_CAP=0.12 \
  BG_SCALE=0.90 \
  SELECT_BG_WEIGHT=0.18 \
  SELECT_EASY_WEIGHT=0.10

log "queue complete"
