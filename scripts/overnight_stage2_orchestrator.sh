#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/gemini/code/FMtrack-main/FM-Track"
cd "${REPO_ROOT}"

POLL_SECONDS="${1:-120}"
LOG_DIR="outputs/botsort_ltra_stage2/orchestrator_logs"
mkdir -p "${LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/overnight_${STAMP}.log"

mot20_summary="outputs/botsort_ltra_stage2/MOT20/summary.csv"
mot17_summary="outputs/botsort_ltra_stage2/MOT17/summary.csv"

is_running_dataset() {
  local dataset="$1"
  pgrep -f "run_botsort_ltra_stage2.sh ${dataset}|laplace_${dataset,,}_val_meanhist|laplace_${dataset,,}_val_single|laplace_${dataset,,}_val_multinorel|collect_trackeval_metrics.py .*botsort_ltra_stage2/${dataset}|eval_botsort_halfval_trackeval.py .*${dataset}" >/dev/null 2>&1
}

{
  echo "[orchestrator] start poll=${POLL_SECONDS}s"
  echo "[orchestrator] mot20_summary=${mot20_summary}"
  echo "[orchestrator] mot17_summary=${mot17_summary}"

  while [[ ! -f "${mot20_summary}" ]]; do
    if is_running_dataset "MOT20"; then
      echo "[orchestrator] MOT20 stage2 still running"
    else
      echo "[orchestrator] MOT20 summary missing and no active process; resuming MOT20 stage2"
      bash scripts/run_botsort_ltra_stage2.sh MOT20
    fi
    sleep "${POLL_SECONDS}"
  done

  echo "[orchestrator] MOT20 stage2 summary detected"

  if [[ -f "${mot17_summary}" ]]; then
    echo "[orchestrator] MOT17 summary already exists; exiting"
    exit 0
  fi

  if is_running_dataset "MOT17"; then
    echo "[orchestrator] MOT17 stage2 already running; waiting for summary"
  else
    echo "[orchestrator] launching MOT17 stage2"
    bash scripts/run_botsort_ltra_stage2.sh MOT17
  fi

  while [[ ! -f "${mot17_summary}" ]]; do
    if is_running_dataset "MOT17"; then
      echo "[orchestrator] MOT17 stage2 still running"
    else
      echo "[orchestrator] MOT17 summary missing and no active process; resuming MOT17 stage2"
      bash scripts/run_botsort_ltra_stage2.sh MOT17
    fi
    sleep "${POLL_SECONDS}"
  done

  echo "[orchestrator] MOT17 stage2 summary detected; overnight chain complete"
} >> "${LOG_FILE}" 2>&1

echo "[orchestrator] log=${LOG_FILE}"
