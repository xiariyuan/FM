#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/gemini/code/FMtrack-main/FM-Track"
cd "${REPO_ROOT}"

WAIT_SUMMARY="${1:?usage: chain_stage2_after_summary.sh <wait_summary_csv> <next_dataset> [poll_seconds]}"
NEXT_DATASET="${2:?usage: chain_stage2_after_summary.sh <wait_summary_csv> <next_dataset> [poll_seconds]}"
POLL_SECONDS="${3:-120}"

LOG_DIR="outputs/botsort_ltra_stage2/chain_logs"
mkdir -p "${LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/${NEXT_DATASET}_after_$(basename "$(dirname "${WAIT_SUMMARY}")")_${STAMP}.log"

{
  echo "[chain] waiting for summary: ${WAIT_SUMMARY}"
  echo "[chain] next dataset: ${NEXT_DATASET}"
  while [[ ! -f "${WAIT_SUMMARY}" ]]; do
    sleep "${POLL_SECONDS}"
  done
  echo "[chain] detected summary: ${WAIT_SUMMARY}"
  echo "[chain] launching stage2 for ${NEXT_DATASET}"
  bash "scripts/run_botsort_ltra_stage2.sh" "${NEXT_DATASET}"
  echo "[chain] stage2 for ${NEXT_DATASET} finished"
} >> "${LOG_FILE}" 2>&1

echo "[chain] log=${LOG_FILE}"
