#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
PLAN_CSV="${EXPERIMENT_PLAN_CSV:-${REPO_ROOT}/outputs/experiment_plan.csv}"
REGISTRY_CSV="${EXPERIMENT_REGISTRY_CSV:-${REPO_ROOT}/outputs/experiment_registry.csv}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
SCRIPT_NAME="${SCRIPT_NAME:-scripts/queue_haca_submit_after_transfer.sh}"

HACA_CHECKPOINT="${HACA_CHECKPOINT:-}"
DATASET="${DATASET:-MOT17}"
HACA_MODE="${HACA_MODE:-haca_v3}"
HACA_LABEL="${HACA_LABEL:-$(basename "${HACA_CHECKPOINT:-checkpoint}" .npz)}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/outputs/haca_submit_${DATASET,,}_${HACA_LABEL}_${TS}}"
OUT_DIR="$(realpath -m "${OUT_DIR}")"
LOG_PATH="${LOG_PATH:-${OUT_DIR}/queue.log}"
PLAN_KEY="${PLAN_KEY:-submit:${OUT_DIR}}"

WAIT_FOR_MOT20_SUMMARY="${WAIT_FOR_MOT20_SUMMARY:-}"
WAIT_FOR_STRONGSORT_SUMMARY="${WAIT_FOR_STRONGSORT_SUMMARY:-}"
WAIT_FOR_STRONGSORT="${WAIT_FOR_STRONGSORT:-1}"
WAIT_FOR_PROCESS_PATTERN="${WAIT_FOR_PROCESS_PATTERN:-}"
WAIT_FOR_PROCESS_GRACE_SECS="${WAIT_FOR_PROCESS_GRACE_SECS:-900}"
POLL_SECS="${POLL_SECS:-60}"
TIMEOUT_SECS="${TIMEOUT_SECS:-43200}"
GPU_IDLE_MEM_MB="${GPU_IDLE_MEM_MB:-200}"

if [[ -z "${HACA_CHECKPOINT}" ]]; then
  echo "HACA_CHECKPOINT is required" >&2
  exit 1
fi
if [[ ! -f "${HACA_CHECKPOINT}" ]]; then
  echo "Missing HACA checkpoint: ${HACA_CHECKPOINT}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"
exec > >(tee -a "${LOG_PATH}") 2>&1

update_plan_status() {
  local status="$1"
  shift || true
  local extras=(
    "dataset=${DATASET}"
    "haca_mode=${HACA_MODE}"
    "wait_for_mot20_summary=${WAIT_FOR_MOT20_SUMMARY}"
    "wait_for_strongsort_summary=${WAIT_FOR_STRONGSORT_SUMMARY}"
    "wait_for_strongsort=${WAIT_FOR_STRONGSORT}"
    "wait_for_process_pattern=${WAIT_FOR_PROCESS_PATTERN}"
    "wait_for_process_grace_secs=${WAIT_FOR_PROCESS_GRACE_SECS}"
    "poll_secs=${POLL_SECS}"
    "gpu_idle_mem_mb=${GPU_IDLE_MEM_MB}"
  )
  if [[ $# -gt 0 ]]; then
    extras+=("$@")
  fi
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/upsert_experiment_plan.py" \
    --csv "${PLAN_CSV}" \
    --key "${PLAN_KEY}" \
    --status "${status}" \
    --kind other \
    --script "${SCRIPT_NAME}" \
    --dataset "${DATASET}" \
    --split test \
    --tracker-family BoT-SORT \
    --variant "${HACA_LABEL}_submit" \
    --run-root "${OUT_DIR}" \
    --checkpoint "${HACA_CHECKPOINT}" \
    --log-path "${LOG_PATH}" \
    --notes "auto_submit_after_transfer" \
    --extra "${extras[@]}"
}

append_registry_record() {
  local zip_path="$1"
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/append_experiment_record.py" \
    --csv "${REGISTRY_CSV}" \
    --kind other \
    --status success \
    --script "${SCRIPT_NAME}" \
    --dataset "${DATASET}" \
    --split test \
    --tracker-family BoT-SORT \
    --variant "${HACA_LABEL}_submit" \
    --run-root "${OUT_DIR}" \
    --checkpoint "${HACA_CHECKPOINT}" \
    --log-path "${LOG_PATH}" \
    --notes "auto_submit_after_transfer" \
    --extra \
      zip_path="${zip_path}" \
      haca_mode="${HACA_MODE}" \
      wait_for_mot20_summary="${WAIT_FOR_MOT20_SUMMARY}" \
      wait_for_strongsort_summary="${WAIT_FOR_STRONGSORT_SUMMARY}"
}

on_exit() {
  local rc=$?
  trap - EXIT
  if [[ ${rc} -ne 0 ]]; then
    update_plan_status failed "exit_code=${rc}" || true
  fi
  exit ${rc}
}

trap on_exit EXIT
update_plan_status queued

wait_for_file() {
  local label="$1"
  local path="$2"
  if [[ -z "${path}" ]]; then
    return 0
  fi
  local start_ts now_ts elapsed
  start_ts="$(date +%s)"
  echo "[submit-queue] waiting for ${label}: ${path}"
  while [[ ! -f "${path}" ]]; do
    now_ts="$(date +%s)"
    elapsed=$((now_ts - start_ts))
    if (( elapsed > TIMEOUT_SECS )); then
      echo "[submit-queue] timeout waiting for ${label}: ${path}" >&2
      return 1
    fi
    sleep "${POLL_SECS}"
  done
  echo "[submit-queue] detected ${label}: ${path}"
}

wait_for_summary_or_process_exit() {
  local label="$1"
  local path="$2"
  local process_pattern="$3"
  if [[ -z "${path}" ]]; then
    return 0
  fi
  if [[ -f "${path}" ]]; then
    echo "[submit-queue] detected ${label}: ${path}"
    return 0
  fi
  if [[ -z "${process_pattern}" ]]; then
    wait_for_file "${label}" "${path}"
    return 0
  fi

  local start_ts now_ts elapsed
  local process_seen=0
  start_ts="$(date +%s)"
  echo "[submit-queue] waiting for ${label} or process exit: ${path}"
  echo "[submit-queue] process pattern: ${process_pattern}"
  while true; do
    if [[ -f "${path}" ]]; then
      echo "[submit-queue] detected ${label}: ${path}"
      return 0
    fi
    if pgrep -f "${process_pattern}" >/dev/null 2>&1; then
      process_seen=1
    elif (( process_seen == 1 )); then
      echo "[submit-queue] process finished without ${label}; continuing with fallback"
      return 0
    fi

    now_ts="$(date +%s)"
    elapsed=$((now_ts - start_ts))
    if (( elapsed > TIMEOUT_SECS )); then
      echo "[submit-queue] timeout waiting for ${label} or process exit" >&2
      return 1
    fi
    if (( process_seen == 0 && elapsed > WAIT_FOR_PROCESS_GRACE_SECS )); then
      echo "[submit-queue] process did not appear within grace window; continuing without ${label}"
      return 0
    fi
    sleep "${POLL_SECS}"
  done
}

is_gpu_idle() {
  local query
  query="$(nvidia-smi --query-compute-apps=used_gpu_memory --format=csv,noheader,nounits 2>/dev/null || true)"
  if [[ -z "${query}" ]]; then
    return 0
  fi
  while IFS= read -r line; do
    [[ -z "${line}" ]] && continue
    local mem="${line%% *}"
    if [[ "${mem}" =~ ^[0-9]+$ ]] && (( mem > GPU_IDLE_MEM_MB )); then
      return 1
    fi
  done <<< "${query}"
  return 0
}

wait_for_gpu_idle() {
  local start_ts now_ts elapsed
  start_ts="$(date +%s)"
  echo "[submit-queue] waiting for gpu idle"
  while ! is_gpu_idle; do
    now_ts="$(date +%s)"
    elapsed=$((now_ts - start_ts))
    if (( elapsed > TIMEOUT_SECS )); then
      echo "[submit-queue] timeout waiting for gpu idle" >&2
      return 1
    fi
    sleep "${POLL_SECS}"
  done
  echo "[submit-queue] gpu idle"
}

echo "[submit-queue] $(date '+%F %T %z') start"
echo "[submit-queue] checkpoint=${HACA_CHECKPOINT}"
echo "[submit-queue] dataset=${DATASET}"
echo "[submit-queue] out_dir=${OUT_DIR}"

update_plan_status running

wait_for_file "mot20_summary" "${WAIT_FOR_MOT20_SUMMARY}"
if [[ "${WAIT_FOR_STRONGSORT}" == "1" ]]; then
  wait_for_summary_or_process_exit "strongsort_summary" "${WAIT_FOR_STRONGSORT_SUMMARY}" "${WAIT_FOR_PROCESS_PATTERN}"
fi
wait_for_gpu_idle

echo "[submit-queue] launching test submit packaging"
bash "${REPO_ROOT}/scripts/run_botsort_haca_submit.sh" \
  "${HACA_CHECKPOINT}" \
  "${DATASET}" \
  "${OUT_DIR}"

ZIP_PATH=""
if [[ -f "${OUT_DIR}/latest_zip.txt" ]]; then
  ZIP_PATH="$(head -n 1 "${OUT_DIR}/latest_zip.txt" | tr -d '\r')"
fi

update_plan_status completed "zip_path=${ZIP_PATH}"
append_registry_record "${ZIP_PATH}"

echo "[submit-queue] done $(date '+%F %T %z')"
echo "[submit-queue] zip=${ZIP_PATH}"
