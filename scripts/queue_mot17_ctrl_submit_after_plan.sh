#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
PLAN_CSV="${EXPERIMENT_PLAN_CSV:-${REPO_ROOT}/outputs/experiment_plan.csv}"
REGISTRY_CSV="${EXPERIMENT_REGISTRY_CSV:-${REPO_ROOT}/outputs/experiment_registry.csv}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
SCRIPT_NAME="${SCRIPT_NAME:-scripts/queue_mot17_ctrl_submit_after_plan.sh}"

WAIT_PLAN_KEY="${WAIT_PLAN_KEY:-}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/outputs/mot17_ctrl_submit_after_${TS}}"
OUT_DIR="$(realpath -m "${OUT_DIR}")"
LOG_PATH="${OUT_DIR}/queue.log"
PLAN_KEY="${PLAN_KEY:-submit_ctrl:${OUT_DIR}}"
POLL_SECS="${POLL_SECS:-120}"
TIMEOUT_SECS="${TIMEOUT_SECS:-172800}"
GPU_IDLE_MEM_MB="${GPU_IDLE_MEM_MB:-200}"

if [[ -z "${WAIT_PLAN_KEY}" ]]; then
  echo "WAIT_PLAN_KEY is required" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"
exec > >(tee -a "${LOG_PATH}") 2>&1

update_plan_status() {
  local status="$1"
  shift || true
  local extras=(
    "wait_plan_key=${WAIT_PLAN_KEY}"
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
    --dataset MOT17 \
    --split test \
    --tracker-family BoT-SORT \
    --variant mot17_ctrl_submit \
    --run-root "${OUT_DIR}" \
    --log-path "${LOG_PATH}" \
    --notes "queue_base_and_heuristic_test_packages_after_safe_runs" \
    --extra "${extras[@]}"
}

append_registry_record() {
  local variant="$1"
  local run_root="$2"
  local zip_path="$3"
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/append_experiment_record.py" \
    --csv "${REGISTRY_CSV}" \
    --kind other \
    --status success \
    --script "${SCRIPT_NAME}" \
    --dataset MOT17 \
    --split test \
    --tracker-family BoT-SORT \
    --variant "${variant}" \
    --run-root "${run_root}" \
    --log-path "${LOG_PATH}" \
    --notes "queued_control_submit_package" \
    --extra \
      zip_path="${zip_path}" \
      wait_plan_key="${WAIT_PLAN_KEY}"
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

plan_status() {
  "${PYTHON_BIN}" - <<'PY' "${PLAN_CSV}" "$1"
import csv
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
plan_key = sys.argv[2]
if not csv_path.is_file():
    print("")
    raise SystemExit(0)
with csv_path.open("r", newline="") as f:
    for row in csv.DictReader(f):
        if row.get("plan_key", "") == plan_key:
            print(row.get("status", ""))
            raise SystemExit(0)
print("")
PY
}

wait_for_plan() {
  local start_ts now_ts elapsed status
  start_ts="$(date +%s)"
  echo "[ctrl-submit] waiting for ${WAIT_PLAN_KEY}"
  while true; do
    status="$(plan_status "${WAIT_PLAN_KEY}" || true)"
    if [[ "${status}" == "completed" || "${status}" == "failed" || "${status}" == "cancelled" ]]; then
      echo "[ctrl-submit] wait plan terminal status=${status}"
      return 0
    fi
    now_ts="$(date +%s)"
    elapsed=$((now_ts - start_ts))
    if (( elapsed > TIMEOUT_SECS )); then
      echo "[ctrl-submit] timeout waiting for ${WAIT_PLAN_KEY}" >&2
      return 1
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
  echo "[ctrl-submit] waiting for gpu idle"
  while ! is_gpu_idle; do
    now_ts="$(date +%s)"
    elapsed=$((now_ts - start_ts))
    if (( elapsed > TIMEOUT_SECS )); then
      echo "[ctrl-submit] timeout waiting for gpu idle" >&2
      return 1
    fi
    sleep "${POLL_SECS}"
  done
  echo "[ctrl-submit] gpu idle"
}

run_mode() {
  local mode="$1"
  local run_root="${OUT_DIR}/${mode}"
  mkdir -p "${run_root}"
  echo "[ctrl-submit] run mode=${mode} root=${run_root}"
  bash "${REPO_ROOT}/scripts/run_botsort_ctrl_submit.sh" "${mode}" MOT17 "${run_root}"
  local zip_path=""
  if [[ -f "${run_root}/latest_zip.txt" ]]; then
    zip_path="$(head -n 1 "${run_root}/latest_zip.txt" | tr -d '\r')"
  fi
  append_registry_record "mot17_${mode}_ctrl_submit" "${run_root}" "${zip_path}"
  echo "[ctrl-submit] mode=${mode} zip=${zip_path}"
}

echo "[ctrl-submit] $(date '+%F %T %z') start"
echo "[ctrl-submit] run_root=${OUT_DIR}"
wait_for_plan
wait_for_gpu_idle
update_plan_status running

run_mode base
run_mode heuristic

update_plan_status completed
echo "[ctrl-submit] $(date '+%F %T %z') done"
