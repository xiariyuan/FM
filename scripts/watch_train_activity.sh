#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"

PID=""
EXP_NAME=""
CONFIG_PATH=""
OUT_DIR=""
RESULT_CSV=""
SUMMARY_CSV=""
REGISTRY_CSV=""
LOG_PATH=""
SCRIPT_NAME="scripts/queue_paper_ctrl_mot17_val0213.sh"
DATASET="MOT17"
SPLIT="val0213_proxy"
TRACKER_FAMILY="ByteTrack"
VARIANT="paper_ctrl_host_control"
TAG="paper_ctrl_mot17_val0213"
POLL_SEC="${POLL_SEC:-60}"
STALL_SEC="${STALL_SEC:-2700}"
EXIT_GRACE_SEC="${EXIT_GRACE_SEC:-180}"
KILL_ON_STALL="${KILL_ON_STALL:-1}"
WATCHDOG_LOG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pid)
      PID="$2"
      shift 2
      ;;
    --exp-name)
      EXP_NAME="$2"
      shift 2
      ;;
    --config-path)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --result-csv)
      RESULT_CSV="$2"
      shift 2
      ;;
    --summary-csv)
      SUMMARY_CSV="$2"
      shift 2
      ;;
    --registry-csv)
      REGISTRY_CSV="$2"
      shift 2
      ;;
    --log-path)
      LOG_PATH="$2"
      shift 2
      ;;
    --script)
      SCRIPT_NAME="$2"
      shift 2
      ;;
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --split)
      SPLIT="$2"
      shift 2
      ;;
    --tracker-family)
      TRACKER_FAMILY="$2"
      shift 2
      ;;
    --variant)
      VARIANT="$2"
      shift 2
      ;;
    --tag)
      TAG="$2"
      shift 2
      ;;
    --poll-sec)
      POLL_SEC="$2"
      shift 2
      ;;
    --stall-sec)
      STALL_SEC="$2"
      shift 2
      ;;
    --exit-grace-sec)
      EXIT_GRACE_SEC="$2"
      shift 2
      ;;
    --kill-on-stall)
      KILL_ON_STALL="$2"
      shift 2
      ;;
    --watchdog-log)
      WATCHDOG_LOG="$2"
      shift 2
      ;;
    *)
      echo "[watchdog] unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${PID}" || -z "${EXP_NAME}" || -z "${CONFIG_PATH}" || -z "${OUT_DIR}" || -z "${RESULT_CSV}" || -z "${SUMMARY_CSV}" || -z "${REGISTRY_CSV}" || -z "${LOG_PATH}" ]]; then
  echo "[watchdog] missing required args" >&2
  exit 2
fi

if [[ -z "${WATCHDOG_LOG}" ]]; then
  WATCHDOG_LOG="${OUT_DIR}/watchdog.log"
fi
mkdir -p "$(dirname "${WATCHDOG_LOG}")"

log() {
  printf '[%s] [watchdog] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >> "${WATCHDOG_LOG}"
}

csv_status() {
  "${PYTHON_BIN}" - <<'PY' "${RESULT_CSV}" "${EXP_NAME}"
import csv
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
exp_name = sys.argv[2]
if not csv_path.is_file():
    print("")
    raise SystemExit(0)
with csv_path.open("r", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row.get("exp_name", "") == exp_name:
            print(row.get("status", ""))
            raise SystemExit(0)
print("")
PY
}

mark_failed_if_running() {
  local note="$1"
  "${PYTHON_BIN}" - <<'PY' "${RESULT_CSV}" "${SUMMARY_CSV}" "${EXP_NAME}" "${CONFIG_PATH}" "${OUT_DIR}" "${note}"
import csv
import sys
from pathlib import Path

result_csv = Path(sys.argv[1])
summary_csv = Path(sys.argv[2])
exp_name, config_path, out_dir, note = sys.argv[3:7]

def rewrite(csv_path: Path) -> bool:
    if not csv_path.is_file():
        return False
    changed = False
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or ["exp_name", "config_path", "out_dir", "best_epoch", "best_hota", "checkpoint", "status"]
        rows = []
        for row in reader:
            if row.get("exp_name", "") == exp_name and row.get("status", "") == "running":
                row = {
                    "exp_name": exp_name,
                    "config_path": config_path,
                    "out_dir": out_dir,
                    "best_epoch": row.get("best_epoch", ""),
                    "best_hota": row.get("best_hota", ""),
                    "checkpoint": row.get("checkpoint", ""),
                    "status": "failed",
                }
                changed = True
            rows.append(row)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return changed

changed = rewrite(result_csv)
rewrite(summary_csv)
print("changed=1" if changed else "changed=0")
PY

  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/append_experiment_record.py" \
    --csv "${REGISTRY_CSV}" \
    --kind train \
    --status failed \
    --script "${SCRIPT_NAME}" \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --tracker-family "${TRACKER_FAMILY}" \
    --variant "${VARIANT}" \
    --tag "${TAG}" \
    --run-root "${OUT_DIR}" \
    --summary-csv "${RESULT_CSV}" \
    --log-path "${LOG_PATH}" \
    --notes "${note}" \
    --extra exp_name="${EXP_NAME}" config_path="${CONFIG_PATH}" out_dir="${OUT_DIR}" >/dev/null 2>&1 || true
}

process_alive() {
  kill -0 "${PID}" 2>/dev/null
}

log_age_seconds() {
  if [[ ! -f "${LOG_PATH}" ]]; then
    echo 0
    return 0
  fi
  local now_ts
  local mtime_ts
  now_ts="$(date +%s)"
  mtime_ts="$(stat -c %Y "${LOG_PATH}" 2>/dev/null || echo 0)"
  echo $((now_ts - mtime_ts))
}

wait_and_finalize_missing() {
  local note="$1"
  log "pid=${PID} missing, grace=${EXIT_GRACE_SEC}s"
  sleep "${EXIT_GRACE_SEC}"
  local status
  status="$(csv_status || true)"
  if [[ "${status}" == "running" ]]; then
    log "mark failed: ${note}"
    mark_failed_if_running "${note}" >> "${WATCHDOG_LOG}" 2>&1 || true
  else
    log "skip mark after missing pid; current status=${status}"
  fi
}

handle_stall() {
  local age="$1"
  local note="watchdog_stall_no_log_update_age_${age}s_pid_${PID}"
  log "stale log detected age=${age}s"
  if [[ "${KILL_ON_STALL}" == "1" ]] && process_alive; then
    kill -TERM "${PID}" 2>/dev/null || true
    sleep 10
    if process_alive; then
      kill -KILL "${PID}" 2>/dev/null || true
    fi
  fi
  sleep "${EXIT_GRACE_SEC}"
  local status
  status="$(csv_status || true)"
  if [[ "${status}" == "running" ]]; then
    log "mark failed after stall: ${note}"
    mark_failed_if_running "${note}" >> "${WATCHDOG_LOG}" 2>&1 || true
  else
    log "skip stall mark; current status=${status}"
  fi
}

log "start pid=${PID} exp=${EXP_NAME} poll=${POLL_SEC}s stall=${STALL_SEC}s grace=${EXIT_GRACE_SEC}s"

while true; do
  status="$(csv_status || true)"
  if [[ "${status}" != "" && "${status}" != "running" ]]; then
    log "stop: status=${status}"
    exit 0
  fi

  if ! process_alive; then
    wait_and_finalize_missing "watchdog_process_missing_pid_${PID}"
    exit 0
  fi

  age="$(log_age_seconds)"
  if (( age > STALL_SEC )); then
    handle_stall "${age}"
    exit 0
  fi

  sleep "${POLL_SEC}"
done
