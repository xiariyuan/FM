#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
POLL_SEC="${POLL_SEC:-60}"

SUMMARY_CSV=""
PROCESS_PATTERN=""
APPEND_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --summary-csv)
      SUMMARY_CSV="$2"
      APPEND_ARGS+=("$1" "$2")
      shift 2
      ;;
    --process-pattern)
      PROCESS_PATTERN="$2"
      shift 2
      ;;
    --poll-sec)
      POLL_SEC="$2"
      shift 2
      ;;
    *)
      APPEND_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "${SUMMARY_CSV}" ]]; then
  echo "Missing --summary-csv" >&2
  exit 2
fi

while true; do
  if [[ -f "${SUMMARY_CSV}" ]]; then
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/append_experiment_record.py" "${APPEND_ARGS[@]}"
    exit 0
  fi
  if [[ -n "${PROCESS_PATTERN}" ]]; then
    if ! ps -eo cmd | grep -E "${PROCESS_PATTERN}" | grep -v grep >/dev/null 2>&1; then
      echo "[watcher] process ended before summary appeared: ${SUMMARY_CSV}" >&2
      exit 1
    fi
  fi
  sleep "${POLL_SEC}"
done
