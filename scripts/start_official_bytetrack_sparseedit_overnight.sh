#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
SESSION_NAME="${SESSION_NAME:-official_sparseedit_overnight}"
POLL_SEC="${POLL_SEC:-180}"
MAIN_RUN_ROOT="${1:-${REPO_ROOT}/outputs/official_bytetrack_stage1_largecomp4_sparseedit_20260326_235955}"
QUEUE_ROOT="${2:-}"

if tmux has-session -t "${SESSION_NAME}" >/dev/null 2>&1; then
  echo "[start] tmux session already exists: ${SESSION_NAME}" >&2
  exit 1
fi

CMD="cd $(printf '%q' "${REPO_ROOT}") && $(printf '%q' "${PYTHON_BIN}") scripts/queue_official_bytetrack_sparseedit_overnight.py --main-run-root $(printf '%q' "${MAIN_RUN_ROOT}") --poll-sec $(printf '%q' "${POLL_SEC}")"
if [[ -n "${QUEUE_ROOT}" ]]; then
  CMD+=" --queue-root $(printf '%q' "${QUEUE_ROOT}")"
fi

tmux new-session -d -s "${SESSION_NAME}" "${CMD}"
echo "[start] session=${SESSION_NAME}"
echo "[start] main_run_root=${MAIN_RUN_ROOT}"
if [[ -n "${QUEUE_ROOT}" ]]; then
  echo "[start] queue_root=${QUEUE_ROOT}"
fi
