#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/gemini/code/FMtrack-main/FM-Track"
SESSION="${1:-fmtrack_strongsort_mot17_val}"

cd "${REPO_ROOT}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "[skip] tmux session already exists: ${SESSION}"
  exit 0
fi

tmux new-session -d -s "${SESSION}" "cd ${REPO_ROOT} && bash scripts/run_strongsort_ltra_mot17_val.sh"
echo "[started] session=${SESSION}"
