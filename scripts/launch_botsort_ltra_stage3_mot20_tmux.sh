#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/gemini/code/FMtrack-main/FM-Track"
cd "${REPO_ROOT}"

SESSION="${1:-fmtrack_stage3_mot20}"

chmod +x scripts/run_botsort_ltra_stage3_mot20.sh

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  tmux kill-session -t "${SESSION}"
fi

tmux new-session -d -s "${SESSION}" "cd ${REPO_ROOT} && bash scripts/run_botsort_ltra_stage3_mot20.sh"

echo "[launched] tmux session=${SESSION}"
