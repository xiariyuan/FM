#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/gemini/code/FMtrack-main/FM-Track"
cd "${REPO_ROOT}"

VARIANT="${1:?usage: launch_stage3_variant_tmux.sh <variant> [session_name]}"
SESSION="${2:-stage3_${VARIANT}}"

chmod +x scripts/run_botsort_ltra_stage3_variant_mot20.sh

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  tmux kill-session -t "${SESSION}"
fi

tmux new-session -d -s "${SESSION}" "cd ${REPO_ROOT} && bash scripts/run_botsort_ltra_stage3_variant_mot20.sh ${VARIANT}"

echo "[launched] variant=${VARIANT} tmux=${SESSION}"
