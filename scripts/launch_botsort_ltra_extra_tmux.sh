#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/gemini/code/FMtrack-main/FM-Track"
DATASET="${1:?usage: launch_botsort_ltra_extra_tmux.sh <dataset> <variant> [track args...]}"
VARIANT="${2:?usage: launch_botsort_ltra_extra_tmux.sh <dataset> <variant> [track args...]}"
shift 2
SESSION="fmtrack_${DATASET,,}_${VARIANT}"

cd "${REPO_ROOT}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "[skip] tmux session already exists: ${SESSION}"
  exit 0
fi

cmd="cd ${REPO_ROOT} && bash scripts/run_botsort_ltra_variant_eval.sh ${DATASET} ${VARIANT}"
for arg in "$@"; do
  cmd+=" $(printf '%q' "${arg}")"
done

tmux new-session -d -s "${SESSION}" "${cmd}"
echo "[started] session=${SESSION}"
