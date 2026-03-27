#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"

export HACA_LABEL="${HACA_LABEL:-haca_v3}"
export HACA_MODE="${HACA_MODE:-haca_v3}"
export SCRIPT_NAME="${SCRIPT_NAME:-scripts/run_botsort_haca_v3_eval.sh}"
export VARIANT_NAME="${VARIANT_NAME:-haca_v3_matrix}"

exec "${REPO_ROOT}/scripts/run_botsort_haca_v1_eval.sh"
