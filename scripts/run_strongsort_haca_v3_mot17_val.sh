#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"

export HACA_LABEL="${HACA_LABEL:-haca_v3}"
export HACA_MODE="${HACA_MODE:-haca_v3}"
export SCRIPT_NAME="${SCRIPT_NAME:-scripts/run_strongsort_haca_v3_mot17_val.sh}"
export VARIANT_NAME="${VARIANT_NAME:-strongsort_haca_v3}"

exec bash "${REPO_ROOT}/scripts/run_strongsort_haca_v2_mot17_val.sh"
