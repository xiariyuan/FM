#!/usr/bin/env bash
set -euo pipefail
cd /gemini/code/FMtrack-main/FM-Track
python3 scripts/dmm_base_tracker.py "$@"
