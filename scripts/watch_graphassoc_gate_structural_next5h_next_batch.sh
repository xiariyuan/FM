#!/usr/bin/env bash
set -euo pipefail

ROOT="/gemini/code/FMtrack-main/FM-Track"
QUEUE_SUMMARY="$ROOT/outputs/20260505_144347_graphassoc_gate_structural_next5h_gatedblend/summary.csv"
LAUNCH_LOG="$ROOT/outputs/20260505_144347_graphassoc_gate_structural_next5h_gatedblend/logs/launch_next_structural.log"

running_row_exists() {
  python - "$QUEUE_SUMMARY" <<'PY'
import csv
import pathlib
import sys

summary = pathlib.Path(sys.argv[1])
if not summary.is_file():
    sys.exit(1)

with summary.open(newline="") as handle:
    rows = list(csv.DictReader(handle))

for row in rows:
    if str(row.get("status", "")).strip() == "running":
        sys.exit(0)
sys.exit(1)
PY
}

mkdir -p "$(dirname "$LAUNCH_LOG")"
echo "[watcher_started] $(date -Iseconds)" >> "$LAUNCH_LOG"

while running_row_exists; do
  sleep 120
done

echo "[watcher_launching_next_batch] $(date -Iseconds)" >> "$LAUNCH_LOG"
cd "$ROOT"
python scripts/run_graphassoc_gate_structural_next5h.py --steps 09 10 11 12 >> "$LAUNCH_LOG" 2>&1
echo "[watcher_finished] $(date -Iseconds)" >> "$LAUNCH_LOG"
