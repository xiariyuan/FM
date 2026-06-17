#!/usr/bin/env bash
set -euo pipefail

unset ORION_TASK_IDLE_TIME || true

REPO_ROOT="/gemini/code/FMtrack-main/FM-Track"
cd "$REPO_ROOT"

SMOKE_SUMMARY="outputs/deep_ocsort_local_contention_export_dance_smoke_20260406_1/summary.csv"
FULL_OUT="outputs/deep_ocsort_local_contention_export_dance_best_20260406_1"
FOLLOW_LOG="$FULL_OUT/followup_launcher.log"

mkdir -p "$FULL_OUT"

{
  echo "[started_at] $(date --iso-8601=seconds)"
  echo "[mode] wait smoke success then launch full DanceTrack export"
} >> "$FOLLOW_LOG"

while pgrep -f "python scripts/run_deep_ocsort_preassoc_competition_dataset_eval.py --benchmark DanceTrack --seq-name dancetrack0004 --out-root outputs/deep_ocsort_local_contention_export_dance_smoke_20260406_1 --reuse-raw-from outputs/deep_ocsort_local_contention_export_dance_smoke_20260406_1" >/dev/null; do
  sleep 60
done

python - <<'PY' >> "$FOLLOW_LOG" 2>&1
import csv
import sys
from pathlib import Path

summary = Path("outputs/deep_ocsort_local_contention_export_dance_smoke_20260406_1/summary.csv")
if not summary.is_file():
    print({"error": "missing_summary", "path": str(summary)})
    sys.exit(2)

rows = list(csv.DictReader(summary.open("r", encoding="utf-8", newline="")))
status = {row["step"]: row["status"] for row in rows}
print({"smoke_status": status})
required = ["raw_track", "raw_eval", "competition_track", "competition_eval", "compare"]
if not all(status.get(step) == "success" for step in required):
    sys.exit(3)
PY

rc=$?
if [ "$rc" -ne 0 ]; then
  {
    echo "[stop_at] $(date --iso-8601=seconds)"
    echo "[reason] smoke_not_success rc=$rc"
  } >> "$FOLLOW_LOG"
  exit "$rc"
fi

if pgrep -f "python scripts/run_deep_ocsort_preassoc_competition_dataset_eval.py --benchmark DanceTrack --out-root outputs/deep_ocsort_local_contention_export_dance_best_20260406_1" >/dev/null; then
  {
    echo "[skip_at] $(date --iso-8601=seconds)"
    echo "[reason] full_export_already_running"
  } >> "$FOLLOW_LOG"
  exit 0
fi

{
  echo "[launch_at] $(date --iso-8601=seconds)"
} >> "$FOLLOW_LOG"

nohup python scripts/run_deep_ocsort_preassoc_competition_dataset_eval.py \
  --benchmark DanceTrack \
  --out-root outputs/deep_ocsort_local_contention_export_dance_best_20260406_1 \
  --preassoc-stale-competition-min-time-since-update 2 \
  --preassoc-stale-competition-max-time-since-update 8 \
  --preassoc-stale-competition-min-hits 20 \
  --preassoc-stale-competition-min-box-iou 0.75 \
  --preassoc-stale-competition-min-edge-score 0.0 \
  --preassoc-stale-competition-bias 0.1 \
  --preassoc-stale-competition-iou-scale 0.0 \
  --preassoc-stale-competition-require-raw-owner \
  --preassoc-stale-competition-min-age-gap-vs-owner 50 \
  --preassoc-stale-competition-owner-max-hits 8 \
  --preassoc-stale-competition-owner-edge-penalty 0.05 \
  --preassoc-stale-competition-max-owner-edge-deficit -1.0 \
  --preassoc-stale-competition-force-owner-edge-deficit-arg \
  --preassoc-stale-competition-block-owner-on-reclaim \
  --local-contention-export-jsonl outputs/deep_ocsort_local_contention_export_dance_best_20260406_1/local_contention_units.jsonl \
  --local-contention-topk 3 \
  --local-contention-min-box-iou 0.5 \
  --local-contention-max-time-since-update 8 \
  --local-contention-min-challenger-hits 3 \
  --local-contention-owner-weak-hits 8 \
  --competition-track-max-frames-per-batch 10000 \
  > outputs/deep_ocsort_local_contention_export_dance_best_20260406_1/launcher.log 2>&1 &

echo "[full_pid] $!" >> "$FOLLOW_LOG"
