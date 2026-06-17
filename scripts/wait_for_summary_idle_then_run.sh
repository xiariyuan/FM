#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 4 ]; then
  echo "usage: $0 <summary_csv> <log_path> <poll_seconds> -- <command...>" >&2
  exit 2
fi

SUMMARY_CSV=$1
LOG_PATH=$2
POLL_SECONDS=$3
shift 3

if [ "${1:-}" != "--" ]; then
  echo "missing -- separator before command" >&2
  exit 2
fi
shift

if [ "$#" -eq 0 ]; then
  echo "missing command to run" >&2
  exit 2
fi

mkdir -p "$(dirname "$LOG_PATH")"

{
  echo "[queued_at] $(date --iso-8601=seconds)"
  echo "[wait_on] $SUMMARY_CSV"
  echo "[poll_seconds] $POLL_SECONDS"
} >>"$LOG_PATH"

if [ ! -f "$SUMMARY_CSV" ]; then
  echo "[error] missing summary csv: $SUMMARY_CSV" >>"$LOG_PATH"
  exit 1
fi

while grep -q ",running," "$SUMMARY_CSV"; do
  echo "[poll] $(date --iso-8601=seconds) current run still active" >>"$LOG_PATH"
  sleep "$POLL_SECONDS"
done

{
  echo "[launch_at] $(date --iso-8601=seconds)"
  echo "[cmd] $*"
} >>"$LOG_PATH"

"$@" >>"$LOG_PATH" 2>&1
rc=$?

echo "[finished_at] $(date --iso-8601=seconds) rc=$rc" >>"$LOG_PATH"
exit "$rc"
