#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
PLAN_CSV="${EXPERIMENT_PLAN_CSV:-${REPO_ROOT}/outputs/experiment_plan.csv}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"

TRAIN_CHECKPOINT="${TRAIN_CHECKPOINT:-}"
TRAIN_RUN_ROOT="${TRAIN_RUN_ROOT:-}"
POLL_SECS="${POLL_SECS:-60}"
TIMEOUT_SECS="${TIMEOUT_SECS:-43200}"
EVAL_RUN_ROOT="${EVAL_RUN_ROOT:-${REPO_ROOT}/outputs/haca_v1_eval_after_${TS}}"

RUN_BASE="${RUN_BASE:-0}"
RUN_HEURISTIC="${RUN_HEURISTIC:-0}"
RUN_CURRENT_LEARNED="${RUN_CURRENT_LEARNED:-0}"
RUN_HACA="${RUN_HACA:-1}"
CURRENT_CALIBRATOR_NPZ="${CURRENT_CALIBRATOR_NPZ:-}"
HACA_DISABLE_SET_ENCODER="${HACA_DISABLE_SET_ENCODER:-0}"
HACA_DISABLE_BACKGROUND="${HACA_DISABLE_BACKGROUND:-0}"
HACA_DELTA_SCALE="${HACA_DELTA_SCALE:-}"

if [[ -z "${TRAIN_CHECKPOINT}" ]]; then
  echo "TRAIN_CHECKPOINT is required" >&2
  exit 1
fi

train_status() {
  "${PYTHON_BIN}" - <<'PY' "${PLAN_CSV}" "${TRAIN_RUN_ROOT}"
import csv
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
run_root = sys.argv[2]
if not run_root or not csv_path.is_file():
    print("")
    raise SystemExit(0)
with csv_path.open("r", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row.get("run_root", "") == run_root:
            print(row.get("status", ""))
            raise SystemExit(0)
print("")
PY
}

echo "[watch] train_checkpoint=${TRAIN_CHECKPOINT}"
echo "[watch] train_run_root=${TRAIN_RUN_ROOT}"
echo "[watch] poll_secs=${POLL_SECS} timeout_secs=${TIMEOUT_SECS}"

start_ts="$(date +%s)"
while [[ ! -f "${TRAIN_CHECKPOINT}" ]]; do
  now_ts="$(date +%s)"
  elapsed=$((now_ts - start_ts))
  if (( elapsed > TIMEOUT_SECS )); then
    echo "[watch] timeout waiting for ${TRAIN_CHECKPOINT}"
    exit 1
  fi
  status="$(train_status || true)"
  if [[ "${status}" == "failed" || "${status}" == "cancelled" ]]; then
    echo "[watch] train status=${status}; aborting eval queue"
    exit 1
  fi
  sleep "${POLL_SECS}"
done

echo "[watch] checkpoint ready: ${TRAIN_CHECKPOINT}"
sleep 5

RUN_ROOT="${EVAL_RUN_ROOT}" \
HACA_NPZ="${TRAIN_CHECKPOINT}" \
RUN_BASE="${RUN_BASE}" \
RUN_HEURISTIC="${RUN_HEURISTIC}" \
RUN_CURRENT_LEARNED="${RUN_CURRENT_LEARNED}" \
RUN_HACA="${RUN_HACA}" \
CURRENT_CALIBRATOR_NPZ="${CURRENT_CALIBRATOR_NPZ}" \
HACA_DISABLE_SET_ENCODER="${HACA_DISABLE_SET_ENCODER}" \
HACA_DISABLE_BACKGROUND="${HACA_DISABLE_BACKGROUND}" \
HACA_DELTA_SCALE="${HACA_DELTA_SCALE}" \
"${REPO_ROOT}/scripts/run_botsort_haca_v1_eval.sh"
