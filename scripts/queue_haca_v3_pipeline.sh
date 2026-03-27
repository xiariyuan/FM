#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
PLAN_CSV="${EXPERIMENT_PLAN_CSV:-${REPO_ROOT}/outputs/experiment_plan.csv}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
SCRIPT_NAME="${SCRIPT_NAME:-scripts/queue_haca_v3_pipeline.sh}"

TRAIN_CHECKPOINT="${TRAIN_CHECKPOINT:-}"
TRAIN_RUN_ROOT="${TRAIN_RUN_ROOT:-}"
POLL_SECS="${POLL_SECS:-60}"
TIMEOUT_SECS="${TIMEOUT_SECS:-43200}"

CURRENT_CALIBRATOR_NPZ="${CURRENT_CALIBRATOR_NPZ:-${REPO_ROOT}/outputs/lpb_ltra_formal_mot17_shrink_20260311_191211/train_vec_20260311_201832/mot17_lpb_ltra_20260311_201832.npz}"
SAMEBASE_RUN_ROOT="${SAMEBASE_RUN_ROOT:-${REPO_ROOT}/outputs/haca_v3_eval_after_${TS}}"
STRONGSORT_RUN_ROOT="${STRONGSORT_RUN_ROOT:-${REPO_ROOT}/outputs/strongsort_haca_v3/MOT17_val_after_${TS}}"
MOT20_RUN_ROOT="${MOT20_RUN_ROOT:-${REPO_ROOT}/outputs/haca_v3_mot20_after_${TS}}"
QUEUE_LOG_PATH="${QUEUE_LOG_PATH:-}"
PLAN_KEY="${PLAN_KEY:-}"

RUN_SAMEBASE="${RUN_SAMEBASE:-1}"
RUN_STRONGSORT="${RUN_STRONGSORT:-1}"
RUN_MOT20="${RUN_MOT20:-1}"
STOP_IF_SAMEBASE_WEAK="${STOP_IF_SAMEBASE_WEAK:-1}"
STOP_IF_STRONGSORT_COLLAPSES="${STOP_IF_STRONGSORT_COLLAPSES:-0}"

SAMEBASE_HOTA_MARGIN="${SAMEBASE_HOTA_MARGIN:-0.0}"
SAMEBASE_ASSA_MARGIN="${SAMEBASE_ASSA_MARGIN:-0.0}"
STRONGSORT_HOTA_DROP_LIMIT="${STRONGSORT_HOTA_DROP_LIMIT:-0.3}"
STRONGSORT_ASSA_DROP_LIMIT="${STRONGSORT_ASSA_DROP_LIMIT:-0.3}"

if [[ -z "${TRAIN_CHECKPOINT}" ]]; then
  echo "TRAIN_CHECKPOINT is required" >&2
  exit 1
fi

if [[ -z "${TRAIN_RUN_ROOT}" ]]; then
  TRAIN_RUN_ROOT="$(dirname "${TRAIN_CHECKPOINT}")"
fi
if [[ -z "${QUEUE_LOG_PATH}" ]]; then
  QUEUE_LOG_PATH="${TRAIN_RUN_ROOT}/pipeline.log"
fi
if [[ -z "${PLAN_KEY}" ]]; then
  PLAN_KEY="pipeline:${TRAIN_RUN_ROOT}"
fi

update_plan_status() {
  local status="$1"
  shift || true
  local extras=(
    "samebase_run_root=${SAMEBASE_RUN_ROOT}"
    "strongsort_run_root=${STRONGSORT_RUN_ROOT}"
    "mot20_run_root=${MOT20_RUN_ROOT}"
    "run_samebase=${RUN_SAMEBASE}"
    "run_strongsort=${RUN_STRONGSORT}"
    "run_mot20=${RUN_MOT20}"
    "stop_if_samebase_weak=${STOP_IF_SAMEBASE_WEAK}"
    "stop_if_strongsort_collapses=${STOP_IF_STRONGSORT_COLLAPSES}"
  )
  if [[ $# -gt 0 ]]; then
    extras+=("$@")
  fi
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/upsert_experiment_plan.py" \
    --csv "${PLAN_CSV}" \
    --key "${PLAN_KEY}" \
    --status "${status}" \
    --kind other \
    --script "${SCRIPT_NAME}" \
    --dataset "MOT17_MOT20_StrongSORT" \
    --split staged \
    --tracker-family "BoT-SORT+StrongSORT" \
    --variant haca_v3_pipeline \
    --run-root "${TRAIN_RUN_ROOT}" \
    --checkpoint "${TRAIN_CHECKPOINT}" \
    --log-path "${QUEUE_LOG_PATH}" \
    --notes "auto_queue_after_haca_v3_train" \
    --extra "${extras[@]}"
}

on_exit() {
  local rc=$?
  trap - EXIT
  if [[ ${rc} -ne 0 ]]; then
    update_plan_status failed "exit_code=${rc}" || true
  fi
  exit ${rc}
}

trap on_exit EXIT
update_plan_status running

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

wait_for_checkpoint() {
  local start_ts
  start_ts="$(date +%s)"
  echo "[queue] waiting for checkpoint ${TRAIN_CHECKPOINT}"
  while [[ ! -f "${TRAIN_CHECKPOINT}" ]]; do
    local now_ts elapsed status
    now_ts="$(date +%s)"
    elapsed=$((now_ts - start_ts))
    if (( elapsed > TIMEOUT_SECS )); then
      echo "[queue] timeout waiting for ${TRAIN_CHECKPOINT}"
      return 1
    fi
    status="$(train_status || true)"
    if [[ "${status}" == "failed" || "${status}" == "cancelled" ]]; then
      echo "[queue] train status=${status}; aborting queue"
      return 1
    fi
    sleep "${POLL_SECS}"
  done
  echo "[queue] checkpoint ready: ${TRAIN_CHECKPOINT}"
}

gate_samebase() {
  "${PYTHON_BIN}" - <<'PY' "${SAMEBASE_RUN_ROOT}/eval/mot17_summary.csv" "${SAMEBASE_HOTA_MARGIN}" "${SAMEBASE_ASSA_MARGIN}"
import csv
import sys

summary_path = sys.argv[1]
hota_margin = float(sys.argv[2])
assa_margin = float(sys.argv[3])

rows = {}
with open(summary_path, newline="") as f:
    for row in csv.DictReader(f):
        rows[row["name"]] = row

haca = rows.get("haca")
heur = rows.get("heuristic")
if haca is None or heur is None:
    print("[queue] missing haca/heuristic rows in same-base summary")
    raise SystemExit(2)

h_hota = float(haca["HOTA"])
h_assa = float(haca["AssA"])
heur_hota = float(heur["HOTA"])
heur_assa = float(heur["AssA"])

print(f"[queue] same-base haca HOTA={h_hota:.3f} AssA={h_assa:.3f}; heuristic HOTA={heur_hota:.3f} AssA={heur_assa:.3f}")
if h_hota >= heur_hota + hota_margin and h_assa >= heur_assa + assa_margin:
    raise SystemExit(0)
raise SystemExit(10)
PY
}

gate_strongsort() {
  "${PYTHON_BIN}" - <<'PY' "${STRONGSORT_RUN_ROOT}/summary.csv" "${STRONGSORT_HOTA_DROP_LIMIT}" "${STRONGSORT_ASSA_DROP_LIMIT}"
import csv
import sys

summary_path = sys.argv[1]
hota_drop = float(sys.argv[2])
assa_drop = float(sys.argv[3])

rows = {}
with open(summary_path, newline="") as f:
    for row in csv.DictReader(f):
        rows[row["name"]] = row

haca = rows.get("haca_eval")
heur = rows.get("laplace_eval")
if haca is None or heur is None:
    print("[queue] missing haca_eval/laplace_eval rows in StrongSORT summary")
    raise SystemExit(2)

h_hota = float(haca["HOTA"])
h_assa = float(haca["AssA"])
heur_hota = float(heur["HOTA"])
heur_assa = float(heur["AssA"])

print(f"[queue] StrongSORT haca HOTA={h_hota:.3f} AssA={h_assa:.3f}; heuristic HOTA={heur_hota:.3f} AssA={heur_assa:.3f}")
if h_hota + hota_drop >= heur_hota and h_assa + assa_drop >= heur_assa:
    raise SystemExit(0)
raise SystemExit(10)
PY
}

echo "[queue] $(date '+%F %T %z') start"
echo "[queue] train_run_root=${TRAIN_RUN_ROOT}"
echo "[queue] samebase_run_root=${SAMEBASE_RUN_ROOT}"
echo "[queue] strongsort_run_root=${STRONGSORT_RUN_ROOT}"
echo "[queue] mot20_run_root=${MOT20_RUN_ROOT}"

wait_for_checkpoint

if [[ "${RUN_SAMEBASE}" == "1" ]]; then
  HACA_NPZ="${TRAIN_CHECKPOINT}" \
  RUN_ROOT="${SAMEBASE_RUN_ROOT}" \
  RUN_BASE=0 RUN_HEURISTIC=1 RUN_CURRENT_LEARNED=1 RUN_HACA=1 \
  CURRENT_CALIBRATOR_NPZ="${CURRENT_CALIBRATOR_NPZ}" \
  bash "${REPO_ROOT}/scripts/run_botsort_haca_v3_eval.sh"

  if ! gate_samebase; then
    if [[ "${STOP_IF_SAMEBASE_WEAK}" == "1" ]]; then
      echo "[queue] stopping after same-base gate"
      update_plan_status completed "stop_reason=samebase_gate"
      exit 0
    fi
  fi
fi

if [[ "${RUN_STRONGSORT}" == "1" ]]; then
  HACA_NPZ="${TRAIN_CHECKPOINT}" \
  OUT_ROOT="${STRONGSORT_RUN_ROOT}" \
  bash "${REPO_ROOT}/scripts/run_strongsort_haca_v3_mot17_val.sh"

  if [[ "${STOP_IF_STRONGSORT_COLLAPSES}" == "1" ]]; then
    if ! gate_strongsort; then
      echo "[queue] stopping after StrongSORT gate"
      update_plan_status completed "stop_reason=strongsort_gate"
      exit 0
    fi
  fi
fi

if [[ "${RUN_MOT20}" == "1" ]]; then
  HACA_NPZ="${TRAIN_CHECKPOINT}" \
  RUN_ROOT="${MOT20_RUN_ROOT}" \
  RUN_BASE=0 RUN_HEURISTIC=1 RUN_CURRENT_LEARNED=0 RUN_HACA=1 \
  bash "${REPO_ROOT}/scripts/run_botsort_haca_v3_mot20_eval.sh"
fi

echo "[queue] $(date '+%F %T %z') finished"
update_plan_status completed "stop_reason=finished"
