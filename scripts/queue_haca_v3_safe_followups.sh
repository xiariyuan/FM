#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
PLAN_CSV="${EXPERIMENT_PLAN_CSV:-${REPO_ROOT}/outputs/experiment_plan.csv}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
SCRIPT_NAME="${SCRIPT_NAME:-scripts/queue_haca_v3_safe_followups.sh}"
QUEUE_LABEL="${QUEUE_LABEL:-haca_v3_safe_followups_${TS}}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/outputs/${QUEUE_LABEL}}"
RUN_ROOT="$(realpath -m "${RUN_ROOT}")"
LOG_PATH="${RUN_ROOT}/queue.log"
PLAN_KEY="${PLAN_KEY:-queue:${RUN_ROOT}}"
WAIT_PLAN_KEY="${WAIT_PLAN_KEY:-}"
POLL_SECS="${POLL_SECS:-120}"
TIMEOUT_SECS="${TIMEOUT_SECS:-172800}"

mkdir -p "${RUN_ROOT}"
exec > >(tee -a "${LOG_PATH}") 2>&1

if [[ -z "${WAIT_PLAN_KEY}" ]]; then
  echo "WAIT_PLAN_KEY is required" >&2
  exit 1
fi

update_plan_status() {
  local status="$1"
  shift || true
  local extras=(
    "wait_plan_key=${WAIT_PLAN_KEY}"
    "poll_secs=${POLL_SECS}"
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
    --dataset "MOT17+MOT20" \
    --split "safe_followups" \
    --tracker-family "BoT-SORT+StrongSORT" \
    --variant "haca_v3_safe_followups" \
    --run-root "${RUN_ROOT}" \
    --log-path "${LOG_PATH}" \
    --notes "queue_safe_variants_after_main_pipeline" \
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

plan_status() {
  "${PYTHON_BIN}" - <<'PY' "${PLAN_CSV}" "$1"
import csv
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
plan_key = sys.argv[2]
if not csv_path.is_file():
    print("")
    raise SystemExit(0)
with csv_path.open("r", newline="") as f:
    for row in csv.DictReader(f):
        if row.get("plan_key", "") == plan_key:
            print(row.get("status", ""))
            raise SystemExit(0)
print("")
PY
}

wait_for_plan() {
  local start_ts now_ts elapsed status
  start_ts="$(date +%s)"
  echo "[followups] waiting for ${WAIT_PLAN_KEY}"
  while true; do
    status="$(plan_status "${WAIT_PLAN_KEY}" || true)"
    if [[ "${status}" == "completed" || "${status}" == "failed" || "${status}" == "cancelled" ]]; then
      echo "[followups] wait plan terminal status=${status}"
      return 0
    fi
    now_ts="$(date +%s)"
    elapsed=$((now_ts - start_ts))
    if (( elapsed > TIMEOUT_SECS )); then
      echo "[followups] timeout waiting for ${WAIT_PLAN_KEY}" >&2
      return 1
    fi
    sleep "${POLL_SECS}"
  done
}

run_variant() {
  local label="$1"
  local comp_topk="$2"
  local comp_margin_quantile="$3"
  local comp_delta_scale="$4"
  local loss_shift_weight="$5"
  local shift_batch_prob="$6"

  local train_root="${REPO_ROOT}/outputs/${label}_train_${TS}"
  local checkpoint="${train_root}/mot17_${label}_${TS}.npz"
  local samebase_root="${REPO_ROOT}/outputs/${label}_eval_${TS}"
  local strongsort_root="${REPO_ROOT}/outputs/strongsort_${label}/MOT17_val_${TS}"
  local mot20_root="${REPO_ROOT}/outputs/${label}_mot20_${TS}"

  echo "[followups] train ${label} topk=${comp_topk} q=${comp_margin_quantile} delta=${comp_delta_scale} shift_w=${loss_shift_weight} shift_p=${shift_batch_prob}"
  env \
    HACA_LABEL="${label}" \
    VARIANT_NAME="${label}_train" \
    OUT_ROOT="${train_root}" \
    OUT_NPZ="${checkpoint}" \
    COMP_TOPK="${comp_topk}" \
    COMP_MARGIN_QUANTILE="${comp_margin_quantile}" \
    COMP_DELTA_SCALE="${comp_delta_scale}" \
    LOSS_SHIFT_WEIGHT="${loss_shift_weight}" \
    SHIFT_BATCH_PROB="${shift_batch_prob}" \
    bash "${REPO_ROOT}/scripts/train_haca_v3_safe_mot17.sh"

  echo "[followups] pipeline ${label}"
  env \
    HACA_LABEL="${label}" \
    TRAIN_CHECKPOINT="${checkpoint}" \
    TRAIN_RUN_ROOT="${train_root}" \
    SAMEBASE_RUN_ROOT="${samebase_root}" \
    STRONGSORT_RUN_ROOT="${strongsort_root}" \
    MOT20_RUN_ROOT="${mot20_root}" \
    STOP_IF_SAMEBASE_WEAK=0 \
    STOP_IF_STRONGSORT_COLLAPSES=0 \
    bash "${REPO_ROOT}/scripts/queue_haca_v3_pipeline.sh"
}

echo "[followups] $(date '+%F %T %z') start"
echo "[followups] run_root=${RUN_ROOT}"
wait_for_plan

run_variant "haca_v3_safe_q45d075" 3 0.45 0.75 0.15 0.35
run_variant "haca_v3_safe_k2q45d075" 2 0.45 0.75 0.15 0.35

echo "[followups] $(date '+%F %T %z') done"
update_plan_status completed
