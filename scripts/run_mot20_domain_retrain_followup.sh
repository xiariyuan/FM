#!/usr/bin/env bash
set -euo pipefail

# Continue the MOT20-domain SCA+LMF experiment after HACA v3 retraining.
# Waits for a specific HACA checkpoint, then runs MOT20 Stage1 training and final eval.

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
PLAN_CSV="${PLAN_CSV:-${REPO_ROOT}/outputs/experiment_plan.csv}"
REGISTRY_CSV="${REGISTRY_CSV:-${REPO_ROOT}/outputs/experiment_registry.csv}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
HACA_RUN_ROOT="${HACA_RUN_ROOT:-${REPO_ROOT}/outputs/haca_mot20_train_20260608_171931}"
HACA_CKPT="${HACA_CKPT:-${HACA_RUN_ROOT}/haca_v3/mot20_haca_v3.npz}"
HACA_SUMMARY="${HACA_SUMMARY:-${HACA_RUN_ROOT}/summary.csv}"
POLL_SECS="${POLL_SECS:-120}"
TIMEOUT_SECS="${TIMEOUT_SECS:-172800}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs/mot20_domain_retrain_followup_${TS}}"
OUT_ROOT="$(realpath -m "${OUT_ROOT}")"
PLAN_KEY="${PLAN_KEY:-run_root:${OUT_ROOT}}"
LOG_PATH="${OUT_ROOT}/queue.log"
SUMMARY_CSV="${OUT_ROOT}/summary.csv"
STAGE1_ROOT="${STAGE1_ROOT:-${REPO_ROOT}/outputs/rgsa_stage1_mot20_pipeline_${TS}}"
EVAL_ROOT="${EVAL_ROOT:-${REPO_ROOT}/outputs/sca_lmf_mot20_eval_${TS}}"
STAGE1_CKPT="${STAGE1_ROOT}/stage1/stage1_best.pt"

mkdir -p "${OUT_ROOT}"
exec > >(tee -a "${LOG_PATH}") 2>&1

HACA_STATUS="waiting"
STAGE1_STATUS="pending"
EVAL_STATUS="pending"
CURRENT_PHASE="haca_wait"

write_summary_status() {
  cat > "${SUMMARY_CSV}" <<EOF
phase,status,artifact
haca_v3,${HACA_STATUS},${HACA_CKPT}
stage1_mot20,${STAGE1_STATUS},${STAGE1_CKPT}
sca_lmf_eval,${EVAL_STATUS},${EVAL_ROOT}/summary.csv
EOF
}

update_plan_status() {
  local status="$1"
  shift || true
  local extras=(
    "haca_run_root=${HACA_RUN_ROOT}"
    "haca_checkpoint=${HACA_CKPT}"
    "stage1_run_root=${STAGE1_ROOT}"
    "eval_run_root=${EVAL_ROOT}"
    "poll_secs=${POLL_SECS}"
  )
  if [[ $# -gt 0 ]]; then
    extras+=("$@")
  fi
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/upsert_experiment_plan.py" \
    --csv "${PLAN_CSV}" \
    --key "${PLAN_KEY}" \
    --status "${status}" \
    --kind train \
    --script "scripts/run_mot20_domain_retrain_followup.sh" \
    --dataset MOT20 \
    --split train_half_val_half \
    --tracker-family BoT-SORT \
    --variant "mot20_domain_retrain_followup" \
    --run-root "${OUT_ROOT}" \
    --summary-csv "${SUMMARY_CSV}" \
    --checkpoint "${HACA_CKPT}" \
    --log-path "${LOG_PATH}" \
    --extra "${extras[@]}"
}

append_registry() {
  local status="$1"
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/append_experiment_record.py" \
    --csv "${REGISTRY_CSV}" \
    --kind train \
    --status "${status}" \
    --script "scripts/run_mot20_domain_retrain_followup.sh" \
    --dataset MOT20 \
    --split train_half_val_half \
    --tracker-family BoT-SORT \
    --variant "mot20_domain_retrain_followup" \
    --run-root "${OUT_ROOT}" \
    --summary-csv "${SUMMARY_CSV}" \
    --checkpoint "${HACA_CKPT}" \
    --log-path "${LOG_PATH}" \
    --extra \
      "haca_run_root=${HACA_RUN_ROOT}" \
      "stage1_run_root=${STAGE1_ROOT}" \
      "eval_run_root=${EVAL_ROOT}"
}

haca_plan_status() {
  "${PYTHON_BIN}" - <<'PY' "${PLAN_CSV}" "${HACA_RUN_ROOT}"
import csv
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
run_root = sys.argv[2]
if not csv_path.is_file():
    print("")
    raise SystemExit(0)
with csv_path.open("r", newline="") as f:
    for row in csv.DictReader(f):
        if row.get("run_root", "") == run_root:
            print(row.get("status", ""))
            raise SystemExit(0)
print("")
PY
}

haca_summary_status() {
  "${PYTHON_BIN}" - <<'PY' "${HACA_SUMMARY}"
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    print("")
    raise SystemExit(0)
with path.open("r", newline="") as f:
    for row in csv.DictReader(f):
        if row.get("phase", "") == "haca_v3":
            print(row.get("status", ""))
            raise SystemExit(0)
print("")
PY
}

on_exit() {
  local rc=$?
  trap - EXIT
  if [[ ${rc} -ne 0 ]]; then
    case "${CURRENT_PHASE}" in
      haca_wait)
        HACA_STATUS="failed"
        ;;
      stage1)
        STAGE1_STATUS="failed"
        ;;
      eval)
        EVAL_STATUS="failed"
        ;;
    esac
    write_summary_status || true
    update_plan_status failed "exit_code=${rc}" || true
    append_registry failed || true
  fi
  exit ${rc}
}

trap on_exit EXIT
write_summary_status
update_plan_status running

echo "[queue] start $(date '+%F %T %z')"
echo "[queue] haca_run_root=${HACA_RUN_ROOT}"
echo "[queue] haca_checkpoint=${HACA_CKPT}"
echo "[queue] stage1_root=${STAGE1_ROOT}"
echo "[queue] eval_root=${EVAL_ROOT}"
echo "[queue] poll_secs=${POLL_SECS} timeout_secs=${TIMEOUT_SECS}"

start_ts="$(date +%s)"
while [[ ! -f "${HACA_CKPT}" ]]; do
  now_ts="$(date +%s)"
  elapsed=$((now_ts - start_ts))
  if (( elapsed > TIMEOUT_SECS )); then
    echo "[queue] timeout waiting for ${HACA_CKPT}"
    exit 1
  fi
  plan_status="$(haca_plan_status || true)"
  summary_status="$(haca_summary_status || true)"
  echo "[queue] waiting haca_v3 plan=${plan_status:-unknown} summary=${summary_status:-unknown} elapsed=${elapsed}s"
  if [[ "${plan_status}" == "failed" || "${summary_status}" == "failed" ]]; then
    echo "[queue] HACA run failed before checkpoint was produced"
    exit 1
  fi
  sleep "${POLL_SECS}"
done

HACA_STATUS="success"
STAGE1_STATUS="running"
CURRENT_PHASE="stage1"
write_summary_status
echo "[queue] HACA checkpoint ready: ${HACA_CKPT}"

TS="${TS}" \
OUT_ROOT="${STAGE1_ROOT}" \
HACA_CKPT="${HACA_CKPT}" \
PLAN_CSV="${PLAN_CSV}" \
REGISTRY_CSV="${REGISTRY_CSV}" \
"${REPO_ROOT}/scripts/run_rgsa_stage1_mot20_train.sh"

STAGE1_STATUS="success"
EVAL_STATUS="running"
CURRENT_PHASE="eval"
write_summary_status
echo "[queue] Stage1 checkpoint ready: ${STAGE1_CKPT}"

TS="${TS}" \
OUT_BASE="${EVAL_ROOT}" \
HACA_CKPT="${HACA_CKPT}" \
S1_CKPT="${STAGE1_CKPT}" \
PLAN_CSV="${PLAN_CSV}" \
REGISTRY_CSV="${REGISTRY_CSV}" \
"${REPO_ROOT}/scripts/run_stage1_freezeonly_mot20_eval.sh"

HACA_STATUS="success"
STAGE1_STATUS="success"
EVAL_STATUS="success"
CURRENT_PHASE="done"
write_summary_status
update_plan_status completed
append_registry success
echo "[queue] done $(date '+%F %T %z')"
