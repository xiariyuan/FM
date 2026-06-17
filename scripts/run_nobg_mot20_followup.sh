#!/usr/bin/env bash
set -euo pipefail

# MOT20 no-background follow-up pipeline.
# Chains: HACA v3 checkpoint → no-bg Stage1 train → no-bg 4-variant eval.
# All track.py calls use --laplace-haca-no-background.

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
PLAN_CSV="${PLAN_CSV:-${REPO_ROOT}/outputs/experiment_plan.csv}"
REGISTRY_CSV="${REGISTRY_CSV:-${REPO_ROOT}/outputs/experiment_registry.csv}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
HACA_CKPT="${HACA_CKPT:-${REPO_ROOT}/outputs/haca_mot20_train_20260608_171931/haca_v3/mot20_haca_v3.npz}"
POLL_SECS="${POLL_SECS:-120}"
TIMEOUT_SECS="${TIMEOUT_SECS:-172800}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs/mot20_nobg_followup_${TS}}"
OUT_ROOT="$(realpath -m "${OUT_ROOT}")"
PLAN_KEY="${PLAN_KEY:-run_root:${OUT_ROOT}}"
LOG_PATH="${OUT_ROOT}/queue.log"
SUMMARY_CSV="${OUT_ROOT}/summary.csv"
STAGE1_ROOT="${STAGE1_ROOT:-${REPO_ROOT}/outputs/rgsa_stage1_mot20_nobg_pipeline_${TS}}"
EVAL_ROOT="${EVAL_ROOT:-${REPO_ROOT}/outputs/sca_lmf_mot20_nobg_eval_${TS}}"
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
stage1_mot20_nobg,${STAGE1_STATUS},${STAGE1_CKPT}
sca_lmf_nobg_eval,${EVAL_STATUS},${EVAL_ROOT}/summary.csv
EOF
}

update_plan_status() {
  local status="$1"
  shift || true
  local extras=(
    "haca_checkpoint=${HACA_CKPT}"
    "stage1_run_root=${STAGE1_ROOT}"
    "eval_run_root=${EVAL_ROOT}"
    "no_background=true"
  )
  if [[ $# -gt 0 ]]; then
    extras+=("$@")
  fi
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/upsert_experiment_plan.py" \
    --csv "${PLAN_CSV}" \
    --key "${PLAN_KEY}" \
    --status "${status}" \
    --kind other \
    --script "scripts/run_nobg_mot20_followup.sh" \
    --dataset MOT20 \
    --split train_half_val_half \
    --tracker-family BoT-SORT \
    --variant "mot20_nobg_followup" \
    --run-root "${OUT_ROOT}" \
    --summary-csv "${OUT_ROOT}/summary.csv" \
    --checkpoint "${HACA_CKPT}" \
    --log-path "${LOG_PATH}" \
    --extra "${extras[@]}"
}

on_exit() {
  local rc=$?
  trap - EXIT
  if [[ ${rc} -ne 0 ]]; then
    case "${CURRENT_PHASE}" in
      stage1_train)
        STAGE1_STATUS="failed"
        ;;
      eval)
        EVAL_STATUS="failed"
        ;;
    esac
    write_summary_status || true
    update_plan_status failed "exit_code=${rc}" || true
  fi
  exit ${rc}
}

trap on_exit EXIT
write_summary_status
update_plan_status running

echo "[info] HACA checkpoint: ${HACA_CKPT}"
echo "[info] NO-BACKGROUND mode: all track.py calls use --laplace-haca-no-background"

# --- Wait for HACA checkpoint ---
echo "[queue] Checking HACA checkpoint..."
if [[ -f "${HACA_CKPT}" ]]; then
  echo "[queue] HACA checkpoint ready: ${HACA_CKPT}"
  HACA_STATUS="success"
  write_summary_status
else
  echo "[FATAL] HACA checkpoint not found: ${HACA_CKPT}"
  HACA_STATUS="failed"
  write_summary_status
  exit 1
fi

# --- Phase 1: No-bg Stage1 training (oracle dump + labels + Stage1) ---
CURRENT_PHASE="stage1_train"
STAGE1_STATUS="running"
write_summary_status
echo "[queue] Starting no-bg Stage1 training pipeline..."

export TS="${TS}" OUT_ROOT="${STAGE1_ROOT}" HACA_CKPT="${HACA_CKPT}"
bash "${REPO_ROOT}/scripts/run_rgsa_stage1_mot20_nobg_train.sh"
STAGE1_CKPT="${STAGE1_ROOT}/stage1/stage1_best.pt"

if [[ ! -f "${STAGE1_CKPT}" ]]; then
  echo "[FATAL] Stage1 checkpoint not produced: ${STAGE1_CKPT}"
  STAGE1_STATUS="failed"
  write_summary_status
  exit 1
fi
STAGE1_STATUS="success"
write_summary_status
echo "[queue] Stage1 checkpoint ready: ${STAGE1_CKPT}"

# --- Phase 2: No-bg 4-variant eval ---
CURRENT_PHASE="eval"
EVAL_STATUS="running"
write_summary_status
echo "[queue] Starting no-bg 4-variant eval..."

export TS="${TS}" OUT_BASE="${EVAL_ROOT}" HACA_CKPT="${HACA_CKPT}" S1_CKPT="${STAGE1_CKPT}"
bash "${REPO_ROOT}/scripts/run_nobg_mot20_eval.sh"

EVAL_STATUS="success"
CURRENT_PHASE="done"
write_summary_status

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/append_experiment_record.py" \
  --csv "${REGISTRY_CSV}" \
  --kind queue \
  --script "scripts/run_nobg_mot20_followup.sh" \
  --dataset MOT20 \
  --split train_half_val_half \
  --tracker-family BoT-SORT \
  --variant "mot20_nobg_followup" \
  --run-root "${OUT_ROOT}" \
  --summary-csv "${OUT_ROOT}/summary.csv" \
  --checkpoint "${HACA_CKPT}" \
  --log-path "${LOG_PATH}" \
  --extra \
    "stage1_checkpoint=${STAGE1_CKPT}" \
    "no_background=true"

update_plan_status completed
echo "[done] All phases complete."
echo "[done] Stage1: ${STAGE1_CKPT}"
echo "[done] Eval: ${EVAL_ROOT}/summary.csv"
