#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"

CURRENT_MOT20_ROOT="${CURRENT_MOT20_ROOT:-${REPO_ROOT}/outputs/lpb_ltra_eval_mot20_learned_20260311_235019}"
CURRENT_MOT20_PATTERN="${CURRENT_MOT20_PATTERN:-mot20_lpb_learned_20260311_235019}"
CURRENT_SMOKE_ROOT="${CURRENT_SMOKE_ROOT:-${REPO_ROOT}/outputs/strongsort_lpb_ltra_smoke_20260312_001304}"
CURRENT_SMOKE_PATTERN="${CURRENT_SMOKE_PATTERN:-strongsort_lpb_ltra_smoke_20260312_001304}"

CALIBRATOR_NPZ="${CALIBRATOR_NPZ:-${REPO_ROOT}/outputs/lpb_ltra_allhalf_final_20260311_221239/mot17_lpb_ltra_allhalf_20260311_221239.npz}"
ALT_CALIBRATOR_NPZ="${ALT_CALIBRATOR_NPZ:-${REPO_ROOT}/outputs/lpb_ltra_formal_mot17_shrink_20260311_191211/train_vec_20260311_201832/mot17_lpb_ltra_20260311_201832.npz}"

SS_FULL_OUT="${SS_FULL_OUT:-${REPO_ROOT}/outputs/strongsort_lpb_ltra/MOT17_val_allhalf}"
SS_ALT_OUT="${SS_ALT_OUT:-${REPO_ROOT}/outputs/strongsort_lpb_ltra/MOT17_val_oldbest}"
MOT20_NODET_OUT="${MOT20_NODET_OUT:-${REPO_ROOT}/outputs/lpb_ltra_eval_mot20_learned_nodetscore}"
MOT20_ALT_OUT="${MOT20_ALT_OUT:-${REPO_ROOT}/outputs/lpb_ltra_eval_mot20_learned_oldbest}"
REPORT_OUT="${REPORT_OUT:-${REPO_ROOT}/outputs/lpb_overnight_report_$(date +%Y%m%d_%H%M%S).txt}"

LOG_PATH="${REPORT_OUT%.txt}.log"
REGISTRY_CSV="${EXPERIMENT_REGISTRY_CSV:-${REPO_ROOT}/outputs/experiment_registry.csv}"
PLAN_CSV="${EXPERIMENT_PLAN_CSV:-${REPO_ROOT}/outputs/experiment_plan.csv}"
PLAN_KEY="${PLAN_KEY:-run_root:${REPORT_OUT}}"
mkdir -p "$(dirname "${REPORT_OUT}")"
exec > >(tee -a "${LOG_PATH}") 2>&1

wait_for_summary_or_fail() {
  local summary_path="$1"
  local process_pattern="$2"
  local label="$3"
  while true; do
    if [[ -f "${summary_path}" ]]; then
      echo "[ready] ${label}: ${summary_path}"
      return 0
    fi
    if ! ps -eo cmd | grep -E "${process_pattern}" | grep -v grep >/dev/null 2>&1; then
      echo "[error] ${label} stopped without summary: ${summary_path}" >&2
      return 1
    fi
    sleep 60
  done
}

append_if_ready() {
  local summary_csv="$1"
  shift
  if [[ ! -f "${summary_csv}" ]]; then
    echo "[skip] summary missing, cannot append: ${summary_csv}"
    return 0
  fi
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/append_experiment_record.py" \
    --csv "${REGISTRY_CSV}" \
    --summary-csv "${summary_csv}" \
    "$@"
}

update_orchestrator_status() {
  local status="$1"
  shift || true
  local extras=(
    "current_mot20_root=${CURRENT_MOT20_ROOT}"
    "current_smoke_root=${CURRENT_SMOKE_ROOT}"
    "ss_full_out=${SS_FULL_OUT}"
    "ss_alt_out=${SS_ALT_OUT}"
    "mot20_nodet_out=${MOT20_NODET_OUT}"
    "mot20_alt_out=${MOT20_ALT_OUT}"
  )
  if [[ $# -gt 0 ]]; then
    extras+=("$@")
  fi
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/upsert_experiment_plan.py" \
    --csv "${PLAN_CSV}" \
    --key "${PLAN_KEY}" \
    --status "${status}" \
    --kind other \
    --script "scripts/overnight_lpb_followups.sh" \
    --dataset "MULTI" \
    --split "overnight" \
    --tracker-family "MULTI" \
    --variant "lpb_followups" \
    --run-root "${REPORT_OUT}" \
    --log-path "${LOG_PATH}" \
    --extra "${extras[@]}"
}

queue_child_plan() {
  local child_key="$1"
  shift
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/upsert_experiment_plan.py" \
    --csv "${PLAN_CSV}" \
    --key "${child_key}" \
    --status queued \
    "$@"
}

on_exit() {
  local rc=$?
  trap - EXIT
  if [[ ${rc} -ne 0 ]]; then
    update_orchestrator_status failed "exit_code=${rc}" || true
  fi
  exit ${rc}
}

trap on_exit EXIT
update_orchestrator_status running

echo "[start] $(date '+%F %T %z')"
echo "[current_mot20_root] ${CURRENT_MOT20_ROOT}"
echo "[current_smoke_root] ${CURRENT_SMOKE_ROOT}"
echo "[calibrator] ${CALIBRATOR_NPZ}"
echo "[alt_calibrator] ${ALT_CALIBRATOR_NPZ}"

wait_for_summary_or_fail "${CURRENT_SMOKE_ROOT}/summary.csv" "${CURRENT_SMOKE_PATTERN}" "strongsort smoke"
append_if_ready "${CURRENT_SMOKE_ROOT}/summary.csv" \
  --kind eval \
  --script "scripts/run_strongsort_lpb_ltra_mot17_val.sh" \
  --dataset MOT17 \
  --split val_half \
  --tracker-family StrongSORT \
  --variant lpb_ltra_learned_smoke \
  --tag overnight_backfill \
  --run-root "${CURRENT_SMOKE_ROOT}" \
  --calibrator-npz "${CALIBRATOR_NPZ}" \
  --log-path "${CURRENT_SMOKE_ROOT}/run.log" \
  --extra seq_override="MOT17-02-FRCNN"
wait_for_summary_or_fail "${CURRENT_MOT20_ROOT}/eval/mot20_summary.csv" "${CURRENT_MOT20_PATTERN}" "mot20 learned"
append_if_ready "${CURRENT_MOT20_ROOT}/eval/mot20_summary.csv" \
  --kind eval \
  --script "scripts/run_botsort_lpb_ltra_mot20_eval.sh" \
  --dataset MOT20 \
  --split val_half \
  --tracker-family BoT-SORT \
  --variant lpb_ltra_learned \
  --tag overnight_backfill \
  --run-root "${CURRENT_MOT20_ROOT}" \
  --calibrator-npz "${CALIBRATOR_NPZ}" \
  --log-path "${CURRENT_MOT20_ROOT}/eval.log" \
  --extra \
    laplace_primary_only="1" \
    laplace_no_det_score="0" \
    laplace_decay_scales="1 2 4" \
    laplace_min_history="3" \
    laplace_proto_mode="multi" \
    cmc_method="file"

echo "[run] StrongSORT full val with all-half final checkpoint"
if [[ -f "${SS_FULL_OUT}/summary.csv" ]]; then
  echo "[skip] StrongSORT all-half summary exists: ${SS_FULL_OUT}/summary.csv"
else
  queue_child_plan "run_root:${SS_FULL_OUT}" \
    --kind eval \
    --script "scripts/run_strongsort_lpb_ltra_mot17_val.sh" \
    --dataset MOT17 \
    --split val_half \
    --tracker-family StrongSORT \
    --variant lpb_ltra_learned \
    --run-root "${SS_FULL_OUT}" \
    --summary-csv "${SS_FULL_OUT}/summary.csv" \
    --calibrator-npz "${CALIBRATOR_NPZ}" \
    --log-path "${SS_FULL_OUT}/run.log"
  CALIBRATOR_NPZ="${CALIBRATOR_NPZ}" OUT_ROOT="${SS_FULL_OUT}" bash "${REPO_ROOT}/scripts/run_strongsort_lpb_ltra_mot17_val.sh"
fi

if [[ -f "${ALT_CALIBRATOR_NPZ}" ]]; then
  echo "[run] StrongSORT full val with older best checkpoint"
  if [[ -f "${SS_ALT_OUT}/summary.csv" ]]; then
    echo "[skip] StrongSORT old-best summary exists: ${SS_ALT_OUT}/summary.csv"
  else
    queue_child_plan "run_root:${SS_ALT_OUT}" \
      --kind eval \
      --script "scripts/run_strongsort_lpb_ltra_mot17_val.sh" \
      --dataset MOT17 \
      --split val_half \
      --tracker-family StrongSORT \
      --variant lpb_ltra_learned \
      --run-root "${SS_ALT_OUT}" \
      --summary-csv "${SS_ALT_OUT}/summary.csv" \
      --calibrator-npz "${ALT_CALIBRATOR_NPZ}" \
      --log-path "${SS_ALT_OUT}/run.log"
    CALIBRATOR_NPZ="${ALT_CALIBRATOR_NPZ}" OUT_ROOT="${SS_ALT_OUT}" bash "${REPO_ROOT}/scripts/run_strongsort_lpb_ltra_mot17_val.sh"
  fi
fi

echo "[run] MOT20 learned no-det-score"
if [[ -f "${MOT20_NODET_OUT}/eval/mot20_summary.csv" ]]; then
  echo "[skip] MOT20 no-det-score summary exists: ${MOT20_NODET_OUT}/eval/mot20_summary.csv"
else
  queue_child_plan "run_root:${MOT20_NODET_OUT}" \
    --kind eval \
    --script "scripts/run_botsort_lpb_ltra_mot20_eval.sh" \
    --dataset MOT20 \
    --split val_half \
    --tracker-family BoT-SORT \
    --variant lpb_ltra_learned \
    --run-root "${MOT20_NODET_OUT}" \
    --summary-csv "${MOT20_NODET_OUT}/eval/mot20_summary.csv" \
    --calibrator-npz "${CALIBRATOR_NPZ}" \
    --log-path "${MOT20_NODET_OUT}/eval.log" \
    --extra laplace_primary_only="1" laplace_no_det_score="1" laplace_decay_scales="1 2 4" laplace_min_history="3" laplace_proto_mode="multi" cmc_method="file"
  RUN_ROOT="${MOT20_NODET_OUT}" CALIBRATOR_NPZ="${CALIBRATOR_NPZ}" LAPLACE_NO_DET_SCORE=1 bash "${REPO_ROOT}/scripts/run_botsort_lpb_ltra_mot20_eval.sh"
fi

if [[ -f "${ALT_CALIBRATOR_NPZ}" ]]; then
  echo "[run] MOT20 learned with older best checkpoint"
  if [[ -f "${MOT20_ALT_OUT}/eval/mot20_summary.csv" ]]; then
    echo "[skip] MOT20 old-best summary exists: ${MOT20_ALT_OUT}/eval/mot20_summary.csv"
  else
    queue_child_plan "run_root:${MOT20_ALT_OUT}" \
      --kind eval \
      --script "scripts/run_botsort_lpb_ltra_mot20_eval.sh" \
      --dataset MOT20 \
      --split val_half \
      --tracker-family BoT-SORT \
      --variant lpb_ltra_learned \
      --run-root "${MOT20_ALT_OUT}" \
      --summary-csv "${MOT20_ALT_OUT}/eval/mot20_summary.csv" \
      --calibrator-npz "${ALT_CALIBRATOR_NPZ}" \
      --log-path "${MOT20_ALT_OUT}/eval.log" \
      --extra laplace_primary_only="1" laplace_no_det_score="0" laplace_decay_scales="1 2 4" laplace_min_history="3" laplace_proto_mode="multi" cmc_method="file"
    RUN_ROOT="${MOT20_ALT_OUT}" CALIBRATOR_NPZ="${ALT_CALIBRATOR_NPZ}" bash "${REPO_ROOT}/scripts/run_botsort_lpb_ltra_mot20_eval.sh"
  fi
fi

{
  echo "# Overnight LPB Follow-ups"
  echo
  echo "Generated: $(date '+%F %T %z')"
  echo
  for path in \
    "${REPO_ROOT}/outputs/lpb_ltra_eval_allhalf_final_20260311_222116/eval/mot17_summary.csv" \
    "${CURRENT_MOT20_ROOT}/eval/mot20_summary.csv" \
    "${SS_FULL_OUT}/summary.csv" \
    "${SS_ALT_OUT}/summary.csv" \
    "${MOT20_NODET_OUT}/eval/mot20_summary.csv" \
    "${MOT20_ALT_OUT}/eval/mot20_summary.csv"
  do
    if [[ -f "${path}" ]]; then
      echo "## ${path}"
      cat "${path}"
      echo
    fi
  done
} > "${REPORT_OUT}"

update_orchestrator_status completed
echo "[done] $(date '+%F %T %z')"
echo "[report] ${REPORT_OUT}"
