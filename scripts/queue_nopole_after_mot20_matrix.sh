#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
PLAN_CSV="${EXPERIMENT_PLAN_CSV:-${REPO_ROOT}/outputs/experiment_plan.csv}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/outputs/queued_runs}"
LOG_PATH="${LOG_DIR}/queue_nopole_after_mot20_matrix_${TS}.log"

WAIT_RUN_ROOT="${WAIT_RUN_ROOT:-${REPO_ROOT}/outputs/lpb_ltra_eval_mot20_matrix_oldbest_20260312_181838}"
WAIT_SUMMARY="${WAIT_SUMMARY:-${WAIT_RUN_ROOT}/eval/mot20_summary.csv}"
CALIBRATOR_NPZ="${CALIBRATOR_NPZ:-${REPO_ROOT}/outputs/lpb_ltra_formal_mot17_shrink_20260311_191211/train_vec_20260311_201832/mot17_lpb_ltra_20260311_201832.npz}"

MOT20_NOPOLE_OUT_ROOT="${MOT20_NOPOLE_OUT_ROOT:-${REPO_ROOT}/outputs/lpb_ltra_eval_mot20_learned_oldbest_nopole_${TS}}"
STRONGSORT_NOPOLE_OUT_ROOT="${STRONGSORT_NOPOLE_OUT_ROOT:-${REPO_ROOT}/outputs/strongsort_lpb_ltra/MOT17_val_oldbest_debug_full_nopole_${TS}}"

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_PATH}") 2>&1

queue_plan_row() {
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/upsert_experiment_plan.py" \
    --csv "${PLAN_CSV}" \
    --status queued \
    "$@"
}

run_step() {
  local label="$1"
  shift
  echo "[step] ${label} $(date '+%F %T %z')"
  set +e
  "$@"
  local rc=$?
  set -e
  echo "[step-exit] ${label} rc=${rc} $(date '+%F %T %z')"
  return ${rc}
}

echo "[start] $(date '+%F %T %z')"
echo "[wait_summary] ${WAIT_SUMMARY}"
echo "[mot20_nopole_out] ${MOT20_NOPOLE_OUT_ROOT}"
echo "[strongsort_nopole_out] ${STRONGSORT_NOPOLE_OUT_ROOT}"

queue_plan_row \
  --kind eval \
  --script "scripts/run_botsort_lpb_ltra_mot20_eval.sh" \
  --dataset MOT20 \
  --split val_half \
  --tracker-family BoT-SORT \
  --variant lpb_ltra_learned_nopole \
  --run-root "${MOT20_NOPOLE_OUT_ROOT}" \
  --summary-csv "${MOT20_NOPOLE_OUT_ROOT}/eval/mot20_summary.csv" \
  --calibrator-npz "${CALIBRATOR_NPZ}" \
  --log-path "${MOT20_NOPOLE_OUT_ROOT}/eval.log" \
  --key "run_root:${MOT20_NOPOLE_OUT_ROOT}" \
  --notes "queued_after_mot20_matrix" \
  --extra "run_base=0" "run_heuristic=0" "run_learned=1" "laplace_primary_only=1" "laplace_no_det_score=0" "laplace_disable_pole_bank=1" "laplace_decay_scales=1 2 4" "laplace_min_history=3" "laplace_proto_mode=multi" "cmc_method=file"

queue_plan_row \
  --kind eval \
  --script "scripts/run_strongsort_lpb_ltra_mot17_val.sh" \
  --dataset MOT17 \
  --split val_half \
  --tracker-family StrongSORT \
  --variant lpb_ltra_learned_nopole_debug \
  --run-root "${STRONGSORT_NOPOLE_OUT_ROOT}" \
  --summary-csv "${STRONGSORT_NOPOLE_OUT_ROOT}/summary.csv" \
  --calibrator-npz "${CALIBRATOR_NPZ}" \
  --log-path "${STRONGSORT_NOPOLE_OUT_ROOT}/run.log" \
  --key "run_root:${STRONGSORT_NOPOLE_OUT_ROOT}" \
  --notes "queued_after_mot20_matrix" \
  --extra "analysis_dir=${STRONGSORT_NOPOLE_OUT_ROOT}/pair_logs" "laplace_disable_pole_bank=1"

while [[ ! -f "${WAIT_SUMMARY}" ]]; do
  echo "[wait] summary missing, sleep 60s: ${WAIT_SUMMARY}"
  sleep 60
done

echo "[resume] found matrix summary $(date '+%F %T %z')"

run_step "mot20_learned_nopole" \
  env \
    RUN_ROOT="${MOT20_NOPOLE_OUT_ROOT}" \
    CALIBRATOR_NPZ="${CALIBRATOR_NPZ}" \
    RUN_BASE=0 \
    RUN_HEURISTIC=0 \
    RUN_LEARNED=1 \
    LAPLACE_PRIMARY_ONLY=1 \
    LAPLACE_NO_DET_SCORE=0 \
    LAPLACE_DISABLE_POLE_BANK=1 \
    /bin/bash "${REPO_ROOT}/scripts/run_botsort_lpb_ltra_mot20_eval.sh" \
  || true

run_step "strongsort_full_debug_nopole" \
  env \
    OUT_ROOT="${STRONGSORT_NOPOLE_OUT_ROOT}" \
    CALIBRATOR_NPZ="${CALIBRATOR_NPZ}" \
    LAPLACE_DISABLE_POLE_BANK=1 \
    /bin/bash "${REPO_ROOT}/scripts/run_strongsort_lpb_ltra_mot17_val.sh" \
  || true

echo "[done] $(date '+%F %T %z')"
