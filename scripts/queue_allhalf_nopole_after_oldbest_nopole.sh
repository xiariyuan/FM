#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
PLAN_CSV="${EXPERIMENT_PLAN_CSV:-${REPO_ROOT}/outputs/experiment_plan.csv}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/outputs/queued_runs}"
LOG_PATH="${LOG_DIR}/queue_allhalf_nopole_after_oldbest_nopole_${TS}.log"

WAIT_RUN_ROOT="${WAIT_RUN_ROOT:-${REPO_ROOT}/outputs/lpb_ltra_eval_mot20_learned_oldbest_nopole_20260312_213634}"
WAIT_SUMMARY="${WAIT_SUMMARY:-${WAIT_RUN_ROOT}/eval/mot20_summary.csv}"
CALIBRATOR_NPZ="${CALIBRATOR_NPZ:-${REPO_ROOT}/outputs/lpb_ltra_allhalf_final_20260311_221239/mot17_lpb_ltra_allhalf_20260311_221239.npz}"
MOT20_ALLHALF_NOPOLE_OUT_ROOT="${MOT20_ALLHALF_NOPOLE_OUT_ROOT:-${REPO_ROOT}/outputs/lpb_ltra_eval_mot20_learned_allhalf_nopole_${TS}}"

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "[start] $(date '+%F %T %z')"
echo "[wait_summary] ${WAIT_SUMMARY}"
echo "[out_root] ${MOT20_ALLHALF_NOPOLE_OUT_ROOT}"
echo "[calibrator] ${CALIBRATOR_NPZ}"

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/upsert_experiment_plan.py" \
  --csv "${PLAN_CSV}" \
  --status queued \
  --kind eval \
  --script "scripts/run_botsort_lpb_ltra_mot20_eval.sh" \
  --dataset MOT20 \
  --split val_half \
  --tracker-family BoT-SORT \
  --variant lpb_ltra_learned_allhalf_nopole \
  --run-root "${MOT20_ALLHALF_NOPOLE_OUT_ROOT}" \
  --summary-csv "${MOT20_ALLHALF_NOPOLE_OUT_ROOT}/eval/mot20_summary.csv" \
  --calibrator-npz "${CALIBRATOR_NPZ}" \
  --log-path "${MOT20_ALLHALF_NOPOLE_OUT_ROOT}/eval.log" \
  --key "run_root:${MOT20_ALLHALF_NOPOLE_OUT_ROOT}" \
  --notes "queued_after_oldbest_nopole" \
  --extra "run_base=0" "run_heuristic=0" "run_learned=1" "laplace_primary_only=1" "laplace_no_det_score=0" "laplace_disable_pole_bank=1" "laplace_decay_scales=1 2 4" "laplace_min_history=3" "laplace_proto_mode=multi" "cmc_method=file"

while [[ ! -f "${WAIT_SUMMARY}" ]]; do
  echo "[wait] summary missing, sleep 60s: ${WAIT_SUMMARY}"
  sleep 60
done

echo "[resume] found oldbest nopole summary $(date '+%F %T %z')"

env \
  RUN_ROOT="${MOT20_ALLHALF_NOPOLE_OUT_ROOT}" \
  CALIBRATOR_NPZ="${CALIBRATOR_NPZ}" \
  RUN_BASE=0 \
  RUN_HEURISTIC=0 \
  RUN_LEARNED=1 \
  LAPLACE_PRIMARY_ONLY=1 \
  LAPLACE_NO_DET_SCORE=0 \
  LAPLACE_DISABLE_POLE_BANK=1 \
  /bin/bash "${REPO_ROOT}/scripts/run_botsort_lpb_ltra_mot20_eval.sh"

echo "[done] $(date '+%F %T %z')"
