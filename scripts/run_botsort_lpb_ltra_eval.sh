#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"
BOT_ROOT="${BOT_ROOT:-${REPO_ROOT}/external/BoT-SORT-main}"
DATA_ROOT="${DATA_ROOT:-/gemini/code/datasets}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/outputs/lpb_ltra_eval_${TS}}"
RUN_ROOT="$(realpath -m "${RUN_ROOT}")"
LOG_PATH="${RUN_ROOT}/eval.log"
REGISTRY_CSV="${EXPERIMENT_REGISTRY_CSV:-${REPO_ROOT}/outputs/experiment_registry.csv}"
PLAN_CSV="${EXPERIMENT_PLAN_CSV:-${REPO_ROOT}/outputs/experiment_plan.csv}"
PLAN_KEY="${PLAN_KEY:-run_root:${RUN_ROOT}}"
SUMMARY_CSV="${RUN_ROOT}/eval/mot17_summary.csv"

SEQ_IDS=(${SEQ_IDS:-2 4 5 9 10 11 13})
RUN_BASE="${RUN_BASE:-1}"
RUN_HEURISTIC="${RUN_HEURISTIC:-1}"
RUN_LEARNED="${RUN_LEARNED:-1}"

EXP_FILE="${EXP_FILE:-./yolox/exps/example/mot/yolox_x_mix_det.py}"
CKPT="${CKPT:-./pretrained/bytetrack_x_mot17.pth.tar}"
REID_CFG="${REID_CFG:-fast_reid/configs/MOT17/sbs_S50.yml}"
REID_WTS="${REID_WTS:-pretrained/mot17_sbs_S50.pth}"

CALIBRATOR_NPZ="${CALIBRATOR_NPZ:-}"
LAPLACE_DECAY_SCALES="${LAPLACE_DECAY_SCALES:-1 2 4}"
LAPLACE_MIN_HISTORY="${LAPLACE_MIN_HISTORY:-3}"
LAPLACE_PROTO_MODE="${LAPLACE_PROTO_MODE:-multi}"
LAPLACE_PRIMARY_ONLY="${LAPLACE_PRIMARY_ONLY:-1}"

mkdir -p "${RUN_ROOT}"
exec > >(tee -a "${LOG_PATH}") 2>&1

update_plan_status() {
  local status="$1"
  shift || true
  local extras=(
    "run_base=${RUN_BASE}"
    "run_heuristic=${RUN_HEURISTIC}"
    "run_learned=${RUN_LEARNED}"
    "laplace_primary_only=${LAPLACE_PRIMARY_ONLY}"
    "laplace_decay_scales=${LAPLACE_DECAY_SCALES}"
    "laplace_min_history=${LAPLACE_MIN_HISTORY}"
    "laplace_proto_mode=${LAPLACE_PROTO_MODE}"
  )
  if [[ $# -gt 0 ]]; then
    extras+=("$@")
  fi
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/upsert_experiment_plan.py" \
    --csv "${PLAN_CSV}" \
    --key "${PLAN_KEY}" \
    --status "${status}" \
    --kind eval \
    --script "scripts/run_botsort_lpb_ltra_eval.sh" \
    --dataset MOT17 \
    --split val_half \
    --tracker-family BoT-SORT \
    --variant lpb_ltra_matrix \
    --run-root "${RUN_ROOT}" \
    --summary-csv "${SUMMARY_CSV}" \
    --calibrator-npz "${CALIBRATOR_NPZ}" \
    --log-path "${LOG_PATH}" \
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

run_track() {
  local experiment_name="$1"
  local analysis_dir="$2"
  shift 2
  (
    cd "${BOT_ROOT}"
    cmd=(
      "${PYTHON_BIN}" -u tools/track.py "${DATA_ROOT}/MOT17"
      --benchmark MOT17
      --eval val
      --seq-ids "${SEQ_IDS[@]}"
      --mot17-detector-exts FRCNN
      -f "${EXP_FILE}"
      -c "${CKPT}"
      --with-reid
      --fast-reid-config "${REID_CFG}"
      --fast-reid-weights "${REID_WTS}"
      --experiment-name "${experiment_name}"
    )
    if [[ -n "${analysis_dir}" ]]; then
      mkdir -p "${analysis_dir}"
      cmd+=(--laplace-analysis-dir "${analysis_dir}")
    fi
    cmd+=("$@")
    echo "[track] ${cmd[*]}"
    "${cmd[@]}"
  )
}

eval_results() {
  local results_dir="$1"
  local tracker_name="$2"
  local work_dir="$3"
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/eval_botsort_halfval_trackeval.py" \
    --dataset MOT17 \
    --data-root "${DATA_ROOT}" \
    --results-dir "${results_dir}" \
    --tracker-name "${tracker_name}" \
    --work-dir "${work_dir}"
}

RESULT_DIRS=()
TRACKER_EVAL_DIRS=()

if [[ "${RUN_BASE}" == "1" ]]; then
  BASE_EXP="mot17_frcnn_base_${TS}"
  BASE_RESULTS="${BOT_ROOT}/YOLOX_outputs/${BASE_EXP}/track_results"
  echo "[step] base"
  run_track "${BASE_EXP}" ""
  eval_results "${BASE_RESULTS}" "${BASE_EXP}" "${RUN_ROOT}/eval/base"
  RESULT_DIRS+=("${RUN_ROOT}/eval/base/eval/${BASE_EXP}")
fi

if [[ "${RUN_HEURISTIC}" == "1" ]]; then
  HEUR_EXP="mot17_frcnn_lpb_heuristic_${TS}"
  HEUR_RESULTS="${BOT_ROOT}/YOLOX_outputs/${HEUR_EXP}/track_results"
  echo "[step] heuristic"
  EXTRA_ARGS=(--laplace-assoc --laplace-decay-scales ${LAPLACE_DECAY_SCALES} --laplace-min-history "${LAPLACE_MIN_HISTORY}" --laplace-proto-mode "${LAPLACE_PROTO_MODE}")
  if [[ "${LAPLACE_PRIMARY_ONLY}" == "1" ]]; then
    EXTRA_ARGS+=(--laplace-primary-only)
  fi
  run_track "${HEUR_EXP}" "${RUN_ROOT}/pair_logs/heuristic" "${EXTRA_ARGS[@]}"
  eval_results "${HEUR_RESULTS}" "${HEUR_EXP}" "${RUN_ROOT}/eval/heuristic"
  RESULT_DIRS+=("${RUN_ROOT}/eval/heuristic/eval/${HEUR_EXP}")
fi

if [[ "${RUN_LEARNED}" == "1" ]]; then
  if [[ -z "${CALIBRATOR_NPZ}" ]]; then
    echo "CALIBRATOR_NPZ is required when RUN_LEARNED=1" >&2
    exit 1
  fi
  LEARNED_EXP="mot17_frcnn_lpb_learned_${TS}"
  LEARNED_RESULTS="${BOT_ROOT}/YOLOX_outputs/${LEARNED_EXP}/track_results"
  echo "[step] learned"
  EXTRA_ARGS=(--laplace-assoc --laplace-decay-scales ${LAPLACE_DECAY_SCALES} --laplace-min-history "${LAPLACE_MIN_HISTORY}" --laplace-proto-mode "${LAPLACE_PROTO_MODE}" --laplace-calibrator "${CALIBRATOR_NPZ}")
  if [[ "${LAPLACE_PRIMARY_ONLY}" == "1" ]]; then
    EXTRA_ARGS+=(--laplace-primary-only)
  fi
  run_track "${LEARNED_EXP}" "${RUN_ROOT}/pair_logs/learned" "${EXTRA_ARGS[@]}"
  eval_results "${LEARNED_RESULTS}" "${LEARNED_EXP}" "${RUN_ROOT}/eval/learned"
  RESULT_DIRS+=("${RUN_ROOT}/eval/learned/eval/${LEARNED_EXP}")
fi

if [[ "${#RESULT_DIRS[@]}" -eq 0 ]]; then
  echo "[skip] no variants enabled; nothing to evaluate"
  exit 0
fi

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/collect_trackeval_metrics.py" "${RESULT_DIRS[@]}" --csv "${SUMMARY_CSV}" | tee "${RUN_ROOT}/eval/mot17_summary.txt"
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/append_experiment_record.py" \
  --csv "${REGISTRY_CSV}" \
  --kind eval \
  --script "scripts/run_botsort_lpb_ltra_eval.sh" \
  --dataset MOT17 \
  --split val_half \
  --tracker-family BoT-SORT \
  --variant lpb_ltra_matrix \
  --run-root "${RUN_ROOT}" \
  --summary-csv "${SUMMARY_CSV}" \
  --calibrator-npz "${CALIBRATOR_NPZ}" \
  --log-path "${LOG_PATH}" \
  --extra \
    run_base="${RUN_BASE}" \
    run_heuristic="${RUN_HEURISTIC}" \
    run_learned="${RUN_LEARNED}" \
    laplace_primary_only="${LAPLACE_PRIMARY_ONLY}" \
    laplace_decay_scales="${LAPLACE_DECAY_SCALES}" \
    laplace_min_history="${LAPLACE_MIN_HISTORY}" \
    laplace_proto_mode="${LAPLACE_PROTO_MODE}"
update_plan_status completed

echo "[done] $(date '+%F %T %z')"
echo "[summary] ${SUMMARY_CSV}"
