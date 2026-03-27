#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"
BOT_ROOT="${BOT_ROOT:-${REPO_ROOT}/external/BoT-SORT-main}"
DATA_ROOT="${DATA_ROOT:-/gemini/code/datasets}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
HACA_LABEL="${HACA_LABEL:-haca_v1}"
HACA_MODE="${HACA_MODE:-haca_v1}"
SCRIPT_NAME="${SCRIPT_NAME:-scripts/run_botsort_haca_v1_mot20_eval.sh}"
VARIANT_NAME="${VARIANT_NAME:-${HACA_LABEL}_mot20}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/outputs/${HACA_LABEL}_mot20_eval_${TS}}"
RUN_ROOT="$(realpath -m "${RUN_ROOT}")"
LOG_PATH="${RUN_ROOT}/eval.log"
REGISTRY_CSV="${EXPERIMENT_REGISTRY_CSV:-${REPO_ROOT}/outputs/experiment_registry.csv}"
PLAN_CSV="${EXPERIMENT_PLAN_CSV:-${REPO_ROOT}/outputs/experiment_plan.csv}"
PLAN_KEY="${PLAN_KEY:-run_root:${RUN_ROOT}}"
SUMMARY_CSV="${RUN_ROOT}/eval/mot20_summary.csv"

SEQ_IDS=(${SEQ_IDS:-1 2 3 5})
RUN_BASE="${RUN_BASE:-0}"
RUN_HEURISTIC="${RUN_HEURISTIC:-0}"
RUN_CURRENT_LEARNED="${RUN_CURRENT_LEARNED:-0}"
RUN_HACA="${RUN_HACA:-1}"

EXP_FILE="${EXP_FILE:-./yolox/exps/example/mot/yolox_x_mix_mot20_ch.py}"
CKPT="${CKPT:-./pretrained/bytetrack_x_mot20.pth.tar}"
REID_CFG="${REID_CFG:-fast_reid/configs/MOT20/sbs_S50.yml}"
REID_WTS="${REID_WTS:-pretrained/mot20_sbs_S50.pth}"

CURRENT_CALIBRATOR_NPZ="${CURRENT_CALIBRATOR_NPZ:-}"
HACA_NPZ="${HACA_NPZ:-}"
LAPLACE_DECAY_SCALES="${LAPLACE_DECAY_SCALES:-1 2 4}"
LAPLACE_MIN_HISTORY="${LAPLACE_MIN_HISTORY:-3}"
LAPLACE_PROTO_MODE="${LAPLACE_PROTO_MODE:-multi}"
LAPLACE_PRIMARY_ONLY="${LAPLACE_PRIMARY_ONLY:-1}"
LAPLACE_NO_DET_SCORE="${LAPLACE_NO_DET_SCORE:-0}"
CMC_METHOD="${CMC_METHOD:-file}"
HACA_DISABLE_SET_ENCODER="${HACA_DISABLE_SET_ENCODER:-0}"
HACA_DISABLE_BACKGROUND="${HACA_DISABLE_BACKGROUND:-0}"
HACA_DELTA_SCALE="${HACA_DELTA_SCALE:-}"

mkdir -p "${RUN_ROOT}"
exec > >(tee -a "${LOG_PATH}") 2>&1

update_plan_status() {
  local status="$1"
  shift || true
  local extras=(
    "run_base=${RUN_BASE}"
    "run_heuristic=${RUN_HEURISTIC}"
    "run_current_learned=${RUN_CURRENT_LEARNED}"
    "run_haca=${RUN_HACA}"
    "laplace_primary_only=${LAPLACE_PRIMARY_ONLY}"
    "laplace_no_det_score=${LAPLACE_NO_DET_SCORE}"
    "laplace_decay_scales=${LAPLACE_DECAY_SCALES}"
    "laplace_min_history=${LAPLACE_MIN_HISTORY}"
    "laplace_proto_mode=${LAPLACE_PROTO_MODE}"
    "cmc_method=${CMC_METHOD}"
    "haca_disable_set_encoder=${HACA_DISABLE_SET_ENCODER}"
    "haca_disable_background=${HACA_DISABLE_BACKGROUND}"
  )
  if [[ $# -gt 0 ]]; then
    extras+=("$@")
  fi
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/upsert_experiment_plan.py" \
    --csv "${PLAN_CSV}" \
    --key "${PLAN_KEY}" \
    --status "${status}" \
    --kind eval \
    --script "${SCRIPT_NAME}" \
    --dataset MOT20 \
    --split val_half \
    --tracker-family BoT-SORT \
    --variant "${VARIANT_NAME}" \
    --run-root "${RUN_ROOT}" \
    --summary-csv "${SUMMARY_CSV}" \
    --checkpoint "${HACA_NPZ}" \
    --calibrator-npz "${CURRENT_CALIBRATOR_NPZ}" \
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
      "${PYTHON_BIN}" -u tools/track.py "${DATA_ROOT}/MOT20"
      --benchmark MOT20
      --eval val
      --seq-ids "${SEQ_IDS[@]}"
      -f "${EXP_FILE}"
      -c "${CKPT}"
      --with-reid
      --fast-reid-config "${REID_CFG}"
      --fast-reid-weights "${REID_WTS}"
      --cmc-method "${CMC_METHOD}"
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
    --dataset MOT20 \
    --data-root "${DATA_ROOT}" \
    --results-dir "${results_dir}" \
    --tracker-name "${tracker_name}" \
    --work-dir "${work_dir}" \
    --remap-results-from-fullval
}

RESULT_DIRS=()

if [[ "${RUN_BASE}" == "1" ]]; then
  BASE_EXP="mot20_haca_base_${TS}"
  BASE_RESULTS="${BOT_ROOT}/YOLOX_outputs/${BASE_EXP}/track_results"
  echo "[step] base"
  run_track "${BASE_EXP}" ""
  eval_results "${BASE_RESULTS}" "${BASE_EXP}" "${RUN_ROOT}/eval/base"
  RESULT_DIRS+=("${RUN_ROOT}/eval/base/eval/${BASE_EXP}")
fi

if [[ "${RUN_HEURISTIC}" == "1" ]]; then
  HEUR_EXP="mot20_${HACA_LABEL}_heuristic_${TS}"
  HEUR_RESULTS="${BOT_ROOT}/YOLOX_outputs/${HEUR_EXP}/track_results"
  echo "[step] heuristic"
  EXTRA_ARGS=(--laplace-assoc --laplace-assoc-mode heuristic --laplace-decay-scales ${LAPLACE_DECAY_SCALES} --laplace-min-history "${LAPLACE_MIN_HISTORY}" --laplace-proto-mode "${LAPLACE_PROTO_MODE}")
  if [[ "${LAPLACE_PRIMARY_ONLY}" == "1" ]]; then
    EXTRA_ARGS+=(--laplace-primary-only)
  fi
  if [[ "${LAPLACE_NO_DET_SCORE}" == "1" ]]; then
    EXTRA_ARGS+=(--laplace-no-det-score)
  fi
  run_track "${HEUR_EXP}" "${RUN_ROOT}/pair_logs/heuristic" "${EXTRA_ARGS[@]}"
  eval_results "${HEUR_RESULTS}" "${HEUR_EXP}" "${RUN_ROOT}/eval/heuristic"
  RESULT_DIRS+=("${RUN_ROOT}/eval/heuristic/eval/${HEUR_EXP}")
fi

if [[ "${RUN_CURRENT_LEARNED}" == "1" ]]; then
  if [[ -z "${CURRENT_CALIBRATOR_NPZ}" ]]; then
    echo "CURRENT_CALIBRATOR_NPZ is required when RUN_CURRENT_LEARNED=1" >&2
    exit 1
  fi
  LEARNED_EXP="mot20_${HACA_LABEL}_currentlearned_${TS}"
  LEARNED_RESULTS="${BOT_ROOT}/YOLOX_outputs/${LEARNED_EXP}/track_results"
  echo "[step] current_learned"
  EXTRA_ARGS=(--laplace-assoc --laplace-assoc-mode current_learned --laplace-decay-scales ${LAPLACE_DECAY_SCALES} --laplace-min-history "${LAPLACE_MIN_HISTORY}" --laplace-proto-mode "${LAPLACE_PROTO_MODE}" --laplace-calibrator "${CURRENT_CALIBRATOR_NPZ}")
  if [[ "${LAPLACE_PRIMARY_ONLY}" == "1" ]]; then
    EXTRA_ARGS+=(--laplace-primary-only)
  fi
  if [[ "${LAPLACE_NO_DET_SCORE}" == "1" ]]; then
    EXTRA_ARGS+=(--laplace-no-det-score)
  fi
  run_track "${LEARNED_EXP}" "${RUN_ROOT}/pair_logs/current_learned" "${EXTRA_ARGS[@]}"
  eval_results "${LEARNED_RESULTS}" "${LEARNED_EXP}" "${RUN_ROOT}/eval/current_learned"
  RESULT_DIRS+=("${RUN_ROOT}/eval/current_learned/eval/${LEARNED_EXP}")
fi

if [[ "${RUN_HACA}" == "1" ]]; then
  if [[ -z "${HACA_NPZ}" ]]; then
    echo "HACA_NPZ is required when RUN_HACA=1" >&2
    exit 1
  fi
  HACA_EXP="mot20_${HACA_LABEL}_${TS}"
  HACA_RESULTS="${BOT_ROOT}/YOLOX_outputs/${HACA_EXP}/track_results"
  echo "[step] ${HACA_LABEL}"
  EXTRA_ARGS=(--laplace-assoc --laplace-assoc-mode "${HACA_MODE}" --laplace-decay-scales ${LAPLACE_DECAY_SCALES} --laplace-min-history "${LAPLACE_MIN_HISTORY}" --laplace-proto-mode "${LAPLACE_PROTO_MODE}" --laplace-haca-checkpoint "${HACA_NPZ}")
  if [[ "${LAPLACE_PRIMARY_ONLY}" == "1" ]]; then
    EXTRA_ARGS+=(--laplace-primary-only)
  fi
  if [[ "${LAPLACE_NO_DET_SCORE}" == "1" ]]; then
    EXTRA_ARGS+=(--laplace-no-det-score)
  fi
  if [[ "${HACA_DISABLE_SET_ENCODER}" == "1" ]]; then
    EXTRA_ARGS+=(--laplace-haca-no-set-encoder)
  fi
  if [[ "${HACA_DISABLE_BACKGROUND}" == "1" ]]; then
    EXTRA_ARGS+=(--laplace-haca-no-background)
  fi
  if [[ -n "${HACA_DELTA_SCALE}" ]]; then
    EXTRA_ARGS+=(--laplace-haca-delta-scale "${HACA_DELTA_SCALE}")
  fi
  run_track "${HACA_EXP}" "${RUN_ROOT}/pair_logs/haca" "${EXTRA_ARGS[@]}"
  eval_results "${HACA_RESULTS}" "${HACA_EXP}" "${RUN_ROOT}/eval/haca"
  RESULT_DIRS+=("${RUN_ROOT}/eval/haca/eval/${HACA_EXP}")
fi

if [[ "${#RESULT_DIRS[@]}" -eq 0 ]]; then
  echo "[skip] no variants enabled; nothing to evaluate"
  exit 0
fi

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/collect_trackeval_metrics.py" \
  "${RESULT_DIRS[@]}" \
  --csv "${SUMMARY_CSV}" | tee "${RUN_ROOT}/eval/mot20_summary.txt"
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/append_experiment_record.py" \
  --csv "${REGISTRY_CSV}" \
  --kind eval \
  --script "${SCRIPT_NAME}" \
  --dataset MOT20 \
  --split val_half \
  --tracker-family BoT-SORT \
  --variant "${VARIANT_NAME}" \
  --run-root "${RUN_ROOT}" \
  --summary-csv "${SUMMARY_CSV}" \
  --checkpoint "${HACA_NPZ}" \
  --calibrator-npz "${CURRENT_CALIBRATOR_NPZ}" \
  --log-path "${LOG_PATH}" \
  --extra \
    run_base="${RUN_BASE}" \
    run_heuristic="${RUN_HEURISTIC}" \
    run_current_learned="${RUN_CURRENT_LEARNED}" \
    run_haca="${RUN_HACA}" \
    laplace_primary_only="${LAPLACE_PRIMARY_ONLY}" \
    laplace_no_det_score="${LAPLACE_NO_DET_SCORE}" \
    laplace_decay_scales="${LAPLACE_DECAY_SCALES}" \
    laplace_min_history="${LAPLACE_MIN_HISTORY}" \
    laplace_proto_mode="${LAPLACE_PROTO_MODE}" \
    haca_disable_set_encoder="${HACA_DISABLE_SET_ENCODER}" \
    haca_disable_background="${HACA_DISABLE_BACKGROUND}"
update_plan_status completed

echo "[done] $(date '+%F %T %z')"
echo "[summary] ${SUMMARY_CSV}"
