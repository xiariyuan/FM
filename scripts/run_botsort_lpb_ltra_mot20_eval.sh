#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"
BOT_ROOT="${BOT_ROOT:-${REPO_ROOT}/external/BoT-SORT-main}"
DATA_ROOT="${DATA_ROOT:-/gemini/code/datasets}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
DEFAULT_TS="$(date +%Y%m%d_%H%M%S)"
TS="${TS:-${DEFAULT_TS}}"
RUN_ROOT="${RUN_ROOT:-${OUT_ROOT:-${REPO_ROOT}/outputs/lpb_ltra_eval_mot20_${TS}}}"
RUN_ROOT="$(realpath -m "${RUN_ROOT}")"
if [[ "${TS}" == "${DEFAULT_TS}" ]]; then
  RUN_BASENAME="$(basename "${RUN_ROOT}")"
  if [[ "${RUN_BASENAME}" =~ ([0-9]{8}_[0-9]{6})$ ]]; then
    TS="${BASH_REMATCH[1]}"
  fi
fi
LOG_PATH="${RUN_ROOT}/eval.log"
REGISTRY_CSV="${EXPERIMENT_REGISTRY_CSV:-${REPO_ROOT}/outputs/experiment_registry.csv}"
PLAN_CSV="${EXPERIMENT_PLAN_CSV:-${REPO_ROOT}/outputs/experiment_plan.csv}"
PLAN_KEY="${PLAN_KEY:-run_root:${RUN_ROOT}}"
SUMMARY_CSV="${RUN_ROOT}/eval/mot20_summary.csv"

SEQ_IDS=(${SEQ_IDS:-1 2 3 5})
RUN_BASE="${RUN_BASE:-0}"
RUN_HEURISTIC="${RUN_HEURISTIC:-0}"
RUN_LEARNED="${RUN_LEARNED:-1}"
EXP_FILE="${EXP_FILE:-./yolox/exps/example/mot/yolox_x_mix_mot20_ch.py}"
CKPT="${CKPT:-./pretrained/bytetrack_x_mot20.pth.tar}"
REID_CFG="${REID_CFG:-fast_reid/configs/MOT20/sbs_S50.yml}"
REID_WTS="${REID_WTS:-pretrained/mot20_sbs_S50.pth}"

CALIBRATOR_NPZ="${CALIBRATOR_NPZ:-}"
LAPLACE_DECAY_SCALES="${LAPLACE_DECAY_SCALES:-1 2 4}"
LAPLACE_MIN_HISTORY="${LAPLACE_MIN_HISTORY:-3}"
LAPLACE_PROTO_MODE="${LAPLACE_PROTO_MODE:-multi}"
LAPLACE_PRIMARY_ONLY="${LAPLACE_PRIMARY_ONLY:-1}"
LAPLACE_NO_DET_SCORE="${LAPLACE_NO_DET_SCORE:-0}"
LAPLACE_DISABLE_POLE_BANK="${LAPLACE_DISABLE_POLE_BANK:-0}"
CMC_METHOD="${CMC_METHOD:-file}"

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
    "laplace_no_det_score=${LAPLACE_NO_DET_SCORE}"
    "laplace_disable_pole_bank=${LAPLACE_DISABLE_POLE_BANK}"
    "laplace_decay_scales=${LAPLACE_DECAY_SCALES}"
    "laplace_min_history=${LAPLACE_MIN_HISTORY}"
    "laplace_proto_mode=${LAPLACE_PROTO_MODE}"
    "cmc_method=${CMC_METHOD}"
  )
  if [[ $# -gt 0 ]]; then
    extras+=("$@")
  fi
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/upsert_experiment_plan.py" \
    --csv "${PLAN_CSV}" \
    --key "${PLAN_KEY}" \
    --status "${status}" \
    --kind eval \
    --script "scripts/run_botsort_lpb_ltra_mot20_eval.sh" \
    --dataset MOT20 \
    --split val_half \
    --tracker-family BoT-SORT \
    --variant lpb_ltra_mot20_matrix \
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

if [[ "${RUN_LEARNED}" == "1" && -z "${CALIBRATOR_NPZ}" ]]; then
  echo "CALIBRATOR_NPZ is required" >&2
  exit 1
fi

run_track() {
  local experiment_name="$1"
  local analysis_dir="${2-}"
  if [[ "$#" -ge 2 ]]; then
    shift 2
  else
    shift "$#"
  fi
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
      --laplace-assoc
      --laplace-decay-scales ${LAPLACE_DECAY_SCALES}
      --laplace-min-history "${LAPLACE_MIN_HISTORY}"
      --laplace-proto-mode "${LAPLACE_PROTO_MODE}"
      --laplace-calibrator "${CALIBRATOR_NPZ}"
    )
    if [[ -n "${analysis_dir}" ]]; then
      cmd+=(--laplace-analysis-dir "${analysis_dir}")
    fi
    if [[ "${LAPLACE_PRIMARY_ONLY}" == "1" ]]; then
      cmd+=(--laplace-primary-only)
    fi
    if [[ "${LAPLACE_NO_DET_SCORE}" == "1" ]]; then
      cmd+=(--laplace-no-det-score)
    fi
    if [[ "${LAPLACE_DISABLE_POLE_BANK}" == "1" ]]; then
      cmd+=(--laplace-disable-pole-bank)
    fi
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

results_complete() {
  local results_dir="$1"
  local seq_id
  local seq_file
  [[ -d "${results_dir}" ]] || return 1
  for seq_id in "${SEQ_IDS[@]}"; do
    printf -v seq_file "MOT20-%02d.txt" "${seq_id}"
    [[ -s "${results_dir}/${seq_file}" ]] || return 1
  done
  return 0
}

eval_complete() {
  local tracker_name="$1"
  local work_dir="$2"
  [[ -s "${work_dir}/eval/${tracker_name}/pedestrian_summary.txt" ]]
}

RESULT_DIRS=()

if [[ "${RUN_BASE}" == "1" ]]; then
  BASE_EXP="mot20_base_${TS}"
  BASE_RESULTS="${BOT_ROOT}/YOLOX_outputs/${BASE_EXP}/track_results"
  echo "[step] base"
  if results_complete "${BASE_RESULTS}"; then
    echo "[reuse] base tracking results: ${BASE_RESULTS}"
  else
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
        --experiment-name "${BASE_EXP}"
      )
      echo "[track] ${cmd[*]}"
      "${cmd[@]}"
    )
  fi
  if eval_complete "${BASE_EXP}" "${RUN_ROOT}/eval/base"; then
    echo "[reuse] base TrackEval: ${RUN_ROOT}/eval/base/eval/${BASE_EXP}"
  else
    eval_results "${BASE_RESULTS}" "${BASE_EXP}" "${RUN_ROOT}/eval/base"
  fi
  RESULT_DIRS+=("${RUN_ROOT}/eval/base/eval/${BASE_EXP}")
fi

if [[ "${RUN_HEURISTIC}" == "1" ]]; then
  HEUR_EXP="mot20_lpb_heuristic_${TS}"
  HEUR_RESULTS="${BOT_ROOT}/YOLOX_outputs/${HEUR_EXP}/track_results"
  echo "[step] heuristic"
  if results_complete "${HEUR_RESULTS}"; then
    echo "[reuse] heuristic tracking results: ${HEUR_RESULTS}"
  else
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
        --experiment-name "${HEUR_EXP}"
        --laplace-analysis-dir "${RUN_ROOT}/pair_logs/heuristic"
        --laplace-assoc
        --laplace-decay-scales ${LAPLACE_DECAY_SCALES}
        --laplace-min-history "${LAPLACE_MIN_HISTORY}"
        --laplace-proto-mode "${LAPLACE_PROTO_MODE}"
      )
      if [[ "${LAPLACE_PRIMARY_ONLY}" == "1" ]]; then
        cmd+=(--laplace-primary-only)
      fi
      if [[ "${LAPLACE_NO_DET_SCORE}" == "1" ]]; then
        cmd+=(--laplace-no-det-score)
      fi
      echo "[track] ${cmd[*]}"
      "${cmd[@]}"
    )
  fi
  if eval_complete "${HEUR_EXP}" "${RUN_ROOT}/eval/heuristic"; then
    echo "[reuse] heuristic TrackEval: ${RUN_ROOT}/eval/heuristic/eval/${HEUR_EXP}"
  else
    eval_results "${HEUR_RESULTS}" "${HEUR_EXP}" "${RUN_ROOT}/eval/heuristic"
  fi
  RESULT_DIRS+=("${RUN_ROOT}/eval/heuristic/eval/${HEUR_EXP}")
fi

if [[ "${RUN_LEARNED}" == "1" ]]; then
  LEARNED_EXP="mot20_lpb_learned_${TS}"
  LEARNED_RESULTS="${BOT_ROOT}/YOLOX_outputs/${LEARNED_EXP}/track_results"
  echo "[step] learned"
  if results_complete "${LEARNED_RESULTS}"; then
    echo "[reuse] learned tracking results: ${LEARNED_RESULTS}"
  else
    run_track "${LEARNED_EXP}" "${RUN_ROOT}/pair_logs/learned"
  fi
  if eval_complete "${LEARNED_EXP}" "${RUN_ROOT}/eval/learned"; then
    echo "[reuse] learned TrackEval: ${RUN_ROOT}/eval/learned/eval/${LEARNED_EXP}"
  else
    eval_results "${LEARNED_RESULTS}" "${LEARNED_EXP}" "${RUN_ROOT}/eval/learned"
  fi
  RESULT_DIRS+=("${RUN_ROOT}/eval/learned/eval/${LEARNED_EXP}")
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
  --script "scripts/run_botsort_lpb_ltra_mot20_eval.sh" \
  --dataset MOT20 \
  --split val_half \
  --tracker-family BoT-SORT \
  --variant lpb_ltra_mot20_matrix \
  --run-root "${RUN_ROOT}" \
  --summary-csv "${SUMMARY_CSV}" \
  --calibrator-npz "${CALIBRATOR_NPZ}" \
  --log-path "${LOG_PATH}" \
  --extra \
    run_base="${RUN_BASE}" \
    run_heuristic="${RUN_HEURISTIC}" \
    run_learned="${RUN_LEARNED}" \
    laplace_primary_only="${LAPLACE_PRIMARY_ONLY}" \
    laplace_no_det_score="${LAPLACE_NO_DET_SCORE}" \
    laplace_disable_pole_bank="${LAPLACE_DISABLE_POLE_BANK}" \
    laplace_decay_scales="${LAPLACE_DECAY_SCALES}" \
    laplace_min_history="${LAPLACE_MIN_HISTORY}" \
    laplace_proto_mode="${LAPLACE_PROTO_MODE}" \
    cmc_method="${CMC_METHOD}"
update_plan_status completed

echo "[done] $(date '+%F %T %z')"
echo "[summary] ${SUMMARY_CSV}"
