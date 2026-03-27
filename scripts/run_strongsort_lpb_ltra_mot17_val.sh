#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"
SS_ROOT="${SS_ROOT:-${REPO_ROOT}/external/StrongSORT-master}"
DATA_ROOT="${DATA_ROOT:-/gemini/code/datasets}"
DET_ROOT="${DET_ROOT:-${SS_ROOT}/MOT17_val_YOLOX+BoT}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs/strongsort_lpb_ltra/MOT17_val_${TS}}"
OUT_ROOT="$(realpath -m "${OUT_ROOT}")"
CALIBRATOR_NPZ="${CALIBRATOR_NPZ:-}"
SEQ_OVERRIDE="${SEQ_OVERRIDE:-}"
REGISTRY_CSV="${EXPERIMENT_REGISTRY_CSV:-${REPO_ROOT}/outputs/experiment_registry.csv}"
PLAN_CSV="${EXPERIMENT_PLAN_CSV:-${REPO_ROOT}/outputs/experiment_plan.csv}"
LAPLACE_DISABLE_POLE_BANK="${LAPLACE_DISABLE_POLE_BANK:-0}"

if [[ -z "${BASE_REF+x}" ]]; then
  BASE_REF="${REPO_ROOT}/outputs/strongsort_ltra/MOT17_val/base_eval/eval/strongsort_mot17_val_base"
fi
if [[ -z "${HEUR_REF+x}" ]]; then
  HEUR_REF="${REPO_ROOT}/outputs/strongsort_ltra/MOT17_val/laplace_eval/eval/strongsort_mot17_val_laplace"
fi

if [[ -z "${CALIBRATOR_NPZ}" ]]; then
  echo "CALIBRATOR_NPZ is required" >&2
  exit 2
fi
if [[ ! -d "${DET_ROOT}" ]]; then
  echo "Missing DET_ROOT: ${DET_ROOT}" >&2
  exit 2
fi

mkdir -p "${OUT_ROOT}"
LOG_PATH="${OUT_ROOT}/run.log"
exec > >(tee -a "${LOG_PATH}") 2>&1

RESULT_ROOT="${OUT_ROOT}/results"
LEARNED_DIR="${RESULT_ROOT}/learned"
LEARNED_EVAL="${OUT_ROOT}/learned_eval"
ANALYSIS_DIR="${OUT_ROOT}/pair_logs"
ANALYSIS_COMBINED="${ANALYSIS_DIR}/_combined/all_pairs.csv"
ANALYSIS_SUMMARY_DIR="${ANALYSIS_DIR}/_combined"
SUMMARY_CSV="${OUT_ROOT}/summary.csv"
PLAN_KEY="${PLAN_KEY:-run_root:${OUT_ROOT}}"

mkdir -p "${LEARNED_DIR}"

combine_pair_logs() {
  local src_dir="$1"
  local dst_csv="$2"
  shopt -s nullglob
  local files=("${src_dir}"/*_pairs.csv)
  shopt -u nullglob
  if [[ "${#files[@]}" -eq 0 ]]; then
    echo "No pair logs found in ${src_dir}" >&2
    return 1
  fi
  mkdir -p "$(dirname "${dst_csv}")"
  head -n 1 "${files[0]}" > "${dst_csv}"
  for f in "${files[@]}"; do
    tail -n +2 "${f}" >> "${dst_csv}"
  done
  echo "[combined] ${dst_csv}"
}

update_plan_status() {
  local status="$1"
  shift || true
  local extras=(
    "seq_override=${SEQ_OVERRIDE}"
    "det_root=${DET_ROOT}"
    "analysis_dir=${ANALYSIS_DIR}"
    "laplace_disable_pole_bank=${LAPLACE_DISABLE_POLE_BANK}"
  )
  if [[ $# -gt 0 ]]; then
    extras+=("$@")
  fi
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/upsert_experiment_plan.py" \
    --csv "${PLAN_CSV}" \
    --key "${PLAN_KEY}" \
    --status "${status}" \
    --kind eval \
    --script "scripts/run_strongsort_lpb_ltra_mot17_val.sh" \
    --dataset MOT17 \
    --split val_half \
    --tracker-family StrongSORT \
    --variant lpb_ltra_learned \
    --run-root "${OUT_ROOT}" \
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

COMMON_ARGS=(
  MOT17
  val
  --BoT
  --NSA
  --EMA
  --MC
  --woC
  --root_dataset "${DATA_ROOT}"
  --dir_dets "${DET_ROOT}"
  --dir_save "${LEARNED_DIR}"
  --LAPLACE
  --laplace-calibrator "${CALIBRATOR_NPZ}"
  --laplace-analysis-dir "${ANALYSIS_DIR}"
)
if [[ "${LAPLACE_DISABLE_POLE_BANK}" == "1" ]]; then
  COMMON_ARGS+=(--laplace-disable-pole-bank)
fi

ECC_PATH="${SS_ROOT}/MOT17_ECC_val.json"
if [[ -f "${ECC_PATH}" ]]; then
  COMMON_ARGS+=(--ECC --path_ECC "${ECC_PATH}")
fi
if [[ -n "${SEQ_OVERRIDE}" ]]; then
  COMMON_ARGS+=(--sequences ${SEQ_OVERRIDE})
fi

echo "[start] $(date '+%F %T %z')"
echo "[out_root] ${OUT_ROOT}"
echo "[calibrator] ${CALIBRATOR_NPZ}"
echo "[seq_override] ${SEQ_OVERRIDE:-<full-val>}"

(
  cd "${SS_ROOT}"
  cmd=("${PYTHON_BIN}" -u strong_sort.py "${COMMON_ARGS[@]}")
  echo "[track] ${cmd[*]}"
  "${cmd[@]}"
)

if combine_pair_logs "${ANALYSIS_DIR}" "${ANALYSIS_COMBINED}"; then
  echo "[check] pair-log header"
  head -n 1 "${ANALYSIS_COMBINED}"
  for required_col in assoc_stage history_len amb_spa amb_lap amb_mot learned_r; do
    if ! head -n 1 "${ANALYSIS_COMBINED}" | grep -q "${required_col}"; then
      echo "Missing required column in StrongSORT pair logs: ${required_col}" >&2
      exit 1
    fi
  done
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/summarize_laplace_pair_logs.py" \
    "${ANALYSIS_COMBINED}" \
    --out-dir "${ANALYSIS_SUMMARY_DIR}"
fi

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/eval_botsort_halfval_trackeval.py" \
  --dataset MOT17 \
  --data-root "${DATA_ROOT}" \
  --results-dir "${LEARNED_DIR}" \
  --tracker-name strongsort_mot17_val_lpb_learned \
  --work-dir "${LEARNED_EVAL}" \
  --remap-results-from-fullval

COLLECT_ARGS=()
if [[ -d "${BASE_REF}" ]]; then
  COLLECT_ARGS+=("${BASE_REF}")
fi
if [[ -d "${HEUR_REF}" ]]; then
  COLLECT_ARGS+=("${HEUR_REF}")
fi
COLLECT_ARGS+=("${LEARNED_EVAL}/eval/strongsort_mot17_val_lpb_learned")

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/collect_trackeval_metrics.py" \
  "${COLLECT_ARGS[@]}" \
  --csv "${SUMMARY_CSV}" | tee "${OUT_ROOT}/summary.txt"
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/append_experiment_record.py" \
  --csv "${REGISTRY_CSV}" \
  --kind eval \
  --script "scripts/run_strongsort_lpb_ltra_mot17_val.sh" \
  --dataset MOT17 \
  --split val_half \
  --tracker-family StrongSORT \
  --variant lpb_ltra_learned \
  --run-root "${OUT_ROOT}" \
  --summary-csv "${SUMMARY_CSV}" \
  --calibrator-npz "${CALIBRATOR_NPZ}" \
  --log-path "${LOG_PATH}" \
  --extra \
    seq_override="${SEQ_OVERRIDE}" \
    det_root="${DET_ROOT}" \
    analysis_dir="${ANALYSIS_DIR}" \
    laplace_disable_pole_bank="${LAPLACE_DISABLE_POLE_BANK}"
update_plan_status completed

echo "[done] $(date '+%F %T %z')"
echo "[summary] ${SUMMARY_CSV}"
