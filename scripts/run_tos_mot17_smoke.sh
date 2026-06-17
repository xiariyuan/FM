#!/usr/bin/env bash
set -euo pipefail

# TOS-Track MOT17 smoke test (single sequence, fast).
# Runs baseline, TOS-analysis-only, and TOS-analysis+freeze variants.

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"
BOT_ROOT="${BOT_ROOT:-${REPO_ROOT}/external/BoT-SORT-main}"
DATA_ROOT="${DATA_ROOT:-/gemini/code/datasets}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
REGISTRY_CSV="${REGISTRY_CSV:-${REPO_ROOT}/outputs/experiment_registry.csv}"
PLAN_CSV="${PLAN_CSV:-${REPO_ROOT}/outputs/experiment_plan.csv}"

# MOT17 smoke: single sequence MOT17-02 (FRCNN)
SMOKE_SEQ="MOT17-02-FRCNN"
EXP_FILE="${EXP_FILE:-./yolox/exps/example/mot/yolox_x_ablation.py}"
CKPT="${CKPT:-./pretrained/bytetrack_x_mot20.pth.tar}"
REID_CFG="${REID_CFG:-fast_reid/configs/MOT20/sbs_S50.yml}"
REID_WTS="${REID_WTS:-pretrained/mot20_sbs_S50.pth}"

# HACA no-bg: use v1 checkpoint for smoke (faster than v3)
HACA_V1_CKP="${HACA_V1_CKP:-${REPO_ROOT}/outputs/haca_mot20_train_20260613_082005/haca_v1/mot20_haca_v1.npz}"
LAPLACE_DECAY_SCALES="${LAPLACE_DECAY_SCALES:-1 2 4}"
LAPLACE_MIN_HISTORY="${LAPLACE_MIN_HISTORY:-3}"
LAPLACE_PROTO_MODE="${LAPLACE_PROTO_MODE:-multi}"

RUN_ROOT="${REPO_ROOT}/outputs/tos_mot17_smoke_${TS}"
RUN_ROOT="$(realpath -m "${RUN_ROOT}")"
LOG_PATH="${RUN_ROOT}/smoke.log"
mkdir -p "${RUN_ROOT}"
exec > >(tee -a "${LOG_PATH}") 2>&1

# TOS config
TOS_HOLD_BUFFER="${TOS_HOLD_BUFFER:-30}"
TOS_NEWBORN_DELAY="${TOS_NEWBORN_DELAY:-5}"
TOS_MEMORY_FRAMES="${TOS_MEMORY_FRAMES:-150}"
TOS_RECONNECT_GAP_MAX="${TOS_RECONNECT_GAP_MAX:-60}"
TOS_RECONNECT_MIN_SIM="${TOS_RECONNECT_MIN_SIM:-0.70}"
TOS_OCCLUSION_THRESH="${TOS_OCCLUSION_THRESH:-0.5}"

log() { echo "[$(date '+%F %T')] $*"; }

run_variant() {
  local label="$1"
  local exp_name="$2"
  shift 2
  local extra_args=("$@")

  local results_dir="${BOT_ROOT}/YOLOX_outputs/${exp_name}/track_results"
  log "[variant] ${label}: ${exp_name}"

  (
    cd "${BOT_ROOT}"
    "${PYTHON_BIN}" -u tools/track.py "${DATA_ROOT}/MOT17" \
      --benchmark MOT17 \
      --eval val \
      --seq-ids 2 \
      -f "${EXP_FILE}" \
      -c "${CKPT}" \
      --with-reid \
      --fast-reid-config "${REID_CFG}" \
      --fast-reid-weights "${REID_WTS}" \
      --experiment-name "${exp_name}" \
      --laplace-assoc \
      --laplace-assoc-mode haca_v1 \
      --laplace-decay-scales ${LAPLACE_DECAY_SCALES} \
      --laplace-min-history "${LAPLACE_MIN_HISTORY}" \
      --laplace-proto-mode "${LAPLACE_PROTO_MODE}" \
      --laplace-primary-only \
      --laplace-haca-checkpoint "${HACA_V1_CKP}" \
      --laplace-haca-no-background \
      "${extra_args[@]}" \
    2>&1 | tee -a "${RUN_ROOT}/${label}.log"
  )

  local rc=$?
  log "[variant] ${label} exit: ${rc}"
  echo "${results_dir}" > "${RUN_ROOT}/${label}.results_dir"
  return ${rc}
}

eval_variant() {
  local label="$1"
  local results_dir="$2"
  local eval_dir="${RUN_ROOT}/eval/${label}"
  mkdir -p "${eval_dir}"

  if [[ ! -d "${results_dir}" ]]; then
    log "[eval:skip] ${label}: no results_dir ${results_dir}"
    return 1
  fi

  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/eval_botsort_halfval_trackeval.py" \
    --dataset MOT17 \
    --data-root "${DATA_ROOT}" \
    --results-dir "${results_dir}" \
    --tracker-name "${label}" \
    --work-dir "${eval_dir}" \
    2>&1 | tee -a "${RUN_ROOT}/${label}.eval.log"
}

write_summary() {
  local label="$1"
  local eval_dir="${RUN_ROOT}/eval/${label}"
  local metrics=""

  if [[ -f "${eval_dir}/pedestrian_summary.txt" ]]; then
    metrics=$(grep -E "^[0-9]+\.[0-9]+" "${eval_dir}/pedestrian_summary.txt" 2>/dev/null | head -1 || echo "")
  fi
  if [[ -z "${metrics}" ]]; then
    metrics=$(grep -E "^HOTA|^MOTA|^IDF1" "${eval_dir}/summary.txt" 2>/dev/null | head -1 || echo "")
  fi

  echo "${label},$([[ -n "${metrics}" ]] && echo "${metrics}" || echo 'failed')" >> "${RUN_ROOT}/summary_raw.txt"
}

# --- Variants ---
declare -A LABELS
declare -A EXP_NAMES
declare -A EXTRA_ARGS

# Variant 1: baseline (no TOS)
LABELS[0]="baseline"
EXP_NAMES[0]="tos_mot17_baseline_${TS}"
EXTRA_ARGS[0]=""

# Variant 2: TOS analysis-only
LABELS[1]="tos_analysis"
EXP_NAMES[1]="tos_mot17_analysis_${TS}"
EXTRA_ARGS[1]="--tos-enable --tos-analysis-only --tos-analysis-dir ${RUN_ROOT}/tos_analysis"

# Variant 3: TOS freeze-on-occlusion
LABELS[2]="tos_freeze"
EXP_NAMES[2]="tos_mot17_freeze_${TS}"
EXTRA_ARGS[2]="--tos-enable --tos-freeze-on-occlusion --tos-occlusion-thresh ${TOS_OCCLUSION_THRESH} --tos-hold-buffer ${TOS_HOLD_BUFFER} --tos-disable-reentry"

# --- Run all variants ---
for i in 0 1 2; do
  label="${LABELS[$i]}"
  exp="${EXP_NAMES[$i]}"
  args="${EXTRA_ARGS[$i]}"
  results_dir=""

  if run_variant "${label}" "${exp}" ${args}; then
    results_dir=$(cat "${RUN_ROOT}/${label}.results_dir" 2>/dev/null || echo "")
    if [[ -n "${results_dir}" ]]; then
      eval_variant "${label}" "${results_dir}" || true
    fi
  else
    log "[error] variant ${label} failed, skipping eval"
  fi

  write_summary "${label}" || true
done

# --- Summary ---
echo "variant,metrics" > "${RUN_ROOT}/summary.csv"
if [[ -f "${RUN_ROOT}/summary_raw.txt" ]]; then
  cat "${RUN_ROOT}/summary_raw.txt" >> "${RUN_ROOT}/summary.csv"
fi

log "[done] smoke run complete"
log "[summary] ${RUN_ROOT}/summary.csv"
echo "=== Summary ==="
cat "${RUN_ROOT}/summary.csv"
