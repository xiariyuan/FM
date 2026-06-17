#!/usr/bin/env bash
set -euo pipefail

# TOS-Track MOT20 smoke test (MOT20-05 only, ~20min with HACA no-bg).
# Phase 1A: verify TOS analysis-only and freeze-on-occlusion don't crash.

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"
BOT_ROOT="${BOT_ROOT:-${REPO_ROOT}/external/BoT-SORT-main}"
DATA_ROOT="${DATA_ROOT:-/gemini/code/datasets}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
REGISTRY_CSV="${REGISTRY_CSV:-${REPO_ROOT}/outputs/experiment_registry.csv}"
PLAN_CSV="${PLAN_CSV:-${REPO_ROOT}/outputs/experiment_plan.csv}"

EXP_FILE="${EXP_FILE:-./yolox/exps/example/mot/yolox_x_mix_mot20_ch.py}"
CKPT="${CKPT:-./pretrained/bytetrack_x_mot20.pth.tar}"
REID_CFG="${REID_CFG:-fast_reid/configs/MOT20/sbs_S50.yml}"
REID_WTS="${REID_WTS:-pretrained/mot20_sbs_S50.pth}"

# HACA no-bg: v1 for smoke (v3 available after fullval completes)
HACA_NPZ="${HACA_NPZ:-${REPO_ROOT}/outputs/haca_mot20_train_20260613_082005/haca_v1/mot20_haca_v1.npz}"
LAPLACE_DECAY_SCALES="${LAPLACE_DECAY_SCALES:-1 2 4}"
LAPLACE_MIN_HISTORY="${LAPLACE_MIN_HISTORY:-3}"
LAPLACE_PROTO_MODE="${LAPLACE_PROTO_MODE:-multi}"

RUN_ROOT="${REPO_ROOT}/outputs/tos_mot20_smoke_${TS}"
RUN_ROOT="$(realpath -m "${RUN_ROOT}")"
LOG_PATH="${RUN_ROOT}/smoke.log"
mkdir -p "${RUN_ROOT}"
exec > >(tee -a "${LOG_PATH}") 2>&1

log() { echo "[$(date '+%F %T')] $*"; }

# TOS defaults
TOS_HOLD_BUFFER="${TOS_HOLD_BUFFER:-30}"
TOS_NEWBORN_DELAY="${TOS_NEWBORN_DELAY:-5}"
TOS_MEMORY_FRAMES="${TOS_MEMORY_FRAMES:-150}"
TOS_RECONNECT_GAP_MAX="${TOS_RECONNECT_GAP_MAX:-60}"
TOS_RECONNECT_MIN_SIM="${TOS_RECONNECT_MIN_SIM:-0.70}"
TOS_OCCLUSION_THRESH="${TOS_OCCLUSION_THRESH:-0.5}"

run_variant() {
  local label="$1"
  local exp_name="$2"
  shift 2
  local extra_args=("$@")

  log "[variant] ${label}: ${exp_name}"
  (
    cd "${BOT_ROOT}"
    "${PYTHON_BIN}" -u tools/track.py "${DATA_ROOT}/MOT20" \
      --benchmark MOT20 \
      --eval train \
      --seq-ids 5 \
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
      --laplace-haca-checkpoint "${HACA_NPZ}" \
      --laplace-haca-no-background \
      "${extra_args[@]}" \
    2>&1 | tee -a "${RUN_ROOT}/${label}.log"
  )
  local rc=$?
  log "[variant] ${label} exit: ${rc}"
  echo "${BOT_ROOT}/YOLOX_outputs/${exp_name}/track_results" > "${RUN_ROOT}/${label}.results_dir"
  return ${rc}
}

eval_variant() {
  local label="$1"
  local results_dir="$2"
  local eval_dir="${RUN_ROOT}/eval/${label}"
  mkdir -p "${eval_dir}"

  if [[ ! -d "${results_dir}" ]]; then
    log "[eval:skip] ${label}: no results"
    return 1
  fi

  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/eval_botsort_halfval_trackeval.py" \
    --dataset MOT20 \
    --data-root "${DATA_ROOT}" \
    --results-dir "${results_dir}" \
    --tracker-name "${label}" \
    --work-dir "${eval_dir}" \
    --remap-results-from-fullval \
    2>&1 | tee -a "${RUN_ROOT}/${label}.eval.log"
}

write_variant_result() {
  local label="$1"
  local eval_dir="${RUN_ROOT}/eval/${label}"

  if [[ -f "${eval_dir}/pedestrian_summary.txt" ]]; then
    local hota deta assa mota idf1 idsw
    hota=$(grep -oP "^\K[0-9.]+" "${eval_dir}/pedestrian_summary.txt" | head -1 || echo "failed")
    echo "${label},success,${hota}" >> "${RUN_ROOT}/summary.csv"
  else
    echo "${label},failed,NA" >> "${RUN_ROOT}/summary.csv"
  fi
}

log "[start] TOS MOT20 smoke (MOT20-05 only)"
log "[HACA] ${HACA_NPZ}"

# Header for summary
echo "variant,status,HOTA" > "${RUN_ROOT}/summary.csv"

# Variant 1: baseline
run_variant "baseline" "tos_mot20_baseline_${TS}" && {
  eval_variant "baseline" "$(cat "${RUN_ROOT}/baseline.results_dir")" || true
  write_variant_result "baseline"
} || { echo "baseline,crashed,NA" >> "${RUN_ROOT}/summary.csv"; }

# Variant 2: TOS analysis-only
run_variant "tos_analysis" "tos_mot20_analysis_${TS}" \
  --tos-enable --tos-analysis-only \
  --tos-analysis-dir "${RUN_ROOT}/tos_analysis" && {
  eval_variant "tos_analysis" "$(cat "${RUN_ROOT}/tos_analysis.results_dir")" || true
  write_variant_result "tos_analysis"
} || { echo "tos_analysis,crashed,NA" >> "${RUN_ROOT}/summary.csv"; }

# Variant 3: TOS freeze-on-occlusion
run_variant "tos_freeze" "tos_mot20_freeze_${TS}" \
  --tos-enable --tos-freeze-on-occlusion \
  --tos-occlusion-thresh "${TOS_OCCLUSION_THRESH}" \
  --tos-hold-buffer "${TOS_HOLD_BUFFER}" \
  --tos-disable-reentry && {
  eval_variant "tos_freeze" "$(cat "${RUN_ROOT}/tos_freeze.results_dir")" || true
  write_variant_result "tos_freeze"
} || { echo "tos_freeze,crashed,NA" >> "${RUN_ROOT}/summary.csv"; }

log "[done] smoke run complete"
log "[summary] ${RUN_ROOT}/summary.csv"
echo "=== Summary ==="
cat "${RUN_ROOT}/summary.csv"
