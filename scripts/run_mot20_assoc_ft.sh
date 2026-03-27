#!/usr/bin/env bash
set -euo pipefail

# MOT20 association adaptation pipeline
#
# Modes:
#   stage1 [DATA_ROOT]
#     - train with held-out val (MOT20-05), save per-epoch ckpt, auto-select best by HOTA.
#
#   stage2 [DATA_ROOT] [STAGE1_EXP_DIR] [BEST_CKPT]
#     - full-train consolidation on all MOT20 train sequences.
#     - if STAGE1_EXP_DIR/BEST_CKPT not provided, tries outputs/latest_mot20_assoc_stage1_dir.txt.
#
#   all [DATA_ROOT]
#     - run stage1 then stage2 automatically.
#
# Examples:
#   bash scripts/run_mot20_assoc_ft.sh stage1
#   bash scripts/run_mot20_assoc_ft.sh stage2 /gemini/code/datasets outputs/xxx outputs/xxx/checkpoint_epoch_3.pth
#   bash scripts/run_mot20_assoc_ft.sh all

MODE="${1:-stage1}"
DATA_ROOT="${2:-/gemini/code/datasets}"
TS="$(date +%Y%m%d_%H%M%S)"

STAGE1_CFG="configs/experiments/bytetrack_fa_mot_mot20_v13_assoc_ft_val05.yaml"
STAGE2_CFG="configs/experiments/bytetrack_fa_mot_mot20_v13_assoc_ft_fulltrain.yaml"
LATEST_STAGE1_FILE="outputs/latest_mot20_assoc_stage1_dir.txt"

run_stage1() {
  local out_dir="outputs/bytetrack_fa_mot_mot20_v13_assoc_ft_val05_${TS}"
  echo "[stage1] out_dir=${out_dir}"
  python -u train_bytetrack.py \
    --config-path "${STAGE1_CFG}" \
    --data-root "${DATA_ROOT}" \
    --outputs-dir "${out_dir}"

  echo "${out_dir}" > "${LATEST_STAGE1_FILE}"
  echo "[stage1] selecting best checkpoint by HOTA..."
  python scripts/select_best_bytetrack_ckpt.py \
    --exp-dir "${out_dir}" \
    --dataset MOT20 \
    --split train \
    --metric HOTA | tee "${out_dir}/best_ckpt.txt"
}

resolve_stage1_dir() {
  local stage1_dir="${1:-}"
  if [[ -n "${stage1_dir}" ]]; then
    echo "${stage1_dir}"
    return 0
  fi
  if [[ -f "${LATEST_STAGE1_FILE}" ]]; then
    cat "${LATEST_STAGE1_FILE}"
    return 0
  fi
  echo ""
}

resolve_best_ckpt() {
  local stage1_dir="${1}"
  local ckpt_arg="${2:-}"
  if [[ -n "${ckpt_arg}" ]]; then
    echo "${ckpt_arg}"
    return 0
  fi
  if [[ -f "${stage1_dir}/best_ckpt.txt" ]]; then
    local ckpt
    ckpt="$(grep -E '^checkpoint=' "${stage1_dir}/best_ckpt.txt" | tail -n 1 | cut -d'=' -f2- || true)"
    if [[ -n "${ckpt}" ]]; then
      echo "${ckpt}"
      return 0
    fi
  fi
  # Fallback: recompute
  local tmp_out
  tmp_out="$(python scripts/select_best_bytetrack_ckpt.py --exp-dir "${stage1_dir}" --dataset MOT20 --split train --metric HOTA)"
  echo "${tmp_out}" | tee "${stage1_dir}/best_ckpt.txt" >/dev/null
  echo "${tmp_out}" | grep -E '^checkpoint=' | tail -n 1 | cut -d'=' -f2-
}

run_stage2() {
  local stage1_dir="${1:-}"
  local ckpt_arg="${2:-}"
  local resolved_stage1
  resolved_stage1="$(resolve_stage1_dir "${stage1_dir}")"
  if [[ -z "${resolved_stage1}" ]]; then
    echo "[stage2] missing stage1 dir. Pass it explicitly or run stage1 first."
    exit 1
  fi
  if [[ ! -d "${resolved_stage1}" ]]; then
    echo "[stage2] stage1 dir not found: ${resolved_stage1}"
    exit 1
  fi

  local best_ckpt
  best_ckpt="$(resolve_best_ckpt "${resolved_stage1}" "${ckpt_arg}")"
  if [[ -z "${best_ckpt}" || ! -f "${best_ckpt}" ]]; then
    echo "[stage2] best checkpoint not found: ${best_ckpt}"
    exit 1
  fi

  local out_dir="outputs/bytetrack_fa_mot_mot20_v13_assoc_ft_fulltrain_${TS}"
  echo "[stage2] stage1_dir=${resolved_stage1}"
  echo "[stage2] resume_ckpt=${best_ckpt}"
  echo "[stage2] out_dir=${out_dir}"

  python -u train_bytetrack.py \
    --config-path "${STAGE2_CFG}" \
    --data-root "${DATA_ROOT}" \
    --resume-model "${best_ckpt}" \
    --outputs-dir "${out_dir}"
}

case "${MODE}" in
  stage1)
    run_stage1
    ;;
  stage2)
    # Optional args:
    #   $3 -> stage1_exp_dir
    #   $4 -> best_ckpt
    run_stage2 "${3:-}" "${4:-}"
    ;;
  all)
    run_stage1
    run_stage2 "" ""
    ;;
  *)
    echo "Unknown mode: ${MODE}"
    echo "Usage:"
    echo "  bash scripts/run_mot20_assoc_ft.sh stage1 [DATA_ROOT]"
    echo "  bash scripts/run_mot20_assoc_ft.sh stage2 [DATA_ROOT] [STAGE1_EXP_DIR] [BEST_CKPT]"
    echo "  bash scripts/run_mot20_assoc_ft.sh all [DATA_ROOT]"
    exit 1
    ;;
esac

