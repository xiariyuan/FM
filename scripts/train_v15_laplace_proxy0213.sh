#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/gemini/code/FMtrack-main/FM-Track"
cd "${REPO_ROOT}"

export LD_LIBRARY_PATH="/root/miniconda3/lib:/root/miniconda3/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
DATA_ROOT="${DATA_ROOT:-/gemini/code/datasets}"
CFG="${CFG:-configs/experiments/bytetrack_fa_mot_mot17_v15_laplace_reid_da_val0213.yaml}"
EPOCHS="${EPOCHS:-6}"
BATCH_SIZE="${BATCH_SIZE:-1}"
ACCUMULATE_STEPS="${ACCUMULATE_STEPS:-6}"
REGISTRY_CSV="${REGISTRY_CSV:-outputs/experiment_registry.csv}"

TS="$(date +%Y%m%d_%H%M%S)"
EXP_NAME="${EXP_NAME:-bytetrack_fa_mot_mot17_v15_laplace_proxy0213_${TS}}"
OUT_DIR="${OUT_DIR:-outputs/${EXP_NAME}}"
SUMMARY_CSV="${OUT_DIR}/summary.csv"

mkdir -p "${OUT_DIR}"
echo "${OUT_DIR}" > outputs/latest_v15_laplace_proxy_dir.txt

record() {
  local status="$1"
  local notes="$2"
  local best_epoch_value="${3:-}"
  local best_hota_value="${4:-}"
  local best_checkpoint_value="${5:-}"
  local summary_target="$6"
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/append_experiment_record.py" \
    --csv "${summary_target}" \
    --kind train \
    --status "${status}" \
    --script "scripts/train_v15_laplace_proxy0213.sh" \
    --dataset MOT17 \
    --split train \
    --tracker-family ByteTrack \
    --variant v15_laplace_proxy0213 \
    --tag laplace_proxy \
    --run-root "${OUT_DIR}" \
    --checkpoint "" \
    --log-path "${OUT_DIR}/train.log" \
    --notes "${notes}" \
    --extra exp_name="${EXP_NAME}" config_path="${CFG}" out_dir="${OUT_DIR}" \
      best_epoch="${best_epoch_value}" best_hota="${best_hota_value}" \
      best_checkpoint="${best_checkpoint_value}" phase=train
}

record running "training started" "" "" "" "${SUMMARY_CSV}"
record running "training started" "" "" "" "${REGISTRY_CSV}"

echo "[train] cfg=${CFG}"
echo "[train] out_dir=${OUT_DIR}"
echo "[train] epochs=${EPOCHS} batch_size=${BATCH_SIZE} accumulate_steps=${ACCUMULATE_STEPS}"

if ! "${PYTHON_BIN}" -u train_bytetrack.py \
  --config-path "${CFG}" \
  --data-root "${DATA_ROOT}" \
  --outputs-dir "${OUT_DIR}" \
  --exp-name "${EXP_NAME}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --accumulate-steps "${ACCUMULATE_STEPS}" \
  2>&1 | tee "${OUT_DIR}/train.log"; then
  record failed "training failed" "" "" "" "${SUMMARY_CSV}"
  record failed "training failed" "" "" "" "${REGISTRY_CSV}"
  exit 1
fi

BEST_TXT="$("${PYTHON_BIN}" -u scripts/select_best_bytetrack_ckpt.py --exp-dir "${OUT_DIR}" --metric HOTA --dataset MOT17 --split train)"
echo "${BEST_TXT}" | tee "${OUT_DIR}/best_ckpt.txt"

BEST_EPOCH="$(printf '%s\n' "${BEST_TXT}" | sed -n 's/^best_epoch=//p')"
BEST_HOTA="$(printf '%s\n' "${BEST_TXT}" | sed -n 's/^best_value=//p')"
BEST_CKPT="$(printf '%s\n' "${BEST_TXT}" | sed -n 's/^checkpoint=//p')"

record success "training complete" "${BEST_EPOCH}" "${BEST_HOTA}" "${BEST_CKPT}" "${SUMMARY_CSV}"
record success "training complete" "${BEST_EPOCH}" "${BEST_HOTA}" "${BEST_CKPT}" "${REGISTRY_CSV}"

echo "[done] ${OUT_DIR}"
