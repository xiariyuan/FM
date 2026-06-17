#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/gemini/code/FMtrack-main/FM-Track"
cd "${REPO_ROOT}"

export LD_LIBRARY_PATH="/root/miniconda3/lib:/root/miniconda3/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
DATA_ROOT="${DATA_ROOT:-/gemini/code/datasets}"
CFG="${CFG:-configs/experiments/bytetrack_fa_mot_mot17_v16_laplace_trainable_val0213.yaml}"
EPOCHS="${EPOCHS:-4}"
BATCH_SIZE="${BATCH_SIZE:-1}"
ACCUMULATE_STEPS="${ACCUMULATE_STEPS:-6}"
REGISTRY_CSV="${REGISTRY_CSV:-outputs/experiment_registry.csv}"

TS="$(date +%Y%m%d_%H%M%S)"
EXP_NAME="${EXP_NAME:-bytetrack_fa_mot_mot17_v16_laplace_gate_proxy0213_${TS}}"
OUT_DIR="${OUT_DIR:-outputs/${EXP_NAME}}"
SUMMARY_CSV="${SUMMARY_CSV:-${OUT_DIR}/summary.csv}"
LOG_PATH="${OUT_DIR}/train.log"
SCRIPT_NAME="scripts/train_v16_laplace_gate_proxy0213.sh"
DATASET="MOT17"
SPLIT="train"
TRACKER_FAMILY="ByteTrack"
VARIANT="v16_laplace_gate_proxy0213"
TAG="laplace_trainable"

mkdir -p "${OUT_DIR}"
echo "${OUT_DIR}" > outputs/latest_v16_laplace_gate_dir.txt

append_summary_record() {
  local csv_path="$1"
  local status="$2"
  local notes="$3"
  shift 3
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/append_experiment_record.py" \
    --csv "${csv_path}" \
    --kind train \
    --status "${status}" \
    --script "${SCRIPT_NAME}" \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --tracker-family "${TRACKER_FAMILY}" \
    --variant "${VARIANT}" \
    --tag "${TAG}" \
    --run-root "${OUT_DIR}" \
    --log-path "${LOG_PATH}" \
    --notes "${notes}" \
    --extra "$@"
}

record_summary() {
  local status="$1"
  local notes="$2"
  shift 2
  append_summary_record "${SUMMARY_CSV}" "${status}" "${notes}" "$@"
}

record_registry() {
  local status="$1"
  local notes="$2"
  shift 2
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/append_experiment_record.py" \
    --csv "${REGISTRY_CSV}" \
    --kind train \
    --status "${status}" \
    --script "${SCRIPT_NAME}" \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --tracker-family "${TRACKER_FAMILY}" \
    --variant "${VARIANT}" \
    --tag "${TAG}" \
    --run-root "${OUT_DIR}" \
    --summary-csv "${SUMMARY_CSV}" \
    --log-path "${LOG_PATH}" \
    --notes "${notes}" \
    --extra "$@"
}

record_failure() {
  local rc="$1"
  record_summary "failed" "training failed" "exit_code=${rc}"
  record_registry "failed" "training failed" "exit_code=${rc}"
}

trap 'rc=$?; if [[ ${rc} -ne 0 ]]; then record_failure "${rc}"; fi' EXIT

record_summary "running" "training started" "phase=train"
record_registry "running" "training started" "phase=train"

echo "[train] cfg=${CFG}"
echo "[train] out_dir=${OUT_DIR}"
echo "[train] epochs=${EPOCHS} batch_size=${BATCH_SIZE} accumulate_steps=${ACCUMULATE_STEPS}"

"${PYTHON_BIN}" -u train_bytetrack.py \
  --config-path "${CFG}" \
  --data-root "${DATA_ROOT}" \
  --outputs-dir "${OUT_DIR}" \
  --exp-name "${EXP_NAME}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --accumulate-steps "${ACCUMULATE_STEPS}" \
  2>&1 | tee "${OUT_DIR}/train.log"

BEST_TXT=""
BEST_EPOCH=""
BEST_HOTA=""
BEST_CKPT="${OUT_DIR}/checkpoint_final.pth"
if "${PYTHON_BIN}" -u scripts/select_best_bytetrack_ckpt.py --exp-dir "${OUT_DIR}" --metric HOTA --dataset MOT17 --split train > "${OUT_DIR}/best_ckpt.txt" 2>/dev/null; then
  BEST_TXT="$(cat "${OUT_DIR}/best_ckpt.txt")"
  echo "${BEST_TXT}"
  BEST_EPOCH="$(sed -n 's/^best_epoch=//p' "${OUT_DIR}/best_ckpt.txt" | tail -n 1)"
  BEST_HOTA="$(sed -n 's/^best_value=//p' "${OUT_DIR}/best_ckpt.txt" | tail -n 1)"
  BEST_CKPT="$(sed -n 's/^checkpoint=//p' "${OUT_DIR}/best_ckpt.txt" | tail -n 1)"
else
  echo "[train] No epoch validation summaries found; using checkpoint_final.pth" | tee "${OUT_DIR}/best_ckpt.txt"
fi

record_summary "success" "training completed" "phase=train" "best_epoch=${BEST_EPOCH}" "best_hota=${BEST_HOTA}" "best_checkpoint=${BEST_CKPT}" "checkpoint=${BEST_CKPT}"
record_registry "success" "training completed" "phase=train" "best_epoch=${BEST_EPOCH}" "best_hota=${BEST_HOTA}" "best_checkpoint=${BEST_CKPT}" "checkpoint=${BEST_CKPT}"
trap - EXIT

echo "[done] ${OUT_DIR}"
