#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
SHARD_ROOT="${SHARD_ROOT:-${REPO_ROOT}/outputs/lpb_ltra_formal_mot17_shrink_20260311_191211/gt_builder}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${1:-${REPO_ROOT}/outputs/mtcr_train_${STAMP}}"

TRAIN_DIR="${TRAIN_DIR:-${SHARD_ROOT}/train_shards}"
VAL_DIR="${VAL_DIR:-${SHARD_ROOT}/val_shards}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-123}"
EPOCHS="${EPOCHS:-12}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
GRAD_CLIP="${GRAD_CLIP:-5.0}"
HIST_HIDDEN="${HIST_HIDDEN:-16}"
COMP_HIDDEN="${COMP_HIDDEN:-64}"
TOPK="${TOPK:-3}"
MARGIN_THRESHOLD="${MARGIN_THRESHOLD:--1}"
MARGIN_QUANTILE="${MARGIN_QUANTILE:-0.35}"
MARGIN_TEMPERATURE="${MARGIN_TEMPERATURE:-0.03}"
DELTA_SCALE="${DELTA_SCALE:-1.0}"
MIN_HISTORY="${MIN_HISTORY:-3}"
TEMPERATURE="${TEMPERATURE:-1.0}"
DUEL_MARGIN="${DUEL_MARGIN:-0.20}"
LOSS_DUEL_WEIGHT="${LOSS_DUEL_WEIGHT:-0.50}"
LOSS_SAFE_WEIGHT="${LOSS_SAFE_WEIGHT:-0.20}"
LOSS_BG_WEIGHT="${LOSS_BG_WEIGHT:-0.75}"
LOSS_GATE_WEIGHT="${LOSS_GATE_WEIGHT:-0.10}"
AMBIGUOUS_WEIGHT="${AMBIGUOUS_WEIGHT:-2.0}"
EASY_WEIGHT="${EASY_WEIGHT:-0.5}"
BACKGROUND_WEIGHT="${BACKGROUND_WEIGHT:-0.75}"
GATE_CAP="${GATE_CAP:-0.20}"
BG_SCALE="${BG_SCALE:-0.75}"
SELECT_AMB_WEIGHT="${SELECT_AMB_WEIGHT:-1.5}"
SELECT_BG_WEIGHT="${SELECT_BG_WEIGHT:-0.10}"
SELECT_EASY_WEIGHT="${SELECT_EASY_WEIGHT:-0.05}"
PATIENCE="${PATIENCE:-3}"

mkdir -p "${OUT_DIR}"

mapfile -t TRAIN_NPZ < <(find "${TRAIN_DIR}" -maxdepth 1 -name '*.npz' | sort)
mapfile -t VAL_NPZ < <(find "${VAL_DIR}" -maxdepth 1 -name '*.npz' | sort)

if [[ "${#TRAIN_NPZ[@]}" -eq 0 ]]; then
  echo "No train shards found under ${TRAIN_DIR}" >&2
  exit 2
fi

CHECKPOINT_PATH="${OUT_DIR}/mot17_mtcr_${STAMP}.npz"
LOG_PATH="${OUT_DIR}/run.log"

{
  echo "[mtcr-train] out_dir=${OUT_DIR}"
  echo "[mtcr-train] train_dir=${TRAIN_DIR}"
  echo "[mtcr-train] val_dir=${VAL_DIR}"
  echo "[mtcr-train] train_shards=${#TRAIN_NPZ[@]} val_shards=${#VAL_NPZ[@]}"
  echo "[mtcr-train] checkpoint=${CHECKPOINT_PATH}"
  echo "[mtcr-train] device=${DEVICE} epochs=${EPOCHS} topk=${TOPK} delta_scale=${DELTA_SCALE}"
} | tee "${LOG_PATH}"

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/train_mtcr_from_gt_tracks.py" \
  --train-npz "${TRAIN_NPZ[@]}" \
  --val-npz "${VAL_NPZ[@]}" \
  --out-npz "${CHECKPOINT_PATH}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --grad-clip "${GRAD_CLIP}" \
  --hist-hidden "${HIST_HIDDEN}" \
  --comp-hidden "${COMP_HIDDEN}" \
  --topk "${TOPK}" \
  --margin-threshold "${MARGIN_THRESHOLD}" \
  --margin-quantile "${MARGIN_QUANTILE}" \
  --margin-temperature "${MARGIN_TEMPERATURE}" \
  --delta-scale "${DELTA_SCALE}" \
  --min-history "${MIN_HISTORY}" \
  --temperature "${TEMPERATURE}" \
  --duel-margin "${DUEL_MARGIN}" \
  --loss-duel-weight "${LOSS_DUEL_WEIGHT}" \
  --loss-safe-weight "${LOSS_SAFE_WEIGHT}" \
  --loss-bg-weight "${LOSS_BG_WEIGHT}" \
  --loss-gate-weight "${LOSS_GATE_WEIGHT}" \
  --ambiguous-weight "${AMBIGUOUS_WEIGHT}" \
  --easy-weight "${EASY_WEIGHT}" \
  --background-weight "${BACKGROUND_WEIGHT}" \
  --gate-cap "${GATE_CAP}" \
  --bg-scale "${BG_SCALE}" \
  --select-amb-weight "${SELECT_AMB_WEIGHT}" \
  --select-bg-weight "${SELECT_BG_WEIGHT}" \
  --select-easy-weight "${SELECT_EASY_WEIGHT}" \
  --patience "${PATIENCE}" \
  2>&1 | tee -a "${LOG_PATH}"

echo "${CHECKPOINT_PATH}" | tee "${OUT_DIR}/latest_checkpoint.txt"
echo "[mtcr-train] done checkpoint=${CHECKPOINT_PATH}" | tee -a "${LOG_PATH}"
