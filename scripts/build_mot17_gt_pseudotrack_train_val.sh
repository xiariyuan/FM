#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"
BOT_ROOT="${BOT_ROOT:-${REPO_ROOT}/external/BoT-SORT-main}"
DATA_ROOT="${DATA_ROOT:-/gemini/code/datasets}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
DEVICE="${DEVICE:-cuda}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs/gt_pseudotrack_mot17_${TS}}"
OUT_ROOT="$(realpath -m "${OUT_ROOT}")"

TRAIN_SPLIT_PART="${TRAIN_SPLIT_PART:-train_half}"
VAL_SPLIT_PART="${VAL_SPLIT_PART:-train_half}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_HISTORY="${MAX_HISTORY:-8}"
MIN_HISTORY="${MIN_HISTORY:-3}"
MAX_GAP="${MAX_GAP:-30}"
MAX_FRAMES="${MAX_FRAMES:-0}"
CANDIDATE_TOPK="${CANDIDATE_TOPK:-16}"
MAX_HARD_NEGATIVES="${MAX_HARD_NEGATIVES:-4}"
MAX_RANDOM_NEGATIVES="${MAX_RANDOM_NEGATIVES:-2}"
CANDIDATE_MIN_MOTION="${CANDIDATE_MIN_MOTION:-0.0}"
SMOOTH_ALPHA="${SMOOTH_ALPHA:-0.9}"
IOU_POS="${IOU_POS:-0.7}"
IOU_IGNORE="${IOU_IGNORE:-0.5}"
FEATURE_DTYPE="${FEATURE_DTYPE:-float16}"
SEED="${SEED:-123}"
WRITE_CSV="${WRITE_CSV:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

TRAIN_SEQS=(${TRAIN_SEQS:-MOT17-02-FRCNN MOT17-04-FRCNN MOT17-05-FRCNN MOT17-09-FRCNN})
VAL_SEQS=(${VAL_SEQS:-MOT17-10-FRCNN MOT17-11-FRCNN MOT17-13-FRCNN})

REID_CFG="${REID_CFG:-${BOT_ROOT}/fast_reid/configs/MOT17/sbs_S50.yml}"
REID_WTS="${REID_WTS:-${BOT_ROOT}/pretrained/mot17_sbs_S50.pth}"

mkdir -p "${OUT_ROOT}"
echo "[out_root] ${OUT_ROOT}"

TRAIN_SHARDS_DIR="${OUT_ROOT}/train_shards"
VAL_SHARDS_DIR="${OUT_ROOT}/val_shards"
TRAIN_LIST="${OUT_ROOT}/train_npz_list.txt"
VAL_LIST="${OUT_ROOT}/val_npz_list.txt"

mkdir -p "${TRAIN_SHARDS_DIR}" "${VAL_SHARDS_DIR}"
: > "${TRAIN_LIST}"
: > "${VAL_LIST}"

build_one_seq() {
  local split_name="$1"
  local split_part="$2"
  local out_dir="$3"
  local list_path="$4"
  local seq_name="$5"
  local shard_prefix="${out_dir}/${seq_name}"
  local out_npz="${shard_prefix}_groups.npz"
  local out_csv="${shard_prefix}_pairs.csv"

  if [[ "${SKIP_EXISTING}" == "1" && -f "${out_npz}" ]]; then
    if [[ "${WRITE_CSV}" != "1" || -f "${out_csv}" ]]; then
      echo "[skip] ${split_name} ${seq_name} existing shard"
      printf '%s\n' "${out_npz}" >> "${list_path}"
      return
    fi
  fi

  echo "[build] ${split_name} ${seq_name} (${split_part})"
  local cmd=(
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/build_gt_pseudotrack_groups.py"
    --dataset MOT17
    --data-root "${DATA_ROOT}"
    --seqs "${seq_name}"
    --split-part "${split_part}"
    --fast-reid-config "${REID_CFG}"
    --fast-reid-weights "${REID_WTS}"
    --device "${DEVICE}"
    --batch-size "${BATCH_SIZE}"
    --max-history "${MAX_HISTORY}"
    --min-history "${MIN_HISTORY}"
    --feature-dtype "${FEATURE_DTYPE}"
    --seed "${SEED}"
    --smooth-alpha "${SMOOTH_ALPHA}"
    --iou-pos "${IOU_POS}"
    --iou-ignore "${IOU_IGNORE}"
    --max-gap "${MAX_GAP}"
    --max-frames "${MAX_FRAMES}"
    --candidate-topk "${CANDIDATE_TOPK}"
    --max-hard-negatives "${MAX_HARD_NEGATIVES}"
    --max-random-negatives "${MAX_RANDOM_NEGATIVES}"
    --candidate-min-motion "${CANDIDATE_MIN_MOTION}"
    --include-background
    --out-npz "${out_npz}"
  )
  if [[ "${WRITE_CSV}" == "1" ]]; then
    cmd+=(--out-csv "${out_csv}")
  fi
  "${cmd[@]}"
  printf '%s\n' "${out_npz}" >> "${list_path}"
}

for seq_name in "${TRAIN_SEQS[@]}"; do
  build_one_seq "train" "${TRAIN_SPLIT_PART}" "${TRAIN_SHARDS_DIR}" "${TRAIN_LIST}" "${seq_name}"
done

for seq_name in "${VAL_SEQS[@]}"; do
  build_one_seq "val" "${VAL_SPLIT_PART}" "${VAL_SHARDS_DIR}" "${VAL_LIST}" "${seq_name}"
done

echo "[done] train_shards=${TRAIN_SHARDS_DIR}"
echo "[done] val_shards=${VAL_SHARDS_DIR}"
echo "[done] train_list=${TRAIN_LIST}"
echo "[done] val_list=${VAL_LIST}"
