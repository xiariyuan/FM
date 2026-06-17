#!/usr/bin/env bash
set -euo pipefail

# Parallel GT pseudo-track shard builder for MOT20 HACA v3 retrain.
# Runs up to N build processes concurrently to maximize GPU/CPU utilization.

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
DATA_ROOT="${DATA_ROOT:-/gemini/code/datasets}"
BOT_ROOT="${REPO_ROOT}/external/BoT-SORT-main"
DEVICE="${DEVICE:-cuda}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs/haca_mot20_train_20260613_082005}"
PSEUDOTRACK_DIR="${OUT_ROOT}/gt_pseudotrack"
MAX_PARALLEL="${MAX_PARALLEL:-8}"
LOG_DIR="${PSEUDOTRACK_DIR}/build_logs"
FRAME_WINDOW="${FRAME_WINDOW:-120}"

mkdir -p "${LOG_DIR}"

# Generate all shard commands
declare -a COMMANDS=()

build_cmd() {
  local seq="$1"
  local split_part="$2"
  local out_dir="$3"
  local start_frame="$4"
  local end_frame="$5"

  local frame_start="${start_frame}"
  while [[ "${frame_start}" -le "${end_frame}" ]]; do
    local frame_end=$((frame_start + FRAME_WINDOW - 1))
    if [[ "${frame_end}" -gt "${end_frame}" ]]; then
      frame_end="${end_frame}"
    fi
    local suffix
    printf -v suffix "f%04d_%04d" "${frame_start}" "${frame_end}"
    local npz_path="${out_dir}/${seq}_${suffix}_groups.npz"

    # Skip if already exists and valid
    if [[ -f "${npz_path}" ]] && "${PYTHON_BIN}" -c "import numpy as np; np.load('${npz_path}', allow_pickle=True)" 2>/dev/null; then
      echo "[skip] ${seq} ${split_part} ${suffix} (already exists)"
    else
      local log_file="${LOG_DIR}/${seq}_${suffix}.log"
      COMMANDS+=("${PYTHON_BIN} ${REPO_ROOT}/scripts/build_gt_pseudotrack_groups.py \
        --dataset MOT20 \
        --data-root ${DATA_ROOT} \
        --seqs ${seq} \
        --split-part ${split_part} \
        --fast-reid-config ${BOT_ROOT}/fast_reid/configs/MOT20/sbs_S50.yml \
        --fast-reid-weights ${BOT_ROOT}/pretrained/mot20_sbs_S50.pth \
        --device ${DEVICE} \
        --batch-size 4 \
        --max-history 8 \
        --min-history 3 \
        --feature-dtype float16 \
        --seed 123 \
        --smooth-alpha 0.9 \
        --iou-pos 0.7 \
        --iou-ignore 0.5 \
        --max-gap 30 \
        --candidate-topk 16 \
        --max-hard-negatives 6 \
        --max-random-negatives 2 \
        --positive-keep-prob 0.7 \
        --include-background \
        --frame-start ${frame_start} \
        --frame-end ${frame_end} \
        --out-npz ${npz_path} \
        --out-csv ${out_dir}/${seq}_${suffix}_pairs.csv \
        > ${log_file} 2>&1")
    fi
    frame_start=$((frame_end + 1))
  done
}

# Generate all shard commands
echo "[plan] Generating shard commands..."
build_cmd MOT20-01 train_half "${PSEUDOTRACK_DIR}/train_shards" 1 215
build_cmd MOT20-02 train_half "${PSEUDOTRACK_DIR}/train_shards" 1 1392
build_cmd MOT20-03 train_half "${PSEUDOTRACK_DIR}/train_shards" 1 1203
build_cmd MOT20-05 val_half   "${PSEUDOTRACK_DIR}/val_shards"   1 1657

TOTAL=${#COMMANDS[@]}
echo "[plan] ${TOTAL} shards to build (max ${MAX_PARALLEL} parallel)"

if [[ ${TOTAL} -eq 0 ]]; then
  echo "[done] All shards already built."
  exit 0
fi

# Run in parallel with GNU xargs
printf '%s\n' "${COMMANDS[@]}" | xargs -P "${MAX_PARALLEL}" -I {} bash -c '{}'

echo "[done] All ${TOTAL} shards built."
