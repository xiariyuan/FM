#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"

DETECTOR="${DETECTOR:-sw_yolox}"
MODE="${MODE:-base}"
SCOPE="${SCOPE:-full7}"
TAG="${TAG:-$(date +%Y%m%d_%H%M%S)}"

DUMP_ROOT="${DUMP_ROOT:-${REPO_ROOT}/outputs/runtime_assoc_dump_${DETECTOR}_${MODE}_${SCOPE}_${TAG}}"
RUN_DIR="${RUN_DIR:-${REPO_ROOT}/outputs/runtime_assoc_dump_run_${DETECTOR}_${MODE}_${SCOPE}_${TAG}}"
REPLAY_ROOT="${REPLAY_ROOT:-${REPO_ROOT}/outputs/runtime_replay_labeled_${DETECTOR}_${MODE}_${SCOPE}_${TAG}}"
SHARD_ROOT="${SHARD_ROOT:-${REPO_ROOT}/outputs/runtime_replay_shards_${DETECTOR}_${MODE}_${SCOPE}_${TAG}}"
LEARNED_ROOT="${LEARNED_ROOT:-${REPO_ROOT}/outputs/runtime_replay_learned_${DETECTOR}_${MODE}_${SCOPE}_${TAG}}"

TEACHER_MODEL="${TEACHER_MODEL:-}"
DUMP_TOPK="${DUMP_TOPK:-0}"
DUMP_MIN_SCORE="${DUMP_MIN_SCORE:-0.0}"
DUMP_NPZ_EVERY_N_GROUPS="${DUMP_NPZ_EVERY_N_GROUPS:-1024}"
GROUPS_PER_SHARD="${GROUPS_PER_SHARD:-1024}"
RANK_SCORE_COL="${RANK_SCORE_COL:-refined_score}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-15}"
TRAIN_BATCH_GROUPS="${TRAIN_BATCH_GROUPS:-24}"
TRAIN_TOPK="${TRAIN_TOPK:-5}"
TRAIN_DEVICE="${TRAIN_DEVICE:-cuda}"
VALID_ONLY="${VALID_ONLY:-1}"
TRAIN_SEQS="${TRAIN_SEQS:-}"
VAL_SEQS="${VAL_SEQS:-}"
VAL_RATIO="${VAL_RATIO:-0.2}"

# Default to a sequence-disjoint split for MOT17 full7 replay training.
# Random group splits can leak near-duplicate groups from the same sequence into both
# train/val and make offline gains look better than they really are.
if [[ -z "${TRAIN_SEQS}" && -z "${VAL_SEQS}" && "${SCOPE}" == "full7" ]]; then
  TRAIN_SEQS="MOT17-02-FRCNN,MOT17-04-FRCNN,MOT17-05-FRCNN,MOT17-09-FRCNN"
  VAL_SEQS="MOT17-10-FRCNN,MOT17-11-FRCNN,MOT17-13-FRCNN"
fi

mkdir -p "${RUN_DIR}" "${REPLAY_ROOT}" "${SHARD_ROOT}" "${LEARNED_ROOT}"

LABEL_CSV="${REPLAY_ROOT}/labeled_replay_allcand.csv"
LABEL_SUMMARY="${REPLAY_ROOT}/labeled_replay_allcand.summary.json"
LABEL_GROUP_JSONL="${REPLAY_ROOT}/labeled_replay_allcand.groups.jsonl"
LABEL_RECOVER="${REPLAY_ROOT}/labeled_replay_allcand.recoverability.json"
LEARNED_CKPT="${LEARNED_ROOT}/runtime_replay_full.pt"
LEARNED_METRICS="${LEARNED_ROOT}/runtime_replay_full.metrics.jsonl"

echo "[pipeline] detector=${DETECTOR} mode=${MODE} scope=${SCOPE} tag=${TAG}"
echo "[pipeline] dump_root=${DUMP_ROOT}"
echo "[pipeline] replay_root=${REPLAY_ROOT}"
echo "[pipeline] shard_root=${SHARD_ROOT}"
echo "[pipeline] learned_root=${LEARNED_ROOT}"
if [[ -n "${TRAIN_SEQS}" || -n "${VAL_SEQS}" ]]; then
  echo "[pipeline] train_seqs=${TRAIN_SEQS}"
  echo "[pipeline] val_seqs=${VAL_SEQS}"
else
  echo "[pipeline] train/val split=random_group_split val_ratio=${VAL_RATIO}"
fi

if [[ ! -f "${LABEL_CSV}" ]]; then
  echo "[step1] runtime dump with tensor shards"
  DUMP_SAVE_TENSORS=1 \
  DUMP_TOPK="${DUMP_TOPK}" \
  DUMP_MIN_SCORE="${DUMP_MIN_SCORE}" \
  DUMP_NPZ_EVERY_N_GROUPS="${DUMP_NPZ_EVERY_N_GROUPS}" \
  /bin/bash "${REPO_ROOT}/scripts/run_bytetrack_assoc_dump_external.sh" \
    "${DETECTOR}" "${MODE}" "${SCOPE}" "${DUMP_ROOT}" "${RUN_DIR}"

  echo "[step2] label runtime replay groups"
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/build_runtime_assoc_replay_labels.py" \
    --dump-root "${DUMP_ROOT}" \
    --dataset MOT17 \
    --data-root /gemini/code/datasets \
    --split train \
    --split-part full \
    --out-csv "${LABEL_CSV}" \
    --summary-json "${LABEL_SUMMARY}" \
    --out-group-jsonl "${LABEL_GROUP_JSONL}" \
    --out-recoverability-json "${LABEL_RECOVER}" \
    --topk 0 \
    --rank-score-col "${RANK_SCORE_COL}" \
    --ambiguity-margin 0.10
else
  echo "[skip] labeled replay already exists: ${LABEL_CSV}"
fi

echo "[step3] build trainable group shards"
BUILD_ARGS=(
  "${REPO_ROOT}/scripts/build_runtime_assoc_group_shards.py"
  --labeled-csv "${LABEL_CSV}"
  --tensor-root "${DUMP_ROOT}"
  --out-dir "${SHARD_ROOT}"
  --rank-score-col "${RANK_SCORE_COL}"
  --groups-per-shard "${GROUPS_PER_SHARD}"
)
if [[ -n "${TEACHER_MODEL}" ]]; then
  BUILD_ARGS+=(--teacher-model "${TEACHER_MODEL}")
fi
"${PYTHON_BIN}" "${BUILD_ARGS[@]}"

echo "[step4] train full learned runtime"
TRAIN_ARGS=(
  "${REPO_ROOT}/scripts/train_runtime_replay_reranker.py"
  --input-dir "${SHARD_ROOT}"
  --out-ckpt "${LEARNED_CKPT}"
  --metrics-path "${LEARNED_METRICS}"
  --device "${TRAIN_DEVICE}"
  --epochs "${TRAIN_EPOCHS}"
  --batch-groups "${TRAIN_BATCH_GROUPS}"
  --topk "${TRAIN_TOPK}"
  --val-ratio "${VAL_RATIO}"
)
if [[ "${VALID_ONLY}" == "1" ]]; then
  TRAIN_ARGS+=(--valid-only)
fi
if [[ -n "${TRAIN_SEQS}" ]]; then
  TRAIN_ARGS+=(--train-seqs "${TRAIN_SEQS}")
fi
if [[ -n "${VAL_SEQS}" ]]; then
  TRAIN_ARGS+=(--val-seqs "${VAL_SEQS}")
fi
"${PYTHON_BIN}" "${TRAIN_ARGS[@]}"

echo "[done] checkpoint=${LEARNED_CKPT}"
