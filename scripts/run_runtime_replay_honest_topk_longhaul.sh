#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"

TAG="${1:-$(date +%Y%m%d_%H%M%S)}"

SHARD_ROOT="${SHARD_ROOT:-${REPO_ROOT}/outputs/runtime_replay_shards_sw_yolox_base_full7_20260318_full_learned_basefull7_fixreview}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs/runtime_replay_honest_topk_longhaul_${TAG}}"
TRAIN_OUT="${OUT_ROOT}/train"
CKPT_PATH="${TRAIN_OUT}/runtime_replay_honest_topk.pt"
METRICS_PATH="${TRAIN_OUT}/runtime_replay_honest_topk.metrics.jsonl"
SWEEP_OUT="${OUT_ROOT}/proxy_epoch_sweep"
LOG_PATH="${OUT_ROOT}/longhaul.log"
STATUS_PATH="${OUT_ROOT}/job_status.txt"
PID_PATH="${OUT_ROOT}/job_pid.txt"

mkdir -p "${TRAIN_OUT}" "${SWEEP_OUT}"

echo "$$" > "${PID_PATH}"
echo "running" > "${STATUS_PATH}"
exec > >(tee -a "${LOG_PATH}") 2>&1
trap 'status=$?; echo "[honest_topk] exit_code=${status} finished_at=$(date --iso-8601=seconds)"; echo "${status}" > "${STATUS_PATH}"' EXIT

echo "[honest_topk] tag=${TAG}"
echo "[honest_topk] shard_root=${SHARD_ROOT}"
echo "[honest_topk] out_root=${OUT_ROOT}"
echo "[honest_topk] log_path=${LOG_PATH}"

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/train_runtime_replay_reranker.py" \
  --input-dir "${SHARD_ROOT}" \
  --out-ckpt "${CKPT_PATH}" \
  --metrics-path "${METRICS_PATH}" \
  --device cuda \
  --epochs 10 \
  --patience 4 \
  --batch-groups 24 \
  --topk 5 \
  --valid-only \
  --fixed-val-sample \
  --honest-topk \
  --exclude-unrecoverable-positive-loss \
  --loss-distill-weight 0.0 \
  --train-seqs MOT17-04-FRCNN,MOT17-05-FRCNN,MOT17-09-FRCNN,MOT17-10-FRCNN,MOT17-11-FRCNN \
  --val-seqs MOT17-02-FRCNN,MOT17-13-FRCNN \
  --train-groups-per-epoch 4096 \
  --val-groups-per-epoch 2048 \
  --sample-groups-per-shard 128 \
  --sample-hard-positive-weight 8.0 \
  --sample-ambiguous-weight 3.5 \
  --sample-easy-weight 0.35 \
  --sample-background-weight 0.55 \
  --hard-positive-weight 4.0 \
  --ambiguous-weight 2.2 \
  --easy-weight 0.5 \
  --background-weight 0.75 \
  --loss-gate-weight 0.04 \
  --gate-positive-target 0.20 \
  --select-hard-weight 2.4 \
  --select-bg-weight 0.25 \
  --select-easy-weight 0.10

echo "[honest_topk] proxy sweep + best full7"
bash "${REPO_ROOT}/scripts/run_runtime_replay_proxy_epoch_sweep.sh" \
  sw_yolox "${TRAIN_OUT}" "${SWEEP_OUT}" 1

echo "[honest_topk] done train_dir=${TRAIN_OUT}"
