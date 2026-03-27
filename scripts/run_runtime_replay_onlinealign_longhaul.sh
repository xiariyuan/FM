#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"

TAG="${1:-$(date +%Y%m%d_%H%M%S)}"

SHARD_ROOT="${SHARD_ROOT:-${REPO_ROOT}/outputs/runtime_replay_shards_sw_yolox_base_full7_20260318_full_learned_basefull7_fixreview}"
INIT_CKPT="${INIT_CKPT:-${REPO_ROOT}/outputs/runtime_replay_learned_sw_yolox_base_full7_hardtrain_mixedval_gate/runtime_replay_hardtrain_mixedval_gate.pt}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs/runtime_replay_onlinealign_longhaul_${TAG}}"
TRAIN_OUT="${OUT_ROOT}/train"
CKPT_PATH="${TRAIN_OUT}/runtime_replay_onlinealign.pt"
METRICS_PATH="${TRAIN_OUT}/runtime_replay_onlinealign.metrics.jsonl"
PROXY_OUT="${OUT_ROOT}/eval_proxy0213"
FULL7_OUT="${OUT_ROOT}/eval_full7"
LOG_PATH="${OUT_ROOT}/longhaul.log"
STATUS_PATH="${OUT_ROOT}/job_status.txt"
PID_PATH="${OUT_ROOT}/job_pid.txt"

mkdir -p "${TRAIN_OUT}" "${PROXY_OUT}" "${FULL7_OUT}"

echo "$$" > "${PID_PATH}"
echo "running" > "${STATUS_PATH}"
exec > >(tee -a "${LOG_PATH}") 2>&1
trap 'status=$?; echo "[onlinealign] exit_code=${status} finished_at=$(date --iso-8601=seconds)"; echo "${status}" > "${STATUS_PATH}"' EXIT

echo "[onlinealign] tag=${TAG}"
echo "[onlinealign] shard_root=${SHARD_ROOT}"
echo "[onlinealign] init_ckpt=${INIT_CKPT}"
echo "[onlinealign] out_root=${OUT_ROOT}"
echo "[onlinealign] log_path=${LOG_PATH}"

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/train_runtime_replay_reranker.py" \
  --input-dir "${SHARD_ROOT}" \
  --out-ckpt "${CKPT_PATH}" \
  --metrics-path "${METRICS_PATH}" \
  --init-ckpt "${INIT_CKPT}" \
  --device cuda \
  --epochs 10 \
  --batch-groups 24 \
  --topk 5 \
  --valid-only \
  --fixed-val-sample \
  --train-seqs MOT17-04-FRCNN,MOT17-05-FRCNN,MOT17-09-FRCNN,MOT17-10-FRCNN,MOT17-11-FRCNN \
  --val-seqs MOT17-02-FRCNN,MOT17-13-FRCNN \
  --train-groups-per-epoch 4096 \
  --val-groups-per-epoch 2048 \
  --sample-groups-per-shard 128 \
  --sample-hard-positive-weight 6.0 \
  --sample-ambiguous-weight 3.0 \
  --sample-easy-weight 0.35 \
  --sample-background-weight 0.6 \
  --hard-positive-weight 3.0 \
  --ambiguous-weight 2.0 \
  --easy-weight 0.6 \
  --background-weight 0.9 \
  --loss-gate-weight 0.08 \
  --gate-positive-target 0.16 \
  --select-hard-weight 2.2 \
  --select-bg-weight 0.35 \
  --select-easy-weight 0.15

echo "[onlinealign] proxy0213 evaluation"
bash "${REPO_ROOT}/scripts/run_bytetrack_runtime_replay_external.sh" \
  sw_yolox proxy0213 "${CKPT_PATH}" "${PROXY_OUT}"

echo "[onlinealign] full7 evaluation"
bash "${REPO_ROOT}/scripts/run_bytetrack_runtime_replay_external.sh" \
  sw_yolox full7 "${CKPT_PATH}" "${FULL7_OUT}"

echo "[onlinealign] done ckpt=${CKPT_PATH}"
