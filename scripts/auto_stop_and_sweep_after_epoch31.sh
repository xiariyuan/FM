#!/usr/bin/env bash
set -euo pipefail

# Auto workflow:
# 1) Wait until epoch 31 validation summary exists for v11b
# 2) Stop the running training tmux session (to free GPU)
# 3) Run a small association-parameter sweep on:
#    - v10 best checkpoint (epoch 1)
#    - v11b checkpoint at epoch 31
#
# This is intended to be launched inside tmux so it can run unattended.

REPO_ROOT="/gemini/code/FMtrack-main/FM-Track"
cd "${REPO_ROOT}"

TRAIN_SESSION="${TRAIN_SESSION:-fmtrack_v11b_train}"
EXP_DIR="${EXP_DIR:-outputs/bytetrack_fa_mot_mot17_v11b_resume_e29_g2_det50_bs4_acc2_lr5e5}"
WAIT_EPOCH="${WAIT_EPOCH:-31}"

VAL_SUMMARY="${EXP_DIR}/val/epoch_${WAIT_EPOCH}/tracker/MOT17-train/pedestrian_summary.txt"
CKPT_V11B="${EXP_DIR}/checkpoint_epoch_${WAIT_EPOCH}.pth"

# Global-best reference checkpoint (from prior v10 run)
CKPT_V10="${CKPT_V10:-outputs/bytetrack_fa_mot_mot17_v10_tune_matchloss_val0405_allDet_fix27_longseq_len6/checkpoint_epoch_1.pth}"

# Use a single shared config for fair inference settings (architecture is identical across these runs).
CONFIG_PATH="${CONFIG_PATH:-configs/bytetrack_fa_mot_mot17_v10_tune_matchloss_val0405_allDet_fix27_longseq_len6.yaml}"

DATA_ROOT="${DATA_ROOT:-/gemini/code/datasets}"
VAL_SEQS="${VAL_SEQS:-MOT17-04,MOT17-05}"

# A compact, high-value sweep range around known-good settings.
ID_LIST="${ID_LIST:-0.002,0.003,0.004}"
IOU_LIST="${IOU_LIST:-0.25,0.30,0.35}"
DETMAX_LIST="${DETMAX_LIST:-50,60}"

echo "[auto] $(date) waiting for val summary: ${VAL_SUMMARY}"
while [[ ! -f "${VAL_SUMMARY}" ]]; do
  sleep 60
done
echo "[auto] $(date) epoch ${WAIT_EPOCH} validation summary detected."
tail -n 2 "${VAL_SUMMARY}" || true

echo "[auto] $(date) attempting to stop training session: ${TRAIN_SESSION}"
if tmux has-session -t "${TRAIN_SESSION}" 2>/dev/null; then
  tmux send-keys -t "${TRAIN_SESSION}" C-c
  echo "[auto] sent Ctrl-C to ${TRAIN_SESSION}"
else
  echo "[auto] WARNING: tmux session not found: ${TRAIN_SESSION}"
fi

# Wait for GPU to become available (best-effort).
echo "[auto] $(date) waiting for GPU memory to drop..."
gpu_ok="false"
for _ in $(seq 1 120); do
  if command -v nvidia-smi >/dev/null 2>&1; then
    mem="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
    if [[ "${mem}" =~ ^[0-9]+$ ]] && [[ "${mem}" -lt 2000 ]]; then
      echo "[auto] GPU memory OK: ${mem} MiB"
      gpu_ok="true"
      break
    fi
  fi
  sleep 10
done
if [[ "${gpu_ok}" != "true" ]]; then
  echo "[auto] ERROR: GPU still busy after waiting; aborting sweeps to avoid OOM."
  nvidia-smi || true
  exit 2
fi

# Ensure torch shared libs are discoverable for any compiled extensions used during inference.
export LD_LIBRARY_PATH="/root/miniconda3/lib:/root/miniconda3/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

OUT_ROOT="outputs/sweep_assoc_val0405_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUT_ROOT}"
echo "[auto] $(date) sweep out root: ${OUT_ROOT}"

run_sweep() {
  local name="$1"
  local ckpt="$2"
  local out_dir="${OUT_ROOT}/${name}"
  echo "[auto] $(date) starting sweep: ${name}"
  echo "[auto] checkpoint: ${ckpt}"
  python -u scripts/sweep_assoc_params.py \
    --config-path "${CONFIG_PATH}" \
    --checkpoint "${ckpt}" \
    --data-root "${DATA_ROOT}" \
    --dataset MOT17 \
    --split train \
    --out-root "${out_dir}" \
    --eval-only-val \
    --val-sequences "${VAL_SEQS}" \
    --id-thresh-list "${ID_LIST}" \
    --assoc-iou-gate-list "${IOU_LIST}" \
    --det-max-per-frame-list "${DETMAX_LIST}" \
    --keep-going
  echo "[auto] $(date) finished sweep: ${name}"
}

if [[ -f "${CKPT_V10}" ]]; then
  run_sweep "v10_epoch1" "${CKPT_V10}"
else
  echo "[auto] WARNING: missing CKPT_V10: ${CKPT_V10}"
fi

if [[ -f "${CKPT_V11B}" ]]; then
  run_sweep "v11b_epoch${WAIT_EPOCH}" "${CKPT_V11B}"
else
  echo "[auto] WARNING: missing CKPT_V11B: ${CKPT_V11B}"
fi

echo "[auto] $(date) all sweeps complete. Results:"
find "${OUT_ROOT}" -maxdepth 2 -name "sweep_assoc_results.csv" -print || true
