#!/usr/bin/env bash
set -euo pipefail

# Queue: v13 Transformer vs (lowfreq) Mamba ablations on MOT17 (ByteTrack feature training).
#
# This script is intended to be launched inside tmux and run unattended.
# It will:
#   1) Stop the currently running detector finetune (single GPU constraint)
#   2) Train tracker modules for each config (v13 TF-only, then TF+lowfreq-Mamba)
#   3) Select best epoch by HOTA on val (02/13 by default via config)
#   4) Run a compact association sweep around known-good settings

REPO_ROOT="/gemini/code/FMtrack-main/FM-Track"
cd "${REPO_ROOT}"

export LD_LIBRARY_PATH="/root/miniconda3/lib:/root/miniconda3/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

DATA_ROOT="${DATA_ROOT:-/gemini/code/datasets}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"

TF_ONLY_CFG="configs/experiments/bytetrack_fa_mot_mot17_v13_tf_only_val0213.yaml"
TF_ONLY_RESUME_CFG="configs/experiments/bytetrack_fa_mot_mot17_v13_tf_only_val0213_resume_epoch0.yaml"
TF_PLUS_CFG="configs/experiments/bytetrack_fa_mot_mot17_v13_tf_plus_lowfreq_mamba_val0213.yaml"

CONFIGS=()
if [[ -f "outputs/bytetrack_fa_mot_mot17_v13_tf_only_val0213/checkpoint_0.pth" ]]; then
  echo "[queue] detected existing TF-only checkpoint_0.pth -> resume from epoch0"
  CONFIGS+=("${TF_ONLY_RESUME_CFG}")
else
  CONFIGS+=("${TF_ONLY_CFG}")
fi
CONFIGS+=("${TF_PLUS_CFG}")

SWEEP_ID_LIST="${SWEEP_ID_LIST:-0.04,0.05,0.08}"
SWEEP_TAU_LIST="${SWEEP_TAU_LIST:-0.03,0.05,0.07}"
SWEEP_IOU_GATE_LIST="${SWEEP_IOU_GATE_LIST:-0.20,0.25,0.30}"
SWEEP_MISS_TOL_LIST="${SWEEP_MISS_TOL_LIST:-30,50}"

stop_detector_tmux() {
  local train_tmux=""
  local watch_tmux=""
  if [[ -f "outputs/detector/latest_tmux_session.txt" ]]; then
    train_tmux="$(cat outputs/detector/latest_tmux_session.txt | tr -d '\n')"
  fi
  if [[ -f "outputs/detector/latest_watch_tmux_session.txt" ]]; then
    watch_tmux="$(cat outputs/detector/latest_watch_tmux_session.txt | tr -d '\n')"
  fi

  if [[ -n "${train_tmux}" ]] && tmux has-session -t "${train_tmux}" 2>/dev/null; then
    echo "[queue] stopping detector training tmux: ${train_tmux}"
    pid="$(tmux list-panes -t "${train_tmux}" -F "#{pane_pid}" | head -n 1 | tr -d '\r')"
    tmux send-keys -t "${train_tmux}" C-c || true
    if [[ -n "${pid}" ]]; then
      for _ in $(seq 1 120); do
        if ! kill -0 "${pid}" 2>/dev/null; then
          break
        fi
        sleep 1
      done
    else
      sleep 15
    fi
    tmux kill-session -t "${train_tmux}" || true
  fi
  if [[ -n "${watch_tmux}" ]] && tmux has-session -t "${watch_tmux}" 2>/dev/null; then
    echo "[queue] stopping detector watchdog tmux: ${watch_tmux}"
    tmux kill-session -t "${watch_tmux}" || true
  fi
}

get_exp_name() {
  local cfg="$1"
  "${PYTHON_BIN}" - <<PY
import yaml
with open("${cfg}", "r") as f:
    d=yaml.safe_load(f)
print(d.get("EXP_NAME","" ).strip())
PY
}

run_train() {
  local cfg="$1"
  local exp
  exp="$(get_exp_name "${cfg}")"
  if [[ -z "${exp}" ]]; then
    echo "[queue] ERROR: EXP_NAME missing in ${cfg}" >&2
    exit 2
  fi
  echo "[queue] ===== Train: ${exp} ====="
  "${PYTHON_BIN}" -u train_bytetrack.py --config-path "${cfg}" --data-root "${DATA_ROOT}" --exp-name "${exp}"
  echo "[queue] ===== Done train: ${exp} ====="
}

select_best_ckpt() {
  local exp="$1"
  "${PYTHON_BIN}" -u scripts/select_best_bytetrack_ckpt.py --exp-dir "outputs/${exp}" --metric HOTA --dataset MOT17 --split train
}

run_sweep() {
  local cfg="$1"
  local exp="$2"
  local ckpt="$3"
  local ts
  ts="$(date +%Y%m%d_%H%M%S)"
  local out_root="outputs/sweep_v13_${exp}_${ts}"
  mkdir -p "${out_root}"
  echo "[queue] sweep out_root=${out_root}"

  "${PYTHON_BIN}" -u scripts/sweep_assoc_params.py \
    --config-path "${cfg}" \
    --checkpoint "${ckpt}" \
    --data-root "${DATA_ROOT}" \
    --dataset MOT17 \
    --split train \
    --out-root "${out_root}" \
    --eval-only-val \
    --val-sequences "MOT17-02,MOT17-13" \
    --assoc-mode-list "logit,hybrid,feature" \
    --id-thresh-list "${SWEEP_ID_LIST}" \
    --assoc-feat-tau-list "${SWEEP_TAU_LIST}" \
    --assoc-iou-gate-list "${SWEEP_IOU_GATE_LIST}" \
    --miss-tolerance-list "${SWEEP_MISS_TOL_LIST}" \
    --det-thresh-list "0.25" \
    --newborn-thresh-list "0.25" \
    --det-max-per-frame-list "60" \
    --keep-going
}

echo "[queue] $(date) starting v13 TF vs Mamba queue"
echo "[queue] single GPU: stopping detector finetune first"
stop_detector_tmux

for cfg in "${CONFIGS[@]}"; do
  exp="$(get_exp_name "${cfg}")"
  run_train "${cfg}"
  echo "[queue] selecting best checkpoint for ${exp}"
  best_out="$(select_best_ckpt "${exp}")"
  echo "${best_out}"
  best_ckpt="$(echo "${best_out}" | awk -F= /^checkpoint=/{print })"
  if [[ -z "${best_ckpt}" ]] || [[ ! -f "${best_ckpt}" ]]; then
    echo "[queue] ERROR: best checkpoint not found for ${exp}: ${best_ckpt}" >&2
    exit 3
  fi
  echo "[queue] running sweep for ${exp} ckpt=${best_ckpt}"
  run_sweep "${cfg}" "${exp}" "${best_ckpt}"
done

echo "[queue] $(date) all experiments finished"
