#!/usr/bin/env bash
set -euo pipefail

# Quick ablations on held-out val sequences (proxy).
# Usage:
#   bash scripts/run_paper_ablations_val.sh <CKPT> [DATASET] [VAL_SEQS]
#
# Examples:
#   bash scripts/run_paper_ablations_val.sh outputs/.../checkpoint_epoch_3.pth MOT20 MOT20-05
#   bash scripts/run_paper_ablations_val.sh outputs/.../checkpoint_epoch_3.pth MOT17 MOT17-02,MOT17-13

CKPT="${1:?checkpoint required}"
DATASET="${2:-MOT20}"
VAL_SEQS="${3:-MOT20-05}"
DATA_ROOT="${DATA_ROOT:-/gemini/code/datasets}"
OUT_ROOT="${OUT_ROOT:-outputs/_paper_ablation_val_${DATASET}}"

if [[ "${DATASET}" == "MOT20" ]]; then
  CONFIG="${CONFIG:-configs/experiments/bytetrack_fa_mot_mot20_v13_assoc_only_val05_reidmot20.yaml}"
  DET_ROOT="${DET_ROOT:-outputs/external_det/sw_yolox}"
  DET_PATTERN="{root}/{dataset}/{split}/{seq}.txt"
  SPLIT="train"
else
  CONFIG="${CONFIG:-configs/experiments/bytetrack_fa_mot_mot17_v13_tf_only_val0213_reid_da.yaml}"
  DET_ROOT="${DET_ROOT:-outputs/external_det/sw_yolox}"
  DET_PATTERN="{root}/{dataset}/{split}/{seq}.txt"
  SPLIT="train"
fi

mkdir -p "${OUT_ROOT}"

run_variant () {
  local name="$1"; shift
  local out_dir="${OUT_ROOT}/${name}"
  mkdir -p "${out_dir}"
  echo "[RUN] ${name}"
  python -u submit_bytetrack.py \
    --config-path "${CONFIG}" \
    --inference-model "${CKPT}" \
    --inference-dataset "${DATASET}" \
    --inference-split "${SPLIT}" \
    --data-root "${DATA_ROOT}" \
    --output-dir "${out_dir}" \
    --eval-only-val \
    --val-sequences "${VAL_SEQS}" \
    --det-source external \
    --external-det-root "${DET_ROOT}" \
    --external-det-pattern "${DET_PATTERN}" \
    "$@"
}

# Baseline (as-configured)
run_variant baseline

# Ablations
run_variant no_freq \
  --use-freq-aware false \
  --use-freq-decoder-v2 false \
  --use-freq-guided-assoc false

run_variant no_reid \
  --assoc-use-reid false \
  --assoc-feat-source yolox

run_variant no_hybrid \
  --assoc-mode reid

run_variant no_two_stage \
  --assoc-two-stage false

run_variant no_interp \
  --use-track-interpolation false

echo "[DONE] Outputs in ${OUT_ROOT}"
