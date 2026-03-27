#!/usr/bin/env bash
set -euo pipefail

ROOT="/gemini/code/FMtrack-main/FM-Track"
CFG="$ROOT/configs/bytetrack_fa_mot_mot17_v5_core_eval.yaml"
EXP="bytetrack_fa_mot_mot17_v5_core"
CKPT_DIR="$ROOT/outputs/$EXP"
OUT_DIR="$CKPT_DIR/val_rev"
PY="/root/miniconda3/bin/python"
export PYTHONUNBUFFERED=1
if command -v stdbuf >/dev/null 2>&1; then
  PY_CMD=(stdbuf -oL -eL "$PY")
else
  PY_CMD=("$PY")
fi

mkdir -p "$OUT_DIR"

mapfile -t ckpts < <(ls "$CKPT_DIR"/checkpoint_epoch_*.pth 2>/dev/null \
  | sed -E 's/.*checkpoint_epoch_([0-9]+)\.pth/\1 \0/' \
  | sort -nr \
  | awk '{print $2}')

if [ "${#ckpts[@]}" -eq 0 ]; then
  echo "No checkpoints found in $CKPT_DIR" >&2
  exit 1
fi

for ckpt in "${ckpts[@]}"; do
  epoch=$(basename "$ckpt" | sed -E 's/checkpoint_epoch_([0-9]+)\.pth/\1/')
  run_dir="$OUT_DIR/epoch_${epoch}"
  if [ -f "$run_dir/pedestrian_summary.txt" ]; then
    echo "[skip] epoch ${epoch} summary exists"
    continue
  fi
  mkdir -p "$run_dir"
  echo "[run] epoch ${epoch} -> $run_dir"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${PY_CMD[@]}" -u "$ROOT/submit_bytetrack.py" \
    --config-path "$CFG" \
    --inference-model "$ckpt" \
    --inference-dataset MOT17 \
    --inference-split train \
    --output-dir "$run_dir" \
    >> "$OUT_DIR/eval_epoch_${epoch}.log" 2>&1
  echo "[done] epoch ${epoch}"
done
