#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/gemini/code/FMtrack-main/FM-Track"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
DATA_ROOT="${DATA_ROOT:-/gemini/code/datasets}"
CKPT="${CKPT:-outputs/bytetrack_fa_mot_mot17_v13_tf_only_val0213/checkpoint_epoch_0.pth}"
DET_ROOT="${DET_ROOT:-outputs/external_det/sw_yolox}"
VAL_SEQS="${VAL_SEQS:-MOT17-02,MOT17-13}"
OUT_ROOT="${OUT_ROOT:-outputs/v15_laplace_proxy0213}"

BASE_CFG="configs/experiments/bytetrack_fa_mot_mot17_v14_ctrl_base_reid_da_val0213.yaml"
LAPLACE_CFG="configs/experiments/bytetrack_fa_mot_mot17_v15_laplace_reid_da_val0213.yaml"

mkdir -p "${OUT_ROOT}"

run_one() {
  local name="$1"
  local cfg="$2"
  local out_dir="${OUT_ROOT}/${name}"
  mkdir -p "${out_dir}"
  echo "[run] ${name}"
  "${PYTHON_BIN}" -u submit_bytetrack.py \
    --config-path "${cfg}" \
    --inference-model "${CKPT}" \
    --inference-dataset MOT17 \
    --inference-split train \
    --data-root "${DATA_ROOT}" \
    --output-dir "${out_dir}" \
    --eval-only-val \
    --val-sequences "${VAL_SEQS}" \
    --det-source external \
    --external-det-root "${DET_ROOT}" \
    --external-det-pattern '{root}/{dataset}/{split}/{seq}.txt' \
    --detector-filter FRCNN
}

run_one base "${BASE_CFG}"
run_one laplace "${LAPLACE_CFG}"

echo "[done] outputs in ${OUT_ROOT}"
