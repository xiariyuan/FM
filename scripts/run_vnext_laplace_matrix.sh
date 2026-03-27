#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/gemini/code/FMtrack-main/FM-Track"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
DATA_ROOT="${DATA_ROOT:-/gemini/code/datasets}"
DATASET="${1:-MOT17}"
CKPT="${2:-outputs/bytetrack_fa_mot_mot17_v13_tf_only_val0213/checkpoint_epoch_0.pth}"
OUT_ROOT="${OUT_ROOT:-outputs/_vnext_laplace_matrix_${DATASET}}"
DET_ROOT="${DET_ROOT:-outputs/external_det/sw_yolox}"

case "${DATASET}" in
  MOT17)
    SPLIT="${SPLIT:-train}"
    VAL_SEQS="${VAL_SEQS:-MOT17-02,MOT17-13}"
    BASE_CFG="${BASE_CFG:-configs/experiments/bytetrack_fa_mot_mot17_v14_ctrl_base_reid_da_val0213.yaml}"
    LAPLACE_CFG="${LAPLACE_CFG:-configs/experiments/bytetrack_fa_mot_mot17_v15_laplace_reid_da_val0213.yaml}"
    DET_PATTERN="${DET_PATTERN:-{root}/{dataset}/{split}/{seq}.txt}"
    EXTRA_ARGS=(--detector-filter FRCNN)
    ;;
  MOT20)
    SPLIT="${SPLIT:-train}"
    VAL_SEQS="${VAL_SEQS:-MOT20-05}"
    BASE_CFG="${BASE_CFG:-configs/experiments/bytetrack_fa_mot_mot20_v13_assoc_only_val05_reidmot20.yaml}"
    LAPLACE_CFG="${LAPLACE_CFG:-configs/experiments/bytetrack_fa_mot_mot20_v15_laplace_assoc_val05_reidmot20.yaml}"
    DET_PATTERN="${DET_PATTERN:-{root}/{dataset}/{split}/{seq}.txt}"
    EXTRA_ARGS=()
    ;;
  *)
    echo "Unsupported dataset: ${DATASET}" >&2
    exit 2
    ;;
esac

mkdir -p "${OUT_ROOT}"

run_one() {
  local name="$1"
  local cfg="$2"
  local out_dir="${OUT_ROOT}/${name}"
  mkdir -p "${out_dir}"
  echo "[run] ${DATASET} ${name}"
  "${PYTHON_BIN}" -u submit_bytetrack.py \
    --config-path "${cfg}" \
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
    "${EXTRA_ARGS[@]}"
}

run_one base "${BASE_CFG}"
run_one laplace "${LAPLACE_CFG}"

"${PYTHON_BIN}" scripts/collect_trackeval_metrics.py \
  "${OUT_ROOT}/base" \
  "${OUT_ROOT}/laplace" \
  --csv "${OUT_ROOT}/summary.csv"

echo "[done] outputs in ${OUT_ROOT}"
