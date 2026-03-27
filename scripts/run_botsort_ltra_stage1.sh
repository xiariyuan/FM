#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/gemini/code/FMtrack-main/FM-Track"
cd "${REPO_ROOT}"

TARGET="${1:-all}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"

run_one_dataset() {
  local dataset="$1"
  local lower
  lower="$(echo "${dataset}" | tr '[:upper:]' '[:lower:]')"
  local out_root="outputs/botsort_ltra_stage1/${dataset}"
  mkdir -p "${out_root}"

  echo "[stage1] running ${dataset} base vs LTRA"
  if [[ "${dataset}" == "MOT17" && "${MOT17_FULL:-0}" != "1" ]]; then
    # Fast validation protocol for MOT17:
    # - Only evaluate a small subset of sequences (02, 13) on a single detector tag (FRCNN).
    # - This keeps ablations practical while still being challenging (crowded/occlusion-heavy).
    export SEQ_IDS_RAW="${MOT17_SEQ_IDS_RAW:-2 13}"
    export MOT17_DETECTOR_EXTS_RAW="${MOT17_DETECTOR_EXTS_RAW:-FRCNN}"
    echo "[stage1] MOT17 fast protocol: SEQ_IDS_RAW='${SEQ_IDS_RAW}', MOT17_DETECTOR_EXTS_RAW='${MOT17_DETECTOR_EXTS_RAW}'"
  else
    export SEQ_IDS_RAW=""
    export MOT17_DETECTOR_EXTS_RAW=""
  fi
  bash scripts/run_botsort_laplace_matrix.sh "${dataset}" val | tee "${out_root}/run.log"

  echo "[stage1] evaluating ${dataset} base"
  "${PYTHON_BIN}" scripts/eval_botsort_halfval_trackeval.py \
    --dataset "${dataset}" \
    --results-dir "external/BoT-SORT-main/YOLOX_outputs/laplace_${lower}_val_base/track_results" \
    --tracker-name "botsort_${lower}_val_base" \
    --work-dir "${out_root}/base_eval"

  echo "[stage1] evaluating ${dataset} laplace"
  "${PYTHON_BIN}" scripts/eval_botsort_halfval_trackeval.py \
    --dataset "${dataset}" \
    --results-dir "external/BoT-SORT-main/YOLOX_outputs/laplace_${lower}_val_laplace/track_results" \
    --tracker-name "botsort_${lower}_val_laplace" \
    --work-dir "${out_root}/laplace_eval"

  "${PYTHON_BIN}" scripts/collect_trackeval_metrics.py \
    "${out_root}/base_eval/eval/botsort_${lower}_val_base" \
    "${out_root}/laplace_eval/eval/botsort_${lower}_val_laplace" \
    --csv "${out_root}/summary.csv" | tee "${out_root}/summary.txt"

  echo "[stage1] finished ${dataset}; summary at ${out_root}/summary.csv"
}

case "${TARGET}" in
  MOT17|mot17)
    run_one_dataset MOT17
    ;;
  MOT20|mot20)
    run_one_dataset MOT20
    ;;
  all)
    run_one_dataset MOT17
    run_one_dataset MOT20
    ;;
  *)
    echo "Usage: bash scripts/run_botsort_ltra_stage1.sh [MOT17|MOT20|all]" >&2
    exit 2
    ;;
esac
