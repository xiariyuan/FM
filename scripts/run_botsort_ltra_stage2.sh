#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/gemini/code/FMtrack-main/FM-Track"
BOT_ROOT="${REPO_ROOT}/external/BoT-SORT-main"
DATA_ROOT="${DATA_ROOT:-/gemini/code/datasets}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
DATASET="${1:-MOT20}"
LAPLACE_CALIBRATOR="${LAPLACE_CALIBRATOR:-}"

cd "${REPO_ROOT}"

case "${DATASET}" in
  MOT17)
    LOWER="mot17"
    REID_CFG="fast_reid/configs/MOT17/sbs_S50.yml"
    REID_WTS="pretrained/mot17_sbs_S50.pth"
    EXP_FILE="./yolox/exps/example/mot/yolox_x_mix_det.py"
    DET_CKPT="./pretrained/bytetrack_x_mot17.pth.tar"
    CMC_ARGS=()
    if [[ "${MOT17_FULL:-0}" == "1" ]]; then
      EXPECTED_TXT=21
      EXPECTED_FILES=(
        MOT17-02-FRCNN MOT17-02-DPM MOT17-02-SDP
        MOT17-04-FRCNN MOT17-04-DPM MOT17-04-SDP
        MOT17-05-FRCNN MOT17-05-DPM MOT17-05-SDP
        MOT17-09-FRCNN MOT17-09-DPM MOT17-09-SDP
        MOT17-10-FRCNN MOT17-10-DPM MOT17-10-SDP
        MOT17-11-FRCNN MOT17-11-DPM MOT17-11-SDP
        MOT17-13-FRCNN MOT17-13-DPM MOT17-13-SDP
      )
    else
      # Fast validation protocol: only FRCNN on 02/13.
      EXPECTED_TXT=2
      EXPECTED_FILES=(MOT17-02-FRCNN MOT17-13-FRCNN)
    fi
    ;;
  MOT20)
    LOWER="mot20"
    REID_CFG="fast_reid/configs/MOT20/sbs_S50.yml"
    REID_WTS="pretrained/mot20_sbs_S50.pth"
    EXP_FILE="./yolox/exps/example/mot/yolox_x_mix_mot20_ch.py"
    DET_CKPT="./pretrained/bytetrack_x_mot20.pth.tar"
    EXPECTED_TXT=4
    # Keep consistent with MOT20 test submissions (GMC from file).
    CMC_ARGS=(--cmc-method file)
    EXPECTED_FILES=(MOT20-01 MOT20-02 MOT20-03 MOT20-05)
    ;;
  *)
    echo "Unsupported dataset: ${DATASET}" >&2
    exit 2
    ;;
esac

STAGE_OUT="outputs/botsort_ltra_stage2/${DATASET}"
mkdir -p "${STAGE_OUT}"

if [[ -n "${LAPLACE_CALIBRATOR}" && ! -f "${LAPLACE_CALIBRATOR}" ]]; then
  echo "Missing LAPLACE_CALIBRATOR: ${LAPLACE_CALIBRATOR}" >&2
  exit 2
fi

run_variant() {
  local variant="$1"
  shift
  local exp_name="laplace_${LOWER}_val_${variant}"
  local result_dir="${BOT_ROOT}/YOLOX_outputs/${exp_name}/track_results"
  local count=0
  local missing_files=()
  local seq_args=()
  if [[ -d "${result_dir}" ]]; then
    count=$(find "${result_dir}" -maxdepth 1 -name '*.txt' | wc -l)
  fi

  if [[ "${count}" -ge "${EXPECTED_TXT}" ]]; then
    echo "[stage2] skip ${variant}: found ${count}/${EXPECTED_TXT} result files"
    return 0
  fi

  if [[ -d "${result_dir}" ]]; then
    for name in "${EXPECTED_FILES[@]}"; do
      if [[ ! -f "${result_dir}/${name}.txt" ]]; then
        missing_files+=("${name}")
      fi
    done
  else
    missing_files=("${EXPECTED_FILES[@]}")
  fi

  if [[ "${#missing_files[@]}" -gt 0 ]]; then
    declare -A need_seq_ids=()
    for name in "${missing_files[@]}"; do
      seq_tail="${name#${DATASET}-}"
      seq_num="${seq_tail%%-*}"
      seq_num="${seq_num#0}"
      need_seq_ids["${seq_num}"]=1
    done
    for seq_id in "${!need_seq_ids[@]}"; do
      seq_args+=("${seq_id}")
    done
  fi

  echo "[stage2] running ${DATASET} ${variant}"
  if [[ "${#seq_args[@]}" -gt 0 ]]; then
    echo "[stage2] resume seq-ids: ${seq_args[*]}"
  fi
  (
    cd "${BOT_ROOT}"
    cmd=(
      python tools/track.py
      "${DATA_ROOT}/${DATASET}"
      --benchmark "${DATASET}"
      --eval val
    )
    if [[ "${#seq_args[@]}" -gt 0 ]]; then
      cmd+=(--seq-ids "${seq_args[@]}")
    fi
    if [[ "${DATASET}" == "MOT17" && "${MOT17_FULL:-0}" != "1" ]]; then
      cmd+=(--mot17-detector-exts FRCNN)
    fi
    cmd+=(
      -f "${EXP_FILE}"
      -c "${DET_CKPT}"
      --with-reid
      --fast-reid-config "${REID_CFG}"
      --fast-reid-weights "${REID_WTS}"
      "${CMC_ARGS[@]}"
      --experiment-name "${exp_name}"
      "$@"
    )
    env PYTHONUNBUFFERED=1 "${cmd[@]}"
  )
}

eval_variant() {
  local label="$1"
  local results_dir="$2"
  local tracker_name="botsort_${LOWER}_val_${label}"
  local work_dir="${STAGE_OUT}/${label}_eval"

  echo "[stage2] evaluating ${label}"
  "${PYTHON_BIN}" scripts/eval_botsort_halfval_trackeval.py \
    --dataset "${DATASET}" \
    --results-dir "${results_dir}" \
    --tracker-name "${tracker_name}" \
    --work-dir "${work_dir}"
}

BASE_RESULTS="${BOT_ROOT}/YOLOX_outputs/laplace_${LOWER}_val_base/track_results"
FULL_RESULTS="${BOT_ROOT}/YOLOX_outputs/laplace_${LOWER}_val_laplace/track_results"
MEAN_RESULTS="${BOT_ROOT}/YOLOX_outputs/laplace_${LOWER}_val_meanhist/track_results"
MEAN_REL_RESULTS="${BOT_ROOT}/YOLOX_outputs/laplace_${LOWER}_val_meanrel/track_results"
SINGLE_RESULTS="${BOT_ROOT}/YOLOX_outputs/laplace_${LOWER}_val_single/track_results"
MULTI_NR_RESULTS="${BOT_ROOT}/YOLOX_outputs/laplace_${LOWER}_val_multinorel/track_results"
LEARNED_RESULTS="${BOT_ROOT}/YOLOX_outputs/laplace_${LOWER}_val_learned/track_results"

run_variant meanhist \
  --laplace-assoc \
  --laplace-primary-only \
  --laplace-weight 0.35 \
  --laplace-decay-scales 1 2 4 \
  --laplace-min-history 3 \
  --laplace-proto-mode mean \
  --laplace-no-reliability

run_variant meanrel \
  --laplace-assoc \
  --laplace-primary-only \
  --laplace-weight 0.35 \
  --laplace-decay-scales 1 2 4 \
  --laplace-min-history 3 \
  --laplace-proto-mode mean

run_variant single \
  --laplace-assoc \
  --laplace-primary-only \
  --laplace-weight 0.35 \
  --laplace-decay-scales 2 \
  --laplace-min-history 3 \
  --laplace-proto-mode single \
  --laplace-no-reliability

run_variant multinorel \
  --laplace-assoc \
  --laplace-primary-only \
  --laplace-weight 0.35 \
  --laplace-decay-scales 1 2 4 \
  --laplace-min-history 3 \
  --laplace-proto-mode multi \
  --laplace-no-reliability

if [[ -n "${LAPLACE_CALIBRATOR}" ]]; then
  run_variant learned \
    --laplace-assoc \
    --laplace-primary-only \
    --laplace-decay-scales 1 2 4 \
    --laplace-min-history 3 \
    --laplace-proto-mode multi \
    --laplace-calibrator "${LAPLACE_CALIBRATOR}"
fi

eval_variant base "${BASE_RESULTS}"
eval_variant meanhist "${MEAN_RESULTS}"
eval_variant meanrel "${MEAN_REL_RESULTS}"
eval_variant single "${SINGLE_RESULTS}"
eval_variant multinorel "${MULTI_NR_RESULTS}"
eval_variant full "${FULL_RESULTS}"
if [[ -n "${LAPLACE_CALIBRATOR}" ]]; then
  eval_variant learned "${LEARNED_RESULTS}"
fi

SUMMARY_DIRS=(
  "${STAGE_OUT}/base_eval/eval/botsort_${LOWER}_val_base"
  "${STAGE_OUT}/meanhist_eval/eval/botsort_${LOWER}_val_meanhist"
  "${STAGE_OUT}/meanrel_eval/eval/botsort_${LOWER}_val_meanrel"
  "${STAGE_OUT}/single_eval/eval/botsort_${LOWER}_val_single"
  "${STAGE_OUT}/multinorel_eval/eval/botsort_${LOWER}_val_multinorel"
  "${STAGE_OUT}/full_eval/eval/botsort_${LOWER}_val_full"
)
if [[ -n "${LAPLACE_CALIBRATOR}" ]]; then
  SUMMARY_DIRS+=("${STAGE_OUT}/learned_eval/eval/botsort_${LOWER}_val_learned")
fi

"${PYTHON_BIN}" scripts/collect_trackeval_metrics.py \
  "${SUMMARY_DIRS[@]}" \
  --csv "${STAGE_OUT}/summary.csv" | tee "${STAGE_OUT}/summary.txt"

echo "[stage2] finished ${DATASET}; summary at ${STAGE_OUT}/summary.csv"
