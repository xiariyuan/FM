#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/gemini/code/FMtrack-main/FM-Track"
BOT_ROOT="${REPO_ROOT}/external/BoT-SORT-main"
DATA_ROOT="${DATA_ROOT:-/gemini/code/datasets}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
DATASET="MOT20"
LOWER="mot20"

cd "${REPO_ROOT}"

REID_CFG="fast_reid/configs/MOT20/sbs_S50.yml"
REID_WTS="pretrained/mot20_sbs_S50.pth"
EXP_FILE="./yolox/exps/example/mot/yolox_x_mix_mot20_ch.py"
DET_CKPT="./pretrained/bytetrack_x_mot20.pth.tar"
# Keep consistent with MOT20 test submissions (GMC from file).
CMC_ARGS=(--cmc-method file)
EXPECTED_FILES=(MOT20-01 MOT20-02 MOT20-03 MOT20-05)

OUT_ROOT="outputs/botsort_ltra_stage3/MOT20"
mkdir -p "${OUT_ROOT}"

run_variant() {
  local label="$1"
  shift
  local exp_name="laplace_${LOWER}_val_${label}"
  local result_dir="${BOT_ROOT}/YOLOX_outputs/${exp_name}/track_results"
  local missing_files=()
  local seq_args=()

  if [[ -d "${result_dir}" ]]; then
    for name in "${EXPECTED_FILES[@]}"; do
      if [[ ! -f "${result_dir}/${name}.txt" ]]; then
        missing_files+=("${name}")
      fi
    done
  else
    missing_files=("${EXPECTED_FILES[@]}")
  fi

  if [[ "${#missing_files[@]}" -eq 0 ]]; then
    echo "[stage3] skip ${label}: all files present"
    return 0
  fi

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

  echo "[stage3] running ${label} seq-ids: ${seq_args[*]}"
  (
    cd "${BOT_ROOT}"
    cmd=(
      python tools/track.py
      "${DATA_ROOT}/${DATASET}"
      --benchmark "${DATASET}"
      --eval val
      --seq-ids "${seq_args[@]}"
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
  local work_dir="${OUT_ROOT}/${label}_eval"
  echo "[stage3] evaluating ${label}"
  "${PYTHON_BIN}" scripts/eval_botsort_halfval_trackeval.py \
    --dataset "${DATASET}" \
    --results-dir "${results_dir}" \
    --tracker-name "${tracker_name}" \
    --work-dir "${work_dir}"
}

BASE_EVAL="outputs/botsort_ltra_stage2/MOT20/base_eval/eval/botsort_mot20_val_base"
MEAN_EVAL="outputs/botsort_ltra_stage2/MOT20/meanhist_eval/eval/botsort_mot20_val_meanhist"
FULL_EVAL="outputs/botsort_ltra_stage2/MOT20/full_eval/eval/botsort_mot20_val_full"

run_variant nodetscore \
  --laplace-assoc \
  --laplace-weight 0.35 \
  --laplace-decay-scales 1 2 4 \
  --laplace-min-history 3 \
  --laplace-proto-mode multi \
  --laplace-no-det-score

run_variant rel075 \
  --laplace-assoc \
  --laplace-weight 0.35 \
  --laplace-decay-scales 1 2 4 \
  --laplace-min-history 3 \
  --laplace-proto-mode multi \
  --laplace-reliability-scale 0.75

run_variant rel050 \
  --laplace-assoc \
  --laplace-weight 0.35 \
  --laplace-decay-scales 1 2 4 \
  --laplace-min-history 3 \
  --laplace-proto-mode multi \
  --laplace-reliability-scale 0.5

run_variant nodetscore_rel075 \
  --laplace-assoc \
  --laplace-weight 0.35 \
  --laplace-decay-scales 1 2 4 \
  --laplace-min-history 3 \
  --laplace-proto-mode multi \
  --laplace-no-det-score \
  --laplace-reliability-scale 0.75

run_variant weight025 \
  --laplace-assoc \
  --laplace-weight 0.25 \
  --laplace-decay-scales 1 2 4 \
  --laplace-min-history 3 \
  --laplace-proto-mode multi

run_variant slowhist \
  --laplace-assoc \
  --laplace-weight 0.35 \
  --laplace-decay-scales 2 4 8 \
  --laplace-min-history 5 \
  --laplace-proto-mode multi

eval_variant nodetscore "${BOT_ROOT}/YOLOX_outputs/laplace_${LOWER}_val_nodetscore/track_results"
eval_variant rel075 "${BOT_ROOT}/YOLOX_outputs/laplace_${LOWER}_val_rel075/track_results"
eval_variant rel050 "${BOT_ROOT}/YOLOX_outputs/laplace_${LOWER}_val_rel050/track_results"
eval_variant nodetscore_rel075 "${BOT_ROOT}/YOLOX_outputs/laplace_${LOWER}_val_nodetscore_rel075/track_results"
eval_variant weight025 "${BOT_ROOT}/YOLOX_outputs/laplace_${LOWER}_val_weight025/track_results"
eval_variant slowhist "${BOT_ROOT}/YOLOX_outputs/laplace_${LOWER}_val_slowhist/track_results"

"${PYTHON_BIN}" scripts/collect_trackeval_metrics.py \
  "${BASE_EVAL}" \
  "${MEAN_EVAL}" \
  "${FULL_EVAL}" \
  "${OUT_ROOT}/nodetscore_eval/eval/botsort_mot20_val_nodetscore" \
  "${OUT_ROOT}/rel075_eval/eval/botsort_mot20_val_rel075" \
  "${OUT_ROOT}/rel050_eval/eval/botsort_mot20_val_rel050" \
  "${OUT_ROOT}/nodetscore_rel075_eval/eval/botsort_mot20_val_nodetscore_rel075" \
  "${OUT_ROOT}/weight025_eval/eval/botsort_mot20_val_weight025" \
  "${OUT_ROOT}/slowhist_eval/eval/botsort_mot20_val_slowhist" \
  --csv "${OUT_ROOT}/summary.csv" | tee "${OUT_ROOT}/summary.txt"

echo "[stage3] finished MOT20; summary at ${OUT_ROOT}/summary.csv"
