#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/gemini/code/FMtrack-main/FM-Track"
BOT_ROOT="${REPO_ROOT}/external/BoT-SORT-main"
DATA_ROOT="${DATA_ROOT:-/gemini/code/datasets}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
DATASET="${1:?usage: run_botsort_ltra_variant_eval.sh <MOT17|MOT20> <variant> [track args...]}"
VARIANT="${2:?usage: run_botsort_ltra_variant_eval.sh <MOT17|MOT20> <variant> [track args...]}"
shift 2
TRACK_ARGS=("$@")

cd "${REPO_ROOT}"

case "${DATASET}" in
  MOT17)
    LOWER="mot17"
    REID_CFG="fast_reid/configs/MOT17/sbs_S50.yml"
    REID_WTS="pretrained/mot17_sbs_S50.pth"
    EXP_FILE="./yolox/exps/example/mot/yolox_x_mix_det.py"
    DET_CKPT="./pretrained/bytetrack_x_mot17.pth.tar"
    EXPECTED_TXT=21
    CMC_ARGS=()
    EXPECTED_FILES=(
      MOT17-02-FRCNN MOT17-02-DPM MOT17-02-SDP
      MOT17-04-FRCNN MOT17-04-DPM MOT17-04-SDP
      MOT17-05-FRCNN MOT17-05-DPM MOT17-05-SDP
      MOT17-09-FRCNN MOT17-09-DPM MOT17-09-SDP
      MOT17-10-FRCNN MOT17-10-DPM MOT17-10-SDP
      MOT17-11-FRCNN MOT17-11-DPM MOT17-11-SDP
      MOT17-13-FRCNN MOT17-13-DPM MOT17-13-SDP
    )
    ;;
  MOT20)
    LOWER="mot20"
    REID_CFG="fast_reid/configs/MOT20/sbs_S50.yml"
    REID_WTS="pretrained/mot20_sbs_S50.pth"
    EXP_FILE="./yolox/exps/example/mot/yolox_x_mix_mot20_ch.py"
    DET_CKPT="./pretrained/bytetrack_x_mot20.pth.tar"
    EXPECTED_TXT=4
    CMC_ARGS=(--cmc-method orb)
    EXPECTED_FILES=(MOT20-01 MOT20-02 MOT20-03 MOT20-05)
    ;;
  *)
    echo "Unsupported dataset: ${DATASET}" >&2
    exit 2
    ;;
esac

OUT_ROOT="outputs/botsort_ltra_extra/${DATASET}"
mkdir -p "${OUT_ROOT}"

EXP_NAME="laplace_${LOWER}_val_${VARIANT}"
RESULT_DIR="${BOT_ROOT}/YOLOX_outputs/${EXP_NAME}/track_results"

count=0
missing_files=()
seq_args=()
if [[ -d "${RESULT_DIR}" ]]; then
  count=$(find "${RESULT_DIR}" -maxdepth 1 -name '*.txt' | wc -l)
fi
if [[ "${count}" -lt "${EXPECTED_TXT}" ]]; then
  if [[ -d "${RESULT_DIR}" ]]; then
    for name in "${EXPECTED_FILES[@]}"; do
      if [[ ! -f "${RESULT_DIR}/${name}.txt" ]]; then
        missing_files+=("${name}")
      fi
    done
  else
    missing_files=("${EXPECTED_FILES[@]}")
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
    cmd+=(
      -f "${EXP_FILE}"
      -c "${DET_CKPT}"
      --with-reid
      --fast-reid-config "${REID_CFG}"
      --fast-reid-weights "${REID_WTS}"
      "${CMC_ARGS[@]}"
      --experiment-name "${EXP_NAME}"
      "${TRACK_ARGS[@]}"
    )
    echo "[variant] running ${DATASET} ${VARIANT} seq-ids: ${seq_args[*]}"
    env PYTHONUNBUFFERED=1 "${cmd[@]}"
  )
else
  echo "[variant] skip tracking ${DATASET} ${VARIANT}: found ${count}/${EXPECTED_TXT}"
fi

"${PYTHON_BIN}" scripts/eval_botsort_halfval_trackeval.py \
  --dataset "${DATASET}" \
  --results-dir "${RESULT_DIR}" \
  --tracker-name "botsort_${LOWER}_val_${VARIANT}" \
  --work-dir "${OUT_ROOT}/${VARIANT}_eval"

echo "[variant] done ${DATASET} ${VARIANT}"
