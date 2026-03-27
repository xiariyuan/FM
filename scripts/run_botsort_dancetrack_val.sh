#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/gemini/code/FMtrack-main/FM-Track"
BOT_ROOT="${REPO_ROOT}/external/BoT-SORT-main"
DATA_ROOT="${DATA_ROOT:-/tmp/DanceTrack_val}"
SPLIT="${SPLIT:-val}"
VARIANT="${1:-base}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
EXTRA_ARGS=("${@:2}")

cd "${BOT_ROOT}"

REID_CFG="fast_reid/configs/MOT17/sbs_S50.yml"
REID_WTS="pretrained/mot17_sbs_S50.pth"
EXP_FILE="./yolox/exps/example/mot/yolox_x_mix_det.py"
DET_CKPT="./pretrained/bytetrack_x_mot17.pth.tar"

case "${VARIANT}" in
  base)
    EXP_NAME="botsort_dancetrack_${SPLIT}_base"
    TRACK_ARGS=()
    ;;
  ltra|full)
    EXP_NAME="botsort_dancetrack_${SPLIT}_ltra"
    TRACK_ARGS=(
      --laplace-assoc
      --laplace-weight 0.35
      --laplace-decay-scales 1 2 4
      --laplace-min-history 3
    )
    ;;
  meanrel)
    EXP_NAME="botsort_dancetrack_${SPLIT}_meanrel"
    TRACK_ARGS=(
      --laplace-assoc
      --laplace-proto-mode mean
      --laplace-weight 0.35
      --laplace-decay-scales 1 2 4
      --laplace-min-history 3
    )
    ;;
  mean)
    EXP_NAME="botsort_dancetrack_${SPLIT}_mean"
    TRACK_ARGS=(
      --laplace-assoc
      --laplace-proto-mode mean
      --laplace-weight 0.35
      --laplace-decay-scales 1 2 4
      --laplace-min-history 3
      --laplace-no-reliability
    )
    ;;
  multinorel)
    EXP_NAME="botsort_dancetrack_${SPLIT}_multinorel"
    TRACK_ARGS=(
      --laplace-assoc
      --laplace-weight 0.35
      --laplace-decay-scales 1 2 4
      --laplace-min-history 3
      --laplace-no-reliability
    )
    ;;
  single)
    EXP_NAME="botsort_dancetrack_${SPLIT}_single"
    TRACK_ARGS=(
      --laplace-assoc
      --laplace-proto-mode single
      --laplace-weight 0.35
      --laplace-decay-scales 1 2 4
      --laplace-min-history 3
      --laplace-no-reliability
    )
    ;;
  rel075)
    EXP_NAME="botsort_dancetrack_${SPLIT}_rel075"
    TRACK_ARGS=(
      --laplace-assoc
      --laplace-weight 0.35
      --laplace-decay-scales 1 2 4
      --laplace-min-history 3
      --laplace-reliability-scale 0.75
    )
    ;;
  rel050)
    EXP_NAME="botsort_dancetrack_${SPLIT}_rel050"
    TRACK_ARGS=(
      --laplace-assoc
      --laplace-weight 0.35
      --laplace-decay-scales 1 2 4
      --laplace-min-history 3
      --laplace-reliability-scale 0.50
    )
    ;;
  nodetscore)
    EXP_NAME="botsort_dancetrack_${SPLIT}_nodetscore"
    TRACK_ARGS=(
      --laplace-assoc
      --laplace-weight 0.35
      --laplace-decay-scales 1 2 4
      --laplace-min-history 3
      --laplace-no-det-score
    )
    ;;
  singlerel)
    EXP_NAME="botsort_dancetrack_${SPLIT}_singlerel"
    TRACK_ARGS=(
      --laplace-assoc
      --laplace-proto-mode single
      --laplace-weight 0.35
      --laplace-decay-scales 1 2 4
      --laplace-min-history 3
    )
    ;;
  weight025)
    EXP_NAME="botsort_dancetrack_${SPLIT}_weight025"
    TRACK_ARGS=(
      --laplace-assoc
      --laplace-weight 0.25
      --laplace-decay-scales 1 2 4
      --laplace-min-history 3
    )
    ;;
  weight050)
    EXP_NAME="botsort_dancetrack_${SPLIT}_weight050"
    TRACK_ARGS=(
      --laplace-assoc
      --laplace-weight 0.50
      --laplace-decay-scales 1 2 4
      --laplace-min-history 3
    )
    ;;
  *)
    echo "Unsupported variant: ${VARIANT}" >&2
    exit 2
    ;;
esac

python tools/track.py \
  "${DATA_ROOT}" \
  --benchmark DanceTrack \
  --eval "${SPLIT}" \
  -f "${EXP_FILE}" \
  -c "${DET_CKPT}" \
  --with-reid \
  --fast-reid-config "${REID_CFG}" \
  --fast-reid-weights "${REID_WTS}" \
  --cmc-method none \
  --experiment-name "${EXP_NAME}" \
  "${TRACK_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"

echo "[done] results: ${BOT_ROOT}/YOLOX_outputs/${EXP_NAME}/track_results"
