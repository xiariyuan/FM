#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/gemini/code/FMtrack-main/FM-Track"
BOT_ROOT="${REPO_ROOT}/external/BoT-SORT-main"
DATA_ROOT="${DATA_ROOT:-/gemini/code/datasets}"
DATASET="${1:-MOT17}"
SPLIT="${2:-val}"

cd "${BOT_ROOT}"

SEQ_IDS_RAW="${SEQ_IDS_RAW:-}"
MOT17_DETECTOR_EXTS_RAW="${MOT17_DETECTOR_EXTS_RAW:-}"

case "${DATASET}" in
  MOT17)
    REID_CFG="fast_reid/configs/MOT17/sbs_S50.yml"
    REID_WTS="pretrained/mot17_sbs_S50.pth"
    EXP_FILE="./yolox/exps/example/mot/yolox_x_mix_det.py"
    DET_CKPT="./pretrained/bytetrack_x_mot17.pth.tar"
    CMC_ARGS=()
    ;;
  MOT20)
    REID_CFG="fast_reid/configs/MOT20/sbs_S50.yml"
    REID_WTS="pretrained/mot20_sbs_S50.pth"
    EXP_FILE="./yolox/exps/example/mot/yolox_x_mix_mot20_ch.py"
    DET_CKPT="./pretrained/bytetrack_x_mot20.pth.tar"
    # Use the same CMC mode as test submissions (file-based GMC).
    # MOT20 val previously used ORB due to an ablation-flag bug which is now fixed.
    CMC_ARGS=(--cmc-method file)
    ;;
  *)
    echo "Unsupported dataset: ${DATASET}" >&2
    exit 2
    ;;
esac

COMMON_ARGS=(
  "${DATA_ROOT}/${DATASET}"
  --benchmark "${DATASET}"
  --eval "${SPLIT}"
  -f "${EXP_FILE}"
  -c "${DET_CKPT}"
  --with-reid
  --fast-reid-config "${REID_CFG}"
  --fast-reid-weights "${REID_WTS}"
)

if [[ -n "${SEQ_IDS_RAW}" ]]; then
  # shellcheck disable=SC2206
  SEQ_IDS_ARR=(${SEQ_IDS_RAW})
  COMMON_ARGS+=(--seq-ids "${SEQ_IDS_ARR[@]}")
fi

if [[ "${DATASET}" == "MOT17" && -n "${MOT17_DETECTOR_EXTS_RAW}" ]]; then
  # shellcheck disable=SC2206
  MOT17_EXTS_ARR=(${MOT17_DETECTOR_EXTS_RAW})
  COMMON_ARGS+=(--mot17-detector-exts "${MOT17_EXTS_ARR[@]}")
fi

python tools/track.py "${COMMON_ARGS[@]}" \
  "${CMC_ARGS[@]}" \
  --experiment-name "laplace_${DATASET,,}_${SPLIT}_base"

python tools/track.py "${COMMON_ARGS[@]}" \
  --experiment-name "laplace_${DATASET,,}_${SPLIT}_laplace" \
  "${CMC_ARGS[@]}" \
  --laplace-assoc \
  --laplace-primary-only \
  --laplace-weight 0.35 \
  --laplace-decay-scales 1 2 4 \
  --laplace-min-history 3

echo "[done] inspect external/BoT-SORT-main/YOLOX_outputs for base and laplace results"
