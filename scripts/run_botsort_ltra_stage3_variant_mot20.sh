#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/gemini/code/FMtrack-main/FM-Track"
BOT_ROOT="${REPO_ROOT}/external/BoT-SORT-main"
DATA_ROOT="${DATA_ROOT:-/gemini/code/datasets}"
VARIANT="${1:?usage: run_botsort_ltra_stage3_variant_mot20.sh <variant>}"

cd "${REPO_ROOT}"

REID_CFG="fast_reid/configs/MOT20/sbs_S50.yml"
REID_WTS="pretrained/mot20_sbs_S50.pth"
EXP_FILE="./yolox/exps/example/mot/yolox_x_mix_mot20_ch.py"
DET_CKPT="./pretrained/bytetrack_x_mot20.pth.tar"
EXPECTED_FILES=(MOT20-01 MOT20-02 MOT20-03 MOT20-05)

ARGS=()
case "${VARIANT}" in
  nodetscore)
    ARGS=(--laplace-assoc --laplace-weight 0.35 --laplace-decay-scales 1 2 4 --laplace-min-history 3 --laplace-proto-mode multi --laplace-no-det-score)
    ;;
  rel075)
    ARGS=(--laplace-assoc --laplace-weight 0.35 --laplace-decay-scales 1 2 4 --laplace-min-history 3 --laplace-proto-mode multi --laplace-reliability-scale 0.75)
    ;;
  rel050)
    ARGS=(--laplace-assoc --laplace-weight 0.35 --laplace-decay-scales 1 2 4 --laplace-min-history 3 --laplace-proto-mode multi --laplace-reliability-scale 0.5)
    ;;
  nodetscore_rel075)
    ARGS=(--laplace-assoc --laplace-weight 0.35 --laplace-decay-scales 1 2 4 --laplace-min-history 3 --laplace-proto-mode multi --laplace-no-det-score --laplace-reliability-scale 0.75)
    ;;
  weight025)
    ARGS=(--laplace-assoc --laplace-weight 0.25 --laplace-decay-scales 1 2 4 --laplace-min-history 3 --laplace-proto-mode multi)
    ;;
  slowhist)
    ARGS=(--laplace-assoc --laplace-weight 0.35 --laplace-decay-scales 2 4 8 --laplace-min-history 5 --laplace-proto-mode multi)
    ;;
  *)
    echo "Unsupported variant: ${VARIANT}" >&2
    exit 2
    ;;
esac

RESULT_DIR="${BOT_ROOT}/YOLOX_outputs/laplace_mot20_val_${VARIANT}/track_results"
missing_files=()
seq_args=()

if [[ -d "${RESULT_DIR}" ]]; then
  for name in "${EXPECTED_FILES[@]}"; do
    if [[ ! -f "${RESULT_DIR}/${name}.txt" ]]; then
      missing_files+=("${name}")
    fi
  done
else
  missing_files=("${EXPECTED_FILES[@]}")
fi

if [[ "${#missing_files[@]}" -eq 0 ]]; then
  echo "[variant] ${VARIANT} already complete"
  exit 0
fi

declare -A need_seq_ids=()
for name in "${missing_files[@]}"; do
  seq_tail="${name#MOT20-}"
  seq_num="${seq_tail%%-*}"
  seq_num="${seq_num#0}"
  need_seq_ids["${seq_num}"]=1
done
for seq_id in "${!need_seq_ids[@]}"; do
  seq_args+=("${seq_id}")
done

echo "[variant] ${VARIANT} seq-ids: ${seq_args[*]}"

cd "${BOT_ROOT}"
cmd=(
  python tools/track.py
  "${DATA_ROOT}/MOT20"
  --benchmark MOT20
  --eval val
  --seq-ids "${seq_args[@]}"
  -f "${EXP_FILE}"
  -c "${DET_CKPT}"
  --with-reid
  --fast-reid-config "${REID_CFG}"
  --fast-reid-weights "${REID_WTS}"
  --cmc-method orb
  --experiment-name "laplace_mot20_val_${VARIANT}"
  "${ARGS[@]}"
)

env PYTHONUNBUFFERED=1 "${cmd[@]}"
