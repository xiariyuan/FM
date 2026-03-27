#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"
BS_ROOT="${REPO_ROOT}/external/BoT-SORT-main"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"

HACA_CHECKPOINT="${1:?HACA checkpoint (.npz) is required}"
DATASET="${2:-MOT17}"
OUT_DIR="${3:-${REPO_ROOT}/outputs/haca_submit_${DATASET,,}_$(date +%Y%m%d_%H%M%S)}"
DATA_ROOT="${4:-/gemini/code/datasets}"

if [[ ! -f "${HACA_CHECKPOINT}" ]]; then
  echo "Missing HACA checkpoint: ${HACA_CHECKPOINT}" >&2
  exit 2
fi

OUT_DIR="$(realpath -m "${OUT_DIR}")"
mkdir -p "${OUT_DIR}"
RUN_LOG="${OUT_DIR}/run.log"

INTERP_N_MIN="${INTERP_N_MIN:-5}"
INTERP_N_DTI="${INTERP_N_DTI:-20}"
RUN_INTERPOLATION="${RUN_INTERPOLATION:-1}"
SANITIZE_RESULTS="${SANITIZE_RESULTS:-1}"
SANITIZE_PRECISION="${SANITIZE_PRECISION:-2}"
HACA_MODE="${HACA_MODE:-haca_v3}"
HACA_LABEL="${HACA_LABEL:-$(basename "${HACA_CHECKPOINT}" .npz)}"
HACA_DISABLE_SET_ENCODER="${HACA_DISABLE_SET_ENCODER:-0}"
HACA_DISABLE_BACKGROUND="${HACA_DISABLE_BACKGROUND:-0}"
HACA_DELTA_SCALE="${HACA_DELTA_SCALE:-}"
ENABLE_REID="${ENABLE_REID:-1}"
ENABLE_FUSE="${ENABLE_FUSE:-1}"

case "${DATASET}" in
  MOT17)
    EXP_FILE="./yolox/exps/example/mot/yolox_x_mix_det.py"
    CKPT="./pretrained/bytetrack_x_mot17.pth.tar"
    REID_CFG="fast_reid/configs/MOT17/sbs_S50.yml"
    REID_WTS="pretrained/mot17_sbs_S50.pth"
    ZIP_NAME="mot17_haca_${HACA_LABEL}_test_submission_$(date +%Y%m%d_%H%M%S).zip"
    CHECK_CMD=("${PYTHON_BIN}" "${REPO_ROOT}/scripts/check_mot17_submission.py" --zip-path "${OUT_DIR}/${ZIP_NAME}" --profile mot17_test_public_21)
    EXPECTED=(
      MOT17-01-DPM.txt MOT17-01-FRCNN.txt MOT17-01-SDP.txt
      MOT17-03-DPM.txt MOT17-03-FRCNN.txt MOT17-03-SDP.txt
      MOT17-06-DPM.txt MOT17-06-FRCNN.txt MOT17-06-SDP.txt
      MOT17-07-DPM.txt MOT17-07-FRCNN.txt MOT17-07-SDP.txt
      MOT17-08-DPM.txt MOT17-08-FRCNN.txt MOT17-08-SDP.txt
      MOT17-12-DPM.txt MOT17-12-FRCNN.txt MOT17-12-SDP.txt
      MOT17-14-DPM.txt MOT17-14-FRCNN.txt MOT17-14-SDP.txt
    )
    TRACK_ARGS=(
      "${PYTHON_BIN}" -u tools/track.py "${DATA_ROOT}/MOT17"
      --benchmark MOT17
      --eval test
      -f "${EXP_FILE}"
      -c "${CKPT}"
      --experiment-name "haca_submit_mot17_${HACA_LABEL}"
      --laplace-assoc
      --laplace-assoc-mode "${HACA_MODE}"
      --laplace-haca-checkpoint "${HACA_CHECKPOINT}"
      --laplace-primary-only
    )
    ;;
  MOT20)
    EXP_FILE="./yolox/exps/example/mot/yolox_x_mix_mot20_ch.py"
    CKPT="./pretrained/bytetrack_x_mot20.pth.tar"
    REID_CFG="fast_reid/configs/MOT20/sbs_S50.yml"
    REID_WTS="pretrained/mot20_sbs_S50.pth"
    ZIP_NAME="mot20_haca_${HACA_LABEL}_test_submission_$(date +%Y%m%d_%H%M%S).zip"
    CHECK_CMD=("${PYTHON_BIN}" "${REPO_ROOT}/scripts/check_mot20_submission.py" --zip-path "${OUT_DIR}/${ZIP_NAME}" --profile mot20_test_4)
    EXPECTED=(MOT20-04.txt MOT20-06.txt MOT20-07.txt MOT20-08.txt)
    TRACK_ARGS=(
      "${PYTHON_BIN}" -u tools/track.py "${DATA_ROOT}/MOT20"
      --benchmark MOT20
      --eval test
      -f "${EXP_FILE}"
      -c "${CKPT}"
      --experiment-name "haca_submit_mot20_${HACA_LABEL}"
      --cmc-method file
      --laplace-assoc
      --laplace-assoc-mode "${HACA_MODE}"
      --laplace-haca-checkpoint "${HACA_CHECKPOINT}"
      --laplace-primary-only
    )
    ;;
  *)
    echo "Unsupported dataset: ${DATASET}" >&2
    exit 2
    ;;
esac

RESULT_DIR="${BS_ROOT}/YOLOX_outputs/haca_submit_${DATASET,,}_${HACA_LABEL}/track_results"
SANITIZED_DIR="${OUT_DIR}/sanitized_results"

{
  echo "dataset=${DATASET}"
  echo "haca_checkpoint=${HACA_CHECKPOINT}"
  echo "haca_mode=${HACA_MODE}"
  echo "out_dir=${OUT_DIR}"
  echo "data_root=${DATA_ROOT}"
  echo "run_interpolation=${RUN_INTERPOLATION}"
  echo "sanitize_results=${SANITIZE_RESULTS}"
  echo "haca_disable_set_encoder=${HACA_DISABLE_SET_ENCODER}"
  echo "haca_disable_background=${HACA_DISABLE_BACKGROUND}"
  echo "haca_delta_scale=${HACA_DELTA_SCALE}"
  echo "start_time=$(date '+%F %T %z')"
} > "${OUT_DIR}/meta.txt"

(
  cd "${BS_ROOT}"
  cmd=("${TRACK_ARGS[@]}")
  if [[ "${ENABLE_REID}" == "1" ]]; then
    cmd+=(--with-reid --fast-reid-config "${REID_CFG}" --fast-reid-weights "${REID_WTS}")
  fi
  if [[ "${ENABLE_FUSE}" == "1" ]]; then
    cmd+=(--fuse)
  fi
  if [[ "${HACA_DISABLE_SET_ENCODER}" == "1" ]]; then
    cmd+=(--laplace-haca-no-set-encoder)
  fi
  if [[ "${HACA_DISABLE_BACKGROUND}" == "1" ]]; then
    cmd+=(--laplace-haca-no-background)
  fi
  if [[ -n "${HACA_DELTA_SCALE}" ]]; then
    cmd+=(--laplace-haca-delta-scale "${HACA_DELTA_SCALE}")
  fi
  echo "[track] ${cmd[*]}" | tee -a "${RUN_LOG}"
  "${cmd[@]}" 2>&1 | tee -a "${RUN_LOG}"
)

if [[ "${RUN_INTERPOLATION}" == "1" ]]; then
  (
    cd "${BS_ROOT}"
    echo "[post] interpolation n_min=${INTERP_N_MIN} n_dti=${INTERP_N_DTI}" | tee -a "${RUN_LOG}"
    "${PYTHON_BIN}" tools/interpolation.py \
      --txt_path "${RESULT_DIR}" \
      --n_min "${INTERP_N_MIN}" \
      --n_dti "${INTERP_N_DTI}" \
      2>&1 | tee -a "${RUN_LOG}"
  )
fi

PACKAGE_DIR="${RESULT_DIR}"
if [[ "${SANITIZE_RESULTS}" == "1" ]]; then
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/sanitize_mot_submission.py" \
    --input-dir "${RESULT_DIR}" \
    --output-dir "${SANITIZED_DIR}" \
    --data-root "${DATA_ROOT}" \
    --benchmark "${DATASET}" \
    --precision "${SANITIZE_PRECISION}" \
    2>&1 | tee -a "${RUN_LOG}"
  PACKAGE_DIR="${SANITIZED_DIR}"
fi

for name in "${EXPECTED[@]}"; do
  if [[ ! -f "${PACKAGE_DIR}/${name}" ]]; then
    echo "Missing expected result file: ${PACKAGE_DIR}/${name}" >&2
    exit 1
  fi
done

(
  cd "${PACKAGE_DIR}"
  zip -q "${OUT_DIR}/${ZIP_NAME}" "${EXPECTED[@]}"
)

printf '%s\n' "${OUT_DIR}/${ZIP_NAME}" > "${OUT_DIR}/latest_zip.txt"
"${CHECK_CMD[@]}" | tee "${OUT_DIR}/precheck.log"

echo "[OK] package_dir=${PACKAGE_DIR}"
echo "[OK] zip=${OUT_DIR}/${ZIP_NAME}"
