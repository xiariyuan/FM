#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"
BS_ROOT="${REPO_ROOT}/external/BoT-SORT-main"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
MANIFEST_TOOL="${REPO_ROOT}/scripts/update_run_manifest.py"

FCAA_CHECKPOINT="${1:?fcaa checkpoint (.pt) is required}"
DATASET="${2:-MOT17}"
OUT_DIR="${3:-${REPO_ROOT}/outputs/fcaa_${DATASET,,}_submit_$(date +%Y%m%d_%H%M%S)}"
DATA_ROOT="${4:-/gemini/code/datasets}"

if [[ "${DATASET}" != "MOT17" ]]; then
  echo "Unsupported DATASET=${DATASET}; only MOT17 is supported here" >&2
  exit 2
fi
if [[ ! -f "${FCAA_CHECKPOINT}" ]]; then
  echo "Missing FCAA checkpoint: ${FCAA_CHECKPOINT}" >&2
  exit 2
fi

OUT_DIR="$(realpath -m "${OUT_DIR}")"
FCAA_CHECKPOINT="$(realpath -m "${FCAA_CHECKPOINT}")"
mkdir -p "${OUT_DIR}"
RUN_LOG="${OUT_DIR}/run.log"

RUN_INTERPOLATION="${RUN_INTERPOLATION:-1}"
INTERP_N_MIN="${INTERP_N_MIN:-5}"
INTERP_N_DTI="${INTERP_N_DTI:-20}"
SANITIZE_RESULTS="${SANITIZE_RESULTS:-1}"
SANITIZE_PRECISION="${SANITIZE_PRECISION:-2}"
FCAA_LABEL="${FCAA_LABEL:-freq}"
FCAA_PRIVATE_DET="${FCAA_PRIVATE_DET:-0}"
FCAA_PRIMARY_EXT="${FCAA_PRIMARY_EXT:-FRCNN}"
FCAA_TRIGGER_MODE="${FCAA_TRIGGER_MODE:-shared_det_top1}"
FCAA_TRIGGER_MARGIN="${FCAA_TRIGGER_MARGIN:-0.05}"
FCAA_LAMBDA="${FCAA_LAMBDA:-0.3}"
FCAA_TOPK="${FCAA_TOPK:-3}"

if [[ "${FCAA_PRIVATE_DET}" == "1" ]]; then
  DEFAULT_PROFILE="mot17_private_ctrl_base"
else
  DEFAULT_PROFILE="mot17_public_ctrl_base"
fi
TRACK_PROFILE="${TRACK_PROFILE:-${DEFAULT_PROFILE}}"

ZIP_NAME="mot17_fcaa_${FCAA_LABEL}_test_submission_$(date +%Y%m%d_%H%M%S).zip"
EXPERIMENT_NAME="fcaa_submit_mot17_${FCAA_LABEL}_$(date +%Y%m%d_%H%M%S)"
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

RESULT_DIR="${BS_ROOT}/YOLOX_outputs/${EXPERIMENT_NAME}/track_results"
SANITIZED_DIR="${OUT_DIR}/sanitized_results"
RUN_MANIFEST="${BS_ROOT}/YOLOX_outputs/${EXPERIMENT_NAME}/run_manifest.json"

expand_mot17_private_detector_results() {
  local target_dir="$1"
  local src_ext="${FCAA_PRIMARY_EXT}"
  for name in "${EXPECTED[@]}"; do
    local src_name="${name/-DPM.txt/-${src_ext}.txt}"
    src_name="${src_name/-SDP.txt/-${src_ext}.txt}"
    if [[ "${src_name}" == "${name}" ]]; then
      continue
    fi
    if [[ ! -f "${target_dir}/${src_name}" ]]; then
      echo "Missing source file for MOT17 private-det expansion: ${target_dir}/${src_name}" >&2
      exit 1
    fi
    cp -f "${target_dir}/${src_name}" "${target_dir}/${name}"
    echo "[expand] ${src_name} -> ${name}" | tee -a "${RUN_LOG}"
  done
}

{
  echo "dataset=${DATASET}"
  echo "out_dir=${OUT_DIR}"
  echo "data_root=${DATA_ROOT}"
  echo "experiment_name=${EXPERIMENT_NAME}"
  echo "fcaa_checkpoint=${FCAA_CHECKPOINT}"
  echo "fcaa_label=${FCAA_LABEL}"
  echo "fcaa_trigger_mode=${FCAA_TRIGGER_MODE}"
  echo "fcaa_trigger_margin=${FCAA_TRIGGER_MARGIN}"
  echo "fcaa_lambda=${FCAA_LAMBDA}"
  echo "fcaa_topk=${FCAA_TOPK}"
  echo "run_interpolation=${RUN_INTERPOLATION}"
  echo "sanitize_results=${SANITIZE_RESULTS}"
  echo "fcaa_private_det=${FCAA_PRIVATE_DET}"
  echo "fcaa_primary_ext=${FCAA_PRIMARY_EXT}"
  echo "track_profile=${TRACK_PROFILE}"
  echo "start_time=$(date '+%F %T %z')"
} > "${OUT_DIR}/meta.txt"

(
  cd "${BS_ROOT}"
  cmd=(
    "${PYTHON_BIN}" -u tools/track.py "${DATA_ROOT}/MOT17"
    --exp-profile "${TRACK_PROFILE}"
    --experiment-name "${EXPERIMENT_NAME}"
    --run-manifest-path "${RUN_MANIFEST}"
    --fcaa-enable
    --fcaa-scorer-checkpoint "${FCAA_CHECKPOINT}"
    --fcaa-trigger-mode "${FCAA_TRIGGER_MODE}"
    --fcaa-trigger-margin "${FCAA_TRIGGER_MARGIN}"
    --fcaa-lambda "${FCAA_LAMBDA}"
    --fcaa-topk "${FCAA_TOPK}"
  )
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

if [[ "${FCAA_PRIVATE_DET}" == "1" ]]; then
  expand_mot17_private_detector_results "${PACKAGE_DIR}"
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

ZIP_MD5="$(md5sum "${OUT_DIR}/${ZIP_NAME}" | awk '{print $1}')"
if [[ -f "${RUN_MANIFEST}" ]]; then
  "${PYTHON_BIN}" "${MANIFEST_TOOL}" \
    --manifest "${RUN_MANIFEST}" \
    --set "submit.mode=\"fcaa\"" \
    --set "submit.fcaa_label=\"${FCAA_LABEL}\"" \
    --set "submit.fcaa_checkpoint=\"${FCAA_CHECKPOINT}\"" \
    --set "submit.fcaa_trigger_mode=\"${FCAA_TRIGGER_MODE}\"" \
    --set "submit.fcaa_trigger_margin=${FCAA_TRIGGER_MARGIN}" \
    --set "submit.fcaa_lambda=${FCAA_LAMBDA}" \
    --set "submit.fcaa_topk=${FCAA_TOPK}" \
    --set "submit.out_dir=\"${OUT_DIR}\"" \
    --set "submit.sanitized_dir=\"${SANITIZED_DIR}\"" \
    --set "submit.package_dir=\"${PACKAGE_DIR}\"" \
    --set "submit.zip_path=\"${OUT_DIR}/${ZIP_NAME}\"" \
    --set "submit.zip_md5=\"${ZIP_MD5}\"" \
    --set "submit.precheck_log=\"${OUT_DIR}/precheck.log\"" \
    --set "submit.meta_path=\"${OUT_DIR}/meta.txt\""
fi

echo "[OK] package_dir=${PACKAGE_DIR}"
echo "[OK] zip=${OUT_DIR}/${ZIP_NAME}"
