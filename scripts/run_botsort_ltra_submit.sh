#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/gemini/code/FMtrack-main/FM-Track"
BS_ROOT="${REPO_ROOT}/external/BoT-SORT-main"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
DATASET="${1:-MOT20}"
PROTO_MODE="${2:-mean}"
OUT_DIR="${3:-${REPO_ROOT}/outputs/botsort_submit_${DATASET,,}_${PROTO_MODE}_$(date +%Y%m%d_%H%M%S)_$$}"
DATA_ROOT="${4:-/gemini/code/datasets}"
REL_SCALE="${REL_SCALE:-1.0}"
LAPLACE_WEIGHT="${LAPLACE_WEIGHT:-0.35}"
LAPLACE_PRIMARY_ONLY="${LAPLACE_PRIMARY_ONLY:-1}"
ENABLE_LAPLACE="${ENABLE_LAPLACE:-1}"
ENABLE_REID="${ENABLE_REID:-1}"
ENABLE_FUSE="${ENABLE_FUSE:-1}"
RUN_INTERPOLATION="${RUN_INTERPOLATION:-1}"
INTERP_N_MIN="${INTERP_N_MIN:-5}"
INTERP_N_DTI="${INTERP_N_DTI:-20}"
SANITIZE_RESULTS="${SANITIZE_RESULTS:-1}"
SANITIZE_PRECISION="${SANITIZE_PRECISION:-2}"
DROP_RAW_NEGXY="${DROP_RAW_NEGXY:-0}"
MOT17_PRIVATE_DET="${MOT17_PRIVATE_DET:-0}"
MOT17_PRIMARY_EXT="${MOT17_PRIMARY_EXT:-FRCNN}"
LAPLACE_CALIBRATOR="${LAPLACE_CALIBRATOR:-}"
OUT_DIR="$(realpath -m "${OUT_DIR}")"
PACKAGE_MODE="${PACKAGE_MODE:-test}"
if [[ "${ENABLE_LAPLACE}" == "1" ]]; then
  if [[ -n "${LAPLACE_CALIBRATOR}" ]]; then
    METHOD_TAG="${METHOD_TAG:-botsort_ltra_learned}"
    EXPERIMENT_NAME="${EXPERIMENT_NAME:-ltra_${DATASET,,}_${PACKAGE_MODE}_${PROTO_MODE}_learned}"
  else
    METHOD_TAG="${METHOD_TAG:-botsort_ltra}"
    EXPERIMENT_NAME="${EXPERIMENT_NAME:-ltra_${DATASET,,}_${PACKAGE_MODE}_${PROTO_MODE}}"
  fi
else
  METHOD_TAG="${METHOD_TAG:-botsort_baseline}"
  EXPERIMENT_NAME="${EXPERIMENT_NAME:-baseline_${DATASET,,}_${PACKAGE_MODE}}"
fi

if [[ -n "${LAPLACE_CALIBRATOR}" && ! -f "${LAPLACE_CALIBRATOR}" ]]; then
  echo "Missing LAPLACE_CALIBRATOR: ${LAPLACE_CALIBRATOR}" >&2
  exit 2
fi

case "${DATASET}" in
  MOT20)
    REID_CFG="fast_reid/configs/MOT20/sbs_S50.yml"
    REID_WEIGHTS="pretrained/mot20_sbs_S50.pth"
    case "${PACKAGE_MODE}" in
      test)
        ZIP_NAME="mot20_${METHOD_TAG}_${PROTO_MODE}_test_submission_$(date +%Y%m%d_%H%M%S).zip"
        CHECK_CMD=("${PYTHON_BIN}" "${REPO_ROOT}/scripts/check_mot20_submission.py" --zip-path "${OUT_DIR}/${ZIP_NAME}" --profile mot20_test_4)
        EXPECTED=(MOT20-04.txt MOT20-06.txt MOT20-07.txt MOT20-08.txt)
        SPLITS=(test)
        SPLIT_EXPECTED_test=(MOT20-04.txt MOT20-06.txt MOT20-07.txt MOT20-08.txt)
        ;;
      train)
        ZIP_NAME="mot20_${METHOD_TAG}_${PROTO_MODE}_train_submission_$(date +%Y%m%d_%H%M%S).zip"
        CHECK_CMD=("${PYTHON_BIN}" "${REPO_ROOT}/scripts/check_mot20_submission.py" --zip-path "${OUT_DIR}/${ZIP_NAME}" --profile mot20_train_4)
        EXPECTED=(MOT20-01.txt MOT20-02.txt MOT20-03.txt MOT20-05.txt)
        SPLITS=(train)
        SPLIT_EXPECTED_train=(MOT20-01.txt MOT20-02.txt MOT20-03.txt MOT20-05.txt)
        ;;
      full)
        ZIP_NAME="mot20_${METHOD_TAG}_${PROTO_MODE}_full_submission_$(date +%Y%m%d_%H%M%S).zip"
        CHECK_CMD=("${PYTHON_BIN}" "${REPO_ROOT}/scripts/check_mot20_submission.py" --zip-path "${OUT_DIR}/${ZIP_NAME}" --profile mot20_full_8)
        EXPECTED=(MOT20-01.txt MOT20-02.txt MOT20-03.txt MOT20-04.txt MOT20-05.txt MOT20-06.txt MOT20-07.txt MOT20-08.txt)
        SPLITS=(train test)
        SPLIT_EXPECTED_train=(MOT20-01.txt MOT20-02.txt MOT20-03.txt MOT20-05.txt)
        SPLIT_EXPECTED_test=(MOT20-04.txt MOT20-06.txt MOT20-07.txt MOT20-08.txt)
        ;;
      *)
        echo "Unsupported PACKAGE_MODE for MOT20: ${PACKAGE_MODE}" >&2
        exit 2
        ;;
    esac
    ;;
  MOT17)
    REID_CFG="fast_reid/configs/MOT17/sbs_S50.yml"
    REID_WEIGHTS="pretrained/mot17_sbs_S50.pth"
    TRAIN_FRCNN_EXPECTED=(
      MOT17-02-FRCNN.txt MOT17-04-FRCNN.txt MOT17-05-FRCNN.txt
      MOT17-09-FRCNN.txt MOT17-10-FRCNN.txt MOT17-11-FRCNN.txt MOT17-13-FRCNN.txt
    )
    TEST_FRCNN_EXPECTED=(
      MOT17-01-FRCNN.txt MOT17-03-FRCNN.txt MOT17-06-FRCNN.txt
      MOT17-07-FRCNN.txt MOT17-08-FRCNN.txt MOT17-12-FRCNN.txt MOT17-14-FRCNN.txt
    )
    TRAIN_EXPECTED=(
      MOT17-02-DPM.txt MOT17-02-FRCNN.txt MOT17-02-SDP.txt
      MOT17-04-DPM.txt MOT17-04-FRCNN.txt MOT17-04-SDP.txt
      MOT17-05-DPM.txt MOT17-05-FRCNN.txt MOT17-05-SDP.txt
      MOT17-09-DPM.txt MOT17-09-FRCNN.txt MOT17-09-SDP.txt
      MOT17-10-DPM.txt MOT17-10-FRCNN.txt MOT17-10-SDP.txt
      MOT17-11-DPM.txt MOT17-11-FRCNN.txt MOT17-11-SDP.txt
      MOT17-13-DPM.txt MOT17-13-FRCNN.txt MOT17-13-SDP.txt
    )
    TEST_EXPECTED=(
      MOT17-01-DPM.txt MOT17-01-FRCNN.txt MOT17-01-SDP.txt
      MOT17-03-DPM.txt MOT17-03-FRCNN.txt MOT17-03-SDP.txt
      MOT17-06-DPM.txt MOT17-06-FRCNN.txt MOT17-06-SDP.txt
      MOT17-07-DPM.txt MOT17-07-FRCNN.txt MOT17-07-SDP.txt
      MOT17-08-DPM.txt MOT17-08-FRCNN.txt MOT17-08-SDP.txt
      MOT17-12-DPM.txt MOT17-12-FRCNN.txt MOT17-12-SDP.txt
      MOT17-14-DPM.txt MOT17-14-FRCNN.txt MOT17-14-SDP.txt
    )
    case "${PACKAGE_MODE}" in
      test)
        ZIP_NAME="mot17_${METHOD_TAG}_${PROTO_MODE}_test_submission_$(date +%Y%m%d_%H%M%S).zip"
        CHECK_CMD=("${PYTHON_BIN}" "${REPO_ROOT}/scripts/check_mot17_submission.py" --zip-path "${OUT_DIR}/${ZIP_NAME}" --profile mot17_test_public_21)
        EXPECTED=("${TEST_EXPECTED[@]}")
        SPLITS=(test)
        SPLIT_EXPECTED_test=("${TEST_EXPECTED[@]}")
        if [[ "${MOT17_PRIVATE_DET}" == "1" ]]; then
          RUN_EXPECTED_test=("${TEST_FRCNN_EXPECTED[@]}")
        fi
        ;;
      train)
        ZIP_NAME="mot17_${METHOD_TAG}_${PROTO_MODE}_train_submission_$(date +%Y%m%d_%H%M%S).zip"
        CHECK_CMD=("${PYTHON_BIN}" "${REPO_ROOT}/scripts/check_mot17_submission.py" --zip-path "${OUT_DIR}/${ZIP_NAME}" --profile mot17_train_frcnn_7)
        EXPECTED=("${TRAIN_FRCNN_EXPECTED[@]}")
        SPLITS=(train)
        SPLIT_EXPECTED_train=("${TRAIN_EXPECTED[@]}")
        if [[ "${MOT17_PRIVATE_DET}" == "1" ]]; then
          RUN_EXPECTED_train=("${TRAIN_FRCNN_EXPECTED[@]}")
        fi
        ;;
      full)
        ZIP_NAME="mot17_${METHOD_TAG}_${PROTO_MODE}_full_submission_$(date +%Y%m%d_%H%M%S).zip"
        CHECK_CMD=("${PYTHON_BIN}" "${REPO_ROOT}/scripts/check_mot17_submission.py" --zip-path "${OUT_DIR}/${ZIP_NAME}" --profile mot17_full_42)
        EXPECTED=("${TRAIN_EXPECTED[@]}" "${TEST_EXPECTED[@]}")
        SPLITS=(train test)
        SPLIT_EXPECTED_train=("${TRAIN_EXPECTED[@]}")
        SPLIT_EXPECTED_test=("${TEST_EXPECTED[@]}")
        if [[ "${MOT17_PRIVATE_DET}" == "1" ]]; then
          RUN_EXPECTED_train=("${TRAIN_FRCNN_EXPECTED[@]}")
          RUN_EXPECTED_test=("${TEST_FRCNN_EXPECTED[@]}")
        fi
        ;;
      *)
        echo "Unsupported PACKAGE_MODE for MOT17: ${PACKAGE_MODE}" >&2
        exit 2
        ;;
    esac
    ;;
  *)
    echo "Unsupported dataset: ${DATASET}" >&2
    exit 2
    ;;
esac

mkdir -p "${OUT_DIR}"
RUN_LOG="${OUT_DIR}/run.log"
META_TXT="${OUT_DIR}/meta.txt"

{
  echo "dataset=${DATASET}"
  echo "proto_mode=${PROTO_MODE}"
  echo "package_mode=${PACKAGE_MODE}"
  echo "experiment_name=${EXPERIMENT_NAME}"
  echo "out_dir=${OUT_DIR}"
  echo "data_root=${DATA_ROOT}"
  echo "rel_scale=${REL_SCALE}"
  echo "laplace_weight=${LAPLACE_WEIGHT}"
  echo "laplace_primary_only=${LAPLACE_PRIMARY_ONLY}"
  echo "enable_laplace=${ENABLE_LAPLACE}"
  echo "enable_reid=${ENABLE_REID}"
  echo "enable_fuse=${ENABLE_FUSE}"
  echo "run_interpolation=${RUN_INTERPOLATION}"
  echo "interp_n_min=${INTERP_N_MIN}"
  echo "interp_n_dti=${INTERP_N_DTI}"
  echo "sanitize_results=${SANITIZE_RESULTS}"
  echo "sanitize_precision=${SANITIZE_PRECISION}"
  echo "drop_raw_negxy=${DROP_RAW_NEGXY}"
  echo "mot17_private_det=${MOT17_PRIVATE_DET}"
  echo "mot17_primary_ext=${MOT17_PRIMARY_EXT}"
  echo "laplace_calibrator=${LAPLACE_CALIBRATOR}"
  echo "method_tag=${METHOD_TAG}"
  echo "start_time=$(date '+%F %T %z')"
} > "${META_TXT}"

cd "${BS_ROOT}"

RESULT_DIR="${BS_ROOT}/YOLOX_outputs/${EXPERIMENT_NAME}/track_results"
mkdir -p "${RESULT_DIR}"

run_split() {
  local split="$1"
  local target_expected_name="SPLIT_EXPECTED_${split}"
  local run_expected_name="RUN_EXPECTED_${split}"
  if declare -p "${run_expected_name}" >/dev/null 2>&1; then
    local -n expected_ref="${run_expected_name}"
  else
    local -n expected_ref="${target_expected_name}"
  fi
  local all_present=1
  for name in "${expected_ref[@]}"; do
    if [[ ! -f "${RESULT_DIR}/${name}" ]]; then
      all_present=0
      break
    fi
  done
  if [[ "${all_present}" -eq 1 ]]; then
    echo "[skip] ${split} already present under ${RESULT_DIR}" | tee -a "${RUN_LOG}"
    return 0
  fi

  echo "[run] split=${split}" | tee -a "${RUN_LOG}"
  cmd=("${PYTHON_BIN}" -u tools/track.py "${DATA_ROOT}/${DATASET}" \
    --benchmark "${DATASET}" \
    --eval "${split}" \
    --default-parameters \
    --fp16 \
    --fast-reid-config "${REID_CFG}" \
    --fast-reid-weights "${REID_WEIGHTS}" \
    --experiment-name "${EXPERIMENT_NAME}")
  if [[ "${ENABLE_REID}" == "1" ]]; then
    cmd+=(--with-reid)
  fi
  if [[ "${ENABLE_LAPLACE}" == "1" ]]; then
    cmd+=(--laplace-assoc --laplace-proto-mode "${PROTO_MODE}" --laplace-weight "${LAPLACE_WEIGHT}" --laplace-reliability-scale "${REL_SCALE}")
    if [[ "${LAPLACE_PRIMARY_ONLY}" == "1" ]]; then
      cmd+=(--laplace-primary-only)
    fi
    if [[ -n "${LAPLACE_CALIBRATOR}" ]]; then
      cmd+=(--laplace-calibrator "${LAPLACE_CALIBRATOR}")
    fi
  fi
  if [[ "${DATASET}" == "MOT17" && "${MOT17_PRIVATE_DET}" == "1" ]]; then
    cmd+=(--mot17-detector-exts "${MOT17_PRIMARY_EXT}")
  fi
  if [[ "${ENABLE_FUSE}" == "1" ]]; then
    cmd+=(--fuse)
  fi
  "${cmd[@]}" 2>&1 | tee -a "${RUN_LOG}"
}

expand_mot17_private_detector_results() {
  local target_dir="$1"
  local split="$2"
  local -n split_expected_ref="SPLIT_EXPECTED_${split}"
  local src_ext="${MOT17_PRIMARY_EXT}"
  for name in "${split_expected_ref[@]}"; do
    local src_name="${name/-DPM.txt/-${src_ext}.txt}"
    src_name="${src_name/-SDP.txt/-${src_ext}.txt}"
    if [[ "${src_name}" == "${name}" ]]; then
      continue
    fi
    if [[ ! -f "${target_dir}/${src_name}" ]]; then
      echo "Missing source file for MOT17 private-det expansion: ${target_dir}/${src_name}" >&2
      exit 1
    fi
    # Always overwrite to avoid mixing previously-generated DPM/SDP files from a different run/config.
    cp -f "${target_dir}/${src_name}" "${target_dir}/${name}"
    echo "[expand] ${src_name} -> ${name}" | tee -a "${RUN_LOG}"
  done
}

for split in "${SPLITS[@]}"; do
  run_split "${split}"
done

if [[ "${RUN_INTERPOLATION}" == "1" ]]; then
  echo "[post] interpolation n_min=${INTERP_N_MIN} n_dti=${INTERP_N_DTI}" | tee -a "${RUN_LOG}"
  "${PYTHON_BIN}" tools/interpolation.py \
    --txt_path "${RESULT_DIR}" \
    --n_min "${INTERP_N_MIN}" \
    --n_dti "${INTERP_N_DTI}" \
    2>&1 | tee -a "${RUN_LOG}"
fi

PACKAGE_RESULT_DIR="${RESULT_DIR}"
if [[ "${SANITIZE_RESULTS}" == "1" ]]; then
  SANITIZED_DIR="${OUT_DIR}/sanitized_results"
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/sanitize_mot_submission.py" \
    --input-dir "${RESULT_DIR}" \
    --output-dir "${SANITIZED_DIR}" \
    --data-root "${DATA_ROOT}" \
    --benchmark "${DATASET}" \
    --precision "${SANITIZE_PRECISION}" \
    $( [[ "${DROP_RAW_NEGXY}" == "1" ]] && printf '%s' "--drop-raw-negxy" ) \
    2>&1 | tee -a "${RUN_LOG}"
  PACKAGE_RESULT_DIR="${SANITIZED_DIR}"
fi

if [[ "${DATASET}" == "MOT17" && "${MOT17_PRIVATE_DET}" == "1" ]]; then
  for split in "${SPLITS[@]}"; do
    expand_mot17_private_detector_results "${PACKAGE_RESULT_DIR}" "${split}"
  done
fi

for name in "${EXPECTED[@]}"; do
  if [[ ! -f "${PACKAGE_RESULT_DIR}/${name}" ]]; then
    echo "Missing expected result file: ${PACKAGE_RESULT_DIR}/${name}" >&2
    exit 1
  fi
done

(
  cd "${PACKAGE_RESULT_DIR}"
  zip -q "${OUT_DIR}/${ZIP_NAME}" "${EXPECTED[@]}"
)

printf '%s\n' "${OUT_DIR}/${ZIP_NAME}" > "${OUT_DIR}/latest_zip.txt"

"${CHECK_CMD[@]}" | tee "${OUT_DIR}/precheck.log"

echo "[OK] result_dir=${RESULT_DIR}"
echo "[OK] package_result_dir=${PACKAGE_RESULT_DIR}"
echo "[OK] zip=${OUT_DIR}/${ZIP_NAME}"
