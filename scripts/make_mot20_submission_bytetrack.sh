#!/usr/bin/env bash
set -euo pipefail

# Build MOT20 submission zip from a tracker checkpoint.
#
# Usage:
#   bash scripts/make_mot20_submission_bytetrack.sh <CHECKPOINT> [CONFIG_PATH] [OUT_DIR] [DATA_ROOT]
#
# Example:
#   bash scripts/make_mot20_submission_bytetrack.sh \
#     outputs/bytetrack_fa_mot_mot20_v13_assoc_ft_fulltrain_xxx/checkpoint_epoch_3.pth

CHECKPOINT="${1:?checkpoint is required}"
CONFIG_PATH="${2:-configs/experiments/bytetrack_fa_mot_mot20_v13_assoc_submit_sw_yolox.yaml}"
OUT_DIR="${3:-outputs/submit_mot20_$(date +%Y%m%d_%H%M%S)}"
DATA_ROOT="${4:-/gemini/code/datasets}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}"
  exit 1
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config not found: ${CONFIG_PATH}"
  exit 1
fi

mkdir -p "${OUT_DIR}"
echo "[submit] out_dir=${OUT_DIR}"
echo "[submit] checkpoint=${CHECKPOINT}"
echo "[submit] config=${CONFIG_PATH}"

python -u submit_bytetrack.py \
  --config-path "${CONFIG_PATH}" \
  --inference-model "${CHECKPOINT}" \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUT_DIR}"

# Locate tracker result dir (usually: OUT_DIR/tracker/MOT20-test)
TRACKER_DIR="$(find "${OUT_DIR}" -maxdepth 3 -type d -name "MOT20-test" | head -n 1 || true)"
if [[ -z "${TRACKER_DIR}" ]]; then
  echo "Cannot find MOT20-test results under ${OUT_DIR}"
  exit 1
fi

for seq in MOT20-04 MOT20-06 MOT20-07 MOT20-08; do
  if [[ ! -f "${TRACKER_DIR}/${seq}.txt" ]]; then
    echo "Missing result file: ${TRACKER_DIR}/${seq}.txt"
    exit 1
  fi
done

ZIP_PATH="${OUT_DIR}/mot20_submission_$(date +%Y%m%d_%H%M%S).zip"
(
  cd "${TRACKER_DIR}"
  zip -q "${OLDPWD}/${ZIP_PATH##*/}" MOT20-04.txt MOT20-06.txt MOT20-07.txt MOT20-08.txt
)
mv "${TRACKER_DIR}/${ZIP_PATH##*/}" "${ZIP_PATH}" 2>/dev/null || true

echo "[submit] tracker_dir=${TRACKER_DIR}"
echo "[submit] zip=${ZIP_PATH}"

if [[ -f "scripts/check_mot20_submission.py" ]]; then
  echo "[submit] running precheck..."
  python scripts/check_mot20_submission.py --zip-path "${ZIP_PATH}" --profile mot20_test_4
fi
