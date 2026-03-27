#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
SRC_DIR="${1:-${REPO_ROOT}/outputs/competition_assoc_base_reid_da_proxy0213_hybriddumpfix}"
OUT_DIR="${2:-${REPO_ROOT}/outputs/competition_assoc_stage1_$(date +%Y%m%d_%H%M%S)}"

CASES_CSV="${SRC_DIR}/competition_cases/competition_cases.csv"
GROUP_JSONL="${SRC_DIR}/labeled_replay_top8.groups.jsonl"
RESULT_CSV="${OUT_DIR}/result.csv"
SUMMARY_CSV="${OUT_DIR}/summary.csv"
LOG_PATH="${OUT_DIR}/run.log"

mkdir -p "${OUT_DIR}"

STATUS="failed"
if "${PYTHON_BIN}" "${REPO_ROOT}/scripts/train_competition_assoc_stage1.py" \
  --cases-csv "${CASES_CSV}" \
  --group-jsonl "${GROUP_JSONL}" \
  --out-dir "${OUT_DIR}" >"${LOG_PATH}" 2>&1; then
  STATUS="success"
fi

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/append_experiment_record.py" \
  --csv "${REPO_ROOT}/outputs/experiment_registry.csv" \
  --kind train \
  --status "${STATUS}" \
  --script "scripts/run_competition_assoc_stage1.sh" \
  --dataset "MOT17" \
  --split "val0213_proxy" \
  --tracker-family "ByteTrack" \
  --variant "competition_assoc_stage1" \
  --tag "competition_assoc_mainline" \
  --run-root "${OUT_DIR}" \
  --summary-csv "${SUMMARY_CSV}" \
  --checkpoint "${OUT_DIR}/best.pt" \
  --log-path "${LOG_PATH}" \
  --notes "first-stage competition controller on base_reid_da proxy0213 conflict cases"

if ! "${PYTHON_BIN}" "${REPO_ROOT}/scripts/post_experiment_pro_bundle.py" \
  --run-root "${OUT_DIR}" \
  --tag "competition_assoc_stage1_bundle" \
  --label "competition_assoc_stage1" \
  --status "${STATUS}"; then
  echo "[competition-stage1] warning: failed to build Pro review bundle for ${OUT_DIR}" >&2
fi

echo "[competition-stage1] status=${STATUS} out_dir=${OUT_DIR}"
