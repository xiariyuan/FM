#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"

MATRIX_SUMMARY="${MATRIX_SUMMARY:-}"
MATRIX_RUN_ROOT="${MATRIX_RUN_ROOT:-}"
FULL_HACA_CKPT="${FULL_HACA_CKPT:-}"
CURRENT_CALIBRATOR_NPZ="${CURRENT_CALIBRATOR_NPZ:-}"
POLL_SECS="${POLL_SECS:-60}"
TIMEOUT_SECS="${TIMEOUT_SECS:-43200}"
LOG_PATH="${LOG_PATH:-${REPO_ROOT}/outputs/overnight_haca_followups_${TS}.log}"

if [[ -z "${MATRIX_SUMMARY}" ]]; then
  echo "MATRIX_SUMMARY is required" >&2
  exit 1
fi
if [[ -z "${FULL_HACA_CKPT}" ]]; then
  echo "FULL_HACA_CKPT is required" >&2
  exit 1
fi

mkdir -p "$(dirname "${LOG_PATH}")"
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "[queue] matrix_summary=${MATRIX_SUMMARY}"
echo "[queue] matrix_run_root=${MATRIX_RUN_ROOT}"
echo "[queue] full_haca_ckpt=${FULL_HACA_CKPT}"
echo "[queue] current_calibrator=${CURRENT_CALIBRATOR_NPZ}"
echo "[queue] poll_secs=${POLL_SECS} timeout_secs=${TIMEOUT_SECS}"

matrix_status() {
  "${PYTHON_BIN}" - <<'PY' "${REPO_ROOT}/outputs/experiment_plan.csv" "${MATRIX_RUN_ROOT}"
import csv
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
run_root = sys.argv[2]
if not run_root or not csv_path.is_file():
    print("")
    raise SystemExit(0)
with csv_path.open("r", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row.get("run_root", "") == run_root:
            print(row.get("status", ""))
            raise SystemExit(0)
print("")
PY
}

start_ts="$(date +%s)"
while [[ ! -f "${MATRIX_SUMMARY}" ]]; do
  now_ts="$(date +%s)"
  elapsed=$((now_ts - start_ts))
  if (( elapsed > TIMEOUT_SECS )); then
    echo "[queue] timeout waiting for matrix summary: ${MATRIX_SUMMARY}"
    exit 1
  fi
  status="$(matrix_status || true)"
  if [[ "${status}" == "failed" || "${status}" == "cancelled" ]]; then
    echo "[queue] matrix status=${status}; aborting follow-up queue"
    exit 1
  fi
  sleep "${POLL_SECS}"
done

echo "[queue] matrix summary ready: ${MATRIX_SUMMARY}"

NSET_TS="$(date +%Y%m%d_%H%M%S)"
NSET_TRAIN_ROOT="${REPO_ROOT}/outputs/haca_v1_train_noset_${NSET_TS}"
NSET_EVAL_ROOT="${REPO_ROOT}/outputs/haca_v1_eval_noset_${NSET_TS}"
echo "[queue] train no-set encoder ablation"
OUT_ROOT="${NSET_TRAIN_ROOT}" \
DISABLE_SET_ENCODER=1 \
./scripts/train_haca_v1_mot17.sh
NSET_CKPT="$(find "${NSET_TRAIN_ROOT}" -maxdepth 1 -type f -name '*.npz' | head -n 1)"
if [[ -z "${NSET_CKPT}" ]]; then
  echo "[queue] missing no-set checkpoint under ${NSET_TRAIN_ROOT}" >&2
  exit 1
fi

echo "[queue] eval no-set encoder ablation"
RUN_ROOT="${NSET_EVAL_ROOT}" \
RUN_BASE=0 \
RUN_HEURISTIC=0 \
RUN_CURRENT_LEARNED=0 \
RUN_HACA=1 \
HACA_NPZ="${NSET_CKPT}" \
./scripts/run_botsort_haca_v1_eval.sh

NBG_TS="$(date +%Y%m%d_%H%M%S)"
NBG_TRAIN_ROOT="${REPO_ROOT}/outputs/haca_v1_train_nobg_${NBG_TS}"
NBG_EVAL_ROOT="${REPO_ROOT}/outputs/haca_v1_eval_nobg_${NBG_TS}"
echo "[queue] train no-background ablation"
OUT_ROOT="${NBG_TRAIN_ROOT}" \
DISABLE_BACKGROUND=1 \
./scripts/train_haca_v1_mot17.sh
NBG_CKPT="$(find "${NBG_TRAIN_ROOT}" -maxdepth 1 -type f -name '*.npz' | head -n 1)"
if [[ -z "${NBG_CKPT}" ]]; then
  echo "[queue] missing no-background checkpoint under ${NBG_TRAIN_ROOT}" >&2
  exit 1
fi

echo "[queue] eval no-background ablation"
RUN_ROOT="${NBG_EVAL_ROOT}" \
RUN_BASE=0 \
RUN_HEURISTIC=0 \
RUN_CURRENT_LEARNED=0 \
RUN_HACA=1 \
HACA_NPZ="${NBG_CKPT}" \
./scripts/run_botsort_haca_v1_eval.sh

MOT20_TS="$(date +%Y%m%d_%H%M%S)"
MOT20_ROOT="${REPO_ROOT}/outputs/haca_v1_mot20_${MOT20_TS}"
echo "[queue] eval MOT20 zero-shot with full HACA checkpoint"
RUN_ROOT="${MOT20_ROOT}" \
RUN_BASE=0 \
RUN_HEURISTIC=0 \
RUN_CURRENT_LEARNED=0 \
RUN_HACA=1 \
HACA_NPZ="${FULL_HACA_CKPT}" \
CURRENT_CALIBRATOR_NPZ="${CURRENT_CALIBRATOR_NPZ}" \
./scripts/run_botsort_haca_v1_mot20_eval.sh

echo "[queue] done"
