#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/gemini/code/FMtrack-main/FM-Track"
cd "${REPO_ROOT}"

WAIT_PID="${1:?usage: bash scripts/continue_botsort_ltra_after_pid.sh <wait_pid> [python_bin]}"
PYTHON_BIN="${2:-/root/miniconda3/bin/python}"

echo "[wait] waiting for PID ${WAIT_PID}"
while kill -0 "${WAIT_PID}" 2>/dev/null; do
  sleep 30
done
echo "[wait] PID ${WAIT_PID} finished"

mkdir -p outputs/botsort_ltra_stage1/MOT17

echo "[eval] MOT17 base"
"${PYTHON_BIN}" scripts/eval_botsort_halfval_trackeval.py \
  --dataset MOT17 \
  --results-dir external/BoT-SORT-main/YOLOX_outputs/laplace_mot17_val_base/track_results \
  --tracker-name botsort_mot17_val_base \
  --work-dir outputs/botsort_ltra_stage1/MOT17/base_eval

echo "[eval] MOT17 laplace"
"${PYTHON_BIN}" scripts/eval_botsort_halfval_trackeval.py \
  --dataset MOT17 \
  --results-dir external/BoT-SORT-main/YOLOX_outputs/laplace_mot17_val_laplace/track_results \
  --tracker-name botsort_mot17_val_laplace \
  --work-dir outputs/botsort_ltra_stage1/MOT17/laplace_eval

echo "[collect] MOT17"
"${PYTHON_BIN}" scripts/collect_trackeval_metrics.py \
  outputs/botsort_ltra_stage1/MOT17/base_eval/eval/botsort_mot17_val_base \
  outputs/botsort_ltra_stage1/MOT17/laplace_eval/eval/botsort_mot17_val_laplace \
  --csv outputs/botsort_ltra_stage1/MOT17/summary.csv | tee outputs/botsort_ltra_stage1/MOT17/summary.txt

echo "[run] MOT20 stage1"
bash scripts/run_botsort_ltra_stage1.sh MOT20
