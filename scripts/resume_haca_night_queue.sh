#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
LOG_PATH="${LOG_PATH:-${REPO_ROOT}/outputs/resume_haca_night_queue_${TS}.log}"
PLAN_CSV="${EXPERIMENT_PLAN_CSV:-${REPO_ROOT}/outputs/experiment_plan.csv}"

FULL_SUMMARY="${FULL_SUMMARY:-${REPO_ROOT}/outputs/haca_v1_matrix_after_20260313_223646/eval/mot17_summary.csv}"
FULL_CKPT="${FULL_CKPT:-${REPO_ROOT}/outputs/haca_v1_train_aligned_20260313_223646/mot17_haca_v1_20260313_223646.npz}"
NOSET_SUMMARY="${NOSET_SUMMARY:-${REPO_ROOT}/outputs/haca_v1_eval_noset_20260314_005127/eval/mot17_summary.csv}"
NOSET_CKPT="${NOSET_CKPT:-${REPO_ROOT}/outputs/haca_v1_train_noset_20260314_005127/mot17_haca_v1_20260314_005127.npz}"
NOBG_STALE_RUN_ROOT="${NOBG_STALE_RUN_ROOT:-${REPO_ROOT}/outputs/haca_v1_eval_nobg_20260314_022445}"
NOBG_CKPT="${NOBG_CKPT:-${REPO_ROOT}/outputs/haca_v1_train_nobg_20260314_022445/mot17_haca_v1_20260314_022446.npz}"
NOBG_RETRY_RUN_ROOT="${NOBG_RETRY_RUN_ROOT:-${REPO_ROOT}/outputs/haca_v1_eval_nobg_retry_${TS}}"
MOT20_RUN_ROOT_PREFIX="${MOT20_RUN_ROOT_PREFIX:-${REPO_ROOT}/outputs/haca_v1_mot20_best}"

mkdir -p "$(dirname "${LOG_PATH}")"
exec > >(tee -a "${LOG_PATH}") 2>&1

mark_plan_failed_if_present() {
  local run_root="$1"
  local summary_csv="$2"
  local checkpoint="$3"
  local log_path="$4"
  if [[ -z "${run_root}" ]]; then
    return 0
  fi
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/upsert_experiment_plan.py" \
    --csv "${PLAN_CSV}" \
    --key "run_root:${run_root}" \
    --status failed \
    --kind eval \
    --script "scripts/run_botsort_haca_v1_eval.sh" \
    --dataset MOT17 \
    --split val_half \
    --tracker-family BoT-SORT \
    --variant haca_v1_matrix \
    --run-root "${run_root}" \
    --summary-csv "${summary_csv}" \
    --checkpoint "${checkpoint}" \
    --log-path "${log_path}" \
    --notes "stale_run_detected_no_live_process"
}

echo "[queue] start $(date '+%F %T %z')"
echo "[queue] log_path=${LOG_PATH}"

NOBG_SUMMARY_CANDIDATE="${NOBG_STALE_RUN_ROOT}/eval/mot17_summary.csv"
if [[ -f "${NOBG_SUMMARY_CANDIDATE}" ]]; then
  echo "[queue] reusing existing no-background summary: ${NOBG_SUMMARY_CANDIDATE}"
  NOBG_SUMMARY="${NOBG_SUMMARY_CANDIDATE}"
else
  if [[ -d "${NOBG_STALE_RUN_ROOT}" ]]; then
    echo "[queue] stale no-background run detected, marking failed: ${NOBG_STALE_RUN_ROOT}"
    mark_plan_failed_if_present \
      "${NOBG_STALE_RUN_ROOT}" \
      "${NOBG_SUMMARY_CANDIDATE}" \
      "${NOBG_CKPT}" \
      "${NOBG_STALE_RUN_ROOT}/eval.log"
  fi

  echo "[queue] rerun no-background evaluation"
  if RUN_ROOT="${NOBG_RETRY_RUN_ROOT}" \
    RUN_BASE=0 \
    RUN_HEURISTIC=0 \
    RUN_CURRENT_LEARNED=0 \
    RUN_HACA=1 \
    HACA_NPZ="${NOBG_CKPT}" \
    "${REPO_ROOT}/scripts/run_botsort_haca_v1_eval.sh"; then
    NOBG_SUMMARY="${NOBG_RETRY_RUN_ROOT}/eval/mot17_summary.csv"
  else
    echo "[queue] no-background rerun failed; continuing with available HACA variants"
    NOBG_SUMMARY=""
  fi
fi

BEST_LINE="$(
  FULL_SUMMARY="${FULL_SUMMARY}" \
  FULL_CKPT="${FULL_CKPT}" \
  NOSET_SUMMARY="${NOSET_SUMMARY}" \
  NOSET_CKPT="${NOSET_CKPT}" \
  NOBG_SUMMARY="${NOBG_SUMMARY:-}" \
  NOBG_CKPT="${NOBG_CKPT}" \
  "${PYTHON_BIN}" - <<'PY'
import csv
import os
import sys
from pathlib import Path

items = [
    ("full", Path(os.environ["FULL_SUMMARY"]), os.environ["FULL_CKPT"]),
    ("noset", Path(os.environ["NOSET_SUMMARY"]), os.environ["NOSET_CKPT"]),
    ("nobg", Path(os.environ["NOBG_SUMMARY"]) if os.environ.get("NOBG_SUMMARY") else None, os.environ.get("NOBG_CKPT", "")),
]

rows = []
for variant, summary_path, ckpt in items:
    if summary_path is None or not summary_path.is_file():
        continue
    with summary_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        picked = None
        for row in reader:
            name = row.get("name", "")
            if name == "haca":
                picked = row
                break
            if picked is None:
                picked = row
        if picked is None:
            continue
        try:
            hota = float(picked.get("HOTA", "nan"))
            assa = float(picked.get("AssA", "nan"))
            idsw = float(picked.get("IDSW", "nan"))
        except ValueError:
            continue
        rows.append((variant, str(summary_path), ckpt, hota, assa, idsw))

if not rows:
    raise SystemExit("no_valid_haca_summary")

rows.sort(key=lambda x: (-x[3], -x[4], x[5], x[0]))
best = rows[0]
print("\t".join([
    best[0],
    best[1],
    best[2],
    f"{best[3]:.3f}",
    f"{best[4]:.3f}",
    f"{best[5]:.0f}",
]))
PY
)"

BEST_VARIANT="$(printf '%s' "${BEST_LINE}" | cut -f1)"
BEST_SUMMARY="$(printf '%s' "${BEST_LINE}" | cut -f2)"
BEST_CKPT="$(printf '%s' "${BEST_LINE}" | cut -f3)"
BEST_HOTA="$(printf '%s' "${BEST_LINE}" | cut -f4)"
BEST_ASSA="$(printf '%s' "${BEST_LINE}" | cut -f5)"
BEST_IDSW="$(printf '%s' "${BEST_LINE}" | cut -f6)"

echo "[queue] selected_haca_variant=${BEST_VARIANT}"
echo "[queue] selected_summary=${BEST_SUMMARY}"
echo "[queue] selected_checkpoint=${BEST_CKPT}"
echo "[queue] selected_metrics HOTA=${BEST_HOTA} AssA=${BEST_ASSA} IDSW=${BEST_IDSW}"

MOT20_RUN_ROOT="${MOT20_RUN_ROOT_PREFIX}_${BEST_VARIANT}_${TS}"
echo "[queue] launch MOT20 zero-shot: ${MOT20_RUN_ROOT}"
RUN_ROOT="${MOT20_RUN_ROOT}" \
RUN_BASE=0 \
RUN_HEURISTIC=0 \
RUN_CURRENT_LEARNED=0 \
RUN_HACA=1 \
HACA_NPZ="${BEST_CKPT}" \
"${REPO_ROOT}/scripts/run_botsort_haca_v1_mot20_eval.sh"

echo "[queue] done $(date '+%F %T %z')"
