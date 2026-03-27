#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
PLAN_CSV="${EXPERIMENT_PLAN_CSV:-${REPO_ROOT}/outputs/experiment_plan.csv}"
SCRIPT_NAME="${SCRIPT_NAME:-scripts/queue_haca_v3_overnight.sh}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"

MAIN_TRAIN_RUN_ROOT="${MAIN_TRAIN_RUN_ROOT:-}"
MAIN_CHECKPOINT="${MAIN_CHECKPOINT:-}"
MAIN_PIPELINE_PLAN_KEY="${MAIN_PIPELINE_PLAN_KEY:-}"
MAIN_SAMEBASE_SUMMARY="${MAIN_SAMEBASE_SUMMARY:-}"
MAIN_STRONGSORT_SUMMARY="${MAIN_STRONGSORT_SUMMARY:-}"
MAIN_MOT20_SUMMARY="${MAIN_MOT20_SUMMARY:-}"

QUEUE_LOG_PATH="${QUEUE_LOG_PATH:-}"
PLAN_KEY="${PLAN_KEY:-}"
POLL_SECS="${POLL_SECS:-60}"
TIMEOUT_SECS="${TIMEOUT_SECS:-64800}"

CURRENT_CALIBRATOR_NPZ="${CURRENT_CALIBRATOR_NPZ:-${REPO_ROOT}/outputs/lpb_ltra_formal_mot17_shrink_20260311_191211/train_vec_20260311_201832/mot17_lpb_ltra_20260311_201832.npz}"
RUN_TRANSFER_FOR_BEST="${RUN_TRANSFER_FOR_BEST:-1}"

if [[ -z "${MAIN_TRAIN_RUN_ROOT}" ]]; then
  echo "MAIN_TRAIN_RUN_ROOT is required" >&2
  exit 1
fi

if [[ -z "${MAIN_CHECKPOINT}" ]]; then
  MAIN_CHECKPOINT="${MAIN_TRAIN_RUN_ROOT}/$(basename "${MAIN_TRAIN_RUN_ROOT}" | sed 's/_train_/_/').npz"
fi

if [[ -z "${MAIN_PIPELINE_PLAN_KEY}" ]]; then
  MAIN_PIPELINE_PLAN_KEY="pipeline:${MAIN_TRAIN_RUN_ROOT}"
fi

if [[ -z "${MAIN_SAMEBASE_SUMMARY}" ]]; then
  MAIN_SAMEBASE_SUMMARY="${REPO_ROOT}/outputs/haca_v3_eval_after_20260314_230101/eval/mot17_summary.csv"
fi
if [[ -z "${MAIN_STRONGSORT_SUMMARY}" ]]; then
  MAIN_STRONGSORT_SUMMARY="${REPO_ROOT}/outputs/strongsort_haca_v3/MOT17_val_after_20260314_230101/summary.csv"
fi
if [[ -z "${MAIN_MOT20_SUMMARY}" ]]; then
  MAIN_MOT20_SUMMARY="${REPO_ROOT}/outputs/haca_v3_mot20_after_20260314_230101/eval/mot20_summary.csv"
fi

if [[ -z "${QUEUE_LOG_PATH}" ]]; then
  QUEUE_LOG_PATH="${MAIN_TRAIN_RUN_ROOT}/overnight_queue.log"
fi
if [[ -z "${PLAN_KEY}" ]]; then
  PLAN_KEY="overnight:${MAIN_TRAIN_RUN_ROOT}"
fi

mkdir -p "$(dirname "${QUEUE_LOG_PATH}")"

update_plan_status() {
  local status="$1"
  shift || true
  local extras=(
    "main_train_run_root=${MAIN_TRAIN_RUN_ROOT}"
    "main_pipeline_plan_key=${MAIN_PIPELINE_PLAN_KEY}"
    "run_transfer_for_best=${RUN_TRANSFER_FOR_BEST}"
    "poll_secs=${POLL_SECS}"
  )
  if [[ $# -gt 0 ]]; then
    extras+=("$@")
  fi
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/upsert_experiment_plan.py" \
    --csv "${PLAN_CSV}" \
    --key "${PLAN_KEY}" \
    --status "${status}" \
    --kind other \
    --script "${SCRIPT_NAME}" \
    --dataset "MOT17_MOT20_StrongSORT" \
    --split overnight \
    --tracker-family "BoT-SORT+StrongSORT" \
    --variant haca_v3_overnight \
    --run-root "${MAIN_TRAIN_RUN_ROOT}" \
    --checkpoint "${MAIN_CHECKPOINT}" \
    --log-path "${QUEUE_LOG_PATH}" \
    --notes "overnight_haca_v3_followups" \
    --extra "${extras[@]}"
}

on_exit() {
  local rc=$?
  trap - EXIT
  if [[ ${rc} -ne 0 ]]; then
    update_plan_status failed "exit_code=${rc}" || true
  fi
  exit ${rc}
}

trap on_exit EXIT
update_plan_status running

plan_status() {
  "${PYTHON_BIN}" - <<'PY' "${PLAN_CSV}" "$1"
import csv
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
plan_key = sys.argv[2]
if not csv_path.is_file():
    print("")
    raise SystemExit(0)
with csv_path.open("r", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row.get("plan_key", "") == plan_key:
            print(row.get("status", ""))
            raise SystemExit(0)
print("")
PY
}

wait_for_pipeline() {
  local start_ts now_ts elapsed status
  start_ts="$(date +%s)"
  echo "[overnight] waiting for main pipeline ${MAIN_PIPELINE_PLAN_KEY}"
  while true; do
    status="$(plan_status "${MAIN_PIPELINE_PLAN_KEY}" || true)"
    if [[ "${status}" == "completed" || "${status}" == "failed" || "${status}" == "cancelled" ]]; then
      echo "[overnight] main pipeline status=${status}"
      break
    fi
    now_ts="$(date +%s)"
    elapsed=$((now_ts - start_ts))
    if (( elapsed > TIMEOUT_SECS )); then
      echo "[overnight] timeout waiting for main pipeline"
      return 1
    fi
    sleep "${POLL_SECS}"
  done
}

run_variant() {
  local label="$1"
  local comp_topk="$2"
  local comp_margin_quantile="$3"
  local comp_delta_scale="$4"

  local train_root="${REPO_ROOT}/outputs/${label}_train_${TS}"
  local train_npz="${train_root}/mot17_${label}_${TS}.npz"
  local eval_root="${REPO_ROOT}/outputs/${label}_eval_${TS}"
  local summary_csv="${eval_root}/eval/mot17_summary.csv"

  echo "[overnight] train ${label} topk=${comp_topk} q=${comp_margin_quantile} delta=${comp_delta_scale}"
  if ! env \
    HACA_LABEL="${label}" \
    VARIANT_NAME="${label}_train" \
    OUT_ROOT="${train_root}" \
    OUT_NPZ="${train_npz}" \
    BASE_NPZ="${REPO_ROOT}/outputs/haca_v2_nobg_train_20260314_2025/mot17_haca_v2_nobg_20260314_202647.npz" \
    EPOCHS=12 \
    COMP_TOPK="${comp_topk}" \
    COMP_MARGIN_QUANTILE="${comp_margin_quantile}" \
    COMP_DELTA_SCALE="${comp_delta_scale}" \
    bash "${REPO_ROOT}/scripts/train_haca_v3_mot17.sh"; then
    echo "[overnight] train failed for ${label}"
    return 1
  fi

  echo "[overnight] eval ${label}"
  if ! env \
    HACA_LABEL="${label}" \
    RUN_ROOT="${eval_root}" \
    HACA_NPZ="${train_npz}" \
    RUN_BASE=0 RUN_HEURISTIC=1 RUN_CURRENT_LEARNED=1 RUN_HACA=1 \
    CURRENT_CALIBRATOR_NPZ="${CURRENT_CALIBRATOR_NPZ}" \
    bash "${REPO_ROOT}/scripts/run_botsort_haca_v3_eval.sh"; then
    echo "[overnight] eval failed for ${label}"
    return 1
  fi

  echo "${label}|${train_npz}|${summary_csv}" >> "${MAIN_TRAIN_RUN_ROOT}/overnight_candidates_${TS}.txt"
  echo "[overnight] candidate recorded ${label}"
  return 0
}

select_best_candidate() {
  "${PYTHON_BIN}" - <<'PY' "${MAIN_TRAIN_RUN_ROOT}/overnight_candidates_${TS}.txt"
import csv
import os
import sys

candidate_file = sys.argv[1]
best = None

def read_summary(path):
    rows = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows[row["name"]] = row
    return rows

with open(candidate_file, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        label, checkpoint, summary = line.split("|")
        if not os.path.isfile(summary):
            continue
        rows = read_summary(summary)
        haca = rows.get("haca")
        heur = rows.get("heuristic")
        if haca is None or heur is None:
            continue
        hota = float(haca["HOTA"])
        assa = float(haca["AssA"])
        idsw = float(haca["IDSW"])
        heur_hota = float(heur["HOTA"])
        heur_assa = float(heur["AssA"])
        beats_heur = 1 if hota > heur_hota and assa > heur_assa else 0
        score = (beats_heur, hota + assa, hota, assa, -idsw)
        if best is None or score > best[0]:
            best = (score, label, checkpoint, summary, beats_heur, hota, assa)

if best is None:
    raise SystemExit(2)

_, label, checkpoint, summary, beats_heur, hota, assa = best
print(f"{label}|{checkpoint}|{summary}|{beats_heur}|{hota:.3f}|{assa:.3f}")
PY
}

run_transfer_for_candidate() {
  local label="$1"
  local checkpoint="$2"
  local strong_root="${REPO_ROOT}/outputs/strongsort_${label}/MOT17_val_${TS}"
  local mot20_root="${REPO_ROOT}/outputs/${label}_mot20_${TS}"

  echo "[overnight] transfer StrongSORT ${label}"
  env \
    HACA_LABEL="${label}" \
    OUT_ROOT="${strong_root}" \
    HACA_NPZ="${checkpoint}" \
    bash "${REPO_ROOT}/scripts/run_strongsort_haca_v3_mot17_val.sh"

  echo "[overnight] transfer MOT20 ${label}"
  env \
    HACA_LABEL="${label}" \
    RUN_ROOT="${mot20_root}" \
    HACA_NPZ="${checkpoint}" \
    RUN_BASE=0 RUN_HEURISTIC=1 RUN_CURRENT_LEARNED=0 RUN_HACA=1 \
    bash "${REPO_ROOT}/scripts/run_botsort_haca_v3_mot20_eval.sh"
}

echo "[overnight] $(date '+%F %T %z') start"
echo "[overnight] main_train_run_root=${MAIN_TRAIN_RUN_ROOT}"
echo "[overnight] queue_log_path=${QUEUE_LOG_PATH}"

wait_for_pipeline

: > "${MAIN_TRAIN_RUN_ROOT}/overnight_candidates_${TS}.txt"

if [[ -f "${MAIN_SAMEBASE_SUMMARY}" ]]; then
  echo "haca_v3_main|${MAIN_CHECKPOINT}|${MAIN_SAMEBASE_SUMMARY}" >> "${MAIN_TRAIN_RUN_ROOT}/overnight_candidates_${TS}.txt"
fi

run_variant "haca_v3_k2safe" 2 0.30 0.75 || true
run_variant "haca_v3_q45" 3 0.45 1.00 || true
run_variant "haca_v3_k4" 4 0.35 1.00 || true
run_variant "haca_v3_d125" 3 0.35 1.25 || true

best_line="$(select_best_candidate || true)"
if [[ -z "${best_line}" ]]; then
  echo "[overnight] no valid candidate summary found"
  update_plan_status completed "stop_reason=no_valid_candidate"
  exit 0
fi

IFS='|' read -r best_label best_ckpt best_summary best_beats_heur best_hota best_assa <<< "${best_line}"
echo "[overnight] best candidate=${best_label} hota=${best_hota} assa=${best_assa} beats_heur=${best_beats_heur}"

if [[ "${RUN_TRANSFER_FOR_BEST}" == "1" && "${best_beats_heur}" == "1" ]]; then
  if [[ "${best_label}" == "haca_v3_main" && -f "${MAIN_STRONGSORT_SUMMARY}" && -f "${MAIN_MOT20_SUMMARY}" ]]; then
    echo "[overnight] main candidate already has transfer results; skip rerun"
  else
    run_transfer_for_candidate "${best_label}" "${best_ckpt}" || true
  fi
fi

echo "[overnight] $(date '+%F %T %z') finished"
update_plan_status completed "best_candidate=${best_label}" "best_hota=${best_hota}" "best_assa=${best_assa}" "best_beats_heur=${best_beats_heur}"
