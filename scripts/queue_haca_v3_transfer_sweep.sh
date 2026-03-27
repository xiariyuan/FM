#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
PLAN_CSV="${EXPERIMENT_PLAN_CSV:-${REPO_ROOT}/outputs/experiment_plan.csv}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
SCRIPT_NAME="${SCRIPT_NAME:-scripts/queue_haca_v3_transfer_sweep.sh}"
QUEUE_LABEL="${QUEUE_LABEL:-haca_v3_transfer_sweep_${TS}}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/outputs/${QUEUE_LABEL}}"
RUN_ROOT="$(realpath -m "${RUN_ROOT}")"
LOG_PATH="${RUN_ROOT}/queue.log"
PLAN_KEY="${PLAN_KEY:-queue:${RUN_ROOT}}"
POLL_SECS="${POLL_SECS:-60}"

mkdir -p "${RUN_ROOT}"
exec > >(tee -a "${LOG_PATH}") 2>&1

update_plan_status() {
  local status="$1"
  shift || true
  local extras=(
    "poll_secs=${POLL_SECS}"
    "queue_label=${QUEUE_LABEL}"
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
    --dataset "MOT17+MOT20" \
    --split "transfer_sweep" \
    --tracker-family "BoT-SORT+StrongSORT" \
    --variant "haca_v3_transfer_sweep" \
    --run-root "${RUN_ROOT}" \
    --log-path "${LOG_PATH}" \
    --notes "strongsort_sweep_then_best_mot20" \
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

echo "[queue] $(date '+%F %T %z') start"

run_strongsort_variant() {
  local label="$1"
  local ckpt="$2"
  local out_root="${REPO_ROOT}/outputs/strongsort_${label}/MOT17_val_${TS}"
  echo "[queue] StrongSORT ${label}"
  env \
    HACA_LABEL="${label}" \
    OUT_ROOT="${out_root}" \
    HACA_NPZ="${ckpt}" \
    bash "${REPO_ROOT}/scripts/run_strongsort_haca_v3_mot17_val.sh"
}

select_best_for_mot20() {
  "${PYTHON_BIN}" - <<'PY' "${TS}"
import csv
import sys
from pathlib import Path

ts = sys.argv[1]
repo = Path("/gemini/code/FMtrack-main/FM-Track")
heur_hota = 69.757
heur_assa = 73.662

samebase_dirs = {
    "haca_v3_k2safe": repo / "outputs/haca_v3_k2safe_eval_20260314_234016/eval/mot17_summary.csv",
    "haca_v3_q45": repo / "outputs/haca_v3_q45_eval_20260314_234016/eval/mot17_summary.csv",
    "haca_v3_k4": repo / "outputs/haca_v3_k4_eval_20260314_234016/eval/mot17_summary.csv",
}
ckpts = {
    "haca_v3_k2safe": repo / "outputs/haca_v3_k2safe_train_20260314_234016/mot17_haca_v3_k2safe_20260314_234016.npz",
    "haca_v3_q45": repo / "outputs/haca_v3_q45_train_20260314_234016/mot17_haca_v3_q45_20260314_234016.npz",
    "haca_v3_k4": repo / "outputs/haca_v3_k4_train_20260314_234016/mot17_haca_v3_k4_20260314_234016.npz",
}
strongsort_dirs = {
    label: repo / f"outputs/strongsort_{label}/MOT17_val_{ts}/summary.csv"
    for label in samebase_dirs
}

def load_row(path, name_key):
    rows = {r["name"]: r for r in csv.DictReader(path.open())}
    return rows[name_key]

candidates = []
for label, path in strongsort_dirs.items():
    if not path.is_file():
        continue
    ss = load_row(path, "haca_eval")
    sb = load_row(samebase_dirs[label], "haca")
    ss_hota = float(ss["HOTA"])
    ss_assa = float(ss["AssA"])
    ss_idsw = float(ss["IDSW"])
    sb_hota = float(sb["HOTA"])
    sb_assa = float(sb["AssA"])
    score = (
        ss_hota >= heur_hota - 0.25,
        ss_assa >= heur_assa - 0.35,
        ss_hota + ss_assa,
        sb_hota + sb_assa,
        -ss_idsw,
    )
    candidates.append((score, label, ckpts[label]))

if not candidates:
    raise SystemExit(2)

candidates.sort(reverse=True)
_, label, ckpt = candidates[0]
print(f"{label}|{ckpt}")
PY
}

run_mot20_variant() {
  local label="$1"
  local ckpt="$2"
  local run_root="${REPO_ROOT}/outputs/${label}_mot20_${TS}"
  echo "[queue] MOT20 ${label}"
  env \
    HACA_LABEL="${label}" \
    RUN_ROOT="${run_root}" \
    HACA_NPZ="${ckpt}" \
    RUN_BASE=0 \
    RUN_HEURISTIC=1 \
    RUN_CURRENT_LEARNED=0 \
    RUN_HACA=1 \
    bash "${REPO_ROOT}/scripts/run_botsort_haca_v3_mot20_eval.sh"
}

run_strongsort_variant "haca_v3_k2safe" "${REPO_ROOT}/outputs/haca_v3_k2safe_train_20260314_234016/mot17_haca_v3_k2safe_20260314_234016.npz"
run_strongsort_variant "haca_v3_q45" "${REPO_ROOT}/outputs/haca_v3_q45_train_20260314_234016/mot17_haca_v3_q45_20260314_234016.npz"
run_strongsort_variant "haca_v3_k4" "${REPO_ROOT}/outputs/haca_v3_k4_train_20260314_234016/mot17_haca_v3_k4_20260314_234016.npz"

BEST="$(select_best_for_mot20)"
BEST_LABEL="${BEST%%|*}"
BEST_CKPT="${BEST#*|}"
echo "[queue] best_for_mot20=${BEST_LABEL}"
update_plan_status running "best_mot20_label=${BEST_LABEL}" "best_mot20_ckpt=${BEST_CKPT}"

run_mot20_variant "${BEST_LABEL}" "${BEST_CKPT}"

echo "[queue] $(date '+%F %T %z') done"
update_plan_status completed "best_mot20_label=${BEST_LABEL}" "best_mot20_ckpt=${BEST_CKPT}"
