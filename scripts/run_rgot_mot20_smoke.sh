#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"
BOT_ROOT="${BOT_ROOT:-${REPO_ROOT}/external/BoT-SORT-main}"
DATA_ROOT="${DATA_ROOT:-/gemini/code/datasets}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"

SCRIPT_NAME="${SCRIPT_NAME:-scripts/run_rgot_mot20_smoke.sh}"
VARIANT_NAME="${VARIANT_NAME:-rgot_mot20_smoke_analysis_only}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/outputs/rgot_analysis_smoke_${TS}}"
RUN_ROOT="$(realpath -m "${RUN_ROOT}")"
LOG_PATH="${RUN_ROOT}/smoke.log"
SUMMARY_CSV="${RUN_ROOT}/summary.csv"
REGISTRY_CSV="${REGISTRY_CSV:-${REPO_ROOT}/outputs/experiment_registry.csv}"
PLAN_CSV="${PLAN_CSV:-${REPO_ROOT}/outputs/experiment_plan.csv}"
PLAN_KEY="${PLAN_KEY:-run_root:${RUN_ROOT}}"

EXP_FILE="${EXP_FILE:-./yolox/exps/example/mot/yolox_x_mix_mot20_ch.py}"
CKPT="${CKPT:-./pretrained/bytetrack_x_mot20.pth.tar}"
REID_CFG="${REID_CFG:-fast_reid/configs/MOT20/sbs_S50.yml}"
REID_WTS="${REID_WTS:-pretrained/mot20_sbs_S50.pth}"
HACA_NPZ="${HACA_NPZ:-${REPO_ROOT}/outputs/haca_mot20_train_20260613_082005/haca_v1/mot20_haca_v1.npz}"

SEQ_ID="${SEQ_ID:-5}"
SEQ_NAME="${SEQ_NAME:-MOT20-05}"
TRACK_EXP_NAME="${TRACK_EXP_NAME:-rgot_mot20_haca_v1_nobg_smoke_${TS}}"
TRACK_RESULTS_DIR="${BOT_ROOT}/YOLOX_outputs/${TRACK_EXP_NAME}/track_results"

RGOT_ANALYSIS_DIR="${RUN_ROOT}/rgot_analysis"
RGOT_SUMMARY_CSV="${RGOT_ANALYSIS_DIR}/${SEQ_NAME}_summary.csv"
RGOT_EVENT_JSONL="${RGOT_ANALYSIS_DIR}/${SEQ_NAME}_events.jsonl"

LAPLACE_DECAY_SCALES="${LAPLACE_DECAY_SCALES:-1 2 4}"
LAPLACE_MIN_HISTORY="${LAPLACE_MIN_HISTORY:-3}"
LAPLACE_PROTO_MODE="${LAPLACE_PROTO_MODE:-multi}"

RGOT_TOP_K="${RGOT_TOP_K:-3}"
RGOT_ROW_MARGIN="${RGOT_ROW_MARGIN:-0.03}"
RGOT_COL_MARGIN="${RGOT_COL_MARGIN:-0.03}"
RGOT_MAX_ROWS="${RGOT_MAX_ROWS:-4}"
RGOT_MAX_COLS="${RGOT_MAX_COLS:-4}"

mkdir -p "${RUN_ROOT}" "${RGOT_ANALYSIS_DIR}"
exec > >(tee -a "${LOG_PATH}") 2>&1

START_TIME="$(date --iso-8601=seconds)"
CURRENT_PID="$$"

log() {
  echo "[$(date '+%F %T')] $*"
}

read_csv_field() {
  local csv_path="$1"
  local field_name="$2"
  if [[ ! -f "${csv_path}" ]]; then
    return 0
  fi
  "${PYTHON_BIN}" - "${csv_path}" "${field_name}" <<'PY'
import csv
import sys

path, field = sys.argv[1], sys.argv[2]
with open(path, "r", newline="") as f:
    row = next(csv.DictReader(f), {})
print(row.get(field, ""))
PY
}

write_summary() {
  local status="$1"
  local end_time="$2"
  local exit_code="$3"
  local event_count="$4"
  local candidate_blocks="$5"
  local trigger_blocks="$6"
  local owneralt_overlap_events="$7"
  local graph_assoc_overlap_events="$8"
  local unexplained_cases="$9"
  local notes="${10}"

  SUMMARY_TARGET="${SUMMARY_CSV}" \
  SUMMARY_TIMESTAMP="$(date --iso-8601=seconds)" \
  SUMMARY_STATUS="${status}" \
  SUMMARY_SCRIPT="${SCRIPT_NAME}" \
  SUMMARY_DATASET="MOT20" \
  SUMMARY_SPLIT="val_half" \
  SUMMARY_TRACKER_FAMILY="BoT-SORT" \
  SUMMARY_VARIANT="${VARIANT_NAME}" \
  SUMMARY_RUN_ROOT="${RUN_ROOT}" \
  SUMMARY_SUMMARY_CSV="${SUMMARY_CSV}" \
  SUMMARY_CHECKPOINT="${HACA_NPZ}" \
  SUMMARY_LOG_PATH="${LOG_PATH}" \
  SUMMARY_SEQ_NAME="${SEQ_NAME}" \
  SUMMARY_SEQ_ID="${SEQ_ID}" \
  SUMMARY_EXPERIMENT_NAME="${TRACK_EXP_NAME}" \
  SUMMARY_TRACKER_OUTPUT_DIR="${TRACK_RESULTS_DIR}" \
  SUMMARY_RGOT_ANALYSIS_DIR="${RGOT_ANALYSIS_DIR}" \
  SUMMARY_RGOT_SUMMARY_CSV="${RGOT_SUMMARY_CSV}" \
  SUMMARY_RGOT_EVENT_JSONL="${RGOT_EVENT_JSONL}" \
  SUMMARY_PID="${CURRENT_PID}" \
  SUMMARY_START_TIME="${START_TIME}" \
  SUMMARY_END_TIME="${end_time}" \
  SUMMARY_EXIT_CODE="${exit_code}" \
  SUMMARY_EVENT_COUNT="${event_count}" \
  SUMMARY_CANDIDATE_BLOCKS="${candidate_blocks}" \
  SUMMARY_TRIGGER_BLOCKS="${trigger_blocks}" \
  SUMMARY_OWNERALT_OVERLAP_EVENTS="${owneralt_overlap_events}" \
  SUMMARY_GRAPH_ASSOC_OVERLAP_EVENTS="${graph_assoc_overlap_events}" \
  SUMMARY_UNEXPLAINED_CASES="${unexplained_cases}" \
  SUMMARY_NOTES="${notes}" \
  "${PYTHON_BIN}" - <<'PY'
import csv
import os
from pathlib import Path

target = Path(os.environ["SUMMARY_TARGET"])
target.parent.mkdir(parents=True, exist_ok=True)
tmp = target.with_suffix(target.suffix + ".tmp")
fieldnames = [
    "timestamp",
    "kind",
    "status",
    "script",
    "dataset",
    "split",
    "tracker_family",
    "variant",
    "run_root",
    "summary_csv",
    "checkpoint",
    "log_path",
    "seq_name",
    "seq_id",
    "experiment_name",
    "tracker_output_dir",
    "rgot_analysis_dir",
    "rgot_summary_csv",
    "rgot_event_jsonl",
    "pid",
    "start_time",
    "end_time",
    "exit_code",
    "event_count",
    "candidate_blocks",
    "trigger_blocks",
    "owneralt_overlap_events",
    "graph_assoc_overlap_events",
    "not_explained_by_buffer_or_reentry_cases",
    "notes",
]
row = {
    "timestamp": os.environ.get("SUMMARY_TIMESTAMP", ""),
    "kind": "analysis",
    "status": os.environ.get("SUMMARY_STATUS", ""),
    "script": os.environ.get("SUMMARY_SCRIPT", ""),
    "dataset": os.environ.get("SUMMARY_DATASET", ""),
    "split": os.environ.get("SUMMARY_SPLIT", ""),
    "tracker_family": os.environ.get("SUMMARY_TRACKER_FAMILY", ""),
    "variant": os.environ.get("SUMMARY_VARIANT", ""),
    "run_root": os.environ.get("SUMMARY_RUN_ROOT", ""),
    "summary_csv": os.environ.get("SUMMARY_SUMMARY_CSV", ""),
    "checkpoint": os.environ.get("SUMMARY_CHECKPOINT", ""),
    "log_path": os.environ.get("SUMMARY_LOG_PATH", ""),
    "seq_name": os.environ.get("SUMMARY_SEQ_NAME", ""),
    "seq_id": os.environ.get("SUMMARY_SEQ_ID", ""),
    "experiment_name": os.environ.get("SUMMARY_EXPERIMENT_NAME", ""),
    "tracker_output_dir": os.environ.get("SUMMARY_TRACKER_OUTPUT_DIR", ""),
    "rgot_analysis_dir": os.environ.get("SUMMARY_RGOT_ANALYSIS_DIR", ""),
    "rgot_summary_csv": os.environ.get("SUMMARY_RGOT_SUMMARY_CSV", ""),
    "rgot_event_jsonl": os.environ.get("SUMMARY_RGOT_EVENT_JSONL", ""),
    "pid": os.environ.get("SUMMARY_PID", ""),
    "start_time": os.environ.get("SUMMARY_START_TIME", ""),
    "end_time": os.environ.get("SUMMARY_END_TIME", ""),
    "exit_code": os.environ.get("SUMMARY_EXIT_CODE", ""),
    "event_count": os.environ.get("SUMMARY_EVENT_COUNT", ""),
    "candidate_blocks": os.environ.get("SUMMARY_CANDIDATE_BLOCKS", ""),
    "trigger_blocks": os.environ.get("SUMMARY_TRIGGER_BLOCKS", ""),
    "owneralt_overlap_events": os.environ.get("SUMMARY_OWNERALT_OVERLAP_EVENTS", ""),
    "graph_assoc_overlap_events": os.environ.get("SUMMARY_GRAPH_ASSOC_OVERLAP_EVENTS", ""),
    "not_explained_by_buffer_or_reentry_cases": os.environ.get("SUMMARY_UNEXPLAINED_CASES", ""),
    "notes": os.environ.get("SUMMARY_NOTES", ""),
}
with tmp.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow(row)
os.replace(tmp, target)
PY
}

update_plan_status() {
  local status="$1"
  shift || true
  local extras=(
    "seq_name=${SEQ_NAME}"
    "seq_id=${SEQ_ID}"
    "experiment_name=${TRACK_EXP_NAME}"
    "tracker_output_dir=${TRACK_RESULTS_DIR}"
    "rgot_analysis_dir=${RGOT_ANALYSIS_DIR}"
    "rgot_analysis_only=1"
    "haca_mode=haca_v1"
    "haca_disable_background=1"
    "laplace_primary_only=1"
    "laplace_decay_scales=${LAPLACE_DECAY_SCALES}"
    "laplace_min_history=${LAPLACE_MIN_HISTORY}"
    "laplace_proto_mode=${LAPLACE_PROTO_MODE}"
    "rgot_top_k=${RGOT_TOP_K}"
    "rgot_row_margin=${RGOT_ROW_MARGIN}"
    "rgot_col_margin=${RGOT_COL_MARGIN}"
    "rgot_max_rows=${RGOT_MAX_ROWS}"
    "rgot_max_cols=${RGOT_MAX_COLS}"
  )
  if [[ $# -gt 0 ]]; then
    extras+=("$@")
  fi
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/upsert_experiment_plan.py" \
    --csv "${PLAN_CSV}" \
    --key "${PLAN_KEY}" \
    --status "${status}" \
    --kind analysis \
    --script "${SCRIPT_NAME}" \
    --dataset MOT20 \
    --split val_half \
    --tracker-family BoT-SORT \
    --variant "${VARIANT_NAME}" \
    --run-root "${RUN_ROOT}" \
    --summary-csv "${SUMMARY_CSV}" \
    --checkpoint "${HACA_NPZ}" \
    --log-path "${LOG_PATH}" \
    --extra "${extras[@]}"
}

append_registry() {
  local status="$1"
  local notes="$2"
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/append_experiment_record.py" \
    --csv "${REGISTRY_CSV}" \
    --kind analysis \
    --status "${status}" \
    --script "${SCRIPT_NAME}" \
    --dataset MOT20 \
    --split val_half \
    --tracker-family BoT-SORT \
    --variant "${VARIANT_NAME}" \
    --run-root "${RUN_ROOT}" \
    --summary-csv "${SUMMARY_CSV}" \
    --checkpoint "${HACA_NPZ}" \
    --log-path "${LOG_PATH}" \
    --notes "${notes}" \
    --extra \
      seq_name="${SEQ_NAME}" \
      seq_id="${SEQ_ID}" \
      experiment_name="${TRACK_EXP_NAME}" \
      tracker_output_dir="${TRACK_RESULTS_DIR}" \
      rgot_analysis_dir="${RGOT_ANALYSIS_DIR}" \
      rgot_analysis_only=1 \
      haca_mode=haca_v1 \
      haca_disable_background=1 \
      laplace_primary_only=1 \
      laplace_decay_scales="${LAPLACE_DECAY_SCALES}" \
      laplace_min_history="${LAPLACE_MIN_HISTORY}" \
      laplace_proto_mode="${LAPLACE_PROTO_MODE}" \
      rgot_top_k="${RGOT_TOP_K}" \
      rgot_row_margin="${RGOT_ROW_MARGIN}" \
      rgot_col_margin="${RGOT_COL_MARGIN}" \
      rgot_max_rows="${RGOT_MAX_ROWS}" \
      rgot_max_cols="${RGOT_MAX_COLS}"
}

finalize_run() {
  local rc="$1"
  set +e
  local status
  local plan_status
  local notes
  local end_time
  local event_count=""
  local candidate_blocks=""
  local trigger_blocks=""
  local owneralt_overlap_events=""
  local graph_assoc_overlap_events=""
  local unexplained_cases=""

  end_time="$(date --iso-8601=seconds)"
  if [[ -f "${RGOT_SUMMARY_CSV}" ]]; then
    event_count="$(read_csv_field "${RGOT_SUMMARY_CSV}" "event_count")"
    candidate_blocks="$(read_csv_field "${RGOT_SUMMARY_CSV}" "candidate_blocks")"
    trigger_blocks="$(read_csv_field "${RGOT_SUMMARY_CSV}" "trigger_blocks")"
    owneralt_overlap_events="$(read_csv_field "${RGOT_SUMMARY_CSV}" "owneralt_overlap_events")"
    graph_assoc_overlap_events="$(read_csv_field "${RGOT_SUMMARY_CSV}" "graph_assoc_overlap_events")"
    unexplained_cases="$(read_csv_field "${RGOT_SUMMARY_CSV}" "not_explained_by_buffer_or_reentry_cases")"
  fi

  if [[ "${rc}" -eq 0 ]]; then
    status="success"
    plan_status="completed"
    notes="RG-OT analysis-only smoke finished"
  elif [[ "${rc}" -eq 130 || "${rc}" -eq 143 ]]; then
    status="interrupted"
    plan_status="interrupted"
    notes="RG-OT analysis-only smoke interrupted"
  else
    status="failed"
    plan_status="failed"
    notes="RG-OT analysis-only smoke failed"
  fi

  write_summary "${status}" "${end_time}" "${rc}" "${event_count}" "${candidate_blocks}" "${trigger_blocks}" "${owneralt_overlap_events}" "${graph_assoc_overlap_events}" "${unexplained_cases}" "${notes}"
  append_registry "${status}" "${notes}"
  update_plan_status "${plan_status}" "exit_code=${rc}" "event_count=${event_count}" "candidate_blocks=${candidate_blocks}" "trigger_blocks=${trigger_blocks}" || true
}

on_exit() {
  local rc=$?
  trap - EXIT INT TERM
  set +e
  finalize_run "${rc}"
  exit "${rc}"
}

trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ ! -f "${HACA_NPZ}" ]]; then
  log "[error] missing HACA checkpoint: ${HACA_NPZ}"
  write_summary "failed" "" "2" "" "" "" "" "" "" "Missing HACA checkpoint"
  append_registry "failed" "Missing HACA checkpoint"
  update_plan_status failed "exit_code=2" "reason=missing_haca_checkpoint" || true
  exit 2
fi

write_summary "running" "" "" "" "" "" "" "" "" "RG-OT analysis-only smoke launched"
append_registry "running" "RG-OT analysis-only smoke launched"
update_plan_status running

CMD=(
  "${PYTHON_BIN}" -u tools/track.py "${DATA_ROOT}/MOT20"
  --benchmark MOT20
  --eval train
  --seq-ids "${SEQ_ID}"
  -f "${EXP_FILE}"
  -c "${CKPT}"
  --with-reid
  --fast-reid-config "${REID_CFG}"
  --fast-reid-weights "${REID_WTS}"
  --experiment-name "${TRACK_EXP_NAME}"
  --laplace-assoc
  --laplace-assoc-mode haca_v1
  --laplace-decay-scales ${LAPLACE_DECAY_SCALES}
  --laplace-min-history "${LAPLACE_MIN_HISTORY}"
  --laplace-proto-mode "${LAPLACE_PROTO_MODE}"
  --laplace-primary-only
  --laplace-haca-checkpoint "${HACA_NPZ}"
  --laplace-haca-no-background
  --rgot-enable
  --rgot-analysis-only
  --rgot-analysis-dir "${RGOT_ANALYSIS_DIR}"
  --rgot-top-k "${RGOT_TOP_K}"
  --rgot-row-margin "${RGOT_ROW_MARGIN}"
  --rgot-col-margin "${RGOT_COL_MARGIN}"
  --rgot-max-rows "${RGOT_MAX_ROWS}"
  --rgot-max-cols "${RGOT_MAX_COLS}"
)

printf "%s\n" "${CMD[@]}" > "${RUN_ROOT}/command.txt"
log "[start] RG-OT MOT20 smoke"
log "[run_root] ${RUN_ROOT}"
log "[command] ${CMD[*]}"

(
  cd "${BOT_ROOT}"
  "${CMD[@]}"
)

if [[ ! -f "${RGOT_SUMMARY_CSV}" ]]; then
  log "[error] missing RG-OT summary: ${RGOT_SUMMARY_CSV}"
  exit 3
fi
if [[ ! -f "${RGOT_EVENT_JSONL}" ]]; then
  log "[error] missing RG-OT event log: ${RGOT_EVENT_JSONL}"
  exit 4
fi

log "[done] RG-OT smoke complete"
log "[rgot_summary] ${RGOT_SUMMARY_CSV}"
log "[rgot_events] ${RGOT_EVENT_JSONL}"
