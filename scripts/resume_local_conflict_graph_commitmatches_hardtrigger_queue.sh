#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
GENERIC_RUNNER="${REPO_ROOT}/scripts/run_local_conflict_graph_commitmatches_hardtrigger_oracle_generic.sh"
QUEUE_ROOT="${1:?usage: resume_local_conflict_graph_commitmatches_hardtrigger_queue.sh <queue_root> [queue_name]}"
QUEUE_NAME="${2:-$(basename "${QUEUE_ROOT}")}"
PLAN_CSV="${REPO_ROOT}/outputs/experiment_plan.csv"
SUMMARY_CSV="${QUEUE_ROOT}/summary.csv"
CURRENT_JOB_TXT="${QUEUE_ROOT}/current_job.txt"
QUEUE_LOG="${QUEUE_ROOT}/queue.log"

if [[ ! -f "${SUMMARY_CSV}" ]]; then
  echo "Missing queue summary: ${SUMMARY_CSV}" >&2
  exit 2
fi
if [[ ! -f "${GENERIC_RUNNER}" ]]; then
  echo "Missing runner: ${GENERIC_RUNNER}" >&2
  exit 2
fi

touch "${QUEUE_LOG}"

update_summary_row() {
  local summary_csv="$1"
  local job_key="$2"
  local status="$3"
  local run_root="$4"
  local started_at="${5:-}"
  local finished_at="${6:-}"
  "${PYTHON_BIN}" - "${summary_csv}" "${job_key}" "${status}" "${run_root}" "${started_at}" "${finished_at}" <<'PY'
import csv
import sys
from pathlib import Path

summary_csv = Path(sys.argv[1])
job_key = sys.argv[2]
status = sys.argv[3]
run_root = sys.argv[4]
started_at = sys.argv[5]
finished_at = sys.argv[6]
run_summary = Path(run_root) / "summary.csv"
metrics = {}
if run_summary.is_file():
    with run_summary.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if rows:
        metrics = rows[-1]

with summary_csv.open("r", encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))

fieldnames = [
    "job_index",
    "job_key",
    "phase",
    "run_name",
    "detector_filter",
    "val_sequences",
    "graph_topk",
    "graph_min_detections",
    "graph_min_committed_matches",
    "out_dir",
    "plan_key",
    "status",
    "HOTA",
    "AssA",
    "IDF1",
    "MOTA",
    "IDSW",
    "started_at",
    "finished_at",
    "notes",
]
for row in rows:
    if row.get("job_key") != job_key:
        continue
    row["status"] = status
    if run_root:
        row["out_dir"] = run_root
    if started_at:
        row["started_at"] = started_at
    if finished_at:
        row["finished_at"] = finished_at
    for key in ("HOTA", "AssA", "IDF1", "MOTA", "IDSW"):
        if key in metrics and metrics[key] != "":
            row[key] = metrics[key]

with summary_csv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
PY
}

sync_existing_statuses() {
  "${PYTHON_BIN}" - "${SUMMARY_CSV}" <<'PY'
import csv
import sys
from pathlib import Path

summary_csv = Path(sys.argv[1])
with summary_csv.open("r", encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))
for row in rows:
    print("\t".join([
        row["job_key"],
        row["phase"],
        row["run_name"],
        row["detector_filter"],
        row["val_sequences"],
        row["graph_topk"],
        row["graph_min_detections"],
        row["graph_min_committed_matches"],
        row["out_dir"],
        row["status"],
        row.get("notes", ""),
    ]))
PY
}

while IFS=$'\t' read -r JOB_KEY PHASE RUN_NAME DETECTOR_FILTER VAL_SEQUENCES TOPK MIN_DETECTIONS MIN_COMMITTED_MATCHES RUN_ROOT STATUS NOTES; do
  RUN_SUMMARY="${RUN_ROOT}/summary.csv"
  PLAN_KEY="queue:${QUEUE_NAME}:${JOB_KEY}"
  if [[ -f "${RUN_SUMMARY}" ]]; then
    RUN_STATUS="$("${PYTHON_BIN}" - "${RUN_SUMMARY}" <<'PY'
import csv, sys
from pathlib import Path
path = Path(sys.argv[1])
with path.open("r", encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))
row = rows[-1] if rows else {}
print(row.get("status", ""))
PY
)"
    if [[ "${RUN_STATUS}" == "ok" ]]; then
      update_summary_row "${SUMMARY_CSV}" "${JOB_KEY}" "completed" "${RUN_ROOT}" "" ""
      "${PYTHON_BIN}" "${REPO_ROOT}/scripts/upsert_experiment_plan.py" \
        --csv "${PLAN_CSV}" \
        --key "${PLAN_KEY}" \
        --status completed \
        --kind eval \
        --script "scripts/run_local_conflict_graph_commitmatches_hardtrigger_oracle_generic.sh" \
        --dataset "MOT17" \
        --split "${PHASE}" \
        --tracker-family "ByteTrack" \
        --variant "${RUN_NAME}" \
        --tag "local_conflict_graph_mainline" \
        --run-root "${RUN_ROOT}" \
        --summary-csv "${RUN_ROOT}/summary.csv" \
        --log-path "${RUN_ROOT}/run.log" \
        --notes "${NOTES}" \
        --extra "queue_name=${QUEUE_NAME}" "queue_root=${QUEUE_ROOT}" "graph_topk=${TOPK}" "graph_min_detections=${MIN_DETECTIONS}" "graph_min_committed_matches=${MIN_COMMITTED_MATCHES}" "detector_filter=${DETECTOR_FILTER}" "val_sequences=${VAL_SEQUENCES}" >/dev/null
      continue
    elif [[ "${RUN_STATUS}" == "failed" ]]; then
      update_summary_row "${SUMMARY_CSV}" "${JOB_KEY}" "failed" "${RUN_ROOT}" "" ""
      "${PYTHON_BIN}" "${REPO_ROOT}/scripts/upsert_experiment_plan.py" \
        --csv "${PLAN_CSV}" \
        --key "${PLAN_KEY}" \
        --status failed \
        --kind eval \
        --script "scripts/run_local_conflict_graph_commitmatches_hardtrigger_oracle_generic.sh" \
        --dataset "MOT17" \
        --split "${PHASE}" \
        --tracker-family "ByteTrack" \
        --variant "${RUN_NAME}" \
        --tag "local_conflict_graph_mainline" \
        --run-root "${RUN_ROOT}" \
        --summary-csv "${RUN_ROOT}/summary.csv" \
        --log-path "${RUN_ROOT}/run.log" \
        --notes "${NOTES}" \
        --extra "queue_name=${QUEUE_NAME}" "queue_root=${QUEUE_ROOT}" "graph_topk=${TOPK}" "graph_min_detections=${MIN_DETECTIONS}" "graph_min_committed_matches=${MIN_COMMITTED_MATCHES}" "detector_filter=${DETECTOR_FILTER}" "val_sequences=${VAL_SEQUENCES}" >/dev/null
    fi
  fi
done < <(sync_existing_statuses)

while IFS=$'\t' read -r JOB_KEY PHASE RUN_NAME DETECTOR_FILTER VAL_SEQUENCES TOPK MIN_DETECTIONS MIN_COMMITTED_MATCHES RUN_ROOT STATUS NOTES; do
  if [[ "${STATUS}" == "completed" ]]; then
    continue
  fi
  PLAN_KEY="queue:${QUEUE_NAME}:${JOB_KEY}"
  STARTED_AT="$(date --iso-8601=seconds)"
  printf '%s\n' "${JOB_KEY}" > "${CURRENT_JOB_TXT}"
  update_summary_row "${SUMMARY_CSV}" "${JOB_KEY}" "running" "${RUN_ROOT}" "${STARTED_AT}" ""
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/upsert_experiment_plan.py" \
    --csv "${PLAN_CSV}" \
    --key "${PLAN_KEY}" \
    --status running \
    --kind eval \
    --script "scripts/run_local_conflict_graph_commitmatches_hardtrigger_oracle_generic.sh" \
    --dataset "MOT17" \
    --split "${PHASE}" \
    --tracker-family "ByteTrack" \
    --variant "${RUN_NAME}" \
    --tag "local_conflict_graph_mainline" \
    --run-root "${RUN_ROOT}" \
    --summary-csv "${RUN_ROOT}/summary.csv" \
    --log-path "${RUN_ROOT}/run.log" \
    --notes "${NOTES}" \
    --extra "queue_name=${QUEUE_NAME}" "queue_root=${QUEUE_ROOT}" "graph_topk=${TOPK}" "graph_min_detections=${MIN_DETECTIONS}" "graph_min_committed_matches=${MIN_COMMITTED_MATCHES}" "detector_filter=${DETECTOR_FILTER}" "val_sequences=${VAL_SEQUENCES}" | tee -a "${QUEUE_LOG}"

  JOB_STATUS="completed"
  if ! bash "${GENERIC_RUNNER}" "${RUN_ROOT}" "${RUN_NAME}" \
    "/gemini/code/FMtrack-main/FM-Track/outputs/competition_assoc_base_reid_da_proxy0213_hybriddumpfix/labeled_replay_top8.groups.jsonl" \
    "${TOPK}" "${MIN_DETECTIONS}" "${MIN_COMMITTED_MATCHES}" "${DETECTOR_FILTER}" "${VAL_SEQUENCES}" "${NOTES}" 2>&1 | tee -a "${QUEUE_LOG}"; then
    JOB_STATUS="failed"
  fi

  FINISHED_AT="$(date --iso-8601=seconds)"
  update_summary_row "${SUMMARY_CSV}" "${JOB_KEY}" "${JOB_STATUS}" "${RUN_ROOT}" "" "${FINISHED_AT}"
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/upsert_experiment_plan.py" \
    --csv "${PLAN_CSV}" \
    --key "${PLAN_KEY}" \
    --status "${JOB_STATUS}" \
    --kind eval \
    --script "scripts/run_local_conflict_graph_commitmatches_hardtrigger_oracle_generic.sh" \
    --dataset "MOT17" \
    --split "${PHASE}" \
    --tracker-family "ByteTrack" \
    --variant "${RUN_NAME}" \
    --tag "local_conflict_graph_mainline" \
    --run-root "${RUN_ROOT}" \
    --summary-csv "${RUN_ROOT}/summary.csv" \
    --log-path "${RUN_ROOT}/run.log" \
    --notes "${NOTES}" \
    --extra "queue_name=${QUEUE_NAME}" "queue_root=${QUEUE_ROOT}" "graph_topk=${TOPK}" "graph_min_detections=${MIN_DETECTIONS}" "graph_min_committed_matches=${MIN_COMMITTED_MATCHES}" "detector_filter=${DETECTOR_FILTER}" "val_sequences=${VAL_SEQUENCES}" | tee -a "${QUEUE_LOG}"
done < <(sync_existing_statuses)

: > "${CURRENT_JOB_TXT}"
echo "[queue-resume] completed queue_root=${QUEUE_ROOT} queue_name=${QUEUE_NAME}" | tee -a "${QUEUE_LOG}"
