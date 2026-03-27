#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
GENERIC_RUNNER="${REPO_ROOT}/scripts/run_local_conflict_graph_commitmatches_hardtrigger_oracle_generic.sh"
ORACLE_JSONL="${ORACLE_JSONL:-${REPO_ROOT}/outputs/competition_assoc_base_reid_da_proxy0213_hybriddumpfix/labeled_replay_top8.groups.jsonl}"
QUEUE_ROOT="${1:-${REPO_ROOT}/outputs/local_conflict_graph_commitmatches_hardtrigger_next12h_$(date +%Y%m%d_%H%M%S)}"
QUEUE_NAME="${2:-local_conflict_graph_commitmatches_hardtrigger_next12h}"
CONTINUE_ON_FAILURE="${QUEUE_CONTINUE_ON_FAILURE:-1}"
PLAN_CSV="${REPO_ROOT}/outputs/experiment_plan.csv"

if [[ ! -f "${GENERIC_RUNNER}" ]]; then
  echo "Missing runner: ${GENERIC_RUNNER}" >&2
  exit 2
fi
if [[ ! -f "${ORACLE_JSONL}" ]]; then
  echo "Missing oracle jsonl: ${ORACLE_JSONL}" >&2
  exit 2
fi

mkdir -p "${QUEUE_ROOT}/runs"

SUMMARY_CSV="${QUEUE_ROOT}/summary.csv"
JOBS_TSV="${QUEUE_ROOT}/jobs.tsv"
MANIFEST_JSON="${QUEUE_ROOT}/queue_manifest.json"
CURRENT_JOB_TXT="${QUEUE_ROOT}/current_job.txt"

"${PYTHON_BIN}" - "${SUMMARY_CSV}" "${JOBS_TSV}" "${MANIFEST_JSON}" "${QUEUE_NAME}" "${QUEUE_ROOT}" "${ORACLE_JSONL}" <<'PY'
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

summary_csv = Path(sys.argv[1])
jobs_tsv = Path(sys.argv[2])
manifest_json = Path(sys.argv[3])
queue_name = sys.argv[4]
queue_root = sys.argv[5]
oracle_jsonl = sys.argv[6]
created_at = datetime.now().astimezone().isoformat(timespec="seconds")

jobs = []

proxy_phase = [
    ("proxy_frcnn", "FRCNN", "MOT17-02,MOT17-13", 8, 2, 2),
    ("proxy_frcnn", "FRCNN", "MOT17-02,MOT17-13", 8, 2, 3),
    ("proxy_frcnn", "FRCNN", "MOT17-02,MOT17-13", 8, 3, 2),
    ("proxy_frcnn", "FRCNN", "MOT17-02,MOT17-13", 8, 4, 2),
]

frcnn_full_phase = [
    ("full_frcnn", "FRCNN", "ALL", 8, 2, 2),
    ("full_frcnn", "FRCNN", "ALL", 8, 2, 3),
    ("full_frcnn", "FRCNN", "ALL", 8, 3, 2),
    ("full_frcnn", "FRCNN", "ALL", 8, 4, 2),
]

public_phase = []
for min_detections in (2, 3, 4, 5):
    for min_committed_matches in (2, 3, 4, 5):
        public_phase.append(("full_public", "ALL", "ALL", 8, min_detections, min_committed_matches))

phase_rows = proxy_phase + frcnn_full_phase + public_phase

for idx, (phase, detector_filter, val_sequences, topk, min_detections, min_committed_matches) in enumerate(phase_rows, start=1):
    run_name = (
        f"{queue_name}_{phase}_topk{topk}_md{min_detections}_mm{min_committed_matches}"
    )
    job_key = f"{idx:02d}_{phase}_topk{topk}_md{min_detections}_mm{min_committed_matches}"
    notes = (
        "local conflict graph matched-edge commit oracle with hard cluster trigger; "
        f"phase={phase}; detector_filter={detector_filter}; val_sequences={val_sequences}; "
        f"topk={topk}; min_detections={min_detections}; min_committed_matches={min_committed_matches}"
    )
    jobs.append(
        {
            "job_index": str(idx),
            "job_key": job_key,
            "phase": phase,
            "run_name": run_name,
            "detector_filter": detector_filter,
            "val_sequences": val_sequences,
            "graph_topk": str(topk),
            "graph_min_detections": str(min_detections),
            "graph_min_committed_matches": str(min_committed_matches),
            "out_dir": str(Path(queue_root) / "runs" / job_key),
            "plan_key": f"queue:{queue_name}:{job_key}",
            "status": "queued",
            "HOTA": "",
            "AssA": "",
            "IDF1": "",
            "MOTA": "",
            "IDSW": "",
            "started_at": "",
            "finished_at": "",
            "notes": notes,
        }
    )

summary_csv.parent.mkdir(parents=True, exist_ok=True)
with summary_csv.open("w", encoding="utf-8", newline="") as f:
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
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(jobs)

with jobs_tsv.open("w", encoding="utf-8", newline="") as f:
    for job in jobs:
        row = [
            job["job_key"],
            job["phase"],
            job["run_name"],
            job["detector_filter"],
            job["val_sequences"],
            job["graph_topk"],
            job["graph_min_detections"],
            job["graph_min_committed_matches"],
            job["notes"],
        ]
        f.write("\t".join(row) + "\n")

manifest = {
    "queue_name": queue_name,
    "queue_root": queue_root,
    "created_at": created_at,
    "oracle_jsonl": oracle_jsonl,
    "job_count": len(jobs),
    "summary_csv": str(summary_csv),
    "jobs_tsv": str(jobs_tsv),
    "phases": {
        "proxy_frcnn": len(proxy_phase),
        "full_frcnn": len(frcnn_full_phase),
        "full_public": len(public_phase),
    },
    "notes": "24-job overnight queue for local conflict graph hard-trigger oracle surface mapping.",
}
manifest_json.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
PY

update_summary_row() {
  local job_key="$1"
  local status="$2"
  local run_root="$3"
  local started_at="${4:-}"
  local finished_at="${5:-}"
  "${PYTHON_BIN}" - "${SUMMARY_CSV}" "${job_key}" "${status}" "${run_root}" "${started_at}" "${finished_at}" <<'PY'
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
    reader = csv.DictReader(f)
    rows = list(reader)

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
    row["out_dir"] = run_root or row.get("out_dir", "")
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

while IFS=$'\t' read -r JOB_KEY PHASE RUN_NAME DETECTOR_FILTER VAL_SEQUENCES TOPK MIN_DETECTIONS MIN_COMMITTED_MATCHES NOTES; do
  RUN_ROOT="${QUEUE_ROOT}/runs/${JOB_KEY}"
  PLAN_KEY="queue:${QUEUE_NAME}:${JOB_KEY}"
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/upsert_experiment_plan.py" \
    --csv "${PLAN_CSV}" \
    --key "${PLAN_KEY}" \
    --status queued \
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
    --extra "queue_name=${QUEUE_NAME}" "queue_root=${QUEUE_ROOT}" "oracle_group_jsonl=${ORACLE_JSONL}" "graph_topk=${TOPK}" "graph_min_detections=${MIN_DETECTIONS}" "graph_min_committed_matches=${MIN_COMMITTED_MATCHES}" "detector_filter=${DETECTOR_FILTER}" "val_sequences=${VAL_SEQUENCES}"
done < "${JOBS_TSV}"

while IFS=$'\t' read -r JOB_KEY PHASE RUN_NAME DETECTOR_FILTER VAL_SEQUENCES TOPK MIN_DETECTIONS MIN_COMMITTED_MATCHES NOTES; do
  RUN_ROOT="${QUEUE_ROOT}/runs/${JOB_KEY}"
  PLAN_KEY="queue:${QUEUE_NAME}:${JOB_KEY}"
  STARTED_AT="$(date --iso-8601=seconds)"
  printf '%s\n' "${JOB_KEY}" > "${CURRENT_JOB_TXT}"
  update_summary_row "${JOB_KEY}" "running" "${RUN_ROOT}" "${STARTED_AT}" ""
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
    --extra "queue_name=${QUEUE_NAME}" "queue_root=${QUEUE_ROOT}" "oracle_group_jsonl=${ORACLE_JSONL}" "graph_topk=${TOPK}" "graph_min_detections=${MIN_DETECTIONS}" "graph_min_committed_matches=${MIN_COMMITTED_MATCHES}" "detector_filter=${DETECTOR_FILTER}" "val_sequences=${VAL_SEQUENCES}"

  JOB_STATUS="completed"
  if ! bash "${GENERIC_RUNNER}" "${RUN_ROOT}" "${RUN_NAME}" "${ORACLE_JSONL}" "${TOPK}" "${MIN_DETECTIONS}" "${MIN_COMMITTED_MATCHES}" "${DETECTOR_FILTER}" "${VAL_SEQUENCES}" "${NOTES}"; then
    JOB_STATUS="failed"
  fi

  FINISHED_AT="$(date --iso-8601=seconds)"
  update_summary_row "${JOB_KEY}" "${JOB_STATUS}" "${RUN_ROOT}" "" "${FINISHED_AT}"
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
    --extra "queue_name=${QUEUE_NAME}" "queue_root=${QUEUE_ROOT}" "oracle_group_jsonl=${ORACLE_JSONL}" "graph_topk=${TOPK}" "graph_min_detections=${MIN_DETECTIONS}" "graph_min_committed_matches=${MIN_COMMITTED_MATCHES}" "detector_filter=${DETECTOR_FILTER}" "val_sequences=${VAL_SEQUENCES}"

  if [[ "${JOB_STATUS}" == "failed" && "${CONTINUE_ON_FAILURE}" != "1" ]]; then
    echo "[queue] stopping on failure at ${JOB_KEY}" >&2
    break
  fi
done < "${JOBS_TSV}"

: > "${CURRENT_JOB_TXT}"
echo "[queue] completed queue_root=${QUEUE_ROOT} queue_name=${QUEUE_NAME}"
