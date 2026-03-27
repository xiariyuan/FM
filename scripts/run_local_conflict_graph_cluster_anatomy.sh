#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
SRC_DIR="${1:-${REPO_ROOT}/outputs/competition_assoc_base_reid_da_proxy0213_hybriddumpfix}"
OUT_DIR="${2:-${REPO_ROOT}/outputs/local_conflict_graph_cluster_anatomy_$(date +%Y%m%d_%H%M%S)}"

CASES_CSV="${SRC_DIR}/competition_cases/competition_cases.csv"
GROUP_JSONL="${SRC_DIR}/labeled_replay_top8.groups.jsonl"
RESULT_CSV="${OUT_DIR}/result.csv"
SUMMARY_CSV="${OUT_DIR}/summary.csv"
LOG_PATH="${OUT_DIR}/run.log"

mkdir -p "${OUT_DIR}"

"${PYTHON_BIN}" - "${RESULT_CSV}" "${SUMMARY_CSV}" "${OUT_DIR}" <<'PY'
import csv
import sys
from pathlib import Path

result_csv = Path(sys.argv[1])
summary_csv = Path(sys.argv[2])
out_dir = sys.argv[3]
fieldnames = [
    "exp_name",
    "out_dir",
    "clusters",
    "frame_bipartite_clusters",
    "recoverable_clusters",
    "bridge_clusters",
    "recoverable_groups_total",
    "recoverable_groups_in_multi_detection_clusters",
    "recoverable_groups_multi_detection_share",
    "recoverable_cluster_avg_detections",
    "recoverable_cluster_avg_tracks",
    "recoverable_overlap_clusters",
    "recoverable_overlap_groups_in_multi_detection_clusters",
    "recoverable_overlap_groups_multi_detection_share",
    "recoverable_overlap_cluster_avg_detections",
    "recoverable_overlap_cluster_avg_tracks",
    "bridge_groups_total",
    "bridge_groups_in_multi_detection_clusters",
    "bridge_groups_multi_detection_share",
    "bridge_cluster_avg_detections",
    "bridge_cluster_avg_tracks",
    "bridge_overlap_clusters",
    "bridge_overlap_groups_in_multi_detection_clusters",
    "bridge_overlap_groups_multi_detection_share",
    "bridge_overlap_cluster_avg_detections",
    "bridge_overlap_cluster_avg_tracks",
    "status",
]
row = {
    "exp_name": "local_conflict_graph_cluster_anatomy",
    "out_dir": out_dir,
    "clusters": "",
    "frame_bipartite_clusters": "",
    "recoverable_clusters": "",
    "bridge_clusters": "",
    "recoverable_groups_total": "",
    "recoverable_groups_in_multi_detection_clusters": "",
    "recoverable_groups_multi_detection_share": "",
    "recoverable_cluster_avg_detections": "",
    "recoverable_cluster_avg_tracks": "",
    "recoverable_overlap_clusters": "",
    "recoverable_overlap_groups_in_multi_detection_clusters": "",
    "recoverable_overlap_groups_multi_detection_share": "",
    "recoverable_overlap_cluster_avg_detections": "",
    "recoverable_overlap_cluster_avg_tracks": "",
    "bridge_groups_total": "",
    "bridge_groups_in_multi_detection_clusters": "",
    "bridge_groups_multi_detection_share": "",
    "bridge_cluster_avg_detections": "",
    "bridge_cluster_avg_tracks": "",
    "bridge_overlap_clusters": "",
    "bridge_overlap_groups_in_multi_detection_clusters": "",
    "bridge_overlap_groups_multi_detection_share": "",
    "bridge_overlap_cluster_avg_detections": "",
    "bridge_overlap_cluster_avg_tracks": "",
    "status": "running",
}
for path in (result_csv, summary_csv):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)
PY

STATUS="failed"
if "${PYTHON_BIN}" "${REPO_ROOT}/scripts/analyze_local_conflict_graph_clusters.py" \
  --cases-csv "${CASES_CSV}" \
  --group-jsonl "${GROUP_JSONL}" \
  --out-dir "${OUT_DIR}" >"${LOG_PATH}" 2>&1; then
  STATUS="success"
fi

"${PYTHON_BIN}" - "${RESULT_CSV}" "${SUMMARY_CSV}" "${OUT_DIR}/summary.json" "${OUT_DIR}" "${STATUS}" <<'PY'
import csv
import json
import sys
from pathlib import Path

result_csv = Path(sys.argv[1])
summary_csv = Path(sys.argv[2])
summary_json = Path(sys.argv[3])
out_dir = sys.argv[4]
status = sys.argv[5]

row = {
    "exp_name": "local_conflict_graph_cluster_anatomy",
    "out_dir": out_dir,
    "clusters": "",
    "frame_bipartite_clusters": "",
    "recoverable_clusters": "",
    "bridge_clusters": "",
    "recoverable_groups_total": "",
    "recoverable_groups_in_multi_detection_clusters": "",
    "recoverable_groups_multi_detection_share": "",
    "recoverable_cluster_avg_detections": "",
    "recoverable_cluster_avg_tracks": "",
    "recoverable_overlap_clusters": "",
    "recoverable_overlap_groups_in_multi_detection_clusters": "",
    "recoverable_overlap_groups_multi_detection_share": "",
    "recoverable_overlap_cluster_avg_detections": "",
    "recoverable_overlap_cluster_avg_tracks": "",
    "bridge_groups_total": "",
    "bridge_groups_in_multi_detection_clusters": "",
    "bridge_groups_multi_detection_share": "",
    "bridge_cluster_avg_detections": "",
    "bridge_cluster_avg_tracks": "",
    "bridge_overlap_clusters": "",
    "bridge_overlap_groups_in_multi_detection_clusters": "",
    "bridge_overlap_groups_multi_detection_share": "",
    "bridge_overlap_cluster_avg_detections": "",
    "bridge_overlap_cluster_avg_tracks": "",
    "status": "ok" if status == "success" else "failed",
}
if summary_json.is_file():
    doc = json.loads(summary_json.read_text())
    for key in list(row.keys()):
        if key in doc:
            row[key] = str(doc[key])
fieldnames = list(row.keys())
for path in (result_csv, summary_csv):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)
PY

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/append_experiment_record.py" \
  --csv "${REPO_ROOT}/outputs/experiment_registry.csv" \
  --kind analysis \
  --status "$( [[ "${STATUS}" == "success" ]] && echo success || echo failed )" \
  --script "scripts/run_local_conflict_graph_cluster_anatomy.sh" \
  --dataset "MOT17" \
  --split "val0213_proxy" \
  --tracker-family "ByteTrack" \
  --variant "local_conflict_graph_cluster_anatomy" \
  --tag "local_conflict_graph_mainline" \
  --run-root "${OUT_DIR}" \
  --summary-csv "${SUMMARY_CSV}" \
  --log-path "${LOG_PATH}" \
  --notes "frame-local conflict graph cluster anatomy on proxy0213 top8 competition cases"

if ! "${PYTHON_BIN}" "${REPO_ROOT}/scripts/post_experiment_pro_bundle.py" \
  --run-root "${OUT_DIR}" \
  --tag "local_conflict_graph_cluster_bundle" \
  --label "local_conflict_graph_cluster_anatomy" \
  --status "$( [[ "${STATUS}" == "success" ]] && echo ok || echo failed )"; then
  echo "[local-conflict-graph] warning: failed to build Pro review bundle for ${OUT_DIR}" >&2
fi

echo "[local-conflict-graph] status=${STATUS} out_dir=${OUT_DIR}"
