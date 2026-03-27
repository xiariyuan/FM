#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
QUEUE_ROOT="${1:-${REPO_ROOT}/outputs/local_conflict_graph_learned_commit_next12h_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${QUEUE_ROOT}"
SUMMARY_CSV="${QUEUE_ROOT}/summary.csv"
QUEUE_LOG="${QUEUE_ROOT}/queue.log"

init_summary() {
  "${PYTHON_BIN}" - "${SUMMARY_CSV}" "${QUEUE_ROOT}" <<'PY'
import csv
import sys
from pathlib import Path

summary_csv = Path(sys.argv[1])
queue_root = Path(sys.argv[2])
rows = [
    {"step": "01_stage1", "script": "scripts/run_local_conflict_commit_stage1.sh", "status": "pending", "out_dir": str(queue_root / "01_stage1"), "checkpoint": "", "HOTA": "", "AssA": "", "IDF1": "", "MOTA": "", "IDSW": "", "notes": "train stage1 learned commit"},
    {"step": "02_proxy_eval", "script": "scripts/run_local_conflict_graph_learned_commit_proxy0213.sh", "status": "pending", "out_dir": str(queue_root / "02_proxy_eval"), "checkpoint": "", "HOTA": "", "AssA": "", "IDF1": "", "MOTA": "", "IDSW": "", "notes": "proxy0213 learned commit eval"},
    {"step": "03_frcnn_md2_mm2", "script": "scripts/run_local_conflict_graph_learned_commit_generic.sh", "status": "pending", "out_dir": str(queue_root / "03_frcnn_md2_mm2"), "checkpoint": "", "HOTA": "", "AssA": "", "IDF1": "", "MOTA": "", "IDSW": "", "notes": "full FRCNN md2/mm2"},
    {"step": "04_frcnn_md2_mm3", "script": "scripts/run_local_conflict_graph_learned_commit_generic.sh", "status": "pending", "out_dir": str(queue_root / "04_frcnn_md2_mm3"), "checkpoint": "", "HOTA": "", "AssA": "", "IDF1": "", "MOTA": "", "IDSW": "", "notes": "full FRCNN md2/mm3"},
    {"step": "05_frcnn_md3_mm2", "script": "scripts/run_local_conflict_graph_learned_commit_generic.sh", "status": "pending", "out_dir": str(queue_root / "05_frcnn_md3_mm2"), "checkpoint": "", "HOTA": "", "AssA": "", "IDF1": "", "MOTA": "", "IDSW": "", "notes": "full FRCNN md3/mm2"},
    {"step": "06_frcnn_md4_mm2", "script": "scripts/run_local_conflict_graph_learned_commit_generic.sh", "status": "pending", "out_dir": str(queue_root / "06_frcnn_md4_mm2"), "checkpoint": "", "HOTA": "", "AssA": "", "IDF1": "", "MOTA": "", "IDSW": "", "notes": "full FRCNN md4/mm2"},
]
with summary_csv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
PY
}

update_summary_status() {
  local step="$1"
  local status="$2"
  local checkpoint="${3:-}"
  "${PYTHON_BIN}" - "${SUMMARY_CSV}" "${step}" "${status}" "${checkpoint}" <<'PY'
import csv
import sys
from pathlib import Path

summary_csv = Path(sys.argv[1])
step = sys.argv[2]
status = sys.argv[3]
checkpoint = sys.argv[4]
with summary_csv.open("r", encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))
fieldnames = list(rows[0].keys())
for row in rows:
    if row.get("step") == step:
        row["status"] = status
        if checkpoint:
            row["checkpoint"] = checkpoint
with summary_csv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
PY
}

merge_run_summary() {
  local step="$1"
  local run_dir="$2"
  local checkpoint="${3:-}"
  "${PYTHON_BIN}" - "${SUMMARY_CSV}" "${step}" "${run_dir}" "${checkpoint}" <<'PY'
import csv
import sys
from pathlib import Path

summary_csv = Path(sys.argv[1])
step = sys.argv[2]
run_dir = Path(sys.argv[3])
checkpoint = sys.argv[4]
run_summary = run_dir / "summary.csv"
merged = {}
if run_summary.is_file():
    with run_summary.open("r", encoding="utf-8", newline="") as f:
        merged = next(csv.DictReader(f), {})
with summary_csv.open("r", encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))
fieldnames = list(rows[0].keys())
for key in ("HOTA", "AssA", "IDF1", "MOTA", "IDSW", "status"):
    if key in merged and key not in fieldnames:
        fieldnames.append(key)
for row in rows:
    if row.get("step") != step:
        continue
    row["out_dir"] = str(run_dir)
    if checkpoint:
        row["checkpoint"] = checkpoint
    for key in ("HOTA", "AssA", "IDF1", "MOTA", "IDSW", "status"):
        if key in merged:
            row[key] = merged.get(key, row.get(key, ""))
with summary_csv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
PY
}

mark_remaining_skipped() {
  local from_step="$1"
  "${PYTHON_BIN}" - "${SUMMARY_CSV}" "${from_step}" <<'PY'
import csv
import sys
from pathlib import Path

summary_csv = Path(sys.argv[1])
from_step = sys.argv[2]
with summary_csv.open("r", encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))
fieldnames = list(rows[0].keys())
found = False
for row in rows:
    if row.get("step") == from_step:
        found = True
        continue
    if found and row.get("status") == "pending":
        row["status"] = "skipped"
with summary_csv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
PY
}

run_step() {
  local step="$1"
  shift
  update_summary_status "${step}" "running" "${CKPT_PATH:-}"
  "$@"
}

init_summary
{
  echo "[queue] root=${QUEUE_ROOT}"
  date

  TRAIN_DIR="${QUEUE_ROOT}/01_stage1"
  update_summary_status "01_stage1" "running"
  if "${REPO_ROOT}/scripts/run_local_conflict_commit_stage1.sh" "${TRAIN_DIR}" "" "" 8 2 2 8 32 "MOT17-02-FRCNN" 12 128 16; then
    CKPT_PATH="${TRAIN_DIR}/best.pt"
    merge_run_summary "01_stage1" "${TRAIN_DIR}" "${CKPT_PATH}"
  else
    merge_run_summary "01_stage1" "${TRAIN_DIR}" ""
    mark_remaining_skipped "01_stage1"
    exit 1
  fi

  PROXY_DIR="${QUEUE_ROOT}/02_proxy_eval"
  update_summary_status "02_proxy_eval" "running" "${CKPT_PATH}"
  if "${REPO_ROOT}/scripts/run_local_conflict_graph_learned_commit_proxy0213.sh" "${PROXY_DIR}" "${CKPT_PATH}" 8 2 2 8 32; then
    merge_run_summary "02_proxy_eval" "${PROXY_DIR}" "${CKPT_PATH}"
  else
    merge_run_summary "02_proxy_eval" "${PROXY_DIR}" "${CKPT_PATH}"
  fi

  FRCNN_MD2_MM2_DIR="${QUEUE_ROOT}/03_frcnn_md2_mm2"
  update_summary_status "03_frcnn_md2_mm2" "running" "${CKPT_PATH}"
  if "${REPO_ROOT}/scripts/run_local_conflict_graph_learned_commit_generic.sh" "${FRCNN_MD2_MM2_DIR}" "local_conflict_graph_learned_commit_frcnn_md2_mm2" "${CKPT_PATH}" 8 2 2 8 32 "FRCNN" ""; then
    merge_run_summary "03_frcnn_md2_mm2" "${FRCNN_MD2_MM2_DIR}" "${CKPT_PATH}"
  else
    merge_run_summary "03_frcnn_md2_mm2" "${FRCNN_MD2_MM2_DIR}" "${CKPT_PATH}"
  fi

  FRCNN_MD2_MM3_DIR="${QUEUE_ROOT}/04_frcnn_md2_mm3"
  update_summary_status "04_frcnn_md2_mm3" "running" "${CKPT_PATH}"
  if "${REPO_ROOT}/scripts/run_local_conflict_graph_learned_commit_generic.sh" "${FRCNN_MD2_MM3_DIR}" "local_conflict_graph_learned_commit_frcnn_md2_mm3" "${CKPT_PATH}" 8 2 3 8 32 "FRCNN" ""; then
    merge_run_summary "04_frcnn_md2_mm3" "${FRCNN_MD2_MM3_DIR}" "${CKPT_PATH}"
  else
    merge_run_summary "04_frcnn_md2_mm3" "${FRCNN_MD2_MM3_DIR}" "${CKPT_PATH}"
  fi

  FRCNN_MD3_MM2_DIR="${QUEUE_ROOT}/05_frcnn_md3_mm2"
  update_summary_status "05_frcnn_md3_mm2" "running" "${CKPT_PATH}"
  if "${REPO_ROOT}/scripts/run_local_conflict_graph_learned_commit_generic.sh" "${FRCNN_MD3_MM2_DIR}" "local_conflict_graph_learned_commit_frcnn_md3_mm2" "${CKPT_PATH}" 8 3 2 8 32 "FRCNN" ""; then
    merge_run_summary "05_frcnn_md3_mm2" "${FRCNN_MD3_MM2_DIR}" "${CKPT_PATH}"
  else
    merge_run_summary "05_frcnn_md3_mm2" "${FRCNN_MD3_MM2_DIR}" "${CKPT_PATH}"
  fi

  FRCNN_MD4_MM2_DIR="${QUEUE_ROOT}/06_frcnn_md4_mm2"
  update_summary_status "06_frcnn_md4_mm2" "running" "${CKPT_PATH}"
  if "${REPO_ROOT}/scripts/run_local_conflict_graph_learned_commit_generic.sh" "${FRCNN_MD4_MM2_DIR}" "local_conflict_graph_learned_commit_frcnn_md4_mm2" "${CKPT_PATH}" 8 4 2 8 32 "FRCNN" ""; then
    merge_run_summary "06_frcnn_md4_mm2" "${FRCNN_MD4_MM2_DIR}" "${CKPT_PATH}"
  else
    merge_run_summary "06_frcnn_md4_mm2" "${FRCNN_MD4_MM2_DIR}" "${CKPT_PATH}"
  fi

  echo "[queue] completed root=${QUEUE_ROOT}"
  date
} >"${QUEUE_LOG}" 2>&1
