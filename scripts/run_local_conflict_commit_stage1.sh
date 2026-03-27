#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
GROUP_JSONL_DEFAULT="${REPO_ROOT}/outputs/competition_assoc_base_reid_da_proxy0213_hybriddumpfix/labeled_replay_top8.groups.jsonl"
CASES_CSV_DEFAULT="${REPO_ROOT}/outputs/competition_assoc_base_reid_da_proxy0213_hybriddumpfix/competition_cases/competition_cases.csv"

OUT_DIR="${1:-${REPO_ROOT}/outputs/local_conflict_commit_stage1_proxy0213_$(date +%Y%m%d_%H%M%S)}"
GROUP_JSONL="${2:-${GROUP_JSONL_DEFAULT}}"
CASES_CSV="${3:-${CASES_CSV_DEFAULT}}"
TOPK="${4:-8}"
MIN_DETECTIONS="${5:-2}"
MIN_COMMITTED_MATCHES="${6:-2}"
MAX_DETECTIONS="${7:-8}"
MAX_TRACKS="${8:-32}"
VAL_SEQUENCES="${9:-MOT17-02-FRCNN}"
EPOCHS="${10:-10}"
HIDDEN_DIM="${11:-128}"
BATCH_SIZE="${12:-8}"

if [[ ! -f "${GROUP_JSONL}" ]]; then
  echo "Missing group jsonl: ${GROUP_JSONL}" >&2
  exit 2
fi
if [[ ! -f "${CASES_CSV}" ]]; then
  echo "Missing cases csv: ${CASES_CSV}" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"
CLUSTER_OUT="${OUT_DIR}/cluster_commit_data"
RESULT_CSV="${OUT_DIR}/result.csv"
SUMMARY_CSV="${OUT_DIR}/summary.csv"
LOG_PATH="${OUT_DIR}/pipeline.log"

"${PYTHON_BIN}" - "${RESULT_CSV}" "${SUMMARY_CSV}" "${OUT_DIR}" "${GROUP_JSONL}" "${CASES_CSV}" "${VAL_SEQUENCES}" "${EPOCHS}" "${HIDDEN_DIM}" "${BATCH_SIZE}" <<'PY'
import csv
import sys
from pathlib import Path

result_csv = Path(sys.argv[1])
summary_csv = Path(sys.argv[2])
out_dir = sys.argv[3]
group_jsonl = sys.argv[4]
cases_csv = sys.argv[5]
val_sequences = sys.argv[6]
epochs = sys.argv[7]
hidden_dim = sys.argv[8]
batch_size = sys.argv[9]

row = {
    "exp_name": "local_conflict_commit_stage1",
    "group_jsonl": group_jsonl,
    "cases_csv": cases_csv,
    "data_jsonl": "",
    "checkpoint": "",
    "val_sequences": val_sequences,
    "epochs": epochs,
    "hidden_dim": hidden_dim,
    "batch_size": batch_size,
    "out_dir": out_dir,
    "status": "running",
}
fieldnames = list(row.keys())
for path in (result_csv, summary_csv):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)
PY

STATUS="failed"
if {
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/build_local_conflict_commit_dataset.py" \
    --group-jsonl "${GROUP_JSONL}" \
    --cases-csv "${CASES_CSV}" \
    --out-dir "${CLUSTER_OUT}" \
    --topk "${TOPK}" \
    --min-detections "${MIN_DETECTIONS}" \
    --min-committed-matches "${MIN_COMMITTED_MATCHES}" \
    --max-detections "${MAX_DETECTIONS}" \
    --max-tracks "${MAX_TRACKS}" &&
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/train_local_conflict_commit_stage1.py" \
    --data-jsonl "${CLUSTER_OUT}/cluster_examples.jsonl" \
    --out-dir "${OUT_DIR}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --hidden-dim "${HIDDEN_DIM}" \
    --val-sequences "${VAL_SEQUENCES}";
} >"${LOG_PATH}" 2>&1; then
  STATUS="success"
fi

"${PYTHON_BIN}" - "${RESULT_CSV}" "${SUMMARY_CSV}" "${OUT_DIR}" "${GROUP_JSONL}" "${CASES_CSV}" "${VAL_SEQUENCES}" "${STATUS}" <<'PY'
import csv
import json
import sys
from pathlib import Path

result_csv = Path(sys.argv[1])
summary_csv = Path(sys.argv[2])
out_dir = Path(sys.argv[3])
group_jsonl = sys.argv[4]
cases_csv = sys.argv[5]
val_sequences = sys.argv[6]
status = sys.argv[7]
best_ckpt = out_dir / "best.pt"
data_jsonl = out_dir / "cluster_commit_data" / "cluster_examples.jsonl"

row = {}
if summary_csv.is_file():
    with summary_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        row = next(reader, {})
row.update(
    {
        "exp_name": "local_conflict_commit_stage1",
        "group_jsonl": group_jsonl,
        "cases_csv": cases_csv,
        "data_jsonl": str(data_jsonl) if data_jsonl.is_file() else row.get("data_jsonl", ""),
        "checkpoint": str(best_ckpt) if best_ckpt.is_file() else row.get("checkpoint", ""),
        "val_sequences": row.get("val_sequences", val_sequences),
        "out_dir": str(out_dir),
        "status": "ok" if status == "success" and best_ckpt.is_file() else "failed",
    }
)
fieldnames = list(row.keys())
for path in (result_csv, summary_csv):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)
PY

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/append_experiment_record.py" \
  --csv "${REPO_ROOT}/outputs/experiment_registry.csv" \
  --kind train \
  --status "$( [[ "${STATUS}" == "success" ]] && echo success || echo failed )" \
  --script "scripts/run_local_conflict_commit_stage1.sh" \
  --dataset "MOT17" \
  --split "val0213_proxy" \
  --tracker-family "ByteTrack" \
  --variant "local_conflict_commit_stage1" \
  --tag "local_conflict_commit_mainline" \
  --run-root "${OUT_DIR}" \
  --summary-csv "${SUMMARY_CSV}" \
  --checkpoint "${OUT_DIR}/best.pt" \
  --log-path "${LOG_PATH}" \
  --notes "local conflict commit stage1 training on proxy0213 groups" \
  --extra "group_jsonl=${GROUP_JSONL}" "cases_csv=${CASES_CSV}" "graph_topk=${TOPK}" "graph_min_detections=${MIN_DETECTIONS}" "graph_min_committed_matches=${MIN_COMMITTED_MATCHES}" "graph_max_detections=${MAX_DETECTIONS}" "graph_max_tracks=${MAX_TRACKS}"

if ! "${PYTHON_BIN}" "${REPO_ROOT}/scripts/post_experiment_pro_bundle.py" \
  --run-root "${OUT_DIR}" \
  --tag "local_conflict_commit_stage1_bundle" \
  --label "local_conflict_commit_stage1" \
  --status "$( [[ "${STATUS}" == "success" ]] && echo ok || echo failed )"; then
  echo "[local-conflict-commit-stage1] warning: failed to build Pro review bundle for ${OUT_DIR}" >&2
fi

echo "[local-conflict-commit-stage1] status=${STATUS} out_dir=${OUT_DIR}"
