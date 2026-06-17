#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"

OUT_DIR="${1:-${REPO_ROOT}/outputs/graph_assoc_commit_stage1_$(date +%Y%m%d_%H%M%S)}"
SOURCE_PATH="${2:-}"
TRAIN_SEQUENCES="${3:-}"
VAL_SEQUENCES="${4:-MOT20-05}"
EPOCHS="${5:-10}"
HIDDEN_DIM="${6:-128}"
BATCH_SIZE="${7:-8}"
TOPK="${8:-8}"
MIN_DETECTIONS="${9:-2}"
MIN_POSITIVE_MATCHES="${10:-1}"
MAX_DETECTIONS="${11:-8}"
MAX_TRACKS="${12:-32}"
DATASET="${13:-MOT20}"
SPLIT_PART="${14:-val_half}"

if [[ -z "${SOURCE_PATH}" ]]; then
  echo "Usage: $0 OUT_DIR SOURCE_PATH [TRAIN_SEQUENCES] [VAL_SEQUENCES] [EPOCHS] [HIDDEN_DIM] [BATCH_SIZE] [TOPK] [MIN_DETECTIONS] [MIN_POSITIVE_MATCHES] [MAX_DETECTIONS] [MAX_TRACKS] [DATASET] [SPLIT_PART]" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"
DATA_OUT="${OUT_DIR}/graph_assoc_commit_data"
RESULT_CSV="${OUT_DIR}/result.csv"
SUMMARY_CSV="${OUT_DIR}/summary.csv"
LOG_PATH="${OUT_DIR}/pipeline.log"

SOURCE_ARG=()
if [[ "${SOURCE_PATH}" == *.csv ]]; then
  SOURCE_ARG=(--source-manifest "${SOURCE_PATH}")
else
  SOURCE_ARG=(--rows-jsonl "${SOURCE_PATH}")
fi

"${PYTHON_BIN}" - "${RESULT_CSV}" "${SUMMARY_CSV}" "${OUT_DIR}" "${SOURCE_PATH}" "${TRAIN_SEQUENCES}" "${VAL_SEQUENCES}" "${EPOCHS}" "${HIDDEN_DIM}" "${BATCH_SIZE}" <<'PY'
import csv
import sys
from pathlib import Path

result_csv = Path(sys.argv[1])
summary_csv = Path(sys.argv[2])
out_dir = sys.argv[3]
source_path = sys.argv[4]
train_sequences = sys.argv[5]
val_sequences = sys.argv[6]
epochs = sys.argv[7]
hidden_dim = sys.argv[8]
batch_size = sys.argv[9]

row = {
    "exp_name": "graph_assoc_commit_stage1",
    "source_path": source_path,
    "data_jsonl": "",
    "checkpoint": "",
    "train_sequences": train_sequences,
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
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/build_graph_assoc_commit_dataset.py" \
    "${SOURCE_ARG[@]}" \
    --out-dir "${DATA_OUT}" \
    --dataset "${DATASET}" \
    --split-part "${SPLIT_PART}" \
    --topk "${TOPK}" \
    --min-detections "${MIN_DETECTIONS}" \
    --min-positive-matches "${MIN_POSITIVE_MATCHES}" \
    --max-detections "${MAX_DETECTIONS}" \
    --max-tracks "${MAX_TRACKS}" \
    --train-sequences "${TRAIN_SEQUENCES}" \
    --val-sequences "${VAL_SEQUENCES}" \
    --dataset-tag "graph_assoc_commit" \
    --feature-version "graph_assoc_v1" &&
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/train_local_conflict_commit_stage1.py" \
    --data-jsonl "${DATA_OUT}/cluster_examples.jsonl" \
    --out-dir "${OUT_DIR}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --hidden-dim "${HIDDEN_DIM}" \
    --train-sequences "${TRAIN_SEQUENCES}" \
    --val-sequences "${VAL_SEQUENCES}";
} >"${LOG_PATH}" 2>&1; then
  STATUS="success"
fi

"${PYTHON_BIN}" - "${RESULT_CSV}" "${SUMMARY_CSV}" "${OUT_DIR}" "${SOURCE_PATH}" "${TRAIN_SEQUENCES}" "${VAL_SEQUENCES}" "${STATUS}" <<'PY'
import csv
import sys
from pathlib import Path

result_csv = Path(sys.argv[1])
summary_csv = Path(sys.argv[2])
out_dir = Path(sys.argv[3])
source_path = sys.argv[4]
train_sequences = sys.argv[5]
val_sequences = sys.argv[6]
status = sys.argv[7]

best_ckpt = out_dir / "best.pt"
data_jsonl = out_dir / "graph_assoc_commit_data" / "cluster_examples.jsonl"
row = {}
if summary_csv.is_file():
    with summary_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        row = next(reader, {})
row.update(
    {
        "exp_name": "graph_assoc_commit_stage1",
        "source_path": source_path,
        "data_jsonl": str(data_jsonl) if data_jsonl.is_file() else row.get("data_jsonl", ""),
        "checkpoint": str(best_ckpt) if best_ckpt.is_file() else row.get("checkpoint", ""),
        "train_sequences": train_sequences,
        "val_sequences": val_sequences,
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
  --script "scripts/run_graph_assoc_commit_stage1.sh" \
  --dataset "${DATASET}" \
  --split "graph_assoc_commit_${SPLIT_PART}" \
  --tracker-family "BoT-SORT" \
  --variant "graph_assoc_commit_stage1" \
  --tag "graph_assoc_commit_stage1" \
  --run-root "${OUT_DIR}" \
  --summary-csv "${SUMMARY_CSV}" \
  --checkpoint "${OUT_DIR}/best.pt" \
  --log-path "${LOG_PATH}" \
  --notes "graph_assoc learned local conflict commit stage1" \
  --extra "source_path=${SOURCE_PATH}" "topk=${TOPK}" "min_detections=${MIN_DETECTIONS}" "min_positive_matches=${MIN_POSITIVE_MATCHES}" "max_detections=${MAX_DETECTIONS}" "max_tracks=${MAX_TRACKS}"

if ! "${PYTHON_BIN}" "${REPO_ROOT}/scripts/post_experiment_pro_bundle.py" \
  --run-root "${OUT_DIR}" \
  --tag "graph_assoc_commit_stage1_bundle" \
  --label "graph_assoc_commit_stage1" \
  --status "$( [[ "${STATUS}" == "success" ]] && echo ok || echo failed )"; then
  echo "[graph-assoc-commit-stage1] warning: failed to build Pro review bundle for ${OUT_DIR}" >&2
fi

echo "[graph-assoc-commit-stage1] status=${STATUS} out_dir=${OUT_DIR}"
