#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
BASE_CONFIG="${REPO_ROOT}/configs/experiments/bytetrack_fa_mot_mot17_v14_ctrl_base_reid_da_val0213.yaml"
BASE_CKPT="${REPO_ROOT}/outputs/paper_ctrl_mot17_val0213/bytetrack_fa_mot_mot17_v14_ctrl_base_reid_da_val0213/checkpoint_epoch_0.pth"
ORACLE_JSONL_DEFAULT="${REPO_ROOT}/outputs/competition_assoc_base_reid_da_proxy0213_hybriddumpfix/labeled_replay_top8.groups.jsonl"

OUT_DIR="${1:-${REPO_ROOT}/outputs/local_conflict_graph_commitmatches_oracle_proxy0213_$(date +%Y%m%d_%H%M%S)}"
ORACLE_JSONL="${2:-${ORACLE_JSONL_DEFAULT}}"
TOPK="${3:-8}"
MIN_DETECTIONS="${4:-2}"

if [[ ! -f "${BASE_CONFIG}" ]]; then
  echo "Missing config: ${BASE_CONFIG}" >&2
  exit 2
fi
if [[ ! -f "${BASE_CKPT}" ]]; then
  echo "Missing base checkpoint: ${BASE_CKPT}" >&2
  exit 2
fi
if [[ ! -f "${ORACLE_JSONL}" ]]; then
  echo "Missing oracle jsonl: ${ORACLE_JSONL}" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"

PROFILE_PATH="${OUT_DIR}/profile.json"
RESULT_CSV="${OUT_DIR}/result.csv"
SUMMARY_CSV="${OUT_DIR}/summary.csv"
LOG_PATH="${OUT_DIR}/run.log"
MANIFEST_PATH="${OUT_DIR}/run_manifest.json"

"${PYTHON_BIN}" - "${RESULT_CSV}" "${SUMMARY_CSV}" "${OUT_DIR}" "${ORACLE_JSONL}" "${TOPK}" "${MIN_DETECTIONS}" <<'PY'
import csv
import sys
from pathlib import Path

result_csv = Path(sys.argv[1])
summary_csv = Path(sys.argv[2])
out_dir = sys.argv[3]
oracle_jsonl = sys.argv[4]
topk = sys.argv[5]
min_detections = sys.argv[6]

fieldnames = [
    "exp_name",
    "config_path",
    "checkpoint",
    "oracle_group_jsonl",
    "graph_mode",
    "graph_topk",
    "graph_min_detections",
    "out_dir",
    "HOTA",
    "AssA",
    "IDF1",
    "MOTA",
    "IDSW",
    "status",
]
row = {
    "exp_name": "local_conflict_graph_commitmatches_oracle_proxy0213",
    "config_path": "configs/experiments/bytetrack_fa_mot_mot17_v14_ctrl_base_reid_da_val0213.yaml",
    "checkpoint": "",
    "oracle_group_jsonl": oracle_jsonl,
    "graph_mode": "oracle_commit_matches",
    "graph_topk": topk,
    "graph_min_detections": min_detections,
    "out_dir": out_dir,
    "HOTA": "",
    "AssA": "",
    "IDF1": "",
    "MOTA": "",
    "IDSW": "",
    "status": "running",
}
for path in (result_csv, summary_csv):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)
PY

"${PYTHON_BIN}" - "${PROFILE_PATH}" "${BASE_CONFIG}" "${BASE_CKPT}" "${ORACLE_JSONL}" "${TOPK}" "${MIN_DETECTIONS}" <<'PY'
import json
import sys
from pathlib import Path

profile_path = Path(sys.argv[1]).resolve()
config_path = sys.argv[2]
base_ckpt = str(Path(sys.argv[3]).resolve())
oracle_jsonl = str(Path(sys.argv[4]).resolve())
topk = int(sys.argv[5])
min_detections = int(sys.argv[6])

overrides = {
    "EXP_NAME": "local_conflict_graph_commitmatches_oracle_proxy0213",
    "EVAL_ONLY_VAL": True,
    "RUN_TRACKEVAL": True,
    "VAL_SEQUENCES": ["MOT17-02", "MOT17-13"],
    "ASSOC_USE_LAPLACE": False,
    "ASSOC_USE_MTCR": False,
    "ASSOC_USE_RUNTIME_REPLAY": False,
    "ASSOC_USE_COMPETITION": False,
    "ASSOC_USE_COMPETITION_ORACLE": False,
    "ASSOC_USE_LOCAL_CONFLICT_GRAPH": True,
    "ASSOC_LOCAL_CONFLICT_GRAPH_MODE": "oracle_commit_matches",
    "ASSOC_USE_LOCAL_CONFLICT_GRAPH_ORACLE": False,
    "ASSOC_LOCAL_CONFLICT_GRAPH_ORACLE_JSONL": oracle_jsonl,
    "ASSOC_LOCAL_CONFLICT_GRAPH_TOPK": topk,
    "ASSOC_LOCAL_CONFLICT_GRAPH_MIN_DETECTIONS": min_detections,
}

doc = {
    "description": "Local conflict graph matched-edge commit oracle with host fallback on proxy0213.",
    "manifest": {
        "line": "local_conflict_graph_mainline",
        "protocol_tier": "proxy0213_internal",
        "host_variant": "base_reid_da",
        "eval_scope": "mot17_proxy0213",
        "inference_model": base_ckpt,
        "oracle_group_jsonl": oracle_jsonl,
        "graph_mode": "oracle_commit_matches",
        "graph_topk": topk,
        "graph_min_detections": min_detections,
    },
    "settings": {
        "config_path": config_path,
        "inference_dataset": "MOT17",
        "inference_split": "train",
        "inference_model": base_ckpt,
        "config_overrides": overrides,
    },
}
profile_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
PY

STATUS="failed"
if "${PYTHON_BIN}" "${REPO_ROOT}/scripts/run_bytetrack_profile.py" \
  --exp-profile "${PROFILE_PATH}" \
  --out-dir "${OUT_DIR}" >"${LOG_PATH}" 2>&1; then
  STATUS="success"
fi

"${PYTHON_BIN}" - "${RESULT_CSV}" "${SUMMARY_CSV}" "${MANIFEST_PATH}" "${BASE_CKPT}" "${OUT_DIR}" "${ORACLE_JSONL}" "${TOPK}" "${MIN_DETECTIONS}" "${STATUS}" <<'PY'
import csv
import json
import sys
from pathlib import Path

result_csv = Path(sys.argv[1])
summary_csv = Path(sys.argv[2])
manifest_path = Path(sys.argv[3])
base_ckpt = sys.argv[4]
out_dir = sys.argv[5]
oracle_jsonl = sys.argv[6]
topk = sys.argv[7]
min_detections = sys.argv[8]
status = sys.argv[9]

row = {
    "exp_name": "local_conflict_graph_commitmatches_oracle_proxy0213",
    "config_path": "configs/experiments/bytetrack_fa_mot_mot17_v14_ctrl_base_reid_da_val0213.yaml",
    "checkpoint": base_ckpt,
    "oracle_group_jsonl": oracle_jsonl,
    "graph_mode": "oracle_commit_matches",
    "graph_topk": topk,
    "graph_min_detections": min_detections,
    "out_dir": out_dir,
    "HOTA": "",
    "AssA": "",
    "IDF1": "",
    "MOTA": "",
    "IDSW": "",
    "status": "ok" if status == "success" else "failed",
}

if manifest_path.is_file():
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    metrics = (((manifest.get("outputs") or {}).get("summary_metrics")) or {})
    for key in ("HOTA", "AssA", "IDF1", "MOTA", "IDSW"):
        if key in metrics:
            row[key] = str(metrics[key])

fieldnames = list(row.keys())
for path in (result_csv, summary_csv):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)
PY

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/append_experiment_record.py" \
  --csv "${REPO_ROOT}/outputs/experiment_registry.csv" \
  --kind eval \
  --status "$( [[ "${STATUS}" == "success" ]] && echo success || echo failed )" \
  --script "scripts/run_local_conflict_graph_commitmatches_oracle_proxy0213.sh" \
  --dataset "MOT17" \
  --split "val0213_proxy" \
  --tracker-family "ByteTrack" \
  --variant "local_conflict_graph_commitmatches_oracle_proxy0213" \
  --tag "local_conflict_graph_mainline" \
  --run-root "${OUT_DIR}" \
  --summary-csv "${SUMMARY_CSV}" \
  --checkpoint "${BASE_CKPT}" \
  --log-path "${LOG_PATH}" \
  --notes "local conflict graph matched-edge commit oracle with host fallback on proxy0213" \
  --extra "oracle_group_jsonl=${ORACLE_JSONL}" "graph_mode=oracle_commit_matches" "graph_topk=${TOPK}" "graph_min_detections=${MIN_DETECTIONS}"

if ! "${PYTHON_BIN}" "${REPO_ROOT}/scripts/post_experiment_pro_bundle.py" \
  --run-root "${OUT_DIR}" \
  --tag "local_conflict_graph_commitmatches_oracle_bundle" \
  --label "local_conflict_graph_commitmatches_oracle_proxy0213" \
  --status "$( [[ "${STATUS}" == "success" ]] && echo ok || echo failed )"; then
  echo "[local-conflict-graph-commitmatches-oracle] warning: failed to build Pro review bundle for ${OUT_DIR}" >&2
fi

echo "[local-conflict-graph-commitmatches-oracle] status=${STATUS} out_dir=${OUT_DIR}"
