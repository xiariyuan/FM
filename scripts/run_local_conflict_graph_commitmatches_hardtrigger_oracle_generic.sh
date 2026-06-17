#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
BASE_CONFIG="${REPO_ROOT}/configs/experiments/bytetrack_fa_mot_mot17_v14_ctrl_base_reid_da_val0213.yaml"
BASE_CKPT="${REPO_ROOT}/outputs/paper_ctrl_mot17_val0213/bytetrack_fa_mot_mot17_v14_ctrl_base_reid_da_val0213/checkpoint_epoch_0.pth"
ORACLE_JSONL_DEFAULT="${REPO_ROOT}/outputs/competition_assoc_base_reid_da_proxy0213_hybriddumpfix/groups.jsonl"

if [[ ! -f "${BASE_CKPT}" ]]; then
  FALLBACK_CKPT="${REPO_ROOT}/outputs/bytetrack_fa_mot_mot17_v16_laplace_gate_proxy0213_20260512_200536/checkpoint_epoch_0.pth"
  if [[ -f "${FALLBACK_CKPT}" ]]; then
    BASE_CKPT="${FALLBACK_CKPT}"
  fi
fi

OUT_DIR="${1:-${REPO_ROOT}/outputs/local_conflict_graph_commitmatches_hardtrigger_oracle_$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${2:-local_conflict_graph_commitmatches_hardtrigger_oracle}"
ORACLE_JSONL="${3:-${ORACLE_JSONL_DEFAULT}}"
TOPK="${4:-8}"
MIN_DETECTIONS="${5:-2}"
MIN_COMMITTED_MATCHES="${6:-2}"
DETECTOR_FILTER_CSV="${7:-}"
VAL_SEQUENCES_CSV="${8:-}"
NOTES="${9:-local conflict graph matched-edge commit oracle with hard cluster trigger}"

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

"${PYTHON_BIN}" - "${RESULT_CSV}" "${SUMMARY_CSV}" "${RUN_NAME}" "${OUT_DIR}" "${ORACLE_JSONL}" "${TOPK}" "${MIN_DETECTIONS}" "${MIN_COMMITTED_MATCHES}" "${DETECTOR_FILTER_CSV}" "${VAL_SEQUENCES_CSV}" <<'PY'
import csv
import sys
from pathlib import Path

result_csv = Path(sys.argv[1])
summary_csv = Path(sys.argv[2])
run_name = sys.argv[3]
out_dir = sys.argv[4]
oracle_jsonl = sys.argv[5]
topk = sys.argv[6]
min_detections = sys.argv[7]
min_committed_matches = sys.argv[8]
detector_filter_csv = sys.argv[9]
val_sequences_csv = sys.argv[10]

fieldnames = [
    "exp_name",
    "config_path",
    "checkpoint",
    "oracle_group_jsonl",
    "graph_mode",
    "graph_topk",
    "graph_min_detections",
    "graph_min_committed_matches",
    "detector_filter",
    "val_sequences",
    "out_dir",
    "HOTA",
    "AssA",
    "IDF1",
    "MOTA",
    "IDSW",
    "status",
]
row = {
    "exp_name": run_name,
    "config_path": "configs/experiments/bytetrack_fa_mot_mot17_v14_ctrl_base_reid_da_val0213.yaml",
    "checkpoint": "",
    "oracle_group_jsonl": oracle_jsonl,
    "graph_mode": "oracle_commit_matches",
    "graph_topk": topk,
    "graph_min_detections": min_detections,
    "graph_min_committed_matches": min_committed_matches,
    "detector_filter": detector_filter_csv,
    "val_sequences": val_sequences_csv,
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

"${PYTHON_BIN}" - "${PROFILE_PATH}" "${BASE_CONFIG}" "${BASE_CKPT}" "${RUN_NAME}" "${ORACLE_JSONL}" "${TOPK}" "${MIN_DETECTIONS}" "${MIN_COMMITTED_MATCHES}" "${DETECTOR_FILTER_CSV}" "${VAL_SEQUENCES_CSV}" "${NOTES}" <<'PY'
import json
import sys
from pathlib import Path

profile_path = Path(sys.argv[1]).resolve()
config_path = sys.argv[2]
base_ckpt = str(Path(sys.argv[3]).resolve())
run_name = sys.argv[4]
oracle_jsonl = str(Path(sys.argv[5]).resolve())
topk = int(sys.argv[6])
min_detections = int(sys.argv[7])
min_committed_matches = int(sys.argv[8])
detector_filter_csv = str(sys.argv[9] or "").strip()
val_sequences_csv = str(sys.argv[10] or "").strip()
notes = str(sys.argv[11] or "").strip()

def normalize_csv_tokens(raw: str) -> list[str]:
    raw = (raw or "").strip()
    if not raw or raw.upper() in {"ALL", "NONE", "NULL", "*"}:
        return []
    return [token.strip() for token in raw.split(",") if token.strip()]


detector_filter = normalize_csv_tokens(detector_filter_csv)
val_sequences = normalize_csv_tokens(val_sequences_csv)

scope_label = "full_public"
if val_sequences:
    scope_label = "proxy_subset"
elif detector_filter:
    scope_label = "full_filtered"

overrides = {
    "EXP_NAME": run_name,
    "RUN_TRACKEVAL": True,
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
    "ASSOC_LOCAL_CONFLICT_GRAPH_MIN_COMMITTED_MATCHES": min_committed_matches,
}
if detector_filter:
    overrides["DETECTOR_FILTER"] = detector_filter
else:
    overrides["DETECTOR_FILTER"] = None
if val_sequences:
    overrides["EVAL_ONLY_VAL"] = True
    overrides["VAL_SEQUENCES"] = val_sequences
else:
    overrides["EVAL_ONLY_VAL"] = False
    overrides["VAL_SEQUENCES"] = []

doc = {
    "description": notes,
    "manifest": {
        "line": "local_conflict_graph_mainline",
        "protocol_tier": scope_label,
        "host_variant": "base_reid_da",
        "eval_scope": scope_label,
        "inference_model": base_ckpt,
        "oracle_group_jsonl": oracle_jsonl,
        "graph_mode": "oracle_commit_matches",
        "graph_topk": topk,
        "graph_min_detections": min_detections,
        "graph_min_committed_matches": min_committed_matches,
        "detector_filter": detector_filter,
        "val_sequences": val_sequences,
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

"${PYTHON_BIN}" - "${RESULT_CSV}" "${SUMMARY_CSV}" "${MANIFEST_PATH}" "${BASE_CKPT}" "${RUN_NAME}" "${OUT_DIR}" "${ORACLE_JSONL}" "${TOPK}" "${MIN_DETECTIONS}" "${MIN_COMMITTED_MATCHES}" "${DETECTOR_FILTER_CSV}" "${VAL_SEQUENCES_CSV}" "${STATUS}" <<'PY'
import csv
import json
import sys
from pathlib import Path

result_csv = Path(sys.argv[1])
summary_csv = Path(sys.argv[2])
manifest_path = Path(sys.argv[3])
base_ckpt = sys.argv[4]
run_name = sys.argv[5]
out_dir = sys.argv[6]
oracle_jsonl = sys.argv[7]
topk = sys.argv[8]
min_detections = sys.argv[9]
min_committed_matches = sys.argv[10]
detector_filter_csv = sys.argv[11]
val_sequences_csv = sys.argv[12]
status = sys.argv[13]

row = {
    "exp_name": run_name,
    "config_path": "configs/experiments/bytetrack_fa_mot_mot17_v14_ctrl_base_reid_da_val0213.yaml",
    "checkpoint": base_ckpt,
    "oracle_group_jsonl": oracle_jsonl,
    "graph_mode": "oracle_commit_matches",
    "graph_topk": topk,
    "graph_min_detections": min_detections,
    "graph_min_committed_matches": min_committed_matches,
    "detector_filter": detector_filter_csv,
    "val_sequences": val_sequences_csv,
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

SPLIT_LABEL="train_full"
if [[ -n "${VAL_SEQUENCES_CSV// /}" ]] && [[ ! "${VAL_SEQUENCES_CSV^^}" =~ ^(ALL|NONE|NULL|\*)$ ]]; then
  SPLIT_LABEL="val_subset"
fi
if [[ -n "${DETECTOR_FILTER_CSV// /}" ]] && [[ ! "${DETECTOR_FILTER_CSV^^}" =~ ^(ALL|NONE|NULL|\*)$ ]]; then
  SPLIT_LABEL="${SPLIT_LABEL}_filtered"
fi

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/append_experiment_record.py" \
  --csv "${REPO_ROOT}/outputs/experiment_registry.csv" \
  --kind eval \
  --status "$( [[ "${STATUS}" == "success" ]] && echo success || echo failed )" \
  --script "scripts/run_local_conflict_graph_commitmatches_hardtrigger_oracle_generic.sh" \
  --dataset "MOT17" \
  --split "${SPLIT_LABEL}" \
  --tracker-family "ByteTrack" \
  --variant "${RUN_NAME}" \
  --tag "local_conflict_graph_mainline" \
  --run-root "${OUT_DIR}" \
  --summary-csv "${SUMMARY_CSV}" \
  --checkpoint "${BASE_CKPT}" \
  --log-path "${LOG_PATH}" \
  --notes "${NOTES}" \
  --extra "oracle_group_jsonl=${ORACLE_JSONL}" "graph_mode=oracle_commit_matches" "graph_topk=${TOPK}" "graph_min_detections=${MIN_DETECTIONS}" "graph_min_committed_matches=${MIN_COMMITTED_MATCHES}" "detector_filter=${DETECTOR_FILTER_CSV}" "val_sequences=${VAL_SEQUENCES_CSV}"

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/append_experiment_record.py" \
  --csv "${REPO_ROOT}/outputs/experiment_registry.csv" \
  --kind eval \
  --status "$( [[ "${STATUS}" == "success" ]] && echo success || echo failed )" \
  --script "scripts/run_local_conflict_graph_commitmatches_hardtrigger_oracle_generic.sh" \
  --dataset "MOT17" \
  --split "train" \
  --tracker-family "ByteTrack" \
  --variant "${RUN_NAME}" \
  --tag "local_conflict_graph_mainline" \
  --run-root "${OUT_DIR}" \
  --summary-csv "${SUMMARY_CSV}" \
  --checkpoint "${BASE_CKPT}" \
  --log-path "${LOG_PATH}" \
  --notes "${NOTES}" \
  --extra "oracle_group_jsonl=${ORACLE_JSONL}" "graph_mode=oracle_commit_matches" "graph_topk=${TOPK}" "graph_min_detections=${MIN_DETECTIONS}" "graph_min_committed_matches=${MIN_COMMITTED_MATCHES}" "detector_filter=${DETECTOR_FILTER_CSV}" "val_sequences=${VAL_SEQUENCES_CSV}"

if ! "${PYTHON_BIN}" "${REPO_ROOT}/scripts/post_experiment_pro_bundle.py" \
  --run-root "${OUT_DIR}" \
  --tag "${RUN_NAME}_bundle" \
  --label "${RUN_NAME}" \
  --status "$( [[ "${STATUS}" == "success" ]] && echo ok || echo failed )"; then
  echo "[${RUN_NAME}] warning: failed to build Pro review bundle for ${OUT_DIR}" >&2
fi

echo "[${RUN_NAME}] status=${STATUS} out_dir=${OUT_DIR}"
