#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
BASE_CONFIG="${BASE_CONFIG:-${REPO_ROOT}/configs/experiments/bytetrack_fa_mot_mot17_v14_ctrl_base_reid_da_val0213.yaml}"
BASE_CKPT="${BASE_CKPT:-${REPO_ROOT}/outputs/paper_ctrl_mot17_val0213/bytetrack_fa_mot_mot17_v14_ctrl_base_reid_da_val0213/checkpoint_epoch_0.pth}"
HOST_VARIANT="${HOST_VARIANT:-base_reid_da}"

OUT_DIR="${1:-${REPO_ROOT}/outputs/bytetrack_host_only_$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${2:-bytetrack_host_only}"
DETECTOR_FILTER_CSV="${3:-}"
VAL_SEQUENCES_CSV="${4:-}"
NOTES="${5:-ByteTrack host-only evaluation}"

if [[ ! -f "${BASE_CONFIG}" ]]; then
  echo "Missing config: ${BASE_CONFIG}" >&2
  exit 2
fi
if [[ ! -f "${BASE_CKPT}" ]]; then
  echo "Missing base checkpoint: ${BASE_CKPT}" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"

PROFILE_PATH="${OUT_DIR}/profile.json"
RESULT_CSV="${OUT_DIR}/result.csv"
SUMMARY_CSV="${OUT_DIR}/summary.csv"
LOG_PATH="${OUT_DIR}/run.log"
MANIFEST_PATH="${OUT_DIR}/run_manifest.json"

"${PYTHON_BIN}" - "${RESULT_CSV}" "${SUMMARY_CSV}" "${RUN_NAME}" "${OUT_DIR}" "${BASE_CONFIG}" "${BASE_CKPT}" "${HOST_VARIANT}" "${DETECTOR_FILTER_CSV}" "${VAL_SEQUENCES_CSV}" <<'PY'
import csv
import sys
from pathlib import Path

result_csv = Path(sys.argv[1])
summary_csv = Path(sys.argv[2])
run_name = sys.argv[3]
out_dir = sys.argv[4]
base_config = sys.argv[5]
base_ckpt = sys.argv[6]
host_variant = sys.argv[7]
detector_filter_csv = sys.argv[8]
val_sequences_csv = sys.argv[9]

fieldnames = [
    "exp_name",
    "module_family",
    "config_path",
    "checkpoint",
    "host_variant",
    "host_config_path",
    "host_checkpoint",
    "graph_mode",
    "graph_checkpoint",
    "graph_topk",
    "graph_min_detections",
    "graph_min_committed_matches",
    "graph_max_detections",
    "graph_max_tracks",
    "graph_cluster_gate_thresh",
    "graph_cluster_gate_temp",
    "graph_cluster_gate_bias",
    "detector_filter",
    "val_sequences",
    "out_dir",
    "eligible_clusters",
    "replaced_clusters",
    "matched_dets",
    "deferred_dets",
    "blocked_tracks",
    "gate_pass_clusters",
    "gate_filtered_clusters",
    "skipped_large_clusters",
    "HOTA",
    "AssA",
    "IDF1",
    "MOTA",
    "IDSW",
    "status",
]
row = {
    "exp_name": run_name,
    "module_family": "bytetrack_host_only",
    "config_path": base_config,
    "checkpoint": base_ckpt,
    "host_variant": host_variant,
    "host_config_path": base_config,
    "host_checkpoint": base_ckpt,
    "graph_mode": "disabled",
    "graph_checkpoint": "",
    "graph_topk": "",
    "graph_min_detections": "",
    "graph_min_committed_matches": "",
    "graph_max_detections": "",
    "graph_max_tracks": "",
    "graph_cluster_gate_thresh": "",
    "graph_cluster_gate_temp": "",
    "graph_cluster_gate_bias": "",
    "detector_filter": detector_filter_csv,
    "val_sequences": val_sequences_csv,
    "out_dir": out_dir,
    "eligible_clusters": "0",
    "replaced_clusters": "0",
    "matched_dets": "0",
    "deferred_dets": "0",
    "blocked_tracks": "0",
    "gate_pass_clusters": "0",
    "gate_filtered_clusters": "0",
    "skipped_large_clusters": "0",
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

"${PYTHON_BIN}" - "${PROFILE_PATH}" "${BASE_CONFIG}" "${BASE_CKPT}" "${HOST_VARIANT}" "${RUN_NAME}" "${DETECTOR_FILTER_CSV}" "${VAL_SEQUENCES_CSV}" "${NOTES}" <<'PY'
import json
import sys
from pathlib import Path

profile_path = Path(sys.argv[1]).resolve()
config_path = sys.argv[2]
base_ckpt = str(Path(sys.argv[3]).resolve())
host_variant = sys.argv[4]
run_name = sys.argv[5]
detector_filter_csv = str(sys.argv[6] or "").strip()
val_sequences_csv = str(sys.argv[7] or "").strip()
notes = str(sys.argv[8] or "").strip()

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
    "ASSOC_USE_LOCAL_CONFLICT_GRAPH": False,
    "ASSOC_LOCAL_CONFLICT_GRAPH_MODE": "disabled",
}
if detector_filter:
    overrides["DETECTOR_FILTER"] = detector_filter
if val_sequences:
    overrides["EVAL_ONLY_VAL"] = True
    overrides["VAL_SEQUENCES"] = val_sequences
else:
    overrides["EVAL_ONLY_VAL"] = False

doc = {
    "description": notes,
    "manifest": {
        "line": "bytetrack_host_only",
        "protocol_tier": scope_label,
        "host_variant": host_variant,
        "host_config_path": config_path,
        "host_checkpoint": base_ckpt,
        "eval_scope": scope_label,
        "inference_model": base_ckpt,
        "graph_mode": "disabled",
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

"${PYTHON_BIN}" - "${RESULT_CSV}" "${SUMMARY_CSV}" "${MANIFEST_PATH}" "${BASE_CONFIG}" "${BASE_CKPT}" "${HOST_VARIANT}" "${RUN_NAME}" "${OUT_DIR}" "${DETECTOR_FILTER_CSV}" "${VAL_SEQUENCES_CSV}" "${STATUS}" <<'PY'
import csv
import json
import sys
from pathlib import Path

result_csv = Path(sys.argv[1])
summary_csv = Path(sys.argv[2])
manifest_path = Path(sys.argv[3])
base_config = sys.argv[4]
base_ckpt = sys.argv[5]
host_variant = sys.argv[6]
run_name = sys.argv[7]
out_dir = sys.argv[8]
detector_filter_csv = sys.argv[9]
val_sequences_csv = sys.argv[10]
status = sys.argv[11]

row = {
    "exp_name": run_name,
    "module_family": "bytetrack_host_only",
    "config_path": base_config,
    "checkpoint": base_ckpt,
    "host_variant": host_variant,
    "host_config_path": base_config,
    "host_checkpoint": base_ckpt,
    "graph_mode": "disabled",
    "graph_checkpoint": "",
    "graph_topk": "",
    "graph_min_detections": "",
    "graph_min_committed_matches": "",
    "graph_max_detections": "",
    "graph_max_tracks": "",
    "graph_cluster_gate_thresh": "",
    "graph_cluster_gate_temp": "",
    "graph_cluster_gate_bias": "",
    "detector_filter": detector_filter_csv,
    "val_sequences": val_sequences_csv,
    "out_dir": out_dir,
    "eligible_clusters": "0",
    "replaced_clusters": "0",
    "matched_dets": "0",
    "deferred_dets": "0",
    "blocked_tracks": "0",
    "gate_pass_clusters": "0",
    "gate_filtered_clusters": "0",
    "skipped_large_clusters": "0",
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
  --script "scripts/run_bytetrack_host_only_generic.sh" \
  --dataset "MOT17" \
  --split "${SPLIT_LABEL}" \
  --tracker-family "ByteTrack" \
  --variant "${RUN_NAME}" \
  --tag "bytetrack_host_only" \
  --run-root "${OUT_DIR}" \
  --summary-csv "${SUMMARY_CSV}" \
  --checkpoint "${BASE_CKPT}" \
  --log-path "${LOG_PATH}" \
  --notes "${NOTES}" \
  --extra "module_family=bytetrack_host_only" "host_variant=${HOST_VARIANT}" "host_config_path=${BASE_CONFIG}" "host_checkpoint=${BASE_CKPT}" "graph_mode=disabled" "detector_filter=${DETECTOR_FILTER_CSV}" "val_sequences=${VAL_SEQUENCES_CSV}"

if ! "${PYTHON_BIN}" "${REPO_ROOT}/scripts/post_experiment_pro_bundle.py" \
  --run-root "${OUT_DIR}" \
  --tag "bytetrack_host_only_bundle" \
  --label "${RUN_NAME}" \
  --status "$( [[ "${STATUS}" == "success" ]] && echo ok || echo failed )"; then
  echo "[bytetrack-host-only-generic] warning: failed to build Pro review bundle for ${OUT_DIR}" >&2
fi

echo "[bytetrack-host-only-generic] status=${STATUS} out_dir=${OUT_DIR} host=${HOST_VARIANT}"
