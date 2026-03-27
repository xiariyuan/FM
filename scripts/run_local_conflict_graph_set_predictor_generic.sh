#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
BASE_CONFIG="${BASE_CONFIG:-${REPO_ROOT}/configs/experiments/bytetrack_fa_mot_mot17_v18_local_conflict_set_predictor_val0213.yaml}"
BASE_CKPT="${BASE_CKPT:-${REPO_ROOT}/outputs/paper_ctrl_mot17_val0213/bytetrack_fa_mot_mot17_v14_ctrl_base_reid_da_val0213/checkpoint_epoch_0.pth}"
HOST_VARIANT="${HOST_VARIANT:-base_reid_da}"

OUT_DIR="${1:-${REPO_ROOT}/outputs/local_conflict_graph_set_predictor_$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${2:-local_conflict_graph_set_predictor}"
GRAPH_CKPT="${3:-}"
TOPK="${4:-8}"
MIN_DETECTIONS="${5:-2}"
MIN_COMMITTED_MATCHES="${6:-2}"
MAX_DETECTIONS="${7:-8}"
MAX_TRACKS="${8:-32}"
CLUSTER_GATE_THRESH="${9:-0.5}"
CLUSTER_GATE_TEMP="${10:-1.0}"
CLUSTER_GATE_BIAS="${11:-0.0}"
DETECTOR_FILTER_CSV="${12:-}"
VAL_SEQUENCES_CSV="${13:-}"
NOTES="${14:-set predictor conservative local-conflict evaluation}"

if [[ ! -f "${BASE_CONFIG}" ]]; then
  echo "Missing config: ${BASE_CONFIG}" >&2
  exit 2
fi
if [[ ! -f "${BASE_CKPT}" ]]; then
  echo "Missing base checkpoint: ${BASE_CKPT}" >&2
  exit 2
fi
if [[ -z "${GRAPH_CKPT}" || ! -f "${GRAPH_CKPT}" ]]; then
  echo "Missing set predictor checkpoint: ${GRAPH_CKPT}" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"

PROFILE_PATH="${OUT_DIR}/profile.json"
RESULT_CSV="${OUT_DIR}/result.csv"
SUMMARY_CSV="${OUT_DIR}/summary.csv"
LOG_PATH="${OUT_DIR}/run.log"
MANIFEST_PATH="${OUT_DIR}/run_manifest.json"

"${PYTHON_BIN}" - "${RESULT_CSV}" "${SUMMARY_CSV}" "${RUN_NAME}" "${OUT_DIR}" "${BASE_CONFIG}" "${BASE_CKPT}" "${HOST_VARIANT}" "${GRAPH_CKPT}" "${TOPK}" "${MIN_DETECTIONS}" "${MIN_COMMITTED_MATCHES}" "${MAX_DETECTIONS}" "${MAX_TRACKS}" "${CLUSTER_GATE_THRESH}" "${CLUSTER_GATE_TEMP}" "${CLUSTER_GATE_BIAS}" "${DETECTOR_FILTER_CSV}" "${VAL_SEQUENCES_CSV}" <<'PY'
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
graph_ckpt = sys.argv[8]
topk = sys.argv[9]
min_detections = sys.argv[10]
min_committed_matches = sys.argv[11]
max_detections = sys.argv[12]
max_tracks = sys.argv[13]
cluster_gate_thresh = sys.argv[14]
cluster_gate_temp = sys.argv[15]
cluster_gate_bias = sys.argv[16]
detector_filter_csv = sys.argv[17]
val_sequences_csv = sys.argv[18]

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
    "module_family": "set_predictor_v2",
    "config_path": base_config,
    "checkpoint": base_ckpt,
    "host_variant": host_variant,
    "host_config_path": base_config,
    "host_checkpoint": base_ckpt,
    "graph_mode": "learned_commit",
    "graph_checkpoint": graph_ckpt,
    "graph_topk": topk,
    "graph_min_detections": min_detections,
    "graph_min_committed_matches": min_committed_matches,
    "graph_max_detections": max_detections,
    "graph_max_tracks": max_tracks,
    "graph_cluster_gate_thresh": cluster_gate_thresh,
    "graph_cluster_gate_temp": cluster_gate_temp,
    "graph_cluster_gate_bias": cluster_gate_bias,
    "detector_filter": detector_filter_csv,
    "val_sequences": val_sequences_csv,
    "out_dir": out_dir,
    "eligible_clusters": "",
    "replaced_clusters": "",
    "matched_dets": "",
    "deferred_dets": "",
    "blocked_tracks": "",
    "gate_pass_clusters": "",
    "gate_filtered_clusters": "",
    "skipped_large_clusters": "",
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

"${PYTHON_BIN}" - "${PROFILE_PATH}" "${BASE_CONFIG}" "${BASE_CKPT}" "${HOST_VARIANT}" "${RUN_NAME}" "${GRAPH_CKPT}" "${TOPK}" "${MIN_DETECTIONS}" "${MIN_COMMITTED_MATCHES}" "${MAX_DETECTIONS}" "${MAX_TRACKS}" "${CLUSTER_GATE_THRESH}" "${CLUSTER_GATE_TEMP}" "${CLUSTER_GATE_BIAS}" "${DETECTOR_FILTER_CSV}" "${VAL_SEQUENCES_CSV}" "${NOTES}" <<'PY'
import json
import sys
from pathlib import Path

profile_path = Path(sys.argv[1]).resolve()
config_path = sys.argv[2]
base_ckpt = str(Path(sys.argv[3]).resolve())
host_variant = sys.argv[4]
run_name = sys.argv[5]
graph_ckpt = str(Path(sys.argv[6]).resolve())
topk = int(sys.argv[7])
min_detections = int(sys.argv[8])
min_committed_matches = int(sys.argv[9])
max_detections = int(sys.argv[10])
max_tracks = int(sys.argv[11])
cluster_gate_thresh = float(sys.argv[12])
cluster_gate_temp = float(sys.argv[13])
cluster_gate_bias = float(sys.argv[14])
detector_filter_csv = str(sys.argv[15] or "").strip()
val_sequences_csv = str(sys.argv[16] or "").strip()
notes = str(sys.argv[17] or "").strip()

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
    "ASSOC_USE_LOCAL_CONFLICT_GRAPH": True,
    "ASSOC_LOCAL_CONFLICT_GRAPH_MODE": "learned_commit",
    "ASSOC_LOCAL_CONFLICT_GRAPH_CHECKPOINT": graph_ckpt,
    "ASSOC_LOCAL_CONFLICT_GRAPH_TOPK": topk,
    "ASSOC_LOCAL_CONFLICT_GRAPH_MIN_DETECTIONS": min_detections,
    "ASSOC_LOCAL_CONFLICT_GRAPH_MIN_COMMITTED_MATCHES": min_committed_matches,
    "ASSOC_LOCAL_CONFLICT_GRAPH_MAX_DETECTIONS": max_detections,
    "ASSOC_LOCAL_CONFLICT_GRAPH_MAX_TRACKS": max_tracks,
    "ASSOC_LOCAL_CONFLICT_GRAPH_CLUSTER_GATE_THRESH": cluster_gate_thresh,
    "ASSOC_LOCAL_CONFLICT_GRAPH_CLUSTER_GATE_TEMP": cluster_gate_temp,
    "ASSOC_LOCAL_CONFLICT_GRAPH_CLUSTER_GATE_BIAS": cluster_gate_bias,
    "ASSOC_LOCAL_CONFLICT_GRAPH_HOST_VARIANT": host_variant,
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
        "line": "local_conflict_set_predictor_mainline",
        "protocol_tier": scope_label,
        "host_variant": host_variant,
        "host_config_path": config_path,
        "host_checkpoint": base_ckpt,
        "eval_scope": scope_label,
        "inference_model": base_ckpt,
        "graph_mode": "learned_commit",
        "graph_checkpoint": graph_ckpt,
        "graph_topk": topk,
        "graph_min_detections": min_detections,
        "graph_min_committed_matches": min_committed_matches,
        "graph_max_detections": max_detections,
        "graph_max_tracks": max_tracks,
        "graph_cluster_gate_thresh": cluster_gate_thresh,
        "graph_cluster_gate_temp": cluster_gate_temp,
        "graph_cluster_gate_bias": cluster_gate_bias,
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

"${PYTHON_BIN}" - "${RESULT_CSV}" "${SUMMARY_CSV}" "${MANIFEST_PATH}" "${BASE_CONFIG}" "${BASE_CKPT}" "${HOST_VARIANT}" "${RUN_NAME}" "${OUT_DIR}" "${GRAPH_CKPT}" "${TOPK}" "${MIN_DETECTIONS}" "${MIN_COMMITTED_MATCHES}" "${MAX_DETECTIONS}" "${MAX_TRACKS}" "${CLUSTER_GATE_THRESH}" "${CLUSTER_GATE_TEMP}" "${CLUSTER_GATE_BIAS}" "${DETECTOR_FILTER_CSV}" "${VAL_SEQUENCES_CSV}" "${STATUS}" <<'PY'
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
graph_ckpt = sys.argv[9]
topk = sys.argv[10]
min_detections = sys.argv[11]
min_committed_matches = sys.argv[12]
max_detections = sys.argv[13]
max_tracks = sys.argv[14]
cluster_gate_thresh = sys.argv[15]
cluster_gate_temp = sys.argv[16]
cluster_gate_bias = sys.argv[17]
detector_filter_csv = sys.argv[18]
val_sequences_csv = sys.argv[19]
status = sys.argv[20]

row = {
    "exp_name": run_name,
    "module_family": "set_predictor_v2",
    "config_path": base_config,
    "checkpoint": base_ckpt,
    "host_variant": host_variant,
    "host_config_path": base_config,
    "host_checkpoint": base_ckpt,
    "graph_mode": "learned_commit",
    "graph_checkpoint": graph_ckpt,
    "graph_topk": topk,
    "graph_min_detections": min_detections,
    "graph_min_committed_matches": min_committed_matches,
    "graph_max_detections": max_detections,
    "graph_max_tracks": max_tracks,
    "graph_cluster_gate_thresh": cluster_gate_thresh,
    "graph_cluster_gate_temp": cluster_gate_temp,
    "graph_cluster_gate_bias": cluster_gate_bias,
    "detector_filter": detector_filter_csv,
    "val_sequences": val_sequences_csv,
    "out_dir": out_dir,
    "eligible_clusters": "",
    "replaced_clusters": "",
    "matched_dets": "",
    "deferred_dets": "",
    "blocked_tracks": "",
    "gate_pass_clusters": "",
    "gate_filtered_clusters": "",
    "skipped_large_clusters": "",
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
    diag = (((manifest.get("outputs") or {}).get("local_conflict_graph_diagnostics")) or {}).get("combined", {})
    for key in (
        "eligible_clusters",
        "replaced_clusters",
        "matched_dets",
        "deferred_dets",
        "blocked_tracks",
        "gate_pass_clusters",
        "gate_filtered_clusters",
        "skipped_large_clusters",
    ):
        if key in diag:
            row[key] = str(diag[key])

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
  --script "scripts/run_local_conflict_graph_set_predictor_generic.sh" \
  --dataset "MOT17" \
  --split "${SPLIT_LABEL}" \
  --tracker-family "ByteTrack" \
  --variant "${RUN_NAME}" \
  --tag "local_conflict_set_predictor_mainline" \
  --run-root "${OUT_DIR}" \
  --summary-csv "${SUMMARY_CSV}" \
  --checkpoint "${GRAPH_CKPT}" \
  --log-path "${LOG_PATH}" \
  --notes "${NOTES}" \
  --extra "module_family=set_predictor_v2" "host_variant=${HOST_VARIANT}" "host_config_path=${BASE_CONFIG}" "host_checkpoint=${BASE_CKPT}" "graph_mode=learned_commit" "graph_topk=${TOPK}" "graph_min_detections=${MIN_DETECTIONS}" "graph_min_committed_matches=${MIN_COMMITTED_MATCHES}" "graph_max_detections=${MAX_DETECTIONS}" "graph_max_tracks=${MAX_TRACKS}" "graph_cluster_gate_thresh=${CLUSTER_GATE_THRESH}" "graph_cluster_gate_temp=${CLUSTER_GATE_TEMP}" "graph_cluster_gate_bias=${CLUSTER_GATE_BIAS}" "detector_filter=${DETECTOR_FILTER_CSV}" "val_sequences=${VAL_SEQUENCES_CSV}"

if ! "${PYTHON_BIN}" "${REPO_ROOT}/scripts/post_experiment_pro_bundle.py" \
  --run-root "${OUT_DIR}" \
  --tag "local_conflict_graph_set_predictor_bundle" \
  --label "${RUN_NAME}" \
  --status "$( [[ "${STATUS}" == "success" ]] && echo ok || echo failed )"; then
  echo "[local-conflict-graph-set-predictor-generic] warning: failed to build Pro review bundle for ${OUT_DIR}" >&2
fi

echo "[local-conflict-graph-set-predictor-generic] status=${STATUS} out_dir=${OUT_DIR} host=${HOST_VARIANT}"
