#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
BASE_CONFIG="${BASE_CONFIG:-${REPO_ROOT}/configs/experiments/bytetrack_fa_mot_mot17_v18_local_conflict_set_predictor_val0213.yaml}"
BASE_CKPT="${BASE_CKPT:-${REPO_ROOT}/outputs/paper_ctrl_mot17_val0213/bytetrack_fa_mot_mot17_v14_ctrl_base_reid_da_val0213/checkpoint_epoch_0.pth}"
HOST_VARIANT="${HOST_VARIANT:-base_reid_da}"

OUT_DIR="${1:-${REPO_ROOT}/outputs/local_conflict_graph_set_predictor_proxy0213_$(date +%Y%m%d_%H%M%S)}"
GRAPH_CKPT="${2:-}"
TOPK="${3:-8}"
MIN_DETECTIONS="${4:-2}"
MIN_COMMITTED_MATCHES="${5:-2}"
MAX_DETECTIONS="${6:-8}"
MAX_TRACKS="${7:-32}"
CLUSTER_GATE_THRESH="${8:-0.5}"
CLUSTER_GATE_TEMP="${9:-1.0}"
CLUSTER_GATE_BIAS="${10:-0.0}"

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
RUN_NAME="local_conflict_graph_set_predictor_proxy0213"

"${PYTHON_BIN}" - "${RESULT_CSV}" "${SUMMARY_CSV}" "${RUN_NAME}" "${OUT_DIR}" "${BASE_CONFIG}" "${BASE_CKPT}" "${HOST_VARIANT}" "${GRAPH_CKPT}" "${TOPK}" "${MIN_DETECTIONS}" "${MIN_COMMITTED_MATCHES}" "${MAX_DETECTIONS}" "${MAX_TRACKS}" "${CLUSTER_GATE_THRESH}" "${CLUSTER_GATE_TEMP}" "${CLUSTER_GATE_BIAS}" <<'PY'
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

"${PYTHON_BIN}" - "${PROFILE_PATH}" "${BASE_CONFIG}" "${BASE_CKPT}" "${HOST_VARIANT}" "${RUN_NAME}" "${GRAPH_CKPT}" "${TOPK}" "${MIN_DETECTIONS}" "${MIN_COMMITTED_MATCHES}" "${MAX_DETECTIONS}" "${MAX_TRACKS}" "${CLUSTER_GATE_THRESH}" "${CLUSTER_GATE_TEMP}" "${CLUSTER_GATE_BIAS}" <<'PY'
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

overrides = {
    "EXP_NAME": run_name,
    "EVAL_ONLY_VAL": True,
    "RUN_TRACKEVAL": True,
    "VAL_SEQUENCES": ["MOT17-02", "MOT17-13"],
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

doc = {
    "description": f"Set-predictor local-conflict evaluation on proxy0213 under host {host_variant}.",
    "manifest": {
        "line": "local_conflict_set_predictor_mainline",
        "protocol_tier": "proxy0213_internal",
        "host_variant": host_variant,
        "host_config_path": config_path,
        "host_checkpoint": base_ckpt,
        "eval_scope": "mot17_proxy0213",
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

"${PYTHON_BIN}" - "${RESULT_CSV}" "${SUMMARY_CSV}" "${MANIFEST_PATH}" "${RUN_NAME}" "${OUT_DIR}" "${BASE_CONFIG}" "${BASE_CKPT}" "${HOST_VARIANT}" "${GRAPH_CKPT}" "${TOPK}" "${MIN_DETECTIONS}" "${MIN_COMMITTED_MATCHES}" "${MAX_DETECTIONS}" "${MAX_TRACKS}" "${CLUSTER_GATE_THRESH}" "${CLUSTER_GATE_TEMP}" "${CLUSTER_GATE_BIAS}" "${STATUS}" <<'PY'
import csv
import json
import sys
from pathlib import Path

result_csv = Path(sys.argv[1])
summary_csv = Path(sys.argv[2])
manifest_path = Path(sys.argv[3])
run_name = sys.argv[4]
out_dir = Path(sys.argv[5])
base_config = sys.argv[6]
base_ckpt = sys.argv[7]
host_variant = sys.argv[8]
graph_ckpt = sys.argv[9]
topk = sys.argv[10]
min_detections = sys.argv[11]
min_committed_matches = sys.argv[12]
max_detections = sys.argv[13]
max_tracks = sys.argv[14]
cluster_gate_thresh = sys.argv[15]
cluster_gate_temp = sys.argv[16]
cluster_gate_bias = sys.argv[17]
status = sys.argv[18]

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
    "out_dir": str(out_dir),
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
    mapping = {
        "eligible_clusters": "eligible_clusters",
        "replaced_clusters": "replaced_clusters",
        "matched_dets": "matched_dets",
        "deferred_dets": "deferred_dets",
        "blocked_tracks": "blocked_tracks",
        "gate_pass_clusters": "gate_pass_clusters",
        "gate_filtered_clusters": "gate_filtered_clusters",
        "skipped_large_clusters": "skipped_large_clusters",
    }
    for dst, src in mapping.items():
        if src in diag:
            row[dst] = str(diag[src])

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
  --script "scripts/run_local_conflict_graph_set_predictor_proxy0213.sh" \
  --dataset "MOT17" \
  --split "val0213_proxy" \
  --tracker-family "ByteTrack" \
  --variant "${RUN_NAME}" \
  --tag "local_conflict_set_predictor_mainline" \
  --run-root "${OUT_DIR}" \
  --summary-csv "${SUMMARY_CSV}" \
  --checkpoint "${GRAPH_CKPT}" \
  --log-path "${LOG_PATH}" \
  --notes "set predictor conservative local-conflict evaluation on proxy0213 under host ${HOST_VARIANT}" \
  --extra "module_family=set_predictor_v2" "host_variant=${HOST_VARIANT}" "host_config_path=${BASE_CONFIG}" "host_checkpoint=${BASE_CKPT}" "graph_mode=learned_commit" "graph_topk=${TOPK}" "graph_min_detections=${MIN_DETECTIONS}" "graph_min_committed_matches=${MIN_COMMITTED_MATCHES}" "graph_max_detections=${MAX_DETECTIONS}" "graph_max_tracks=${MAX_TRACKS}" "graph_cluster_gate_thresh=${CLUSTER_GATE_THRESH}" "graph_cluster_gate_temp=${CLUSTER_GATE_TEMP}" "graph_cluster_gate_bias=${CLUSTER_GATE_BIAS}"

if ! "${PYTHON_BIN}" "${REPO_ROOT}/scripts/post_experiment_pro_bundle.py" \
  --run-root "${OUT_DIR}" \
  --tag "local_conflict_graph_set_predictor_bundle" \
  --label "${RUN_NAME}" \
  --status "$( [[ "${STATUS}" == "success" ]] && echo ok || echo failed )"; then
  echo "[local-conflict-graph-set-predictor-proxy0213] warning: failed to build Pro review bundle for ${OUT_DIR}" >&2
fi

echo "[local-conflict-graph-set-predictor-proxy0213] status=${STATUS} out_dir=${OUT_DIR} host=${HOST_VARIANT}"
