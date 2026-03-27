#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"

OUT_DIR="${1:-${REPO_ROOT}/outputs/local_conflict_graph_host_migration_proxy0213_$(date +%Y%m%d_%H%M%S)}"
HOST_CONFIG="${2:-${REPO_ROOT}/configs/experiments/bytetrack_fa_mot_mot17_v17_local_conflict_commit_hostv15_val0213.yaml}"
HOST_CKPT="${3:-${REPO_ROOT}/outputs/paper_ctrl_mot17_val0213/bytetrack_fa_mot_mot17_v14_ctrl_base_reid_da_val0213/checkpoint_epoch_0.pth}"
HOST_VARIANT="${4:-v15_laplace_reid_da_val0213}"
GRAPH_CKPT="${5:-${REPO_ROOT}/outputs/local_conflict_graph_learned_commit_next12h_20260324_165558/01_stage1/best.pt}"
TOPK="${6:-8}"
MIN_DETECTIONS="${7:-2}"
MIN_COMMITTED_MATCHES="${8:-2}"
MAX_DETECTIONS="${9:-8}"
MAX_TRACKS="${10:-32}"

if [[ ! -f "${HOST_CONFIG}" ]]; then
  echo "Missing host config: ${HOST_CONFIG}" >&2
  exit 2
fi
if [[ ! -f "${HOST_CKPT}" ]]; then
  echo "Missing host checkpoint: ${HOST_CKPT}" >&2
  exit 2
fi
if [[ ! -f "${GRAPH_CKPT}" ]]; then
  echo "Missing graph checkpoint: ${GRAPH_CKPT}" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"

HOST_ONLY_DIR="${OUT_DIR}/00_host_only"
HOST_PLUS_DIR="${OUT_DIR}/01_host_plus_learned_commit"
SUMMARY_CSV="${OUT_DIR}/summary.csv"
RESULT_CSV="${OUT_DIR}/result.csv"
RUN_LOG="${OUT_DIR}/run.log"
MANIFEST_PATH="${OUT_DIR}/run_manifest.json"

refresh_pair_records() {
  "${PYTHON_BIN}" - "${SUMMARY_CSV}" "${RESULT_CSV}" "${HOST_ONLY_DIR}/summary.csv" "${HOST_PLUS_DIR}/summary.csv" "${OUT_DIR}" "${HOST_VARIANT}" "${HOST_CONFIG}" "${HOST_CKPT}" "${GRAPH_CKPT}" "${TOPK}" "${MIN_DETECTIONS}" "${MIN_COMMITTED_MATCHES}" "${MAX_DETECTIONS}" "${MAX_TRACKS}" <<'PY'
import csv
import math
import sys
from pathlib import Path

summary_csv = Path(sys.argv[1])
result_csv = Path(sys.argv[2])
host_only_summary = Path(sys.argv[3])
host_plus_summary = Path(sys.argv[4])
out_dir = sys.argv[5]
host_variant = sys.argv[6]
host_config = sys.argv[7]
host_ckpt = sys.argv[8]
graph_ckpt = sys.argv[9]
topk = sys.argv[10]
min_detections = sys.argv[11]
min_committed_matches = sys.argv[12]
max_detections = sys.argv[13]
max_tracks = sys.argv[14]

summary_fields = [
    "arm",
    "exp_name",
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
    "out_dir",
    "eligible_clusters",
    "replaced_clusters",
    "matched_dets",
    "deferred_dets",
    "blocked_tracks",
    "skipped_large_clusters",
    "HOTA",
    "AssA",
    "IDF1",
    "MOTA",
    "IDSW",
    "status",
]

result_fields = [
    "exp_name",
    "host_variant",
    "host_config_path",
    "host_checkpoint",
    "graph_checkpoint",
    "graph_topk",
    "graph_min_detections",
    "graph_min_committed_matches",
    "graph_max_detections",
    "graph_max_tracks",
    "delta_HOTA",
    "delta_AssA",
    "delta_IDF1",
    "delta_MOTA",
    "delta_IDSW",
    "host_only_dir",
    "host_plus_learned_dir",
    "status",
]

def read_single_row(path: Path):
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            return dict(row)
    return None

def running_row(arm: str, graph_mode: str, arm_out_dir: str):
    return {
        "arm": arm,
        "exp_name": f"local_conflict_graph_host_migration_proxy0213_{host_variant}",
        "config_path": host_config,
        "checkpoint": host_ckpt,
        "host_variant": host_variant,
        "host_config_path": host_config,
        "host_checkpoint": host_ckpt,
        "graph_mode": graph_mode,
        "graph_checkpoint": graph_ckpt if graph_mode != "disabled" else "",
        "graph_topk": topk,
        "graph_min_detections": min_detections,
        "graph_min_committed_matches": min_committed_matches,
        "graph_max_detections": max_detections,
        "graph_max_tracks": max_tracks,
        "out_dir": arm_out_dir,
        "eligible_clusters": "",
        "replaced_clusters": "",
        "matched_dets": "",
        "deferred_dets": "",
        "blocked_tracks": "",
        "skipped_large_clusters": "",
        "HOTA": "",
        "AssA": "",
        "IDF1": "",
        "MOTA": "",
        "IDSW": "",
        "status": "running",
    }

host_only_row = read_single_row(host_only_summary) or running_row("host_only", "disabled", str(host_only_summary.parent))
host_plus_row = read_single_row(host_plus_summary) or running_row("host_plus_learned_commit", "learned_commit", str(host_plus_summary.parent))
host_only_row["arm"] = "host_only"
host_plus_row["arm"] = "host_plus_learned_commit"

for row in (host_only_row, host_plus_row):
    for field in summary_fields:
        row.setdefault(field, "")

with summary_csv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=summary_fields)
    writer.writeheader()
    writer.writerow({field: host_only_row.get(field, "") for field in summary_fields})
    writer.writerow({field: host_plus_row.get(field, "") for field in summary_fields})

pair_status = "running"
if "failed" in {host_only_row.get("status", ""), host_plus_row.get("status", "")}:
    pair_status = "failed"
elif host_only_row.get("status") == "ok" and host_plus_row.get("status") == "ok":
    pair_status = "ok"

def parse_float(value: str) -> float | None:
    try:
        return float(value)
    except Exception:
        return None

result_row = {
    "exp_name": f"local_conflict_graph_host_migration_proxy0213_{host_variant}",
    "host_variant": host_variant,
    "host_config_path": host_config,
    "host_checkpoint": host_ckpt,
    "graph_checkpoint": graph_ckpt,
    "graph_topk": topk,
    "graph_min_detections": min_detections,
    "graph_min_committed_matches": min_committed_matches,
    "graph_max_detections": max_detections,
    "graph_max_tracks": max_tracks,
    "delta_HOTA": "",
    "delta_AssA": "",
    "delta_IDF1": "",
    "delta_MOTA": "",
    "delta_IDSW": "",
    "host_only_dir": str(host_only_summary.parent),
    "host_plus_learned_dir": str(host_plus_summary.parent),
    "status": pair_status,
}

if pair_status == "ok":
    for metric in ("HOTA", "AssA", "IDF1", "MOTA", "IDSW"):
        base = parse_float(host_only_row.get(metric, ""))
        plus = parse_float(host_plus_row.get(metric, ""))
        if base is not None and plus is not None:
            result_row[f"delta_{metric}"] = str(plus - base)

with result_csv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=result_fields)
    writer.writeheader()
    writer.writerow(result_row)
PY
}

run_host_only() {
  local out_dir="${HOST_ONLY_DIR}"
  local profile_path="${out_dir}/profile.json"
  local result_csv="${out_dir}/result.csv"
  local summary_csv="${out_dir}/summary.csv"
  local log_path="${out_dir}/run.log"
  local manifest_path="${out_dir}/run_manifest.json"
  mkdir -p "${out_dir}"

  "${PYTHON_BIN}" - "${result_csv}" "${summary_csv}" "${HOST_CONFIG}" "${HOST_CKPT}" "${HOST_VARIANT}" "${out_dir}" "${TOPK}" "${MIN_DETECTIONS}" "${MIN_COMMITTED_MATCHES}" "${MAX_DETECTIONS}" "${MAX_TRACKS}" <<'PY'
import csv
import sys
from pathlib import Path

result_csv = Path(sys.argv[1])
summary_csv = Path(sys.argv[2])
host_config = sys.argv[3]
host_ckpt = sys.argv[4]
host_variant = sys.argv[5]
out_dir = sys.argv[6]
topk = sys.argv[7]
min_detections = sys.argv[8]
min_committed_matches = sys.argv[9]
max_detections = sys.argv[10]
max_tracks = sys.argv[11]

fieldnames = [
    "exp_name",
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
    "out_dir",
    "eligible_clusters",
    "replaced_clusters",
    "matched_dets",
    "deferred_dets",
    "blocked_tracks",
    "skipped_large_clusters",
    "HOTA",
    "AssA",
    "IDF1",
    "MOTA",
    "IDSW",
    "status",
]
row = {
    "exp_name": f"{host_variant}_proxy0213_host_only",
    "config_path": host_config,
    "checkpoint": host_ckpt,
    "host_variant": host_variant,
    "host_config_path": host_config,
    "host_checkpoint": host_ckpt,
    "graph_mode": "disabled",
    "graph_checkpoint": "",
    "graph_topk": topk,
    "graph_min_detections": min_detections,
    "graph_min_committed_matches": min_committed_matches,
    "graph_max_detections": max_detections,
    "graph_max_tracks": max_tracks,
    "out_dir": out_dir,
    "eligible_clusters": "0",
    "replaced_clusters": "0",
    "matched_dets": "0",
    "deferred_dets": "0",
    "blocked_tracks": "0",
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

  "${PYTHON_BIN}" - "${profile_path}" "${HOST_CONFIG}" "${HOST_CKPT}" "${HOST_VARIANT}" <<'PY'
import json
import sys
from pathlib import Path

profile_path = Path(sys.argv[1]).resolve()
host_config = sys.argv[2]
host_ckpt = str(Path(sys.argv[3]).resolve())
host_variant = sys.argv[4]

doc = {
    "description": f"Host-only proxy0213 baseline for host migration check under {host_variant}.",
    "manifest": {
        "line": "local_conflict_commit_host_migration",
        "protocol_tier": "proxy0213_internal",
        "host_variant": host_variant,
        "host_config_path": host_config,
        "host_checkpoint": host_ckpt,
        "eval_scope": "mot17_proxy0213",
        "inference_model": host_ckpt,
        "graph_mode": "disabled",
    },
    "settings": {
        "config_path": host_config,
        "inference_dataset": "MOT17",
        "inference_split": "train",
        "inference_model": host_ckpt,
        "config_overrides": {
            "EXP_NAME": f"{host_variant}_proxy0213_host_only",
            "EVAL_ONLY_VAL": True,
            "RUN_TRACKEVAL": True,
            "VAL_SEQUENCES": ["MOT17-02", "MOT17-13"],
            "ASSOC_USE_LOCAL_CONFLICT_GRAPH": False,
            "ASSOC_LOCAL_CONFLICT_GRAPH_MODE": "disabled",
        },
    },
}
profile_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
PY

  local status="failed"
  if "${PYTHON_BIN}" "${REPO_ROOT}/scripts/run_bytetrack_profile.py" \
    --exp-profile "${profile_path}" \
    --out-dir "${out_dir}" >"${log_path}" 2>&1; then
    status="success"
  fi

  "${PYTHON_BIN}" - "${result_csv}" "${summary_csv}" "${manifest_path}" "${HOST_CONFIG}" "${HOST_CKPT}" "${HOST_VARIANT}" "${out_dir}" "${TOPK}" "${MIN_DETECTIONS}" "${MIN_COMMITTED_MATCHES}" "${MAX_DETECTIONS}" "${MAX_TRACKS}" "${status}" <<'PY'
import csv
import json
import sys
from pathlib import Path

result_csv = Path(sys.argv[1])
summary_csv = Path(sys.argv[2])
manifest_path = Path(sys.argv[3])
host_config = sys.argv[4]
host_ckpt = sys.argv[5]
host_variant = sys.argv[6]
out_dir = sys.argv[7]
topk = sys.argv[8]
min_detections = sys.argv[9]
min_committed_matches = sys.argv[10]
max_detections = sys.argv[11]
max_tracks = sys.argv[12]
status = sys.argv[13]

row = {
    "exp_name": f"{host_variant}_proxy0213_host_only",
    "config_path": host_config,
    "checkpoint": host_ckpt,
    "host_variant": host_variant,
    "host_config_path": host_config,
    "host_checkpoint": host_ckpt,
    "graph_mode": "disabled",
    "graph_checkpoint": "",
    "graph_topk": topk,
    "graph_min_detections": min_detections,
    "graph_min_committed_matches": min_committed_matches,
    "graph_max_detections": max_detections,
    "graph_max_tracks": max_tracks,
    "out_dir": out_dir,
    "eligible_clusters": "0",
    "replaced_clusters": "0",
    "matched_dets": "0",
    "deferred_dets": "0",
    "blocked_tracks": "0",
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

  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/append_experiment_record.py" \
    --csv "${REPO_ROOT}/outputs/experiment_registry.csv" \
    --kind eval \
    --status "$( [[ "${status}" == "success" ]] && echo success || echo failed )" \
    --script "scripts/run_local_conflict_graph_host_migration_proxy0213.sh" \
    --dataset "MOT17" \
    --split "val0213_proxy" \
    --tracker-family "ByteTrack" \
    --variant "${HOST_VARIANT}_proxy0213_host_only" \
    --tag "local_conflict_commit_host_migration" \
    --run-root "${out_dir}" \
    --summary-csv "${summary_csv}" \
    --checkpoint "${HOST_CKPT}" \
    --log-path "${log_path}" \
    --notes "host-only proxy0213 baseline for host migration under ${HOST_VARIANT}" \
    --extra "host_variant=${HOST_VARIANT}" "host_config_path=${HOST_CONFIG}" "host_checkpoint=${HOST_CKPT}" "graph_mode=disabled"

  if ! "${PYTHON_BIN}" "${REPO_ROOT}/scripts/post_experiment_pro_bundle.py" \
    --run-root "${out_dir}" \
    --tag "local_conflict_graph_host_migration_bundle" \
    --label "${HOST_VARIANT}_proxy0213_host_only" \
    --status "$( [[ "${status}" == "success" ]] && echo ok || echo failed )"; then
    echo "[host-migration-proxy0213] warning: failed to build Pro review bundle for ${out_dir}" >&2
  fi
}

{
  echo "[host-migration-proxy0213] out_dir=${OUT_DIR}"
  echo "[host-migration-proxy0213] host_variant=${HOST_VARIANT}"
  echo "[host-migration-proxy0213] host_config=${HOST_CONFIG}"
  echo "[host-migration-proxy0213] host_ckpt=${HOST_CKPT}"
  echo "[host-migration-proxy0213] graph_ckpt=${GRAPH_CKPT}"
} >"${RUN_LOG}"

refresh_pair_records

run_host_only
refresh_pair_records

BASE_CONFIG="${HOST_CONFIG}" \
BASE_CKPT="${HOST_CKPT}" \
HOST_VARIANT="${HOST_VARIANT}" \
bash "${REPO_ROOT}/scripts/run_local_conflict_graph_learned_commit_proxy0213.sh" \
  "${HOST_PLUS_DIR}" \
  "${GRAPH_CKPT}" \
  "${TOPK}" \
  "${MIN_DETECTIONS}" \
  "${MIN_COMMITTED_MATCHES}" \
  "${MAX_DETECTIONS}" \
  "${MAX_TRACKS}" >>"${RUN_LOG}" 2>&1

refresh_pair_records

PAIR_STATUS="$("${PYTHON_BIN}" - "${RESULT_CSV}" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
with path.open("r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(row.get("status", "failed"))
        break
PY
)"

"${PYTHON_BIN}" - "${MANIFEST_PATH}" "${OUT_DIR}" "${HOST_VARIANT}" "${HOST_CONFIG}" "${HOST_CKPT}" "${GRAPH_CKPT}" "${PAIR_STATUS}" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
host_variant = sys.argv[3]
host_config = sys.argv[4]
host_ckpt = sys.argv[5]
graph_ckpt = sys.argv[6]
status = sys.argv[7]

doc = {
    "created_at": __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds"),
    "runner": "scripts/run_local_conflict_graph_host_migration_proxy0213.sh",
    "status": status,
    "host_variant": host_variant,
    "host_config_path": host_config,
    "host_checkpoint": host_ckpt,
    "graph_checkpoint": graph_ckpt,
    "outputs": {
        "summary_csv": str(out_dir / "summary.csv"),
        "result_csv": str(out_dir / "result.csv"),
        "host_only_dir": str(out_dir / "00_host_only"),
        "host_plus_learned_dir": str(out_dir / "01_host_plus_learned_commit"),
    },
}
manifest_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
PY

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/append_experiment_record.py" \
  --csv "${REPO_ROOT}/outputs/experiment_registry.csv" \
  --kind analysis \
  --status "${PAIR_STATUS}" \
  --script "scripts/run_local_conflict_graph_host_migration_proxy0213.sh" \
  --dataset "MOT17" \
  --split "val0213_proxy" \
  --tracker-family "ByteTrack" \
  --variant "local_conflict_graph_host_migration_proxy0213_${HOST_VARIANT}" \
  --tag "local_conflict_commit_host_migration" \
  --run-root "${OUT_DIR}" \
  --summary-csv "${RESULT_CSV}" \
  --checkpoint "${GRAPH_CKPT}" \
  --log-path "${RUN_LOG}" \
  --notes "paired host migration proxy0213 for ${HOST_VARIANT}" \
  --extra "host_variant=${HOST_VARIANT}" "host_config_path=${HOST_CONFIG}" "host_checkpoint=${HOST_CKPT}" "graph_checkpoint=${GRAPH_CKPT}"

if ! "${PYTHON_BIN}" "${REPO_ROOT}/scripts/post_experiment_pro_bundle.py" \
  --run-root "${OUT_DIR}" \
  --tag "local_conflict_graph_host_migration_bundle" \
  --label "local_conflict_graph_host_migration_proxy0213_${HOST_VARIANT}" \
  --status "${PAIR_STATUS}"; then
  echo "[host-migration-proxy0213] warning: failed to build Pro review bundle for ${OUT_DIR}" >&2
fi

echo "[host-migration-proxy0213] status=${PAIR_STATUS} out_dir=${OUT_DIR}"
