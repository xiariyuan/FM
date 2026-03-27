#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
BASE_CONFIG="${REPO_ROOT}/configs/experiments/bytetrack_fa_mot_mot17_v14_ctrl_base_reid_da_val0213.yaml"
BASE_CKPT="${REPO_ROOT}/outputs/paper_ctrl_mot17_val0213/bytetrack_fa_mot_mot17_v14_ctrl_base_reid_da_val0213/checkpoint_epoch_0.pth"
CTRL_CKPT_DEFAULT="${REPO_ROOT}/outputs/competition_assoc_stage1_fix1_full12/best.pt"
ORACLE_CSV_DEFAULT="${REPO_ROOT}/outputs/competition_assoc_base_reid_da_proxy0213_hybriddumpfix/competition_cases/competition_cases.csv"

MODE="${1:-noop}"
OUT_DIR="${2:-${REPO_ROOT}/outputs/competition_assoc_online_${MODE}_proxy0213_$(date +%Y%m%d_%H%M%S)}"
CTRL_CKPT="${3:-${CTRL_CKPT_DEFAULT}}"
TOPK="${4:-8}"
DELTA_SCALE="${5:-0.05}"
MARGIN_THRESHOLD="${6:-}"
ORACLE_CSV="${7:-${ORACLE_CSV_DEFAULT}}"

if [[ "${MODE}" != "noop" && "${MODE}" != "rerank_only" && "${MODE}" != "rerank_minimal" && "${MODE}" != "oracle_rerank" ]]; then
  echo "Unsupported MODE=${MODE}; expected noop, rerank_only, rerank_minimal, or oracle_rerank" >&2
  exit 2
fi
if [[ ! -f "${BASE_CONFIG}" ]]; then
  echo "Missing config: ${BASE_CONFIG}" >&2
  exit 2
fi
if [[ ! -f "${BASE_CKPT}" ]]; then
  echo "Missing base checkpoint: ${BASE_CKPT}" >&2
  exit 2
fi
if [[ "${MODE}" != "oracle_rerank" && ! -f "${CTRL_CKPT}" ]]; then
  echo "Missing controller checkpoint: ${CTRL_CKPT}" >&2
  exit 2
fi
if [[ "${MODE}" == "oracle_rerank" && ! -f "${ORACLE_CSV}" ]]; then
  echo "Missing oracle csv: ${ORACLE_CSV}" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"

PROFILE_PATH="${OUT_DIR}/profile.json"
RESULT_CSV="${OUT_DIR}/result.csv"
SUMMARY_CSV="${OUT_DIR}/summary.csv"
LOG_PATH="${OUT_DIR}/run.log"
MANIFEST_PATH="${OUT_DIR}/run_manifest.json"

"${PYTHON_BIN}" - "${RESULT_CSV}" "${SUMMARY_CSV}" "${OUT_DIR}" "${CTRL_CKPT}" "${MODE}" "${TOPK}" "${DELTA_SCALE}" "${MARGIN_THRESHOLD}" "${ORACLE_CSV}" <<'PY'
import csv
import sys
from pathlib import Path

result_csv = Path(sys.argv[1])
summary_csv = Path(sys.argv[2])
out_dir = sys.argv[3]
ctrl_ckpt = sys.argv[4]
mode = sys.argv[5]
topk = sys.argv[6]
delta_scale = sys.argv[7]
margin_threshold = sys.argv[8]
oracle_csv = sys.argv[9]

fieldnames = [
    "exp_name",
    "config_path",
    "checkpoint",
    "controller_checkpoint",
    "oracle_csv",
    "controller_mode",
    "competition_topk",
    "competition_delta_scale",
    "competition_margin_threshold",
    "out_dir",
    "HOTA",
    "AssA",
    "IDF1",
    "MOTA",
    "IDSW",
    "status",
]
row = {
    "exp_name": f"competition_assoc_online_{mode}_proxy0213",
    "config_path": "configs/experiments/bytetrack_fa_mot_mot17_v14_ctrl_base_reid_da_val0213.yaml",
    "checkpoint": "",
    "controller_checkpoint": ctrl_ckpt,
    "oracle_csv": oracle_csv if mode == "oracle_rerank" else "",
    "controller_mode": mode,
    "competition_topk": topk,
    "competition_delta_scale": delta_scale,
    "competition_margin_threshold": margin_threshold,
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

"${PYTHON_BIN}" - "${PROFILE_PATH}" "${BASE_CONFIG}" "${BASE_CKPT}" "${CTRL_CKPT}" "${MODE}" "${TOPK}" "${DELTA_SCALE}" "${MARGIN_THRESHOLD}" "${ORACLE_CSV}" <<'PY'
import json
import sys
from pathlib import Path

profile_path = Path(sys.argv[1]).resolve()
config_path = sys.argv[2]
base_ckpt = str(Path(sys.argv[3]).resolve())
ctrl_ckpt = str(Path(sys.argv[4]).resolve())
mode = sys.argv[5]
topk = int(sys.argv[6])
delta_scale = float(sys.argv[7])
margin_threshold_raw = sys.argv[8]
oracle_csv = str(Path(sys.argv[9]).resolve()) if sys.argv[9] else ""

overrides = {
    "EXP_NAME": f"competition_assoc_online_{mode}_proxy0213",
    "EVAL_ONLY_VAL": True,
    "RUN_TRACKEVAL": True,
    "VAL_SEQUENCES": ["MOT17-02", "MOT17-13"],
    "ASSOC_USE_LAPLACE": False,
    "ASSOC_USE_MTCR": False,
    "ASSOC_USE_RUNTIME_REPLAY": False,
    "ASSOC_USE_COMPETITION": True,
    "ASSOC_COMPETITION_CHECKPOINT": ctrl_ckpt,
    "ASSOC_COMPETITION_MODE": mode,
    "ASSOC_COMPETITION_TOPK": topk,
    "ASSOC_COMPETITION_DELTA_SCALE": delta_scale,
}
if margin_threshold_raw:
    overrides["ASSOC_COMPETITION_MARGIN_THRESHOLD"] = float(margin_threshold_raw)
if mode == "oracle_rerank":
    overrides["ASSOC_USE_COMPETITION_ORACLE"] = True
    overrides["ASSOC_COMPETITION_ORACLE_CSV"] = oracle_csv

doc = {
    "description": f"Competition association online proxy0213 run ({mode}).",
    "manifest": {
        "line": "competition_assoc_mainline",
        "protocol_tier": "proxy0213_internal",
        "host_variant": "base_reid_da",
        "eval_scope": "mot17_proxy0213",
        "inference_model": base_ckpt,
      "controller_checkpoint": ctrl_ckpt,
      "oracle_csv": oracle_csv if mode == "oracle_rerank" else "",
      "controller_mode": mode,
      "competition_topk": topk,
      "competition_delta_scale": delta_scale,
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

"${PYTHON_BIN}" - "${RESULT_CSV}" "${SUMMARY_CSV}" "${MANIFEST_PATH}" "${BASE_CKPT}" "${CTRL_CKPT}" "${OUT_DIR}" "${MODE}" "${TOPK}" "${DELTA_SCALE}" "${MARGIN_THRESHOLD}" "${STATUS}" "${ORACLE_CSV}" <<'PY'
import csv
import json
import sys
from pathlib import Path

result_csv = Path(sys.argv[1])
summary_csv = Path(sys.argv[2])
manifest_path = Path(sys.argv[3])
base_ckpt = sys.argv[4]
ctrl_ckpt = sys.argv[5]
out_dir = sys.argv[6]
mode = sys.argv[7]
topk = sys.argv[8]
delta_scale = sys.argv[9]
margin_threshold = sys.argv[10]
status = sys.argv[11]
oracle_csv = sys.argv[12]

row = {
    "exp_name": f"competition_assoc_online_{mode}_proxy0213",
    "config_path": "configs/experiments/bytetrack_fa_mot_mot17_v14_ctrl_base_reid_da_val0213.yaml",
    "checkpoint": base_ckpt,
    "controller_checkpoint": ctrl_ckpt,
    "oracle_csv": oracle_csv if mode == "oracle_rerank" else "",
    "controller_mode": mode,
    "competition_topk": topk,
    "competition_delta_scale": delta_scale,
    "competition_margin_threshold": margin_threshold,
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
  --script "scripts/run_competition_assoc_online_proxy0213.sh" \
  --dataset "MOT17" \
  --split "val0213_proxy" \
  --tracker-family "ByteTrack" \
  --variant "competition_assoc_online_proxy0213" \
  --tag "competition_assoc_mainline" \
  --run-root "${OUT_DIR}" \
  --summary-csv "${SUMMARY_CSV}" \
  --checkpoint "${BASE_CKPT}" \
  --log-path "${LOG_PATH}" \
  --notes "competition-aware association online proxy0213 run" \
  --extra "controller_checkpoint=${CTRL_CKPT}" "controller_mode=${MODE}" "competition_topk=${TOPK}" "competition_delta_scale=${DELTA_SCALE}" "competition_margin_threshold=${MARGIN_THRESHOLD}" "oracle_csv=${ORACLE_CSV}"

if ! "${PYTHON_BIN}" "${REPO_ROOT}/scripts/post_experiment_pro_bundle.py" \
  --run-root "${OUT_DIR}" \
  --tag "competition_assoc_online_bundle" \
  --label "competition_assoc_online_proxy0213" \
  --status "$( [[ "${STATUS}" == "success" ]] && echo ok || echo failed )"; then
  echo "[competition-online] warning: failed to build Pro review bundle for ${OUT_DIR}" >&2
fi

echo "[competition-online] status=${STATUS} mode=${MODE} out_dir=${OUT_DIR}"
