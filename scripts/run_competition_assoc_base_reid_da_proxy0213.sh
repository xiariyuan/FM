#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
BASE_CONFIG="configs/experiments/bytetrack_fa_mot_mot17_v14_ctrl_base_reid_da_val0213.yaml"
BASE_CKPT="${REPO_ROOT}/outputs/paper_ctrl_mot17_val0213/bytetrack_fa_mot_mot17_v14_ctrl_base_reid_da_val0213/checkpoint_epoch_0.pth"
OUT_DIR="${1:-${REPO_ROOT}/outputs/competition_assoc_base_reid_da_proxy0213_$(date +%Y%m%d_%H%M%S)}"
TOPK="${2:-8}"
HORIZON="${3:-16}"

mkdir -p "${OUT_DIR}"
PROFILE_PATH="${OUT_DIR}/profile.json"
RESULT_CSV="${OUT_DIR}/result.csv"
SUMMARY_CSV="${OUT_DIR}/summary.csv"
LOG_PATH="${OUT_DIR}/pipeline.log"
MANIFEST_PATH="${OUT_DIR}/run_manifest.json"
DUMP_ROOT="${OUT_DIR}/runtime_dump"
LABELED_CSV="${OUT_DIR}/labeled_replay_top${TOPK}.csv"
GROUP_JSONL="${OUT_DIR}/labeled_replay_top${TOPK}.groups.jsonl"
RECOVERABILITY_JSON="${OUT_DIR}/labeled_replay_top${TOPK}.recoverability.json"
COMP_DIR="${OUT_DIR}/competition_cases"

if [[ ! -f "${BASE_CKPT}" ]]; then
  echo "Missing checkpoint: ${BASE_CKPT}" >&2
  exit 2
fi

"${PYTHON_BIN}" - "${RESULT_CSV}" "${SUMMARY_CSV}" "${OUT_DIR}" "${TOPK}" "${HORIZON}" <<'PY'
import csv
import sys
from pathlib import Path

result_csv = Path(sys.argv[1])
summary_csv = Path(sys.argv[2])
out_dir = sys.argv[3]
topk = int(sys.argv[4])
horizon = int(sys.argv[5])
fieldnames = [
    "exp_name",
    "config_path",
    "checkpoint",
    "out_dir",
    "topk",
    "horizon",
    "groups",
    "positive_groups",
    "background_groups",
    "ambiguous_groups",
    "recoverable_groups",
    "action_keep",
    "action_rerank",
    "action_null",
    "continuity_bridge",
    "hard_case_groups",
    "ambiguous_rate",
    "recoverable_rate_among_positive",
    "rerank_rate_among_positive",
    "bridge_rate_among_positive",
    "hard_case_rate",
    "status",
]
row = {
    "exp_name": "competition_assoc_base_reid_da_proxy0213",
    "config_path": "configs/experiments/bytetrack_fa_mot_mot17_v14_ctrl_base_reid_da_val0213.yaml",
    "checkpoint": "",
    "out_dir": out_dir,
    "topk": str(topk),
    "horizon": str(horizon),
    "groups": "",
    "positive_groups": "",
    "background_groups": "",
    "ambiguous_groups": "",
    "recoverable_groups": "",
    "action_keep": "",
    "action_rerank": "",
    "action_null": "",
    "continuity_bridge": "",
    "hard_case_groups": "",
    "ambiguous_rate": "",
    "recoverable_rate_among_positive": "",
    "rerank_rate_among_positive": "",
    "bridge_rate_among_positive": "",
    "hard_case_rate": "",
    "status": "running",
}
for path in (result_csv, summary_csv):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)
PY

"${PYTHON_BIN}" - "${PROFILE_PATH}" "${BASE_CONFIG}" "${BASE_CKPT}" "${DUMP_ROOT}" "${TOPK}" <<'PY'
import json
import sys
from pathlib import Path

profile_path = Path(sys.argv[1]).resolve()
config_path = sys.argv[2]
checkpoint = str(Path(sys.argv[3]).resolve())
dump_root = str(Path(sys.argv[4]).resolve())
topk = int(sys.argv[5])

doc = {
    "description": "Base ReID host dump for competition-aware association on MOT17 proxy0213.",
    "manifest": {
        "line": "competition_assoc_mainline",
        "protocol_tier": "proxy0213_internal",
        "host_variant": "base_reid_da",
        "eval_scope": "mot17_proxy0213",
        "inference_model": checkpoint,
        "dump_root": dump_root,
        "topk": topk,
    },
    "settings": {
        "config_path": config_path,
        "inference_dataset": "MOT17",
        "inference_split": "train",
        "inference_model": checkpoint,
        "config_overrides": {
            "EXP_NAME": "competition_assoc_base_reid_da_proxy0213_dump",
            "EVAL_ONLY_VAL": True,
            "RUN_TRACKEVAL": True,
            "VAL_SEQUENCES": ["MOT17-02", "MOT17-13"],
            "ASSOC_USE_LAPLACE": False,
            "ASSOC_USE_MTCR": False,
            "ASSOC_USE_RUNTIME_REPLAY": False,
            "ASSOC_RUNTIME_DUMP_PATH": dump_root,
            "ASSOC_RUNTIME_DUMP_TOPK": topk,
            "ASSOC_RUNTIME_DUMP_MIN_SCORE": 0.0,
            "ASSOC_RUNTIME_DUMP_SAVE_TENSORS": True,
            "ASSOC_RUNTIME_DUMP_NPZ_EVERY_N_GROUPS": 2048,
        },
    },
}
profile_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
print(profile_path)
PY

STATUS="failed"
{
  echo "[step] run_bytetrack_profile"
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/run_bytetrack_profile.py" \
    --exp-profile "${PROFILE_PATH}" \
    --out-dir "${OUT_DIR}"

  echo "[step] build_runtime_assoc_replay_labels"
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/build_runtime_assoc_replay_labels.py" \
    --dump-root "${DUMP_ROOT}" \
    --dataset MOT17 \
    --data-root /gemini/code/datasets \
    --split train \
    --split-part full \
    --out-csv "${LABELED_CSV}" \
    --out-group-jsonl "${GROUP_JSONL}" \
    --out-recoverability-json "${RECOVERABILITY_JSON}" \
    --topk "${TOPK}" \
    --rank-score-col refined_score \
    --ambiguity-margin 0.10

  echo "[step] build_competition_assoc_cases"
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/build_competition_assoc_cases.py" \
    --group-jsonl "${GROUP_JSONL}" \
    --out-dir "${COMP_DIR}" \
    --horizon "${HORIZON}"

  STATUS="success"
} >"${LOG_PATH}" 2>&1 || STATUS="failed"

if [[ "${STATUS}" == "success" ]]; then
  if [[ ! -f "${GROUP_JSONL}" || ! -f "${COMP_DIR}/summary.json" ]]; then
    echo "[check] missing expected downstream artifacts; forcing failed status" >>"${LOG_PATH}"
    STATUS="failed"
  fi
fi

"${PYTHON_BIN}" - "${RESULT_CSV}" "${SUMMARY_CSV}" "${COMP_DIR}/summary.json" "${BASE_CKPT}" "${OUT_DIR}" "${TOPK}" "${HORIZON}" "${STATUS}" <<'PY'
import csv
import json
import sys
from pathlib import Path

result_csv = Path(sys.argv[1])
summary_csv = Path(sys.argv[2])
summary_json = Path(sys.argv[3])
checkpoint = sys.argv[4]
out_dir = sys.argv[5]
topk = sys.argv[6]
horizon = sys.argv[7]
status = sys.argv[8]

base = {
    "exp_name": "competition_assoc_base_reid_da_proxy0213",
    "config_path": "configs/experiments/bytetrack_fa_mot_mot17_v14_ctrl_base_reid_da_val0213.yaml",
    "checkpoint": checkpoint,
    "out_dir": out_dir,
    "topk": topk,
    "horizon": horizon,
    "groups": "",
    "positive_groups": "",
    "background_groups": "",
    "ambiguous_groups": "",
    "recoverable_groups": "",
    "action_keep": "",
    "action_rerank": "",
    "action_null": "",
    "continuity_bridge": "",
    "hard_case_groups": "",
    "ambiguous_rate": "",
    "recoverable_rate_among_positive": "",
    "rerank_rate_among_positive": "",
    "bridge_rate_among_positive": "",
    "hard_case_rate": "",
    "status": "ok" if status == "success" else "failed",
}
if summary_json.is_file():
    with summary_json.open("r", encoding="utf-8") as f:
        doc = json.load(f)
    for key in (
        "groups",
        "positive_groups",
        "background_groups",
        "ambiguous_groups",
        "recoverable_groups",
        "action_keep",
        "action_rerank",
        "action_null",
        "continuity_bridge",
        "hard_case_groups",
        "ambiguous_rate",
        "recoverable_rate_among_positive",
        "rerank_rate_among_positive",
        "bridge_rate_among_positive",
        "hard_case_rate",
    ):
        base[key] = str(doc.get(key, ""))
fieldnames = list(base.keys())
for path in (result_csv, summary_csv):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(base)
PY

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/update_run_manifest.py" \
  --manifest "${MANIFEST_PATH}" \
  --set "status=$( [[ "${STATUS}" == "success" ]] && echo ok || echo failed )" \
  --set "topk=${TOPK}" \
  --set "horizon=${HORIZON}"

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/append_experiment_record.py" \
  --csv "${REPO_ROOT}/outputs/experiment_registry.csv" \
  --kind analysis \
  --status "$( [[ "${STATUS}" == "success" ]] && echo success || echo failed )" \
  --script "scripts/run_competition_assoc_base_reid_da_proxy0213.sh" \
  --dataset "MOT17" \
  --split "val0213_proxy" \
  --tracker-family "ByteTrack" \
  --variant "competition_assoc_base_reid_da_proxy0213" \
  --tag "competition_assoc_mainline" \
  --run-root "${OUT_DIR}" \
  --summary-csv "${SUMMARY_CSV}" \
  --log-path "${LOG_PATH}" \
  --notes "base_reid_da host proxy0213 dump + labels + competition cases" \
  --extra "checkpoint=${BASE_CKPT}" "topk=${TOPK}" "horizon=${HORIZON}"

if ! "${PYTHON_BIN}" "${REPO_ROOT}/scripts/post_experiment_pro_bundle.py" \
  --run-root "${OUT_DIR}" \
  --tag "competition_assoc_proxy_bundle" \
  --label "competition_assoc_proxy0213" \
  --status "$( [[ "${STATUS}" == "success" ]] && echo ok || echo failed )"; then
  echo "[competition-assoc] warning: failed to build Pro review bundle for ${OUT_DIR}" >&2
fi

echo "[competition-assoc] status=${STATUS} out_dir=${OUT_DIR}"
