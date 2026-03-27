#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GROUP_JSONL="${1:-${REPO_ROOT}/outputs/runtime_replay_labeled_sw_yolox_base_full7_20260318_full_learned_basefull7_fixreview/labeled_replay_allcand.groups.jsonl}"
OUT_DIR="${2:-${REPO_ROOT}/outputs/competition_assoc_stage0_$(date +%Y%m%d_%H%M%S)}"
HORIZON="${3:-16}"

mkdir -p "${OUT_DIR}"
LOG_PATH="${OUT_DIR}/run.log"
RESULT_CSV="${OUT_DIR}/result.csv"
SUMMARY_CSV="${OUT_DIR}/summary.csv"
MANIFEST_PATH="${OUT_DIR}/run_manifest.json"

"${PYTHON_BIN}" - "${RESULT_CSV}" "${SUMMARY_CSV}" "${GROUP_JSONL}" "${OUT_DIR}" "${HORIZON}" <<'PY'
import csv
import sys
from pathlib import Path

result_csv = Path(sys.argv[1])
summary_csv = Path(sys.argv[2])
group_jsonl = sys.argv[3]
out_dir = sys.argv[4]
horizon = int(sys.argv[5])
fieldnames = [
    "exp_name",
    "group_jsonl",
    "out_dir",
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
    "exp_name": "competition_assoc_stage0",
    "group_jsonl": group_jsonl,
    "out_dir": out_dir,
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
    "status": "running",
}
for path in (result_csv, summary_csv):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)
PY

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/update_run_manifest.py" \
  --manifest "${MANIFEST_PATH}" \
  --set "line=competition_assoc_stage0" \
  --set "group_jsonl=${GROUP_JSONL}" \
  --set "horizon=${HORIZON}" \
  --set "status=running"

if "${PYTHON_BIN}" "${REPO_ROOT}/scripts/build_competition_assoc_cases.py" \
  --group-jsonl "${GROUP_JSONL}" \
  --out-dir "${OUT_DIR}" \
  --horizon "${HORIZON}" >"${LOG_PATH}" 2>&1; then
  STATUS="success"
else
  STATUS="failed"
fi

"${PYTHON_BIN}" - "${OUT_DIR}" "${RESULT_CSV}" "${SUMMARY_CSV}" "${STATUS}" <<'PY'
import csv
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
result_csv = Path(sys.argv[2])
summary_csv = Path(sys.argv[3])
status = sys.argv[4]
summary_json = out_dir / "summary.json"
base = {
    "exp_name": "competition_assoc_stage0",
    "group_jsonl": str(out_dir),
    "out_dir": str(out_dir),
    "horizon": "",
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
    "status": "failed" if status != "success" else "ok",
}
if summary_json.is_file():
    with summary_json.open("r", encoding="utf-8") as f:
        doc = json.load(f)
    base.update(
        {
            "group_jsonl": str(doc.get("group_jsonl", "")),
            "horizon": str(doc.get("horizon", "")),
            "groups": str(doc.get("groups", "")),
            "positive_groups": str(doc.get("positive_groups", "")),
            "background_groups": str(doc.get("background_groups", "")),
            "ambiguous_groups": str(doc.get("ambiguous_groups", "")),
            "recoverable_groups": str(doc.get("recoverable_groups", "")),
            "action_keep": str(doc.get("action_keep", "")),
            "action_rerank": str(doc.get("action_rerank", "")),
            "action_null": str(doc.get("action_null", "")),
            "continuity_bridge": str(doc.get("continuity_bridge", "")),
            "hard_case_groups": str(doc.get("hard_case_groups", "")),
            "ambiguous_rate": str(doc.get("ambiguous_rate", "")),
            "recoverable_rate_among_positive": str(doc.get("recoverable_rate_among_positive", "")),
            "rerank_rate_among_positive": str(doc.get("rerank_rate_among_positive", "")),
            "bridge_rate_among_positive": str(doc.get("bridge_rate_among_positive", "")),
            "hard_case_rate": str(doc.get("hard_case_rate", "")),
        }
    )
fieldnames = list(base.keys())
for path in (result_csv, summary_csv):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(base)
PY

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/update_run_manifest.py" \
  --manifest "${MANIFEST_PATH}" \
  --set "status=$( [[ "${STATUS}" == "success" ]] && echo ok || echo failed )"

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/append_experiment_record.py" \
  --csv "${REPO_ROOT}/outputs/experiment_registry.csv" \
  --kind analysis \
  --status "$( [[ "${STATUS}" == "success" ]] && echo success || echo failed )" \
  --script "scripts/run_competition_assoc_stage0.sh" \
  --dataset "MOT17" \
  --split "train_full7_private_det_dump" \
  --tracker-family "ByteTrack" \
  --variant "competition_assoc_stage0" \
  --tag "competition_assoc" \
  --run-root "${OUT_DIR}" \
  --summary-csv "${SUMMARY_CSV}" \
  --log-path "${LOG_PATH}" \
  --notes "stage0 competition diagnostics from existing runtime replay labeled groups"

echo "[stage0] status=${STATUS} out_dir=${OUT_DIR}"
