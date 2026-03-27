#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"

DETECTOR="${1:?detector is required: sw_yolox|sgt}"
SCOPE="${2:-proxy0213}"
MTCR_CHECKPOINT="${3:?mtcr checkpoint path is required}"
OUT_DIR="${4:-${REPO_ROOT}/outputs/mot17_external_${DETECTOR}_mtcr_${SCOPE}_$(date +%Y%m%d_%H%M%S)}"

if [[ "${DETECTOR}" != "sw_yolox" && "${DETECTOR}" != "sgt" ]]; then
  echo "Unsupported detector=${DETECTOR}" >&2
  exit 2
fi
if [[ "${SCOPE}" != "proxy0213" && "${SCOPE}" != "full7" ]]; then
  echo "Unsupported scope=${SCOPE}" >&2
  exit 2
fi
if [[ ! -f "${MTCR_CHECKPOINT}" ]]; then
  echo "MTCR checkpoint not found: ${MTCR_CHECKPOINT}" >&2
  exit 2
fi

BASE_PROFILE="${REPO_ROOT}/configs/profiles/mot17_external_${DETECTOR}_base_${SCOPE}.json"
if [[ ! -f "${BASE_PROFILE}" ]]; then
  echo "Base profile not found: ${BASE_PROFILE}" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"
GENERATED_PROFILE="${OUT_DIR}/mtcr_profile.json"

"${PYTHON_BIN}" - "${REPO_ROOT}" "${BASE_PROFILE}" "${MTCR_CHECKPOINT}" "${GENERATED_PROFILE}" "${DETECTOR}" "${SCOPE}" <<'PY'
import json
import sys
from pathlib import Path

repo_root = Path(sys.argv[1]).resolve()
base_profile = Path(sys.argv[2]).resolve()
checkpoint = str(Path(sys.argv[3]).resolve())
out_profile = Path(sys.argv[4]).resolve()
detector = sys.argv[5]
scope = sys.argv[6]

base_doc = json.loads(base_profile.read_text(encoding="utf-8"))
settings = dict(base_doc["settings"])
overrides = dict(settings.get("config_overrides", {}) or {})
overrides.update(
    {
        "EXP_NAME": f"mot17_external_{detector}_mtcr_{scope}",
        "ASSOC_USE_LAPLACE": False,
        "ASSOC_USE_MTCR": True,
        "ASSOC_MTCR_CHECKPOINT": checkpoint,
    }
)
settings["config_overrides"] = overrides

doc = {
    "description": (
        f"External-det system-recovery line: {detector} detections, clean feature-only host "
        f"plus MTCR association, MOT17 {scope} FRCNN evaluation."
    ),
    "manifest": {
        "line": "external-det/system-rescue",
        "protocol_tier": "private-det",
        "detector_source": detector,
        "association_variant": "mtcr",
        "eval_scope": f"mot17_{scope}_frcnn",
        "mtcr_checkpoint": checkpoint,
    },
    "settings": settings,
}
out_profile.write_text(json.dumps(doc, indent=2, sort_keys=False) + "\n", encoding="utf-8")
print(out_profile)
PY

echo "[run] detector=${DETECTOR} scope=${SCOPE}"
echo "[run] checkpoint=${MTCR_CHECKPOINT}"
echo "[run] profile=${GENERATED_PROFILE}"
echo "[run] out_dir=${OUT_DIR}"

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/run_bytetrack_profile.py" \
  --exp-profile "${GENERATED_PROFILE}" \
  --out-dir "${OUT_DIR}"
