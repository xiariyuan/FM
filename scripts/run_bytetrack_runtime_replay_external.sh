#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"

DETECTOR="${1:?detector is required: sw_yolox|sgt}"
SCOPE="${2:-full7}"
RUNTIME_REPLAY_CKPT="${3:?runtime replay checkpoint path is required}"
OUT_DIR="${4:-${REPO_ROOT}/outputs/mot17_external_${DETECTOR}_runtime_replay_${SCOPE}_$(date +%Y%m%d_%H%M%S)}"

if [[ "${DETECTOR}" != "sw_yolox" && "${DETECTOR}" != "sgt" ]]; then
  echo "Unsupported detector=${DETECTOR}" >&2
  exit 2
fi
if [[ "${SCOPE}" != "proxy0213" && "${SCOPE}" != "full7" ]]; then
  echo "Unsupported scope=${SCOPE}" >&2
  exit 2
fi
if [[ ! -f "${RUNTIME_REPLAY_CKPT}" ]]; then
  echo "Runtime replay checkpoint not found: ${RUNTIME_REPLAY_CKPT}" >&2
  exit 2
fi

BASE_PROFILE="${REPO_ROOT}/configs/profiles/mot17_external_${DETECTOR}_base_${SCOPE}.json"
if [[ ! -f "${BASE_PROFILE}" ]]; then
  echo "Base profile not found: ${BASE_PROFILE}" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"
GENERATED_PROFILE="${OUT_DIR}/runtime_replay_profile.json"

"${PYTHON_BIN}" - "${BASE_PROFILE}" "${RUNTIME_REPLAY_CKPT}" "${GENERATED_PROFILE}" "${DETECTOR}" "${SCOPE}" <<'PY'
import json
import sys
from pathlib import Path

base_profile = Path(sys.argv[1]).resolve()
checkpoint = str(Path(sys.argv[2]).resolve())
out_profile = Path(sys.argv[3]).resolve()
detector = sys.argv[4]
scope = sys.argv[5]

base_doc = json.loads(base_profile.read_text(encoding="utf-8"))
settings = dict(base_doc["settings"])
overrides = dict(settings.get("config_overrides", {}) or {})
overrides.update(
    {
        "EXP_NAME": f"mot17_external_{detector}_runtime_replay_{scope}",
        "ASSOC_USE_LAPLACE": False,
        "ASSOC_USE_MTCR": False,
        "ASSOC_USE_RUNTIME_REPLAY": True,
        "ASSOC_RUNTIME_REPLAY_CHECKPOINT": checkpoint,
    }
)
settings["config_overrides"] = overrides

doc = {
    "description": (
        f"External-det system-recovery line: {detector} detections, clean feature-only host "
        f"plus runtime replay reranker, MOT17 {scope} FRCNN evaluation."
    ),
    "manifest": {
        "line": "external-det/system-rescue",
        "protocol_tier": "private-det",
        "detector_source": detector,
        "association_variant": "runtime_replay",
        "eval_scope": f"mot17_{scope}_frcnn",
        "runtime_replay_checkpoint": checkpoint,
    },
    "settings": settings,
}
out_profile.write_text(json.dumps(doc, indent=2, sort_keys=False) + "\n", encoding="utf-8")
print(out_profile)
PY

echo "[run] detector=${DETECTOR} scope=${SCOPE}"
echo "[run] checkpoint=${RUNTIME_REPLAY_CKPT}"
echo "[run] profile=${GENERATED_PROFILE}"
echo "[run] out_dir=${OUT_DIR}"

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/run_bytetrack_profile.py" \
  --exp-profile "${GENERATED_PROFILE}" \
  --out-dir "${OUT_DIR}"
