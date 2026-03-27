#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"

DETECTOR="${1:?detector is required: sw_yolox|sgt}"
MODE="${2:?mode is required: base|heuristic|mtcr}"
SCOPE="${3:-proxy0213}"
DUMP_DIR="${4:?dump dir is required}"
OUT_DIR="${5:-${REPO_ROOT}/outputs/mot17_external_${DETECTOR}_${MODE}_${SCOPE}_dump_$(date +%Y%m%d_%H%M%S)}"
MTCR_CHECKPOINT="${MTCR_CHECKPOINT:-}"
DUMP_TOPK="${DUMP_TOPK:-8}"
DUMP_MIN_SCORE="${DUMP_MIN_SCORE:-0.0}"
DUMP_SAVE_TENSORS="${DUMP_SAVE_TENSORS:-0}"
DUMP_NPZ_EVERY_N_GROUPS="${DUMP_NPZ_EVERY_N_GROUPS:-2048}"

if [[ "${DETECTOR}" != "sw_yolox" && "${DETECTOR}" != "sgt" ]]; then
  echo "Unsupported detector=${DETECTOR}" >&2
  exit 2
fi
if [[ "${MODE}" != "base" && "${MODE}" != "heuristic" && "${MODE}" != "mtcr" ]]; then
  echo "Unsupported mode=${MODE}" >&2
  exit 2
fi
if [[ "${SCOPE}" != "proxy0213" && "${SCOPE}" != "full7" ]]; then
  echo "Unsupported scope=${SCOPE}" >&2
  exit 2
fi
if [[ "${MODE}" == "mtcr" && ! -f "${MTCR_CHECKPOINT}" ]]; then
  echo "MODE=mtcr requires MTCR_CHECKPOINT to point to an existing checkpoint" >&2
  exit 2
fi

if [[ "${MODE}" == "mtcr" ]]; then
  BASE_PROFILE="${REPO_ROOT}/configs/profiles/mot17_external_${DETECTOR}_base_${SCOPE}.json"
else
  BASE_PROFILE="${REPO_ROOT}/configs/profiles/mot17_external_${DETECTOR}_${MODE}_${SCOPE}.json"
fi
if [[ ! -f "${BASE_PROFILE}" ]]; then
  echo "Base profile not found: ${BASE_PROFILE}" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}" "${DUMP_DIR}"
GENERATED_PROFILE="${OUT_DIR}/assoc_dump_profile.json"

"${PYTHON_BIN}" - "${BASE_PROFILE}" "${GENERATED_PROFILE}" "${DETECTOR}" "${MODE}" "${SCOPE}" "${DUMP_DIR}" "${DUMP_TOPK}" "${DUMP_MIN_SCORE}" "${MTCR_CHECKPOINT}" <<'PY'
import json
import sys
from pathlib import Path

base_profile = Path(sys.argv[1]).resolve()
out_profile = Path(sys.argv[2]).resolve()
detector = sys.argv[3]
mode = sys.argv[4]
scope = sys.argv[5]
dump_dir = str(Path(sys.argv[6]).resolve())
dump_topk = int(sys.argv[7])
dump_min_score = float(sys.argv[8])
mtcr_checkpoint = sys.argv[9]
dump_save_tensors = int(__import__("os").environ.get("DUMP_SAVE_TENSORS", "0"))
dump_npz_every_n_groups = int(__import__("os").environ.get("DUMP_NPZ_EVERY_N_GROUPS", "2048"))

base_doc = json.loads(base_profile.read_text(encoding="utf-8"))
settings = dict(base_doc["settings"])
overrides = dict(settings.get("config_overrides", {}) or {})
overrides.update(
    {
        "EXP_NAME": f"mot17_external_{detector}_{mode}_{scope}_assocdump",
        "ASSOC_RUNTIME_DUMP_PATH": dump_dir,
        "ASSOC_RUNTIME_DUMP_TOPK": dump_topk,
        "ASSOC_RUNTIME_DUMP_MIN_SCORE": dump_min_score,
        "ASSOC_RUNTIME_DUMP_SAVE_TENSORS": dump_save_tensors,
        "ASSOC_RUNTIME_DUMP_NPZ_EVERY_N_GROUPS": dump_npz_every_n_groups,
    }
)
if mode == "mtcr":
    overrides["ASSOC_USE_LAPLACE"] = False
    overrides["ASSOC_USE_MTCR"] = True
    overrides["ASSOC_MTCR_CHECKPOINT"] = mtcr_checkpoint

settings["config_overrides"] = overrides
doc = {
    "description": (
        f"External-det association runtime dump: {detector}, mode={mode}, scope={scope}. "
        "Used for runtime candidate replay collection."
    ),
    "manifest": {
        "line": "external-det/runtime-replay",
        "protocol_tier": "private-det",
        "detector_source": detector,
        "association_variant": mode,
        "eval_scope": f"mot17_{scope}_frcnn",
        "assoc_runtime_dump_path": dump_dir,
        "mtcr_checkpoint": mtcr_checkpoint if mode == "mtcr" else "",
    },
    "settings": settings,
}
out_profile.write_text(json.dumps(doc, indent=2, sort_keys=False) + "\n", encoding="utf-8")
print(out_profile)
PY

echo "[run] detector=${DETECTOR} mode=${MODE} scope=${SCOPE}"
echo "[run] dump_dir=${DUMP_DIR}"
echo "[run] profile=${GENERATED_PROFILE}"
echo "[run] out_dir=${OUT_DIR}"

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/run_bytetrack_profile.py" \
  --exp-profile "${GENERATED_PROFILE}" \
  --out-dir "${OUT_DIR}"
