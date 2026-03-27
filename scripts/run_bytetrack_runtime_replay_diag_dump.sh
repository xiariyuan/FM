#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"

OUT_DIR="${1:-${REPO_ROOT}/outputs/runtime_replay_diag_dump_$(date +%Y%m%d_%H%M%S)}"
CHECKPOINT="${2:-${REPO_ROOT}/outputs/runtime_replay_learned_sw_yolox_base_full7_hardtrain_mixedval_gate/runtime_replay_hardtrain_mixedval_gate.pt}"
SEQS="${3:-MOT17-02,MOT17-09,MOT17-11,MOT17-13}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"
PROFILE_PATH="${OUT_DIR}/diag_profile.json"
DUMP_ROOT="${OUT_DIR}/runtime_dump"

"${PYTHON_BIN}" - "${PROFILE_PATH}" "${CHECKPOINT}" "${SEQS}" "${DUMP_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

profile_path = Path(sys.argv[1]).resolve()
checkpoint = str(Path(sys.argv[2]).resolve())
seqs = [s.strip() for s in sys.argv[3].split(",") if s.strip()]
dump_root = str(Path(sys.argv[4]).resolve())

doc = {
    "description": (
        "Runtime replay diagnostic dump on external/sw_yolox for targeted MOT17 FRCNN sequences. "
        "Used to compare positive sequences and regressing sequences at candidate-group level."
    ),
    "manifest": {
        "line": "external-det/system-rescue",
        "protocol_tier": "private-det",
        "detector_source": "sw_yolox",
        "association_variant": "runtime_replay_diag_dump",
        "eval_scope": "mot17_diag_targeted_frcnn",
        "runtime_replay_checkpoint": checkpoint,
        "val_sequences": seqs,
        "dump_root": dump_root,
    },
    "settings": {
        "config_path": "configs/bytetrack_fa_mot_mot17_proxy0213_train_reidain_feature_stageBbest_detft_nooverlap.yaml",
        "inference_dataset": "MOT17",
        "inference_split": "train",
        "config_overrides": {
            "EXP_NAME": "mot17_external_sw_yolox_runtime_replay_diag_dump",
            "BYTETRACK_DET_SOURCE": "external",
            "EXTERNAL_DET_ROOT": "outputs/external_det/sw_yolox/MOT17",
            "EXTERNAL_DET_PATTERN": "{root}/{split}/{seq}.txt",
            "DETECTOR_FILTER": ["FRCNN"],
            "RUN_TRACKEVAL": True,
            "EVAL_ONLY_VAL": True,
            "VAL_SEQUENCES": seqs,
            "USE_MEMORY_BANK": False,
            "ASSOC_USE_LAPLACE": False,
            "ASSOC_USE_MTCR": False,
            "ASSOC_USE_RUNTIME_REPLAY": True,
            "ASSOC_RUNTIME_REPLAY_CHECKPOINT": checkpoint,
            "ASSOC_RUNTIME_DUMP_PATH": dump_root,
            "ASSOC_RUNTIME_DUMP_TOPK": 0,
            "ASSOC_RUNTIME_DUMP_MIN_SCORE": 0.0,
            "ASSOC_RUNTIME_DUMP_SAVE_TENSORS": False,
            "ASSOC_RUNTIME_DUMP_NPZ_EVERY_N_GROUPS": 4096,
        },
    },
}
profile_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
print(profile_path)
PY

echo "[run] profile=${PROFILE_PATH}"
echo "[run] checkpoint=${CHECKPOINT}"
echo "[run] sequences=${SEQS}"
echo "[run] dump_root=${DUMP_ROOT}"

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/run_bytetrack_profile.py" \
  --exp-profile "${PROFILE_PATH}" \
  --out-dir "${OUT_DIR}"

