#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"

DETECTOR="${1:?detector is required: sw_yolox|sgt}"
TRAIN_DIR="${2:?train dir is required}"
OUT_ROOT="${3:-${TRAIN_DIR}/proxy_epoch_sweep}"
RUN_FULL7_BEST="${4:-0}"

if [[ "${DETECTOR}" != "sw_yolox" && "${DETECTOR}" != "sgt" ]]; then
  echo "Unsupported detector=${DETECTOR}" >&2
  exit 2
fi

TRAIN_DIR="$(cd "${TRAIN_DIR}" && pwd)"
OUT_ROOT="$(mkdir -p "${OUT_ROOT}" && cd "${OUT_ROOT}" && pwd)"
EPOCH_DIR="${TRAIN_DIR}/epoch_ckpts"

if [[ ! -d "${EPOCH_DIR}" ]]; then
  echo "Epoch checkpoint directory not found: ${EPOCH_DIR}" >&2
  exit 2
fi

mapfile -t CKPTS < <(find "${EPOCH_DIR}" -maxdepth 1 -type f -name 'epoch_*.pt' | sort)
if [[ "${#CKPTS[@]}" -eq 0 ]]; then
  echo "No epoch checkpoints found under ${EPOCH_DIR}" >&2
  exit 2
fi

SUMMARY_CSV="${OUT_ROOT}/epoch_proxy0213_scores.csv"
BEST_TXT="${OUT_ROOT}/best_epoch.txt"
RUN_LOG="${OUT_ROOT}/sweep.log"

echo "epoch,checkpoint,hota,assa,idf1,idsw,summary_path,run_dir" > "${SUMMARY_CSV}"
exec > >(tee -a "${RUN_LOG}") 2>&1

echo "[proxy-sweep] detector=${DETECTOR}"
echo "[proxy-sweep] train_dir=${TRAIN_DIR}"
echo "[proxy-sweep] out_root=${OUT_ROOT}"

best_epoch=""
best_hota=""
best_run_dir=""
best_ckpt=""

for ckpt in "${CKPTS[@]}"; do
  base="$(basename "${ckpt}")"
  epoch="${base#epoch_}"
  epoch="${epoch%.pt}"
  run_dir="${OUT_ROOT}/epoch_${epoch}"
  echo "[proxy-sweep] epoch=${epoch} ckpt=${ckpt}"
  bash "${REPO_ROOT}/scripts/run_bytetrack_runtime_replay_external.sh" \
    "${DETECTOR}" proxy0213 "${ckpt}" "${run_dir}"

  summary_path="${run_dir}/tracker/MOT17-train/pedestrian_summary.txt"
  if [[ ! -f "${summary_path}" ]]; then
    echo "Missing summary for epoch ${epoch}: ${summary_path}" >&2
    exit 3
  fi

  metrics="$("${PYTHON_BIN}" - "${summary_path}" <<'PY'
import sys
from pathlib import Path
txt=Path(sys.argv[1]).read_text().strip().splitlines()
hdr=txt[0].split()
vals=txt[1].split()
d={k:v for k,v in zip(hdr, vals)}
print(",".join([d["HOTA"], d["AssA"], d["IDF1"], d["IDSW"]]))
PY
)"
  IFS=',' read -r hota assa idf1 idsw <<< "${metrics}"
  echo "${epoch},${ckpt},${hota},${assa},${idf1},${idsw},${summary_path},${run_dir}" >> "${SUMMARY_CSV}"

  if [[ -z "${best_hota}" ]] || awk "BEGIN {exit !(${hota} > ${best_hota})}"; then
    best_epoch="${epoch}"
    best_hota="${hota}"
    best_run_dir="${run_dir}"
    best_ckpt="${ckpt}"
  fi
done

{
  echo "best_epoch=${best_epoch}"
  echo "best_hota=${best_hota}"
  echo "best_checkpoint=${best_ckpt}"
  echo "best_proxy_run_dir=${best_run_dir}"
} > "${BEST_TXT}"

echo "[proxy-sweep] best_epoch=${best_epoch} best_hota=${best_hota}"

if [[ "${RUN_FULL7_BEST}" == "1" ]]; then
  full7_dir="${OUT_ROOT}/best_epoch_full7"
  echo "[proxy-sweep] running full7 for best epoch=${best_epoch}"
  bash "${REPO_ROOT}/scripts/run_bytetrack_runtime_replay_external.sh" \
    "${DETECTOR}" full7 "${best_ckpt}" "${full7_dir}"
fi
