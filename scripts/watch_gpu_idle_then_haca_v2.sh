#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
POLL_SECS="${POLL_SECS:-120}"
GPU_IDLE_MEM_MB="${GPU_IDLE_MEM_MB:-200}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
LOG_PATH="${LOG_PATH:-${REPO_ROOT}/outputs/watch_gpu_idle_then_haca_v2_${TS}.log}"

mkdir -p "$(dirname "${LOG_PATH}")"
exec > >(tee -a "${LOG_PATH}") 2>&1

is_gpu_idle() {
  local query
  query="$(nvidia-smi --query-compute-apps=used_gpu_memory --format=csv,noheader,nounits 2>/dev/null || true)"
  if [[ -z "${query}" ]]; then
    return 0
  fi
  while IFS= read -r line; do
    [[ -z "${line}" ]] && continue
    local mem="${line%% *}"
    if [[ "${mem}" =~ ^[0-9]+$ ]] && (( mem > GPU_IDLE_MEM_MB )); then
      return 1
    fi
  done <<< "${query}"
  return 0
}

echo "[watch] start $(date '+%F %T %z')"
echo "[watch] poll_secs=${POLL_SECS} gpu_idle_mem_mb=${GPU_IDLE_MEM_MB}"

until is_gpu_idle; do
  echo "[watch] gpu busy, sleep ${POLL_SECS}s"
  sleep "${POLL_SECS}"
done

echo "[watch] gpu idle, launch HACA-v2 train"
TRAIN_ROOT="${REPO_ROOT}/outputs/haca_v2_train_${TS}"
EVAL_ROOT="${REPO_ROOT}/outputs/haca_v2_eval_${TS}"
MOT20_ROOT="${REPO_ROOT}/outputs/haca_v2_mot20_${TS}"

OUT_ROOT="${TRAIN_ROOT}" \
"${REPO_ROOT}/scripts/train_haca_v2_mot17.sh"

HACA_NPZ="$(find "${TRAIN_ROOT}" -maxdepth 1 -type f -name '*.npz' | head -n 1)"
if [[ -z "${HACA_NPZ}" ]]; then
  echo "[watch] missing HACA-v2 checkpoint under ${TRAIN_ROOT}" >&2
  exit 1
fi

echo "[watch] eval HACA-v2 same-base"
RUN_ROOT="${EVAL_ROOT}" \
RUN_BASE=0 \
RUN_HEURISTIC=0 \
RUN_CURRENT_LEARNED=0 \
RUN_HACA=1 \
HACA_NPZ="${HACA_NPZ}" \
"${REPO_ROOT}/scripts/run_botsort_haca_v2_eval.sh"

SUMMARY_CSV="${EVAL_ROOT}/eval/mot17_summary.csv"
DECISION="$("${PYTHON_BIN}" - <<'PY' "${SUMMARY_CSV}"
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit("missing_summary")
with path.open("r", newline="") as f:
    rows = list(csv.DictReader(f))
if not rows:
    raise SystemExit("empty_summary")
row = rows[0]
hota = float(row["HOTA"])
assa = float(row["AssA"])
should_continue = hota >= 78.65 and assa >= 77.50
print(f"{int(should_continue)}\t{hota:.3f}\t{assa:.3f}")
PY
)"

SHOULD_CONTINUE="$(printf '%s' "${DECISION}" | cut -f1)"
HOTA_VALUE="$(printf '%s' "${DECISION}" | cut -f2)"
ASSA_VALUE="$(printf '%s' "${DECISION}" | cut -f3)"
echo "[watch] same-base metrics HOTA=${HOTA_VALUE} AssA=${ASSA_VALUE}"

if [[ "${SHOULD_CONTINUE}" != "1" ]]; then
  echo "[watch] HACA-v2 same-base did not clear threshold, stop before MOT20"
  exit 0
fi

echo "[watch] launch HACA-v2 MOT20 zero-shot"
RUN_ROOT="${MOT20_ROOT}" \
RUN_BASE=0 \
RUN_HEURISTIC=0 \
RUN_CURRENT_LEARNED=0 \
RUN_HACA=1 \
HACA_NPZ="${HACA_NPZ}" \
"${REPO_ROOT}/scripts/run_botsort_haca_v2_mot20_eval.sh"

echo "[watch] done $(date '+%F %T %z')"
