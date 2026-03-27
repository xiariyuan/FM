#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/gemini/code/FMtrack-main/FM-Track"
LOG_DIR="${REPO_ROOT}/outputs/dancetrack"
LOG_FILE="${LOG_DIR}/overnight_queue.log"
SUMMARY_FILE="${LOG_DIR}/overnight_summary.tsv"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
WAIT_SESSION="${WAIT_SESSION:-dancetrack_full_ltra}"

mkdir -p "${LOG_DIR}"

variants=(
  meanrel
  mean
  multinorel
  single
  singlerel
  rel075
  rel050
  nodetscore
  weight025
  weight050
)

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${LOG_FILE}"
}

run_eval_bg() {
  local variant="$1"
  local results_dir="${REPO_ROOT}/external/BoT-SORT-main/YOLOX_outputs/botsort_dancetrack_val_${variant}/track_results"
  local work_dir="${REPO_ROOT}/outputs/dancetrack/${variant}_eval"
  if [[ ! -d "${results_dir}" ]]; then
    log "skip eval ${variant}: results dir missing"
    return 0
  fi
  (
    cd "${REPO_ROOT}"
    "${PYTHON_BIN}" scripts/eval_motstyle_trackeval.py \
      --benchmark-name DanceTrack \
      --split-to-eval val \
      --gt-root /tmp/DanceTrack_val/val \
      --results-dir "${results_dir}" \
      --tracker-name "botsort_dancetrack_val_${variant}" \
      --work-dir "${work_dir}"
  ) >> "${LOG_FILE}" 2>&1 &
  echo $!
}

write_summary() {
  {
    echo -e "variant\tHOTA\tDetA\tAssA\tMOTA\tIDF1\tIDSW"
    for variant in base ltra "${variants[@]}"; do
      local summary="${REPO_ROOT}/outputs/dancetrack/${variant}_eval/eval/botsort_dancetrack_val_${variant}/pedestrian_summary.txt"
      if [[ -f "${summary}" ]]; then
        python - "$variant" "$summary" <<'PY'
import sys
variant = sys.argv[1]
path = sys.argv[2]
with open(path, "r") as f:
    header = f.readline().strip().split()
    values = f.readline().strip().split()
data = dict(zip(header, values))
print("\t".join([
    variant,
    data.get("HOTA", ""),
    data.get("DetA", ""),
    data.get("AssA", ""),
    data.get("MOTA", ""),
    data.get("IDF1", ""),
    data.get("IDSW", ""),
]))
PY
      fi
    done
  } > "${SUMMARY_FILE}"
}

log "overnight queue start"
log "waiting for current session: ${WAIT_SESSION}"
while tmux has-session -t "${WAIT_SESSION}" 2>/dev/null; do
  sleep 20
done
log "detected ${WAIT_SESSION} finished"

eval_pids=()

for variant in "${variants[@]}"; do
  summary="${REPO_ROOT}/outputs/dancetrack/${variant}_eval/eval/botsort_dancetrack_val_${variant}/pedestrian_summary.txt"
  if [[ -f "${summary}" ]]; then
    log "skip ${variant}: summary already exists"
    continue
  fi

  log "start variant ${variant}"
  (
    cd "${REPO_ROOT}"
    env PYTHONUNBUFFERED=1 bash scripts/run_botsort_dancetrack_val.sh "${variant}" --fp16
  ) >> "${LOG_FILE}" 2>&1
  log "finished tracking ${variant}; launch eval in background"
  pid="$(run_eval_bg "${variant}")"
  eval_pids+=("${pid}")
done

log "waiting for ${#eval_pids[@]} background eval jobs"
for pid in "${eval_pids[@]}"; do
  wait "${pid}" || true
done

write_summary
log "overnight queue finished"
log "summary saved to ${SUMMARY_FILE}"
