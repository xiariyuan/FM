#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
POLL_SECS="${POLL_SECS:-120}"
REGISTRY_CSV="${REGISTRY_CSV:-${REPO_ROOT}/outputs/experiment_registry.csv}"

HONEST_ROOT="${HONEST_ROOT:-${REPO_ROOT}/outputs/runtime_replay_honest_topk_longhaul_20260321_honest_topk_v1}"
HONEST_STATUS="${HONEST_STATUS:-${HONEST_ROOT}/job_status.txt}"
HONEST_BEST_TXT="${HONEST_BEST_TXT:-${HONEST_ROOT}/proxy_epoch_sweep/best_epoch.txt}"
HONEST_SUMMARY="${HONEST_SUMMARY:-${HONEST_ROOT}/proxy_epoch_sweep/best_epoch_full7/tracker/MOT17-train/pedestrian_summary.txt}"

CTRL_OUT_ROOT="${CTRL_OUT_ROOT:-${REPO_ROOT}/outputs/paper_ctrl_mot17_val0213}"
CTRL_EPOCHS="${CTRL_EPOCHS:-6}"
CTRL_BATCH_SIZE="${CTRL_BATCH_SIZE:-1}"
CTRL_ACCUMULATE_STEPS="${CTRL_ACCUMULATE_STEPS:-6}"
CTRL_SUMMARY="${CTRL_SUMMARY:-${CTRL_OUT_ROOT}/summary.csv}"

ABLATION_OUT_ROOT="${ABLATION_OUT_ROOT:-${REPO_ROOT}/outputs/_paper_ablation_val_MOT17}"
QUEUE_ROOT="${QUEUE_ROOT:-${REPO_ROOT}/outputs/overnight_mainline_queue_$(date +%Y%m%d_%H%M%S)}"
QUEUE_LOG="${QUEUE_ROOT}/queue.log"

mkdir -p "${QUEUE_ROOT}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${QUEUE_LOG}"
}

wait_for_honest_topk() {
  log "waiting for honest-topk job status path=${HONEST_STATUS}"
  while true; do
    local status="missing"
    if [[ -f "${HONEST_STATUS}" ]]; then
      status="$(tr -d '\r\n' < "${HONEST_STATUS}")"
    fi
    log "honest-topk status=${status}"
    if [[ "${status}" == "running" || "${status}" == "initialized" || "${status}" == "missing" ]]; then
      sleep "${POLL_SECS}"
      continue
    fi
    return 0
  done
}

summarize_honest_topk() {
  log "honest-topk finished; collecting summary"
  if [[ -f "${HONEST_BEST_TXT}" ]]; then
    log "best-epoch:"
    sed 's/^/[best] /' "${HONEST_BEST_TXT}" | tee -a "${QUEUE_LOG}"
  else
    log "best-epoch file missing: ${HONEST_BEST_TXT}"
  fi
  if [[ -f "${HONEST_SUMMARY}" ]]; then
    log "best full7 summary:"
    sed 's/^/[full7] /' "${HONEST_SUMMARY}" | tee -a "${QUEUE_LOG}"
  else
    log "best full7 summary missing: ${HONEST_SUMMARY}"
  fi
}

run_paper_ctrl_queue() {
  log "launch paper control queue"
  (
    cd "${REPO_ROOT}"
    EPOCHS="${CTRL_EPOCHS}" \
    BATCH_SIZE="${CTRL_BATCH_SIZE}" \
    ACCUMULATE_STEPS="${CTRL_ACCUMULATE_STEPS}" \
    OUT_ROOT="${CTRL_OUT_ROOT}" \
    bash scripts/queue_paper_ctrl_mot17_val0213.sh
  ) 2>&1 | tee -a "${QUEUE_LOG}"
}

sync_ctrl_summary_registry() {
  if [[ ! -f "${CTRL_SUMMARY}" ]]; then
    log "control summary missing; skip registry sync path=${CTRL_SUMMARY}"
    return 0
  fi
  log "sync control summary to registry"
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/append_experiment_record.py" \
    --csv "${REGISTRY_CSV}" \
    --summary-csv "${CTRL_SUMMARY}" \
    --kind train \
    --status success \
    --script "scripts/queue_paper_ctrl_mot17_val0213.sh" \
    --dataset MOT17 \
    --split val0213_proxy \
    --tracker-family ByteTrack \
    --variant paper_ctrl_host_control \
    --tag "$(basename "${CTRL_OUT_ROOT}")" \
    --run-root "${CTRL_OUT_ROOT}" \
    --log-path "${CTRL_OUT_ROOT}/queue.log" >/dev/null 2>&1 || true
}

find_best_ctrl_ckpt() {
  "${PYTHON_BIN}" - <<'PY' "${CTRL_SUMMARY}"
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(1)
best = None
with path.open("r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if str(row.get("status", "")).strip() != "ok":
            continue
        try:
            hota = float(row.get("best_hota", ""))
        except Exception:
            continue
        ckpt = str(row.get("checkpoint", "")).strip()
        if not ckpt:
            continue
        if best is None or hota > best[0]:
            best = (hota, ckpt, str(row.get("exp_name", "")).strip())
if best is None:
    raise SystemExit(2)
print(best[1])
print(best[2])
print(best[0])
PY
}

run_best_ctrl_ablation() {
  log "select best control checkpoint for ablation"
  local best_info
  if ! best_info="$(find_best_ctrl_ckpt)"; then
    log "no successful control checkpoint found; skip ablation"
    return 0
  fi
  local best_ckpt best_exp best_hota
  best_ckpt="$(echo "${best_info}" | sed -n '1p')"
  best_exp="$(echo "${best_info}" | sed -n '2p')"
  best_hota="$(echo "${best_info}" | sed -n '3p')"
  log "best control checkpoint exp=${best_exp} best_hota=${best_hota} ckpt=${best_ckpt}"
  (
    cd "${REPO_ROOT}"
    OUT_ROOT="${ABLATION_OUT_ROOT}/${best_exp}" \
    bash scripts/run_paper_ablations_val.sh "${best_ckpt}" MOT17 MOT17-02,MOT17-13
  ) 2>&1 | tee -a "${QUEUE_LOG}"
}

main() {
  log "=== overnight mainline queue started ==="
  wait_for_honest_topk
  summarize_honest_topk
  run_paper_ctrl_queue
  sync_ctrl_summary_registry
  run_best_ctrl_ablation
  log "=== overnight mainline queue finished ==="
}

main "$@"
