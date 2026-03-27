#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/gemini/code/FMtrack-main/FM-Track"
cd "${REPO_ROOT}"

export LD_LIBRARY_PATH="/root/miniconda3/lib:/root/miniconda3/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
DATA_ROOT="${DATA_ROOT:-/gemini/code/datasets}"
OUT_ROOT="${OUT_ROOT:-outputs/paper_ctrl_mot17_val0213}"
REGISTRY_CSV="${REGISTRY_CSV:-outputs/experiment_registry.csv}"
EPOCHS="${EPOCHS:-6}"
BATCH_SIZE="${BATCH_SIZE:-1}"
ACCUMULATE_STEPS="${ACCUMULATE_STEPS:-6}"
WATCHDOG_POLL_SEC="${WATCHDOG_POLL_SEC:-60}"
WATCHDOG_STALL_SEC="${WATCHDOG_STALL_SEC:-2700}"
WATCHDOG_EXIT_GRACE_SEC="${WATCHDOG_EXIT_GRACE_SEC:-180}"

DEFAULT_CONFIGS=(
  "configs/experiments/bytetrack_fa_mot_mot17_v14_ctrl_base_spatial_val0213.yaml"
  "configs/experiments/bytetrack_fa_mot_mot17_v14_ctrl_freq_spatial_val0213.yaml"
  "configs/experiments/bytetrack_fa_mot_mot17_v14_ctrl_base_reid_da_val0213.yaml"
  "configs/experiments/bytetrack_fa_mot_mot17_v14_ctrl_full_reid_da_val0213.yaml"
)

CONFIGS=()
if [[ -n "${CONFIGS_OVERRIDE:-}" ]]; then
  IFS=',' read -r -a CONFIGS <<< "${CONFIGS_OVERRIDE}"
else
  CONFIGS=("${DEFAULT_CONFIGS[@]}")
fi

mkdir -p "${OUT_ROOT}"
SUMMARY_CSV="${OUT_ROOT}/summary.csv"
if [[ ! -f "${SUMMARY_CSV}" ]]; then
  echo "exp_name,config_path,out_dir,best_epoch,best_hota,checkpoint,status" > "${SUMMARY_CSV}"
fi
QUEUE_LOG="${OUT_ROOT}/queue.log"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${QUEUE_LOG}"
}

write_single_row_csv() {
  local csv_path="$1"
  local exp="$2"
  local cfg="$3"
  local out_dir="$4"
  local best_epoch="$5"
  local best_hota="$6"
  local best_ckpt="$7"
  local status="$8"
  "${PYTHON_BIN}" - <<'PY' "${csv_path}" "${exp}" "${cfg}" "${out_dir}" "${best_epoch}" "${best_hota}" "${best_ckpt}" "${status}"
import csv
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
exp, cfg, out_dir, best_epoch, best_hota, best_ckpt, status = sys.argv[2:9]
fieldnames = ["exp_name", "config_path", "out_dir", "best_epoch", "best_hota", "checkpoint", "status"]
row = {
    "exp_name": exp,
    "config_path": cfg,
    "out_dir": out_dir,
    "best_epoch": best_epoch,
    "best_hota": best_hota,
    "checkpoint": best_ckpt,
    "status": status,
}
csv_path.parent.mkdir(parents=True, exist_ok=True)
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow(row)
PY
}

upsert_summary_row() {
  local exp="$1"
  local cfg="$2"
  local out_dir="$3"
  local best_epoch="$4"
  local best_hota="$5"
  local best_ckpt="$6"
  local status="$7"
  "${PYTHON_BIN}" - <<'PY' "${SUMMARY_CSV}" "${exp}" "${cfg}" "${out_dir}" "${best_epoch}" "${best_hota}" "${best_ckpt}" "${status}"
import csv
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
exp, cfg, out_dir, best_epoch, best_hota, best_ckpt, status = sys.argv[2:9]
fieldnames = ["exp_name", "config_path", "out_dir", "best_epoch", "best_hota", "checkpoint", "status"]
rows = []
if summary_path.is_file():
    with summary_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [dict(row) for row in reader if str(row.get("exp_name", "")).strip() != exp]
rows.append({
    "exp_name": exp,
    "config_path": cfg,
    "out_dir": out_dir,
    "best_epoch": best_epoch,
    "best_hota": best_hota,
    "checkpoint": best_ckpt,
    "status": status,
})
summary_path.parent.mkdir(parents=True, exist_ok=True)
with summary_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
PY
}

append_registry_row() {
  local result_csv="$1"
  local status="$2"
  local out_dir="$3"
  local log_path="$4"
  local exp="$5"
  local cfg="$6"
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/append_experiment_record.py" \
    --csv "${REGISTRY_CSV}" \
    --kind train \
    --status "${status}" \
    --script "scripts/queue_paper_ctrl_mot17_val0213.sh" \
    --dataset MOT17 \
    --split val0213_proxy \
    --tracker-family ByteTrack \
    --variant paper_ctrl_host_control \
    --tag "$(basename "${OUT_ROOT}")" \
    --run-root "${out_dir}" \
    --summary-csv "${result_csv}" \
    --log-path "${log_path}" \
    --extra exp_name="${exp}" config_path="${cfg}" out_dir="${out_dir}" >/dev/null 2>&1 || true
}

get_exp_name() {
  local cfg="$1"
  "${PYTHON_BIN}" - <<PY
import yaml
with open("${cfg}", "r", encoding="utf-8") as f:
    d = yaml.safe_load(f)
print(str(d.get("EXP_NAME", "")).strip())
PY
}

run_one() {
  local cfg="$1"
  local exp out_dir log_path result_csv best_txt best_epoch best_hota best_ckpt train_pid watchdog_pid watchdog_log train_rc
  exp="$(get_exp_name "${cfg}")"
  if [[ -z "${exp}" ]]; then
    echo "[queue] missing EXP_NAME in ${cfg}" >&2
    exit 2
  fi
  out_dir="${OUT_ROOT}/${exp}"
  log_path="${out_dir}/queue_train.log"
  result_csv="${out_dir}/result.csv"
  watchdog_log="${out_dir}/watchdog.log"
  mkdir -p "${out_dir}"

  if grep -q "^${exp},.*,[^,]*,[^,]*,[^,]*,[^,]*,ok$" "${SUMMARY_CSV}" 2>/dev/null; then
    log "skip completed experiment exp=${exp}"
    return 0
  fi

  log "===== ${exp} ====="
  log "cfg=${cfg}"
  log "out_dir=${out_dir}"
  upsert_summary_row "${exp}" "${cfg}" "${out_dir}" "" "" "" "running"
  write_single_row_csv "${result_csv}" "${exp}" "${cfg}" "${out_dir}" "" "" "" "running"

  : > "${watchdog_log}"
  "${PYTHON_BIN}" -u train_bytetrack.py \
    --config-path "${cfg}" \
    --data-root "${DATA_ROOT}" \
    --outputs-dir "${out_dir}" \
    --exp-name "${exp}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --accumulate-steps "${ACCUMULATE_STEPS}" \
    > >(tee "${log_path}") 2>&1 &
  train_pid=$!

  nohup bash "${REPO_ROOT}/scripts/watch_train_activity.sh" \
    --pid "${train_pid}" \
    --exp-name "${exp}" \
    --config-path "${cfg}" \
    --out-dir "${out_dir}" \
    --result-csv "${result_csv}" \
    --summary-csv "${SUMMARY_CSV}" \
    --registry-csv "${REGISTRY_CSV}" \
    --log-path "${log_path}" \
    --script "scripts/queue_paper_ctrl_mot17_val0213.sh" \
    --dataset "MOT17" \
    --split "val0213_proxy" \
    --tracker-family "ByteTrack" \
    --variant "paper_ctrl_host_control" \
    --tag "$(basename "${OUT_ROOT}")" \
    --poll-sec "${WATCHDOG_POLL_SEC}" \
    --stall-sec "${WATCHDOG_STALL_SEC}" \
    --exit-grace-sec "${WATCHDOG_EXIT_GRACE_SEC}" \
    --watchdog-log "${watchdog_log}" >/dev/null 2>&1 &
  watchdog_pid=$!
  echo "${watchdog_pid}" > "${out_dir}/watchdog.pid"

  set +e
  wait "${train_pid}"
  train_rc=$?
  set -e
  if kill -0 "${watchdog_pid}" 2>/dev/null; then
    kill "${watchdog_pid}" 2>/dev/null || true
    wait "${watchdog_pid}" 2>/dev/null || true
  fi
  rm -f "${out_dir}/watchdog.pid"

  if [[ "${train_rc}" -ne 0 ]]; then
    log "train failed exp=${exp}"
    upsert_summary_row "${exp}" "${cfg}" "${out_dir}" "" "" "" "failed"
    write_single_row_csv "${result_csv}" "${exp}" "${cfg}" "${out_dir}" "" "" "" "failed"
    append_registry_row "${result_csv}" "failed" "${out_dir}" "${log_path}" "${exp}" "${cfg}"
    return 1
  fi

  if ! best_txt="$("${PYTHON_BIN}" -u scripts/select_best_bytetrack_ckpt.py --exp-dir "${out_dir}" --metric HOTA --dataset MOT17 --split train)"; then
    log "best checkpoint selection failed exp=${exp}"
    upsert_summary_row "${exp}" "${cfg}" "${out_dir}" "" "" "" "select_failed"
    write_single_row_csv "${result_csv}" "${exp}" "${cfg}" "${out_dir}" "" "" "" "select_failed"
    append_registry_row "${result_csv}" "select_failed" "${out_dir}" "${log_path}" "${exp}" "${cfg}"
    return 1
  fi
  echo "${best_txt}" | tee "${out_dir}/best_ckpt.txt"

  best_epoch="$(echo "${best_txt}" | awk -F'best_epoch=' '/best_epoch=/{split($2,a,\" \"); print a[1]}')"
  best_hota="$(echo "${best_txt}" | awk -F'best_value=' '/best_value=/{split($2,a,\" \"); print a[1]}')"
  best_ckpt="$(echo "${best_txt}" | awk -F= '/^checkpoint=/{print $2}')"

  upsert_summary_row "${exp}" "${cfg}" "${out_dir}" "${best_epoch}" "${best_hota}" "${best_ckpt}" "ok"
  write_single_row_csv "${result_csv}" "${exp}" "${cfg}" "${out_dir}" "${best_epoch}" "${best_hota}" "${best_ckpt}" "ok"
  append_registry_row "${result_csv}" "success" "${out_dir}" "${log_path}" "${exp}" "${cfg}"
  log "finished exp=${exp} best_epoch=${best_epoch} best_hota=${best_hota}"
}

log "starting paper-control 4-way proxy training"
log "epochs=${EPOCHS} batch_size=${BATCH_SIZE} accumulate_steps=${ACCUMULATE_STEPS}"

for cfg in "${CONFIGS[@]}"; do
  if ! run_one "${cfg}"; then
    log "continue after failure cfg=${cfg}"
  fi
done

log "all four control runs finished"
