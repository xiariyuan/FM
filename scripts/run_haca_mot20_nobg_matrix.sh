#!/usr/bin/env bash
set -euo pipefail

# MOT20 HACA no-background matrix evaluation.
# Evaluates HACA v1 / v2 / v3 checkpoints as plain no-background baselines.

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
DATA_ROOT="${DATA_ROOT:-/gemini/code/datasets}"
BOT_ROOT="${REPO_ROOT}/external/BoT-SORT-main"
REGISTRY_CSV="${REGISTRY_CSV:-${REPO_ROOT}/outputs/experiment_registry.csv}"
PLAN_CSV="${PLAN_CSV:-${REPO_ROOT}/outputs/experiment_plan.csv}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
TRAIN_ROOT="${TRAIN_ROOT:-${REPO_ROOT}/outputs/haca_mot20_train_20260613_082005}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs/haca_mot20_nobg_matrix_${TS}}"
OUT_ROOT="$(realpath -m "${OUT_ROOT}")"
PLAN_KEY="${PLAN_KEY:-run_root:${OUT_ROOT}}"
LOG_PATH="${OUT_ROOT}/matrix.log"
SUMMARY_CSV="${OUT_ROOT}/summary.csv"

mkdir -p "${OUT_ROOT}"
exec > >(tee -a "${LOG_PATH}") 2>&1

V1_CKPT="${V1_CKPT:-${TRAIN_ROOT}/haca_v1/mot20_haca_v1.npz}"
V2_CKPT="${V2_CKPT:-${TRAIN_ROOT}/haca_v2/mot20_haca_v2.npz}"
V3_CKPT="${V3_CKPT:-${TRAIN_ROOT}/haca_v3/mot20_haca_v3.npz}"

V1_STATUS="pending"
V2_STATUS="pending"
V3_STATUS="pending"
CURRENT_VARIANT="init"

write_summary() {
  OUT_ROOT="${OUT_ROOT}" \
  TS="${TS}" \
  V1_STATUS="${V1_STATUS}" \
  V2_STATUS="${V2_STATUS}" \
  V3_STATUS="${V3_STATUS}" \
  python - <<'PY'
import csv
import os
from pathlib import Path

out_root = Path(os.environ["OUT_ROOT"])
ts = os.environ["TS"]
status_map = {
    "haca_v1": os.environ["V1_STATUS"],
    "haca_v2": os.environ["V2_STATUS"],
    "haca_v3": os.environ["V3_STATUS"],
}
rows = []
for mode, status in status_map.items():
    row = {
        "variant": mode,
        "status": status,
        "checkpoint": str(out_root / mode / "checkpoint.npz"),
        "HOTA": "",
        "AssA": "",
        "IDF1": "",
        "MOTA": "",
        "IDSW": "",
    }
    metrics_txt = out_root / mode / "trackeval" / "eval" / f"haca_mot20_nobg_{mode}_{ts}" / "pedestrian_summary.txt"
    if metrics_txt.exists():
      with metrics_txt.open() as f:
        header = f.readline().strip().split()
        values = f.readline().strip().split()
      metrics = dict(zip(header, values))
      for key in ["HOTA", "AssA", "IDF1", "MOTA", "IDSW"]:
        row[key] = metrics.get(key, "")
    rows.append(row)

summary_csv = out_root / "summary.csv"
with summary_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["variant", "status", "checkpoint", "HOTA", "AssA", "IDF1", "MOTA", "IDSW"])
    writer.writeheader()
    writer.writerows(rows)
PY
}

update_plan_status() {
  local status="$1"
  shift || true
  local extras=(
    "train_root=${TRAIN_ROOT}"
    "no_background=true"
    "seq_ids=1,2,3,5"
  )
  if [[ $# -gt 0 ]]; then
    extras+=("$@")
  fi
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/upsert_experiment_plan.py" \
    --csv "${PLAN_CSV}" \
    --key "${PLAN_KEY}" \
    --status "${status}" \
    --kind eval \
    --script "scripts/run_haca_mot20_nobg_matrix.sh" \
    --dataset MOT20 \
    --split val_half \
    --tracker-family BoT-SORT \
    --variant "haca_mot20_nobg_matrix" \
    --run-root "${OUT_ROOT}" \
    --summary-csv "${SUMMARY_CSV}" \
    --checkpoint "${V3_CKPT}" \
    --log-path "${LOG_PATH}" \
    --extra "${extras[@]}"
}

on_exit() {
  local rc=$?
  trap - EXIT
  if [[ ${rc} -ne 0 ]]; then
    case "${CURRENT_VARIANT}" in
      haca_v1)
        V1_STATUS="failed"
        ;;
      haca_v2)
        V2_STATUS="failed"
        ;;
      haca_v3)
        V3_STATUS="failed"
        ;;
    esac
    write_summary || true
    update_plan_status failed "exit_code=${rc}" || true
  fi
  exit ${rc}
}

trap on_exit EXIT
write_summary
update_plan_status running

for ckpt in "${V1_CKPT}" "${V2_CKPT}" "${V3_CKPT}"; do
  if [[ ! -f "${ckpt}" ]]; then
    echo "[FATAL] missing checkpoint: ${ckpt}"
    exit 1
  fi
done

SEQ_IDS=(1 2 3 5)
COMMON_ARGS=(
  --laplace-assoc
  --laplace-decay-scales 1 2 4
  --laplace-min-history 3
  --laplace-proto-mode multi
  --laplace-primary-only
  --laplace-haca-no-background
)

run_variant() {
  local mode="$1"
  local ckpt="$2"
  local out_dir="${OUT_ROOT}/${mode}"
  local exp="haca_mot20_nobg_${mode}_${TS}"
  local results_dir
  mkdir -p "${out_dir}"
  ln -sfn "${ckpt}" "${out_dir}/checkpoint.npz"

  echo "[run] ${mode} ckpt=${ckpt}"
  (
    cd "${BOT_ROOT}"
    "${PYTHON_BIN}" -u tools/track.py "${DATA_ROOT}/MOT20" \
      --benchmark MOT20 \
      --eval train \
      --seq-ids "${SEQ_IDS[@]}" \
      -f ./yolox/exps/example/mot/yolox_x_mix_mot20_ch.py \
      -c ./pretrained/bytetrack_x_mot20.pth.tar \
      --with-reid \
      --fast-reid-config fast_reid/configs/MOT20/sbs_S50.yml \
      --fast-reid-weights pretrained/mot20_sbs_S50.pth \
      --experiment-name "${exp}" \
      --laplace-assoc-mode "${mode}" \
      --laplace-haca-checkpoint "${ckpt}" \
      "${COMMON_ARGS[@]}" \
      > "${out_dir}/track.log" 2>&1
  )

  results_dir="${BOT_ROOT}/YOLOX_outputs/${exp}/track_results"
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/eval_botsort_halfval_trackeval.py" \
    --dataset MOT20 \
    --data-root "${DATA_ROOT}" \
    --results-dir "${results_dir}" \
    --tracker-name "${exp}" \
    --work-dir "${out_dir}/trackeval" \
    --remap-results-from-fullval \
    > "${out_dir}/eval_trackeval.log" 2>&1

  local metrics_txt="${out_dir}/trackeval/eval/${exp}/pedestrian_summary.txt"
  if [[ ! -f "${metrics_txt}" ]]; then
    echo "[FATAL] Missing TrackEval summary for ${mode}: ${metrics_txt}"
    exit 1
  fi
  echo "[done] ${mode}"
}

CURRENT_VARIANT="haca_v1"
V1_STATUS="running"
write_summary
run_variant "haca_v1" "${V1_CKPT}"
V1_STATUS="success"
write_summary

CURRENT_VARIANT="haca_v2"
V2_STATUS="running"
write_summary
run_variant "haca_v2" "${V2_CKPT}"
V2_STATUS="success"
write_summary

CURRENT_VARIANT="haca_v3"
V3_STATUS="running"
write_summary
run_variant "haca_v3" "${V3_CKPT}"
V3_STATUS="success"
write_summary

CURRENT_VARIANT="done"

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/append_experiment_record.py" \
  --csv "${REGISTRY_CSV}" \
  --kind eval \
  --script "scripts/run_haca_mot20_nobg_matrix.sh" \
  --dataset MOT20 \
  --split val_half \
  --tracker-family BoT-SORT \
  --variant "haca_mot20_nobg_matrix" \
  --run-root "${OUT_ROOT}" \
  --summary-csv "${SUMMARY_CSV}" \
  --checkpoint "${V3_CKPT}" \
  --log-path "${LOG_PATH}" \
  --extra \
    "train_root=${TRAIN_ROOT}" \
    "v1_checkpoint=${V1_CKPT}" \
    "v2_checkpoint=${V2_CKPT}" \
    "v3_checkpoint=${V3_CKPT}" \
    "no_background=true" \
    "seq_ids=1,2,3,5"

update_plan_status completed
echo "[done] matrix summary: ${SUMMARY_CSV}"
