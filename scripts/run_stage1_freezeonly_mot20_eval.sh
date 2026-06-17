#!/usr/bin/env bash
set -euo pipefail

# MOT20 SCA+LMF mainline evaluation.
# Runs baseline / Stage1 / Stage1+freeze and evaluates with TrackEval.

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
DATA_ROOT="${DATA_ROOT:-/gemini/code/datasets}"
BOT_ROOT="${REPO_ROOT}/external/BoT-SORT-main"
REGISTRY_CSV="${REGISTRY_CSV:-${REPO_ROOT}/outputs/experiment_registry.csv}"
PLAN_CSV="${PLAN_CSV:-${REPO_ROOT}/outputs/experiment_plan.csv}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
OUT_BASE="${OUT_BASE:-${REPO_ROOT}/outputs/sca_lmf_mot20_eval_${TS}}"
OUT_BASE="$(realpath -m "${OUT_BASE}")"
PLAN_KEY="${PLAN_KEY:-run_root:${OUT_BASE}}"
LOG_PATH="${OUT_BASE}/eval.log"
SUMMARY_CSV="${OUT_BASE}/summary.csv"
HACA_CKPT="${HACA_CKPT:-}"
S1_CKPT="${S1_CKPT:-${STAGE1_CKPT:-}}"

mkdir -p "${OUT_BASE}"
exec > >(tee -a "${LOG_PATH}") 2>&1

BASELINE_STATUS="pending"
STAGE1_ONLY_STATUS="pending"
STAGE1_FREEZE_STATUS="pending"
CURRENT_VARIANT="init"

write_eval_summary() {
  OUT_BASE="${OUT_BASE}" \
  TS="${TS}" \
  BASELINE_STATUS="${BASELINE_STATUS}" \
  STAGE1_ONLY_STATUS="${STAGE1_ONLY_STATUS}" \
  STAGE1_FREEZE_STATUS="${STAGE1_FREEZE_STATUS}" \
  python - <<'PY'
import csv
import os
from pathlib import Path

out_base = Path(os.environ["OUT_BASE"])
ts = os.environ["TS"]
status_by_variant = {
    "baseline": os.environ["BASELINE_STATUS"],
    "stage1_only": os.environ["STAGE1_ONLY_STATUS"],
    "stage1_freeze": os.environ["STAGE1_FREEZE_STATUS"],
}
rows = []
for variant, status in status_by_variant.items():
    row = {
        "variant": variant,
        "status": status,
        "HOTA": "",
        "AssA": "",
        "IDF1": "",
        "MOTA": "",
        "IDSW": "",
    }
    metrics_txt = out_base / variant / "trackeval" / "eval" / f"sca_lmf_mot20_{variant}_{ts}" / "pedestrian_summary.txt"
    if metrics_txt.exists():
        with metrics_txt.open() as f:
            header = f.readline().strip().split()
            values = f.readline().strip().split()
        metrics = dict(zip(header, values))
        for key in ["HOTA", "AssA", "IDF1", "MOTA", "IDSW"]:
            row[key] = metrics.get(key, "")
    rows.append(row)

summary_csv = out_base / "summary.csv"
with summary_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["variant", "status", "HOTA", "AssA", "IDF1", "MOTA", "IDSW"])
    writer.writeheader()
    writer.writerows(rows)
PY
}

write_eval_summary

if [[ -z "${HACA_CKPT}" ]]; then
  HACA_CKPT=$(find "${REPO_ROOT}/outputs" -path '*/haca_v3/mot20_haca_v3.npz' | sort | tail -1)
fi
if [[ -z "${S1_CKPT}" ]]; then
  S1_CKPT=$(find "${REPO_ROOT}/outputs" -path '*/stage1/stage1_best.pt' | grep '/rgsa_stage1_mot20_pipeline_' | sort | tail -1 || true)
fi

if [[ -z "${HACA_CKPT}" ]]; then
  BASELINE_STATUS="failed"
  STAGE1_ONLY_STATUS="skipped"
  STAGE1_FREEZE_STATUS="skipped"
  write_eval_summary
  echo "[FATAL] No MOT20 HACA v3 checkpoint found."
  echo "Run: bash scripts/run_haca_v3_mot20_train.sh"
  exit 1
fi

update_plan_status() {
  local status="$1"
  shift || true
  local extras=(
    "haca_checkpoint=${HACA_CKPT}"
    "stage1_checkpoint=${S1_CKPT:-}"
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
    --script "scripts/run_stage1_freezeonly_mot20_eval.sh" \
    --dataset MOT20 \
    --split val_half \
    --tracker-family BoT-SORT \
    --variant "sca_lmf_mot20_eval" \
    --run-root "${OUT_BASE}" \
    --summary-csv "${OUT_BASE}/summary.csv" \
    --checkpoint "${HACA_CKPT}" \
    --log-path "${LOG_PATH}" \
    --extra "${extras[@]}"
}

on_exit() {
  local rc=$?
  trap - EXIT
  if [[ ${rc} -ne 0 ]]; then
    case "${CURRENT_VARIANT}" in
      baseline)
        BASELINE_STATUS="failed"
        ;;
      stage1_only)
        STAGE1_ONLY_STATUS="failed"
        ;;
      stage1_freeze)
        STAGE1_FREEZE_STATUS="failed"
        ;;
    esac
    write_eval_summary || true
    update_plan_status failed "exit_code=${rc}" || true
  fi
  exit ${rc}
}

trap on_exit EXIT
update_plan_status running

echo "[info] HACA: ${HACA_CKPT}"
echo "[info] Stage1: ${S1_CKPT:-none}"

SEQ_IDS=(1 2 3 5)

run_variant() {
  local variant="$1"
  shift
  local exp="sca_lmf_mot20_${variant}_${TS}"
  local out_root="${OUT_BASE}/${variant}"
  local results_dir
  mkdir -p "${out_root}"

  CURRENT_VARIANT="${variant}"
  case "${variant}" in
    baseline)
      BASELINE_STATUS="running"
      ;;
    stage1_only)
      STAGE1_ONLY_STATUS="running"
      ;;
    stage1_freeze)
      STAGE1_FREEZE_STATUS="running"
      ;;
  esac
  write_eval_summary

  echo "[run] ${variant}"
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
      --laplace-assoc \
      --laplace-assoc-mode haca_v3 \
      --laplace-decay-scales 1 2 4 \
      --laplace-min-history 3 \
      --laplace-proto-mode multi \
      --laplace-primary-only \
      --laplace-haca-checkpoint "${HACA_CKPT}" \
      "$@" \
      > "${out_root}/track.log" 2>&1
  )

  results_dir="${BOT_ROOT}/YOLOX_outputs/${exp}/track_results"
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/eval_botsort_halfval_trackeval.py" \
    --dataset MOT20 \
    --data-root "${DATA_ROOT}" \
    --results-dir "${results_dir}" \
    --tracker-name "${exp}" \
    --work-dir "${out_root}/trackeval" \
    --remap-results-from-fullval \
    > "${out_root}/eval_trackeval.log" 2>&1

  local metrics_txt="${out_root}/trackeval/eval/${exp}/pedestrian_summary.txt"
  if [[ ! -f "${metrics_txt}" ]]; then
    echo "[FATAL] Missing TrackEval summary for ${variant}: ${metrics_txt}"
    exit 1
  fi
  case "${variant}" in
    baseline)
      BASELINE_STATUS="success"
      ;;
    stage1_only)
      STAGE1_ONLY_STATUS="success"
      ;;
    stage1_freeze)
      STAGE1_FREEZE_STATUS="success"
      ;;
  esac
  write_eval_summary
  echo "[done] ${variant}"
}

run_variant "baseline"

if [[ -n "${S1_CKPT}" ]]; then
  run_variant "stage1_only" \
    --rgsa-enable \
    --rgsa-stage1-checkpoint "${S1_CKPT}" \
    --rgsa-topk 5 \
    --rgsa-stage1-lambda-defer 0.15 \
    --rgsa-stage1-lambda-reject 0.0

  run_variant "stage1_freeze" \
    --rgsa-enable \
    --rgsa-stage1-checkpoint "${S1_CKPT}" \
    --rgsa-topk 5 \
    --rgsa-stage1-lambda-defer 0.15 \
    --rgsa-stage1-lambda-reject 0.0 \
    --tcgau-enable \
    --tcgau-freeze-thresh 0.03 \
    --tcgau-soft-thresh 0.0
else
  echo "[skip] Stage1 checkpoint missing; baseline only"
  STAGE1_ONLY_STATUS="skipped"
  STAGE1_FREEZE_STATUS="skipped"
  write_eval_summary
fi

CURRENT_VARIANT="done"
write_eval_summary

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/append_experiment_record.py" \
  --csv "${REGISTRY_CSV}" \
  --kind eval \
  --script "scripts/run_stage1_freezeonly_mot20_eval.sh" \
  --dataset MOT20 \
  --split val_half \
  --tracker-family BoT-SORT \
  --variant "sca_lmf_mot20_eval" \
  --run-root "${OUT_BASE}" \
  --summary-csv "${OUT_BASE}/summary.csv" \
  --checkpoint "${HACA_CKPT}" \
  --log-path "${LOG_PATH}" \
  --extra \
    "stage1_checkpoint=${S1_CKPT:-}" \
    "seq_ids=1,2,3,5"

update_plan_status completed
echo "[done] All variants complete. Summary: ${OUT_BASE}/summary.csv"
