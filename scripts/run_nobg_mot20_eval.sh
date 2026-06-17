#!/usr/bin/env bash
set -euo pipefail

# MOT20 SCA+LMF evaluation — NO BACKGROUND GATE variant.
# Runs baseline / freeze_only / Stage1 / Stage1+freeze and evaluates with TrackEval.

# GUARD: 4-concurrent MOT20 eval is DISABLED (2026-06-18).
# MOT20 eval must be strictly serial. Use scripts/run_haca_mot20_nobg_matrix.sh
# or the serial smoke/fullval scripts instead.
if [[ "${ALLOW_PARALLEL_MOT20_EVAL:-0}" != "1" ]]; then
  echo "[ERROR] 4-concurrent MOT20 eval is disabled (2026-06-18)."
  echo "  Set ALLOW_PARALLEL_MOT20_EVAL=1 to override at your own risk."
  echo "  Prefer serial: scripts/run_haca_mot20_nobg_matrix.sh or run_tos_mot20_smoke.sh"
  exit 1
fi
# All variants use --laplace-haca-no-background.

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
DATA_ROOT="${DATA_ROOT:-/gemini/code/datasets}"
BOT_ROOT="${REPO_ROOT}/external/BoT-SORT-main"
REGISTRY_CSV="${REGISTRY_CSV:-${REPO_ROOT}/outputs/experiment_registry.csv}"
PLAN_CSV="${PLAN_CSV:-${REPO_ROOT}/outputs/experiment_plan.csv}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
OUT_BASE="${OUT_BASE:-${REPO_ROOT}/outputs/sca_lmf_mot20_nobg_eval_${TS}}"
OUT_BASE="$(realpath -m "${OUT_BASE}")"
PLAN_KEY="${PLAN_KEY:-run_root:${OUT_BASE}}"
LOG_PATH="${OUT_BASE}/eval.log"
SUMMARY_CSV="${OUT_BASE}/summary.csv"
HACA_CKPT="${HACA_CKPT:-}"
S1_CKPT="${S1_CKPT:-${STAGE1_CKPT:-}}"

mkdir -p "${OUT_BASE}"
exec > >(tee -a "${LOG_PATH}") 2>&1

BASELINE_STATUS="pending"
FREEZE_ONLY_STATUS="pending"
STAGE1_ONLY_STATUS="pending"
STAGE1_FREEZE_STATUS="pending"
CURRENT_VARIANT="init"

write_eval_summary() {
  OUT_BASE="${OUT_BASE}" \
  TS="${TS}" \
  BASELINE_STATUS="${BASELINE_STATUS}" \
  FREEZE_ONLY_STATUS="${FREEZE_ONLY_STATUS}" \
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
    "freeze_only": os.environ["FREEZE_ONLY_STATUS"],
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
    metrics_txt = out_base / variant / "trackeval" / "eval" / f"sca_lmf_mot20_nobg_{variant}_{ts}" / "pedestrian_summary.txt"
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
  # Prefer no-bg Stage1, fall back to original
  S1_CKPT=$(find "${REPO_ROOT}/outputs" -path '*/stage1/stage1_best.pt' | grep '/rgsa_stage1_mot20_nobg_pipeline_' | sort | tail -1 || true)
  if [[ -z "${S1_CKPT}" ]]; then
    S1_CKPT=$(find "${REPO_ROOT}/outputs" -path '*/stage1/stage1_best.pt' | grep '/rgsa_stage1_mot20_pipeline_' | sort | tail -1 || true)
  fi
fi

if [[ -z "${HACA_CKPT}" ]]; then
  BASELINE_STATUS="failed"
  FREEZE_ONLY_STATUS="skipped"
  STAGE1_ONLY_STATUS="skipped"
  STAGE1_FREEZE_STATUS="skipped"
  write_eval_summary
  echo "[FATAL] No MOT20 HACA v3 checkpoint found."
  exit 1
fi

update_plan_status() {
  local status="$1"
  shift || true
  local extras=(
    "haca_checkpoint=${HACA_CKPT}"
    "stage1_checkpoint=${S1_CKPT:-}"
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
    --script "scripts/run_nobg_mot20_eval.sh" \
    --dataset MOT20 \
    --split val_half \
    --tracker-family BoT-SORT \
    --variant "sca_lmf_mot20_nobg_eval" \
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
      freeze_only)
        FREEZE_ONLY_STATUS="failed"
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
echo "[info] NO-BACKGROUND mode on all variants"

SEQ_IDS=(1 2 3 5)

# Common track.py args (always includes --laplace-haca-no-background)
COMMON_ARGS=(
  --laplace-assoc
  --laplace-assoc-mode haca_v3
  --laplace-decay-scales 1 2 4
  --laplace-min-history 3
  --laplace-proto-mode multi
  --laplace-primary-only
  --laplace-haca-checkpoint "${HACA_CKPT}"
  --laplace-haca-no-background
)

run_variant() {
  local variant="$1"
  shift
  local exp="sca_lmf_mot20_nobg_${variant}_${TS}"
  local out_root="${OUT_BASE}/${variant}"
  local results_dir
  mkdir -p "${out_root}"

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
      "${COMMON_ARGS[@]}" \
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
  echo "[done] ${variant}"
}

# Collect all variant definitions, then run tracking in parallel
declare -A VARIANT_ARGS
VARIANT_ARGS[baseline]=""
VARIANT_ARGS[freeze_only]="--tcgau-enable --tcgau-freeze-thresh 0.03 --tcgau-soft-thresh 0.0"

if [[ -n "${S1_CKPT}" ]]; then
  VARIANT_ARGS[stage1_only]="--rgsa-enable --rgsa-stage1-checkpoint ${S1_CKPT} --rgsa-topk 5 --rgsa-stage1-lambda-defer 0.15 --rgsa-stage1-lambda-reject 0.0"
  VARIANT_ARGS[stage1_freeze]="--rgsa-enable --rgsa-stage1-checkpoint ${S1_CKPT} --rgsa-topk 5 --rgsa-stage1-lambda-defer 0.15 --rgsa-stage1-lambda-reject 0.0 --tcgau-enable --tcgau-freeze-thresh 0.03 --tcgau-soft-thresh 0.0"
else
  echo "[skip] Stage1 checkpoint missing; baseline + freeze_only only"
  STAGE1_ONLY_STATUS="skipped"
  STAGE1_FREEZE_STATUS="skipped"
  write_eval_summary
fi

# Mark all as running
BASELINE_STATUS="running"
FREEZE_ONLY_STATUS="running"
[[ "${STAGE1_ONLY_STATUS}" != "skipped" ]] && STAGE1_ONLY_STATUS="running"
[[ "${STAGE1_FREEZE_STATUS}" != "skipped" ]] && STAGE1_FREEZE_STATUS="running"
write_eval_summary

# Launch all tracking in parallel (GPU memory: ~3GB each, 4 variants = ~12GB, fits in 24GB)
EVAL_PIDS=()
EVAL_VARIANTS=()
for variant in "${!VARIANT_ARGS[@]}"; do
  echo "[parallel-track] launching ${variant}"
  run_variant "${variant}" ${VARIANT_ARGS[${variant}]} &
  EVAL_PIDS+=($!)
  EVAL_VARIANTS+=("${variant}")
done

# Wait for all tracking + eval to finish
EVAL_FAIL=0
for i in "${!EVAL_PIDS[@]}"; do
  pid="${EVAL_PIDS[$i]}"
  variant="${EVAL_VARIANTS[$i]}"
  if wait "${pid}"; then
    case "${variant}" in
      baseline) BASELINE_STATUS="success" ;;
      freeze_only) FREEZE_ONLY_STATUS="success" ;;
      stage1_only) STAGE1_ONLY_STATUS="success" ;;
      stage1_freeze) STAGE1_FREEZE_STATUS="success" ;;
    esac
    write_eval_summary
  else
    echo "[FATAL] ${variant} (PID ${pid}) failed"
    case "${variant}" in
      baseline) BASELINE_STATUS="failed" ;;
      freeze_only) FREEZE_ONLY_STATUS="failed" ;;
      stage1_only) STAGE1_ONLY_STATUS="failed" ;;
      stage1_freeze) STAGE1_FREEZE_STATUS="failed" ;;
    esac
    write_eval_summary
    EVAL_FAIL=1
  fi
done
if [[ ${EVAL_FAIL} -ne 0 ]]; then
  echo "[FATAL] One or more eval variants failed"
  exit 1
fi

CURRENT_VARIANT="done"
write_eval_summary

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/append_experiment_record.py" \
  --csv "${REGISTRY_CSV}" \
  --kind eval \
  --script "scripts/run_nobg_mot20_eval.sh" \
  --dataset MOT20 \
  --split val_half \
  --tracker-family BoT-SORT \
  --variant "sca_lmf_mot20_nobg_eval" \
  --run-root "${OUT_BASE}" \
  --summary-csv "${OUT_BASE}/summary.csv" \
  --checkpoint "${HACA_CKPT}" \
  --log-path "${LOG_PATH}" \
  --extra \
    "stage1_checkpoint=${S1_CKPT:-}" \
    "no_background=true" \
    "seq_ids=1,2,3,5"

update_plan_status completed
echo "[done] All variants complete. Summary: ${OUT_BASE}/summary.csv"
