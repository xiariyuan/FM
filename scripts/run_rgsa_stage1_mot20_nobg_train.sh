#!/usr/bin/env bash
set -euo pipefail

# MOT20 RGSA Stage1 training pipeline — NO BACKGROUND GATE variant.
# Same as run_rgsa_stage1_mot20_train.sh but with --laplace-haca-no-background
# on all track.py calls.  This fixes the background gate misfire that caused
# HOTA=9.36 in the original pipeline.

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
DATA_ROOT="${DATA_ROOT:-/gemini/code/datasets}"
BOT_ROOT="${REPO_ROOT}/external/BoT-SORT-main"
DEVICE="${DEVICE:-cuda}"
REGISTRY_CSV="${REGISTRY_CSV:-${REPO_ROOT}/outputs/experiment_registry.csv}"
PLAN_CSV="${PLAN_CSV:-${REPO_ROOT}/outputs/experiment_plan.csv}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs/rgsa_stage1_mot20_nobg_pipeline_${TS}}"
OUT_ROOT="$(realpath -m "${OUT_ROOT}")"
PLAN_KEY="${PLAN_KEY:-run_root:${OUT_ROOT}}"
LOG_PATH="${OUT_ROOT}/pipeline.log"
SUMMARY_CSV="${OUT_ROOT}/summary.csv"
HACA_CKPT="${HACA_CKPT:-}"

TRAIN_SEQS=(MOT20-01 MOT20-02 MOT20-03)
VAL_SEQ="MOT20-05"
DUMP_SEQS=(1 2 3 5)

ORACLE_OUT="${OUT_ROOT}/oracle_dump"
LABELS_OUT="${OUT_ROOT}/labels"
STAGE1_OUT="${OUT_ROOT}/stage1"

mkdir -p "${OUT_ROOT}"
exec > >(tee -a "${LOG_PATH}") 2>&1

HACA_STATUS="pending"
ORACLE_STATUS="pending"
LABELS_STATUS="pending"
STAGE1_STATUS="pending"
CURRENT_PHASE="haca_checkpoint"

write_summary_status() {
  cat > "${SUMMARY_CSV}" <<EOF
phase,status,artifact
haca_checkpoint,${HACA_STATUS},${HACA_CKPT}
oracle_dump,${ORACLE_STATUS},${ORACLE_OUT}
labels,${LABELS_STATUS},${LABELS_OUT}
stage1,${STAGE1_STATUS},${STAGE1_OUT}/stage1_best.pt
EOF
}

update_plan_status() {
  local status="$1"
  shift || true
  local extras=(
    "device=${DEVICE}"
    "train_seqs=${TRAIN_SEQS[*]}"
    "val_seq=${VAL_SEQ}"
    "no_background=true"
  )
  if [[ $# -gt 0 ]]; then
    extras+=("$@")
  fi
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/upsert_experiment_plan.py" \
    --csv "${PLAN_CSV}" \
    --key "${PLAN_KEY}" \
    --status "${status}" \
    --kind train \
    --script "scripts/run_rgsa_stage1_mot20_nobg_train.sh" \
    --dataset MOT20 \
    --split train_half_val_half \
    --tracker-family BoT-SORT \
    --variant "rgsa_stage1_mot20_nobg_train" \
    --run-root "${OUT_ROOT}" \
    --summary-csv "${OUT_ROOT}/summary.csv" \
    --checkpoint "${OUT_ROOT}/stage1/stage1_best.pt" \
    --log-path "${LOG_PATH}" \
    --extra "${extras[@]}"
}

on_exit() {
  local rc=$?
  trap - EXIT
  if [[ ${rc} -ne 0 ]]; then
    case "${CURRENT_PHASE}" in
      haca_checkpoint)
        HACA_STATUS="failed"
        ;;
      oracle_dump)
        ORACLE_STATUS="failed"
        ;;
      labels)
        LABELS_STATUS="failed"
        ;;
      stage1)
        STAGE1_STATUS="failed"
        ;;
    esac
    write_summary_status || true
    update_plan_status failed "exit_code=${rc}" || true
  fi
  exit ${rc}
}

trap on_exit EXIT
HACA_STATUS="running"
write_summary_status
update_plan_status running

if [[ -z "${HACA_CKPT}" ]]; then
  HACA_CKPT=$(find "${REPO_ROOT}/outputs" -path '*/haca_v3/mot20_haca_v3.npz' | sort | tail -1)
fi
if [[ -z "${HACA_CKPT}" ]]; then
  HACA_STATUS="failed"
  write_summary_status
  echo "[FATAL] No MOT20 HACA v3 checkpoint found."
  echo "Run: bash scripts/run_haca_v3_mot20_train.sh"
  exit 1
fi
HACA_STATUS="success"
ORACLE_STATUS="running"
CURRENT_PHASE="oracle_dump"
write_summary_status
echo "[info] Using HACA checkpoint: ${HACA_CKPT}"
echo "[info] NO-BACKGROUND mode: --laplace-haca-no-background on all track.py calls"

echo "=== Phase 1: RGSA Oracle Dump (no-background, PARALLEL) ==="
mkdir -p "${ORACLE_OUT}"

# Build list of sequences that need dumping
DUMP_PIDS=()
for SEQ_ID in "${DUMP_SEQS[@]}"; do
  SEQ_NAME="MOT20-$(printf '%02d' "${SEQ_ID}")"
  mkdir -p "${ORACLE_OUT}/${SEQ_NAME}"
  if [[ -f "${ORACLE_OUT}/${SEQ_NAME}/.complete" && -s "${ORACLE_OUT}/${SEQ_NAME}/pairbank.csv" ]]; then
    echo "[dump] ${SEQ_NAME} already complete; skipping"
    continue
  fi
  rm -f "${ORACLE_OUT}/${SEQ_NAME}/.complete"
  echo "[dump] ${SEQ_NAME} (parallel)"
  (
    cd "${BOT_ROOT}"
    "${PYTHON_BIN}" -u tools/track.py "${DATA_ROOT}/MOT20" \
      --benchmark MOT20 \
      --eval train \
      --seq-ids "${SEQ_ID}" \
      -f ./yolox/exps/example/mot/yolox_x_mix_mot20_ch.py \
      -c ./pretrained/bytetrack_x_mot20.pth.tar \
      --with-reid \
      --fast-reid-config fast_reid/configs/MOT20/sbs_S50.yml \
      --fast-reid-weights pretrained/mot20_sbs_S50.pth \
      --experiment-name "rgsa_oracle_mot20_nobg_${SEQ_NAME}_${TS}" \
      --laplace-assoc \
      --laplace-assoc-mode haca_v3 \
      --laplace-decay-scales 1 2 4 \
      --laplace-min-history 3 \
      --laplace-proto-mode multi \
      --laplace-primary-only \
      --laplace-haca-checkpoint "${HACA_CKPT}" \
      --laplace-haca-no-background \
      --rgsa-dump-dir "${ORACLE_OUT}/${SEQ_NAME}" \
      > "${ORACLE_OUT}/${SEQ_NAME}.log" 2>&1 \
      && touch "${ORACLE_OUT}/${SEQ_NAME}/.complete"
  ) &
  DUMP_PIDS+=($!)
done

# Wait for all parallel dumps to finish
DUMP_FAIL=0
for pid in "${DUMP_PIDS[@]}"; do
  if ! wait "${pid}"; then
    echo "[FATAL] Oracle dump process ${pid} failed"
    DUMP_FAIL=1
  fi
done
if [[ ${DUMP_FAIL} -ne 0 ]]; then
  echo "[FATAL] One or more oracle dumps failed"
  exit 1
fi

echo "=== Phase 2: Build RGSA Labels ==="
ORACLE_STATUS="success"
LABELS_STATUS="running"
CURRENT_PHASE="labels"
write_summary_status
"${PYTHON_BIN}" -u "${REPO_ROOT}/scripts/build_rgsa_labels.py" \
  --oracle-dir "${ORACLE_OUT}" \
  --data-root "${DATA_ROOT}" \
  --dataset MOT20 \
  --seqs "${TRAIN_SEQS[@]}" "${VAL_SEQ}" \
  --out-dir "${LABELS_OUT}" \
  --margin-threshold 0.05 \
  --topk 5 \
  --iou-threshold 0.7

echo "=== Phase 3: Train Stage1 ==="
LABELS_STATUS="success"
STAGE1_STATUS="running"
CURRENT_PHASE="stage1"
write_summary_status
"${PYTHON_BIN}" -u "${REPO_ROOT}/scripts/train_rgsa_stage1.py" \
  --labels-dir "${LABELS_OUT}" \
  --train-seqs "${TRAIN_SEQS[@]}" \
  --val-seqs "${VAL_SEQ}" \
  --epochs 30 \
  --batch-size 256 \
  --lr 1e-3 \
  --hidden-dims 32 16 \
  --dropout 0.1 \
  --patience 5 \
  --out-dir "${STAGE1_OUT}" \
  --device "${DEVICE}"

HACA_STATUS="success"
ORACLE_STATUS="success"
LABELS_STATUS="success"
STAGE1_STATUS="success"
CURRENT_PHASE="done"
write_summary_status

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/append_experiment_record.py" \
  --csv "${REGISTRY_CSV}" \
  --kind train \
  --script "scripts/run_rgsa_stage1_mot20_nobg_train.sh" \
  --dataset MOT20 \
  --split train_half_val_half \
  --tracker-family BoT-SORT \
  --variant "rgsa_stage1_mot20_nobg_train" \
  --run-root "${OUT_ROOT}" \
  --summary-csv "${OUT_ROOT}/summary.csv" \
  --checkpoint "${STAGE1_OUT}/stage1_best.pt" \
  --log-path "${LOG_PATH}" \
  --extra \
    "device=${DEVICE}" \
    "haca_checkpoint=${HACA_CKPT}" \
    "no_background=true" \
    "train_seqs=MOT20-01,MOT20-02,MOT20-03" \
    "val_seq=MOT20-05"

update_plan_status completed
echo "[done] Stage1 checkpoint (no-bg): ${STAGE1_OUT}/stage1_best.pt"
