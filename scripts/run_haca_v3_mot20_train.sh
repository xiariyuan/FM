#!/usr/bin/env bash
set -euo pipefail

# MOT20 HACA v3 training pipeline.
# Builds GT pseudo-track shards, then trains HACA v1 -> v2 -> v3.

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
DATA_ROOT="${DATA_ROOT:-/gemini/code/datasets}"
DEVICE="${DEVICE:-cuda}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
REGISTRY_CSV="${REGISTRY_CSV:-${REPO_ROOT}/outputs/experiment_registry.csv}"
PLAN_CSV="${PLAN_CSV:-${REPO_ROOT}/outputs/experiment_plan.csv}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs/haca_mot20_train_${TS}}"
OUT_ROOT="$(realpath -m "${OUT_ROOT}")"
PLAN_KEY="${PLAN_KEY:-run_root:${OUT_ROOT}}"
LOG_PATH="${OUT_ROOT}/train.log"
SUMMARY_CSV="${OUT_ROOT}/summary.csv"

TRAIN_SEQS=(MOT20-01 MOT20-02 MOT20-03)
VAL_SEQ="MOT20-05"
FRAME_WINDOW="${FRAME_WINDOW:-120}"
PSEUDOTRACK_CANDIDATE_TOPK="${PSEUDOTRACK_CANDIDATE_TOPK:-16}"
PSEUDOTRACK_MAX_HARD_NEGATIVES="${PSEUDOTRACK_MAX_HARD_NEGATIVES:-6}"
PSEUDOTRACK_MAX_RANDOM_NEGATIVES="${PSEUDOTRACK_MAX_RANDOM_NEGATIVES:-2}"
PSEUDOTRACK_POSITIVE_KEEP_PROB="${PSEUDOTRACK_POSITIVE_KEEP_PROB:-0.7}"

mkdir -p "${OUT_ROOT}"
exec > >(tee -a "${LOG_PATH}") 2>&1

MOT20_OUT="${OUT_ROOT}/gt_pseudotrack"
HACA_V1_OUT="${OUT_ROOT}/haca_v1"
HACA_V2_OUT="${OUT_ROOT}/haca_v2"
HACA_V3_OUT="${OUT_ROOT}/haca_v3"
HACA_V3_POSITIVE_INJECTION_PROB="${HACA_V3_POSITIVE_INJECTION_PROB:-0.7}"

GT_STATUS="pending"
HACA_V1_STATUS="pending"
HACA_V2_STATUS="pending"
HACA_V3_STATUS="pending"
CURRENT_PHASE="init"

is_valid_npz() {
  local path="$1"
  [[ -f "${path}" ]] && "${PYTHON_BIN}" -c "import numpy as np; np.load('${path}', allow_pickle=True)" >/dev/null 2>&1
}

write_summary_status() {
  cat > "${SUMMARY_CSV}" <<EOF
phase,status,artifact
gt_pseudotrack,${GT_STATUS},${MOT20_OUT}
haca_v1,${HACA_V1_STATUS},${HACA_V1_OUT}/mot20_haca_v1.npz
haca_v2,${HACA_V2_STATUS},${HACA_V2_OUT}/mot20_haca_v2.npz
haca_v3,${HACA_V3_STATUS},${HACA_V3_OUT}/mot20_haca_v3.npz
EOF
}

update_plan_status() {
  local status="$1"
  shift || true
  local extras=(
    "device=${DEVICE}"
    "train_seqs=${TRAIN_SEQS[*]}"
    "val_seq=${VAL_SEQ}"
  )
  if [[ $# -gt 0 ]]; then
    extras+=("$@")
  fi
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/upsert_experiment_plan.py" \
    --csv "${PLAN_CSV}" \
    --key "${PLAN_KEY}" \
    --status "${status}" \
    --kind train \
    --script "scripts/run_haca_v3_mot20_train.sh" \
    --dataset MOT20 \
    --split train_half_val_half \
    --tracker-family BoT-SORT \
    --variant "haca_v3_mot20_train" \
    --run-root "${OUT_ROOT}" \
    --summary-csv "${OUT_ROOT}/summary.csv" \
    --checkpoint "${OUT_ROOT}/haca_v3/mot20_haca_v3.npz" \
    --log-path "${LOG_PATH}" \
    --extra "${extras[@]}"
}

on_exit() {
  local rc=$?
  trap - EXIT
  if [[ ${rc} -ne 0 ]]; then
    # Distinguish exit semantics by signal that killed us
    local exit_semantic="failed"
    if [[ ${rc} -eq 137 ]] || [[ ${rc} -eq 143 ]] || [[ ${rc} -eq 141 ]]; then
      # SIGKILL(137), SIGTERM(143), SIGPIPE(141) — external kill, not code bug
      exit_semantic="interrupted"
    fi
    case "${CURRENT_PHASE}" in
      gt_pseudotrack)
        GT_STATUS="${exit_semantic}"
        ;;
      haca_v1)
        HACA_V1_STATUS="${exit_semantic}"
        ;;
      haca_v2)
        HACA_V2_STATUS="${exit_semantic}"
        ;;
      haca_v3)
        HACA_V3_STATUS="${exit_semantic}"
        ;;
    esac
    write_summary_status || true
    update_plan_status "${exit_semantic}" "exit_code=${rc}" || true
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/append_experiment_record.py" \
      --csv "${REGISTRY_CSV}" \
      --status "${exit_semantic}" \
      --kind train \
      --script "scripts/run_haca_v3_mot20_train.sh" \
      --dataset MOT20 \
      --split train_half_val_half \
      --tracker-family BoT-SORT \
      --variant "haca_v3_mot20_train" \
      --run-root "${OUT_ROOT}" \
      --summary-csv "${OUT_ROOT}/summary.csv" \
      --log-path "${LOG_PATH}" \
      --extra \
        "exit_code=${rc}" \
        "exit_semantic=${exit_semantic}" \
        "failed_phase=${CURRENT_PHASE}" \
        "train_seqs=MOT20-01,MOT20-02,MOT20-03" \
        "val_seq=MOT20-05" || true
  fi
  exit ${rc}
}

trap on_exit EXIT
update_plan_status running
GT_STATUS="running"
write_summary_status

BOT_ROOT="${REPO_ROOT}/external/BoT-SORT-main"
mkdir -p "${MOT20_OUT}/train_shards" "${MOT20_OUT}/val_shards"
mkdir -p "${HACA_V1_OUT}" "${HACA_V2_OUT}" "${HACA_V3_OUT}"

echo "=== Phase 0.1: Build MOT20 GT pseudotrack shards ==="
CURRENT_PHASE="gt_pseudotrack"
build_shards() {
  local seq="$1"
  local split_part="$2"
  local out_dir="$3"
  local start_frame="$4"
  local end_frame="$5"

  local shard_idx=0
  local frame_start="${start_frame}"
  while [[ "${frame_start}" -le "${end_frame}" ]]; do
    local frame_end=$((frame_start + FRAME_WINDOW - 1))
    if [[ "${frame_end}" -gt "${end_frame}" ]]; then
      frame_end="${end_frame}"
    fi
    local suffix
    printf -v suffix "f%04d_%04d" "${frame_start}" "${frame_end}"
    local npz_path="${out_dir}/${seq}_${suffix}_groups.npz"
    if [[ -f "${npz_path}" ]] && "${PYTHON_BIN}" -c "import numpy as np; np.load('${npz_path}', allow_pickle=True)" 2>/dev/null; then
      echo "[skip] ${seq} ${split_part} ${suffix} (already exists and valid)"
      shard_idx=$((shard_idx + 1))
      frame_start=$((frame_end + 1))
      continue
    fi
    echo "[build] ${seq} ${split_part} ${suffix}"
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/build_gt_pseudotrack_groups.py" \
      --dataset MOT20 \
      --data-root "${DATA_ROOT}" \
      --seqs "${seq}" \
      --split-part "${split_part}" \
      --fast-reid-config "${BOT_ROOT}/fast_reid/configs/MOT20/sbs_S50.yml" \
      --fast-reid-weights "${BOT_ROOT}/pretrained/mot20_sbs_S50.pth" \
      --device "${DEVICE}" \
      --batch-size 4 \
      --max-history 8 \
      --min-history 3 \
      --feature-dtype float16 \
      --seed 123 \
      --smooth-alpha 0.9 \
      --iou-pos 0.7 \
      --iou-ignore 0.5 \
      --max-gap 30 \
      --candidate-topk "${PSEUDOTRACK_CANDIDATE_TOPK}" \
      --max-hard-negatives "${PSEUDOTRACK_MAX_HARD_NEGATIVES}" \
      --max-random-negatives "${PSEUDOTRACK_MAX_RANDOM_NEGATIVES}" \
      --positive-keep-prob "${PSEUDOTRACK_POSITIVE_KEEP_PROB}" \
      --frame-start "${frame_start}" \
      --frame-end "${frame_end}" \
      --out-npz "${out_dir}/${seq}_${suffix}_groups.npz" \
      --out-csv "${out_dir}/${seq}_${suffix}_pairs.csv"
    shard_idx=$((shard_idx + 1))
    frame_start=$((frame_end + 1))
  done
  echo "[done] ${seq} ${split_part} shards=${shard_idx}"
}

build_shards MOT20-01 train_half "${MOT20_OUT}/train_shards" 1 215
build_shards MOT20-02 train_half "${MOT20_OUT}/train_shards" 1 1392
build_shards MOT20-03 train_half "${MOT20_OUT}/train_shards" 1 1203
build_shards MOT20-05 val_half "${MOT20_OUT}/val_shards" 1 1657

find "${MOT20_OUT}/train_shards" -maxdepth 1 -type f -name '*_groups.npz' | sort > "${MOT20_OUT}/train_npz_list.txt"
find "${MOT20_OUT}/val_shards" -maxdepth 1 -type f -name '*_groups.npz' | sort > "${MOT20_OUT}/val_npz_list.txt"

TRAIN_NPZ_COUNT=$(wc -l < "${MOT20_OUT}/train_npz_list.txt")
VAL_NPZ_COUNT=$(wc -l < "${MOT20_OUT}/val_npz_list.txt")
echo "[check] train shards=${TRAIN_NPZ_COUNT} val shards=${VAL_NPZ_COUNT}"
if [[ "${TRAIN_NPZ_COUNT}" -lt 3 ]]; then
  echo "[FATAL] expected 3 training shards, got ${TRAIN_NPZ_COUNT}"
  exit 1
fi
if [[ "${VAL_NPZ_COUNT}" -lt 1 ]]; then
  echo "[FATAL] expected >=1 val shard, got ${VAL_NPZ_COUNT}"
  exit 1
fi

mapfile -t TRAIN_NPZS < "${MOT20_OUT}/train_npz_list.txt"
mapfile -t VAL_NPZS < "${MOT20_OUT}/val_npz_list.txt"

echo "=== Phase 0.2: Train HACA v1 ==="
CURRENT_PHASE="haca_v1"
GT_STATUS="success"
if is_valid_npz "${HACA_V1_OUT}/mot20_haca_v1.npz"; then
  echo "[skip] HACA v1 checkpoint already exists: ${HACA_V1_OUT}/mot20_haca_v1.npz"
  HACA_V1_STATUS="success"
  write_summary_status
else
  HACA_V1_STATUS="running"
  write_summary_status
  "${PYTHON_BIN}" -u "${REPO_ROOT}/scripts/train_haca_v1_from_gt_tracks.py" \
    --version haca_v1 \
    --train-npz "${TRAIN_NPZS[@]}" \
    --val-npz "${VAL_NPZS[@]}" \
    --out-npz "${HACA_V1_OUT}/mot20_haca_v1.npz" \
    --device "${DEVICE}" \
    --disable-background \
    --epochs 20 \
    --batch-groups 128 \
    --lr 1e-4 \
    --patience 4 > "${HACA_V1_OUT}/train.log" 2>&1
fi

echo "=== Phase 0.3: Train HACA v2 ==="
CURRENT_PHASE="haca_v2"
HACA_V1_STATUS="success"
if is_valid_npz "${HACA_V2_OUT}/mot20_haca_v2.npz"; then
  echo "[skip] HACA v2 checkpoint already exists: ${HACA_V2_OUT}/mot20_haca_v2.npz"
  HACA_V2_STATUS="success"
  write_summary_status
else
  HACA_V2_STATUS="running"
  write_summary_status
  "${PYTHON_BIN}" -u "${REPO_ROOT}/scripts/train_haca_v1_from_gt_tracks.py" \
    --version haca_v2 \
    --train-npz "${TRAIN_NPZS[@]}" \
    --val-npz "${VAL_NPZS[@]}" \
    --out-npz "${HACA_V2_OUT}/mot20_haca_v2.npz" \
    --device "${DEVICE}" \
    --disable-background \
    --epochs 20 \
    --batch-groups 128 \
    --lr 1e-4 \
    --patience 4 > "${HACA_V2_OUT}/train.log" 2>&1
fi

echo "=== Phase 0.4: Train HACA v3 ==="
CURRENT_PHASE="haca_v3"
HACA_V2_STATUS="success"
if is_valid_npz "${HACA_V3_OUT}/mot20_haca_v3.npz"; then
  echo "[skip] HACA v3 checkpoint already exists: ${HACA_V3_OUT}/mot20_haca_v3.npz"
  HACA_V3_STATUS="success"
  write_summary_status
else
  HACA_V3_STATUS="running"
  write_summary_status
  "${PYTHON_BIN}" -u "${REPO_ROOT}/scripts/train_haca_v3_from_gt_tracks.py" \
    --version haca_v3 \
    --base-npz "${HACA_V2_OUT}/mot20_haca_v2.npz" \
    --train-npz "${TRAIN_NPZS[@]}" \
    --val-npz "${VAL_NPZS[@]}" \
    --out-npz "${HACA_V3_OUT}/mot20_haca_v3.npz" \
    --device "${DEVICE}" \
    --disable-background \
    --epochs 12 \
    --batch-groups 128 \
    --lr 1e-4 \
    --positive-injection-prob "${HACA_V3_POSITIVE_INJECTION_PROB}" \
    --patience 3 > "${HACA_V3_OUT}/train.log" 2>&1
fi

GT_STATUS="success"
HACA_V1_STATUS="success"
HACA_V2_STATUS="success"
HACA_V3_STATUS="success"
write_summary_status

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/append_experiment_record.py" \
  --csv "${REGISTRY_CSV}" \
  --kind train \
  --script "scripts/run_haca_v3_mot20_train.sh" \
  --dataset MOT20 \
  --split train_half_val_half \
  --tracker-family BoT-SORT \
  --variant "haca_v3_mot20_train" \
  --run-root "${OUT_ROOT}" \
  --summary-csv "${OUT_ROOT}/summary.csv" \
  --checkpoint "${HACA_V3_OUT}/mot20_haca_v3.npz" \
  --log-path "${LOG_PATH}" \
  --extra \
    "device=${DEVICE}" \
    "pseudotrack_candidate_topk=${PSEUDOTRACK_CANDIDATE_TOPK}" \
    "pseudotrack_max_hard_negatives=${PSEUDOTRACK_MAX_HARD_NEGATIVES}" \
    "pseudotrack_max_random_negatives=${PSEUDOTRACK_MAX_RANDOM_NEGATIVES}" \
    "pseudotrack_positive_keep_prob=${PSEUDOTRACK_POSITIVE_KEEP_PROB}" \
    "haca_v3_positive_injection_prob=${HACA_V3_POSITIVE_INJECTION_PROB}" \
    "train_seqs=MOT20-01,MOT20-02,MOT20-03" \
    "val_seq=MOT20-05"

CURRENT_PHASE="done"
update_plan_status completed
echo "[done] MOT20 HACA v3 checkpoint: ${HACA_V3_OUT}/mot20_haca_v3.npz"
