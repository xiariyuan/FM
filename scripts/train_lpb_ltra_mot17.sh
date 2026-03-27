#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
DEVICE="${DEVICE:-cuda}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
REGISTRY_CSV="${EXPERIMENT_REGISTRY_CSV:-${REPO_ROOT}/outputs/experiment_registry.csv}"
PLAN_CSV="${EXPERIMENT_PLAN_CSV:-${REPO_ROOT}/outputs/experiment_plan.csv}"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/outputs/gt_pseudotrack_mot17_latest}"
TRAIN_SHARDS_DIR="${TRAIN_SHARDS_DIR:-${DATA_ROOT}/train_shards}"
VAL_SHARDS_DIR="${VAL_SHARDS_DIR:-${DATA_ROOT}/val_shards}"
TRAIN_LIST="${TRAIN_LIST:-${DATA_ROOT}/train_npz_list.txt}"
VAL_LIST="${VAL_LIST:-${DATA_ROOT}/val_npz_list.txt}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs/lpb_ltra_train_${TS}}"
OUT_ROOT="$(realpath -m "${OUT_ROOT}")"
OUT_NPZ="${OUT_NPZ:-${OUT_ROOT}/mot17_lpb_ltra_${TS}.npz}"
PLAN_KEY="${PLAN_KEY:-run_root:${OUT_ROOT}}"

EPOCHS="${EPOCHS:-12}"
BATCH_GROUPS="${BATCH_GROUPS:-256}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
PAIR_HIDDEN="${PAIR_HIDDEN:-16}"
TRACK_HIDDEN="${TRACK_HIDDEN:-8}"
TAU_INIT="${TAU_INIT:-1 2 4 8}"
TAU_MIN="${TAU_MIN:-0.5}"
TAU_MAX="${TAU_MAX:-32}"
TEMPERATURE="${TEMPERATURE:-0.5}"
RANK_MARGIN="${RANK_MARGIN:-0.05}"
TRUST_MARGIN="${TRUST_MARGIN:-0.03}"
LOSS_BG_WEIGHT="${LOSS_BG_WEIGHT:-0.25}"
LOSS_RANK_WEIGHT="${LOSS_RANK_WEIGHT:-0.25}"
LOSS_TRUST_WEIGHT="${LOSS_TRUST_WEIGHT:-0.10}"
LOSS_POLE_WEIGHT="${LOSS_POLE_WEIGHT:-0.02}"
PATIENCE="${PATIENCE:-2}"
ALLOW_EMPTY_VAL="${ALLOW_EMPTY_VAL:-0}"

mkdir -p "${OUT_ROOT}"
LOG_PATH="${OUT_ROOT}/train.log"
exec > >(tee -a "${LOG_PATH}") 2>&1

update_plan_status() {
  local status="$1"
  shift || true
  local extras=(
    "epochs=${EPOCHS}"
    "batch_groups=${BATCH_GROUPS}"
    "lr=${LR}"
    "pair_hidden=${PAIR_HIDDEN}"
    "track_hidden=${TRACK_HIDDEN}"
    "tau_init=${TAU_INIT}"
    "tau_min=${TAU_MIN}"
    "tau_max=${TAU_MAX}"
    "patience=${PATIENCE}"
    "allow_empty_val=${ALLOW_EMPTY_VAL}"
  )
  if [[ $# -gt 0 ]]; then
    extras+=("$@")
  fi
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/upsert_experiment_plan.py" \
    --csv "${PLAN_CSV}" \
    --key "${PLAN_KEY}" \
    --status "${status}" \
    --kind train \
    --script "scripts/train_lpb_ltra_mot17.sh" \
    --dataset MOT17 \
    --split train_val_split \
    --tracker-family BoT-SORT \
    --variant lpb_ltra_train \
    --run-root "${OUT_ROOT}" \
    --checkpoint "${OUT_NPZ}" \
    --log-path "${LOG_PATH}" \
    --extra "${extras[@]}"
}

on_exit() {
  local rc=$?
  trap - EXIT
  if [[ ${rc} -ne 0 ]]; then
    update_plan_status failed "exit_code=${rc}" || true
  fi
  exit ${rc}
}

trap on_exit EXIT
update_plan_status running

collect_npzs() {
  local list_path="$1"
  local shard_dir="$2"
  local -n out_ref="$3"
  out_ref=()
  if [[ -f "${list_path}" ]]; then
    while IFS= read -r line; do
      [[ -n "${line}" ]] && out_ref+=("${line}")
    done < "${list_path}"
  fi
  if [[ ${#out_ref[@]} -eq 0 && -d "${shard_dir}" ]]; then
    mapfile -t out_ref < <(find "${shard_dir}" -maxdepth 1 -type f -name '*_groups.npz' | sort)
  fi
}

TRAIN_NPZS=()
VAL_NPZS=()
collect_npzs "${TRAIN_LIST}" "${TRAIN_SHARDS_DIR}" TRAIN_NPZS
collect_npzs "${VAL_LIST}" "${VAL_SHARDS_DIR}" VAL_NPZS

if [[ ${#TRAIN_NPZS[@]} -eq 0 ]]; then
  echo "[error] no training shard NPZ found under ${TRAIN_SHARDS_DIR}"
  exit 1
fi
if [[ ${#VAL_NPZS[@]} -eq 0 && "${ALLOW_EMPTY_VAL}" != "1" ]]; then
  echo "[error] no validation shard NPZ found under ${VAL_SHARDS_DIR}"
  exit 1
fi

echo "[start] $(date '+%F %T %z')"
echo "[train_shards] ${#TRAIN_NPZS[@]}"
printf '  %s\n' "${TRAIN_NPZS[@]}"
echo "[val_shards] ${#VAL_NPZS[@]}"
if [[ ${#VAL_NPZS[@]} -gt 0 ]]; then
  printf '  %s\n' "${VAL_NPZS[@]}"
else
  echo "  [disabled]"
fi
echo "[out_npz] ${OUT_NPZ}"

CMD=(
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/train_lpb_ltra_from_gt_tracks.py"
  --train-npz "${TRAIN_NPZS[@]}"
  --out-npz "${OUT_NPZ}"
  --device "${DEVICE}"
  --epochs "${EPOCHS}"
  --batch-groups "${BATCH_GROUPS}"
  --lr "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --pair-hidden "${PAIR_HIDDEN}"
  --track-hidden "${TRACK_HIDDEN}"
  --tau-init ${TAU_INIT}
  --tau-min "${TAU_MIN}"
  --tau-max "${TAU_MAX}"
  --temperature "${TEMPERATURE}"
  --rank-margin "${RANK_MARGIN}"
  --trust-margin "${TRUST_MARGIN}"
  --loss-bg-weight "${LOSS_BG_WEIGHT}"
  --loss-rank-weight "${LOSS_RANK_WEIGHT}"
  --loss-trust-weight "${LOSS_TRUST_WEIGHT}"
  --loss-pole-weight "${LOSS_POLE_WEIGHT}"
  --patience "${PATIENCE}"
)
if [[ ${#VAL_NPZS[@]} -gt 0 ]]; then
  CMD+=(--val-npz "${VAL_NPZS[@]}")
fi
"${CMD[@]}"
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/append_experiment_record.py" \
  --csv "${REGISTRY_CSV}" \
  --kind train \
  --script "scripts/train_lpb_ltra_mot17.sh" \
  --dataset MOT17 \
  --split train_val_split \
  --tracker-family BoT-SORT \
  --variant lpb_ltra_train \
  --run-root "${OUT_ROOT}" \
  --checkpoint "${OUT_NPZ}" \
  --log-path "${LOG_PATH}" \
  --extra \
    epochs="${EPOCHS}" \
    batch_groups="${BATCH_GROUPS}" \
    lr="${LR}" \
    pair_hidden="${PAIR_HIDDEN}" \
    track_hidden="${TRACK_HIDDEN}" \
    tau_init="${TAU_INIT}" \
    tau_min="${TAU_MIN}" \
    tau_max="${TAU_MAX}" \
    patience="${PATIENCE}" \
    allow_empty_val="${ALLOW_EMPTY_VAL}"
update_plan_status completed
echo "[done] $(date '+%F %T %z')"
echo "[checkpoint] ${OUT_NPZ}"
