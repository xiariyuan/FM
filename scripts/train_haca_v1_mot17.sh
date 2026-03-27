#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
DEVICE="${DEVICE:-cuda}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
REGISTRY_CSV="${EXPERIMENT_REGISTRY_CSV:-${REPO_ROOT}/outputs/experiment_registry.csv}"
PLAN_CSV="${EXPERIMENT_PLAN_CSV:-${REPO_ROOT}/outputs/experiment_plan.csv}"
HACA_LABEL="${HACA_LABEL:-haca_v1}"
HACA_VERSION="${HACA_VERSION:-haca_v1}"
SCRIPT_NAME="${SCRIPT_NAME:-scripts/train_haca_v1_mot17.sh}"
VARIANT_NAME="${VARIANT_NAME:-${HACA_LABEL}_train}"
TRAIN_SCRIPT_PATH="${TRAIN_SCRIPT_PATH:-${REPO_ROOT}/scripts/train_haca_v1_from_gt_tracks.py}"
BASE_NPZ="${BASE_NPZ:-}"
COMP_HIDDEN="${COMP_HIDDEN:-}"
COMP_TOPK="${COMP_TOPK:-}"
COMP_MARGIN_QUANTILE="${COMP_MARGIN_QUANTILE:-}"
COMP_MARGIN_TEMPERATURE="${COMP_MARGIN_TEMPERATURE:-}"
COMP_DELTA_SCALE="${COMP_DELTA_SCALE:-}"
DUEL_MARGIN="${DUEL_MARGIN:-}"
LOSS_DUEL_WEIGHT="${LOSS_DUEL_WEIGHT:-}"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/outputs/lpb_ltra_formal_mot17_shrink_20260311_191211/gt_builder}"
TRAIN_SHARDS_DIR="${TRAIN_SHARDS_DIR:-${DATA_ROOT}/train_shards}"
VAL_SHARDS_DIR="${VAL_SHARDS_DIR:-${DATA_ROOT}/val_shards}"
TRAIN_LIST="${TRAIN_LIST:-${DATA_ROOT}/train_npz_list.txt}"
VAL_LIST="${VAL_LIST:-${DATA_ROOT}/val_npz_list.txt}"

OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs/${HACA_LABEL}_train_${TS}}"
OUT_ROOT="$(realpath -m "${OUT_ROOT}")"
OUT_NPZ="${OUT_NPZ:-${OUT_ROOT}/mot17_${HACA_LABEL}_${TS}.npz}"
PLAN_KEY="${PLAN_KEY:-run_root:${OUT_ROOT}}"

EPOCHS="${EPOCHS:-20}"
BATCH_GROUPS="${BATCH_GROUPS:-256}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
GRAD_CLIP="${GRAD_CLIP:-5.0}"
HIST_HIDDEN="${HIST_HIDDEN:-16}"
PAIR_HIDDEN="${PAIR_HIDDEN:-64}"
ANCHOR_ALPHA="${ANCHOR_ALPHA:-0.35}"
DELTA_SCALE="${DELTA_SCALE:-1.5}"
LAPLACE_DECAY_SCALES="${LAPLACE_DECAY_SCALES:-1 2 4}"
MIN_HISTORY="${MIN_HISTORY:-3}"
MAX_HISTORY="${MAX_HISTORY:-0}"
TEMPERATURE="${TEMPERATURE:-1.0}"
MARGIN="${MARGIN:-0.05}"
SAFE_MARGIN="${SAFE_MARGIN:-0.10}"
LOSS_BG_WEIGHT="${LOSS_BG_WEIGHT:-0.25}"
LOSS_MARGIN_WEIGHT="${LOSS_MARGIN_WEIGHT:-0.25}"
LOSS_SAFE_WEIGHT="${LOSS_SAFE_WEIGHT:-0.10}"
LOSS_RES_WEIGHT="${LOSS_RES_WEIGHT:-0.02}"
LOSS_SHIFT_WEIGHT="${LOSS_SHIFT_WEIGHT:-0.0}"
SHIFT_BATCH_PROB="${SHIFT_BATCH_PROB:-0.0}"
HIST_GATE_HIDDEN="${HIST_GATE_HIDDEN:-8}"
OOD_SCALE="${OOD_SCALE:-6.0}"
OOD_QUANTILE="${OOD_QUANTILE:-0.95}"
CORRUPT_FEAT_NOISE="${CORRUPT_FEAT_NOISE:-0.03}"
CORRUPT_SCORE_NOISE="${CORRUPT_SCORE_NOISE:-0.08}"
CORRUPT_HISTORY_MIN_RATIO="${CORRUPT_HISTORY_MIN_RATIO:-0.35}"
PATIENCE="${PATIENCE:-3}"
DISABLE_SET_ENCODER="${DISABLE_SET_ENCODER:-0}"
DISABLE_BACKGROUND="${DISABLE_BACKGROUND:-0}"
DISABLE_HIST_GATE="${DISABLE_HIST_GATE:-0}"
DISABLE_OOD_GATE="${DISABLE_OOD_GATE:-0}"
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
    "hist_hidden=${HIST_HIDDEN}"
    "pair_hidden=${PAIR_HIDDEN}"
    "anchor_alpha=${ANCHOR_ALPHA}"
    "delta_scale=${DELTA_SCALE}"
    "haca_version=${HACA_VERSION}"
    "min_history=${MIN_HISTORY}"
    "max_history=${MAX_HISTORY}"
    "disable_set_encoder=${DISABLE_SET_ENCODER}"
    "disable_background=${DISABLE_BACKGROUND}"
    "disable_hist_gate=${DISABLE_HIST_GATE}"
    "disable_ood_gate=${DISABLE_OOD_GATE}"
    "loss_shift_weight=${LOSS_SHIFT_WEIGHT}"
    "allow_empty_val=${ALLOW_EMPTY_VAL}"
  )
  if [[ $# -gt 0 ]]; then
    extras+=("$@")
  fi
  if [[ -n "${BASE_NPZ}" ]]; then
    extras+=("base_npz=${BASE_NPZ}")
  fi
  if [[ -n "${COMP_TOPK}" ]]; then
    extras+=("comp_topk=${COMP_TOPK}")
  fi
  if [[ -n "${COMP_MARGIN_QUANTILE}" ]]; then
    extras+=("comp_margin_quantile=${COMP_MARGIN_QUANTILE}")
  fi
  if [[ -n "${COMP_MARGIN_TEMPERATURE}" ]]; then
    extras+=("comp_margin_temperature=${COMP_MARGIN_TEMPERATURE}")
  fi
  if [[ -n "${COMP_DELTA_SCALE}" ]]; then
    extras+=("comp_delta_scale=${COMP_DELTA_SCALE}")
  fi
  if [[ -n "${DUEL_MARGIN}" ]]; then
    extras+=("duel_margin=${DUEL_MARGIN}")
  fi
  if [[ -n "${LOSS_DUEL_WEIGHT}" ]]; then
    extras+=("loss_duel_weight=${LOSS_DUEL_WEIGHT}")
  fi
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/upsert_experiment_plan.py" \
    --csv "${PLAN_CSV}" \
    --key "${PLAN_KEY}" \
    --status "${status}" \
    --kind train \
    --script "${SCRIPT_NAME}" \
    --dataset MOT17 \
    --split train_val_split \
    --tracker-family BoT-SORT \
    --variant "${VARIANT_NAME}" \
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
  "${PYTHON_BIN}" "${TRAIN_SCRIPT_PATH}"
  --version "${HACA_VERSION}"
  --train-npz "${TRAIN_NPZS[@]}"
  --out-npz "${OUT_NPZ}"
  --device "${DEVICE}"
  --epochs "${EPOCHS}"
  --batch-groups "${BATCH_GROUPS}"
  --lr "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --grad-clip "${GRAD_CLIP}"
  --hist-hidden "${HIST_HIDDEN}"
  --pair-hidden "${PAIR_HIDDEN}"
  --hist-gate-hidden "${HIST_GATE_HIDDEN}"
  --anchor-alpha "${ANCHOR_ALPHA}"
  --delta-scale "${DELTA_SCALE}"
  --laplace-decay-scales ${LAPLACE_DECAY_SCALES}
  --min-history "${MIN_HISTORY}"
  --max-history "${MAX_HISTORY}"
  --temperature "${TEMPERATURE}"
  --margin "${MARGIN}"
  --safe-margin "${SAFE_MARGIN}"
  --loss-bg-weight "${LOSS_BG_WEIGHT}"
  --loss-margin-weight "${LOSS_MARGIN_WEIGHT}"
  --loss-safe-weight "${LOSS_SAFE_WEIGHT}"
  --loss-res-weight "${LOSS_RES_WEIGHT}"
  --loss-shift-weight "${LOSS_SHIFT_WEIGHT}"
  --shift-batch-prob "${SHIFT_BATCH_PROB}"
  --ood-scale "${OOD_SCALE}"
  --ood-quantile "${OOD_QUANTILE}"
  --corrupt-feat-noise "${CORRUPT_FEAT_NOISE}"
  --corrupt-score-noise "${CORRUPT_SCORE_NOISE}"
  --corrupt-history-min-ratio "${CORRUPT_HISTORY_MIN_RATIO}"
  --patience "${PATIENCE}"
)
if [[ ${#VAL_NPZS[@]} -gt 0 ]]; then
  CMD+=(--val-npz "${VAL_NPZS[@]}")
fi
if [[ -n "${BASE_NPZ}" ]]; then
  CMD+=(--base-npz "${BASE_NPZ}")
fi
if [[ -n "${COMP_HIDDEN}" ]]; then
  CMD+=(--comp-hidden "${COMP_HIDDEN}")
fi
if [[ -n "${COMP_TOPK}" ]]; then
  CMD+=(--comp-topk "${COMP_TOPK}")
fi
if [[ -n "${COMP_MARGIN_QUANTILE}" ]]; then
  CMD+=(--comp-margin-quantile "${COMP_MARGIN_QUANTILE}")
fi
if [[ -n "${COMP_MARGIN_TEMPERATURE}" ]]; then
  CMD+=(--comp-margin-temperature "${COMP_MARGIN_TEMPERATURE}")
fi
if [[ -n "${COMP_DELTA_SCALE}" ]]; then
  CMD+=(--comp-delta-scale "${COMP_DELTA_SCALE}")
fi
if [[ -n "${DUEL_MARGIN}" ]]; then
  CMD+=(--duel-margin "${DUEL_MARGIN}")
fi
if [[ -n "${LOSS_DUEL_WEIGHT}" ]]; then
  CMD+=(--loss-duel-weight "${LOSS_DUEL_WEIGHT}")
fi
if [[ "${DISABLE_SET_ENCODER}" == "1" ]]; then
  CMD+=(--disable-set-encoder)
fi
if [[ "${DISABLE_BACKGROUND}" == "1" ]]; then
  CMD+=(--disable-background)
fi
if [[ "${DISABLE_HIST_GATE}" == "1" ]]; then
  CMD+=(--disable-hist-gate)
fi
if [[ "${DISABLE_OOD_GATE}" == "1" ]]; then
  CMD+=(--disable-ood-gate)
fi

"${CMD[@]}"

RECORD_EXTRAS=(
  "epochs=${EPOCHS}"
  "batch_groups=${BATCH_GROUPS}"
  "lr=${LR}"
  "hist_hidden=${HIST_HIDDEN}"
  "pair_hidden=${PAIR_HIDDEN}"
  "anchor_alpha=${ANCHOR_ALPHA}"
  "delta_scale=${DELTA_SCALE}"
  "haca_version=${HACA_VERSION}"
  "min_history=${MIN_HISTORY}"
  "max_history=${MAX_HISTORY}"
  "disable_set_encoder=${DISABLE_SET_ENCODER}"
  "disable_background=${DISABLE_BACKGROUND}"
  "disable_hist_gate=${DISABLE_HIST_GATE}"
  "disable_ood_gate=${DISABLE_OOD_GATE}"
  "loss_shift_weight=${LOSS_SHIFT_WEIGHT}"
  "allow_empty_val=${ALLOW_EMPTY_VAL}"
)
if [[ -n "${BASE_NPZ}" ]]; then
  RECORD_EXTRAS+=("base_npz=${BASE_NPZ}")
fi
if [[ -n "${COMP_TOPK}" ]]; then
  RECORD_EXTRAS+=("comp_topk=${COMP_TOPK}")
fi
if [[ -n "${COMP_MARGIN_QUANTILE}" ]]; then
  RECORD_EXTRAS+=("comp_margin_quantile=${COMP_MARGIN_QUANTILE}")
fi
if [[ -n "${COMP_MARGIN_TEMPERATURE}" ]]; then
  RECORD_EXTRAS+=("comp_margin_temperature=${COMP_MARGIN_TEMPERATURE}")
fi
if [[ -n "${COMP_DELTA_SCALE}" ]]; then
  RECORD_EXTRAS+=("comp_delta_scale=${COMP_DELTA_SCALE}")
fi
if [[ -n "${DUEL_MARGIN}" ]]; then
  RECORD_EXTRAS+=("duel_margin=${DUEL_MARGIN}")
fi
if [[ -n "${LOSS_DUEL_WEIGHT}" ]]; then
  RECORD_EXTRAS+=("loss_duel_weight=${LOSS_DUEL_WEIGHT}")
fi

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/append_experiment_record.py" \
  --csv "${REGISTRY_CSV}" \
  --kind train \
  --script "${SCRIPT_NAME}" \
  --dataset MOT17 \
  --split train_val_split \
  --tracker-family BoT-SORT \
  --variant "${VARIANT_NAME}" \
  --run-root "${OUT_ROOT}" \
  --checkpoint "${OUT_NPZ}" \
  --log-path "${LOG_PATH}" \
  --extra "${RECORD_EXTRAS[@]}"

update_plan_status completed
echo "[done] $(date '+%F %T %z')"
echo "[checkpoint] ${OUT_NPZ}"
