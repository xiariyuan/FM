#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"

export HACA_LABEL="${HACA_LABEL:-haca_v3}"
export HACA_VERSION="${HACA_VERSION:-haca_v3}"
export SCRIPT_NAME="${SCRIPT_NAME:-scripts/train_haca_v3_mot17.sh}"
export VARIANT_NAME="${VARIANT_NAME:-haca_v3_train}"
export TRAIN_SCRIPT_PATH="${TRAIN_SCRIPT_PATH:-${REPO_ROOT}/scripts/train_haca_v3_from_gt_tracks.py}"
export BASE_NPZ="${BASE_NPZ:-${REPO_ROOT}/outputs/haca_v2_nobg_train_20260314_2025/mot17_haca_v2_nobg_20260314_202647.npz}"
export EPOCHS="${EPOCHS:-12}"
export DISABLE_SET_ENCODER="${DISABLE_SET_ENCODER:-1}"
export DISABLE_BACKGROUND="${DISABLE_BACKGROUND:-1}"
export LOSS_BG_WEIGHT="${LOSS_BG_WEIGHT:-0.0}"
export LOSS_MARGIN_WEIGHT="${LOSS_MARGIN_WEIGHT:-0.0}"
export LOSS_SAFE_WEIGHT="${LOSS_SAFE_WEIGHT:-0.20}"
export LOSS_RES_WEIGHT="${LOSS_RES_WEIGHT:-0.0}"
export LOSS_SHIFT_WEIGHT="${LOSS_SHIFT_WEIGHT:-0.0}"
export COMP_HIDDEN="${COMP_HIDDEN:-64}"
export COMP_TOPK="${COMP_TOPK:-3}"
export COMP_MARGIN_QUANTILE="${COMP_MARGIN_QUANTILE:-0.35}"
export COMP_MARGIN_TEMPERATURE="${COMP_MARGIN_TEMPERATURE:-0.03}"
export COMP_DELTA_SCALE="${COMP_DELTA_SCALE:-1.0}"
export DUEL_MARGIN="${DUEL_MARGIN:-0.20}"
export LOSS_DUEL_WEIGHT="${LOSS_DUEL_WEIGHT:-0.50}"

exec "${REPO_ROOT}/scripts/train_haca_v1_mot17.sh"
