#!/usr/bin/env bash
set -euo pipefail

# Extract DanceTrack zips (val/train/test) into a unified folder layout:
#   ${DATASET_BASE}/extracted/{train,val,test}/dancetrackXXXX/...
#
# Download zips are expected under:
#   ${DATASET_BASE}/_downloads/{val.zip,train1.zip,train2.zip,test1.zip,test2.zip}

DATASET_BASE="${DATASET_BASE:-/gemini/code/datasets/DanceTrack}"
DOWNLOAD_DIR="${DATASET_BASE}/_downloads"
OUT_DIR="${DATASET_BASE}/extracted"

mkdir -p "${OUT_DIR}"

require_zip() {
  local name="$1"
  local path="${DOWNLOAD_DIR}/${name}"
  if [[ ! -f "${path}" ]]; then
    echo "[ERR] missing zip: ${path}" >&2
    exit 2
  fi
}

extract_val() {
  local marker="${OUT_DIR}/val/.extracted.ok"
  if [[ -f "${marker}" ]]; then
    echo "[skip] val already extracted: ${marker}"
    return 0
  fi
  echo "[run] extract val.zip -> ${OUT_DIR}"
  rm -rf "${OUT_DIR}/val"
  bsdtar -xf "${DOWNLOAD_DIR}/val.zip" -C "${OUT_DIR}"
  if [[ ! -d "${OUT_DIR}/val" ]]; then
    echo "[ERR] expected ${OUT_DIR}/val after extracting val.zip" >&2
    exit 3
  fi
  touch "${marker}"
  echo "[ok] val extracted"
}

extract_merge_split() {
  local split="$1"  # train | test
  local z1="$2"
  local z2="$3"
  local prefix1="$4"
  local prefix2="$5"

  local marker="${OUT_DIR}/${split}/.extracted.ok"
  if [[ -f "${marker}" ]]; then
    echo "[skip] ${split} already extracted: ${marker}"
    return 0
  fi

  echo "[run] extract ${z1}, ${z2} -> ${OUT_DIR}/${split}"
  local tmp="${OUT_DIR}/._tmp_${split}_$$"
  rm -rf "${tmp}"
  mkdir -p "${tmp}"

  bsdtar -xf "${DOWNLOAD_DIR}/${z1}" -C "${tmp}"
  bsdtar -xf "${DOWNLOAD_DIR}/${z2}" -C "${tmp}"

  rm -rf "${OUT_DIR:?}/${split}"
  mkdir -p "${OUT_DIR}/${split}"

  shopt -s nullglob
  for d in "${tmp}/${prefix1}"/dancetrack* "${tmp}/${prefix2}"/dancetrack*; do
    if [[ -d "${d}" ]]; then
      mv "${d}" "${OUT_DIR}/${split}/"
    fi
  done
  shopt -u nullglob

  rm -rf "${tmp}"
  touch "${marker}"
  echo "[ok] ${split} extracted"
}

require_zip "val.zip"
require_zip "train1.zip"
require_zip "train2.zip"
require_zip "test1.zip"
require_zip "test2.zip"

extract_val
extract_merge_split "train" "train1.zip" "train2.zip" "train1" "train2"
extract_merge_split "test" "test1.zip" "test2.zip" "test1" "test2"

echo "[done] DanceTrack extracted to: ${OUT_DIR}"

