#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-outputs/external_det}"
DOWNLOAD_DIR="${ROOT_DIR}/_downloads"

mkdir -p "${DOWNLOAD_DIR}"
cd "${DOWNLOAD_DIR}"

declare -A URLS=(
  [sw_yolox_mot20.zip]="https://motchallenge.net/download_results.php?shakey=f854d506dfec2edd7ee3e2d01a44e724d7460c7e&name=SW_YoloX_det&chl=14"
  [sw_yolox_mot17.zip]="https://motchallenge.net/download_results.php?shakey=e1773360e89521a4a15f031d28727d02c23d1b8e&name=SW_YoloX_det&chl=9"
  [sgt_mot20.zip]="https://motchallenge.net/download_results.php?shakey=143bb6f71766c7c1232ef5ffb89fd304a26139d1&name=SGT_det&chl=14"
  [sgt_mot17.zip]="https://motchallenge.net/download_results.php?shakey=30af6ec44b1b78fd1f065ed675fda12a9b04474d&name=SGT_det&chl=9"
)

FILES=(
  sw_yolox_mot20.zip
  sw_yolox_mot17.zip
  sgt_mot20.zip
  sgt_mot17.zip
)

for file in "${FILES[@]}"; do
  url="${URLS[$file]}"
  echo "[`date '+%F %T'`] Downloading ${file}"
  wget -c --tries=20 --read-timeout=30 --timeout=30 -O "${file}" "${url}"
  echo "[`date '+%F %T'`] Downloaded ${file}"
done

echo "[`date '+%F %T'`] All downloads completed in ${DOWNLOAD_DIR}"
