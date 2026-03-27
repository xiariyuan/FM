#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/gemini/code/FMtrack-main/FM-Track"
BOT_ROOT="${REPO_ROOT}/external/BoT-SORT-main"
DATA_ROOT="${DATA_ROOT:-/gemini/code/datasets}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/outputs/ltra_learned_mot17_overnight_${TS}}"
RUN_ROOT="$(realpath -m "${RUN_ROOT}")"
LOG_PATH="${RUN_ROOT}/overnight.log"

MOT17_SEQ_IDS=(2 4 5 9 10 11 13)
MOT17_TRAIN_SEQS=(MOT17-02-FRCNN MOT17-04-FRCNN MOT17-05-FRCNN MOT17-09-FRCNN)
MOT17_VAL_SEQS=(MOT17-10-FRCNN MOT17-11-FRCNN MOT17-13-FRCNN)

MOT17_REID_CFG="fast_reid/configs/MOT17/sbs_S50.yml"
MOT17_REID_WTS="pretrained/mot17_sbs_S50.pth"
MOT17_EXP="./yolox/exps/example/mot/yolox_x_mix_det.py"
MOT17_CKPT="./pretrained/bytetrack_x_mot17.pth.tar"

MOT20_REID_CFG="fast_reid/configs/MOT20/sbs_S50.yml"
MOT20_REID_WTS="pretrained/mot20_sbs_S50.pth"
MOT20_EXP="./yolox/exps/example/mot/yolox_x_mix_mot20_ch.py"
MOT20_CKPT="./pretrained/bytetrack_x_mot20.pth.tar"

BASE_EXP="mot17_frcnn_base_${TS}"
HEUR_EXP="mot17_frcnn_heuristic_${TS}"
LEARNED_EXP="mot17_frcnn_learned_${TS}"
MOT20_LEARNED_EXP="mot20_learned_${TS}"

BASE_RESULTS="${BOT_ROOT}/YOLOX_outputs/${BASE_EXP}/track_results"
HEUR_RESULTS="${BOT_ROOT}/YOLOX_outputs/${HEUR_EXP}/track_results"
LEARNED_RESULTS="${BOT_ROOT}/YOLOX_outputs/${LEARNED_EXP}/track_results"
MOT20_LEARNED_RESULTS="${BOT_ROOT}/YOLOX_outputs/${MOT20_LEARNED_EXP}/track_results"

HEUR_LOG_DIR="${RUN_ROOT}/pair_logs/mot17_heuristic"
LEARNED_LOG_DIR="${RUN_ROOT}/pair_logs/mot17_learned"
MOT20_LEARNED_LOG_DIR="${RUN_ROOT}/pair_logs/mot20_learned"
CALIB_DIR="${RUN_ROOT}/calibrators"
CALIB_NPZ="${CALIB_DIR}/mot17_primary_alpha_r_${TS}.npz"

mkdir -p "${RUN_ROOT}" "${CALIB_DIR}"
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "[start] $(date '+%F %T %z')"
echo "[run_root] ${RUN_ROOT}"

combine_pair_logs() {
  local src_dir="$1"
  local dst_csv="$2"
  shopt -s nullglob
  local files=("${src_dir}"/*_pairs.csv)
  shopt -u nullglob
  if [[ "${#files[@]}" -eq 0 ]]; then
    echo "No pair logs found in ${src_dir}" >&2
    return 1
  fi
  mkdir -p "$(dirname "${dst_csv}")"
  head -n 1 "${files[0]}" > "${dst_csv}"
  for f in "${files[@]}"; do
    tail -n +2 "${f}" >> "${dst_csv}"
  done
  echo "[combined] ${dst_csv}"
}

run_track() {
  local benchmark="$1"
  local experiment_name="$2"
  local analysis_dir="$3"
  shift 3
  (
    cd "${BOT_ROOT}"
    cmd=("${PYTHON_BIN}" -u tools/track.py "${DATA_ROOT}/${benchmark}" --benchmark "${benchmark}" --eval val "$@")
    if [[ -n "${analysis_dir}" ]]; then
      mkdir -p "${analysis_dir}"
      cmd+=(--laplace-analysis-dir "${analysis_dir}")
    fi
    cmd+=(--experiment-name "${experiment_name}")
    echo "[track] ${cmd[*]}"
    "${cmd[@]}"
  )
}

eval_results() {
  local dataset="$1"
  local results_dir="$2"
  local tracker_name="$3"
  local work_dir="$4"
  echo "[eval] ${dataset} ${tracker_name}"
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/eval_botsort_halfval_trackeval.py" \
    --dataset "${dataset}" \
    --data-root "${DATA_ROOT}" \
    --results-dir "${results_dir}" \
    --tracker-name "${tracker_name}" \
    --work-dir "${work_dir}"
}

echo "[step] MOT17 base run"
run_track MOT17 "${BASE_EXP}" "" \
  --seq-ids "${MOT17_SEQ_IDS[@]}" \
  --mot17-detector-exts FRCNN \
  -f "${MOT17_EXP}" \
  -c "${MOT17_CKPT}" \
  --with-reid \
  --fast-reid-config "${MOT17_REID_CFG}" \
  --fast-reid-weights "${MOT17_REID_WTS}"

echo "[step] MOT17 heuristic+log run"
run_track MOT17 "${HEUR_EXP}" "${HEUR_LOG_DIR}" \
  --seq-ids "${MOT17_SEQ_IDS[@]}" \
  --mot17-detector-exts FRCNN \
  -f "${MOT17_EXP}" \
  -c "${MOT17_CKPT}" \
  --with-reid \
  --fast-reid-config "${MOT17_REID_CFG}" \
  --fast-reid-weights "${MOT17_REID_WTS}" \
  --laplace-assoc \
  --laplace-primary-only \
  --laplace-decay-scales 1 2 4 \
  --laplace-min-history 3 \
  --laplace-proto-mode multi

HEUR_COMBINED="${HEUR_LOG_DIR}/_combined/all_pairs.csv"
combine_pair_logs "${HEUR_LOG_DIR}" "${HEUR_COMBINED}"
echo "[check] heuristic pair-log header"
head -n 1 "${HEUR_COMBINED}"
for required_col in assoc_stage history_len amb_spa amb_lap amb_mot; do
  if ! head -n 1 "${HEUR_COMBINED}" | grep -q "${required_col}"; then
    echo "Missing required column in fresh pair logs: ${required_col}" >&2
    exit 1
  fi
done

echo "[step] summarize heuristic logs"
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/summarize_laplace_pair_logs.py" \
  "${HEUR_COMBINED}" \
  --out-dir "${HEUR_LOG_DIR}/_combined"

shopt -s nullglob
PAIR_FILES=("${HEUR_LOG_DIR}"/MOT17-*-FRCNN_pairs.csv)
shopt -u nullglob
if [[ "${#PAIR_FILES[@]}" -eq 0 ]]; then
  echo "No MOT17 FRCNN pair logs found under ${HEUR_LOG_DIR}" >&2
  exit 1
fi

echo "[step] train alpha/r calibrator"
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/train_ltra_calibrator_from_pairs.py" \
  --pair-csv "${PAIR_FILES[@]}" \
  --out-npz "${CALIB_NPZ}" \
  --assoc-stage primary \
  --train-seqs "${MOT17_TRAIN_SEQS[@]}" \
  --val-seqs "${MOT17_VAL_SEQS[@]}" \
  --device cuda \
  --epochs 12 \
  --batch-size 512 \
  --hidden-dim 16 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --rank-margin 0.05 \
  --trust-margin 0.03 \
  --loss-bg-weight 0.25 \
  --loss-rank-weight 0.25 \
  --loss-trust-weight 0.10

echo "[step] MOT17 learned+log run"
run_track MOT17 "${LEARNED_EXP}" "${LEARNED_LOG_DIR}" \
  --seq-ids "${MOT17_SEQ_IDS[@]}" \
  --mot17-detector-exts FRCNN \
  -f "${MOT17_EXP}" \
  -c "${MOT17_CKPT}" \
  --with-reid \
  --fast-reid-config "${MOT17_REID_CFG}" \
  --fast-reid-weights "${MOT17_REID_WTS}" \
  --laplace-assoc \
  --laplace-primary-only \
  --laplace-decay-scales 1 2 4 \
  --laplace-min-history 3 \
  --laplace-proto-mode multi \
  --laplace-calibrator "${CALIB_NPZ}"

LEARNED_COMBINED="${LEARNED_LOG_DIR}/_combined/all_pairs.csv"
combine_pair_logs "${LEARNED_LOG_DIR}" "${LEARNED_COMBINED}"

echo "[step] summarize learned logs"
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/summarize_laplace_pair_logs.py" \
  "${LEARNED_COMBINED}" \
  --out-dir "${LEARNED_LOG_DIR}/_combined"

echo "[step] evaluate MOT17 base/heuristic/learned"
eval_results MOT17 "${BASE_RESULTS}" "mot17_frcnn_base_${TS}" "${RUN_ROOT}/eval/mot17_base"
eval_results MOT17 "${HEUR_RESULTS}" "mot17_frcnn_heuristic_${TS}" "${RUN_ROOT}/eval/mot17_heuristic"
eval_results MOT17 "${LEARNED_RESULTS}" "mot17_frcnn_learned_${TS}" "${RUN_ROOT}/eval/mot17_learned"

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/collect_trackeval_metrics.py" \
  "${RUN_ROOT}/eval/mot17_base/eval/mot17_frcnn_base_${TS}" \
  "${RUN_ROOT}/eval/mot17_heuristic/eval/mot17_frcnn_heuristic_${TS}" \
  "${RUN_ROOT}/eval/mot17_learned/eval/mot17_frcnn_learned_${TS}" \
  --csv "${RUN_ROOT}/eval/mot17_summary.csv" | tee "${RUN_ROOT}/eval/mot17_summary.txt"

echo "[step] MOT20 learned follow-up"
run_track MOT20 "${MOT20_LEARNED_EXP}" "${MOT20_LEARNED_LOG_DIR}" \
  -f "${MOT20_EXP}" \
  -c "${MOT20_CKPT}" \
  --with-reid \
  --fast-reid-config "${MOT20_REID_CFG}" \
  --fast-reid-weights "${MOT20_REID_WTS}" \
  --cmc-method file \
  --laplace-assoc \
  --laplace-primary-only \
  --laplace-decay-scales 1 2 4 \
  --laplace-min-history 3 \
  --laplace-proto-mode multi \
  --laplace-calibrator "${CALIB_NPZ}"

MOT20_LEARNED_COMBINED="${MOT20_LEARNED_LOG_DIR}/_combined/all_pairs.csv"
combine_pair_logs "${MOT20_LEARNED_LOG_DIR}" "${MOT20_LEARNED_COMBINED}"
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/summarize_laplace_pair_logs.py" \
  "${MOT20_LEARNED_COMBINED}" \
  --out-dir "${MOT20_LEARNED_LOG_DIR}/_combined"
eval_results MOT20 "${MOT20_LEARNED_RESULTS}" "mot20_learned_${TS}" "${RUN_ROOT}/eval/mot20_learned"

echo "[done] $(date '+%F %T %z')"
echo "[artifacts] calibrator=${CALIB_NPZ}"
echo "[artifacts] mot17_summary=${RUN_ROOT}/eval/mot17_summary.csv"
echo "[artifacts] mot20_eval=${RUN_ROOT}/eval/mot20_learned"
