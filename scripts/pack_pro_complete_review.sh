#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"
cd "${REPO_ROOT}"

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/outputs}"
ZIP_PATH="${OUT_DIR}/pro_complete_project_review_${TS}.zip"

mkdir -p "${OUT_DIR}"
rm -f "${ZIP_PATH}"

echo "[pack] creating comprehensive review archive"
echo "[pack] repo: ${REPO_ROOT}"
echo "[pack] out:  ${ZIP_PATH}"

# Phase 1: pack a broad whitelist of project code / configs / docs.
# This is intentionally comprehensive, but avoids scanning giant raw-output trees.
INCLUDE_TREES=(
  "README.md"
  "LICENSE"
  "requirements.txt"
  "requirements_dino.txt"
  "TrackEval"
  "analysis"
  "assets"
  "config"
  "configs"
  "data"
  "demo"
  "docs"
  "models"
  "paper"
  "scripts"
  "create_seqinfo.py"
  "diagnostic_test.py"
  "migrate_v1_to_v2.py"
  "runtime_option.py"
  "submit_and_evaluate.py"
  "submit_bytetrack.py"
  "submit_public.py"
  "sweep_thresholds.py"
  "test_val_split.py"
  "train.py"
  "train_bytetrack.py"
  "run_mot20_sweep.sh"
  "run_mot20_val.py"
  "run_mot20_val.sh"
  "external/BoT-SORT-main/LICENSE"
  "external/BoT-SORT-main/README.md"
  "external/BoT-SORT-main/requirements.txt"
  "external/BoT-SORT-main/setup.cfg"
  "external/BoT-SORT-main/setup.py"
  "external/BoT-SORT-main/VideoCameraCorrection"
  "external/BoT-SORT-main/assets"
  "external/BoT-SORT-main/fast_reid"
  "external/BoT-SORT-main/tools"
  "external/BoT-SORT-main/tracker"
  "external/BoT-SORT-main/yolov7"
  "external/BoT-SORT-main/yolox"
  "external/StrongSORT-master/LICENSE"
  "external/StrongSORT-master/README.md"
  "external/StrongSORT-master/GSI.py"
  "external/StrongSORT-master/opts.py"
  "external/StrongSORT-master/strong_sort.py"
  "external/StrongSORT-master/MOT17_ECC_val.json"
  "external/StrongSORT-master/MOT17_ECC_test.json"
  "external/StrongSORT-master/AFLink"
  "external/StrongSORT-master/application_util"
  "external/StrongSORT-master/assets"
  "external/StrongSORT-master/deep_sort"
  "external/StrongSORT-master/others"
  "external/StrongSORT-master/tools"
)

for path in "${INCLUDE_TREES[@]}"; do
  if [[ -e "${path}" ]]; then
    zip -qr "${ZIP_PATH}" "${path}" \
      -x "__pycache__/*" \
      -x "*/__pycache__/*" \
      -x "*/*/__pycache__/*" \
      -x "*/*/*/__pycache__/*" \
      -x "*.pyc" \
      -x "*.pyo" \
      -x "*.so" \
      -x "*.whl" \
      -x "*.pth" \
      -x "*.pt" \
      -x "*.ckpt" \
      -x "*.zip" \
      -x "*.tar.gz" \
      -x "*.pdf" \
      -x "external/BoT-SORT-main/YOLOX_outputs/*" \
      -x "external/BoT-SORT-main/pretrained/*" \
      -x "external/StrongSORT-master/MOT17_val_YOLOX+BoT/*" \
      -x "external/StrongSORT-master/MOT17_test_YOLOX+BoT/*" \
      -x "external/StrongSORT-master/outputs/*"
  else
    echo "[warn] missing source tree: ${path}"
  fi
done

# Phase 2: append the key current outputs and review prompts that are needed
# to understand the current project state.
# Do not append any weight / checkpoint files.
INCLUDE_PATHS=(
  "outputs/pro_design_prompt_20260313.md"
  "outputs/pro_review_prompt_20260313.md"
  "outputs/experiment_plan.csv"
  "outputs/experiment_registry.csv"
  "outputs/lpb_ltra_eval_full_20260311_204037/eval/mot17_summary.csv"
  "outputs/lpb_ltra_eval_mot20_20260312_183923/eval/mot20_summary.csv"
  "outputs/lpb_ltra_eval_mot20_20260312_183923/eval.log"
  "outputs/lpb_ltra_eval_mot20_learned_oldbest_nopole_20260312_213634/eval/mot20_summary.csv"
  "outputs/lpb_ltra_eval_mot20_learned_oldbest_nopole_20260312_213634/eval.log"
  "outputs/lpb_ltra_eval_mot20_learned_allhalf_nopole_20260312_215353/eval.log"
  "outputs/strongsort_lpb_ltra/MOT17_val_oldbest/summary.csv"
  "outputs/strongsort_lpb_ltra/MOT17_val_oldbest_debug_smoke/pair_logs/_combined/learned_r_calibration.csv"
  "outputs/strongsort_lpb_ltra/MOT17_val_oldbest_debug_full_nopole_20260312_213634/run.log"
  "outputs/queued_runs/queue_nopole_after_mot20_matrix_20260313_105929.log"
  "outputs/queued_runs/queue_allhalf_nopole_after_oldbest_nopole_20260312_215353.log"
  "outputs/queued_runs/queue_allhalf_followups_after_nopole_20260313_105929.log"
)

for path in "${INCLUDE_PATHS[@]}"; do
  if [[ -e "${path}" ]]; then
    zip -qr "${ZIP_PATH}" "${path}"
  else
    echo "[warn] missing include: ${path}"
  fi
done

echo "[pack] done"
echo "[pack] size: $(du -h "${ZIP_PATH}" | cut -f1)"
echo "[pack] file: ${ZIP_PATH}"
