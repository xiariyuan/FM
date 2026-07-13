#!/usr/bin/env bash
# Run improved AFLink v2 on both train and test sets.
# Compares: A23 original (thr0.20) vs v2 (gap-aware + competition filter)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
A23_ROOT="${REPO_ROOT}/outputs/spot_runtime_gate_20260628/A23_appearance_aflink"
PYTHON_BIN="${PYTHON_BIN:-python}"

echo "=== A24 AFLink v2 Pipeline ==="
echo "Start: $(date -Iseconds)"

# --- Step 1: Evaluate v2 on TRAIN (using OOF predictions) ---
# The OOF predictions are in A23_03 (out-of-fold predictions from HGB model)
TRAIN_PAIRS="${A23_ROOT}/A23_03_aflink_reid_hgb/oof_predictions.csv"
TRAIN_TRACKS="${A23_ROOT}/A23_00_control_gap30_all_train/track_results"
TRAIN_OUT="${REPO_ROOT}/outputs/spot_runtime_gate_20260628/A24_aflink_v2/train_oof"
TRAIN_EVAL="${REPO_ROOT}/outputs/spot_runtime_gate_20260628/A24_aflink_v2/train_oof_eval"

echo "--- Step 1: Run v2 linker on train OOF ---"
mkdir -p "${TRAIN_OUT}"
cd "${REPO_ROOT}"
"${PYTHON_BIN}" scripts/postprocess/link_tracks_by_score_v2.py \
  --input-dir "${TRAIN_TRACKS}" \
  --scores "${TRAIN_PAIRS}" \
  --output-dir "${TRAIN_OUT}/linked_results" \
  --base-thr 0.20 \
  --max-gap 40 \
  --min-len 5 \
  --min-appearance 0.25 \
  --competition-margin 0.03

echo "--- Step 1b: Evaluate v2 on train ---"
GT_ROOT="${GT_ROOT:-/gemini/code/datasets/MOT20/train}"
mkdir -p "${TRAIN_EVAL}"
"${PYTHON_BIN}" scripts/eval_motstyle_trackeval.py \
  --benchmark-name MOT20 \
  --split-to-eval train \
  --gt-root "${GT_ROOT}" \
  --results-dir "${TRAIN_OUT}/linked_results" \
  --tracker-name A24_aflink_v2_train_oof \
  --work-dir "${TRAIN_EVAL}" \
  --keep-workdir \
  --seqs MOT20-01 MOT20-02 MOT20-03 MOT20-05 \
  2>&1 | tail -5

# --- Step 2: Run v2 on TEST ---
# Use test pair predictions scored by the HGB model
TEST_PAIRS="${A23_ROOT}/A23_test_pair_dataset_with_reid/aflink_pair_predictions.csv"
A18_TEST="${REPO_ROOT}/outputs/spot_runtime_gate_20260628/A18_test_submission_control_interp_gap30/track_results"
TEST_OUT="${REPO_ROOT}/outputs/spot_runtime_gate_20260628/A24_aflink_v2/test"

echo "--- Step 2: Run v2 linker on test ---"
mkdir -p "${TEST_OUT}"
"${PYTHON_BIN}" scripts/postprocess/link_tracks_by_score_v2.py \
  --input-dir "${A18_TEST}" \
  --scores "${TEST_PAIRS}" \
  --output-dir "${TEST_OUT}/linked_results" \
  --base-thr 0.20 \
  --max-gap 40 \
  --min-len 5 \
  --min-appearance 0.25 \
  --competition-margin 0.03

# --- Step 3: Package test submission ---
SUBMISSION_DIR="${REPO_ROOT}/outputs/spot_runtime_gate_20260628/A24_aflink_v2/test_submission"
mkdir -p "${SUBMISSION_DIR}"
cp "${TEST_OUT}/linked_results"/MOT20-*.txt "${SUBMISSION_DIR}/"

cd "${REPO_ROOT}"
"${PYTHON_BIN}" -c "
import zipfile, os
out = 'outputs/spot_runtime_gate_20260628/A24_aflink_v2/test_submission/MOT20_A24_aflink_v2_thr020_gap40_submission.zip'
with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
    for f in sorted(os.listdir('outputs/spot_runtime_gate_20260628/A24_aflink_v2/test_submission')):
        if f.endswith('.txt'):
            z.write(f'outputs/spot_runtime_gate_20260628/A24_aflink_v2/test_submission/{f}', f)
            print(f'  added {f}')
print(f'Created: {out}')
"

# --- Step 4: Write summary ---
echo "--- Step 3: Write summary ---"
cat > "${REPO_ROOT}/outputs/spot_runtime_gate_20260628/A24_aflink_v2/summary.csv" << 'CSV'
status,phase,base_tracks,linker,base_thr,max_gap,min_len,min_appearance,competition_margin,test_links,notes
completed,aflink_v2,A18_control_interp_gap30,link_v2_gap_aware,0.20,40,5,0.25,0.03,TBD,"Improved linker with gap-aware thresholds and competition filtering"
CSV

echo "=== Done: $(date -Iseconds) ==="
echo ""
echo "Test submission:"
echo "  outputs/spot_runtime_gate_20260628/A24_aflink_v2/test_submission/MOT20_A24_aflink_v2_thr020_gap40_submission.zip"
echo ""
echo "Train evaluation:"
echo "  ${TRAIN_EVAL}/eval/A24_aflink_v2_train_oof/pedestrian_summary.txt"
