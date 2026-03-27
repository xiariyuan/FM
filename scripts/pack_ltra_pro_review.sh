#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/gemini/code/FMtrack-main/FM-Track"
cd "${REPO_ROOT}"

TS="$(date +%Y%m%d_%H%M%S)"
PACK_ROOT="${REPO_ROOT}/pack"
PACK_DIR="${PACK_ROOT}/LTRA_PRO_REVIEW_${TS}"
ZIP_PATH="${PACK_ROOT}/LTRA_PRO_REVIEW_${TS}.zip"

mkdir -p "${PACK_DIR}"

copy_file() {
  local src="$1"
  local dst_dir="$2"
  if [[ -f "${src}" ]]; then
    mkdir -p "${dst_dir}"
    cp -f "${src}" "${dst_dir}/"
  fi
}

copy_file_as() {
  local src="$1"
  local dst="$2"
  if [[ -f "${src}" ]]; then
    mkdir -p "$(dirname "${dst}")"
    cp -f "${src}" "${dst}"
  fi
}

copy_tree() {
  local src="$1"
  local dst="$2"
  if [[ -d "${src}" ]]; then
    mkdir -p "$(dirname "${dst}")"
    rm -rf "${dst}"
    cp -a "${src}" "${dst}"
    find "${dst}" -type d -name "__pycache__" -prune -exec rm -rf {} +
    find "${dst}" -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete
  fi
}

echo "[pack] create: ${PACK_DIR}"

cat > "${PACK_DIR}/README_FIRST.md" <<'MD'
# LTRA Pro Review Pack

This pack is built from the current local workspace on 2026-03-11.

The mainline is now:
- **BoT-SORT + learned LTRA**
- **primary association only**
- **Laplace-inspired temporal signatures + tiny alpha/r trust calibrator**

This pack is intentionally centered on the current paper line and excludes the deprecated FM / frequency / Mamba mainline except where old scripts still remain in the repository.

## Start here
1. Read `PRO_REVIEW_CONTEXT.md`
2. Read `PROJECT_STATUS.md`
3. Paste `PROMPT_TO_PASTE.md` into GPT Pro web
4. If you want code first, inspect:
   - `external/BoT-SORT-main/tracker/laplace_assoc.py`
   - `external/BoT-SORT-main/tracker/laplace_calibrator.py`
   - `external/BoT-SORT-main/tracker/bot_sort.py`
   - `external/BoT-SORT-main/tools/track.py`
   - `scripts/train_ltra_calibrator_from_pairs.py`

## Important caveats
- The **code path** is ahead of the **paper tables**. `paper/sections/experiments.tex` still contains stale mixed numbers and should not be treated as source-of-truth.
- The learned alpha/r runtime and grouped training pipeline are implemented, but the **first full learned experiment after the cleanup has not been run yet**.
- GT pseudo-track training data generation is **not yet implemented**. Current learned training still consumes tracker pair logs.

## Evidence included
- Current runtime code and training code
- Paper draft and experiment checklist
- MOT17/MOT20 pair logs and summary CSVs
- DanceTrack and StrongSORT result snapshots
- Current packing script so future packs can be reproduced
MD

cat > "${PACK_DIR}/PRO_REVIEW_CONTEXT.md" <<'MD'
# PRO Review Context

## Goal
Prepare a NeurIPS 2026 paper around a single narrow claim:

**When should a MOT tracker trust appearance for association?**

The carrier is a strong tracking-by-detection baseline:
- **BoT-SORT** as the main carrier
- **StrongSORT** as plug-in transfer evidence

## Mainline decision already made
The project has pivoted away from the old FM / frequency / Mamba direction.

The current paper line is:
- **BoT-SORT + LTRA**
- **LTRA = Laplace-inspired temporal signatures + learned trust calibrator**
- **Only the primary association stage is modified**
- Detector / ReID / Kalman / CMC / Hungarian remain standard

Training-free heuristic LTRA is still present, but only as:
- lower-bound ablation
- heuristic teacher / debugging reference
- evidence for why trainable calibration is needed

## Current learned design
For each detection-track pair:
- `S_spa`: instantaneous spatial appearance similarity
- `S_lap`: Laplace-inspired temporal signature similarity
- `S_mot`: motion similarity

A tiny calibrator predicts:
- `alpha in [0,1]`: how to mix `S_spa` and `S_lap`
- `r in [0,1]`: how much to trust appearance versus motion

Formula:
- `S_app = (1 - alpha) * S_spa + alpha * S_lap`
- `S_fuse = r * S_app + (1 - r) * S_mot`
- `cost = 1 - S_fuse`

## Implemented code paths
### Runtime
- `external/BoT-SORT-main/tracker/laplace_assoc.py`
- `external/BoT-SORT-main/tracker/laplace_calibrator.py`
- `external/BoT-SORT-main/tracker/bot_sort.py`
- `external/BoT-SORT-main/tools/track.py`

### Analysis
- `external/BoT-SORT-main/tracker/laplace_analysis.py`
- `scripts/summarize_laplace_pair_logs.py`

### Training
- `scripts/train_ltra_calibrator_from_pairs.py`

## What is already implemented
1. `--laplace-primary-only`
   - Mainline now isolates primary association only.
2. `--laplace-calibrator <npz>`
   - Learned runtime path is wired into BoT-SORT.
3. Alpha/r learned fusion
   - `laplace_assoc.py` builds pair features and calls the calibrator.
4. Grouped candidate-set training
   - The trainer no longer uses simple pairwise weighted BCE as the main loss.
   - It now uses grouped CE + background loss + ranking loss + trust auxiliary loss.
5. Richer pair features
   - ambiguity margins
   - history length normalization
   - gap
   - agreement / stability / coherence
6. Pair-log improvements
   - `assoc_stage`
   - ambiguity fields
   - `prod_sim`
   - current-frame IoU-based `track_gt_id`

## Current local evidence snapshot
### Same-base validation
- MOT17 val summary:
  - base: HOTA 78.446 / AssA 76.952 / IDF1 86.113 / IDSW 555
  - heuristic full LTRA: HOTA 78.749 / AssA 77.554 / IDF1 86.435 / IDSW 537
  - source: `outputs/botsort_ltra_stage2/MOT17/summary.csv`

- MOT20 val summary:
  - base: HOTA 77.772 / AssA 75.011 / IDF1 89.414 / IDSW 768
  - meanhist: HOTA 78.027 / AssA 75.484 / IDF1 89.796 / IDSW 770
  - heuristic full LTRA: HOTA 77.875 / AssA 75.211 / IDF1 89.652 / IDSW 756
  - source: `outputs/botsort_ltra_stage3/MOT20/summary.csv`

- DanceTrack val:
  - base: HOTA 48.012 / AssA 31.947 / IDF1 48.859 / IDSW 2523
  - meanrel: HOTA 48.427 / AssA 32.522 / IDF1 48.891 / IDSW 2361
  - full ltra: HOTA 47.901 / AssA 31.850 / IDF1 48.377 / IDSW 2313
  - source: `outputs/dancetrack/overnight_summary.tsv`

- StrongSORT transfer:
  - base: HOTA 69.581 / AssA 73.370 / IDF1 82.275 / IDSW 233
  - +LTRA: HOTA 69.757 / AssA 73.662 / IDF1 82.818 / IDSW 203
  - source: `outputs/strongsort_ltra/MOT17_val/summary.csv`

### Official benchmark signals
These are manual records from official submissions and are included here for context:
- MOT17 private-detection test on 2026-03-10:
  - baseline: HOTA 59.96
  - LTRA multi: HOTA 59.92
- MOT20 official test:
  - last successful stable submission snapshot: HOTA 58.13

Interpretation:
- heuristic LTRA can help on local val and association-heavy regimes
- but current official MOT17 test signal is not yet positive
- this is why the project has moved to learned calibration as the mainline

## What is still missing
1. **First real learned-LTRA experiment after code cleanup**
   - not yet run
2. **GT pseudo-track builder**
   - not yet implemented
   - current learned training still uses tracker pair logs
3. **Paper/source-of-truth alignment**
   - `paper/sections/experiments.tex` still contains stale mixed numbers
4. **Clean locked evidence chain**
   - needs one full same-base learned run on MOT17 full-FRCNN val before more submissions

## What we want you to review
Please inspect the current implemented code and answer:

1. Is the current **alpha/r calibrator design** the right minimal learned mainline, or is there a cleaner variant?
2. Is the **grouped training protocol** clean enough for a paper, or is GT pseudo-track supervision mandatory before large-scale experiments?
3. Are there any remaining **structural code/design issues** that should be fixed before launching the first real learned run?
4. Is the current **paper positioning** credible:
   - "Laplace-inspired temporal signatures"
   - "when to trust appearance"
   - plug-in reliability calibration at association time
5. What are the top **reviewer attack points** now, and what is the cheapest defense evidence for each?

## Non-goals
Please do **not** push the design toward:
- detector retraining
- ReID backbone redesign
- end-to-end tracker training
- a broader frequency/Mamba story

We want the narrowest defensible paper.
MD

cat > "${PACK_DIR}/PROJECT_STATUS.md" <<'MD'
# Project Status

## Strategic state
- Mainline locked: **BoT-SORT + learned LTRA**
- Old FM / frequency / Mamba line is deprecated
- StrongSORT remains a transfer baseline, not a replacement carrier

## What changed recently
- Added `--laplace-primary-only`
- Wired learned calibrator into runtime via `--laplace-calibrator`
- Reworked calibrator training into grouped candidate-set learning
- Added ambiguity features and richer pair logging
- Clarified paper wording to `Laplace-inspired temporal signatures`

## Current strongest local signals
- MOT17 val: heuristic full LTRA is positive vs base
- MOT20 val: mean-history variant is stronger than full heuristic multi-scale
- DanceTrack val: meanrel is best, full multi-scale heuristic is not best
- StrongSORT transfer is positive

## Key interpretation
The current evidence suggests:
- the core idea is probably **trust calibration**
- the exact heuristic fusion is not stable enough
- a learned but tiny calibrator is a better paper line than heuristic-only LTRA

## Current risks
1. Official MOT17 test is still not positive.
2. Learned code exists, but the first cleaned learned run has not started yet.
3. Training still uses tracker pair logs rather than GT pseudo-tracks.
4. Paper tables are not yet cleaned enough to be submission-safe.

## Immediate next experiment after this review
1. Generate fresh `primary-only` pair logs on full MOT17 FRCNN val
2. Train alpha/r calibrator with grouped script
3. Evaluate `base / best heuristic / learned`
4. If positive, extend to MOT20 val and DanceTrack val

## What should not happen next
- no detector-centric work
- no blind sweeps
- no expansion back into old FM lines
- no paper claims based on proxy numbers alone
MD

cat > "${PACK_DIR}/FUTURE_PLAN.md" <<'MD'
# Immediate Plan

## Step 1
Run the first cleaned learned-LTRA experiment on full MOT17 FRCNN val:
- fresh primary-only pair logs
- grouped alpha/r training
- base vs heuristic vs learned

## Step 2
If MOT17 full val is positive:
- run MOT20 val
- run DanceTrack val
- re-check StrongSORT transfer if needed

## Step 3
Implement GT pseudo-track builder if review confirms it is necessary before serious paper claims.

## Step 4
Clean `paper/sections/experiments.tex` so every number comes from a local summary file instead of stale hand-written tables.

## Step 5
Only after the learned config is locked on val:
- resume official MOT17 / MOT20 submissions
- build ablation matrix
- freeze paper narrative
MD

cat > "${PACK_DIR}/PROMPT_TO_PASTE.md" <<'MD'
Please review this pack as a code-and-paper auditor, not as a brainstormer.

I need you to inspect the CURRENT IMPLEMENTED learned-LTRA pipeline for MOT, not the old heuristic-only version.

Mainline decision already made:
- carrier: BoT-SORT
- transfer baseline: StrongSORT
- contribution: when to trust appearance for association
- module: Laplace-inspired temporal signatures + tiny learned alpha/r trust calibrator
- insertion point: primary association only
- non-goals: detector retraining, ReID redesign, end-to-end tracker training, broader FM/Mamba/frequency story

What to inspect first:
1. `PRO_REVIEW_CONTEXT.md`
2. `PROJECT_STATUS.md`
3. Runtime code:
   - `external/BoT-SORT-main/tracker/laplace_assoc.py`
   - `external/BoT-SORT-main/tracker/laplace_calibrator.py`
   - `external/BoT-SORT-main/tracker/bot_sort.py`
   - `external/BoT-SORT-main/tools/track.py`
4. Training code:
   - `scripts/train_ltra_calibrator_from_pairs.py`
5. Pair-log analysis:
   - `external/BoT-SORT-main/tracker/laplace_analysis.py`
   - `scripts/summarize_laplace_pair_logs.py`
6. Current evidence:
   - `outputs/botsort_ltra_stage2/MOT17/summary.csv`
   - `outputs/botsort_ltra_stage3/MOT20/summary.csv`
   - `outputs/dancetrack/overnight_summary.tsv`
   - `outputs/strongsort_ltra/MOT17_val/summary.csv`

Questions I need answered:
1. Is the current alpha/r design the right minimal learned module, or should it be simplified/changed before running real experiments?
2. Is the grouped training protocol conceptually clean enough for a paper, or is a GT pseudo-track builder mandatory before I trust any learned result?
3. Do you see any remaining structural mistakes or hidden confounds in the implemented code path?
4. What are the top reviewer attack points now, and what is the fastest defense evidence for each?
5. Is the current naming/positioning credible:
   - "Laplace-inspired temporal signatures"
   - "learned trust calibration"
   - "when to trust appearance"
6. If you were responsible for the next 3 days of work, exactly what would you run or fix first?

Please give:
- a hard strategic judgment first
- then concrete code/design findings with file-level references
- then the next experiment order
- then paper-positioning advice

Do not propose broadening the project into a larger tracker redesign unless you think the current line is fundamentally broken.
MD

mkdir -p "${PACK_DIR}/external/BoT-SORT-main/tracker"
for f in external/BoT-SORT-main/tracker/*.py; do
  copy_file "${f}" "${PACK_DIR}/external/BoT-SORT-main/tracker"
done
copy_file "external/BoT-SORT-main/tools/track.py" "${PACK_DIR}/external/BoT-SORT-main/tools"

for f in \
  scripts/pack_ltra_pro_review.sh \
  scripts/train_ltra_calibrator_from_pairs.py \
  scripts/summarize_laplace_pair_logs.py \
  scripts/run_botsort_laplace_matrix.sh \
  scripts/run_botsort_ltra_stage1.sh \
  scripts/run_botsort_ltra_stage2.sh \
  scripts/run_botsort_ltra_submit.sh \
  scripts/run_botsort_dancetrack_val.sh \
  scripts/run_botsort_dancetrack_submit.sh \
  scripts/run_strongsort_laplace_matrix.sh \
  scripts/run_strongsort_ltra_mot17_val.sh \
  scripts/queue_dancetrack_overnight.sh \
  scripts/queue_dancetrack_test_submit.sh \
  scripts/make_dancetrack_submission_zip.py \
  scripts/extract_dancetrack_full.sh \
  scripts/make_mot17_private_det_zip.py \
  scripts/sanitize_mot_submission.py \
  scripts/check_mot17_submission.py \
  scripts/check_mot20_submission.py \
  scripts/collect_trackeval_metrics.py \
  scripts/eval_motstyle_trackeval.py \
; do
  copy_file "${f}" "${PACK_DIR}/$(dirname "${f}")"
done

copy_tree "vnext_laplace_assoc" "${PACK_DIR}/vnext_laplace_assoc"
copy_tree "paper" "${PACK_DIR}/paper"

mkdir -p "${PACK_DIR}/outputs/botsort_ltra_stage1/MOT17"
mkdir -p "${PACK_DIR}/outputs/botsort_ltra_stage1/MOT20"
mkdir -p "${PACK_DIR}/outputs/botsort_ltra_stage2/MOT17"
mkdir -p "${PACK_DIR}/outputs/botsort_ltra_stage2/MOT20"
mkdir -p "${PACK_DIR}/outputs/botsort_ltra_stage3/MOT20"
mkdir -p "${PACK_DIR}/outputs/dancetrack"
mkdir -p "${PACK_DIR}/outputs/strongsort_ltra/MOT17_val"
mkdir -p "${PACK_DIR}/outputs/paper_ctrl_mot17_val0213"

copy_file_as "outputs/botsort_ltra_stage1/MOT17/summary.csv" "${PACK_DIR}/outputs/botsort_ltra_stage1/MOT17/summary.csv"
copy_file_as "outputs/botsort_ltra_stage1/MOT20/summary.csv" "${PACK_DIR}/outputs/botsort_ltra_stage1/MOT20/summary.csv"
copy_file_as "outputs/botsort_ltra_stage2/MOT17/summary.csv" "${PACK_DIR}/outputs/botsort_ltra_stage2/MOT17/summary.csv"
copy_file_as "outputs/botsort_ltra_stage2/MOT20/summary.csv" "${PACK_DIR}/outputs/botsort_ltra_stage2/MOT20/summary.csv"
copy_file_as "outputs/botsort_ltra_stage3/MOT20/summary.csv" "${PACK_DIR}/outputs/botsort_ltra_stage3/MOT20/summary.csv"
copy_file_as "outputs/dancetrack/overnight_summary.tsv" "${PACK_DIR}/outputs/dancetrack/overnight_summary.tsv"
copy_file_as "outputs/strongsort_ltra/MOT17_val/summary.csv" "${PACK_DIR}/outputs/strongsort_ltra/MOT17_val/summary.csv"
copy_file_as "outputs/strongsort_ltra/MOT17_val/summary.txt" "${PACK_DIR}/outputs/strongsort_ltra/MOT17_val/summary.txt"
copy_file_as "outputs/paper_ctrl_mot17_val0213/summary.csv" "${PACK_DIR}/outputs/paper_ctrl_mot17_val0213/summary.csv"

copy_tree "outputs/laplace_pair_logs/MOT17_full" "${PACK_DIR}/outputs/laplace_pair_logs/MOT17_full"
copy_tree "outputs/laplace_pair_logs/MOT20_meanrel" "${PACK_DIR}/outputs/laplace_pair_logs/MOT20_meanrel"
copy_tree "outputs/ltra_analysis/mot20_val_seq05_meanrel_20260310_190310/summary" "${PACK_DIR}/outputs/ltra_analysis/mot20_val_seq05_meanrel_20260310_190310/summary"
copy_tree "outputs/ltra_analysis/mot20_val_seq05_meanrel_20260310_190310/summary_full" "${PACK_DIR}/outputs/ltra_analysis/mot20_val_seq05_meanrel_20260310_190310/summary_full"

(
  cd "${PACK_DIR}"
  {
    echo "# Files Included"
    echo
    echo "Generated from local workspace on $(date '+%Y-%m-%d %H:%M:%S %Z')."
    echo
    find . -type f | sort
  } > FILES_INCLUDED.md
)

echo "[pack] zip -> ${ZIP_PATH}"
rm -f "${ZIP_PATH}"
(
  cd "${PACK_ROOT}"
  zip -qr "${ZIP_PATH}" "$(basename "${PACK_DIR}")"
)

echo "[OK] ${ZIP_PATH}"
echo "[OK] size: $(du -h "${ZIP_PATH}" | cut -f1)"
