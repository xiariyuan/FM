# FCAA Mainline

This document is the execution anchor for the `FCAA` project line.

## Project Definition

- carrier tracker: `BoT-SORT-style`
- frozen components: detector, ReID backbone, motion model
- novelty: `frequency-conditioned appearance rescoring`
- scope: `selective ambiguity resolution`

## What This Line Is Not

The following are not part of the FCAA line:

- the old heavy trajectory-frequency family
- full `freq_aware_trajectory_modeling`
- Mamba-based frequency mainline
- whole-tracker rewrite
- end-to-end detector + ReID + association retraining

## First Milestone

The first milestone was intentionally narrow:

1. build a GT-aligned pseudo-track pair-bank
2. train `control` scorer on `s_reid`
3. train `freq` scorer on `[s_reid, s_low, s_mid, s_high]`
4. prove offline ambiguous-pair signal before online integration

## March 30, 2026 Status

The original `row-margin` formulation is now closed as the main offline bank.

What was verified:

- the pair-bank is now detector-driven and BoT-SORT-aligned
- pseudo tracks now use `STrack + Kalman`, not raw GT boxes
- high-score stage-1 row ambiguity is structurally sparse on the hard MOT17 carrier slices

Key evidence:

- `outputs/fcaa_pairbank_hardseq_conflict_diag_20260330_1/pairbank/summary.csv`
  - `groups=16508`
  - `ambiguous_groups=4`
  - `negative_rows=7`
  - `top1_conflict_tracks=1825`
  - `top1_conflict_detections=893`

Interpretation:

- row-style ambiguity is too sparse to support meaningful training
- shared-detection top-1 conflicts are abundant on the same carrier
- the live mainline has therefore pivoted from `row_margin` to `shared_det_top1`

Current live offline bank:

- `outputs/fcaa_pairbank_sharedet_diag_20260330_1/pairbank/summary.csv`
  - `grouping=shared_det_top1`
  - `groups=846`
  - `rows=1727`
  - `positive_rows=846`
  - `negative_rows=881`

Current offline A/B runs:

- control, seed 42: `outputs/fcaa_sharedet_control_20260330_1/summary.csv`
- freq, seed 42: `outputs/fcaa_sharedet_freq_20260330_1/summary.csv`
- control, seed 7: `outputs/fcaa_sharedet_control_seed7_20260330_1/summary.csv`
- freq, seed 7: `outputs/fcaa_sharedet_freq_seed7_20260330_1/summary.csv`

Current readout:

- seed 42: `freq` beats `control` on `val_ambiguous_top1` and `val_auc`
- seed 7: `freq` still beats `control` on `val_ambiguous_top1`, but `val_auc` drops

Management conclusion:

- keep `shared_det_top1` alive
- do not resume `row_margin` as the main offline bank
- do not claim stable frequency advantage yet
- require additional offline stability or online confirmation before paper-level claims

## Required Controls

Every FCAA experiment must keep these comparisons:

1. raw carrier
2. same-gate no-frequency control
3. same-gate frequency scorer

Without the same-gate control, the line is not considered diagnostic.
