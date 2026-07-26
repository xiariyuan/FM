# M23-67 — Source Boundary Failure Root-Cause Audit — Preregistration

Date: 2026-07-24

## Identity and scope

M23-67 is an independent post-hoc source-side diagnostic following the closed M23-66 result. It determines why corrected source boundary performance is low before any remediation training is authorized.

Fixed declarations:

- `post_hoc_diagnostic_only=true`
- `uses_frozen_gt_derived_label_sidecars=true`
- `not_deployable=true`
- `not_a_strict_result=true`
- `training_authorized=false`
- `next_policy_authorized=false`
- `hota=null`

No result from M23-67 authorizes M23-68 automatically.

## Immutable inputs

Only read-only inputs under R62, R63, R64-R1, R65, and R66 are allowed. Raw MOT17 GT and raw MOT20 GT are forbidden. Labels may be read only from `R63/row_supervision.parquet`; corrected validation score artifacts may be read from R66. The R64-R1 checkpoint may be loaded only in `eval()` / `no_grad()` mode for deterministic source inference. All source and output artifacts under R62–R66 remain immutable.

Core inputs to hash and freeze before real audits:

- `R64/frozen_checkpoint/relation_v3_frozen.pt`
- `R63/row_supervision.parquet`
- `R63/source_windows.parquet`
- `R63/source_chunks.parquet`
- `R63/candidate_pool.parquet`
- `R64/external_validation_metrics.json`
- `R66/boundary_metrics.json`
- `R66/source_target_comparison.json`
- `R66/final_summary.json`

## Absolute prohibitions

No model training, optimizer construction or step, checkpoint creation/modification, warm start, threshold search, score reversal as repair, calibration, temperature scaling, candidate-K change, gap change, gate change, tracker, TrackEval, HOTA, MOT20 test access/submission, teacher action, held-outer access, M23-54, M23-58, P0/P1/P2 policy, raw MOT17 GT read, raw MOT20 GT read, or modification of M23-65/M23-66 artifacts is allowed.

## Boundary row mapping contract

For every boundary observation, `(sequence, src_row_index, dst_row_index)` must map uniquely to observable source rows. Both rows must exist, belong to the same `track_id`, preserve the source track’s stable temporal/line/row order, and satisfy `dst_frame > src_frame`. `row_index` is the explicit semantic row identifier, not a parquet physical offset. `line_index` is distinct metadata and must not be substituted for `row_index`. `score_to_source_row.parquet` must match the corresponding observable rows exactly on `row_index, frame, line_index, track_id, x1, y1, x2, y2`.

Manual traces are fixed at 10 positive, 10 negative, 10 repeated-observation transitions, and 10 unknown-endpoint observations, with both MOT17-11 and MOT17-13 represented whenever each category exists.

## Boundary label contract

For adjacent source rows in one boundary observation:

- `y_boundary=1` iff both endpoints have `supervision_status=matched` and their sequence-namespaced `gt_identity_key` values differ.
- `y_boundary=0` iff both endpoints are matched and their `gt_identity_key` values are equal.
- If either endpoint is unknown, distractor, ambiguous, or otherwise not matched, the observation is excluded from binary metrics and is never a negative.

`source_track_id` and unnamespaced `gt_id` are never boundary identity labels. All repeated observations of the same physical transition must have identical labels.

## Physical-transition aggregation

The physical key is `(sequence, src_row_index, dst_row_index)`.

Primary corrected score:

1. retain finite observations only;
2. arithmetic-mean all `boundary_logit` observations for the physical key;
3. apply sigmoid to the mean logit.

Sensitivity score: arithmetic mean of `boundary_probability` for the same key.

Legacy observation-weighted, corrected mean-logit primary, and mean-probability sensitivity are all reported. No aggregation is selected based on results.

## Population audit

Three fixed populations are compared:

1. R64-R1 training examples;
2. R64-R1 validation examples;
3. full corrected MOT17-11/13 source-window physical transitions used by M23-66.

The audit reports windows, observations, unique transitions, positives, negatives, base rate, tracks, sequences, frame-delta, box-size, feature-142 crowd density, `appearance_mapped`, missing appearance, matched/unknown status, and per-sequence counts.

Sampling-bias indicators include fixed-length-window coverage, per-sequence counts, track-start-offset distribution, positive/negative rebalancing, unknown exclusion, density coverage, frame-delta coverage, and appearance-missing coverage. The following ratios are descriptive and fixed:

- `train_to_audit_positive_ratio = train_positive_rate / audit_positive_rate`
- `train_to_audit_unique_transition_ratio = train_unique_transitions / audit_unique_transitions`

Population mismatch is classified only from predeclared distribution diagnostics; no result-driven resampling is performed.

## Score distribution and orientation

For `boundary_logit` and `boundary_probability`, positive and negative classes report count, mean, standard deviation, min, q00/q01/q05/q25/q50/q75/q95/q99/q100.

Orientation is descriptive only: positive/negative means and medians, pairwise probability `P(score_positive > score_negative)` with half credit for ties, and rank statistics are reported. An apparent reverse orientation sets `score_orientation_anomaly=true`; reversed scores are never used as a repaired model or gate result.

Saturation/tie diagnostics report probability fractions `<=1e-6` and `>=1-1e-6`, logit fractions `|logit|>=10` and `|logit|>=20`, exact unique score count, exact tie rate, and top-k distinct score counts for fixed k in `{10,100,1000}`.

## Fixed source strata

Sequence strata are fixed to MOT17-02/04/05/09/10/11/13. M23-66 primary interpretation remains focused on MOT17-11/13; train sequences are descriptive comparisons.

Boundary observations are adjacent rows inside frozen source windows. Their direct row frame delta is reported. The v3 topology convention is separately reconciled as `topology_gap = dst_first_frame - src_last_frame`, with fixed buckets `1-30, 31-90, 91-180, 181-600`; `topology_gap-1` may only be descriptive. Boundary row transitions are not silently reinterpreted as candidate topology edges.

Crowd-density feature is zero-based feature 142, `geometry_14_crowd_density_over_100_clipped`, with fixed buckets:

- `[0,0.25)`
- `[0.25,0.50)`
- `[0.50,1.00)`
- `[1.00,2.00)`
- `[2.00,5.00]`

Zero-based feature 143 is `geometry_15_nearest_neighbor_distance` and is never used as crowd density.

`appearance_mapped` is a GT-free row sidecar and is not feature 143. Transition/window mapped fractions use fixed buckets:

- `0`
- `(0,0.50)`
- `[0.50,0.90)`
- `[0.90,1.00)`
- `1.00 exact`

Track/chunk purity is computed from frozen matched supervision only and uses fixed buckets:

- `[0.80,0.90)`
- `[0.90,0.99)`
- `[0.99,1.00)`
- `1.00 exact`

## Fixed root-cause decision order

Fields:

- `measurement_integrity_decision`
- `boundary_label_mapping_status`
- `training_population_alignment_status`
- `score_collapse_status`
- `source_capacity_status`
- `scientific_primary_failure`
- `overall_primary_classification`

Allowed overall classifications:

- `PASS_BOUNDARY_IMPLEMENTATION`
- `FAIL_BOUNDARY_LABEL_MAPPING`
- `FAIL_BOUNDARY_POPULATION_SHIFT`
- `FAIL_BOUNDARY_SCORE_COLLAPSE`
- `FAIL_SOURCE_BOUNDARY_CAPACITY`
- `INCONCLUSIVE_BOUNDARY_ROOT_CAUSE`

Priority:

A. Any row/score/label mapping failure -> primary `boundary_label_mapping_failure`, overall `FAIL_BOUNDARY_LABEL_MAPPING`.

B. Mapping passes, and severe population mismatch is present -> primary `boundary_population_mismatch`, overall `FAIL_BOUNDARY_POPULATION_SHIFT`.

C. Mapping and population pass, but scores are near-constant or class distributions effectively overlap -> primary `boundary_score_collapse`, overall `FAIL_BOUNDARY_SCORE_COLLAPSE`.

D. Mapping, population, and score implementation/orientation are acceptable, while corrected source capacity remains extremely low -> primary `source_boundary_capacity_failure`, overall `FAIL_SOURCE_BOUNDARY_CAPACITY`.

E. Multiple failures retain one primary by A>B>C>D; others are secondary. Insufficient evidence yields `INCONCLUSIVE_BOUNDARY_ROOT_CAUSE`.

Fixed severe-population-mismatch indicators are any of:

- training or validation positive rate differs from audit by a factor greater than 5 or less than 0.2;
- train unique physical-transition coverage is below 20% of audit while examples are claimed to represent full source behavior;
- at least one audit crowd-density or appearance-missing stratum containing >=5% of audit transitions has zero train support;
- train and audit sequence support are structurally non-comparable beyond the predeclared train/validation split disclosure.

Fixed score-collapse indicators require both weak discrimination and low variability: corrected pooled MOT17-11/13 ROC-AUC in `[0.45,0.55]` and either logit standard deviation `<0.10`, exact tie rate `>=0.95`, or fewer than 100 distinct finite scores. Reverse orientation alone is an anomaly, not a repaired score.

Source capacity remains failed when M23-66 corrected MOT17-11/13 macro metrics remain below the unchanged reference: PR-AUC 0.283, precision@actual 0.35, recall@95P 0.05, and every-sequence precision@actual 0.20.

## Synthetic implementation tests

Before implementation freeze, tests must cover unique row mapping, row-index versus line-index distinction, unknown exclusion, repeated-label consistency, mean-logit aggregation, population ratio calculations, fixed bucket boundaries, orientation anomaly reporting, score-collapse rule, and decision-priority ordering.

## Closure

M23-67 produces diagnostics only. It must close all summary stages, close its registry row, preserve R62–R66 byte hashes, record zero training/tracker/TrackEval/HOTA/raw-GT counters, and state `next_policy_authorized=false`. M23-68 is not started in this experiment.
