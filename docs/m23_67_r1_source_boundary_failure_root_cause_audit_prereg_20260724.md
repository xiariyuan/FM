# M23-67-R1 — Source Boundary Failure Root-Cause Audit Repair Preregistration

- Experiment ID: `M23-67-R1`
- Nature: post-hoc diagnostic repair only
- Date: 2026-07-24
- Run root: `outputs/mot20_m23_20260718/m23_67_r1_source_boundary_failure_root_cause_audit`
- Script: `scripts/m23_research/m23_67_r1_source_boundary_failure_root_cause_audit.py`
- Test: `scripts/m23_research/test_m23_67_r1_source_boundary_audit.py`
- Predecessor: M23-67, fail-closed as `FAIL_IMPLEMENTATION`

## 1. Purpose and only allowed repair

M23-67-R1 repeats the frozen M23-67 scientific protocol after repairing exactly one implementation-definition error. M23-67 incorrectly treated semantic non-interchangeability of `row_index` and `line_index` as a requirement that at least one numeric value differ. R1 permits the two explicit columns to be numerically equal and instead proves that the real score/label mapping is keyed by explicit semantic `row_index` values.

The repaired validation must establish all of the following through the actual mapping/join helper used by the experiment:

1. `row_index` and `line_index` are explicit columns.
2. `row_index` is present and unique in the observable and supervision tables used as row maps.
3. score `src_row_index` and `dst_row_index` map uniquely through `row_index`.
4. label rows map through `row_index`.
5. `line_index` is retained only as metadata and is not accepted as the join key.
6. parquet physical row position is not used as a semantic identifier; arbitrary physical reordering does not change the join result.
7. missing or duplicate semantic `row_index` values fail closed.
8. numeric equality between `row_index` and `line_index` is allowed and does not by itself establish or refute semantic correctness.

No other metric, threshold, aggregation, population, stratum, classification priority, input, or scientific rule may change from M23-67.

## 2. Immutable inputs

All inputs are read-only. R1 may use only frozen artifacts from:

- R62: `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_gtfree_source_regeneration`
- R63: `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_supervision_join`
- R64-R1: `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_pair_reconstruction_training_repair1`
- R65: `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_mot20_representation_gate`
- R66: `outputs/mot20_m23_20260718/m23_66_v3_metric_correctness_source_target_decomposition_audit`
- failed/closed R67: `outputs/mot20_m23_20260718/m23_67_source_boundary_failure_root_cause_audit`

R67 is provenance only and must not be modified, rerun, or used as an output root. R1 freezes and re-verifies every file SHA listed by the predecessor input manifest plus the R67 script, test, prereg, result, final summary, closure, independent closure, implementation failure, artifact manifest, and input manifest.

Labels come only from R63 frozen supervision sidecars. Scores come only from frozen R66 source-score artifacts. The R64-R1 checkpoint may be loaded only in `eval()` under `torch.no_grad()` for frozen source inference needed to compare the R64 train and validation populations. No raw MOT17 or MOT20 GT may be read.

## 3. Absolute prohibitions

M23-67-R1 prohibits:

- model training, optimizer creation or optimizer steps;
- new or modified checkpoints, warm start, checkpoint selection, or seed/epoch changes;
- candidate-K, topology-gap, gate, policy P0/P1/P2, or metric-threshold changes;
- threshold search, score reversal as repair/result, calibration, temperature scaling, or result-driven aggregation selection;
- modifying R62–R67 artifacts;
- tracker generation, TrackEval, HOTA, MOT20 test reads/submission;
- teacher, held-outer, M23-54, M23-58, or M23-68 execution;
- raw MOT17/MOT20 GT reads;
- repository commit, push, reset, clean, checkout-revert, or unrelated cleanup.

The experiment must end with:

- `post_hoc_diagnostic_only=true`
- `uses_frozen_gt_derived_label_sidecars=true`
- `not_deployable=true`
- `not_a_strict_result=true`
- `hota=null`
- `next_policy_authorized=false`

## 4. Pre-freeze required tests

Before `implementation_frozen`, the R1 test suite must call the same mapping/join helper used by `run-mapping-audit` and pass at least:

1. equal-valued `row_index`/`line_index` fixture;
2. different-valued fixture;
3. deliberate `line_index`-keyed join rejection/failure;
4. physical row shuffle invariance under `row_index` join;
5. missing semantic row-index failure;
6. duplicate semantic row-index failure;
7. unknown and ambiguity endpoint exclusion;
8. repeated-label consistency;
9. physical-transition aggregation (`sigmoid(mean(logit))` primary; mean probability sensitivity);
10. fixed bucket boundaries;
11. score-collapse rule;
12. fixed A>B>C>D decision priority.

A fail-close end-to-end selftest must also inject a post-freeze failure into a disposable temporary experiment root and registry copy and prove that the R1 fail-close implementation itself:

- removes all running/pending statuses;
- writes summary CSV, protocol JSONL, a failure record, final summary, closure validation, independent closure validation, and artifact manifest;
- re-verifies frozen inputs;
- records zero scope counts;
- closes the disposable registry row as failed/closed;
- derives `closure_integrity_passed` and `independent_closure_passed` from actual checks rather than assigning unconditional true.

## 5. Boundary labels and physical-transition aggregation

For two adjacent frozen source rows whose endpoints are eligible matched, non-distractor, non-ambiguous, and non-tied:

- `y_boundary=1` iff sequence-namespaced `gt_identity_key` differs;
- `y_boundary=0` iff it is equal;
- any unknown, distractor, ambiguity, tie, or otherwise non-matched endpoint is excluded from binary metrics and never treated as negative.

Physical-transition key is fixed:

`(sequence, src_row_index, dst_row_index)`

Primary score is fixed:

`primary_probability = sigmoid(arithmetic_mean(all finite boundary_logit observations))`

Sensitivity only:

`mean_probability = arithmetic_mean(boundary_probability observations)`

No result-dependent selection between aggregations is allowed.

## 6. Fixed populations

The population audit compares exactly:

1. M23-64-R1 frozen train node-window examples;
2. M23-64-R1 frozen validation node-window examples;
3. M23-66 full source diagnostic for MOT17-11 and MOT17-13.

Report fixed counts and distributions for windows, observations, unique transitions, positives, negatives, base rate, tracks, sequences, frame delta, box size, crowd density, appearance mapping/missingness, visible/unknown state, and per-sequence counts. Report:

- `train_positive_rate`
- `validation_positive_rate`
- `audit_positive_rate`
- `train_to_audit_positive_ratio`
- `train_to_audit_unique_transition_ratio`

The inherited M23-67 severe-population-mismatch rule is unchanged. It is true if any of:

- train/audit positive-rate ratio is above `5.0` or below `0.2`;
- validation/audit positive-rate ratio is above `5.0` or below `0.2`;
- train/audit unique-transition-count ratio is below `0.2`;
- a fixed crowd-density or appearance bucket represents at least 5% of eligible audit observations and has zero eligible train observations.

These inherited rules are diagnostic and may not be tuned or supplemented after results are seen.

## 7. Fixed score diagnostics

For `boundary_logit` and `boundary_probability`, report positive/negative count, mean, std, min, q00, q01, q05, q25, q50, q75, q95, q99, q100. Descriptively report positive/negative means and medians, pairwise probability that a positive score exceeds a negative score, rank orientation, reversed-score diagnostic metrics, saturation fractions, absolute-logit tail, unique-score count, tie rate, top-k distinct score count, and raw versus unique-transition variance.

The inherited M23-67 orientation rule is unchanged: `score_orientation_anomaly=true` when pooled corrected source ROC-AUC under the declared orientation is below `0.5`. It is descriptive only. Reversed scores are never a model result or gate result.

The inherited M23-67 score-collapse rule is unchanged. `FAIL_BOUNDARY_SCORE_COLLAPSE` requires mapping/label and population alignment to pass, pooled corrected source ROC-AUC in `[0.45,0.55]`, and at least one low-variation condition:

- unique-transition logit standard deviation `<0.10`;
- exact logit tie rate `>=0.95`;
- unique finite logit count `<100`.

## 8. Fixed strata

No bucket search is allowed.

Sequence:
`MOT17-02`, `MOT17-04`, `MOT17-05`, `MOT17-09`, `MOT17-10`, `MOT17-11`, `MOT17-13`.

Frame/topology-gap descriptive buckets:
`1-30`, `31-90`, `91-180`, `181-600`, using `dst_frame-src_frame` for boundary rows and preserving the topology definition `dst_first_frame-src_last_frame` where chunk topology is reported. Subtracting one is sensitivity metadata only.

Crowd density uses zero-based feature 142, `geometry_14_crowd_density_over_100_clipped`:
`[0,0.25)`, `[0.25,0.50)`, `[0.50,1.00)`, `[1.00,2.00)`, `[2.00,5.00]`.
Feature 143 is nearest-neighbor distance and must not be used as crowd density.

Appearance uses frozen GT-free `appearance_mapped`:
`0`, `(0,0.50)`, `[0.50,0.90)`, `[0.90,1.00)`, `1.00`.

Track purity from frozen supervision:
`[0.80,0.90)`, `[0.90,0.99)`, `[0.99,1.00)`, `1.00 exact`.

Each stratum reports counts, base rate, PR-AUC, PR/base-rate lift, ROC-AUC, precision@actual, recall@95P, positive/negative score distributions, and score variance where defined.

## 9. Fixed source reference and classification priority

Frozen source reference thresholds:

- macro corrected PR-AUC `>=0.283`;
- macro precision@actual `>=0.35`;
- macro recall@95% precision `>=0.05`;
- each source sequence precision@actual `>=0.20`.

Unique primary classification priority is fixed:

A. Any genuine row/label mapping failure:
- `measurement_integrity_decision=FAIL_BOUNDARY_LABEL_MAPPING`
- `scientific_primary_failure=boundary_label_mapping_failure`
- `overall_primary_classification=FAIL_BOUNDARY_LABEL_MAPPING`

B. Mapping/label pass but severe frozen population mismatch:
- `FAIL_BOUNDARY_POPULATION_SHIFT`
- primary `boundary_population_mismatch`

C. Mapping/label and population pass but frozen score-collapse rule passes:
- `FAIL_BOUNDARY_SCORE_COLLAPSE`
- primary `boundary_score_collapse`

D. Mapping, population, and score implementation pass, but any fixed source reference fails:
- `FAIL_SOURCE_BOUNDARY_CAPACITY`
- primary `source_boundary_capacity_failure`

Otherwise:
- `PASS_BOUNDARY_IMPLEMENTATION`, or `INCONCLUSIVE_BOUNDARY_ROOT_CAUSE` only when required evidence is undefined despite valid implementation.

Multiple failures must have exactly one primary according to A>B>C>D, with others listed as secondary. “Domain shift” is not an allowed catch-all.

## 10. Execution and closure

Required command order:

1. `init`
2. `verify-inputs`
3. `run-mapping-audit`
4. `run-label-audit`
5. `run-population-audit`
6. `run-score-distribution-audit`
7. `run-stratified-boundary-audit`
8. `diagnose`
9. `validate`
10. `summarize`

A genuine mapping/label failure may skip downstream scientific stages only under the frozen priority rule, but R1 must still close itself completely.

At closure, recompute all frozen input SHAs, verify R62–R67 unchanged, ensure scope counts are zero, confirm no relevant process/GPU compute use, ensure no stale running/pending stage, write a complete artifact SHA manifest, derive closure and independent closure checks, write the result document, and update the existing R1 registry row to `completed/closed` or `failed/closed`.

M23-67-R1 is diagnostic only and does not automatically authorize M23-68 or any training/policy/tracker stage.
