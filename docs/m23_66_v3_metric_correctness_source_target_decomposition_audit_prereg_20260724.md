# M23-66 — M23-59 v3 Metric Correctness and Source-to-Target Decomposition Audit

Pre-registered: 2026-07-24

## Status and scope

M23-66 is an independent post-hoc diagnostic following the closed failed M23-65 result. It is not an M23-65 policy continuation and does not authorize deployment, strict evaluation, tracker construction, TrackEval, or HOTA.

Fixed declarations:

- `post_hoc_diagnostic_only=true`
- `uses_frozen_gt_derived_label_sidecars=true`
- `not_deployable=true`
- `not_a_strict_result=true`
- `hota=null`
- `next_policy_authorized=false`

The immutable inputs are R62, R63, R64-R1, and R65. They are read-only. The only checkpoint is R64-R1 `frozen_checkpoint/relation_v3_frozen.pt`, expected SHA-256 `dc24211ff8c050f03920711aadac009d4feac26568ff356f8d6bd29dabf7d329` and parameter count 881124. The model source is the frozen v2 implementation with expected SHA-256 `50e12cdc6259903f9a7a76adfd3329622899c0c727f1466c2399034c49f6451d`. No v2 checkpoint is loaded.

## Prohibited operations

No training, optimizer construction or step, checkpoint creation/overwrite, warm start, checkpoint modification, candidate-K/horizon/bucket modification, gate/risk modification, temperature scaling, calibration fitting, threshold search, score-sign reversal, model or score selection, tracker output, TrackEval, HOTA, MOT20 test read/submission, teacher action, held-outer access, M23-54/M23-58, policy P0/P1/P2, M23-59 v1 artifact read, or write under R62/R63/R64/R65 is permitted.

Raw GT is not reopened. MOT20 labels come only from `R65/<sequence>/labels/row_labels.parquet`; MOT17 labels come only from `R63/row_supervision.parquet`. Therefore the preregistered counters are `new_raw_mot20_gt_reads=0`, `new_raw_mot17_gt_reads=0`, and `frozen_label_sidecar_reads=true`.

## Frozen feature and topology contract

The required v3 contract hash is `90cfab7d3a1fc87cc46b26441d3d883b9fc72f27d8852d1e2074b00e628428f5`. Zero-based feature 143 must be `geometry_15_nearest_neighbor_distance`. `appearance_mapped` is an independent GT-free row-sidecar field and is not feature 143.

Frozen topology rules:

- `MAX_NODE_ROWS=30`
- `NODE_STRIDE=15`
- `CHUNK_MAX_ROWS=30`
- `CHUNK_MAX_GAP=30`
- `CANDIDATE_MAX_GAP=600`
- `K=32` per source/gap bucket
- gap buckets: `1-30`, `31-90`, `91-180`, `181-600`
- candidate score: `0.70 * appearance_cosine + 0.30 * exp(-4 * geometry_distance)`

The primary gap is `topology_gap = dst_first_frame - src_last_frame`; eligible candidates satisfy `1 <= topology_gap <= 600`. `intervening_empty_frames = topology_gap - 1` is descriptive sensitivity only and never changes pools or buckets.

## Source inference

The primary source domain is MOT17-11 and MOT17-13, the R64 validation split. Source findings are capability diagnostics, not independently held-out generalization claims.

Inference uses R62 observables, R63 frozen windows/chunks/candidate pools/pairs, and the R64-R1 checkpoint. The model is in `eval()` under `torch.no_grad()`. No optimizer is created. Parameter count and checkpoint SHA are checked before and after inference. Inputs, masks, and outputs must be finite; masks must be contiguous prefixes. Deterministic settings are fixed, and a fixed small sample is evaluated twice for numeric identity. Source artifacts are written only under R66 `source_scores/MOT17-11` and `source_scores/MOT17-13`.

Source scores are frozen before R63 row supervision is opened by any metric stage.

## Trusted chunks

For a chunk, only rows with `supervision_status=matched` participate. A trusted chunk requires at least two matched rows and modal GT-identity purity at least 0.80. If two or more identities tie for the maximum count, the chunk is `untrusted_identity_tie`. Unknown, distractor, and ambiguous rows never become negatives. Chunk identity is the modal `gt_identity_key`, never source `track_id`.

Preregistered chunk descriptors used only for decomposition are:

- crowd density: arithmetic mean of zero-based feature 142 over all chunk rows;
- appearance-mapped fraction: arithmetic mean of the boolean `appearance_mapped` row sidecar over all chunk rows;
- trusted purity: modal matched identity count divided by matched row count;
- frame density: arithmetic mean, over distinct frames represented by the chunk, of the full observable sequence row count at that frame.

## Canonical queries independent of candidate pools

Canonical construction reads only chunks, frozen label/supervision sidecars, and observable row metadata/features. It does not read candidate pools, relation scores, or model rankings. `canonical_queries.parquet`, `valid_targets.parquet`, and their manifest are frozen before candidate pools are opened for coverage/ranking.

### Outgoing canonical successor

For every trusted source chunk, search the complete trusted same-sequence chunk universe for the same `gt_identity_key`, with `dst_first_frame > src_last_frame` and `1 <= topology_gap <= 600`. The valid-target set is independent of the candidate pool. Sort valid targets by destination first frame ascending, destination last frame ascending, then destination chunk ID lexicographically; the first is canonical.

A trusted source with a temporal future same-identity chunk but no target within the horizon is `horizon_outside`. A trusted source with no temporal future same-identity chunk is `no_successor`. Only `eligible` queries enter rank denominators.

### Incoming canonical predecessor

For every trusted destination chunk, search the complete trusted same-sequence universe for the same identity, with `src_last_frame < dst_first_frame` and `1 <= topology_gap <= 600`. Sort valid predecessors by source last frame descending, source first frame descending, then source chunk ID lexicographically; the first is canonical.

A trusted destination with temporal past but no predecessor within the horizon is `horizon_outside`; no temporal past is `no_predecessor`. Only `eligible` queries enter rank denominators.

The manifest reports trusted chunks, temporal future/past counts, eligible, horizon-outside, no-successor/no-predecessor, canonical queries, valid targets, and multi-positive queries.

## Candidate coverage

Coverage is reported for canonical exact target and any valid positive, for all eligible queries. Candidate-missing queries remain in all-query denominators and contribute zero to R@1, R@3, and MRR. Reports include all-query, candidate-present, canonical-present, any-positive-present, missing count, coverage, sequence, direction, frozen gap bucket, pooled, and macro views.

## Corrected relation ranking

Primary ranking sorts `relation_logit` descending and uses `candidate_id` lexicographically as the stable tie break. Canonical target absent from the pool has infinite rank. Any-valid-positive rank is the minimum rank of any valid target. `pos_idx[0]` is never used to define ground truth, and all-query is never copied from candidate-present.

Metrics are canonical exact-target and any-valid-positive R@1/R@3/MRR, each in all-query and candidate-present-conditional forms, outgoing/incoming, per sequence, macro and pooled. Also reported are score tie rate, top1-top2 margin, and catastrophic false-link rate.

The fixed diagnostic baselines are evaluated without tuning or selection:

- frozen `relation_logit` descending;
- `candidate_score` descending;
- `appearance_cosine` descending;
- `geometry_distance` ascending;
- `topology_gap` ascending.

## Paired metric correction

The frozen pair fields are `original_margin`, `original_mean_logit`, `cross_mean_logit`, and `paired_probability`. Every row must satisfy `original_margin == original_mean_logit - cross_mean_logit` within numeric tolerance.

A legal original-vs-cross pair requires all four endpoint chunks trusted, both original edges same-identity, the two original identities different, and both crossed edges present in the frozen candidate pool. The primary metric is

`valid_pair_original_over_cross_accuracy = mean(original_margin > 0)`

on legal pairs only. A zero margin is counted separately and fails the strict primary accuracy. Half-credit is sensitivity only.

For classification over the evaluable trusted-endpoint pair universe, legal pairs are positives and other trusted-endpoint configurations are negatives. Report positive base rate, threshold accuracy at 0.5, balanced accuracy, PR-AUC, AUROC, valid/invalid/unknown counts, valid-margin mean/std/quantiles, and tie rate. The M23-65 field is named only `legacy_misnamed_paired_replacement_R@1`.

## Boundary duplicate-window audit

The physical transition key is `(sequence, src_row_index, dst_row_index)`. The audit validates unique score-to-row mappings, same source track, strict forward time, duplicate-label consistency, finite logits/probabilities, sigmoid agreement, and exclusion of unknown endpoints.

Three fixed views are reported:

A. `legacy_observation_weighted`: each frozen score row is one observation.

B. `corrected_unique_transition_primary`: arithmetic mean of all finite `boundary_logit` values for a physical transition, followed by sigmoid of the mean logit. Each transition appears once. This is primary.

C. `corrected_unique_transition_probability_sensitivity`: arithmetic mean of frozen boundary probabilities per transition. This is sensitivity only.

For each view report raw/matched observations, unique transitions, duplicate count and multiplicity distribution, positives, base rate, PR-AUC, PR-AUC/base-rate lift, ROC-AUC, precision at actual-positive count, recall at 90/95/99 percent precision, positive/negative mean/std, q00/q01/q05/q25/q50/q75/q95/q99/q100, saturation, and tie rate.

Precision at actual-positive count uses `k = number of positives`, score descending, and transition key lexicographically as stable tie break. No threshold is searched.

M23-65 legacy boundary metrics are independently reproduced from frozen artifacts. Counts must match exactly; absolute metric differences and tolerance `1e-10` are recorded. Failure is disclosed as `legacy_behavioral_reproduction=false`.

## Source-to-target decomposition

The same corrected definitions compare source MOT17-11/13 and target MOT20-01/02/03/05.

Boundary comparisons include PR-AUC, base rate, lift over base rate, ROC-AUC, precision@actual, recall@95P, positive/negative quantiles, duplicate factor, and score mean/std. Absolute AP is never interpreted without base rate/lift and ROC-AUC.

Relation comparisons include canonical and any-valid all-query R@1/R@3/MRR, coverage, fixed baselines, retention ratios, top1 margin, and tie rate. Paired comparisons include valid-pair original-over-cross accuracy, threshold accuracy, PR-AUC, base rate, and margin distribution.

Fixed decomposition strata are sequence, topology gap bucket, crowd density feature 142, appearance-mapped chunk fraction, trusted purity, source/destination frame density, canonical target present/missing, relation score saturation/tie, and boundary duplicate multiplicity.

Fixed buckets:

- crowd density: `[0,0.25)`, `[0.25,0.50)`, `[0.50,1.00)`, `[1.00,2.00)`, `[2.00,5.00]`;
- trusted purity: `[0.80,0.90)`, `[0.90,0.99)`, `[0.99,1.00)`, `1.00 exact`;
- appearance-mapped fraction: `0`, `(0,0.50)`, `[0.50,0.90)`, `[0.90,1.00)`, `1.00 exact`.

Frame-density buckets are fixed before results as sequence-domain quartiles computed from all trusted chunks in that sequence, with duplicate quantile edges collapsed and labeled by numeric interval. This affects decomposition only and never queries, candidates, ranking, or primary metrics.

Relation-score saturation is `abs(relation_logit) >= 20`; a query-level top tie is exact equality of the top two finite logits after stable ordering. Boundary multiplicity strata are exact counts `1`, `2`, `3`, and `4+`.

## Decisions

`measurement_integrity_decision` is one of `PASS_METRIC_DEFINITIONS`, `FAIL_METRIC_DEFINITION_BUG`, `FAIL_HISTORICAL_IMPLEMENTATION_PROVENANCE`, or `FAIL_INPUT_REVERIFICATION`. Priority is input reverification failure, then demonstrated metric-definition bug, then unavailable historical byte-exact provenance, then pass.

M23-65 gate stability applies the original fixed boundary thresholds to target corrected unique-transition primary metrics:

- macro PR-AUC >= 0.283;
- macro precision@actual >= 0.35;
- macro recall@95P >= 0.05;
- every-sequence precision@actual >= 0.20.

It is `STABLE_FAIL`, `WOULD_PASS_UNDER_CORRECTED_METRIC`, or `INDETERMINATE`; it never edits M23-65.

The unique `scientific_primary_failure` follows this fixed priority:

1. If source MOT17-11/13 fails the same boundary reference, `source_boundary_capacity_failure`.
2. Else if source passes and target fails, `target_boundary_transfer_failure`.
3. Else if target any-valid pooled coverage is below 0.80, or at least two target sequences are below 0.80, `candidate_coverage_bottleneck`.
4. Else if coverage is at least 0.80 and target any-valid all-query R@1 is below 50% of source with absolute decline at least 0.20, `relation_ranking_transfer_failure`.
5. Else `mixed_or_inconclusive`.

If corrected metrics change the M23-65 boundary branch from fail to pass, `overall_primary_classification=metric_definition_bug`; otherwise it equals the unique scientific primary failure. Historical source drift is always a separate critical limitation.

## Closure invariants

Final scope counters are fixed to zero for training, optimizer steps, checkpoint outputs, tracker outputs, TrackEval, HOTA, MOT20 test reads/submissions, teacher/held-outer reads, warm starts, v2 checkpoint loads, M23-54/M23-58, and policy starts. `hota=null` and `next_policy_authorized=false`. Every summary stage must be closed as completed, failed, skipped, or closed; no effective running/pending row may remain. The registry ends with `current_stage=closed` and a complete result/run/summary reference.
