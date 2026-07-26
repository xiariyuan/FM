# M23-68 Boundary Label-Eligibility and Population Decomposition Audit — Preregistration

Date: 2026-07-25

Status: preregistered before creation of any M23-68 run root or result artifact. M23-68 was confirmed unused in both paths and the central registry. The user explicitly authorized this new diagnostic after M23-67-R2; predecessor statements that M23-68 was not automatically authorized remain historically true and are not modified.

## Purpose

M23-67-R2 validly closed with `FAIL_BOUNDARY_POPULATION_SHIFT`, using a train positive rate of 0.007685874970475771 and a MOT17-11/13 full-audit positive rate of 0.0011678293739819908. That comparison mixes two potentially different eligibility definitions and different physical sequence sets. M23-68 must separate, without remediation or training:

1. frozen M23-63 training-label implementation correctness;
2. M23-63-native versus M23-66-strict endpoint eligibility;
3. endpoint/example selection versus the complete frozen window population;
4. overlapping-window observation weighting versus unique physical transitions;
5. train-sequence versus validation-sequence composition;
6. boundary-score capacity under each explicitly named label definition.

M23-67, M23-67-R1, and M23-67-R2 are immutable. M23-68 may assess stability but never overwrite their classifications or artifacts.

## Frozen inputs

The run freezes and rehashes all referenced M23-67-R2 artifacts and predecessor-declared inputs, plus the M23-62 feature contract/observables, M23-63 windows/supervision/example metadata and preregistration, M23-64-R1 tensors/checkpoint/manifests, and M23-66 corrected metric protocol. The central registry is not an input because M23-68 must update it. Any missing file or SHA change fails closed.

Fixed sequences are:

- train composition: MOT17-02, MOT17-04, MOT17-05, MOT17-09, MOT17-10;
- validation composition: MOT17-11, MOT17-13.

No raw MOT17 or MOT20 GT may be opened. Only the already frozen M23-63 GT-derived row-supervision sidecar may be read.

## Scope exclusions

All of the following counters must remain zero: training runs, optimizer steps, checkpoint outputs or modifications, new model inference, tracker outputs, TrackEval runs, HOTA evaluations, raw MOT17/MOT20 GT reads, MOT20 test reads/submissions, teacher reads, held-outer reads, threshold searches, calibration fits, score reversal, policy runs, and M23-54/M23-58 starts. `hota=null`; this is post-hoc, non-deployable, and not a strict result.

The frozen R2 boundary logits are reused. No checkpoint is executed because every M23-63 window already has a frozen R2 score observation.

## Two fixed label definitions

### M23-63-native

A boundary is labeled only when both adjacent endpoints have `supervision_status=matched`. Different `gt_identity_key` is positive, equal identity is negative, and all other pairs are ignored. This exactly follows the frozen M23-63 preregistration and tensor constructor. Ambiguity/tie flags do not independently remove a row after it has status `matched`.

### M23-66-strict

A boundary is labeled only when both endpoints are matched and neither endpoint is distractor-removed, ambiguity-flagged, nor tie-flagged. Different identity is positive, equal identity is negative, and all other pairs are ignored. This exactly follows the corrected M23-66/M23-67 audit semantics.

The two definitions are never silently pooled. Every table names its definition.

## Hard implementation checks

The run fails as `FAIL_IMPLEMENTATION` if any of these exact checks fails:

- source-window and node-example IDs/row sequences are not one-to-one and byte-semantically identical;
- tensor indices are not exactly 0..N-1, masks are not prefix-valid, or node tensors differ from frozen R62 features/padding;
- any frozen `boundary_y` differs from the reconstructed M23-63-native label;
- any R2 train/validation observation label differs from frozen `boundary_y`;
- any R2 audit observation label differs from reconstructed M23-66-strict semantics;
- an expected frozen score observation is missing, duplicated, nonfinite, or has inconsistent sigmoid probability;
- duplicate observations of one physical transition disagree under either fixed definition;
- input, preregistration, script, or test hashes change after implementation freeze.

These checks distinguish an implementation defect from a protocol-definition difference. A native-known row excluded by the strict definition is not itself an M23-63 implementation failure.

## Population units

An observation is one adjacent row pair inside one frozen M23-63 source window. A physical transition key is `(sequence, src_row_index, dst_row_index)`. The primary population unit is one unique physical transition; its score is sigmoid of the arithmetic mean finite logit across duplicate windows. Raw observation-weighted results are sensitivity only.

The complete frozen-window universe contains every adjacent pair from all 6,943 frozen source windows. Frozen example construction is absent only if the M23-64 node-example metadata exactly equals this universe; this is an exact identity check, not an inferred rate comparison. The actual M23-64 optimizer adapter additionally retained only windows with `node_label>=0`; that frozen adapter subset is reconstructed separately and must not be conflated with either the complete window universe or R2's all-example summary.

## Fixed decomposition tests

### Cross-protocol eligibility integrity

`mixed_label_eligibility_definitions=true` when the historical R2 numerator uses M23-63-native labels, its audit denominator uses M23-66-strict labels, and at least one native-labeled observation is strict-excluded. In that case `measurement_integrity_decision=FAIL_MIXED_LABEL_ELIGIBILITY_DEFINITIONS`. Zero tolerance is used because the compared estimands must be identical by definition.

The original R2 mixed ratio must be reproduced within absolute tolerance `1e-15`. Harmonized native/native and strict/strict ratios are reported separately.

### Sequence-composition component

The primary test uses M23-66-strict unique physical transitions. A material sequence-composition shift requires all of:

- pooled train/validation risk ratio outside `[0.5, 2.0]`;
- absolute positive-rate difference at least `0.001`;
- two-sided Fisher exact p-value below `0.01`;
- at least 20 positives in each pooled composition;
- at least four of five train-sequence rates lie on the same side of the pooled validation rate.

No threshold is selected from M23-68 outputs.

### Optimizer-adapter sampling component

Using M23-66-strict unique transitions, compare the actual `node_label>=0` optimizer-adapter subset against the complete train-window population. A material example-sampling shift requires all of: optimizer/full risk ratio outside `[0.8, 1.25]`, absolute rate difference at least `0.0005`, at least one percent of train windows excluded, and at least three train sequences with the same signed direction.

### Duplicate-observation weighting component

For each composition, compare strict raw-observation and strict unique-transition positive rates. A material weighting shift requires an aggregate raw/unique ratio outside `[0.8, 1.25]`, absolute difference at least `0.0005`, and at least three physical sequences with the same signed direction.

### Scientific component classification

With hard checks passed, the harmonized population component is uniquely classified as:

- `multiple_population_components` if more than one of sequence composition, optimizer-adapter sampling, and duplicate weighting is material;
- `sequence_composition_shift`, `example_sampling_shift`, or `observation_weighting_shift` if exactly that component is material;
- `no_material_population_component` if none is material and support requirements are met;
- `inconclusive_population_decomposition` if required support is absent.

Endpoint/example subset selection is reported separately and must be exactly absent; a mismatch is an implementation failure, not a post-hoc scientific class.

## M23-67-R2 stability

The frozen R2 severe positive-rate rule is reapplied without changing its threshold: ratio `>5.0` or `<0.2` is severe. The harmonized strict raw ratio yields:

- `STABLE_FAIL` if both original mixed and harmonized strict comparisons fail;
- `UNSTABLE_FAIL` if the original fails and harmonized strict does not;
- `STABLE_PASS` if neither fails;
- `REVERSED` only if the original does not fail and harmonized strict does.

Unique-transition results remain primary scientifically even though raw rates are used for exact R2 rule stability.

## Score-capacity audit

For native and strict definitions, raw and unique views report count, positives, base rate, PR-AUC, ROC-AUC, precision at actual-positive count, recall at 95% precision, positive/negative score means, and per-sequence/macro/pooled results. Stable ordering uses transition key. The historical reference thresholds 0.283 PR-AUC, 0.35 precision@actual, 0.05 recall@95P, and 0.20 minimum sequence precision are diagnostic only and cannot authorize a gate, tracker, or policy.

## Classification priority

`overall_primary_classification` follows this fixed priority:

1. `implementation_failure` if a hard check fails;
2. `cross_protocol_label_eligibility_mismatch` if mixed definitions are confirmed;
3. the harmonized scientific population component otherwise.

The harmonized component and capacity findings are always retained as separate fields, so measurement priority cannot erase scientific evidence.

## Closure

The run must create `summary.csv`, `protocol_events.jsonl`, input/implementation manifests, label reconstruction artifacts, population identity validation, decomposition CSV/JSON, score-capacity CSV/JSON, final diagnosis, validation report, final summary, closure validation, independent closure validation, artifact SHA manifest/validation, and a result document. The important run is recorded in `outputs/experiment_registry.csv` as running during execution and completed/closed or failed/closed at termination.

Before closure, inputs and implementation hashes are rechecked, summary has no running/pending rows, the registry agrees, and the robust M23-67-R2 `/proc` identity classifier samples processes/GPU twice one second apart. Parent orchestration shells, self, and ancestors are ignored; real sibling experiment/TrackEval/tracker processes remain blocking. No Notion writeback is claimed.
