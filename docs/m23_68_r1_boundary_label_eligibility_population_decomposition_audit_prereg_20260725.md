# M23-68-R1 Boundary Label-Eligibility and Population Decomposition Audit — Repair Preregistration

Date: 2026-07-25

Status: preregistered before creation of any M23-68-R1 run root, registry row, or result artifact.

## Predecessor state

M23-68 is immutable and validly closed as FAIL_IMPLEMENTATION. Its scientific stages stopped during population-identity validation. The recorded join counts were exact:

- train: both=5662, left_only=0, right_only=0;
- validation: both=1281, left_only=0, right_only=0.

The frozen implementation nevertheless failed because pandas categorical value_counts retained the zero-count left_only/right_only categories and the code required len(join_counts)==1. That length condition does not test join integrity.

M23-68-R1 must freeze and rehash the complete M23-68 artifact/input closure plus all original frozen inputs. It never modifies or resumes M23-68.

## Sole allowed repair

The population-identity join predicate is replaced by the explicit semantic condition:

- both equals source_window_count;
- both equals node_example_count;
- left_only equals zero;
- right_only equals zero.

Missing keys are interpreted as zero. No other scientific function, label definition, input, threshold, population unit, score, classification priority, or closure rule may change. M23-68-R1 imports the immutable M23-68 implementation and overrides only this command plus predecessor/input plumbing required for a new root.

Synthetic tests must include categorical mappings with explicit zero-count categories, true left-only failure, true right-only failure, and deficient both-count failure.

## Scientific protocol

All scientific definitions and fixed thresholds are exactly those in docs/m23_68_boundary_label_eligibility_population_decomposition_audit_prereg_20260725.md:

- M23-63-native labels use matched endpoints;
- M23-66-strict labels additionally exclude distractor, ambiguity, and tie endpoints;
- unique physical transition is primary;
- frozen examples, node_label>=0 optimizer subset, overlapping-window weighting, and sequence composition are decomposed separately;
- the original M23-67-R2 ratio is reproduced at absolute tolerance 1e-15;
- sequence composition, optimizer-adapter sampling, and duplicate weighting use the frozen thresholds;
- overall priority is implementation failure, then cross-protocol label-eligibility mismatch, then the harmonized scientific component;
- score-capacity references are diagnostic only.

## Scope

The following remain zero: training, optimizer steps, checkpoint outputs/modifications, new model inference, tracker output, TrackEval, HOTA, raw MOT17/MOT20 GT, MOT20 test, teacher, held-outer, threshold search, calibration, score reversal, policy, M23-54, and M23-58. HOTA is null. No next policy or M23-69 is automatically authorized.

## Closure

M23-68-R1 must maintain summary.csv, protocol_events.jsonl, a running then terminal central registry record, frozen input and implementation manifests, predecessor closure verification, all scientific artifacts, validation, final summary/result, closure and independent closure, and artifact SHA validation. Real process/GPU state and persisted records are checked together. Any post-freeze implementation failure closes this new root without same-root repair.

No Notion writeback is claimed.
