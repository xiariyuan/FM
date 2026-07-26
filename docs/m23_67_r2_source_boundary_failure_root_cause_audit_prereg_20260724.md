# M23-67-R2 — Process Identity Repair and Deterministic Reproduction — Preregistration

Date: 2026-07-24

## Identity

M23-67-R2 is a confirmatory post-hoc diagnostic repair following two immutable failed predecessors:

- M23-67 failed because row-index and line-index semantic distinction was incorrectly tested as numerical inequality.
- M23-67-R1 repaired the semantic join and completed all scientific stages, but failed validation because its process detector treated the parent orchestration shell as an active experiment runner.

M23-67-R2 does not modify or reopen either predecessor. It independently regenerates the scientific payload from the frozen R62–R66 inputs, compares it with the frozen R1 provisional payload, repairs only process identity classification, and closes with no policy or training authorization.

Fixed declarations:

- post_hoc_diagnostic_only=true
- uses_frozen_gt_derived_label_sidecars=true
- not_deployable=true
- not_a_strict_result=true
- training_authorized=false
- next_policy_authorized=false
- hota=null
- m23_68_started=false
- m23_68_authorized=false

## Immutable inputs

The complete M23-67-R1 input manifest, artifact manifest, implementation, test, preregistration, result, failure record, closure records, scientific payload, and their transitive frozen predecessor inputs are read-only inputs. The R62–R66 scientific sources remain the only inputs used to regenerate mapping, labels, populations, scores, strata, and diagnosis. R1 scientific artifacts are comparison references only and are not copied into the R2 result.

All input SHA-256 values are frozen during init and reverified before scientific work, during validation, and at closure.

Raw MOT17 GT, raw MOT20 GT, MOT20 test, teacher actions, held-outer data, and any mutable external source are forbidden.

## Allowed implementation difference

Only the following R2 changes are allowed:

1. experiment identity, paths, manifests, and structured-record plumbing;
2. process identity collection and classification;
3. process-classifier tests;
4. deterministic comparison against the frozen R1 scientific payload;
5. validation requirements that include the reproduction report.

All mapping, label, aggregation, population, score, strata, capacity, and diagnosis functions are imported directly from the frozen R1 implementation. Their normalized AST hashes are frozen in the R2 implementation manifest. Any change in the frozen R1 base SHA or scientific function hashes is an implementation failure.

## Process identity contract

Process identity is derived from structured Linux proc records: PID, PPID, state, executable, null-separated argv, and process start time. The classifier must not search an entire shell command line for substrings.

The current process and its recursively discovered ancestor chain are orchestration processes and are nonblocking. A bash, sh, dash, zsh, fish, or timeout process is not a runner merely because its command payload mentions an M23, TrackEval, or tracker path. A real sibling or independent Python process is blocking only when its executed script token identifies M23-59 through M23-69, TrackEval, or a tracker. Direct TrackEval or tracker executables are also blocking. Zombie and disappeared processes are nonblocking.

Every operational snapshot contains two samples separated by exactly one second. Validation passes only when both samples contain zero blocking processes, zero GPU compute processes, and zero GPU memory use. Ignored ancestor processes and nonblocking textual mentions are retained as diagnostics.

## Frozen process tests

Before implementation freeze, tests must verify:

- parent shell text is ignored;
- current process and ancestors are ignored;
- an independent Python M23 runner is detected;
- a TrackEval runner is detected;
- unrelated Python is ignored;
- zombie processes are ignored;
- a live sibling R2 fixture is detected and then fully reaped;
- zero-valued GPU placeholder rows are ignored;
- nonzero GPU compute rows are detected;
- no fixture remains after test completion.

The R1 row-index, line-index, label, aggregation, bucket, decision-priority, and fail-close tests remain mandatory.

## Scientific protocol

The scientific stage order and definitions remain frozen from R1:

1. semantic row-index mapping audit;
2. boundary label audit;
3. training, validation, and full-audit population audit;
4. score distribution and orientation audit;
5. fixed stratified boundary audit;
6. fixed A greater than B greater than C greater than D diagnosis.

The physical transition key is sequence, source row index, and destination row index. The primary score is sigmoid of the arithmetic mean finite boundary logit. Mean probability is sensitivity only. No aggregation is selected from results.

Population mismatch remains frozen when any preregistered R1 criterion is true, including a training-to-audit or validation-to-audit positive-rate ratio above 5 or below 0.2. Score collapse requires pooled corrected ROC-AUC between 0.45 and 0.55 together with low logit variability, extreme ties, or fewer than 100 distinct scores. Source capacity references remain macro PR-AUC 0.283, macro precision-at-actual 0.35, macro recall-at-95-percent-precision 0.05, and every-sequence precision-at-actual 0.20.

No threshold, score orientation, bucket, population rule, gate, candidate topology, or decision priority may be changed.

## Deterministic R1 reproduction

R2 independently regenerates the scientific artifacts from R62–R66 and compares them with the frozen R1 provisional payload.

For JSON, experiment identifiers and timestamps are removed before recursive comparison. Integers, booleans, strings, keys, list order, and categorical decisions must match exactly. Floating-point absolute tolerance is 1e-12 with zero relative tolerance.

CSV and Parquet artifacts are compared as ordered data frames after removing any experiment-id column. Columns, row order, integer values, booleans, strings, and categories must match; floating-point absolute tolerance is 1e-12 with zero relative tolerance.

A mismatch closes R2 as FAIL_REPRODUCTION. A frozen implementation or process-classifier failure closes R2 as FAIL_IMPLEMENTATION. Neither failure authorizes a policy or follow-up experiment.

If every scientific artifact reproduces and validation succeeds, R2 reports the diagnosis computed by the unchanged frozen decision logic. The expected R1 provisional classification is disclosed but is not hard-coded: boundary_population_mismatch with overall FAIL_BOUNDARY_POPULATION_SHIFT and source_boundary_capacity_failure as secondary.

## Required stages

The structured summary contains:

1. init
2. verify-inputs
3. run-mapping-audit
4. run-label-audit
5. run-population-audit
6. run-score-distribution-audit
7. run-stratified-boundary-audit
8. diagnose
9. reproduce-r1-scientific-payload
10. validate
11. summarize
12. closed

Every stage is recorded in summary.csv and protocol_events.jsonl. The central experiment registry receives one running row that is updated to completed/closed or failed/closed.

## Absolute prohibitions

The following remain zero:

- training runs and optimizer steps;
- checkpoint outputs and modifications;
- tracker outputs;
- TrackEval runs;
- HOTA evaluations;
- raw MOT17 or MOT20 GT reads;
- MOT20 test reads or submissions;
- teacher and held-outer reads;
- threshold searches, calibration fits, temperature scaling, or score reversal;
- candidate, gap, gate, risk, or policy changes;
- M23-54, M23-58, or M23-68 starts.

No Git commit, push, reset, clean, checkout-based restoration, or unrelated workspace cleanup is performed. Notion writeback is not claimed.

## Closure

Before closure, R2 reverifies every frozen input, verifies the R1 predecessor remains failed/closed and unchanged, validates the deterministic reproduction report, confirms all scope counters are zero, records two-sample process and GPU state, closes the summary and registry, writes final summary and result documents, creates an artifact SHA manifest, recomputes every artifact SHA, and performs an independent persisted-state closure validation.

M23-67-R2 never authorizes M23-68 automatically.
