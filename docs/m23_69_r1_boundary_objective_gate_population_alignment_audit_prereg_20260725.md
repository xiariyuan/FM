# M23-69-R1 Boundary Objective–Gate Population Alignment Audit Repair — Preregistration

## Repair status

M23-69-R1 is a new run root and a two-predicate bookkeeping/lifecycle repair over immutable failed M23-69. The original M23-69 remains `FAIL_IMPLEMENTATION`; it was not modified, resumed, or scientifically rerun.

Original M23-69 failed during `verify-inputs` before any scientific stage. All 274 frozen input SHA values matched. Its frozen `predecessor_check` nonetheless required the current M23-69 registry row to remain absent after `registry_start`, so `all_passed=false` with `first_mismatch=null`. Its fail-close registry update then encountered a second schema bug because the frozen `registry_close` attempted to add unsupported lowercase `hota` and `result` dictionary keys. A dedicated reconciliation performed no scientific work and closed the original registry/summary/closure/artifact bookkeeping as failed.

## Immutable identity

- Experiment ID: `M23-69-R1`
- Run root: `outputs/mot20_m23_20260718/m23_69_r1_boundary_objective_gate_population_alignment_audit`
- Script: `scripts/m23_research/m23_69_r1_boundary_objective_gate_population_alignment_audit.py`
- Script SHA-256: `e905ab852622178cba0f46225c82bda5f57a24fb6eb250be700f0aebbf0b580d`
- Production-path test: `scripts/m23_research/test_m23_69_r1_boundary_objective_gate_population_alignment_audit.py`
- Test SHA-256: `9acb834f29ab043922a6121e44543efaf1998dee2e15fd5775eee5995d8aef64`
- Immutable base M23-69 script SHA-256: `f04245c335401010056f04d8ed8d5a45e22bbdc0ad5cfa4fef7b4b5d5e0dd234`
- M23-69 failure reconciler SHA-256: `7006f39c48d35e628a256e19eae498405e4e8f58b9aa67148a52321a0f770b7b`
- M23-69 closed artifact manifest SHA-256: `312e1f47defe26c30f6bed00c852fbadf547577d4ab496297c4da9a062b0e49d`
- Git HEAD at repair preflight: `a04dfa0012114c1663ea1cb543158bc4d1d975a5`

The R1 script, test, and this preregistration become immutable at the R1 `implementation_frozen` event. Any later implementation defect must fail-close R1 and cannot be fixed in this root.

## Sole repairs

R1 changes exactly two lifecycle predicates:

1. **Registry lifecycle check**: before initialization, the R1 registry row may be absent; after `registry_start`, exactly one R1 row may exist in a valid `running`, `completed`, or `failed` lifecycle state. The predecessor check no longer treats the expected current R1 row as an occupied experiment ID.
2. **Registry schema close**: closing R1 writes only keys present in the actual central registry header. It does not add unsupported lowercase `hota` or `result` keys.

These repairs are exercised by production-path tests. They do not replace a scientific predicate with a constant and do not change model inference, tensor masks, label definitions, metrics, aggregation, thresholds, classification priority, scope, or result interpretation.

## Inherited scientific protocol

The complete scientific protocol is inherited byte-for-byte in implementation from:

`docs/m23_69_boundary_objective_gate_population_alignment_audit_prereg_20260725.md`

The inherited protocol remains binding, including:

- exact reconstruction of full, auxiliary, conditional, pure, and node-unknown populations;
- M23-63-native matched-endpoint labels as primary and strict labels only as sensitivity;
- exact conditional focal/checkpoint-selection mask reconstruction;
- explicit audit of count consistency, sparsity, and unknown-label clamping;
- fixed hierarchical score `sigmoid(node_impurity_logit) * sigmoid(conditional_boundary_logit)`;
- `sigmoid(mean(logit(score)))` unique-transition aggregation;
- exactly three retained seed-best checkpoint loads and CPU validation inference runs;
- counterfactual composites that replace only the historical boundary PR-AUC term;
- no all-epoch winner claim unless 90 distinct epoch weight states exist;
- inherited gate and diagnostic thresholds;
- inherited classification priority;
- all hard integrity checks and fail-close rules.

## Additional predecessor freeze

R1 additionally freezes and verifies the failed original M23-69:

- `status=failed`, `current_stage=closed`, `decision=FAIL_IMPLEMENTATION`;
- exact registry-lifecycle failure identified;
- all original frozen inputs unchanged;
- no original scientific output stage executed;
- summary contains no stale running/pending row;
- registry is `failed/closed`;
- closure integrity and independent closure pass;
- artifact manifest validation passes;
- HOTA is null, no next policy is authorized, and M23-70 was not started.

The original M23-69 script, test, preregistration, result, reconciliation records, complete run root, artifact manifest, and all original frozen input records become R1 inputs.

## Scope

The sole nonzero scientific execution counts remain:

- `retained_checkpoint_loads=3`;
- `retained_checkpoint_inference_runs=3`;
- `retained_checkpoint_validation_windows=3843`.

Training, optimizer steps, checkpoint outputs/modifications, tracker generation, TrackEval, HOTA, raw MOT17/MOT20 GT reads, MOT20 test reads/submissions, teacher/held-outer reads, threshold search, calibration, score reversal, and policy construction remain zero.

R1 is post-hoc, uses frozen GT-derived sidecars, is not deployable, is not a strict result, reports `HOTA=null`, leaves M23-65 unchanged as `FAIL_MOT20_REPRESENTATION_GATE`, sets `next_policy_authorized=false`, and does not start M23-70.

## Execution

```bash
python -B -u scripts/m23_research/m23_69_r1_boundary_objective_gate_population_alignment_audit.py run
```

The inherited stage order, structured `summary.csv`, protocol JSONL, central registry update, input SHA checks, result document, closure validation, independent closure, and artifact manifest are mandatory.

Notion writeback is not available and must not be claimed.
