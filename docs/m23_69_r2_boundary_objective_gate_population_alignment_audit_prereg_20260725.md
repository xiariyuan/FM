# M23-69-R2 Boundary Objective–Gate Population Alignment Audit Repair — Preregistration

## Repair status

M23-69-R2 is a new root over immutable failed M23-69 and M23-69-R1. It inherits the complete M23-69 scientific protocol and the R1 registry-lifecycle repairs. Its only new scientific-integrity repair is a fixed numerical tolerance for comparing metrics produced by different frozen inference paths.

M23-69-R1 successfully repaired registry lifecycle handling, reverified all 295 frozen inputs, and reconstructed every objective-population, label, mask, row-count, positive-count, P@actual, and R@95P check. It failed before checkpoint inventory or checkpoint inference because the selected frozen R66/R68 source-score path produced conditional PR-AUC `0.07760925017688275`, while the M64 training-time validation record stored `0.07761571889450053`. The absolute difference is `0.000006468717617780229`. Rows (`2534`), positives (`149`), precision@actual (`0.04697986577181208`), and recall@95P (`0.020134228187919462`) are identical.

R1 remains `FAIL_IMPLEMENTATION`, failed/closed, with no checkpoint load or inference, no tracker/TrackEval/HOTA, and complete closure/artifact validation. R2 does not modify or resume either predecessor.

## Immutable identity

- Experiment ID: `M23-69-R2`
- Run root: `outputs/mot20_m23_20260718/m23_69_r2_boundary_objective_gate_population_alignment_audit`
- Script: `scripts/m23_research/m23_69_r2_boundary_objective_gate_population_alignment_audit.py`
- Script SHA-256: `c81b1a3154e91457b59e4e5ee10fb0e26a1a589346fc38108495b1b74c7776b2`
- Production-path test: `scripts/m23_research/test_m23_69_r2_boundary_objective_gate_population_alignment_audit.py`
- Test SHA-256: `d2c19d36c967ea293ec98bb7261cd17bbb0bc726223353be0dc5106bcecb0829`
- Immutable base M23-69 script SHA-256: `f04245c335401010056f04d8ed8d5a45e22bbdc0ad5cfa4fef7b4b5d5e0dd234`
- Immutable R1 wrapper SHA-256: `e905ab852622178cba0f46225c82bda5f57a24fb6eb250be700f0aebbf0b580d`
- R1 failure reconciler SHA-256: `73b2bc5724b6473de8b9a0087b6708d9509b1fe7960e8f652c19b2b42c46116f`
- R1 closed artifact manifest SHA-256: `6d4c7264ab799f16f505c53aae282ba5a79490ec9339f0ead4436a32a552a797`
- Git HEAD at repair preflight: `a04dfa0012114c1663ea1cb543158bc4d1d975a5`

The R2 script, test, and this preregistration become immutable at `implementation_frozen`. Any later defect must fail-close R2 and cannot be repaired in this root.

## Fixed numerical repair

The absolute tolerance for **cross-inference historical metric hard checks** is fixed to:

`1e-4`

This tolerance applies only when comparing a recomputed metric against an M64 training-time metric generated through a different frozen inference path/device. It is applied to the boolean reproduction check, not to scores or published numbers.

The following remain exact and are not relaxed:

- input SHA values;
- sequence/window/tensor/position mapping;
- node and boundary labels;
- valid masks;
- rows and positive counts;
- duplicate-label consistency;
- state-dict and checkpoint SHA;
- parameter count;
- selected frozen node/boundary logit comparison tolerance already fixed at `2e-5` in the original protocol;
- M23-65 raw metric reproduction from the same frozen artifacts (`1e-12`);
- gate thresholds, ranking values, counterfactual composites, and final classifications.

No metric value is rounded, replaced, snapped to its historical value, or used after tolerance-based mutation. Production tests execute the real frozen source-score reproduction path and require a positive nonzero AP difference below `1e-4` while retaining the unmodified recomputed value.

## Inherited registry repairs

R2 retains the validated R1 lifecycle behavior:

1. the current R2 registry row may be absent before init or present in one valid lifecycle state afterward;
2. registry close writes only columns present in the actual registry header and does not add unsupported lowercase `hota` or `result` keys.

## Inherited scientific protocol

All scientific definitions, thresholds, and priorities remain exactly those frozen in:

`docs/m23_69_boundary_objective_gate_population_alignment_audit_prereg_20260725.md`

They include:

- exact full/auxiliary/conditional/pure/node-unknown population reconstruction;
- M23-63-native labels as primary and strict labels as sensitivity;
- the oracle-impure conditional focal/checkpoint-selection population;
- explicit count-consistency, sparsity, and unknown-label-clamping audit;
- fixed joint score `sigmoid(node_impurity_logit) * sigmoid(conditional_boundary_logit)`;
- unique transition score `sigmoid(mean(logit(score)))`;
- exactly three retained seed-best CPU inference runs over 1,281 MOT17 validation windows each;
- fixed counterfactual composite and lower-seed/earlier-epoch tie rule;
- no all-epoch winner claim when 90 epoch weight states do not exist;
- inherited source/target gate thresholds and diagnostic thresholds;
- inherited classification priority led by `boundary_objective_gate_population_mismatch` when its fixed rule triggers;
- post-hoc/nondeployable/non-strict interpretation.

## Additional predecessor freeze

R2 freezes and verifies both failed predecessors, including their scripts, tests, preregistrations, results, reconcilers, input manifests, partial artifacts, closure records, artifact manifests, and central registry rows.

M23-69 must remain failed/closed with zero scientific stages. M23-69-R1 must remain failed/closed with only partial objective-population artifacts, no checkpoint inventory, no retained-checkpoint inference, and an explicitly non-final scientific interpretation. Both must have unchanged inputs, passing closure/independent closure/artifact validation, null HOTA, and no next policy.

## Scope

The only nonzero scientific execution counts remain:

- `retained_checkpoint_loads=3`;
- `retained_checkpoint_inference_runs=3`;
- `retained_checkpoint_validation_windows=3843`.

Training, optimizer steps, checkpoint outputs/modifications, trackers, TrackEval, HOTA, raw MOT17/MOT20 GT, MOT20 test, teacher/held-outer data, threshold search, calibration, score reversal, and policy runs remain zero.

R2 reports `HOTA=null`, leaves M23-65 unchanged, sets `next_policy_authorized=false`, and does not start M23-70.

## Execution

```bash
python -B -u scripts/m23_research/m23_69_r2_boundary_objective_gate_population_alignment_audit.py run
```

The original M23-69 stage order and all summary, event, registry, SHA, validation, closure, independent-closure, and artifact-manifest requirements remain mandatory.

Notion writeback is unavailable and must not be claimed.
