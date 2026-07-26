# M23-69 Boundary Objective–Gate Population Alignment Audit — Preregistration

## Status and purpose

This document freezes **M23-69** before any M23-69 scientific metric is produced. M23-69 is a post-hoc diagnostic over immutable M23-63 through M23-68-R1 artifacts. It asks whether the M23-64 boundary optimization/checkpoint-selection population and the M23-65/M23-66 full-window representation-gate population estimate the same quantity, whether the conditional boundary head was published as an unconditional score, and whether the original checkpoint choice is stable among the only three retained seed-best weights.

M23-69 is not deployable and is not a strict tracking result. It uses frozen GT-derived label sidecars, reports `HOTA=null`, and cannot authorize a tracker, TrackEval, HOTA evaluation, threshold search, same-run training, or M23-70.

## Immutable experiment identity

- Experiment ID: `M23-69`
- Run root: `outputs/mot20_m23_20260718/m23_69_boundary_objective_gate_population_alignment_audit`
- Script: `scripts/m23_research/m23_69_boundary_objective_gate_population_alignment_audit.py`
- Script SHA-256: `f04245c335401010056f04d8ed8d5a45e22bbdc0ad5cfa4fef7b4b5d5e0dd234`
- Production-path test: `scripts/m23_research/test_m23_69_boundary_objective_gate_population_alignment_audit.py`
- Test SHA-256: `c2662775bf96228ff7faf1b42a138ea6087705418c42f2dfeb39a7065f7a86c9`
- Process-classifier helper SHA-256: `f10eec3e6d5d4256aaceb6697ed80c641db97803af1cde123224e7f55b8aa3b3`
- Frozen v2 model/objective source SHA-256: `50e12cdc6259903f9a7a76adfd3329622899c0c727f1466c2399034c49f6451d`
- Git HEAD at preflight: `a04dfa0012114c1663ea1cb543158bc4d1d975a5`

The script, test, and this preregistration become immutable when the `implementation_frozen` protocol event is written. Any implementation defect discovered afterward must fail-close M23-69 and be repaired only in a new root.

## Frozen predecessors

M23-69 freezes and rehashes the predecessor chain and directly used artifacts, including:

- M23-63 source windows and supervision labels;
- M23-64-R1 train/validation tensors, metadata, training configuration, three seed-best checkpoints, 90 epoch metric rows, checkpoint manifests, selected frozen checkpoint, and closure;
- M23-65 frozen MOT20 node/boundary scores, frozen row-label sidecars, representation metrics, failed gate, and closure;
- M23-66 frozen MOT17 validation node/boundary scores, corrected boundary metrics, diagnostic records, and closure;
- M23-68-R1 reconstructed native/strict labels, selected-checkpoint source scores, unique-transition population, population decomposition, score capacity, artifact manifest, and closure.

The central experiment registry is excluded from the frozen input hash because M23-69 must update it. No predecessor artifact may be modified.

## Fixed sequences and labels

Source train sequences:

`MOT17-02`, `MOT17-04`, `MOT17-05`, `MOT17-09`, `MOT17-10`

Source validation sequences:

`MOT17-11`, `MOT17-13`

Target diagnostic sequences:

`MOT20-01`, `MOT20-02`, `MOT20-03`, `MOT20-05`

Primary boundary labels use the **M23-63-native** definition: both adjacent endpoints must have `supervision_status=matched`; a GT identity change is positive and equal identity is negative. All other endpoints are ignored and never converted to negatives.

The **strict sensitivity** additionally excludes endpoints marked ambiguous, tied, distractor-removed, or nonmatched. Strict labels cannot replace the primary native definition because M23-65/M23-66 published their gate using matched endpoints.

## Fixed objective populations

M23-69 reconstructs the following populations from frozen tensors and source code:

1. `full`: every valid adjacent transition in every frozen window, followed by the selected label-eligibility definition;
2. `auxiliary`: every valid adjacent transition in a node-known window (`M23-63 node_y >= 0`), corresponding to the broad population seen by boundary count consistency and sparsity;
3. `conditional`: every native-known transition in an oracle-impure window (`M23-63 node_y = 0`, transformed by M23-64 to model `node_label = 1`), corresponding to the conditional boundary focal loss and the effective known-only checkpoint-selection metric;
4. `pure`: the native-known transitions in oracle-pure windows (`M23-63 node_y = 1`);
5. `node_unknown`: transitions in windows whose M23-63 node label is ignored.

M23-69 must prove directly that the optimizer focal mask and repaired checkpoint-selection mask are identical on frozen validation tensors. It must separately report that count consistency clamps unknown `boundary_y=-1` values to zero and that sparsity acts over all valid positions; those auxiliary losses must not be falsely described as conditional-only.

Raw-observation views retain duplicate-window weighting. The primary physical-transition view groups by `(sequence, src_row_index, dst_row_index)` and uses `sigmoid(mean(logit))`; duplicate labels must be consistent.

## Fixed score semantics

The v2 source names the heads `node_impurity_head` and `conditional_boundary_head`. M23-69 therefore freezes one non-tuned hierarchical sensitivity score:

`joint_probability = sigmoid(node_impurity_logit) * sigmoid(conditional_boundary_logit)`

No fitted coefficient, threshold, calibration, sign reversal, sequence-specific rule, or target feedback is allowed. For duplicate physical transitions, the primary joint score is `sigmoid(mean(logit(joint_probability)))`. Arithmetic-mean probability may be retained only as a labeled sensitivity artifact and cannot drive a classification.

The raw published score remains `sigmoid(conditional_boundary_logit)`. M23-69 compares raw and fixed joint scores on frozen source-validation and target artifacts without rerunning M23-65.

## Retained-checkpoint counterfactual

Preflight found exactly three retained training weight files, one seed-best checkpoint per seed. M23-69 will load each exactly once on CPU and run exactly one MOT17 validation inference pass over 1,281 windows. It will not run inference on MOT20 for these alternate checkpoints.

For each retained checkpoint, M23-69 must reproduce its historical conditional-boundary rows, positives, PR-AUC, precision@actual, and recall@95% precision. It then computes fixed counterfactual composites by replacing only the historical boundary PR-AUC component while retaining the frozen node, relation, and catastrophic-risk components:

`0.30 * node_PR_AUC + 0.35 * aligned_boundary_PR_AUC + 0.25 * mean(outgoing_R1,incoming_R1) + 0.10 * (1-catastrophic_false_link_rate)`

The primary retained-checkpoint counterfactual is the M23-63-native, full-window, unique-transition, per-sequence macro PR-AUC using the fixed hierarchical joint score. Ties select lower seed then earlier epoch. Raw-boundary, pooled-raw, and strict-label variants are sensitivities.

The 90 epoch metric rows are not 90 restorable weight states. Unless 90 distinct epoch weights are found, M23-69 must record `all_epoch_counterfactual_status=unavailable_only_seed_best_weights_retained` and must not claim an all-epoch winner or that the historical epoch-19 selection would definitively change.

## Fixed metrics and thresholds

Binary metrics are PR-AUC, ROC-AUC, precision at the actual positive count, recall at 95% precision, base rate, positive/negative score means, rows, positives, and negatives. Tie-breaking is deterministic.

The inherited fixed gate is:

- macro PR-AUC `>= 0.283`;
- macro precision@actual `>= 0.35`;
- macro recall@95% precision `>= 0.05`;
- every-sequence precision@actual `>= 0.20`.

The objective/gate population mismatch is material only when, in both source train and source validation, conditional native-known row coverage is below `0.50` and at least `0.10` of the full native-known population lies outside the conditional population.

A negative-support diagnosis additionally requires conditional positive coverage of at least `0.95` in both source splits.

A fixed source generalization failure requires the conditional native raw-observation PR-AUC to drop by at least `0.20` from source train to validation and the validation/train PR-AUC ratio to be at most `0.50`.

A fixed hierarchical-score gain is called material only if native full unique-transition macro PR-AUC improves by at least `0.02` on both source validation and target. This sensitivity does not authorize a gate rerun or policy even if material.

## Classification priority

The unique overall primary classification follows this frozen priority:

1. `implementation_or_measurement_failure` if any hard mapping, label, score, SHA, historical reproduction, or process/scope check fails;
2. `boundary_objective_gate_population_mismatch` if the fixed material population rule triggers;
3. `checkpoint_selection_population_mismatch` if the retained seed-best winner changes under the primary aligned composite and the population rule does not trigger;
4. `source_sequence_generalization_failure` if the fixed generalization rule triggers and neither prior scientific rule triggers;
5. `no_fixed_failure_component_identified` otherwise.

Checkpoint-selection instability and source generalization may be retained as secondary findings when a higher-priority objective/gate mismatch is primary.

## Hard integrity checks

M23-69 must fail-close if any of the following fails:

- predecessor closure and registry checks;
- frozen input SHA verification at start and finish;
- exact M23-63 node/boundary tensor reconstruction;
- exact optimizer/checkpoint population reconstruction;
- exact reproduction of the selected historical conditional metric;
- exact 30/30 finite history and manifest SHA for all three seeds;
- exactly three retained training checkpoints and one frozen duplicate;
- model parameter count `881124`, state-dict strict load, finite outputs, and exact valid masks;
- selected CPU inference agrees with frozen selected node/boundary logits within `2e-5`;
- frozen M23-65 raw boundary metrics reproduce within `1e-12`;
- duplicate transition labels are consistent;
- robust `/proc` process classifier reports no external M23/TrackEval/tracker process and GPU is idle at preflight/closure;
- structured summary has no stale `running` or `pending` row at closure;
- central registry is closed and all output artifact hashes verify.

## Scope

The only nonzero scientific execution counts allowed are:

- `retained_checkpoint_loads=3`;
- `retained_checkpoint_inference_runs=3`;
- `retained_checkpoint_validation_windows=3843`.

All of the following remain zero:

- training runs and optimizer steps;
- checkpoint outputs or modifications;
- tracker outputs, TrackEval, and HOTA;
- raw MOT17 GT reads and raw MOT20 GT reads;
- MOT20 test reads/submissions;
- teacher or held-outer reads;
- threshold searches, calibration fits, score reversals, and policy runs.

Frozen GT-derived source/target label sidecars may be read. M23-65 remains closed as `FAIL_MOT20_REPRESENTATION_GATE`. HOTA remains null. `next_policy_authorized=false` and `m23_70_started=false`.

## Execution order

```bash
python -u scripts/m23_research/m23_69_boundary_objective_gate_population_alignment_audit.py run
```

The run command must execute, in order:

1. `init`;
2. `verify-inputs`;
3. `audit-objective-population`;
4. `inventory-checkpoints`;
5. `evaluate-retained-checkpoints`;
6. `audit-gate-score-semantics`;
7. `diagnose`;
8. `validate`;
9. `summarize`;
10. `closed`.

Each stage must be recorded in queue-level `summary.csv`, protocol events in JSONL, and the central `outputs/experiment_registry.csv`. Any post-freeze implementation error must close as `FAIL_IMPLEMENTATION`; the same root cannot be repaired or rerun.

Notion writeback is outside the available connector scope and must not be claimed.
