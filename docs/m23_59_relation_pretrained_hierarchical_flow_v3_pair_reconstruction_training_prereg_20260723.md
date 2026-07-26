# M23-64 preregistration — frozen-pool pair reconstruction and from-scratch relation training

Frozen before any M23-64 corrected example construction or optimizer step. Git HEAD is recorded in the input manifest. R62 and R63 are immutable inputs.

## Scope
Stage A reconstructs paired examples from the complete frozen R63 pool. Stage B starts automatically only if every Stage A gate passes. This experiment never reads MOT20 GT, teacher action, held-outer labels or MOT20 test; never generates a tracker; never runs TrackEval or HOTA; never loads an M23-59 v2 checkpoint; and never starts M23-54/M23-58.

## Frozen inputs
Required contract hash: `90cfab7d3a1fc87cc46b26441d3d883b9fc72f27d8852d1e2074b00e628428f5`. R63 must be closed as `FAIL_EXAMPLE_VALIDATION` with exactly one failed scientific gate: `train_minimum_support`; train paired support must be 2<5, while join semantics, topology, provenance, split isolation, unknown handling and scope guards pass. Every declared R62/R63 artifact is rehashed before Stage A and again at closure.

## Stage A diagnostic
Trust is fixed at known rows >= 2, majority purity >= 0.80; identity keys must be sequence-qualified. All 140,000 rows of `paired_candidate_pool.parquet` are labeled. For edge1 a->b and edge2 c->d, `valid_paired_positive` is exactly: all four endpoint chunks trusted; a.identity=b.identity; c.identity=d.identity; a.identity!=c.identity; and edge1, edge2, cross1 and cross2 exist in the pre-label frozen candidate pool. No pair or edge may be created, replaced or searched outside the pool.

The prior diagnostic is frozen only as a replication target, not as an input label: train=182, validation=99; per sequence `{"MOT17-02": 42, "MOT17-04": 17, "MOT17-05": 17, "MOT17-09": 55, "MOT17-10": 51, "MOT17-11": 63, "MOT17-13": 36}`. Any mismatch closes as `FAIL_PAIR_DIAGNOSTIC_REPLICATION` without changing rules.

Corrected examples contain every unique valid pair exactly once in pair_id order, without random choice, oversampling or duplication. M23-63 node and relation NPZ members and metadata must remain byte-exact; only pair_x, pair_mask and paired metadata change. Each NPZ member is loaded once into memory for full provenance validation. Stage A records wall time, rchar delta and peak RSS; TB-scale repeated member reads fail the performance guard.

## Stage A gates
M23-63 input closure and SHA valid; diagnostic exact; 182/99 pair counts and nonzero support in every physical sequence; train pair support >=5 and validation >=1; node/relation byte-exact; all pair IDs and all four edges frozen; no topology or input mutation; tensors finite; prefix masks; full row and pair provenance; physical split isolation; no unknown-as-negative; all scope counts zero; no checkpoint loaded; training not started.

## Stage B frozen protocol
From scratch only: `from_scratch=true`, `v2_checkpoint_reuse=false`, `warm_start=false`. Model is v2 `HierarchicalRelationEncoder`, exact parameter count 881124. Seeds `[2359001, 2359002, 2359003]`, epochs=30, optimizer AdamW, LR=0.0003, weight decay=0.0001, node/relation batch sizes=256/256, gradient clipping 5.0, gap buckets `[('1-30', 1, 30), ('31-90', 31, 90), ('91-180', 91, 180), ('181-600', 181, 600)]`. Loss weights are `{"boundary_count_consistency": 0.2, "catastrophic_risk": 1.5, "conditional_boundary_focal": 1.0, "edit_sparsity_source_anchor": 0.02, "incoming_listwise": 0.5, "node_focal": 1.0, "outgoing_listwise": 0.5, "paired_replacement": 0.5, "sequence_group_dro": 0.5}`. Group-DRO update, class-balanced focal losses, listwise relation losses, catastrophic-risk terms, edit sparsity, batch permutation and modulo cycling are inherited exactly from v2.

The unchanged frozen relation example set is adapted into v2 triplets deterministically. Each labeled positive edge uses the highest frozen candidate_score labeled negative with the same source as outgoing negative and the highest score labeled negative with the same destination as incoming negative; ties use candidate_id. No candidate is added and no GT search occurs.

Checkpoint selection composite is fixed to: 0.30 node PR-AUC + 0.35 conditional-boundary PR-AUC + 0.25 mean(outgoing R@1,incoming R@1) + 0.10*(1-catastrophic false-link rate). Within a seed, ties select earlier epoch; across seeds, ties select lower seed then earlier epoch. No HOTA/MOT20 metric participates.

The v2 source defines selection/health metrics but no separate external acceptance threshold. Therefore Stage B requires all 90 epoch records complete, finite, reloadable, exact parameter count, stable example/checkpoint SHA, calculable validation metrics and nonconstant score outputs. No post-hoc numerical acceptance threshold may be invented.

## Decisions
Stage A failure closes using the first applicable root cause: `FAIL_M23_63_INPUT_REVERIFICATION`, `FAIL_PAIR_DIAGNOSTIC_REPLICATION`, `FAIL_CORRECTED_PAIR_RECONSTRUCTION`, `FAIL_NONPAIRED_INVARIANCE`, `FAIL_TENSOR_PROVENANCE`, `FAIL_SPLIT_LEAKAGE`, `FAIL_SCOPE_GUARD`. Stage B failure closes as `FAIL_TRAINING_NONFINITE`, `FAIL_TRAINING_INCOMPLETE`, `FAIL_CHECKPOINT_SELECTION`, `FAIL_EXTERNAL_VALIDATION_HEALTH`, or `FAIL_SCOPE_GUARD`. Full pass is `PASS_V3_FROM_SCRATCH_RELATION_TRAINING` and authorizes only a new M23-65 to reverify checkpoint/observable SHA and run the MOT20 representation gate.
