# M23-70A Identity-Conditioned Causal Branch Capacity — Preregistration

Date: 2026-07-25

## Decision question

Can the existing M23-57 teacher capacity be preserved under an action space that is meaningfully closer to deployment: exactly three identity hypotheses per source, an eight-frame evidence horizon, branch-conditioned candidate regeneration, and an unchanged detector?

This is a **teacher-only capacity audit**, not a deployable result.

## Frozen baseline and target

- Strict host: M23-46, combined MOT20-train HOTA `79.123193`.
- Unconstrained teacher reference: M23-57 v2, combined HOTA `81.148046`.
- Capacity gate: combined HOTA `>= 80.80`.
- Every sequence must be no worse than its M23-46 fold.
- Passing authorizes M23-70B training only; it does not authorize a test submission.

## Fixed causal branch universe

- `K=3`.
- `B0`: M23-46 parent/no-op or terminate.
- `B1`: deterministic highest-scoring observable causal alternative.
- `B2`: deterministic second observable causal alternative or split/reseed.
- Decision horizon: exactly `8` frames.
- No per-sequence threshold, branch count, delay, score weight, policy, or top-k search.

A GT ownership transition is available to the capacity teacher only when both identity runs contain at least two matched observations and the second right-side confirmation is visible within eight frames of the branch decision.

The candidate graph is regenerated after the causally admissible split state. For a non-parent successor, the gap must be in `[0, 8]`. Its score may use the completed source history, but destination appearance and motion may use only rows visible by `source_last_frame + 8`. Complete future destination-tracklet descriptors are explicitly forbidden. Each source retains at most two such alternatives.

The global GT teacher is used only after this candidate graph is frozen, to measure the capacity of the action space. It is not a deployment policy.

## Integrity requirements

- Parent-ID reconstruction is byte-exact M23-46.
- Detection payload is unchanged.
- Selected path cover is one-to-one, time-forward, and acyclic.
- No model training, optimizer step, checkpoint modification, MOT20 test read, or test submission.
- All outputs are written below a new M23-70A root; no M23-46 or M23-57 artifact may be overwritten.

## Fixed decisions

**PASS:** HOTA at least 80.80, no sequence degradation, and all integrity gates pass. Authorize M23-70B.

**FAIL:** do not train the branch model. Close or mechanically redesign the branch action space before further representation work.

Machine-readable protocol SHA-256: `0e4f875ad0445088d90e9a6c9b5fe72f7656b27eb84db1cda0841cecd1b810be`
