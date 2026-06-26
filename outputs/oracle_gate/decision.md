# Oracle Gate Decision

## Status

- protocol_lock: completed
- gt_alignment: completed_verified
- oracle_0a: completed_positive
- oracle_0c_inline_gt: completed_partial_trusted
- oracle_0b_smoke: completed
- oracle_0d_smoke: completed
- oracle_0e: closed (with --allow-partial-0c)

## Key Results

### Oracle 0A (State Protection)
- switch_events: 3935
- protectable_events: 287
- idsw_reduction_percent: 7.29%
- median_recovery_latency: 9.0 frames
- **Verdict: POSITIVE** (above 3% threshold for SPOT ablation)

### Oracle 0C (Cost Reranking, inline GT)
- groups_with_gt: 1803
- wrong_selected_groups: 134
- fixable_groups: 58
- fixable_percent: 43.28%
- median_positive_rank: 0.0
- **Verdict: MODERATE SIGNAL** (partial, trusted inline GT)

### Oracle 0E (Joint Decision, allow-partial-0c)
- state_gain: 7.29%
- rerank_gain_proxy: 43.28%
- decision_confidence: closed
- runtime_patch_allowed: 1
- block_reason: (none)
- final_route: SPOT_MAINLINE
- pcc_role: strong_support
- p5_role: skip
- **Verdict: CLOSED** (with partial 0C allowance)

## Decision

Runtime tracker patches are now allowed to proceed. The 0C result is trusted (inline GT) and partial, but combined with the strong 0A signal, the decision is closed.

**Final Route: SPOT_MAINLINE**

- P4 ADG-freeze / State Protection: main novelty
- PCC: strong support module
- P5 delayed commitment: skip (not required)

## Recommendation

Given:
- 0A has positive signal (7.29% IDSW reduction)
- 0C has moderate signal (43.28% fixable)
- 0C is partial but trusted (inline GT)
- 0E decision is closed with SPOT_MAINLINE

**Proceed with SPOT state protection (P4 ADG-freeze) as main novelty, with PCC as support module.**

This matches the strategic direction from the design document: "SPOT-Track = State-Protected Online Tracking under Ambiguous Association"
