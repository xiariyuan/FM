# Oracle Gate Decision

## Status

- protocol_lock: completed
- gt_alignment: completed_verified
- oracle_0a: completed_positive
- oracle_0c_inline_gt: completed_partial_trusted
- oracle_0b_smoke: completed
- oracle_0d_smoke: completed
- oracle_0e: not_closed

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

### Oracle 0E (Joint Decision)
- state_gain: 7.29%
- rerank_gain_proxy: 43.28%
- decision_confidence: not_closed
- runtime_patch_allowed: 0
- block_reason: 0C rerank oracle is not full-file
- **Verdict: NOT_CLOSED**

## Decision

Runtime tracker patches remain frozen. The 0C result is trusted (inline GT) but not full-file. To proceed:

1. Either run 0C full-file (requires significant compute time)
2. Or formally exclude 0C and proceed with SPOT state protection only

## Recommendation

Given:
- 0A has positive signal (7.29% IDSW reduction)
- 0C has moderate signal (43.28% fixable)
- 0C is not full-file but is trusted

**Recommendation: Proceed with SPOT state protection (P4 ADG-freeze) as main novelty, with PCC as support module. P5 delayed commitment remains optional extension.**

This matches the strategic direction from the design document: "SPOT-Track = State-Protected Online Tracking under Ambiguous Association"
