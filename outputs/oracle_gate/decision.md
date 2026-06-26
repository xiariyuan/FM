# Oracle Gate Decision

## ⚠️ CORRECTED (2026-06-26)

Previous version incorrectly used oracle ceiling as go/kill criterion for runtime patches.
Oracle numbers are upper bounds, NOT runtime improvements. Real paired eval required.

## Status

- protocol_lock: completed
- gt_alignment: completed_verified
- oracle_0a: completed_positive (oracle ceiling, not runtime gain)
- oracle_0c_inline_gt: completed_partial_trusted
- oracle_0b_smoke: completed (evidence_latency semantics need correction)
- oracle_0d_smoke: completed
- oracle_0e: provisional (runtime_patch_allowed=0, requires real paired eval)

## Key Results

### Oracle 0A (State Protection)
- switch_events: 3935
- protectable_events: 287
- oracle_recoverable_rate: 7.29% (ORACLE CEILING, NOT RUNTIME GAIN)
- median_recovery_latency: 9.0 frames
- **Verdict: POSITIVE DIRECTION** (oracle ceiling indicates headroom exists)

### Oracle 0C (Cost Reranking, inline GT)
- groups_with_gt: 1803
- wrong_selected_groups: 134
- fixable_groups: 58
- fixable_percent: 43.28%
- median_positive_rank: 0.0
- **Verdict: MODERATE SIGNAL** (partial, trusted inline GT)

### Oracle 0E (Joint Decision)
- oracle_recoverable_rate: 7.29% (ceiling, not runtime)
- rerank_gain_proxy: 43.28%
- decision_confidence: provisional
- runtime_patch_allowed: 0
- block_reason: requires real paired eval; oracle ceiling is not sufficient
- final_route: SPOT_PROVISIONAL
- pcc_role: strong_support
- p5_role: skip
- **Verdict: PROVISIONAL** (oracle indicates direction, but runtime patches require paired eval)

## Decision

Runtime tracker patches are NOT yet allowed. Oracle evidence indicates SPOT direction has headroom,
but the go/kill criterion must be real paired eval (HOTA/IDSW delta), not oracle ceiling.

**Final Route: SPOT_PROVISIONAL**

- P4 ADG-freeze / State Protection: candidate novelty (needs paired eval)
- PCC: support module (needs paired eval)
- P5 delayed commitment: skip

## Next Steps

1. Implement minimal P4 ADG-freeze runtime patch (appearance/history freeze only)
2. Run spot_enable=0 parity test (must equal baseline exactly)
3. Run MOT20-05 paired eval (baseline vs SPOT)
4. Only if paired eval positive: unlock runtime_patch_allowed=1
5. Only then: expand to DanceTrack/SportsMOT

## Recommendation

Given:
- 0A oracle ceiling: 7.29% (indicates headroom exists)
- 0C moderate signal: 43.28% fixable
- 0C is partial but trusted (inline GT)
- 0B evidence_latency semantics need correction
- Only MOT20-05 single sequence evidence

**Proceed with minimal P4 implementation + paired eval. Do NOT unlock runtime patches until paired eval confirms positive.**
