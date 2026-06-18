# TOS Analysis-Only Verdict

**Date:** 2026-06-18
**Dataset:** MOT20-05 (3315 frames)
**Checkpoint:** HACA v1 no-bg (77.595 baseline)
**Run:** `outputs/tos_analysis_v5_20260618_084245/`

## Summary

TOS analysis-only provides evidence that the mechanism **exists but is too sparse** to drive behavioral improvement.

## 1. Is TOS working?

**Partially yes, but barely.** The `_compute_tos_occlusion_score` produces non-zero values on ALL matched pairs (due to `det_ambiguity = 1.0 - det_score` noise floor), but meaningful high-occlusion events are very rare.

| Bucket | Count | % of Total | Notes |
|--------|-------|-----------|-------|
| occlusion 0.0-0.1 | 605,445 | 98.8% | Noise floor from det_score |
| occlusion 0.1-0.2 | 6,808 | 1.1% | Minimal signal |
| occlusion 0.2-0.3 | 278 | 0.045% | Weak signal |
| occlusion 0.3-0.4 | 178 | 0.029% | Moderate |
| occlusion 0.4-0.5 | 66 | 0.011% | Strong |
| occlusion 0.5-0.6 | 24 | 0.004% | Very strong |
| occlusion 0.6-0.7 | 4 | 0.0007% | Extreme |

With default threshold (0.5): **28 freeze events in 612,803 matches (0.0046%)**.

## 2. Are events concentrated in occlusion/sparse recovery zones?

**Weakly.** The 28 events above 0.5 threshold span 20 of 33 hundred-frame windows, with max 3 events/window. There is no heavy concentration in any occlusion zone.

All 28 events share:
- gap ≥ 11 frames (track missing before reconnection)
- app_sim = 0.0 (appearance similarity unavailable with HACA v1)
- haca_active = 0.0 (competition head not available in HACA v1)
- det_score between 0.60–0.89

The occlusion score formula is dominated by `gap_factor` and `det_ambiguity`, not HACA signals.

## 3. Can gains be explained by larger track_buffer?

**Yes, almost certainly.** Track_buffer default=30 frames. The 28 TOS events all involve gaps of 11-26 frames — well within the existing lost-track recovery window (max_time_lost=30). Simply increasing track_buffer from 30 to 60 or 90 would let these tracks survive longer without needing TOS freeze.

## 4. Signal quality issues

Two meta-problems made the analysis harder than expected:

1. **HACA v1 lacks competition head.** `haca_comp_active` is always 0, so the occlusion score cannot use this signal. Only `final_sim` is available, which for long-gap reconnections is naturally low.

2. **app_sim / pair_rel always 0.** The `_safe_cosine_similarity` fallback returns 0 when `smooth_feat` is uninitialized. This means TCGAU would also unconditionally freeze everything — and v1 freeze-only results confirm this is catastrophic (-5.628 HOTA).

## 5. Judgment: Does NOT pass analysis-only gate

| Criterion | Status | Detail |
|-----------|--------|--------|
| Sufficient event count | ❌ | Only 28/612803 > thresh 0.5 |
| Dense/occlusion concentration | ❌ | Evenly spread across all frames |
| Beats larger track_buffer | ❌ | Events within existing max_time_lost |
| Not repeat of reentry memory | ❌ | Reentry already covers lost-track recovery |

## Recommendation

**TOS behavioral experiments should NOT proceed.** The analysis-only evidence fails all four pass criteria. TOS is downgraded from active research direction to **archived negative result**.

**Next direction to investigate:** Larger track_buffer as a simpler control, then RG-OT (Resolved-Gram Online Tracking) as the next mainline.
