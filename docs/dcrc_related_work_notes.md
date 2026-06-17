# DCRC + AGSA Related Work Notes

## Date: 2026-06-06

## Our Position

**Core claim**: The same HACA reliability score (activation, margin, entropy) has different semantic meaning under different crowd densities. Our method calibrates the commit/abstain decision conditioned on local density and ambiguity, rather than rewriting association results.

**Key differentiator from all prior work**: We do NOT do local rewrite / re-ranking / re-assignment. We only decide whether to commit or abstain from the host tracker's existing association.

## Comparison Table

| Method | Year | Core Mechanism | Density-Aware? | Selective Association? | Local Rewrite? | Carrier |
|--------|------|---------------|----------------|----------------------|----------------|---------|
| **UTrack** | ECCV 2022 | Uncertainty-aware tracking with probabilistic detection model | No | No | No | Standalone |
| **UncertaintyTrack** | 2023 | Kalman filter uncertainty propagation for association | No | No | No | ByteTrack variant |
| **DeconfuseTrack** | ECCV 2023 | Deconfusion of crowded scenes via feature disambiguation | Implicit (crowded handling) | No | Yes (feature reranking) | Standalone |
| **PKF** | 2023 | Prioritized Kalman Filter with confidence-gated association | No | Partial (confidence threshold) | No | BoT-SORT variant |
| **TrackTrack** | 2024 | Track-to-track association with temporal consistency | No | No | Yes (track merging) | Standalone |
| **MOTIP** | 2024 | ID prediction with set-level matching | No | No | Yes (set prediction) | Standalone |
| **PlugTrack** | 2024 | Plug-and-play learned association module | No | Yes (learned accept/reject) | No (plug-in) | Modular |
| **TCEI** | 2024 | Temporal confidence estimation with iterative refinement | No | Partial | Yes (iterative) | Standalone |
| **Dual-level Adaptation** | 2024 | Domain adaptation for MOT (detection + association) | No | No | No (domain transfer) | BoT-SORT |
| **CHTracker** | TASE 2026 | Crowd-aware tracking with head estimation | **Yes** (crowd density) | Unknown | Unknown | Unknown |
| **FC-Track** | 2025 | False correction tracking with reliability gating | No | Yes (gating) | Partial | BoT-SORT |
| **DCRC+AGSA (ours)** | 2026 | Density-conditioned calibration + ambiguity-gated abstention | **Yes** (explicit density features) | **Yes** (commit/abstain) | **No** (we never rewrite) | BoT-SORT+HACA v3 |

## Key Differentiators

### vs CHTracker (closest competitor)
- CHTracker likely uses crowd density as input to a learned tracker. We use density as a **calibration conditioning variable** for an existing high-quality carrier (HACA v3).
- CHTracker likely modifies the association mechanism. We **keep association unchanged** and only add a post-hoc commit/abstain gate.
- Our abstention is not "don't match" — it's "don't commit to this particular host edge, let newborn/reentry handle it."

### vs PlugTrack
- PlugTrack is a plug-and-play learned association module. We are a plug-and-play **calibration module** that sits on top of an existing association.
- PlugTrack likely does learned accept/reject at the pair level. Our commit/abstain is density-conditioned and operates on the host's existing decision.

### vs FC-Track
- FC-Track does false correction tracking. We do **false prevention** — abstaining before a wrong commit, rather than correcting after.
- FC-Track likely needs to identify and fix errors post-hoc. We prevent errors by conditioning the commit threshold on local context.

### vs RGSA (our own previous approach)
- RGSA tried rewrite/defer/reject with a learned Stage2. Failed because rewrite signal was too sparse (7/1238).
- DCRC+AGSA explicitly **forbids local rewrite**. Only commit/abstain.
- RGSA Stage1 soft deferral (+0.279 HOTA) is the starting point. DCRC adds density conditioning to make it robust across domains.

## What We Need to Verify for CHTracker

- [ ] Does CHTracker use density as a calibration variable or as a feature?
- [ ] Does CHTracker do selective association (commit/abstain)?
- [ ] Does CHTracker modify the host tracker's association or add a gate?
- [ ] What carrier does CHTracker use?

## Open Questions

1. Is the density-conditioned calibration the right framing, or should we call it "scene-adaptive thresholding"?
2. Should density be estimated per-frame or per-det-track-pair?
3. How does our abstention interact with BoT-SORT's secondary association (low-score matching)?
