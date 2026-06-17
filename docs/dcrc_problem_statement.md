# DCRC + AGSA Problem Statement

## Date: 2026-06-06

## The Problem

The same HACA reliability score (activation, margin, entropy, s_final) has **different semantic meaning under different crowd densities**. A margin of 0.1 may indicate a confident match in a sparse scene but an ambiguous one in a crowded scene.

### Evidence from RGSA experiments

1. **MOT17 positive**: Stage1 soft deferral (lambda=0.15) on MOT17 val_half gives HOTA +0.279, IDSW -2
2. **MOT20 negative**: Same Stage1 head on MOT20-02 gives HOTA -4.2, IDSW +137
3. **Root cause**: MOT17-trained Stage1 head's decision boundary does not transfer to MOT20
   - MOT17-10 reject rate: 27.6% (1436/5206)
   - MOT20-02 reject rate: 54.2% (77661/143342)

### Why rewrite-based approaches failed

| Approach | Failure Mode | Evidence |
|----------|-------------|----------|
| Learned Stage2 rewrite | rewrite positive samples = 7/1238 | outputs/rgsa_stage2_v3_formal_20260606/ |
| dual_haca_v2 cost discount | 7230/15250 rewrite on MOT20, too aggressive | outputs/rgsa_dual_haca_v2_fullval_20260606/ |
| Heuristic verifier | Vetoed 161/184 correct matches on MOT17-10 | outputs/rgsa_verifier_fullval_20260606/ |

### The insight

All rewrite/verify approaches failed because they tried to **change the host's match**. The host (HACA v3) is already very good (98.5% top-1 accuracy). The problem is not "how to find a better match" but "when to trust the host's match."

## The Approach: DCRC + AGSA

### DCRC (Density-Conditioned Reliability Calibration)

- Input: HACA runtime features (activation, margin, entropy, ...) + local density features
- Output: p(host edge is correct | features, density)
- The key insight: **calibrate the reliability score conditioned on local context**

### AGSA (Ambiguity-Gated Selective Association)

- If p >= tau_commit: commit (accept host's match)
- If p < tau_commit: abstain (let newborn/reentry handle it)
- **Never rewrite** — only commit or abstain

### Why this should work

1. The host is already good — we only need to identify the ~1.5% wrong cases
2. Density conditioning should fix the MOT17→MOT20 transfer failure
3. Abstention is safer than rewrite — worst case is a missed match, not a wrong match
4. BoT-SORT's existing newborn/reentry pipeline handles abstained detections

## Frozen Results (from RGSA experiments)

See: `outputs/rgsa_definitive_summary_20260606.csv`
See: `outputs/rgsa_failed_experiments_20260606.md`

## Success Criteria for DCRC+AGSA

1. MOT17 val_half: HOTA >= 74.104 (current stage1_only best)
2. MOT20 transfer: HOTA >= 66.85 (baseline) — no more negative transfer
3. Density-conditioned calibration > global-only calibration
4. ECE/Brier/NLL improvement over uncalibrated baseline
5. Abstention improves risk-coverage tradeoff
