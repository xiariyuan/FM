# Stage1 + TCGAU Combination Evidence

## Date: 2026-06-07

## Main Result (MOT17 val_half: seq 10, 11, 13)

| Method | HOTA | AssA | IDF1 | MOTA | IDSW | dHOTA |
|--------|------|------|------|------|------|-------|
| Baseline (HACA v3) | 73.825 | 72.740 | 86.753 | 88.695 | 41 | — |
| Stage1 only (λ=0.15) | 74.104 | 73.062 | 86.565 | 88.695 | 39 | +0.279 |
| TCGAU only (freeze=0.03) | 74.068 | 73.214 | 87.121 | 88.695 | 41 | +0.243 |
| **Stage1 + TCGAU** | **74.512** | **73.653** | **87.211** | **88.695** | **39** | **+0.687** |

Source: `outputs/tcgau_stage1_combo_20260607/summary.csv`

## Per-Sequence Breakdown

| Seq | Metric | Baseline | Stage1 | TCGAU | S1+TCGAU | Delta |
|-----|--------|----------|--------|-------|----------|-------|
| MOT17-10 | HOTA | 0.6775 | 0.6847 | 0.6834 | **0.6944** | **+0.017** |
| MOT17-10 | AssA | 0.6578 | 0.6678 | 0.6692 | **0.6821** | **+0.024** |
| MOT17-10 | IDSW | 25 | 23 | 25 | **23** | -2 |
| MOT17-11 | HOTA | 0.7670 | 0.7670 | 0.7670 | 0.7670 | 0 |
| MOT17-13 | HOTA | 0.7983 | 0.7983 | 0.7982 | 0.7982 | -0.0001 |

**All gains from MOT17-10 only. No adverse effects on MOT17-11/13.**

## TCGAU Mechanism Analysis

- Freeze triggered by `q_update <= threshold` where q_update = `app_sim * pair_rel * stability * coherence * hist_norm * margin_gate`
- `margin_gate = 1.0 if margin > 0.05 else 0.0` — this is the primary discriminant
- Freeze rate: 33/12052 = 0.27% — extremely selective
- Frozen samples characterised by: margin < 0.05, app_sim ~0.84 (vs normal ~0.92), track_gap ~3.2 (vs normal ~1.0)
- These are rare but high-impact memory pollution events

## Parameter Robustness

All combinations of λ ∈ {0.15, 0.18} × freeze ∈ {0.03, 0.05} produce identical results (HOTA=74.512). This indicates the gain is from the mechanism, not from parameter tuning.

## Go Criteria Assessment

| Criterion | Target | Result | Pass? |
|-----------|--------|--------|-------|
| HOTA ≥ +0.4 | ≥ 74.225 | 74.512 (+0.687) | ✓ |
| IDSW ≤ baseline | ≤ 41 | 39 | ✓ |
| HOTA > Stage1 only | > 74.104 | 74.512 | ✓ |
| Additive gain | S1+T > max(S1,T) | +0.687 > max(0.279, 0.243) | ✓ |

**Strong Go: PASSED**

## Pending

- MOT20-02 transfer check (running)
- Per-sequence MOT20 analysis
- Threshold boundary analysis (freeze=0.10 shows degradation)
