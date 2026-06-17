# SCA+LMF Mainline Evidence

## Method: Selective Conservative Association with Low-Reliability Memory Freezing

### Final Main Table (MOT17 val_half: seq 10, 11, 13)

| variant | HOTA | AssA | IDF1 | IDSW | dHOTA | freeze |
|---------|------|------|------|------|-------|--------|
| baseline | 73.825 | 72.740 | 86.753 | 41 | — | — |
| Stage1 only | 74.104 | 73.062 | 86.565 | 39 | +0.279 | — |
| freeze-only TCGAU | 74.068 | 73.214 | 87.121 | 41 | +0.243 | 33/12052 |
| **Stage1 + freeze-only** | **74.512** | **73.653** | **87.211** | **39** | **+0.687** | 52/12052 |

### Mechanism Summary

- **Stage 1** (soft deferral): Selective conservative commit before Hungarian — penalizes uncertain host edges by raising their cost.
- **TCGAU freeze-only**: After match acceptance, freezes appearance memory update for extremely low-quality matches (q_update < 0.03).
- Only **33-52 out of 12,052** primary matches are frozen (0.27-0.43%). These are high-impact memory pollution points.
- Frozen samples characterised by: margin < 0.05, app_sim ~0.87, track_gap elevated.

### Per-Sequence Analysis

All gains from MOT17-10 (crowded sequence). MOT17-11/13 unchanged. Stage1 and freeze-only additively combine.

### Soft Mode Rejected

Three-tier TCGAU (with soft-thresh=0.70) was experimentally rejected after semantic fix:
- Soft mode activates on ~20% of matches, degrading appearance memory maintenance
- HOTA drops to 73.730 (TCGAU only) and 73.048 (combo) with soft enabled

### MOT20 Boundary

Cross-domain transfer to MOT20-02 fails (HOTA -4.4). Method scope limited to MOT17-style moderate-crowd. Not claimed as universal.

### Threshold Ablation

- freeze_thresh 0.01-0.09: All produce identical result (HOTA=74.068). Same 33 samples frozen.
- freeze_thresh 0.10+: Over-freezing (208+ samples), degradation begins.
- Sweet spot: freeze_thresh ∈ [0.01, 0.09], effectively any value that catches the ~33 worst samples.
