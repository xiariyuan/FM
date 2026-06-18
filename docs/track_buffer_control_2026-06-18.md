# Track Buffer Control Experiment

**Date:** 2026-06-18
**Dataset:** MOT20-05 (3315 frames)
**Checkpoint:** HACA v1 no-bg (same across all variants)
**Purpose:** Test whether TOS verdict is valid — can larger track_buffer alone explain occlusion-related gains?

## Results

| Variant | track_buffer | HOTA | Δ HOTA | AssA | Δ AssA | IDSW | Δ IDSW | IDF1 |
|---------|:-----------:|:----:|:-----:|:----:|:------:|:---:|:------:|:----:|
| baseline (tb30) | 30 | 79.121 | — | 76.566 | — | 328 | — | 90.666 |
| tb60 | 60 | 79.191 | +0.070 | 76.727 | +0.161 | 326 | -2 | 90.902 |
| tb90 | 90 | 79.279 | +0.158 | 76.918 | +0.352 | 328 | 0 | 91.068 |

## Conclusions

1. **Larger track_buffer provides positive but minuscule gains.** Increasing from 30→60→90 adds only +0.158 HOTA. This is within run-to-run noise.

2. **IDSW does not improve.** Despite the longer memory window, IDSW = 328 at both tb30 and tb90. The -2 at tb60 is not meaningful.

3. **AssA improves modestly** (+0.352 from 30→90), suggesting slightly better identity association with longer memory — but this does not translate into IDSW reduction.

4. **TOS verdict confirmed.** The TOS analysis-only found only 28 occlusion events above threshold 0.5. If these were meaningful, larger track_buffer would also capture them. It doesn't. Both TOS and larger buffer produce negligible improvements on MOT20-05.

5. **Both TOS and track_buffer tuning are dead ends.** Track_buffer saturation at +0.158 HOTA confirms that the occlusion problem on MOT20-05 cannot be solved by simply keeping tracks alive longer. TOS analysis showed events too sparse to matter; track_buffer control proves the gap cannot be closed by buffering alone.

## Raw Data

- baseline: `outputs/tos_mot20_smoke_20260617_012506/eval/baseline/`
- tb60: `outputs/tb60_control_20260618_103823/`
- tb90: `outputs/tb90_control_20260618_122920/`
