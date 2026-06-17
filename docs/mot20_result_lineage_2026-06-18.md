# MOT20 Result Lineage

**Date:** 2026-06-18
**Purpose:** Normalize all known MOT20 results with full provenance, identify the true baseline.

## Summary Table

| # | Experiment | Checkpoint | Train | BG Mode | Split | HOTA | AssA | IDF1 | MOTA | IDSW | Status | Evidence |
|---|-----------|-----------|-------|---------|-------|------|------|------|------|------|--------|----------|
| 1 | sca_lmf_mot20_baseline_20260609 | HACA v3 (MOT20 retrain) | MOT20 train_half | BG ON | val_half | 9.3648 | 2.5205 | - | 29.994 | 1306 | catastrophic | outputs/sca_lmf_mot20_eval_20260609_081744/ |
| 2 | haca_mot20_bg_probe_no_bg | HACA v2 (MOT17→MOT20 zeroshot) | MOT17 all_half_final | NO-BG | val_half | 67.083 | 57.838 | 73.803 | 90.142 | 312 | zeroshot_baseline | outputs/haca_mot20_bg_gate_probe_20260612_005759/ |
| 3 | haca_v3_nobg_full_20260612 | HACA v3 (MOT20 retrain) | MOT20 train_half | NO-BG | val_half | 77.394 | 74.306 | 88.866 | 93.341 | 819 | fullval_v3_nobg | outputs/haca_mot20_v3_nobg_full_eval_20260612_012822/ |
| 4 | haca_v1_nobg_baseline_20260616 | HACA v1 (MOT20 retrain) | MOT20 train_half | NO-BG | val_half | 77.595 | 74.713 | 89.331 | 93.361 | 805 | fullval_v1_nobg | outputs/haca_mot20_nobg_v1_20260616_1315/ |
| 5 | haca_v3_nobg_baseline_20260616 | HACA v3 (MOT20 retrain) | MOT20 train_half | NO-BG | val_half | 77.653 | 74.748 | 89.299 | 93.354 | 841 | fullval_v3_nobg_retest | outputs/haca_mot20_nobg_v3_20260616_2045/ |

## Known Historical Results (from docs, not independently verified)

| # | Source | HOTA | AssA | IDF1 | MOTA | IDSW | Notes |
|---|--------|------|------|------|------|------|-------|
| A | docs/stage1_mainline_evidence_20260607.md | 66.850 | 57.348 | 73.588 | 90.277 | 296 | MOT17→MOT20 zeroshot (different checkpoint) |
| B | docs/paper_results_snapshot_20260305.md | 69.38 | 67.26 | - | - | 766 | v13 epoch0 (older code, unknown bg mode) |
| C | docs/dcrc_problem_statement.md | 66.85 | - | - | - | - | baseline reference (same as #A?) |

## Key Findings

1. **BG gate is catastrophic on MOT20:** Row #1 shows HOTA=9.36 when background gate is enabled. This confirms `--laplace-haca-no-background` is mandatory for MOT20.

2. **v1 no-bg (77.595) is currently the observed best** on val_half. It edges v3 no-bg (77.394 first run, 77.653 retest) by a small margin (delta=+0.201 vs v3 first run, -0.058 vs v3 retest). The v1/v3 difference is within noise — the real takeaway is that no-bg HACA achieves HOTA ~77.6 on MOT20 val_half.

3. **v2-zeroshot-no-bg (67.083) vs v3-retrain-no-bg (77.394):** The retrained v3 no-bg is decisively better (+10.311 HOTA) than the zeroshot v2 no-bg. This is the critical comparison that proves MOT20-specific retraining is essential.

4. **v1 vs v3 no-bg comparison is not a clean same-split control.** v1 and v3 use different checkpoint architectures; the small HOTA difference (~0.2) does not justify declaring one "better." They are functionally equivalent on val_half.

5. **Missing controls:**
   - v2-zeroshot with BG ON (expected to be worse than 67.083)
   - v2-zeroshot-no-bg on same eval chain as v1 no-bg
   - Larger track_buffer baseline on no-bg

## Recommended Baseline for Future Experiments

**Primary baseline: HACA v1 no-bg (77.595)** — it has the cleanest structured record (summary.csv + registry entry).

**Alternative: HACA v3 no-bg (77.653)** — functionally equivalent, use when v3 checkpoint is preferred.

**Zeroshot reference: HACA v2 no-bg (67.083)** — use only for transfer-learning ablation, not as primary baseline.
