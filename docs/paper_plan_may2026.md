# FM-Track Paper Plan (to May 2026)

Last updated: 2026-03-05

## 1) Strategy

Goal: finish a submission-quality draft first, then do targeted score improvement without breaking evidence attribution.

Working mode:
- `Paper track` (stable): lock protocol, fill required tables/figures.
- `Score track` (aggressive): limited extra runs with strict stop rules.

## 2) Locked Protocol (Paper Track)

- Main benchmarks: `MOT17` and `MOT20`.
- Core metrics: `HOTA`, `MOTA`, `IDF1`, `DetA`, `AssA`, `IDSW`, `Frag`.
- Association validation protocol:
  - MOT17 proxy: full7 / 02+13 (as used in prior runs).
  - MOT20 proxy: `MOT20-05` holdout.
- Detector stays fixed during association analysis.
- ReID and association changes are evaluated with controlled A/B only.

## 3) What Is Already Available

- Multiple MOT17 submissions and csv summaries:
  - `mot17_metrics_tabbed_v3.csv`
  - `mot17_metrics_tabbed_v4.csv`
  - `mot17_metrics_continued.csv`
- ReID MOT20-only training complete:
  - `outputs/reid_mot20_only_osnetain_20260304_203842/reid_best.pth`
  - best val: Rank-1 `51.40`, mAP `29.86`
- Extensive parameter sweeps in `outputs/*/sweep_assoc_results.csv`.

## 4) Missing Data Before Writing "Experiments" Final

P0 (must-have):
- Final unified main-result table under one locked protocol:
  - MOT17 final
  - MOT20 final
- 2x2 attribution table:
  - Base
  - Base + ReID
  - Base + Assoc strategy
  - Base + ReID + Assoc strategy
- Efficiency table:
  - inference speed (FPS), GPU memory, params.

P1 (strongly recommended):
- Robustness over seeds (`mean ± std`, >=2 seeds).
- Error analysis by difficult sequences (high IDSW/Frag).
- 3 success + 3 failure qualitative examples.

## 5) Current Decision Log (important for paper consistency)

- v14-SFI early probe (MOT20-05, epoch0) underperformed vs v13 baseline:
  - v13 epoch0: HOTA `69.38`, AssA `67.26`, IDSW `766`
  - v14 epoch0: HOTA `68.65`, AssA `66.02`, IDSW `834`
- Action taken:
  - stopped v14 to avoid compute waste.
  - switched to `v13 + MOT20-only ReID best` controlled run.

This keeps the paper narrative clean: first prove complementarity, then revisit structure changes only if needed.

## 6) Writing Milestones

Week 1 (now):
- Complete Abstract, Intro, Related Work, Method draft.
- Freeze figure placeholders and table schemas.

Week 2:
- Fill main-result and attribution tables with current stable runs.
- Add qualitative analysis and failure cases.

Week 3:
- Add efficiency + robustness experiments.
- Full draft v1 with appendix (training details, configs).

Week 4:
- Internal review pass and response-driven fixes.
- Finalize camera-ready style experiment section.

## 7) Stop Rules (Score Track)

- Any new idea must beat baseline on proxy by at least one:
  - `HOTA +0.3`, or
  - `AssA +0.5`, with non-worsening `IDSW/Frag`.
- If not met by epoch 2-3, stop that branch.
- No new module additions after drafting deadline lock.

