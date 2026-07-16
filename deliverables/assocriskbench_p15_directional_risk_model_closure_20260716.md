# AssocRiskBench P15｜Directional Utility Risk Modeling Closure｜2026-07-16

## 1. Scope and locked-data guarantee

This stage continued from the frozen directional handoff locked-LOSO experiment and evaluated whether a stronger risk model could safely reopen directional edits.

- Training labels: ranks 21–100 only.
- Locked pool before exclusion: 160 directional rows.
- Previously revealed rows excluded from all new prediction stages: 4.
- Remaining locked directional rows: 156.
- New locked utility labels read: 0.
- New locked TrackEval calls: 0.
- Final selected transactions: 0.

## 2. Oracle feasibility diagnostic

All 16 train-only top20-like windows contain at least one positive executable directional edit.

- Positive-oracle windows: 16 / 16.
- Oracle best-candidate HOTA sum: +0.877901.
- Per-sequence oracle sum:
  - MOT20-01: +0.570522
  - MOT20-02: +0.238180
  - MOT20-03: +0.032115
  - MOT20-05: +0.037084

Therefore, the primary bottleneck is candidate identification and risk control, not executor feasibility or absence of positive counterfactual edits.

## 3. Nested window-conditioned conformal top-one

Artifact: `outputs/assocriskbench_p15_20260716/directional_window_conformal_topone_v1`

Protocol:

- Preserve the exact directional ranker family and seed schedule.
- Split every train sequence into four top20-like windows.
- Use pairwise cross-fitted calibration sequences inside each outer sequence fold.
- Evaluate 12 preregistered pooled/group conformal LCB and null-p certificates.

Result:

- Eligible conformal gates: 0.
- Outer OOF selections: 0.
- Locked selections: 0.
- Deployment allowed: false.
- Report SHA-256: `90fdf50c448d2f7000281247bf266d3bc1bf4641abd1a7a072d0cc62748829e7`.

Interpretation: calibrated lower bounds avoid loss only by abstaining completely. The absolute utility model does not provide useful lower-tail separation.

## 4. Nested pairwise/listwise window ranker

Artifact: `outputs/assocriskbench_p15_20260716/directional_pairwise_window_ranker_v1`

Protocol:

- Learn within-window pairwise preferences instead of cross-sequence absolute utility.
- Compare ExtraTrees and HistGradientBoosting rankers.
- Compare raw and sequence-normalized targets.
- Compare capped and tail-risk pair weighting.
- Apply train-observable abstention gates.
- Exclude all four previously revealed locked candidates before locked prediction.

Development diagnostics showed that fixed pairwise rankers can change the raw 16-window utility from strongly negative to approximately +0.10 to +0.12 and avoid the MOT20-01 −0.528063 catastrophic top-one error.

Strict nested result:

- Only the MOT20-03 outer fold selected a ranker/gate.
- Outer selected windows: 3.
- Positive windows: 2.
- Negative windows: 1.
- Outer utility: −0.039815.
- Worst sequence utility: −0.039815.
- Locked selections after global gate: 0.
- Deployment allowed: false.
- Report SHA-256: `d881e9b8ca9579a059ad29b864c283fb78decea9cf8656850d6625cd5001457a`.

Interpretation: relative ranking substantially improves average utility, but inner model/gate selection is unstable under sequence shift and can still promote a high-impact false positive.

## 5. Sequence-conditioned nearest-neighbor retrieval

Artifact: `outputs/assocriskbench_p15_20260716/directional_within_sequence_knn_v1`

Motivation: cross-sequence bias remained after pairwise learning, so each sequence was adapted using its own ranks 21–100 only.

Protocol:

- Outer leave-one-window-out inside each sequence.
- Nested inner leave-one-window-out over the other three windows.
- Mechanism and compact feature spaces.
- Standard and robust scaling.
- 1/3/5/7-neighbor retrieval certificates.
- Sequences without local positive evidence remain no-op.

Outer OOF result:

- Selected windows: 5.
- Positive windows: 3.
- Negative windows: 2.
- Positive precision: 0.60.
- Positive sum: +0.053648.
- Negative mass: 0.025359.
- Utility sum: +0.028289.
- Worst selected-sequence utility: −0.009540.
- Deployment allowed: false.
- Locked selections after global gate: 0.
- Report SHA-256: `89f33ff61bb18140037fbbf4cafc91dd1d5c87e7acf69cd117a252f6870a0dba`.

Interpretation: same-sequence retrieval reveals useful repeated mechanisms in MOT20-01 and MOT20-05, but sparse inner evidence can still authorize a false positive in an unseen window.

## 6. Locked prediction stability audit

Artifact: `outputs/assocriskbench_p15_20260716/directional_within_sequence_knn_stability_v1`

No locked labels were used. The audit compared all train-eligible retrieval specifications and leave-one-window retraining variants.

Stability rule:

- At least 3 eligible specifications.
- At least 80% full-train agreement.
- At least 70% leave-one-window agreement.
- At least 70% overall agreement on one exact directional key.

Results:

- MOT20-01: 1 eligible specification, 5 predictions, mode fraction 0.80; rejected because the evidence family is too narrow.
- MOT20-02: no eligible specifications.
- MOT20-03: no eligible specifications.
- MOT20-05: 7 eligible specifications, 33 predictions, 7 unique candidates, full-train mode fraction 0.50, leave-one-window mode fraction 0.259259, overall mode fraction 0.303030; rejected because candidate identity is unstable.

Final stable sequences: none.

Final selected transactions: 0.

Report SHA-256: `a31e275a123327dddafae0d67bad1390b222d2d948cd55ed35aba79f0fe76f2f`.

## 7. Reproducibility

The four formal evidence chains were independently rerun. All report, prediction, calibration, fold-selection, and summary files reproduced byte-identically.

- Conformal top-one: byte-identical reproduction passed.
- Pairwise window ranker: byte-identical reproduction passed.
- Sequence-conditioned KNN: byte-identical reproduction passed.
- Locked stability audit: byte-identical reproduction passed.

Key final hashes:

- Conformal report: `90fdf50c448d2f7000281247bf266d3bc1bf4641abd1a7a072d0cc62748829e7`
- Pairwise report: `d881e9b8ca9579a059ad29b864c283fb78decea9cf8656850d6625cd5001457a`
- KNN report: `89f33ff61bb18140037fbbf4cafc91dd1d5c87e7acf69cd117a252f6870a0dba`
- Stability report: `a31e275a123327dddafae0d67bad1390b222d2d948cd55ed35aba79f0fe76f2f`

## 8. Scientific decision

The current directional utility-learning family is closed for the frozen P15 dataset.

Reasons:

1. Positive edits are abundant in oracle analysis, so the research premise remains valid.
2. Absolute models cannot calibrate the lower tail without complete abstention.
3. Pairwise ranking improves mean utility but fails worst-sequence control.
4. Same-sequence retrieval finds repeated mechanisms but does not generalize reliably across unseen windows.
5. Remaining locked predictions are not stable across admissible train-only models.
6. Reading more locked labels would convert the test pool into a model-selection set and invalidate the protocol.

Decision:

- Keep the final locked policy at no-op.
- Keep all 156 remaining locked directional labels unread.
- Do not promote any directional edit to the mainline.
- Do not continue threshold or gate sweeps on this frozen dataset.

## 9. Next research direction

The next improvement path must increase counterfactual supervision rather than reuse the P15 locked pool.

Priority direction:

1. Build a broader train-only directional utility bank from additional sequences or independently generated counterfactual training splits.
2. Replace sparse global HOTA deltas with denser mechanism-aware supervision, including local association contribution, identity-transition duration, and affected-track support.
3. Train a hierarchical utility model with sequence/domain random effects and explicit uncertainty decomposition.
4. Require sequence-disjoint nested evaluation and stable candidate identity before creating any new P15 locked manifest.

This direction preserves the positive oracle signal while addressing the demonstrated sample-size and domain-shift bottleneck.
