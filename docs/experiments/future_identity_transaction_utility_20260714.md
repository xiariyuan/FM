# Future-aware Identity Transaction Utility (FITR) — 2026-07-14

## Scope

This experiment studies post-tracking identity repair as a constrained transaction-utility problem. It is a diagnostic experiment on MOT20 train. The current utility ranker is trained and evaluated with grouped out-of-fold predictions on MOT20-02 only; therefore, the result is **not yet a cross-sequence deployable result**.

## Baseline

The starting point is `aggressive15_merge_then_swap` from Identity Transaction Fusion:

| Metric | COMBINED | Mean over M01/02/03/05 | M02 |
|---|---:|---:|---:|
| HOTA | 78.813840 | 77.628159 | 71.570820 |
| AssA | 76.218560 | — | 63.637130 |
| IDF1 | 90.912357 | — | 79.197229 |
| IDSW | 899 | — | 290 |

## Method

1. Canonicalize reciprocal pair-state proposals into events `(u, v, t)`.
2. Evaluate permanent single-event counterfactual utility on the top 100 MOT20-02 events.
3. Build future-aware observable features from tracker outputs, detector/ReID dumps, and future video frames only:
   - pair/track survival;
   - future co-presence and overlap recurrence;
   - ReID keep/swap consistency at 30/60/120/300 frames;
   - swap-margin trajectories;
   - local motion, appearance, and identity-debt features.
4. Exclude GT labels, counterfactual outcomes, post-action changed-row counts, track IDs, canonical rank, and action-end statistics from the model.
5. Train ExtraTrees regression, HistGradientBoosting regression, and an ExtraTrees positive-utility classifier using 5-fold GroupKFold by unordered track pair.
6. Rank events with the fixed OOF score:

```
0.4 * rank(ET utility)
+ 0.4 * rank(HGB utility)
+ 0.2 * rank(ET positive probability)
```

7. Apply the top-K permanent transactions jointly with collision and overlap safety gates.

## Event utility bank

The top-100 event bank contains:

- 100 events from 91 unordered pairs;
- 44 positive-utility events;
- 56 non-positive events;
- mean single-event delta HOTA: -0.017932;
- positive single-event utility sum: +3.085505 HOTA points.

The strongest individual event is pair `140 <-> 190` at frame 1432:

- M02 HOTA: 71.570820 -> 71.866970;
- delta HOTA: +0.296150;
- delta AssA: +0.365753;
- IDSW: 290 -> 288.

## Grouped OOF ranking quality

| Ranker | Spearman | Positive AP | Positive AUC | Top-5 precision | Top-10 precision | Top-20 precision |
|---|---:|---:|---:|---:|---:|---:|
| ExtraTrees utility | 0.2126 | 0.5952 | 0.6189 | 1.00 | 1.00 | 0.65 |
| HGB utility | 0.2815 | 0.5952 | 0.6189 | 0.60 | 0.70 | 0.70 |
| OOF ensemble rank | 0.2599 | 0.5952 | 0.6189 | 1.00 | 0.80 | 0.80 |

Descriptive full-data feature importance is led by future overlap recurrence, future swap margin, and track age. These importances are descriptive only and are not used to select the OOF test events.

## Joint replay on MOT20-02

| Selection | Accepted | M02 HOTA | Delta HOTA | M02 AssA | M02 IDF1 | IDSW |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | — | 71.570820 | — | 63.637130 | 79.197229 | 290 |
| OOF top-5 | 5/5 | 72.097284 | +0.526464 | 64.536800 | 79.933680 | 286 |
| OOF top-10 | 10/10 | 72.187190 | +0.616370 | 64.702684 | 80.101769 | 288 |
| OOF top-20 | 19/20 | **72.348195** | **+0.777375** | **64.946560** | **80.476535** | **278** |

The top-20 result is not the sum of independent utilities; it is a measured joint TrackEval result after transaction interaction and safety-gate handling.

## Full four-sequence TrackEval

For this diagnostic full evaluation, only MOT20-02 is replaced by the OOF top-20 FITR result. MOT20-01, MOT20-03, and MOT20-05 remain unchanged from the prior global best.

| Sequence | HOTA | AssA | IDF1 | IDSW |
|---|---:|---:|---:|---:|
| MOT20-01 | 78.789973 | 76.044060 | 90.678290 | 41 |
| MOT20-02 | **72.348195** | **64.946560** | **80.476535** | **278** |
| MOT20-03 | 80.570410 | 79.993110 | 94.723794 | 138 |
| MOT20-05 | 79.581434 | 77.318590 | 91.869819 | 430 |
| **COMBINED** | **78.911320** | **76.394870** | **91.086279** | **887** |

Four-sequence arithmetic mean HOTA: **77.822503**.

Relative to the prior global best:

- COMBINED HOTA: 78.813840 -> **78.911320**, delta **+0.097480**;
- arithmetic mean HOTA: 77.628159 -> **77.822503**, delta **+0.194344**;
- COMBINED AssA: 76.218560 -> **76.394870**, delta **+0.176310**;
- COMBINED IDF1: 90.912357 -> **91.086279**, delta **+0.173922**;
- IDSW: 899 -> **887**, delta **-12**.

## Interpretation

The result supports the central hypothesis: pair-state probability is useful for candidate generation, but transaction utility requires future identity-survival evidence. Future-aware utility ranking changes a harmful average intervention pool into a high-precision commit set.

## Limitations and next protocol

The current ranker remains a MOT20-02 same-sequence diagnostic, although event folds are separated by unordered pair. It must not be presented as a deployable generalization result. The next required experiment is a multi-sequence utility bank with leave-one-sequence-out training and evaluation, followed by one fixed transaction policy evaluated on all held-out sequences.
