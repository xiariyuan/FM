# Strict outer-clean FITR pilot on MOT20 train

Date: 2026-07-15

## Scope

This report separates the earlier same-sequence diagnostic FITR gain from the stricter cross-sequence experiment.

- Baseline: `identity_transaction_fusion_eval_v2/aggressive15_merge_then_swap`
- Baseline COMBINED HOTA: **78.813840**
- Earlier same-sequence diagnostic FITR result: **78.919023**
- The 78.919023 result is not an end-to-end cross-sequence estimate because MOT20-02 candidate and utility models used same-sequence grouped-OOF labels.
- The experiment below rebuilds unary candidates, pair scores, utility labels, utility prediction, and transaction selection with sequence-level holdout constraints.

All reported HOTA values are produced by the repository TrackEval wrapper on MOT20 train. Future tracker, detector, ReID, and overlap observations are allowed because the target method is offline tracking. GT is used only to create training/evaluation labels and is excluded from deployed features.

## 1. Cross-sequence unary change proposal

Output:

`outputs/assocriskbench_p15_20260714/segment_change_detector_loso`

Protocol:

- Train on three MOT20 train sequences.
- Score the fourth sequence.
- Candidate rule: `overlap_max_ioa >= 0.1`.
- Target: persistent identity switch `p5` within +/-3 frames.
- No held-out GT is used for model fitting.

Results:

| Variant | Mean sequence AP | Minimum sequence AP | Mean sequence AUC |
|---|---:|---:|---:|
| Raw | 0.056183 | 0.025185 | 0.874110 |
| Raw + within-sequence percentiles | **0.063653** | **0.036937** | **0.877885** |

Top-5000 exact-event recall for the selected percentile model:

| Sequence | Recall |
|---|---:|
| MOT20-01 | 1.0000 |
| MOT20-02 | 0.7706 |
| MOT20-03 | 0.7787 |
| MOT20-05 | 0.5382 |

Interpretation: cross-sequence unary ranking is real but recall is the first strict bottleneck, especially on MOT20-05.

## 2. Outer-clean pair-state reasoning

Output:

`outputs/assocriskbench_p15_20260714/segment_pair_state_ranker_outer_clean`

Outer-clean protocol:

- Training sequences use their own grouped-OOF unary candidate banks.
- The held-out sequence uses a unary model trained only on the other three sequences.
- Pair-model inputs exclude GT-derived columns, IDs, exact ranks, unary learned scores, and debt labels.
- Training weights are balanced by source sequence and class.

Results:

| Target | Variant | Mean sequence AP | Minimum sequence AP | Mean sequence AUC | Non-learned heuristic AP |
|---|---|---:|---:|---:|---:|
| Reciprocal swap | Raw | 0.266668 | 0.198019 | 0.965105 | 0.207038 |
| Reciprocal swap | Raw + percentiles | **0.282649** | **0.233711** | 0.960742 | 0.207038 |
| Related identity event | Raw | 0.251319 | 0.234390 | **0.949192** | 0.083226 |
| Related identity event | Raw + percentiles | **0.256455** | **0.251921** | 0.947186 | 0.083226 |

Interpretation: the pair-state representation transfers across sequences. This is the strongest positive result in the strict pilot.

## 3. Pre-registered outer-clean transaction candidates

Output:

`outputs/assocriskbench_p15_20260715/outer_clean_transaction_candidates`

Ranking was fixed before utility labeling:

`max(P(reciprocal), P(related))`

Canonicalization:

- unordered-pair cluster radius: 3 frames;
- per-track spacing: 30 frames;
- utility labels are not used for ranking, clustering, or spacing.

Top-30 single-event executor acceptance:

| Sequence | Accepted | Rejected |
|---|---:|---:|
| MOT20-01 | 24 | 6 |
| MOT20-02 | 30 | 0 |
| MOT20-03 | 28 | 2 |
| MOT20-05 | 27 | 3 |

## 4. Four-sequence counterfactual utility labels

Output:

`outputs/assocriskbench_p15_20260715/outer_clean_transaction_utility_top20`

Protocol:

- Use the pre-registered Top-20 transaction list from each sequence.
- Evaluate every event independently with permanent transaction replay.
- Use actual single-sequence TrackEval delta HOTA as the utility label.
- Rejected events receive zero utility because the deterministic executor makes no change.

Dataset:

- 80 events total;
- 77 executed;
- 3 rejected;
- 29 positive-HOTA events;
- 48 negative-HOTA events;
- 3 zero-utility events.

| Sequence | Positive | Negative | Zero | Mean delta HOTA | Best | Worst |
|---|---:|---:|---:|---:|---:|---:|
| MOT20-01 | 5 | 13 | 2 | -0.126807 | +0.375767 | **-0.783693** |
| MOT20-02 | 7 | 13 | 0 | -0.048701 | +0.134056 | -0.232310 |
| MOT20-03 | 8 | 12 | 0 | -0.003957 | +0.103994 | -0.127475 |
| MOT20-05 | 9 | 10 | 1 | -0.000485 | +0.072466 | -0.047954 |

Interpretation: high pair-state probability is not equivalent to positive transaction utility. A utility/risk layer is necessary. MOT20-01 contains rare catastrophic actions that dominate Top-K utility sums.

## 5. Sequence-level LOSO future utility ranking

Outputs:

- `outputs/assocriskbench_p15_20260715/outer_clean_transaction_survival_top20`
- `outputs/assocriskbench_p15_20260715/loso_future_transaction_utility_top20`

Observable future features include:

- pair future lifetime and co-presence;
- future overlap count, duration, and strength;
- ReID coverage and IoU quality at 30/60/120/300-frame horizons;
- keep/swap margin trajectories;
- chunk-wise future swap margins;
- outer-clean unary and pair probabilities.

Validation:

- train on three sequences;
- predict the fourth;
- held-out utility labels are excluded from fitting and preprocessing;
- only 20 labeled candidates are available per sequence.

Main compact variant:

- mean sequence rank Spearman: **0.117624**;
- pair-score baseline Spearman: 0.111290;
- positive AP: 0.501673;
- positive AUC: 0.639902;
- total actual Top-3 utility over four held-out folds: **-1.266063**;
- pair-score Top-3 utility: -0.083270.

The utility ranker slightly improves rank correlation but fails the actual risk-sensitive Top-K objective. It over-ranks catastrophic MOT20-01 and MOT20-02 events.

## 6. Nested risk-gated utility selection

Output:

`outputs/assocriskbench_p15_20260715/nested_risk_gated_transaction_utility_top20`

Protocol:

- Outer fold: hold out one sequence.
- Inner folds: leave one of the other three sequences out to estimate prediction residuals.
- Inner predictions choose among five fixed conservative gates and caps `{1,2,3,5}`.
- Objective: normalized utility sum with an additional 2x penalty for the worst negative training sequence.
- Final selection is greedy conflict-free by raw track ID.

Selection result:

| Held-out sequence | Selected | Positive | Negative | Isolated utility sum |
|---|---:|---:|---:|---:|
| MOT20-01 | 0 | 0 | 0 | 0.000000 |
| MOT20-02 | 1 | 0 | 1 | -0.232310 |
| MOT20-03 | 1 | 1 | 0 | +0.017007 |
| MOT20-05 | 0 | 0 | 0 | 0.000000 |
| Total | 2 | 1 | 1 | **-0.215303** |

## 7. Actual four-sequence TrackEval

Output:

`outputs/assocriskbench_p15_20260715/nested_risk_gated_fitr_full_top20`

| Method | COMBINED HOTA | AssA | IDF1 | IDSW |
|---|---:|---:|---:|---:|
| Identity Transaction Fusion baseline | **78.813840** | **76.218560** | **90.912357** | 899 |
| Strict outer-clean FITR Top-20 pilot | 78.789250 | 76.170677 | 90.886037 | 899 |
| Delta | **-0.024590** | -0.047883 | -0.026319 | 0 |

Per-sequence HOTA delta:

| Sequence | Delta HOTA |
|---|---:|
| MOT20-01 | 0.000000 |
| MOT20-02 | -0.232310 |
| MOT20-03 | +0.017007 |
| MOT20-05 | 0.000000 |

## Conclusion

The strict pilot does **not** improve TrackEval HOTA. It should not replace the baseline and cannot support a claim that FITR currently generalizes end to end.

What is supported:

1. Cross-sequence unary change signals exist, although MOT20-05 candidate recall is insufficient at a fixed Top-5000 budget.
2. Pair-state reasoning transfers across sequences and substantially improves related-event AP over non-learned pair heuristics.
3. Transaction utility is highly heterogeneous and contains rare large negative actions; a risk-aware utility layer is scientifically necessary.

What is not supported:

1. The current 20-label-per-sequence future utility model is not reliable.
2. The earlier 78.919023 result remains a same-sequence diagnostic, not the strict result.
3. A target of 79.3 or 80 cannot be justified from the current strict evidence.

## Next pre-registered experiment

Do not tune thresholds on the current 80 labels and re-evaluate the same Top-20 pool. Instead:

1. Freeze all candidate, pair, feature, and executor code from this report.
2. Label ranks 21-100 independently for all four sequences, producing 320 additional counterfactual examples.
3. Keep ranks 1-20 as a locked evaluation subset.
4. Train utility/risk models only on ranks 21-100 from the three non-held-out sequences.
5. Evaluate once on the locked ranks 1-20 of the held-out sequence.
6. Increase the strict unary candidate budget for MOT20-05 only through a pre-registered recall-budget study, not utility feedback.

This split converts the present exploratory pilot into a defensible train/validation protocol.
