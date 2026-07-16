# AssocRiskBench P15｜Local Counterfactual Supervision｜2026-07-16

## 1. Objective

The previous frozen-P15 directional utility models established that positive directional edits exist, but sparse sequence-level `delta_HOTA` supervision was not sufficient for reliable cross-sequence selection.

This stage therefore decomposes every accepted train-only directional edit into dense mechanism-level counterfactual supervision and tests whether the resulting signal can be learned from deployment-observable features under strict sequence-disjoint evaluation.

The locked protocol remains unchanged:

- Training scope: directional ranks 21–100 only.
- Remaining locked directional rows: 156.
- New locked utility labels read: 0.
- New locked TrackEval calls: 0.
- New locked candidates frozen: 0.

## 2. Exact counterfactual reconstruction

Artifact:

`outputs/assocriskbench_p15_20260716/directional_local_counterfactual_labels_v1`

Builder:

`scripts/build_directional_local_counterfactual_labels.py`

The local labels are not derived from a guessed raw-tracker provenance. For every accepted train event, the builder uses:

1. The formal aggressive15 merge-then-swap baseline trajectory.
2. The previously audited edited trajectory saved for that exact event.
3. The stored edited-trajectory SHA-256 from the formal utility audit.
4. A unique MOT row key consisting of all row fields except the identity label.

All 1,115,049 baseline trajectory rows have unique non-ID row keys across the four sequences. This permits an exact row-by-row counterfactual difference:

- Every edited file is SHA-verified.
- Every edited row is geometry-matched to exactly one baseline row.
- The edited and baseline geometry multisets must be identical.
- Changed rows are precisely those whose identity label differs.
- Recovered changed-row counts must equal the formal executor counts.
- Recovered minimum changed frame must equal `effective_start_frame`.

Formal verification result:

- Accepted train events: 225.
- Counterfactual files SHA-verified: 225 / 225.
- Changed-row count mismatches: 0.
- Changed rows labeled: 31,711.

One event contains a multi-receiver composition created by earlier identity operations:

- Event: `mot20_05_r0034_638_996_f1643_u_to_v`.
- 29 rows change `996 -> 638`.
- 2 rows change `590 -> 638`.

The metric implementation therefore treats the baseline as multiple receiver partitions merged into one donor partition, rather than assuming a single receiver identity.

## 3. GT matching and local labels

The formal baseline is matched to MOT20 train GT once per sequence using frame-level Hungarian assignment at IoU >= 0.5. Only MOT rows with `mark=1` and `class=1` are used.

Matching coverage:

| Sequence | Matched tracker rows | Total tracker rows | Coverage |
|---|---:|---:|---:|
| MOT20-01 | 19,087 | 19,372 | 98.53% |
| MOT20-02 | 148,089 | 151,152 | 97.97% |
| MOT20-03 | 301,574 | 305,937 | 98.57% |
| MOT20-05 | 625,796 | 638,588 | 98.00% |

Local labels are computed at 30, 60, 120, 300, and full-future horizons.

### 3.1 Local IDTP delta

For each event and horizon, the baseline partition is:

- donor history;
- each receiver history combined with its own future changed rows.

The edited partition is:

- donor history combined with all changed future rows;
- each receiver history without those future rows.

The label is the difference in maximum tracker-partition-to-GT assignment support, normalized by the local matched support.

### 3.2 Pairwise association delta

For every donor-history/future and receiver-history/future pair, the label counts the counterfactual change in:

`correct same-GT association pairs - incorrect different-GT association pairs`.

The result is normalized by the number of affected cross-partition pairs.

### 3.3 Row-level supervision

Every changed row is labeled as one of:

- benefit;
- harm;
- shared-history identity;
- other GT identity;
- unmatched.

This produces 31,711 row-level labels in addition to the 225 event-level targets.

## 4. Preregistered target qualification

Qualification requirements were frozen before reading the results:

- pooled Spearman correlation with `delta_AssA` >= 0.50;
- positive-`delta_HOTA` ROC AUC >= 0.75;
- positive sequence-level Spearman correlation in at least 3 of 4 sequences;
- 16-window top-one total `delta_HOTA` > 0;
- worst sequence top-one utility >= 0;
- at least 8 positive top-one windows.

Thirteen local targets pass all conditions.

The strongest target is `full_idtp_delta_norm`:

- Spearman with `delta_AssA`: 0.803645.
- Positive-`delta_HOTA` AUC: 0.950506.
- Positive sequence correlations: 4 / 4.
- Positive top-one windows: 15 / 16.
- Top-one `delta_HOTA` sum: +0.703428.
- Worst sequence utility: +0.012615.

Another particularly discriminative target is `full_pairwise_net_norm`:

- Positive-`delta_HOTA` AUC: 0.972340.
- Top-one `delta_HOTA` sum: +0.783056.
- Worst sequence utility: +0.008315.

This confirms that the dense local counterfactual labels accurately capture association utility and are substantially more informative than direct sparse global-utility regression.

Formal local-label report SHA-256:

`6fbef81a1c759fd186ae1a465b923657c3b8aa58fb500253eed4591e0616e62d`

## 5. Sequence-disjoint observable-feature ensemble

Artifact:

`outputs/assocriskbench_p15_20260716/directional_local_counterfactual_loso_ensemble_v1`

Trainer:

`scripts/train_loso_local_counterfactual_ensemble.py`

Protocol:

- Primary target fixed to `full_idtp_delta_norm` before training.
- 90 deployment-observable features only.
- No GT, local-label, TrackEval, or global-delta field is present in the feature whitelist.
- Four outer folds, each holding out one complete sequence.
- Six fixed components:
  - two ExtraTrees regressors;
  - one HistGradientBoosting regressor;
  - two ExtraTrees pairwise rankers;
  - one HistGradientBoosting pairwise ranker.
- Component scores are converted to within-window percentile ranks.
- The ensemble uses the 25th percentile across component ranks.

Outer LOSO event metrics:

- Spearman with local target: 0.357470.
- Spearman with `delta_AssA`: 0.405938.
- Spearman with `delta_HOTA`: 0.414236.
- Positive-`delta_HOTA` AUC: 0.678143.

Outer top-one result:

- Selected windows: 16.
- Positive / negative / zero: 9 / 6 / 1.
- Positive sum: +0.194571.
- Negative mass: 0.058202.
- Total `delta_HOTA`: +0.136369.
- Negative-mass fraction: 0.299130.
- Worst sequence utility: -0.040599.
- Catastrophic windows with `delta_HOTA <= -0.05`: 0.

Per sequence:

- MOT20-01: +0.167938.
- MOT20-02: -0.009058.
- MOT20-03: -0.040599.
- MOT20-05: +0.018088.

The deployment gate fails because the positive-HOTA AUC, positive-window count, negative-mass fraction, and worst-sequence constraint are not satisfied.

The strongest individual students are the two ExtraTrees pairwise models. Each achieves approximately:

- Top-one `delta_HOTA`: +0.211898.
- Positive windows: 10 / 16.
- Negative-mass fraction: 0.191772.

However, both retain the MOT20-03 domain failure and therefore cannot be deployed.

Formal LOSO ensemble report SHA-256:

`83dc32daf2df66d2988848c6cfc85eb14e73a74ae2e9747415da3a9cbbe79505`

## 6. Nested local-positive gate

Artifact:

`outputs/assocriskbench_p15_20260716/directional_nested_local_positive_gate_v1`

Trainer:

`scripts/train_nested_local_positive_gate.py`

The second model family separates sign prediction from ranking:

1. Two ExtraTrees classifiers predict whether `full_idtp_delta_norm > 0`.
2. The minimum of their probabilities is the conservative gate score.
3. Two ExtraTrees pairwise rankers order candidates that pass the gate.
4. Every outer fold calibrates its threshold using an inner sequence-disjoint LOSO over the other three sequences.
5. Inner calibration uses local counterfactual labels only.

No threshold passes the preregistered inner conditions in any outer fold. All four folds therefore choose no-op.

Event-level outer metrics:

- Local-positive AUC: 0.674435.
- Positive-`delta_HOTA` AUC: 0.628884.

Representative inner near-misses show that the failure is not a minor threshold issue:

- For the MOT20-01 outer fold, the best broad thresholds select 8–10 windows but local-positive precision is only 0.40–0.50 and worst-sequence local utility is strongly negative.
- For the MOT20-02 outer fold, broad thresholds have local-positive precision 0.40 and large negative local utility.
- For the MOT20-03 outer fold, one high threshold is perfectly precise but covers only one sequence and one window; broader thresholds fail precision or worst-sequence constraints.
- For the MOT20-05 outer fold, high thresholds cover only two sequences and retain negative local utility.

Formal result:

- Eligible outer folds: 0 / 4.
- Selected windows: 0.
- Deployment allowed: false.
- Locked predictions generated: 0.

Formal nested-gate report SHA-256:

`9dbe3b7245cdc3cc14d58bc8e518900dbdf3c727dd8187e8ba019e36a9055a11`

## 7. Reproducibility and leakage audit

The local-label builder, observable-feature LOSO ensemble, and nested positive gate were each independently rerun.

All files in all three formal/reproduction directory pairs are byte-identical.

Additional checks:

- 225 / 225 counterfactual trajectory SHA values verified.
- 0 changed-row replay mismatches.
- Feature whitelist contains no GT, TrackEval, local-label, or delta-derived feature.
- Every formal report states that locked artifacts were not read.
- The 156-row remaining locked pool was not opened by these scripts.

## 8. Scientific interpretation

The stage separates two questions that were previously conflated:

1. **Does a dense mechanism-level target faithfully represent directional utility?**
   - Yes. The full local IDTP and pairwise targets strongly correlate with `delta_AssA`, achieve AUC values above 0.95, and produce highly positive train-window oracle rankings.

2. **Can the existing observable feature representation predict that target across unseen sequences?**
   - Not reliably. The student models learn useful ordinal signal and positive average utility, but do not transfer the local sign boundary to every held-out sequence.

The main bottleneck is therefore no longer label sparsity alone. It is the interaction of:

- only four training domains;
- sequence-specific calibration and identity graph structure;
- insufficient observable representation of the GT-consistent donor/receiver mechanism;
- strong domain shift in the local positive boundary.

This is a useful paper-level result: the local counterfactual target provides a high-quality teacher and an interpretable oracle upper bound, while the student gap identifies a concrete generalization problem rather than an executor or optimization failure.

## 9. Decision

- Keep the P15 locked policy at no-op.
- Keep all 156 remaining locked directional labels unread.
- Do not freeze a new locked manifest.
- Stop threshold sweeps for the current local-positive gate family.
- Preserve the dense local counterfactual labels as a train-only supervision asset.

## 10. Next research direction

The next improvement should target domain generalization rather than another threshold family:

1. Expand independent train-only directional counterfactual supervision to additional MOT17/MOT20 sequences or separately generated domains.
2. Train a hierarchical model with sequence/domain random effects and explicit uncertainty decomposition.
3. Add observable identity-graph mechanism features that approximate the teacher quantities without using GT, such as donor/receiver partition consistency, multi-anchor composition, and temporal support concentration.
4. Evaluate leave-domain-out calibration with candidate-identity stability, not only utility averages.
5. Reopen P15 locked prediction only after the expanded student passes sequence-disjoint positive-boundary and worst-domain constraints.
