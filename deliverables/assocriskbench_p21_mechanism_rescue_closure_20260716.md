# AssocRiskBench P21 — Mechanism-Aware Rescue Closure

Date: 2026-07-16

## Objective

P21 tests whether the single positive-block miss left by the frozen P20 rank-conformal candidate set can be recovered by mechanism-specific retrieval without enlarging the generic P20 rank radius.

Two mechanisms are modeled separately:

1. **Receiver split / ephemeral-anchor namespace split**: a positive intervention whose donor history is teacher-unmatched while the receiver history and future map to different dominant identities.
2. **Same-identity merge**: a positive intervention whose donor, receiver, and future teacher identities agree.

The mechanism labels are used only as train/evaluation targets. All model inputs are deployable tracker/event features.

## Deployable mechanism feature bank

P21 adds 75 features computed from frozen executable-event metadata and raw tracker rows:

- receiver pre/post change-point residuals;
- pre-boundary back-test residuals;
- pre/post velocity and log-size velocity changes;
- donor-to-receiver future transfer errors;
- receiver self-prediction errors;
- donor lifespan and ephemeral-anchor ratios;
- donor/receiver overlap IoU and center-distance statistics;
- boundary geometry and confidence discontinuities.

Feature-bank integrity:

- events: 11,705;
- sequences: 7;
- feature columns: 75;
- duplicate event keys: 0;
- forbidden GT/utility/TrackEval/locked feature columns: 0;
- maximum missing fraction: 0.199572832123;
- TrackEval calls: 0;
- P15 locked reads: 0.

## Fixed nested protocol

- Outer leave-one-sequence-out evaluation over all seven MOT17-FRCNN sequences.
- Six inner leave-one-sequence-out folds inside every outer fold.
- Frozen P20 rank-conformal set remains unchanged.
- Two fixed ExtraTrees classifiers, one per mechanism:
  - 256 trees;
  - maximum depth 10;
  - minimum leaf size 3;
  - maximum features 0.75;
  - balanced classes;
  - equal total sample weight per sequence.
- Total model inputs: 130 deployable features.
- Mechanism rescue radius: finite-sample higher conformal quantile at alpha 0.10 of the minimum mechanism-positive rank in completely inner-OOF source blocks.
- No model-family, parameter, alpha, radius, threshold, or utility-gate sweep.

Preregistered acceptance requires all of the following:

- positive-block coverage = 23/23;
- at least one newly covered block;
- mean added candidates <= 12;
- maximum combined set size <= 100;
- nonnegative worst-sequence set-oracle utility.

## Results

### Mechanism support

- receiver-split positive events: 301 across all seven sequences;
- same-identity-merge positive events: 24 across all seven sequences.

Thus the P20 miss is not an isolated hand-written exception; both mechanisms have sequence-disjoint source support.

### Coverage

- P20 base set: 22/23 positive-available blocks;
- P21 combined set: **23/23**;
- conditional positive-block coverage: **1.0**;
- newly covered blocks: 1;
- newly covered block: MOT17-11-FRCNN temporal block 3.

The three previously missed positive events are recovered by the mechanism models.

### Efficiency

- mean P20 base-set size: 46.571429;
- mean rescue additions: **98.857143**;
- mean combined-set size: **145.428571**;
- maximum combined-set size: **218**;
- rescue unique events: 2,768.

The combined set contains:

- 545 positive events;
- 3,301 negative events;
- 226 zero-utility events.

Set-oracle utility remains positive:

- combined set-oracle utility sum: +7.350931296431;
- worst-sequence set-oracle utility: +0.286057692308.

However, the fixed efficiency constraints are violated by a large margin.

## Why the conformal rescue set becomes too large

The mechanism classifiers are useful in typical source blocks:

- receiver-split median minimum-positive rank: approximately 5–10;
- same-identity-merge median minimum-positive rank: approximately 2–5.

The failure arises from finite-sample calibration rather than a complete lack of mechanism signal.

At alpha = 0.10, the finite-sample higher conformal quantile is strictly below the maximum only when at least 19 calibration observations are available. P21 has:

- receiver-split positive calibration blocks: 14–17 per outer fold;
- same-identity-merge positive calibration blocks: 8–11 per outer fold.

Therefore all 14 outer-fold × mechanism calibrations are mathematically forced to use the maximum observed source-block rank.

Consequences:

- median receiver-split radius inflation over the median positive rank: 13.5×;
- median same-identity-merge radius inflation: 13.5×;
- receiver-split conformal radii: 76–162;
- same-identity-merge conformal radii: 24–52.

This explains why a model that usually places a mechanism positive near the top still produces an impractically broad conformal rescue set.

## Decision

**Do not retain the P21 block-rank conformal rescue set.**

P21 establishes two separate facts:

1. Mechanism-aware deployable features and sequence-disjoint learning can recover the final P20 coverage miss.
2. Positive-block rank is too coarse and too sparse a calibration unit for 90% finite-sample conformal efficiency in this seven-domain bank.

Deployment remains disallowed. No locked manifest is created, and the P15 policy remains no-op.

## Closed directions

- Enlarging the generic P20 rank radius.
- Retaining the maximum-rank P21 rescue set.
- Hard `donor <= 20 frames` ephemeral rules: they cover split positives but retain roughly 139 candidates per block.
- Simple merge geometry rules: they retain roughly 39 candidates per block.
- Returning to scalar utility thresholds or pooled-score gates.

## Next stage

P22 should keep the learned mechanism ranking and frozen P20 retrieval layer, but replace positive-block rank calibration with one of the following structured risk-control formulations:

1. cluster-aware event-level risk control with sequence/block dependence explicitly represented;
2. mechanism-conditioned pairwise certificates that compare a candidate against local hard negatives;
3. hierarchical risk sets whose calibration unit contains substantially more than 19 exchangeable observations while preserving sequence-level outer evaluation.

The goal is no longer to improve average classifier AUC. It is to preserve the mechanism-ranking signal while avoiding the small-sample maximum-rank degeneracy.

## Reproducibility

Both formal chains reproduce byte-identically:

- receiver-change feature bank: all four files identical;
- nested mechanism rescue audit: all ten files identical.

Key hashes:

- feature report SHA256: `4e1a52057a51307fc3c0206f252548485cd3c6cea97153d9c2969f72ff9306bb`;
- mechanism audit report SHA256: `26cc7d7d0cc8b2ae93fed3da6b12c61e8d447ff508c494154f263df61c81378a`;
- unified audit SHA256: `5b3ab03ef0357b99a8ef19dccee27fa80a8a36de3accb2219cbc2a11211e23a2`.

## Locked-state statement

- new P15 locked-label reads: 0;
- new P15 locked TrackEval calls: 0;
- new global TrackEval calls: 0;
- remaining locked rows untouched: 156;
- P15 policy: no-op.
