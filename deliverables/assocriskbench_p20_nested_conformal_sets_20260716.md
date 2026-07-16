# AssocRiskBench P20 — Nested Conformal Candidate Sets

Date: 2026-07-16
Status: completed, fail-closed for deployment

## Objective

P19 showed that expanding from 705 canonical events to all 11,705 executable events recovered most positive counterfactual teachers and made the pooled motion-multitask top-one utility positive. However, the worst sequence remained negative and no scalar abstention threshold was safe.

P20 therefore replaces single-score authorization with set-valued prediction. The goal is to retrieve a compact candidate set that contains at least one positive event whenever a positive event exists in the block, while preserving strict sequence-disjoint evaluation.

## Protocol

- Dataset: seven MOT17-FRCNN train sequences, 11,705 executable directional events.
- Evaluation unit: four temporal blocks per sequence, 28 blocks total.
- Outer evaluation: leave one complete sequence out.
- Inner calibration: each remaining sequence is predicted completely out of fold.
- Base views: the three fixed P19 views only:
  - geometry positive probability;
  - geometry utility prediction;
  - actual-anchor motion 13-target multitask score.
- No model-family or threshold sweep.
- No global TrackEval call.
- No P15 locked-label or locked-TrackEval read.

### Rank-conformal set

For every block, each event receives a descending rank under each of the three fixed views. The event union rank is the minimum of these three ranks.

For every inner-OOF source block containing at least one positive event, the calibration statistic is the minimum union rank among its positive events. The outer set radius is the finite-sample higher conformal quantile at alpha = 0.10. The outer candidate set contains all events whose union rank is no larger than this source-only radius.

### Block OOD certificate

Each block is summarized by label-free score-distribution, top-gap, and cross-view overlap descriptors. The nonconformity score is the nearest cross-sequence block distance after robust scaling. A pooled conformal p-value of at least 0.10 is required for the exploratory OOD certificate.

### Positive-support set

As a coverage upper bound, source-only geometry and compact motion features are robust-scaled and projected to 15 PCA components. Positive support nonconformity is the log ratio between nearest positive and nearest negative distances. Calibration positives are always queried against positives and negatives from other source sequences.

## Results

### Rank-conformal candidate set

| Metric | Result |
|---|---:|
| Positive-available blocks | 23 |
| Covered positive blocks | 22 |
| Conditional positive coverage | **95.6522%** |
| Mean set size | **46.5714** |
| Median set size | 50 |
| Maximum set size | 87 |
| Positive members | 207 |
| Negative members | 932 |
| Oracle set utility sum | +6.314904 |
| Oracle set worst sequence | **+0.286058** |

The observed conditional coverage exceeds the fixed 90% target. Unlike P19 scalar top-one policies, the set retains nonnegative oracle potential on every sequence.

This is a candidate-retrieval result, not a deployable single-action policy. The set still contains many negative members and cannot itself decide which transaction to execute.

### Positive-support set

| Metric | Result |
|---|---:|
| Covered positive blocks | **23 / 23** |
| Conditional positive coverage | 100% |
| Mean set size | 310.5714 |
| Median set size | 293 |
| Maximum set size | 647 |
| Negative members | 7,392 |

The positive-support set closes the one rank-set miss, but only by retaining hundreds of candidates per block. It is not retained as the main method.

### Block OOD certificate

The fixed p >= 0.10 certificate retained 24 blocks across all seven sequences. Among 22 certified positive-available blocks, it covered 21. The same rank-set miss remained certified, so block-level distribution shift is not the cause of that failure.

The current OOD certificate is therefore rejected.

## Uncovered-event forensic

The only missed positive block is MOT17-11-FRCNN temporal block 3. It contains three positive events outside the rank-conformal set:

- one `same_identity_merge` event;
- two `ephemeral_anchor_receiver_split` events.

The split events expose a mechanism not represented by the current actual-anchor transfer model. A short or GT-unmatched donor anchor is used as a fresh identity namespace to split a receiver trajectory whose identity changes across the boundary. In this mechanism, donor-to-future motion agreement is not expected to be positive; the useful signal is receiver self-discontinuity plus low anchor cost.

This explains why generic positive-neighbor support and source-transfer motion both rank these events poorly.

## Decision

Retain:

- the full 11,705-event teacher bank;
- the full actual-anchor motion bank;
- the nested rank-conformal set as a candidate-retrieval layer;
- the 95.65% conditional coverage result as formal evidence.

Reject:

- direct deployment of the candidate set;
- the current block OOD certificate;
- the broad positive-support set;
- generic radius enlargement;
- further scalar threshold tuning.

Deployment remains false. No locked manifest is created. P15 remains `no_op`, and all 156 locked rows remain unread.

## Next stage

Build mechanism-specific, label-free features for:

1. receiver pre/post-boundary change points;
2. receiver velocity and geometry discontinuity;
3. ephemeral donor-anchor cost;
4. namespace-split opportunity;
5. same-identity merge continuity.

These features should be used as a rescue head inside the rank-conformal candidate set, not as a new global scalar gate.

## Reproducibility

Formal directory:

`outputs/assocriskbench_p20_20260716/nested_conformal_candidate_sets_v1`

Independent reproduction:

`outputs/assocriskbench_p20_20260716/nested_conformal_candidate_sets_v1_repro`

All files are byte-identical.

Report SHA256:

`b69e52c52ae9084839aa2da58860e8fec4ac387d73c7981cf26f9b65666c1564`

Unified audit:

`deliverables/assocriskbench_p20_audit_20260716.json`

Audit SHA256:

`1ff6b3a1204b4cfbd2bf46cc042d8edcb7272ba4f8ee3e12d16bb10fe0880b86`
