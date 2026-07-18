# M23-25 Sequence-Calibrated Transaction Conflict Graph

## Decision

M23-24 confirms that upstream FastReID fine-tuning is not sufficient by itself.
On the strict outer-held MOT20-02 fold, the fine-tuned candidate graph changed the
inner-selected action budget from 2 to 32, but the frozen tracker reached only
72.040470 HOTA and 64.406186 AssA. This is approximately +0.466 HOTA over the
fixed parent, while IDSW increased from 286 to 328.

The next bottleneck is therefore the mapping from candidate transactions to a
safe globally consistent action set, not detector recall or ReID capacity alone.

## Evidence

- Fixed exploratory parent: 78.763497 combined HOTA.
- Best nested exploratory policy: 78.790003 combined HOTA.
- Link-only GT oracle: 79.200060 combined HOTA.
- Chain-transaction GT oracle: 80.980587 combined HOTA.
- M02 parent: 71.574910 HOTA, 63.638735 AssA.
- M02 chain oracle: 76.780814 HOTA, 72.634435 AssA.

The link-only ceiling below 80 and the chain ceiling above 80 show that split and
relink transactions are necessary. The deployable model needs to recover about
55 percent of the available oracle association margin to cross 80 combined HOTA.

## Model

M23-25 is a lightweight structured model suitable for only four MOT20 training
sequences. It deliberately avoids a large end-to-end graph transformer.

1. Base transaction representation
   - GT-free appearance, motion, gap, endpoint, cut-position and chain-coherence
     features from the M23 microtracklet graph.

2. Transaction conflict graph
   - Each candidate split/relink transaction is an edge between its two original
     parent tracks.
   - Added features describe source and destination conflict degree, competing
     ranks, pair multiplicity, sequence-relative percentiles, appearance upgrade,
     motion-appearance agreement and action complexity.

3. Sequence-normalized distributional utility
   - A classifier predicts positive transaction probability.
   - Separate regressors predict normalized positive gain and negative loss.
   - Training weights balance both sequences and utility signs.
   - The decision score is probability times gain minus a calibrated multiple of
     failure probability times loss.

4. Sequence-adaptive selection
   - There is no fixed number of actions.
   - A sequence-relative score quantile changes the action count with candidate
     density.
   - Inner leave-one-training-sequence-out calibration maximizes the worst
     sequence-normalized utility, preventing MOT20-05 from dominating the policy.

5. Global conflict resolution
   - Eligible transactions are selected with exact maximum-weight matching over
     the original-track conflict graph.
   - This replaces independent greedy acceptance and guarantees a maximum-weight
     disjoint action set under the one-transaction-per-parent constraint.

## Strict protocol

- Outer-held sequence GT is not opened during feature construction, model fitting,
  calibration or tracker generation.
- The other three sequences provide transaction utility labels.
- Every calibration candidate is evaluated only on inner held training sequences.
- The outer tracker is frozen before TrackEval opens held GT.
- All four held trackers must eventually be concatenated and evaluated once for
  the final combined score.

## Promotion criteria

- Immediate M02 signal: HOTA above 72.040470 without DetA collapse.
- Strong M02 signal: at least +1.0 HOTA over the 71.574910 parent.
- Final promotion: strict four-fold combined HOTA above 80.0.
- If M23-25 cannot outperform M23-24, the next change will target exact
  counterfactual HOTA/AssA supervision and temporal tracklet encoding rather than
  additional threshold grids.

## First strict outer-fold result

The full MOT20-02 fold passed the strong-signal gate:

- Parent: 71.574910 HOTA / 63.638735 AssA / 286 IDSW.
- M23-24 fine-tuned fixed-budget policy: 72.040470 HOTA / 64.406186 AssA /
  328 IDSW with 32 actions.
- M23-25: 72.868943 HOTA / 65.885675 AssA / 317 IDSW with 18 actions.
- Delta from parent: +1.294033 HOTA / +2.246940 AssA / +31 IDSW.
- Delta from M23-24: +0.828473 HOTA / +1.479489 AssA / -11 IDSW.

Inner training-sequence calibration froze loss multiplier 1.0, minimum positive
probability 0.5 and sequence-relative score quantile 0.999 before the outer
TrackEval. The result uses no MOT20-02 GT before tracker freezing.

This result validates the structured decision mechanism but is not a combined
score. The remaining MOT20-01, MOT20-03 and MOT20-05 FastReID outer folds are
queued sequentially. Only their concatenated four-fold TrackEval can trigger the
80.0 promotion rule.
