# M23-50 expert lower-bound transaction student audit

Date: 2026-07-20

## Protocol

M23-50 retained M23-49's compact pooled five-leaf HGB replacement ranking and added one five-leaf expert per outer-training sequence. Candidate eligibility required the minimum expert probability to exceed an inner-selected threshold. The pre-registered grid was:

- expert-min thresholds: 0.25, 0.30, 0.35, 0.40, 0.45, 0.50
- top-k: 1 or 2
- plus no-op

Policy selection maximized the worst inner-fold exact HOTA delta, then mean delta and stitched inner HOTA. M01/M02/M03 M23-41 labels were the only training labels for outer-M05. The M05 label file and GT remained unread until tracker freeze.

## Inner selection

The robust winner was `replace_top1_p0p45`:

- M01: +0.122400 HOTA
- M02: +0.001057 HOTA
- M03: +0.028938 HOTA
- worst fold: +0.001057
- mean fold: +0.050798
- stitched inner HOTA: 78.208053

Thresholds 0.25 through 0.45 produced the same three inner actions; the tie-break selected the most conservative equivalent threshold, 0.45. Top-2 policies were rejected due to a large M01 degradation.

## Frozen outer-M05 result

The frozen policy selected one GT-free replacement (`source_index=178152`) from five eligible actions. Its minimum source-domain expert probability was 0.572265. Official TrackEval after freeze produced:

- HOTA: **79.701775**
- DetA: **81.956017**
- AssA: **77.551734**
- IDSW: **475**
- Delta versus M23-25: **+0.048572 HOTA**
- Delta versus M23-39: **-0.031075 HOTA**
- Delta versus M23-46: **-0.068552 HOTA**

## Decision

Expert lower-bound abstention successfully removed the M23-49 false positive and selected a genuinely beneficial action relative to M23-25. However, rebuilding the transaction policy from the M23-25 state remains weaker than the already deployed M23-39 transaction tracker and the current M23-46 strict best. M23-50 is closed and must not be expanded to four folds.

The next experiment should keep the frozen deployable M23-39 transaction state as its baseline and label only residual transaction actions around that state. This reduces the learning problem from selecting an entire transaction set to deciding whether one additional replacement/drop improves an already strong tracker.
