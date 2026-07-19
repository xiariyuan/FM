# M23-49 compact pure-HGB transaction student audit

Date: 2026-07-20

## Protocol

- Strict nested sequence-LOSO.
- Reused the already completed M23-41 diverse exact-HOTA transaction label banks; no label experiments were rerun.
- Outer-M05 training read only M01/M02/M03 labels.
- Held M05 candidates were generated from the GT-free M23-25 transaction graph.
- Model: one five-leaf HistGradientBoostingClassifier for replacement actions.
- Policies: no-op and replacement top-k for k in {1, 2, 4, 8}, with maximum-weight matching.
- Freeze rule: maximum worst inner-fold HOTA delta, then mean fold delta and stitched inner HOTA.
- Held label file and M05 GT remained unread until the held tracker was frozen.

## Runtime smoke

The no-op smoke reproduced the three M23-25 inner baselines exactly:

- M01: 79.181480 HOTA
- M02: 72.868943 HOTA
- M03: 80.571616 HOTA

The generated held bank contained 256 replacement actions and no GT/teacher columns. The no-op M05 tracker was byte-identical to the M23-25 tracker.

## Inner policy selection

`replace_top1` was the only nonzero policy positive on all three inner folds:

- M01: +0.116310 HOTA
- M02: +0.001057 HOTA
- M03: +0.028938 HOTA
- Worst fold: +0.001057
- Mean fold: +0.048768
- Stitched inner HOTA: 78.207800 versus 78.183484 no-op

Top-2, top-4 and top-8 each degraded at least one inner sequence and were rejected.

## Frozen outer-M05 result

The frozen policy selected one GT-free replacement (`source_index=122636`). Official TrackEval after freeze produced:

- HOTA: **79.643667**
- DetA: **81.953650**
- AssA: **77.440940**
- IDSW: **475**
- Delta versus M23-25: **-0.009536 HOTA**
- Delta versus M23-39: **-0.089183 HOTA**
- Delta versus M23-46: **-0.126660 HOTA**

M23-49 is rejected and must not be expanded to four outer folds.

## Post-freeze diagnosis

After the tracker was frozen and M05 GT had been opened, the selected action's exact teacher delta was -0.009537 HOTA. Its pooled probability was 0.726, but single-domain expert probabilities were:

- M01 expert: 0.722
- M02 expert: 0.657
- M03 expert: 0.300

Thus the pooled model hid a strong source-domain disagreement. The three positive inner top-1 actions had minimum expert probabilities 0.322, 0.543 and 0.494 respectively. The next experiment should retain pooled ranking but introduce an expert-consensus lower-bound abstention threshold selected exclusively by inner TrackEval. Thresholds must be pre-registered as a grid; no threshold may be set directly from the held M05 failure.
