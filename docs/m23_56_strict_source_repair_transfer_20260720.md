# M23-56 Strict Source-Repair Structured Transfer Feasibility

Date: 2026-07-20

## Final decision

- **Protocol valid:** yes.
- **Four frozen outer policies:** P0 / P0 / P0 / P0.
- **New strict improvement:** no.
- **Strict deployable best remains M23-46:** HOTA **79.123193**, DetA 81.543470, AssA 76.825150, IDSW 996.
- M23-55 remains a teacher-only capacity result at HOTA 80.385540 and is not deployable.
- M23-54 was not started and no MOT20 test submission was made.

## Protocol invariants

- Preregistration SHA-256: `8ba1c85b791614bae498afe30f2d2e8441c11d8f6340412438644370c7d71583`.
- Frozen implementation SHA-256: `aa7999169548df30679369d7901c7caf095088bb8a4acf2ab0cbaa49840d33d2`.
- One shared two-layer MLP, hidden dimension 64, six fixed heads, 20 epochs.
- Immutable M23-55 candidate graph; no candidate K, gap quota, flow-weight, risk-level, threshold, conformal-coverage or edit-count scan.
- P0/P1/P2 only. P1 risk UCB <=2%; P2 risk UCB <=5%.
- All four outer policy manifests were frozen before the outer-evaluation event.

## GT-free observable transaction freeze

| Sequence | Source transactions | Cross transactions | Runtime | Peak RSS | M23-55 graph SHA |
|---|---:|---:|---:|---:|---|
| MOT20-01 | 606 | 90325 | 17.6 s | 1.08 GiB | `f877b0197ee66e54...` |
| MOT20-02 | 4852 | 907822 | 47.3 s | 2.80 GiB | `67ca9ffa5bd96178...` |
| MOT20-03 | 9848 | 1765291 | 89.5 s | 4.92 GiB | `e9ef7c7a804e3fab...` |
| MOT20-05 | 20647 | 3934884 | 208.6 s | 10.34 GiB | `a50b20d78d447558...` |

Every observable parquet and normalized feature matrix was frozen before M23-56 labels were opened. Forbidden GT-derived input columns were empty on all folds.

## Strict nested inner gates

| Outer | P1 deltas | P1 worst / mean | P1 pass | P2 deltas | P2 worst / mean | P2 pass | Frozen policy |
|---|---|---:|:---:|---|---:|:---:|:---:|
| MOT20-01 | MOT20-02:+0.000000, MOT20-03:+0.000000, MOT20-05:+0.000000 | +0.000000 / +0.000000 | False | MOT20-02:+0.000000, MOT20-03:+0.000000, MOT20-05:+0.000000 | +0.000000 / +0.000000 | False | **P0** |
| MOT20-02 | MOT20-01:+0.000000, MOT20-03:+0.000000, MOT20-05:+0.000000 | +0.000000 / +0.000000 | False | MOT20-01:+0.000000, MOT20-03:+0.000000, MOT20-05:+0.000000 | +0.000000 / +0.000000 | False | **P0** |
| MOT20-03 | MOT20-01:+0.000000, MOT20-02:+0.000000, MOT20-05:+0.000000 | +0.000000 / +0.000000 | False | MOT20-01:+0.000000, MOT20-02:+0.000000, MOT20-05:+0.000000 | +0.000000 / +0.000000 | False | **P0** |
| MOT20-05 | MOT20-01:+0.000000, MOT20-02:+0.000000, MOT20-03:+0.000000 | +0.000000 / +0.000000 | False | MOT20-01:+0.000000, MOT20-02:+0.000000, MOT20-03:+0.000000 | +0.000000 / +0.000000 | False | **P0** |

All 24 P1/P2 training-calibration decisions produced zero preliminarily eligible transactions. Their exact catastrophic-risk UCB was therefore 1.0 and both policies were disabled before every validation tracker was constructed.

## Observable rankability diagnostics

| Metric across 12 inner validations | Mean | Min | Max |
|---|---:|---:|---:|
| Source-error PR-AUC | 0.1830 | 0.0987 | 0.3203 |
| Precision at score >=0.5 | 0.0930 | 0.0440 | 0.1492 |
| Recall at score >=0.5 | 0.8420 | 0.7274 | 0.9310 |
| Precision at actual edit count | 0.2191 | 0.1324 | 0.3103 |
| Recall at 95% repair precision | 0.0120 | 0.0000 | 0.1034 |
| Recall at 99% repair precision | 0.0120 | 0.0000 | 0.1034 |
| Cross alternative MRR | 0.3718 | 0.2798 | 0.4419 |

The source head frequently detects suspicious source edges (mean recall 84.20%) but cannot isolate them precisely (mean precision 9.30%; precision at the true edit count 21.91%). Mean recall available at 95% or 99% repair precision is only 1.20%. This is the direct reason strict risk calibration selects no edits.

False-cut and catastrophic false-repair rates are both zero only because every policy froze to no-op; they are not evidence that the learned editor is safe.

## Compute and provenance

- 12 full inner models trained on 172,620 source rows and 3,208,776 sampled cross transactions.
- Total model-training time: 37.4 s; maximum process RSS 5.11 GiB; maximum GPU allocation 0.91 GiB.
- Full validation scoring covered 20,094,966 cross edges in 365.4 s.
- 36 inner policy metric evaluations were required. Four unique byte streams ran official TrackEval; 32 results were reused only after exact tracker SHA equality.
- Outer TrackEval runs: 0; COMBINED TrackEval runs: 0, because all four frozen trackers are byte-exact M23-46.

## Frozen outer provenance

| Outer | Policy manifest SHA | Tracker SHA |
|---|---|---|
| MOT20-01 | `b64a3de49a5984a622d1c2daddaa824f1c227023b45d25e1a69c986d9d1c6ccb` | `7daadfdb4815bbe56c0969ec6a9e860ab5371a7f2ef767ab7485881c7c0b0fb4` |
| MOT20-02 | `799dc459ba5ba297c7522c3b5cccb16d0b6ffa5bd3095ff1197b996475b5720c` | `f6f5600da5fd630d07a496e3d6eda5dab5c194fedc5fbfc2e2f8e184be70a714` |
| MOT20-03 | `adfe98a18ff37ad8d893b66a799329ee442be5a0b1ec391972f3857e13e687f5` | `753d1f6e042afd9501589cc0d5aec82f1b11ad6c89b3b60db517da97cfca3cf1` |
| MOT20-05 | `211c71d7902165f8b96267c09c92ae3a7f4c2e59cb7ce4c85f0a7324eb9028fe` | `5c665fd61a75048c3c8ef8538491b0b5f421658f3046d5be585d28c87a2a7dae` |

## Scientific conclusion

M23-55 already established that candidate recall and the teacher flow solver can support HOTA 80.385540. M23-56 shows that a small observable source-anchored transfer model cannot certify useful replacements across sequences under 2% or 5% catastrophic-risk bounds. The failure should therefore be attributed to observable source-integrity transfer, not to candidate recall or the global flow solver.

M23-56 is closed. The only future directions recorded, not executed in this round, are relation-specific external pretraining and an observable node-impurity change-point model.
