# MOT20 M23-16—M23-23 domain-robust transaction-policy audit

Date: 2026-07-18
Dataset: MOT20 train sequences `01/02/03/05`
Outer protocol: sequence LOSO; every held sequence is excluded from model fitting and policy calibration
Inference GT use: none
Parent: fixed GT-free exploratory fused A42 tracker, HOTA `78.763497`
Formal deployable anchor: HOTA `77.699`
Best exploratory result before this batch: M23-14 HOTA `78.790003`
Nondeployable action-space lower bound: M23-12 HOTA `80.980587`

## Outcome

This batch did not produce a new deployable or exploratory best. The corrected transaction executor, sequence normalization, domain ensembles, pair-state representation, expanded budgets, and GT-free sequence-specific appearance adaptation all failed to convert the `80.980587` GT-oracle action space into a robust held-sequence policy.

The strongest result in this batch was M23-16 at HOTA `78.783100`, still below M23-14 by `0.006903`. M23-23 was the first strict policy in this family to act on both MOT20-02 and MOT20-05 without the old OOD gate, but its two MOT20-05 actions had zero positive utility and a post-freeze transaction-utility sum of `-6258.783443`; combined HOTA fell to `78.503050`.

No result in this document is promoted as a formal deployable score because the fixed fused parent itself remains exploratory. The formal deployable anchor therefore remains `77.699` HOTA.

## Results

| Phase | Method | Frozen held-sequence actions | HOTA | DetA | AssA | IDSW | Delta vs parent | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| M23-16 | corrected truncate-then-disjoint execution | M01 4, M02 0, M03 6, M05 0 | 78.783100 | 81.550795 | 76.161880 | 912 | +0.019603 | mechanism gain; below M23-14 |
| M23-17 | sequence-normalized utility | M01 0, M02 8, M03 0, M05 0 | 78.766270 | 81.551665 | 76.128656 | 904 | +0.002773 | first strict M02 recall, gain too small |
| M23-18 | domain ensemble and uncertainty | M01 5, M02 6, M03 7, M05 0 | 78.757430 | 81.548100 | 76.115020 | 910 | -0.006067 | agreement did not imply correctness |
| M23-19 | sequence-normalized sign classifier | M01 0, M02 0, M03 5, M05 0 | 78.720660 | 81.551576 | 76.041216 | 909 | -0.042837 | sign-only target discarded utility magnitude |
| M23-20 | conflict-graph normalized utility | all folds no-op | 78.763497 | 81.551695 | 76.123375 | 901 | +0.000000 | conflict features did not transfer |
| M23-21 | 128D pair-state metric MLP | all folds no-op | 78.763497 | 81.551695 | 76.123375 | 901 | +0.000000 | partner-aware supervised representation still domain-bound |
| M23-22 | pair-state score, no old OOD gate, K up to 2500 | all folds no-op | 78.763497 | 81.551695 | 76.123375 | 901 | +0.000000 | proves the ranking, not the gate, was the blocker |
| M23-23 | GT-free per-sequence self-supervised metric | M01 0, M02 9, M03 2, M05 2 | 78.503050 | 81.551380 | 75.620824 | 911 | -0.260447 | M02 improved, M05 high-cost false links; reject |

## Protocol controls

For M23-16 through M23-23, outer policy and transaction manifests were written and SHA-256 hashed before TrackEval or held-sequence utility diagnostics. Model labels and inner policy calibration use only the three outer-training sequences. Held-sequence tracker inference does not read GT.

M23-21 adds a fixed-seed 32-dimensional projection of 128-dimensional prefix/suffix and whole-track representations, represented by partner-aware elementwise products and absolute differences. A two-head MLP is trained on sequence-normalized utility and sign targets from fit sequences only.

M23-23 performs additional test-time adaptation without GT: consecutive microtracklets inside each parent track are pseudo positives, while temporally overlapping microtracklets from different parent tracks are hard negatives. This produces a deterministic per-sequence diagonal appearance metric. The adapted features are then consumed by the same nested sequence-LOSO utility policy. The old M23-14 probability OOD gate is disabled before freeze, and the budget grid is fixed to `10/25/50/100/250/500/1000/2500`.

## Frozen artifact hashes

| Phase | Selection SHA-256 | Policy SHA-256 |
|---|---|---|
| M23-16 | `8baa0a73982383c8821b4482b3458543a7bf4b5b8d682bf5bff07bb858ca70e3` | `c9eaa58fc4efcf13ffee4fcac112c24571e943ee3d347e359bd3c5af737da47c` |
| M23-17 | `ada8438e21f4dc73f1c4cab7593e86d97871b29a8412cf4a570ff771e2eed4be` | `eeeb73b1c449ae9cdf849680e91cc158b498c869b150ac0a3f652deb7dcfebad` |
| M23-18 | `d7fb2fdb4791c8abb109e2ecea9c5a15ddd2762318f267701561426dac90acc9` | `1a8cfce0bcf9befe378fd6ac50cf0614d8a73720e66fd7b13c33488958f31cc7` |
| M23-19 | `009340a05291ef23680d6a755023cf6907ffe397d6f2d7408dab166549c92192` | `72a8bc9468b20c33fa6d5feab932c45b4f7e832782cb5cd7c8e5850d38ca8749` |
| M23-20 | `224ef5814723ca409044da6b54664cb258a080efa128b0da7300a56351d204eb` | `1301f3d06d738699d0f200a48e96963aab11288f8e823af6537138f66f3cbf02` |
| M23-21 | `2167f1c88db68f82f59b05ca581a84fb0770262ef9d5a94f027e4a77ae0dec43` | `437b33721f5c8b400b05cc9a7a51b3b816eb0a2f6ddde22c4b47b3dbdf50e419` |
| M23-22 | `2167f1c88db68f82f59b05ca581a84fb0770262ef9d5a94f027e4a77ae0dec43` | `8c96576ed17f8e79a3934067c59ec824b998f62284f275b076104947f7bddecd` |
| M23-23 | `342ad69def4a6bf026ccdc705a323ffce241e3be2973db891ec64d562cfa040c` | `a98006af6f520c89db51bb7caca3dd33f677730656e6d3e31df1b5150300dc26` |

M23-23 self-supervised coordinate-weight hashes are recorded in `outputs/mot20_m23_20260718/m23_23_self_supervised_metric_utility_v1/self_supervised_metric_adaptation.json`.

## M23-23 held-sequence diagnosis

| Sequence | Actions | Positive actions | Utility sum | HOTA | AssA | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| MOT20-01 | 0 | 0 | 0.000000 | 78.819050 | 76.060380 | no-op |
| MOT20-02 | 9 | 5 | +376.827221 | 71.681005 | 63.831407 | real association gain, but still far below the required scale |
| MOT20-03 | 2 | 1 | +82.082278 | 80.584913 | 80.021670 | small gain |
| MOT20-05 | 2 | 0 | -6258.783443 | 79.008790 | 76.215374 | catastrophic high-stake false links |

The key mechanism result is not that self-supervision failed everywhere: it improved MOT20-02 HOTA by `+0.106095` relative to the parent sequence result. It failed because the same observable rule assigned high confidence to two extremely costly MOT20-05 transactions. A deployable policy therefore needs a parent representation and candidate generator whose identity evidence is calibrated before the transaction stage; another threshold or tree variant on the current graph is not sufficient.

## Decision and next mainline

Close the current transaction-selector family for score chasing. It has established a viable nondeployable action space (`80.980587`) and several GT-free local mechanisms, but the held-sequence error surface is not transferable enough to reach `80`.

The next 80+ mainline must move upstream:

1. fine-tune or train the parent association/ReID representation under strict sequence LOSO, rather than learning only a post-hoc transaction selector;
2. regenerate candidates from that representation, so MOT20-02 and MOT20-05 receive calibrated identity evidence before chain editing;
3. retain the same freeze-before-TrackEval and no-test-GT controls;
4. require a combined held-sequence HOTA above `80.000000` before promotion.

The gap is `+1.209997` HOTA from the current exploratory high (`78.790003`) and `+2.301000` from the formal deployable anchor (`77.699`). Results at the `+0.01` scale are mechanism evidence only, not goal completion.

## Reproduction artifacts

- Scripts: `scripts/m23_research/m23_16_truncate_then_disjoint_policy.py` through `m23_23_self_supervised_metric_utility.py`
- Outputs: `outputs/mot20_m23_20260718/m23_16_truncate_then_disjoint_policy_v1` through `m23_23_self_supervised_metric_utility_v1`
- Queue summary: `outputs/mot20_m23_20260718/summary.csv`
- Global experiment registry: `outputs/experiment_registry.csv`
