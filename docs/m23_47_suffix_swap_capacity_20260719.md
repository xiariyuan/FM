# M23-47 two-boundary suffix-swap capacity audit

Date: 2026-07-19

## Protocol

- Baseline: strict deployable M23-46 outer-held trackers and byte-identical GT-free applied graphs.
- Action: remove two selected source edges from two different current chains, then cross-connect both suffixes.
- Constraints: time-forward, one-to-one, acyclic, different current chains.
- Candidate construction: GT-free microtracklet structure, projected appearance prototypes, motion and temporal features.
- Exact HOTA teacher is opened only after each sequence shortlist is frozen.
- All results in this document are teacher-only and `deployable=false`.

## Runtime validation

The first sparse-graph smoke produced only one reciprocal candidate. The implementation was then extended to synthesize reciprocal cross edges from GT-free microtracklet prototypes and geometry rather than requiring both edges to exist in the original TOPK=16 candidate graph.

The v2 M02 smoke froze 8 GT-free swaps. All 8 passed one-to-one and acyclic reconstruction and exact teacher evaluation; one swap improved HOTA by +0.012922.

## Full 64-swap capacity results

| Sequence | M23-46 baseline HOTA | M23-47 teacher HOTA | Delta | Positive swaps | Successful labels |
|---|---:|---:|---:|---:|---:|
| MOT20-01 | 78.805125 | 78.812253 | +0.007129 | 1 | 64/64 |
| MOT20-02 | 73.098153 | 73.199564 | +0.101411 | 3 | 64/64 |
| MOT20-03 | 80.603278 | 80.671161 | +0.067884 | 2 | 64/64 |
| MOT20-05 | 79.770327 | 79.777288 | +0.006962 | 2 | 64/64 |

Candidate parquet audit found no `same_gt`, `modal_gt`, `purity`, `label_confidence`, `delta_HOTA`, or related GT/teacher fields.

## Official stitched capacity

Four per-sequence teacher-best trackers were concatenated and evaluated once with official TrackEval:

- HOTA: **79.159210**
- DetA: **81.543493**
- AssA: **76.894796**
- IDSW: **993**
- Delta versus strict M23-46: **+0.036017 HOTA**
- Margin to 80: **-0.840790 HOTA**

Conclusion: two-boundary reciprocal suffix swap is mechanically valid and has independent positive actions, but this action family and shortlist are far below the capacity required for COMBINED HOTA > 80. It must not be reported as deployable or as a strict OOF result.

## Decision

M23-47 is closed as a capacity-insufficient action family. Do not tune its classifier or thresholds.

The next action family must alter a larger identity segment than a reciprocal suffix swap. The next probe should support a directed interval transfer: cut two boundaries on one donor chain and one insertion boundary on another chain, detach the donor interval, and splice it into the receiver while reconnecting the donor prefix/suffix. Candidate generation must remain GT-free, with one-to-one, acyclic and time-direction constraints verified before exact teacher evaluation.
