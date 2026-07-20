# M23-53 Candidate Coverage and Flow Conversion Audit

Date: 2026-07-20

## Status

- Role: post-freeze teacher-only diagnostic; deployable=false.
- Frozen candidate graphs, teacher utilities, selected flows, trackers and official TrackEval outputs were read-only.
- No candidate regeneration, GT-driven tuning, teacher-weight change, TrackEval rerun, student training or test submission was performed.
- Audited Git input: `d5fc3407c7e4960ed6835781fae0b04d753b39af`.

## Integrity checks

All four folds passed candidate node/edge SHA-256 verification, forbidden-column audit, freeze-before-GT certification, and M23-46 byte-exact baseline reconstruction.
The existing M02 smoke was also reverified without rerunning it: baseline byte-exact=true, frozen edges=6900, forbidden frozen columns=0, one-to-one/acyclic/time-forward=true, official HOTA=73.896456.

## Existing-code reuse audit

- `m23_11_eval_utility_graph.py`: reused sparse maximum-weight bipartite matching, reconstruction and official-evaluation patterns.
- `m23_12_chain_transaction_oracle.py`: reused chain transaction and tracker-writing conventions.
- `m23_37_fast_exact_hota_teacher.py`: reused exact-HOTA/official-verification infrastructure, but single-action HOTA delta was intentionally not used as the M23-53 objective.
- `eval_max_weight_identity_path_cover.py`: provided the legacy matching/path-cover precedent, but only supported track-level positive merge links and lacked frozen source/cross/dummy semantics and freeze-before-GT evidence.
- M23-53 therefore retained the existing matching/reconstruction machinery and added only the unified source/cross/dummy flow representation required by the preregistered protocol.

## Candidate coverage

| Fold | GT-successors | Candidate recall | Recall gap<=600 | Selected/present | Official HOTA |
|---|---:|---:|---:|---:|---:|
| MOT20-01 | 612 | 97.55% | 97.55% | 98.99% | 80.84379 |
| MOT20-02 | 4905 | 96.33% | 96.33% | 98.98% | 75.02046 |
| MOT20-03 | 9775 | 99.36% | 99.36% | 99.78% | 81.342626 |
| MOT20-05 | 20521 | 98.69% | 98.69% | 99.61% | 80.35353400000001 |
| POOLED | 35813 | 98.53% | 98.53% | 99.57% | 79.95059 |

Primary successor definition: earliest strictly time-forward, non-overlapping chunk with the same post-freeze teacher GT identity. This diagnostic definition does not alter the flow objective.

## Current source-edge errors and alternatives

| Fold | Source edges | Cross-GT | Unmatched | Cross-GT any alt | Cross-GT exact alt |
|---|---:|---:|---:|---:|---:|
| MOT20-01 | 606 | 3.80% | 0.50% | 47.83% | 39.13% |
| MOT20-02 | 4852 | 3.85% | 1.05% | 51.87% | 36.90% |
| MOT20-03 | 9848 | 1.04% | 1.02% | 23.53% | 22.55% |
| MOT20-05 | 20647 | 1.13% | 1.59% | 24.46% | 18.45% |
| TOTAL | 35953 | 1.52% | 1.34% | 34.68% | 26.42% |

## Teacher flow actions relative to M23-46

| Fold | Keep | Cut | Cross | Swap pairs | Affected rows | Affected rate |
|---|---:|---:|---:|---:|---:|---:|
| MOT20-01 | 577 | 29 | 20 | 0 | 1689 | 8.72% |
| MOT20-02 | 4594 | 258 | 104 | 1 | 13977 | 9.25% |
| MOT20-03 | 9636 | 212 | 58 | 1 | 10170 | 3.32% |
| MOT20-05 | 20036 | 611 | 158 | 3 | 29028 | 4.55% |
| TOTAL | 34843 | 1110 | 340 | 5 | 54864 | 4.92% |

`Affected rows` counts frozen detection rows in chunks incident to an edited edge. Actual detection row values, boxes and scores changed: **0**.

## Stratified pooled recall

### By temporal gap

| Gap | Events | Recall | Selected/present |
|---|---:|---:|---:|
| 0 | 34963 | 99.75% | 99.83% |
| 1-30 | 682 | 53.08% | 75.14% |
| 31-90 | 133 | 27.82% | 94.59% |
| 91-180 | 26 | 30.77% | 100.00% |
| 181-600 | 9 | 22.22% | 100.00% |

### By sequence-normalized crowd tertile

| Crowd | Events | Recall | Selected/present |
|---|---:|---:|---:|
| low | 11264 | 99.04% | 99.59% |
| medium | 11588 | 98.56% | 99.72% |
| high | 12961 | 98.06% | 99.41% |

### By GT trajectory-support tertile

| Length | Events | Recall | Selected/present |
|---|---:|---:|---:|
| high | 21987 | 98.97% | 99.56% |
| low | 3489 | 97.33% | 99.59% |
| medium | 10337 | 98.00% | 99.58% |

## Correct-continuation ranks when present

| Rank | N | P50 | P90 | Top-5 | Top-32 |
|---|---:|---:|---:|---:|---:|
| out_rank | 35286 | 1.0 | 4.0 | 92.56% | 100.00% |
| in_rank | 35286 | 1.0 | 4.0 | 93.25% | 99.70% |
| max_rank | 35286 | 1.0 | 6.0 | 89.65% | 99.70% |
| appearance_out_rank | 35286 | 1.0 | 2.0 | 94.75% | 100.00% |
| appearance_in_rank | 35286 | 1.0 | 2.0 | 95.58% | 99.84% |
| motion_out_rank | 35286 | 1.0 | 1.0 | 99.93% | 100.00% |
| motion_in_rank | 35286 | 1.0 | 1.0 | 99.90% | 100.00% |

## Capacity decision

- Frozen fixed-chunk teacher COMBINED: HOTA **79.950590**, DetA **81.820434**, AssA **78.174204**, IDSW **1337**.
- The result remains below the preregistered 80.300000 no-student floor.
- The already-completed uniform GT-free adaptive repair reached 79.920490 and is also closed.
- M23-54 remains prohibited under the preregistered gate.
- Best strict deployable result remains M23-46 HOTA 79.123193.

## Output artifacts

- `outputs/mot20_m23_20260718/m23_53_candidate_coverage_audit_v1/report.json`
- `outputs/mot20_m23_20260718/m23_53_candidate_coverage_audit_v1/successor_events.parquet`
- `docs/generated/M23_53_CANDIDATE_COVERAGE_AUDIT_20260720.json`
