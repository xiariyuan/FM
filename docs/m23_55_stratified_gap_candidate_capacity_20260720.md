# M23-55 Stratified Long-Gap Candidate Recall Expansion

Date: 2026-07-20

## Protocol

- GT-free candidate ranking pools were frozen and SHA-256 recorded before GT diagnostics.
- Gap buckets 1-30, 31-90, 91-180 and 181-600 have independent outgoing/incoming quotas.
- Ranking pool K=256; teacher-flow subgraph K=32 per direction/bucket; all M23-53 edges retained.
- M23-54 was not started; no MOT20 test submission was made.

## Coverage

| Fold | Nonzero events | Expanded recall | Exact wrong-source alternative |
|---|---:|---:|---:|
| MOT20-01 | 25 | 96.00% | 56.52% |
| MOT20-02 | 272 | 95.59% | 77.54% |
| MOT20-03 | 122 | 95.90% | 35.29% |
| MOT20-05 | 431 | 93.74% | 42.92% |
| POOLED | 850 | 94.71% | 53.94% |

### Pooled recall by gap

| Gap | Events | Recall |
|---|---:|---:|
| 1-30 | 682 | 95.60% |
| 31-90 | 133 | 91.73% |
| 91-180 | 26 | 100.00% |
| 181-600 | 9 | 55.56% |

### Candidate scale and runtime

| Fold | Flow edges | Out K=256 | In K=256 | Frozen disk | Runtime |
|---|---:|---:|---:|---:|---:|
| MOT20-01 | 90931 | 197881 | 201384 | 0.03 GiB | 44.7 s |
| MOT20-02 | 912674 | 3125275 | 3121707 | 0.29 GiB | 94.1 s |
| MOT20-03 | 1775139 | 8670511 | 8986148 | 0.75 GiB | 259.3 s |
| MOT20-05 | 3955531 | 20694026 | 20605150 | 1.73 GiB | 704.1 s |

Peak process RAM/GPU memory was not instrumented in v1; the table reports exact frozen artifact disk size rather than inventing a peak-memory value.

### Independent ranking contribution

| Score view | Pooled weighted MRR | Pooled weighted R@32 |
|---|---:|---:|
| fixed multiview + motion score | 0.6438 | 88.00% |
| multiview appearance | 0.5419 | 83.65% |
| whole-track prototype | 0.5035 | 82.00% |
| motion-only rank | 0.7054 | 90.47% |

### Legacy rejection reasons for missing nonzero-gap successors

| First rejection reason | Events |
|---|---:|
| temporal eligibility / gap gate | 3 |
| spatial / motion gate | 0 |
| appearance threshold | 0 |
| global top-k filled by gap=0 | 4 |
| incoming/outgoing one-way omission | 57 |
| endpoint quality / node exclusion | 0 |
| source edge consumed cross budget | 0 |
| node identity impurity | 179 |
| legacy outgoing top-16 / unmodeled | 198 |

## Teacher capacity

| Fold | HOTA | DetA | AssA | IDSW |
|---|---:|---:|---:|---:|
| MOT20-01 | 80.816746 | 81.996185 | 79.791980 | 55 |
| MOT20-02 | 76.171464 | 81.599605 | 71.189874 | 435 |
| MOT20-03 | 81.552810 | 81.388250 | 81.747620 | 189 |
| MOT20-05 | 80.754274 | 82.070845 | 79.495770 | 580 |
| **COMBINED** | **80.385540** | **81.810826** | **79.031010** | **1259** |

- Relative to M23-53 teacher: **+0.434950 HOTA**.
- Relative to strict M23-46: **+1.262347 HOTA**, but this result is teacher-only and deployable=false.
- Capacity gate: **B (80.300000-80.700000)**. M23-54 remains locked.

## Observable OOF rankability

| Held fold | Canonical MRR | R@32 | Out top-1 precision | In top-1 precision | Wrong-source repair precision | Catastrophic false-link |
|---|---:|---:|---:|---:|---:|---:|
| MOT20-01 | 0.3579 | 88.00% | 89.83% | 88.80% | 0.00% | 10.17% |
| MOT20-02 | 0.1360 | 52.94% | 87.85% | 88.18% | 0.00% | 12.15% |
| MOT20-03 | 0.2107 | 79.51% | 92.67% | 91.79% | 0.00% | 7.33% |
| MOT20-05 | 0.2486 | 67.75% | 91.31% | 91.43% | 0.43% | 8.69% |

The recall-first candidate pool raises teacher capacity above 80.3, but observable cross-sequence ranking does not safely repair wrong source edges. Teacher conversion must not be treated as deployable rankability evidence.

## Implementation limitations

- M23-55 v1 gave every source the same per-gap outgoing/incoming quota and retained all prior source/cross edges. It did **not** add a separate suspicious-source-only extra quota. That conservative GT-free subset was frozen before GT and was not modified post hoc.
- The OOF rankability model is a diagnostic RobustScaler + logistic regression, not the separately gated sequence-normalized structured transfer student.
- Peak process memory was not instrumented; only exact frozen artifact disk size and wall-clock runtime are claimed.

## Decision

- Do not start M23-54 under the preregistered 80.7 gate.
- Do not train a large student and do not submit a new MOT20 test result.
- The only unlocked follow-up is a separately preregistered small strict nested-LOSO structured transfer feasibility audit; every failed inner gate must freeze the outer tracker to no-op.
- Best strict deployable result remains M23-46 HOTA **79.123193**.
