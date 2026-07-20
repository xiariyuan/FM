# M23-57 Optional Intra-node Change-point Capacity and Observability Audit

## Final decision

```text
M23-57 v2 protocol valid
teacher capacity gate = PASS
strict observability decision = branch C
M23-58 = prohibited
strict deployable best remains M23-46
```

M23-57 demonstrates that optional internal cuts provide substantial **teacher-only** capacity, but the frozen local observable representation does not transfer safely across sequences. Therefore the action-space hypothesis is supported, while the deployable representation hypothesis is rejected.

## Valid-run provenance

- M23-57 v1 is **invalid**: after GT had been opened, a post-freeze diagnostic attempted to convert a one-way missing rank represented by infinity to an integer. M03/M05 were stopped, every v1 metric was excluded from selection, and no fold was resumed or locally patched.
- M23-57 v2 is a complete rerun from empty GT-free boundary directories. Its only code change maps non-finite diagnostic ranks to `null`; scientific protocol, features, actions, candidates, teacher objective and gates are unchanged.
- v2 did not copy, hard-link or reuse v1 boundary, teacher, tracker or metric artifacts. Recomputed v2 boundary artifacts are byte-identical to v1 because the pre-GT scientific input is unchanged.

## M23-53B equivalence audit

M23-53B is **not** semantically equivalent to M23-57:

1. It enumerates many observable boundaries but admits at most one preselected boundary per fixed chunk.
2. It rebuilds nodes and the source/cross/dummy graph after that preselection.
3. The teacher can cut the one introduced internal parent edge, but cannot choose an unselected boundary.
4. Unselected boundary features remain diagnostic records rather than independent actions.
5. Multiple internal cuts and complete A→B→A repair are therefore outside the M23-53B action space.

## GT-free boundary universe

| Fold | Detection rows | Fixed nodes | Independent boundaries | Features | Freeze time | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|
| 01 | 19,372 | 698 | 18,674 | 108 | 20.4s | 1.08 GiB |
| 02 | 151,152 | 5,290 | 145,862 | 108 | 139.4s | 1.93 GiB |
| 03 | 305,937 | 10,602 | 295,335 | 108 | 269.1s | 3.37 GiB |
| 05 | 638,588 | 22,228 | 616,360 | 108 | 544.7s | 6.45 GiB |

Total independent boundary actions: **1,076,231**. All 108-dimensional schemas are identical, all forbidden-column audits are empty, and all four freezes occurred before any teacher label was opened. No chunk contained an internal M23-46 parent-ID transition.

## Teacher change-point taxonomy

| Fold | Impure nodes | Change points | Supported audit positives | Unsupported CP | A→B→A | Split-affected rows |
|---|---:|---:|---:|---:|---:|---:|
| 01 | 49 | 99 | 40 | 59 | 43 | 1,393 |
| 02 | 390 | 823 | 430 | 393 | 341 | 11,508 |
| 03 | 177 | 347 | 210 | 137 | 151 | 5,003 |
| 05 | 437 | 836 | 471 | 365 | 354 | 12,511 |

Across four folds there are **2,105** deterministic teacher change points in **1,053** impure fixed nodes, including **889** A→B→A patterns. Unmatched rows were ignored as identities; nonchosen equivalent boundaries and unsupported transitions were marked ignored rather than negative.

## Oracle split + M23-55-style flow capacity

| Fold | Split nodes | Candidate recall | Flow conversion | HOTA | ΔHOTA vs M23-55 | AssA | DetA | IDSW |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 01 | 797 | 99.7183% | 98.7288% | **81.742230** | +0.925484 | 81.363344 | 82.259690 | 19 |
| 02 | 6,113 | 98.1323% | 98.5414% | **77.593540** | +1.422076 | 73.571980 | 81.907290 | 193 |
| 03 | 10,949 | 99.9210% | 99.6342% | **81.885120** | +0.332310 | 82.330110 | 81.469610 | 65 |
| 05 | 23,064 | 99.7143% | 99.3425% | **81.573320** | +0.819046 | 81.043880 | 82.141143 | 277 |
| **COMBINED** | — | — | — | **81.148046** | **+0.762506** | 80.424900 | 81.920034 | 554 |

The COMBINED teacher HOTA is **81.148046**, exceeding the preregistered 80.700000 gate by **0.448046** and improving on M23-55 by **+0.762506**. This is teacher-only and not deployable.

The split graph adds 2,105 nodes. It contains 7,548,198 edges; teacher flow keeps 35,145 parent edges, cuts 2,913, and selects 2,424 cross edges. Overall canonical recall is 99.5305% and conversion when present is 99.2898%. Of 504 newly expressible internal canonical successors, 492 are present and 480 are selected.

Every fold reconstructs M23-46 byte-exact from parent edges, preserves detection rows/boxes/scores, and passes one-to-one, time-forward and acyclic checks. Four fold TrackEval runs and exactly one COMBINED TrackEval run were executed.

## Strict sequence-LOSO boundary observability

| Held | Base rate | PR-AUC | ROC-AUC* | Precision | Recall | Precision@actual count | Recall@95% precision | Pure-node false-split | Median abs offset |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 01 | 0.2174% | **0.032318** | 0.884671 | 5.9406% | 15.0000% | 5.0000% | 0.0000% | 0.2554% | 2.5 rows |
| 02 | 0.3010% | **0.029489** | 0.866030 | 4.1424% | 14.8837% | 4.6512% | 0.0000% | 0.6833% | 3.0 rows |
| 03 | 0.0721% | **0.021688** | 0.923907 | 2.9192% | 18.5714% | 2.3810% | 0.4762% | 0.3361% | 3.0 rows |
| 05 | 0.0779% | **0.016987** | 0.879234 | 2.8846% | 13.3758% | 4.6709% | 0.0000% | 0.2713% | 3.0 rows |

\* ROC-AUC is auxiliary only; the base rate is 0.072%–0.301%, so PR metrics control the decision.

Four-fold mean PR-AUC is **0.025120**, mean precision at the actual split count is **4.1758%**, and mean recall at 95% precision is **0.1190%**. Mean pure-node false-split rate is **0.3865%**.

## Comparison with M23-56

| Metric | M23-56 source error | M23-57 local boundary | Difference |
|---|---:|---:|---:|
| PR-AUC | 0.182984 | 0.025120 | -0.157863 |
| Precision@actual count | 21.9072% | 4.1758% | -17.73 pp |
| Recall@95% precision | 1.2017% | 0.1190% | -1.08 pp |

The local change-point task is not more separable. It is substantially worse on all three preregistered evidence metrics. M02/M05 do not uniquely collapse—the failure is broad across all four sequences.

## Strict decision

```text
Branch C
teacher action-space capacity = sufficient
current local observable representation = not safely transferable
do not train a deployable change-point student
do not start M23-58
next candidate only = relation-specific external pretraining
```

The strict deployable best therefore remains M23-46 at HOTA 79.123193. M23-57 becomes the highest teacher capacity at HOTA 81.148046, but must never be presented as deployable.

## Resources and audit

- Boundary freeze total: 973.7s.
- Four capacity folds total: 1688.2s; maximum RSS 15.34 GiB; maximum GPU allocation 350.7 MiB.
- Four observability folds total: 122.9s.
- Host: 48 physical / 96 logical CPUs, 503.56 GiB RAM, B1.gpu.large with 23.25 GiB GPU memory.
- The MCP environment snapshot helper returned `name 'sys' is not defined`; an equivalent manual JSON snapshot was saved and hashed.
- M23-54 was not started. M23-58 was not started. No MOT20 test submission was made.

## Key SHA-256

- Preregistration: `06ef9d70f2fe319de8e787e7aedcf74895c39736b31da1146c84cec24a5778d5`
- Implementation manifest: `faca4cd7f9d98485037de4fb3eeed1228bd3165005da8b5a680b66e29b62ae08`
- Capacity script: `3314e8e900553efd77d06ef5f8ebc944fdfffbf01bf6136d7b10052d69361501`
- Observability script: `43f272f772f754912471e340f137fbc8a1297a7463d83c08048596d65f041697`
- COMBINED report: `e9e74ad402fd0def243d95e6933ecc27f847533522953e61b4a25e71d2a64226`
- Observability summary: `0f055c5df932140ac472b44c0b51e372dbcaa0cf9e80d8a856c463c7f6569d5f`
- Invalid v1 manifest: `377914f807b8601fc9c40dc488861f741f211a17f139aca4ad8454a2cbb6ff59`
