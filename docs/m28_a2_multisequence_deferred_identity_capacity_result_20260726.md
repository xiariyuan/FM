# M28-A2 Multi-Sequence Deferred Identity Inheritance Capacity — Result

Decision: **PASS_M28_A2_AUTHORIZE_STRONG_HOST_CAPACITY_GATE**

| Scope | Baseline HOTA | Teacher HOTA | Delta | Positive actions | Selected actions |
|---|---:|---:|---:|---:|---:|
| MOT20-01 | 76.737285 | 78.277445 | +1.540160 | 26 | 14 |
| MOT20-02 | 68.300265 | 71.654910 | +3.354645 | 124 | 92 |
| MOT20-03 | 79.866225 | 80.357569 | +0.491345 | 38 | 33 |
| MOT20-05 | 78.728026 | 79.800642 | +1.072615 | 165 | 123 |
| COMBINED | 77.698000 | 78.889000 | +1.191000 | 353 | 262 |

- Combined AssA: `74.700000 -> 77.013000`.
- Combined IDSW: `1156 -> 961`.
- All four sequence HOTA deltas are positive and every sequence has at least five positive actions.
- Candidate generation was frozen before GT; all results remain teacher-only and non-deployable.
- The current online-host ceiling `78.889000` is below strict M23-46 `79.123193` and below 80. Student training is therefore not yet authorized.

Next: M28-A3 strong-host M23-46 capacity gate.
