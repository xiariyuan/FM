# M23-70A Genuine Causal Branch Capacity — Result

Decision: **FAIL_CAUSAL_BRANCH_CAPACITY_CLOSE_OR_REDESIGN_BEFORE_TRAINING**

This is a teacher-only capacity audit, not a deployable result.

| Scope | HOTA | DetA | AssA | IDSW | ΔHOTA vs M23-46 |
|---|---:|---:|---:|---:|---:|
| MOT20-01 | 81.082183 | 82.201654 | 80.118054 | 34 | +2.277058 |
| MOT20-02 | 75.087000 | 81.891197 | 68.940365 | 400 | +1.988850 |
| MOT20-03 | 80.946570 | 81.452477 | 80.478716 | 183 | +0.343290 |
| MOT20-05 | 80.790570 | 82.131820 | 79.511570 | 532 | +1.020243 |
| COMBINED | 80.103000 | 81.906784 | 78.389130 | 1149 | +0.979807 |

- K: `3`.
- Delay: `8` frames.
- Gate: `80.80` HOTA.
- Combined HOTA: `80.103000`.
- M23-70B authorized: `false`.
- MOT20 test submission: `false`.
