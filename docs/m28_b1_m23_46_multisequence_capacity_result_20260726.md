# M28-B1 Deferred Identity Inheritance on M23-46 — Result

Decision: **FAIL_M28_B1_CAPACITY_BELOW_80P8_NO_STUDENT_ON_M23_46**

| Scope | Baseline HOTA | Teacher HOTA | ΔHOTA | Baseline IDSW | Teacher IDSW | ΔIDSW |
|---|---:|---:|---:|---:|---:|---:|
| MOT20-01 | 78.805125 | 79.696350 | +0.891225 | 46 | 42 | -4 |
| MOT20-02 | 73.098150 | 73.402210 | +0.304060 | 325 | 314 | -11 |
| MOT20-03 | 80.603280 | 80.698250 | +0.094970 | 146 | 141 | -5 |
| MOT20-05 | 79.770327 | 80.049000 | +0.278673 | 479 | 438 | -41 |
| COMBINED | 79.123193 | 79.363860 | +0.240667 | 996 | 935 | -61 |


- Combined HOTA: `79.123193 → 79.363860` (`+0.240667`).
- Combined AssA delta: `+0.471155`.
- Combined IDSW: `996 → 935` (`-61`).
- Every sequence is HOTA-positive and candidate generation has zero future-row reads.
- The preregistered teacher ceiling was `80.80`; the observed ceiling is `79.363860`.
- Therefore no strict LOSO student is authorized on the M23-46 host.
- Teacher-only, deployable=false, MOT20 test reads/submissions=0.
