# M23-71A M23-46 Gap-20 Interpolation — Result

Decision: **FAIL_INTERPOLATION_NON_DEGRADATION_GATE**

| Scope | HOTA | ΔHOTA | AssA | DetA | IDSW | ΔIDSW |
|---|---:|---:|---:|---:|---:|---:|
| MOT20-01 | 78.740060 | -0.065065 | 76.010084 | 81.738955 | 46 | +0 |
| MOT20-02 | 73.110910 | +0.012760 | 66.485155 | 80.517790 | 317 | -8 |
| MOT20-03 | 80.601860 | -0.001420 | 80.081110 | 81.158346 | 140 | -6 |
| MOT20-05 | 79.769600 | -0.000727 | 77.694560 | 81.944100 | 463 | -16 |
| COMBINED | 79.122174 | -0.001019 | 76.843470 | 81.521916 | 966 | -30 |

- GT-free and reproducible: yes.
- Original rows retained semantically and no duplicate frame/ID: yes.
- Combined HOTA regressed by `-0.001019`; MOT20-01 regressed by `-0.065065`.
- Do not include this interpolation in the deployment stack.
