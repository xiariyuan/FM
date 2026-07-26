# M28-A2-R1 Strict Causal Multi-Sequence Capacity — Result

Decision: **PASS_M28_A2R1_AUTHORIZE_STRICT_SEQUENCE_LOSO_STUDENT**

All candidate generation is strictly causal: every old-identity row and ReID feature precedes the decision frame, every young feature is observed by the decision frame, and the measured future-row read count is zero.

| Scope | Baseline HOTA | Teacher HOTA | ΔHOTA | Baseline IDSW | Teacher IDSW | ΔIDSW |
|---|---:|---:|---:|---:|---:|---:|
| MOT20-01 | 76.737285 | 78.035563 | +1.298278 | 49 | 39 | -10 |
| MOT20-02 | 68.300265 | 70.647480 | +2.347215 | 420 | 366 | -54 |
| MOT20-03 | 79.866225 | 80.239064 | +0.372839 | 160 | 134 | -26 |
| MOT20-05 | 78.728026 | 79.699653 | +0.971627 | 527 | 426 | -101 |
| COMBINED | 77.697810 | 78.669953 | +0.972143 | 1156 | 965 | -191 |


- Combined HOTA: `77.697810 → 78.669953` (`+0.972143`).
- Combined AssA delta: `+1.883787`.
- Combined IDSW: `1156 → 965` (`-191`).
- Capacity retained versus the earlier noncausal audit: `81.62%`.
- Four sequences are HOTA-positive and contain at least five positive valid actions.
- This is teacher-only capacity evidence; no deployable tracker or test submission was produced.
- MOT20 test reads: `0`.

The strict sequence-LOSO student is authorized only for a host whose teacher ceiling can exceed the project target. The original online host reaches `78.669953`, so stronger-host stacking must be audited before student training.
