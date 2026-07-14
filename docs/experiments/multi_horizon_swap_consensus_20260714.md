# Multi-horizon Swap Consensus — 2026-07-14

## Goal

Evaluate deployable, sequence-agnostic consensus rules across h30, h60 and permanent AssA-aware swap rankers. The purpose is to reduce risky permanent swaps without using train-sequence GT outcomes as deployment rules.

## Policies and results

All policies are applied after the current best 70-link merge result.

| Policy | Events | COMBINED HOTA | Mean HOTA | M02 HOTA | M05 HOTA |
|---|---:|---:|---:|---:|---:|
| h30 safe | 5 | 78.765820 | 77.618516 | 71.576230 | 79.495050 |
| h60 safe | 6 | 78.767600 | **77.632552** | 71.575123 | 79.496676 |
| 3-way consensus permanent | 3 | 78.774714 | 77.621719 | 71.588230 | 79.508406 |
| 2-way consensus permanent aggressive | 6 | 78.782540 | 77.625115 | 71.588230 | 79.521990 |
| 2-way consensus permanent safe | 7 | **78.782636** | 77.626525 | 71.588230 | 79.521990 |
| 2-way consensus shortest finite | 6 | 78.765010 | 77.615119 | 71.576230 | 79.494005 |
| 2-way consensus longest finite | 6 | 78.767030 | 77.615783 | 71.575123 | 79.497770 |

The prior permanent aggressive15 fusion remains best in COMBINED HOTA at 78.813840. The h60 policy has the best macro mean among the multi-horizon candidates and slightly exceeds aggressive15 on mean HOTA, illustrating a small objective conflict between COMBINED and macro-average HOTA.

## Findings

- Three-way consensus is too conservative and retains only three events.
- Two-way permanent consensus removes the obvious M01/M02 risky events without GT-conditioned sequence selection, but also removes several M05-only long-horizon gains.
- Finite h30/h60 transactions cannot validate several useful long-term identity permutations.
- Threshold tuning on the current overlap candidate set is approaching saturation.

## Next direction

Move from event-level threshold tuning to family-level candidate coverage and large-scale identity-fragmentation recovery. Candidate generation, not only ranking, is now the main bottleneck.
