# M23-3-1 Deployable-Form Segment Replacement Oracle Audit

Date: 2026-07-17

## Research question

M23-3-0 showed that candidate-default actions are structurally useful under an oracle-enhanced post-NMS context. M23-3-1 asks a stricter question: **starting from the raw baseline, can a realistically executable same-ID box replacement action provide enough ceiling to justify training a segment-conditioned graph scorer?**

## Fixed executable action

The raw baseline tracker is reproduced exactly. The only permitted modification is to replace the box and score of an existing row by a frozen-budget pre-NMS candidate in the same frame whose `candidate_track_id` equals the original baseline tracker ID.

Hard constraints:

- no row additions or deletions;
- no tracker-ID changes;
- no tail-column changes;
- no suppressor routing;
- no spawn or new-ID creation;
- at most one replacement per `(sequence, frame, baseline tracker ID)`;
- no-op unless the best oracle utility is strictly positive.

Utility is candidate IoU to the contiguous segment’s unique modal train-GT identity minus the original baseline-row IoU to that identity. This is a train-GT ceiling, not a deployable result.

## Preregistration and precision correction

- v1 preregistration SHA: `eebe086d4f607a08d4d867b9405bbeeaa47fc8397b1aa6b9228ec732a3cade47`
- The complete v1 run reproduced every per-sequence detailed metric exactly, but `pedestrian_summary.txt` rounds combined values to three decimals. This caused a false exact-reproduction failure of 2e-5 to 5e-4.
- v1 trackers, replacement plans, metrics, and TrackEval outputs were preserved locally.
- v2 changed only the combined metric source to the detailed `COMBINED` row. Candidate sets, trackers, gates, and action definitions were unchanged.

- v2 preregistration SHA: `79115f2f16c4fbd91f8b16b381fca50a95d864f8c75ba54aecea99fd28849efe`
- final action script SHA: `0b931e4dd54b14ea48644001af464fa3d4927955d7eba41259abf957c72cf355`

## Candidate and replacement support

| Sequence | Frozen candidates | Eligible | M23-selected positive groups | Dense positive groups |
|---|---:|---:|---:|---:|
| MOT20-01 | 6,712 | 5,818 | 2,104 | 2,212 |
| MOT20-02 | 57,734 | 51,412 | 19,429 | 20,103 |
| MOT20-03 | 122,108 | 105,996 | 45,217 | 45,829 |
| MOT20-05 | 273,345 | 251,161 | 104,462 | 106,353 |
| COMBINED | 459,899 | 414,387 | 171,212 | 174,497 |

- Frozen candidates: **459,899**.
- Eligible candidates with observable baseline row, segment support, and target GT: **414,387**.
- M23-selected positive replacement groups: **171,212**.
- Dense positive replacement groups: **174,497**.
- The dense plan adds only 3,285 groups beyond the M23-selected plan.

## TrackEval results

| Variant | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline_raw` | 77.699020 | 80.907240 | 74.672127 | 89.308469 | 93.606460 | 1222 | 18,566 | 52,754 |
| `m23_selected_replacement_oracle` | 77.987070 | 81.252920 | 74.901140 | 89.324651 | 93.657931 | 1219 | 18,288 | 52,451 |
| `dense_replacement_oracle` | 78.001330 | 81.267375 | 74.915320 | 89.336889 | 93.688250 | 1220 | 18,118 | 52,276 |

### Gains

- M23-selected replacement: **+0.288050 HOTA**.
- Dense replacement: **+0.302310 HOTA**.
- Dense over selected: only **+0.014260 HOTA**.
- Dense replacement reduces FP by **448**, FN by **478**, and IDSW by **2** after MOTChallenge preprocessing.
- All four sequences improve, so the action is directionally safe in oracle space.

### Per-sequence HOTA

| Sequence | Baseline | Selected replacement | Delta | Dense replacement | Delta |
|---|---:|---:|---:|---:|---:|
| MOT20-01 | 77.029973 | 77.266490 | +0.236517 | 77.311534 | +0.281561 |
| MOT20-02 | 68.748330 | 68.958074 | +0.209744 | 68.966925 | +0.218595 |
| MOT20-03 | 80.054490 | 80.323430 | +0.268940 | 80.338050 | +0.283560 |
| MOT20-05 | 78.539010 | 78.856720 | +0.317710 | 78.871155 | +0.332145 |

## Why 174,497 oracle-positive replacements yield only +0.302 HOTA

- **164,268/174,497 (94.14%)** of dense group-best utilities are at most 0.05 IoU.
- Only **10,229** replacement groups improve IoU by more than 0.05.
- Replacement cannot create a missing observation, extend a trajectory into an absent frame, or repair a wrong global identity link.
- Raw tracker row count and `(frame, ID, tail)` skeleton are unchanged; TrackEval `Dets` can differ slightly because MOTChallenge distractor preprocessing retains or removes different rows after boxes move.
- Consequently, the dominant M23-1 ceiling is not box refinement. It comes from **observation recovery and identity reattachment**.

## Preregistered gate decision

| Gate | Requirement | Result |
|---|---:|---:|
| Selected combined HOTA | >=78.2 | 77.987070 - fail |
| Dense combined HOTA | >=78.5 | 78.001330 - fail |
| Selected sequence deltas | all nonnegative | pass |
| Dense sequence deltas | all nonnegative | pass |
| Baseline reproduction | exact | pass |

**Final decision:** close the direct segment-replacement graph as a learned main branch. Do not spend sequence-LOSO model capacity on predicting these replacements. The action may remain a diagnostic or secondary deterministic refinement, but it does not provide the ceiling required for the paper’s main innovation.

## Next stage: M23-4-0

Run a fixed **30-frame inter-segment gap-bridge addition oracle** from the raw baseline:

1. Preserve all baseline rows and IDs.
2. Permit a pre-NMS observation to be added only when the inherited existing tracker ID is absent in that frame.
3. Bridge only bounded gaps between compatible existing track segments; do not create a new ID.
4. Use at most one added observation per `(frame, inherited track ID)`.
5. Combine the bridge with the already audited dense replacement plan.
6. Keep suppressor routing and arbitrary spawn disabled.
7. Preregister one 30-frame gap, one IoU matching rule, and sequence-level nonnegative gates before TrackEval.

This directly tests the missing capability identified by M23-3-0/3-1: recovery of absent observations while retaining an existing identity chain.

## Reproducibility

- Compact outputs: **6/6 byte-identical**.
- Tracker files: **12/12 byte-identical**.
- TrackEval summary/detailed files: **6/6 byte-identical**.
- v1 precision-diagnostic trackers and TrackEval files are identical to v2.
- Unified audit: **90/90 checks passed**.

- formal report SHA: `95b1e75d18c1ee225bd30f96eba834c2121ab9dfd8037a4f09f10cf3a0b67223`
- formal manifest SHA: `92b1d6e0451420859a9a2c777e8d03dccc2c060dee0287735eb3b09e0e4eaf6f`
- unified audit SHA: `d037853e178ca4312e4bd651bb7907cf76c293a6f27cd936739f1ef87b5c4efa`

## Locked-state compliance

- P15 policy: no-op.
- Locked-label reads: 0.
- Locked TrackEval calls: 0.
- Remaining locked rows untouched: 156.
