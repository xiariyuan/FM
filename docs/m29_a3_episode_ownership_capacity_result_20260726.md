# M29-A3 Episode Ownership Capacity — M01 Kill-Gate Result

## Decision

`FAIL_M29_A3_CLOSE_EPISODE_OWNERSHIP`

M29-A3 used the byte-frozen M28-B1 teacher on strict M23-46 geometry as C0. All C1/C2 candidates and feature-time audits were frozen before GT was opened. Detection boxes, scores, row order and geometry IDs were unchanged; candidate scoring used no feature after each three-observation decision frame.

## Exact and official results

| Stage | HOTA | Delta | AssA | Official IDSW |
|---|---:|---:|---:|---:|
| C0: M28-B1 teacher | 79.696351 | — | 77.762997 | 42 |
| C1: mature episode re-certification | 80.157995 | +0.461644 | 78.660697 | 40 |
| C2: add reciprocal ownership arbitration | 80.157995 | +0.000000 over C1 | 78.660697 | 40 |

C1 had 2 positive actions and selected 2. C2 had 1 individually positive action, but selected 0 after C1. The only positive reciprocal action overlapped the rows already modified by a stronger C1 action and provided no compatible incremental gain.

## Mechanism interpretation

C1 crossed 80 on M01, but its output transform is an episode-bounded identity reassignment already contained by historical interval/transaction action families. The proposed new element was the composed ownership state with independent reciprocal arbitration. That component failed its preregistered `+0.15` incremental gate (`0.000000`). Therefore this result cannot support a new ownership-state contribution or four-sequence expansion.

The frozen candidate universe was also too narrow for the diagnosed mature-error population: only 7 C1 events and 8 C2 actions were produced on M01. Most mature ID switches occur inside continuous visibility runs rather than at frame-gap episode boundaries.

## Closure

- Do not expand M29-A3 to M02/M03/M05.
- Do not train an ownership student.
- Do not rename the C1 interval gains as a new contribution.
- M23-46 remains the strict deployable best at 79.123193 HOTA.
- MOT20 test reads/submissions remain zero.
