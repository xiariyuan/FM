# M26-A0 Motion-Prompted Pair-Conditioned ReID — Result

Decision: **FAIL_M26_A0_CLOSE_MOTION_PROMPTED_REID**

## Protocol integrity

- Online B0 remained byte-identical: `6cfb242bea622253d3e653c9c604d38624fab640cfb5cdec41a6cd78752a7bf8`.
- GT-free raw events: `16,523`; frozen events: `64`; frozen top-8 pairs: `512`.
- Detector and MOT20 FastReID weights were frozen; no model was trained.
- Prompt pair scores and hashes were frozen before MOT20-01 GT was opened.
- Full-frame GT matching was used for candidate labels; no top-k subset matching bias.
- MOT20 test reads/submissions: `0`.

## FastReID reproduction

- Median cached/re-extracted global cosine: `0.999994`.
- 5th percentile: `0.998839`.
- Minimum: `0.994772`.

The pre-registered integrity gate passed.

## Representation gate

| Measure | Original cost | Motion-prompted cost |
|---|---:|---:|
| Top-1 accuracy | 0.958333 | 0.895833 |
| Correct top-1 events | 46 / 48 | 43 / 48 |
| Fixes | — | 0 |
| Breaks | — | 3 |
| Net fixes | — | -3 |
| Oracle-union gain | — | +0.000000 |

## Conclusion

The primary frozen BoT-SORT cost already ranks the correct detection first for 95.83% of evaluable selected events. The fixed track-conditioned spatial prompt repairs none of the two original errors and breaks three correct decisions. Do not tune Gaussian widths, negative weights, thresholds, or train a prompt head on this event space.

The next authorized work is an exact IDSW source attribution audit, because the remaining association deficit is not explained by primary per-frame top-1 ranking.
