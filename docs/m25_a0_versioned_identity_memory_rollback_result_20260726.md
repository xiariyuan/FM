# M25-A0 Versioned Identity Memory Rollback — Result

Decision: **FAIL_M25_A0_CLOSE_VERSIONED_MEMORY_ROLLBACK**

## Protocol integrity

- Host: deterministic online `DMMBaseTracker` / `botsort_reid` replay.
- MOT20-01 B0 rows: `18,842`.
- B0 SHA-256: `6cfb242bea622253d3e653c9c604d38624fab640cfb5cdec41a6cd78752a7bf8` (byte-identical to the independent replay).
- Frozen GT-free events: `24` from `16,249` eligible primary matches.
- Frozen candidate trackers: `96` (`freeze0`, rollback-4/8/16).
- All target updates were hit exactly once; no duplicate `(frame, track_id)` rows.
- Candidate hashes were frozen before MOT20-01 GT was opened.
- MOT20 test reads/submissions: `0`.

## Official TrackEval

| Variant | HOTA | ΔHOTA | DetA | AssA | IDSW |
|---|---:|---:|---:|---:|---:|
| Online B0 | 76.737 | — | 80.237 | 73.541 | 49 |
| Best freeze0 | 76.742 | +0.005 | 80.244 | 73.543 | 51 |
| Best rollback | 76.740 | +0.003 | 80.241 | 73.543 | 49 |

- Positive rollback candidates: `1` / `72`.
- Maximum paired rollback gain over the same event's `freeze0`: `+0.000` HOTA.
- Only `11/96` candidates changed the final tracker, yielding just `5` unique tracker hashes including B0.

## Conclusion

Exact rollback to older versions of the same global 2048D identity vector does not create useful corrective capacity. The rollback depth is not the limiting factor; most appearance-state differences never cross an online association decision boundary, and the few that do are not beneficial. Do not expand to other sequences, scan depths/hold windows, or train a rollback selector.

The next authorized probe must introduce new localized visual evidence rather than another update rule over the same global embedding.
