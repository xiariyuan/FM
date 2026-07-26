# M27-A0 Exact IDSW Source Attribution — Result

## Integrity

- Runtime association log and online B0 were frozen before GT was opened.
- B0 SHA-256: `6cfb242bea622253d3e653c9c604d38624fab640cfb5cdec41a6cd78752a7bf8`.
- Recorded state updates: `19,018`.
- Reconstructed CLEAR IDSW: `49`; official CLEAR IDSW: `49`.
- MOT20 test reads/submissions: `0`.

## Decomposition

| Source | IDSW | Share |
|---|---:|---:|
| Unconfirmed-track confirmation | 23 | 46.9% |
| Established primary update | 18 | 36.7% |
| Low-score second stage | 5 | 10.2% |
| Lost-track reactivation | 3 | 6.1% |

- After-gap switches: `39/49` (79.6%).
- Immediate switches: `10/49` (20.4%).
- Current tracker age 1–5: `25/49` (51.0%); all occurred after a gap.

## Conclusion

The dominant failure is not primary same-frame top-1 ranking. Almost half of all switches occur when a recently born unconfirmed track is promoted after the true identity has disappeared, and four-fifths occur after a temporal gap. The next mechanism must decouple geometric track confirmation from permanent identity allocation and allow a confirmed anonymous tracklet to inherit a lost identity after evidence accumulation.
