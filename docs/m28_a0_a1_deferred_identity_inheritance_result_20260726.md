# M28-A0/A1 Deferred Identity Inheritance — Result

## M28-A0 capacity

Decision: **PASS_M28_A0_AUTHORIZE_TRUE_ONLINE_ANONYMOUS_IDENTITY**

- Frozen unconfirmed-confirmation events: `60`.
- Frozen GT-free inheritance candidates: `388`.
- Successful exact labels: `388`.
- Positive actions: `26`.
- Best single exact HOTA gain: `+0.348842`.
- Exact compatible selected actions: `14`.
- Baseline HOTA: `76.737285`.
- Combined teacher HOTA: `78.277445`.
- Combined gain: `+1.540160`.
- Official TrackEval: HOTA `78.277`, AssA `76.552`, IDSW `39`.

Candidate analysis after freeze:

- 23 exact unconfirmed-source IDSW events.
- Correct predecessor identity appeared in top-8 for `12/23` events.
- Correct predecessor rank: 11 at rank-1 and 1 at rank-2.
- `11/12` covered predecessor actions had positive exact HOTA.
- Directly inheriting top-1 for every event is unsafe; no-op prediction is mandatory.

## M28-A1 bounded online parity

- Internal geometric track IDs and association state remained unchanged.
- External identity decisions were committed exactly three frames after confirmation.
- Maximum output latency: `3` frames.
- B0 online output SHA: `6cfb242bea622253d3e653c9c604d38624fab640cfb5cdec41a6cd78752a7bf8`; byte-identical to independent B0: `True`.
- Online identity-layer teacher SHA: `2b7bec11897bffddadebdeb11351c8ed4958ce83bb53e990a09f5e7319e0d588`.
- Static teacher SHA: `2b7bec11897bffddadebdeb11351c8ed4958ce83bb53e990a09f5e7319e0d588`.
- Online/static teacher byte identity: `True`.
- Duplicate `(frame, external identity)` rows: `0`.

## Scientific interpretation

The dominant M01 failure is premature permanent identity allocation, not object-trajectory initialization. A geometrically valid young tracklet can inherit a lost identity after a bounded evidence delay without perturbing the host tracker. This separates object existence, geometric continuity and identity naming into distinct decisions.

The result is teacher-only and not deployable. It authorizes multi-sequence candidate/label construction and strict sequence-LOSO learning; it does not authorize MOT20 test submission.
