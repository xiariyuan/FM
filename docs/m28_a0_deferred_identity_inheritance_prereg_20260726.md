# M28-A0 — Deferred Identity Inheritance Capacity

Pre-registered: 2026-07-26

## Motivation

Exact CLEAR attribution on the deterministic M01 online host found that 23/49 ID switches originate when an unconfirmed young track is promoted after a temporal gap. M28 decouples geometric track confirmation from permanent identity allocation: a young confirmed tracklet may inherit an older lost identity rather than permanently keeping the ID allocated at geometric birth.

## Frozen host

- `DMMBaseTracker`, default `botsort_reid`, `dmm_v3_enable=false`.
- MOT20-01 phase0 detection/ReID dump.
- Online B0 SHA-256: `6cfb242bea622253d3e653c9c604d38624fab640cfb5cdec41a6cd78752a7bf8`.
- Runtime update log was frozen before GT in M27-A0.

## GT-free event/candidate construction

Events are all `update:unconfirmed` confirmations in the frozen runtime log.

For each young track, candidate inherited identities must:

- be a different tracker ID;
- have ended strictly before the confirmation frame;
- have gap in `[1, 120]`;
- contain at least four output observations and at least two valid ReID updates;
- have no frame overlap with the young track suffix.

Candidate score is fixed as:

`appearance_cos - 0.35 * motion_error - 0.03 * log(1 + gap) - 0.08 * abs(log(height_ratio))`.

The top eight candidates per event are frozen. No GT-derived field is allowed in the candidate parquet.

## Teacher action

The static capacity action replaces the young track ID by the candidate inherited ID from the confirmation frame onward, leaving every detection box, score, row order and all other IDs unchanged. Candidate actions must have no duplicate `(frame, ID)` rows.

After the candidate parquet and hashes are frozen, exact official-preprocessed HOTA labels are computed. Positive non-conflicting actions are greedily considered by exact individual HOTA delta and accepted only if their exact incremental HOTA gain remains positive. One inherited identity and one young track may be used at most once.

## M01 GO gate

Authorize a true online anonymous-track implementation only if:

- at least eight candidate actions have positive exact HOTA delta;
- best single action improves HOTA by at least `+0.10`;
- exact compatible combined teacher improves HOTA by at least `+0.50`;
- output integrity passes and official TrackEval verifies the final teacher tracker.

Otherwise fail-close deferred identity inheritance without training.

## Scope

Teacher-only capacity diagnostic. Not deployable. MOT20 test reads/submissions are prohibited.
