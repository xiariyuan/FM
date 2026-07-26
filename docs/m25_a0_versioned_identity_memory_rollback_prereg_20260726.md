# M25-A0 — Versioned Identity Memory Rollback Capacity

Pre-registered: 2026-07-26

## Question

Can an online MOT tracker recover association capacity by rolling a track's appearance memory back to an earlier exact version, without changing the current detection assignment or motion update?

## Host

- DMMBaseTracker default `botsort_reid` replay.
- Frozen input: `outputs/alink_train_inputs/phase0_root/MOT20-01/dump_yolox_reid.npz`.
- Sequence: MOT20-01 train only for the kill gate.
- `dmm_v3_enable=false`.
- B0 must be byte-identical to the independently reproduced online baseline.

## Event freeze

Before GT is opened, record primary chosen matches with exact pre-update appearance versions. Eligible events require:

- track age and version count at least 16;
- valid track and detection appearance features;
- event frame at least 8 frames before sequence end;
- chosen primary assignment below its fixed matching threshold.

The fixed channel-diverse shortlist uses four GT-free risks:

1. low row/column pair margin;
2. low smooth-memory/current-detection cosine;
3. high recent-history drift;
4. large advantage of an older exact memory version over the current memory.

Round-robin selection uses at most 24 events, with 24-frame same-track NMS and at most two events per frame.

## Fixed actions

All actions preserve the current assignment and all geometric/Kalman/lifecycle updates. Only appearance memory is changed after the chosen update.

- `freeze0`: restore the exact appearance state immediately before the event update;
- `rollback4`: restore the exact state four accepted updates earlier;
- `rollback8`: restore the exact state eight accepted updates earlier;
- `rollback16`: restore the exact state sixteen accepted updates earlier.

The selected state is restored after each update to the same track for 8 frames, including the event frame. Appearance state consists of `smooth_feat`, `curr_feat`, the feature deque, and `alpha`.

## Freeze-before-GT contract

The event parquet, baseline tracker, every candidate tracker, implementation hash, input hash, candidate hashes, row counts, duplicate checks, and action-hit counters are frozen before TrackEval reads MOT20-01 GT.

## M01 kill gate

GO only if all conditions hold:

- best rollback action improves official HOTA by at least `+0.15`;
- at least four rollback candidates have positive HOTA delta;
- at least one paired rollback candidate beats `freeze0` at the same event by `+0.10` HOTA;
- no candidate has duplicate `(frame, track_id)` rows;
- B0 is byte-identical to the independent online baseline.

Otherwise M25 is fail-closed and is not expanded to M02/M03/M05 or model training.

## Scope

Teacher-only capacity diagnostic. Not deployable. MOT20 test reads/submissions are prohibited.
