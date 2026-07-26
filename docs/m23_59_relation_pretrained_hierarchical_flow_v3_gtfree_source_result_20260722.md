# M23-59 v3 GT-Free Source-Row Regeneration — M23-62 Result (2026-07-22)

## Decision

**PASS_GT_FREE_SOURCE_REGENERATION**.

M23-62 froze one canonical 144-D feature contract and regenerated seven MOT17 plus four MOT20 row-observable sets from source-tracker rows. All formula, semantic, GT-free lineage, phase0 compatibility, geometry round-trip and feature-143 round-trip checks passed.

## Canonical repairs

- global column 134 / geometry local 6 is the constant visibility-unavailable sentinel `1.0`, not measured visibility;
- global column 143 / geometry local 15 / one-based column 144 is nearest same-frame normalized source-row center distance clipped to `[0,1]`, singleton sentinel `1.0`;
- temporal features use source `track_id` in both domains;
- crowd density and nearest-neighbor populations use same-frame source rows in both domains;
- appearance uses the same frozen MOT20 detector/ReID phase0 settings and seed-2310 projection in both domains.

## Scope boundary

No MOT17 GT, MOT20 GT, MOT20 test, teacher action or held-outer label was read. No relation model was trained or replayed. No tracker was generated, no TrackEval was run and no test submission was created. The historical v2 checkpoint is incompatible with the changed training-side contract and was not reused.

## Next authorized stage

A later fresh experiment may join the already frozen MOT17 source rows to external supervision labels and retrain the unchanged v2 architecture from scratch. That experiment must reverify all M23-62 SHA before label access. MOT20 labels remain locked until four future v3 observables are regenerated and frozen under the strict outer event order.
