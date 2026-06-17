## FGAS

`FGAS` stands for `Frequency-Guided Association System`.

This project is the tracker-upgrade line for closing the gap to stronger MOT17
test methods. It is not another small pairwise rescoring plugin.

Core management decision:

- freeze `FCAA` as the narrow validation line that proved a small online signal
- start a new `FGAS` line that upgrades the association subsystem itself
- keep BoT-SORT as the carrier shell for now, but cover more than the primary
  appearance term

### Scope

`FGAS` covers four connected pieces of the tracking pipeline:

1. primary association: tracked + lost tracks against high-score detections
2. recovery association: residual tracked tracks against low-score detections
3. unconfirmed association: one-frame tracks against remaining detections
4. lifecycle scoring: initialization, re-activation, and retirement support

The first milestone still starts with a bounded problem:

- export a `block-bank` instead of a pair-bank
- each sample is a local conflict block with multiple rows and columns
- learn edge scores and row-level no-match decisions jointly
- keep frequency as one signal among appearance, motion, quality, and lifecycle

### Why this line exists

`FCAA` only refined the first-stage appearance matrix and delivered a real but
small gain. The remaining gap to stronger 2025 MOT17 methods is much larger
than that plugin can plausibly recover.

`FGAS` moves the contribution from:

- pair-level rescoring

to:

- stage-aware conflict resolution across the association subsystem

### First implementation target

Phase 1 is offline only:

1. build `block-bank` samples from teacher-forced replay
2. train a minimal block resolver
3. compare against row-wise / pair-wise controls on the same block-bank

Phase 2 is online integration:

1. replace primary-stage local assignment on triggered blocks
2. extend to recovery and unconfirmed stages only if Phase 1 is positive

### Stop discipline

This line gets one clean mainline.

Stop if the first block-level resolver fails to show:

- clear offline block-ranking gains over pair-wise control
- stable online gains on strict half-val after bounded runtime integration

Do not reopen the old heavy frequency family.
Do not keep growing the old `FCAA` plugin if this line becomes the new mainline.
