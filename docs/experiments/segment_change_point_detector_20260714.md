# Segment Change-Point Detector — MOT20-02 pilot 2026-07-14

## Goal

Move from whole-track identity edits to segment-level reconstruction. The previous coverage audit showed 620/660 fragmented families require segmentation rather than ranking.

## Engineering

- Phase0 YOLOX + FastReID dump is accessed with direct mmap from uncompressed npz members.
- M02 detection rows: 160051, ReID dimension 2048.
- A43 track rows are aligned to detections by same-frame IoU.
- ReID alignment coverage: 99.6202%.

## Change-point feature dataset

MOT20-02:

- track rows: 151152
- boundaries: 148050
- persistent p5 switches: 218
- p5 transition-window positives (+/-3 frames): 1478

Features:

- local ReID continuity/change
- motion discontinuity
- bbox scale change
- overlap context
- frame density
- IoU alignment quality

## Detector audit

Validation:

- GroupKFold by tracker ID (diagnostic only, not cross-sequence validation).
- Candidate pool: overlap_max_ioa >= 0.1.

OOF:

- ExtraTrees AUC: 0.88107, AP: 0.08480
- HGB AUC: 0.87572, AP: 0.08020
- Ensemble AUC: 0.88331

Event NMS:

- budget 100: HGB recalls 16/218 exact events, precision upper bound 16%
- budget 500: HGB recalls 31/218, precision upper bound 15.5%

## Debt gating experiment

Adding identity debt as a hard gate:

- debt top 75% keeps 94.47% event-track recall while removing about 19.5% rows;
- debt top 50% loses too many events.

Conclusion: debt should be a feature or soft prior, not a strict candidate filter.

## Conclusion

Unary change-point detection has learnable signal but is insufficient as a direct segmentation controller. Identity switches are pair-state events. Next step should combine:

- segment boundary proposal;
- partner-aware pair state;
- AssA utility decision.

The segment detector is a candidate generator, not the final edit decision.
