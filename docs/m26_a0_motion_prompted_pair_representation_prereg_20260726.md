# M26-A0 — Motion-Prompted Pair-Conditioned ReID Gate

Pre-registered: 2026-07-26

## Hypothesis

A detection should not have one fixed appearance embedding for every candidate track. In crowded frames, the candidate track's predicted spatial state can act as an automatic positive prompt, while competing track states act as negative prompts. Spatially prompted pooling of the frozen FastReID feature map may isolate the identity evidence relevant to the specific track-detection pair.

## Host and evidence

- Host: deterministic online `DMMBaseTracker`, default `botsort_reid`, `dmm_v3_enable=false`.
- Frozen sequence: MOT20-01 train.
- Frozen detector/ReID dump: `outputs/alink_train_inputs/phase0_root/MOT20-01/dump_yolox_reid.npz`.
- Frozen FastReID: MOT20 SBS-S50; no training or parameter update.
- B0 replay must be byte-identical to SHA-256 `6cfb242bea622253d3e653c9c604d38624fab640cfb5cdec41a6cd78752a7bf8`.

## GT-free event and candidate freeze

From primary Hungarian matches, keep events with track age >= 16 and valid appearance. Freeze at most 64 events through round-robin channels:

1. lowest chosen-pair row/column margin;
2. highest overlap of the chosen detection with another detection;
3. highest number of track-prediction centers inside the chosen detection;
4. largest mismatch between predicted-track box and chosen detection box.

Use 12-frame same-track NMS and at most four events per frame. Freeze the top eight candidate detections by the original final association cost for each event. Store track predicted boxes, all competing predicted boxes, track smooth features, candidate boxes, cached global features, IoU costs, embedding costs and original final costs before opening GT.

## Fixed motion prompt

For each track-detection pair on the 24x8 FastReID backbone map:

- positive prompt: Gaussian derived from the candidate track predicted box mapped into the detection crop;
- negative prompt: maximum Gaussian response from other predicted tracks overlapping the same detection;
- final spatial weight: `positive * (1 - 0.9 * negative) + 0.02`;
- pooling: weighted generalized-mean pooling using the frozen head's learned `p`;
- neck: the unchanged frozen FastReID BN neck.

The prompted appearance distance replaces only the embedding branch of the original BoT-SORT cost. IoU masking, appearance threshold and `min(IoU, appearance)` fusion remain unchanged.

## Integrity gate

Re-extracted full-map embeddings must reproduce cached phase0 embeddings with median cosine >= 0.999 and 5th percentile >= 0.995.

## Label-after-freeze gate

After pair features and hashes are frozen, open MOT20-01 GT only to label event-track identity and candidate detections.

GO to an online HOTA experiment only if, among events where the correct detection is in the frozen top-8:

- prompted top-1 accuracy improves over original final-cost top-1 by at least 8 percentage points;
- prompted scoring fixes at least six original top-1 errors;
- fixes minus breaks is at least four;
- the oracle union of original and prompted top-1 improves by at least 10 percentage points;
- integrity checks pass.

Otherwise fail-close Motion-Prompted ReID without online TrackEval or training.

## Scope

Teacher-after-freeze representation gate. Not deployable. MOT20 test reads/submissions are prohibited.
