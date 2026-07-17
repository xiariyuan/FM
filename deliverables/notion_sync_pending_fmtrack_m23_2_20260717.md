# FM-Track / MOT20 M23-2 — Appearance Evidence Audit

Status: completed, fail-closed
Date: 2026-07-17

## Final decision

- Appearance authenticity selector: rejected.
- Direct appearance identity router: rejected after disclosed strong-baseline correction.
- Appearance feature bank: retained only as auxiliary evidence for a candidate-default global graph.
- Deployment: false.
- Locked manifest: false.
- P15: no-op; 156 locked rows untouched.

## Core numbers

- Frozen candidates: 459,899; unique crops: 459,891.
- FastReID cache: 8,931 frames, 12 shards, 1,883,713,536 bytes, nonfinite=0.
- Compact features: 35, nonfinite=0.
- Additive oracle positives for selector: 17,715.
- Geometry AP 0.102922 vs appearance AP 0.096596; appearance delta -0.006326.
- Geometry top10 precision/recall 0.137437/0.356816.
- Appearance top10 precision/recall 0.137154/0.356082.
- Identity cases: 857.
- Appearance 61.84%, geometry 19.72%, candidate-only 88.91%, suppressor-only 11.09%.
- Appearance is 27.07 percentage points below candidate-only.

## Integrity corrections

1. Candidate key table originally used integer fancy indexing, which returned a copy. Preflight rejected the invalid table before training. Rebuilt keys are 459,899/459,899 unique; feature SHA unchanged.
2. Selector target was corrected before prediction from replace/add positives to additive positives because fixed top10 recall 40% was otherwise mathematically impossible.
3. Original identity audit omitted candidate-only. A disclosed post-hoc correction preserved the preregistered output and added the stronger baseline. This correction is authoritative.

## Reproduction

- Key rebuild 4/4 files byte-identical.
- LOSO 8/8 files byte-identical.
- Strong-baseline correction 6/6 files byte-identical.
- Unified audit 29/29 passed.

## Next stage

M23-3 candidate-default counterfactual global tracklet graph:
- candidate track is default edge;
- suppressor assignment is a minority override edge;
- appearance is auxiliary evidence only;
- integrate temporal path consistency, tracklet motion and global constraints;
- use K-best solutions and baseline-relative solution-level risk control;
- require nonnegative four-sequence cross-fitted results.

Notion connector is not writable in this session. This file is the pending synchronization payload.
