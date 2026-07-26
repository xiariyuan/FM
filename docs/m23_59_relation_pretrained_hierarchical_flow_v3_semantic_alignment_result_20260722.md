# M23-59 v3 Semantic Alignment Validation — M23-61 Result (2026-07-22)

## Decision

**BLOCKED_GT_FREE_LINEAGE_AND_FULL_CONTRACT**. No counterfactual B/C, training, tracker, TrackEval, label unlock, strict outer evaluation, MOT20 test read or submission occurred.

## Canonical feature 143

Global zero-based column **143** = one-based column **144** = geometry local index **15**. Its unique canonical meaning is nearest same-frame normalized center distance, clipped to `[0,1]`, singleton sentinel `1.0`. This comes from the v2 preregistration and generic generator, not MOT20 results.

## Full contract result

- unique canonical definitions: **143/144**
- as-run formula/semantic parity: **141/144**
- MOT17 strict GT-free feature lineage: **0/144**
- MOT20 proposed GT-free lineage: **143/144**

Blocking findings: visibility has no uniquely preregistered GT-free sentinel/proxy; MOT17 features are generated from GT rows, temporal derivatives group by GT identity, crowd/nearest-neighbor populations are GT-derived, while MOT20 uses source-tracker rows. Therefore a feature-143-only repair cannot be certified as a 144-D GT-free semantic repair.

## Source split

Physical-video split and canonical FRCNN policy pass; physical overlap=0, exact image SHA overlap=0. This does not cure feature lineage failure.

## Historical distinctions

- v2 as-run output: byte-exact M23-46 P0 fallback, historical metrics unchanged.
- v2 scientific comparison: confounded by semantic mismatch.
- M23-60: post-hoc GT diagnosis only, not deployable and not strict.
- v3: front-end contract record only; no deployable tracker.

## Next allowed action

Create a new preregistered external source-row construction that does not use GT boxes/visibility/identity in feature generation, while keeping GT only as external supervision labels, and uniquely freeze the visibility missing-value contract. Then regenerate both MOT17 train/validation features and MOT20 observables from raw inputs in a fresh root before any replay or training.
