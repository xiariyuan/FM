# ShadowPDA Online Constraints, Reflection, and Next Step

> Purpose: prevent offline audit conclusions from leaking into the final online algorithm.

---

## 1. Hard Boundary: Final Method Must Be Online

ShadowPDA must run causally. At frame `t`, the algorithm may only use:

```text
current detections at frame t
current detection scores and features
current active/lost/removed track states
historical track memory before or at frame t
historical shadow state before or at frame t
historical pending/hold state before or at frame t
current active-track competition information
```

The final algorithm must not use:

```text
future frames
future detections
GT labels
TrackEval feedback
sequence name
sequence-specific manual choice
post-hoc release ratio over the whole video
offline what-if selection
```

If a signal requires GT or future frames, it may only be used for diagnosis, ablation, or design explanation. It must not enter the online inference path.

---

## 2. What Was Offline Audit Used For?

The following analyses were diagnostic only:

```text
GT audit of recovered low detections
future 10-frame audit after hold candidates
offline what-if threshold sweeps
comparison of correct / wrong_id / det_fp recovery events
```

These analyses answered questions such as:

```text
why HOTA dropped
whether wrong recoveries were wrong_id or det_fp
whether hold candidates had future high-score opportunities
which online-observable variables correlate with safer recovery
```

They do not define the final inference procedure.

---

## 3. Current Online Modules and Status

### 3.1 Direct Contact-Supported Low Recovery

Online rule:

```text
low evidence can public recover only if:
  identity_margin >= threshold
  contact_overlap >= threshold
  det_score >= threshold
  lost_age <= threshold
  consecutive_support >= threshold
  low-to-low consistency is stable
```

Status:

```text
works online
filters wrong_id and isolated det_fp better than earlier versions
safe but small gain
```

Best safe variant so far:

```text
identity_margin >= 0.03
contact_overlap >= 0.20
det_score >= 0.50
lost_age <= 6
```

More aggressive audited variant:

```text
identity_margin >= 0.03
contact_overlap >= 0.20
det_score >= 0.35
ultra_reliability >= 0.65
lost_age <= 6
```

On MOT20-02 full, offline audit showed the aggressive variant produced 9 recovered events, all correct. This is only diagnostic until tested on hard negative sequences.

---

### 3.2 PendingPublic

Online rule:

```text
medium low evidence enters pending state
next frame must confirm by stable low evidence before public recovery
```

Status:

```text
mechanism works online
pending candidates in MOT20-02 full were correct in audit
promotion opportunities are rare
no meaningful TrackEval gain yet
```

Reflection:

```text
PendingPublic is safe but too dependent on immediate next-frame evidence.
It should not be the main recovery path unless promotion opportunities increase.
```

---

### 3.3 ShadowHold

Online rule:

```text
medium low evidence reinforces hidden shadow state
no public recovery
no re_activate
no output
```

Status:

```text
mechanism works online
hold_preserved > 0
but no high_recovered increase in MOT20-02 full
no metric improvement over pending/direct variant
```

Reflection:

```text
Hold preserves shadow state but current high recovery channel does not use it effectively.
Increasing hold duration alone is unlikely to help unless a safe high-score recovery channel is added.
```

---

## 4. Important Baseline Parity Issue

The recovered ShadowPDA script with recovery disabled gives a different MOT20-02 baseline than the older ParamBest baseline.

Old historical baseline:

```text
MOT20-02 HOTA = 68.821
IDF1 = 75.131
IDSW = 417
FN = 11561
FP = 2336
```

Current same-script no-recovery baseline:

```text
MOT20-02 HOTA = 68.748
IDF1 = 75.205
IDSW = 428
FN = 11556
FP = 2329
```

Therefore, while developing ShadowPDA variants, internal comparisons must use the same-script no-recovery baseline. Final paper-level comparisons require restoring parity with the historical ParamBest baseline or clearly reporting the exact baseline implementation used.

---

## 5. Current Main Reflection

The current results suggest three lessons:

```text
1. Low-score detections can help, but direct public recovery must be highly constrained.
2. Contact-supported identity dominance is a meaningful online risk model.
3. ShadowHold and PendingPublic are conceptually valid but currently do not create enough additional public recoveries.
```

The most promising near-term direction is not adding more states, but validating whether the aggressive online direct-recovery variant remains safe on high-risk sequences.

---

## 6. Next Step

Do not continue adding new mechanisms yet.

Next step:

```text
Run the aggressive online direct-recovery variant on MOT20-05 short window, then audit.
```

Use the online-only rule:

```text
identity_margin >= 0.03
contact_overlap >= 0.20
det_score >= 0.35
ultra_reliability >= 0.65
lost_age <= 6
consecutive_support >= 2
low_iou >= 0.20 or low_center_step <= 80
```

Run order:

```text
1. MOT20-05 first 500 frames
2. GT audit only for diagnosis
3. If safe, MOT20-05 full
4. If safe, MOT20-03 full
5. If safe, MOT20 train all with same parameters
```

Success criteria for MOT20-05 short window:

```text
no wrong_id recovery
no det_fp recovery
IDSW not increased in TrackEval short/full evaluation
```

If MOT20-05 fails, the aggressive variant is not safe and the safe direct variant should remain the main online module.

If MOT20-05 passes, then run full-sequence evaluation and compare against the same-script no-recovery baseline.

---

## 7. One-Sentence Method Direction

```text
ShadowPDA should remain an online, causal, proposal-level method: low-confidence detections accumulate hidden identity evidence, and public recovery is allowed only when current evidence is identity-dominant, contact-supported, temporally stable, and short-age.
```
