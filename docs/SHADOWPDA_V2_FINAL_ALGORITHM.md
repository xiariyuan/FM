# ShadowPDA-V2 Final Algorithm Definition

## 1. Motivation

ShadowPDA-V2 stops treating low-confidence recovery as a collection of hand-tuned gates. The final method is an online, causal shadow-state confirmation framework.

Core claim:

```text
Low-confidence detections are probabilistic evidence for hidden identity existence under clutter, not immediate public recovery observations.
```

Therefore, a low-score detection first updates a shadow state. Public recovery is allowed only when the current online evidence produces a sufficiently high public recovery posterior.

---

## 2. Online Constraint

At frame `t`, ShadowPDA-V2 may use only:

```text
current detections and scores
current active/lost tracks
historical track feature memory
historical shadow state
current active-track competition
current low-to-low temporal consistency
```

It must not use:

```text
GT labels
future frames
future detections
TrackEval feedback
sequence-specific switches
post-hoc sequence statistics
```

Offline audit is allowed only for diagnosis and ablation.

---

## 3. State Variables

For each lost track, ShadowPDA maintains a hidden state:

```text
existence_logit
reliability = sigmoid(existence_logit)
support_count
consecutive_support
p_best
p_clutter
entropy
margin
avg_memory
avg_det_score
last_low_box
low_iou
low_center_step
active_overlap
lost_age
```

These are updated online from low-confidence detections and historical memory.

---

## 4. Public Recovery Posterior

For the current low detection `d` associated with shadow identity `i`, define a public recovery score:

```text
S_public =
    w_R       * z_reliability
  + w_id      * z_identity_margin
  + w_app     * z_lost_app
  + w_det     * z_det_score
  + w_contact * z_contact
  + w_temp    * z_temporal
  - w_clutter * z_p_clutter
  - w_entropy * z_entropy
  - w_age     * z_lost_age
```

Where:

```text
z_reliability     = reliability
z_identity_margin = clipped identity margin
z_lost_app        = lost-track memory similarity
z_det_score       = detection confidence
z_contact         = current contact / occlusion context
z_temporal        = low-to-low consistency from previous evidence
z_p_clutter       = clutter posterior
z_entropy         = PDA ambiguity entropy
z_lost_age        = normalized lost age
```

The recovery decision is:

```text
if shadow is confirmed and S_public >= theta_public:
    public recover
else:
    keep shadow
```

This replaces separate hard gates such as `det_score >= a` and `contact_overlap >= b` as the main decision mechanism.

---

## 5. Minimal Safety Constraints

A few constraints remain as physically meaningful guards rather than the method core:

```text
lost_age <= max_lost_age
support_count >= min_support
consecutive_support >= min_consecutive_support
temporal consistency must not be degenerate
```

These prevent impossible long-gap or single-frame recovery. They are not the main idea.

---

## 6. V2 Default Parameters

Default global parameters are fixed for all sequences:

```text
theta_public = 2.0
w_R       = 1.0
w_id      = 1.2
w_app     = 0.8
w_det     = 0.7
w_contact = 0.8
w_temp    = 0.8
w_clutter = 1.0
w_entropy = 0.6
w_age     = 0.4
max_lost_age = 6
min_support = 2
min_consecutive_support = 2
```

They should be evaluated globally, not tuned per sequence.

---

## 7. What Becomes Ablation

The following modules are not the final method core:

```text
instant OSR
pending-high
sequence-level warm-up
primary guidance
PendingPublic
ShadowHold
hard contact-threshold sweeps
hard det-score-threshold sweeps
```

They are useful ablations for understanding failure modes.

---

## 8. Evaluation Plan

Run V2 with one fixed parameter set:

```text
1. MOT20-02 full
2. MOT20-05 first 500-frame diagnostic audit
3. MOT20-05 full if diagnostic is acceptable
4. MOT20-03 full
5. MOT20-01 full
6. MOT20 train all
```

Compare against:

```text
same-script no-recovery baseline
safe direct contact+identity recovery
ShadowPDA-V2 posterior
```

Important metrics:

```text
HOTA
IDF1
FN
FP
IDSW
Frag
recovered event correctness via offline audit only
```

---

## 9. One-Sentence Method

```text
ShadowPDA-V2 performs online track-before-recover by accumulating hidden low-confidence evidence under a clutter hypothesis and using a causal public-recovery posterior to decide when a lost identity may safely re-enter the public tracker state.
```
