# M23-3-0 Candidate-Default Graph Action-Space Oracle Audit

Date: 2026-07-17

## Scope and validity

This stage is a **MOT20-train oracle action-space audit**, not a deployable tracker result. The M23-1 post-NMS replace/add oracle is held fixed as context. Only the identity/fallback action of frozen-budget pre-NMS suppressed observations is changed. No model or parameter sweep is used.

The purpose is to determine whether a candidate-default global graph has enough structural ceiling before training a graph scorer.

## Fixed action space

Baseline tracker IDs are split into contiguous segments whenever the observed frame gap is greater than one. A segment obtains an oracle support identity from the unique modal train-GT identity among same-frame IoU>=0.5 Hungarian matches; tied modes are left unlabeled.

Actions use the fixed priority:

1. `candidate_default`: candidate segment supports the event target.
2. `suppressor_override`: candidate does not support the target but suppressor segment does.
3. `spawn`: neither existing segment supports the target.

The safe variants restore the corresponding post-NMS context row when it exists, otherwise they abstain. Spawn episodes use a fixed 30-frame gap and a new synthetic identity.

## Preregistration and fail-fast correction

- Original preregistration SHA: `6bfa5ba3882798434a500160293a1ee0234fdf9b590309976d905e8651b287e3`
- The first run stopped before episode-spawn metrics because synthetic IDs near 1.5e9 collided after TrackEval numeric loading. Candidate-safe partial metrics were diagnostic only.
- The failed run directory is preserved locally as `candidate_default_action_space_failed_floatid_v1`.
- v2 changed only the synthetic ID namespace to 10M–14M, below 2^24 and disjoint from all source tracker IDs. No action definitions, gates, or metrics changed.

- Corrected preregistration SHA: `b2f90ad8770c5a25aa7295b1a12512e12a8bfd5ba73bcd7142ac4417acac776f`
- Final action script SHA: `6bd62425faa18fd916540dc6e92e2bb091db44e71b52de7ad473410748f294f0`

## Structural action coverage

| Sequence | Candidate-default | Suppressor override | Spawn | Spawn fraction |
|---|---:|---:|---:|---:|
| MOT20-01 | 2,104 | 3 | 380 | 15.28% |
| MOT20-02 | 19,328 | 28 | 4,349 | 18.35% |
| MOT20-03 | 45,173 | 29 | 4,112 | 8.34% |
| MOT20-05 | 104,204 | 63 | 7,823 | 6.98% |
| COMBINED | 170,809 | 123 | 16,664 | 8.88% |

- Total selected suppressed events: **187,596**.
- Candidate-default covers **170,809 (91.05%)**.
- Suppressor override covers only **123 (0.066%)**.
- Spawn remains **16,664 (8.88%)**, concentrated most strongly in MOT20-02 (18.35%).
- Of the candidate-default events, **169,624/170,809 (99.31%)** are replacements rather than additions.

## TrackEval results

| Variant | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| `postnms_context` | 83.676 | 82.480 | 84.924 | 97.881 | 95.784 | 6 | 30,071 |
| `candidate_default_safe` | 84.226 | 83.054 | 85.447 | 98.142 | 96.286 | 71 | 24,304 |
| `candidate_suppressor_safe` | 84.227 | 83.054 | 85.448 | 98.142 | 96.286 | 71 | 24,302 |
| `candidate_suppressor_episode_spawn` | 83.597 | 83.352 | 83.887 | 96.931 | 95.278 | 17190 | 18,629 |
| `oracle_linked_spawn` | 84.523 | 83.357 | 85.738 | 98.396 | 96.792 | 8 | 18,630 |

### Candidate-default is the useful action

- Candidate-default safe improves HOTA from **83.676 to 84.226 (+0.550)**.
- Adding the oracle suppressor override changes HOTA only from **84.226 to 84.227 (+0.001 rounded)** and recovers only two additional true positives after preprocessing.
- Candidate+suppressor safe improves over post-NMS context on all four sequences.
- Its worst-sequence HOTA is **83.893** on MOT20-02, above the preregistered 83.5 gate.

### New-track spawn is not a valid substitute for re-linking

- The 30-frame episode-spawn variant drops to **83.597 HOTA** and creates **17,190 ID switches**.
- Its worst sequence is MOT20-02 at **82.816 HOTA**. Both preregistered spawn gates fail.
- Oracle-linked spawn restores **84.523 HOTA** and reproduces M23-1 exactly with zero metric delta.
- Therefore the missing capability is identity reattachment to an existing trajectory, not merely maintaining a locally continuous new ID.

## Failure-mechanism evidence

- **11,198** spawn events have candidate and suppressor segments both supporting another GT: these are genuine identity conflicts.
- **5,026** spawn events have both segments missing: these are the main short-gap propagation/evidence-coverage cases.
- There are 2,527 spawn episodes; 1,754 are multi-frame, with a maximum of 200 events. Local continuity alone still fragments global identity.
- 170,071 candidate-default events also have the suppressor supporting the same target, showing that the NMS suppressor is usually not an independent identity route.

## Formal decision

- **Retain:** candidate-default action space.
- **Do not train as a separate branch:** suppressor override; its oracle incremental value is negligible and M23-2 already rejected direct appearance routing.
- **Reject:** direct episode-spawn branch. Unsupported events must fail closed to the context/no-op path.
- **Appearance:** auxiliary edge evidence only.
- **Deployment:** false.
- **Locked manifest:** false.

## M23-3-1 next protocol

Proceed with a **sequence-LOSO segment-conditioned observation replacement graph**:

1. Each baseline contiguous segment is one graph component.
2. The baseline observation is always available as the no-op node.
3. Pre-NMS observations connect only through `candidate_track_id`; suppressor edges remain diagnostic.
4. Enforce at most one observation per segment per frame.
5. Disable spawn in the first learned graph; unsupported components abstain.
6. Score replacements using geometry, detector confidence, segment motion consistency, and appearance only as auxiliary evidence.
7. Evaluate with four outer sequence folds and require every sequence to be nonnegative against its baseline/no-op context.
8. Do not sweep outer thresholds, graph variants, or feature families after seeing outer metrics.

## Reproducibility

- Compact formal/reproduction files: **8/8 byte-identical**.
- Tracker files: **20/20 byte-identical**.
- TrackEval summary/detailed files: **10/10 byte-identical**.
- Unified audit: **60/60 checks passed**.

- Formal report SHA: `7e94c270cff5a9d83d056421ddb016431b5a24c353b8de69a89cb998797906c3`
- Formal manifest SHA: `c7ccac422d6b7d2f07691cd9bcc0a27537c356738d750a70d82ee2ac8a395164`
- Unified audit SHA: `5e395083183d49dadb08ad198a1ed4e85f22cfd0a414e716ebb3d36b57a2b5b1`

## Locked-state compliance

- P15 policy: no-op.
- Locked-label reads: 0.
- Locked TrackEval calls: 0.
- Remaining locked rows untouched: 156.
