# MOT20 M23-35 to M23-37 audit — 2026-07-19

## Standing strict protocol

- Outer-held sequence GT is excluded from model fitting, candidate generation,
  policy calibration, and tracker construction.
- Outer-held GT is opened only after the tracker and policy are frozen.
- Final success remains a single four-sequence stitched TrackEval run with
  `COMBINED HOTA > 80.000000`; per-fold HOTA averaging is not accepted.
- Current accepted strict stitched result remains `79.025010 HOTA`.

## M23-35: component-risk matcher

M23-35 introduced a catastrophic-risk head, local conflict-component features,
and an implicit no-op. The formal outer-M05 run failed:

- HOTA: `78.758836`
- DetA: `81.852967`
- AssA: `75.834644`
- IDSW: `714`
- Delta versus strict M23-25 M05: `-0.894367 HOTA`

The exact same-graph post-hoc audit found 104 positive and 192 negative
selected transactions. The selected transaction precision was 35.02%, and the
summed proxy utility was negative.

A second post-hoc audit showed that the pooled-inner policy selection criterion
had hidden an inner-M01 regression. A risk-heavier policy improved all three
inner folds and reached `80.068330 HOTA` on M05 post hoc. This result is
strictly diagnostic and is not a reportable outer score.

## M23-36: worst-inner-fold robust freezing

M23-36 changed only the policy-freezing rule. Each inner TrackEval candidate is
parsed per sequence and ranked primarily by the worst per-sequence HOTA delta
versus no-op.

For formal outer-M02, the frozen policy was
`r2_u025_b025_q0975_p002`:

- Inner COMBINED HOTA: `80.552435`
- M01 delta: `+0.349930`
- M03 delta: `+0.483654`
- M05 delta: `+0.843873`
- Worst-fold delta: `+0.349930`

The outer-M02 tracker was frozen before GT access, but the final result failed:

- HOTA: `71.834040`
- DetA: `80.541800`
- AssA: `64.178073`
- IDSW: `357`
- Delta versus strict M23-25 M02: `-1.034903 HOTA`

Post-hoc evaluation of all five unique policy action sets showed that the entire
M23-36 policy family was below the M23-25 M02 baseline. Importantly, several
sets had positive summed transaction proxy utility while reducing true HOTA.
This established that the remaining error was a set-level objective mismatch,
not only a threshold or policy-freezing mistake.

## M23-37: exact incremental HOTA teacher

M23-37 uses official TrackEval MOT20 preprocessing and the official HOTA
matching equations in process. Tracker boxes and GT are cached; candidate
trackers replace only tracker IDs. The incremental evaluator caches baseline
potential-match contributions, global alignment, Hungarian assignments, and
19-alpha statistics. It recomputes only changed partitions and affected frames.

### Numerical validation

On MOT20-02:

| Tracker | Official HOTA | Incremental HOTA |
|---|---:|---:|
| M23-25 | 72.868943 | 72.868943214 |
| M23-36 | 71.834040 | 71.834039688 |

The evaluator remains exact to normal floating-point output precision. Candidate
evaluation dropped from roughly 84 seconds for a full official in-process HOTA
recomputation to about 2.3–2.5 seconds even when all 2,782 M02 frames were
affected. On M01 single transactions, the mean was about 0.31 seconds.

### GT-free shortlist and single-action bank on M01

A 96-action shortlist was frozen using only model predictions and structural,
motion, and appearance features. GT was opened only after shortlist freezing to
produce teacher labels.

- Exact-HOTA-positive actions: `33 / 96`
- Positive rate: `34.375%`
- Best single-action delta: `+0.524598 HOTA`
- Worst single-action delta: `-1.181680 HOTA`

Single-action proxy utility had high M01 rank correlation with exact delta HOTA
(Spearman `0.9854`). Therefore the principal failure is interaction among
multiple selected transactions rather than lack of a usable individual-action
signal.

### Exact-HOTA forward set teacher on M01

The set teacher greedily evaluated the exact HOTA marginal of every disjoint
eligible action under the currently selected set. It stopped after eight
positive steps.

- Parent HOTA: `78.819048`
- Selected actions: `8`
- Incremental final HOTA: `79.955351`
- Official TrackEval check: `79.955350`
- Final DetA: `81.931650`
- Final AssA: `78.168090`
- IDSW: `41`
- Delta versus parent: `+1.136302 HOTA`
- Delta versus strict M23-25 M01: approximately `+0.773870 HOTA`

This is teacher-only evidence, not a deployable or strict outer result. It shows
that a GT-free shortlist contains a substantially better action set and that
true set-level HOTA marginals can identify it.

## Next registered direction

1. Build exact-HOTA teacher banks and forward set trajectories on at least two
   additional outer-training sequences.
2. Train a structured imitation model to predict the next-action marginal and
   stop/no-op decision from GT-free state and candidate features.
3. Validate by nested LOSO: teacher trajectories may use only each fold's three
   outer-training sequences; the outer-held sequence remains GT-free until one
   final TrackEval.
4. Do not treat teacher-selected trackers or post-hoc policy probes as strict
   scores.
