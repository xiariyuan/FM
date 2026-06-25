# RG-OT Phase 0.5: Interface Audit

**Date:** 2026-06-18  
**Auditor:** Codex  
**Status:** COMPLETE

## Audit Objective

Define a viable interface for **RG-OT (Resolved-Gram Online Tracking)** inside the current BoT-SORT/FM-Track codebase.

This audit is not an implementation plan for a full method. It answers four narrower questions:

1. What existing runtime mechanisms already overlap with RG-OT?
2. Where can RG-OT attach without duplicating TOS, plain `track_buffer`, or reentry memory?
3. What signals already exist at runtime and are strong enough to support an `analysis-only` RG-OT pass?
4. What structured outputs should RG-OT produce before any behavioral rewrite is allowed?

---

## Executive Judgment

**RG-OT is viable as a new primary-association recovery line, but only if it is scoped as a local online reassignment mechanism, not as another long-gap archive/reentry heuristic.**

The codebase already contains three adjacent mechanisms:

1. **Reentry memory / query engine** for removed-track recovery.
2. **OwnerAlt competition** for stale-challenger reclaim against a weak recent owner.
3. **Local graph reassociation** for local block-level reassignment under utility and safety constraints.

RG-OT should therefore be defined as:

> A runtime mechanism that detects ambiguous local association neighborhoods and resolves them using structured recovery signals, especially when the baseline current owner is weak but the case is not well explained by plain long-gap reentry.

If RG-OT is instead defined as "hold tracks longer", "freeze appearance updates", or "recover removed tracks from archive", it collapses into already explored dead ends or already existing modules.

---

## What RG-OT Must NOT Be

### 1. Not TOS under a new name

TOS already failed its `analysis-only` gate on MOT20-05:

- event count too small
- no strong concentration in occlusion zones
- weak explanatory power beyond baseline memory

RG-OT must not reuse the TOS framing of:

- shadow track holding
- freeze-on-occlusion
- newborn hold
- "longer survival solves the problem"

Those paths are already weak or negative.

### 2. Not plain `track_buffer` scaling

The latest `tb30 -> tb60 -> tb90` control shows only minimal gains on MOT20-05. RG-OT cannot claim novelty if its main effect is just preserving identities longer inside the same recovery window.

### 3. Not plain unmatched reentry recovery

The tracker already has:

- `reentry_memory_enable`
- `reentry_memory_compete_primary`
- `reentry_engine_enable`

These cover removed-track archive search and reactivation. If RG-OT only queries old identities after they are removed, it is functionally a reentry wrapper.

### 4. Not another generic learned score blender

The codebase already contains multiple score-rescaling or selective rewrite layers:

- HACA / Laplace
- TCGAU
- RGSA
- graph-assoc learned commit / gate

RG-OT should start as a **mechanism audit + neighborhood decision problem**, not as another early-stage scalar controller.

---

## Existing Runtime Mechanisms Relevant to RG-OT

### A. Reentry memory and query engine

**Key locations**

- `external/BoT-SORT-main/tracker/bot_sort.py:602-712`
- `external/BoT-SORT-main/tracker/bot_sort.py:2598-2795`
- `models/reentry_query_engine/engine.py`

**What it already does**

1. Keeps removed tracks in archive for a bounded gap window.
2. Scores unmatched detections against archived tracks.
3. Can optionally let archived tracks **compete with primary matches** before newborn creation.
4. Exposes structured metrics:
   - candidate tracks
   - candidate detections
   - candidate pairs
   - reactivated tracks
   - competitive reactivated tracks

**Why this matters for RG-OT**

This is the cleanest existing implementation of **identity recovery after disappearance**.

**Why RG-OT should not collapse into it**

Reentry operates after a track has already moved into archive or removed-track logic. RG-OT should target the harder middle zone:

- still-local ambiguity
- stale-but-not-removed competition
- reclaim vs owner swap decisions
- neighborhood rewrite before full loss/removal

### B. OwnerAlt competition

**Key locations**

- `external/BoT-SORT-main/tracker/owneralt_competition.py`
- `external/BoT-SORT-main/tracker/bot_sort.py:1490-1498`
- config wiring at `external/BoT-SORT-main/tracker/bot_sort.py:884-907`

**What it already does**

OwnerAlt identifies cases where:

- a challenger track is stale enough to be interesting,
- the current owner is weak enough to be replaced,
- an alternative detection exists for the current owner,
- the challenger-vs-owner edge deficit remains bounded.

Then it rewrites local costs so the challenger can reclaim the detection.

**Why this matters for RG-OT**

Mechanically, OwnerAlt is already very close to an RG-OT-style "resolved local ownership transfer" idea.

**Its current limitation**

It is still primarily a handcrafted reclaim rule centered on:

- gap threshold
- tracklet length
- box IoU
- bounded owner-edge deficit

It does not define a richer local neighborhood state or a broader recovery taxonomy.

### C. Local graph reassociation

**Key locations**

- `external/BoT-SORT-main/tracker/local_graph_reassoc.py`
- `external/BoT-SORT-main/tracker/bot_sort.py:908-958`
- runtime call at `external/BoT-SORT-main/tracker/bot_sort.py:1499-1507`

**What it already does**

1. Forms a local ambiguous block.
2. Enumerates assignment alternatives.
3. Applies reclaim bonuses / owner penalties / safety rules.
4. Optionally uses learned commit scorers.
5. Exports structured event rows and summary metrics.

**Why this matters for RG-OT**

This is the strongest immediate substrate for RG-OT.

If RG-OT is about resolving local ambiguity using online evidence, then local graph reassociation is already the closest carrier:

- block structure exists
- candidate rows exist
- utility reasoning exists
- safety gates exist
- JSONL/summary style instrumentation already exists

**Current limitation**

Graph-assoc is still framed as a local reassignment/refinement utility mechanism, not as a broader online recovery theory.

---

## Best Attachment Point for RG-OT

## Recommendation: attach RG-OT to the primary-association rewrite layer, not to removed-track archive recovery

The relevant primary association path is:

1. build `dists`
2. optional `owneralt_refiner.refine_primary_cost(...)`
3. optional `graph_assoc_refiner.refine_primary_cost(...)`
4. linear assignment
5. post-match updates / second association / newborns

This happens in:

- `external/BoT-SORT-main/tracker/bot_sort.py:1490-1585`

### Why this is the right place

RG-OT should answer:

> When the baseline local assignment is ambiguous, under what structured online conditions should we rewrite ownership before the tracker falls back to lost/archive logic?

That is a primary-association question, not a late archive query question.

### Why not attach RG-OT only after removal

If attached only to:

- `_recover_removed_tracks()`
- `ReentryQueryEngine.query()`

then RG-OT degenerates into:

- archive search
- delayed reactivation
- long-gap identity recall

which is already covered by reentry.

---

## Proposed RG-OT Scope

RG-OT should be scoped as:

### RG-OT = local resolved recovery on ambiguous assignment neighborhoods

The target case is:

- a detection has multiple plausible owners, or
- a current owner is weak / recent / unstable, while
- an alternate continuation or reclaim path exists, and
- the decision is not reducible to plain archive reentry.

This places RG-OT between:

- **simple current-owner matching**
- and **full removed-track reentry recovery**

In other words:

| Mechanism | Intended zone |
|---|---|
| baseline matching | easy current-frame assignment |
| RG-OT | ambiguous local recovery / reassignment |
| reentry memory / engine | removed-track long-gap return |

---

## Runtime Signals Already Available

The existing code already exposes enough signals for an `analysis-only` RG-OT pass.

### 1. Track gap / freshness

Available in:

- owneralt
- graph-assoc
- reentry
- TOS analysis rows

This is useful but must not be the dominant signal, otherwise RG-OT collapses into `track_buffer` logic.

### 2. Tracklet length / owner maturity

Already used by:

- `owner_max_tracklet_len`
- `min_tracklet_len`
- young-active protections

This is important for distinguishing:

- weak recent owner
- mature stable owner
- reclaim-worthy stale alternative

### 3. Raw IoU and alternative edge structure

Already used by:

- `OwnerAltCompetitionRefiner`
- `LocalGraphReassocRefiner`

Signals include:

- challenger box IoU
- owner alt box IoU
- owner edge deficit
- cost delta
- assignment gain

These are central RG-OT candidates because they describe **local transfer geometry**, not just memory duration.

### 4. Appearance similarity availability

Available through:

- `laplace_debug`
- `_safe_cosine_similarity`
- HACA / Laplace features

But caution:

- HACA v1 on MOT20 can lack competition-head signals.
- missing `smooth_feat` can zero out appearance-derived terms.

RG-OT should explicitly log signal availability, not silently assume appearance is reliable.

### 5. Neighborhood ambiguity

Already represented by:

- graph-assoc block rows / cols
- ambiguous rows / cols
- enumerated assignments
- candidate rows

This is the strongest direct input for RG-OT, because the method should be justified by **hard local ambiguity**, not by generic low confidence.

### 6. Existing outcome metrics

Already available or derivable:

- changed block rate
- candidate accept rate
- event rows
- competitive reactivated tracks
- paired flip analyses
- gap bucket evaluation scripts

This means RG-OT can start with rigorous diagnostics before any behavioral intervention.

---

## Signals RG-OT Should Prefer

For a first `analysis-only` pass, RG-OT should prioritize these features:

1. `gap`
2. `tracklet_len`
3. `owner_age` or recent-owner status
4. `challenger vs owner edge deficit`
5. `assignment utility gain`
6. `candidate rank / margin to second`
7. `alternative edge availability for displaced owner`
8. `appearance availability flag`
9. `block size` and `enumerated_assignments`
10. `state` of candidate rows: `Tracked`, `Lost`, `LongLost`

This feature set is much better aligned with "resolved online recovery" than the TOS-style occlusion score.

---

## Proposed RG-OT Analysis-Only Output

Before any behavior change, RG-OT should produce:

### 1. `rgot_events.jsonl`

One row per triggered local neighborhood:

- frame id
- seq name
- rows / cols in block
- owner row
- challenger row(s)
- owner state
- challenger state
- owner gap
- challenger gap
- owner tracklet len
- challenger tracklet len
- owner edge score / cost
- challenger edge score / cost
- owner edge deficit
- candidate alternative for owner exists or not
- utility gain of rewrite candidate
- whether case is also explainable by reentry-only logic
- whether case is also explainable by simple larger buffer logic

### 2. `rgot_summary.csv`

One-row summary per run:

- triggered blocks
- owner-weak cases
- reclaim-candidate cases
- local ambiguous blocks
- cases explainable by plain reentry
- cases not explainable by plain reentry
- cases with positive utility gain
- mean / median owner edge deficit
- mean / median utility gain

### 3. Optional `rgot_candidate_rows.jsonl`

Only if needed for calibration:

- full local candidate ranking
- selected rewrite candidate
- rejected reasons

This mirrors the style already used by graph-assoc and owneralt.

---

## Pass Criteria for RG-OT Analysis-Only

RG-OT should only proceed to behavioral mode if all of the following hold:

1. **Event count is non-trivial**  
   Not another `28 / 612803` situation.

2. **Events concentrate in true ambiguous local neighborhoods**  
   Not uniformly spread across easy frames.

3. **A substantial fraction is not reducible to plain reentry**

4. **A substantial fraction is not reducible to plain larger track_buffer**

5. **Rewrite utility signal exists before intervention**

If these fail, RG-OT should be stopped early, same as TOS.

---

## Recommended First Implementation Shape

### Phase 0.6: RG-OT analysis-only hook

Do **not** change matching behavior yet.

Implement:

- `--rgot-enable`
- `--rgot-analysis-only`
- `--rgot-analysis-dir`

Attach after local cost matrix construction and before final assignment, ideally alongside:

- `owneralt_refiner`
- `graph_assoc_refiner`

but without modifying `dists`.

The hook should:

1. detect ambiguous local blocks
2. score RG-OT candidate neighborhoods
3. dump event rows
4. export summary

### Phase 0.7: overlap audit

For each RG-OT event, annotate:

- would reentry memory also fire?
- would competitive archive recovery also explain this?
- would owneralt already handle it?
- would graph-assoc already rewrite it?

If most cases are already fully covered, RG-OT has no independent problem definition.

---

## Risks and Hard Constraints

### Risk 1: RG-OT collapses into graph-assoc rebranding

This is the biggest danger.

Mitigation:

- define RG-OT as a problem diagnosis layer first
- log explicit overlap with graph-assoc and owneralt
- do not claim novelty from a renamed rewrite block

### Risk 2: Missing runtime implementation of some imported modules

`bot_sort.py` imports `ReentryQueryEngine` through `models.reentry_query_engine`, which exists as a package in the repo. This is usable, but the audit should treat the query-engine path as optional support, not as the only RG-OT carrier.

### Risk 3: MOT20 carrier mismatch

Some reentry and graph-assoc evidence in the registry comes from DanceTrack or earlier standalone runs. RG-OT must be re-validated on the current MOT20 no-bg carrier before any paper claim.

### Risk 4: another negative branch with no event density

TOS failed because the trigger population was too sparse. RG-OT should therefore start from **existing ambiguous blocks** instead of inventing a new scalar trigger over all matched pairs.

---

## Final Recommendation

**Proceed with RG-OT, but only as an `analysis-only` local-neighborhood recovery audit built on top of existing graph-assoc / owneralt / reentry instrumentation.**

Do not start with:

- feature freeze
- longer buffer
- archive-only recovery
- learned policy training

Start with:

1. ambiguous local block discovery
2. event export
3. overlap accounting vs owneralt / graph-assoc / reentry
4. pass/fail gate

If RG-OT clears that gate, then and only then move to behavioral rewrite experiments.

---

## Immediate Next Step

Create an `RG-OT Phase 0.6 analysis-only` task with:

- CLI flags
- event schema
- summary schema
- overlap labels with existing mechanisms
- MOT20-05 smoke first, then broader validation only if event density is real

## Phase 0.6 Execution Checklist

The repo now has a minimal `analysis-only` RG-OT skeleton target. The execution order should be fixed as below.

### 1. File-level implementation scope

Only these files should change in Phase 0.6:

- `external/BoT-SORT-main/tracker/rgot_analysis.py`
- `external/BoT-SORT-main/tracker/bot_sort.py`
- `external/BoT-SORT-main/tools/track.py`
- one short run script under `scripts/` if needed

Do not modify matching behavior yet. No cost rewrite, no reassignment, no learned gate.

### 2. Required CLI surface

Phase 0.6 should expose:

- `--rgot-enable`
- `--rgot-analysis-only`
- `--rgot-analysis-dir`
- `--rgot-top-k`
- `--rgot-row-margin`
- `--rgot-col-margin`
- `--rgot-max-rows`
- `--rgot-max-cols`

Optional proxy-threshold flags are acceptable if they are only used for diagnostics.

### 3. Required runtime insertion point

RG-OT analysis should run:

1. after the primary `dists` matrix is available,
2. after `owneralt_refiner` and `graph_assoc_refiner` have populated their current-frame debug rows,
3. before Hungarian assignment finalization.

This keeps RG-OT aligned with the intended problem:

- local ambiguous recovery,
- not removed-track archive reentry,
- not post-hoc result auditing only.

### 4. Required output files

Each sequence run must produce:

- `rgot_analysis/<seq>_summary.csv`
- `rgot_analysis/<seq>_events.jsonl`

The summary must be checkpoint-safe during long runs, following the same partial-write pattern already used by:

- `owneralt_analysis`
- `graph_assoc_analysis`

### 5. Required summary fields

At minimum:

- `enabled`
- `analysis_only`
- `frames`
- `candidate_blocks`
- `trigger_blocks`
- `event_count`
- `owner_weak_cases`
- `challenger_reclaim_cases`
- `alt_available_cases`
- `buffer_like_cases`
- `reentry_like_cases`
- `owneralt_overlap_events`
- `graph_assoc_overlap_events`
- `not_explained_by_buffer_or_reentry_cases`
- `mean_owner_edge_deficit`
- `median_owner_edge_deficit`
- `mean_joint_cost_delta_proxy`
- `median_joint_cost_delta_proxy`

### 6. Required event schema

Each event row should capture one triggered local neighborhood and include:

- block geometry: `rows`, `cols`, `ambiguous_rows`, `ambiguous_cols`
- focus target: `focus_det_col`, `focus_det_score`, `focus_margin`
- owner descriptor: `owner_row`, `owner_track_id`, `owner_state`, `owner_gap`, `owner_tracklet_len`, `owner_cost`
- challenger descriptor: `challenger_row`, `challenger_track_id`, `challenger_state`, `challenger_gap`, `challenger_tracklet_len`, `challenger_cost`
- geometric support: `owner_box_iou`, `challenger_box_iou`
- ownership contrast: `owner_edge_deficit`
- displaced-owner fallback: `alt_exists`, `alt_col`, `alt_cost`, `alt_box_iou`
- overlap/proxy tags:
  - `buffer_like_proxy`
  - `reentry_like_proxy`
  - `owneralt_like_proxy`
  - `owneralt_overlap_event`
  - `graph_assoc_overlap_event`
  - `not_explained_by_buffer_or_reentry`
- appearance/debug support if available:
  - `appearance_available`
  - `appearance_cosine`
  - `haca_margin`
  - `haca_bg_prob`

### 7. First smoke run

Run first on:

- MOT20-05
- current no-bg HACA carrier
- `analysis-only` only

Do not start with full MOT20 queue.

Expected artifact root:

- `outputs/rgot_analysis_smoke_<ts>/`

Required records:

- queue or run-level `summary.csv`
- `rgot_analysis/MOT20-05_summary.csv`
- `rgot_analysis/MOT20-05_events.jsonl`
- append row in `outputs/experiment_registry.csv`

### 8. Pass / stop gate

Proceed to Phase 0.7 only if all are true:

1. `event_count` is materially larger than the TOS failure regime.
2. A useful portion of events are `not_explained_by_buffer_or_reentry`.
3. Overlap with `owneralt` and `graph_assoc` is informative but not total.
4. The events concentrate on real local ambiguity rather than trivial easy rows.

Stop early if:

- event density is near-zero,
- almost all events are plain buffer/reentry proxies,
- or RG-OT is fully subsumed by current `owneralt` / `graph_assoc` rules.
