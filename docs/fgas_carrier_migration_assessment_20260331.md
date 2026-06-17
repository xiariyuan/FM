# FGAS Carrier Migration Assessment (2026-03-31)

## Decision

Primary migration target: `Deep-OC-SORT`

Fallback target: `BoostTrack`

Rejected for current mainline: `TrackTrack`

## Why `Deep-OC-SORT` is the best fit

`FGAS-v4-nofreq` is no longer just an edge rescoring hook. It now emits explicit
controller actions before assignment:

- `forced_matches`
- `blocked_rows`
- `blocked_cols`

This contract needs a carrier with:

- a clear primary association stage
- an explicit cost matrix before Hungarian/LAP
- strong motion + appearance interaction
- a research narrative centered on association quality rather than whole-system glue

`Deep-OC-SORT` matches that best.

Relevant local inspection:

- primary tracker:
  [external/carrier_candidates/deep_ocsort/Deep-OC-SORT-main/trackers/integrated_ocsort_embedding/ocsort.py](/gemini/code/FMtrack-main/FM-Track/external/carrier_candidates/deep_ocsort/Deep-OC-SORT-main/trackers/integrated_ocsort_embedding/ocsort.py)
- primary association:
  [external/carrier_candidates/deep_ocsort/Deep-OC-SORT-main/trackers/integrated_ocsort_embedding/association.py](/gemini/code/FMtrack-main/FM-Track/external/carrier_candidates/deep_ocsort/Deep-OC-SORT-main/trackers/integrated_ocsort_embedding/association.py)

The key structure is:

- `ocsort.py:update(...)` builds the tracked pool and detection pool
- `association.py:associate(...)` builds the primary match score / cost
- LAP/Hungarian is applied once on that primary matrix
- a second OCR-style cleanup pass happens only after the first round

That is exactly the place where FGAS controller actions belong: inside the first
association round, before LAP.

## Why `BoostTrack` is second, not first

Relevant local inspection:

- tracker:
  [external/carrier_candidates/boosttrack/BoostTrack-master/tracker/boost_track.py](/gemini/code/FMtrack-main/FM-Track/external/carrier_candidates/boosttrack/BoostTrack-master/tracker/boost_track.py)
- association:
  [external/carrier_candidates/boosttrack/BoostTrack-master/tracker/assoc.py](/gemini/code/FMtrack-main/FM-Track/external/carrier_candidates/boosttrack/BoostTrack-master/tracker/assoc.py)

`BoostTrack` is clean and easy to modify. It has a single explicit association
matrix and would be straightforward to hook.

It is not the first choice because:

- the paper story is more about boosted similarity/confidence engineering
- the carrier is simpler but less naturally aligned with our current
  "association controller over ambiguous local blocks" framing
- `Deep-OC-SORT` provides a stronger appearance-aware association baseline for
  FGAS to improve on

If `Deep-OC-SORT` integration cost or environment friction becomes excessive,
`BoostTrack` is the correct fallback.

## Why `TrackTrack` is rejected

Relevant local inspection:

- tracker:
  [external/carrier_candidates/tracktrack/TrackTrack-main/3. Tracker/trackers/tracker.py](/gemini/code/FMtrack-main/FM-Track/external/carrier_candidates/tracktrack/TrackTrack-main/3.%20Tracker/trackers/tracker.py)
- assignment logic:
  [external/carrier_candidates/tracktrack/TrackTrack-main/3. Tracker/trackers/utils.py](/gemini/code/FMtrack-main/FM-Track/external/carrier_candidates/tracktrack/TrackTrack-main/3.%20Tracker/trackers/utils.py)

`TrackTrack` does not expose the kind of standard primary LAP/Hungarian stage we
want. Its main association is an `iterative_assignment(...)` greedy loop with a
custom repeated mutual-minimum procedure.

That makes our controller contract awkward:

- `forced_matches` is less natural
- `blocked_rows` / `blocked_cols` are less clean
- the migration would turn into a carrier-specific rewrite instead of a
  portable FGAS migration

So `TrackTrack` is not the right next host.

## Migration plan for `Deep-OC-SORT`

Initial files to touch:

- `external/Deep-OC-SORT-main/trackers/integrated_ocsort_embedding/ocsort.py`
- `external/Deep-OC-SORT-main/trackers/integrated_ocsort_embedding/association.py`
- a new small FGAS bridge module or direct import path into `projects/fgas`
- a runner entry that records smoke/full results under `outputs/`

Integration strategy:

1. Keep `FGAS-v4-nofreq` only.
2. Do not port the frequency branch.
3. Inject FGAS only into the first association round.
4. Preserve the second OCR rematch round unchanged at first.
5. First target is one smoke sequence: `MOT17-05-FRCNN`.
6. Only if smoke stays positive, expand to full7 half-val.

## Stop / continue rule

Continue only if `Deep-OC-SORT + FGAS-v4-nofreq` can produce a positive smoke
without broad controller overreach.

If migration is technically smooth but online signal is negative, the next step
is not more frequency work. The next step is either:

- ambiguity-gate calibration on the new carrier, once
- or fallback migration to `BoostTrack`
