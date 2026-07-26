# M28-A2 — Multi-Sequence Deferred Identity Inheritance Capacity

Pre-registered: 2026-07-26

## Scope

MOT20 train sequences `MOT20-01`, `MOT20-02`, `MOT20-03`, and `MOT20-05`. Each sequence uses an independently replayed deterministic `DMMBaseTracker` host with default `botsort_reid` and `dmm_v3_enable=false`.

## Freeze-before-GT

For every sequence:

1. independently reproduce the online host tracker;
2. freeze all runtime update events and the baseline tracker;
3. freeze all `update:unconfirmed` events;
4. freeze top-8 old-identity candidates using exactly the M28-A0 GT-free appearance/motion/scale score;
5. write input, runtime, event, candidate and implementation hashes.

Only after these artifacts are frozen may sequence GT be opened.

## Identity-consistent teacher

After freeze, official TrackEval preprocessing is used to obtain the dominant GT identity of each baseline tracker ID. Exact HOTA is evaluated only for candidate actions whose young and old tracker IDs share the same dominant GT identity. M01 diagnostics showed this screen retains 98.4% of M28-A0 capacity while eliminating most unnecessary exact evaluations.

Positive identity-consistent actions are greedily considered in descending exact single-action HOTA delta. Each young ID and inherited old ID may be used at most once. An action is accepted only when its exact incremental HOTA gain remains positive. One official TrackEval run verifies each final teacher tracker.

## GO gate

Authorize strict sequence-LOSO student construction only if:

- every sequence has positive final exact HOTA gain;
- every sequence has at least five positive identity-consistent actions;
- the four-sequence official combined teacher improves HOTA by at least `+0.80`;
- combined IDSW does not increase;
- candidate generation and runtime replay remain GT-free and integrity-clean.

Otherwise fail-close or redesign the candidate retrieval before training.

## Status

Teacher-only, `deployable=false`. MOT20 test reads/submissions are prohibited.
