# Implementation Redesign Plan: Local Graph Phase 1 (2026-03-23)

## Decision

Phase 1 only does one thing:

- replace the current weak `local_conflict_graph_oracle` semantics
- with `full cluster replacement oracle`

This phase does **not** introduce the learned cluster scorer yet.

## Scope

Keep:

- frozen host: `base_reid_da`
- primary association only
- pre-Hungarian injection
- top-k local conflict clusters

Change:

- decision unit from partial row pre-assign to cluster-level complete local assignment

Do not do in Phase 1:

- learned cluster model
- continuity / stitching
- cluster construction sweeps
- full7 expansion
- old row-local controller tuning

## Real Repo Paths

These are the actual files to change in this repository:

1. `models/local_conflict_graph_common.py`
2. `models/runtime_tracker_bytetrack.py`
3. `submit_bytetrack.py`
4. `train_bytetrack.py`
5. `scripts/run_local_conflict_graph_fullcluster_oracle_proxy0213.sh`

## Change 1

- File: `models/local_conflict_graph_common.py`
- Reason: runtime, analysis, and later dataset building must share one cluster construction helper
- New responsibility:
  - build top-k bipartite connected components
  - solve local assignment with private null columns
- Phase 1 status: implemented

## Change 2

- File: `models/runtime_tracker_bytetrack.py`
- Current anchor:
  - `__init__` local graph flags
  - `_load_local_conflict_graph_oracle`
  - `_match_with_scores`
- Problem in old path:
  - old local graph oracle only pre-assigned a subset of positive pairs
  - host Hungarian still decided the cluster afterwards
  - cluster was not the real decision unit
- Phase 1 replacement:
  - add `ASSOC_USE_LOCAL_CONFLICT_GRAPH`
  - add `ASSOC_LOCAL_CONFLICT_GRAPH_MODE=oracle_full`
  - build local top-k conflict clusters from current score matrix
  - solve each eligible cluster completely
  - commit cluster decisions directly
  - remove those cluster rows / cols from host stage-1 Hungarian
- Phase 1 status: implemented

## Change 3

- File: `submit_bytetrack.py`
- Reason: inference config must pass the new local graph mode into the runtime tracker
- Phase 1 additions:
  - `ASSOC_USE_LOCAL_CONFLICT_GRAPH`
  - `ASSOC_LOCAL_CONFLICT_GRAPH_MODE`
- Phase 1 status: implemented

## Change 4

- File: `train_bytetrack.py`
- Reason: tracker construction inside train/validation must stay config-compatible with submit
- Phase 1 additions:
  - `ASSOC_USE_LOCAL_CONFLICT_GRAPH`
  - `ASSOC_LOCAL_CONFLICT_GRAPH_MODE`
- Phase 1 status: implemented

## Change 5

- File: `scripts/run_local_conflict_graph_fullcluster_oracle_proxy0213.sh`
- Reason: Phase 1 needs a clean runner and a clean registry line, separate from legacy partial oracle
- Phase 1 behavior:
  - sets `ASSOC_USE_LOCAL_CONFLICT_GRAPH=True`
  - sets `ASSOC_LOCAL_CONFLICT_GRAPH_MODE=oracle_full`
  - disables legacy partial oracle path
  - writes `result.csv`, `summary.csv`, `experiment_registry.csv`, and Pro bundle
- Phase 1 status: implemented

## Expected Verification

Phase 1 success criterion:

- `fullcluster oracle` should outperform the current weak graph oracle
- especially in `HOTA` / `AssA`
- and should clarify whether the stronger cluster-level online semantics are worth learning

## Next Step After Phase 1

Only if Phase 1 is positive:

- build `LocalConflictAssignmentRefiner`
- train a minimal learned cluster scorer
- deploy it with the same cluster-level direct-commit semantics
