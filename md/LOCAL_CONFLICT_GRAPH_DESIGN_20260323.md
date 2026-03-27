# Local Conflict Graph Design (2026-03-23)

## Decision

The previous `single-row / row-local rerank` mainline is stopped.

The new mainline is:

`competition-aware local conflict graph`

This mainline upgrades the decision unit from:

- one detection row with its top-k candidates

to:

- a frame-local conflict cluster containing multiple detections and their competing candidate tracks

## Problem Definition

At primary association time, before the host's global Hungarian step:

- build a local bipartite graph from the frozen host's gated candidate matrix
- left nodes are current-frame detections
- right nodes are active tracks
- edges are host-gated candidate links

Each connected component of this bipartite graph is treated as a local conflict cluster.

The cluster-level decision is:

- a one-to-one partial assignment inside the cluster
- detections may remain unmatched
- tracks may remain unmatched
- cluster-external rows and columns remain untouched

The first version remains:

- primary-only
- pre-Hungarian
- local-only
- frozen-host outside the cluster

## Why This Mainline

The previous row-local line is killed because:

- `rerank_only` lost to `noop`
- `rerank_minimal` lost to `noop`
- `oracle_rerank(top-8 + minimal winner override)` also lost to `noop`

This is not just a bad learned model.
It is evidence that the row-local decision unit itself is too small.

The current residual errors are better explained by:

- cross-row coupling
- one-to-one joint assignment inside local conflict regions

not by:

- local winner correction within a single row
- continuity / long-gap stitching as the first-order bottleneck

## First Diagnostics

The first experiment batch should not train a model.

It should produce:

1. cluster anatomy
   - cluster size distribution
   - detections per cluster
   - tracks per cluster
   - recoverable / bridge coverage at cluster level

2. oracle local conflict graph upper bound
   - compare against `noop`
   - compare against old `oracle_rerank`
   - prove whether the larger decision unit has real online value

## Cluster Construction v1

For each frame:

- create one detection node per detection row
- create one track node per candidate track
- add an edge if the track is in the detection's host-gated top-k set

Then:

- take connected components as candidate clusters
- focus on components with at least one conflict
  - multiple detections, or
  - multiple tracks, or
  - a node with degree > 1

The first cluster definition should stay simple.
Do not make IoU the primary cluster criterion in v1.
Candidate overlap and graph connectivity are the primary signals.

## Oracle Upper Bound v1

The first oracle should not output per-edge labels only.

It should output:

- cluster-level one-to-one assignment oracle

Inside each cluster:

- use replay / GT feasibility to determine positive edges
- solve a cluster-internal one-to-one assignment
- allow unmatched detections
- prefer minimal deviation from host as a tie-break

This tests the value of the new decision unit directly.

## Learned Model v1

Only after the cluster-level oracle clearly beats `noop`:

- implement a small learned local graph module
- observed-only features
- primary-only
- cluster-local only

The first learned model can be:

- a small bipartite edge scorer
- a message-passing graph model
- a lightweight transformer over cluster nodes/edges

The goal is not model sophistication.
The goal is validating the new decision unit.

## Immediate Files / Modules

Suggested new files:

- `models/conflict_graph_assoc.py`
- `scripts/analyze_local_conflict_graph_clusters.py`
- `scripts/run_local_conflict_graph_cluster_anatomy.sh`
- later: `scripts/run_local_conflict_graph_oracle_proxy0213.sh`

## Current Evidence To Preserve

Keep these outputs as negative evidence / background:

- `outputs/competition_assoc_stage1_fix1_full12`
- `outputs/competition_assoc_base_reid_da_proxy0213_hybriddumpfix`
- `outputs/competition_assoc_online_noop_proxy0213_20260323_094948`
- `outputs/competition_assoc_online_rerank_only_proxy0213_20260323_113046`
- `outputs/competition_assoc_online_rerank_minimal_proxy0213_20260323_114447`
- `outputs/competition_assoc_online_oracle_rerank_proxy0213_20260323_141625`

These are no longer the active mainline, but they define why the decision unit changed.
