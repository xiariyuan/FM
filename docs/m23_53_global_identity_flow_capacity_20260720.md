# M23-53 Global Identity Flow Capacity Audit

Date: 2026-07-20

## Status

- Experiment family: M23-53
- Role: teacher-only action-space capacity audit
- Deployable: false
- Strict parent: frozen M23-46 sequence-LOSO trackers and applied graphs
- Fixed segmentation: chunk-30 microtracklets
- Final decision: fixed chunk-30 capacity is insufficient for M23-54; continue only with M23-53B GT-free adaptive microtracklet segmentation

## Motivation

M23-51 and M23-52 showed that independent residual actions have unstable utility across sequence domains and parent graph states. M23-53 therefore represents cuts, retained source edges, cross-track merges/relinks, swaps, termination and restart as one global, one-to-one, time-forward identity-flow/path-cover problem.

## Frozen candidate protocol

The same protocol was used for all four sequences:

- Parent tracker and applied graph: strict frozen M23-46.
- Nodes: fixed chunk-30 microtracklets; detection boxes, confidence scores and detection rows are unchanged.
- Cross-edge source: `micrograph_chunk30_v1/<seq>/candidate_edges.parquet` read through an explicit GT-free column allowlist.
- Current M23-46 applied edges are always retained as parent candidates.
- Cross edges must be time-forward and satisfy the preregistered reachability limits:
  - maximum gap: 600 frames;
  - maximum motion error: 5.0;
  - maximum endpoint displacement: 6.0;
  - maximum absolute log-height ratio: 1.2.
- Candidate retention is the union of top-32 outgoing/incoming appearance rank and top-32 outgoing/incoming motion rank, plus every parent edge.
- Dummy terminate/restart is an explicit zero-weight no-link option in the bipartite path-cover solve.
- Candidate nodes and edges are written before GT access, with SHA-256 hashes in `freeze_manifest.json`.
- Frozen files are audited to exclude `same_gt`, `modal_gt`, `purity`, `label_confidence`, `actual_assa`, association-delta labels, matched-GT counts and related GT-derived fields.

Each fold reconstructed the M23-46 baseline byte-exactly before GT was opened. The protocol event log records candidate freeze before teacher GT loading.

## Teacher objective

After the candidate graph is frozen, GT is used only to construct teacher structural edge utilities:

- forward edges connecting chunks with the same dominant matched GT identity receive positive utility;
- utility emphasizes temporal continuity, recoverable association contribution, endpoint lengths, identity coverage and chunk purity;
- different-identity or unmatched edges receive negative utility and lose to the zero-weight no-link option;
- one maximum-weight bipartite path-cover solve enforces one successor, one predecessor, time direction and acyclicity.

This is not an independent-action `delta_HOTA` label and is not a deployable policy.

## Smoke audit

`m23_53_identity_flow_capacity_m02_smoke_v1`

- Active smoke region: first 512 chunks.
- Cross-edge cap: 2,048.
- Frozen candidates: 6,900 edges = 4,852 parent + 2,048 cross.
- Forbidden frozen columns: none.
- Baseline reconstruction: byte-exact.
- Selected path-cover edges: 4,623, including 10 cross edges.
- One-to-one, acyclic and time-forward: yes.
- Official M02 HOTA: 73.896456.
- Gain over strict M23-46 M02: +0.798306.

The smoke validated the complete freeze-before-GT and official-TrackEval execution path.

## Full per-sequence capacity

| Fold | Frozen edges | Parent edges | Cross candidates | Selected cross | HOTA | DetA | AssA | IDSW |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M01 | 8,214 | 606 | 7,608 | 20 | 80.843790 | 82.058820 | 79.788980 | 56 |
| M02 | 62,016 | 4,852 | 57,164 | 104 | 75.020460 | 81.605670 | 69.072660 | 462 |
| M03 | 105,741 | 9,848 | 95,893 | 58 | 81.342626 | 81.399490 | 81.317570 | 199 |
| M05 | 199,983 | 20,647 | 179,336 | 158 | 80.353534 | 82.078904 | 78.705120 | 620 |

All folds passed:

- byte-exact M23-46 reconstruction;
- candidate freeze before GT access;
- empty forbidden-column audit;
- one-to-one path cover;
- acyclicity;
- time-forward selected edges;
- complete tracker writing and official TrackEval.

## Single official combined result

Experiment: `m23_53_identity_flow_capacity_combined_v1`

Exactly one official COMBINED TrackEval was run after copying and hashing the four frozen teacher trackers.

- COMBINED HOTA: **79.950590**
- DetA: **81.820434**
- AssA: **78.174204**
- IDSW: **1337**
- Gain over strict M23-46 COMBINED 79.123193: **+0.827397**
- Gain over M23-51 residual interaction oracle 79.719687: **+0.230903**
- Gap to 80: **0.049410**
- Gap to preregistered 80.300000 capacity floor: **0.349410**

## Preregistered interpretation

The result is below 80.300000. Therefore:

1. M23-53 fixed chunk-30 identity flow is useful mechanism evidence but lacks the deployment margin required for M23-54.
2. No HOTA-aligned identity-flow student is trained from this graph.
3. Teacher GT weights, K, gap limits and per-sequence rules must not be tuned to recover the missing margin.
4. The only allowed next step is M23-53B: add GT-free adaptive boundaries at appearance discontinuities, motion-residual changes, occlusion recovery and crowding change-points, preserve all chunk-30 boundaries, freeze the new graph before GT, and rerun the same global identity-flow capacity audit.
5. M23-54 is eligible only if the resulting teacher COMBINED capacity reaches at least 80.700000.

## Output roots

- `outputs/mot20_m23_20260718/m23_53_identity_flow_capacity_m02_smoke_v1`
- `outputs/mot20_m23_20260718/m23_53_identity_flow_capacity_m01_v1`
- `outputs/mot20_m23_20260718/m23_53_identity_flow_capacity_m02_v1`
- `outputs/mot20_m23_20260718/m23_53_identity_flow_capacity_m03_v1`
- `outputs/mot20_m23_20260718/m23_53_identity_flow_capacity_m05_v1`
- `outputs/mot20_m23_20260718/m23_53_identity_flow_capacity_combined_v1`
