# M23-53B Adaptive Microtracklet Identity-Flow Capacity Audit

Date: 2026-07-20

## Status

- Experiment family: M23-53B
- Role: teacher-only capacity audit after M23-53 fixed chunk-30 failed the 80.300000 floor
- Deployable: false
- Strict parent: frozen M23-46 trackers
- Final decision: M23-53B is insufficient and closed; M23-54 must not start

## Motivation

M23-53 fixed chunk-30 global identity flow reached COMBINED HOTA 79.950590. This was only 0.049410 below 80 but 0.349410 below the preregistered 80.300000 minimum capacity margin. M23-53B tested the only authorized repair: add GT-free adaptive boundaries inside fixed chunk-30 nodes and rerun the same global identity-flow teacher without changing teacher weights or using per-sequence GT rules.

## Uniform GT-free segmentation protocol

The same protocol and constants were applied to all four sequences:

- preserve every existing chunk-30 and temporal-gap boundary;
- use only frozen tracker rows and phase-0 ReID assets;
- project 2048D phase-0 embeddings to 128D with fixed seed 2353;
- calculate candidate change points from:
  - consecutive ReID discontinuity;
  - forward motion residual;
  - local crowd-density change;
  - box-scale change;
  - confidence change;
  - ReID recovery after an unavailable embedding;
- normalize signals with sequence-level median/MAD using no GT;
- require at least five rows on each side of a cut;
- use the fixed 75th-percentile composite change-score threshold;
- add at most one adaptive cut per fixed chunk;
- construct a GT-free candidate bank with appearance top-32 plus motion top-8 and maximum gap 600;
- map each adaptive node to the frozen M23-46 tracker ID using exact frame/box alignment;
- build the adaptive M23-46 parent graph and require byte-exact reconstruction before teacher GT access.

The M23-53 identity-flow engine was generalized to reconstruct arbitrary frozen segmentations from node row counts and source ordinals. For adaptive graphs, baseline reconstruction preserves the explicit `parent_tracker_id`; the final teacher tracker still uses the global path-cover assignment. The original fixed chunk-30 M02 reconstruction remained byte-exact after this change.

## Frozen adaptive graph sizes

| Fold | Fixed chunks | Adaptive chunks | Added boundaries | Candidate bank edges | Parent edges | M23-46 byte-exact |
|---|---:|---:|---:|---:|---:|---:|
| M01 | 698 | 1,321 | 623 | 44,833 | 1,229 | yes |
| M02 | 5,290 | 10,086 | 4,796 | 370,296 | 9,648 | yes |
| M03 | 10,602 | 20,536 | 9,934 | 744,092 | 19,782 | yes |
| M05 | 22,228 | 43,022 | 20,794 | 1,581,079 | 41,441 | yes |

No GT was used by the graph builder. Each graph includes a freeze manifest and SHA-256 hashes for nodes, prototypes, candidate edges, boundary scores, parent edges and the copied M23-46 tracker.

## M02 smoke

Experiment: `m23_53b_identity_flow_capacity_m02_smoke_v1`

- adaptive nodes: 10,086;
- smoke frozen edges: 13,744 = 9,648 parent + 4,096 cross;
- forbidden frozen candidate columns: none;
- baseline reconstruction: byte-exact;
- selected cross edges: 12;
- one-to-one, acyclic and time-forward: yes;
- official M02 HOTA: 72.857136.

The low smoke score was attributed to the deliberately restricted cross-edge set and was not used to alter the uniform protocol.

## Full teacher-only capacity

| Fold | HOTA | DetA | AssA | IDSW | Delta vs fixed M23-53 |
|---|---:|---:|---:|---:|---:|
| M01 | 80.061990 | 82.239550 | 78.101580 | 63 | -0.781800 |
| M02 | 74.572086 | 81.765190 | 68.102280 | 484 | -0.448374 |
| M03 | 81.308920 | 81.442480 | 81.208030 | 211 | -0.033706 |
| M05 | 80.442330 | 82.112646 | 78.846020 | 591 | +0.088796 |

All folds passed:

- adaptive M23-46 byte-exact reconstruction;
- graph freeze before GT access;
- no forbidden GT-derived columns in frozen candidate nodes or edges;
- one-to-one, acyclic and time-forward path cover;
- complete tracker writing and official TrackEval.

## Single official combined result

Experiment: `m23_53b_identity_flow_capacity_combined_v1`

Exactly one official COMBINED TrackEval was run after the four adaptive teacher trackers were frozen and hashed.

- COMBINED HOTA: **79.920490**
- DetA: **81.876564**
- AssA: **78.059750**
- IDSW: **1349**
- Delta versus fixed M23-53 79.950590: **-0.030100**
- Gain over strict M23-46 79.123193: **+0.797297**
- Gap to 80: **0.079510**
- Gap to M23-54 capacity floor 80.700000: **0.779510**

## Interpretation and stopping decision

The uniform adaptive segmentation over-fragmented M01 and M02, was nearly neutral on M03, and improved only M05 slightly. The extra boundaries exposed more teacher-positive candidate edges, but the final global path cover did not convert them into higher HOTA across domains. This shows that simply adding high-signal GT-free cuts does not solve the remaining candidate/state mismatch.

The preregistered decision is therefore:

1. close this M23-53B protocol;
2. do not tune the teacher GT weights, cut quantile, per-sequence thresholds, K or gap using these outer results;
3. do not start M23-54 because capacity is below 80.700000;
4. retain M23-53 fixed chunk-30 HOTA 79.950590 as the best global-flow teacher capacity, still teacher-only and nondeployable;
5. retain M23-46 HOTA 79.123193 as the strict deployable best.

## Output roots

- `outputs/mot20_m23_20260718/m23_53b_adaptive_micrograph_v1`
- `outputs/mot20_m23_20260718/m23_53b_m23_46_adaptive_baseline_cache_v1`
- `outputs/mot20_m23_20260718/m23_53b_identity_flow_capacity_m02_smoke_v1`
- `outputs/mot20_m23_20260718/m23_53b_identity_flow_capacity_m01_v1`
- `outputs/mot20_m23_20260718/m23_53b_identity_flow_capacity_m02_v1`
- `outputs/mot20_m23_20260718/m23_53b_identity_flow_capacity_m03_v1`
- `outputs/mot20_m23_20260718/m23_53b_identity_flow_capacity_m05_v1`
- `outputs/mot20_m23_20260718/m23_53b_identity_flow_capacity_combined_v1`
