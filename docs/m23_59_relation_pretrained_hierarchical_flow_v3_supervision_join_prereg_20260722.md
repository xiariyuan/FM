# M23-63 preregistration — v3 supervision join and example audit

Frozen before any MOT17 GT read. Experiment root: `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_supervision_join`.

## Scope
Stage-A audit only. Training, optimizer steps, checkpoints, tracker generation, TrackEval, MOT20 GT/teacher/held-outer/test access, M23-54 and M23-58 are prohibited. R62 is immutable. M23-59 v2 checkpoint reuse is prohibited.

## Physical split and namespace
Train: MOT17-02/04/05/09/10-FRCNN. Validation: MOT17-11/13-FRCNN. DPM/SDP variants are excluded. Every row, track, GT identity, topology entity, candidate and example uses a sequence-qualified stable key.

## Frozen M23-62 contract
Required hash: `90cfab7d3a1fc87cc46b26441d3d883b9fc72f27d8852d1e2074b00e628428f5`. All five root authorization/closure files and all seven MOT17 rows/features/manifests are rehashed before label unlock. Any mismatch closes as FAIL_M23_62_REVERIFICATION.

## Supervision join
GT may be read only after topology/candidate freeze and a label_unlock event. Allowed path pattern: `datasets/MOT17/train/<sequence>-FRCNN/gt/gt.txt`.
TrackEval MOT box semantics are xywh converted to xyxy. IoU threshold is exactly `0.50`; no search. Valid pedestrian GT is mark != 0 and class == 1; visibility is not an eligibility threshold. Input feature 134 remains the unavailable sentinel and is never replaced by GT visibility.
MOT17 distractor classes are frozen from TrackEval as `[2, 7, 8, 12]` (person_on_vehicle, static_person, distractor, reflection). Per frame, all-GT Hungarian matching first removes source rows matched to these distractors; then remaining rows are matched one-to-one to valid pedestrian GT.
Source order is (line_index,row_index), GT order is original GT line order. IoU is quantized to `1e-09` and a `1e-12` lexicographic bonus resolves equivalent optima. Unknown source rows remain ignore and are never negative. No majority vote, temporal fill, GT-box feature recomputation, source-track rewrite, or GT-driven candidate insertion is permitted.

## Label-blind topology
Node opportunities follow each source track in (frame,line_index,row_index) order with MAX_NODE_ROWS=30, stride=15, minimum rows=3.
Non-overlapping chunks split at source gap > 30 or 30 rows. Candidate edges are time-forward with gap 1..600, stratified by `[('1-30', 1, 30), ('31-90', 31, 90), ('91-180', 91, 180), ('181-600', 181, 600)]`, top K=32 per source chunk/bucket. Ranking is 0.70 appearance cosine + 0.30 exp(-4*normalized center distance), then destination frame/track/chunk lexicographic order. GT cannot add, remove or reorder candidates.
Paired topology requires two same-bucket original edges, distinct endpoints, destination-start distance <= 30, and both crossed edges already present. At most 20000 stable combinations per sequence are frozen. Relation examples use the first 512 frozen edges per sequence/bucket; pair examples use the first 512 frozen combinations. Selection is label-blind.

## Labels and examples
Trusted chunk identity requires at least 2 known rows and purity >= 0.80. Node labels require at least 3 known rows: all one identity=pure, multiple identities=impure; unknown rows never imply impurity. Boundary is defined only when both adjacent rows are known (identity change=positive, same=negative, otherwise ignore). Candidate edge labels are positive only for two trusted same-identity chunks, negative only for two trusted different-identity chunks, otherwise ignore. Missing GT successor candidates are reported and never inserted. Paired samples are emitted only when original edges are positive and crossed edges exist in the frozen pool.

Minimum support gates: `{"train": {"boundary_negative": 100, "boundary_positive": 5, "impure_node": 5, "paired_replacement": 5, "pure_node": 100, "successor_negative": 100, "successor_positive": 20}, "validation": {"boundary_negative": 20, "boundary_positive": 1, "impure_node": 1, "paired_replacement": 1, "pure_node": 20, "successor_negative": 20, "successor_positive": 5}}`. These gates are fixed before labels and will not be lowered. Join rate, purity and sequence dominance are descriptive only.

## Provenance disclosure
Source tracker inference is image-only/GT-free, but `full7_best_raw` was historically selected using MOT17 comparison. Claims are limited to sequence-disjoint relation supervision/validation under a historically selected frozen source host. Source-host and M23-62 feature-extractor provenance are recorded separately.

## Decision
All positive gates pass: PASS_SUPERVISION_JOIN_AND_EXAMPLE_CONSTRUCTION and authorize only a future fresh experiment to train v3 from scratch using frozen example SHA. Otherwise close on the first root cause as FAIL_M23_62_REVERIFICATION, FAIL_JOIN_SEMANTICS, FAIL_TOPOLOGY_COMPATIBILITY, FAIL_EXAMPLE_VALIDATION, FAIL_SPLIT_LEAKAGE, or FAIL_SCOPE_GUARD. This experiment always ends with training_runs=0.
