# M23-59 v2 Relation-pretrained Hierarchical Identity Segmentation and Flow Feasibility — Preregistration

Date: 2026-07-20  
Status at freeze: **no MOT17 GT row parsed, no M23-59 model trained, no MOT20 teacher label opened**  
Data protocol: `external_supervision=true`  
Internal-data strict reference: M23-46, HOTA 79.123193  
Teacher-only capacity reference: M23-57 v2, HOTA 81.148046

## 0. v2 implementation-fix scope

M23-59 v1 is invalid because the first external-pretraining run exposed nondeterministic CuBLAS `Linear` execution: `CUBLAS_WORKSPACE_CONFIG` was not set before importing PyTorch, and deterministic algorithms were warning-only. No v1 artifact or metric may be reused.

v2 preserves the scientific route, data split, feature dimensions, four-head architecture, parameter cap, loss weights, seeds, epoch counts, optimizer, P0/P1/P2 family, representation/HOTA gates, tie-break, M23-55 candidate generator, flow solver, and evaluation rule. Before v2 freeze, the implementation was aligned exactly with those definitions: `CUBLAS_WORKSPACE_CONFIG=:4096:8` is set before importing `torch`; deterministic algorithms are hard-fail; CUDA/CuDNN TF32 are disabled; outgoing and incoming listwise negatives are represented separately; paired replacement uses two true relations versus both crossed relations; class-balanced focal and the explicit 0.5 group-DRO regularizer are used; false-split and wrong-link risks are calibrated on training sequences only; and every learned held tracker/graph is frozen before outer evaluation. Every stage is rerun from raw inputs in an empty v2 root, with no v1 artifact reuse.

## 1. Scientific question

M23-57 established that optional intra-node cuts plus the unchanged long-gap global path cover have enough teacher capacity, but an independent classifier over 1,076,231 boundaries is not observable across sequences. M23-59 tests one new route only:

1. detect whether a fixed chunk-30 node is impure;
2. conditionally localize one or more internal change points;
3. score outgoing and incoming successor relations for the resulting segments;
4. apply the unchanged one-to-one, time-forward, acyclic global path cover.

M23-59 does not revive M23-58, does not start M23-54, and does not change the M23-55 candidate K, gap quotas, source graph semantics, detector rows, boxes, or scores.

## 2. External data audit and physical grouping

The only new relation-label source is **MOT17 train** at the resolved local path `/gemini/code/datasets/MOT17/train`.

The 21 detector-variant folders are grouped into seven physical videos:

- MOT17-02
- MOT17-04
- MOT17-05
- MOT17-09
- MOT17-10
- MOT17-11
- MOT17-13

DPM, FRCNN, and SDP are not independent scenes. A full byte manifest verifies that their image trees and GT are identical within each physical ID. Only the FRCNN folder is admitted as the canonical copy. Detector result files are recorded but do not create additional samples.

Fixed physical split, chosen before parsing GT rows:

- external train: MOT17-02, MOT17-04, MOT17-05, MOT17-09, MOT17-10;
- external validation: MOT17-11, MOT17-13.

MOT17 test is excluded entirely. No MOT17 test image, detector result, or GT content is admitted. MOT20 test GT and MOT20 test images are prohibited.

External dataset manifest:

`outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_dataset_manifest.json`

Manifest SHA-256 at preregistration: `3c555e708a7341f6747a4a24ab9200bb9bf848d37b847dde6bc646b44e28e406`.

Counts:

- 7 unique physical sequences;
- 21 detector-variant directories;
- 5,316 canonical unique frames;
- 15,948 all-variant image files;
- 21 GT files;
- 21 detector result files;
- 21 sequence metadata files.

## 3. Frozen inherited appearance extractor

M23-59 inherits the existing M23-46/M23-57 per-detection appearance extractor and does not fine-tune it or train generic identity classification.

- config: `external/BoT-SORT-main/fast_reid/configs/MOT20/sbs_S50.yml`
- config SHA-256: `900ab06e87e407b30b1a5ee6167d2de543c9039d1a6d64254f4a62f150276065`
- weights: `external/BoT-SORT-main/pretrained/mot20_sbs_S50.pth`
- weights SHA-256: `d3a39a1ab54a63ac6724f2864a36485b0b5762fefced98941fca2a0784180656`
- interface: `external/BoT-SORT-main/fast_reid/fast_reid_interfece.py`
- interface SHA-256: `821a921b116b4ae2e4e1a0b32855180e29d534d35f5b32f4be009bdd8c0aa191`
- projection: deterministic Gaussian 2048→128, seed 2310, divided by `sqrt(128)`, followed by L2 normalization;
- projection implementation source: `scripts/m23_research/m23_10_build_micrograph.py`.

This inherited component is declared separately from the new external relation supervision. M23-59 does not claim an MOT17-only appearance extractor.

## 4. Label eligibility and external feature freeze

After implementation freeze, canonical MOT17-FRCNN GT rows are eligible only when:

- mark = 1;
- class = pedestrian (class 1);
- visibility ≥ 0.1;
- width and height > 1 pixel.

The frozen extractor is applied frame-wise to eligible boxes. The 2048-D output is projected to the fixed 128-D M23 representation. Geometry/context is 16-D and contains normalized center, width, height, aspect/area, visibility, velocity, scale changes, acceleration, frame delta, crowd density, and nearest-neighbor distance.

Per-sequence metadata, 128-D appearance, 16-D geometry, schema, counts, and SHA-256 are frozen before synthetic relation examples are built.

## 5. Fixed synthetic relation supervision

No random easy-negative pool is allowed. All negatives are from the same physical video and a different identity.

### 5.1 Node examples

- maximum node length: 30 observations;
- continuous run permits adjacent observation step ≤2 frames;
- pure windows: stride 15, cap 2,048 per physical sequence;
- A→B: 15 observations from A followed by 15 from B, cap 2,048 per sequence;
- A→B→A: 10 A + 10 B + 10 A, two separate boundary labels, cap 1,024 per sequence;
- a node may have multiple positive boundaries; no top-1 restriction.

The B partner must be from the same physical sequence, a different identity, within 30 frames of the A window center, normalized geometry distance ≤3, and scale discrepancy ≤1.4. Fixed hardness score:

`0.55 * appearance cosine + 0.30 * exp(-geometry distance) + 0.15 * exp(-|time offset| / 30)`.

Tie-break: higher hardness, lower identity ID, earlier frame, lower deterministic window index.

### 5.2 Relation examples

- endpoint segment length: 8 observations;
- gap buckets: 1–30, 31–90, 91–180, 181–600 frames;
- cap: 1,024 positive continuation triplets per gap bucket and physical sequence;
- paired replacement/swap cap: 1,024 per sequence.

Each positive continuation `A→B` is paired with two independently frozen hard alternatives from the same physical video:

- outgoing alternative `A→N`: `N` is a different identity, its start is within 30 frames of `B`, geometry distance ≤3, and it is ranked by `0.70 * cos(A,N) + 0.30 * exp(-geometry distance)`;
- incoming alternative `C→B`: `C` is a different identity, its end is within 30 frames of the true predecessor `A`, `C→B` lies in the same gap bucket, geometry distance ≤3, and it is ranked by `0.70 * cos(C,B) + 0.30 * exp(-geometry distance)`.

Paired replacement uses two true same-bucket relations `A→B` and `C→D` from different identities. Their destination starts must be within 30 frames and both crossed relations `A→D`, `C→B` must remain time-forward. The loss compares the summed true energy against the summed crossed energy.

Tie-break for every hard alternative: higher hardness, lower identity ID, earlier frame, lower deterministic segment index.

## 6. Fixed architecture

One model family only:

- appearance projection: Linear 128→128 + LayerNorm + GELU;
- geometry/context projection: Linear 16→32 + LayerNorm + GELU;
- shared temporal encoder: two-layer bidirectional GRU;
- GRU hidden size: 128 per direction;
- GRU dropout: 0.1;
- parameter cap: 5,000,000;
- frozen implementation parameter count from synthetic construction: 881,124.

Exactly four heads:

1. node impurity head over mean/max pooled node representation;
2. conditional boundary head over left/right/absolute-difference GRU states, producing all 29 potential boundaries;
3. cross-relation head over source/destination pooled segment pairs;
4. catastrophic-risk head over the same relation pair representation.

No Transformer, GNN, hidden-size scan, layer scan, architecture scan, generic ID softmax, ordinary 2048-D ReID reranker, or delta-HOTA regression is allowed.

## 7. Fixed losses

The fixed objective is:

- node class-balanced focal loss, weight 1.0;
- conditional multi-boundary class-balanced focal loss, weight 1.0;
- boundary-count consistency, weight 0.2;
- outgoing listwise `softplus(-(s(A,B)-s(A,N)))`, weight 0.5;
- incoming listwise `softplus(-(s(A,B)-s(C,B)))`, weight 0.5;
- paired replacement `softplus(-((s(A,B)+s(C,D))-(s(A,D)+s(C,B))))`, weight 0.5;
- catastrophic-risk BCE over true relations as safe and outgoing/incoming/crossed relations as catastrophic, weight 1.5;
- sequence group-DRO regularizer, weight 0.5, fixed eta 0.1, equal to the mean of node-focal group-DRO and mean outgoing/incoming relation group-DRO;
- edit sparsity/source-anchor regularization, weight 0.02.

The total is the exact weighted sum above. No task term is silently replaced by group-DRO, and paired replacement is applied only to paired examples. HOTA is never an action-level or boundary-level training label.

## 8. External training protocol

- optimizer: AdamW;
- learning rate: 3e-4;
- weight decay: 1e-4;
- epochs: exactly 30;
- node batch size: 256;
- relation batch size: 256;
- gradient norm cap: 5.0;
- random seeds: 2359001, 2359002, 2359003;
- `CUBLAS_WORKSPACE_CONFIG=:4096:8` fixed before importing PyTorch; deterministic algorithms are hard-fail; CUDA/CuDNN TF32 disabled; no early stopping.

Checkpoint selection uses only MOT17 physical validation sequences. Fixed validation composite:

`0.30 * node impurity PR-AUC + 0.35 * conditional boundary PR-AUC + 0.25 * mean(outgoing successor R@1, incoming predecessor R@1) + 0.10 * (1 - catastrophic false-link rate)`.

Tie-break: higher composite, lower seed, earlier epoch. No MOT20 metric participates in external checkpoint selection.

The selected checkpoint, identity scaler declaration, external train/validation example manifests, model config, parameter count, and SHA-256 are frozen. The same checkpoint initializes all four MOT20 outer folds.

## 9. MOT20 GT-free observable freeze

Before opening any M23-57 teacher label, all four MOT20 sequences freeze:

- M23-46 source tracker SHA;
- exact M23-57 fixed chunk membership SHA;
- per-row 128-D appearance and 16-D geometry/context tensor;
- schema and artifact SHA;
- `gt_opened=false`;
- `outer_teacher_action_read=false`.

All four observable freeze events must precede the first MOT20 label unlock event.

## 10. Strict nested sequence-LOSO

For outer held sequence S:

- the common external checkpoint is fixed;
- only the other three MOT20 sequences may be used;
- three inner LOSO folds are run: two sequences fine-tune, one validates;
- fine-tuning epochs: exactly 10;
- learning rate: 1e-4;
- optimizer: AdamW;
- weight decay: 1e-4;
- the validation model, identity scaler declaration, node threshold, boundary threshold, and P1/P2 risk-calibration manifest are written before that fold first reads its validation labels;
- node threshold is the training-only threshold with maximum recall at precision ≥95%;
- boundary threshold is the training-only threshold with maximum recall at precision ≥90%;
- the risk head is oriented as `wrong/different relation probability`; link risk uses it directly, while false-split risk for a proposed cut is `1 - risk-head probability`;
- for each policy and action type, the training-only calibrated score threshold is the largest lower-is-safer cutoff whose selected-set one-sided 95% Clopper-Pearson UCB is ≤2% for P1 or ≤5% for P2; insufficient support freezes the cutoff to `-1`, admitting no such action;
- no threshold grid, risk level, or coverage target is scanned.

The outer held sequence is excluded from fine-tuning, normalization, calibration, checkpoint selection, threshold selection, policy selection, and teacher-action construction.

## 11. Fixed representation gate

For each outer, aggregate the three inner validation folds. Passing requires all conditions:

- mean conditional boundary PR-AUC ≥0.283;
- mean precision at actual boundary count ≥0.35;
- mean recall at 95% precision ≥0.05;
- every inner fold precision at actual boundary count ≥0.20;
- every inner fold pure-node false-split rate ≤0.002.

ROC-AUC is reported only as an auxiliary metric and cannot pass the gate.

Also report node impurity PR-AUC, precision at actual impure-node count, recall at 95% precision, pure-node false-split rate, boundary offsets, A→B→A event recall, successor MRR, successor R@1/R@8/R@32, top-cross repair precision, and catastrophic top-1 false-link rate.

If the representation gate fails, the outer policy is immediately frozen to P0. No learned inner tracker or TrackEval is run for that outer.

## 12. Fixed inference graph and policy family

Only after a representation pass:

1. start from byte-exact M23-46;
2. score all fixed nodes and all valid internal positions;
3. admit a cut only when node, boundary, and catastrophic-risk conditions all pass;
4. allow multiple cuts in one node;
5. deterministically rebuild segments;
6. regenerate the unchanged M23-55 candidate pools with ranking K=256 and flow K=32, legacy max gap 600, appearance bank K=32, motion bank K=8;
7. score source/cross relations;
8. use one-to-one sparse global path cover with zero-weight dummy terminate/restart;
9. preserve detection rows, boxes, scores, time-forward order, and acyclicity.

Fixed learned edge utility:

`relation logit + 0.25 * parent_edge - 0.25 * cross_edge`.

A cut or link is eligible only when its training-only calibrated lower-is-safer score is at or below the frozen policy-specific cutoff. The cutoff itself is selected solely by the fixed Clopper-Pearson rule above; the numeric probability 0.02/0.05 is a UCB limit, not an uncalibrated neural-score threshold. No candidate K, gap quota, edit budget, source-anchor weight, or edge-utility scan is allowed.

Policies:

- P0: byte-exact M23-46 no-op;
- P1: training-only calibrated cut/link cutoffs with actual one-sided 95% Clopper-Pearson catastrophic-risk UCB ≤2%;
- P2: training-only calibrated cut/link cutoffs with actual one-sided 95% Clopper-Pearson catastrophic-risk UCB ≤5%.

No other risk level exists.

## 13. Inner exact TrackEval gate

Only representation-passing outers evaluate P1/P2 inner trackers. For each policy:

- all three inner validation ΔHOTA must be ≥+0.05;
- mean inner validation ΔHOTA must be ≥+0.20;
- all three catastrophic-risk UCB checks must pass;
- rows, boxes, and scores must remain unchanged.

If both policies pass, fixed tie-break:

1. maximum worst-fold ΔHOTA;
2. maximum mean ΔHOTA;
3. fewest identity edits;
4. prefer P1.

Otherwise freeze P0. No near-pass relaxation is allowed.

## 14. Outer model and policy freeze

A learned outer policy, if any, is fine-tuned for exactly 10 epochs on all three outer-training sequences. Node and boundary thresholds are the medians of the three previously frozen inner-training-only thresholds. P1/P2 cut/link score cutoffs are recalibrated only on the three outer-training sequences with the same fixed Clopper-Pearson rule. The held sequence is then inferred GT-free; its rebuilt segment graph, full candidate graph, selected flow, tracker, and all SHA values are frozen before the outer policy manifest is written. No held sequence label or statistic is used.

Each outer manifest records:

- external checkpoint SHA;
- fine-tuned model SHA or null for P0;
- identity scaler SHA;
- risk calibration SHA or null for P0;
- boundary policy;
- flow policy;
- candidate graph SHA;
- selected-flow SHA and frozen held-tracker SHA for a learned policy;
- `outer_tracker_frozen_before_outer_gt=true` for a learned policy;
- `outer_gt_read=false`;
- `outer_teacher_action_read=false`.

All four outer policy manifests must be frozen before any outer TrackEval.

## 15. Official evaluation and stopping rules

If all four outers are P0:

- outer TrackEval runs = 0;
- COMBINED TrackEval runs = 0;
- byte-exact tracker SHA proves the result remains M23-46;
- decision branch C closes M23-59.

If at least one outer is learned:

- each held fold is evaluated exactly once using only the tracker already referenced by its frozen outer manifest;
- no held tracker is regenerated after manifest freeze;
- four frozen trackers are stitched;
- COMBINED is evaluated exactly once;
- outer metrics cannot change any model, calibration, threshold, or policy.

Final classification:

- A: COMBINED HOTA >80, report separately as strict external-pretrained best;
- B: strict gain but HOTA ≤80, report without post-hoc scan;
- C: representation gate, risk gate, or exact-HOTA gate fails, freeze P0 and close.

M23-46 remains the independent internal-data-only strict best unless a later internal-only protocol supersedes it. M23-59 can never silently replace that category.

No MOT20 test submission is permitted. If branch A occurs, only prepare a package and wait for explicit user approval plus a current MOTChallenge external-supervision rule check.

## 16. Error handling

If any implementation error is discovered after MOT17 labels or MOT20 teacher labels are opened:

- invalidate the entire current version;
- stop remaining same-version processes;
- preserve logs and artifacts;
- do not use invalid metrics for gating;
- fix under a new version and rerun from an empty output root;
- no partial continuation or artifact reuse.

## 17. Prohibited actions

- no M23-54;
- no M23-58;
- no MOT20 test submission;
- no MOT17 test content;
- no MOT20 test images or transductive adaptation;
- no private or unknown data;
- no undeclared checkpoint;
- no generic ID softmax;
- no large Transformer/GNN;
- no delta-HOTA supervision;
- no candidate-K, gap-quota, risk-level, edit-budget, threshold-grid, or conformal scan;
- no change to detection rows, boxes, or scores;
- no use of an outer-held MOT20 label before policy freeze.

## 18. Superseded invalid version

- invalid version: M23-59 v1
- invalidation report: `docs/m23_59_v1_invalidated_determinism_20260720.md`
- v1 artifacts and metrics are prohibited from reuse.
