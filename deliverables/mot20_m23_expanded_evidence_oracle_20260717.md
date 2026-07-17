# M23-0 MOT20 Post-NMS Expanded-Evidence Oracle Audit

Date: 2026-07-17

## 1. Research question

Can the existing MOT20 Phase-0 low-confidence detection pool create enough oracle headroom to make a train HOTA target of 82.5 technically plausible without changing the detector?

This stage is an offline oracle audit only. It does not train or authorize a deployable policy.

## 2. Fixed protocol

Sequences: MOT20-01, MOT20-02, MOT20-03, MOT20-05.

Evidence:

- Frozen TrustTrack all-four baseline tracker rows.
- Existing Phase-0 YOLOX detections with score >= 0.09.
- Source-code inspection confirmed that Phase-0 calls YOLOX `postprocess`, so this evidence is post-NMS rather than pre-NMS.
- A Phase-0 row is treated as geometrically novel when its maximum same-frame IoU against baseline rows is below 0.90.

Oracle assignment:

- MOTChallenge distractor preprocessing at IoU 0.50.
- Valid pedestrian matching at IoU 0.50.
- Threshold-before-Hungarian assignment.
- No threshold, model, or evidence variant was chosen after seeing results.

Variants:

1. `baseline_raw`: unchanged baseline output.
2. `baseline_id_oracle`: preserve every baseline row and box; change IDs only.
3. `expanded_additive_oracle`: preserve baseline rows and add Phase-0 candidates only for baseline-missed GT.
4. `expanded_replace_add_oracle`: preserve baseline unmatched/FP burden, replace matched baseline observations when an expanded candidate is better, and add newly recovered GT.
5. `expanded_selected_ceiling`: retain only the best expanded candidate for each matched GT. This is an optimistic pool ceiling because unmatched rows are removed by GT.

Pre-registered evidence target:

- Expanded oracle HOTA >= 84.5.
- Expanded candidate rows <= 1.5 times baseline rows.

## 3. Combined MOT20-train results

| Variant | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | Delta HOTA |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 77.699 | 80.907 | 74.672 | 89.308 | 93.606 | 1,222 | 0.000 |
| Baseline ID oracle | 82.571 | 81.324 | 83.872 | 96.882 | 93.857 | 0 | +4.872 |
| Expanded additive oracle | 83.630 | 82.430 | 84.884 | 97.855 | 95.734 | 13 | +5.931 |
| Expanded replace/add oracle | **83.676** | **82.480** | **84.924** | **97.881** | **95.784** | **6** | **+5.977** |
| Expanded selected ceiling | **84.265** | **83.561** | **85.011** | **98.657** | **97.349** | **6** | **+6.566** |

The row-preserving expanded oracle exceeds the desired 82.5 target by 1.176 HOTA. The optimistic selected ceiling exceeds it by 1.765 HOTA.

However, neither expanded oracle reaches the pre-registered 84.5 safety-margin target:

- Replace/add oracle: 83.676, short by 0.824.
- Selected pool ceiling: 84.265, short by 0.235.

## 4. Per-sequence replace/add oracle

| Sequence | Baseline | Baseline ID oracle | Expanded replace/add | Expanded selected ceiling |
|---|---:|---:|---:|---:|
| MOT20-01 | 77.030 | 82.178 | **83.936** | 84.342 |
| MOT20-02 | 68.748 | 81.684 | **82.984** | 83.457 |
| MOT20-03 | 80.054 | 82.178 | **83.581** | 84.012 |
| MOT20-05 | 78.539 | 82.954 | **83.850** | 84.551 |

All four sequences improve over their baseline ID oracle. The prior MOT20-05 negative-transfer failure is absent at the oracle level. MOT20-02 remains the limiting domain.

## 5. Candidate-pool findings

Baseline source rows: 1,100,703.

Phase-0 post-NMS rows: 1,188,001.

After same-frame IoU >= 0.90 duplicate removal:

- Novel Phase-0 rows: 90,451.
- Expanded pool rows: 1,191,154.
- Expanded/baseline row ratio: **1.08218**.
- Candidate-budget condition: passed.

At IoU 0.50:

- Baseline matched valid GT rows: 1,082,673.
- Expanded replace/add matched GT rows: 1,104,551.
- Newly recovered GT rows: **21,878**.
- Coverage: 95.4221% -> 97.3504%, an increase of 1.9282 percentage points.

The additive oracle recovered 21,309 GT rows. Their Phase-0 score distribution was:

- score >= 0.60: 5,391.
- 0.10 <= score < 0.60: 15,121.
- 0.09 <= score < 0.10: 797.

Therefore **74.70%** of recovered GT rows came from score below 0.60. Low-confidence evidence is the main source of added coverage, rather than duplicated high-confidence boxes.

## 6. What this changes about the 82.5 target

Under the old baseline-only ID oracle, reaching 82.5 required recovering 98.54% of the entire oracle gain from 77.699 to 82.571. That was nearly an exact-oracle requirement.

Under the row-preserving expanded oracle, reaching 82.5 requires recovering approximately **80.32%** of the gain from 77.699 to 83.676.

Under the optimistic selected ceiling, it requires approximately **73.12%** of the gain from 77.699 to 84.265.

Thus 82.5 is no longer mathematically pinned to the old oracle boundary. It is technically plausible, but it still requires a strong global selector that captures most useful low-confidence observations while rejecting the majority of distractors.

## 7. Formal decision

- `candidate_budget_passed = true`
- `expanded_replace_add_target_84p5_passed = false`
- `expanded_selected_ceiling_target_84p5_passed = false`
- `deployment_allowed = false`
- `locked_manifest_created = false`
- P15 policy remains `no_op`
- New P15 locked-label reads: 0
- New locked TrackEval calls: 0
- Remaining locked rows untouched: 156

The existing post-NMS pool is sufficient to justify continuing toward HOTA 82.5, but it does not provide the pre-registered 84.5 safety margin.

## 8. Required next stage

The next stage should be **M23-1: Pre-NMS Suppressed-Evidence Oracle Audit**.

It should modify the Phase-0 detector path to save detector candidates before class-aware NMS, without initially extracting ReID for every row. The audit should test:

1. NMS-suppressed high-confidence boxes.
2. Post-NMS low-confidence boxes already validated here.
3. Their deduplicated union with baseline observations.
4. The same row-preserving replace/add and optimistic selected ceilings.

Acceptance rule:

- Selected expanded ceiling >= 84.5.
- Worst sequence selected ceiling >= 84.0.
- Deduplicated candidate pool <= 1.5 times baseline rows.

Only after this ceiling is established should expensive appearance extraction and global tracklet-graph training begin. If pre-NMS evidence still fails, the next evidence source should be short-horizon propagation rather than additional score-threshold tuning.

## 9. Reproducibility

The independently generated formal and reproduction chains are byte-identical for:

- Candidate inventory.
- Selected-event table.
- Combined metrics.
- Per-sequence metrics.
- Report and manifest.
- All 20 generated tracker result hashes.
- All five TrackEval summary files.

Key SHA256 values:

- Script: `d9472db58eb27a712bcd7d65040b0f2d0c3183d6c3a35aea5b6c5e430047342b`
- Candidate inventory: `4c8d342f79301e462ee72d90a36677847e7ab375fd013483939c2118205f9cee`
- Selected events: `76ed4e41e09fd6d7408e4d79ab4907dbcd308f9767f601cb944c7f65c1cbb715`
- Per-sequence metrics: `1535d94c7f77af48f5212b5cbe68bf67b02acaf1ee05637fdaa2590dacb694cd`
- Variant metrics: `7fc6a558e50a5b9e7c3caebb106f5fa34e46ea705f5fe9ae73797674221bb6d9`
- Report: `032655a04cc3e8b734d04af432d21be717869d581260de2680d87e1bf04566f4`
- Manifest: `dc72c8fea28571adc48944bc4ccf17bc8ee1788fb6acc785f52a0849242971c4`
