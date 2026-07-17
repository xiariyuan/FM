# Notion Sync Pending — FM-Track M23-0

Target page ID: `39dabff3-387a-8125-99da-caec2ec8f7ec`

Status: pending because no writable Notion connector is available in this session.

## Stage

M23-0 MOT20 post-NMS expanded-evidence oracle audit.

## Protocol

- MOT20 train sequences 01/02/03/05.
- Frozen all-four baseline rows.
- Existing Phase-0 YOLOX postprocess/NMS detections, score >= 0.09.
- Novel candidate criterion: maximum same-frame IoU to baseline < 0.90.
- MOTChallenge distractor preprocessing and GT matching at IoU 0.50.
- No model training or threshold sweep.

## Main results

- Baseline HOTA: 77.699.
- Baseline ID oracle HOTA: 82.571.
- Expanded additive oracle HOTA: 83.630.
- Expanded replace/add oracle HOTA: 83.676.
- Expanded selected pool ceiling HOTA: 84.265.
- Candidate pool ratio: 1.08218, below the 1.5 budget.
- Newly recovered valid GT rows: 21,878.
- 74.70% of additive recoveries came from detection score below 0.60.

Per-sequence replace/add HOTA:

- MOT20-01: 83.936.
- MOT20-02: 82.984.
- MOT20-03: 83.581.
- MOT20-05: 83.850.

## Decision

The existing post-NMS low-confidence pool is sufficient to make HOTA 82.5 technically plausible and removes the old near-exact-oracle requirement. However, the optimistic selected ceiling is 84.265 and therefore misses the pre-registered 84.5 safety-margin target by 0.235.

Proceed to M23-1: capture NMS-suppressed detector candidates before NMS and repeat the same oracle audit. Do not begin expensive full-pool ReID extraction until the pre-NMS ceiling is established.

Deployment remains false. P15 remains no-op. Locked-label reads and locked TrackEval calls remain zero; 156 locked rows remain untouched.

## Reproducibility

Formal and independent reproduction chains are byte-identical for all compact outputs, all 20 generated tracker hashes, and all five TrackEval summaries.

Formal report:

`deliverables/mot20_m23_expanded_evidence_oracle_20260717.md`

Audit:

`deliverables/mot20_m23_audit_20260717.json`

Formal output:

`outputs/mot20_m23_20260717/expanded_evidence_oracle_v1`

Key report SHA256:

`032655a04cc3e8b734d04af432d21be717869d581260de2680d87e1bf04566f4`
