# Segment Pair-State Graph Pilot — 2026-07-14

## Motivation

Unary change-point proposals identify suspicious boundaries but cannot determine the identity transition partner. This pilot builds partner-aware pair-state candidates.

## Pair bank

MOT20-02:

- unary proposals: 5000
- proposals with observable overlap partner: 4907
- pair rows: 11713
- valid pair ReID rows: 11475
- reciprocal swap rows: 91
- identity-related rows: 141

## Leakage correction

The first pair ranker accidentally used the inherited GT-derived `target` column. This run is invalid and discarded.

The clean rerun excludes:

- target
- label_*
- a_gt_*
- b_gt_*
- distance_to_switch_*

Clean feature count: 57.

## Clean diagnostic results

GroupKFold by track_a:

Reciprocal swap:

- unary AP: 0.02890
- pair ensemble AP: 0.31560
- pair ensemble AUC: 0.97019

Identity-related events:

- unary AP: 0.05372
- HGB AP: 0.36564
- HGB AUC: 0.94527

## Next

Canonicalize mirrored events and evaluate fixed transaction replay.
