# Pending Notion Sync｜P16 Actual-Anchor Motion Closure

Target page ID: `39dabff3-387a-8125-99da-caec2ec8f7ec`

Reason pending: the current tool session does not expose a writable Notion connector. No successful Notion write is claimed.

Source of truth:

`deliverables/assocriskbench_p16_actual_anchor_motion_closure_20260716.md`

## Content to append

- Corrected directional transaction semantics from nominal `u/v` to actual source `edited_label` plus row-conditioned receiver-family `baseline_label`.
- Identified 15/225 aggregate-anchor or remapped transactions.
- Built boundary-local appearance and third-party competition features for all 225 events with 99.58% mean future ReID coverage and no label leakage.
- Third-party appearance provides open-set signal (`pair_vs_third_h120_q25` oriented AUC 0.705761) but fails on the visually similar MOT20-02 rank72 negative event.
- Built 367 actual-anchor motion features and a 36-feature compact model set.
- Actual-anchor motion improves sequence-disjoint local Spearman from 0.417656 to 0.495464 and HOTA-positive AUC from 0.641008 to 0.699388.
- Non-nested expanded-motion top-one reaches +0.497074 HOTA but has worst-sequence HOTA −0.064545 and one catastrophic window; it is not a deployment estimate.
- Strict nested multitask motion veto finds zero eligible gate in every outer fold; all four outer decisions are `no_op`.
- Candidate stability is not the bottleneck: all gate failures arise from insufficient positive precision, excessive negative mass, or sequence-negative/catastrophic utility.
- Constrained top-k fallback has zero configuration with both nonnegative worst HOTA sequence and nonnegative worst local-teacher sequence.
- All five formal evidence chains reproduce byte-identically.
- New locked labels read: 0.
- New locked TrackEval calls: 0.
- Locked manifest created: no.
- Remaining locked rows unread: 156.
- Final decision: close the current P16 gate/veto/fallback family and expand independent actual-anchor counterfactual domains before any future locked prediction.
