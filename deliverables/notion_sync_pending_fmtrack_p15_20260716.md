# Pending Notion Sync

Target page ID: `39dabff3-387a-8125-99da-caec2ec8f7ec`

Reason pending: the Notion connector was not exposed in the current tool session on 2026-07-16. No successful Notion write is claimed.

Content to append: `deliverables/assocriskbench_p15_directional_sequence_abstention_20260716.md`

Required status after sync:

- Mark directional sequence abstention V1 as train-only and locked-label clean.
- Record that no preregistered gate passed and the frozen decision is `no_op`.
- Record zero additional locked labels consumed and 156 remaining unread.
- Preserve the next-step requirement: redesign candidate-level lower-tail risk calibration before any additional locked reveal.

## Pending section｜Directional Utility Risk Modeling Closure

Source of truth: `deliverables/assocriskbench_p15_directional_risk_model_closure_20260716.md`

- Completed nested window conformal, nested pairwise/listwise ranking, sequence-conditioned KNN retrieval, and locked prediction stability audits.
- All four evidence chains reproduced byte-identically.
- Oracle analysis: 16/16 train windows contain a positive executable direction; oracle best-candidate sum is +0.877901 HOTA.
- No risk model met the deployment constraints.
- Locked selections remain 0.
- No new locked utility labels or TrackEval results were read.
- All 156 remaining locked directional rows remain unread.
- Decision: close the current directional utility-learning family on frozen P15; next work must expand train-only counterfactual supervision rather than tune against the locked pool.

## Pending section｜Local Counterfactual Supervision

Source of truth: `deliverables/assocriskbench_p15_local_counterfactual_supervision_20260716.md`

- Built exact dense local counterfactual supervision for 225 accepted ranks21–100 directional events.
- Verified 225/225 edited trajectory SHA values and recovered 31,711 changed-row labels with zero replay-count mismatches.
- Generalized the label definition to support one event containing multiple baseline receiver anchors merged into one donor.
- `full_idtp_delta_norm` strongly matches global association utility: Spearman with delta_AssA 0.803645, positive-HOTA AUC 0.950506, 15/16 positive train-window top-one choices, and +0.703428 HOTA total.
- Thirteen preregistered local targets passed all qualification constraints.
- The sequence-disjoint observable-feature ensemble learned partial signal but failed deployment: +0.136369 HOTA total, 9/16 positive windows, worst sequence -0.040599, positive-HOTA AUC 0.678143.
- A nested local-positive gate found no eligible threshold in any outer fold; all four folds selected no-op.
- All three formal evidence chains reproduced byte-identically.
- No locked artifact, locked utility label, or new locked TrackEval result was read; all 156 remaining locked directional rows remain unread.
- Decision: preserve the dense labels as a train-only teacher, stop the current threshold/gate family, and next expand independent counterfactual domains plus hierarchical/domain-generalized observable features.
