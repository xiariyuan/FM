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
