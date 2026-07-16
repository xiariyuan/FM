# AssocRiskBench P16｜Actual-Anchor Appearance and Motion Risk Closure｜2026-07-16

## 1. Scope and locked-data guarantee

This stage tested whether denser appearance and motion mechanisms can convert the P16 local counterfactual teacher into a sequence-disjoint directional deployment policy.

- Training events: 225 accepted directional events from ranks 21–100.
- Dense changed-row supervision available for offline training audit: 31,711 rows.
- Remaining locked directional rows before and after this stage: 156.
- New locked utility labels read: 0.
- New locked TrackEval calls: 0.
- New locked manifest created: no.
- Final deployment policy: `no_op`.

The stage does not reopen the frozen P15 locked pool and does not use locked artifacts for feature extraction, model selection, veto selection, or diagnostics.

## 2. Executor semantic correction: nominal IDs are not always the actual transaction

The geometry diff audit found that the executable transaction is not always the nominal `u_to_v` or `v_to_u` pair.

- Actual source identity: the unique `edited_label` in the executor geometry diff.
- Actual receiver family: the per-row `baseline_label` in the executor geometry diff.
- Events using an aggregate or remapped anchor: 15 / 225.
- One transaction can affect more than one receiver-family label.

All appearance and motion features in this stage therefore use the actual source and row-conditioned receiver family. This correction is essential: using nominal `u/v` IDs would describe the wrong causal transaction for aggregate-anchor events.

## 3. Boundary-local appearance and third-party competition

Artifact: `outputs/assocriskbench_p15_20260716/directional_boundary_local_appearance_v2`

Builder: `scripts/build_directional_boundary_local_appearance_features.py`

Protocol:

- Memory-map the uncompressed YOLOX/FastReID detection feature dumps.
- Build event-pre source and receiver-family appearance prototypes.
- Match event-post changed receiver rows to detection embeddings.
- Compare each future row with source, receiver-family, and all safe third-party tracklet-start prototypes whose fifth baseline row predates the event.
- Read only execution keys, geometry, actual baseline labels, and actual edited labels from the changed-row table.

Data audit:

- Events: 225.
- Feature columns: 556.
- Duplicate directional keys: 0.
- Forbidden GT, `row_class`, utility, HOTA, AssA, or IDTP-delta columns: 0.
- Mean future ReID coverage: 0.995802.
- Minimum event coverage: 0.875000.
- Minimum third-party candidates per event: 37.
- Events without a third-party candidate: 0.

The strongest predefined open-set feature, `pair_vs_third_h120_q25`, reaches an oriented AUC of 0.705761 for events containing `other_gt` or `unmatched` rows. However, it does not reject the critical MOT20-02 rank72 failure:

- `full_idtp_delta_norm`: −0.136961.
- `delta_HOTA`: −0.055850.
- `pair_vs_third_h120_q25`: +0.220323.

Interpretation: third-party appearance competition contains useful open-set signal, but visually similar different identities remain indistinguishable. Appearance cannot provide a sufficient deployment certificate by itself.

Key hashes:

- Appearance feature CSV: `db628ae408a0d4837c75c223f73a51ce984624b4893f5191fe5221a372d06a27`.
- Report: `504018be41b691c555a8a9734f1e0b0acb8445d099d26ac8460b57f0b93b7c0b`.
- Manifest: `94b49d3cb759b42c1c47d33254225c1f9c9ad1b48e49a1cefac820cf22674cd1`.

## 4. Actual-anchor future motion transfer

Artifact: `outputs/assocriskbench_p15_20260716/directional_actual_anchor_motion_v1`

Builder: `scripts/build_directional_actual_anchor_motion_features.py`

Protocol:

- Fit event-pre constant-velocity and log-size states for the actual source and every actual receiver-family member.
- Evaluate event-post trajectory residuals at h30, h60, h120, and full horizons.
- Define motion error as center residual + 0.25 × bottom residual + 0.10 × log-size residual.
- Record source-vs-receiver residual margin, source-win fraction, and future velocity mismatch.
- Use 36 preregistered compact motion features for downstream learning.

Data audit:

- Events: 225.
- Total motion feature columns: 367.
- Compact model features: 36.
- Duplicate keys: 0.
- Forbidden label or metric columns: 0.
- Maximum missing fraction: 0.013333.
- Aggregate-anchor events: 15.

Sequence-disjoint fixed LOSO diagnostics show a real incremental representation gain:

| Metric | Existing 90 features | Existing + 36 actual-anchor motion features |
|---|---:|---:|
| Local-utility Spearman | 0.417656 | **0.495464** |
| HOTA Spearman from local prediction | 0.476386 | **0.548914** |
| HOTA-positive AUC from local prediction | 0.641008 | **0.699388** |
| Direct HOTA classifier AUC | 0.597575 | **0.609228** |

Therefore, actual-anchor future motion is useful model information. It improves domain-disjoint ranking correlation and sign discrimination, unlike the direct high-dimensional appearance concatenation.

Key hashes:

- Motion feature CSV: `932f4ea78c73a23745f8799c8e6bf10f2ce1b2b478524894cfbd0f8eade1b9f3`.
- Compact feature list: `16968c599d291a190224e0db5e8fa41b083c0b89dfea591890b6fa70668c3ac6`.
- Report: `8f98886a6562e9f0db8264ae40ce2c2b7e8ec4e82eb6eea3828d33011d59bb40`.

## 5. Multitask uncertainty gate without motion

Artifact: `outputs/assocriskbench_p15_20260716/directional_nested_multitask_uncertainty_gate_v1`

Script: `scripts/train_nested_multitask_uncertainty_gate.py`

The model predicts all 13 qualified local teacher targets with five seeds and uses only inner sequence-disjoint local utility for model, score, and gate selection.

Strict nested result:

- MOT20-01 outer fold: no eligible configuration.
- MOT20-02 outer fold: no eligible configuration.
- MOT20-03 outer fold: one configuration family authorized four windows.
- MOT20-05 outer fold: no eligible configuration.
- Outer selected windows: 4.
- Outer HOTA sum: −0.062665.
- Outer local utility: −0.145178.
- HOTA catastrophic windows: 1.
- Deployment allowed: false.

This result shows that prediction stability and low seed dispersion can remain high while the selected candidate is systematically wrong in a new sequence.

Report hash: `30c3a4d37a986ca11dffde4672378e9a6161031c9f05c2f115eacd61bfb5c7e9`.

## 6. Strict nested multitask ranker plus motion hazard veto

Artifact: `outputs/assocriskbench_p15_20260716/directional_nested_multitask_motion_hazard_veto_v1`

Script: `scripts/train_nested_multitask_motion_hazard_veto.py`

Fixed ranker:

- 90 existing observable features + 36 actual-anchor motion features.
- 13 qualified local targets.
- ExtraTrees, 300 trees, depth 7, minimum leaf 3, max-feature fraction 0.65.
- Five fixed seeds.
- Candidate score: mean normalized prediction across the 13 tasks.

Motion policy:

- Motion never freely reranks the window.
- The ranker first chooses top-one.
- A preregistered motion certificate may retain or abstain only.
- Inner three-sequence LOSO selects the certificate with `full_idtp_delta_norm`; HOTA remains outer audit only.

Strict nested result:

- Eligible gates in every outer fold: 0.
- Outer choices: `no_op`, `no_op`, `no_op`, `no_op`.
- Outer selected windows: 0.
- Deployment allowed: false.

Failure decomposition over all 48 outer-fold/gate combinations:

- Positive precision below 0.70: 48 / 48.
- Negative mass above the allowed ratio: 48 / 48.
- At least one catastrophic local window: 43 / 48.
- Negative worst-sequence utility: 36 / 48.
- Candidate-stability failure: 0 / 48.
- Coverage failure: 0 / 48.

The bottleneck is therefore not seed stability or candidate coverage. The ranking errors are stable but wrong under sequence shift, and a one-dimensional abstention threshold cannot repair them.

Report hash: `6d1fa30da1ebdd815ebc37355bb8e04e461a851b432ad2353ed3e20885451181`.

## 7. Non-nested upper diagnostics and constrained fallback bound

Artifact: `outputs/assocriskbench_p15_20260716/directional_actual_anchor_motion_closure_diagnostics_v1`

Script: `scripts/audit_directional_actual_anchor_motion_closure.py`

The non-nested expanded-motion top-one diagnostic demonstrates potential but is not a deployment estimate:

- HOTA sum: +0.497074.
- MOT20-01: +0.570522.
- MOT20-02: −0.027367.
- MOT20-03: −0.064545.
- MOT20-05: +0.018464.
- Worst local-teacher sequence: −0.145178.
- Catastrophic HOTA windows: 1.

The registered constrained fallback diagnostic permits a fallback only among the top 2/3/5 utility candidates when the top-one motion certificate fails. Across all previously tested thresholds and fallback rules:

- Configurations with nonnegative worst HOTA sequence and nonnegative worst local-teacher sequence: 0.
- Best robust diagnostic threshold: 0.40.
- Top-k: 2.
- HOTA sum: +0.128210.
- Worst HOTA sequence: −0.004005.
- Local utility sum: +0.381441.
- Worst local-teacher sequence: −0.035280.
- Catastrophic HOTA windows: 0.
- Fallbacks: 3.
- Abstentions: 4.

The fallback removes catastrophic errors but still cannot make all sequence domains nonnegative. It is therefore not promoted to another nested search stage.

Report hash: `7a87b31521ddb101a451409913d18abc9b789edbf8bad755e0e9cd10436b9bea`.

## 8. Reproducibility and final audit

All five formal evidence chains were independently reproduced:

1. Boundary-local appearance v2.
2. Actual-anchor motion v1.
3. Nested multitask uncertainty gate v1.
4. Nested multitask motion hazard veto v1.
5. Actual-anchor motion closure diagnostics v1.

For every chain, the formal and reproduction directories contain the same files and every corresponding file has the same SHA-256 hash.

Final audit artifact:

`deliverables/assocriskbench_p16_actual_anchor_motion_audit_20260716.json`

Audit SHA-256: `9af34ef6464a450997c9fd4a10c1d7b3b61c6dbbd8f04edb6eb45984d8b377ec`.

## 9. Scientific decision

The actual-anchor representation correction is valid and materially improves sequence-disjoint predictive signal. However, the current P16 deployment family is closed on the frozen four-sequence training bank.

Reasons:

1. Appearance competition provides useful open-set evidence but fails on visually similar different identities.
2. Actual-anchor motion materially improves LOSO correlation and HOTA sign AUC.
3. The non-nested top-one result is strongly positive in aggregate but contains sequence-negative and catastrophic failures.
4. Strict nested motion veto authorizes no outer fold.
5. Stable seed agreement does not imply cross-sequence correctness.
6. A constrained top-k fallback eliminates catastrophes but still leaves negative worst-sequence utility.
7. Continuing threshold or gate sweeps on the same 225 events would optimize against the frozen evaluation structure rather than solve the domain-shift problem.

Decision:

- Keep the final policy at `no_op`.
- Create no locked manifest.
- Read no additional locked labels.
- Keep all 156 remaining locked directional rows unread.
- Stop the current gate, veto, and fallback family on frozen P15/P16.

## 10. Next research stage

The next stage must expand independent training domains rather than tune this frozen bank.

Priority design:

1. Generate a broader actual-anchor counterfactual bank from additional MOT sequences or independent temporal/domain splits.
2. Preserve the actual source and row-conditioned receiver-family transaction semantics.
3. Add event-local open-set negatives with visually similar third-party identities.
4. Train a hierarchical mechanism model with separate appearance, motion, and executor-remap random effects.
5. Evaluate with sequence-disjoint outer folds and domain-level lower confidence bounds.
6. Require nonnegative worst-domain utility and stable candidate identity before any future locked manifest is considered.

The research contribution from this stage is not a deployable edit rule. It is the identification and validation of actual-anchor motion transfer as a useful representation, together with a rigorous demonstration that representation gain alone does not solve sequence-level selection risk under the current sample size.
