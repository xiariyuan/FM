# Experiment Index

This document is the main navigation page for reading the repository's experiment history, diagnosis reports, and current conclusions without scanning the whole `outputs/` tree manually.

The intended reader is either:

- a human reviewer who wants the shortest path to the main evidence, or
- a GitHub-connected GPT that needs to reconstruct project context from repository contents alone.

## 1. Start here

If the reader has no prior context, read these files in this order:

1. `outputs/experiment_registry.csv`
2. `docs/experiment_index.md`
3. `outputs/cross_host_carrier_uniqueness_audit_mot17_frcnn_valhalf_20260327_030500/report.md`
4. `outputs/cross_host_baseline_defect_audit_mot17_frcnn_valhalf_20260327_104500/report.md`
5. `outputs/legacy_module_forensic_audit_20260327_161709/report.md`
6. `outputs/local_conflict_set_predictor_large_base_stable_20260325_023500/summary.csv`
7. `outputs/official_bytetrack_bridgecommit_smoke_decision_20260327.md`
8. `outputs/official_bytetrack_posthost_one_edit_oracle_decision_20260327/report.md`
9. `outputs/official_bytetrack_posthost_one_edit_offline_smoke_decision_20260327/report.md`
10. `outputs/official_bytetrack_posthost_one_edit_hierarchical_smoke_decision_20260328/report.md`
11. `outputs/official_bytetrack_posthost_one_edit_hierarchical_followup_decision_20260328/report.md`
12. `outputs/official_bytetrack_posthost_one_edit_hierarchical_stop_decision_20260328/report.md`
13. `outputs/official_bytetrack_posthost_one_edit_oracle_defer_only_decision_20260328/report.md`
14. `outputs/official_bytetrack_posthost_one_edit_rule_decision_20260329/report.md`

That list is the current "minimal complete context" path.

## 2. Current top-level conclusions

These are the current repository-wide conclusions as of the latest indexed experiments:

- Canonical paper carrier: `official_bytetrack`
- Test-oriented transfer carrier: `botsort_base`
- Specialist-only reference carrier: `strongsort_base`
- Strongest internal positive line: `base_reid_da + set_predictor_v2`
- Current official ByteTrack learned pre-Hungarian line: stop-gated
- Current official ByteTrack post-host oracle ceiling: executable and globally positive on HOTA / AssA / IDF1, but not yet switch-safe
- Current official ByteTrack strongest learned post-host family before stop-gate: hierarchical one-edit scorer
- Current official ByteTrack bounded utility-aware rerun: safe but too sparse, and not better than the earlier tiny learned non-zero point
- Current official ByteTrack learned post-host hierarchical family: stop-gated
- Current official ByteTrack defer-only oracle decomposition: materially positive on HOTA / AssA / IDF1, but even less switch-safe than the full mixed oracle
- Current official ByteTrack defer-only learned replacement line: rejected
- Current official ByteTrack legal rule-controller reference: small positive on HOTA / AssA / IDF1, but not strong enough for hidden-test submission

The important nuance is:

- learned local operators are not globally disproven across all hosts
- but the current `set_predictor_v2` family has not yet produced executable online commits under the frozen `official_bytetrack` pre-Hungarian partial-commit contract
- after changing the contract to a post-host one-edit oracle, executable local correction headroom does appear
- the hierarchical offline learner was the strongest learned family under that changed contract
- the bounded utility-aware rerun then showed the learned line can be made safe, but only by becoming too sparse
- so the learned post-host hierarchical family is now stop-gated rather than left open for more sweeps
- the later defer-only oracle decomposition showed that narrowing to pure defer is not the safe simplification either
- the later rule-based controller proved that a legal non-zero positive point exists, but only as a small bounded gain

## 3. Question-oriented navigation

This section is the fastest route for GPT-style reading.

### Q1. Which baseline should be treated as the main paper baseline?

Read:

- `outputs/cross_host_carrier_uniqueness_audit_mot17_frcnn_valhalf_20260327_030500/report.md`
- `outputs/cross_host_carrier_uniqueness_audit_mot17_frcnn_valhalf_20260327_030500/summary.csv`

Answer:

- `official_bytetrack` is the canonical paper carrier.
- `botsort_base` is the transfer carrier.
- `strongsort_base` is a specialist reference, not the main carrier.

### Q2. What defect is each clean baseline best understood as having?

Read:

- `outputs/cross_host_baseline_defect_audit_mot17_frcnn_valhalf_20260327_104500/report.md`
- `outputs/cross_host_baseline_defect_audit_mot17_frcnn_valhalf_20260327_104500/summary.csv`

Answer:

- `official_bytetrack`: crowded local-association failure plus large-component coverage gap
- `botsort_base`: stronger on hard crowded slices, but pays switch instability on official-favorable slices
- `strongsort_base`: broader coverage / detection deficit rather than pure local ranking weakness

### Q3. Did the learned `set_predictor_v2` idea ever work anywhere?

Read:

- `outputs/local_conflict_set_predictor_large_base_stable_20260325_023500/summary.csv`
- `outputs/local_conflict_set_predictor_large_base_stable_20260325_023500/02_proxy_eval/result.csv`
- `outputs/local_conflict_set_predictor_large_base_stable_20260325_023500/03_full_eval_md2_mm2/result.csv`

Answer:

- yes, on the internal `base_reid_da` host
- this is the strongest positive evidence for the operator direction
- but it is not the clean canonical paper carrier

Key recorded numbers:

- `proxy0213`: `HOTA=53.118`, `AssA=44.577`, `IDF1=58.73`, `MOTA=73.437`, `IDSW=811`
- `full md2/mm2`: `HOTA=63.257`, `AssA=60.191`, `IDF1=72.128`, `MOTA=76.055`, `IDSW=1481`

### Q4. What happened when the idea was moved onto official ByteTrack?

Read:

- `outputs/official_bytetrack_stage1_largecomp4_sparseedit_posboost_lc4_20260327_015600/summary.csv`
- `outputs/official_bytetrack_possampler_followup_queue_20260327_103300/02_possampler8_retry/summary.csv`
- `outputs/official_bytetrack_bridgecommit_smoke_decision_20260327.md`

Answer:

- the sparse-edit line failed as an exact online no-op
- the oversample retry also remained an exact no-op
- the later bridge-commit redesign fixed target density but still failed to produce executable commits
- therefore the official ByteTrack learned line is currently stop-gated

### Q5. What is the cleanest summary of the current stop decision?

Read:

- `outputs/official_bytetrack_bridgecommit_smoke_decision_20260327.md`

Answer:

- do not launch a strict full official paired run for the current bridge family
- the teacher is dense enough
- the gate can open
- but the assignment head still does not produce executable bridge commits under the frozen official ByteTrack runtime contract

### Q6. What happened after the contract was changed to post-host one-edit?

Read:

- `outputs/official_bytetrack_posthost_one_edit_oracle_decision_20260327/report.md`
- `outputs/official_bytetrack_posthost_one_edit_oracle_halfval_20260327_215036/summary.csv`
- `outputs/official_bytetrack_posthost_one_edit_oracle_halfval_20260327_215036/result.csv`

Answer:

- the line is no longer an execution-level no-op
- the post-host oracle performs real edits on the official carrier
- global half-val paired deltas are positive on `HOTA`, `AssA`, and `IDF1`
- most profitable edits are `swap` or `defer`, not additive bridge commits
- the direction has real headroom, but still needs switch-risk control on official-favorable slices such as `MOT17-02`

### Q7. Were the older frequency and Laplace idea families actually untested?

Read:

- `outputs/legacy_module_forensic_audit_20260327_161709/report.md`
- `outputs/legacy_module_forensic_audit_20260327_161709/summary.csv`
- `outputs/legacy_module_forensic_audit_20260327_161709/family_runs.csv`

Answer:

- no
- `frequency` was run and failed first through optimization instability, then through semantic collapse
- `laplace` had a real positive proxy regime, but its learned gate version regressed

### Q8. What did the first learned post-host offline smoke show?

Read:

- `outputs/official_bytetrack_posthost_one_edit_dataset_20260327_234041/summary.csv`
- `outputs/official_bytetrack_posthost_one_edit_scorer_smoke_20260327_234041/summary.csv`
- `outputs/official_bytetrack_posthost_one_edit_scorer_swapfocus_20260327_234816/summary.csv`
- `outputs/official_bytetrack_posthost_one_edit_offline_smoke_decision_20260327/report.md`
- `outputs/official_bytetrack_posthost_one_edit_hierarchical_smoke_20260328_000238/summary.csv`
- `outputs/official_bytetrack_posthost_one_edit_hierarchical_smoke_decision_20260328/report.md`

Answer:

- the post-host action dataset is dense enough and dominated by `defer`, with a small `swap` tail
- a flat one-stage scorer already learns `keep` vs `edit`
- the base scorer collapses rare `swap` to `defer`
- aggressive swap reweighting recovers `swap` recall but damages defer precision and exact candidate ranking
- the next learned family should therefore be hierarchical, not a single flat softmax over all candidates

### Q9. Did the hierarchical post-host learner actually improve over the flat learned baselines?

Read:

- `outputs/official_bytetrack_posthost_one_edit_hierarchical_smoke_20260328_000238/summary.csv`
- `outputs/official_bytetrack_posthost_one_edit_hierarchical_smoke_decision_20260328/report.md`

Answer:

- yes
- it is the first learned post-host family that avoids the flat tradeoff between zero-`swap` collapse and over-aggressive swap oversampling
- it reaches `val_action_type_acc = 0.7363`, `val_keep_vs_edit_acc = 1.0000`, `val_swap_action_recall = 0.7647`, and `val_exact_top1_acc = 0.3363`
- it does not yet fully solve candidate-level defer ranking, so the next step is conservative online integration rather than further flat offline sweeps

### Q10. What did the first hierarchical online follow-up cycle actually prove?

Read:

- `outputs/official_bytetrack_posthost_one_edit_hierarchical_followup_decision_20260328/report.md`
- `outputs/official_bytetrack_posthost_one_edit_hierarchical_swap_priority_halfval_smoke_20260328_102900/result.csv`
- `outputs/official_bytetrack_posthost_one_edit_hierarchical_swap_priority_halfval_smoke_keep097_20260328_110500/result.csv`
- `outputs/official_bytetrack_posthost_one_edit_hierarchical_balanced_gate_swap_20260328_131500/summary.csv`
- `outputs/official_bytetrack_posthost_one_edit_hierarchical_balanced_gate_swap_halfval_smoke_keep0988_20260328_131900/result.csv`
- `outputs/official_bytetrack_posthost_one_edit_hierarchical_balanced_gate_swap_halfval_smoke_keep097_20260328_142200/result.csv`

Answer:

- the online hierarchical line is no longer blocked by pure execution failure
- but it still does not have a useful safe operating window
- tighter gate gives exact no-op
- looser gate gives one or two executed `swap`s but still negative paired outcome
- therefore the next bottleneck is candidate utility / ranking quality, not threshold calibration alone

### Q11. What happened in the bounded utility-aware rerun, and what is the current stop decision?

Read:

- `outputs/official_bytetrack_posthost_one_edit_dataset_utilityaware_20260328_212500/summary.csv`
- `outputs/official_bytetrack_posthost_one_edit_hierarchical_utilityaware_20260328_215800/summary.csv`
- `outputs/official_bytetrack_posthost_one_edit_hierarchical_utilityaware_halfval_rerun_20260328_220500/result.csv`
- `outputs/official_bytetrack_posthost_one_edit_hierarchical_stop_decision_20260328/report.md`

Answer:

- the utility-aware redesign restored the correct defer-heavy target mix and produced a non-degenerate offline checkpoint
- the strict paired rerun was not a no-op and not a runtime bug
- but it executed only two `defer` edits, touched `MOT17-02` and `MOT17-05`, and still missed `MOT17-10` and `MOT17-13`
- global paired delta became `HOTA +0.002 / AssA +0.002 / IDF1 -0.001 / MOTA +0.005 / IDSW -2`
- this was safer than some earlier runs, but worse than the earlier `runtime_safe_zero` learned point on the main association metrics
- under the agreed "one bounded utility-aware chance" rule, the learned post-host hierarchical family is now stop-gated

### Q12. Did the post-host `defer-only` oracle validate a simpler replacement line?

Read:

- `outputs/official_bytetrack_posthost_one_edit_oracle_defer_only_halfval_20260328_233113/result.csv`
- `outputs/official_bytetrack_posthost_one_edit_oracle_defer_only_halfval_20260328_233113/summary.csv`
- `outputs/official_bytetrack_posthost_one_edit_oracle_defer_only_decision_20260328/report.md`

Answer:

- it retained a large share of the oracle `HOTA / AssA / IDF1` gains:
  - `delta_HOTA = +1.238`
  - `delta_AssA = +2.803`
  - `delta_IDF1 = +2.217`
- it touched all hard slices and executed `583` pure defer edits
- but its switch cost was much worse than the full mixed oracle:
  - `delta_IDSW = +90` versus `+28` for the mixed oracle
- therefore `defer-only` is not a safe simplified replacement contract
- a new learned defer-only branch should not be opened

### Q13. Is there any test-legal post-host line that is actually non-zero and positive?

Read:

- `outputs/official_bytetrack_posthost_one_edit_rule_halfval_rerun_20260329_002100/result.csv`
- `outputs/official_bytetrack_posthost_one_edit_rule_c4_halfval_20260329_005000/result.csv`
- `outputs/official_bytetrack_posthost_one_edit_rule_decision_20260329/report.md`

Answer:

- yes
- a conservative rule-based post-host defer controller is executable without GT and gives:
  - `delta_HOTA = +0.118`
  - `delta_AssA = +0.302`
  - `delta_IDF1 = +0.060`
  - `delta_MOTA = -0.097`
  - `delta_IDSW = +5`
- this is better than the stop-gated learned utility-aware rerun on the main association metrics
- but the gain is still small
- and a looser follow-up already made the result worse
- so it should be kept as a legal reference point, not treated as the final submission line

## 4. Baseline map

This section groups the repository by baseline family and role.

### official_bytetrack

Role:

- canonical paper carrier

Why it matters:

- defines the frozen `primary-only / pre-Hungarian / conservative partial-commit + defer to host` contract
- the main paper claim must eventually stand or fall here

Core records:

- `outputs/cross_host_carrier_uniqueness_audit_mot17_frcnn_valhalf_20260327_030500/report.md`
- `outputs/cross_host_baseline_defect_audit_mot17_frcnn_valhalf_20260327_104500/report.md`
- `outputs/official_bytetrack_stage1_largecomp4_sparseedit_posboost_lc4_20260327_015600/summary.csv`
- `outputs/official_bytetrack_possampler_followup_queue_20260327_103300/02_possampler8_retry/summary.csv`
- `outputs/official_bytetrack_bridgecommit_smoke_dataset_20260327_1/summary.csv`
- `outputs/official_bytetrack_bridgecommit_smoke_train_20260327_3/summary.csv`
- `outputs/official_bytetrack_bridgecommit_smoke_decision_20260327.md`
- `outputs/official_bytetrack_posthost_one_edit_oracle_decision_20260327/report.md`
- `outputs/official_bytetrack_posthost_one_edit_oracle_halfval_20260327_215036/summary.csv`
- `outputs/official_bytetrack_posthost_one_edit_dataset_20260327_234041/summary.csv`
- `outputs/official_bytetrack_posthost_one_edit_scorer_smoke_20260327_234041/summary.csv`
- `outputs/official_bytetrack_posthost_one_edit_scorer_swapfocus_20260327_234816/summary.csv`
- `outputs/official_bytetrack_posthost_one_edit_offline_smoke_decision_20260327/report.md`
- `outputs/official_bytetrack_posthost_one_edit_hierarchical_smoke_20260328_000238/summary.csv`
- `outputs/official_bytetrack_posthost_one_edit_hierarchical_smoke_decision_20260328/report.md`

Current state:

- diagnosis complete enough to justify stop-gating the current learned family
- a post-host oracle ceiling confirms executable edit headroom under a changed contract
- the flat one-stage scorer is not the right learned architecture
- the hierarchical one-edit scorer was the best learned offline family on this contract
- the bounded utility-aware rerun proved the learned line can become safe but too sparse
- the current learned post-host hierarchical family is therefore stopped
- the later defer-only oracle decomposition proved that narrowing the action space to pure defer is not the safe replacement either
- the later conservative rule controller proved that a legal non-zero positive point exists, but only as a small bounded gain

### botsort_base

Role:

- test-oriented transfer carrier

Why it matters:

- strongest clean carrier on harder official failure slices such as `MOT17-05/10/13`
- useful for asking whether a method transfers to a stronger host

Core records:

- `outputs/cross_host_carrier_uniqueness_audit_mot17_frcnn_valhalf_20260327_030500/report.md`
- `outputs/cross_host_baseline_defect_audit_mot17_frcnn_valhalf_20260327_104500/report.md`

Current state:

- chosen as transfer carrier, not canonical paper baseline

### strongsort_base

Role:

- specialist-only reference

Why it matters:

- useful as a counterexample carrier with low-switch strengths on some slices
- not a clean main carrier for this project's current contract

Core records:

- `outputs/cross_host_carrier_uniqueness_audit_mot17_frcnn_valhalf_20260327_030500/report.md`
- `outputs/cross_host_baseline_defect_audit_mot17_frcnn_valhalf_20260327_104500/report.md`

Current state:

- useful reference, not promoted to paper mainline

### base_reid_da internal host

Role:

- internal positive reference only

Why it matters:

- strongest positive result for `set_predictor_v2`
- proves the direction can be useful under at least one host family

Core records:

- `outputs/local_conflict_set_predictor_large_base_stable_20260325_023500/summary.csv`
- `outputs/local_conflict_set_predictor_large_base_stable_20260325_023500/02_proxy_eval/result.csv`
- `outputs/local_conflict_set_predictor_large_base_stable_20260325_023500/03_full_eval_md2_mm2/result.csv`

Current state:

- evidence of directional value
- not the clean paper carrier

## 5. Timeline of the project logic

This is the shortest temporal map of how the current state was reached.

### Phase A. Internal learned local conflict line

Representative records:

- `outputs/local_conflict_commit_large_base_20260324_222409/summary.csv`
- `outputs/local_conflict_set_predictor_large_base_stable_20260325_023500/summary.csv`

Outcome:

- internal host work produced meaningful positive evidence

### Phase B. Baseline and carrier selection

Representative records:

- `outputs/cross_host_carrier_uniqueness_audit_mot17_frcnn_valhalf_20260327_030500/report.md`
- `outputs/cross_host_baseline_defect_audit_mot17_frcnn_valhalf_20260327_104500/report.md`

Outcome:

- `official_bytetrack` fixed as canonical
- `botsort_base` fixed as transfer
- `strongsort_base` fixed as specialist reference

### Phase C. Official ByteTrack sparse-edit attempts

Representative records:

- `outputs/official_bytetrack_stage1_largecomp4_sparseedit_posboost_lc4_20260327_015600/summary.csv`
- `outputs/official_bytetrack_possampler_followup_queue_20260327_103300/02_possampler8_retry/summary.csv`

Outcome:

- exact online no-op
- oversampling did not rescue execution-level behavior

### Phase D. Official ByteTrack bridge smoke stop decision

Representative records:

- `outputs/official_bytetrack_bridgecommit_smoke_dataset_20260327_1/summary.csv`
- `outputs/official_bytetrack_bridgecommit_smoke_train_20260327_3/summary.csv`
- `outputs/official_bytetrack_bridgecommit_smoke_decision_20260327.md`

Outcome:

- the teacher became dense enough
- the gate could open
- executable commits still stayed at zero
- the learned pre-Hungarian line was stopped under the frozen contract

### Phase E. Post-host one-edit oracle ceiling

Representative records:

- `outputs/official_bytetrack_posthost_one_edit_oracle_halfval_20260327_215036/summary.csv`
- `outputs/official_bytetrack_posthost_one_edit_oracle_halfval_20260327_215036/result.csv`
- `outputs/official_bytetrack_posthost_one_edit_oracle_decision_20260327/report.md`

Outcome:

- contract change produced real executable edits
- global paired `HOTA / AssA / IDF1` moved positive
- dominant profitable action type is `swap` or `defer`, not `add`
- the new direction still needs switch-risk control before any learned successor is justified

### Phase F. Post-host offline learned smoke

Representative records:

- `outputs/official_bytetrack_posthost_one_edit_dataset_20260327_234041/summary.csv`
- `outputs/official_bytetrack_posthost_one_edit_scorer_smoke_20260327_234041/summary.csv`
- `outputs/official_bytetrack_posthost_one_edit_scorer_swapfocus_20260327_234816/summary.csv`
- `outputs/official_bytetrack_posthost_one_edit_offline_smoke_decision_20260327/report.md`

Outcome:

- the coarse `keep` vs `edit` decision is learnable
- the dominant `defer` action family is learnable
- rare `swap` actions require explicit treatment
- a flat one-stage softmax over all candidates is not yet adequate

### Phase G. Hierarchical post-host learned smoke

Representative records:

- `outputs/official_bytetrack_posthost_one_edit_hierarchical_smoke_20260328_000238/summary.csv`
- `outputs/official_bytetrack_posthost_one_edit_hierarchical_smoke_decision_20260328/report.md`

Outcome:

- hierarchical decomposition resolves the flat-family tradeoff much better
- `keep` vs `edit` is preserved perfectly on the validation split
- useful `swap` coverage is retained without collapsing exact action quality as badly as the flat swap-focused scorer
- the next step is conservative online integration smoke

### Phase H. Hierarchical online follow-up and threshold exhaustion

Representative records:

- `outputs/official_bytetrack_posthost_one_edit_hierarchical_followup_decision_20260328/report.md`
- `outputs/official_bytetrack_posthost_one_edit_hierarchical_swap_priority_halfval_smoke_20260328_102900/result.csv`
- `outputs/official_bytetrack_posthost_one_edit_hierarchical_swap_priority_halfval_smoke_keep097_20260328_110500/result.csv`
- `outputs/official_bytetrack_posthost_one_edit_hierarchical_balanced_gate_swap_20260328_131500/summary.csv`
- `outputs/official_bytetrack_posthost_one_edit_hierarchical_balanced_gate_swap_halfval_smoke_keep0988_20260328_131900/result.csv`
- `outputs/official_bytetrack_posthost_one_edit_hierarchical_balanced_gate_swap_halfval_smoke_keep097_20260328_142200/result.csv`

Outcome:

- online execution is now possible, so the line is no longer a pure wiring no-op
- however the main tradeoff became exact no-op versus sparse-but-negative execution
- the hard official slices are still not being improved in a useful way
- the next redesign question is candidate utility / ranking quality, not another broad threshold sweep

### Phase I. Bounded utility-aware rerun and stop gate

Representative records:

- `outputs/official_bytetrack_posthost_one_edit_dataset_utilityaware_20260328_212500/summary.csv`
- `outputs/official_bytetrack_posthost_one_edit_hierarchical_utilityaware_20260328_215800/summary.csv`
- `outputs/official_bytetrack_posthost_one_edit_hierarchical_utilityaware_halfval_rerun_20260328_220500/result.csv`
- `outputs/official_bytetrack_posthost_one_edit_hierarchical_stop_decision_20260328/report.md`

Outcome:

- the redesign restored an oracle-consistent defer-heavy target mix
- the offline checkpoint became non-degenerate on utility-aware metrics
- the online rerun was no longer a no-op and executed two real `defer` edits
- but it still did not beat the earlier `runtime_safe_zero` learned point
- therefore the learned post-host hierarchical family is now stop-gated

### Phase J. Defer-only oracle decomposition

Representative records:

- `outputs/official_bytetrack_posthost_one_edit_oracle_defer_only_halfval_20260328_233113/result.csv`
- `outputs/official_bytetrack_posthost_one_edit_oracle_defer_only_halfval_20260328_233113/summary.csv`
- `outputs/official_bytetrack_posthost_one_edit_oracle_defer_only_decision_20260328/report.md`

Outcome:

- the narrow `defer-only` post-host oracle kept a large fraction of the association-quality headroom
- but it exploded switch cost to `delta_IDSW = +90`
- therefore a simpler learned defer-only replacement line is not justified

### Phase K. Test-legal rule-controller probe

Representative records:

- `outputs/official_bytetrack_posthost_one_edit_rule_halfval_rerun_20260329_002100/result.csv`
- `outputs/official_bytetrack_posthost_one_edit_rule_c4_halfval_20260329_005000/result.csv`
- `outputs/official_bytetrack_posthost_one_edit_rule_decision_20260329/report.md`

Outcome:

- a conservative rule-based post-host defer controller is the first test-legal official-carrier line in this repo that is both executable and globally positive on `HOTA / AssA / IDF1`
- the best legal point is:
  - `delta_HOTA = +0.118`
  - `delta_AssA = +0.302`
  - `delta_IDF1 = +0.060`
  - `delta_MOTA = -0.097`
  - `delta_IDSW = +5`
- a looser follow-up is worse, so the current legal rule point should be treated as a bounded reference rather than an open threshold-sweep branch

### Phase L. Legacy idea forensic audit

Representative records:

- `outputs/legacy_module_forensic_audit_20260327_161709/report.md`

Outcome:

- `frequency` and `laplace` were not "forgotten"
- their failure modes and partial successes are now explicitly documented

## 6. Current code paths

This section distinguishes the current mainline from archived-but-important historical paths.

### Latest official ByteTrack post-host path

If the reader wants the latest learned official ByteTrack implementation path that was actually run, start here:

- `scripts/run_official_bytetrack_local_conflict_halfval_pair.py`
- `scripts/run_official_bytetrack_shared_detection_pair_core.py`
- `third_party/ByteTrack/yolox/tracker/byte_tracker_local_conflict.py`
- `third_party/ByteTrack/tools/track.py`
- `third_party/ByteTrack/yolox/evaluators/mot_evaluator.py`
- `third_party/ByteTrack/exps/example/mot/yolox_x_mix_det_valhalf.py`
- `scripts/build_posthost_one_edit_dataset.py`
- `scripts/train_posthost_one_edit_hierarchical.py`
- `models/posthost_one_edit_hierarchical.py`
- `models/posthost_one_edit_scorer.py`

Interpretation:

- the current runtime carrier remains `official_bytetrack`
- the stopped `pre-Hungarian` learned line is no longer the active implementation focus
- the latest learned continuation that was actually tested is the `post-host one-edit` path
- the latest learned family on that path is hierarchical
- that learned hierarchical line is now stop-gated after one bounded utility-aware rerun

### Archived but still important official ByteTrack pre-Hungarian path

If the reader needs to understand why the earlier learned line was stop-gated, read these files:

- `scripts/build_local_conflict_set_predictor_dataset.py`
- `scripts/train_local_conflict_set_predictor.py`
- `scripts/run_official_bytetrack_local_conflict_stage1_trainhalf.py`
- `scripts/queue_official_bytetrack_possampler_followup.py`
- `models/local_conflict_set_predictor.py`
- `models/local_conflict_graph_common.py`
- `third_party/ByteTrack/yolox/tracker/byte_tracker_local_conflict.py`

## 7. Record format policy

This repository intentionally stores lightweight structured experiment evidence in git.

Included under `outputs/`:

- `summary.csv`
- `result.csv`
- `metrics.csv`
- `metrics.jsonl`
- `*.metrics.jsonl`
- `report.md`
- `summary.json`
- `sequence_cluster_summary.csv`
- `family_runs.csv`
- `experiment_registry.csv`

Intentionally excluded:

- checkpoints
- `.pth` / `.pt` / `.npz` / `.npy`
- dataset dumps
- raw runtime shards
- full logs
- packaged archives

This keeps the repository GPT-readable without pushing heavy artifacts.

## 8. How to sync new experiment records

### Stage only

```bash
scripts/git_stage_experiment_records.sh
```

### Stage, commit, and push in one command

```bash
scripts/git_sync_experiment_records.sh
```

Optional custom commit message:

```bash
scripts/git_sync_experiment_records.sh "Sync experiment records after official ByteTrack rerun"
```

## 9. What a GitHub-connected GPT should infer first

If a GPT is reading this repository fresh, the correct first-pass interpretation is:

- the project already ran multiple learned local-operator families
- the best positive result is currently internal-host, not official ByteTrack
- official ByteTrack is still the canonical contract and baseline
- the latest strict official learned line is not merely weak; it is stop-gated at execution level
- cross-host diagnosis and legacy-module forensics are part of the evidence base, not side notes

That is the correct context before proposing any next redesign.
