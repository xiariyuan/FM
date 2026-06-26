# GitHub Reader Guide

This file is the shortest high-signal guide for a human reviewer or GitHub-connected GPT that needs to understand the repository **without any chat history**.

## ⚠️ IMPORTANT: Current Mainline Has Changed

**As of 2026-06-26, the current mainline is SPOT-Track, not the old `official_bytetrack / post-host one-edit` line.**

Read `docs/current_mainline_2026-06-26.md` first for the current state.

## 1. What this repository is actually about

This repository contains SPOT-Track: State-Protected Online Tracking under Ambiguous Association.

The current question is:

- "under a frozen detector protocol, can we learn when NOT to update tracker state under ambiguous association?"

The oracle gate has confirmed:
- Oracle 0A: 7.29% oracle recoverable ceiling (NOT runtime gain)
- Oracle 0C: 43.28% fixable (moderate, partial inline GT)
- Oracle 0E: PROVISIONAL → SPOT_PROVISIONAL (requires paired eval)

## 2. Current high-level project state

As of `2026-06-26`:

- **Current mainline:** SPOT-Track
- **Oracle Gate:** PROVISIONAL → SPOT_PROVISIONAL
- **Runtime patch allowed:** NO
- **Main novelty:** P4 ADG-freeze / State Protection (candidate; needs paired eval)
- **Support module:** PCC (support only; not runtime gain)
- **P5 delayed commitment:** SKIP

Historical (2026-03) context:
- old canonical carrier: `official_bytetrack`
- old transfer carrier: `botsort_base`
- old specialist reference: `strongsort_base`
- old learned pre-Hungarian line: stopped
- old post-host one-edit: stop-gated

## 3. Shortest reading order

If you want the minimum complete **current SPOT** context, read these files in order:

1. `docs/current_mainline_2026-06-26.md`
2. `outputs/oracle_gate/decision.md`
3. `outputs/oracle_gate/summary.csv`
4. `docs/oracle_gate_recap_2026-06-25.md`
5. `plan/spot_implementation_plan.md`
6. `scripts/spot_oracle/run_oracle_state_protection.py`
7. `scripts/spot_oracle/run_oracle_cost_rerank_inline.py`
8. `scripts/spot_oracle/run_joint_oracle.py`
9. `external/BoT-SORT-main/tracker/bot_sort.py`
10. `external/BoT-SORT-main/tools/track.py`

For the old 2026-03 ByteTrack/post-host history, read `docs/experiment_index.md` after the current SPOT files above.

## 4. What was already decided

These decisions are fixed unless a later report explicitly overturns them:

- `official_bytetrack` is the canonical carrier.
- `botsort_base` is for transfer evidence, not the main paper baseline.
- `strongsort_base` is a specialist reference, not the main carrier.
- The learned `pre-Hungarian` official ByteTrack line is stop-gated.
- The repository should not spend more budget sweeping sparse-edit / bridge variants on that old contract.
- The `post-host one-edit` oracle contract is validated.
- The current learned hierarchical line on that contract is now stop-gated after one bounded utility-aware rerun.
- The later `defer-only` oracle decomposition shows that pure defer is not a safe simplified replacement contract.
- The later rule-based post-host controller gives the first test-legal non-zero positive point on the canonical carrier, but only as a small bounded gain.

## 5. What the post-host contract means

The post-host contract is **not** a whole-tracker rewrite.

It means:

- keep the `official_bytetrack` host
- let the host produce its first-stage local matches
- then allow one conservative local edit inside a hard cluster

That is why the current code and reports focus on:

- oracle post-host one-edit evaluation
- post-host one-edit dataset construction
- offline learned one-edit scorers
- hierarchical decomposition of `keep/edit -> defer/swap -> candidate rank`

The important current nuance is:

- this contract still matters because the oracle ceiling is real
- but the latest learned hierarchical family under this contract is no longer an open tuning line
- and the simplest narrowed replacement hypothesis, `defer-only`, also does not justify a new learned line

## 6. Current code entrypoints

### Current SPOT runtime / paired evaluation

SPOT runtime is **not yet unlocked**. The next implementation target is a minimal BoT-SORT `freeze_app` experiment with `spot_enable=0` parity first:

- `external/BoT-SORT-main/tracker/bot_sort.py`
- `external/BoT-SORT-main/tools/track.py`
- `scripts/eval_botsort_halfval_trackeval.py`
- `plan/spot_implementation_plan.md`

Historical ByteTrack paired-eval scripts are references only, not the active runtime path:

- `scripts/run_official_bytetrack_local_conflict_halfval_pair.py`
- `scripts/run_official_bytetrack_shared_detection_pair_core.py`

### Historical learned post-host path

Read this only for the old 2026-03 offline learned redesign:

- `scripts/build_posthost_one_edit_dataset.py`
- `scripts/train_posthost_one_edit_scorer.py`
- `scripts/train_posthost_one_edit_hierarchical.py`
- `models/posthost_one_edit_scorer.py`
- `models/posthost_one_edit_hierarchical.py`

### Historical stopped official pre-Hungarian path

Read these only to understand why the earlier line was stopped:

- `scripts/build_local_conflict_set_predictor_dataset.py`
- `scripts/train_local_conflict_set_predictor.py`
- `scripts/run_official_bytetrack_local_conflict_stage1_trainhalf.py`
- `scripts/queue_official_bytetrack_possampler_followup.py`
- `models/local_conflict_set_predictor.py`
- `models/local_conflict_graph_common.py`

## 7. Most important structured experiment records

### Canonical carrier / defect diagnosis

- `outputs/cross_host_carrier_uniqueness_audit_mot17_frcnn_valhalf_20260327_030500/report.md`
- `outputs/cross_host_baseline_defect_audit_mot17_frcnn_valhalf_20260327_104500/report.md`

### Old official ByteTrack line stop decision

- `outputs/official_bytetrack_bridgecommit_smoke_decision_20260327.md`

### Active contract oracle headroom

- `outputs/official_bytetrack_posthost_one_edit_oracle_decision_20260327/report.md`
- `outputs/official_bytetrack_posthost_one_edit_oracle_halfval_20260327_215036/summary.csv`
- `outputs/official_bytetrack_posthost_one_edit_oracle_halfval_20260327_215036/result.csv`

### Active contract learned offline results

- `outputs/official_bytetrack_posthost_one_edit_dataset_20260327_234041/summary.csv`
- `outputs/official_bytetrack_posthost_one_edit_scorer_smoke_20260327_234041/summary.csv`
- `outputs/official_bytetrack_posthost_one_edit_scorer_swapfocus_20260327_234816/summary.csv`
- `outputs/official_bytetrack_posthost_one_edit_offline_smoke_decision_20260327/report.md`
- `outputs/official_bytetrack_posthost_one_edit_hierarchical_smoke_20260328_000238/summary.csv`
- `outputs/official_bytetrack_posthost_one_edit_hierarchical_smoke_decision_20260328/report.md`
- `outputs/official_bytetrack_posthost_one_edit_hierarchical_followup_decision_20260328/report.md`
- `outputs/official_bytetrack_posthost_one_edit_dataset_utilityaware_20260328_212500/summary.csv`
- `outputs/official_bytetrack_posthost_one_edit_hierarchical_utilityaware_20260328_215800/summary.csv`
- `outputs/official_bytetrack_posthost_one_edit_hierarchical_utilityaware_halfval_rerun_20260328_220500/result.csv`
- `outputs/official_bytetrack_posthost_one_edit_hierarchical_stop_decision_20260328/report.md`
- `outputs/official_bytetrack_posthost_one_edit_oracle_defer_only_halfval_20260328_233113/result.csv`
- `outputs/official_bytetrack_posthost_one_edit_oracle_defer_only_decision_20260328/report.md`
- `outputs/official_bytetrack_posthost_one_edit_rule_halfval_rerun_20260329_002100/result.csv`
- `outputs/official_bytetrack_posthost_one_edit_rule_c4_halfval_20260329_005000/result.csv`
- `outputs/official_bytetrack_posthost_one_edit_rule_decision_20260329/report.md`

## 8. What is intentionally missing from git

This repo is designed to be GPT-readable, not to vendor every heavy artifact.

So GitHub readers should expect that these may be referenced but not tracked:

- large checkpoints
- detector weights
- local dataset roots
- raw logs
- large intermediate dumps

That is normal.

The project stores the lightweight evidence that explains what happened:

- `summary.csv`
- `result.csv`
- `metrics.jsonl`
- `report.md`
- `outputs/experiment_registry.csv`

## 9. If you only have one minute

Read these first for the **current SPOT state**:

1. `docs/current_mainline_2026-06-26.md`
2. `outputs/oracle_gate/decision.md`
3. `outputs/oracle_gate/summary.csv`
4. `docs/oracle_gate_recap_2026-06-25.md`
5. `plan/spot_implementation_plan.md`

That is the shortest path to understanding:

- current mainline is SPOT-Track, not old ByteTrack/post-host
- oracle gate is PROVISIONAL
- runtime patches are not allowed yet
- oracle ceiling is not runtime gain
- next step is minimal paired eval, not opening PCC/P5
