# GitHub Reader Guide

This file is the shortest high-signal guide for a human reviewer or GitHub-connected GPT that needs to understand the repository **without any chat history**.

## 1. What this repository is actually about

This repository contains several historical FM-Track / FA-MOT research directions, but the current paper-oriented tracking question has already been narrowed.

The important current question is not:

- "how do I train the old generic FM-Track model?"

The important current question is:

- "under a clean canonical carrier, can a local learned operator improve tracking without changing the host into something else?"

That carrier was fixed to `official_bytetrack`.

## 2. Current high-level project state

As of `2026-03-28`:

- canonical paper carrier: `official_bytetrack`
- transfer carrier: `botsort_base`
- specialist reference carrier: `strongsort_base`
- strongest internal positive evidence: `base_reid_da + set_predictor_v2`
- official ByteTrack learned `pre-Hungarian` line: stopped
- current active redesign direction: `post-host one-edit`
- latest best learned offline family on the active direction: `hierarchical post-host one-edit`

This matters because there are old code paths in the repo that are still relevant historically but are **not** the current mainline.

## 3. Shortest reading order

If you want the minimum complete context, read these files in order:

1. `docs/github_reader_guide.md`
2. `docs/experiment_index.md`
3. `outputs/experiment_registry.csv`
4. `outputs/cross_host_carrier_uniqueness_audit_mot17_frcnn_valhalf_20260327_030500/report.md`
5. `outputs/cross_host_baseline_defect_audit_mot17_frcnn_valhalf_20260327_104500/report.md`
6. `outputs/legacy_module_forensic_audit_20260327_161709/report.md`
7. `outputs/official_bytetrack_bridgecommit_smoke_decision_20260327.md`
8. `outputs/official_bytetrack_posthost_one_edit_oracle_decision_20260327/report.md`
9. `outputs/official_bytetrack_posthost_one_edit_offline_smoke_decision_20260327/report.md`
10. `outputs/official_bytetrack_posthost_one_edit_hierarchical_smoke_decision_20260328/report.md`

## 4. What was already decided

These decisions are fixed unless a later report explicitly overturns them:

- `official_bytetrack` is the canonical carrier.
- `botsort_base` is for transfer evidence, not the main paper baseline.
- `strongsort_base` is a specialist reference, not the main carrier.
- The learned `pre-Hungarian` official ByteTrack line is stop-gated.
- The repository should not spend more budget sweeping sparse-edit / bridge variants on that old contract.
- The current active contract is `post-host one-edit`.

## 5. What the current active line means

The current active line is **not** a whole-tracker rewrite.

It means:

- keep the `official_bytetrack` host
- let the host produce its first-stage local matches
- then allow one conservative local edit inside a hard cluster

That is why the current code and reports focus on:

- oracle post-host one-edit evaluation
- post-host one-edit dataset construction
- offline learned one-edit scorers
- hierarchical decomposition of `keep/edit -> defer/swap -> candidate rank`

## 6. Current code entrypoints

### Runtime / paired evaluation

Start here for the current runnable official ByteTrack path:

- `scripts/run_official_bytetrack_local_conflict_halfval_pair.py`
- `scripts/run_official_bytetrack_shared_detection_pair_core.py`
- `third_party/ByteTrack/yolox/tracker/byte_tracker_local_conflict.py`
- `third_party/ByteTrack/tools/track.py`
- `third_party/ByteTrack/yolox/evaluators/mot_evaluator.py`
- `third_party/ByteTrack/exps/example/mot/yolox_x_mix_det_valhalf.py`

### Current learned post-host path

Start here for the current offline learned redesign:

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

Read these four files:

1. `docs/experiment_index.md`
2. `outputs/official_bytetrack_bridgecommit_smoke_decision_20260327.md`
3. `outputs/official_bytetrack_posthost_one_edit_oracle_decision_20260327/report.md`
4. `outputs/official_bytetrack_posthost_one_edit_hierarchical_smoke_decision_20260328/report.md`

That is the shortest path to understanding:

- what failed
- what was stopped
- what changed
- what is active now
