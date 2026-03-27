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

That list is the current "minimal complete context" path.

## 2. Current top-level conclusions

These are the current repository-wide conclusions as of the latest indexed experiments:

- Canonical paper carrier: `official_bytetrack`
- Test-oriented transfer carrier: `botsort_base`
- Specialist-only reference carrier: `strongsort_base`
- Strongest internal positive line: `base_reid_da + set_predictor_v2`
- Current official ByteTrack learned pre-Hungarian line: stop-gated

The important nuance is:

- learned local operators are not globally disproven across all hosts
- but the current `set_predictor_v2` family has not yet produced executable online commits under the frozen `official_bytetrack` pre-Hungarian partial-commit contract

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

### Q6. Were the older frequency and Laplace idea families actually untested?

Read:

- `outputs/legacy_module_forensic_audit_20260327_161709/report.md`
- `outputs/legacy_module_forensic_audit_20260327_161709/summary.csv`
- `outputs/legacy_module_forensic_audit_20260327_161709/family_runs.csv`

Answer:

- no
- `frequency` was run and failed first through optimization instability, then through semantic collapse
- `laplace` had a real positive proxy regime, but its learned gate version regressed

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

Current state:

- diagnosis complete enough to justify stop-gating the current learned family
- not enough evidence yet for a positive learned plugin under the strict official contract

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

### Phase D. Official ByteTrack bridge-commit redesign

Representative records:

- `outputs/official_bytetrack_bridgecommit_smoke_dataset_20260327_1/summary.csv`
- `outputs/official_bytetrack_bridgecommit_smoke_train_20260327_1/summary.csv`
- `outputs/official_bytetrack_bridgecommit_smoke_train_20260327_2/summary.csv`
- `outputs/official_bytetrack_bridgecommit_smoke_train_20260327_3/summary.csv`
- `outputs/official_bytetrack_bridgecommit_smoke_decision_20260327.md`

Outcome:

- dataset target density became healthy enough
- gate stopped being totally dead
- assignment still failed to produce executable commits
- line stop-gated before expensive full paired evaluation

### Phase E. Legacy idea forensic audit

Representative records:

- `outputs/legacy_module_forensic_audit_20260327_161709/report.md`

Outcome:

- `frequency` and `laplace` were not "forgotten"
- their failure modes and partial successes are now explicitly documented

## 6. Current code paths for the latest official-ByteTrack redesign

If the reader wants the latest implementation that led to the current stop decision, start here:

- `scripts/build_local_conflict_set_predictor_dataset.py`
- `scripts/train_local_conflict_set_predictor.py`
- `scripts/run_official_bytetrack_local_conflict_stage1_trainhalf.py`
- `scripts/queue_official_bytetrack_possampler_followup.py`
- `models/local_conflict_set_predictor.py`
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
