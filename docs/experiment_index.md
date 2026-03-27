# Experiment Index

This document is the repository entry point for reading the diagnosis-driven MOT experiments without scanning the whole `outputs/` tree manually.

## Reading order

If a GitHub-connected GPT needs the shortest path through the project, start here:

1. `outputs/experiment_registry.csv`
2. `outputs/cross_host_carrier_uniqueness_audit_mot17_frcnn_valhalf_20260327_030500/report.md`
3. `outputs/cross_host_baseline_defect_audit_mot17_frcnn_valhalf_20260327_104500/report.md`
4. `outputs/legacy_module_forensic_audit_20260327_161709/report.md`
5. `outputs/local_conflict_set_predictor_large_base_stable_20260325_023500/summary.csv`
6. `outputs/official_bytetrack_bridgecommit_smoke_decision_20260327.md`

## Current high-level status

- Canonical paper carrier: `official_bytetrack`
- Test-oriented transfer carrier: `botsort_base`
- Specialist-only reference carrier: `strongsort_base`
- Internal positive reference: `base_reid_da + set_predictor_v2`
- Current learned pre-Hungarian official-ByteTrack line: stop-gated

The current conclusion is not that learned local operators are universally invalid. The narrower conclusion is that the current `set_predictor_v2` family, under the frozen official ByteTrack pre-Hungarian partial-commit contract, has not yet produced executable online commits in a clean strict paired run.

## Key experiment groups

### 1. Cross-host carrier decision

Primary reports:

- `outputs/cross_host_carrier_uniqueness_audit_mot17_frcnn_valhalf_20260327_030500/report.md`
- `outputs/cross_host_carrier_uniqueness_audit_mot17_frcnn_valhalf_20260327_030500/summary.csv`
- `outputs/cross_host_baseline_defect_audit_mot17_frcnn_valhalf_20260327_104500/report.md`
- `outputs/cross_host_baseline_defect_audit_mot17_frcnn_valhalf_20260327_104500/summary.csv`

Main takeaways:

- `official_bytetrack` stays the canonical paper baseline because it uniquely protects official-favorable slices such as `MOT17-02-FRCNN` while also defining the clean pre-Hungarian carrier contract.
- `botsort_base` is the best clean transfer carrier on the harder official failure slices, especially `MOT17-05/10/13`.
- `strongsort_base` behaves as a specialist reference, not the main paper carrier.

### 2. Internal-host positive reference

Primary records:

- `outputs/local_conflict_set_predictor_large_base_stable_20260325_023500/summary.csv`
- `outputs/local_conflict_set_predictor_large_base_stable_20260325_023500/02_proxy_eval/result.csv`
- `outputs/local_conflict_set_predictor_large_base_stable_20260325_023500/03_full_eval_md2_mm2/result.csv`

Main takeaways:

- This is the strongest positive result for `set_predictor_v2`, but it lives on the internal `base_reid_da` carrier rather than the canonical official ByteTrack carrier.
- Recorded summary:
  - `proxy0213`: `HOTA=53.118`, `AssA=44.577`, `IDF1=58.73`, `MOTA=73.437`, `IDSW=811`
  - `full md2/mm2`: `HOTA=63.257`, `AssA=60.191`, `IDF1=72.128`, `MOTA=76.055`, `IDSW=1481`
- This experiment is the main evidence that the operator direction can be useful somewhere, but it is not the clean paper baseline.

### 3. Official ByteTrack sparse-edit line

Primary records:

- `outputs/official_bytetrack_stage1_largecomp4_sparseedit_posboost_lc4_20260327_015600/summary.csv`
- `outputs/official_bytetrack_possampler_followup_queue_20260327_103300/02_possampler8_retry/summary.csv`

Main takeaways:

- The strict official-host sparse-edit line failed as an exact online no-op.
- In both the main sparse-edit run and the oversample retry:
  - `plugin_replaced_clusters=0`
  - `plugin_matched_dets=0`
  - `delta_HOTA=0`
  - `delta_AssA=0`
  - `delta_IDF1=0`
- The oversample retry increased offline signal density but did not change execution-level behavior under the runtime contract.

### 4. Official ByteTrack bridge-commit redesign smoke

Primary records:

- `outputs/official_bytetrack_bridgecommit_smoke_dataset_20260327_1/summary.csv`
- `outputs/official_bytetrack_bridgecommit_smoke_train_20260327_1/summary.csv`
- `outputs/official_bytetrack_bridgecommit_smoke_train_20260327_2/summary.csv`
- `outputs/official_bytetrack_bridgecommit_smoke_train_20260327_3/summary.csv`
- `outputs/official_bytetrack_bridgecommit_smoke_decision_20260327.md`

Main takeaways:

- The bridge teacher fixed the worst density problem at dataset level:
  - `eligible_clusters=2657`
  - `cluster_should_intervene_bridge_clusters=81`
  - positives concentrate on hard slices `MOT17-05/10/13`
- Training smoke still hit the stop gate:
  - later epochs can open calibrated gate coverage
  - but `val_commit_precision=0` and `val_commit_recall=0` remain zero
- Result: do not spend budget on a full strict official paired run for this family without a new family-level redesign decision.

### 5. Legacy idea families

Primary records:

- `outputs/legacy_module_forensic_audit_20260327_161709/report.md`
- `outputs/legacy_module_forensic_audit_20260327_161709/summary.csv`
- `outputs/legacy_module_forensic_audit_20260327_161709/family_runs.csv`

Main takeaways:

- `frequency family`:
  - not merely unvalidated
  - failed first through optimization instability, then through identity-semantic collapse
  - strongest validated rescue attempt on MOT17 still collapsed to `HOTA=24.79`, `IDF1=18.72`, `IDSW=19693`
- `laplace family`:
  - had a genuine positive regime on `proxy0213`
  - fixed branch achieved `delta HOTA=+1.947`, `delta AssA=+2.886`, `delta IDF1=+2.436`, `delta MOTA=+0.264`, `delta IDSW=-56`
  - but the learned/trainable gate version regressed

## Key code paths

If the reader wants the latest official-ByteTrack redesign implementation, start with:

- `scripts/build_local_conflict_set_predictor_dataset.py`
- `scripts/train_local_conflict_set_predictor.py`
- `scripts/run_official_bytetrack_local_conflict_stage1_trainhalf.py`
- `scripts/queue_official_bytetrack_possampler_followup.py`
- `models/local_conflict_set_predictor.py`
- `third_party/ByteTrack/yolox/tracker/byte_tracker_local_conflict.py`

## Record sync policy

The repository intentionally stores lightweight structured experiment evidence, not full heavy artifacts.

Uploaded under `outputs/`:

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
- `.pth` / `.npz` / `.npy`
- dataset dumps
- raw runtime shards
- full logs
- packaged archives

## One-command experiment record sync

To stage, commit, and push the latest structured experiment records:

```bash
scripts/git_sync_experiment_records.sh
```

Optional custom commit message:

```bash
scripts/git_sync_experiment_records.sh "Sync experiment records after official ByteTrack rerun"
```
