# M23-65 result — MOT20 frozen-checkpoint representation gate

## Decision

**FAIL_MOT20_REPRESENTATION_GATE**. The failure is a representation/target-transfer score-collapse result, not an input-reverification or scope failure. Frozen candidate-present diagnostic recall is 1.0 for all four sequences, so candidate presence is not the primary failure signal; boundary scores remain near the target base rates, especially MOT20-03/05.

## Frozen inputs and commands

- Git HEAD: `a04dfa0012114c1663ea1cb543158bc4d1d975a5`
- Prereg SHA: `1f6e6beb7333a8c15f8eeac4959869f58ea77e0afcc502b1bc7a1757873d6446`
- Final script SHA: `9f690c7f667ce45d7c8bc1d1a67b68190bfde7c2923904abf5bab40a0d5710f5`
- Checkpoint SHA: `dc24211ff8c050f03920711aadac009d4feac26568ff356f8d6bd29dabf7d329`
- Contract SHA: `90cfab7d3a1fc87cc46b26441d3d883b9fc72d8852d1e2074b00e628428f5`
- Topology manifest SHA: `2f5d3a20d882cc4d630ef89e2abb3692e2acf3dbb3f2cb7ce0126cdfc5019445`
- Score-freeze manifest SHA: `e43484f7e40e9ef3660bb96e64d442b56b5ed17e946380f3c76f10f09927b3a5`
- Label-join manifest SHA: `b65978f906438b9d21bc143eb959897cef83b2006cf810f788455feca2a67fed`
- Reconciled input manifest SHA: `58e2deca17e76aa643b161232225d3be733dad4c59d429921583b2736d4f2bd4`
- Corrected-example reconciliation SHA: `4ef3f920d19f996d13ac8737ddbb6ce58fea30114e174f955159540a5104a25b`

The original pre-GT gate verified the immutable R64 closure, authorization, checkpoint manifest, selection, parameter count and contract, but did not explicitly compare the current corrected train/validation example files and manifests. A post-closure reconciliation compared all four files against the independent R64 closure/checkpoint/training records: train `f002826704a31f4e8fa7e8ba0da0d818c10c6d8514eeee4378d973bdcb7d086f`, validation `aa02f35de40fa64f8d0e4b5600d108a7a7eaed80a5ed714a0edc3e83f5960735`, train manifest `1524735e216ca8cc1d65a75cc4165ef0f8c8061b9682d7f72a6f003585521050`, validation manifest `b15e6415b40db33794597fd14e30d4680457da2a25ea0de9fb1907510c580453`; all matched. The explicit current-file comparison occurred after label unlock and cannot be represented as a pre-GT event. These files are not consumed by M23-65 topology, inference, label join or metrics; no scientific artifact was regenerated.

Final command sequence:
```text
python -u scripts/m23_research/m23_65_v3_mot20_representation_gate.py init
python -u scripts/m23_research/m23_65_v3_mot20_representation_gate.py topology
python -u scripts/m23_research/m23_65_v3_mot20_representation_gate.py scores
python -u scripts/m23_research/m23_65_v3_mot20_representation_gate.py labels
python -u scripts/m23_research/m23_65_v3_mot20_representation_gate.py metrics
python -u scripts/m23_research/m23_65_v3_mot20_representation_gate.py gate
```
`metrics` had two implementation-only performance restarts (pair-trust lookup caching and scalar rank accumulation), and `gate` had one (vectorized pooled join); no frozen scientific artifact was changed.

## Freeze/unlock order

- `2026-07-23T10:50:47.570622+00:00` — **topology_frozen**; gt_reads=0
- `2026-07-23T11:08:51.414905+00:00` — **score_freeze_complete**; gt_reads=0
- `2026-07-23T11:12:15.904185+00:00` — **label_unlock**; gt_reads=0
- `2026-07-23T11:12:16.142482+00:00` — **mot20_gt_opened** (MOT20-01); gt_reads=1
- `2026-07-23T11:12:20.317300+00:00` — **sequence_label_join_complete** (MOT20-01); gt_reads=1
- `2026-07-23T11:12:21.764123+00:00` — **mot20_gt_opened** (MOT20-02); gt_reads=1
- `2026-07-23T11:12:47.032191+00:00` — **sequence_label_join_complete** (MOT20-02); gt_reads=1
- `2026-07-23T11:12:47.671240+00:00` — **mot20_gt_opened** (MOT20-03); gt_reads=1
- `2026-07-23T11:13:32.657495+00:00` — **sequence_label_join_complete** (MOT20-03); gt_reads=1
- `2026-07-23T11:13:33.902083+00:00` — **mot20_gt_opened** (MOT20-05); gt_reads=1
- `2026-07-23T11:15:15.599998+00:00` — **sequence_label_join_complete** (MOT20-05); gt_reads=1
- `2026-07-23T11:15:22.313999+00:00` — **label_unlock_complete**; gt_reads=4
- `2026-07-23T11:45:36.506928+00:00` — **representation_gate_closed**; gt_reads=4

The score freeze completed at gt_reads=0; label unlock followed; all four GT reads occurred only after that event. Score immutability validation is true.

## Per-sequence representation metrics

| sequence | boundary base | PR-AUC | precision@actual | recall@90 | recall@95 | recall@99 | ROC-AUC | node PR-AUC | node ROC-AUC | outgoing R@1/R@3/MRR | incoming R@1/R@3/MRR | paired R@1 | catastrophic false-link | candidate recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|
| MOT20-01 | 0.004639 | 0.009428 | 0.049080 | 0.000000 | 0.000000 | 0.000000 | 0.609760 | 0.159547 | 0.651274 | 0.429060/0.625641/0.553331 | 0.131624/0.292308/0.249487 | 0.506650 | 0.416667 | 1.000000 |
| MOT20-02 | 0.005156 | 0.008274 | 0.011088 | 0.000000 | 0.000000 | 0.000000 | 0.616840 | 0.124643 | 0.651541 | 0.322498/0.487382/0.441940 | 0.045134/0.110374/0.124110 | 0.492448 | 0.492124 | 1.000000 |
| MOT20-03 | 0.001059 | 0.001061 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.511474 | 0.018258 | 0.554199 | 0.445977/0.619038/0.560925 | 0.039055/0.103491/0.116174 | 0.483723 | 0.451842 | 1.000000 |
| MOT20-05 | 0.001247 | 0.001389 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.548730 | 0.023455 | 0.583418 | 0.343446/0.520276/0.466526 | 0.038319/0.098002/0.110824 | 0.527819 | 0.484053 | 1.000000 |

Score diagnostics (min/max/mean/std, ties, saturation) are in `representation_metrics.json` and the expanded CSV. `ABA_exact_two_boundary_recall` is explicitly undefined under the frozen pairwise observation schema and is not used for the gate; MOT17 source-to-target retention is not computed because this experiment does not retarget against MOT17.

## Frozen gate

| check | threshold | observed | result |
|---|---:|---:|---|
| macro boundary PR-AUC | >= 0.283 | 0.005038048 | FAIL |
| macro precision@actual | >= 0.35 | 0.015041941 | FAIL |
| macro recall@95 precision | >= 0.05 | 0.000000 | FAIL |
| minimum sequence precision@actual | >= 0.20 | 0.000000 | FAIL |

Pooled boundary PR-AUC is 0.003626583 and pooled precision@actual is 0.011120152; pooled values are descriptive only. No score reversal, calibration, temperature scaling or threshold search was used.

## Scope and closure

- training_runs=0; optimizer_steps=0; checkpoint_outputs=0
- mot20_gt_reads=4 (only after score freeze); mot20_test_reads=0; mot20_test_submissions=0
- teacher_reads=0; held_outer_teacher_reads=0; tracker_outputs=0; trackeval_runs=0; hota_evaluations=0
- m23_54_starts=0; m23_58_starts=0; v2_checkpoint_loads=0; warm_starts=0
- HOTA is empty (`null`); no tracker or TrackEval was run.
- `closure_validation.json` reports `closure_integrity_passed=true`.
- `next_stage_authorization.json` reports `authorized=false`; no M23-66 authorization was issued.
- Notion writeback: **未执行Notion写回**.

Result artifacts: `representation_metrics.json`, `representation_metrics.csv`, `per_sequence_gate.csv`, `representation_gate.json`, `score_immutability_validation.json`, `performance_validation.json`, `leakage_scope_validation.json`, `final_summary.json`, and `closure_validation.json`.
