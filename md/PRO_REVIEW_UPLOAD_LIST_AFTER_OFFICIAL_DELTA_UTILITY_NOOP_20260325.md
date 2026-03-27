# Pro Review Upload List (2026-03-25, Official ByteTrack Delta-Utility Mainline Near No-Op)

这份清单对应：

- `md/PRO_REVIEW_SEND_TO_PRO_AFTER_OFFICIAL_DELTA_UTILITY_NOOP_20260325.md`

建议至少携带下面这些文件。不要只发提示词而不发上下文。

## 1. 必带上下文文档

- `md/PRO_REVIEW_CANONICAL_CONTEXT_20260324.md`
- `md/PRO_REVIEW_LATEST_DELTA_20260324.md`
- `md/PRO_REVIEW_LATEST_DELTA_20260325_BASELINE_PIVOT.md`
- `md/PRO_REVIEW_LATEST_DELTA_20260325_OFFICIAL_BYTETRACK_STRICT_NEGATIVE.md`
- `md/PRO_REVIEW_LATEST_DELTA_20260325_OFFICIAL_DELTA_UTILITY_NOOP.md`
- `md/PRO_REVIEW_EXPERIMENT_CHAIN_INDEX_20260325.md`
- `md/PRO_REVIEW_INTERACTION_LOG.md`
- `official_bytetrack_redesign_decision_20260325.md`
- `md/PRO_REVIEW_SEND_TO_PRO_AFTER_OFFICIAL_DELTA_UTILITY_NOOP_20260325.md`

## 2. 必带关键结果文件

### A. strict official ByteTrack zero-shot slight negative

- `outputs/official_bytetrack_local_conflict_halfval_pair_20260325_184000/result.csv`
- `outputs/official_bytetrack_local_conflict_halfval_pair_20260325_184000/summary.csv`

### B. official-host retrain aggressive negative

- `outputs/official_bytetrack_local_conflict_stage1_trainhalf_20260325_195300/summary.csv`
- `outputs/official_bytetrack_local_conflict_stage1_trainhalf_20260325_195300/result.csv`
- `outputs/official_bytetrack_local_conflict_stage1_trainhalf_20260325_195300/01_stage1/summary.csv`
- `outputs/official_bytetrack_local_conflict_stage1_trainhalf_20260325_195300/01_stage1/metrics.jsonl`
- `outputs/official_bytetrack_local_conflict_stage1_trainhalf_20260325_195300/02_official_halfval_pair/result.csv`
- `outputs/official_bytetrack_local_conflict_stage1_trainhalf_20260325_195300/02_official_halfval_pair/summary.csv`
- `outputs/official_bytetrack_local_conflict_stage1_trainhalf_20260325_195300/02_official_halfval_pair/01_host_plus_plugin/summary.csv`
- `outputs/official_bytetrack_local_conflict_stage1_trainhalf_20260325_195300/cluster_set_predictor_data/summary.json`
- `outputs/official_bytetrack_local_conflict_stage1_trainhalf_20260325_195300/cluster_set_predictor_data/sequence_cluster_summary.csv`

### C. current delta-utility near no-op negative

- `outputs/official_bytetrack_delta_utility_mainline_20260325_224718/summary.csv`
- `outputs/official_bytetrack_delta_utility_mainline_20260325_224718/result.csv`
- `outputs/official_bytetrack_delta_utility_mainline_20260325_224718/pipeline.log`
- `outputs/official_bytetrack_delta_utility_mainline_20260325_224718/01_stage1/summary.csv`
- `outputs/official_bytetrack_delta_utility_mainline_20260325_224718/01_stage1/metrics.jsonl`
- `outputs/official_bytetrack_delta_utility_mainline_20260325_224718/02_official_halfval_pair/result.csv`
- `outputs/official_bytetrack_delta_utility_mainline_20260325_224718/02_official_halfval_pair/summary.csv`
- `outputs/official_bytetrack_delta_utility_mainline_20260325_224718/02_official_halfval_pair/01_host_plus_plugin/summary.csv`
- `outputs/official_bytetrack_delta_utility_mainline_20260325_224718/cluster_set_predictor_data/summary.csv`
- `outputs/official_bytetrack_delta_utility_mainline_20260325_224718/cluster_set_predictor_data/sequence_cluster_summary.csv`
- `outputs/official_bytetrack_delta_utility_mainline_20260325_224718/cluster_set_predictor_data/cluster_examples.sample.jsonl`

## 3. 必带代码锚点

- `third_party/ByteTrack/yolox/tracker/byte_tracker_local_conflict.py`
- `third_party/ByteTrack/tools/track.py`
- `third_party/ByteTrack/yolox/evaluators/mot_evaluator.py`
- `scripts/run_official_bytetrack_local_conflict_halfval_pair.py`
- `scripts/run_official_bytetrack_local_conflict_stage1_trainhalf.py`
- `scripts/build_local_conflict_set_predictor_dataset.py`
- `scripts/train_local_conflict_set_predictor.py`
- `models/local_conflict_set_predictor.py`

## 4. 可选但建议带上

- `outputs/official_bytetrack_delta_utility_mainline_20260325_224718/source_manifest.csv`
- `outputs/official_bytetrack_delta_utility_mainline_20260325_224718/labeled_replay.summary.json`
- `outputs/official_bytetrack_delta_utility_mainline_20260325_224718/01_stage1/best.pt`
- `outputs/official_bytetrack_local_conflict_stage1_trainhalf_20260325_195300/cluster_set_predictor_data/cluster_examples.sample.jsonl`

## 5. 当前不需要重点带的内容

当前这次提问不需要再重点带：

- `base_reid_da` internal positive 的全套旧工件
- `v15` host migration 那套 runner 细节
- baseline selection 之前的全量争论材料

理由：

- 这些问题已经被收束过了；
- 这次问题不是“baseline 选谁”，也不是“official ByteTrack 能不能当主线”；
- 这次问题是：
  - 为什么 old official-host retrain 是 `aggressive replacer`
  - 为什么 new `delta_utility` redesign 又是 `near no-op`
  - 下一步怎么把 operator 拉回 “非零覆盖但仍保守”的区间

