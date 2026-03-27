# Pro Review Experiment Chain Index (2026-03-25)

这份索引的目的不是重复 canonical context，而是把“当前真正相关的实验链”按时间顺序串起来，方便新的 Pro 在没有本地上下文时快速定位证据。

## 0. 方向已经固定的背景

当前已经固定:

- 方法不是 whole-tracker rewrite
- 方法是 frozen host 上的 plugin-style local association operator
- 在线语义固定为:
  - cluster-level
  - conservative partial commit
  - defer to host
  - primary-only
  - pre-Hungarian
- row-local rerank 已死
- full cluster replacement 已死
- 当前 stronger module family 是:
  - `HostConditionedLocalConflictSetPredictor`
  - `set_predictor_v2`

## 1. internal host 上的 enlarged-data `v1` 证明 operator 不是幻觉

运行根目录:

- `outputs/local_conflict_commit_large_base_20260324_222409`

关键信息:

- enlarged-data `v1` 在 internal host `base_reid_da` 上拿到真实正号
- 这一步证明:
  - `conservative partial commit` 方向不是幻觉

推荐先看:

- `outputs/local_conflict_commit_large_base_20260324_222409/summary.csv`
- `outputs/local_conflict_commit_large_base_20260324_222409/01_stage1/summary.csv`
- `outputs/local_conflict_commit_large_base_20260324_222409/02_proxy_eval/summary.csv`
- `outputs/local_conflict_commit_large_base_20260324_222409/03_full_eval_md2_mm2/summary.csv`

## 2. `v15` host migration 负证据说明 current old line 不具备简单 portability

运行根目录:

- `outputs/local_conflict_graph_hostmig_v15_proxy0213_20260324_194915`

关键信息:

- `v15` stronger-host first-shot migration 为负
- 这一步把问题从“继续磨旧 learned commit”推向:
  - larger data
  - stronger structured module

推荐先看:

- `outputs/local_conflict_graph_hostmig_v15_proxy0213_20260324_194915/result.csv`
- `outputs/local_conflict_graph_hostmig_v15_proxy0213_20260324_194915/summary.csv`

## 3. internal host 上的 stable `v2` 明确为正

运行根目录:

- `outputs/local_conflict_set_predictor_large_base_stable_20260325_023500`

关键信息:

- 相对 enlarged-data `v1`
- stable `v2` 在 internal host `base_reid_da` 上已经拿到真实正号

关键结果:

- proxy:
  - `HOTA 53.118`
  - `AssA 44.577`
  - `IDF1 58.730`
  - `MOTA 73.437`
  - `IDSW 811`
- full md2/mm2:
  - `HOTA 63.257`
  - `AssA 60.191`
  - `IDF1 72.128`
  - `MOTA 76.055`
  - `IDSW 1481`

这一步说明:

- `set_predictor_v2` 在 internal host 上不是幻觉
- baseline selection 才因此成为下一步焦点

推荐先看:

- `outputs/local_conflict_set_predictor_large_base_stable_20260325_023500/summary.csv`
- `outputs/local_conflict_set_predictor_large_base_stable_20260325_023500/01_stage1/summary.csv`
- `outputs/local_conflict_set_predictor_large_base_stable_20260325_023500/02_proxy_eval/summary.csv`
- `outputs/local_conflict_set_predictor_large_base_stable_20260325_023500/03_full_eval_md2_mm2/summary.csv`

## 4. baseline hierarchy 已固定为 official ByteTrack 主线

当前已经固定，不再争论:

- primary paper baseline: `official ByteTrack`
- secondary transfer baseline: `BoT-SORT`
- internal ablation-only baseline:
  - `base_reid_da`
  - internal ByteTrack-style hosts
  - 当前仓内 StrongSORT 线
- exclude for now:
  - `MOTIP`

对应文档:

- `md/PRO_REVIEW_LATEST_DELTA_20260325_BASELINE_PIVOT.md`
- `md/PRO_REVIEW_SEND_TO_PRO_BASELINE_SELECTION_AND_PAPER_PROTOCOL_20260325.md`

## 5. strict official ByteTrack zero-shot paired eval: 轻微负且几乎不介入

运行根目录:

- `outputs/official_bytetrack_local_conflict_halfval_pair_20260325_184000`

关键结果:

- host-only:
  - `77.594 / 76.534 / 86.235 / 90.186 / 183`
- host + internal-host-trained `v2`:
  - `77.447 / 76.266 / 86.140 / 90.195 / 177`
- delta:
  - `HOTA -0.147`
  - `AssA -0.268`
  - `IDF1 -0.095`
  - `MOTA +0.009`
  - `IDSW -6`

关键诊断:

- `eligible_clusters = 7015`
- `replaced_clusters = 0`
- `matched_dets = 0`
- `trigger_filtered_clusters = 7015`

这一步说明:

- internal-host-trained checkpoint 不能 zero-shot 迁到 official ByteTrack
- 但当时主要问题还是“不介入”

推荐先看:

- `outputs/official_bytetrack_local_conflict_halfval_pair_20260325_184000/result.csv`
- `outputs/official_bytetrack_local_conflict_halfval_pair_20260325_184000/summary.csv`

## 6. strict official ByteTrack official-host retrain: 明显负且大规模过度介入

运行根目录:

- `outputs/official_bytetrack_local_conflict_stage1_trainhalf_20260325_195300`

### 6.1 official-host dataset 事实

关键文件:

- `outputs/official_bytetrack_local_conflict_stage1_trainhalf_20260325_195300/cluster_set_predictor_data/summary.json`
- `outputs/official_bytetrack_local_conflict_stage1_trainhalf_20260325_195300/cluster_set_predictor_data/sequence_cluster_summary.csv`

关键数字:

- `eligible_clusters = 557`
- `trigger_pass_clusters = 557`
- `trigger_fail_clusters = 0`
- `train_examples = 458`
- `val_examples = 99`
- fallback split:
  - train: `MOT17-05-FRCNN,MOT17-09-FRCNN`
  - val: `MOT17-11-FRCNN`

最关键的一点:

- current official-host stage1 dataset 下，gate supervision 几乎没有负样本

### 6.2 stage1 训练本身是成功的

关键文件:

- `outputs/official_bytetrack_local_conflict_stage1_trainhalf_20260325_195300/01_stage1/summary.csv`
- `outputs/official_bytetrack_local_conflict_stage1_trainhalf_20260325_195300/01_stage1/metrics.jsonl`

best checkpoint:

- `best_epoch = 11`
- `val_loss = 0.6323`
- `val_row_acc = 0.8845`
- `val_edge_ap = 0.8180`

但 gate 指标几乎全满:

- `val_cluster_f1 = 1.0`
- `val_cluster_gate_precision_cal = 1.0`
- `val_cluster_gate_recall_cal = 1.0`
- `val_cluster_gate_coverage_cal = 1.0`

### 6.3 strict paired result 显著为负

关键文件:

- `outputs/official_bytetrack_local_conflict_stage1_trainhalf_20260325_195300/result.csv`
- `outputs/official_bytetrack_local_conflict_stage1_trainhalf_20260325_195300/summary.csv`
- `outputs/official_bytetrack_local_conflict_stage1_trainhalf_20260325_195300/02_official_halfval_pair/result.csv`
- `outputs/official_bytetrack_local_conflict_stage1_trainhalf_20260325_195300/02_official_halfval_pair/01_host_plus_plugin/summary.csv`

paired result:

- host-only:
  - `HOTA 77.594`
  - `AssA 76.534`
  - `IDF1 86.235`
  - `MOTA 90.186`
  - `IDSW 183`
- official-host-trained plugin:
  - `HOTA 73.925`
  - `AssA 70.038`
  - `IDF1 81.520`
  - `MOTA 89.568`
  - `IDSW 335`
- delta:
  - `HOTA -3.669`
  - `AssA -6.496`
  - `IDF1 -4.715`
  - `MOTA -0.618`
  - `IDSW +152`

### 6.4 这次不是没介入，而是介入过头

plugin arm 诊断:

- `eligible_clusters = 7025`
- `replaced_clusters = 5935`
- `matched_dets = 18532`
- `deferred_dets = 4793`
- `gate_pass_clusters = 7025`
- `gate_filtered_clusters = 0`
- `trigger_filtered_clusters = 1090`

最重要的结论:

- 当前问题不是 integration bug
- 当前问题也不是 baseline 不干净
- 当前问题是:
  - `official-host retrain` 之后，current `set_predictor_v2` 学成了 aggressive replacer

## 7. 当前真正未决的问题

下一次 Pro 需要回答的，不再是:

- baseline 选谁
- operator 要不要 kill

而是:

- 在固定 `official ByteTrack` 主线下
- 当前 negative 的主病因到底更接近:

## 8. strict official ByteTrack `delta_utility` mainline: 从 aggressive replacer 摆到 near no-op

运行根目录:

- `outputs/official_bytetrack_delta_utility_mainline_20260325_224718`

### 8.1 这条线的目标

这一步不是 baseline 选择，也不是 host migration。

它是在 fixed `official ByteTrack` 主线下，按新的 Pro 裁决，把 old:

- `trigger_pass + full oracle commit`

重写成:

- `delta_utility teacher`
- 更保守的 runtime constraints

目标是把模块从 `aggressive replacer` 拉回 `selective conservative operator`。

### 8.2 数据分布已经被明显压稀

关键文件:

- `outputs/official_bytetrack_delta_utility_mainline_20260325_224718/cluster_set_predictor_data/summary.csv`

关键数字:

- `eligible_clusters = 557`
- `trigger_pass_clusters = 20`
- `trigger_fail_clusters = 537`
- `cluster_should_intervene_clusters = 20`
- `delta_committed_matches = 24`
- `train_examples = 458`
- `val_examples = 99`

这一步说明:

- old official-host teacher 太宽
- current strict `delta_utility` teacher 又太稀

### 8.3 stage1 没炸，但学成了近零覆盖

关键文件:

- `outputs/official_bytetrack_delta_utility_mainline_20260325_224718/01_stage1/summary.csv`
- `outputs/official_bytetrack_delta_utility_mainline_20260325_224718/01_stage1/metrics.jsonl`

best checkpoint:

- `best_epoch = 1`
- `val_loss = 0.02717`
- `val_row_acc = 0.9986`
- `val_commit_precision = 0.0`
- `val_commit_recall = 0.0`
- `val_edge_ap = 0.00324`
- `val_cluster_f1 = 0.0`
- `val_cluster_gate_utility_cal = 0.0`
- `val_cluster_gate_coverage_cal = 0.0`

这说明:

- 当前 trainer 没崩
- 但在当前目标下，最容易学到的策略就是 “全 defer / 近全关”

### 8.4 strict paired result 仍然为负，但形态已变

关键文件:

- `outputs/official_bytetrack_delta_utility_mainline_20260325_224718/result.csv`
- `outputs/official_bytetrack_delta_utility_mainline_20260325_224718/summary.csv`
- `outputs/official_bytetrack_delta_utility_mainline_20260325_224718/02_official_halfval_pair/result.csv`
- `outputs/official_bytetrack_delta_utility_mainline_20260325_224718/02_official_halfval_pair/01_host_plus_plugin/summary.csv`

paired result:

- host-only:
  - `HOTA 77.594`
  - `AssA 76.534`
  - `IDF1 86.235`
  - `MOTA 90.186`
  - `IDSW 183`
- plugin:
  - `HOTA 77.447`
  - `AssA 76.266`
  - `IDF1 86.140`
  - `MOTA 90.195`
  - `IDSW 177`
- delta:
  - `HOTA -0.147`
  - `AssA -0.268`
  - `IDF1 -0.095`
  - `MOTA +0.009`
  - `IDSW -6`

plugin arm 诊断:

- `eligible_clusters = 7015`
- `gate_pass_clusters = 3245`
- `gate_filtered_clusters = 3770`
- `trigger_filtered_clusters = 3245`
- `replaced_clusters = 0`
- `matched_dets = 0`

这一步最关键的结论是:

- 当前失败形态已经从 `too open` 变成 `too closed`
- 现在不是 aggressive replacement
- 而是 `near no-op`

## 9. 当前真正未决的问题已经进一步收窄

下一次 Pro 需要回答的，不再是:

- baseline 选谁
- operator 要不要 kill
- 当前模块 family 要不要立即推翻

现在真正的问题是:

- 在 fixed `official ByteTrack` 主线下
- 如何把 operator 从:
  - old official-host retrain 的 `aggressive replacer`
  - current delta-utility mainline 的 `near no-op`
- 拉回到:
  - 非零覆盖
  - 但仍保守
  - 且能 strict paired 转正的 selective conservative regime
  - dataset / target semantics
  - gate / abstain supervision
  - runtime conservative constraints
  - 还是模块 family 本身
- 下一步唯一主 redesign 应该是什么

## 8. 当前最推荐先看的文件集合

如果时间有限，建议 Pro 至少按下面顺序看:

1. `md/PRO_REVIEW_CANONICAL_CONTEXT_20260324.md`
2. `md/PRO_REVIEW_LATEST_DELTA_20260325_BASELINE_PIVOT.md`
3. `md/PRO_REVIEW_LATEST_DELTA_20260325_OFFICIAL_BYTETRACK_STRICT_NEGATIVE.md`
4. `md/PRO_REVIEW_INTERACTION_LOG.md`
5. `outputs/official_bytetrack_local_conflict_halfval_pair_20260325_184000/result.csv`
6. `outputs/official_bytetrack_local_conflict_stage1_trainhalf_20260325_195300/result.csv`
7. `outputs/official_bytetrack_local_conflict_stage1_trainhalf_20260325_195300/cluster_set_predictor_data/summary.json`
8. `outputs/official_bytetrack_local_conflict_stage1_trainhalf_20260325_195300/01_stage1/summary.csv`
9. `scripts/build_local_conflict_set_predictor_dataset.py`
10. `scripts/train_local_conflict_set_predictor.py`
11. `models/local_conflict_set_predictor.py`
12. `third_party/ByteTrack/yolox/tracker/byte_tracker_local_conflict.py`
