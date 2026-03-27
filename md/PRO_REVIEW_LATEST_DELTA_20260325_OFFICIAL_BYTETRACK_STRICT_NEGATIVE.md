# Pro Review Latest Delta (2026-03-25, Official ByteTrack Strict Negative)

这份 delta 只记录一个新的收束点:

- `official ByteTrack` 现在已经固定为论文主 carrier，不再继续争论 baseline selection；
- 插件式 local-conflict operator 已经成功接进 `official ByteTrack` 的严格 paired half-val protocol；
- 但在 `official ByteTrack` 上，当前 `set_predictor_v2` 仍然没有得到正结果；
- 而且这次负结果已经从“zero-shot 不介入”升级成“official-host retrain 后过度介入并明显伤指标”。

因此，下一次向 Pro 的问题不再是:

- baseline 应该选谁
- official ByteTrack 是否排第一
- host migration 还要不要做

而是收窄成:

- 在 `official ByteTrack` 这条严格主线上，当前 operator 为什么会学成 aggressive replacer；
- 下一步唯一最值钱的 redesign，应该落在:
  - supervision / target semantics
  - selective gate / abstain
  - runtime conservative commit constraints
  - 还是更换模块 family。

## 1. 已经固定的新事实

### 1.1 baseline hierarchy 已固定

当前已经固定:

- primary paper baseline: `official ByteTrack`
- secondary transfer baseline: `BoT-SORT`
- internal ablation-only baseline:
  - `base_reid_da`
  - internal ByteTrack-style hosts
  - 当前仓内 StrongSORT 线
- currently exclude:
  - `MOTIP`

这次不要再把问题重新转回 baseline selection。

### 1.2 official ByteTrack 插件接入已经打通

关键代码路径:

- `third_party/ByteTrack/yolox/tracker/byte_tracker_local_conflict.py`
- `third_party/ByteTrack/tools/track.py`
- `third_party/ByteTrack/yolox/evaluators/mot_evaluator.py`
- `scripts/run_official_bytetrack_local_conflict_halfval_pair.py`
- `scripts/run_official_bytetrack_local_conflict_stage1_trainhalf.py`

当前不是“还没接进去”，而是已经能在严格 paired protocol 下稳定评估。

## 2. 第一阶段负证据: zero-shot 官方宿主迁移没过

运行根目录:

- `/gemini/code/FMtrack-main/FM-Track/outputs/official_bytetrack_local_conflict_halfval_pair_20260325_184000`

关键结果:

- host-only:
  - `HOTA 77.594`
  - `AssA 76.534`
  - `IDF1 86.235`
  - `MOTA 90.186`
  - `IDSW 183`
- host + internal-host-trained `v2`:
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

关键诊断:

- `eligible_clusters = 7015`
- `replaced_clusters = 0`
- `matched_dets = 0`
- `trigger_filtered_clusters = 7015`

结论:

- 这一步说明 internal-host-trained checkpoint 不能 zero-shot 迁到 official ByteTrack；
- 但它当时主要是“没真正出手”，不是 aggressive failure。

## 3. 第二阶段负证据: official-host retrain 后变成过度介入

运行根目录:

- `/gemini/code/FMtrack-main/FM-Track/outputs/official_bytetrack_local_conflict_stage1_trainhalf_20260325_195300`

### 3.1 official-host 数据集构建成功，但分布很窄

关键文件:

- `/gemini/code/FMtrack-main/FM-Track/outputs/official_bytetrack_local_conflict_stage1_trainhalf_20260325_195300/cluster_set_predictor_data/summary.json`
- `/gemini/code/FMtrack-main/FM-Track/outputs/official_bytetrack_local_conflict_stage1_trainhalf_20260325_195300/cluster_set_predictor_data/sequence_cluster_summary.csv`

在当前 cluster caps:

- `topk = 8`
- `min_detections = 2`
- `min_committed_matches = 2`
- `max_detections = 8`
- `max_tracks = 32`

下，official-host train-half 只产生了:

- `eligible_clusters = 557`
- `trigger_pass_clusters = 557`
- `trigger_fail_clusters = 0`

并且只有 3 个序列有 eligible clusters:

- `MOT17-05-FRCNN: 285`
- `MOT17-09-FRCNN: 173`
- `MOT17-11-FRCNN: 99`

fallback split 因此自动变成:

- train: `MOT17-05-FRCNN,MOT17-09-FRCNN`
- val: `MOT17-11-FRCNN`

训练样本量:

- `train_examples = 458`
- `val_examples = 99`

这一步的关键问题是:

- stage1 official-host 数据里 `trigger_fail_clusters = 0`
- 也就是说当前 gate supervision 在 official host 下几乎没有真正负样本

### 3.2 stage1 训练本身成功

关键文件:

- `/gemini/code/FMtrack-main/FM-Track/outputs/official_bytetrack_local_conflict_stage1_trainhalf_20260325_195300/01_stage1/summary.csv`
- `/gemini/code/FMtrack-main/FM-Track/outputs/official_bytetrack_local_conflict_stage1_trainhalf_20260325_195300/01_stage1/metrics.jsonl`

best checkpoint:

- `best_epoch = 11`
- `val_loss = 0.6323`
- `val_row_acc = 0.8845`
- `val_commit_precision = 0.9530`
- `val_commit_recall = 0.9971`
- `val_edge_ap = 0.8180`

但 gate 相关指标几乎是全满:

- `val_cluster_f1 = 1.0`
- `val_cluster_gate_precision_cal = 1.0`
- `val_cluster_gate_recall_cal = 1.0`
- `val_cluster_gate_coverage_cal = 1.0`

这和上面的数据分布一起看，不能解释成“gate 真学好了”，更可能是:

- 当前训练目标下 gate 没有被迫学会 selective abstain

### 3.3 strict paired official ByteTrack half-val 结果明显为负

关键文件:

- `/gemini/code/FMtrack-main/FM-Track/outputs/official_bytetrack_local_conflict_stage1_trainhalf_20260325_195300/02_official_halfval_pair/result.csv`
- `/gemini/code/FMtrack-main/FM-Track/outputs/official_bytetrack_local_conflict_stage1_trainhalf_20260325_195300/02_official_halfval_pair/summary.csv`
- `/gemini/code/FMtrack-main/FM-Track/outputs/official_bytetrack_local_conflict_stage1_trainhalf_20260325_195300/02_official_halfval_pair/01_host_plus_plugin/summary.csv`

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

### 3.4 这次不是没介入，而是介入过头

plugin arm 诊断汇总:

- `eligible_clusters = 7025`
- `replaced_clusters = 5935`
- `matched_dets = 18532`
- `deferred_dets = 4793`
- `gate_pass_clusters = 7025`
- `gate_filtered_clusters = 0`
- `trigger_filtered_clusters = 1090`
- `skipped_large_clusters = 1010`

按序列看，前两个样例也已经足够说明问题:

- `MOT17-02-FRCNN`
  - `eligible_clusters = 951`
  - `replaced_clusters = 685`
  - `gate_pass_clusters = 951`
  - `gate_filtered_clusters = 0`
- `MOT17-04-FRCNN`
  - `eligible_clusters = 2474`
  - `replaced_clusters = 2270`
  - `gate_pass_clusters = 2474`
  - `gate_filtered_clusters = 0`

结论:

- 这次负结果不是“模块没触发”；
- 也不是“只提交了少量错误 commit”；
- 当前 official-host-trained `v2` 更像是一个近乎全开门的 aggressive replacer。

## 4. 当前最有可能的根因层

现在最值得怀疑的，不再是 baseline cleanliness，也不再是 integration bug。

当前最可能的根因是:

1. current official-host stage1 dataset 的 gate supervision 退化
   - `trigger_fail_clusters = 0`
   - 导致 cluster gate 学不到“别出手”
2. current target semantics 过于贴近 `trigger_pass / oracle commit`
   - 但没有表达“介入的在线净收益是否为正”
3. runtime contract 缺少更保守的 operator 约束
   - 当前 plugin arm 在 official host 上几乎全放行
4. 因此，哪怕 row/edge ranking 学到了一些东西，整体在线行为依然会过度替换 host

## 5. 下一次向 Pro 真正要问什么

下一次不要再问:

- official ByteTrack 要不要做主线
- host migration 还要不要继续
- 当前模块方向要不要 kill

现在真正要问的是:

1. 在 `official ByteTrack` 这个固定主线上，当前 negative 的主病因更接近:
   - gate supervision / abstain target
   - commit utility target construction
   - runtime conservative constraint
   - 还是模块 family 本身不对
2. 下一步唯一主 redesign 应该是什么
3. 这一步 redesign 应该改哪些文件
4. 第一枪应该跑什么 paired experiment

## 6. 当前不要重开的事情

- 不要重开 baseline selection
- 不要把 `base_reid_da` 再推回论文主 carrier
- 不要把这次失败误读成“插件没接进去”
- 不要继续重复同版 `v2` official-host retrain
- 不要重新打开 row-local / full replacement / continuity
- 不要先做 host migration / BoT-SORT transfer

## 7. 当前本地判断

当前本地判断已经收紧为:

- `official ByteTrack` 严格主线已经建立完成；
- 当前问题不再是“是否严格可比”，而是“当前 official-host 训练目标把 operator 学坏了”；
- 下一步最值钱的 Pro 问题，应该是:
  - 在固定 official ByteTrack 主线下
  - 如何把 `set_predictor_v2` 从 aggressive replacer 改回 selective conservative operator。
