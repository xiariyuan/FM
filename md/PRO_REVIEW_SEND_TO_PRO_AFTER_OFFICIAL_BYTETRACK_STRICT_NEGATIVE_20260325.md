# Pro Review Request: Under Fixed Official ByteTrack, What Is The Single Best Redesign After Strict Official-Host Retrain Failed?

## Zero-Context Opening

你现在面对的是一个没有共享上下文的项目裁决问题。请只依赖我附带的文档、结果文件和代码锚点，不要脑补我们之前已经讨论过什么。

这次问题已经被强力收窄，不要重新发散。

当前有 6 条固定事实:

1. 论文主 carrier 已固定为 `official ByteTrack`。
2. baseline hierarchy 已固定为:
   - primary paper baseline: `official ByteTrack`
   - secondary transfer baseline: `BoT-SORT`
   - internal ablation-only baseline: internal ByteTrack-style hosts 与当前仓内 StrongSORT 线
   - exclude for now: `MOTIP`
3. 当前方法不是 whole-tracker rewrite，而是一个:
   - frozen host
   - plugin-style local association operator
   - cluster-level
   - conservative partial commit
   - defer to host
   - primary-only
   - pre-Hungarian
4. 已死路线不要重开:
   - row-local rerank
   - full cluster replacement
   - continuity / stitching
5. 当前 stronger module family 已固定为:
   - `HostConditionedLocalConflictSetPredictor`
   - 即 `set_predictor_v2`
6. 现在不再问 baseline 选谁，也不再问模块线要不要 kill。现在唯一问题是:
   - 在 `official ByteTrack` 这条固定主线上，为什么 official-host retrain 之后模块会学成 aggressive replacer；
   - 下一步唯一最值钱的 redesign 应该是什么。

## 这次请把下面这些文件当作权威上下文

- `md/PRO_REVIEW_CANONICAL_CONTEXT_20260324.md`
- `md/PRO_REVIEW_LATEST_DELTA_20260324.md`
- `md/PRO_REVIEW_LATEST_DELTA_20260325_BASELINE_PIVOT.md`
- `md/PRO_REVIEW_LATEST_DELTA_20260325_OFFICIAL_BYTETRACK_STRICT_NEGATIVE.md`
- `md/PRO_REVIEW_EXPERIMENT_CHAIN_INDEX_20260325.md`
- `md/PRO_REVIEW_INTERACTION_LOG.md`

## 如果你要核对代码，请优先看这些锚点

- `third_party/ByteTrack/yolox/tracker/byte_tracker_local_conflict.py`
- `third_party/ByteTrack/tools/track.py`
- `third_party/ByteTrack/yolox/evaluators/mot_evaluator.py`
- `scripts/run_official_bytetrack_local_conflict_halfval_pair.py`
- `scripts/run_official_bytetrack_local_conflict_stage1_trainhalf.py`
- `scripts/build_local_conflict_set_predictor_dataset.py`
- `scripts/train_local_conflict_set_predictor.py`
- `models/local_conflict_set_predictor.py`

## 当前最新实验事实

### 1. strict official ByteTrack paired protocol 已经打通

也就是说:

- 现在不是“我们还没在严格论文 baseline 上比较”
- 而是“已经严格比较过了”

### 2. zero-shot official-host migration 没过，但当时主要是没介入

运行根目录:

- `outputs/official_bytetrack_local_conflict_halfval_pair_20260325_184000`

paired result:

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

当时关键诊断:

- `eligible_clusters = 7015`
- `replaced_clusters = 0`
- `matched_dets = 0`
- `trigger_filtered_clusters = 7015`

所以第一阶段问题更像是:

- internal-host-trained checkpoint 不能 zero-shot 迁到 official host

### 3. official-host retrain 成功了，但结果更差

运行根目录:

- `outputs/official_bytetrack_local_conflict_stage1_trainhalf_20260325_195300`

### 3.1 official-host stage1 split

在当前 caps 下:

- `topk = 8`
- `min_detections = 2`
- `min_committed_matches = 2`
- `max_detections = 8`
- `max_tracks = 32`

official-host eligible clusters 只出现在 3 个序列:

- `MOT17-05-FRCNN: 285`
- `MOT17-09-FRCNN: 173`
- `MOT17-11-FRCNN: 99`

因此 fallback split 自动变成:

- train: `MOT17-05-FRCNN,MOT17-09-FRCNN`
- val: `MOT17-11-FRCNN`

### 3.2 official-host dataset 分布很关键

关键文件:

- `outputs/official_bytetrack_local_conflict_stage1_trainhalf_20260325_195300/cluster_set_predictor_data/summary.json`
- `outputs/official_bytetrack_local_conflict_stage1_trainhalf_20260325_195300/cluster_set_predictor_data/sequence_cluster_summary.csv`

核心数字:

- `eligible_clusters = 557`
- `trigger_pass_clusters = 557`
- `trigger_fail_clusters = 0`
- `train_examples = 458`
- `val_examples = 99`

也就是说:

- current official-host stage1 dataset 几乎没有真正的 gate 负样本

### 3.3 stage1 训练本身没有崩

关键文件:

- `outputs/official_bytetrack_local_conflict_stage1_trainhalf_20260325_195300/01_stage1/summary.csv`

best checkpoint:

- `best_epoch = 11`
- `val_loss = 0.6323`
- `val_row_acc = 0.8845`
- `val_commit_precision = 0.9530`
- `val_commit_recall = 0.9971`
- `val_edge_ap = 0.8180`

但 gate 指标几乎全满:

- `val_cluster_f1 = 1.0`
- `val_cluster_gate_precision_cal = 1.0`
- `val_cluster_gate_recall_cal = 1.0`
- `val_cluster_gate_coverage_cal = 1.0`

### 3.4 strict paired official ByteTrack half-val 结果明显为负

关键文件:

- `outputs/official_bytetrack_local_conflict_stage1_trainhalf_20260325_195300/summary.csv`
- `outputs/official_bytetrack_local_conflict_stage1_trainhalf_20260325_195300/result.csv`
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

### 3.5 这次不是没触发，而是介入过头

plugin arm 诊断:

- `eligible_clusters = 7025`
- `replaced_clusters = 5935`
- `matched_dets = 18532`
- `deferred_dets = 4793`
- `gate_pass_clusters = 7025`
- `gate_filtered_clusters = 0`
- `trigger_filtered_clusters = 1090`
- `skipped_large_clusters = 1010`

所以当前 negative 的关键事实不是:

- 模块没插进去
- 模块没学到 row ranking

而是:

- official-host retrain 后，当前 operator 更像 aggressive replacer，而不是 conservative partial commit module

## 这次不要重开的结论

以下内容已经固定，请不要回头重开:

- 不要重新讨论 baseline selection
- 不要再建议把 `base_reid_da` 推回论文主 carrier
- 不要再建议 row-local / full replacement / continuity
- 不要再建议先做 host migration
- 不要把当前 negative 简化成“再多训几轮试试”

## 这次你必须回答的唯一问题

> 在固定 `official ByteTrack` 为论文主 carrier 的前提下，当前 strict official-host retrain negative 最应该如何 redesign，才能把 current `set_predictor_v2` 从 aggressive replacer 改回 selective conservative operator？

## 你必须在下面这些解释中做主次判断

你可以提出自己的判断，但请至少明确区分这几个层次里哪个是主病因、哪个只是次病因:

1. `dataset / target semantics`
   - 例如 `trigger_pass` 在 official host 下退化成近乎全正
   - current labels 没表达“在线净收益”
2. `gate / abstain supervision`
   - 模型没真正学会什么时候不要碰 cluster
3. `runtime operator constraints`
   - 当前 online semantics 过于常开，缺少保守约束
4. `model family itself`
   - 当前 `set_predictor_v2` 这个 family 在 official ByteTrack 上就不合适

## 请你给唯一主方案和一个备选

我不要“可以都试试”。请你只给:

- 一个唯一主方案
- 一个备选

并说明:

- 为什么主方案排第一
- 为什么另外几种先不要做

## 额外硬要求: 请直接给代码级重设计文档

这次请不要只给方向判断，我还需要你直接对当前代码做“可执行级别的重设计”。

你的回答必须包含两部分：

### Part A. 管理级决策

你必须明确给出：

- `GO / NARROW GO / KILL`
- 为什么
- 下一步唯一 first-priority experiment 是什么
- 哪些事情现在不要再做

### Part B. 代码级实现设计文档

请你基于我提供的代码包，直接写一份可执行的 `MD` 设计文档。
这份文档不能停留在概念层，必须具体到代码改动层。

你的实现设计文档，至少必须包含以下内容：

1. 总体模块设计
- 新主模块叫什么
- 它是升级当前 `set_predictor_v2`、还是替换它、还是保留它只改 supervision / runtime
- 在线注入点在哪
- 输入是什么
- 输出是什么
- 与 host 的接口是什么

2. 文件级改动清单
对于每一个需要修改或新增的文件，你都必须写清楚：

- 文件路径
- 为什么改这个文件
- 当前这个文件里相关逻辑是什么
- 需要修改的函数 / 类 / 配置段是什么
- 明确代码锚点：
  - 函数名
  - 类名
  - 关键变量名
  - 可 grep 的关键字符串
- 修改后应该变成什么
- 是局部改写还是新增模块后接入

3. 你必须具体回答这些工程问题

- current official-host dataset 的 gate negatives 该如何构造
- current `trigger_pass` 是否应该被替换成 utility-style cluster label
- 是否需要从 paired online outcome 反推 cluster-level intervention target
- runtime 是否应该加入:
  - 更严格的 cluster gate
  - replacement budget
  - stronger abstain path
  - 每簇最大提交数 / 更高 `min_committed_matches`
- 当前 `set_predictor_v2` family 是保留还是更换

4. 第一批实验必须只给一个 first-priority experiment

而且这一步必须是:

- 官方 ByteTrack 主线上可直接落地的 paired experiment

不能再回到:

- baseline selection
- host migration
- internal host sweep

## 我当前自己的判断，你可以直接反驳

我当前本地判断是:

- 当前官方 negative 并不证明 operator 方向应该 kill
- 但它已经证明 current official-host training objective 把模块学坏了
- 现在最像主病因的是:
  - gate / abstain supervision 退化
  - target semantics 没表达在线净收益
  - runtime conservative constraints 不够
- 我不确定应不应该优先:
  - 只改 supervision 和 runtime，不换 `set_predictor_v2`
  - 还是直接换一个更适合 selective intervention 的 family

如果你不同意我的判断，请直接反驳，并给出更强的主方案。

## 希望的回答格式

请直接按下面结构回答:

1. `管理级决策`
2. `为什么当前 official-host negative 的主病因是什么`
3. `唯一主方案`
4. `一个备选`
5. `文件级与 runner 级落点`
6. `唯一 first-priority experiment`
7. `当前不要做什么`

我要的是一份能直接指导下一轮工程改动的回答，不要泛泛空话。
