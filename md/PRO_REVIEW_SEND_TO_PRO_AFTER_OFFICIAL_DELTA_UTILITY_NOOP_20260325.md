# Pro Review Request: Under Fixed Official ByteTrack, The Current Official-Host Redesign Has Collapsed From Aggressive Replacer To Near No-Op. What Is The Single Best Next Redesign?

## Zero-Context Opening

你现在面对的是一个没有共享上下文的项目裁决问题。请只依赖我附带的文档、结果文件和代码锚点，不要脑补我们之前已经讨论过什么。

这次问题已经被进一步收窄，不要重新发散。

当前有 7 条固定事实：

1. 论文主 carrier 已固定为 `official ByteTrack`。
2. baseline hierarchy 已固定为：
   - primary paper baseline: `official ByteTrack`
   - secondary transfer baseline: `BoT-SORT`
   - internal ablation-only baseline: internal ByteTrack-style hosts 与当前仓内 StrongSORT 线
   - exclude for now: `MOTIP`
3. 当前方法不是 whole-tracker rewrite，而是一个：
   - frozen host
   - plugin-style local association operator
   - cluster-level
   - conservative partial commit
   - defer to host
   - primary-only
   - pre-Hungarian
4. 已死路线不要重开：
   - row-local rerank
   - full cluster replacement
   - continuity / stitching
5. 当前 stronger module family 仍固定为：
   - `HostConditionedLocalConflictSetPredictor`
   - 即 `set_predictor_v2`
6. 现在不再问 baseline 选谁，也不再问这条线要不要 kill。
7. 当前唯一问题是：
   - old official-host retrain 会把模块学成 `aggressive replacer`
   - new `delta_utility + conservative runtime` 又把模块压成 `near no-op`
   - 在 fixed `official ByteTrack` 主线上，下一步唯一最值钱的 redesign 应该是什么

## 这次请把下面这些文件当作权威上下文

- `md/PRO_REVIEW_CANONICAL_CONTEXT_20260324.md`
- `md/PRO_REVIEW_LATEST_DELTA_20260324.md`
- `md/PRO_REVIEW_LATEST_DELTA_20260325_BASELINE_PIVOT.md`
- `md/PRO_REVIEW_LATEST_DELTA_20260325_OFFICIAL_BYTETRACK_STRICT_NEGATIVE.md`
- `md/PRO_REVIEW_LATEST_DELTA_20260325_OFFICIAL_DELTA_UTILITY_NOOP.md`
- `md/PRO_REVIEW_EXPERIMENT_CHAIN_INDEX_20260325.md`
- `md/PRO_REVIEW_INTERACTION_LOG.md`
- `official_bytetrack_redesign_decision_20260325.md`

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

### 1. strict official ByteTrack paired protocol 已经稳定打通

也就是说：

- 现在不是“我们还没在严格论文 baseline 上比较”
- 而是“已经严格比较过多次了”

### 2. first strict official negative：zero-shot 时主要是不介入

运行根目录：

- `outputs/official_bytetrack_local_conflict_halfval_pair_20260325_184000`

paired result：

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

当时关键诊断：

- `eligible_clusters = 7015`
- `replaced_clusters = 0`
- `matched_dets = 0`
- `trigger_filtered_clusters = 7015`

所以第一阶段问题更像：

- internal-host-trained checkpoint 不能 zero-shot 迁到 official host

### 3. second strict official negative：official-host retrain 后变成 aggressive replacer

运行根目录：

- `outputs/official_bytetrack_local_conflict_stage1_trainhalf_20260325_195300`

paired result：

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

关键诊断：

- `eligible_clusters = 7025`
- `replaced_clusters = 5935`
- `matched_dets = 18532`
- `gate_pass_clusters = 7025`
- `gate_filtered_clusters = 0`

所以第二阶段问题很明确：

- 不是没介入
- 而是 current official-host objective 把模块学成了 `aggressive replacer`

### 4. third strict official negative：delta_utility redesign 之后又退化成 near no-op

运行根目录：

- `outputs/official_bytetrack_delta_utility_mainline_20260325_224718`

paired result：

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

关键诊断：

- `eligible_clusters = 7015`
- `gate_pass_clusters = 3245`
- `gate_filtered_clusters = 3770`
- `trigger_filtered_clusters = 3245`
- `replaced_clusters = 0`
- `matched_dets = 0`
- `deferred_dets = 23315`

这一步说明：

- 现在不是 aggressive replacer 了
- 但也不是有效 selective operator
- 它现在更像 `pass gate but still defer everything`

### 5. 当前新 teacher 的分布已经被压得非常稀

关键文件：

- `outputs/official_bytetrack_delta_utility_mainline_20260325_224718/cluster_set_predictor_data/summary.csv`

核心数字：

- `eligible_clusters = 557`
- `trigger_pass_clusters = 20`
- `trigger_fail_clusters = 537`
- `cluster_should_intervene_clusters = 20`
- `delta_committed_matches = 24`
- `train_examples = 458`
- `val_examples = 99`

这一步说明：

- old teacher 太宽
- current strict `delta_utility` teacher 又太稀

### 6. stage1 训练也支持 “too sparse -> all defer / all close” 这个解释

关键文件：

- `outputs/official_bytetrack_delta_utility_mainline_20260325_224718/01_stage1/summary.csv`

best checkpoint：

- `best_epoch = 1`
- `val_loss = 0.02717`
- `val_row_acc = 0.9986`
- `val_commit_precision = 0.0`
- `val_commit_recall = 0.0`
- `val_edge_ap = 0.00324`
- `val_cluster_f1 = 0.0`
- `val_cluster_gate_precision_cal = 0.0`
- `val_cluster_gate_recall_cal = 0.0`
- `val_cluster_gate_utility_cal = 0.0`
- `val_cluster_gate_coverage_cal = 0.0`

所以当前现实不是：

- trainer 崩了
- 或 runtime bug

而是：

- 现有 redesign 把目标压得过稀，最容易学到的策略就是“全 defer / 近全关”

## 这次不要重开的结论

以下内容已经固定，请不要回头重开：

- 不要重新讨论 baseline selection
- 不要再建议把 `base_reid_da` 推回论文主 carrier
- 不要再建议 row-local / full replacement / continuity
- 不要再建议先做 host migration
- 不要再建议 whole-tracker rewrite
- 不要再回答“多训几轮看看”

## 这次你必须回答的唯一问题

> 在固定 `official ByteTrack` 为论文主 carrier、固定 plugin-style pre-Hungarian operator contract 的前提下，当前官方主线已经从 `aggressive replacer` 摆到 `near no-op`。下一步唯一最值钱的 redesign，应该如何把它拉回 “非零覆盖、但仍保守”的 selective conservative regime？

## 你必须在下面这些解释中做主次判断

你可以提出自己的判断，但请至少明确区分下面几层中哪个是主病因、哪个只是次病因：

1. `teacher positive density / utility target semantics`
   - strict `delta = oracle - host` 是否过稀
   - 是否需要更 dense 的 local utility target
   - 是否需要从 hard set-diff 改成 weighted / soft utility target
2. `gate / abstain objective`
   - 当前 gate 学成零覆盖或近零覆盖
   - checkpoint / calibration / selection metric 是否把模型推向全关
3. `assignment / commit objective`
   - 当前 row-wise CE + sparse delta edge supervision 是否天然鼓励 all-defer
4. `runtime operator contract`
   - current conservative runtime 是否压得过头
   - 还是 runtime 其实不是主因，因为这次 `budget_filtered_clusters` 和 `margin_filtered_pairs` 还没有真正成为主过滤项
5. `model family itself`
   - 当前 `set_predictor_v2` family 是否根本不适合 official ByteTrack

## 请你给唯一主方案和一个备选

我不要“可以都试试”。请你只给：

- 一个唯一主方案
- 一个备选

并说明：

- 为什么主方案排第一
- 为什么另外几种现在先不要做

## 额外硬要求：请直接给代码级重设计文档

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
- 它是继续保留 `set_predictor_v2` family、还是局部替换 head / target / runtime contract、还是彻底更换 family
- 在线注入点在哪
- 输入是什么
- 输出是什么
- 与 official ByteTrack host 的接口是什么

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

- strict `delta_utility` teacher 为什么会稀到只剩 `20` 个 positive clusters / `24` 个 delta commits
- 下一步是否应该继续用 hard `delta = oracle - host`，还是改成更 dense 的 utility target
- 是否应该从 row-level local rescue / weighted utility / soft gain 角度重写 target
- gate negatives 现在应如何构造
- checkpoint selection 是否应该从 `val_loss / gate_utility` 改成 coverage-aware selective utility
- runtime 是否应该继续保留:
  - max commits per cluster
  - replacement budget
  - margin filter
  - 更高或更低的 `min_committed_matches`
- 当前 `set_predictor_v2` family 是保留还是更换

4. 第一批实验必须只给一个 first-priority experiment

而且这一步必须是：

- `official ByteTrack host-only` vs `new redesign plugin`
- 同 detector / 同 checkpoint / 同 split / 同 evaluator / 同 pre-Hungarian injection 的 strict paired half-val experiment

不能再回到：

- baseline selection
- host migration
- internal host sweep

## 我当前自己的判断，你可以直接反驳

我当前本地判断是：

- 当前 official-host negative 并不证明 operator 方向应该 kill
- 但它已经证明：
  - old official-host target semantics 太宽，会学成 replacer
  - current strict `delta_utility` 又太稀，会学成 no-op
- 所以现在最像主病因的是：
  - target semantics / positive density 不对
  - gate / assignment objective 在当前极稀 supervision 下把模型推向全 defer
  - runtime 现在更像放大器，不像主病因
- 我不确定下一步应不应该优先：
  - 保留 family，只把 target 从 strict set-diff 改成更 dense utility target
  - 还是直接换一个更适合 selective coverage 的 family

如果你不同意我的判断，请直接反驳，并给出更强的主方案。

## 希望的回答格式

请直接按下面结构回答：

1. `管理级决策`
2. `为什么 current official-host redesign 会从 aggressive replacer 摆到 near no-op`
3. `唯一主方案`
4. `一个备选`
5. `文件级与 runner 级落点`
6. `唯一 first-priority experiment`
7. `当前不要做什么`

我要的是一份能直接指导下一轮工程改动的回答，不要泛泛空话。

