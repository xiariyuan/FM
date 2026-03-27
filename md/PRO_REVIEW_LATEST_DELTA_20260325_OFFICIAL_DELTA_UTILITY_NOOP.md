# Pro Review Latest Delta (2026-03-25, Official ByteTrack Delta-Utility Mainline Near No-Op)

这份 delta 只记录一个新的收束点：

- `official ByteTrack` 仍然固定为论文主 carrier；
- `set_predictor_v2` 仍然固定为当前 stronger module family；
- 但在 strict official ByteTrack 主线上，我们现在已经拿到了 **两种相反失败形态**；
- 所以下一次向 Pro 的问题，不再是“这条线要不要继续”，而是：
  - 为什么 old official-host retrain 会学成 `aggressive replacer`
  - 为什么新的 `delta_utility + conservative runtime` 又会退化成 `near no-op`
  - 下一步唯一最值钱的 redesign，究竟该优先落在：
    - teacher densification / utility target redesign
    - gate-abstain objective
    - runtime contract
    - 还是 family 本身

## 1. 这次新增的关键事实

### 1.1 新主线实验已经跑完

运行根目录：

- `outputs/official_bytetrack_delta_utility_mainline_20260325_224718`

这条 run 不是 smoke，也不是局部脚本验证，而是完整 official ByteTrack 主线：

- official train-half dump
- rebuilt `delta_utility` dataset
- stage1 retrain
- strict half-val paired eval

### 1.2 这次不是 aggressive 了，而是几乎不提交

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

这组数和最早那枪 strict official zero-shot negative 几乎同型，不是大幅恶化，而是接近 no-op。

### 1.3 plugin arm 诊断证明：不是没触发 gate，而是没有形成任何 commit

关键文件：

- `outputs/official_bytetrack_delta_utility_mainline_20260325_224718/02_official_halfval_pair/01_host_plus_plugin/summary.csv`

核心数字：

- `eligible_clusters = 7015`
- `gate_pass_clusters = 3245`
- `gate_filtered_clusters = 3770`
- `trigger_filtered_clusters = 3245`
- `replaced_clusters = 0`
- `matched_dets = 0`
- `deferred_dets = 23315`
- `blocked_tracks = 0`

这说明当前失败形态不是：

- gate 全关
- integration bug
- 运行时根本没进入 plugin

而是：

- 有相当多 cluster 通过了 gate
- 但这些 cluster 最终没有形成任何满足 runtime commit 条件的匹配
- 所以整个 operator 在线行为退化成 “pass gate but still defer everything”

### 1.4 新 teacher 的确把正样本压回来了，而且压得非常狠

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

对比 old official-host retrain：

- old:
  - `trigger_pass_clusters = 557`
  - `trigger_fail_clusters = 0`
  - failure = aggressive replacer
- new delta_utility:
  - `trigger_pass_clusters = 20`
  - `trigger_fail_clusters = 537`
  - failure = near no-op

这已经把问题收得非常清楚：

- old teacher 太宽，学成几乎逢簇必替
- new teacher 太稀，学成几乎没有可提交覆盖

### 1.5 stage1 训练也支持“太稀 -> 学成全 defer / 全关”这个解释

关键文件：

- `outputs/official_bytetrack_delta_utility_mainline_20260325_224718/01_stage1/summary.csv`

核心数字：

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

这说明：

- 现在不是数值炸了
- 也不是 trainer 崩了
- 而是当前目标下，最容易学到的策略就是：
  - row-wise 基本全 defer
  - edge 不提交
  - gate utility 接近零覆盖

## 2. 现在已经固定的新判断

### 2.1 baseline selection 已经彻底不是问题

不要再回到：

- official ByteTrack 要不要当主线
- byter / botsort / strongsort 应该先跑谁

当前最有价值的问题已经不在 baseline。

### 2.2 当前也还不是 family kill 的时点

因为现在我们拿到了两个相反 failure modes：

1. old official-host retrain:
   - 明显是 `too open`
   - `replaced_clusters = 5935`
2. new delta_utility mainline:
   - 明显是 `too closed`
   - `replaced_clusters = 0`

这更像：

- current official-host supervision / runtime contract 还没有把 `set_predictor_v2` 调到正确的 selective coverage 区间

而不是：

- `set_predictor_v2` family 一定不适合 official ByteTrack

### 2.3 下一次问 Pro 时，问题必须严格收窄

现在不要再问开放式问题，例如：

- 要不要重新设计更强大模块
- 要不要换 baseline
- 要不要继续扫 host

现在真正要问的是：

> 在 fixed official ByteTrack + fixed plugin contract 下，如何把 operator 从 `too open` 和 `too closed` 两个极端之间，拉回到 “小覆盖但非零提交”的 selective conservative regime？

## 3. 建议下一次向 Pro 必须强制回答的层次

下一次不要让 Pro 再泛泛列可能性，而是要强制他在下面几层里做主次判断：

1. `teacher positive density / utility target semantics`
   - strict `delta = oracle - host` 是否过稀
   - 是否需要更 dense 的 local utility teacher
   - 是否需要从 hard set-diff 改成 weighted / soft utility target
2. `gate / abstain objective`
   - 当前 gate 学到全关或近全关
   - 是否需要不同的 gate label、class balance、checkpoint selection
3. `assignment / commit objective`
   - 当前 row-wise CE + sparse delta edges 是否把最优策略推成全 defer
   - 是否需要明确鼓励“非零但保守提交”
4. `runtime contract`
   - 当前 conservative constraints 是否压过头
   - 还是 runtime 其实不是主因，因为 `margin_filtered_pairs` / `budget_filtered_clusters` 还没真正起作用

## 4. 这次向 Pro 真正该问什么

下一次 prompt 不应再问：

- 是否继续这条线
- 是否要换 tracker family
- 是否重开 baseline selection

而应只问：

1. 为什么 old official-host retrain 是 `aggressive replacer`
2. 为什么 new `delta_utility` mainline 是 `near no-op`
3. 下一步唯一 first-priority redesign 是什么
4. 这一步应该具体改哪些文件、哪些函数、哪些 target、哪些 runner

## 5. 需要 Pro 直接处理的当前事实张力

当前最重要的证据张力只有一句话：

- old official-host teacher 太宽，导致 `5935` 个 replaced clusters
- new delta-utility teacher 太稀，导致 `0` 个 replaced clusters

所以现在最值钱的问题不是：

- “再把 gate 调一点”

而是：

- “怎样定义一个 **既不是 full replacement、也不是 strict zero-delta-only** 的 official-host selective intervention target”

