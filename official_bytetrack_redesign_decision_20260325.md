# official ByteTrack strict negative 之后的收束裁决与代码级重设计文档

## 1. 管理级决策

决策：`NARROW GO`

原因：

1. 这次 strict negative 不是“模块根本插不进去”。zero-shot official-host migration 那枪几乎没有介入：`eligible_clusters=7015`，但 `replaced_clusters=0`、`matched_dets=0`，因此它说明的是 host migration 失败，而不是 operator 概念无效。
2. official-host retrain 那枪则相反：模块大规模介入，`eligible_clusters=7025`，`replaced_clusters=5935`，`matched_dets=18532`，且 paired 指标显著变差：`HOTA -3.669 / AssA -6.496 / IDF1 -4.715 / MOTA -0.618 / IDSW +152`。这说明主问题不是“没接进去”，而是“训成了 aggressive replacer”。
3. 当前最强 family `HostConditionedLocalConflictSetPredictor` 在 internal host 上已拿到真实正号，因此现在不应先 kill family，而应先修正 official-host 训练语义与 runtime 约束。

结论：

- 继续这条论文主线。
- 但只允许做一次收束式 redesign。
- 这次 redesign 的目标不是“变强的 whole tracker”，而是把当前插件重新压回 `selective conservative operator` 的角色。

---

## 2. 当前 official-host negative 的主病因排序

### 主病因：`dataset / target semantics`

当前 teacher 语义是：

- 先用 GT-positive mask 做局部 Hungarian
- 再看 `matched_count >= min_committed_matches`
- 若通过，则整簇 `trigger_pass=1`
- `target_by_det` 直接监督为 `oracle_commit_pairs`

也就是说，当前监督表达的是：

- “这个 cluster 能不能形成一组 GT 合法提交”
- 而不是
- “在 official ByteTrack 当前 host 上，介入这个 cluster 是否真的有在线净收益”

这在 official host 上是致命错配。

当前 official-host stage1 数据中：

- `eligible_clusters = 557`
- `trigger_pass_clusters = 557`
- `trigger_fail_clusters = 0`

而且 replay 统计已经显示：

- `positive_groups = 46048`
- `rank_top1_correct_groups = 45901`
- `rank_top1_acc_positive = 0.9968`
- `recoverable_groups = 147`
- `recoverable_rate_among_positive = 0.00319`

这说明 official ByteTrack 的 first-stage 本地排序本来就已经几乎全对。当前 teacher 却仍然把绝大多数“host 本来就会做对”的 cluster 也标成可干预正例，于是模型自然会学成“只要 cluster eligible 就替换”。

我进一步核对了 `cluster_examples.jsonl`：

- 557/557 个样本 `trigger_pass` 全为 1
- 323/557 个 cluster，局部 host assignment 与当前 oracle target 完全相同
- 535/557 个 cluster，`oracle_commit - host_commit` 的增量提交数为 0
- 只有 22/557 个 cluster 存在任何正增量提交
- 只有 4/557 个 cluster 的正增量提交数 `>= 2`

这组数说明当前数据集的大部分样本，并不支持“应该替换 host”；它们只支持“这个 cluster 里存在一套 GT 合法匹配”。这正是 aggressive replacer 的主病因。

### 次主病因：`gate / abstain supervision`

这其实是主病因的直接表征。

因为 `trigger_pass` 在 official-host dataset 上退化成全正：

- `val_cluster_f1 = 1.0`
- `val_cluster_gate_precision_cal = 1.0`
- `val_cluster_gate_recall_cal = 1.0`
- `val_cluster_gate_coverage_cal = 1.0`

这些数字不代表 gate 真会 abstain，只代表 target 没有 negatives。于是 `cluster_commit_logit` 学到的是“eligible 即开”，不是“什么时候不要碰”。

### 次病因：`runtime operator constraints`

当前 runtime 过于常开，具体表现在：

- gate 只要过阈值就直接放行
- 没有 replacement budget
- 没有每簇最大提交数上限
- 没有基于 commit 置信度 / margin 的二次保守过滤
- 没有“只提交 host 的增量改正项”的运行时限制

因此一旦 gate 被训坏，runtime 会把错误成倍放大。

### 不是当前主病因：`model family itself`

当前不应该先判 `set_predictor_v2` family 不适合 official ByteTrack。

更准确的说法是：

- 这个 family 当前的 teacher 和 runtime 合同不适合 official ByteTrack
- 不是 backbone / token mixer / head family 先验上不适合

因此第一步不该换 family，而该先换 supervision semantics 与 runtime 约束。

---

## 3. 唯一主方案

### 方案名

`Utility-Gated Delta-Commit Set Predictor`

建议代码名：

- `set_predictor_v2_utilitygate`
- 中文语义：`效用门控的增量提交集预测器`

### 核心判断

主方案不是换 backbone，而是：

- **保留当前 `HostConditionedLocalConflictSetPredictor` family**
- **重写 official-host dataset target semantics**
- **把 cluster gate 从 `trigger_pass` 改成 `should_intervene` / `utility-positive`**
- **把 assignment supervision 从 `full oracle commit` 改成 `delta commit against host`**
- **再在 runtime 叠加更强的 conservative constraints**

### 为什么它排第一

因为当前 negative 的根因首先是 teacher 错了。

在 official ByteTrack 上，你不是要让模型学会“给 eligible cluster 都找一套更完整的匹配”，而是要让它学会：

- 哪些 cluster **不要碰**
- 哪些 cluster 只值得做**少量增量提交**
- 剩余一切继续 defer 回 official host

换句话说，当前该学的是：

- `intervene only when host-local first-stage is predictably wrong`

而不是：

- `replace whenever a GT-consistent cluster matching exists`

### 主方案的 teacher 语义

在 `build_local_conflict_set_predictor_dataset.py` 里，不再使用当前：

- `trigger_pass`
- `oracle_commit_pairs`
- `edge_is_oracle_commit`
- `target_by_det = oracle target`

改为构造以下新标签：

#### 1) host-local baseline assignment

对每个 eligible cluster，使用和 official ByteTrack first-stage 一致的本地 assignment surrogate：

- score: 当前 cluster 内的 `refined_score_raw` / runtime `dense_refined_scores`
- feasible mask: 与 runtime 一致的 score feasibility threshold
- solve: Hungarian without rewriting global host

得到：

- `host_pairs_local`

#### 2) oracle local assignment

保留现有 GT-positive Hungarian，得到：

- `oracle_pairs_local`

#### 3) delta commit teacher

定义：

- `delta_commit_pairs = oracle_pairs_local - host_pairs_local`

只把 **host 当前没选到，但 oracle 局部认为该选到** 的 pair 作为插件应提交的 pair。

其余 detection 默认 `defer`。

这一步非常关键。它把 teacher 从“替换 host 的完整解”改成“只做 host 的增量改正”。

#### 4) utility-style cluster label

定义：

- `cluster_should_intervene = 1` 当且仅当 `delta_commit_pairs` 带来正局部效用
- 否则 `cluster_should_intervene = 0`

第一版可以用最简 surrogate：

- `len(delta_commit_pairs) > 0`

但更推荐用一个带误配惩罚的局部 utility：

- `utility = gain_true_delta - alpha * added_fp_delta - beta * host_true_pairs_lost`
- `cluster_should_intervene = 1[utility > 0 and len(delta_commit_pairs) <= cap]`

其中第一轮可以先取保守实现：

- `gain_true_delta = len(delta_commit_pairs)`
- `added_fp_delta = 0`（因为 teacher 直接用 oracle-positive）
- `host_true_pairs_lost = 0`（delta teacher 不允许替换 host 已正确 pair）

也就是先上**最小增量 teacher**，不先做复杂全局 counterfactual。

### 主方案的 runtime 语义

runtime 仍保持：

- official ByteTrack first-stage pre-Hungarian injection
- partial commit
- residual defer back to host

但要新增三层保守约束。

#### A. utility gate 取代 trigger gate

当前 `cluster_commit_logit` 的语义改为：

- `cluster_utility_logit`
- 预测“这个 cluster 是否值得 intervention”

#### B. per-cluster commit cap

新增：

- `local_conflict_max_commits_per_cluster`

默认建议：

- 第一轮设为 `1` 或 `2`

因为 official host 下真正有净收益的 delta cluster 极少，不应该允许一簇一次性接管 5~8 个 det。

#### C. replacement budget

新增：

- `local_conflict_replacement_budget_ratio`
- 或 `local_conflict_max_replaced_clusters`

作用：

- 对整段评测过程中的 intervention coverage 设上限
- 防止 gate 一次校准失误后全局失控

第一轮建议：

- 使用简单的 `max_replaced_clusters_per_seq` 或 `budget_ratio`
- 先把 coverage 明确压低到远小于当前 7025/7025 gate-pass 的水平

#### D. stronger abstain path

新增：

- commit 置信度二次过滤
- 对被 gate 通过的 cluster，只有 top commit logits / margin 足够强的 pair 才允许提交
- 其他全部 defer 回 host

### 为什么第一轮不直接做 online paired outcome mining

这可以做，但不是 first-priority。

原因：

1. 你当前 operator 的合同就是 local、primary-only、pre-Hungarian。
2. 本轮最重要的是把 teacher 对齐到这个局部合同，而不是引入更复杂的全轨迹 credit assignment。
3. 先用 `host-local vs oracle-local delta` 做 teacher，已经足以把 “always replace” 修成 “rare selective intervention”。

因此第一轮主方案是：

- **先做 local utility / delta-commit teacher**
- 不先做全序列 paired outcome 反推

---

## 4. 一个备选

### 备选方案

`Runtime Hard-Conservative Wrapper on top of current v2`

即：

- 不先改模型 family
- 也不先重做 teacher
- 只在 runtime 上加硬保守壳：
  - 更高 gate threshold
  - replacement budget
  - 每簇最多 1~2 个提交
  - 更严格 `min_committed_matches` 或更高 pair margin
  - 更强 defer

### 为什么它只能做备选

因为它只修放大器，不修老师。

当前问题的根在于：

- 训练目标把“几乎所有 eligible cluster”都教成了可替换正例

如果不改数据语义，runtime 再保守，本质上还是在压一个被训坏的 replacer。它可能让指标回升，但通常很难给出稳定、可解释、可扩展的论文主故事。

因此这个方案只能作为：

- 主方案改完后仍然 coverage 偏大时的二次保险
- 或者你需要一个 1 天内可落地的 emergency patch

不应排在主方案前面。

---

## 5. 文件级与 runner 级落点

下面按文件给出具体改法。

### 文件 1：`code/scripts/build_local_conflict_set_predictor_dataset.py`

#### 为什么改

这是当前主病因所在文件。`_cluster_example()` 里定义了整个 teacher semantics。

#### 当前关键锚点

可 grep：

- `def _cluster_example(`
- `trigger_pass = int(matched_count >= int(min_committed_matches))`
- `oracle_commit_pairs = matched_pairs if trigger_pass else set()`
- `edge_is_oracle_commit`
- `target_by_det`

#### 当前逻辑

- 用 GT-positive Hungarian 生成 `matched_pairs`
- 用 `matched_count >= min_committed_matches` 生成 `trigger_pass`
- 如果 `trigger_pass=1`，整个 cluster 监督成 oracle commit

#### 修改目标

新增 teacher mode：

- `--teacher-mode {trigger_pass, delta_utility}`
- 默认改成 `delta_utility`

新增 cluster-level metadata：

- `host_pairs_local`
- `oracle_pairs_local`
- `delta_commit_pairs`
- `cluster_should_intervene`
- `cluster_utility_gain`
- `host_equals_oracle`
- `host_match_count`
- `oracle_match_count`
- `delta_commit_count`

新增 edge / det target：

- `edge_is_delta_commit`
- `target_by_det_delta` 或直接覆写 `target_by_det`

#### 修改后核心逻辑

1. 先构造 `host_score_sub`
   - 使用 cluster 内原始 `refined_score_raw`
   - feasible mask 与 official runtime 一致
2. 求解 `host_pairs_local`
3. 保留当前 `oracle_pairs_local`
4. 构造 `delta_commit_pairs = oracle_pairs_local - host_pairs_local`
5. 令：
   - `cluster_should_intervene = int(len(delta_commit_pairs) > 0)`
6. 令 `target_by_det` 只对 `delta_commit_pairs` 写 track idx，其余全设 `-1`
7. 令 `edge_is_oracle_commit` 不再作为主训练标签；新主标签变成 `edge_is_delta_commit`
8. summary.json 与 sequence_cluster_summary.csv 追加统计：
   - `utility_positive_clusters`
   - `utility_negative_clusters`
   - `host_equals_oracle_clusters`
   - `delta_commit_zero_clusters`
   - `delta_commit_positive_clusters`

#### 建议实现方式

局部改写，不新开脚本。

原因：

- 数据构造主干已经对齐当前 feature pack
- 只需替换 supervision semantics
- 不需要另起 dataset pipeline

---

### 文件 2：`code/scripts/train_local_conflict_set_predictor.py`

#### 为什么改

当前 trainer 把 cluster gate 学成了“eligible 即开”，且默认 model selection 偏向 `val_loss`。

#### 当前关键锚点

可 grep：

- `cluster_target = torch.tensor(float(sample["trigger_pass"])`
- `_cluster_gate_loss(`
- `fit_cluster_gate_calibration(`
- `select_cluster_gate_threshold(`
- `cluster_gate_select_metric`
- `model_selection_metric`

#### 当前逻辑

- cluster head 监督目标是 `trigger_pass`
- assignment / edge 目标是 oracle commit
- calibration 指标支持 `f0.5` / `utility`
- 但 dataset 全正时 calibration 失去意义
- model selection 当前走 `val_loss`

#### 修改目标

1. 把 cluster target 改成：
   - `sample["cluster_should_intervene"]`
2. edge loss 改为：
   - `edge_is_delta_commit`
3. assignment loss 改为：
   - delta commit target（非 delta 全部 defer）
4. calibration 默认改为 precision / utility 导向
5. model selection 不再默认 `val_loss`

#### 新增参数

- `--cluster-target-key cluster_should_intervene`
- `--edge-target-key edge_is_delta_commit`
- `--target-gate-coverage 0.02`（示例）
- `--coverage-penalty-weight 1.0`
- `--model-selection-metric val_cluster_gate_utility`
- `--cluster-gate-select-metric utility`
- `--cluster-gate-fp-weight` 保留，但默认更大

#### 修改后训练语义

- cluster head 学“值不值得 intervene”
- edge / assignment 学“若 intervene，仅提交哪些增量 pair”
- calibration 目标优先压低 false intervention

#### 额外建议

增加一个 coverage-aware 正则，不必改网络，只需在 trainer 里加：

- batch 内 gate 平均开启率如果高于 target coverage，额外罚 loss

这能把 official host 下的 intervention coverage 主动压低。

---

### 文件 3：`code/models/local_conflict_set_predictor.py`

#### 为什么改

模型 family 建议保留，但 head 语义要改清楚。

#### 当前关键锚点

可 grep：

- `class HostConditionedLocalConflictSetPredictor`
- `cluster_commit_logit`
- `edge_logits`
- `defer_logits`
- `build_dense_assignment_logits`

#### 当前逻辑

- 三个头：edge / defer / cluster_commit
- cluster head 被解释成 commit gate

#### 修改目标

最小改法：

- 保留 backbone 与 token 交互不动
- 将 `cluster_commit_logit` 语义重命名为 `cluster_utility_logit`
- 为兼容旧代码，可保留：
  - 输出字典里同时给 `cluster_commit_logit` 和 `cluster_utility_logit`
  - 二者值相同

可选增强：

- 新增 `edge_margin_logits` 或 pair confidence head
- 但这不是第一轮必须项

#### 推荐结论

- **保留 family，不换 backbone**
- 只改 cluster head 的语义与命名

---

### 文件 4：`code/third_party/ByteTrack/yolox/tracker/byte_tracker_local_conflict.py`

#### 为什么改

这是 runtime aggressive replacement 的直接放大器。

#### 当前关键锚点

可 grep：

- `def _get_local_conflict_plan(`
- `cluster_gate_prob`
- `solve_assignment_with_private_defer`
- `local_conflict_min_committed_matches`
- `replaced_clusters`
- `gate_filtered_clusters`
- `trigger_filtered_clusters`

#### 当前逻辑

- gate 过阈值就整簇放行
- Hungarian 直接输出 matched pairs
- 只要 `matched_pairs >= min_committed_matches` 就替换
- 无 budget、无 cap、无二次 pair filtering

#### 修改目标

新增 runtime 参数：

- `local_conflict_max_commits_per_cluster`
- `local_conflict_replacement_budget_ratio`
- `local_conflict_max_replaced_clusters`
- `local_conflict_min_commit_margin`
- `local_conflict_utility_thresh`（若与 gate thresh 分离）

新增诊断字段：

- `budget_filtered_clusters`
- `cap_filtered_matches`
- `margin_filtered_matches`
- `utility_filtered_clusters`

#### 修改后逻辑

1. `cluster_gate_prob` 解释为 `should_intervene_prob`
2. gate 通过后，不立即全量提交；先构造 candidate matched pairs
3. 对 candidate matched pairs 做二次保守过滤：
   - 按 pair logit / margin 排序
   - 截断到 `max_commits_per_cluster`
4. 检查 global budget：
   - 超预算则整簇 defer
5. 若过滤后提交数不足要求，则 defer
6. 其余 detection 一律 defer 给 host

#### 重要边界

- 不要改注入点
- 不要改 official host 的 residual Hungarian
- 仍保持 primary-only / pre-Hungarian / defer-to-host

---

### 文件 5：`code/third_party/ByteTrack/tools/track.py`

#### 为什么改

需要把新的 runtime conservative knobs 暴露到官方 paired runner。

#### 当前关键锚点

可 grep：

- `--use-local-conflict`
- `--local-conflict-min-committed-matches`
- `--local-conflict-cluster-gate-thresh`

#### 修改目标

新增 CLI：

- `--local-conflict-max-commits-per-cluster`
- `--local-conflict-replacement-budget-ratio`
- `--local-conflict-max-replaced-clusters`
- `--local-conflict-min-commit-margin`

建议全部给默认值，但默认要偏保守。

---

### 文件 6：`code/scripts/run_official_bytetrack_local_conflict_stage1_trainhalf.py`

#### 为什么改

这是 official trainhalf -> stage1 -> paired halfval 的主 pipeline，必须接入新 teacher 与新 runtime knobs。

#### 当前关键锚点

可 grep：

- `--min-committed-matches`
- `--cluster-gate-select-metric`
- `--cluster-gate-fp-weight`
- 调 `build_local_conflict_set_predictor_dataset.py`
- 调 `train_local_conflict_set_predictor.py`
- 调 `run_official_bytetrack_local_conflict_halfval_pair.py`

#### 修改目标

1. dataset builder 部分新增：
   - `--teacher-mode delta_utility`
2. trainer 部分新增：
   - utility gate training args
   - target coverage args
   - model selection 改成 `val_cluster_gate_utility`
3. pair eval 部分新增：
   - `--graph-max-commits-per-cluster`
   - `--graph-replacement-budget-ratio`
   - `--graph-min-commit-margin`
4. root summary.csv 追加字段：
   - utility positive/negative counts
   - plugin budget filtered clusters
   - plugin cap filtered matches

---

### 文件 7：`code/scripts/run_official_bytetrack_local_conflict_halfval_pair.py`

#### 为什么改

strict paired eval runner 需要把新 runtime 参数从命令行传进 official `track.py`。

#### 当前关键锚点

可 grep：

- `--graph-min-committed-matches`
- `--graph-cluster-gate-thresh`
- 组装 `track.py` command 的地方

#### 修改目标

新增 pass-through：

- `--graph-max-commits-per-cluster`
- `--graph-replacement-budget-ratio`
- `--graph-max-replaced-clusters`
- `--graph-min-commit-margin`

并把聚合 summary.csv / result.csv 中的新诊断字段写出来。

---

### 文件 8：`code/third_party/ByteTrack/yolox/evaluators/mot_evaluator.py`

#### 为什么改

如果你想在每序列诊断 JSON 里保留新的 budget / cap 统计，这里需要跟进聚合输出。

#### 当前关键锚点

可 grep：

- `write_tracker_diagnostics`
- `get_local_conflict_diagnostics`

#### 修改目标

这是轻改：

- 保持现有 diagnostics 写盘逻辑
- 若 tracker stats 字段扩展，这里无需大改，只需确认序列级 JSON 原样写出

因此这个文件不是主战场。

---

## 6. 工程问题的直接回答

### Q1. current official-host dataset 的 gate negatives 该如何构造？

答案：

- 不要再靠 `trigger_fail`
- 要把 **host-correct / no-benefit eligible clusters** 直接当作 gate negatives

具体定义：

- 若 `host_pairs_local == oracle_pairs_local`，则 negative
- 或更一般地，若 `delta_commit_pairs` 为空，则 negative

这些负样本现在正是官方 host 数据中的大头。

### Q2. current `trigger_pass` 是否应该被替换成 utility-style cluster label？

答案：**应该，而且要作为第一优先级改。**

建议：

- `trigger_pass` 保留为 debug metadata
- 但不再作为训练 gate 主标签
- 主标签改为 `cluster_should_intervene`

### Q3. 是否需要从 paired online outcome 反推 cluster-level intervention target？

答案：**第一轮不需要。**

先做：

- `local host vs local oracle delta` teacher

理由：

- 与 current operator 合同一致
- 实现成本低
- 直接打在当前主病因上

只有在这一步仍然 coverage 过高或指标仍负时，再考虑第二阶段的全轨迹 paired outcome mining。

### Q4. runtime 是否应该加入更严格 gate / budget / stronger abstain / 每簇最大提交数？

答案：**都应该加，但它们是辅助，不是主病因修复。**

优先级：

1. 更严格 cluster gate：是
2. replacement budget：是
3. stronger abstain path：是
4. 每簇最大提交数：是
5. 更高 `min_committed_matches`：谨慎，不作为第一主杠杆

### Q5. 当前 `set_predictor_v2` family 是保留还是更换？

答案：**保留。**

至少在这一轮 redesign 中：

- 不换 family
- 不做 whole new operator
- 不把失败归因为 backbone 架构本身

---

## 7. 唯一 first-priority experiment

只做这一个：

`official ByteTrack host-only vs utility-gated delta-commit plugin on strict MOT17 half-val paired protocol`

具体要求：

1. 数据仍来自 official ByteTrack trainhalf runtime dump
2. dataset builder 改为 `--teacher-mode delta_utility`
3. trainer 改为：
   - cluster target = `cluster_should_intervene`
   - edge target = `edge_is_delta_commit`
   - select metric = `utility`
   - model selection = `val_cluster_gate_utility`
4. runtime 打开：
   - `max_commits_per_cluster = 1 or 2`
   - replacement budget
   - stronger gate threshold
5. paired eval 仍然严格同：
   - same official host
   - same detector
   - same checkpoint
   - same split
   - same evaluator

这一步的成功判据，不只是最终 HOTA / IDF1 回正，还包括诊断必须出现：

- `gate_pass_clusters` 明显低于当前的 7025
- `replaced_clusters` 明显低于当前的 5935
- 插件覆盖率从“几乎全开”收缩为“小比例高精度 intervention”

---

## 8. 当前不要做什么

1. 不要重新讨论 baseline selection。
2. 不要把 `base_reid_da` 拉回论文主 carrier。
3. 不要再做 host migration。
4. 不要回到 row-local / full replacement / continuity。
5. 不要先换模型 family。
6. 不要把这次 negative 简化成“多训几轮”。
7. 不要先做全轨迹 paired outcome mining 作为第一步。
8. 不要先跑一轮大规模 runtime 网格搜索来代替 teacher 改造。

---

## 9. 最终收束

一句话总结：

当前 official-host strict negative 的主病因不是“插件 family 不行”，而是**teacher 语义把 official ByteTrack 上本该 abstain 的大多数 cluster 全部教成了 intervention positive**，再叠加过于常开的 runtime，最终把 `set_predictor_v2` 训成了 aggressive replacer。

因此下一步唯一最值钱的 redesign 是：

- **保留 `set_predictor_v2` family**
- **把训练目标改成 utility-gated delta-commit teacher**
- **再用 runtime budget / per-cluster cap / stronger abstain 把在线语义压回 selective conservative operator**

这条路比“先换 family”更对症，也更符合你当前已经固定的论文合同。
