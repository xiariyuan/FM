
# official ByteTrack 主线：从 `aggressive replacer` / `near no-op` 拉回 selective conservative regime 的代码级重设计文档
日期：2026-03-25

## 1. 管理级决策

决策：`NARROW GO`

原因：

1. 这条线现在已经不是 baseline 问题，也不是 integration 问题，而是 **official-host supervision 与 selective operator 合同不对齐**。
2. 当前已经出现两个相反 failure modes：
   - old official-host retrain：`aggressive replacer`
   - new `delta_utility + conservative runtime`：`near no-op`
3. 这说明当前问题更像是 **target semantics / objective geometry / coverage selection** 没把 `set_predictor_v2` 拉到“非零覆盖但仍保守”的区间，而不是 family 本身天然不适合。
4. 因为 strict paired official ByteTrack protocol 已经打通，下一步最值钱的是 **只做一次收束式 redesign**，直接对 `official ByteTrack host-only` vs `new redesign plugin` 做 strict half-val paired experiment。

结论：

- 不 kill 方向
- 不换论文主 carrier
- 不换 whole family
- 下一步只允许做一次 **target/objective 主导** 的 redesign

---

## 2. 当前问题的再诊断：为什么会从 `aggressive replacer` 摆到 `near no-op`

### 2.1 old official-host retrain 为什么会学成 aggressive replacer

旧 teacher 的本质是：

- 只要 cluster 内存在 GT-合法局部匹配
- 且 `matched_count >= min_committed_matches`
- 就把整个 cluster 监督成 `trigger_pass=1`
- 再把 `target_by_det` 监督成 `oracle_commit_pairs`

这等价于在教模型：

- “存在一组 GT 合法解 = 应该介入”

而不是：

- “在 official ByteTrack 当前 host 上，这个 cluster 的 intervention 是否有净收益”

在 official ByteTrack 上，这会把大多数 **host 本来就够好 / 不值得碰** 的 cluster 也当作正例，于是自然学成 almost-always-open / aggressive replacer。

### 2.2 new strict `delta_utility` 为什么又会摆到 near no-op

当前 `delta_utility` 的关键定义是：

- `host_pairs_local`
- `oracle_pairs_local`
- `delta_commit_pairs = oracle_pairs_local - host_pairs_local`
- `cluster_should_intervene = int(len(delta_commit_pairs) > 0)`

这一步修掉了“太宽”，但又引入了更深的一层错配：

#### (A) 它不是单纯“太稀”，而是 **语义错在 hard set-diff 只看 added oracle pairs**

从最新 `cluster_summary.csv` 可见：

- `eligible_clusters = 557`
- `cluster_should_intervene = 20`
- `delta_committed_matches = 24`
- `host_equals_oracle = 285`
- `host_committed_matches > oracle_committed_matches` 的 cluster 有 `269`
- `oracle_committed_matches > host_committed_matches` 的 cluster 是 `0`
- `delta_commit_count` 分布：
  - `0: 537`
  - `1: 17`
  - `2: 2`
  - `3: 1`
- `oracle_committed_matches - host_committed_matches` 分布：
  - `-5: 1`
  - `-4: 19`
  - `-3: 31`
  - `-2: 55`
  - `-1: 163`
  - `0: 288`

这说明：

- 当前 positives 几乎都不是 “oracle 比 host 多恢复了很多 pair”
- 它们大多是 **swap / edit** 场景：host 已经提交了同数或更多 pair，但部分 pair 是错的
- 然而 `delta_commit_pairs = oracle - host` 只保留 oracle-exclusive added pair，却**不表达被 oracle 替换掉的 host-wrong rows**

也就是说，teacher 只告诉模型：

- “这里新增这 1 个 pair”

却没有告诉模型：

- “这一簇真正的问题是 host 的哪些 row action 应该被编辑 / displaced / 改成 defer / 改成别的 pair”

这会让 target 变成又稀又偏：大多数正例只剩 1 个孤立 added pair。

#### (B) 当前 assignment objective 在这种 target 下天然鼓励 all-defer

在当前训练里，assignment 仍是 dense row-wise CE：

- 每个 detection row 都要在 `[tracks + defer]` 上做分类
- 但 positives 里真正需要改的 row 很少
- 其余 rows 不是 “不重要”，而是被当作大量 easy defer/no-change 行混在一起

对当前 official-host `delta_utility` 数据，positives 只有 20 个 cluster、24 个 positive delta commits，其中：

- `17` 个 positive cluster 只含 `1` 个 delta commit
- `2` 个 cluster 含 `2` 个 delta commit
- `1` 个 cluster 含 `3` 个 delta commit

这意味着当前 dense CE 最容易学到的策略就是：

- row-wise 全 defer
- edge 不拉高
- gate 即使偶尔开，也没有任何 pair 会过 commit 约束

也正因此你在 runtime 上看到的是：

- `gate_pass_clusters = 3245`
- `replaced_clusters = 0`
- `matched_dets = 0`

不是 gate 没开，而是 **gate 之后 assignment / commit 没有可执行 pair**。

### 2.3 所以新的主病因排序

#### 主病因：`teacher positive density / utility target semantics`
但必须精确说成：

- **hard `delta = oracle - host` 不只是过稀，更关键是它没有表达 local edit utility**
- 它只表达 added pairs，不表达 decision-bearing rows / displaced host-wrong rows

#### 次主病因：`assignment / commit objective`
- 当前 dense row CE + very sparse delta edge supervision
- 会把最优策略推向 `all defer / all close`

#### 第三病因：`gate / abstain objective`
- gate 依然重要，但这次不是第一病因
- 因为已经有 `3245` 个 cluster 过 gate，却 0 commit
- 所以 gate calibration 不是当前 zero-submit 的主杠杆

#### 第四病因：`runtime operator contract`
- runtime 现在是放大器，不是主病因
- 这次 near no-op 里，真正致命的不是 budget/margin/cap，而是 **根本没有 pair 形成 commit**
- 因此 runtime 先不要再被当作第一杠杆

#### 不是主病因：`model family itself`
- 当前 `HostConditionedLocalConflictSetPredictor` family 不应先换
- 真正该换的是 target / objective，而不是 backbone family

---

## 3. 唯一主方案

## 方案名

### 论文/概念名
`Edit-Utility Gated Selective Commit Predictor`

### 代码建议名
`set_predictor_v2_editutility`

## 核心原则

- **保留 `HostConditionedLocalConflictSetPredictor` family**
- **不改 online 注入点**
- **不改 official ByteTrack host residual Hungarian**
- **不走 whole-tracker rewrite**
- **主改 teacher semantics + assignment objective**
- runtime 只做轻量保守辅助，不再当主杠杆

## 3.1 这一步到底改什么

不是回退到 old broad teacher，也不是继续 strict hard set-diff。

而是把 target 改成：

### `host-local vs oracle-local` 的 **edit-aware weighted utility target**

关键点：

1. 不再只抽 `oracle - host` 的 added pairs
2. 改为先定义每个 detection row 的：
   - `host_action`
   - `oracle_action`
3. 只要某个 row 上 `host_action != oracle_action`，这个 row 就是 **decision-bearing row**
4. gate 的 utility 不再由 `len(delta_commit_pairs) > 0` 定义，而由 **这些 decision-bearing rows 的加权 edit gain** 定义
5. assignment loss 不再在所有 rows 上做 dense CE，而是只在 **decision-bearing rows** 上做 masked CE / weighted CE

## 3.2 为什么它排第一

因为你当前的问题不是：

- gate 太松还是太紧
- runtime cap 多一点还是少一点

而是：

- teacher 现在只给了极少数 added pairs，却没有把“这簇哪里需要编辑 host 决策”讲给模型

换句话说，当前应该学的是：

- **selective edit against host**

不是：

- full replacement
- strict zero-delta-only added pairs

这正好落在你现在最想要的区间：

- 非零覆盖
- 仍然保守
- 不越过 plugin-style partial commit contract

## 3.3 新 teacher 的定义

在每个 eligible cluster 内，继续保留：

- `host_pairs_local`
- `oracle_pairs_local`

但新增 row-level action view：

### 对每个 det row 定义

- `host_action_by_det[d]`
  - 若 host local 选了某个 track，则为该 `track_idx`
  - 否则为 `DEFER`
- `oracle_action_by_det[d]`
  - 若 oracle local 选了某个 track，则为该 `track_idx`
  - 否则为 `DEFER`

### 定义 decision-bearing rows

- `row_edit_mask[d] = 1` 当且仅当 `host_action_by_det[d] != oracle_action_by_det[d]`
- 否则为 `0`

### 定义 row action type

建议记录：

- `row_action_type = keep | add_commit | reassign_commit | force_defer`

其中：

- `add_commit`: host=DEFER, oracle=track
- `reassign_commit`: host=track_a, oracle=track_b
- `force_defer`: host=track, oracle=DEFER
- `keep`: host=oracle

### 定义 edit-aware commit pairs

- `edit_commit_pairs = {(d, oracle_track[d]) | row_edit_mask[d] = 1 and oracle_action_by_det[d] != DEFER}`

### 定义 cluster utility gain

第一版建议：

- `cluster_edit_gain = sum(row_gain[d])`

其中：

- `row_gain = +1.0` for `add_commit`
- `row_gain = +1.0` for `reassign_commit`
- `row_gain = +0.5` for `force_defer`
- `row_gain = 0.0` for `keep`

再加一个轻量保守罚项：

- `cluster_edit_cost = lambda_commit * len(edit_commit_pairs)`

最终：

- `cluster_utility_gain = cluster_edit_gain - cluster_edit_cost`

建议默认：

- `lambda_commit = 0.15 ~ 0.25`

### 定义新的 gate label

- `cluster_should_intervene = 1[cluster_utility_gain >= min_cluster_gain]`

建议默认：

- `min_cluster_gain = 0.75`

这样做的效果：

- 不再只剩 strict hard delta 的 20 个 positives
- 但也不会回到 old teacher 的 557/557 全正
- 它会把 “真实需要 edit 的 cluster” 提回一个 **小而非零** 的正样本区间

## 3.4 新 assignment / commit objective

### 关键改变：masked decision-row CE

当前不要再让所有 rows 都参与 equal-weight dense CE。

改成：

- 只对 `row_edit_mask == 1` 的 rows 计算主 assignment loss
- `row_edit_mask == 0` 的 rows：
  - 不进入主 assignment CE
  - 或只加极小权重 `w_keep = 0.05`

### 建议 loss

#### cluster gate loss
- BCE on `cluster_should_intervene`
- + target-coverage regularizer

#### row assignment loss
- masked CE on `target_by_det_edit`
- `target_by_det_edit[d] = oracle_action_by_det[d]`
- only if `row_edit_mask[d] == 1`

#### edge loss
- focal / BCE on `edge_is_edit_commit`
- 只在与 decision-bearing rows 相连的 edges 上计算
- positive weight 明显放大

### 为什么这一步关键

因为当前 near no-op 不是单纯 gate 关掉，而是：

- gate 开了
- 但 assignment 仍然喜欢 all-defer

masked decision-row CE 的作用就是：

- 让“需要编辑的 row”真正成为优化重心
- 不让 95%+ 的 no-change rows 把 loss 几何推向 all-defer

## 3.5 runtime contract：保留，但只做辅助

保留以下 knobs：

- `max_commits_per_cluster`
- `replacement_budget`
- `margin filter`
- `min_committed_matches = 1`

但它们的角色降为：

- **保险丝**
- 不是主病因修复器

### 默认建议

- `max_commits_per_cluster = 1`
- `replacement_budget_ratio = 0.05`
- `max_replaced_clusters = 0`（不用额外 hard cap 时可不启）
- `min_commit_margin = 0.05`
- `min_committed_matches = 1`

解释：

- 当前你要的是小覆盖、非零提交
- 所以 `max_commits_per_cluster` 要保守
- 但 `margin` 不要继续设太高，否则又会和新训练目标一起压回 no-op
- `min_committed_matches` 不应上调，因为当前很多 edit cluster 天生就是 single-edit

---

## 4. 一个备选

备选方案：

### `Action-Balanced Masked Delta-Utility`
也就是：

- 继续保留 current `delta_utility`
- 不改 gate label 主定义
- 但训练时：
  - 引入 `row_edit_mask`
  - 对 assignment 只在 changed rows 上做 masked CE
  - 对 positives / edit rows / positive edges 大幅 reweight
  - checkpoint selection 改成 coverage-aware selective utility

### 为什么它只是备选

因为它修的是 objective geometry，但 teacher 语义仍然是 strict hard set-diff。

换句话说，它可能让 current no-op 好转，但它仍然没有正面回答：

- swap / displaced host-wrong rows 到底该如何在 target 中被表达

所以：

- 主方案：改成 edit-aware weighted utility target
- 备选：不改 target semantics，只改 masked loss / reweight / selection

---

## 5. 文件级与 runner 级落点

## 文件 1：`scripts/build_local_conflict_set_predictor_dataset.py`

### 为什么改
当前主病因就在这里：teacher semantics 由 `_cluster_example()` 定义。

### 当前关键锚点
可 grep：

- `def _cluster_example(`
- `teacher_mode`
- `host_pairs_local`
- `oracle_pairs_local`
- `delta_commit_pairs`
- `cluster_should_intervene`
- `target_by_det`
- `target_by_det_delta`
- `edge_is_delta_commit`

### 当前逻辑
在 `delta_utility` 模式下：

- `delta_commit_pairs = oracle - host`
- `cluster_should_intervene = 1[len(delta_commit_pairs) > 0]`
- `target_by_det = target_by_det_delta`
- 其余 rows 默认 `defer`

### 修改后应该变成什么

新增 teacher mode：

- `--teacher-mode edit_utility`

新增字段：

- `host_action_by_det`
- `oracle_action_by_det`
- `row_edit_mask`
- `row_action_type`
- `target_by_det_edit`
- `edge_is_edit_commit`
- `cluster_edit_gain`
- `cluster_edit_cost`
- `cluster_utility_gain`
- `edit_commit_pairs`
- `num_edit_rows`
- `num_add_commit_rows`
- `num_reassign_rows`
- `num_force_defer_rows`

### 具体实现

在 `_cluster_example()` 内：

1. 继续保留：
   - `host_pairs_local`
   - `oracle_pairs_local`
2. 构造：
   - `host_action_by_det`（长度=`num_dets`，值域=`track_idx or -1`）
   - `oracle_action_by_det`
3. 计算：
   - `row_edit_mask[d] = int(host_action_by_det[d] != oracle_action_by_det[d])`
4. 分类每个 row 的 `row_action_type`
5. 构造：
   - `edit_commit_pairs`
6. 计算：
   - `cluster_edit_gain`
   - `cluster_edit_cost`
   - `cluster_utility_gain`
   - `cluster_should_intervene`
7. 对 `teacher_mode == "edit_utility"`：
   - `target_by_det = target_by_det_edit`
   - `edge target = edge_is_edit_commit`
   - `trigger_pass` 保留为 debug，不再作为主标签

### 是局部改写还是新增模块
- **局部改写**
- 不需要新脚本
- 直接扩展现有 `teacher_mode`

---

## 文件 2：`scripts/train_local_conflict_set_predictor.py`

### 为什么改
当前 near no-op 的第二主病因在这里：dense row CE 在极稀 target 下把模型推向 all-defer。

### 当前关键锚点
可 grep：

- `cluster_target = torch.tensor(float(sample.get("cluster_should_intervene", sample["trigger_pass"])))`
- `target_by_det`
- `edge_is_delta_commit`
- `_cluster_gate_loss(`
- `fit_cluster_gate_calibration(`
- `select_cluster_gate_threshold(`
- `cluster_gate_select_metric`
- `model_selection_metric`

### 修改后应该变成什么

新增参数：

- `--assignment-row-mask-key row_edit_mask`
- `--edge-target-key edge_is_edit_commit`
- `--cluster-target-key cluster_should_intervene`
- `--target-gate-coverage 0.03`
- `--coverage-penalty-weight 1.0`
- `--keep-row-loss-weight 0.05`
- `--edit-row-loss-weight 1.0`
- `--edit-edge-positive-weight 8.0`
- `--model-selection-metric val_selective_utility_targetcov`
- `--cluster-gate-select-metric bounded_utility`

### 训练逻辑修改

#### cluster loss
- 用 `cluster_should_intervene`
- 不再回退到 `trigger_pass`

#### assignment loss
- 主 CE 只在 `row_edit_mask == 1` 的 rows 上计算
- `keep rows` 只加极小权重或不参与

#### edge loss
- 用 `edge_is_edit_commit`
- 只在 `row_edit_mask` 覆盖到的 decision rows 附近计算
- positive weight 拉高

#### coverage-aware regularizer
- 若 batch gate 开启率低于 `min_target_gate_coverage`，加 penalty
- 若高于 `max_target_gate_coverage`，也加 penalty

建议 coverage band：

- `[0.02, 0.08]`

这样避免模型再次掉到零覆盖。

### checkpoint selection 修改

不要再用：

- `val_loss`
- 或纯 `val_cluster_gate_utility`

改为新增：

- `val_selective_utility_targetcov`

定义建议：

- 只在 `gate_coverage` 落在 target band 内时比较 utility
- 超带宽的 checkpoint 直接扣分

### 是局部改写还是新增模块
- **局部改写**
- 训练主循环与 calibration 逻辑都需要改
- 但 family 和数据加载主骨架保留

---

## 文件 3：`models/local_conflict_set_predictor.py`

### 为什么改
family 保留，但 head 语义要和新训练目标对齐。

### 当前关键锚点
可 grep：

- `class HostConditionedLocalConflictSetPredictor`
- `cluster_commit_logit`
- `cluster_utility_logit`
- `edge_logits`
- `defer_logits`
- `build_dense_assignment_logits`

### 修改后应该变成什么

**第一轮不换 family，也不强行加新 backbone**

建议最小修改：

1. 保留：
   - token encoder
   - edge head
   - defer head
   - cluster head
2. 只在输出字典中标准化语义：
   - `cluster_utility_logit` 作为主 key
   - `cluster_commit_logit` 仅作为兼容 alias
3. 可选新增一个非常轻量的：
   - `row_edit_logit`
   - 但第一轮不是必须

### 建议结论
- **保留 family**
- 第一轮先不加新 row head
- 先靠 teacher + masked loss 把 regime 拉回来

---

## 文件 4：`third_party/ByteTrack/yolox/tracker/byte_tracker_local_conflict.py`

### 为什么改
runtime 不是主病因，但仍需要保留保守保险丝。

### 当前关键锚点
可 grep：

- `def _get_local_conflict_plan(`
- `cluster_gate_prob`
- `solve_assignment_with_private_defer`
- `local_conflict_min_committed_matches`
- `local_conflict_max_commits_per_cluster`
- `local_conflict_replacement_budget_ratio`
- `local_conflict_min_commit_margin`
- `replaced_clusters`
- `trigger_filtered_clusters`

### 当前逻辑
这些保守 knobs 基本已经接好了。

### 修改后应该变成什么

第一轮只做两类轻改：

1. **保持 current contract 不变**
   - 仍是 pre-Hungarian
   - partial commit
   - defer to host
2. **调默认值，但不重写逻辑**
   - `max_commits_per_cluster` 默认 `1`
   - `replacement_budget_ratio` 默认 `0.05`
   - `min_commit_margin` 默认 `0.05`
   - `min_committed_matches` 继续 `1`

### 额外新增诊断（若尚未写出）
- `decision_row_clusters`
- `clusters_with_any_candidate_pairs`
- `clusters_failed_after_assignment`
- `clusters_failed_after_cap`
- `clusters_failed_after_margin`
- `clusters_failed_after_budget`

目的：

- 下一轮若仍 no-op，要一眼看出死在 gate、assignment、margin 还是 budget

### 是局部改写还是新增模块
- **局部改写**
- 不换 runtime plan 框架

---

## 文件 5：`third_party/ByteTrack/tools/track.py`

### 为什么改
CLI 暴露与默认值同步。

### 当前关键锚点
可 grep：

- `--use-local-conflict`
- `--local-conflict-min-committed-matches`
- `--local-conflict-cluster-gate-thresh`
- `--local-conflict-max-commits-per-cluster`
- `--local-conflict-replacement-budget-ratio`
- `--local-conflict-min-commit-margin`

### 修改后应该变成什么
- 如参数已经存在，仅调整默认值与 help 文案
- help 中明确：
  - `max-commits-per-cluster` 是 conservative selective cap
  - `replacement-budget-ratio` 是 global coverage fuse

---

## 文件 6：`scripts/run_official_bytetrack_local_conflict_stage1_trainhalf.py`

### 为什么改
这是 official trainhalf -> dataset -> stage1 -> paired eval 的主 pipeline。

### 当前关键锚点
可 grep：

- `--teacher-mode`
- `--cluster-gate-select-metric`
- `--model-selection-metric`
- `--graph-max-commits-per-cluster`
- `--graph-replacement-budget-ratio`
- `--graph-min-commit-margin`

### 修改后应该变成什么

默认改成：

- `--teacher-mode edit_utility`
- `--cluster-gate-select-metric bounded_utility`
- `--model-selection-metric val_selective_utility_targetcov`
- `--graph-max-commits-per-cluster 1`
- `--graph-replacement-budget-ratio 0.05`
- `--graph-min-commit-margin 0.05`

并追加 trainer 新参数：

- `--assignment-row-mask-key row_edit_mask`
- `--keep-row-loss-weight 0.05`
- `--edit-row-loss-weight 1.0`
- `--edit-edge-positive-weight 8.0`
- `--target-gate-coverage-min 0.02`
- `--target-gate-coverage-max 0.08`
- `--coverage-penalty-weight 1.0`

---

## 文件 7：`scripts/run_official_bytetrack_local_conflict_halfval_pair.py`

### 为什么改
strict paired runner 要把新 regime 的参数与 summary 写干净。

### 当前关键锚点
可 grep：

- `--graph-min-committed-matches`
- `--graph-cluster-gate-thresh`
- 组装 `track.py` command 的地方

### 修改后应该变成什么

新增 / 保留 pass-through：

- `--graph-max-commits-per-cluster`
- `--graph-replacement-budget-ratio`
- `--graph-max-replaced-clusters`
- `--graph-min-commit-margin`

新增 summary 字段：

- `teacher_mode`
- `assignment_row_mask_key`
- `target_gate_coverage_min`
- `target_gate_coverage_max`
- `clusters_with_any_candidate_pairs`
- `clusters_failed_after_assignment`
- `clusters_failed_after_margin`
- `clusters_failed_after_budget`

---

## 文件 8：`third_party/ByteTrack/yolox/evaluators/mot_evaluator.py`

### 为什么改
不是主战场，只要 diagnostics 能透传。

### 当前关键锚点
可 grep：

- `write_tracker_diagnostics`
- `get_local_conflict_diagnostics`

### 修改后应该变成什么
- 确认新的 tracker diagnostic 字段原样写出
- 不需要大改 evaluator 主逻辑

---

## 6. 工程问题的直接回答

### Q1. strict `delta_utility` teacher 为什么会稀到只剩 20 个 positive clusters / 24 个 delta commits？

不是单纯“official host 太强”这一个原因。

更精确地说：

1. official ByteTrack local host 确实已经很强；
2. 但更关键的是当前 target 定义为 `oracle - host` 的 **hard set-diff**；
3. 在当前数据里，positives 大多是 **swap / edit**，不是 “oracle simply adds more pairs than host”；
4. 所以 hard set-diff 只留下 oracle-exclusive added pairs，把被替换掉的 host-wrong rows 丢了；
5. 结果就是：
   - positives 极少
   - 且大多只剩 1 个单点 commit
   - dense CE 最终偏向全 defer

### Q2. 下一步是否应该继续用 hard `delta = oracle - host`？

**不应该。**

下一步应该改成：

- `edit-aware weighted local utility target`

也就是：

- 不再只看 pair set difference
- 改看 row action 的 edit utility

### Q3. 是否应该从 row-level local rescue / weighted utility / soft gain 角度重写 target？

**应该。并且这是主方案的核心。**

但不是退回 row-local rerank 那条死路线。

这里说的 row-level，是：

- 用 row action 差异来定义 **哪些 row 真正是 decision-bearing rows**
- 然后仍在 cluster-level operator 里做 selective partial commit

不是把方法改回 row-local 方法本身。

### Q4. gate negatives 现在应如何构造？

应该分两类：

1. `keep negatives`
   - `host_action == oracle_action`
2. `no-benefit negatives`
   - 有局部差异，但 `cluster_utility_gain < min_cluster_gain`

也就是说：

- 不再只靠 `trigger_fail`
- 也不再只靠 `delta empty`
- 而是用 **edit-aware utility** 定义 gate negatives

### Q5. checkpoint selection 是否应该从 `val_loss / gate_utility` 改成 coverage-aware selective utility？

**应该。**

建议新增：

- `val_selective_utility_targetcov`

只有当 gate coverage 落在目标带宽内，checkpoint 才有资格竞争最优。

### Q6. runtime 是否应该继续保留这些 conservative knobs？

- `max commits per cluster`：**保留**
- `replacement budget`：**保留**
- `margin filter`：**保留，但默认放松到 0.05**
- `min_committed_matches`：**保留为 1，不要上调**

因为：

- 当前 many useful edits 本来就是 single-edit
- 如果再把 `min_committed_matches` 提高，会再次把 regime 压回 no-op

### Q7. 当前 `set_predictor_v2` family 是保留还是更换？

**保留。**

这次 first-priority redesign 不换 family。

---

## 7. 唯一 first-priority experiment

只做这一个：

### `official ByteTrack host-only` vs `set_predictor_v2_editutility`
在 **strict MOT17 half-val paired protocol** 上做 paired experiment

要求：

- same detector
- same checkpoint
- same split
- same evaluator
- same pre-Hungarian injection
- host-only 与 plugin 只差：
  - `teacher_mode = edit_utility`
  - masked decision-row assignment loss
  - coverage-aware checkpoint selection
  - conservative runtime defaults (`max_commits=1`, `budget=0.05`, `margin=0.05`, `min_committed_matches=1`)

### 成功判据

这一步不要求一下子大幅正号，但必须同时满足：

1. `replaced_clusters > 0`
2. `matched_dets > 0`
3. coverage 明显小于 old aggressive retrain
4. 不回到 current near no-op
5. paired metrics 至少不再出现大负号
6. 最好 sequence 级诊断里能看到：
   - intervention coverage 落在小比例区间
   - single-edit clusters 开始出现非零提交

---

## 8. 当前不要做什么

1. 不要重新讨论 baseline selection
2. 不要把 `base_reid_da` 推回论文主 carrier
3. 不要重开 row-local / full replacement / continuity
4. 不要建议 host migration
5. 不要做 whole-tracker rewrite
6. 不要把主方案切成“只调 runtime”
7. 不要先换 `set_predictor_v2` family
8. 不要再把问题简化成“多训几轮”
9. 不要先做全轨迹 paired outcome mining
10. 不要先做大规模 runtime 网格搜索

---

## 9. 最终收束

一句话结论：

当前 official ByteTrack 主线的下一步，最值钱的 redesign 不是继续 strict hard `delta = oracle - host`，也不是只调 runtime，而是：

- **保留 `set_predictor_v2` family**
- **把 teacher 从 hard pair set-diff 改成 edit-aware weighted utility target**
- **把 assignment 从 dense all-row CE 改成 masked decision-row CE**
- **再保留轻量 conservative runtime 作为保险丝**

这一步最可能把 current regime 从：

- old 的 `too open / aggressive replacer`
- 和 new 的 `too closed / near no-op`

拉回到：

- **小覆盖、非零提交、仍然保守** 的 selective conservative regime。
