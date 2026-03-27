# 完整版 Learned Runtime Reranker 设计稿

日期：2026-03-18

目标：

- 不再做最小版的 “表格特征 MLP”
- 也不回到旧的 GT pseudo-group / pair-wise calibrator
- 而是设计一个**完整但仍可落地**的 learned runtime reranker

这个版本的定位是：

> 在强 tracking-by-detection host 上，对真实 runtime 导出的 detection-centered candidate groups 做 ambiguity-triggered、bounded、history-aware 的竞争重排序。

这份设计是为了后续直接实现，不是纯概念讨论。

---

## 1. 先说结论：我们到底要做什么

我们要做的不是：

- 一个更大的 pair-wise alpha/r MLP
- 一个更黑的 score calibrator
- 一个只吃 CSV 标量特征的普通表格分类器

我们要做的是：

**一个完整的 runtime candidate-group reranker，包含四个层次：**

1. 结构化标量证据
2. 原始时序历史证据
3. 候选间竞争建模
4. 安全触发与 bounded residual 输出

它的目标不是替代整个 host，而是：

- 在 easy groups 上尽量 no-op
- 在 ambiguous groups 上修正 top-1 排序
- 在 background groups 上避免乱抬分

---

## 2. 为什么不能只做“最小 learned runtime”

现在已经有两个事实成立：

1. `GBDT + ambiguity-only` 已经证明 runtime replay 数据里有真实信号
2. 强 host 上可修复空间很小，而且集中在少量 ambiguity groups

因此，如果我们只做“最小 learned runtime”：

- 输入只用现有 CSV 标量特征
- 模型只是一个 MLP / 小 Transformer
- 没有原始历史张量
- 没有候选竞争建模

那它很可能只能逼近 GBDT，而很难显著超过 GBDT。

这对论文不够。

所以完整版 learned runtime 必须引入**GBDT 当前看不到的信息**：

- 原始 detection embedding
- track history embedding 序列
- detection 和 history 的逐时刻相似度序列
- candidate 之间的相互作用

也就是说：

> 完整版 learned runtime 的意义，不是把 GBDT 再神经网络化一次，而是把 GBDT 吃不到的 runtime 时序与竞争信息纳入模型。

---

## 3. 完整版方法总览

建议名称：

**RR-CTR**

全称：

**Runtime-Replay Competitive Temporal Reranker**

一句话：

> RR-CTR 是一个基于真实 runtime candidate groups 训练的安全型重排序模块，它用结构化特征、时序历史证据和局部候选竞争建模，只在 ambiguity groups 上对强 host 的 primary association 做 bounded reranking。

---

## 4. 输入对象：必须是 detection-centered runtime group

### 4.1 训练 / 推理统一对象

每个训练样本与每次推理调用都统一为：

- 一个 detection `d_j`
- 它通过 primary gating 后保留下来的 candidate tracks 集合 `C_j`

也就是：

`group_j = { detection_j, candidates_i in C_j }`

这比旧路线的单个 pair 正确得多。

### 4.2 group 内每个 candidate 的输入

对每个 candidate `i`，需要两类输入。

#### A. 标量结构化特征

这部分先保留和当前 GBDT 一致的强特征：

- `base_score`
- `refined_score`
- `motion_score`
- `det_score`
- `track_gap`
- `track_hist_len`
- `base_margin`
- `refined_margin`
- `rank_margin`
- `rank_entropy`
- `rank_frac`
- `dx_norm`
- `dy_norm`
- `log_w_ratio`
- `log_h_ratio`
- `log_area_ratio`
- `det_track_iou`

#### B. 原始张量特征

这部分是完整版与最小版的根本区别：

- detection appearance feature `det_feat_j`
- candidate track history features `hist_feat_i[t]`
- history mask `hist_mask_i[t]`
- history timestamps / ages `hist_time_i[t]`
- candidate last box / predicted box
- detection box

如果后续可行，也可以增加：

- current track aggregated feature
- motion state uncertainty
- appearance reliability flag

---

## 5. 数据导出设计：CSV 不够，必须加 tensor shard

### 5.1 为什么必须改 dump

当前 runtime dump 只有 CSV 标量特征。

这意味着：

- GBDT 可以吃
- logistic 可以吃
- 小 MLP 也可以吃

但这不足以支持真正的“完整版 learned runtime”，因为：

- 历史序列本身已经在 dump 前被压缩丢失
- detection 与 history 的逐时刻关系没有被保存
- 候选交互只能靠后验拼接出来

因此，完整版必须改成：

**CSV + tensor shard 双轨导出**

### 5.2 CSV 继续保留什么

CSV 继续保留：

- group_id
- seq / frame / det_idx
- track_id / rank
- all scalar features
- label / ambiguity / recoverability diagnostics

它用于：

- baseline
- 可视化
- 诊断
- debug

### 5.3 需要新增的 tensor shard

建议新建 shard npz 或 pt 文件，按 group 打包，至少包含：

- `group_offsets`
- `group_ids`
- `det_feat`
- `det_box`
- `det_score`
- `cand_track_ids`
- `cand_anchor_score`
- `cand_refined_score`
- `cand_motion_score`
- `cand_track_box`
- `cand_track_gap`
- `cand_hist_len`
- `cand_hist_feat`
- `cand_hist_mask`
- `cand_hist_time`

推荐按固定 group 数分 shard，例如：

- 每 `2048` 或 `4096` 个 groups 写一个 shard

### 5.4 Top-K 还是 full-candidate

建议：

- dump 阶段保留 full-candidate
- 训练阶段动态裁成 top-K

原因：

- full-candidate 用于 recoverability 统计和诊断
- model 真正训练时只看 top-K，计算更可控

推荐训练 `K = 5` 或 `K = 8`

规则：

- inference 时也只对 top-K 做 rerank
- 训练时如果 GT positive 不在 top-K，强制塞进去

---

## 6. 模型结构：四塔合一

完整版 learned runtime 不建议只做一个统一大网络，而建议拆成 4 个有明确职责的子模块。

---

### 6.1 模块 A：Scalar Evidence Tower

作用：

- 吃结构化特征
- 学会 GBDT 已经证明有效的那部分边界

输入：

- 现有 17~20 维结构化特征

结构：

- `Linear -> GELU -> LayerNorm -> Linear -> GELU -> LayerNorm`
- 输出 candidate scalar embedding `e_scalar_i`

输出：

- `e_scalar_i`
- 可选 scalar score prior `z_scalar_i`

这个模块的本质：

- 先把 GBDT 的强项保住

---

### 6.2 模块 B：Temporal Evidence Tower

作用：

- 提取 GBDT 看不到的时序证据
- 这是你原始 idea 最应该真正落脚的地方

#### 输入

对每个 candidate `i`：

- detection feature `d_j`
- history feature sequence `h_i[t]`
- history age / timestamp
- history valid mask

#### 中间构造

先构造 detection-conditioned temporal sequence：

- cosine sim `sim_t = cos(d_j, h_i[t])`
- similarity delta
- age embedding
- optional feature delta

#### 建议结构

不要直接上很重的 Transformer。
完整版但可控的设计是：

**Multi-Scale Temporal Encoder**

包含三个分支：

- 短期 TCN 分支：kernel 3
- 中期 TCN 分支：kernel 5
- 长期 TCN 分支：kernel 9 或 dilated conv

然后：

- 分支输出拼接
- 再用 gated attention pooling 汇总

也可以加一个 detection-conditioned attention：

- query = detection token
- key/value = history tokens

最终输出：

- `e_temp_i`
- `temp_conf_i`
- `temp_uncertainty_i`

#### 为什么这样设计

因为：

- 你的时序信号是短而非平稳的
- 直接上频域说服力不够
- multi-scale temporal evidence 比“Laplace 模块”更容易解释，也更接近真实任务结构

---

### 6.3 模块 C：Candidate Competition Tower

作用：

- 真正建模 candidate-set competition
- 这是从 pair-wise 走向 group-wise 的关键

#### 输入

对 top-K candidates：

- `e_scalar_i`
- `e_temp_i`
- anchor logit / rank features

先拼成：

- `token_i = [e_scalar_i, e_temp_i, anchor_meta_i]`

#### 建议结构

推荐两层结构：

##### 第一层：Set Self-Attention

对 top-K candidates 做 2 层小型 multi-head self-attention

作用：

- 感知其它 rivals 的存在
- 建模 group context

##### 第二层：Pairwise Duel Aggregation

对每个 candidate `i`，和组内其它 `k != i` 形成 duel token：

- `token_i`
- `token_k`
- `token_i - token_k`
- `z_i - z_k`

再用 attention 聚合 hardest rivals。

输出：

- `e_comp_i`
- `duel_score_i`

#### 为什么不只做 Set Encoder

因为：

- 单纯 mean/max group context 已经被 HACA-v1 证明不够
- 真正关键的是 rival interaction

---

### 6.4 模块 D：Safety / Activation Controller

作用：

- 决定这组要不要激活 learned rerank
- 决定 residual 可以多大
- 决定是否倾向 background / no-match

#### 输入

group-level features：

- top1-top2 margin
- entropy
- candidate_count
- mean history sufficiency
- mean temporal confidence
- det score

#### 输出

- `a_j`：group activation gate
- `z_null_j`：background / null logit
- `delta_cap_j`：本组 residual 上限

#### 原则

- easy groups：`a_j` 小
- ambiguous groups：`a_j` 大
- history 不足：`a_j` 小
- background 倾向高：`z_null_j` 高

---

## 7. 输出形式：必须是 bounded residual，不是重写分数

### 7.1 为什么

强 host 已经很强。
我们不能让 learned runtime 直接重写整张 cost matrix。

### 7.2 推荐公式

先定义 host anchor：

- `z_i^0 = logit(anchor_score_i)`

model 预测：

- candidate residual `r_i`
- candidate trust `beta_i`
- group activation `a_j`
- null prob `p_null_j = sigmoid(z_null_j)`

先做 zero-sum residual：

- `r_i_centered = r_i - mean(r_k over top-K)`

最终：

- `z_i = z_i^0 + a_j * beta_i * delta_cap_j * tanh(r_i_centered)`

如果考虑 background：

- `s_i = (1 - p_null_j) * sigmoid(z_i)`

这个形式的好处：

- bounded
- zero-sum
- 不会把整组都整体抬高
- easy groups 容易退化成 no-op

---

## 8. 训练目标：不是纯 BCE

完整版至少需要 5 个 loss。

---

### 8.1 主损失：Listwise CE

对 top-K + null candidate 做 listwise 分类：

- 正样本组：GT candidate 为正确类
- background 组：null 为正确类

这是最核心的 loss。

---

### 8.2 Hard Duel Loss

只对 ambiguous positive groups：

- 让 GT candidate 压过 hardest rival

形式：

- margin ranking / hinge loss

目的：

- 直接优化 hardest case

---

### 8.3 Safe No-op Loss

对 easy groups：

- 约束 final logits 不要偏离 anchor 太多

形式：

- `|| z_final - z_anchor ||^2`

目的：

- 保住 easy case

---

### 8.4 Teacher Distillation Loss

因为当前 GBDT 已经是强 teacher，
完整版 learned runtime 最合理的启动方式不是随机学，而是蒸馏 teacher。

蒸馏方式：

- score distillation
- 或 pairwise order distillation

作用：

- 先学到 GBDT 证明有效的边界
- 再靠 temporal / competition tower 去超越 GBDT

---

### 8.5 Activation Sparsity / Gate Regularization

对 group activation `a_j` 加轻正则：

- 鼓励 sparse activation

作用：

- 避免 all-group always-on

---

## 9. 训练流程：建议三阶段

完整版不建议一步训到底。

### 阶段 1：Scalar Student

输入：

- 只有结构化特征

目标：

- 先追平 GBDT

loss：

- listwise + distill + safe

### 阶段 2：Add Temporal Tower

输入：

- 加 det/history tensors

目标：

- 在 ambiguity groups 上超过 scalar-only 模型

### 阶段 3：Add Competition Tower

输入：

- full token + top-K rival interaction

目标：

- 进一步提升 hard groups

---

## 10. 推理接入点：必须锁死

只能接：

- primary association only
- gating 之后
- Hungarian 之前

不能动：

- secondary matching
- newborn logic
- lifecycle
- CMC / KF 主干

这一步对论文 attribution 非常重要。

---

## 11. 调试与可解释性：必须保留日志

完整版 learned runtime 很容易变黑。

因此 runtime debug 字段必须保存：

- `group_id`
- `group_is_ambiguous`
- `activation`
- `null_prob`
- `topk_before`
- `topk_after`
- `anchor_margin`
- `entropy`
- `teacher_score`
- `residual_norm`
- `rank_swap`

这样才能知道：

- 模型是在修 hard groups
- 还是在乱动 easy groups

---

## 12. 完整实验矩阵

### 12.1 Offline replay

必须有：

- logistic
- GBDT
- learned scalar-only
- learned scalar + teacher
- learned scalar + temporal
- learned scalar + temporal + competition

### 12.2 Activation ablation

- `apply_on=all`
- `apply_on=ambiguous`
- no activation controller

### 12.3 Output ablation

- bounded residual
- unbounded residual
- with null head
- without null head

### 12.4 Input ablation

- scalar only
- scalar + temporal
- scalar + temporal + competition

### 12.5 Online frozen-host

必须有：

- host base
- host heuristic
- host + learned plugin

宿主至少两个：

- heuristic full7 host
- base full7 host

### 12.6 Generalization

- leave-one-sequence-out
- second host

---

## 13. 成功标准

完整版 learned runtime 如果要成立，至少要达到：

### Offline replay

- 不弱于 GBDT
- ambiguity groups 稳定正增益
- easy groups 不退化

### Online runtime

- frozen host 上 `AssA / IDF1 / HOTA` 有稳定正增益
- `IDSW` 不明显恶化

### Incremental value

- temporal tower 比 scalar-only 更强
- competition tower 比 temporal-only 更强

如果这三条做不到，那这套“完整版”虽然复杂，但论文不一定更强。

---

## 14. 和最小版的区别总结

最小版 learned runtime：

- 只有结构化特征
- 本质是神经版 GBDT
- 用来证明神经模型能否追平树模型

完整版 learned runtime：

- 结构化特征 + det/history tensors
- 有 temporal evidence tower
- 有 top-K competition tower
- 有 null/background head
- 有 activation controller

也就是说：

> 最小版是“能不能站住”，完整版是“能不能真正超过 GBDT 并把你的原创时序想法装回来”。

---

## 15. 最后一句结论

如果我们现在就开始实现 learned runtime，
最正确的顺序不是：

- 直接上完整版 end-to-end 大模型

而是：

1. 先实现 scalar-only student，追平 GBDT
2. 再补 tensor dump
3. 再上完整版 RR-CTR

这样做不是保守，而是为了保证：

- 证据链可解释
- 每一层增量都知道是谁带来的
- 最终能把你的时序 / Laplace 思想重新放回论文主线里，而不是又做成一个黑盒大模型

