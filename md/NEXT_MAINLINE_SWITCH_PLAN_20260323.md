# Next Mainline Switch Plan (2026-03-23)

## 当前决策

当前 `single-row / row-local learned online controller` 主线停止。

停止的原因不是绝对值 `52.x` 低，而是：
- `rerank_only` 没打过 `noop`
- `rerank_minimal` 没打过 `noop`
- 连 `oracle_rerank(top-8 + minimal winner override)` 也没打过 `noop`

因此，继续在这条 line 上做 trigger sweep、loss tweak、delta 调整、更多 learned rerank 试验，不再有主线价值。

## 新主线

新主线切到：

`competition-aware local conflict graph`

一句话定义：
不再把 decision unit 定义为“单个 detection 的一行 top-k 重排”，而是把同一局部冲突区内的多个 detections、多个 candidate tracks 联合起来做一个小规模结构化决策。

## 为什么是它，不是 continuity

当前不选 continuity / stitching 作为主线，原因有三条：

1. 当前 proxy 数据里 `bridge_rate_among_positive = 0.03801`，bridge 占比很低。
2. 当前 stage1 里的 continuity/bridge head 没形成有效 offline signal。
3. 当前 oracle negative 更像是在说明“单行局部 winner correction 不够”，不是“长时连续性是第一矛盾”。

所以现在最合理的升级方向，不是把时间跨度拉长，而是先把 decision unit 从单行扩成局部冲突子图。

## 新主线的最小问题定义

在每一帧的 primary association 前：

- 先从 host 的 gated score matrix 中抽取局部冲突子图
- 子图节点包括：
  - 当前冲突 detections
  - 它们共享的 top-k candidate tracks
- 在这个子图上做联合决策，而不是一行一行分别重排

第一版目标不是做 end-to-end tracker replacement，而是做：

- `edge scoring + conflict-aware joint assignment`

也就是：
- detection-track 边仍然是基础单元
- 但判定时要看局部冲突图上下文
- 输出仍然接回原有 Hungarian / matching 体系，或者直接在子图内部做局部 one-to-one assignment

## 第一批代码改造

### 1. 保留不动的部分

- `base_reid_da` host 继续保留
- runtime dump / replay label / competition case build 这条数据链继续保留
- experiment recording / bundling 继续保留

### 2. 停止继续扩展的部分

- 不再继续扩展 `row-local controller`
- 不再新增 `rerank_only / rerank_minimal / oracle_rerank` 变体
- 不再为当前 learned row operator 继续做 loss / trigger 调整

### 3. 新增的核心模块

建议新增一个模块，例如：

- `models/conflict_graph_assoc.py`

最小职责：

- 输入：
  - 一个局部冲突 cluster
  - cluster 内 detections 的 observed features
  - cluster 内 candidate tracks 的 observed features
  - detection-track edge features
- 输出：
  - edge scores 或局部 assignment logits

### 4. host 接回点

仍然放在：

- `pre-Hungarian`
- `primary association only`

但这次不是一行一行改 score，而是：

- 先识别局部冲突 cluster
- 在 cluster 内联合计算
- 输出 cluster 内部的更一致的局部 assignment proposal

## 第一批实验顺序

### 实验 1

做一个纯分析实验，不训练模型。

目标：
- 在当前 proxy0213 上统计“真正的局部冲突 cluster”长什么样

至少输出：
- cluster size 分布
- 每个 cluster 包含的 detections 数
- 每个 cluster 覆盖的 candidate tracks 数
- row-local oracle 失败时，是否存在跨行冲突

这个实验的作用是验证：
- 为什么单行 oracle 都不行
- 局部图决策是否真的更合理

### 实验 2

做 `oracle local conflict graph upper bound`

目标：
- 不训练 learned model
- 先用 replay labels / GT 信息在局部 cluster 内做一个 oracle joint assignment

这一步是新主线最重要的 upper bound。

如果连局部 conflict graph oracle 也不明显打过 `noop`，那说明问题还不在 decision unit 升级，而要重新审 host 或 benchmark。

### 实验 3

在 oracle conflict graph 确认有 upper bound 后，再做第一版 learned conflict graph 模型。

第一版必须保守：
- observed-only features
- primary-only
- local cluster only
- 不碰 continuity
- 不碰 lifecycle

## 第一批实验不要做什么

- 不要先做 continuity 主线
- 不要先做 tracklet stitching 主线
- 不要继续扫 row-local learned rerank 的参数
- 不要先改 detector
- 不要先做 full benchmark 扩展

## 近期交付物

近期应该交付三样东西：

1. 一份最终 kill 决策材料
2. 一份 local conflict graph 的设计文档
3. 一份第一批 cluster/oracle graph 诊断实验结果

## 当前仓库里建议保留的参考证据

- row-local offline stage1：
  - `outputs/competition_assoc_stage1_fix1_full12`
- proxy conflict-case build：
  - `outputs/competition_assoc_base_reid_da_proxy0213_hybriddumpfix`
- online negative chain：
  - `outputs/competition_assoc_online_noop_proxy0213_20260323_094948`
  - `outputs/competition_assoc_online_rerank_only_proxy0213_20260323_113046`
  - `outputs/competition_assoc_online_rerank_minimal_proxy0213_20260323_114447`
  - `outputs/competition_assoc_online_oracle_rerank_proxy0213_20260323_141625`

这些不是要继续扩展的主线结果，而是新主线的反证背景。
