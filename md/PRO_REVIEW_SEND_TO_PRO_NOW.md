# Pro Review Context: Current Mainline

这是一份用于立即审阅当前主线与下一步 online integration 的发送版提示词。

```markdown
请你把自己当成独立研究顾问，只审下面这条已经收敛下来的 MOT 主线。不要重新发散到旧方向，也不要泛泛给建议。请基于我给出的完整历史和当前证据，重点审阅最后的具体问题。

## 1. 项目背景

我当前做的是 tracking-by-detection 范式下的 MOT 研究。
当前宿主是 ByteTrack 风格 host，但当前固定对照宿主是 `base_reid_da`。
当前研究目标不是直接改 detector，而是在固定宿主上研究一个可插拔的 learned association module。

当前我认定的主线是：
- 宿主：`base_reid_da`
- 主角方法：`competition-aware association controller`
- 当前阶段目标：在真实 runtime ambiguity/conflict groups 上输出 `keep / rerank / null`
- 下一阶段目标：如果第一阶段在线成立，再补 `short-gap continuity / tracklet-level continuity`

注意：
`base_reid_da` 不是主角方法本身，它只是当前固定宿主。

## 2. 已经做过的路线，以及当前结论

### 路线 A：Laplace / MTCR / HACA / pairwise residual safe plugin

核心思路：
在强宿主上加一个保守、安全、只改 top-k 歧义组的 residual 插件，用 laplace / temporal / pairwise competition 证据修正关联分数。

当前结论：
- heuristic cue 不是完全没用
- 但 learned pairwise residual 作为主线不成立
- 主要问题是训练对象长期是 proxy / pseudo-group，不是 runtime competition 本体
- 这条线现在只保留成 control，不再当主线

### 路线 B：runtime replay safe plugin

核心思路：
训练对象改成更接近 runtime 的 candidate groups，做 runtime replay reranker。

当前结论：
- 这条线没有被彻底证伪
- 也拿到过小幅真实在线收益
- 但 effect size 太小，且 ceiling 可疑
- 现在只保留成 secondary line / appendix line，不再吞主线预算

### 路线 C：host-control / frequency family

核心思路：
用 `base_spatial -> base_reid_da -> full_reid_da` 这条控制链验证 appearance 与 frequency family。

当前结论：
- `base_reid_da` 对 `base_spatial` 有真实正收益
- `full_reid_da` 没有继续变强，反而给出负信号
- 因此当前 frequency-aware / spatial-freq interaction 这版 formulation 已经停止作为主线

因此当前明确结论是：
- 旧 safe residual 主线：停
- runtime replay：降级为二线
- frequency 主线：停
- 当前唯一主线：`base_reid_da` host 上的 `competition-aware association controller`

## 3. 当前 host / baseline 对照证据

这是 MOT17 proxy0213（MOT17-02 / 13）上的关键对照。

### `base_spatial` 最好点
- HOTA = 52.233
- AssA = 43.089
- IDF1 = 58.557
- MOTA = 72.382
- IDSW = 1148

### `base_reid_da` 最好点
- HOTA = 52.704
- AssA = 43.997
- IDF1 = 58.608
- MOTA = 73.361
- IDSW = 834

### `full_reid_da` epoch0
- HOTA = 51.733
- AssA = 42.365
- IDF1 = 57.969
- MOTA = 73.471
- IDSW = 791

结论：
- `base_reid_da` 相对 `base_spatial` 有真实净提升，大致是 `HOTA +0.471 / AssA +0.908 / IDSW -314`
- 当前固定 `base_reid_da`，因为它是最稳的宿主
- `full_reid_da` 已经给出负信号，因此我不再继续 frequency family 主线

## 4. 当前主线的核心动机

我现在认为真正的问题不是：
- 不是 feature 还不够 fancy
- 也不是再堆一个频域模块就会自然变强

而是：
- runtime association 里的局部冲突和竞争关系没有被直接建模

因此我把主线切成：
- 固定 `base_reid_da` 作为 host
- 从真实 runtime dump 中构建 `ambiguity/conflict groups`
- 训练 `competition-aware association controller` 来输出 `keep / rerank / null`
- 如果第一阶段在线成立，再扩到 `continuity / tracklet-level continuity`

## 5. 当前主线已经跑通的事实

### 5.1 数据 / runtime pipeline

我已经在 `base_reid_da` host 上成功跑通：
- runtime dump
- replay labels
- competition case build

proxy0213 competition summary 目前是：
- groups = 29319
- positive_groups = 19756
- ambiguous_groups = 19756
- recoverable_groups = 4751
- recoverable_rate_among_positive = 0.24048
- bridge_rate_among_positive = 0.03801

这说明：
- 当前真实 hard groups 数量是够的
- 可恢复重排组约占 positive groups 的 24%
- continuity bridge 只占很小一部分，大约 3.8%
- 所以第一阶段先做 rerank / conflict decision 是合理的

### 5.2 当前训练状态

最开始 stage1 训练有 NaN，原因不是数据坏，而是：
- 有一部分 group 是 `valid_mask` 全 false
- `MultiheadAttention` 在全 mask 行上产出了 NaN
- 训练脚本之前也没有把非有限值强制标失败

这个问题现在已经修复：
- 模型前向对 all-invalid rows 做了数值稳定处理
- 训练脚本加了 finite checks
- 非有限 loss 会直接把结构化记录写成 failed，不再假装 ok

### 5.3 最新正式结果

当前正式离线 run：
- run = `outputs/competition_assoc_stage1_fix1_full12`
- best_epoch = 10
- val_action_acc = 0.618739
- val_rerank_candidate_acc = 0.915464
- val_rerank_action_acc = 0.657732
- status = ok

这说明：
- 这条 controller 线至少已经学到有效的 conflict rerank signal
- 当前还没有在线 integration 结果
- 下一个动作就是把它接回 host，做 proxy0213 在线闭环

## 6. 当前最需要你审的具体问题

现在请你不要再回头讨论：
- 要不要继续 frequency
- 要不要继续旧 HACA / Laplace
- 要不要继续 runtime replay 老插件

这些路线我已经做了明确去留判断。

我接下来的动作是：
- 把 stage1 best checkpoint 接回 `base_reid_da` host，做 proxy0213 在线评测

请你只回答下面这些问题：
1. 这个 `competition-aware association controller` 接回 host 时，最稳的注入方式应该是什么？
2. `keep / rerank / null` 三头输出，应该怎样作用到原始关联分数或匹配流程，才能最大化离线到在线的一致性？
3. 第一轮 online ablation 最合理的顺序是什么？
4. 如果出现“离线强、线上不涨”，最优先排查的 3 个 failure point 是什么？

## 7. 我希望你的输出方式

请直接给我：
- go / kill 风险判断
- 推荐的 online integration 设计
- 必做 ablation 清单
- 最需要警惕的 failure modes

不要泛泛谈方向，不要重复旧历史，不要只说“先做实验看看”。
我需要的是：在当前上下文完整的前提下，你对下一步 online integration 的深度审阅。
```

