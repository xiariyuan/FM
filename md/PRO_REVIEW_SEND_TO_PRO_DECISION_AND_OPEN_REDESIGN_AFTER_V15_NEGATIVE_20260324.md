# Send To Pro: Decision And Open Redesign After V15 Negative (2026-03-24)

这份提示词用于当前阶段的“双任务”提问：既要当前最现实的下一步裁决，也要在更开放的视角下，让 Pro 结合更多论文与代码给出下一版更强模块设计。

```markdown
请你把自己当成独立研究顾问兼实现 reviewer。你这次没有历史上下文，所以请先把我附带的以下文档视为当前项目的权威上下文：

1. `PRO_REVIEW_CANONICAL_CONTEXT_20260324.md`
2. `PRO_REVIEW_LATEST_DELTA_20260324.md`
3. `PRO_REVIEW_INTERACTION_LOG.md`

另外我会附上最新代码包、当前 learned commit 队列结果，以及刚跑完的 `v15` host migration 结果。请你严格基于这些上下文回答。

## 0. 这次你要同时完成两件事

### 任务 A. 当前现实决策

基于现有证据，给出下一步唯一主方案：

- 是直接转 `B. larger training data`
- 还是还值得补打一枪 `v13` portability check

### 任务 B. 开放式更强模块设计

不要只局限当前最小 edge-MLP + pooling。

请你在充分阅读当前代码包、上下文文档，并参考你认为有代表性的近年 MOT / graph matching / structured assignment / set prediction / ID prediction 方向后，开放地思考：

- 如果不被当前最小实现束缚
- 但仍然尊重我们已经拿到的正负证据

那么下一版真正更强、也更像论文主角的模块应该是什么。

注意：

- 任务 A 和任务 B 都要做。
- 任务 A 是近期执行决策。
- 任务 B 是中期模块设计。
- 不允许只回答其中一个。

## 1. 当前已经明确的主线与硬约束

当前主线已经固定成：

- host-level tracking-by-detection runtime
- module: `LocalConflictCommitRefiner`
- online semantics: `cluster-level conservative partial commit + defer to host`
- injection point: `primary-only`, `pre-Hungarian`

已经明确停掉或不进入当前主故事的：

- row-local rerank
- full cluster replacement
- continuity / stitching

## 2. 当前已经确认的正号

在 `base_reid_da` 上，learned commit 已经拿到真实但不大的正号：

- proxy0213: `53.755 / 46.125 / 59.856 / 73.166 / 869`
- full FRCNN `md2/mm2`: `61.995 / 58.274 / 70.930 / 75.868 / 1605`
- full FRCNN `md2/mm3`: `61.763 / 57.705 / 70.497 / 75.885 / 1583`
- full FRCNN `md3/mm2`: `61.995 / 58.274 / 70.930 / 75.868 / 1605`
- full FRCNN `md4/mm2`: `61.800 / 57.765 / 70.619 / 75.879 / 1582`

这说明：

- operator 在当前 host 上不是空故事
- 但绝对指标还远没有到论文终局

## 3. 新的关键负证据：v15 paired proxy0213 migration 为负

run root:

- `outputs/local_conflict_graph_hostmig_v15_proxy0213_20260324_194915`

paired result:

- `v15 host_only`: `HOTA 53.069 / AssA 44.392 / IDF1 59.099 / MOTA 73.752 / IDSW 684`
- `v15 + learned_commit`: `HOTA 52.850 / AssA 43.973 / IDF1 58.978 / MOTA 73.722 / IDSW 693`

delta:

- `delta_HOTA = -0.219`
- `delta_AssA = -0.419`
- `delta_IDF1 = -0.121`
- `delta_MOTA = -0.030`
- `delta_IDSW = +9`

但这不是空跑：

- `eligible_clusters = 5769`
- `replaced_clusters = 138`
- `matched_dets = 278`

所以当前已经明确知道：

- `v15` 上的 first-shot zero-shot portability 没过
- 问题不是模块没触发
- 问题更像是：
  - portability 不成立
  - 或 current training data / supervision 面太窄
  - 或 current module 只够在 base host 上成立，不够成为更通用的 stronger-host operator

## 4. 你这次不能回避的两个问题

### 问题 A. 现在到底该怎么继续

请你只在下面两个候选里选一个主方案：

#### 候选 A2. still run one last portability check on v13

- 不立刻转大数据
- 用同样 paired proxy0213 protocol，再补打一枪 `v13_tf_only_val0213_reid_da`
- 看 `v15` 失败是不是 host-specific，而不是 portability 整体失败

#### 候选 B. larger training data

- 不再继续 stronger-host zero-shot 迁移
- 直接回到更大的 cluster-commit 数据构建与重训
- 重训 learned commit 后再决定是否做 host migration

### 问题 B. 如果跳出当前最小实现，你建议的更强模块是什么

这里我不要你只说“可以考虑 GNN / Transformer / Sinkhorn”。

我要你结合：

- 当前代码包
- 当前 operator 的正负证据
- 近年的相关论文与开源实现思路

直接选出一个你认为最值得作为下一版论文主角的模块方案。

这个方案可以比当前更开放，但必须满足：

- 仍然与当前证据链相容
- 不回到已经判死的 row-local / full replacement / continuity 主线
- 要能落到当前 repo 的文件级实现

## 5. 你的任务

请你回答下面 8 个问题：

1. 当前应判定为 `GO / NARROW GO / KILL` 中的哪一个？
2. 对任务 A，在 `A2 / B` 中唯一主方案是什么？为什么？
3. 如果你还建议打 `v13`，请明确说为什么 `v15` 的负结果不足以直接转 `B`。
4. 如果你建议直接转 `B`，请明确说为什么 `v13` 的额外信息增益不值成本。
5. 对任务 B，如果不被当前最小实现束缚，你认为下一版唯一最值得设计的更强模块是什么？
6. 这个更强模块，与当前 `LocalConflictCommitRefiner` 的关系是什么？
   - 替代
   - 升级
   - 两阶段协同
   你必须明确选一种。
7. 这个更强模块最小可落地版本应该是什么？
8. 下一步唯一 first-priority experiment 是什么？这里请区分：
   - 近期执行决策
   - 中期模块设计验证

## 6. 输出硬要求

不要重新建议我去：

- row-local rerank
- full cluster replacement
- continuity / stitching
- 先随便大 sweep

你的回答必须包含三部分：

### Part A. 管理级决策

- `GO / NARROW GO / KILL`
- 为什么
- 任务 A 的唯一主方案
- 一个备选
- 当前不要再做什么

### Part B. 开放式模块判断

- 你建议的唯一更强模块
- 为什么它比当前最小 learned commit 更像论文主角
- 它与当前主线证据如何对齐
- 为什么不是别的几个明显备选

### Part C. 可执行实现设计

请直接给一个可执行的实现设计说明，至少包含：

- 要改哪些文件
- 哪些 runner / config 入口要新增或重写
- 新的 supervision / dataset builder 是否要改
- `summary.csv / result.csv / experiment_registry.csv` 应如何记录
- 第一批实验顺序

如果你认为任务 A 和任务 B 应分两阶段执行，也请明确分成：

1. 近期执行项
2. 中期重设计项
```
