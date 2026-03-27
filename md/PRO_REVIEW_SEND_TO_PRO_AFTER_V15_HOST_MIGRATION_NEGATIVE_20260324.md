# Send To Pro: After V15 Host Migration Negative (2026-03-24)

这份提示词用于 `v15` stronger-host zero-shot migration 已跑完且为负之后的下一步裁决。不要再重开旧路线，也不要重新问“方向值不值”。

```markdown
请你把自己当成独立研究顾问兼实现 reviewer。你这次没有历史上下文，所以请先把我附带的以下文档视为当前项目的权威上下文：

1. `PRO_REVIEW_CANONICAL_CONTEXT_20260324.md`
2. `PRO_REVIEW_LATEST_DELTA_20260324.md`
3. `PRO_REVIEW_INTERACTION_LOG.md`

另外我会附上最新代码包和这次 `v15` host migration 的实验 bundle。请你严格基于这些上下文回答，不要重新打开 row-local、full replacement 或 continuity。

## 1. 当前已经确认的主线

当前主线已经固定成：

- host-level tracking-by-detection runtime
- module: `LocalConflictCommitRefiner`
- online semantics: `cluster-level conservative partial commit + defer to host`
- injection point: `primary-only`, `pre-Hungarian`

## 2. 已经确认的正号

在 `base_reid_da` 上，learned commit 已经拿到真实但不大的正号：

- proxy0213: `53.755 / 46.125 / 59.856 / 73.166 / 869`
- full FRCNN `md2/mm2`: `61.995 / 58.274 / 70.930 / 75.868 / 1605`

这说明 operator 在当前 host 上不是空故事。

## 3. 上一次你的裁决

你上一次给的唯一主方案是：

- `A. stronger host migration`

你选的 stronger host 是：

- 主选: `v15_laplace_reid_da_val0213`
- 备选: `v13_tf_only_val0213_reid_da`

## 4. 新的关键证据：v15 paired proxy0213 migration 已经跑完，而且是负的

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
- 问题更像是 portability 不成立，或者 current training data / supervision 面太窄

## 5. 现在我只要你裁决一个问题

请你只在下面两个候选中，选一个唯一主方案，并给一个备选：

### 候选 A2. still run one last portability check on v13

也就是：

- 不立刻转大数据
- 先用同样 paired proxy0213 protocol，再补打一枪 `v13_tf_only_val0213_reid_da`
- 看 `v15` 失败是不是 host-specific，而不是 portability 整体失败

### 候选 B. larger training data

也就是：

- 不再继续 stronger-host zero-shot 迁移
- 直接回到当前 `base_reid_da` 或未来目标 host 的更大 cluster-commit 数据构建
- 重训 learned commit 后再决定是否做 host migration

## 6. 你的任务

请你只回答下面 5 个问题：

1. 现在该判定为 `GO / NARROW GO / KILL` 中的哪一个？
2. 在 `A2 / B` 两个候选里，唯一主方案是什么？为什么？
3. 如果你还建议打 `v13`，请明确说为什么 `v15` 的负结果不足以直接转 `B`。
4. 如果你建议直接转 `B`，请明确说为什么 `v13` 这枪的额外信息增益不值成本。
5. 下一步唯一 first-priority experiment 是什么？请写到可以直接执行的程度。

## 7. 输出硬要求

不要重新建议我去：

- row-local rerank
- full cluster replacement
- continuity / stitching
- 先强模型再说

你的回答必须包含两部分：

### Part A. 管理级决策

- `GO / NARROW GO / KILL`
- 为什么
- 当前唯一主方案
- 一个备选
- 下一步唯一 first-priority experiment
- 哪些事情现在不要再做

### Part B. 可执行实现说明

如果你选择的主方案需要改代码或实验脚本，请直接给一个简洁但可执行的实现说明，至少包含：

- 改哪些文件
- 用哪个 runner / config 入口
- 结构化记录怎么写到 `summary.csv` / `result.csv`
```
