# Send To Pro With Context (2026-03-24)

这是一份新的默认发送模板。用途是: 在新的 Pro 没有上下文时，先把固定背景、最新增量、交互日志明确塞进去，再问单一问题。

## 使用方法

发送时建议附上这 4 份文件:

1. `md/PRO_REVIEW_CANONICAL_CONTEXT_20260324.md`
2. `md/PRO_REVIEW_LATEST_DELTA_20260324.md`
3. `md/PRO_REVIEW_INTERACTION_LOG.md`
4. 当前这份真正的提问模板

如果本次还要附代码包或 bundle，也一并说明。

## 可直接复制的模板

```markdown
请你把自己当成独立研究顾问兼实现 reviewer。你这次没有历史上下文，所以请先把我附带的以下 3 份文档视为当前项目的权威上下文：

1. `PRO_REVIEW_CANONICAL_CONTEXT_20260324.md`
2. `PRO_REVIEW_LATEST_DELTA_20260324.md`
3. `PRO_REVIEW_INTERACTION_LOG.md`

要求：

- 不要重新打开这些文档里已经判死的旧路线。
- 不要忽略 interaction log 里已经被证据修正过的结论。
- 你必须在这些上下文基础上回答我最后的单一问题。
- 除非我明确要求开放式 brainstorm，否则请只给一个主方案，不要给一堆平行备选。

当前我要你回答的唯一问题是：

[把本轮真正的问题写在这里，例如:
“现在 learned commit 已经在 current host 上拿到稳定小正号，下一步应该优先迁移到哪一个更强 host？请只在以下候选里选择: v13_tf_only_val0213_reid_da, v15_laplace_reid_da_val0213, v16_laplace_trainable_val0213。请给出唯一主选和一个备选，并说明为什么其余不选。”]

你的回答必须包含两部分：

Part A. 管理级决策
- `GO / NARROW GO / KILL`
- 为什么
- 下一步唯一 first-priority experiment 是什么
- 哪些事情现在不要再做

Part B. 代码级实现设计文档
- 如果这次问题需要改代码，请直接写成可执行级别的 Markdown 文档正文
- 必须落到文件级改动、配置与脚本接入、结构化记录方式
- 不允许只给高层建议
```

## 推荐提问范围

这份模板最适合问:

- stronger host selection
- next-step redesign after a concrete queue finishes
- why a specific learned operator plateaued
- what the next single decisive experiment should be

不适合问:

- 开放式“还有什么好方向”
- 已经被 interaction log 判死的旧线
- 没有具体候选集合的泛泛 host 建议
