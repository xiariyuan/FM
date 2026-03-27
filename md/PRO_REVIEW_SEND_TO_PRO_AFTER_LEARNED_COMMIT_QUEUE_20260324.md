# Send To Pro: After Learned Commit Queue (2026-03-24)

这份提示词用于当前 learned commit 队列完整结束后的下一步决策。不要再让 Pro 重新判断旧路线，而是只裁决下一步最值钱的实验。

```markdown
请你把自己当成独立研究顾问兼实现 reviewer。你这次没有历史上下文，所以请先把我附带的以下文档视为当前项目的权威上下文：

1. `PRO_REVIEW_CANONICAL_CONTEXT_20260324.md`
2. `PRO_REVIEW_LATEST_DELTA_20260324.md`
3. `PRO_REVIEW_INTERACTION_LOG.md`

另外我会附上最新代码包和本轮实验 bundle。请你严格基于这些上下文给判断，不要重新打开已经判死的旧路线。

## 1. 当前我已经确认的事实

当前主线不是：

- row-local rerank
- full cluster replacement
- continuity / stitching

当前主线已经收敛成：

- host: `base_reid_da`
- module: `LocalConflictCommitRefiner`
- online semantics: cluster-level `conservative partial commit + defer to host`
- injection point: `primary-only`, `pre-Hungarian`

## 2. 这轮刚跑完的完整 learned commit 队列

queue root:

- `outputs/local_conflict_graph_learned_commit_next12h_20260324_165558`

queue summary:

- `02 proxy0213`: `HOTA 53.755 / AssA 46.125 / IDF1 59.856 / MOTA 73.166 / IDSW 869`
- `03 full FRCNN md2/mm2`: `61.995 / 58.274 / 70.930 / 75.868 / 1605`
- `04 full FRCNN md2/mm3`: `61.763 / 57.705 / 70.497 / 75.885 / 1583`
- `05 full FRCNN md3/mm2`: `61.995 / 58.274 / 70.930 / 75.868 / 1605`
- `06 full FRCNN md4/mm2`: `61.800 / 57.765 / 70.619 / 75.879 / 1582`

## 3. 对照 control

oracle hard-trigger control:

- proxy0213: `53.175 / 44.949 / 59.036 / 73.219 / 873`
- full FRCNN md2/mm2: `61.858 / 57.957 / 70.705 / 75.882 / 1609`
- full FRCNN md2/mm3: `61.844 / 57.856 / 70.739 / 75.897 / 1574`
- full FRCNN md3/mm2: `61.698 / 57.615 / 70.412 / 75.946 / 1581`
- full FRCNN md4/mm2: `61.837 / 57.859 / 70.709 / 75.892 / 1597`

我的当前解释是：

- learned commit 已经拿到真实但不大的正号
- 这足以说明 operator 有价值
- 但绝对指标还不够，不足以直接成为论文终局

## 4. 现在我只要你回答一个具体决策

请你只在下面这三个下一步候选中，选出一个唯一主方案，并给一个备选：

### 候选 A. stronger host migration

把当前 learned commit 迁移到更强 host 之一做最小验证。

候选 host 限定为：

1. `configs/experiments/bytetrack_fa_mot_mot17_v13_tf_only_val0213_reid_da.yaml`
2. `configs/experiments/bytetrack_fa_mot_mot17_v15_laplace_reid_da_val0213.yaml`
3. `configs/experiments/bytetrack_fa_mot_mot17_v16_laplace_trainable_val0213.yaml`

### 候选 B. larger training data

不换 host，先扩大 cluster-commit 训练数据，再在当前 `base_reid_da` 上重训 learned commit。

### 候选 C. model strengthening

不换 host、不扩数据，直接在当前主线上把最小 edge-MLP + pooling 模型升级成更强模块。

## 5. 你的任务

请你只回答下面 5 个问题：

1. 现在该判定为 `GO / NARROW GO / KILL` 中的哪一个？
2. 在 A/B/C 三个候选里，唯一主方案是什么？为什么？
3. 如果你选 A，请在 3 个 host 中只选 1 个主选和 1 个备选，并说明剩下那个为什么不选。
4. 如果你不选 A，请说明为什么此时不该换 host。
5. 下一步唯一 first-priority experiment 是什么？请写到可以直接执行的程度。

## 6. 输出硬要求

请不要开放式 brainstorm，也不要重新建议我回到 row-local、full replacement 或 continuity。

你的回答必须包含两部分：

### Part A. 管理级决策

- `GO / NARROW GO / KILL`
- 为什么
- 当前唯一主方案
- 一个备选
- 下一步唯一 first-priority experiment
- 哪些事情现在不要再做

### Part B. 可执行实现说明

如果你选择的主方案需要改代码，请直接给一个简洁但可执行的实现说明，至少包含：

- 改哪些文件
- 入口函数或配置锚点
- 新 runner / config 该怎么落
- 结果应该怎么结构化记录到 `summary.csv` / `result.csv`
```
