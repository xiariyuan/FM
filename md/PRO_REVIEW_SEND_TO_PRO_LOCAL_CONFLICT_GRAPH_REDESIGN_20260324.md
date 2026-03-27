# Send To Pro: Local Conflict Graph Redesign (2026-03-24)

这是一份当前阶段可直接发送给 Pro 的强约束提示词。目标不是再问“方向对不对”，而是基于已经收敛的证据链，让 Pro 直接给出更强模块的代码级重设计文档。

```markdown
请你把自己当成独立研究顾问兼实现设计 reviewer。不要重新开放式发散到旧方向，也不要只给泛泛方法建议。我现在不是要你再判断一次“要不要做 graph”，而是要你基于完整证据链和当前代码包，直接为当前主线设计一个更强的模块，并给出可执行级别的代码重设计文档。

你的回答必须包含两部分：

Part A. 管理级决策
- `GO / NARROW GO / KILL`
- 为什么
- 当前唯一 first-priority experiment 是什么
- 哪些事情现在不要再做

Part B. 代码级实现设计文档
- 请直接输出一份完整 Markdown 文档正文
- 文档标题请写成：`IMPLEMENTATION_REDESIGN_PLAN.md`
- 必须落到文件级改动
- 必须说明修改文件、函数锚点 / 行号范围、修改前逻辑、修改后逻辑、配置与脚本接入方式
- 不允许只给概念性方向

## 1. 当前项目已经收敛后的主线

我当前做的是 tracking-by-detection 范式下的 MOT 研究。
当前固定宿主是 `base_reid_da`。宿主不是主角方法，它只是当前最稳定的 host baseline。

当前唯一主线已经切成：
- 宿主：`base_reid_da`
- 主角方法：`competition-aware local conflict graph`
- 当前阶段：只动 `primary association`，只在 `pre-Hungarian` 做局部竞争修正
- 当前在线语义：`oracle_commit_matches + hard cluster trigger`
- 下一阶段目标：把当前 oracle/heuristic cluster policy 升级成真正的 learned cluster-level assignment/refinement module

## 2. 已经正式停掉或降级的路线

### 已停主线
- `Laplace / MTCR / HACA / pairwise residual safe plugin`
- `frequency-aware / spatial-freq interaction`
- `single-row / row-local rerank` 作为主 decision unit

### 已降级为二线
- `runtime replay safe plugin`

因此请你不要再建议我回到这些路线当主线，除非你能基于下面证据明确推翻现有结论。

## 3. 已经判死的旧线证据

### 3.1 proxy0213 noop baseline

`outputs/competition_assoc_online_noop_proxy0213_20260323_094948/result.csv`

- HOTA = `52.758`
- AssA = `44.038`
- IDF1 = `58.276`
- MOTA = `73.232`
- IDSW = `847`

### 3.2 row-local oracle rerank 仍输给 noop

`outputs/competition_assoc_online_oracle_rerank_proxy0213_20260323_141625/result.csv`

- HOTA = `52.220`
- AssA = `43.202`
- IDF1 = `58.316`
- MOTA = `72.753`
- IDSW = `1012`

结论：
- 这不是 learned 没学到，而是 `single-row / row-local winner correction` 这个 decision unit 本身不值主线预算
- 所以 `row-local rerank` 已经正式 `KILL`

## 4. 为什么主矛盾不在 continuity，而在 local competition

cluster anatomy 结果来自：
`outputs/local_conflict_graph_cluster_anatomy_20260323_154625/summary.json`
`outputs/local_conflict_graph_cluster_anatomy_20260323_154625/sequence_cluster_summary.csv`

关键统计：
- `recoverable_groups_total = 4751`
- `recoverable_overlap_groups_in_multi_detection_clusters = 4416`
- `recoverable_overlap_groups_multi_detection_share = 0.92949`
- `recoverable_overlap_cluster_avg_detections = 4.676`
- `recoverable_overlap_cluster_avg_tracks = 16.844`
- `bridge_groups_total = 751`
- `bridge_overlap_groups_in_multi_detection_clusters = 373`
- `bridge_overlap_groups_multi_detection_share = 0.49667`
- `bridge_overlap_cluster_avg_detections = 1.401`
- `bridge_overlap_cluster_avg_tracks = 9.690`

结论：
- 当前主导残差不是 long-gap continuity
- 当前主矛盾是同帧局部竞争里的多 detection / 多 track 耦合
- 所以当前主线不该转去 stitching / continuity

## 5. 当前 graph 线已经拿到的关键正负证据

### 5.1 语义过强的 full cluster replacement oracle 会伤主流程

`outputs/local_conflict_graph_fullcluster_oracle_proxy0213_20260323_201852/result.csv`

- HOTA = `45.669`
- AssA = `47.405`
- IDF1 = `54.349`
- MOTA = `49.495`
- IDSW = `599`

结论：
- 直接 full cluster replacement 这版在线语义过强，虽然某些 identity 指标有局部收益，但整体主流程被伤到
- 所以“更强模块”不能简单等于“更激进地整块替换 host”

### 5.2 语义更弱的 commit-matches oracle 略正

`outputs/local_conflict_graph_commitmatches_oracle_proxy0213_20260323_205803/result.csv`

- HOTA = `52.824`
- AssA = `44.241`
- IDF1 = `58.477`
- MOTA = `73.123`
- IDSW = `886`

相对 noop：
- HOTA / AssA 略正
- 但 IDSW 仍偏高

### 5.3 加 hard cluster trigger 后，proxy0213 变成明确正号

`outputs/local_conflict_graph_commitmatches_hardtrigger_oracle_proxy0213_20260323_212612/result.csv`

主点 `topk=8, min_detections=2, min_committed_matches=2`：
- HOTA = `53.175`
- AssA = `44.949`
- IDF1 = `59.036`
- MOTA = `73.219`
- IDSW = `873`

同批 proxy surface：
- `md2/mm2`: `53.175 / 44.949 / 59.036 / 73.219 / 873`
- `md2/mm3`: `53.094 / 44.582 / 59.160 / 73.275 / 838`
- `md3/mm2`: `52.498 / 43.690 / 57.966 / 73.457 / 845`
- `md4/mm2`: `53.072 / 44.601 / 59.049 / 73.259 / 861`

结论：
- `min_detections=2` 是甜点
- `mm=2` 最优于 `HOTA/AssA`
- `mm=3` 更保守，能压 `IDSW`

### 5.4 full FRCNN 四点面已经收齐

来自：
`outputs/local_conflict_graph_commitmatches_hardtrigger_next12h_20260323_230551/summary.csv`

- `05 md2/mm2`: `HOTA 61.858 / AssA 57.957 / IDF1 70.705 / MOTA 75.882 / IDSW 1609`
- `06 md2/mm3`: `HOTA 61.844 / AssA 57.856 / IDF1 70.739 / MOTA 75.897 / IDSW 1574`
- `07 md3/mm2`: `HOTA 61.698 / AssA 57.615 / IDF1 70.412 / MOTA 75.946 / IDSW 1581`
- `08 md4/mm2`: `HOTA 61.837 / AssA 57.859 / IDF1 70.709 / MOTA 75.892 / IDSW 1597`

当前结论：
- `md2` 仍是最稳主设置
- `mm=2` 有最高 `HOTA/AssA`
- `mm=3` 基本守住主指标，同时把 `IDSW` 从 `1609` 降到 `1574`
- `md>=3` 不值得作为主配置继续扩

## 6. 当前我对这条线的判断

我现在的判断不是 KILL，而是：
- `NARROW GO`

但我认为当前继续的前提不是再磨 oracle/hard-trigger，而是直接重设计更强模块。

也就是说，我不是要你告诉我“要不要继续 graph”。
我要你告诉我：

1. 这条主线下一版真正该学的模块是什么？
2. 这个模块应如何替代当前 `oracle_commit_matches + hard trigger` 的弱实现？
3. 该如何在当前代码基础上落地到工程级实现？

## 7. 当前代码入口，请直接基于这些文件做设计

### 核心 tracker 注入点

`models/runtime_tracker_bytetrack.py`

你可以直接从这些锚点看当前实现：
- `__init__` 中 local-conflict 配置接入：约 `246`, `477-520`
- `_get_local_conflict_graph_oracle_commit_matches_plan(...)`：约 `1764`
- 旧 row-local 路径 `_refine_assoc_scores_with_competition(...)`：约 `1945`
- 主匹配流程 `_match_with_scores(...)`：约 `2298`
- local graph branch / commit path：约 `2387-2412`

### config / submit / train 接口

`submit_bytetrack.py`
- local-conflict config 传参：约 `647-653`
- diagnostics 汇总：约 `782-798`
- CLI/config override：约 `1080-1094`

`train_bytetrack.py`
- local-conflict config 传参：约 `465-471`

### 当前 graph 相关脚本

- `scripts/analyze_local_conflict_graph_clusters.py`
- `scripts/run_local_conflict_graph_fullcluster_oracle_proxy0213.sh`
- `scripts/run_local_conflict_graph_commitmatches_hardtrigger_oracle_proxy0213.sh`
- `scripts/run_local_conflict_graph_commitmatches_hardtrigger_oracle_generic.sh`

请注意：
- 当前真正在线生效的 still 是 oracle/heuristic cluster policy
- 当前还没有一个与你建议的新 decision unit 真正对齐的 learned cluster module

## 8. 这次我希望你直接完成的任务

请你直接回答下面 6 个问题，并在后半部分给出完整实现设计文档：

1. 基于上面的证据链，这条主线现在应判定为 `GO / NARROW GO / KILL` 中的哪一个？为什么？
2. 如果继续，当前最应该设计的“更强模块”是什么？你必须只选一个主方案，不要给我很多备选方向。
3. 这个更强模块，应该如何继承当前已经验证过的有效约束：
   - `primary-only`
   - `pre-Hungarian`
   - `top-k observed-only`
   - `conservative cluster trigger`
4. 这个模块的输出应该是什么？
   - edge logits?
   - cluster commit policy?
   - local assignment logits?
   - row/track dual heads?
   你必须明确选一种主输出设计。
5. 训练 supervision 和 online operator 应该如何对齐？当前 mismatch 在哪里？
6. 第一版最小可落地实现应该是什么？先跑什么唯一 first-priority experiment？

## 9. 额外硬要求：这次必须直接给代码级重设计文档

这次请不要只给方向判断，我还需要你直接对当前代码做“可执行级别的重设计”。

你的回答必须包含两部分：

### Part A. 管理级决策

你必须明确给出：
- `GO / NARROW GO / KILL`
- 为什么
- 下一步唯一 first-priority experiment 是什么
- 哪些事情现在不要再做

### Part B. 代码级实现设计文档

请你基于我提供的代码包，直接写一份可执行的 `MD` 设计文档。
这份文档不能停留在概念层，必须具体到代码改动层。

你的实现设计文档，至少必须包含以下内容：

1. 总体模块设计
- 新主模块叫什么
- 它替代 / 升级的是哪条旧线
- 在线注入点在哪
- 输入是什么
- 输出是什么
- 与 host 的接口是什么

2. 文件级改动清单
对于每一个需要修改或新增的文件，你都必须写清楚：
- 文件路径
- 为什么改这个文件
- 当前这个文件里相关逻辑是什么
- 需要修改的函数 / 类 / 配置段是什么
- 修改前大致在哪几行到哪几行
- 如果无法保证精确行号，也必须给明确代码锚点：
  - 函数名
  - 类名
  - 关键变量名
  - 可 grep 的关键字符串
- 修改后应该变成什么
- 是“局部改写”还是“新增模块后接入”
- 这部分改动的风险点是什么

3. 请按下面这种格式逐项写

### Change 1
- File:
- Current code responsibility:
- Current anchor / lines:
- Problem in current implementation:
- New implementation:
- Exact modification plan:
- Whether old path stays for ablation:
- Dependencies:

### Change 2
- File:
- Current code responsibility:
- Current anchor / lines:
- Problem in current implementation:
- New implementation:
- Exact modification plan:
- Whether old path stays for ablation:
- Dependencies:

依次列完。

4. 新增文件
如果你建议新增文件，也必须写清楚：
- 新文件路径
- 这个文件负责什么
- 为什么不能只在旧文件里硬改
- 这个文件的核心类 / 函数列表
- 与现有哪些文件交互

5. 配置与脚本
你必须明确指出：
- 哪些 config 文件要新增参数
- 哪些 shell runner 要新增
- 哪些训练 / 推理 / oracle / eval 脚本要改
- 结构化记录怎么记：
  - `result.csv`
  - `summary.csv`
  - `experiment_registry.csv`
  - bundle

6. 最小可落地版本
请你明确给出：
- 第一版最小实现是什么
- 它应该先跑什么实验
- 这个最小版本不做什么
- 哪些复杂部分先延后

7. 不允许只给高层建议
以下形式都不算合格回答：
- 只说“建议引入图模块”
- 只说“可以做 joint assignment”
- 只说“建议改成 transformer / graph network”
- 只给方向，不给文件级落点
- 只给伪代码，不给代码入口

我需要的是：
- 能直接指导工程修改的文档
- 最好是我拿到文档后，就能按文件逐项实现

8. 输出格式
请你最后直接输出一份完整的 Markdown 文档正文。
文档标题请写成：

`IMPLEMENTATION_REDESIGN_PLAN.md`

文档中必须有：
- 总体结论
- 文件级改动表
- 逐文件详细修改说明
- 第一批实验执行顺序
- 风险与回退策略

## 10. 最后再强调一次边界

这次不要再回答：
- “方向没问题，可以继续看看”
- “建议试试更强 GNN”
- “可以考虑 Sinkhorn / transformer / graph matching”

这些都不够。

我需要的是：
- 一个单一主设计
- 一个工程上可落地的代码改造计划
- 能直接对当前代码仓实现的文件级设计文档
```
