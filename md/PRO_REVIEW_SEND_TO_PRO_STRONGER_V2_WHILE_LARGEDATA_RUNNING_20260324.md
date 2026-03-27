# Pro Review Request: Stronger V2 Design While Large-Data Run Is In Flight

请把下面这些文件视为本次问题的权威上下文，并在此基础上回答:

- `md/PRO_REVIEW_CANONICAL_CONTEXT_20260324.md`
- `md/PRO_REVIEW_LATEST_DELTA_20260324.md`
- `md/PRO_REVIEW_INTERACTION_LOG.md`

如果你打开了我附的代码包，请优先核对:

- `models/local_conflict_commit.py`
- `models/local_conflict_graph_common.py`
- `models/runtime_tracker_bytetrack.py`
- `scripts/build_local_conflict_commit_dataset.py`
- `scripts/train_local_conflict_commit_stage1.py`
- `scripts/run_local_conflict_commit_stage1_large_base.sh`
- `scripts/run_local_conflict_graph_learned_commit_proxy0213.sh`
- `scripts/run_local_conflict_graph_learned_commit_generic.sh`

## 这次不要重复裁决的事情

这些结论都已经固定，请不要回头重开:

- row-local rerank 已死
- full cluster replacement 已死
- 当前正确 operator 语义是:
  - cluster-level
  - conservative partial commit
  - defer to host
  - primary-only
  - pre-Hungarian
- 当前近期 first-priority experiment 已经在跑:
  - `large_data_base_retrain_v1`

## 当前最新事实

### 已完成的旧证据

1. tiny-data learned commit 曾在 `base_reid_da` 上拿到真实但不大的正号:

- proxy0213:
  - `HOTA 53.755`
  - `AssA 46.125`
  - `IDF1 59.856`
  - `MOTA 73.166`
  - `IDSW 869`
- full FRCNN `md2/mm2`:
  - `HOTA 61.995`
  - `AssA 58.274`
  - `IDF1 70.930`
  - `MOTA 75.868`
  - `IDSW 1605`

2. 但 tiny-data stage1 本身很不健康:

- `train_examples = 282`
- `val_examples = 4`
- `val_commit_precision = 0`
- `val_commit_recall = 0`

3. `v15` stronger-host zero-shot portability 已被否掉:

- `delta_HOTA = -0.219`
- `delta_AssA = -0.419`
- `delta_IDF1 = -0.121`
- 且不是空跑:
  - `eligible_clusters = 5769`
  - `replaced_clusters = 138`
  - `matched_dets = 278`

### 当前正在运行的 larger-data 结果

当前运行根目录:

- `/gemini/code/FMtrack-main/FM-Track/outputs/local_conflict_commit_large_base_20260324_222409`

当前状态:

- top-level `summary.csv` 显示:
  - `status=running`
  - `current_stage=06_proxy_eval`

larger-data dataset summary:

- `sequences = 7`
- `eligible_clusters = 1103`
- `train_examples = 817`
- `val_examples = 286`
- `train_frames = 3961`
- `val_frames = 1348`

larger-data stage1 summary:

- `best_epoch = 5`
- `train_loss = 0.4932`
- `val_loss = 1.5500`
- `train_row_acc = 0.7891`
- `val_row_acc = 0.4540`
- `val_commit_precision = 0.4736`
- `val_commit_recall = 0.8366`

所以当前已经确定:

- larger-data + strict sequence split 这件事已经真正落地
- 旧的 `282 train / 4 val` 问题已被修正
- 但最终 proxy/full eval 还没全部收尾

## 我现在真正要你做的事情

我不需要你再回答“要不要继续这条线”。

我需要你直接做下面两件事:

### Part A. 管理级设计裁决

请只回答这一个问题:

> 在当前 operator 语义固定不变的前提下，下一代 stronger-`v2` 模块应该如何设计，才能比当前 `LocalConflictCommitRefiner v1` 明显更稳、更强、对 host shift 更不敏感？

请直接给:

- 唯一主方案
- 一个备选
- 为什么不是几个明显替代方案

这里的“固定不变”指:

- 不回 row-local
- 不回 full replacement
- 不回 continuity / stitching 主线
- 不改在线注入点:
  - primary-only
  - pre-Hungarian
- 不改最终 operator 语义:
  - cluster-local conservative partial commit
  - unmatched / abstained rows defer to host

### Part B. 工程级实现说明

请直接输出一个可落地的 `v2` 设计文档，至少覆盖:

1. 新模块名称
2. 结构设计
3. 输入特征
4. host-shift robustness 要怎么做
5. 输出头设计
6. loss 设计
7. runtime 接口
8. dataset / supervision 要怎么升级
9. 文件级改动清单
10. 唯一 first-priority experiment

## 这次回答的硬要求

### 1. 不要泛泛地说“可以考虑 Transformer / GNN / Sinkhorn”

我需要的是:

- 一个唯一主方案
- 明确到模块命名、输入、输出、loss、脚本、配置、文件改动

### 2. 不要让我等 larger-data 最终结果再来问

你现在就可以给 stronger-`v2` 的完整设计。

因为我现在的目的不是替代当前 large-data 队列，而是并行准备:

- 如果 large-data 仍然不能让 `v1` 稳住
- 那么下一步要立刻转哪个 `v2`

### 3. 必须回答 loss 设计

当前 `v1` 的核心问题之一是 loss 太弱:

- 只有 row-wise CE over `[local tracks + defer]`
- 没有 edge-level auxiliary loss
- 没有 cluster-level trigger/safety loss

你必须明确:

- `v2` 的 loss 组合是什么
- 各项 loss 分别解决什么问题
- 为什么它会比 current CE baseline 更稳

### 4. 必须回答 host-shift robustness

因为 `v15` negative portability 是当前最关键的负证据之一。

你必须明确:

- current `v1` 为什么会 host-shift 脆弱
- `v2` 的 feature / normalization / conditioning 怎么缓解这个问题

## 我当前自己的判断，你可以直接反驳

我当前的主判断是:

- operator 方向仍然成立
- current `v1` 的主病因像是四层叠加:
  1. tiny-data + bad split
  2. host-shift sensitive raw feature semantics
  3. loss too weak
  4. model expression too weak

如果你同意，请把 stronger-`v2` 的设计建立在这个判断上。
如果你不同意，请明确指出你认为哪个病因判断是错的。

## 希望的回答格式

请直接按下面格式回答:

1. `管理级决策`
2. `为什么`
3. `唯一主方案`
4. `为什么不是其他几个明显方案`
5. `实现设计文档`
6. `唯一 first-priority experiment`
7. `当前不要做什么`

我要的是可以直接照着改仓库的回答，不要高层空话。
