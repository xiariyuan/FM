# Pro Review Request: Stable V2 Base-Host Results Are In, What Is The Single Best Next Enhancement?

请把下面这些文件视为本次问题的权威上下文，并在此基础上回答:

- `md/PRO_REVIEW_CANONICAL_CONTEXT_20260324.md`
- `md/PRO_REVIEW_LATEST_DELTA_20260324.md`
- `md/PRO_REVIEW_INTERACTION_LOG.md`

如果你打开了代码包，请优先核对:

- `models/local_conflict_set_predictor.py`
- `models/runtime_tracker_bytetrack.py`
- `scripts/build_local_conflict_set_predictor_dataset.py`
- `scripts/train_local_conflict_set_predictor.py`
- `scripts/run_local_conflict_graph_set_predictor_proxy0213.sh`
- `scripts/run_local_conflict_graph_set_predictor_generic.sh`
- `scripts/run_local_conflict_set_predictor_stage1_large_base.sh`

## 这次不要重开的结论

这些结论都已经固定，请不要回头重开:

- row-local rerank 已死
- full cluster replacement 已死
- 当前 operator 语义固定为:
  - cluster-level
  - conservative partial commit
  - defer to host
  - primary-only
  - pre-Hungarian
- 当前 stronger module 主线已经固定为:
  - `HostConditionedLocalConflictSetPredictor`

## 当前最新已完成结果

### A. enlarged-data `v1`

run root:

- `outputs/local_conflict_commit_large_base_20260324_222409`

top-level:

- `proxy0213`: `53.046 / 44.414 / 59.627 / 73.461 / 796`
- `full md2/mm2`: `62.691 / 59.151 / 71.814 / 75.996 / 1525`

### B. first `v2` run, training fragile but online stronger

run root:

- `outputs/local_conflict_set_predictor_large_base_20260325_013200`

top-level:

- `proxy0213`: `53.434 / 45.051 / 59.735 / 73.636 / 755`
- `full md2/mm2`: `63.418 / 60.614 / 72.605 / 76.180 / 1386`

但训练器本身不稳定:

- `best_epoch = 1`
- `epoch 2+` 出现大面积 `Infinity`
- 线上结果本质上是靠 `epoch1` checkpoint 顶住

### C. stable `v2` run after stability fixes

run root:

- `outputs/local_conflict_set_predictor_large_base_stable_20260325_023500`

stage1:

- `best_epoch = 3`
- 前 12 轮不再出现整片 `Infinity`
- 当前 trainer 已不再在 `epoch2` 之后数值爆炸

stable online:

- `proxy0213`: `53.118 / 44.577 / 58.730 / 73.437 / 811`
- `full md2/mm2`: `63.257 / 60.191 / 72.128 / 76.055 / 1481`

## 当前已经可以确认的事实

1. `v2` 方向是真的有效，不是幻觉

相对 enlarged-data `v1`:

- `stable full md2/mm2` 仍然为正:
  - `HOTA +0.566`
  - `AssA +1.040`
  - `IDF1 +0.314`
  - `MOTA +0.059`
  - `IDSW -44`

2. 但 `stable proxy0213` 明显收缩

相对 enlarged-data `v1`:

- `HOTA +0.072`
- `AssA +0.163`
- `IDF1 -0.897`
- `MOTA -0.024`
- `IDSW +15`

3. 这意味着当前主问题已经不是:

- 要不要继续这条线
- 要不要重新发明一个全新更强模块

而更像是:

- 为什么 stable `v2` 在 `full` 仍然正，而在 `proxy` 上收缩
- 以及下一步唯一最值钱的增强点应该落在什么地方

## 我现在真正要你回答的唯一问题

请只回答这一个收束问题:

> 在当前 `stable v2` 已经证明 full 为正、proxy 收缩的前提下，下一步唯一 first-priority enhancement 应该是什么？

## 请只允许在下面几类里选一个主方案

你必须选一个唯一主方案，并给一个备选:

1. `gate / trigger calibration`
2. `loss reweight / objective rewrite`
3. `feature semantics / normalization refinement`
4. `proxy-vs-full distribution mismatch handling`

## 这次回答的硬要求

### 1. 不要重新设计一个全新大模块

不要再回答:

- “可以考虑更大的 transformer”
- “可以考虑 memory bank”
- “可以考虑多帧 decoder”
- “可以考虑 whole-tracker rewrite”

这次我只接受 `v2.1` 级别增强，不接受 `v3 from scratch`。

### 2. 必须解释为什么 stable full 还正、stable proxy 却收缩

你必须给一个主解释，而不是列一堆可能性。

### 3. 必须给出文件级改动点

请明确:

- 改哪个文件
- 改哪个函数 / 配置键 / runner
- 当前不要改哪些文件

### 4. 必须给唯一 first-priority experiment

而且只允许是:

- `base host` 上的下一轮 `v2.1` paired rerun

不允许把 first-priority experiment 再切回:

- stronger-host migration
- 新大模型
- 大 sweep

## 我当前自己的判断，你可以直接反驳

我当前的判断是:

- 当前大方向已经成立
- 现在最像主瓶颈的，不再是表达力不够，而更像:
  - `stable training` 把某些 aggressive commit 压回去了
  - `proxy` 对当前 gate / score calibration 更敏感
  - `full` 仍然能从更稳的局部提交里受益

如果你同意，请基于这个判断给 `v2.1` 方案。
如果你不同意，请明确指出你认为主瓶颈其实是什么。

## 希望的回答格式

请直接按下面结构回答:

1. `管理级决策`
2. `为什么 stable full 正而 proxy 收缩`
3. `唯一主方案`
4. `一个备选`
5. `文件级改动清单`
6. `唯一 first-priority experiment`
7. `当前不要做什么`

我要的是可以直接照着改仓库继续跑的回答，不要高层空话。
