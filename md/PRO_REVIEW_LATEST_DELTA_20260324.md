# Pro Review Latest Delta (2026-03-24)

这份文档只记录相对长期上下文的新变化。每次新的 Pro 提问前，优先更新这份文档，而不是重写长期背景。

## 1. 本轮新增的关键事实

- learned conservative partial commit 主线已经从设计文档变成了实际代码。
- 当前 runtime 已支持 `ASSOC_LOCAL_CONFLICT_GRAPH_MODE=learned_commit`。
- 新主模块不是 full replacement，也不是 row-local controller，而是 cluster-level conservative partial commit。

## 2. 本轮已落地的代码

- `models/local_conflict_commit.py`
- `models/local_conflict_graph_common.py`
- `models/runtime_tracker_bytetrack.py`
- `scripts/build_local_conflict_commit_dataset.py`
- `scripts/train_local_conflict_commit_stage1.py`
- `scripts/run_local_conflict_commit_stage1.sh`
- `scripts/run_local_conflict_graph_learned_commit_proxy0213.sh`
- `scripts/run_local_conflict_graph_learned_commit_generic.sh`
- `scripts/queue_local_conflict_graph_learned_commit_next12h.sh`
- `configs/experiments/bytetrack_fa_mot_mot17_v17_local_conflict_commit_val0213.yaml`
- `submit_bytetrack.py`
- `train_bytetrack.py`
- `scripts/analyze_local_conflict_graph_clusters.py`
- `scripts/build_pro_review_bundle.py`

## 3. 本轮实验队列

队列根目录:

- `/gemini/code/FMtrack-main/FM-Track/outputs/local_conflict_graph_learned_commit_next12h_20260324_165558`

当前队列设计:

1. stage1 train
2. proxy eval
3. full FRCNN `md2/mm2`
4. full FRCNN `md2/mm3`
5. full FRCNN `md3/mm2`
6. full FRCNN `md4/mm2`

## 4. 截至当前已完成结果

### 4.1 stage1

- `best_epoch = 12`
- `train_examples = 282`
- `val_examples = 4`
- `train_loss = 1.0702726181751738`
- `val_loss = 0.7789765996858478`

### 4.2 proxy0213 learned commit

- `HOTA 53.755`
- `AssA 46.125`
- `IDF1 59.856`
- `MOTA 73.166`
- `IDSW 869`

### 4.3 full FRCNN learned commit

- `md2/mm2`: `61.995 / 58.274 / 70.930 / 75.868 / 1605`
- `md2/mm3`: `61.763 / 57.705 / 70.497 / 75.885 / 1583`
- `md3/mm2`: `61.995 / 58.274 / 70.930 / 75.868 / 1605`
- `md4/mm2`: `61.800 / 57.765 / 70.619 / 75.879 / 1582`

### 4.4 v15 host migration paired proxy0213

run root:

- `/gemini/code/FMtrack-main/FM-Track/outputs/local_conflict_graph_hostmig_v15_proxy0213_20260324_194915`

paired result:

- `v15 host_only`: `HOTA 53.069 / AssA 44.392 / IDF1 59.099 / MOTA 73.752 / IDSW 684`
- `v15 + learned_commit`: `HOTA 52.850 / AssA 43.973 / IDF1 58.978 / MOTA 73.722 / IDSW 693`

delta:

- `delta_HOTA = -0.219`
- `delta_AssA = -0.419`
- `delta_IDF1 = -0.121`
- `delta_MOTA = -0.030`
- `delta_IDSW = +9`

但这不是空跑:

- `eligible_clusters = 5769`
- `replaced_clusters = 138`
- `matched_dets = 278`

## 5. 这轮结果相对旧 control 的意义

对照 `oracle_commit_matches + hard trigger`:

- proxy 上 learned commit 明确更好。
- full FRCNN 至少在 `md2/mm2` 上有稳定小幅增益。
- 这说明真正有价值的不是 oracle 细节本身，而是 conservative partial commit 这个 operator。

但也必须明确:

- 绝对指标仍然不够强。
- 当前收益量级更像“确认 operator 有价值”，而不是“已经到论文终局”。

## 6. 当前主判断

当前最合理的研究判断是:

- operator 方向成立
- module 已初步转正
- 但 `v15` 上的 first-shot zero-shot portability 当前为负
- 这说明它不是“拿来就能迁移到更强 host”的宿主无关模块
- 当前最优先的下一阶段，更像在:
  - `B. larger training data`
  - 是否还有必要补 `v13` 这个备选 host
  - 之后才考虑更强模型

## 7. 下一次真正值得问 Pro 的问题

当前不值得再问泛泛的“方向是否继续”。

下一次问 Pro，应该只在以下两类决策点二选一:

1. stronger host selection
2. learned commit 的 next-step redesign only after current queue fully closes

更具体地说，应等当前 full FRCNN 队列跑完，再拿两三个具体 host 候选去问，而不是开放式地问“该换什么 host”。

## 8. 新增 Pro 裁决

当前这一步已经问完 Pro，结论是:

- 主方案: `A. stronger host migration`
- 主选 host: `v15_laplace_reid_da_val0213`
- 备选 host: `v13_tf_only_val0213_reid_da`
- 当前不选: `v16_laplace_trainable_val0213`

这意味着当前最优先的直接动作不是:

- 在 `base_reid_da` 上继续做一轮小 sweep
- 先扩模型容量

而是:

- 做 `v15` 上的 zero-shot paired proxy0213 migration

## 9. 对上一次 Pro 裁决的最新证据修正

`v15` 这枪已经打完，结果是负的。

因此当前已经明确知道:

- `A. stronger host migration` 的第一主选 `v15` 没过
- 问题不是空跑，因为模块确实介入了 cluster commit
- 问题更像是:
  - current learned commit 对 stronger host 不具备直接 zero-shot portability
  - 或者 current training data / target 面太窄，无法支撑跨 host 迁移

所以当前新的单一决策点变成:

- 现在是否应直接切 `B. larger training data`
- 还是还值得先补打一枪 `v13` 作为最后一个 portability check

## 10. 新增 Pro 裁决

这个决策点已经再次问完 Pro，当前最新裁决是:

- 近期主方案: `B. larger training data`
- 近期备选: `A2. v13 portability check`
- 中期唯一更强模块: `LocalConflictSetPredictor`

这意味着:

- 当前 first priority 已不再是 stronger-host zero-shot portability
- 当前也不该先做 model strengthening
- 近期最值钱的一枪是:
  - `large_data_base_retrain_v1`

## 11. 当前仓库层面的现实约束

当前本地现成可直接复用的 cluster-commit 训练数据，仍然只有一份:

- `outputs/competition_assoc_base_reid_da_proxy0213_hybriddumpfix/labeled_replay_top8.groups.jsonl`
- `outputs/competition_assoc_base_reid_da_proxy0213_hybriddumpfix/competition_cases/competition_cases.csv`

也就是说:

- 仓库里还没有已经现成准备好的 larger-data cluster dataset
- 要真正执行 `B`，还需要补:
  - larger base runtime dump
  - runtime replay label build
  - manifest-based dataset builder
  - strict sequence-split trainer

## 12. 更新规则

每次新一轮实验结束后，这份文档至少更新以下内容:

- 新增了哪些代码文件
- 新拿到的关键结构化结果
- 当前主结论有没有变化
- 下一次需要问 Pro 的唯一具体问题是什么

## 13. 当前正在运行的 `large_data_base_retrain_v1`

当前运行根目录:

- `/gemini/code/FMtrack-main/FM-Track/outputs/local_conflict_commit_large_base_20260324_222409`

当前 top-level 状态:

- `status=running`
- `current_stage=06_proxy_eval`

这轮已经不是旧的 tiny proxy stage1，而是更大的 base-host 数据面:

- dataset summary:
  - `sequences = 7`
  - `eligible_clusters = 1103`
  - `train_examples = 817`
  - `val_examples = 286`
  - `train_frames = 3961`
  - `val_frames = 1348`
- stage1 summary:
  - `best_epoch = 5`
  - `train_loss = 0.4932`
  - `val_loss = 1.5500`
  - `train_row_acc = 0.7891`
  - `val_row_acc = 0.4540`
  - `val_commit_precision = 0.4736`
  - `val_commit_recall = 0.8366`

这说明:

- 这次已经补上了 Pro 要求的 larger-data + strict sequence split
- 旧的 `282 train / 4 val` 问题已被修正
- 当前正在等 proxy0213 与后续 full FRCNN `md2/mm2` 回测

## 14. 当前新增的真实判断

截至当前，最像主病因的仍然是四层叠加:

1. tiny-data + 极差 val split
2. current `v1_raw` feature / score semantics 对 host shift 过敏
3. loss 只有 `[local tracks + defer]` 的 row-wise CE，缺 edge-level 与 cluster-level safety supervision
4. `LocalConflictCommitRefiner v1` 作为最小 MLP baseline，表达力不足以稳定处理更复杂冲突

当前需要 Pro 回答的新问题已经不再是:

- “这条主线还要不要继续”

而是:

- 在当前 operator 语义固定不变的前提下，如何设计一个明显强于 `LocalConflictCommitRefiner v1` 的 `v2` 模块
- 以及它的最小可落地 loss / feature / runtime 接口 / first-priority experiment 是什么

## 15. 下一次要问 Pro 的唯一具体问题

当前推荐的新提问是:

- 已完成，不再是待问问题。

## 16. 这次新增的 stronger v2 裁决

这次 Pro 已经对 stronger `v2` 给出明确裁决:

- 不再补 `v13` portability check
- 近期主方案仍是 `B. larger training data`
- 中期唯一更强模块是 `HostConditionedLocalConflictSetPredictor`

这意味着当前主线切成两层:

1. 近期:
   - 让 larger-data `v1` 跑完，回答“数据面扩大后 current operator 能否先稳住”
2. 中期:
   - 直接把 stronger `v2` 落地成 host-conditioned local set predictor

## 17. 已落地的 stronger v2 代码

当前仓库已经实现完整 `set_predictor_v2` 主链，而不是最小 MVP:

- `models/local_conflict_set_predictor.py`
- `scripts/build_local_conflict_set_predictor_dataset_manifest.py`
- `scripts/build_local_conflict_set_predictor_dataset.py`
- `scripts/train_local_conflict_set_predictor.py`
- `scripts/run_local_conflict_graph_set_predictor_proxy0213.sh`
- `scripts/run_local_conflict_graph_set_predictor_generic.sh`
- `scripts/run_local_conflict_set_predictor_stage1_large_base.sh`
- `configs/experiments/bytetrack_fa_mot_mot17_v18_local_conflict_set_predictor_val0213.yaml`

runtime 接口也已接入:

- `models/runtime_tracker_bytetrack.py`
- `submit_bytetrack.py`
- `train_bytetrack.py`

## 18. v2 smoke 结果

smoke run root:

- `/gemini/code/FMtrack-main/FM-Track/outputs/local_conflict_set_predictor_smoke_20260325_005923`

smoke dataset:

- `eligible_clusters = 1103`
- `train_examples = 817`
- `val_examples = 286`

smoke trainer 最终通过:

- `status = ok`
- `best_epoch = 1`
- `train_loss = 0.7833`
- `val_loss = 1.9033`
- `train_row_acc = 0.7220`
- `val_row_acc = 0.4265`
- `val_commit_precision = 0.4240`
- `val_commit_recall = 0.6340`
- `val_edge_ap = 0.3934`
- `val_cluster_f1 = 0.8560`

中途还拿到一个有价值的实现层证据:

- 早期 `NaN` 问题不是方向问题，而是实现细节问题。
- 已通过以下修复解决:
  - 去掉 in-place host conditioning
  - 默认学习率从 `1e-3` 降到 `3e-4`
  - trainer 增加 non-finite sample / batch guards

## 19. 当前正在运行的正式 v2 队列

当前正式运行根目录:

- `/gemini/code/FMtrack-main/FM-Track/outputs/local_conflict_set_predictor_large_base_20260325_013200`

当前设计:

1. 复用现成 large-base runtime rows/group jsonl
2. 构建 `cluster_set_predictor_data`
3. 训练 `set_predictor_v2`
4. 跑 proxy0213
5. 跑 full FRCNN `md2/mm2`

当前固定参数:

- `topk = 8`
- `epochs = 12`
- `hidden_dim = 128`
- `num_heads = 4`
- `num_conflict_blocks = 2`
- `batch_size = 8`
- `min_val_examples = 64`
- `min_detections = 2`
- `min_committed_matches = 2`
- `max_detections = 8`
- `max_tracks = 32`
- `cluster_gate_thresh = 0.5`

当前要求:

- 以后对外报告状态时，统一以该目录下的:
  - `summary.csv`
  - `result.csv`
  - 分阶段 `summary.csv`
  为准

## 20. 当前主判断更新

当前仓库层面已经不只是“知道该做 v2”，而是:

- `v2` 代码已落地
- smoke 已过
- 正式 large-base `v2` 队列已启动

所以当前真正等待回答的问题变成:

- `set_predictor_v2` 相对 enlarged-data `v1`，在 base host 上能否先拿到更强、更稳的 proxy/full 正号

- 不等 `large_data_base_retrain_v1` 最终收尾
- 直接并行向 Pro 请求 stronger-`v2` 设计
- 但约束必须锁死:
  - 不回 row-local rerank
  - 不回 full cluster replacement
  - 不改 operator 语义
  - 固定为 `cluster-level conservative partial commit + defer to host`

对应提示词:

- `md/PRO_REVIEW_SEND_TO_PRO_STRONGER_V2_WHILE_LARGEDATA_RUNNING_20260324.md`

## 21. stable v2 正式结果已完成

stable run root:

- `/gemini/code/FMtrack-main/FM-Track/outputs/local_conflict_set_predictor_large_base_stable_20260325_023500`

stable stage1:

- `best_epoch = 3`
- 训练不再在 `epoch2+` 发生整片 `Infinity`
- 当前 trainer 稳定性问题已显著修正

stable proxy0213:

- `HOTA 53.118`
- `AssA 44.577`
- `IDF1 58.730`
- `MOTA 73.437`
- `IDSW 811`

stable full FRCNN `md2/mm2`:

- `HOTA 63.257`
- `AssA 60.191`
- `IDF1 72.128`
- `MOTA 76.055`
- `IDSW 1481`

## 22. stable v2 相对 enlarged-data v1 的意义

相对 enlarged-data `v1`:

- proxy:
  - `HOTA +0.072`
  - `AssA +0.163`
  - `IDF1 -0.897`
  - `MOTA -0.024`
  - `IDSW +15`
- full `md2/mm2`:
  - `HOTA +0.566`
  - `AssA +1.040`
  - `IDF1 +0.314`
  - `MOTA +0.059`
  - `IDSW -44`

这说明:

- `v2` 方向已经不再需要重新证明
- stable `v2` 在 `full` 上仍然有真实正号
- 但 stable `proxy` 收缩明显

## 23. 当前主判断再次收束

当前主问题已经不再是:

- 要不要继续这条线
- 要不要重新设计 whole new stronger module

而更像是:

- 为什么 stable `full` 仍然正而 stable `proxy` 收缩
- 以及下一步唯一最值钱的增强点到底是:
  - `gate calibration`
  - `loss/objective`
  - `feature semantics`
  - `proxy-full mismatch`

## 24. 下一次值得问 Pro 的唯一具体问题

当前值得问 Pro 的问题已经收窄为:

- 在 stable `v2` 已证明 full 为正、proxy 收缩的前提下
- 下一步唯一 first-priority enhancement 应该是什么
- 只允许 `v2.1` 级别增强，不允许重开大架构

对应提示词:

- `md/PRO_REVIEW_SEND_TO_PRO_AFTER_STABLE_V2_BASE_RESULTS_20260325.md`
