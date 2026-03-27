# Pro Reply: After V15 Negative (2026-03-24)

这份文件保存 `v15` host migration 为负之后，最新一轮 Pro 审阅的核心回答，供后续新的 Pro 与本地实现继续继承。

## 对应提问

- prompt: `md/PRO_REVIEW_SEND_TO_PRO_DECISION_AND_OPEN_REDESIGN_AFTER_V15_NEGATIVE_20260324.md`

## Pro 核心结论

- 判定: `NARROW GO`
- 近期执行主方案: `B. larger training data`
- 近期备选: `A2. still run one last portability check on v13`
- 当前不要再做:
  - 先补 `v13`
  - 先做 `C. model strengthening`
  - 回 row-local / full replacement / continuity

## 为什么现在应先转 B

- `v15` 负迁移已经足够回答当前最关键的问题:
  - 当前 tiny-data / v1-MLP checkpoint 不能直接 zero-shot 迁到 stronger host
- 当前模型和特征强依赖 host score 分布:
  - 直接吃 `base_score / refined_score / motion_score / track_gap / hist_len / rank_frac`
  - 没有 host embedding
  - 没有 cluster 内 score normalization
  - 没有显式 same-row / same-col 结构交互
- 所以再打一枪 `v13`，测到的更多还是 covariate shift，不是更本质的 operator 真假

## 为什么 v13 不是当前 first priority

- `v13` 若再负，结论不变，仍然要回 `B`
- `v13` 若侥幸转正，也更像 host-specific compatibility 信息，而不是改变近期工程动作
- 因此 `v13` 的额外信息增益，不足以压过 larger-data 重训

## 近期唯一 first-priority experiment

`large_data_base_retrain_v1`

目标:

- 先把当前 `v1 LocalConflictCommitRefiner` 放到更大的 `base_reid_da` cluster-commit 数据上重训
- 再回 base host 验证

不是:

- 先跑 `v13`
- 先做 `v2`
- 先扩 sweep

## 近期执行顺序

1. 生成更大的 `base_reid_da` runtime dump
2. 构建 larger cluster-commit dataset
3. 用更大数据重训当前 `v1`
4. 只回测:
   - `proxy0213`
   - `full FRCNN md2/mm2`

## 近期实现建议

### 要改的文件

- `scripts/build_local_conflict_commit_dataset.py`
- `scripts/train_local_conflict_commit_stage1.py`
- `scripts/run_local_conflict_graph_learned_commit_proxy0213.sh`
- `scripts/run_local_conflict_graph_learned_commit_generic.sh`

### 要新增的文件

- `scripts/build_local_conflict_commit_dataset_manifest.py`
- `scripts/run_local_conflict_commit_stage1_large_base.sh`

## 近期实现要点

### Dataset builder

从“单一 `group_jsonl + cases_csv`”升级为“多 source manifest”。

建议新增参数:

- `--source-manifest`
- `--train-sequences`
- `--val-sequences`
- `--strict-sequence-split`
- `--min-val-examples`
- `--feature-version`

每条 example 新增字段:

- `source_tag`
- `host_variant`
- `split_tag`
- `feature_version`

### Trainer

必须解决当前:

- 验证集过小
- silent fallback 到 80/20

建议新增:

- `--strict-sequence-split`
- `--min-val-examples`

当 val 不满足阈值时:

- 直接 fail
- 不允许 silent fallback

### New runner

`scripts/run_local_conflict_commit_stage1_large_base.sh`

负责:

1. 生成或收集 full base dump
2. 生成 `source-manifest.csv`
3. 调 dataset builder
4. 调 trainer
5. 自动接着跑:
   - `proxy0213`
   - `full FRCNN md2/mm2`

## 中期唯一更强模块

`LocalConflictSetPredictor`

一句话定义:

- 一个 host-conditioned 的局部 ID set-prediction transformer
- 在每个 local conflict cluster 内
- 把 `detection -> [local tracks + defer]` 当成小型动态 ID 预测 / 集合预测问题

## 它与当前 v1 的关系

- 选择: `升级`

不是替代整条主线，也不是两阶段协同。

在线语义不变:

- 仍是 `partial commit + defer to host`
- 仍是 `primary-only / pre-Hungarian`
- 仍是 `edge logits + defer logits`

变的是:

- 从 `edge-MLP + pooling` 升级成更结构化的 cluster set predictor

## 中期最小可落地版本

建议做:

- `models/local_conflict_set_predictor.py`

结构:

- detection token
- track token
- feasible edge token
- cluster token
- 2 层 transformer / attention block
- same-row / same-col 显式结构 bias
- 输出:
  - `edge_logits`
  - `defer_logits`
  - `cluster_commit_logit`

推理:

- 仍保留 `private defer + Hungarian`
- 再叠 conservative gate

## 为什么不是别的明显备选

- 不是纯 GNN
- 不是纯 Sinkhorn 头
- 不是直接抄整机 `MOTR / TrackFormer / MOTIP`

核心原因:

- 当前 repo 已经围绕 `Hungarian + private defer + plugin operator` 打通
- 更稳的路线是吸收 set prediction / ID prediction 思想，但不重写整机 tracker
