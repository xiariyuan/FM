# Pro Reply: After Learned Commit Queue (2026-03-24)

这份文件保存 2026-03-24 这一轮 Pro 审阅的核心回答，供后续新的 Pro 或本地实现继续继承。

## 对应提问

- prompt: `md/PRO_REVIEW_SEND_TO_PRO_AFTER_LEARNED_COMMIT_QUEUE_20260324.md`

## Pro 核心结论

- 判定: `NARROW GO`
- 当前唯一主方案: `A. stronger host migration`
- 备选: `B. larger training data`
- 当前不应先做: `C. model strengthening`

## 为什么不是 B/C

- 当前 learned commit 已经回答了“在 base_reid_da 上是否有 operator 价值”，答案是有，但不大。
- 下一步最高信息增益不是继续在当前 host 上堆训练或堆模型，而是先回答它对更强 host 是否仍可迁移。
- 当前训练面极小，约 `282 train / 4 val`。在这个阶段直接增强模型容量，只会把数据瓶颈和 host 迁移问题混在一起。

## A 里的 host 选择

### 主选

- `configs/experiments/bytetrack_fa_mot_mot17_v15_laplace_reid_da_val0213.yaml`

### 备选

- `configs/experiments/bytetrack_fa_mot_mot17_v13_tf_only_val0213_reid_da.yaml`

### 不选

- `configs/experiments/bytetrack_fa_mot_mot17_v16_laplace_trainable_val0213.yaml`

## 选择 v15 的理由

- v15 是最干净的“更强 host 最小迁移”。
- 它在当前 repo 里有现成 Laplace proxy/matrix runner 参考。
- 研究问题更干净: `operator` 能否和更强但仍固定的 association cue 叠加。

## 为什么 v13 只是备选

- v13 也能做固定 checkpoint 的 portability check。
- 但它更像一次 host 语义大切换，解释性不如 v15 干净。

## 为什么 v16 现在不该碰

- v16 会把 `host retraining + operator migration` 两件事绑在一起。
- 这样做出来无法解释增益到底来自哪个部分。

## 下一步唯一 first-priority experiment

做一个 `v15` 上的 zero-shot stronger-host paired proxy0213 migration:

1. `host-only baseline`: v15 host，不挂 learned commit
2. `host + learned commit`: 同一个 v15 host、同一个 host checkpoint，挂当前 learned commit checkpoint

固定参数:

- `topk=8`
- `min_detections=2`
- `min_committed_matches=2`
- `max_detections=8`
- `max_tracks=32`
- 评测面: `proxy0213`

## 验收标准

- 相对 v15 host-only，`HOTA / AssA / IDF1` 能否仍为正
- `eligible_clusters / replaced_clusters` 不能是空跑
- 如果 proxy0213 成立，再去 full FRCNN `md2/mm2`
- 如果 proxy0213 不成立，直接切到 `B. larger training data`，不要先上 `C`

## Pro 给的实现建议

- 不需要再改 `models/runtime_tracker_bytetrack.py` 主逻辑
- 主要改 runner:
  - `scripts/run_local_conflict_graph_learned_commit_proxy0213.sh`
  - `scripts/run_local_conflict_graph_learned_commit_generic.sh`
- 让 host config / host checkpoint / host variant 参数化
- 删除把 host 模块强行关掉的 override
- 新增:
  - `scripts/run_local_conflict_graph_host_migration_proxy0213.sh`
  - `configs/experiments/bytetrack_fa_mot_mot17_v17_local_conflict_commit_hostv15_val0213.yaml`

## 当前不该再做

- 不回 row-local rerank
- 不回 full cluster replacement
- 不提前拉 continuity / stitching
- 不在当前 base_reid_da 上继续做一轮 md/mm/topk 小 sweep 来代替 host migration
- 不先上 model strengthening
- 不把 v16 当第一次 stronger-host migration
