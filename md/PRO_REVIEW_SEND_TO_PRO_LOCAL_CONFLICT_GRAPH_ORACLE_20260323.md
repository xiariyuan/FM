# 发给 Pro 的完整提示词：Local Conflict Graph Oracle（2026-03-23）

你现在不用补历史上下文，我把这条线到今天的关键证据压缩如下，请你直接给 go/kill 与下一步实验决策。

## 1. 已经正式 kill 的旧主线

旧主线是 `single-row / row-local rerank controller`。

这条线不是只停当前 learned deployment，而是把这个 decision unit 一并 kill。

最硬证据：

- `noop` online proxy0213：
  - `HOTA 52.758 / AssA 44.038 / IDF1 58.276 / MOTA 73.232 / IDSW 847`
  - 路径：
    - `outputs/competition_assoc_online_noop_proxy0213_20260323_094948/result.csv`
- `oracle_rerank` online proxy0213：
  - `HOTA 52.220 / AssA 43.202 / IDF1 58.316 / MOTA 72.753 / IDSW 1012`
  - 路径：
    - `outputs/competition_assoc_online_oracle_rerank_proxy0213_20260323_141625/result.csv`

也就是说，`top-8 + oracle winner + minimal winner override` 仍然输给同一 harness 下的 `noop`。

所以我们已经确认：

- 不是简单的“模型没学到”
- 不是“trigger 还能再调”
- 不是“operator 再磨一磨”

而是 `row-local winner correction` 这个 decision unit 本身不值得继续做主线。

## 2. 新主线已经正式切到什么

新主线已经切到：

`competition-aware local conflict graph`

理解方式：

- 冻结 `base_reid_da` host
- 只动 `primary association`
- `pre-Hungarian`
- 把 decision unit 从 “单 detection row 的 top-k rerank”
  升级到 “同帧局部冲突 cluster 内的联合 one-to-one assignment”

当前只是 oracle upper bound，不是 learned graph model。

## 3. cluster anatomy 修正后的硬统计

我们先把 cluster anatomy 重跑并修正了 summary 字段。

可信 run：

- `outputs/local_conflict_graph_cluster_anatomy_20260323_154625/result.csv`
- `outputs/local_conflict_graph_cluster_anatomy_20260323_154625/summary.json`

关键统计：

- coarse frame bipartite clusters：
  - `1350`
- recoverable overlap clusters：
  - `1016`
- recoverable groups in multi-detection overlap clusters：
  - `4416 / 4751 = 0.9295`
- recoverable overlap cluster avg size：
  - `4.676 detections / 16.844 tracks`
- bridge overlap clusters：
  - `536`
- bridge groups in multi-detection overlap clusters：
  - `373 / 751 = 0.4967`
- bridge overlap cluster avg size：
  - `1.401 detections / 9.690 tracks`

这组数支持：

- 当前主导残差更像 multi-detection local conflict
- 不像 continuity / stitching 是第一主矛盾

## 4. 新的 oracle local conflict graph upper bound 结果

我们已经实现并跑完了第一版：

- `primary-only`
- `pre-Hungarian`
- `top-k = 8`
- `min multi-detection cluster size = 2`
- oracle 来源：
  - `outputs/competition_assoc_base_reid_da_proxy0213_hybriddumpfix/labeled_replay_top8.groups.jsonl`

run 路径：

- `outputs/local_conflict_graph_oracle_proxy0213_20260323_161845/result.csv`
- `outputs/local_conflict_graph_oracle_proxy0213_20260323_161845/run_manifest.json`
- bundle：
  - `outputs/local_conflict_graph_oracle_bundle_20260323_162433.zip`

总体结果：

- `HOTA 52.998 / AssA 44.821 / IDF1 58.699 / MOTA 72.958 / IDSW 930`

相对 `noop`：

- `HOTA +0.240`
- `AssA +0.783`
- `IDF1 +0.423`
- `MOTA -0.274`
- `IDSW +83`

相对旧 `oracle_rerank`：

- `HOTA +0.778`
- `AssA +1.619`
- `IDF1 +0.383`
- `MOTA +0.205`
- `IDSW -82`

也就是说：

- 新 decision unit 的 oracle upper bound 已经是正号
- 它明显强于旧的 row-local oracle
- 但还不是“全面大胜 noop”，因为 `MOTA` 仍略降，`IDSW` 也还偏高

## 5. 序列分解

按 `pedestrian_detailed.csv` 拆到序列：

### MOT17-02-FRCNN

graph oracle 相对 noop：

- `HOTA +0.009`
- `AssA +0.020`
- `IDF1 +0.011`
- `MOTA -0.002`
- `IDSW +36`

### MOT17-13-FRCNN

graph oracle 相对 noop：

- `HOTA -0.007`
- `AssA -0.011`
- `IDF1 -0.006`
- `MOTA -0.004`
- `IDSW +47`

也就是说：

- 总体正号是成立的
- 但这版 graph oracle 还不是稳到每条序列同号
- 它已经证明 decision unit 升级是对的
- 但 cluster definition / tie-break / unmatched policy 可能还需要一轮结构性收紧

## 6. 现在我希望你直接回答的 6 个问题

1. 你是否同意：  
   现在已经可以正式确认 `local conflict graph` 作为新主 direction 是 `GO`，因为它的 oracle upper bound 已经打过了 `noop`，而且显著强于旧 `oracle_rerank`？

2. 这个结果在你看来属于什么强度？  
   是：
   - 仅仅说明 “比 row-local 更对”，但还不够支撑 learned mainline  
   还是
   - 已经足够支持下一步直接做 learned graph baseline

3. 如果继续，你建议下一步最有信息增益的是哪个？
   只在下面几项里选一个作为 first priority：
   - `cluster construction ablation`
   - `oracle graph policy ablation`
   - `simple learned graph baseline`
   - `seq-held-out validation rebuild`

4. 如果你选 `oracle graph policy ablation`，你最建议先改哪一个？
   - cluster 构造
   - unmatched/null policy
   - oracle tie-break
   - top-k / score-window 稀疏化

5. 如果你选 `simple learned graph baseline`，你建议第一版最小实现是什么？
   请只给“最小可验证版本”，不要给大模型方案。

6. 现在 continuity / stitching 这条线是否仍然应该继续延后？
   我当前理解是：继续延后，不进这版主故事。请你明确表态同不同意。

## 7. 我自己的当前判断

我当前判断是：

- `row-local rerank`：正式 kill，作为 negative evidence 保留
- `local conflict graph`：正式 go
- 下一步优先级：
  - 先做一轮很窄的 graph-side structure ablation
  - 然后直接上 `simple learned local graph baseline`
- `continuity / stitching`：继续延后，不进当前主故事

但我希望你不是复述我，而是直接按上面的硬证据链给独立 go/kill judgement。

## 8. 你回答时请尽量直接给管理级决策

我最需要的是这种格式：

- `GO / NARROW GO / KILL`
- 为什么
- 下一步唯一 first-priority experiment 是什么
- 哪些东西现在不要做

不要泛泛综述文献，我现在需要的是明确实验决策。
