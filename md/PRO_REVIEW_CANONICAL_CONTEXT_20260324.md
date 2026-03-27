# Pro Review Canonical Context (2026-03-24)

这份文档是给新的、没有上下文的 Pro reviewer 的长期固定背景。后续每次提问，都应把这份文档和最新增量文档一起发送，而不是只发单次问题。

## 1. 这份文档的用途

- 告诉新的 Pro: 哪些路线已经正式停掉，不要重新讨论。
- 告诉新的 Pro: 当前唯一主线是什么，当前实现已经走到哪一步。
- 告诉新的 Pro: 哪些证据已经足够成为硬约束。
- 减少每次从零重讲历史的成本。

## 2. 项目当前的基本设定

- 任务: tracking-by-detection 范式下的 MOT 研究。
- 当前代码宿主: ByteTrack 风格 runtime host。
- 当前固定验证宿主: `base_reid_da`。
- 当前研究对象: 可插拔 learned association module。
- 当前主故事: 不是 detector，不是 continuity，不是大而全 graph，而是 primary association 中的局部竞争决策单元。

## 3. 已经明确停掉或降级的旧路线

### 3.1 已停主线

- `single-row / row-local rerank controller`
- `Laplace / MTCR / HACA / pairwise residual safe plugin`
- `frequency-aware / spatial-freq interaction` 作为主线

### 3.2 已降级二线

- `runtime replay safe plugin`

### 3.3 当前不进入主故事

- `continuity / stitching / bridge`
- `full cluster replacement` 作为在线主语义

## 4. 为什么旧 row-local 线已经判死

proxy0213 noop baseline:

- `HOTA 52.758`
- `AssA 44.038`
- `IDF1 58.276`
- `MOTA 73.232`
- `IDSW 847`

row-local oracle rerank:

- `HOTA 52.220`
- `AssA 43.202`
- `IDF1 58.316`
- `MOTA 72.753`
- `IDSW 1012`

结论:

- 这不是 learned 没学到，而是 decision unit 本身不对。
- 所以 row-local winner correction 已经正式退出主线。

## 5. 为什么当前主矛盾是 local competition

cluster anatomy 的结论已经足够稳定:

- recoverable overlap groups 的主体集中在 multi-detection local conflict。
- continuity / bridge 占比和重心都不是当前第一矛盾。

因此当前主线应围绕:

- frame-local conflict cluster
- one-to-one competitive assignment
- conservative online intervention

而不是 continuity。

## 6. 当前真正有效的在线语义

### 6.1 被负证据否掉的语义

`full cluster replacement oracle`:

- `HOTA 45.669`
- `AssA 47.405`
- `IDF1 54.349`
- `MOTA 49.495`
- `IDSW 599`

结论:

- full replacement 会伤 host 主流程。
- 所以更强模块不等于更激进地整块替换 host。

### 6.2 已转正的语义

`oracle_commit_matches + hard trigger` on proxy0213:

- `HOTA 53.175`
- `AssA 44.949`
- `IDF1 59.036`
- `MOTA 73.219`
- `IDSW 873`

这条线说明:

- cluster-level operator 是有价值的。
- 但正确语义是 conservative partial commit，而不是 full replacement。

## 7. 当前唯一主线

当前唯一主线已经收敛为:

- 模块名: `LocalConflictCommitRefiner`
- decision unit: frame-local bipartite conflict cluster
- online semantics: `partial commit + defer to host`
- 注入时机: `primary-only`, `pre-Hungarian`
- 输入约束: `top-k observed-only`
- one-to-one 约束: cluster-local Hungarian with private defer columns
- 输出语义: 对每个 detection 在 `[local tracks + defer]` 上做局部 assignment

换句话说:

- 不学 `keep / rerank / null`
- 不学 full cluster replace
- 只学哪些局部匹配值得提前提交，剩余 detection defer 给 host 收尾

## 8. 当前实现已经落地到代码

本仓库当前已经实现了 learned commit 主线，核心文件包括:

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

## 9. 当前已拿到的 learned evidence

### 9.1 stage1

- train examples: `282`
- val examples: `4`
- best epoch: `12`
- val loss: `0.7789765996858478`

### 9.2 proxy0213 learned commit

- `HOTA 53.755`
- `AssA 46.125`
- `IDF1 59.856`
- `MOTA 73.166`
- `IDSW 869`

相对当前 proxy oracle hard-trigger control:

- HOTA 更高
- AssA 更高
- IDF1 更高
- MOTA 略低
- IDSW 量级相近

### 9.3 full FRCNN learned commit 已完成点

- `md2/mm2`: `61.995 / 58.274 / 70.930 / 75.868 / 1605`
- `md2/mm3`: `61.763 / 57.705 / 70.497 / 75.885 / 1583`
- `md3/mm2`: `61.995 / 58.274 / 70.930 / 75.868 / 1605`

对照 oracle hard-trigger control:

- `md2/mm2`: `61.858 / 57.957 / 70.705 / 75.882 / 1609`
- `md2/mm3`: `61.844 / 57.856 / 70.739 / 75.897 / 1574`
- `md3/mm2`: `61.698 / 57.615 / 70.412 / 75.946 / 1581`

当前解释:

- learned commit 已经不是纯方向证明，而是有真实小正号。
- 但绝对指标仍不够强，不足以单独支撑论文终局。
- 下一阶段更可能是 stronger host 或 larger data，而不是回旧路线。

## 10. 当前甜点与部署偏好

当前已知更像主配置的是:

- `md2/mm2` 作为默认主点
- `md3/mm2` 作为接近对照点

当前不建议作为主配置继续扩:

- `md2/mm3`, 除非明确要更保守地压 IDSW
- `md>=4`

## 11. 当前最可能的下一步

如果 learned commit 在当前队列结束后仍维持稳定正号，下一步更合理的是:

1. 换更强 host 验证 operator 的可迁移价值。
2. 扩大训练数据规模，而不是先做大 sweep。
3. 只在 operator 稳定后，再考虑 trigger learning 或更强模型。

## 12. 当前不应该再建议的事情

不要再把以下内容当成主线建议:

- 回到 row-local rerank
- 把 full replacement 当主语义继续推进
- 提前拉 continuity / stitching 回主故事
- 在旧 `competition_assoc.py` 路径上继续加头
- 大规模扫一堆 top-k / tie-break / trigger 小补丁来替代模块重设计

## 13. 使用方式

每次向新的 Pro 提问，最少一起携带:

1. 这份 `canonical context`
2. 最新的 `delta` 文档
3. `interaction log`
4. 本轮真正要提问的 `send-to-pro` 提示词

如果只发第 4 份，不发前 3 份，就很容易让新的 Pro 重复打开旧路线。
