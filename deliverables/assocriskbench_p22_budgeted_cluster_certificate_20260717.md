# AssocRiskBench P22：固定预算机制补救与序列聚类证书审计

日期：2026-07-17

## 1. 研究问题

P21 已证明 receiver-split 与 same-identity-merge 机制排序能够补回 P20 唯一漏失的正例时间块，但 block-rank conformal 校准因为每折只有 8–17 个机制正例块，被迫采用最大源域排名，导致平均新增 98.86 个候选、最大集合 218。

P22 不再扩大 P20 通用 rank-conformal 半径，也不重新训练或调节 P21 机制模型，而是回答两个独立问题：

1. 在 P21 预注册的平均新增上限 12 内，机制联合排序能否形成紧致、高召回的候选集合？
2. 该集合能否在考虑同序列时间块相关性的情况下，获得目标为 10% miss risk、90% confidence 的统计证书？

这两个问题必须分开。候选检索成功不等于部署授权成功。

## 2. 固定协议

- 基础集合：冻结的 P20 rank-conformal set，共 1,304 个成员。
- 机制分数：冻结的 P21 sequence-OOF receiver-split 与 same-identity-merge 分数。
- 每个时间块的补救预算：固定为 12。
- 预算来源：直接继承 P21 预注册的 maximum mean added candidates = 12，不做 top-k sweep。
- 排序：两种机制块内排名的最小值；并按 receiver-split rank、merge rank 与事件键确定性打破并列。
- 通用 P20 半径不变。
- 不进行模型、预算、阈值、alpha 或序列特定参数搜索。
- 不调用 TrackEval，不读取 P15 locked labels，156 条剩余 locked rows 保持未读。

## 3. 紧致候选集合结果

固定预算策略在 28 个时间块中各增加 12 个 P20 集合外候选，共增加 336 个事件。

| 指标 | P20 | P22 组合集合 |
|---|---:|---:|
| 正例可用时间块 | 23 | 23 |
| 覆盖正例时间块 | 22 | 23 |
| 条件覆盖率 | 95.65% | **100%** |
| 平均集合大小 | 46.57 | **58.57** |
| 平均新增候选 | 0 | **12.00** |
| 最大集合大小 | 87 | **99** |
| set-oracle utility sum | 6.3149 | **6.7787** |
| 最差序列 set-oracle utility | 0.2861 | **0.2861** |

因此，P22 同时满足固定的检索接受条件：

- 23/23 正例时间块覆盖；
- 至少补回 1 个 P20 漏失块；
- 平均新增不超过 12；
- 最大组合集合不超过 100；
- 所有序列 set-oracle utility 非负。

正式决定：

> `compact_candidate_set_retained = true`

这意味着 P20+P21 可以形成一个平均约 59 个候选、最大 99 个候选的完整检索层，不需要采用 P21 的大半径 conformal rescue set。

## 4. 唯一漏失块如何被恢复

恢复的时间块仍是 `MOT17-11-FRCNN / temporal block 3`。

预算集合中的第 6 个候选为：

- canonical rank：532
- transaction：`1 -> 59`，`u_to_v`
- boundary frame：736
- teacher utility：`+0.183036`
- mechanism：same-identity merge
- merge block rank：7
- union rank：7
- budget rank：6

因此，紧致策略不是通过扩大 generic P20 半径恢复该块，而是利用 P21 的机制专属排序，将一个 base-set 外 merge 正例加入固定预算集合。

两个 receiver-split 正例没有进入前 12。这说明当前紧致恢复主要依赖 merge 分支；split 分支仍存在明显的高分伪阳性问题。

## 5. 伪阳性负担

336 个新增候选中：

- 正效用：39
- 负效用：293
- 零效用：4

组合集合共包含：

- 正效用事件：246
- 负效用事件：1,225
- 零效用事件：169

因此，P22 解决的是 candidate retrieval，而不是单事件 authorization。集合中仍有大量不能执行的负效用候选。

机制覆盖也不均衡：

| 机制 | 事件召回 | 正例块覆盖 |
|---|---:|---:|
| Receiver split | 19.60% | 14/18 |
| Same-identity merge | 70.83% | 12/12 |

不能依据当前机制分数直接选 top-one 或开放执行权限。

## 6. Cluster-aware 统计证书

如果错误地把 23 个正例时间块视为独立样本，观察到 0 次漏失时，90% 单侧 Clopper–Pearson miss-risk 上界为：

`0.095264`

它低于目标 0.10，表面上会得到通过结论。

但是同一序列中的四个时间块共享场景、相机、运动模式、密度和 tracker failure mode，不能视为 23 个独立试验。使用序列作为独立 cluster：

- 独立序列数：7
- 出现正例块漏失的序列数：0
- 90% 单侧 miss-risk 上界：`0.280314`

该上界显著高于目标 0.10。

正式决定：

> `cluster_risk_certificate_passed = false`

> `deployment_allowed = false`

P22 的关键统计结论是：

> block-level 样本量足以制造一个看似通过的 9.53% 上界，但在正确处理序列内相关性后，七个独立 domain clusters 只能支持 28.03% 的 miss-risk 上界。

因此，不能把时间块级证书作为部署安全保证。

## 7. Appearance 可观测性准备

P22 的多组诊断表明，仅使用 tracker geometry、score、motion 和 P21 change-point 特征时：

- receiver-split 正例经常被高分负例压过；
- base-conditioned rescue classifier 不能跨域恢复 MOT17-11；
- base-oracle improvement classifier 的 sequence-OOF AP 仅约 0.019；
- receiver change-point 在部分 namespace split 中并不表现为明显运动突变。

这支持一个新的可观测性判断：namespace split 需要 appearance 或原始 association evidence，而不仅是导出的轨迹几何。

为此已建立确定性的 sparse boundary ReID manifest：

- 事件：11,705
- 角色窗口索引：163,958
- 去重 crop：60,708
- 图像帧：3,284
- 每个事件角色使用 1–5 帧
- 角色：donor history、receiver history、receiver future
- crop 重复键：0
- event-role-position 重复键：0

使用资产已经固定：

- FastReID config SHA256：`5410bfe270162e062ca2e6d14c73a8bcb9f9bb8fb3eabb8650329c2afac4168d`
- MOT17 FastReID weights SHA256：`eb2e83afe774c85f20b735a7fabc423659c022387187eba449419c18dd6b4fa9`

当前宿主推理服务在 FastReID 初始化前返回 VM service unavailable，因此本阶段没有生成或声称生成 appearance embeddings。

正式状态：

- `manifest_ready = true`
- `appearance_features_ready = false`

## 8. 研究结论

P22 得到一个正负并存但非常重要的结果：

1. **正结果**：机制分解可以在固定 12-event budget 内补回 P20 唯一漏失块，形成满足规模约束的完整候选检索层。
2. **负结果**：七个独立序列不足以为该检索层提供 10% sequence-level miss-risk 的 90% 置信证书。
3. **可观测性结果**：纯 geometry/motion 仍无法有效清除 split 机制的极端伪阳性；appearance 或原始 association evidence 是下一步必要输入。

论文主线应将 P22 表述为：

> Budgeted mechanism routing closes the retrieval gap, while cluster-aware uncertainty exposes the insufficiency of block-wise certification under domain dependence.

## 9. 下一阶段

P23 应固定 P22 的紧致集合，不改变预算 12，执行：

1. 按 sparse manifest 提取 donor-history、receiver-history 和 receiver-future appearance prototypes；
2. 构建 donor-future similarity、receiver-future similarity、appearance margin、prototype stability 和 missingness 特征；
3. 在严格 sequence-LOSO 下只对 P22 的 336 个新增候选进行 appearance-conditioned pruning/reranking；
4. 检查能否保留 23/23 覆盖，同时显著降低 293 个 rescue negatives；
5. 在增加更多独立域之前，继续禁止部署授权。

## 10. 可复现性与锁定状态

- Budgeted cluster certificate formal/repro：全部文件字节一致。
- Sparse ReID manifest formal/repro：全部文件字节一致。
- P15 locked labels read：0。
- P15 locked TrackEval calls：0。
- Global TrackEval calls：0。
- Remaining locked rows：156，全部未读。
- Locked deployment manifest：未创建。
- P15 policy：保持 no-op。

关键报告 SHA256：

- Budgeted cluster report：`bf1acddfca1003682e76c752e4dd57a61b46e45a396cb05e8662d5358260f4cf`
- Sparse ReID manifest report：`dc3df76f527d454dfc7a8a99e384e1ddd3b47d9fe5ffdd090fc8a681535befa9`
- Unified audit：`d72bd85983ec91274061ec2f0814d46d6e5e74afbb955f14525d55479b1f9617`
