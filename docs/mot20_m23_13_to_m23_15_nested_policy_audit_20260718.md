# MOT20 M23-13—M23-15 nested policy audit（2026-07-18）

## 结论

- 正式可部署锚点仍为 **77.699 HOTA**。本批实验使用的父 tracker 是固定的、推理不读 GT 的历史探索候选 **78.763497 HOTA**，但其上游策略族曾在同一开发集上后验选择，因此不能替代正式锚点。
- M23-13 的严格 nested sequence-LOSO 最坏序列风险约束在四个外层折全部选择 no-op；TrackEval 与父结果逐项一致。它证明全局损失倍率可以安全拒绝域外风险，但召回为零。
- M23-14 在内层三序列采用中位效用选择损失倍率，并用 GT-free 正例概率 q90 的单侧上界做 OOD/no-op 门控。冻结后得到 **78.790003 HOTA / 76.174194 AssA / 908 IDSW**，相对父结果 **+0.026506 HOTA / +0.050819 AssA / +7 IDSW**。这是当前最佳探索性 nested 结果，不是正式可部署成绩。
- M23-15 将绝对风险阈值替换为互斥 top-K 秩预算，得到 **78.738510 HOTA / 76.076126 AssA / 919 IDSW**，低于父结果 **−0.024987 HOTA**，因此关闭该方向。
- M02 仍完全 no-op，HOTA 保持 **71.574910**、AssA **63.638735**。下一阶段应改变 M02 的域稳健表征或候选级校准，而不是继续扫描全局 λ/K。

## 共同协议与边界

1. 外层 held-out 序列不参与该折的模型训练或策略参数选择。
2. M23-13 的内层折在其余三个序列中再留一；模型只用另外两个序列训练。M23-14/M23-15 复用这些已落盘的 inner-LOSO 预测。
3. 推理特征来自显式 GT-free allowlist。GT 仅用于训练序列的 transaction utility 标签，以及所有冻结文件写出之后的诊断和 TrackEval。
4. 每个父 track 在一个折中最多参与一个新事务；事务执行语义为 source 前缀接 destination 后缀，并移除被替代的 source/destination 段。
5. M23-14 的 q90 OOD 规则是在同一四序列开发过程内根据 M23-12/M23-13 失败形态提出；即使外层选择本身不读 GT，方法族仍有开发集适配偏差，所以 `deployment_allowed=false`。

## M23-13：最坏内层序列风险校准

预注册损失倍率：`1, 2, 4, 8, 16, 32, 64, 128, 256, 512`。

评分为：

`P(positive) * predicted_gain - lambda * (1 - P(positive)) * predicted_loss`

选择规则：取最小的、能让三个 inner-held 序列真实事务效用总和都非负且至少选择一个事务的 λ；不存在则该外层折 no-op。

四个外层折均不存在安全非空 λ：

| Outer held | 选择 λ | 事务数 | 结果 |
|---|---:|---:|---|
| MOT20-01 | no-op | 0 | 与父结果一致 |
| MOT20-02 | no-op | 0 | 与父结果一致 |
| MOT20-03 | no-op | 0 | 与父结果一致 |
| MOT20-05 | no-op | 0 | 与父结果一致 |

冻结选择 SHA-256：`9c8fe5a4f20c6d42821e314c7ece1f72bd3061aad82372dae2722140d4a543ac`。

TrackEval：

| Tracker | HOTA | DetA | AssA | IDSW |
|---|---:|---:|---:|---:|
| 固定探索父结果 | 78.763497 | 81.551695 | 76.123375 | 901 |
| M23-13 | 78.763497 | 81.551695 | 76.123375 | 901 |

解释：约束确实阻止了 M05 的重尾灾难，但任何一个内层负序列都会否决整个外层策略，导致可用正事务也全部丢失。

## M23-14：中位效用 + 单侧 q90 OOD 门

### 冻结规则

- λ：选择三个 inner-held 序列真实事务效用总和的中位数最大的正值 λ；不存在正中位数则 no-op。
- OOD：仅使用预测正例概率 `pred_transaction_positive_prob_unweighted`。若 outer q90 高于三个 inner-held q90 的最大值，则判为“过度自信外推”并强制 no-op。
- 单侧而非双侧：本批建模的失败是高置信过量选择；低置信外推不会放大动作数，因此不触发 OOD。

冻结策略：

| Outer held | 校准 λ | outer q90 | inner q90 max | OOD | 应用 λ | 事务数 |
|---|---:|---:|---:|---|---:|---:|
| MOT20-01 | 8 | 0.098974 | 0.140491 | 否 | 8 | 2 |
| MOT20-02 | no-op | 0.097506 | 0.106479 | 否 | no-op | 0 |
| MOT20-03 | 2 | 0.055211 | 0.199268 | 否 | 2 | 17 |
| MOT20-05 | 16 | 0.156460 | 0.087010 | **是** | no-op | 0 |

- 选择清单 SHA-256：`6bfe522112a874ff7ac3efed0b9f877d8cf89cf6743170b5ac72428a388c9a4e`
- 策略表 SHA-256：`b05135c6ec4edf1bb5aca91e482b3f31232f4c587326712699d705176ebef7d2`

冻结后诊断：

| 序列 | 事务数 | 正事务 | 真实 transaction proxy 总和 | HOTA | 相对父序列 HOTA | IDSW |
|---|---:|---:|---:|---:|---:|---:|
| MOT20-01 | 2 | 0 | −1.037829 | 78.818700 | −0.000350 | 40 |
| MOT20-02 | 0 | 0 | 0 | 71.574910 | 0 | 286 |
| MOT20-03 | 17 | 10 | +691.695154 | 80.665165 | +0.093549 | 147 |
| MOT20-05 | 0 | 0 | 0 | 79.491440 | 0 | 435 |
| **COMBINED** | **19** | **10** | **+690.657325** | **78.790003** | **+0.026506** | **908** |

M05 的 q90 门控是本批最重要的正证据：它只使用 test 可得预测分布，准确识别并拒绝了此前 −21454.71（top-10）到百万量级（宽阈值）的域偏移风险。与此同时，M01 的两个极低分动作仍为轻微负效用，M02 则没有恢复任何召回。

## M23-15：nested rank-budget 反证

使用 M23-12 已存在的排名预算子网格 `K={10,25,50,100,250}`。每个 inner-held 序列先按预测 expected utility 排序，再做 track-disjoint 贪心；选择 inner 真实效用中位数最大的正 K，并复用同一 q90 OOD 门。

冻结结果只有 M03 应用 K=10；M01/M02 找不到正中位预算，M05 被 OOD 门拒绝。

- 选择清单 SHA-256：`25d02a87b69859de5f573fab91bdb07ce3831f920a33b9be5e7bc2bfd1c4ba60`
- 策略表 SHA-256：`86d1dcc333c4aa20095c73a793f7a5e2e8acc90bb66582d0659a49d7366012b4`
- M03 的 10 个事务冻结后真实 proxy 总和为 **−608.372315**；M03 HOTA 从 80.571616 降至 **80.483025**，IDSW 从 139 增至 157。
- combined 为 **78.738510 HOTA / 81.550664 DetA / 76.076126 AssA / 919 IDSW**。

这解释了历史 top-10 诊断与可执行策略的差异：历史排名审计直接取最高的十行，未施加 track-disjoint 事务约束；实际执行为避免一个父 track 被多次改写，会跳过冲突行并补入后续候选，动作集合已经不同。因此不能再引用非互斥 top-K 的正效用作为部署证据。

## 决策

- 保留 M23-14 作为 **GT-free inference、nested selection 的探索性机制新高**；不得写成正式 deployable benchmark，也不得替代 77.699 锚点。
- 关闭 M23-13 全局最坏序列 λ 门和 M23-15 全局 rank-budget。
- 下一实验必须直接针对 M02：优先研究 sequence-normalized/域稳健的 transaction 表征，以及候选级 uncertainty/OOD，而不是继续调整 λ、K 或 q90 分位。

## 产物

- `scripts/m23_research/m23_13_nested_chain_risk_policy.py`
- `scripts/m23_research/m23_14_sequence_ood_gate.py`
- `scripts/m23_research/m23_15_nested_rank_budget_policy.py`
- `outputs/mot20_m23_20260718/m23_13_nested_chain_risk_policy_v1/nested_chain_risk_policy_v1/report.json`
- `outputs/mot20_m23_20260718/m23_14_sequence_ood_gate_v1/sequence_ood_median_risk_policy_v1/report.json`
- `outputs/mot20_m23_20260718/m23_15_nested_rank_budget_policy_v1/nested_rank_budget_ood_policy_v1/report.json`
