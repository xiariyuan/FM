# AssocRiskBench P18–P19｜域条件校准闭环与 Full Executable Bank 扩展｜2026-07-16

## 1. 阶段问题

P17 已构建 MOT17 七序列独立反事实 teacher bank，但使用 geometry canonicalization 与 30-frame per-track spacing 后仅保留 705 个事件。固定 13-target multitask 虽将跨序列 positive AUC 提升至 0.7021，仍无法实现最差序列 utility 非负。

P18–P19 继续回答两个问题：

1. 对 705-event canonical bank 做严格 nested domain calibration，能否将 event discrimination 转化为安全授权？
2. canonicalization 是否丢失了关键正例；若保留全部 executable events，跨域排序与最差域风险能否改善？

冻结约束始终保持：

- P15 locked label 新读取：0。
- P15 locked TrackEval 新调用：0。
- 新 global TrackEval：0。
- P15 剩余 156 条 locked rows：全部未读。
- 未创建 locked manifest。
- P15 policy：继续 `no_op`。

## 2. P18：严格 Nested Domain Calibration

正式资产：

- `outputs/assocriskbench_p18_20260716/nested_domain_calibration_v1`

### 2.1 协议

外层持出一整条 MOT17 序列。对其余六条源序列，再执行内层 sequence-LOSO，生成完全 OOF 的三类 base view：

1. absolute local utility regression；
2. 13-target robust-normalized multitask regression；
3. sequence-rank-normalized positive-event probability。

Meta calibrator 仅使用内层 OOF base predictions、块内相对 rank、top margin 与三视角一致性特征。授权阈值只能从内层源序列选择，并必须同时满足：

- 覆盖全部六条源序列；
- 总 local utility 严格为正；
- 最差源序列 utility 非负。

若不存在这样的阈值，该 outer fold 必须 fail-closed。

### 2.2 结果

- Events：705。
- Sequences：7。
- Temporal blocks：28。
- Meta positive AUC：0.642519。
- Meta average precision：0.223438。
- Meta Spearman：0.063552。
- Authorized outer folds：0 / 7。
- Selected outer blocks：0。
- Selected utility：0。

七个 outer folds 的内层授权均为 0。说明在 canonical bank 上，即使使用严格 nested、多视角 domain calibration，也不存在同时满足源域覆盖与最差域非负的授权阈值。

**P18 决策：关闭 canonical-bank scalar calibration family。**

## 3. P19：Full Executable Event Teacher

正式资产：

- `outputs/assocriskbench_p19_20260716/full_executable_event_teacher_v1`

### 3.1 候选冻结协议

对每条 MOT17 训练序列：

1. 枚举所有存在 temporal overlap 的 unordered raw-track pair；
2. 在 pair 的最后共同帧审计两个 directional handoff；
3. 使用冻结 executor 检查 boundary state、continuation、donor reappearance、duplicate identity 和 changed-row 非零约束；
4. 保留全部 executable directions，不再做 geometry priority、spacing 或数量上限筛选；
5. 完整 event set 冻结后才读取 GT，生成 event-level 多时间尺度 local teacher。

候选枚举和 event freeze 均不使用 GT、utility 或 TrackEval。

### 3.2 数据规模

| 项目 | Full executable bank |
|---|---:|
| Ordered candidate directions | 27,336 |
| Executable events | 11,705 |
| Represented changed rows | 1,786,622 |
| Positive events | 1,050 |
| Negative events | 10,147 |
| Zero events | 508 |
| Positive fraction | 8.9705% |

### 3.3 Canonical bank 的正例召回

| 项目 | 数量 |
|---|---:|
| Full events | 11,705 |
| Canonical events | 705 |
| Full positive events | 1,050 |
| Canonical positive events | 104 |
| Positive recall | **9.9048%** |

P17 canonicalization 虽将正例率从 8.97% 提升至 14.75%，却丢失了约 90.1% 的 executable positive events。该发现证明此前失败不仅是模型问题，也包含 candidate-support truncation。

## 4. Full Executable Actual-Anchor Motion

正式资产：

- `outputs/assocriskbench_p19_20260716/full_executable_actual_anchor_motion_v1`

Motion builder 新增 metadata reconstruction 模式：

- 输入冻结 event 的 `donor_anchor`、`receiver_anchor`、`effective_start_frame` 与 `changed_rows`；
- 从 tracker rows 重建 receiver continuation geometry；
- 不读取 GT matching、row class、utility 或 TrackEval；
- 复用 P16 已验证的 actual-anchor future-motion 计算。

结果：

- Events：11,705。
- Feature columns：367。
- Compact preregistered features：36。
- Duplicate event keys：0。
- Maximum missing fraction：1.0423%。
- Aggregate-anchor exceptions：0。
- Forbidden feature columns：0。

兼容性回归：

- P16 原 changed-row 模式四文件全部字节一致；feature SHA-256 仍为 `932f4ea78c73a23745f8799c8e6bf10f2ce1b2b478524894cfbd0f8eade1b9f3`。
- P17 原 changed-row 模式四文件全部字节一致；feature SHA-256 仍为 `cff6631654548365075b74357dcb187ea92e3a59055f437de0362b9d1f62368a`。

## 5. P19 Full-Bank Sequence-LOSO

正式资产：

- `outputs/assocriskbench_p19_20260716/full_executable_domain_generalization_v1`

固定比较三条预注册证据轴，不执行模型或参数 sweep：

1. `geometry_positive`：26 个 topology/lifecycle/boundary geometry 特征上的 positive classifier；
2. `geometry_utility`：同一 geometry 特征上的 absolute utility regressor；
3. `motion_multitask`：36 个 actual-anchor motion 特征上的 13-target multitask regressor。

共同设置：500 trees、max depth 9、min leaf 4、max features 0.70、每条训练序列总 sample weight 相同。

### 5.1 Event-level 泛化

| Model | Spearman | Positive AUC | AP |
|---|---:|---:|---:|
| Geometry positive | 0.003643 | **0.845246** | **0.276505** |
| Geometry utility | **0.673480** | 0.499532 | 0.083500 |
| Motion multitask | 0.083582 | 0.536587 | 0.125215 |

三个模型呈现互补但不一致的信号：

- geometry classifier 能区分正例群体，但最高分尾部包含严重伪阳性；
- geometry utility 能拟合整体 utility 顺序，却不能识别稀有正例；
- motion multitask 的 pooled event 指标一般，但块内 top-one 更有效。

### 5.2 每序列四时间块 Top-One

| Model | Positive | Negative | Zero | Utility sum | Worst sequence | Positive sequences |
|---|---:|---:|---:|---:|---:|---:|
| Geometry positive | 5 | 23 | 0 | −4.910293 | −1.080373 | 1 |
| Geometry utility | 1 | 14 | 13 | −0.757051 | −0.671411 | 0 |
| Motion multitask | **12** | **12** | 4 | **+0.202141** | −0.769903 | **4** |

Full-bank motion multitask 首次令 28 个强制 top-one 的 pooled utility 转正，并使 MOT17-02、05、10、13 四条序列为正。这验证了扩大候选支持域是有效方向。

但最差序列仍失败：

- MOT17-04：−0.769903。
- MOT17-09：−0.264062。
- MOT17-11：−0.643856。

因此 pooled utility 转正不能替代 worst-domain 约束。

### 5.3 Oracle 可用性

- Temporal blocks：28。
- 无 positive event 的 blocks：5。
- Oracle block-top-one utility sum：+7.372292。
- Oracle worst-sequence top-one utility：+0.286058。

与 canonical bank 的 8 个无正例块相比，full bank 将无正例块降至 5，并显著提高 oracle lower bound。候选空间本身已经足够支撑七域正 utility，剩余瓶颈是极端伪阳性识别。

### 5.4 Retrospective 单分数 Abstention 上界

为判断是否值得继续做 scalar threshold tuning，对每个固定模型的 outer top-one 分数事后枚举所有唯一阈值。该步骤仅是可行性上界，不用于训练或部署。

最佳 motion-multitask 全覆盖阈值：

- Selected blocks：24。
- Positive blocks：12。
- Negative blocks：8。
- Utility sum：+0.995300。
- Worst sequence：−0.643856。
- Eligible：false。

三个模型均不存在一个全局单分数阈值，可以同时实现七序列覆盖、总 utility 为正与最差序列非负。

**P19 决策：关闭 scalar score + scalar abstention threshold family。**

## 6. 完整性与复现

四条正式证据链全部独立重跑并逐文件 SHA-256 一致：

| Chain | Report SHA-256 |
|---|---|
| P18 nested calibration | `d094618a6e70849ae60df02309c5d70402e1226bafdde926941af142efbddcd0` |
| P19 full event teacher | `a95321588811c15b77e7599b4dbd862521c82218c3aa40c95b16f56875052d20` |
| P19 full motion | `6b05378baeb5a775449dccc9cc6f20d8fe386d53e2a0ce4017c6bbec5b1c1f68` |
| P19 domain generalization | `890cab3bb87b720c2c9056766ed2cf59c5886584d7a2cc8542a856ca44ae4ab3` |

统一审计：

- `deliverables/assocriskbench_p19_fullbank_audit_20260716.json`
- SHA-256：`71273898e2c88831efb136c6a76dc63607e76b3e3687e1716464b1b4e8e5a59c`

## 7. 科学决策

### 保留

- 11,705-event full executable teacher bank。
- 11,705-event actual-anchor motion bank。
- Geometry positive classifier 作为 support evidence。
- Motion multitask 作为 candidate ranking baseline。
- Full-bank oracle 与 canonical positive-recall 作为论文分析证据。

### 关闭

- Canonical-bank nested scalar calibration。
- Geometry-only top-one。
- Utility-only top-one。
- Motion-only直接部署。
- Geometry/motion scalar fusion。
- 单一或简单组合 scalar threshold tuning。

### 不变

- Deployment：false。
- P15 policy：`no_op`。
- Locked manifest：未创建。
- 156 条 P15 locked rows：保持未读。

## 8. 下一创新阶段

下一阶段不再继续 scalar threshold 微调，而应转向：

1. **Set-valued selective prediction**：输出一个小候选集合与 abstain，而不是强迫单事件 top-one。
2. **Extreme false-positive tail certificate**：显式学习高分伪阳性的 support/OOD 风险。
3. **Cross-domain support intersection**：要求候选在多个源域机制邻域中均有支持，而不是 pooled probability 高。
4. **Conformal risk set**：校准“候选集合包含至少一个非负事件”的覆盖率，再由独立安全规则选择或 abstain。
5. **Worst-domain objective**：优化 sequence-wise CVaR / lower confidence bound，而非 pooled AUC、Spearman 或 mean utility。

论文方法主线可进一步明确为：

> **Domain-generalized set-valued authorization for counterfactual association transactions under extreme false-positive tail risk.**
