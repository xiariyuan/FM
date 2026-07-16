# AssocRiskBench P17｜MOT17 七域反事实扩展与跨域泛化闭环｜2026-07-16

## 1. 阶段目标

P16 已证明：P15 的 225 个 train-only directional events 中存在可学习信号，但在四条 MOT20 序列之间，absolute utility、pairwise ranking、same-sequence retrieval、multitask uncertainty、actual-anchor motion veto 与受限 fallback 均未达到最差序列非负的部署约束。

P17 不再读取或调节冻结的 P15 locked pool，而是扩展独立反事实训练域，回答三个问题：

1. P15 学到的 actual-anchor motion utility 是否能零样本迁移到 MOT17？
2. 扩展为多个独立序列后，跨序列 utility signal 是否可学习？
3. 将 MOT17 teacher 直接并入 P15 训练是否能够改善 P15 的最差序列风险？

本阶段始终保持：

- 新 P15 locked labels 读取：0。
- 新 P15 locked TrackEval 调用：0。
- 新全局 TrackEval 调用：0。
- P15 剩余 156 条 locked directional rows 全部未读。
- 未创建 locked manifest。
- P15 最终策略仍为 `no_op`。

## 2. 方法一：无 GT 候选构造、冻结后 teacher 标注

### 2.1 候选枚举

对每条 MOT17 训练序列：

1. 从 baseline tracker rows 中枚举存在时间重叠的 unordered raw-track pair。
2. 取 pair 的最后共同帧作为 boundary。
3. 使用冻结的 directional planner 分别审计 `u_to_v` 与 `v_to_u`。
4. 仅保留满足以下 executor 约束的方向：
   - boundary 两轨迹均存在；
   - donor/receiver boundary label 唯一；
   - receiver 存在 donor 缺失后的 continuation；
   - donor 不在 continuation 中重现；
   - relabel 后不产生 duplicate identity；
   - changed rows 非零。

候选枚举及 executor audit 不读取 GT、local utility 或 TrackEval。

### 2.2 七域 canonicalization

全部 executable direction 先使用纯 tracker geometry priority 排序：

```text
boundary_center_distance_norm
+ 0.25 × boundary_bottom_gap_norm
+ 0.10 × (|log width ratio| + |log height ratio|)
− 0.25 × boundary_iou
```

较低 priority 优先。之后执行：

- 每个 raw track 的 boundary 至少间隔 30 帧；
- 每条序列最多 500 个事件；
- 不依据 GT 或 utility 筛选正例。

选定事件集合冻结后，才使用 GT 做 per-frame Hungarian matching，并生成 dense local counterfactual teacher。

### 2.3 Teacher 定义

沿用 P16 已验证的 actual-anchor 语义：

- actual source：changed rows 中唯一 `edited_label`；
- receiver：每条 changed row 的 `baseline_label`；
- primary teacher：`full_idtp_delta_norm`；
- 辅助 teacher：P15 预注册并通过资格审计的 13 个 local association / IDTP targets。

全程不需要执行 global TrackEval。

## 3. MOT17-09 单域 pilot

正式资产：

- `outputs/assocriskbench_p17_20260716/mot17_09_directional_local_counterfactual_bank_v1`
- `outputs/assocriskbench_p17_20260716/mot17_09_actual_anchor_motion_v1`
- `outputs/assocriskbench_p17_20260716/mot17_09_cross_domain_transfer_v1`

### 3.1 数据规模

- audited ordered directions：800。
- executable events：363。
- changed-row labels：39,037。
- positive teacher events：39。
- negative teacher events：296。
- zero teacher events：28。
- positive fraction：10.7438%。
- GT matching rate：97.8847%。

相较 P15 的 225 个事件，单条 MOT17 序列已将可用事件数量增加 161%，并提供大量 hard negatives。

### 3.2 P15 → MOT17-09 零样本迁移

使用 P16 预注册的 36 个 compact actual-anchor motion features 和固定 ExtraTrees 模型：

- Spearman：0.074421。
- positive AUC：0.574391。
- average precision：0.121233。
- predicted top-20 positive events：0 / 20。
- predicted top-20 utility sum：−4.514720。

P15 模型在新域的最高分候选全部为负，说明失败不是小幅 calibration shift，而是候选机制支持域发生明显变化。

### 3.3 MOT17-09 域内时间块 OOF

同一固定模型在 MOT17-09 内做四个 temporal block OOF：

- Spearman：0.409314。
- positive AUC：0.715337。
- average precision：0.235384。

这说明新域 teacher 并非纯噪声；信号在域内可学习，但无法由 P15 零样本迁移。

### 3.4 单域直接混池

将 MOT17-09 全部事件直接加入 P15 LOSO 训练后：

- P15-only local worst sequence：−0.333561。
- augmented local worst sequence：−0.465160。
- P15-only HOTA worst sequence：−0.104836。
- augmented HOTA worst sequence：−0.160374。
- catastrophic HOTA windows：2 → 3。

因此单域 teacher bank 被保留，但朴素混池被拒绝。

## 4. MOT17 七序列独立反事实 bank

正式资产：

- `outputs/assocriskbench_p17_20260716/mot17_train7_directional_local_counterfactual_bank_v1`
- `outputs/assocriskbench_p17_20260716/mot17_train7_actual_anchor_motion_v1`

### 4.1 总规模

- sequences：7。
- ordered candidate directions：27,336。
- executable events before canonicalization：11,705。
- selected teacher events：705。
- changed-row labels：85,291。
- positive events：104。
- negative events：571。
- zero events：30。
- positive fraction：14.7518%。

结合 P15 后，正式可用 train-only directional teacher events 从 225 增至 930。

### 4.2 每序列分布

| Sequence | Selected | Positive | Negative | Zero | Positive fraction | Changed rows |
|---|---:|---:|---:|---:|---:|---:|
| MOT17-02-FRCNN | 164 | 44 | 116 | 4 | 26.8293% | 20,446 |
| MOT17-04-FRCNN | 39 | 1 | 36 | 2 | 2.5641% | 12,819 |
| MOT17-05-FRCNN | 158 | 20 | 136 | 2 | 12.6582% | 13,914 |
| MOT17-09-FRCNN | 46 | 9 | 31 | 6 | 19.5652% | 5,783 |
| MOT17-10-FRCNN | 97 | 8 | 87 | 2 | 8.2474% | 12,565 |
| MOT17-11-FRCNN | 73 | 2 | 68 | 3 | 2.7397% | 10,709 |
| MOT17-13-FRCNN | 128 | 20 | 97 | 11 | 15.6250% | 9,055 |

正例率从 2.56% 到 26.83%，证明跨序列 prior 与 lower-tail risk 差异显著。该差异正是后续 domain-conditioned calibration 必须建模的对象。

## 5. 七域 sequence-disjoint generalization

正式资产：

- `outputs/assocriskbench_p17_20260716/mot17_train7_domain_generalization_v1`

固定比较两个理论上预先定义的模型族：

1. `raw`：直接回归 `full_idtp_delta_norm`。
2. `multitask`：回归 13 个预注册 local targets；每个训练序列内分别做 median/IQR robust normalization，最终分数为 13 个预测目标的均值。

共同设置：

- 36 个 compact actual-anchor motion features；
- ExtraTrees，500 trees；
- max depth 7；
- min leaf 3；
- max features 0.65；
- 每条训练序列总 sample weight 相同；
- 无模型参数 sweep；
- 无 gate 或 threshold sweep。

### 5.1 Event-level discrimination

| Model | Spearman | Positive AUC | Average precision |
|---|---:|---:|---:|
| Raw | 0.179771 | 0.524990 | 0.160287 |
| 13-target multitask | **0.222977** | **0.702051** | **0.260891** |

多目标 teacher 显著提升了跨序列正负区分能力，支持“mechanism-aware dense supervision”作为论文方法方向。

### 5.2 四时间块 top-one 风险

每个 held-out sequence 被分成四个 deterministic temporal blocks，每块选择模型 top-one，共 28 个选择。

| Model | Positive | Negative | Zero | Utility sum | Worst sequence |
|---|---:|---:|---:|---:|---:|
| Raw | 5 | 22 | 1 | −2.660749 | −1.076624 |
| 13-target multitask | **7** | **18** | 3 | **−2.286110** | **−0.642163** |

尽管 multitask 在 event-level AUC 上达到 0.7021，lower-tail top-one 仍严重失败；7 条序列中仅 MOT17-13 的序列效用为正。

因此：

- event discrimination 改善不等价于安全 candidate authorization；
- 不能仅以 pooled AUC 或 Spearman 作为部署证据；
- 论文主方法必须显式建模 sequence/domain prior 与 lower-tail calibration。

## 6. 七域 teacher 对 P15 的增量作用

P15 仍只使用 train ranks 21–100，四条序列 LOSO；HOTA 仅用于审计，不参与训练或选择。

### 6.1 Event-level 指标

| Model | Training | Spearman | Positive AUC | AP |
|---|---|---:|---:|---:|
| Raw | P15 only | 0.310576 | 0.559675 | 0.264600 |
| Raw | P15 + P17 | 0.257004 | 0.550965 | 0.253278 |
| Multitask | P15 only | 0.303681 | 0.558145 | 0.250654 |
| Multitask | P15 + P17 | **0.317510** | **0.598399** | **0.263373** |

七域 multitask augmentation 对 pooled event discrimination 有正增益，但仍需检查最差序列。

### 6.2 P15 每窗口 top-one 审计

| Model | Training | Local sum | Local worst seq | HOTA sum | HOTA worst seq | Catastrophic |
|---|---|---:|---:|---:|---:|---:|
| Raw | P15 only | −0.324457 | −0.333561 | −0.144661 | −0.104836 | 3 |
| Raw | P15 + P17 | −0.574186 | −0.372345 | +0.025285 | −0.140528 | 1 |
| Multitask | P15 only | −0.336499 | −0.233681 | −0.114580 | −0.066016 | 2 |
| Multitask | P15 + P17 | −0.295805 | −0.334845 | −0.107813 | −0.056277 | 2 |

结论：

- multitask augmentation 提升 pooled AUC；
- 但 local worst sequence 从 −0.233681 恶化到 −0.334845；
- HOTA worst sequence 仍为 −0.056277；
- 仍有 2 个 catastrophic windows；
- 四种固定组合的 deployment eligibility 均为 0。

因此不能将七域 bank 直接推广为 P15 policy。

## 7. 完整性与泄漏审计

- 七域 motion rows：705。
- motion feature columns：367。
- duplicate event keys：0。
- forbidden feature columns：0。
- candidate selection 使用 GT：false。
- GT teacher 在候选冻结后应用：true。
- global TrackEval calls：0。
- P15 locked labels read：0。
- P15 locked TrackEval calls：0。
- P15 remaining locked rows：156。

Motion builder 已从硬编码 MOT20 四序列改为从 executability 动态推断序列集合。修改后重新执行 P16 全链回归测试，四个 P16 motion 文件全部字节一致：

- P16 motion feature SHA-256：`932f4ea78c73a23745f8799c8e6bf10f2ce1b2b478524894cfbd0f8eade1b9f3`。

## 8. 复现性

六条 P17 正式证据链均独立重跑，并逐文件 SHA-256 一致：

1. MOT17-09 local teacher pilot。
2. MOT17-09 actual-anchor motion。
3. MOT17-09 cross-domain transfer。
4. MOT17 train7 local teacher bank。
5. MOT17 train7 actual-anchor motion。
6. MOT17 train7 domain generalization。

关键 report hashes：

- MOT17-09 teacher：`1c6eee941a7e3da159fc296be7a2ddeb86e6f0c29b19e9f41cb504564a3f7bb6`
- MOT17-09 motion：`6716800e45fd1e3e7bc2d6f14038323bd4b0e4c3ddedb1bda1f6da1d793f5ab4`
- MOT17-09 transfer：`bdcd77198583de56eb8b023d57c71ece3a9ba5c4a05bed3de882019fbef47051`
- MOT17 train7 teacher：`6c494f4890717ae09b54d610bc73743e170dbbb694fcee83ae7dd41511c4296d`
- MOT17 train7 motion：`a014a58cdb676411b2b63a2d415a82c4587b3134d2e82a97a87a22108ed3937f`
- MOT17 train7 generalization：`c0d4e5fda0571a588b6ca1a391f102bc8a408981d09a06ff9a6472407c809e65`

最终统一审计：

- `deliverables/assocriskbench_p17_mot17_train7_domain_audit_20260716.json`
- SHA-256：`e2d038bb5129a25458aae92ff7a3d92cce548e70a234855a64e6c228ebcc8e18`

## 9. 科学决策

### 保留

- 保留七序列 705-event / 85,291-row teacher bank。
- 保留 13-target multitask 表示作为后续方法基线。
- 保留无 GT candidate enumeration + post-freeze teacher 标注协议。

### 拒绝

- 拒绝 P15 → MOT17 零样本直接迁移。
- 拒绝单 MOT17 序列朴素混池。
- 拒绝七序列 raw pooling。
- 拒绝七序列 multitask pooling 直接作为 deployment policy。
- 不创建 P15 locked manifest。
- 不改变 P15 `no_op` 策略。

## 10. 下一方法阶段

P17 证明“增加独立 teacher 域”是必要但不充分的。下一阶段应从 pooled utility regression 转向：

1. **Domain-conditioned calibration**：显式估计每条序列的 positive prior、utility scale 与 residual tail。
2. **Invariant mechanism representation**：将 motion transfer、appearance competition、history support 与 receiver-family composition 分解为机制子空间，而非共享一个绝对 utility 轴。
3. **Leave-one-domain-out authorization**：外层持出整条序列；内层只使用其余域选择 calibration 与 abstention，禁止 held-out utility 参与门槛确定。
4. **Lower-tail objective**：优化 sequence-wise CVaR / worst-domain risk，而不是 pooled Spearman 或 AUC。
5. **Policy acceptance**：同时要求 MOT17 七域与 P15 四域覆盖，且所有域的 worst-sequence utility 非负；否则保持 no-op。

这一路径具有更清晰的论文创新点：从 association utility prediction 转向 **domain-generalized, lower-tail-safe counterfactual transaction authorization**。
