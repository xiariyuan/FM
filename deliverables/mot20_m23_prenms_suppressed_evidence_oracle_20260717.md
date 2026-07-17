# M23-1 MOT20 Pre-NMS Suppressed-Evidence Oracle Audit

日期：2026-07-17

## 1. 研究问题

M23-0 已证明 post-NMS 低置信观测能够把 MOT20 train 的 row-preserving oracle 从 82.571 提升到 83.676，但其乐观 selected ceiling 只有 84.265，未达到预注册的 84.5 安全余量。

M23-1 回答：**YOLOX class-aware NMS 删除的候选中，是否存在足够且跨四序列稳定的有效观测，使 HOTA 82.5 不再贴近检测空间天花板？**

本阶段是离线 ceiling 审计，不训练 selector，不授权部署。

## 2. 固定协议

### 2.1 真正的 pre-NMS 提取

- 检测器：冻结的 MOT20 YOLOX-X checkpoint。
- 精度：FP32。
- 模型：unfused。
- batch size：1。
- confidence threshold：0.09。
- NMS：原始 YOLOX class-aware batched NMS。
- ReID：不提取。
- GT：提取阶段不读取。

batch size 2 曾进行 smoke 验证，但因为 14/20 帧出现数值漂移而被拒绝；正式提取保持与 Phase-0 完全一致的 batch size 1。

### 2.2 冻结 Phase-0 等价检查

| 序列 | Pre-NMS | Post-NMS kept | NMS suppressed |
|---|---:|---:|---:|
| MOT20-01 | 180,743 | 21,423 | 159,320 |
| MOT20-02 | 1,377,872 | 160,051 | 1,217,821 |
| MOT20-03 | 2,852,310 | 332,280 | 2,520,030 |
| MOT20-05 | 5,923,128 | 674,247 | 5,248,881 |
| **合计** | **10,334,053** | **1,188,001** | **9,146,052** |

共 8,931 帧全部通过：

- count mismatch：0；
- class mismatch：0；
- set mismatch：0；
- suppression orphan：0；
- 最大 box absolute difference：0.0001221 像素；
- 最大 score difference：0；
- 最低 matched IoU：大于 0.99997。

### 2.3 GT-free candidate plan

Candidate plan 在任何 GT 读取和 TrackEval 调用之前独立冻结：

1. 删除 suppressor IoU >= 0.99 的近精确重复；
2. 同一 suppression family 内以 IoU >= 0.95 去重；
3. 固定 priority = score × normalized suppressor novelty；
4. 先按 family rank 保持多样性，再按 frame-local rank 与稳定索引排序；
5. 总 observable pool 不超过 baseline 的 1.5 倍。

| 序列 | Full survivors | Budget selected | Budget pool ratio |
|---|---:|---:|---:|
| MOT20-01 | 28,312 | 6,712 | 1.500000 |
| MOT20-02 | 210,300 | 57,734 | 1.499997 |
| MOT20-03 | 423,395 | 122,108 | 1.499998 |
| MOT20-05 | 878,229 | 273,345 | 1.499999 |
| **合计** | **1,540,236** | **459,899** | **1.499998637** |

Full survivor pool ratio 为 2.481496，仅作为非预算上限；正式接受判断使用固定 1.5× budget。

## 3. MOT20 train 组合结果

| Variant | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | Delta vs baseline |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 77.699 | 80.907 | 74.672 | 89.308 | 93.606 | 1,222 | 0.000 |
| Baseline ID oracle | 82.571 | 81.324 | 83.872 | 96.882 | 93.857 | 0 | +4.872 |
| Post-NMS replace/add oracle | 83.676 | 82.480 | 84.924 | 97.881 | 95.784 | 6 | +5.977 |
| Pre-NMS budget additive oracle | 84.135 | 82.941 | 85.382 | 98.363 | 96.724 | 25 | +6.436 |
| **Pre-NMS budget replace/add oracle** | **84.523** | **83.357** | **85.738** | **98.396** | **96.792** | **8** | **+6.824** |
| **Pre-NMS budget selected ceiling** | **85.132** | **84.502** | **85.799** | **99.172** | **98.357** | **8** | **+7.433** |
| Full pre-NMS selected ceiling | 85.915 | 85.326 | 86.539 | 99.483 | 98.970 | 7 | +8.216 |

最重要的结果不是只在 GT 删除 FP 后超过 84.5，而是：

> 保留原有 unmatched/FP 负担的 budget replace/add oracle 也达到 **84.523 HOTA**。

相对 M23-0 post-NMS replace/add：

- HOTA：83.676 -> **84.523**，+0.847；
- FN：30,071 -> **18,630**，减少 11,441；
- FP：17,753 -> **17,755**，仅增加 2；
- IDSW：6 -> 8。

这证明 NMS-suppressed evidence 的主要价值来自恢复漏检和替换定位不足的观测，而不只是依靠删除 FP 制造乐观上限。

## 4. 分序列结果

| 序列 | Post-NMS replace/add | Pre-NMS budget replace/add | Budget selected ceiling | Full selected ceiling |
|---|---:|---:|---:|---:|
| MOT20-01 | 83.936 | **84.898** | **85.313** | 86.674 |
| MOT20-02 | 82.984 | **84.667** | **85.166** | 86.360 |
| MOT20-03 | 83.581 | **84.198** | **84.641** | 85.290 |
| MOT20-05 | 83.850 | **84.610** | **85.331** | 86.065 |


固定 budget selected ceiling 的最差序列为 MOT20-03，HOTA = **84.641**，高于预注册 worst-sequence 门槛 84.0。四个序列均超过 84.5 左右的 oracle 水平，没有 MOT20-05 负迁移。

## 5. Suppression evidence 的结构

固定 budget 相对 post-NMS 新增恢复 17,715 个 GT 观测：

### Score 分布

- score >= 0.60：10,143（57.26%）；
- 0.30 <= score < 0.60：4,275（24.13%）；
- 0.10 <= score < 0.30：3,081（17.39%）；
- 0.09 <= score < 0.10：216（1.22%）。

与 M23-0 不同，pre-NMS 的新增价值多数来自高置信但被 NMS 删除的候选，而不是单纯降低 detection threshold。

### 与 suppressor 的 IoU

- 0.70–0.75：1,661；
- 0.75–0.80：1,383；
- 0.80–0.90：3,466；
- 0.90–0.95：4,705；
- 0.95–0.99：6,500。

其中 **63.25%** 的新增恢复候选与 suppressor IoU >= 0.90。这说明密集遮挡场景中的 NMS 冲突常常不是“重复框 vs 重复框”，而是两个高度重叠、分别对应不同人物或具有不同定位质量的有效观测。

## 6. 对 HOTA 82.5 目标的影响

达到 82.5 所需恢复的 oracle 增益比例：

- 旧 baseline ID oracle：98.54%；
- M23-0 post-NMS replace/add：80.32%；
- M23-1 pre-NMS budget replace/add：**70.35%**；
- M23-1 budget selected ceiling：**64.59%**。

因此 82.5 已从“近乎完整复现 GT ID oracle”变成“需要恢复约 70% 固定预算扩展证据增益”的目标。仍然困难，但已经具备合理研究余量。

## 7. 预注册判断

- Combined selected ceiling >= 84.5：**通过**（85.132）；
- Worst-sequence selected ceiling >= 84.0：**通过**（84.641）；
- Candidate pool <= 1.5× baseline：**通过**（1.499998637）；
- Baseline ID oracle reproduction：通过；
- M23-0 post-NMS oracle reproduction：通过；
- `expanded_evidence_ceiling_sufficient = true`；
- `deployment_allowed = false`；
- `locked_manifest_created = false`；
- P15 policy：`no_op`；
- 新增 locked-label reads：0；
- 新增 locked TrackEval calls：0；
- 156 条剩余 locked rows 未触碰。

## 8. 研究结论

M23-1 支持以下论文主张：

> In crowded MOT, class-aware NMS removes a substantial amount of identity-relevant evidence. A GT-free, fixed-budget recovery pool raises the row-preserving MOT20 oracle from 83.676 to 84.523 HOTA and the selected ceiling to 85.132, while satisfying all sequence-level ceiling constraints.

更重要的是，真正的瓶颈已经从“检测证据不存在”转变为：

> 如何在 459,899 个固定 budget suppressed candidates 中，使用可部署证据识别 17,715 个有价值的新增观测，并把它们与全局身份图安全结合。

## 9. 下一阶段：M23-2

下一阶段应固定本阶段 candidate manifest，不允许再扩大候选池或调 NMS/score 阈值：

1. 只为 459,899 个 budget suppressed candidates、其 suppressor 和相关 track history 提取 appearance；
2. 构建 suppression-pair 特征：candidate/suppressor appearance margin、track-prototype margin、局部 hard-negative margin、box quality、temporal consistency；
3. 在严格 MOT20 sequence-LOSO 下训练 observation selector，不在 outer sequence 调阈值；
4. 将被授权观测作为 tracklet graph 的 observation alternatives，而不是直接改 ID；
5. 以 cross-fitted HOTA、四序列 worst-case、MOT20-05 no-regression 为验收条件；
6. 不触碰 P15 locked evaluation。

建议的第一轮实际目标：

- cross-fitted MOT20 train HOTA >= 79.5；
- 四序列均不低于 baseline；
- 对预算 oracle 的恢复率达到至少 25%，再决定是否进入完整 global graph。

## 10. 可复现性

独立 reproduction 已验证：

- candidate plan：43/43 文件字节一致；
- oracle compact files：7/7 字节一致；
- generated tracker files：28/28 哈希一致；
- TrackEval summaries：7/7 哈希一致。

关键 SHA256：

- pre-NMS dump script：`e5d720c31462f084f0a05c6b56b33db9bd5fc895464e4558b1dd5dbf7fadf0d0`；
- candidate-plan script：`1234817aa6c14e335da9b52030f5424d1ba9d2ad3b42e0db8afd3eb1fc3055ce`；
- oracle audit script：`ce43ea5fa60670479cb45f76b1d530c250c562d3219cf0984a51694c92729ee7`；
- suppression dump manifest：`9b7e4db25c6a96dadc82dcd0aa7c37a385ee31320a63ed63fa79699e5c9e7a9f`；
- candidate-plan manifest：`c9dab3959da330ef5ca5bebc85a0d87807666cc6db82e3d70f07e724e2b9a7ad`；
- oracle report：`5d5acf57f6cbf6434916b9964366ab334ccd7a382e9d7133cf017e9fe3e1ba28`；
- oracle manifest：`c72676559c5dde842a6dec77082950ece703d7edb1af8e58cc822b7dc8930c50`。
