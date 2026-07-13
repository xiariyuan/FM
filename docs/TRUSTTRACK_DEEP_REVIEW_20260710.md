# TrustTrack 深度复核与下一步决策

> 完成时间：2026-07-10
>
> **状态更新：本报告中的 frame 51 / frame 82 最终贡献判断已被全序列反事实推翻。请以 `docs/TRUSTTRACK_DEEP_REVIEW_V2_20260710.md` 为准。**
>
> 复核对象：true soft v1、MOT20-02/03 same-script 实验、current-output ID oracle、Association v1。
>
> 原则：本报告只做离线诊断与实验复核，不把 GT / oracle 信息引入在线方法。

---

## 1. 最终结论摘要

当前证据支持：

```text
1. cue collapse / local ambiguity 是有效的失败风险诊断信号。
2. binary freeze 是可信负结果。
3. true soft v1 在 MOT20-02 为小幅正结果，在 MOT20-03 为小幅负结果；
   02+03 合并后 HOTA -0.018、AssA -0.039、IDSW +3，不能称为跨序列正向模块。
4. soft v1 的核心问题不是只需继续调 alpha，而是风险计算与真实关联阶段错位，
   并且触发后 alpha 从默认 0.9 硬跳到至少 0.98。
5. MOT20-03 的首个负向分叉被严格归因于第 82 帧的一次正确同身份更新被错误软化。
6. MOT20-02 的首个分叉被严格归因于第 51 帧的一次跨身份污染候选被正确软化。
7. current-output oracle 的旧实现存在“先 Hungarian、后阈值过滤”的偏差；
   修正后 HOTA 主结论几乎不变，但旧 IDSW=22/18 不再有效。
8. 已存在的 Association v1 只是 appearance veto，不是完整 adaptive multi-cue weighting；
   它在 MOT20-02 上 AssA、IDF1、IDSW 均变差，属于 mixed/negative ablation。
```

因此当前方法定位应更新为：

```text
TrustTrack 的已验证价值目前仍是 ambiguity diagnosis，
而不是 soft memory action 或现有 Association v1 action。

下一阶段必须先把风险估计迁移到真实、预测后的 association matrix，
再设计 identity-consistency-aware action。
```

---

## 2. 实验公平性与复现性复核

### 2.1 MOT20-03 same-script 对照可靠

baseline 与 soft v1 均使用：

```text
scripts/dmm_base_tracker_trusttrack_soft.py
```

只通过是否传入 `--trust-soft-enable` 区分。

严格复现：

```text
baseline MD5 = ff47a3ec83f31709033ce7b0b16e8013
soft v1 MD5  = 9ec3b40f3d006a3681c1c6de38bdb161
```

baseline 与历史 no-recovery 输出完全一致；soft 与此前 true-soft 输出完全一致。

因此 MOT20-03 的差值不是随机性或 runner 漂移。

### 2.2 02+03 合并结果

TrackEval 对 MOT20-02 + MOT20-03 联合评估：

```text
                         baseline      soft v1      delta
HOTA                       76.544       76.526      -0.018
DetA                       80.209       80.215      +0.006
AssA                       73.104       73.065      -0.039
MOTA                       93.131       93.143      +0.012
IDF1                       87.961       87.990      +0.029
IDSW                          606          609          +3
FN                          25549        25503         -46
FP                           6018         6008         -10
Frag                         1251         1252          +1
```

结论：

```text
在当前已完成的两个序列上，soft v1 整体接近中性但偏负，
不能依据 MOT20-02 单序列 +0.201 HOTA 宣称稳定正收益。
```

---

## 3. soft v1 代码机制复核

### 3.1 EMA 方向正确

真实 `DMMTrack.update_features()` 为：

```text
smooth_feat = alpha * old_smooth_feat + (1 - alpha) * new_feat
```

默认：

```text
alpha = 0.9
```

soft v1 使用：

```text
alpha = 0.98 ~ 0.995
```

因此确实是在降低当前检测特征对长期 identity memory 的影响。

### 3.2 “soft”并非从 baseline 连续变化

当前映射：

```text
不触发：alpha = 0.9
刚触发：alpha ≈ 0.98
极端：  alpha = 0.995
```

所以真实行为是：

```text
hard trigger
  + strong soft freeze
```

而不是：

```text
alpha 从 0.9 平滑连续增加
```

MOT20-03 第 82 帧敏感性实验：

```text
alpha <= 0.905：输出完全等于 baseline
alpha >= 0.910：完整复现第 87~106 帧的 soft 分叉
```

只提高约 1 个百分点就跨越离散决策边界，说明该事件高度敏感。

### 3.3 风险计算与真实 association 错位

soft v1 在调用 `tracker.update()` 之前计算 cue matrix，此时：

```text
- 只读取 activated tracked_stracks；
- 没有包含 lost_stracks；
- 使用的是 Kalman multi_predict 之前的 track.tlbr；
- 没有应用真实 association 的 proximity / appearance hard mask；
- 使用自定义 app / motion / IoU / shape similarity；
- 真实 baseline association 使用预测后的 strack_pool，
  并通过 raw IoU 与 embedding cost 的 BoT-SORT min fusion 做匹配。
```

因此当前 collapse 不是“真实匹配矩阵的局部歧义”，只是一个前置近似代理。

这会造成：

```text
- 不可能匹配的 pair 也进入行列竞争；
- stale geometry 可能把正确 appearance update 判为高风险；
- risk 与真实 Hungarian decision margin 不完全对应。
```

### 3.4 soft 覆盖范围不完整

monkeypatch 只覆盖：

```text
DMMTrack.update()
```

没有覆盖：

```text
DMMTrack.re_activate()
```

而 lost-track reactivation 直接调用 `update_features()`。

同时 `compute_soft_map()` 只包含 activated tracked tracks，所以当前模块实际上主要作用于：

```text
primary high-score active-track update
```

不覆盖：

```text
- lost-track reactivation
- unconfirmed track update
- low-score secondary update
```

这些恰恰可能是 identity memory 风险最高的阶段。

### 3.5 `soft_pairs_predicted` 命名具有误导性

统计：

```text
MOT20-02:
  candidate_pairs       = 8,227,605
  soft_pairs_predicted  = 7,845,643  (95.36%)
  actual soft updates   = 1,716      (1.179% of feature updates)

MOT20-03:
  candidate_pairs       = 41,166,465
  soft_pairs_predicted  = 40,307,631 (97.91%)
  actual soft updates   = 529        (0.172% of feature updates)
```

`soft_pairs_predicted` 是候选矩阵中满足 trigger 的 pair 数，不是实际匹配、也不是风险预测准确数。

建议改名：

```text
potential_trigger_pairs
```

并新增：

```text
actual_matched_trigger_pairs
primary_update_triggers
reactivation_triggers
assignment_changed_events
```

---

## 4. 首个因果分叉：MOT20-03 有害事件

### 4.1 严格反事实结果

完整 soft v1 的首次输出分叉：

```text
frame 87
```

此前实际 soft 更新只有：

```text
frame 15, track 77
frame 17, track 77
frame 82, track 60
frame 92, track 82
```

反事实：

```text
只保留 frame 82 soft update：
  完整复现 frame 87~106 的全部 soft 分叉。

排除 frame 82 soft update：
  前 120 帧完全恢复 baseline。

只保留 frame 15/17 或 frame 92：
  完全等于 baseline。
```

因此首个分叉可严格归因于：

```text
frame 82, track 60, det_global_idx 7150
```

### 4.2 该更新实际上是正确同身份更新

```text
track 60 历史身份 = GT 531
current detection best GT = 531
IoU = 0.5515
下一匹配身份仍为 GT 531
```

cue：

```text
app_pair_margin    = +0.1030
motion_pair_margin = -0.1369
iou_pair_margin    = -0.2798
shape_pair_margin  = -0.3849
collapse           = 0.8970
alpha              = 0.9873
```

关键解释：

```text
appearance 对该身份仍有正区分力；
geometry 因局部拥挤和位置竞争表现为负 margin；
当前 max-based collapse 将整体判为风险，抑制了正确外观记忆更新。
```

短暂缺失后：

```text
baseline 在 frame 87 重新将 GT 531 接回 track 60；
soft 则被 track 72 抢走。
```

这证明：

```text
geometry ambiguity 不能自动推出“应保护 memory”；
当 appearance identity evidence 仍可靠时，当前观测可能恰恰是后续重连需要的正样本。
```

---

## 5. 首个因果分叉：MOT20-02 符合污染保护假设的事件

### 5.1 严格反事实结果

完整 soft v1 的首次输出分叉：

```text
frame 53
```

反事实：

```text
只保留 frame 51 soft update：
  完整复现 frame 53~61 的全部 soft 分叉。

排除 frame 51：
  前 70 帧完全恢复 baseline。
```

因果事件：

```text
frame 51, track 35, det_global_idx 2031
```

### 5.2 该更新是跨身份污染候选

```text
track 35 之前主要对应 GT 230
current detection best GT = 224
IoU = 0.6085
```

cue：

```text
app_pair_margin    = -0.0206
motion_pair_margin = -0.0511
iou_pair_margin    = +0.0416
shape_pair_margin  = -0.1505
collapse           = 0.9584
alpha              = 0.9919
```

这与 MOT20-03 frame 82 的区别是：

```text
MOT20-02 该事件：appearance 本身也不支持当前 pair，符合污染风险。
MOT20-03 有害事件：appearance 仍支持当前正确身份，只是 geometry 竞争严重。
```

因此真正需要预测的是：

```text
identity-memory contamination risk
```

而不是：

```text
generic multi-cue ambiguity
```

---

## 6. 触发质量统计

使用检测自身最佳有效 pedestrian GT 做离线标签：

```text
MOT20-02 soft updates = 1716
  same-identity  = 1474 (85.90%)
  cross-identity = 232  (13.52%)
  unknown        = 10   (0.58%)

MOT20-03 soft updates = 529
  same-identity  = 413  (78.07%)
  cross-identity = 95   (17.96%)
  unknown        = 21   (3.97%)
```

注意：

```text
cross-identity 只是离线污染候选标签，不等价于在线必然错误；
同样，same-identity 也不保证软化一定有害。
```

但可以确认：

```text
当前 soft trigger 的大多数动作发生在同身份正确更新上，
因此它不是高精度 memory-contamination detector。
```

仅在已触发样本内部做离线规则审计发现：

```text
collapse 单独提高到 0.95~0.975 可提高 cross-identity 比例，
但 frame 82 的 harmful event 需要同时利用 appearance consistency 才能排除。
```

不能直接把离线 GT 规则转为在线阈值；需要在真实 matched-pair 日志上重新校准。

---

## 7. current-output ID oracle 修正

### 7.1 旧实现偏差

旧实现：

```text
1. 对完整 IoU matrix 做 Hungarian；
2. 再删除 IoU < 0.5 的 pair。
```

严格做法应为：

```text
1. 先将 IoU < threshold 的 pair 置为无效/零分；
2. 再做 Hungarian；
3. 保留有效 assignment。
```

旧实现少匹配：

```text
MOT20-02: 20 rows
MOT20-03: 13 rows
```

### 7.2 修正后的 0.5 oracle

MOT20-02：

```text
HOTA = 81.683
DetA = 80.292
AssA = 83.138
MOTA = 91.496
IDF1 = 95.618
IDSW = 2
```

相对 baseline：

```text
HOTA +12.935
AssA +23.085
IDF1 +20.413
IDSW -426
```

MOT20-03：

```text
HOTA = 82.178
DetA = 81.183
AssA = 83.208
MOTA = 94.402
IDF1 = 97.154
IDSW = 0
```

相对 baseline：

```text
HOTA +2.124
AssA +3.967
IDF1 +2.985
IDSW -178
```

旧 HOTA 81.681 / 82.176 与新值几乎一致，因此 association headroom 主结论不变。

但是：

```text
旧 oracle IDSW=22 / 18 不应再引用；
修正后为 2 / 0。
```

### 7.3 oracle 不能称为数学严格 ceiling

阈值敏感性：

```text
MOT20-02:
  threshold 0.3: HOTA 81.724
  threshold 0.5: HOTA 81.683
  threshold 0.7: HOTA 80.733

MOT20-03:
  threshold 0.3: HOTA 82.225
  threshold 0.5: HOTA 82.178
  threshold 0.7: HOTA 80.946
```

0.3 与 0.5 的结论稳定，但 HOTA 本身跨多个定位阈值平均，而 oracle 只按一个固定 IoU 阈值分配 ID。

因此推荐名称：

```text
fixed-IoU current-output ID oracle
```

而不是：

```text
strict association ceiling
```

更严格的 upper bound 需要：

```text
- 复用 TrackEval preprocessing；
- 正确处理 distractor/ignored GT；
- 对不同 HOTA alpha 或全局 identity objective 做优化；
- 明确 unmatched output ID 策略。
```

---

## 8. 已遗漏的 Association v1 结果

现有脚本：

```text
scripts/dmm_base_tracker_trusttrack_assoc_v1.py
```

它实现的不是完整 adaptive multi-cue weighting，而是：

```text
当 appearance cost 赢过 IoU、但 appearance margin 弱且 IoU margin 更好时，
对 appearance cost 加一个小 penalty。
```

严格 same-wrapper disabled 对照 MD5 与 baseline 完全一致。

MOT20-02：

```text
                         disabled      enabled      delta
HOTA                       68.748       68.826      +0.078
DetA                       78.844       79.123      +0.279
AssA                       60.053       59.977      -0.076
MOTA                       90.750       90.737      -0.013
IDF1                       75.205       75.142      -0.063
IDSW                          428          447         +19
FN                          11556        11577         +21
FP                           2329         2310         -19
Frag                          765          771          +6
```

解释：

```text
HOTA 的 +0.078 来自 DetA 侧变化，
但 association 指标 AssA / IDF1 / IDSW 全部变差。
```

结论：

```text
当前 Association v1 是 mixed/negative ablation，
不能作为已验证正向 association 主模块。
```

---

## 9. 对此前表述的反思与修正

此前存在以下过度表述或遗漏：

```text
1. 把 soft v1 过早称为“可信正向模块”。
   修正：只能称为 MOT20-02 单序列小幅正向，跨序列尚未成立。

2. 写成“匹配/生命周期均不改”。
   修正：本帧匹配代码不直接改，但 memory 会改变未来匹配、输出数量和生命周期结果。

3. 把 current-output oracle 称为严格 ceiling。
   修正：它是固定 IoU 阈值下的诊断 oracle，不是数学全局上界。

4. 未检查 oracle Hungarian 阈值顺序。
   修正：旧 IDSW 数字作废，HOTA 主结论保留。

5. 未记录已经存在的 Association v1 与负向 association 指标。

6. 只从最终指标猜测 soft 失败原因，没有先做事件级反事实。
   修正：现在 02 frame 51 与 03 frame 82 已有严格单事件因果证据。

7. 过度把 generic ambiguity 等同于 memory contamination。
   修正：memory action 必须显式估计 identity consistency。
```

---

## 10. 更新后的下一步

### Priority 0：冻结当前 action 结论

立即停止：

```text
- soft_start / alpha 的盲目 sweep
- binary freeze sweep
- 现有 appearance-veto Association v1 的参数 sweep
```

当前两种 action 都不应作为默认主线。

### Priority 1：修复诊断基础设施

#### 1. Oracle v2

新建独立脚本，不覆盖历史结果：

```text
scripts/current_output_id_oracle_v2.py
```

要求：

```text
- threshold before Hungarian
- 明确 assignment mode
- TrackEval preprocessing 对齐
- 输出 geometry/score/row strict consistency
- 默认称 fixed-IoU diagnostic oracle
```

#### 2. Association-aligned matched-pair logger

风险必须在真实关联位置采集：

```text
after GMC / Kalman multi_predict
on actual strack_pool + high_dets
using original assoc_cost debug matrices
using actual valid masks and Hungarian matches
```

每个实际 match 记录：

```text
frame
stage: primary / secondary / re_activate / unconfirmed
track state / age / lost gap
raw_iou cost
embedding cost
final baseline cost
row/column assignment margin
best-second gap
chosen feature similarity
historical feature consistency
whether assignment changed
GT diagnostic label（仅离线分析）
```

这是下一阶段最重要的基础设施。

### Priority 2：完成 MOT20-05 与 MOT20-01 ablation

仍需用相同参数完成：

```text
same-script no-trust baseline
true soft v1
corrected fixed-IoU oracle v2
association-aligned matched-pair log
```

顺序：

```text
1. MOT20-05
2. MOT20-01
```

目的不是继续优化 v1，而是补齐四序列行为矩阵。

最终表：

```text
Seq | baseline HOTA | soft delta | AssA delta | IDF1 delta | IDSW delta | oracle delta
01
02
03
05
```

### Priority 3：重新设计 memory action

候选原则：

```text
generic ambiguity 是“需要谨慎”的信号；
identity inconsistency 才是“需要保护 memory”的信号。
```

新 action 至少应满足：

```text
1. 使用预测后、真实 association-aligned cue。
2. geometry ambiguity 单独存在时，不能抑制可靠 appearance update。
3. 只有 appearance / historical identity consistency 同时弱时才提高 alpha。
4. alpha 从 baseline 0.9 连续变化：
   alpha = 0.9 + (alpha_max - 0.9) * p_contamination
5. 分开建模 active update 与 reactivation update。
6. 先做 observe-only / counterfactual audit，再上线 action。
```

### Priority 4：Association v2

不要继续一侧 appearance veto。

应设计对称的动态决策：

```text
appearance 可靠、geometry ambiguous：保留 appearance；
appearance ambiguous、geometry 可靠：降低 appearance；
全部 cue ambiguous：考虑 abstention / delay，而不是强行 commit。
```

第一版建议只改真实 primary association cost：

```text
- 不改生命周期
- 不改 detector
- 不做 delay
- 只记录真实 assignment changed count
```

动态权重必须基于校准后的实际 cost margin，不使用未归一化 similarity margin 混加。

### Priority 5：验证协议

由于 MOT20 只有 4 个 train 序列，必须避免继续在所有序列上调参。

建议：

```text
开发/机制分析：MOT20-02 + MOT20-03
冻结参数后验证：MOT20-05 + MOT20-01
```

必须同时报告：

```text
逐序列结果
02+03 开发集合并结果
01+05 holdout 合并结果
四序列最终结果
trigger/action count
actual assignment changed count
```

---

## 11. 当前一句话状态

```text
cue collapse 作为关联风险诊断仍成立；
但 generic cue collapse 不能直接映射为 memory suppression。

soft v1 的正负差异已被事件级反事实解释：
它在 MOT20-02 frame 51 成功保护了跨身份污染候选，
却在 MOT20-03 frame 82 抑制了正确同身份更新并导致后续错误重连。

因此下一步不是继续调 alpha，而是先建立真实 association-aligned matched-pair diagnostics，
再开发 identity-consistency-aware memory action 与对称 adaptive association v2。
```
