# TrustTrack 第二轮反向审稿复核

> 日期：2026-07-10
>
> 本报告专门审查上一版 `TRUSTTRACK_DEEP_REVIEW_20260710.md` 的证据边界。
>
> **优先级：本报告高于上一版报告及交接文档第 16 节。**

> **状态更新：本报告中的检测 GT 标签解释、跨运行 raw tracker ID 比较及最新 action 结论已被第三轮审计修正。请以 `docs/TRUSTTRACK_DEEP_REVIEW_V3_20260711.md` 为准。**
>
---

## 1. 最重要的修正

上一版有两处事件级方向判断过度，现正式撤销：

```text
1. MOT20-03 frame 82 不是造成最终负收益的“有害事件”。
2. MOT20-02 frame 51 不能解释最终 +0.201 HOTA，也不能称为最终收益事件。
```

更准确的事实是：

```text
- 两个事件都能严格解释“首次局部输出分叉”；
- 但首次分叉不等于最终指标贡献；
- 必须运行全序列 only-event / exclude-event 反事实后才能判断边际贡献。
```

### MOT20-02 frame 51 全序列反事实

```text
only frame 51:
  HOTA / AssA / IDF1 / IDSW 在正式输出精度上与 baseline 相同。

exclude frame 51 from full soft:
  最终指标与 full soft 相同。
```

更高精度 HOTA AUC：

```text
baseline     = 0.6874833
only51      = 0.6874831
full soft   = 0.68948674
exclude51   = 0.6894869
```

因此：

```text
frame 51 造成 frame 53~61 的局部分叉，
但对最终序列指标的边际贡献近似为零。
```

### MOT20-03 frame 82 全序列反事实

```text
baseline:
  HOTA 80.054, IDF1 94.169, IDSW 178

only frame 82:
  HOTA 80.056, IDF1 94.174, IDSW 176

full soft:
  HOTA 79.943, IDF1 94.015, IDSW 186

exclude frame 82:
  HOTA 79.942, IDF1 94.010, IDSW 188
```

高精度 HOTA AUC：

```text
baseline   = 0.8005449
only82    = 0.80056316
full soft = 0.7994334
exclude82 = 0.7994151
```

因此：

```text
frame 82 是微弱正向事件，而不是负向事件；
MOT20-03 总体 -0.111 HOTA 来自其他 soft 事件或事件交互。
```

GT-centric transition 分析显示：

```text
baseline 在 frame 87 / 90 将 GT 531 / GT 553 从 track 72 切到 track 60，
产生两个额外 GT→tracker ID change；
only82 保持 track 72，消除了这两个 transition。
```

上一版只看 `track 60` 的历史身份，忽略了 GT 在上一有效帧已经对应 `track 72`，属于 track-centric 误判。

后续事件诊断必须以：

```text
GT identity -> tracker ID transition
```

为主，而不能只用：

```text
tracker ID -> majority GT
```

---

## 2. cue-collapse 命题重新审查

### 2.1 原始诊断器的候选集问题

`scripts/trusttrack_rrs_diagnostic.py` 会把所有历史出现过的 tracker ID 永久保留在 `states` 中，作为后续 col-margin 竞争者。

原始候选数量：

```text
MOT20-02:
  全历史候选平均约 244
  按真实 max_time_lost=58 近似约 69

MOT20-03:
  全历史候选平均约 356
  按真实 max_time_lost=58 近似约 151
```

注意：

```text
track_buffer=70 在 frame_rate=25 时，
真实 max_time_lost = int(25/30*70) = 58，
不是 70。
```

输出框回配检测的贪心方法影响很小：

```text
MOT20-02 greedy 与 Hungarian 匹配数量相同，仅 2 帧 pair 选择不同；
MOT20-03 完全相同。
```

真正需要修正的是陈旧列竞争者。

### 2.2 58 帧候选修正后的信号

重算 chosen detection 的 col margin，候选仅保留过去 58 帧内轨迹；chosen score 重构误差约 1e-7~1e-6，说明复现正确。

MOT20-02：

```text
                         original      gap58
AUC                        0.7817      0.7872
AP                         0.2407      0.2475
top-10% precision          0.2391      0.2519
lift                        3.45x       3.63x
```

MOT20-03：

```text
                         original      gap58
AUC                        0.8403      0.7809
AP                         0.1952      0.1770
top-10% precision          0.1464      0.1229
lift                        5.73x       4.81x
```

结论：

```text
- 陈旧列竞争明显放大了 MOT20-03 的诊断强度；
- 但修正后 02/03 的 collapse 仍保持稳定风险相关性；
- 因此信号没有被推翻，但“已被证明强预测”应降级为：
  在当前离线近似诊断中具有跨 02/03 的稳定相关性。
```

### 2.3 row-only 与 col-dependent

```text
row-only collapse AUC:
  MOT20-02 = 0.724
  MOT20-03 = 0.690

58-frame pair collapse AUC:
  MOT20-02 = 0.787
  MOT20-03 = 0.781
```

说明：

```text
row ambiguity 自身有信息；
col competition 提供主要增益；
但 col margin 必须使用真实候选池与 actual assignment。
```

### 2.4 标签拆分

58 帧修正后：

```text
wrong_id AUC:
  MOT20-02 = 0.802
  MOT20-03 = 0.861

当前正确、未来 10 帧 tracker identity 改变 AUC:
  MOT20-02 = 0.825
  MOT20-03 = 0.864

det_fp AUC:
  MOT20-02 = 0.614
  MOT20-03 = 0.659
```

因此 collapse 更接近：

```text
identity / association risk
```

而不是单纯检测 FP 风险。

### 2.5 confounder 与增量价值

collapse 不是唯一、也不总是最强的单变量：

```text
MOT20-03 future-10：
  low det score AUC = 0.884
  collapse AUC      = 0.864

MOT20-03 wrong_id：
  negative app-col margin AUC = 0.900
  collapse AUC                = 0.861
```

但跨序列 logistic audit 显示 collapse 提供增量信息：

```text
train 02 -> test 03:
  simple cues               AUC 0.871
  simple cues + collapse    AUC 0.898

train 03 -> test 02:
  simple cues               AUC 0.813
  simple cues + collapse    AUC 0.859
```

因此最准确的结论是：

```text
collapse 是有增量价值的风险特征，
但不是已完成的独立理论贡献，也不是可直接用于 action 的校准概率。
```

固定阈值覆盖率也明显跨序列漂移。以 future-10 / currently-correct 为例：

```text
collapse >= 0.80:
  MOT20-02 coverage 2.7%, precision 15.9%, recall 16.3%
  MOT20-03 coverage 0.6%, precision 13.3%, recall 12.0%
```

所以固定 `soft_start=0.80` 没有跨序列校准意义。

---

## 3. MOT20-02 实验公平性补齐

此前只有 100 帧 no-trust smoke，没有 full same-script no-trust 证据。

现已补跑：

```text
outputs/trusttrack_review_20260710/m02_same_script_full_notrust/
```

MD5：

```text
same-script no-trust = 8cd345b4de6efd9cb9d6f6b1427bd2ec
historical baseline  = 8cd345b4de6efd9cb9d6f6b1427bd2ec
```

指标完全一致：

```text
HOTA 68.748
DetA 78.844
AssA 60.053
IDF1 75.205
IDSW 428
```

因此 M02 对照公平性正式闭环。

但必须增加选择偏差说明：

```text
MOT20-02 上已经比较 binary freeze、soft v1、soft_start=0.85 等多个方案，
最终引用 +0.201 的 v1，本质上是 development / selected result，
不能作为独立泛化证据。
```

MOT20-03 才更接近固定方案后的第一次跨序列验证。

---

## 4. soft v1 的进一步代码边界

### 4.1 只保护 smooth prototype，不保护完整 feature history

真实 `update_features()` 在 EMA 后仍会执行：

```text
features.append(current_feature)
```

soft v1 只临时提高 `self.alpha`，因此：

```text
- smooth_feat 更新被减弱；
- features deque 仍无条件写入当前观测。
```

所以当前模块只能称：

```text
smooth identity prototype attenuation
```

不能宽泛称为：

```text
完整长期 identity memory protection
```

当前 no-DMM baseline 的主要 ReID 匹配使用 smooth feature，因此实验输出仍真实有效；但未来若启用 history-based 模块，这个区别会变得重要。

### 4.2 stage 覆盖不完整

当前只 monkeypatch：

```text
DMMTrack.update()
```

没有覆盖：

```text
DMMTrack.re_activate()
```

且风险图只包含 activated tracked tracks 与高分检测。

未覆盖：

```text
- lost-track reactivation
- low-score secondary association
- unconfirmed association
```

此外 `tracklet_len` 在 reactivation 后重置，当前 min-age 语义也不等于完整身份年龄。

### 4.3 诊断 cue 与实际关联 cue 不一致

当前四 cue：

```text
appearance / simple center-motion / IoU / shape
```

但真实 primary baseline 主要是：

```text
Kalman/GMC 后的 IoU cost
+ ReID embedding cost
+ hard proximity/appearance masks
+ min-cost fusion
```

simple motion 与 shape 目前只是辅助诊断特征，不是 baseline 中独立参与匹配的四个 cue。

因此在真正把它们并入成本前，不应把当前系统表述为已经实现：

```text
adaptive four-cue association
```

---

## 5. oracle v2 复核

### 5.1 两个必要修正

正式 oracle v2 应包含：

```text
1. IoU threshold before Hungarian；
2. TrackEval distractor preprocessing before pedestrian assignment。
```

TrackEval MOT20 distractor 类：

```text
person_on_vehicle, static_person, distractor, reflection, non_mot_vehicle
```

预处理影响：

```text
MOT20-02 removed rows = 52
MOT20-03 removed rows = 1
```

最终修正版：

```text
MOT20-02 oracle:
  HOTA 81.684
  AssA 83.139
  IDF1 95.618
  IDSW 0

MOT20-03 oracle:
  HOTA 82.178
  AssA 83.208
  IDF1 97.154
  IDSW 0
```

### 5.2 01/05 既有输出的修正版 headroom

MOT20-01：

```text
baseline HOTA 77.030 -> oracle 82.178  (+5.148)
AssA          73.930 ->        83.550  (+9.620)
IDF1          88.333 ->        96.218  (+7.885)
IDSW              50 ->             0
```

MOT20-05：

```text
baseline HOTA 78.539 -> oracle 82.954  (+4.415)
AssA          75.802 ->        84.292  (+8.490)
IDF1          90.304 ->        97.067  (+6.763)
IDSW             566 ->             0
```

四序列：

```text
Seq       baseline HOTA   oracle HOTA   delta
MOT20-01      77.030         82.178     +5.148
MOT20-02      68.748         81.684    +12.936
MOT20-03      80.054         82.178     +2.124
MOT20-05      78.539         82.954     +4.415
```

四序列 TrackEval combined：

```text
baseline HOTA 77.699
oracle   HOTA 82.571
Delta         +4.872

baseline AssA 74.672
oracle   AssA 83.872
Delta         +9.200
```

MOT20-02 是明显的 association-headroom outlier，不能围绕它单独设计规则。

### 5.3 oracle 不是纯 AssA 增益

虽然输出框完全不变，只修改 ID，但 HOTA / CLEAR 的帧内匹配会使用身份一致性或全局 alignment 信息，因此：

```text
DetA、FN、FP 也可能随 ID 重赋值小幅改变。
```

四序列 oracle：

```text
DetA +0.417
FN   -813
FP   -813
```

因此应称：

```text
current-output ID reassignment evaluation upper bound
```

而不能把全部 HOTA delta 机械解释为纯 association-only HOTA 增益。

它仍然不是数学严格 ceiling，因为：

```text
- 使用固定 IoU threshold；
- HOTA 跨多个 alpha；
- unmatched ID 策略固定；
- 未直接优化全局 HOTA objective。
```

---

## 6. 内置 debug_assoc 的可复用性与缺陷

`DMMBaseTracker._record_assoc_debug()` 已位于真实流程：

```text
GMC / Kalman multi_predict
-> real assoc_cost
-> Hungarian
-> debug logging
```

它已经记录：

```text
stage
track_id / det_global_idx
final cost
raw IoU cost
embedding cost
chosen
track age / lost age
detection score
```

因此不应再从头开发一套前置 approximate logger。

但现有实现有三个关键缺陷：

```text
1. 每条 track 只记录 cost 最低的 top-3 detections；
   Hungarian 选择 rank>3 时，真实 chosen pair 会被漏掉。

2. row_col_margins 记录的是该行/列“全局 best 与 second-best 的差”，
   不是 actual chosen pair 相对其他候选的 signed margin。
   当 Hungarian 因冲突选择第二名时，现有 margin 可能方向错误。

3. 缺少 smooth_feat-current_feat consistency、track state、intervention flag。
```

正式 observe logger 应：

```text
- 记录 top-K 与所有 chosen pair 的并集；
- 对 chosen(i,j) 计算：
    row_signed = min(cost[i, k != j]) - cost[i,j]
    col_signed = min(cost[k != i,j]) - cost[i,j]
- 记录 chosen rank；
- 记录实际 valid masks；
- 记录 smooth/current cosine；
- 记录 active/lost/unconfirmed state；
- soft 运行时记录 soft_applied / alpha；
- 在线日志不使用 GT，GT 仅后处理标注。
```

---

## 7. 研究设计修正

01/05 在仓库中已经存在大量 ShadowPDA、oracle 与历史实验，不能再称：

```text
完全未见 holdout
```

更准确的定位：

```text
locked evaluation sequences
```

要求：

```text
- 新 action 只在 02/03 设计；
- 参数和决策逻辑冻结后运行 01/05；
- 看见 01/05 结果后不得回调参数；
- 若回调，必须重新定义下一轮验证协议；
- 最终泛化依赖 MOT20 test 一次性提交或外部数据集。
```

由于 MOT20-05 dump 约 2.79 GB、输出约 63 万行，而 MOT20-01 dump 约 89 MB、输出约 1.9 万行，执行顺序应改为：

```text
先 MOT20-01 做低成本第三序列 falsification；
再 MOT20-05 做拥挤大规模 stress test。
```

---

## 8. 更新后的下一步

### Step 1：正式固化 oracle v2

将已验证的临时实现整理为：

```text
scripts/current_output_id_oracle_v2.py
```

必须包含：

```text
threshold-before-Hungarian
TrackEval distractor preprocessing
geometry/score/tail strict consistency
assignment-mode metadata
fixed-IoU 命名与免责声明
```

### Step 2：建立 association-aligned observe wrapper

建议新脚本：

```text
scripts/dmm_base_tracker_trusttrack_observe_v2.py
```

不要修改 `scripts/dmm_base_tracker.py`。

基于现有 `_record_assoc_debug` 扩展：

```text
- all chosen pairs
- chosen-specific signed margins
- chosen rank
- valid masks
- feature consistency
- stage/state/lost age
- optional soft intervention fields
```

### Step 3：先在 M02/M03 baseline 上生成真实日志

目标：

```text
1. 验证日志 chosen 数与实际 assignment 数一致；
2. GT 后处理使用 TrackEval-aligned matching；
3. 标签以 GT→tracker transition 为主；
4. 重新评估 collapse / app margin / det score 的跨序列 AUC 与 calibration；
5. 明确哪些 cue 只是相关，哪些能提供增量。
```

### Step 4：在 M03 soft 上加入真实日志，定位净负事件群

当前已经确认：

```text
frame 82 不是负收益来源。
```

下一步不应继续猜单事件，而应：

```text
- join soft pair log 与 actual chosen-pair log；
- 按真实 GT transition 标注；
- 按时间块/identity cluster 做 leave-block-out；
- 找出对 HOTA/AssA/IDSW 有真实最终边际影响的事件群。
```

只有全序列 counterfactual 才能称为最终贡献。

### Step 5：固定 soft v1 参数，先跑 MOT20-01，再跑 MOT20-05

每个序列同时输出：

```text
same-script baseline
soft v1
observe-v2 logs
oracle v2
per-sequence TrackEval
```

01/05 结果只用于 locked evaluation，不调参。

### Step 6：再决定 action 方向

在真实日志完成前，不应直接开发四 cue weighted sum。

原因：

```text
当前 baseline 的真实 primary decision 主要只有 IoU 与 ReID，
motion/shape 还只是辅助诊断量。
```

优先级应为：

```text
A. identity-prototype update action
   - 使用 actual chosen pair
   - contamination probability
   - alpha 从 0.9 连续变化
   - 明确定义是否保护 features deque
   - active 与 reactivation 分开

B. two-cue reliability gating
   - 先在真实 IoU/ReID cost 间做可靠性门控
   - 不急于引入未经校准的四 cue 加权和

C. abstention/delay
   - 只有当真实 chosen-specific row/col margin 都低时再考虑
```

### Step 7：停止旧 sweep

继续停止：

```text
binary freeze sweep
soft_start / alpha 盲扫
现有 appearance-veto Association v1 参数扫
```

除非新的 actual-association logger 给出明确、预注册的假设。

---

## 9. 最新可信状态

```text
1. cue collapse 在修正候选池后仍与 wrong-ID / future identity transition 稳定相关，
   并对简单 cue 模型有增量价值；但它尚未在真实 association matrix 上完成验证。

2. soft v1 在 M02 是经过选择的 development 小正结果，在 M03 是跨序列负结果；
   02+03 combined 仍为 HOTA -0.018、AssA -0.039、IDSW +3。

3. frame 51 / 82 只能解释首次局部分叉，不能解释最终序列收益；
   frame 82 实际是微弱正向事件，M03 净负来源仍未定位。

4. 四序列 fixed-IoU ID oracle 显示 association 仍有显著空间，
   但 MOT20-02 是异常大 outlier，四序列 combined 上界约 HOTA +4.872。

5. 下一步不是设计新规则，而是把现有真实 debug_assoc 扩展为完整 chosen-pair observe logger，
   再以 GT-centric、全序列 counterfactual 的标准定位真正可干预事件。
```
