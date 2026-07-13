# TrustTrack / ShadowPDA 最新交接文档

> 更新时间：2026-07-10
>
> 工作目录：`/gemini/code/FMtrack-main/FM-Track`
>
> 新会话请先完整阅读本文档，再继续实验。
>
> **重要保护原则：**
> - 不要修改 `scripts/dmm_base_tracker.py`。
> - 不要覆盖任何 recovery pyc 或稳定 wrapper。
> - 不要使用 `apply_patch` 修改 baseline/稳定核心文件。
> - 新方法继续使用独立脚本和独立 `outputs/` 目录。
> - GT / TrackEval / oracle 只能用于诊断，不得进入在线算法。

---

## 1. 当前最重要结论

当前已经明确区分了三个层次：

```text
1. ShadowPDA：低分证据 / lost-track fallback，收益局部，不能作为冲高 HOTA 主引擎。
2. TrustTrack 诊断：cue collapse / local ambiguity 能强预测坏匹配，这个发现成立。
3. TrustTrack action：binary freeze 是负结果；soft memory update 是当前已验证的小幅正向模块。
```

主线应从：

```text
low-score public recovery
```

升级为：

```text
ambiguity-aware association framework
  + adaptive cue weighting
  + soft identity-memory update
  + necessary match delay
```

---

## 2. TrustTrack 核心命题修正

原始 idea：

```text
Reliability Reversal Score (RRS) 高时容易发生 ID switch。
```

MOT20-02 / MOT20-03 诊断表明：

```text
原始 RRS 不是稳定单调风险信号。
```

真正稳定的信号是：

```text
cue_collapse
max_pair_margin
app_col_margin
motion_col_margin
```

推荐论文命题：

```text
Association failures are better predicted by local cue ambiguity collapse
than by raw cue similarity or fixed cue reliability reversal.
```

中文：

```text
关联失败更直接地由局部线索区分力塌陷预测，
而不是由某条线索相似度高低或简单线索可靠性反转预测。
```

推荐方法定位：

```text
TrustTrack: Adaptive Multi-Cue Association via Local Ambiguity Estimation
for Crowded Multi-Object Tracking
```

---

## 3. 已完成的 cue-collapse 诊断

### 3.1 诊断文件

```text
scripts/trusttrack_rrs_diagnostic.py
outputs/analyze_rrs_csv.py
outputs/trusttrack_rrs_m02_norecovery.csv
outputs/analyze_rrs_m02.out
outputs/trusttrack_rrs_m03_norecovery.csv
outputs/analyze_rrs_m03.out
```

### 3.2 MOT20-02

```text
rows = 145060
correct = 138672
det_fp = 1615
wrong_id = 4715
unknown = 58
```

cue collapse 分箱：

```text
collapse 0.001~0.206: bad_rate = 0.45%
collapse 0.524~0.675: bad_rate = 7.28%
collapse 0.784~0.843: bad_rate = 16.43%
collapse 0.843~0.951: bad_rate = 26.44%
collapse 0.951~1.335: bad_rate = 51.14%
```

app-col 最低 10%：

```text
bad_rate = 26.94%
wrong/FP rate = 18.13%
```

### 3.3 MOT20-03

```text
rows = 302553
bad = 7733
wrongfp = 5619
```

cue collapse 分箱：

```text
collapse 0.001~0.116: bad_rate = 0.48%
collapse 0.489~0.642: bad_rate = 3.49%
collapse 0.642~0.732: bad_rate = 9.22%
collapse 0.732~0.884: bad_rate = 16.40%
collapse 0.884~1.343: bad_rate = 34.70%
```

结论：02/03 一致支持 local cue ambiguity / collapse。

### 3.4 MOT20-05

05 full 诊断尚未完成，不应继续盲跑大 CSV。后续若需要，先裁剪小 dump 再跑。

---

## 4. 同参数 MOT20-02 baseline

严格对照结果：

```text
结果文件：
outputs/shadow_pda_m02_full_norecovery/track_results/MOT20-02.txt

评估：
outputs/trust_lite_m02_baseline_eval/eval.out
```

指标：

```text
HOTA  = 68.748
DetA  = 78.844
AssA  = 60.053
MOTA  = 90.750
IDF1  = 75.205
IDSW  = 428
FN    = 11556
FP    = 2329
Frag  = 765
Dets  = 145515
IDs   = 475
```

所有后续 MOT20-02 实验必须优先与这组 same-script baseline 对比，不要混用历史 ParamBest 指标。

---

## 5. Binary memory freeze：可信负结果

实现：

```text
scripts/dmm_base_tracker_trusttrack_lite.py
```

结果目录：

```text
outputs/trust_lite_m02_full_primary/
```

配置：

```text
trust_min_det_score = 0.6
collapse_thresh = 0.85
app_col_thresh = 0.05
motion_col_thresh = 0.05
```

指标：

```text
HOTA  = 68.047
DetA  = 78.878
AssA  = 58.816
MOTA  = 90.789
IDF1  = 74.723
IDSW  = 419
FN    = 11521
FP    = 2313
Frag  = 759
Dets  = 145534
IDs   = 466
```

相对 baseline：

```text
HOTA  -0.701
AssA  -1.237
IDF1  -0.482
IDSW  -9
FN    -35
FP    -16
Frag  -6
```

解释：

```text
freeze 减少了部分错误传播和 IDSW，
但冻结太多正确更新，造成 identity memory 老化，AssA/HOTA 明显下降。
```

结论：

```text
binary freeze 不作为正向主模块，只保留为 negative ablation。
```

---

## 6. True TrustTrack-soft 实现状态

### 6.1 可信脚本

```text
scripts/dmm_base_tracker_trusttrack_soft.py
```

该脚本已经重新构建，并做过三重验证：

```text
1. 真实服务器源码含 --trust-soft-* 参数。
2. 删除旧 pyc 后重新 py_compile。
3. 运行时 inspect.getsource(parse_args) 与实际 parse_args() 均确认 soft 参数被正确解析。
```

soft 行为：

```text
匹配/Kalman/检测/生命周期均不改；
命中高 ambiguity pair 时，临时提高该 track 的 EMA alpha，
调用原 DMMTrack.update() 后恢复原 alpha。
```

### 6.2 100 帧严格 smoke

目录：

```text
outputs/trust_soft_true_m02_100_notrust/
outputs/trust_soft_true_m02_100_v1/
```

结果：

```text
no-trust MD5 = 306ba3d4a52f7402c400418247cc93d5
true-soft MD5 = 6e2f3081d72417bf0bf9e02caa911a0b
```

true-soft：

```text
feature_updates_soft = 28
feature_updates_normal = 3799
mean_alpha ≈ 0.98617
```

说明 soft 逻辑真实生效且输出发生变化。

### 6.3 旧目录警告

```text
outputs/trust_soft_m02_full_v1/
```

该目录是在服务器真实源码/parser 尚未最终核实时生成，过去曾被标记为不可信。

现在新的 true-soft 独立复现实验得到完全相同的主指标，因此结果本身已经被重新验证；但新会话仍应只引用：

```text
outputs/trust_soft_true_m02_full_v1/
```

不要引用旧目录作为证据。

---

## 7. True soft-memory v1：可信正向结果

目录：

```text
outputs/trust_soft_true_m02_full_v1/
```

运行脚本：

```text
outputs/run_trust_soft_true_m02_full_v1.sh
```

配置：

```text
soft_start = 0.80
soft_extreme = 1.00
soft_alpha = 0.98
extreme_alpha = 0.995
min_det_score = 0.60
min_track_age = 5
app/motion col gate disabled
```

实际 soft 统计：

```text
feature_updates_soft = 1716
feature_updates_normal = 143841
soft update rate ≈ 1.179%
mean soft alpha = 0.985077
```

指标：

```text
HOTA  = 68.949
DetA  = 78.876
AssA  = 60.378
MOTA  = 90.787
IDF1  = 75.610
IDSW  = 423
FN    = 11514
FP    = 2320
Frag  = 760
Dets  = 145548
IDs   = 474
```

相对 baseline：

```text
HOTA  +0.201
DetA  +0.032
AssA  +0.325
MOTA  +0.037
IDF1  +0.405
IDSW  -5
FN    -42
FP    -9
Frag  -5
```

结论：

```text
少量、连续型的 ambiguity-aware soft update 是可信正向模块；
它明显优于 binary freeze。
```

---

## 8. True soft-memory v2：soft_start=0.85

目录：

```text
outputs/trust_soft_true_m02_full_v2_start085/
```

配置仅改变：

```text
soft_start: 0.80 -> 0.85
```

统计：

```text
feature_updates_soft = 844
feature_updates_normal = 144715
soft update rate ≈ 0.580%
mean alpha = 0.986531
```

指标：

```text
HOTA  = 68.835
DetA  = 78.808
AssA  = 60.232
MOTA  = 90.771
IDF1  = 75.548
IDSW  = 422
FN    = 11524
FP    = 2335
Frag  = 758
Dets  = 145553
IDs   = 473
```

相对 baseline：

```text
HOTA  +0.087
DetA  -0.036
AssA  +0.179
MOTA  +0.021
IDF1  +0.343
IDSW  -6
FN    -32
FP    +6
Frag  -7
```

对比 v1：

```text
v2 冻结/软化更少，IDSW/Frag 略好，
但 HOTA/AssA/IDF1 不如 v1。
```

当前推荐默认：

```text
soft_start = 0.80
```

但在跨序列验证前不能宣称它是最终最优参数。

---

## 9. Current-output ID oracle：最关键新结果

### 9.1 Oracle 工具

```text
scripts/current_output_id_oracle.py
outputs/run_current_output_id_oracle_m02.sh
```

Oracle 规则：

```text
保持每个 tracker 输出的 frame / box / score / row 数完全不变；
每帧用 Hungarian + IoU>=0.5 匹配有效 pedestrian GT；
matched output ID 替换为 GT ID；
unmatched output 使用 offset ID 保留为 FP。
```

严格一致性检查：

```text
baseline rows = 145567
oracle rows = 145567
除 ID 外 frame/box/score mismatch = 0
```

因此 oracle 的变化确实只来自 ID 重赋值。

### 9.2 Oracle 输出

```text
outputs/current_output_id_oracle_m02_norecovery/
```

匹配统计：

```text
tracker_rows = 145567
matched_to_valid_pedestrian_gt = 143553
unmatched_tracker_rows = 2014
match_rate = 98.6164%
mean IoU = 0.88040
median IoU = 0.89408
p10 IoU = 0.79336
```

Oracle TrackEval：

```text
HOTA  = 81.681
DetA  = 80.294
AssA  = 83.131
MOTA  = 91.454
IDF1  = 95.606
IDSW  = 22
FN    = 11215
FP    = 1988
Frag  = 1046
```

相对 baseline：

```text
HOTA  +12.933
AssA  +23.078
IDF1  +20.401
IDSW  -406
```

最重要结论：

```text
MOT20-02 在完全不改变检测框的情况下，HOTA ceiling 已达到 81.681。
所以当前检测框质量足以支撑 80+；首要瓶颈是 association / identity assignment。
```

注意：HOTA DetA / CLEAR TP-FN-FP 也会变化，因为 TrackEval 的匹配过程受 ID continuity / global alignment 影响；但输入框已严格验证完全一致。

Oracle 只能用于诊断，绝不能进入在线方法。

---

## 10. 对检测问题的最新判断

MOT20-02 oracle 已经证明：

```text
短期达到 80+ 的主要限制不是必须先换 detector，
而是关联不能把现有高质量输出框正确串成身份轨迹。
```

因此优先级应调整为：

```text
1. association v1 / adaptive cue weighting
2. ambiguous commit/delay
3. soft memory update
4. tracklet-conditioned detection score calibration
5. 最后才是 detector retraining / visible-full box / quality head
```

检测仍然会限制更高上限，但不是当前第一瓶颈。

---

## 11. 下一步最高优先级实验

### Priority 1：跨序列验证 true soft v1

先不要继续在 MOT20-02 调更多 alpha。

直接使用 v1 参数跑：

```text
MOT20-03
MOT20-05
MOT20-01（确认中性/不伤）
```

推荐顺序：

```text
1. MOT20-03
2. MOT20-05
3. MOT20-01
```

使用脚本：

```text
scripts/dmm_base_tracker_trusttrack_soft.py
```

参数：

```text
--trust-soft-enable
--trust-soft-start 0.80
--trust-soft-extreme 1.00
--trust-soft-alpha 0.98
--trust-extreme-alpha 0.995
--trust-min-track-age 5
```

每序列必须和 same-script no-trust baseline 对比：

```text
HOTA / DetA / AssA / IDF1 / IDSW / FN / FP / Frag
soft update count / rate / mean alpha
```

### Priority 2：ID oracle 扩展到 01/03/05

复用：

```text
scripts/current_output_id_oracle.py
```

目标：

```text
得到各序列 current-output association ceiling，
确认 02 是否特殊，还是四序列普遍存在巨大关联空间。
```

### Priority 3：TrustTrack Association v1

soft memory 已经是正向辅助模块，但不会带来 10+ HOTA。

真正主模块应进入 cost/commit 决策：

```text
pair_margin_c = min(row_margin_c, col_margin_c)
reliability_c = calibrated positive function(pair_margin_c)

cost(i,j) = sum_c reliability_c(i,j) * normalized_cost_c(i,j)
            / sum_c reliability_c(i,j)
```

第一版只做：

```text
appearance / motion / IoU 三 cue 动态权重
```

先不改生命周期、不做 delay。

### Priority 4：Ambiguous match delay / abstention

当：

```text
max_pair_margin 非常低
且 best-second association gap 很小
```

再考虑一帧 pending / abstain。

这一步需要可编辑的真实 association 阶段；当前 DMMBaseTracker 来自 pyc，需考虑迁移可编辑源码或新 wrapper。

---

## 12. 建议新会话立即执行的命令方向

新会话第一句话：

```text
请读取 docs/TRUSTTRACK_HANDOFF_NEXT_SESSION.md，
从第 11 节开始，先跑 true soft v1 在 MOT20-03 的 same-script baseline 与 soft 对照，
然后做 MOT20-03 current-output ID oracle。
```

不要先做：

```text
- 不要继续 binary freeze sweep。
- 不要继续调 ShadowPDA-V3 release threshold。
- 不要把旧 outputs/trust_soft_m02_full_v1 当独立证据。
- 不要先重训 detector。
```

---

## 13. 关键文件清单

### TrustTrack 诊断

```text
scripts/trusttrack_rrs_diagnostic.py
outputs/analyze_rrs_csv.py
outputs/trusttrack_rrs_m02_norecovery.csv
outputs/analyze_rrs_m02.out
outputs/trusttrack_rrs_m03_norecovery.csv
outputs/analyze_rrs_m03.out
```

### Freeze negative ablation

```text
scripts/dmm_base_tracker_trusttrack_lite.py
outputs/trust_lite_m02_full_primary/
```

### True soft implementation/results

```text
scripts/dmm_base_tracker_trusttrack_soft.py
outputs/trust_soft_true_m02_100_notrust/
outputs/trust_soft_true_m02_100_v1/
outputs/trust_soft_true_m02_full_v1/
outputs/trust_soft_true_m02_full_v2_start085/
outputs/run_trust_soft_true_m02_full_v1.sh
```

### Oracle

```text
scripts/current_output_id_oracle.py
outputs/run_current_output_id_oracle_m02.sh
outputs/current_output_id_oracle_m02_norecovery/
```

### Baseline

```text
outputs/shadow_pda_m02_full_norecovery/track_results/MOT20-02.txt
outputs/trust_lite_m02_baseline_eval/eval.out
```

---

## 14. 当前一句话状态

```text
cue collapse 已被证明能定位关联风险；
binary freeze 是负结果；
true ambiguity-aware soft memory 在 MOT20-02 上可信提升 HOTA +0.201；
current-output ID oracle 在不改框的情况下达到 HOTA 81.681，
证明下一阶段必须主攻 association，而不是继续围绕 detector 或低分恢复打转。
```

---

## 15. 本轮新增：MOT20-03 same-script soft v1 与 current-output ID oracle

> 完成时间：2026-07-10 15:56

### 15.1 严格 same-script baseline / soft v1

本轮使用同一个脚本：

```text
scripts/dmm_base_tracker_trusttrack_soft.py
```

只通过是否传入 `--trust-soft-enable` 区分控制组和 soft v1；其余 tracker 参数完全一致。

结果目录：

```text
outputs/trust_soft_true_m03_same_script_baseline/
outputs/trust_soft_true_m03_same_script_soft_v1/
outputs/trust_soft_true_m03_same_script_compare_20260710/
```

统一 runner：

```text
outputs/trust_soft_true_m03_same_script_compare_20260710/run.sh
```

baseline 严格检查：

```text
trust_soft.enable = false
feature_updates_soft = 0
rows(summary) = 303355
MD5 = ff47a3ec83f31709033ce7b0b16e8013
```

该 MD5 与历史 `outputs/shadow_pda_v3_norecovery_m03_full/track_results/MOT20-03.txt` 完全一致，说明 same-script no-trust 控制组复现成功。

soft v1 参数：

```text
soft_start = 0.80
soft_extreme = 1.00
soft_alpha = 0.98
extreme_alpha = 0.995
min_det_score = 0.60
min_track_age = 5
app/motion col gate disabled
```

soft 实际统计：

```text
feature_updates_soft = 529
feature_updates_normal = 307338
soft update rate = 0.171827%
mean soft alpha = 0.9854779
MD5 = 9ec3b40f3d006a3681c1c6de38bdb161
```

soft 输出 MD5 与此前 `outputs/trust_soft_true_m03_full_v1/` 完全一致，说明结果可复现。

指标：

```text
                         baseline     soft v1      delta
HOTA                      80.054       79.943      -0.111
DetA                      80.911       80.903      -0.008
AssA                      79.241       79.029      -0.212
MOTA                      94.306       94.305      -0.001
IDF1                      94.169       94.015      -0.154
IDSW                         178          186          +8
FN                         13993        13989          -4
FP                          3689         3688          -1
Frag                         486          492          +6
Dets                      303354       303357          +3
IDs                          777          778          +1
```

结论：

```text
true soft v1 在 MOT20-03 上是小幅负结果：
HOTA / AssA / IDF1 均下降，IDSW 和 Frag 增加。

因此 MOT20-02 的 +0.201 HOTA 不能直接外推为跨序列稳定收益；
当前 soft memory 应降级为 sequence-sensitive auxiliary / ablation，
至少完成 MOT20-05 与 MOT20-01 验证前，不能作为默认通用模块。
```

### 15.2 MOT20-03 current-output ID oracle

Oracle 源输出使用本轮 same-script baseline：

```text
outputs/trust_soft_true_m03_same_script_baseline/track_results/MOT20-03.txt
```

Oracle 目录：

```text
outputs/current_output_id_oracle_m03_same_script_baseline/
```

严格一致性检查：

```text
baseline rows = 303355
oracle rows = 303355
row_count_equal = true
frame mismatch = 0
box mismatch = 0
score mismatch = 0
tail mismatch = 0
non-ID mismatch total = 0
```

匹配统计：

```text
tracker_rows = 303355
matched_to_valid_pedestrian_gt = 299713
unmatched_tracker_rows = 3642
match_rate = 98.799426%
mean IoU = 0.8643776
median IoU = 0.8755860
p10 IoU = 0.7785360
duplicate_id_frames = 0
```

Oracle TrackEval：

```text
                         baseline     ID oracle    delta
HOTA                      80.054       82.176      +2.122
DetA                      80.911       81.185      +0.274
AssA                      79.241       83.203      +3.962
MOTA                      94.306       94.389      +0.083
IDF1                      94.169       97.150      +2.981
IDSW                         178           18        -160
FN                         13993        13943         -50
FP                          3689         3639         -50
Frag                         486          642        +156
Dets                      303354       303354           0
IDs                          777         1069        +292
```

最重要结论：

```text
MOT20-03 当前 baseline 已经达到 HOTA 80.054；
只重赋 ID 的 current-output oracle ceiling 为 HOTA 82.176。

因此 MOT20-03 仍有明确 association 空间，
但增量仅 +2.122 HOTA，远小于 MOT20-02 的 +12.933。
这证明 MOT20-02 的巨大 identity gap 不是四序列都可默认共享的量级，
association headroom 具有很强的序列差异。
```

注意：

```text
oracle 的 Frag / IDs 增加来自逐帧 GT ID 重赋值及 unmatched offset ID 机制；
它仍然只用于诊断 current-output identity ceiling，不能作为在线算法。
```

### 15.3 更新后的下一步优先级

```text
1. MOT20-05：同样执行 same-script no-trust baseline / true soft v1，再做 current-output ID oracle。
2. MOT20-01：确认 soft v1 是否中性或有伤害，并补 oracle。
3. 汇总 01/02/03/05 后，再决定 soft memory 是否保留为默认模块、条件模块或仅 ablation。
4. Association v1 必须报告逐序列收益，不能只看 MOT20-02 或四序列合并均值。
```

### 15.4 最新一句话状态

```text
MOT20-03 same-script 对照已完成：true soft v1 HOTA -0.111，属于小幅负结果；
MOT20-03 current-output ID oracle 在不改框的前提下达到 HOTA 82.176（+2.122），
说明该序列已有 80+ baseline，仍有中等 association 空间，但远小于 MOT20-02 的巨大 identity gap。
```

---

## 16. 深度复核修正：soft action、oracle 与 Association v1

> 详细报告：`docs/TRUSTTRACK_DEEP_REVIEW_20260710.md`

### 16.1 对第 7 / 14 / 15 节表述的修正

```text
1. true soft v1 只能称为 MOT20-02 单序列小幅正结果，
   不能再称为跨序列可信正向模块。

2. MOT20-02 + MOT20-03 合并：
   HOTA -0.018，AssA -0.039，IDF1 +0.029，IDSW +3。

3. soft 不直接改变当前帧 Hungarian 代码，
   但它改变 memory，进而会改变未来关联、输出和生命周期结果。

4. current-output oracle 应称 fixed-IoU diagnostic oracle，
   不是数学严格 association ceiling。
```

### 16.2 soft v1 机制缺陷

```text
- cue matrix 在 tracker.update() 前计算；
- 使用 Kalman multi_predict 前的旧 track.tlbr；
- 只包含 activated tracked_stracks，不含 lost_stracks；
- 没有对齐真实 assoc_cost valid mask；
- 使用自定义 app/motion/IoU/shape similarity，而不是实际 baseline cost；
- monkeypatch 只覆盖 DMMTrack.update，不覆盖 re_activate；
- 触发后 alpha 从默认 0.9 至少跳到 0.98，并非从 baseline 连续变化。
```

### 16.3 MOT20-03 首个有害分叉：严格因果证据

```text
frame 82, track 60, det 7150
```

反事实：

```text
只保留 frame 82 soft update：完整复现 frame 87~106 分叉；
排除 frame 82：前 120 帧完全恢复 baseline。
```

身份：

```text
track 历史 GT = 531
current det best GT = 531
这是正确同身份更新。
```

cue：

```text
app_pair_margin    = +0.1030
motion_pair_margin = -0.1369
iou_pair_margin    = -0.2798
collapse           = 0.8970
```

结论：

```text
geometry ambiguity 误触发 memory suppression，
抑制了后续重连所需的正确 appearance evidence。
```

敏感性：

```text
alpha <= 0.905：等于 baseline
alpha >= 0.910：进入完整 soft 分叉
```

### 16.4 MOT20-02 首个分叉：符合污染保护假设的严格因果证据

```text
frame 51, track 35, det 2031
```

反事实：

```text
只保留 frame 51 soft update：完整复现 frame 53~61 分叉；
排除 frame 51：前 70 帧完全恢复 baseline。
```

身份：

```text
track 历史 GT = 230
current det best GT = 224, IoU = 0.6085
属于跨身份污染候选。
```

cue：

```text
app_pair_margin    = -0.0206
motion_pair_margin = -0.0511
iou_pair_margin    = +0.0416
collapse           = 0.9584
```

核心修正：

```text
generic ambiguity != memory contamination risk。
真正 action 需要 identity consistency，而不是只看 collapse。
```

### 16.5 触发质量

使用检测自身最佳有效 GT 做离线标签：

```text
MOT20-02：
  same identity  = 1474 / 1716 = 85.90%
  cross identity =  232 / 1716 = 13.52%

MOT20-03：
  same identity  = 413 / 529 = 78.07%
  cross identity =  95 / 529 = 17.96%
```

大多数 soft 动作发生在同身份更新上，当前 trigger 不是高精度 contamination detector。

### 16.6 Oracle 实现修正

旧实现是：

```text
Hungarian -> 再删除 IoU<0.5 pair
```

修正为：

```text
先 threshold valid mask -> 再 Hungarian
```

修正后：

```text
MOT20-02:
  HOTA 81.683
  AssA 83.138
  IDF1 95.618
  IDSW 2

MOT20-03:
  HOTA 82.178
  AssA 83.208
  IDF1 97.154
  IDSW 0
```

旧 HOTA 主结论基本不变，但旧 IDSW=22/18 作废。

阈值敏感性：

```text
MOT20-02: t0.3 HOTA 81.724 / t0.5 81.683 / t0.7 80.733
MOT20-03: t0.3 HOTA 82.225 / t0.5 82.178 / t0.7 80.946
```

### 16.7 已存在但遗漏的 Association v1

```text
scripts/dmm_base_tracker_trusttrack_assoc_v1.py
outputs/trust_assoc_v1_m02_full/
```

严格 same-wrapper disabled 对照已补：

```text
HOTA  +0.078
DetA  +0.279
AssA  -0.076
IDF1  -0.063
IDSW  +19
Frag  +6
```

结论：

```text
这是 appearance-veto mixed/negative ablation，
不是已经验证的 adaptive association 正向模块。
```

### 16.8 更新后的执行顺序

```text
Priority 0:
  停止 binary freeze / alpha / soft_start / appearance-veto 参数盲扫。

Priority 1:
  新建 current_output_id_oracle_v2.py；
  threshold-before-Hungarian，并与 TrackEval preprocessing 对齐。

Priority 2:
  建立 association-aligned matched-pair logger：
  在 GMC/Kalman predict 后、真实 strack_pool/high_dets/assoc_cost 上记录实际 match。

Priority 3:
  使用新 logger 跑 MOT20-05 same-script baseline/soft/oracle；
  再跑 MOT20-01。

Priority 4:
  四序列完成后冻结 soft v1 为 negative/mixed/conditional ablation。

Priority 5:
  开发 identity-consistency-aware memory action：
  alpha 从 0.9 连续变化，geometry ambiguity 不能单独触发 suppression。

Priority 6:
  开发对称 adaptive association v2：
  appearance 可靠时保留 appearance；
  appearance 弱、geometry 强时降低 appearance；
  全 cue ambiguous 时再考虑 abstention/delay。
```

### 16.9 最新一句话状态

```text
cue collapse 的诊断价值成立，但 action 映射尚未成立；
soft v1 在 02 能保护一次跨身份污染候选，
在 03 却抑制了一次关键正确同身份更新。

下一步必须先做真实 association-aligned diagnostics，
再设计 identity-consistency-aware action，不能继续围绕旧 collapse 阈值盲调。
```

---

## 17. 第二轮反向审稿修正：撤销单事件最终贡献判断

> **本节优先级高于第 16 节。**
>
> 详细报告：`docs/TRUSTTRACK_DEEP_REVIEW_V2_20260710.md`

### 17.1 必须撤销的旧结论

```text
1. MOT20-03 frame 82 不是最终负收益事件。
2. MOT20-02 frame 51 不能解释最终 +0.201 HOTA。
3. 首次局部分叉因果 != 最终指标边际贡献。
```

全序列反事实：

```text
MOT20-02 frame 51:
  only51 与 baseline 在正式指标精度上相同；
  exclude51 与 full soft 相同。
  该事件只造成 frame 53~61 局部分叉，最终边际贡献近似 0。

MOT20-03 frame 82:
  only82 相对 baseline：HOTA +0.002，IDF1 +0.005，IDSW -2；
  exclude82 相对 full soft：HOTA -0.001，IDF1 -0.005，IDSW +2。
  该事件实际是微弱正向，M03 总体负收益来自其他事件或交互。
```

GT-centric transition：

```text
baseline 在 frame 87/90 将 GT 531/553 从 track 72 切到 track 60；
only82 避免这两个 ID change。
```

后续必须以：

```text
GT -> tracker ID transition
```

作为事件方向标准，不能只看 tracker 的 majority GT。

### 17.2 cue-collapse 诊断强度修正

原诊断把所有历史 tracker ID 永久保留为 col competitor。

真实配置：

```text
track_buffer=70, frame_rate=25
=> max_time_lost=58
```

按 58 帧候选重算：

```text
MOT20-02:
  AUC 0.782 -> 0.787
  top10 precision 23.9% -> 25.2%

MOT20-03:
  AUC 0.840 -> 0.781
  top10 precision 14.6% -> 12.3%
```

结论：

```text
collapse 信号仍成立，但 MOT20-03 原强度被陈旧列竞争放大。
准确表述应为：在当前离线近似诊断中跨 02/03 稳定相关，
尚未在真实 association matrix 上完成验证。
```

目标拆分：

```text
wrong_id AUC:
  M02 0.802 / M03 0.861

currently-correct -> future10 identity change AUC:
  M02 0.825 / M03 0.864

det_fp AUC:
  M02 0.614 / M03 0.659
```

跨序列增量：

```text
02 -> 03:
  simple cues 0.871
  + collapse  0.898

03 -> 02:
  simple cues 0.813
  + collapse  0.859
```

因此 collapse 是增量风险特征，不是已校准 action probability。

### 17.3 M02 same-script full baseline 已补齐

```text
outputs/trusttrack_review_20260710/m02_same_script_full_notrust/
```

MD5：

```text
same-script no-trust = 8cd345b4de6efd9cb9d6f6b1427bd2ec
historical baseline  = 8cd345b4de6efd9cb9d6f6b1427bd2ec
```

M02 对照公平性闭环。

但 M02 soft v1 是在多个方案比较后选出的 development result，不是独立验证结果。

### 17.4 soft v1 的额外机制边界

```text
- 只减弱 smooth_feat EMA；
- features deque 仍写入 current feature；
- 不覆盖 re_activate；
- 不覆盖 low-score / unconfirmed stage；
- 当前 simple motion / shape 不是 baseline 真实独立 cost cue。
```

因此不应称为完整长期 memory protection 或已实现四 cue adaptive association。

### 17.5 oracle v2 最终修正

必须包含：

```text
threshold-before-Hungarian
TrackEval distractor preprocessing
strict non-ID consistency
```

最终修正版：

```text
Seq       baseline HOTA   oracle HOTA   delta
MOT20-01      77.030         82.178     +5.148
MOT20-02      68.748         81.684    +12.936
MOT20-03      80.054         82.178     +2.124
MOT20-05      78.539         82.954     +4.415

All-4 combined:
  baseline HOTA 77.699
  oracle HOTA   82.571
  delta         +4.872
```

注意：只改输入 ID 仍会改变 HOTA/CLEAR 内部匹配，因此 DetA/FN/FP 可小幅变化；不能把全部 HOTA delta 称为纯 AssA 增益。

### 17.6 内置 debug_assoc 的复用与缺陷

现有 `_record_assoc_debug()` 已在真实 predict/association/Hungarian 后，可作为基础。

但必须修复：

```text
1. top-3 日志可能漏掉 rank>3 的实际 chosen pair；
2. 当前 row/col margin 是全局 best-second，不是 chosen-specific signed margin；
3. 缺少 feature consistency、state、soft intervention fields。
```

正式 logger 必须记录：

```text
all chosen pairs
chosen rank
chosen-specific signed row/col margin
raw IoU / embedding / final cost
valid masks
smooth-current cosine
stage / state / lost age
soft_applied / alpha
```

### 17.7 最新下一步顺序

```text
Step 1:
  固化 scripts/current_output_id_oracle_v2.py。

Step 2:
  新建 dmm_base_tracker_trusttrack_observe_v2.py，
  扩展真实 debug_assoc，不修改核心 baseline。

Step 3:
  在 M02/M03 baseline 上生成 actual chosen-pair logs，
  用 TrackEval-aligned GT 做 GT-centric 后标注。

Step 4:
  在 M03 full soft 上 join intervention log，
  做时间块/identity cluster leave-out，定位真正净负事件群。

Step 5:
  固定参数后先跑较小的 MOT20-01，
  再跑 2.79GB 的 MOT20-05 stress test。
  01/05 称 locked evaluation sequences，不再称 pristine holdout。

Step 6:
  四序列证据完成后再决定 action：
  优先 actual-pair prototype update 与真实 IoU/ReID two-cue gating，
  暂不直接做未经校准的四 cue weighted sum。
```

### 17.8 最新可信一句话状态

```text
cue collapse 是有增量价值的 identity-risk 特征，但仍停留在离线近似诊断层；
soft v1 跨 02/03 不稳定，且此前 frame 51/82 的最终贡献方向判断已撤销；
M03 净负来源尚未定位。

下一步必须扩展真实 debug_assoc，记录所有 actual chosen pair 与 chosen-specific margin，
再做 GT-centric、全序列 counterfactual，之后才能设计新 action。
```

---

## 18. 第三轮证据链审计：双键协同、精确正向链与 Commit Override

> **本节优先级高于第 17 节。**
>
> 详细报告：`docs/TRUSTTRACK_DEEP_REVIEW_V3_20260711.md`

### 18.1 必须修正的旧解释

```text
1. MOT20-03 frame1808 不能继续称为“干净的正确GT678同身份更新”。
   chosen-only Hungarian标GT678，但direct-best GT是385，IoU 0.8195；
   同帧另一个检测也direct-best GT385，属于严重one-to-one拥挤冲突。

2. 跨运行raw tracker ID不能直接比较。
   M02 raw同ID率仅18.75%，最常见只是baseline_id = soft_id + 3；
   旧different_track=1395绝大部分是编号漂移，不是身份变化。

3. block exclude后的soft count变化不是直接删除事件数。
   消融改变状态后，后续soft候选也会重排；
   所有block delta都是条件边际效应，不能直接相加。
```

### 18.2 Exact-key ablation 已验证

```text
scripts/dmm_base_tracker_trusttrack_soft_key_ablation.py
```

精确排除：

```text
(1808, track483, det221523)
```

得到MD5：

```text
79ddb078569378a6c6df6dd5ceb596da
```

与整帧exclude-1808逐字节一致。

### 18.3 MOT20-03 严格双键协同

```text
K1 = (1808, 483, 221523)
K2 = (1809, 483, 221696)
```

实际关联：

```text
K1:
  rank3
  pair margin -0.08183
  raw IoU cost 0.33340
  embedding cost 0.19695
  alpha 0.995
  detection direct-best GT385，legacy output-state GT678

K2:
  rank1
  pair margin +0.03849
  raw IoU cost 0.06816
  alpha 0.98893
  direct/output GT385
```

必要性：

```text
从full soft排除K1或K2，输出逐字节相同；
相对full soft：HOTA +0.126、AssA +0.237、IDF1 +0.172、IDSW -3。
```

充分性：

```text
only K1 -> exact baseline
only K2 -> exact baseline
only K1+K2 -> 相对baseline仅在1813~2133连续321帧分叉
```

only-both指标：

```text
HOTA 79.929
AssA 79.003
IDF1 93.996
IDSW 181
```

结论：

```text
每个键单独无输出效应；
两键组合对长期分叉既必要又充分。
```

### 18.4 MOT20-02 精确正向链

A：track258双键：

```text
(1704,258,85536)
(1708,258,85823)
```

联合删除逐字节复现exclude1701~1712：

```text
HOTA -0.103
AssA -0.195
IDF1 -0.315
IDSW +2
```

B：单键：

```text
(1763,250,90017)
```

单键删除逐字节复现exclude1763~1769：

```text
HOTA -0.059
AssA -0.080
IDF1 -0.054
IDSW +2
```

C：跨轨迹三键：

```text
(1933,461,104523)
(1934,380,104614)
(1935,380,104699)
```

联合删除逐字节复现exclude1932~1937：

```text
HOTA -0.082
AssA -0.109
IDF1 -0.035
IDSW +2
```

以下直觉事件均被exact-key消融否定为最终贡献来源：

```text
track377@1768/1769
track291@1767
track307@1933/1934
track289 / track388@1932~1937
```

### 18.5 GT后标注偏差

当前chosen-only detection Hungarian与direct-best不同：

```text
M02约0.47%
M03约0.20%
```

与同stage全检测候选Hungarian不同：

```text
M02约0.21%
M03约0.10%
```

后续annotation v3必须同时记录：

```text
legacy chosen-only GT
direct-best GT + top1/top2/gap
stage-all-detection Hungarian GT
multi-GT overlap count
output TrackEval-aligned GT
```

### 18.6 跨运行轨迹对齐

用所有共同 `(frame, det_global_idx)` 建立soft->baseline多数映射：

```text
M02 mapping weighted purity 99.17%
    intervention aligned same 97.31%
    high-confidence aligned same 99.07%

M03 mapping weighted purity 99.73%
    intervention aligned same 95.45%
    high-confidence aligned same 97.86%
```

旧 `same_baseline_track` / `different_track` 字段不再作为证据。

### 18.7 Commit override 候选

无GT在线可计算条件：

```text
同一track上一帧实际执行soft；
当前也计划soft；
primary actual chosen：
  gap=1
  rank=1
  pair margin>=0.02
  raw IoU cost<=0.10
  det score>=0.60

=> 当前取消soft，恢复alpha=0.9。
```

离线exact-key replay：

```text
M02:
  requested/hit 233/233
  HOTA 68.949
  AssA 60.377
  IDF1 75.610
  IDSW 423
  基本保留full-soft正收益。

M03:
  requested/hit 31/30
  HOTA 80.073
  AssA 79.271
  IDF1 94.187
  IDSW 180
  相对baseline HOTA +0.019、AssA +0.030、IDF1 +0.018。
```

02+03 combined相对baseline：

```text
HOTA +0.073
DetA +0.015
AssA +0.125
MOTA +0.012
IDF1 +0.145
IDSW -3
FN -41
FP -10
Frag -3
```

证据边界：

```text
这是基于full-soft路径选择keys的离线replay，
不是在线动态方法结果；
而且规则在看过M03失败后设计，02/03均属于开发序列。
```

### 18.8 最新下一步

```text
Priority 1:
  新建 scripts/dmm_base_tracker_trusttrack_commit_v1.py；
  在真实primary Hungarian后动态执行commit override；
  不修改核心baseline。

Priority 2:
  M02/M03在线运行，硬检查：
  - override来自actual chosen pair；
  - planned/applied一一对应；
  - 与离线233/31 keys比较交并集；
  - 若在线结果不能接近离线矩阵，停止分析路径差异，不立即调参。

Priority 3:
  annotation v3：direct-best / stage-all-Hungarian / multi-GT overlap。

Priority 4:
  intervention summary加入trajectory alignment，废止raw ID比较。

Priority 5:
  对关键事件保存smooth_feat before/after hash与cosine，
  直接验证stale/intermediate prototype假设。

Priority 6:
  阈值冻结后先跑MOT20-01，再跑MOT20-05；
  二者称locked evaluation，不调参。

Priority 7:
  补temporal-block与identity-cluster bootstrap置信区间。
```

### 18.9 最新一句话状态

```text
TrustTrack的核心问题已从“单帧ambiguity是否高”推进到：
连续soft action如何与真实association commitment发生时序交互。

M03存在严格的双键协同负效应；
M02正收益来自少量精确、非线性的action链；
commit override是第一个在离线replay中让02不退、03转正的候选，
但下一步必须完成无GT在线实现并在冻结后跑01/05。
```

---

## 19. 在线 Commit v1 与 Locked Evaluation 最终结论

> **本节优先级高于第18节。**
>
> 详细报告：`docs/TRUSTTRACK_DEEP_REVIEW_V4_20260711.md`

### 19.1 在线方法已完成

```text
scripts/dmm_base_tracker_trusttrack_commit_v1.py
```

冻结规则：

```text
上一帧同track实际执行soft；
当前真实primary Hungarian match：
  rank=1
  pair margin>=0.02
  raw IoU cost<=0.10
  det score>=0.60
=> 当前取消soft，恢复alpha=0.9。
```

不读取GT，不改变当前assignment，只控制当前memory update。

### 19.2 机械验证

```text
- commit关闭逐字节复现原soft；
- planned override == actual normal update；
- online actual-key replay逐字节一致；
- callback与当前soft-observe在rank/margin/IoU/embedding/alpha上零误差；
- M02/M03/M01/M05均有确定性或独立重复验证。
```

M03 K1/K2定点：

```text
K1=(1808,483,221523) 保留soft；
K2=(1809,483,221696) 被在线override；
online prefix1820逐字节等于历史exclude-K2修复分支。
```

### 19.3 Development 02/03

M02 online：

```text
185 overrides
HOTA 68.949
AssA 60.377
IDF1 75.610
IDSW 423
vs baseline: HOTA +0.201, AssA +0.324, IDF1 +0.405, IDSW -5
```

M03 online：

```text
28 overrides
HOTA 80.073
AssA 79.271
IDF1 94.187
IDSW 180
vs baseline: HOTA +0.019, AssA +0.030, IDF1 +0.018
vs soft: HOTA +0.130, AssA +0.242, IDF1 +0.172, IDSW -6
```

02+03 combined相对baseline：

```text
HOTA +0.073
AssA +0.125
IDF1 +0.145
IDSW -3
```

全部预注册条件通过，阈值随后冻结。

### 19.4 Locked MOT20-01

```text
baseline MD5 3d062bf5846b1873699b46b89fbcf0dc
soft/commit MD5 bedd4773445a429799793b7dbd8e467e
```

32次override，但commit输出逐字节等于soft。

相对baseline：

```text
HOTA +0.225
AssA +0.312
IDF1 +0.117
IDSW 0
```

结论：soft在01正向；commit安全但output-level no-op。

### 19.5 Locked MOT20-05 反证

两套独立全量结果完全一致：

```text
baseline MD5 a26cc8314215801d5aeb1417f7942c48
soft MD5     4a5e05b8d48d17eafe61a57069bb511b
commit MD5   2c5429e5f49c5fac54f23e1b52ebc7ab
```

117次override，planned/actual一致。

Commit相对baseline：

```text
HOTA -0.124
AssA -0.223
IDF1 -0.218
IDSW +21
```

Commit只比soft改善约HOTA +0.001。

01+05 locked combined相对baseline：

```text
HOTA -0.115
AssA -0.206
IDF1 -0.208
IDSW +21
```

Locked generalization失败。

### 19.6 四序列总体

Commit相对soft：

```text
HOTA +0.037
AssA +0.067
IDF1 +0.048
IDSW -6
```

Commit相对baseline：

```text
HOTA -0.038
AssA -0.071
IDF1 -0.063
IDSW +18
```

结论：commit v1是soft v1的有效局部修补，但不是最终跨序列方法。

### 19.7 MOT20-05 首个负键

首次输出分叉：frame93。

```text
K05=(92,148,20880)
```

实际关联：

```text
rank1
pair margin +0.04217
embedding cost 0.11212
raw IoU cost 0.35066
alpha 0.98867
```

只保留K05：

```text
前100帧逐字节复现原soft的frame93起分叉；
全序列相对baseline HOTA -0.010、IDSW +1。
```

M05总损失还有多条后续负链；K05不是唯一根因。

### 19.8 最新机制判断

```text
M03：连续两帧soft + 下一帧强commit，导致memory无法跟随；
M05：间隔多帧的孤立/稀疏soft，在关键适应点也可能阻止更新并触发新ID。
```

Commit v1只处理前一种，无法处理后一种。

### 19.9 下一步

停止调commit阈值，冻结V1。

优先实现只读 memory snapshot wrapper，记录预注册关键键：

```text
M02:
(1704,258,85536)
(1708,258,85823)
(1763,250,90017)
(1933,461,104523)
(1934,380,104614)
(1935,380,104699)

M03:
(1808,483,221523)
(1809,483,221696)

M05:
(92,148,20880)
```

每次update记录：

```text
smooth_feat before/after hash
current_feat hash
alpha used
cos(before,current)
cos(after,current)
cos(before,after)
feature history length与recent cosine分布
下一帧actual association rank/cost/margin
```

硬要求：

```text
snapshot关闭/开启均逐字节复现对应baseline/soft输出。
```

若后续使用M05设计新动作，M05不再是locked sequence，必须重新定义外部验证协议。

