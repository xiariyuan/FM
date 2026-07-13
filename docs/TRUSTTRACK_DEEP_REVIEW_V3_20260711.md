# TrustTrack 第三轮证据链审计与下一步

> 日期：2026-07-11
>
> 本报告继续审查 `TRUSTTRACK_DEEP_REVIEW_V2_20260710.md` 的标签、跨运行比较、因果归因和 action 设计边界。
>
> **优先级：本报告高于 V2 报告及交接文档第 17 节。**

> **状态更新：在线 Commit v1、MOT20-01/05 locked evaluation 及 MOT20-05 反证已经完成。请以 `docs/TRUSTTRACK_DEEP_REVIEW_V4_20260711.md` 为准。**
>
---

## 1. 最新结论摘要

本轮新增的最重要结论有五项：

```text
1. MOT20-03 的主要负效应由两个精确 soft key 协同产生：
   (1808, track483, det221523)
   (1809, track483, det221696)

   任一键单独 soft：输出等于 baseline；
   两键同时 soft：足以单独制造 frame 1813~2133 连续 321 帧分叉。

2. frame 1808 不能继续描述为“正确 GT678 同身份更新”。
   chosen-only Hungarian 标为 GT678，
   但该检测 direct-best valid GT 是 GT385，IoU 0.8195。
   它处于多个 GT 与两个检测严重重叠的 one-to-one 冲突中。

3. MOT20-02 的 +0.201 HOTA 主要来自少量精确 action 链，
   而不是 1716 次更新的平均效果。

4. 跨运行直接比较 raw tracker ID 是错误的。
   M02 soft 与 baseline 最常见只是整体 ID 编号偏移 +3；
   旧的 different_track=1395 不能解释为 1395 次身份改变。

5. 一个无 GT、在线可计算的 commit override 候选，在离线 exact-key replay 中：
   - 保留 M02 soft 正收益；
   - 将 M03 从负收益转为小幅正收益；
   - 02+03 combined 相对 baseline：HOTA +0.073、AssA +0.125、IDF1 +0.145、IDSW -3。

   但它尚未被实现为在线动态方法，不能作为最终结果。
```

---

## 2. 已完成的正式基础设施

### 2.1 Oracle v2

```text
scripts/current_output_id_oracle_v2.py
```

已经包含：

```text
- threshold-before-Hungarian；
- MOTChallenge / TrackEval distractor preprocessing；
- frame / box / score / tail 非 ID 一致性检查；
- diagnostic-only 与非严格 HOTA ceiling 元数据。
```

MOT20-01/02/03 生成输出与此前修正参考逐字节一致：

```text
MOT20-01 MD5 = 8e09587ef735cc757f5a494ae3744fb4
MOT20-02 MD5 = 8ea315684be38394e9ceaa81d99005ce
MOT20-03 MD5 = 4873b7f1fab31c28e6b49a2b11813e7b
```

全部：

```text
non-ID mismatch = 0
duplicate ID frames = 0
```

### 2.2 Association-aligned observe v2

```text
scripts/dmm_base_tracker_trusttrack_observe_v2.py
scripts/dmm_base_tracker_trusttrack_soft_observe_v2.py
```

它记录真实：

```text
GMC/Kalman predict
-> baseline assoc_cost + masks/fusion
-> Hungarian actual match
-> chosen-specific signed margin
```

Baseline no-op 验证：

```text
MOT20-02:
  output MD5 exact baseline
  chosen total/logged = 146061 / 146061
  rank > 1 = 329
  max chosen rank = 4
  actual pair margin < 0 = 733

MOT20-03:
  output MD5 exact baseline
  chosen total/logged = 308103 / 308103
  rank > 1 = 74
  max chosen rank = 3
  actual pair margin < 0 = 192
```

因此旧 logger 的两个缺陷已被实证确认：

```text
- top-3 可能漏掉 rank=4 的真实 Hungarian match；
- 全局 best-second margin 不是 actual chosen pair 的 signed margin。
```

### 2.3 精确键消融

```text
scripts/dmm_base_tracker_trusttrack_soft_key_ablation.py
```

已正式通过运行验证。

排除精确键：

```text
(1808, 483, 221523)
```

得到 MD5：

```text
79ddb078569378a6c6df6dd5ceb596da
```

与整帧 exclude-1808 输出逐字节一致。

注意：

```text
当前 selected_key_hits 是 soft map 命中数；
它不是形式上严格的“实际 DMMTrack.update 命中数”。
后续正式工具仍应增加 unique actual update hit / missing key 统计。
```

---

## 3. Observe GT 标签仍有竞争偏差

### 3.1 当前标注方式

`annotate_trusttrack_observe_v2.py` 当前将每帧已选中的检测集合，与 GT 做 Hungarian。

问题是：

```text
未被 chosen 的检测没有参与 GT 竞争；
多个 chosen detection 重叠同一个 GT 时，
one-to-one Hungarian 可能把次优检测分给另一个 GT。
```

### 3.2 全量偏差审计

与 direct-best valid pedestrian GT 比较：

```text
MOT20-02:
  current known = 143739
  direct-best label difference = 673  (约 0.47%)

MOT20-03:
  current known = 302801
  direct-best label difference = 597  (约 0.20%)
```

与同 stage 全检测候选 Hungarian 比较：

```text
MOT20-02 difference = 301  (约 0.21%)
MOT20-03 difference = 296  (约 0.10%)
```

整体比例不高，但高拥挤事件中可能恰好落在关键样本上。

### 3.3 frame 1808 的重大修正

旧标注：

```text
frame 1808
track483 / det221523
chosen-only GT = 678
track_history = same identity
```

重新审查该检测与有效 GT 的 IoU：

```text
GT385  IoU = 0.8195
GT392  IoU = 0.5548
GT678  IoU = 0.5336
GT389  IoU = 0.2576
```

同帧另一个检测 `det221530` 也 direct-best GT385：

```text
GT385 IoU = 0.8240
GT678 IoU = 0.4171
```

因此 frame 1808 是：

```text
两个检测、多个高度重叠 GT、one-to-one assignment 与 tracker state 冲突的拥挤事件。
```

不能再表述为：

```text
soft 抑制了一次干净的正确 GT678 同身份更新。
```

后续 annotation v3 必须同时输出：

```text
legacy chosen-only Hungarian GT
direct-best GT + IoU top1/top2/gap
stage-all-detection Hungarian GT
multi-GT overlap count
```

---

## 4. 跨运行 raw track ID 比较必须废止

此前 `intervention_vs_baseline.csv` 直接比较：

```text
soft track_id == baseline track_id
```

这是不可靠的，因为软干预改变了创建/删除顺序，后续 ID 编号会整体平移。

### 4.1 MOT20-02

```text
1712 个可比较事件中：
raw ID equal rate = 18.75%
```

最常见编号差：

```text
baseline_id - soft_id = +3 : 863 次
+6 : 212 次
+2 : 142 次
```

因此旧的：

```text
different_track = 1395
```

绝大部分只是编号漂移，不是 1395 次身份改变。

### 4.2 共现映射修正

使用所有 `(frame, det_global_idx)` 共现建立 soft track -> baseline track 多数映射：

```text
MOT20-02:
  shared detection keys = 146053
  mapping weighted purity = 99.17%
  intervention aligned same rate = 97.31%
  high-confidence aligned same rate = 99.07%

MOT20-03:
  shared detection keys = 308100
  mapping weighted purity = 99.73%
  intervention aligned same rate = 95.45%
  high-confidence aligned same rate = 97.86%
```

结论：

```text
跨运行比较必须先做轨迹对齐；
raw same_track / different_track 字段不再作为证据。
```

相关输出：

```text
outputs/trusttrack_review_20260710/cross_run_raw_track_id_audit.json
outputs/trusttrack_review_20260710/cross_run_track_alignment_audit.json
```

---

## 5. MOT20-03：严格双键协同负效应

### 5.1 两个精确键

```text
K1 = (frame 1808, track 483, det 221523)
K2 = (frame 1809, track 483, det 221696)
```

K1 实际关联数据：

```text
chosen rank = 3
final cost = 0.19695
raw IoU cost = 0.33340
embedding cost = 0.19695
chosen-specific pair margin = -0.08183
smooth-current cosine = 0.60610
soft alpha = 0.995

detection direct-best GT385 IoU = 0.8195
legacy output-state assignment GT678
```

K2：

```text
chosen rank = 1
final/raw IoU cost = 0.06816  (IoU约0.932)
embedding cost = 0.18340
chosen-specific pair margin = +0.03849
smooth-current cosine = 0.63321
soft alpha = 0.98893

detection direct/output GT = 385
```

### 5.2 必要性

从 full soft 排除 K1 或排除 K2：

```text
两者输出逐字节完全相同：
MD5 = 79ddb078569378a6c6df6dd5ceb596da
```

相对 full soft：

```text
HOTA +0.126
AssA +0.237
IDF1 +0.172
IDSW -3
```

两个排除分支都从 frame 1813 开始与 full soft 分叉，连续到 2133：

```text
1813~2133，共321帧
```

### 5.3 充分性

只启用 K1、其余所有 soft 关闭：

```text
输出逐字节等于 baseline
```

只启用 K2：

```text
输出逐字节等于 baseline
```

只启用 K1 + K2：

```text
相对 baseline 仅在 1813~2133 连续321帧分叉
```

指标：

```text
HOTA  = 79.929
DetA  = 80.900
AssA  = 79.003
MOTA  = 94.307
IDF1  = 93.996
IDSW  = 181
FN    = 13987
FP    = 3689
Frag  = 489
```

因此：

```text
K1、K2 单独均无输出效应；
两键组合对该长期分叉既必要又充分。
```

这是严格的二键协同，而不是可相加的两个单事件效果。

### 5.4 当前可支持的机制表述

可以支持：

```text
K1 是高冲突、rank3、负margin的全局 assignment 事件；
K2 是同一track下一帧的强几何、rank1、正margin commit；
连续两帧均使用高alpha抑制更新，足以触发长期轨迹状态分叉。
```

不能直接声称：

```text
prototype 已被证明变成旧身份与新身份之间的中间向量。
```

因为目前没有保存 K1/K2 前后 `smooth_feat` 完整向量或 hash。

“stale/intermediate prototype”仍是被输出因果支持的机制推断，不是直接观测事实。

---

## 6. MOT20-02：精确正向 action 链

M02 +0.201 HOTA 并非均匀来自1716次更新。

分块与精确键消融已定位以下三组主要条件效应。

### 6.1 正向链 A：track258 双键冗余/协同

```text
A1 = (1704, track258, det85536)
A2 = (1708, track258, det85823)
```

精确排除 A1+A2，逐字节复现删除整个 1701~1712 的结果：

```text
MD5 = 80fba2163eb9a91468d123def62e10ec
```

相对 full soft：

```text
HOTA -0.103
AssA -0.195
IDF1 -0.315
IDSW +2
```

但：

```text
排除1701~1706：输出等于full soft
排除1707~1712：输出等于full soft
```

因此更严谨的表述是：

```text
A1/A2 形成输出级冗余或协同链；
单独移除任一半段可被另一段或下游状态补偿，
同时移除才失去正收益。
```

不能仅凭此证明 memory 内部具有数学冗余。

A1 特征：

```text
rank1
pair margin +0.0217
IoU cost 0.4522
embedding cost 0.1047
alpha 0.9920
```

A2：

```text
rank2
pair margin -0.00467
IoU cost 0.3795
embedding cost 0.0739
alpha 0.995
```

### 6.2 正向链 B：单一精确键

```text
B = (1763, track250, det90017)
```

排除该单键，逐字节复现删除整个1763~1769窗口：

```text
MD5 = 17344fd4849f15c96b1dac9252a887ef
```

相对 full soft：

```text
HOTA -0.059
AssA -0.080
IDF1 -0.054
IDSW +2
```

特征：

```text
rank2
pair margin -0.1083
IoU cost 0.4604
embedding cost 0.1335
alpha 0.995

detection direct GT34 IoU 0.5557
second GT31 IoU 0.5409
top1-top2 gap仅0.0147
```

它是高度拥挤、几何身份近乎平局的关键保护事件。

### 6.3 正向链 C：跨轨迹三键交互

```text
C1 = (1933, track461, det104523)
C2 = (1934, track380, det104614)
C3 = (1935, track380, det104699)
```

排除 C1+C2+C3，逐字节复现删除整个1932~1937：

```text
MD5 = 7934f8f3d483324c916df4d377e2408c
```

组合相对 full soft：

```text
HOTA -0.082
AssA -0.109
IDF1 -0.035
IDSW +2
```

单独贡献：

```text
C1 alone:
  HOTA -0.064
  AssA -0.083
  IDF1 -0.041
  IDSW +1

C2+C3 alone:
  HOTA -0.018
  AssA -0.026
  IDF1 +0.005
  IDSW +1
```

组合明显非线性。

旧猜测 `track307 @ 1933/1934` 被精确消融否定：

```text
删除后输出完全等于full soft。
```

同样被否定的直觉猜测包括：

```text
track377 @1768/1769
track291 @1767
track289 / track388 @1932~1937
```

这些事件看起来有 cross identity 或负margin，但删除后输出为 full-soft no-op。

结论：

```text
不能按“看起来像错误关联”的单事件猜最终贡献；
必须做 exact-key / grouped counterfactual。
```

---

## 7. 分块消融的正确解释

此前文档中使用：

```text
excluded_soft_updates = full_soft_count - remaining_soft_count
```

该字段容易被误读。

因为消融改变状态后，后续 soft map 本身也会变化：

```text
排除一个直接key，可能连带使若干后续key消失或新出现。
```

所以正确称呼应是：

```text
remaining soft update count delta
```

而不是直接删除事件数。

同理：

```text
leave-one-block-out delta 是条件边际效应，
不是可加和的独立贡献，也不是 Shapley value。
```

正向链 A/B/C 的 HOTA delta 不应直接相加解释 full soft +0.201。

---

## 8. 实际关联风险信号：强，但尚未可直接部署

真实 chosen-specific margin / cost 的跨序列 AUC 很高。

GT→tracker changed：

```text
train02 -> test03:
  costs only             AUC 0.946
  chosen margins only    AUC 0.950
  costs + margins        AUC 0.976

train03 -> test02:
  costs only             AUC 0.921
  chosen margins only    AUC 0.901
  costs + margins        AUC 0.955
```

但正样本极稀少，top-10% precision 仍低。例如 M03：

```text
GT changed base rate约0.07%
top-10% precision约0.63%
```

因此：

```text
高AUC证明排序价值，
不等于已经得到高精度可执行gate。
```

另外 `gt_changed` 只是 GT→tracker transition：

```text
它可能包含断轨后重新初始化等合法transition；
不能把每个changed都等同于HOTA错误。
```

仍需：

```text
identity-cluster / temporal-block bootstrap
分stage置信区间
事件级precision/recall与最终序列counterfactual同时报告
```

---

## 9. Commit override：当前最有价值的 action 线索

### 9.1 无GT规则

在 full-soft 的真实 online-observable日志中定义：

```text
如果同一track上一帧刚实际执行soft，
当前帧也计划soft，并且真实primary association满足：

- gap = 1
- chosen rank = 1
- chosen-specific pair margin >= 0.02
- raw IoU cost <= 0.10  （IoU >= 0.90）
- detection score >= 0.60

则当前帧取消soft，使用baseline alpha=0.9正常更新。
```

该规则：

```text
命中 M03 K2（frame1809）；
不命中任何已确认 M02 正向 exact key。
```

离线扫描覆盖：

```text
MOT20-02: 233 keys
MOT20-03: 31 requested keys
```

### 9.2 Exact-key replay：MOT20-02

请求并命中233键，剩余soft更新1482。

结果：

```text
HOTA  68.949
DetA  78.876
AssA  60.377
MOTA  90.787
IDF1  75.610
IDSW  423
FN    11514
FP    2320
Frag  760
```

相对 full soft：

```text
仅 AssA -0.001，其余主指标同显示精度一致。
```

因此该候选在M02没有破坏已确认正向链。

### 9.3 Exact-key replay：MOT20-03

请求31键，路径变化后实际命中30键，剩余soft更新497。

结果：

```text
HOTA  80.073
DetA  80.917
AssA  79.271
MOTA  94.305
IDF1  94.187
IDSW  180
FN    13994
FP    3688
Frag  488
```

相对 baseline：

```text
HOTA +0.019
AssA +0.030
IDF1 +0.018
IDSW +2
FN +1
FP -1
Frag +2
```

相对 full soft：

```text
HOTA +0.130
AssA +0.242
IDF1 +0.172
IDSW -6
```

### 9.4 02+03 combined

```text
HOTA  = 76.617
DetA  = 80.224
AssA  = 73.229
MOTA  = 93.143
IDF1  = 88.106
IDSW  = 603
FN    = 25508
FP    = 6008
Frag  = 1248
```

相对 combined baseline：

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

相对原 full soft：

```text
HOTA +0.091
AssA +0.164
IDF1 +0.116
IDSW -6
```

### 9.5 必须保留的证据边界

这些仍然是：

```text
基于 full-soft 已发生路径选择 exact keys 的离线 replay。
```

它不是：

```text
在线动态方法运行结果。
```

原因：

```text
在线取消某个key会改变后续track状态、actual association和soft候选；
M03的31个请求键已经只命中30个，说明路径依赖真实存在。
```

此外规则是在观察M03失败后设计，M02/03现在都属于开发序列。

不能宣称泛化。

---

## 10. 对当前研究过程的反思

### 10.1 首次分叉、局部标签和最终贡献是三件不同的事

此前多次犯过：

```text
首次输出分叉 -> 推断事件最终好坏
```

正确标准是：

```text
first divergence：证明局部因果起点
full-sequence exact-key counterfactual：证明最终条件效应
only-key sufficiency：证明某key组合是否足以产生结果
```

### 10.2 单轨 majority GT 仍不足

在严重拥挤中：

```text
检测可能同时高IoU重叠多个GT；
one-to-one assignment会把它分配给非direct-best GT；
输出state GT与检测几何direct GT也可能不同。
```

后续必须同时观察：

```text
track state identity
chosen detection direct geometry identity
frame-level one-to-one assignment identity
GT-centric output transition
```

### 10.3 相关特征不能直接变成action

真实margin具有高AUC，但 M03 K1/K2 说明：

```text
第一帧负margin的soft本身无输出效应；
第二帧强正margin的soft本身也无输出效应；
两帧组合才有害。
```

所以 action 必须建模时序状态，而不是单帧分类。

### 10.4 跨运行ID必须先对齐

raw tracker ID 比较会制造大规模伪差异。

任何未来 baseline-vs-variant event join 必须使用：

```text
共同 detection key
或 trajectory overlap / GT mapping
或显式 bipartite trajectory alignment
```

### 10.5 机制解释应晚于观测

目前直接观测到的是：

```text
association cost / margin
soft alpha
output divergence
GT/output assignment
```

没有直接观测：

```text
完整 smooth_feat 前后向量
prototype方向
feature deque污染程度
```

所以“stale/intermediate prototype”应明确标注为 inference。

---

## 11. 更新后的下一步

### Priority 0：冻结开发结论

停止：

```text
soft_start / alpha sweep
binary freeze sweep
旧四cue weighted-sum设计
appearance-veto v1 sweep
```

M02/03 现在均视为开发集。

### Priority 1：实现在线 commit-aware v1

建议新增：

```text
scripts/dmm_base_tracker_trusttrack_commit_v1.py
```

不能修改稳定核心 `scripts/dmm_base_tracker.py`。

实现原则：

```text
1. 先按soft v1计算planned soft map；
2. 在真实 primary assoc_cost + Hungarian 后读取 actual chosen pair；
3. 记录每个track上一帧是否实际执行soft；
4. 满足commit override时，从当前soft map移除该key；
5. 当前DMMTrack.update使用正常alpha=0.9；
6. 记录override planned/applied、actual rank/margin/IoU cost。
```

第一版固定阈值：

```text
gap = 1
rank = 1
pair margin >= 0.02
raw IoU cost <= 0.10
det score >= 0.60
```

不要在02/03结果出来后继续调阈值。

### Priority 2：严格在线复现检查

在M02/M03运行：

```text
same-script soft baseline
commit-aware v1
observe-v2 logs
TrackEval
```

硬检查：

```text
- no override时输出 exact soft MD5；
- override key必须来自实际chosen primary pair；
- planned override / actual normal update一一对应；
- 记录在线override keys与离线233/31 keys的交并集；
- 若在线指标不能接近离线正向矩阵，停止并分析路径差异，不立即调参。
```

### Priority 3：Annotation v3

新增或升级：

```text
scripts/annotate_trusttrack_observe_v3.py
```

字段：

```text
det_gt_legacy_chosen_hungarian
det_gt_direct_best
det_direct_iou_top1 / top2 / gap
det_num_gt_iou_ge_03 / ge_05
det_gt_stage_all_hungarian
output_gt_trackeval_aligned
```

旧 `det_gt` 字段不应继续作为唯一真值。

### Priority 4：跨运行轨迹对齐

升级 intervention summary：

```text
same_raw_tracker_id
aligned_baseline_tracker_id
alignment_support
alignment_purity
same_aligned_trajectory
```

旧 `same_baseline_track` 重命名并标记 deprecated。

### Priority 5：Memory snapshot

对开发序列关键事件保存：

```text
smooth_feat before / after hash
current feature hash
cosine(old,new)
cosine(prototype_before,current)
cosine(prototype_after,current)
features deque length与最近若干feature cosine
```

目的是直接验证：

```text
K1/K2 是否形成stale/intermediate prototype；
正向A/B/C键如何改变identity memory方向。
```

### Priority 6：冻结后运行 locked evaluation

在线 commit v1 在02/03固定后：

```text
先 MOT20-01
再 MOT20-05
```

01/05称：

```text
locked evaluation sequences
```

而不是 pristine holdout。

看到01/05结果后不得回调参数。

若调参，则必须重新声明新的验证协议。

### Priority 7：统计可靠性

补充：

```text
temporal-block bootstrap
identity-cluster bootstrap
per-stage AUC/AP/confidence interval
事件级precision/recall
序列级counterfactual delta
```

不能只报告高AUC。

---

## 12. 最新可信状态

```text
1. actual chosen-specific cost/margin 是可靠的identity-risk排序信号，
   但单帧风险不足以决定memory action。

2. M03负效应由K1/K2严格双键协同产生：
   每键单独无效，两键组合必要且充分地制造1813~2133长期分叉。

3. frame1808不是干净GT678同身份事件，而是严重拥挤的一对多/多对一竞争事件。

4. M02正收益由少量精确action链产生：
   track258双键、track250单键、track461+track380三键交互。

5. 跨运行raw tracker ID比较已废止，必须先做trajectory alignment。

6. commit override是目前第一个同时让02不退、03转正的action线索：
   exact replay下02+03 combined HOTA +0.073、AssA +0.125、IDF1 +0.145、IDSW -3。

7. 该结果仍是路径依赖的离线exact-key replay，下一步必须实现在线commit-aware v1，
   冻结后再运行01/05 locked evaluation。
```
