# TrustTrack 第四轮：在线 Commit v1、Locked Evaluation 与 MOT20-05 反证

> 日期：2026-07-11
>
> **优先级：本报告高于 V3、V2 及交接文档第 18 节。**
>
> 本报告记录真正在线的 commit-aware v1、开发序列 MOT20-02/03、冻结后的 MOT20-01/05 locked evaluation，以及 MOT20-05 对当前机制假设的反证。

---

## 1. 最终结论摘要

```text
1. commit-aware v1 已被真正在线实现，不读取GT，不改变当前Hungarian，
   只在真实primary assignment之后、DMMTrack.update之前决定是否取消soft alpha。

2. 机械正确性已通过：
   - commit关闭逐字节复现原soft；
   - planned override与actual normal update一一对应；
   - 在线actual-key replay逐字节一致；
   - M02/M03/M01/M05均有确定性或独立重复验证。

3. 开发序列02/03通过预注册标准：
   combined相对baseline：
   HOTA +0.073，AssA +0.125，IDF1 +0.145，IDSW -3。

4. 冻结后的MOT20-01为安全正向：
   commit输出逐字节等于soft；相对baseline HOTA +0.225。

5. 冻结后的MOT20-05构成明确反证：
   commit相对baseline HOTA -0.124，AssA -0.223，IDF1 -0.218，IDSW +21。
   commit仅比原soft改善约0.001 HOTA，基本无修复作用。

6. 01+05 locked combined为负：
   HOTA -0.115，AssA -0.206，IDF1 -0.208，IDSW +21。

7. 四序列combined中，commit优于原soft，但仍弱于baseline：
   commit vs soft：HOTA +0.037，AssA +0.067，IDF1 +0.048，IDSW -6；
   commit vs baseline：HOTA -0.038，AssA -0.071，IDF1 -0.063，IDSW +18。

8. 因此commit v1是对soft v1的有效局部修补，不是跨序列成立的最终方法。
   不能只报告02/03，也不能在看过05后继续回调阈值。
```

---

## 2. 在线 Commit v1 实现

脚本：

```text
scripts/dmm_base_tracker_trusttrack_commit_v1.py
```

冻结规则：

```text
同一track上一帧实际执行过soft；
当前也计划soft；
当前真实primary Hungarian chosen pair满足：

- gap = 1
- chosen rank = 1
- chosen-specific pair margin >= 0.02
- raw IoU cost <= 0.10
- det score >= 0.60

=> 当前从soft map移除该key，DMMTrack.update使用baseline alpha=0.9。
```

执行顺序已从恢复源码确认：

```text
GMC/Kalman predict
-> real assoc_cost
-> Hungarian
-> _record_assoc_debug callback
-> track.update / re_activate
```

因此 callback 删除 soft key：

```text
不改变当前assignment；
只改变当前匹配对的ReID memory update强度。
```

方法定位应写为：

```text
association-commit-aware memory update control
```

而不是 association correction。

---

## 3. 机械正确性与数值验证

### 3.1 MOT20-02 100帧 smoke

```text
original soft MD5 = 6e2f3081d72417bf0bf9e02caa911a0b
commit-off MD5    = 6e2f3081d72417bf0bf9e02caa911a0b
commit-on MD5     = 6e2f3081d72417bf0bf9e02caa911a0b
```

commit-on：

```text
planned override = 5
actual override  = 5
soft updates     = 28 -> 23
missing update   = 0
feature missing  = 0
soft key residual= 0
```

在线实际5键的 exact-key replay：

```text
online output == replay output byte-for-byte
```

### 3.2 重复运行确定性

M02 100帧 commit-on 两次：

```text
output MD5 exact
commit CSV exact
override keys exact
stats exact
```

### 3.3 Callback 数值口径

与当前代码重新生成的 soft-observe-v2 比较：

```text
28个事件key集合完全一致
chosen rank最大误差 = 0
pair margin最大误差 = 0
raw IoU cost最大误差 = 0
embedding cost最大误差 = 0
soft alpha最大误差 = 0
```

此前与历史 `observe_chosen_gt.csv` 比较出现的 embedding/margin 微差，来自历史产物版本差异；历史CSV不再用于精确浮点对照。

### 3.4 离线规则扫描的修正

旧离线 key 选择把：

```text
上一帧计划soft
```

误当成：

```text
上一帧实际执行soft
```

例如 MOT20-03 track134：

```text
原soft路径：508/509/510/511连续soft
在线commit：
  508 soft
  509 override(normal)
  510 soft
  511 override(normal)
```

因此在线规则天然形成一帧迟滞/恢复节奏。

旧 `MOT20-03_commit_override_keys.json` 只能视为 retrospective key proposal，不能作为在线expected key列表。

---

## 4. MOT20-03 定点前缀验证

运行至 frame1820：

```text
original soft
commit-off
commit-on
online actual-key exact replay
```

结果：

```text
commit-off == original soft byte-for-byte
commit-on == exact replay byte-for-byte
```

关键键：

```text
K1 = (1808, 483, 221523)
K2 = (1809, 483, 221696)
```

K1：

```text
rank3
pair margin -0.08183
raw IoU cost 0.33340
未override
实际执行soft
```

K2：

```text
上一帧K1实际soft
rank1
pair margin +0.03849
raw IoU cost 0.06816
被override
实际使用normal alpha
```

commit-on 从 frame1813 开始与原soft分叉。

更强验证：

```text
commit-on prefix1820
== 历史 exclude-K2 修复分支 prefix1820 byte-for-byte
```

这证明在线规则真实切断了M03已知双键负链。

动态路径变化：

```text
14次直接override后，soft updates从263降到247；
并非简单263-14，因为3个原soft候选消失、1个新soft候选出现。
```

---

## 5. 开发序列 MOT20-02

### 5.1 Baseline/no-op

完整 commit-off：

```text
MD5 = 6c67a373d264f8fd4d744f92dbff24b9
```

与原 full soft 逐字节一致。

```text
soft updates = 1716
planned/actual equal = true
```

### 5.2 在线 commit

两次独立运行：

```text
MD5 = cbfe03e4111949b5efbc5022db5b31f8
output exact
commit CSV exact
185个override key exact
stats exact
```

```text
primary chosen soft pairs = 1715
actual override = 185
remaining actual soft updates = 1530
```

### 5.3 TrackEval

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

相对baseline：

```text
HOTA +0.201
AssA +0.324
IDF1 +0.405
IDSW -5
FN -42
FP -9
```

相对原soft：

```text
显示精度上HOTA/IDF1/IDSW相同；AssA -0.001。
```

M02通过预注册标准。

---

## 6. 开发序列 MOT20-03

### 6.1 Baseline/no-op

完整 commit-off：

```text
MD5 = 9ec3b40f3d006a3681c1c6de38bdb161
```

与原 full soft 逐字节一致。

```text
soft updates = 529
```

### 6.2 在线 commit

两次独立运行：

```text
MD5 = 2500e42a595a7397d3f0949d923ccc62
output exact
commit CSV exact
28个override key exact
stats exact
```

```text
primary chosen soft pairs = 527
actual override = 28
remaining actual soft updates = 499
```

### 6.3 TrackEval

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

相对baseline：

```text
HOTA +0.019
AssA +0.030
IDF1 +0.018
IDSW +2
```

相对原soft：

```text
HOTA +0.130
AssA +0.242
IDF1 +0.172
IDSW -6
```

M03的HOTA/AssA/IDF1均通过预注册标准；IDSW仍比baseline多2。

---

## 7. 02+03 Development Combined

```text
HOTA  76.617
DetA  80.224
AssA  73.229
MOTA  93.143
IDF1  88.106
IDSW  603
FN    25508
FP    6008
Frag  1248
```

相对baseline：

```text
HOTA +0.073
AssA +0.125
IDF1 +0.145
IDSW -3
FN -41
FP -10
Frag -3
```

全部预注册条件通过，随后阈值正式冻结。

---

## 8. Locked Evaluation：MOT20-01

同脚本现场生成：

```text
baseline MD5 = 3d062bf5846b1873699b46b89fbcf0dc
soft MD5     = bedd4773445a429799793b7dbd8e467e
commit MD5   = bedd4773445a429799793b7dbd8e467e
```

两次commit：

```text
output exact
commit CSV exact
32个override key exact
stats exact
```

```text
soft updates = 324
commit remaining soft = 293
override = 32
```

但：

```text
commit output == soft byte-for-byte
```

TrackEval：

```text
baseline:
  HOTA 77.030
  AssA 73.930
  IDF1 88.333
  IDSW 50

soft/commit:
  HOTA 77.255
  AssA 74.242
  IDF1 88.450
  IDSW 50
```

相对baseline：

```text
HOTA +0.225
AssA +0.312
IDF1 +0.117
IDSW 0
```

解释：

```text
M01支持soft v1本身；
commit v1安全，但32次override均为output-level no-op，未提供额外收益。
```

---

## 9. Locked Evaluation：MOT20-05

### 9.1 参考口径

必须使用 same-script no-recovery baseline，不能使用历史不一致baseline。

两套独立全量产物验证：

```text
baseline MD5 = a26cc8314215801d5aeb1417f7942c48
soft MD5     = 4a5e05b8d48d17eafe61a57069bb511b
commit MD5   = 2c5429e5f49c5fac54f23e1b52ebc7ab
```

两套产物：

```text
baseline exact
soft exact
commit exact
commit CSV exact
override keys exact
stats exact
```

### 9.2 Online action

```text
frames = 3315
original soft updates = 2034
primary chosen soft pairs on commit path = 2035
actual override = 117
remaining actual soft updates = 1918
planned/actual equal = true
```

### 9.3 TrackEval

Baseline：

```text
HOTA  78.539
DetA  81.427
AssA  75.802
MOTA  93.998
IDF1  90.304
IDSW  566
FN    25925
FP    12302
Frag  1285
```

原soft：

```text
HOTA  78.414
AssA  75.579
IDF1  90.085
IDSW  587
```

Commit：

```text
HOTA  78.415
DetA  81.409
AssA  75.579
MOTA  93.997
IDF1  90.086
IDSW  587
FN    25916
FP    12299
Frag  1291
```

Commit相对baseline：

```text
HOTA -0.124
AssA -0.223
IDF1 -0.218
IDSW +21
FN -9
FP -3
Frag +6
```

Commit相对soft：

```text
HOTA +0.001
IDF1 +0.001
IDSW 0
```

结论：

```text
M05的主要问题来自soft v1本体；
commit v1只修复连续强commit链，基本不影响M05主负链。
```

---

## 10. Locked Combined 与四序列总体

### 10.1 MOT20-01 + MOT20-05 locked combined

Commit相对baseline：

```text
HOTA -0.115
AssA -0.206
IDF1 -0.208
IDSW +21
FN -16
FP -5
```

Locked generalization失败。

### 10.2 MOT20-01/02/03/05 all-four combined

Commit：

```text
HOTA  77.661
DetA  80.905
AssA  74.601
MOTA  93.611
IDF1  89.245
IDSW  1240
```

Commit相对原soft：

```text
HOTA +0.037
AssA +0.067
IDF1 +0.048
IDSW -6
Frag -4
```

Commit相对baseline：

```text
HOTA -0.038
AssA -0.071
IDF1 -0.063
IDSW +18
```

最终定位：

```text
commit v1显著优于soft v1，证明temporal commit修补方向有效；
但soft v1整体仍未跨序列超过baseline。
```

---

## 11. MOT20-05 首次负链

Baseline vs soft/commit首次输出分叉：

```text
frame 93
```

GT-centric timeline：

```text
frame 92之前：
  GT171 -> track148（三者一致）

frame 93开始：
  baseline: GT171 -> track148
  soft:     GT171 -> new track291
  commit:   GT171 -> new track291
```

track148在此前的soft事件：

```text
frame68  det15637  alpha0.9821  rank1  pair margin0.8363  IoU cost0.3951
frame76  det17416  alpha0.9814  rank1  pair margin0.0595  IoU cost0.1464
frame92  det20880  alpha0.9887  rank1  pair margin0.0422  IoU cost0.3507
```

三次间隔为8帧和16帧，不属于连续soft链；且IoU cost均不满足commit的<=0.10条件。

### 11.1 frame92精确键

```text
K05 = (92, 148, 20880)
```

近似soft cue：

```text
collapse = 0.91565
app pair margin = +0.08435
motion pair margin = -0.01117
iou pair margin = -0.03672
shape pair margin = -0.08750
alpha = 0.98867
```

真实association：

```text
rank1
pair margin +0.04217
embedding cost 0.11212
raw IoU cost 0.35066
```

### 11.2 前100帧充分性

只保留K05、关闭其它soft：

```text
输出逐字节等于原soft前100帧；
frame93~100与baseline分叉。
```

因此K05单独足以制造首次ID分叉。

### 11.3 全序列作用

只保留K05：

```text
HOTA 78.529
AssA 75.784
IDF1 90.291
IDSW 567
```

相对baseline：

```text
HOTA -0.010
AssA -0.018
IDF1 -0.013
IDSW +1
```

所以K05是一个真实负事件，但只解释M05总损失的一小部分。

从full soft排除K05：

```text
HOTA 78.411
```

比full soft 78.414还低0.003，说明其它soft事件对K05造成的路径存在补偿/交互；leave-one-out不能作为独立可加贡献。

---

## 12. 对 Commit v1 的最终反思

### 12.1 成功之处

```text
- 真正在线、无GT；
- 严格不改变当前匹配；
- 切断M03双键负链；
- M02正收益基本保留；
- 02/03 combined由负转正；
- 四序列总体明显优于原soft。
```

### 12.2 失败之处

```text
- 只处理上一帧soft + 当前强commit；
- 无法处理间隔多帧的稀疏soft累积；
- 无法处理单次孤立soft本身就阻碍必要适应的情况；
- M05中117次override只在44帧改变soft输出，且未触及frame92首个负键；
- locked combined失败。
```

### 12.3 不能做的事

```text
- 不得在看过M05后继续微调0.02/0.10阈值并仍称其为locked泛化；
- 不得只报告02/03；
- 不得把四序列相对soft的改善写成相对baseline的成功；
- 不得把frame92单键当作M05唯一根因。
```

---

## 13. 下一步研究协议

Commit v1阶段结束，代码与结果冻结。

下一阶段不直接设计新阈值，先做 memory-state observation。

### Priority 1：Memory Snapshot Wrapper

新增只读诊断脚本，记录预注册关键键：

```text
M02正向：
  (1704,258,85536)
  (1708,258,85823)
  (1763,250,90017)
  (1933,461,104523)
  (1934,380,104614)
  (1935,380,104699)

M03负向双键：
  (1808,483,221523)
  (1809,483,221696)

M05负向：
  (92,148,20880)
```

每次update记录：

```text
smooth_feat_before hash
smooth_feat_after hash
current_feat hash
alpha used
cos(before,current)
cos(after,current)
cos(before,after)
feature history length
recent memory cosine分布
next-frame actual association cost/margin/rank
```

要求：

```text
诊断关闭/开启均逐字节复现对应baseline/soft输出。
```

### Priority 2：直接检验两个竞争假设

```text
H1：M03是连续soft导致prototype无法跟随已经commit的新观测；
H2：M05是孤立/稀疏soft在关键适应点阻止prototype更新，
    即使没有连续soft也会在下一帧断轨/换ID。
```

比较正向M02键，寻找区分变量，不能仅按rank1/正margin取消soft，因为M02正向键中也存在这些模式。

### Priority 3：新的验证协议

一旦使用M05设计新动作：

```text
M05不再是locked sequence，而进入development；
必须重新定义未使用的外部验证协议，或采用sequence-level leave-one-out/cross-validation。
```

在新协议确定前，不宣称新方法泛化。

---

## 14. 最新一句话状态

```text
Commit v1证明：memory action必须感知真实association commitment；
它成功修复了M03连续双键失败并保住M02收益。

但M05证明：soft v1还存在另一类独立失败——
单次或稀疏的高alpha更新抑制，也可能阻止必要适应并在下一帧触发新ID。

因此下一步不是继续调commit阈值，而是直接观测prototype before/after，
把“连续commit失败”和“孤立适应失败”分成两个机制后再设计动作。
```
